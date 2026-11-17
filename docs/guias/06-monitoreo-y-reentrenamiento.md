# Monitoreo y reentrenamiento

**Objetivo.** Vigilar el modelo en producción con dos señales: el drift de los datos (PSI), que ves desde el primer día, y la performance con etiquetas maduras, que llega un año después. Luego recorrer tres años simulados de producción con alertas, un reentrenamiento, el gate y el cambio de alias.
Al final sabes qué detector atrapa cada tipo de incidente y por qué un modelo nuevo no se puede confirmar hasta que maduran sus propias etiquetas.

**Problema del prototipo que trata.** El prototipo nunca llega a producción, así que nadie revisa qué pasa después del AUC. Esta guía retoma el problema 2 (etiquetas usadas antes de existir) del lado de producción. En el mes M solo conoces el resultado de los préstamos originados 12 meses antes o más.

Necesitas haber terminado la [guía 04](04-sirviendo-el-modelo.md), con el servidor de MLflow corriendo en la segunda terminal y la versión 1 como `@champion`. Las salidas salen de la Mac del instructor. Tus números deben coincidir, porque el simulador y el entrenamiento usan una semilla fija.

## Paso 1. El PSI, a mano

El PSI (Population Stability Index) mide cuánto cambió la distribución de una variable entre una muestra de referencia y el mes actual. Para cada categoría (o cada tramo, si la variable es numérica) suma este término:

```text
PSI = suma de (actual% - referencia%) * ln(actual% / referencia%)
```

Un ejemplo con una variable de dos categorías. En la referencia, A y B aparecen 50% y 50%. Este mes aparecen 80% y 20%. Calcula cada término:

```bash
uv run python -c "import math; a = (0.8 - 0.5) * math.log(0.8 / 0.5); b = (0.2 - 0.5) * math.log(0.2 / 0.5); print(round(a, 4), round(b, 4), round(a + b, 4))"
```

```text
0.141 0.2749 0.4159
```

Los dos términos son positivos, porque el signo de la diferencia y el del logaritmo siempre coinciden. Ahora con la función del paquete, `monitoring/psi.py`:

```bash
uv run python -c "import pandas as pd; from credit_mlops.monitoring.psi import psi_categorical; ref = pd.Series(['A'] * 50 + ['B'] * 50); cur = pd.Series(['A'] * 80 + ['B'] * 20); print(round(psi_categorical(ref, cur), 4))"
```

```text
0.4159
```

El mismo valor. Una prueba (`tests/monitoring/test_psi.py`) verifica este cálculo. Para variables numéricas, la función corta 10 tramos en los deciles de la referencia y aplica la misma suma. Si un tramo queda vacío, usa una proporción mínima de 0.0001 para que el logaritmo no explote.

Los umbrales están en `config/settings.toml` y siguen la convención bancaria:

| PSI | Lectura | Qué hace el monitoreo |
| --- | --- | --- |
| menor que 0.10 | estable | nada |
| de 0.10 a 0.25 | vigilar (`watch`) | lo anota como razón, el estado no cambia |
| 0.25 o más | actuar (`act`) | estado `investigate` |

Con 0.4159, el ejemplo estaría en `investigate`.

## Paso 2. La referencia pertenece al modelo

¿Contra qué se compara cada mes? Contra una muestra de 5000 filas de la ventana de entrenamiento, que `train` guardó como artefacto de la corrida junto con la probabilidad que el modelo predijo para cada fila. Léela desde el champion:

```bash
uv run python -c "import mlflow, pandas as pd; mlflow.set_tracking_uri('http://127.0.0.1:5000'); v = mlflow.MlflowClient().get_model_version_by_alias('credit_default_pd', 'champion'); ref = pd.read_parquet(mlflow.artifacts.download_artifacts(run_id=v.run_id, artifact_path='reference/reference.parquet')); print('champion v' + v.version, 'run', v.run_id[:8], ref.shape); print(ref['channel'].value_counts(normalize=True).round(3).to_dict())"
```

```text
champion v1 run 3b9f8c20 (5000, 13)
{'branch': 0.789, 'digital': 0.211}
```

(salida recortada: quitamos la barra `Downloading artifacts`). El id de la corrida cambia en tu máquina. En la referencia, el 21.1% de las solicitudes llegó por el canal digital. Compáralo con julio de 2024:

```bash
uv run python -c "import pandas as pd; df = pd.read_parquet('data/raw/applications/2024-07.parquet'); print(df['channel'].value_counts(normalize=True).round(3).to_dict())"
```

```text
{'branch': 0.609, 'digital': 0.391}
```

La referencia es la del modelo y no la del mes anterior. Un cambio gradual se ve pequeño de un mes al siguiente, y aun así se acumula. La pregunta que importa es si los datos de hoy se parecen a los datos con los que el modelo aprendió. Cuando cambia el champion, cambia también la referencia.

## Paso 3. Las etiquetas maduran 12 meses después

La etiqueta es "cayó en default dentro de sus primeros 12 meses". Un préstamo originado en julio de 2023 recién tiene etiqueta en julio de 2024:

```text
2023-07                                    2024-07
   |---------------- 12 meses ----------------|
originación                          ya sabes si cayó en default
```

En cada mes M, el monitoreo junta las 3 cohortes más recientes que ya maduraron (M-12, M-13 y M-14). Una sola cohorte tiene pocos defaults por región; tres juntas alcanzan para juzgar. Para julio de 2024:

```bash
uv run python -c "from datetime import date; from credit_mlops.months import add_months, month_label; m = date(2024, 7, 1); print([month_label(add_months(m, -12 - i)) for i in range(3)])"
```

```text
['2023-07', '2023-06', '2023-05']
```

El monitoreo usa las predicciones que el champion actual **hizo** para esas cohortes (los archivos de `data/predictions/batch`), no unas recalculadas hoy. Solo cuentan las regiones con al menos 500 préstamos en las 3 cohortes (`region_min_loans`). Para cada una mide el AUC y la razón de calibración (PD promedio dividida entre la tasa de default observada).

El drift lo ves el mismo mes; la performance, 12 meses después.

## Paso 4. `credit-mlops monitor`

Puntúa julio de 2024 y monitorea ese mes:

```bash
uv run credit-mlops score --month 2024-07
```

```text
Wrote data/predictions/batch/2024-07.parquet
  approve: 2166
  review: 334
```

```bash
uv run credit-mlops monitor --month 2024-07
```

```text
2024-07: status ok (model v1)
  - PSI watch: channel = 0.157 (>= 0.1)
  - labels not yet matured for cohorts scored by v1
Top PSI:
  channel: 0.157
  months_at_job: 0.063
  age: 0.048
  loan_amount_pen: 0.016
  prediction: 0.011
```

(salidas recortadas: quitamos la barra `Downloading artifacts` y las dos líneas `View run` y `View experiment` que imprime MLflow). El canal digital pasó de 21.1% a 39.1%, y su PSI es 0.157, en la zona de vigilancia. `prediction` es el PSI de la probabilidad predicha.

El mensaje sobre las etiquetas dice "not yet matured", pero las cohortes 2023-05 a 2023-07 sí están maduras. Lo que falta son las predicciones. Nunca puntuaste esos meses, así que no hay nada que comparar con los defaults reales. Puntúalos:

```bash
uv run credit-mlops score --month 2023-05
uv run credit-mlops score --month 2023-06
uv run credit-mlops score --month 2023-07
```

```text
Wrote data/predictions/batch/2023-05.parquet
  approve: 2143
  review: 357
Wrote data/predictions/batch/2023-06.parquet
  approve: 2196
  review: 304
Wrote data/predictions/batch/2023-07.parquet
  approve: 2173
  review: 327
```

Y monitorea otra vez:

```bash
uv run credit-mlops monitor --month 2024-07
```

```text
2024-07: status retrain (model v1)
  - PSI watch: channel = 0.157 (>= 0.1)
  - calibration ratio for region norte = 0.71 outside 1 ± 0.15 (z = 5.9, n=1530)
Top PSI:
  channel: 0.157
  months_at_job: 0.063
  age: 0.048
  loan_amount_pen: 0.016
  prediction: 0.011
```

En la región norte el modelo predice en promedio el 71% de los defaults que de verdad ocurrieron. Subestima el riesgo. Los PSI siguen bajos, porque el shock no cambia las variables de norte. Desde julio de 2023 hay un shock económico en esa región, y los mismos perfiles caen más en default. Ninguna medida de drift lo ve; solo lo ven las etiquetas maduras.

El estado de un mes sigue esta prioridad:

1. `block`: el contrato de datos falló. El mes no se puntúa.
2. `investigate`: algún PSI (de una variable o de la predicción) llegó a 0.25.
3. `retrain`: alerta de performance. El AUC cayó más de 0.03 frente al AUC fuera de tiempo del champion, o la calibración salió de 1 ± 0.15 en total o en una región, con evidencia estadística (paso 5).
4. `ok`.

Cada `monitor` queda como una corrida en el experimento `credit-default-monitoring` de MLflow, con los PSI y las métricas maduras como métricas, y el reporte completo en `report.json`. Para cada variable con PSI de 0.10 o más guarda además un gráfico de su distribución frente a la referencia. Ábrelo en http://127.0.0.1:5000.

## Paso 5. El laboratorio: tres años de producción

El notebook `notebooks/06_laboratorio_monitoreo.ipynb` repite el ciclo mensual de enero de 2023 a diciembre de 2025. Trabaja en una carpeta temporal con su propio lago y su propia base de MLflow, así que no toca tu servidor ni tu carpeta `data/`. Ábrelo:

```bash
uv run jupyter notebook notebooks/06_laboratorio_monitoreo.ipynb
```

En el menú **Run**, elige **Run All Cells**. En la Mac del instructor, ejecutar el notebook completo tardó entre 42 y 52 segundos. Mientras corre, lee lo que sigue. Los números citados abajo son las salidas del notebook.

### El estado de cada mes

La tabla de estados muestra `ok` en todo 2023 y de enero a junio de 2024, `retrain` de julio a diciembre de 2024, y `ok` en todo 2025. El gráfico de PSI muestra a `channel` en la zona de vigilancia durante 2024, sin llegar a 0.25, y el PSI de la predicción siempre cerca de cero:

| Mes | PSI de `channel` | PSI de la predicción | Versión del champion |
| --- | --- | --- | --- |
| 2023-01 | 0.070 | 0.012 | 1 |
| 2024-07 | 0.157 | 0.011 | 1 |
| 2024-12 | 0.149 | 0.008 | 1 |
| 2025-01 | 0.031 | 0.006 | 2 |
| 2025-12 | 0.065 | 0.005 | 2 |

En 2025-01 el PSI de `channel` baja de golpe. Los datos no cambiaron; cambió la referencia. El nuevo champion trae la suya, tomada de su ventana de entrenamiento (2022-10 a 2023-09), más reciente que la del primer modelo.

### Por qué una alerta de calibración exige |z| ≥ 4

El notebook muestra la razón de calibración y el estadístico z de norte y oriente en 2024, con los préstamos de oriente en las 3 cohortes (extracto):

| Mes | Estado | Racha | Norte, razón | Norte, z | Oriente, razón | Oriente, z | Oriente, n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-04 | ok | 0 | 1.04 | -0.63 | 0.74 | 3.31 | 592 |
| 2024-06 | ok | 0 | 0.92 | 1.26 | 0.89 | 1.25 | 632 |
| 2024-07 | retrain | 1 | 0.71 | 5.86 | 1.08 | -0.68 | 651 |
| 2024-09 | retrain | 3 | 0.52 | 13.75 | 1.19 | -1.44 | 630 |
| 2024-12 | retrain | 6 | 0.59 | 10.17 | 1.05 | -0.44 | 568 |

En 2024-04, oriente tiene una razón de 0.74, fuera de la banda 1 ± 0.15, y el mes sigue en `ok`. Una alerta de calibración necesita dos condiciones a la vez:

- la razón sale de la banda: |razón − 1| > 0.15 (`calibration_tolerance`);
- la diferencia es estadísticamente significativa: |z| ≥ 4 (`calibration_min_z`).

El z compara los defaults observados con los esperados según el modelo: z = (observados − Σp) / raíz(Σ p(1 − p)), sumando sobre los préstamos de esa región.

La razón de exigir 4 y no 2 son las comparaciones múltiples. Cada mes se hacen unas 6 pruebas de calibración (la global y una por región), unas 72 al año. Con tantas pruebas, alguna sale extrema por azar aunque el modelo esté perfecto. El notebook cuenta las falsas alarmas esperadas en un año:

```text
|z| >= 2.0: 3.276 falsas alarmas esperadas al año
|z| >= 3.0: 0.194 falsas alarmas esperadas al año
|z| >= 4.0: 0.005 falsas alarmas esperadas al año
Bonferroni con 1% al año: |z| >= 3.81
```

Con |z| ≥ 2 tendrías unas 3 alertas falsas al año. La corrección de Bonferroni reparte un 1% de error entre las 72 pruebas y pide |z| ≥ 3.81. Usamos 4 porque la varianza binomial supone que el modelo conoce la probabilidad exacta, y el modelo también tiene error de estimación. El z de 3.31 en oriente, con 592 préstamos, es una desviación grande, pero es compatible con el azar entre 72 pruebas. Norte, con z entre 5.86 y 13.75 desde julio, no lo es.

### La política de 6 alertas seguidas

`monthly_cycle` lleva una racha de alertas. Cada mes con alerta de performance suma 1, un mes `block` no la cambia (no hubo datos para juzgar) y cualquier otro mes la vuelve a 0. Con la racha en 6 (`retrain_after_consecutive_alerts`), el flujo reentrena. En la tabla, la racha va de 1 en julio a 6 en diciembre.

¿Por qué esperar 6 meses con el problema a la vista? Por dos razones. Un mes malo aislado puede ser ruido, y reaccionar a cada uno cambia el modelo sin motivo. Además, en julio solo hay una cohorte madura posterior al shock; en diciembre hay seis, y el challenger necesita esos ejemplos para aprender el nuevo riesgo de norte. Mientras espera, el equipo puede actuar de otra forma, por ejemplo con un ajuste manual de la política de crédito en norte.

### El reentrenamiento de 2024-12

El challenger entrena con `TrainingWindow.rolled`, una ventana que termina en la última cohorte madura:

```bash
uv run python -c "from datetime import date; from credit_mlops.settings import TrainingWindow; w = TrainingWindow.rolled(date(2024, 12, 1)); print(w.as_params())"
```

```text
{'train_start': '2022-10', 'train_end': '2023-09', 'oot_start': '2023-10', 'oot_end': '2023-12', 'as_of': '2024-12', 'validation_months': '3', 'label_window_months': '12', 'half_life_months': '3.0', 'validation_start': '2023-07'}
```

Los meses fuera de tiempo (2023-10 a 2023-12) son las cohortes maduras más recientes. Los 12 meses de entrenamiento terminan justo antes. Además, cada préstamo pesa según su antigüedad, con una vida media de 3 meses (`retrain_half_life_months`):

```bash
uv run python -c "from datetime import date; import pandas as pd; from credit_mlops.months import month_label, month_range; from credit_mlops.training.evaluate import recency_weights; months = month_range(date(2022, 10, 1), date(2023, 9, 1)); w = recency_weights(pd.Series(months), months[-1], 3.0); print(pd.Series(w, index=[month_label(m) for m in months]).round(3).to_string())"
```

```text
2022-10    0.079
2022-11    0.099
2022-12    0.125
2023-01    0.157
2023-02    0.198
2023-03    0.250
2023-04    0.315
2023-05    0.397
2023-06    0.500
2023-07    0.630
2023-08    0.794
2023-09    1.000
```

Un préstamo de 2023-06 pesa la mitad que uno de 2023-09, y uno de 2023-03 la cuarta parte. Así el challenger aprende más de los meses posteriores al shock, que empezó en 2023-07.

El gate compara challenger y champion sobre los meses fuera de tiempo del challenger, que ya incluyen el shock. La tabla del notebook:

| | challenger | champion |
| --- | --- | --- |
| AUC | 0.827 | 0.817 |
| calibración global | 1.011 | 0.944 |
| calibración norte | 0.843 | 0.594 |

El challenger gana en AUC y en calibración global y pasa el gate: `reentrenamiento en 2024-12: gate aprobado, nuevo champion v2`. En norte mejora de 0.594 a 0.843, aunque sigue algo por debajo de 0.85. El gate revisa la calibración global y el AUC por región, no la calibración por región. Si en tu dominio la calibración de un segmento importa, agrégala como regla del gate.

### Por qué no puedes confirmar el nuevo champion hasta 2026

El nuevo champion puntúa desde 2025-01. Esa cohorte madura en 2026-01, después del final de la simulación. Durante todo 2025 el monitoreo dice lo mismo (salida del notebook para 2025-12):

```text
ignored 7500 matured rows scored by other model versions (v1)
labels not yet matured for cohorts scored by v2
```

Las cohortes que maduran en 2025 (originadas en 2024) las puntuó la versión 1. Sirven para saber que la versión 1 fallaba, no para saber si la versión 2 acierta. El gate dio evidencia sobre datos de 2023; la confirmación en producción llega 12 meses después del cambio. Hasta entonces, la única señal temprana es el PSI. Por eso los estados de 2025 son `ok`. No hay alerta, pero tampoco hay confirmación.

### Tres incidentes y quién atrapa cada uno

El notebook termina regenerando el lago con dos incidentes y corriendo el ciclo de ese mes:

```text
2025-03 (bug de ingresos): block, puntuado: False
  - data contract failed
  - monthly_income_pen: values outside [500, 50000] (2500 rows)

2025-06 (cambio en el buró): investigate, puntuado: True
  - PSI act: bureau_score = 0.829 (>= 0.25)
  - PSI act: prediction = 0.604 (>= 0.25)
  - labels not yet matured for cohorts scored by v2
```

| Incidente | Qué cambia | Quién lo detecta | Cuándo | Estado |
| --- | --- | --- | --- | --- |
| Bug de ingresos (2025-03) | `monthly_income_pen` llega en miles de soles | El contrato de datos | Ese mes, antes de puntuar | `block` |
| Cambio en el buró (2025-06) | Todos los scores bajan 80 puntos | El PSI de `bureau_score` (0.829) y el de la predicción | Ese mes | `investigate` |
| Shock en norte (desde 2023-07) | El riesgo real, con las mismas variables | La calibración de norte con z ≥ 4 | 12 meses después (2024-07) | `retrain`, y reentrenamiento en 2024-12 tras 6 alertas |

El bug de ingresos produce valores imposibles (6.5 soles al mes en vez de 6500), y el contrato los rechaza. El cambio en el buró produce valores posibles. Pasa el contrato, pero el PSI lo ve el mismo mes. Su estado es `investigate` y no `retrain`, porque reentrenar sobre un error de datos lo volvería parte del modelo. Primero una persona averigua si el proveedor cambió la escala o si los clientes cambiaron.

Cuando termines, cierra la pestaña, detén Jupyter con `Ctrl+C` y deshaz el guardado automático del notebook:

```bash
git restore notebooks/06_laboratorio_monitoreo.ipynb
```

## Paso 6. El ciclo mensual con Prefect

`flows.py` define el ciclo como un flujo de Prefect. `monthly_cycle` encadena las tareas de validar, puntuar y monitorear, y entrena y promueve cuando la racha lo pide. `simulate_production` lo llama mes por mes. El notebook usó estas mismas funciones. La CLI las ejecuta contra tu servidor de MLflow:

```bash
uv run credit-mlops simulate-production --start 2023-01 --end 2023-03
```

```text
15:44:07.016 | INFO    | prefect - Starting temporary server on http://127.0.0.1:8703
...
2023-01  ok          scored=True
2023-02  ok          scored=True
2023-03  ok          scored=True
```

(salida recortada: Prefect y MLflow imprimen una línea por cada tarea). El puerto del servidor temporal cambia en cada ejecución. Sin un servidor de Prefect configurado, Prefect levanta uno temporal que vive mientras corre el comando. Tardó unos 12 segundos en la Mac del instructor.

### Opcional: la interfaz de Prefect y la programación mensual

Abre una **tercera terminal** en la raíz del repositorio e inicia el servidor de Prefect:

```bash
uv run prefect server start
```

```text
Configure Prefect to communicate with the server with:

    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

View the API reference documentation at http://127.0.0.1:4200/docs

Check out the dashboard at http://127.0.0.1:4200
```

(salida recortada). En la primera terminal, apunta Prefect a ese servidor. El comando guarda el valor en tu perfil de Prefect, así que funciona igual en macOS y PowerShell:

```bash
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
```

```text
Set 'PREFECT_API_URL' to 'http://127.0.0.1:4200/api'.
Updated profile 'ephemeral'.
```

Corre tres meses más y lista las ejecuciones:

```bash
uv run credit-mlops simulate-production --start 2023-04 --end 2023-06
```

```text
2023-04  ok          scored=True
2023-05  ok          scored=True
2023-06  ok          scored=True
```

```bash
uv run prefect flow-run ls
```

```text
                                                   Flow Runs
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃                                   ID ┃ Flow                ┃ Name               ┃ State     ┃ When           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 8a492937-9f5d-4bfb-b8ed-7b5745f9bd62 │ monthly-cycle       │ alluring-monkey    │ COMPLETED │ 8 seconds ago  │
│ 372b05ce-578a-44bd-979b-e5e704e5d7eb │ monthly-cycle       │ nickel-grasshopper │ COMPLETED │ 9 seconds ago  │
│ 63a853d6-6c4d-43c3-90c9-5a3317e65063 │ monthly-cycle       │ invincible-jackal  │ COMPLETED │ 10 seconds ago │
│ 6a6f4726-ba72-4742-94b2-2addfd538998 │ simulate-production │ flexible-mammoth   │ COMPLETED │ 8 seconds ago  │
└──────────────────────────────────────┴─────────────────────┴────────────────────┴───────────┴────────────────┘
```

Los ids y los nombres son aleatorios. Hay una ejecución de `simulate-production` y una de `monthly-cycle` por mes, como subflujos. Ábrelas en http://127.0.0.1:4200 para ver cada tarea con su duración y su estado.

Para programar el ciclo, `serve-flow` publica `monthly_cycle` con un cron que corre a las 06:00 UTC del día 1 de cada mes y procesa el mes anterior:

```bash
uv run credit-mlops serve-flow
```

```text
Your flow 'monthly-cycle' is being served and polling for scheduled runs!

To trigger a run for this flow, use the following command:

        $ prefect deployment run 'monthly-cycle/monthly-cycle'

You can also run your flow via the Prefect UI: http://127.0.0.1:4200/deployments/deployment/0aa0e602-dfd0-412e-84fe-de2d4d8906f5
```

En la interfaz aparece el deployment `monthly-cycle` con el cron `0 6 1 * *`.

Este deployment nunca reentrena. Cada ejecución programada empieza con la racha de alertas en 0, porque `monthly_cycle` recibe la racha de quien lo llama y aquí nadie se la pasa. Así nunca llega a 6. `simulate_production` sí pasa la racha de un mes al siguiente. En producción tendrías que guardar la racha entre ejecuciones, por ejemplo en MLflow o en una tabla, y leerla al empezar cada mes.

No lances una ejecución. El mes anterior a hoy no existe en el lago simulado, que termina en 2025-12. Detén `serve-flow` con `Ctrl+C`, detén el servidor de Prefect con `Ctrl+C` en la tercera terminal y quita la configuración:

```bash
uv run prefect config unset PREFECT_API_URL
```

```text
Unset 'PREFECT_API_URL'.
Updated profile 'ephemeral'.
```

### Prefect y Airflow

Muchos equipos de datos usan Airflow. Los conceptos se corresponden:

| Prefect | Airflow | En este repositorio |
| --- | --- | --- |
| Flow (`@flow`) | DAG | `monthly_cycle`, `simulate_production` |
| Task (`@task`) | Task (un operador, por ejemplo `PythonOperator`) | `validate_task`, `score_task`, `monitor_task`, `train_task`, `promote_task` |
| Deployment con schedule (`flow.serve(cron=...)`) | El `schedule` del DAG | `serve-flow` con `0 6 1 * *` |
| Prefect UI (`prefect server start`) | Airflow UI (webserver) | http://127.0.0.1:4200 |

Una diferencia práctica es que un flujo de Prefect es una función de Python que puedes llamar directamente, como hicimos en el notebook. En Airflow el DAG lo ejecuta el scheduler.

## Por qué así

**PSI escrito en casa.** Es un archivo corto, `monitoring/psi.py`, probado contra valores calculados a mano. Ves exactamente cómo se cortan los tramos y qué pasa con una categoría nueva. Herramientas como Evidently o NannyML hacen lo mismo y más, y quedan fuera de la sesión.

**La referencia guardada con el modelo.** Cada versión trae su muestra de referencia y su probabilidad predicha. Al promover un modelo cambia la referencia sin que nadie la actualice a mano, y siempre puedes reconstruir contra qué se comparó un mes.

**Cohortes agrupadas y z ≥ 4.** Una región pequeña tiene pocos defaults en cada cohorte, y con tan pocos casos la razón de calibración salta mucho de un mes a otro. Agrupar 3 cohortes y exigir significancia evita que el ruido dispare reentrenamientos.

**Una racha antes de reentrenar.** Reentrenar con calendario fijo ("cada mes") parece prudente y puede hacer daño. Mete al modelo meses con errores de datos, cambia las predicciones sin una razón que puedas explicar y no deja tiempo para reunir ejemplos del cambio real. Aquí el reentrenamiento necesita evidencia y pasa por el mismo gate que el primer modelo.

**`investigate` es para una persona.** Un PSI alto puede ser un bug o un cambio real del mercado, y el código no puede distinguirlos. El flujo no reentrena en ese caso.

**Prefect.** Lo usamos porque el flujo es Python normal con decoradores, corre sin servidor y la misma función sirve para la simulación y para la programación mensual.

**Reject inference (solo para mencionar).** En un banco real solo conoces el resultado de los préstamos que aprobaste. A los rechazados nunca les diste el préstamo, así que no sabes si habrían pagado. Un modelo reentrenado solo con aprobados aprende de una población recortada por el modelo anterior. El simulador le da resultado a todas las solicitudes, así que este sesgo no aparece en el laboratorio.

## Si algo falla

**`monitor` termina con `no batch predictions for 2024-08 at data/predictions/batch/2024-08.parquet; run: credit-mlops score --month 2024-08`.** Pasó durante la preparación de esta guía. El monitoreo necesita las predicciones del mes para calcular el PSI de la predicción. Ejecuta el `score` que sugiere el mensaje y repite.

**`monitor` responde `MLflow server at http://127.0.0.1:5000 is unreachable ... start it with: uv run credit-mlops mlflow-server`.** El servidor de MLflow no está corriendo. Inícialo en la segunda terminal (`uv run credit-mlops mlflow-server`) y repite.

**`simulate-production` termina con `RuntimeError: Failed to reach API at http://127.0.0.1:4200/api/`.** Configuraste `PREFECT_API_URL` y el servidor de Prefect está detenido. Inícialo (`uv run prefect server start`) o quita la configuración con `uv run prefect config unset PREFECT_API_URL`.

**El notebook falla con `ModuleNotFoundError` en VS Code.** El kernel no es el de tu proyecto. Elige el intérprete de `.venv`, o abre el notebook con `uv run jupyter notebook` como en el paso 5.

## Para el instructor

45 minutos. En la Mac del instructor, `score` tardó unos 4 s, `monitor` entre 3 y 6 s, el notebook completo entre 42 y 52 s, `simulate-production` de 3 meses unos 12 s con el servidor temporal de Prefect y unos 6 s con `prefect server start`, que estuvo listo en unos 3 s. Lanza el notebook (paso 5) apenas termines el paso 4 y comenta los resultados mientras corre. Dedica el tiempo a la tabla de incidentes y a la pregunta de por qué el champion de 2024-12 no se puede confirmar hasta 2026. Si vas tarde, recorta la parte opcional del paso 6 (interfaz de Prefect y `serve-flow`). La tabla Prefect y Airflow se lee en un minuto.
