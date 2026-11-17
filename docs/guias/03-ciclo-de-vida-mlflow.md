# Ciclo de vida con MLflow

**Objetivo.** Entrenar el modelo con un comando, registrarlo en MLflow con todo lo necesario para reconstruirlo y promoverlo a producción con un quality gate.
Al final tienes un modelo `@champion` y sabes reproducirlo desde sus tags.

**Problemas del prototipo que resuelve.** 4 (parámetros sin registro), 5 (el modelo no se guarda) y 9 (no queda registro de qué datos produjeron el resultado).

Necesitas haber terminado la [guía 02](02-del-notebook-al-paquete.md), con el lago en `data/raw` y `data/raw.dvc` en un commit. Todos los comandos se ejecutan desde la raíz de tu repositorio. Los run ids, el SHA de git y el hash de DVC que ves aquí son de la máquina del instructor; las métricas deben coincidir con las tuyas.

## Paso 1. Inicia el servidor de MLflow

Abre una **segunda terminal** en la raíz del repositorio. El servidor se queda corriendo ahí durante esta guía y la siguiente.

```bash
uv run credit-mlops mlflow-server
```

```text
[MLflow] Security middleware enabled. Allowed hosts: localhost:*, 127.0.0.1:*, host.docker.internal:*.
INFO:     Uvicorn running on http://127.0.0.1:5000 (Press CTRL+C to quit)
...
INFO:     Application startup complete.
```

(salida recortada: el servidor imprime más líneas de inicio y avisos de librerías que puedes ignorar). Abre http://127.0.0.1:5000 en el navegador. Debes ver la interfaz de MLflow, todavía sin corridas.

El comando guarda los metadatos en `mlflow.db` (SQLite) y los artefactos en `mlartifacts/`. Ambos están en `.gitignore`.

## Paso 2. Entrena

Vuelve a la primera terminal:

```bash
uv run credit-mlops train
```

```text
UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. ...
...
🏃 View run hpo-n100-d2 at: http://127.0.0.1:5000/#/experiments/1/runs/e185751f11da47ae84c5d7a2fab91c17
...
🏃 View run hpo-n400-d4 at: http://127.0.0.1:5000/#/experiments/1/runs/05cc126d236a4b879fccd6bf7aff83c9
...
Successfully registered model 'credit_default_pd'.
Created version '1' of model 'credit_default_pd'.
🏃 View run train-2023-01 at: http://127.0.0.1:5000/#/experiments/1/runs/44fdeaaeeb3a46cdaca52fef6419ffd3
{"name":"credit_default_pd","version":"1","run_id":"44fdeaaeeb3a46cdaca52fef6419ffd3"}
```

(salida recortada). Tarda unos 5 segundos. En ese tiempo el comando:

1. Carga la ventana por defecto (guía 02, paso 7) con etiquetas maduras.
2. Prueba las 9 combinaciones de la grilla y elige la mejor por AUC de validación.
3. Reentrena la ganadora con todo el periodo de entrenamiento y la evalúa una vez en el OOT.
4. Registra en MLflow una corrida padre (`train-2023-01`) con 9 corridas hijas (`hpo-n<árboles>-d<profundidad>`).
5. Guarda el modelo y lo registra como versión 1 de `credit_default_pd` con el alias `@challenger`.

La última línea es la referencia al modelo en JSON. El `UserWarning` sobre columnas enteras es un aviso de MLflow al construir la firma (ver paso 6).

## Paso 3. Recorre la corrida en la interfaz

En http://127.0.0.1:5000:

1. Entra a **Experiments** y abre `credit-default-training`. Si ves un selector de tipo de experimento, elige **Model training**.
2. En la tabla de **Runs** aparece `train-2023-01`. Expándela con el botón de su fila y verás las 9 corridas hijas. Cada hija tiene dos parámetros (`n_estimators`, `max_depth`) y una métrica (`validation_auc`).
3. Abre `train-2023-01`. En **Overview** revisa tres bloques.
   - En **Parameters** está la ventana completa (`train_start`, `oot_end`, `as_of`...), los hiperparámetros elegidos, `threshold` y `top_k`. Nuestro código no registra estos dos últimos como parámetros. Son la configuración del modelo pyfunc (`model_config`), y MLflow los copia a la corrida al guardar el modelo.
   - En **Metrics** todas las métricas empiezan con `oot_`. Las que terminan en `__lima`, `__norte`, etc. son por región.
   - En **Tags** están `git_sha`, `git_dirty`, `data_version` (el hash de DVC) y `settings_hash`. Con ellas se reconstruye el modelo (paso 9).
4. En **Artifacts** están `model_card.md` (la ficha del modelo) y `reference/reference.parquet`, una muestra de 5000 filas de entrenamiento con su PD. Esa muestra es la referencia contra la que el monitoreo mide el drift (guía 06).
5. En la navegación de MLflow, entra a **Models**. Ahí está `credit_default_pd` con la versión 1 y el alias `challenger`.

Ahora los mismos datos desde la terminal.

## Paso 4. La grilla del HPO como corridas hijas

```bash
uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); r = mlflow.search_runs(experiment_names=['credit-default-training']); r = r[r['tags.mlflow.runName'].str.startswith('hpo-')]; print(r[['params.n_estimators', 'params.max_depth', 'metrics.validation_auc']].sort_values('metrics.validation_auc', ascending=False).round(4).to_string(index=False))"
```

```text
params.n_estimators params.max_depth  metrics.validation_auc
                200                3                  0.7948
                100                3                  0.7944
                200                2                  0.7939
                400                2                  0.7938
                100                4                  0.7938
                400                3                  0.7937
                200                4                  0.7934
                100                2                  0.7933
                400                4                  0.7910
```

Gana 200 árboles con profundidad 3, el centro de la grilla. Eso es bueno, porque si el ganador estuviera en una esquina, el óptimo podría estar fuera de la grilla. Pero no le des más valor del que tiene. Entre el mejor y el peor hay 0.0038 de AUC, y los ocho primeros están a 0.0015 o menos entre sí. La superficie es plana. Casi cualquier combinación sirve, y con otra semilla podría ganar otra. Lo que importa es que la elección se hizo con datos de validación posteriores al ajuste, nunca con el OOT.

## Paso 5. La ficha del modelo (model card)

```bash
uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); run_id = mlflow.MlflowClient().get_model_version('credit_default_pd', '1').run_id; print(mlflow.artifacts.load_text('runs:/' + run_id + '/model_card.md'))"
```

```text
# Model card: credit_default_pd v1

## Intended use

Estimate the 12-month probability of default (PD) of a new consumer loan application and return a decision at the configured threshold (0.15) with the top 3 TreeSHAP factors. Teaching model: not for real lending decisions.

## Training window

- Training months: 2019-01 to 2021-06 (validation: last 6 months)
- Out-of-time (OOT) months: 2021-07 to 2021-12
- Labels observed as of: 2023-01 (12-month default window)
- Recency half-life (months): None

## Data version

- Data version: d6ccf41697b4c83a6db930f432fa8264.dir
- Git commit: f668a0d5f6981a8d96c6972817e525c051d0e627
- Settings hash: a02f4e64ab05

## Metrics

Computed on the OOT months (2021-07 to 2021-12); calibration ratio is mean predicted PD divided by the observed default rate (1.0 is perfectly calibrated).

| Scope | Loans | Default rate | ROC AUC | Calibration ratio |
|---|---|---|---|---|
| overall | 15000 | 0.101 | 0.804 | 0.973 |
| centro | 1767 | 0.098 | 0.807 | 1.017 |
| lima | 6828 | 0.097 | 0.808 | 0.989 |
| norte | 3011 | 0.112 | 0.794 | 0.913 |
| oriente | 1147 | 0.100 | 0.829 | 0.985 |
| sur | 2247 | 0.105 | 0.795 | 0.978 |

## Known limits
...
```

(salida recortada; antes del texto aparece una barra de progreso `Downloading artifacts`). El código genera la ficha en cada entrenamiento (`registry/model_card.py`), así que nunca queda desactualizada. Data version y Git commit son de la máquina del instructor.

Lee la tabla. El AUC fuera de tiempo es 0.804 y la razón de calibración 0.973. En promedio, el modelo predice el 97 % de los defaults que ocurren. Norte es la región más débil (AUC 0.794, calibración 0.913). Recuérdalo para la guía 06.

## Paso 6. La firma (signature)

MLflow guarda junto al modelo una firma: qué columnas recibe, de qué tipo, y qué devuelve.

```bash
uv run python -c "import mlflow; mlflow.set_tracking_uri('http://127.0.0.1:5000'); print(mlflow.models.get_model_info('models:/credit_default_pd/1').signature)"
```

```text
inputs: 
  ['bureau_score': long (required), 'monthly_income_pen': double (required), 'debt_to_income': double (required), 'loan_amount_pen': double (required), 'term_months': long (required), 'age': long (required), 'months_at_job': long (required), 'num_active_lenders': long (required), 'max_days_past_due_12m': long (required), 'product': string (required), 'region': string (required), 'channel': string (required)]
outputs: 
  ['probability_default': float (required), 'decision': string (required), 'base_log_odds': double (required), 'top_factors': string (required)]
params: 
  None
```

(salida recortada: quitamos la barra de progreso). Además de la probabilidad, el modelo devuelve la decisión al umbral configurado y la explicación SHAP de cada fila. Es un modelo pyfunc de MLflow (`registry/pyfunc.py`) que envuelve el `Pipeline`, así que quien cargue el modelo recibe también la explicación sin escribir código extra.

Las variables enteras (`long`) no admiten valores faltantes. De ahí viene el `UserWarning` del paso 2. Es correcto aquí, porque el contrato de datos ya prohíbe nulos.

## Paso 7. Registry y aliases

```bash
uv run python -c "from mlflow import MlflowClient; print(MlflowClient('http://127.0.0.1:5000').get_registered_model('credit_default_pd').aliases)"
```

```text
{'challenger': '1'}
```

El registry numera las versiones (1, 2, 3...) y los aliases son nombres que apuntan a una versión. Usamos tres:

| Alias | Significado |
| --- | --- |
| `@challenger` | El último modelo entrenado, candidato a producción |
| `@champion` | El modelo en producción. La API y el scoring cargan `models:/credit_default_pd@champion` |
| `@previous_champion` | El champion anterior, para volver atrás con `credit-mlops rollback` |

Cambiar de modelo en producción es mover un alias. No hay que copiar archivos ni cambiar la configuración de la API.

## Paso 8. Promueve con el quality gate

```bash
uv run credit-mlops promote
```

```text
PROMOTED: challenger v1 is now champion (previous: none)
  - no champion yet
```

(salida recortada: quitamos la barra de progreso `Downloading artifacts`).

```bash
uv run python -c "from mlflow import MlflowClient; print(MlflowClient('http://127.0.0.1:5000').get_registered_model('credit_default_pd').aliases)"
```

```text
{'challenger': '1', 'champion': '1'}
```

Como no había champion, el gate aprueba sin comparar. Cuando ya hay uno, `registry/gate.py` evalúa a los dos modelos en los meses OOT del challenger (meses que el challenger no usó para entrenar) y exige tres reglas, todas configurables en `config/settings.toml`:

| Regla | Valor por defecto |
| --- | --- |
| El AUC del challenger es al menos el del champion más `gate_min_auc_gain` | 0.0 |
| La razón de calibración del challenger está dentro de 1 ± `calibration_tolerance` | 0.15 |
| Ninguna región pierde más de `gate_max_region_auc_drop` de AUC | 0.02 |

Si pasa, el champion actual pasa a `@previous_champion` y el challenger a `@champion`. Pase o no, la decisión queda registrada en la corrida del challenger, con un tag `gate` (`passed` o `failed`) y el artefacto `gate_decision.json` con las métricas de ambos modelos. Verás el gate rechazar un modelo en el paso 10.

## Paso 9. Reproduce un modelo desde sus tags

Con el SHA de git y el hash de DVC de un modelo, puedes reconstruirlo. Pruébalo. Primero lee los tags del modelo 1:

```bash
uv run python -c "from mlflow import MlflowClient; c = MlflowClient('http://127.0.0.1:5000'); tags = c.get_run(c.get_model_version('credit_default_pd', '1').run_id).data.tags; print(tags['git_sha'], tags['data_version'])"
```

```text
f668a0d5f6981a8d96c6972817e525c051d0e627 d6ccf41697b4c83a6db930f432fa8264.dir
```

Ahora cambia los datos, como si el lago se hubiera regenerado después del entrenamiento:

```bash
uv run credit-mlops simulate --loans-per-month 1000
```

```text
Wrote 84 months (84000 loans, 2019-01 to 2025-12) to data/raw
Incidents applied: none
```

```bash
uv run dvc status
```

```text
data/raw.dvc:
	changed outs:
		modified:           data/raw
```

Vuelve al commit del modelo. Copia el SHA que imprimió **tu** comando (el de abajo es el del instructor):

```bash
git checkout f668a0d5f6981a8d96c6972817e525c051d0e627
```

```text
Note: switching to 'f668a0d5f6981a8d96c6972817e525c051d0e627'.

You are in 'detached HEAD' state. ...

HEAD is now at f668a0d data: track the simulated lake with DVC
```

(salida recortada). En tu repositorio ese commit es probablemente el último, así que el código no cambia. En un proyecto real sería un commit de hace semanas, y `git checkout` te devolvería el código de ese momento.

Trae los datos de ese commit:

```bash
uv run dvc checkout --force
```

```text
M       data/raw/
```

```bash
uv run dvc status
```

```text
Data and pipelines are up to date.
```

Entrena de nuevo:

```bash
uv run credit-mlops train
```

```text
...
{"name":"credit_default_pd","version":"2","run_id":"724a0fb984134198aa6b36cf0fbba50b"}
```

(salida recortada). Compara las dos versiones. Este comando funciona igual en macOS y PowerShell:

```bash
uv run python -c "
from mlflow import MlflowClient
c = MlflowClient('http://127.0.0.1:5000')
for v in ['1', '2']:
    run = c.get_run(c.get_model_version('credit_default_pd', v).run_id).data
    print('v' + v, run.tags['git_sha'][:7], run.tags['data_version'][:8], round(run.metrics['oot_roc_auc'], 6), round(run.metrics['oot_calibration_ratio'], 6))
"
```

```text
v1 f668a0d d6ccf416 0.804272 0.973481
v2 f668a0d d6ccf416 0.804272 0.973481
```

Mismo código, mismos datos, mismas métricas hasta el sexto decimal. Vuelve a tu rama:

```bash
git switch -
```

```text
Switched to branch 'main'
```

En la interfaz puedes marcar las dos corridas `train-2023-01` en la tabla de **Runs** y pulsar **Compare** para verlas lado a lado.

## Paso 10. Mira al gate rechazar un modelo

La versión 2 es ahora `@challenger` y es idéntica al champion. Con las reglas por defecto pasaría, porque la ganancia mínima de AUC es 0.0. Exige una ganancia de 0.01 con un `.env`. Copia la plantilla (`cp .env.example .env` en macOS y PowerShell) y reemplaza su contenido por esta única línea. Si usas el Bloc de notas, elige **Guardar como** y **Todos los archivos**, o Windows guardará `.env.txt`:

```text
CREDIT_MLOPS_GATE_MIN_AUC_GAIN=0.01
```

```bash
uv run credit-mlops promote
```

```text
REJECTED: challenger v2 vs champion v1
  - AUC gain +0.0000 is below the required +0.0100
```

(salida recortada: quitamos la barra de progreso). El champion sigue siendo la versión 1. Borra el `.env` (el comando funciona también en PowerShell):

```bash
rm .env
```

```bash
uv run python -c "from mlflow import MlflowClient; print(MlflowClient('http://127.0.0.1:5000').get_registered_model('credit_default_pd').aliases)"
```

```text
{'challenger': '2', 'champion': '1'}
```

No promuevas la versión 2. La guía 04 sirve la versión 1.

## Por qué así

**Un servidor de MLflow, no archivos locales.** La API, el contenedor de Docker y el epílogo en Databricks hablan con un servidor por HTTP. Con `mlflow-server` todos ven el mismo registry.

**Un modelo pyfunc.** En la parte 2 registraste parámetros y métricas, pero no el modelo. Aquí el artefacto es el pipeline completo más la lógica de decisión y explicación. Quien lo cargue con `mlflow.pyfunc.load_model` obtiene exactamente lo mismo que la API, sin conocer XGBoost.

**Tags de trazabilidad.** `git_sha` y `data_version` responden "¿con qué código y con qué datos?". `git_dirty` avisa si había cambios sin commit al entrenar. En ese caso el SHA no basta para reproducir el modelo. Haz commit antes de entrenar un modelo que vayas a promover.

**Aliases en vez de etapas.** Las versiones antiguas de MLflow usaban etapas fijas (Staging, Production). Un alias es un nombre libre que mueves con una llamada, sin una lista fija de etapas, y es lo que usa Unity Catalog en Databricks (guía 07).

**El gate en código.** La decisión de promover queda escrita en `registry/gate.py`, se prueba con `pytest` y deja registro. No depende de que alguien revise la interfaz a mano.

## Si algo falla

**`train`, `promote` o `rollback` terminan al instante con `MLflow server at http://127.0.0.1:5000 is unreachable (<urlopen error [Errno 61] Connection refused>); start it with: uv run credit-mlops mlflow-server`.** El servidor de MLflow no está corriendo. Inícialo en otra terminal (paso 1) y repite el comando. `uv run credit-mlops doctor` también te dice si el servidor responde. En Windows el texto entre paréntesis cambia, pero el resto es igual.

**`promote` responde `version 1 is already champion`** (código de salida 1). No hay un challenger nuevo que evaluar, porque `@challenger` apunta a la misma versión que `@champion`. Entrena primero con `uv run credit-mlops train`.

## Para el instructor

40 minutos. En la Mac del instructor, `mlflow-server` estuvo listo entre 3 y 9 s, `train` tardó unos 7,5 s (ahora valida el contrato de datos de los 36 meses antes de entrenar) y `promote` entre 2 y 3 s. La mayor parte del bloque es el recorrido por la interfaz (paso 3) y la lectura de la grilla plana (paso 4). Si vas tarde, recorta la ficha del modelo (paso 5) y el rechazo del gate (paso 10); el paso 9 (reproducir) es el cierre del bloque y no conviene saltarlo. Pide a los estudiantes que dejen corriendo la terminal del servidor para la guía 04.
