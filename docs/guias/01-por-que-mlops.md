# Por qué MLOps

**Objetivo.** Ejecutar el notebook prototipo, encontrar sus nueve problemas y ver cómo los resuelve el resto de la sesión.
Al final sabes qué bloque del día corrige cada problema y qué herramienta usamos para hacerlo.

**Problemas del prototipo que trata esta guía.** Los nueve (1 a 9). Aquí solo los identificamos; las guías siguientes los corrigen.

Antes de empezar, termina [la preparación](00-preparacion.md). `uv run credit-mlops doctor` no debe mostrar ningún `✗`. Todos los comandos se ejecutan desde la carpeta raíz de tu repositorio. Las salidas que ves aquí salen de la Mac del instructor. Tus números deben coincidir, porque el simulador usa una semilla fija.

## El ciclo de vida de un modelo

El trabajo con un modelo no termina cuando el notebook imprime un buen AUC. El modelo pasa por este ciclo, y cada flecha es un lugar donde algo puede romperse:

```text
 datos  ->  validar  ->  entrenar  ->  registrar  ->  servir  ->  monitorear
   ^                                                                  |
   |                                                                  |
   +--------------------------  reentrenar  <-------------------------+
```

| Etapa | Pregunta que responde | Herramienta en este repo | Guía |
| --- | --- | --- | --- |
| Datos | ¿Con qué datos exactos entrené? | DVC | 02 |
| Validar | ¿Los datos de este mes cumplen el contrato? | Great Expectations | 02 |
| Entrenar | ¿Evalué como si estuviera en el futuro? | `TrainingWindow`, `Pipeline` de scikit-learn | 02 |
| Registrar | ¿Qué modelo está en producción y de dónde salió? | MLflow (tracking y registry) | 03 |
| Servir | ¿Cómo llega una predicción a otra aplicación? | FastAPI, Docker | 04 |
| Monitorear | ¿El modelo sigue funcionando? | PSI, etiquetas maduras, Prefect | 06 |
| Reentrenar | ¿Cuándo lo reemplazo, y con qué garantía? | Quality gate y aliases de MLflow | 03 y 06 |

MLOps es el conjunto de prácticas que hace que cada flecha sea repetible y verificable, sin depender de que alguien recuerde qué hizo en su notebook.

## Paso 1. Genera los datos

El repositorio trae un simulador de préstamos de consumo. Genera 7 años de solicitudes (el "lago" de datos) en `data/raw`:

```bash
uv run credit-mlops simulate
```

```text
Wrote 84 months (210000 loans, 2019-01 to 2025-12) to data/raw
Incidents applied: none
```

En Windows la ruta se imprime como `data\raw`. Son 84 archivos mensuales de solicitudes más un archivo de resultados (`outcomes.parquet`) con la fecha de default de cada préstamo que entró en default hasta el 31 de diciembre de 2025.

## Paso 2. Ejecuta el prototipo

Abre el notebook con Jupyter:

```bash
uv run jupyter notebook notebooks/00_prototipo.ipynb
```

Se abre una pestaña en tu navegador. En el menú **Run**, elige **Run All Cells** y espera a que termine (entre 20 y 40 segundos en la Mac del instructor; la celda de SHAP es la más lenta). Si prefieres VS Code, abre el archivo y elige el kernel de la carpeta `.venv`.

Estas son las salidas de las celdas de código, en orden.

Carga de datos:

```text
84 archivos, 210000 préstamos
```

Construcción de la etiqueta:

```text
tasa de default: 0.0948
último mes de originación: 2025-12-01
origination_month
2019    0.0944
2020    0.1015
2021    0.0944
2022    0.0986
2023    0.1030
2024    0.1201
2025    0.0518
Name: default_12m, dtype: float64
```

Split aleatorio:

```text
AUC split aleatorio: 0.797
```

Evaluación hacia el futuro (entrena hasta 2023-12 y evalúa en 2024):

```text
AUC split aleatorio: 0.797
AUC fuera de tiempo: 0.789
diferencia: 0.008
```

La última celda dibuja un gráfico SHAP (beeswarm) con la importancia de cada variable.

Cuando termines, cierra la pestaña y detén Jupyter con `Ctrl+C` en la terminal. Jupyter guarda el notebook solo, así que deshaz esos cambios para que tu repositorio quede limpio:

```bash
git restore notebooks/00_prototipo.ipynb
git status --short
```

`git status --short` no debe imprimir nada.

## Paso 3. Lee los números con desconfianza

**La tasa de default de 2025 engaña.** Cae de 0.1201 en 2024 a 0.0518 en 2025. Los préstamos de 2025 no son más seguros. La mayoría todavía no cumple sus 12 meses, así que sus defaults aún no ocurrieron. El notebook les puso etiqueta 0 igual. Un modelo entrenado con esas filas aprende que los préstamos recientes son más seguros de lo que son. Este es el problema 2, y es el más caro de la lista.

**El split aleatorio da un AUC algo optimista.** Aquí la diferencia es pequeña (0.008), porque estos datos cambian poco de un año a otro. No te confíes. La diferencia depende de cuánto cambie tu población en el tiempo, y un split aleatorio no te deja medirla. En la guía 06 verás un shock regional que un split aleatorio esconde por completo.

**El AUC no dice si la probabilidad es correcta.** Un banco usa la PD (probabilidad de default) para provisionar y fijar precios, así que necesita que una PD de 0.10 signifique 10 de cada 100. El notebook nunca lo mide. Lo haremos con la razón de calibración en la guía 02.

## Los nueve problemas

Es la lista de la última celda del notebook. La columna de la derecha dice dónde se corrige.

| # | Problema | Lo corregimos con | Guía |
| --- | --- | --- | --- |
| 1 | Rutas fijas (hardcoded): `../data/raw/...` solo funciona desde esa carpeta | `Settings` con `data_dir` en `config/settings.toml` | 02 |
| 2 | Etiquetas usadas antes de existir | `attach_labels` y los validadores de `TrainingWindow` | 02 |
| 3 | Split aleatorio sobre datos ordenados en el tiempo | Ventanas fuera de tiempo (OOT) y validación temporal para el HPO | 02 |
| 4 | Parámetros sin registro | Grilla en `config/settings.toml`; cada corrida guarda sus parámetros en MLflow | 02 y 03 |
| 5 | El modelo no se guarda | Modelo pyfunc registrado en MLflow con aliases | 03 |
| 6 | No hay validación de entradas | Contrato de datos con Great Expectations y esquema pydantic en la API | 02 y 04 |
| 7 | El preprocesamiento vive fuera del modelo | `Pipeline` de scikit-learn guardado completo dentro del modelo | 02 |
| 8 | Imposible de programar o servir | CLI, scoring batch, API con FastAPI, imagen Docker | 04 |
| 9 | No queda registro de qué datos produjeron el resultado | Hash de DVC y SHA de git como tags de cada modelo | 02 y 03 |

El documento [Del notebook al código](../mapa-notebook-a-codigo.md) recorre cada celda del prototipo y dice qué archivo del paquete la reemplaza.

## El plan del día

| # | Bloque | Guía | Tiempo |
| --- | --- | --- | --- |
| 0 | Preparación (antes de clase) | [00-preparacion.md](00-preparacion.md) | antes de clase |
| 1 | Por qué MLOps (esta guía) | [01-por-que-mlops.md](01-por-que-mlops.md) | 20 min |
| 2 | Del notebook al paquete | [02-del-notebook-al-paquete.md](02-del-notebook-al-paquete.md) | 40 min |
| 3 | Ciclo de vida con MLflow | [03-ciclo-de-vida-mlflow.md](03-ciclo-de-vida-mlflow.md) | 40 min |
| | Pausa | | 10 min |
| 4 | Sirviendo el modelo | [04-sirviendo-el-modelo.md](04-sirviendo-el-modelo.md) | 45 min |
| 5 | CI con GitHub Actions | [05-ci.md](05-ci.md) | 20 min |
| 6 | Monitoreo y reentrenamiento | [06-monitoreo-y-reentrenamiento.md](06-monitoreo-y-reentrenamiento.md) | 45 min |
| 7 | Epílogo en Databricks (demo del instructor) | [07-epilogo-databricks.md](07-epilogo-databricks.md) | 15 min, si hay tiempo |
| 8 | Checklist del capstone y cierre | [08-cierre.md](08-cierre.md) | 10 min |

Cada guía sigue la misma estructura: objetivo, problema que resuelve, pasos con la salida real, "Por qué así" para cada herramienta y "Si algo falla".

## Por qué así

**Un simulador en vez de datos reales.** Los datos de crédito reales son confidenciales. El simulador genera la misma estructura (buró, ingresos, región, canal) con una semilla fija, así que todos ven los mismos números. También permite inyectar incidentes a propósito (un bug de unidades, un cambio en el buró, un shock regional) para practicar cómo detectarlos.

**El notebook se queda.** Un notebook es una buena herramienta para explorar. El problema es usarlo como sistema de producción. Lo conservamos como punto de partida y lo comparamos celda por celda con el paquete.

## Si algo falla

**El notebook falla en la primera celda con `ValueError: No objects to concatenate`.** No hay archivos en `data/raw/applications`, así que `glob` devolvió una lista vacía. Ejecuta `uv run credit-mlops simulate` desde la raíz del repositorio y vuelve a correr el notebook. Es el problema 1. El notebook no te dice qué ruta buscó.

**`ModuleNotFoundError: No module named 'xgboost'` (o `shap`) en VS Code.** El kernel no es el de tu proyecto. Elige el intérprete de `.venv` en el selector de kernel, o abre el notebook con `uv run jupyter notebook` como en el paso 2.

## Para el instructor

20 minutos. En la Mac del instructor (Apple M5 Pro), `simulate` tardó entre 1 y 2 s y el notebook completo entre 20 y 40 s. Mientras el notebook corre, lee en voz alta la lista de problemas. Dedica la mayor parte del bloque a la caída de la tasa de 2025 (problema 2), el error más caro de la lista. La diferencia de AUC fuera de tiempo es pequeña en estos datos (0.008). Dilo tal cual y no la exageres.
