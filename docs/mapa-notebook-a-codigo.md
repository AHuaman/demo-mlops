# Del notebook al código

Este mapa recorre las seis celdas de código de `notebooks/00_prototipo.ipynb` en orden y dice qué parte del paquete `credit_mlops` reemplaza cada una. Las celdas se nombran por el título de su sección en el notebook. Los números entre paréntesis son los problemas de la lista final del notebook (ver [Por qué MLOps](guias/01-por-que-mlops.md)).

Las rutas de archivo son relativas a `src/credit_mlops/`. Las excepciones están en la raíz del repositorio: lo que empieza con `config/`, `tests/` o `.github/`, y los archivos `pyproject.toml`, `Dockerfile`, `compose.yaml` y `data/raw.dvc`.

## Celda por celda

| Celda del prototipo | Reemplazo | Archivo | Por qué |
| --- | --- | --- | --- |
| **1. Cargar los datos.** `glob("../data/raw/applications/*.parquet")` y `pd.read_parquet` | `Settings.data_dir` y sus propiedades `applications_dir` y `outcomes_path` | `settings.py`, `config/settings.toml` | La ruta sale de la configuración y se arma con `pathlib`, así que funciona desde cualquier carpeta y en Windows (1) |
| | `load_applications(settings, months)` y `load_outcomes(settings)` | `data/lake.py` | Se pide una lista explícita de meses. Si falta uno, el error dice cuál y en qué carpeta, en vez de concatenar una lista vacía |
| | `credit-mlops validate --month` y `validate_applications` | `data/contract.py`, `cli.py` | Cada mes pasa por el contrato de Great Expectations antes de entrenar o puntuar (6) |
| | `dvc add data/raw` y el tag `data_version` | `data/raw.dvc`, `data/versioning.py` | El hash de los datos queda en git y en cada modelo (9) |
| | `warnings.filterwarnings("ignore")` se elimina | `pyproject.toml` (`filterwarnings` de pytest) | Silenciar todos los avisos esconde problemas reales. Las pruebas ignoran solo avisos concretos de librerías, por su mensaje exacto |
| **2. Construir la etiqueta.** `merge` con `outcomes` y `default_12m` calculado para todos los meses | `attach_labels(apps, outcomes, as_of)` e `is_matured` | `data/lake.py` | Solo etiqueta las cohortes que ya cumplieron su ventana de 12 meses a la fecha `as_of`; las demás no entran al entrenamiento (2) |
| | `TrainingWindow` y su validador "OOT labels are not mature at as_of" | `settings.py` | Una ventana con etiquetas inmaduras falla al crearse, antes de leer datos (2) |
| | `label_window_months = 12` | `config/settings.toml` | La ventana de la etiqueta es un parámetro con nombre, no un `DateOffset(months=12)` escondido en una celda |
| **3. Preparar las variables.** Listas `cat_cols` y `num_cols`, `astype("category")` a mano | `FEATURES`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` | `schema.py` | Una sola definición de las variables, sus rangos y sus categorías. El simulador, el contrato, el modelo y la API la importan |
| | `FeatureBuilder` y `CategoricalCaster` dentro de `build_pipeline` | `features.py` | El preprocesamiento es parte del modelo guardado. Las categorías son fijas, así que una categoría nueva se vuelve `NaN` en vez de romper el modelo (7) |
| | Variables derivadas `payment_to_income` y `loan_to_income` | `features.py` | Se calculan igual al entrenar, en batch y en la API, porque las calcula el mismo objeto |
| **4. Entrenar y evaluar con un split aleatorio.** `train_test_split` y `XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, ...)` | `train_model(settings, window)` | `training/train.py` | El HPO elige con los últimos meses del periodo de entrenamiento (validación temporal), nunca con un split aleatorio (3) |
| | Grilla `hpo_n_estimators`, `hpo_max_depth` y `learning_rate` | `config/settings.toml` | Los hiperparámetros dejan de ser números mágicos y quedan como parámetros de cada corrida en MLflow (4) |
| | `scale_pos_weight=1.0` | `training/train.py` | La salida se usa como PD calibrada. Ver el paso 9 de [la guía 02](guias/02-del-notebook-al-paquete.md) |
| | `credit-mlops train` y `log_and_register` | `cli.py`, `registry/logging.py` | El modelo se guarda como pyfunc en MLflow, con firma, ejemplo de entrada, tags de trazabilidad y alias `@challenger` (5, 9) |
| | `evaluate` en vez de un `roc_auc_score` suelto | `training/evaluate.py` | AUC, PR-AUC, Brier y razón de calibración, en total y por región |
| **5. ¿Y si evaluamos hacia el futuro?** Máscaras `month <= "2023-12-01"` y un segundo modelo | `DEFAULT_WINDOW` (`oot_start`, `oot_end`) | `settings.py` | La evaluación fuera de tiempo es la forma estándar de medir, no un experimento aparte (3) |
| | `credit-mlops promote` y `decide` | `registry/gate.py` | Challenger y champion se comparan en los mismos meses OOT con reglas escritas: AUC, calibración y caída máxima por región |
| | Ficha del modelo | `registry/model_card.py` | Las métricas OOT por región quedan escritas junto a cada versión |
| **6. Explicabilidad con SHAP.** `shap.TreeExplainer` y un gráfico beeswarm global | `explain` (contribuciones TreeSHAP por fila, con `pred_contribs` de XGBoost) | `registry/pyfunc.py` | La explicación es parte de cada predicción. El modelo devuelve el log-odds base y los 3 factores principales de cada solicitante |
| | `POST /predict` con `explanation` | `api/app.py`, `api/schemas.py` | Quien recibe una decisión ve sus razones (reason codes). El gráfico global sigue siendo útil para explorar, pero no explica una decisión individual |
| | Pruebas de consistencia de las contribuciones | `tests/registry/test_pyfunc.py` | Verifican que el log-odds base más las contribuciones sea igual al log-odds del modelo, y que coincidan con `shap.TreeExplainer` |

## Lo que el notebook no tenía

El problema 8 (imposible de programar o servir) no corresponde a una celda. Es lo que le falta al notebook completo.

| Necesidad | Reemplazo | Archivo | Guía |
| --- | --- | --- | --- |
| Ejecutar cada paso sin abrir un notebook | Comandos `credit-mlops` | `cli.py` | 02 |
| Puntuar un mes completo | `credit-mlops score --month` | `scoring.py` | 04 |
| Responder a otra aplicación | API con FastAPI | `api/app.py` | 04 |
| Correr igual en cualquier máquina | Imagen Docker slim | `Dockerfile`, `compose.yaml` | 04 |
| Saber si el código se rompió | Pruebas y CI | `tests/`, `.github/workflows/ci.yml` | 05 |
| Saber si el modelo se degradó | `credit-mlops monitor --month` | `monitoring/report.py` | 06 |
| Ejecutar el ciclo cada mes | Flujo de Prefect | `flows.py` | 06 |

## Los nueve problemas y dónde se corrigen

| # | Problema | Dónde se corrige |
| --- | --- | --- |
| 1 | Rutas fijas | `settings.py`, `config/settings.toml` |
| 2 | Etiquetas usadas antes de existir | `data/lake.py` (`attach_labels`), `settings.py` (`TrainingWindow`) |
| 3 | Split aleatorio | `training/train.py`, `settings.py` (`DEFAULT_WINDOW`) |
| 4 | Parámetros sin registro | `config/settings.toml`, `registry/logging.py` |
| 5 | El modelo no se guarda | `registry/logging.py`, `registry/pyfunc.py` |
| 6 | No hay validación de entradas | `data/contract.py`, `api/schemas.py` |
| 7 | Preprocesamiento fuera del modelo | `features.py` |
| 8 | Imposible de programar o servir | `cli.py`, `scoring.py`, `api/app.py`, `Dockerfile`, `flows.py` |
| 9 | No queda registro de qué datos produjeron el resultado | `data/versioning.py`, `data/raw.dvc`, `registry/logging.py` |
