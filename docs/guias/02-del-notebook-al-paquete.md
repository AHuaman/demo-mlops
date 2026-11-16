# Del notebook al paquete

**Objetivo.** Recorrer el paquete `credit_mlops` que reemplaza al prototipo: configuración, CLI, datos versionados con DVC, contrato de datos y ventanas de entrenamiento.
Al final tu lago de datos tiene una versión registrada en git y sabes por qué el modelo se entrena y evalúa como lo hace.

**Problemas del prototipo que resuelve.** 1 (rutas fijas), 2 (etiquetas antes de tiempo), 3 (split aleatorio), 6 (sin validación de entradas) y 7 (preprocesamiento fuera del modelo). Prepara también el 4 (parámetros) y el 9 (qué datos), que se cierran en la guía 03.

Todos los comandos se ejecutan desde la raíz de tu repositorio. Las salidas salen de la Mac del instructor; donde un valor depende de tu máquina, lo indicamos. Ten a mano [el mapa del notebook al código](../mapa-notebook-a-codigo.md). Dice qué archivo reemplaza cada celda del prototipo.

## Paso 1. Recorre el paquete

```bash
git ls-files src
```

```text
src/credit_mlops/__init__.py
src/credit_mlops/api/__init__.py
src/credit_mlops/api/app.py
src/credit_mlops/api/schemas.py
src/credit_mlops/cli.py
src/credit_mlops/data/__init__.py
src/credit_mlops/data/contract.py
src/credit_mlops/data/lake.py
src/credit_mlops/data/versioning.py
src/credit_mlops/doctor.py
src/credit_mlops/features.py
src/credit_mlops/flows.py
src/credit_mlops/monitoring/__init__.py
src/credit_mlops/monitoring/plots.py
src/credit_mlops/monitoring/psi.py
src/credit_mlops/monitoring/report.py
src/credit_mlops/months.py
src/credit_mlops/prediction_log.py
src/credit_mlops/registry/__init__.py
src/credit_mlops/registry/gate.py
src/credit_mlops/registry/health.py
src/credit_mlops/registry/loading.py
src/credit_mlops/registry/logging.py
src/credit_mlops/registry/mlflow_setup.py
src/credit_mlops/registry/model_card.py
src/credit_mlops/registry/pyfunc.py
src/credit_mlops/schema.py
src/credit_mlops/scoring.py
src/credit_mlops/settings.py
src/credit_mlops/simulation/__init__.py
src/credit_mlops/simulation/config.py
src/credit_mlops/simulation/generator.py
src/credit_mlops/training/__init__.py
src/credit_mlops/training/evaluate.py
src/credit_mlops/training/train.py
```

Los archivos que más usamos hoy:

| Archivo o carpeta | Qué contiene |
| --- | --- |
| `schema.py` | La lista de variables, sus rangos y sus categorías. Es la única definición. El simulador, el contrato, el modelo y la API la importan de aquí |
| `settings.py` | `Settings` (la configuración) y `TrainingWindow` (las ventanas de entrenamiento) |
| `data/` | Lectura del lago (`lake.py`), contrato de datos (`contract.py`) y versiones (`versioning.py`) |
| `features.py` | El `Pipeline` de scikit-learn: variables derivadas, categorías fijas y XGBoost |
| `training/` | Entrenamiento con HPO temporal y evaluación fuera de tiempo |
| `registry/` | Todo lo de MLflow: registrar, promover, cargar el modelo (guía 03) |
| `api/` y `scoring.py` | Servir el modelo (guía 04) |
| `monitoring/` y `flows.py` | Monitoreo y ciclo mensual (guía 06) |
| `cli.py` | Los comandos `credit-mlops` |

Las pruebas están en `tests/`, con una carpeta por módulo.

## Paso 2. La configuración en `config/settings.toml` y `.env`

Abre `config/settings.toml`. Ahí viven los valores que en el notebook eran números sueltos: la carpeta de datos, la semilla, el umbral de decisión, la grilla del HPO, los umbrales de monitoreo. El código los lee a través de la clase `Settings` de `settings.py`:

```bash
uv run python -c "from credit_mlops.settings import Settings; s = Settings(); print(s.decision_threshold, s.data_dir, s.mlflow_tracking_uri)"
```

```text
0.15 data http://127.0.0.1:5000
```

Puedes sobrescribir cualquier valor sin tocar el código, con un archivo `.env`. Cópialo desde el ejemplo (este comando funciona igual en macOS y en PowerShell):

```bash
cp .env.example .env
```

Abre `.env` en tu editor, busca la línea `# CREDIT_MLOPS_DECISION_THRESHOLD=0.15`, quítale el `#` del inicio y cambia el valor a `0.20`. Guarda y repite el comando:

```bash
uv run python -c "from credit_mlops.settings import Settings; s = Settings(); print(s.decision_threshold, s.data_dir, s.mlflow_tracking_uri)"
```

```text
0.2 data http://127.0.0.1:5000
```

De mayor a menor prioridad, `Settings` lee los argumentos en el código, las variables de entorno `CREDIT_MLOPS_*`, el `.env` y por último `config/settings.toml`. Borra el `.env` para volver a los valores por defecto (también funciona en PowerShell):

```bash
rm .env
```

`.env` está en `.gitignore`. Es el lugar para valores de tu máquina o secretos, que nunca van a git. En la guía 07 el mismo mecanismo apunta el proyecto a Databricks sin cambiar una línea de código.

## Paso 3. La CLI

Cada paso del proceso es un comando. No hay que abrir un notebook ni recordar en qué orden correr las celdas:

```bash
uv run credit-mlops --help
```

```text
 Usage: credit-mlops [OPTIONS] COMMAND [ARGS]...

 credit-mlops command line interface.

╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────╮
│ version              Print the package version.                                                  │
│ validate             Check one month of applications against the data contract (exit 1 on        │
│                      failure).                                                                   │
│ simulate             Generate the synthetic loan lake under `data_dir`.                          │
│ train                Train, log to MLflow and register the model as the `challenger`.            │
│ mlflow-server        Run a local MLflow tracking server (SQLite backend, proxied artifacts).     │
│ promote              Run the promotion gate on `@challenger` and move `@champion` if it passes.  │
│ rollback             Point `@champion` back at `@previous_champion`.                             │
│ score                Score one month of applications with `@champion` and write a parquet file.  │
│ monitor              Monitor one production month: contract, drift and matured-label             │
│                      performance.                                                                │
│ serve-api            Serve `@champion` over HTTP with per-prediction explanations.               │
│ simulate-production  Replay the monthly cycle (validate, score, monitor, retrain) over a range   │
│                      of months.                                                                  │
│ serve-flow           Serve `monthly_cycle` on a cron (06:00 on day 1) for the previous month.    │
│ doctor               Check this machine before the session (exit 1 if a required check fails).   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

(salida recortada: quitamos el recuadro de opciones globales). Cada comando tiene su propia ayuda, por ejemplo `uv run credit-mlops simulate --help`.

## Paso 4. Genera el lago con `simulate`

Ya lo hiciste en la guía 01. Repetirlo produce exactamente los mismos archivos, porque la semilla está fija:

```bash
uv run credit-mlops simulate
```

```text
Wrote 84 months (210000 loans, 2019-01 to 2025-12) to data/raw
Incidents applied: none
```

Las opciones del comando:

```bash
uv run credit-mlops simulate --help
```

```text
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────╮
│ --loans-per-month        <int>                            Loans originated per month.            │
│                                                           [default: 2500]                        │
│ --incident               <list[income-bug|bureau-shift]>  Data incident to inject (repeatable).  │
│ --help                   <bool>                           Show this message and exit.            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

(salida recortada). `--incident` inyecta errores de datos a propósito. Usaremos `income-bug` en el paso 6.

## Paso 5. Versiona los datos con DVC

git no está hecho para archivos de datos grandes. DVC guarda los datos en una caché y en un remote, y deja en git un archivo pequeño (`data/raw.dvc`) con el hash del contenido. El remote ya está configurado en `.dvc/config` y apunta a la carpeta `.dvc-remote` de tu repositorio, que git ignora.

Registra el lago:

```bash
uv run dvc add data/raw
```

```text
To track the changes with git, run:

	git add data/raw.dvc

To enable auto staging, run:

	dvc config core.autostage true
```

DVC escribió `data/raw.dvc` y no tocó git. Mira qué ve git:

```bash
git status --short
```

```text
?? data/
```

git muestra la carpeta `data/` porque contiene un archivo nuevo, `data/raw.dvc`. La carpeta `data/raw` está en `.gitignore`. Haz commit de ese archivo:

```bash
git add data/raw.dvc
git commit -m "data: track the simulated lake with DVC"
```

```text
[main f668a0d] data: track the simulated lake with DVC
 1 file changed, 6 insertions(+)
 create mode 100644 data/raw.dvc
```

Mira su contenido:

```bash
git show HEAD:data/raw.dvc
```

```text
outs:
- md5: d6ccf41697b4c83a6db930f432fa8264.dir
  size: 8909035
  nfiles: 85
  hash: md5
  path: raw
```

El SHA del commit (`f668a0d`), el `md5` y el `size` son de la máquina del instructor y los tuyos serán distintos, porque los bytes de un archivo parquet cambian con la plataforma aunque los valores sean idénticos. `nfiles: 85` sí coincide, porque son 84 meses más `outcomes.parquet`.

Sube los datos al remote:

```bash
uv run dvc push
```

```text
86 files pushed
```

Son 86 porque DVC también sube el índice del directorio. Ahora simula que perdiste los datos. Borra el lago y la caché local de DVC.

macOS:

```bash
rm -rf data/raw .dvc/cache
```

Windows (PowerShell):

```powershell
Remove-Item -Recurse -Force data/raw, .dvc/cache
```

Y recupéralos desde el remote:

```bash
uv run dvc pull
```

```text
A       data/raw/
86 files fetched and 85 files added
```

```bash
uv run dvc status
```

```text
Data and pipelines are up to date.
```

Un compañero, o el servidor de CI, obtiene tus datos exactos de la misma forma. Clona el repositorio y ejecuta `dvc pull`.

## Paso 6. Valida los datos con el contrato

El contrato (`data/contract.py`) revisa cada mes antes de entrenar o predecir: columnas presentes, tipos, rangos, categorías permitidas, sin nulos. Se genera a partir de `schema.py`, así que no hay una segunda lista de reglas que mantener.

Un mes limpio:

```bash
uv run credit-mlops validate --month 2025-03
```

```text
2025-03: contract OK (2500 rows)
```

Ahora inyecta un bug realista. Desde marzo de 2025, el sistema de origen registra el ingreso en miles de soles (6.5 en vez de 6500).

```bash
uv run credit-mlops simulate --incident income-bug
```

```text
Wrote 84 months (210000 loans, 2019-01 to 2025-12) to data/raw
Incidents applied: income-bug
```

```bash
uv run credit-mlops validate --month 2025-02
```

```text
2025-02: contract OK (2500 rows)
```

```bash
uv run credit-mlops validate --month 2025-03
```

```text
2025-03: contract FAILED (2500 rows)
  - monthly_income_pen: values outside [500, 50000] (2500 rows)
```

El comando termina con código de salida 1, así que cualquier proceso automático que lo llame se detiene ahí. El prototipo habría entrenado o predicho con esos ingresos sin decir nada (problema 6).

DVC también se dio cuenta de que los datos cambiaron:

```bash
uv run dvc status
```

```text
data/raw.dvc:
	changed outs:
		modified:           data/raw
```

Vuelve a la versión registrada en git. Necesitas `--force` porque DVC no borra cambios sin guardar si no se lo pides (ver "Si algo falla"):

```bash
uv run dvc checkout --force
```

```text
M       data/raw/
```

```bash
uv run credit-mlops validate --month 2025-03
```

```text
2025-03: contract OK (2500 rows)
```

## Paso 7. Ventanas de entrenamiento y madurez de las etiquetas

La etiqueta `default_12m` dice si el préstamo cayó en default en sus primeros 12 meses. Un préstamo originado en enero de 2025 solo tiene etiqueta definitiva en enero de 2026. Antes de eso, un 0 significa "todavía no sabemos", y ese fue el error del prototipo (problema 2).

`data/lake.py` solo etiqueta las cohortes maduras, es decir, las que ya cumplieron su ventana de 12 meses a la fecha de corte (`as_of`).

```bash
uv run python -c "from datetime import date; from credit_mlops.data.lake import is_matured; print(is_matured(date(2024, 12, 1), as_of=date(2025, 12, 1)), is_matured(date(2025, 1, 1), as_of=date(2025, 12, 1)))"
```

```text
True False
```

`TrainingWindow` (en `settings.py`) define todas las fechas de un entrenamiento. La ventana por defecto del primer modelo:

```bash
uv run python -c "from credit_mlops.settings import DEFAULT_WINDOW; print(DEFAULT_WINDOW.as_params())"
```

```text
{'train_start': '2019-01', 'train_end': '2021-06', 'oot_start': '2021-07', 'oot_end': '2021-12', 'as_of': '2023-01', 'validation_months': '6', 'label_window_months': '12', 'half_life_months': 'None', 'validation_start': '2021-01'}
```

En una línea de tiempo:

```text
 2019-01            2020-12 2021-01       2021-06 2021-07       2021-12 2022-01         2022-12 2023-01
    |---- ajuste del HPO ---|---- validación ----|------ OOT ---------|--- espera de 12 meses ----|
    |------------------ entrenamiento ----------|                                              as_of
```

- **Entrenamiento** (2019-01 a 2021-06). El HPO ajusta cada candidato con los primeros 24 meses y lo compara en los últimos 6 (validación). Luego el ganador se reentrena con los 30 meses.
- **OOT** (fuera de tiempo, 2021-07 a 2021-12). Meses que el modelo nunca vio, posteriores al entrenamiento. Aquí se miden el AUC y la calibración que reportamos. Reemplaza al split aleatorio (problema 3).
- **as_of** (2023-01). La fecha en que "estamos". La última cohorte OOT (2021-12) cumple sus 12 meses en 2022-12, así que en 2023-01 todas las etiquetas están completas.

Los validadores de `TrainingWindow` rechazan una ventana cuyas etiquetas no han madurado. Por ejemplo, un OOT que llega hasta 2022-06 con `as_of` 2023-01:

```bash
uv run python -c "from datetime import date; from credit_mlops.settings import TrainingWindow; TrainingWindow(train_start=date(2019, 1, 1), train_end=date(2021, 6, 1), oot_start=date(2021, 7, 1), oot_end=date(2022, 6, 1), as_of=date(2023, 1, 1))"
```

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for TrainingWindow
  Value error, OOT labels are not mature at as_of [type=value_error, input_value={'train_start': datetime....tetime.date(2023, 1, 1)}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/value_error
```

(salida recortada: quitamos el traceback). El error aparece al crear la ventana, antes de leer un solo dato.

## Paso 8. El preprocesamiento dentro del `Pipeline`

En el prototipo, el `astype("category")` vivía en una celda aparte, y quien usara el modelo tenía que repetirlo a mano (problema 7). Aquí todo va en un `Pipeline` de scikit-learn (`features.py`), y lo que se guarda en MLflow es el pipeline completo:

```bash
uv run python -c "import json, pandas as pd; from credit_mlops.features import build_pipeline; apps = pd.DataFrame(json.load(open('docs/guias/ejemplo_solicitud.json'))['applications']); apps.loc[1, 'region'] = 'arequipa'; p = build_pipeline(200, 3, 0.05, 1.0, 42); print([name for name, _ in p.steps]); print(p[:-1].transform(apps)[['payment_to_income', 'loan_to_income', 'region']].to_string())"
```

```text
['features', 'categoricals', 'model']
   payment_to_income  loan_to_income region
0             0.0769          0.1538   lima
1             0.4464          1.7857    NaN
```

El comando toma las dos solicitudes de `docs/guias/ejemplo_solicitud.json`, cambia la región de la segunda por una que no existe (`arequipa`) y aplica los dos primeros pasos:

1. `features` (`FeatureBuilder`) calcula dos variables derivadas: la cuota sobre el ingreso (`payment_to_income`) y el monto sobre el ingreso anual (`loan_to_income`).
2. `categoricals` (`CategoricalCaster`) fija las categorías a la lista de `schema.py`. Una categoría desconocida se vuelve `NaN` en vez de crear una categoría nueva que el modelo nunca vio.
3. `model` es el `XGBClassifier`.

## Paso 9. Por qué no usamos `scale_pos_weight`

En la parte 1 del curso usaste `scale_pos_weight = negativos / positivos` para compensar el desbalance de clases. El peso cambia el significado de la salida, y el modelo deja de producir una probabilidad calibrada.

Nuestro modelo produce una PD que se usa como probabilidad. El umbral de decisión (0.15), el quality gate y el monitoreo la comparan con tasas de default observadas. Este comando entrena el mismo modelo dos veces, con y sin el peso, y mide la razón de calibración en el OOT (PD media predicha dividida entre la tasa de default observada; 1.0 es perfecto). Funciona igual en macOS y PowerShell:

```bash
uv run python -c "
from credit_mlops.data.lake import attach_labels, load_applications, load_outcomes
from credit_mlops.features import build_pipeline
from credit_mlops.months import month_range
from credit_mlops.settings import DEFAULT_WINDOW as w, Settings
s = Settings()
outcomes = load_outcomes(s)
def load(start, end):
    return attach_labels(load_applications(s, month_range(start, end)), outcomes, w.as_of)
train, oot = load(w.train_start, w.train_end), load(w.oot_start, w.oot_end)
y, rate = train['default_12m'], oot['default_12m'].mean()
print(f'tasa de default observada (OOT): {rate:.4f}')
for weight in [(y == 0).sum() / y.sum(), 1.0]:
    model = build_pipeline(200, 3, s.learning_rate, weight, s.seed).fit(train, y)
    pd_mean = model.predict_proba(oot)[:, 1].mean()
    print(f'scale_pos_weight={weight:.2f}  PD media={pd_mean:.4f}  ratio={pd_mean / rate:.3f}')
"
```

```text
tasa de default observada (OOT): 0.1014
scale_pos_weight=9.44  PD media=0.3890  ratio=3.837
scale_pos_weight=1.00  PD media=0.0987  ratio=0.973
```

Con el peso, el modelo predice en promedio una PD de 0.389 cuando la tasa real es 0.101, casi cuatro veces más. Con ese modelo, una PD de 0.15 ya no significa 15 de cada 100, así que el umbral de decisión pierde su sentido. Además, la regla de calibración del quality gate (guía 03) rechazaría a un challenger con razón 3.837, porque está fuera de 1 ± 0.15. Por eso `training/train.py` fija `scale_pos_weight=1.0`. Si tu problema solo necesita ordenar (ranking) y no una probabilidad, el peso puede servirte. Si la salida se usa como probabilidad, no.

## Paso 10. La grilla del HPO

La búsqueda de hiperparámetros es pequeña a propósito (la parte 2 ya cubrió Optuna). Está en `config/settings.toml`:

```bash
uv run python -c "from credit_mlops.settings import Settings; s = Settings(); print(s.hpo_n_estimators, s.hpo_max_depth, s.learning_rate)"
```

```text
[100, 200, 400] [2, 3, 4] 0.05
```

Son 9 combinaciones de `n_estimators` por `max_depth`. `training/train.py` ajusta cada una con los meses de ajuste, mide su AUC en los meses de validación (nunca en el OOT) y elige la mejor. En empate gana el modelo más simple. Cuando entrenes en la guía 03, cada combinación aparece en MLflow como una corrida hija (child run) de la corrida principal, con sus parámetros y su `validation_auc`.

## Por qué así

**pydantic-settings para la configuración.** Valida tipos al arrancar (un umbral escrito como texto falla de inmediato) y lee el TOML, el `.env` y las variables de entorno con una prioridad clara. El mismo código corre en tu laptop, en CI y en Databricks cambiando solo el entorno.

**Typer para la CLI.** Convierte funciones de Python en comandos con ayuda automática. Un comando se puede programar (cron, Prefect, CI); una celda de notebook no.

**DVC para los datos.** git guarda el hash y DVC guarda el contenido. Cada modelo registra ese hash (guía 03), así que siempre puedes recuperar los datos exactos con los que se entrenó (problema 9). En tu propio proyecto configuras el remote una vez con `uv run dvc remote add -d local .dvc-remote`; en un equipo real el remote es un bucket (S3, GCS, Azure Blob).

**Great Expectations para el contrato.** Las reglas se escriben en código a partir de `schema.py`, sin carpetas de configuración. Un fallo bloquea el paso y dice qué columna falló y en cuántas filas.

**`TrainingWindow` como objeto.** Las fechas de entrenamiento son parámetros con validadores, no filtros escritos a mano en cada script. Una ventana mal armada falla antes de entrenar, y la ventana completa queda registrada en MLflow.

**Un `Pipeline` en vez de pasos sueltos.** Lo que se entrena es lo que se sirve. La API y el scoring batch llaman al mismo objeto y no pueden olvidar un paso de preprocesamiento.

## Si algo falla

**`dvc checkout` termina con `ERROR: Can't remove the following unsaved files without confirmation. Use --force to force.`** Pasó durante la preparación de esta guía. DVC no sobrescribe archivos que cambiaron desde el último `dvc add`, para no perder trabajo. Si quieres volver a la versión registrada, usa `uv run dvc checkout --force`. Si en cambio quieres conservar los datos nuevos, regístralos con `uv run dvc add data/raw` y haz commit del `.dvc`.

**`validate` responde `missing application files for months 2026-01 in data/raw/applications`.** El lago va de 2019-01 a 2025-12, o todavía no lo generaste. Revisa el mes y, si `data/raw` no existe, ejecuta `uv run dvc pull` (o `uv run credit-mlops simulate`).

## Para el instructor

40 minutos. Tiempos en la Mac del instructor: `simulate` entre 1 y 2 s, `dvc add` hasta 2 s, `dvc push` y `dvc pull` menos de 1 s, `validate` entre 2 y 5 s, y el comando de calibración del paso 9 algo más de 1 s. El tiempo se va en explicar. Si vas tarde, muestra los pasos 1 a 3 desde tu pantalla y deja que los estudiantes ejecuten desde el paso 5 (DVC). No recortes los pasos 7 y 9, porque son la base de la guía 06.
