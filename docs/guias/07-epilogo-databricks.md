# Epílogo en Databricks

**Objetivo.** Correr los mismos `credit-mlops train` y `credit-mlops promote` de la guía 03, pero con el tracking y el registry en Databricks Free Edition. La corrida queda en un experimento del workspace y el modelo en Unity Catalog con el alias `@champion`. Solo cambian variables de entorno.
Al final sabes qué hace falta para mover el proyecto a una plataforma administrada y qué te daría el serving administrado, que aquí no ejecutamos.

**Problema del prototipo que trata.** Todo vivía en una laptop. El servidor de MLflow, el registry y la API los operabas tú. Esta guía muestra que el código ya está listo para que otro los opere.

Es una demo del instructor de unos 15 minutos. Si quieres repetirla por tu cuenta, necesitas la cuenta de Free Edition, el CLI de Databricks y un profile autenticado, como explica el primer [Nube y Databricks](../primers/04-nube-y-databricks.md). También necesitas el lago en `data/raw` de la [guía 02](02-del-notebook-al-paquete.md). Las salidas salen de la Mac del instructor. Cambiamos su correo por `<tu-correo>` y la dirección de su workspace por `<tu-workspace>`. En las tuyas cambian también los ids de experimento y de corrida.

## Paso 1. Qué cambia y qué no

El código no cambia. `Settings` lee cuatro variables y el SDK de Databricks lee una más:

| Variable | En tu laptop (guía 03) | En Databricks |
| --- | --- | --- |
| `CREDIT_MLOPS_MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | `databricks`, el MLflow administrado del workspace |
| `CREDIT_MLOPS_MLFLOW_REGISTRY_URI` | sin valor (el mismo servidor) | `databricks-uc`, el registry de Unity Catalog |
| `CREDIT_MLOPS_REGISTERED_MODEL_NAME` | `credit_default_pd` | `workspace.default.credit_default_pd` |
| `CREDIT_MLOPS_TRAINING_EXPERIMENT` | `credit-default-training` | `/Users/<tu-correo>/credit-default-training` |
| `DATABRICKS_CONFIG_PROFILE` | no se usa | el profile que creaste en el primer 04 |

Unity Catalog nombra los modelos con tres niveles, `catalog.schema.model`. `Settings` rechaza un nombre de dos niveles cuando el registry es `databricks-uc`, antes de conectarse. En Databricks un experimento es una ruta del workspace, por eso lleva tu carpeta de usuario.

El entrenamiento sigue corriendo en tu laptop. Databricks recibe los parámetros, las métricas, los artefactos y el modelo. La demo no crea clusters, jobs ni endpoints.

## Paso 2. El archivo de entorno

El repositorio trae `.env.databricks.example`:

```text
# Copia este archivo a .env.databricks y reemplaza <tu-correo> y <tu-perfil> (guía 07).
CREDIT_MLOPS_MLFLOW_TRACKING_URI=databricks
CREDIT_MLOPS_MLFLOW_REGISTRY_URI=databricks-uc
CREDIT_MLOPS_REGISTERED_MODEL_NAME=workspace.default.credit_default_pd
CREDIT_MLOPS_TRAINING_EXPERIMENT=/Users/<tu-correo>/credit-default-training
DATABRICKS_CONFIG_PROFILE=<tu-perfil>
# La telemetría de MLflow pide un token en paralelo y el CLI de Databricks falla
# al refrescarlo dos veces a la vez (exit status 45 en macOS). Apagarla lo evita.
MLFLOW_DISABLE_TELEMETRY=true
```

La última línea apaga la telemetría de MLflow. Sin ella, el entrenamiento falló durante la preparación de esta guía (ver "Si algo falla").

Cópialo (este comando funciona igual en macOS y en PowerShell):

```bash
cp .env.databricks.example .env.databricks
```

`.env.databricks` está en `.gitignore`, porque lleva tu correo y el nombre de tu profile. Para saber qué correo usa el workspace, pregúntale al CLI:

```bash
databricks current-user me --profile <tu-perfil>
```

```text
...
  "userName": "<tu-correo>"
}
```

(salida recortada). Abre `.env.databricks` en tu editor y reemplaza `<tu-correo>` por ese `userName` y `<tu-perfil>` por el nombre de tu profile, por ejemplo `free`.

`Settings` solo lee `.env` por su cuenta. Para usar otro archivo, `uv run --env-file` carga cada línea como variable de entorno del comando. Así le llegan a `Settings` las variables `CREDIT_MLOPS_*`, al SDK de Databricks las de autenticación y a MLflow `MLFLOW_DISABLE_TELEMETRY`. La opción funciona igual en macOS y en PowerShell. Comprueba qué ve `Settings`:

```bash
uv run --env-file .env.databricks python -c "from credit_mlops.settings import Settings; s = Settings(); print(s.mlflow_tracking_uri, s.mlflow_registry_uri, s.registered_model_name)"
```

```text
databricks databricks-uc workspace.default.credit_default_pd
```

Sin `--env-file`, el mismo comando imprime los valores locales de la guía 03.

## Paso 3. Entrena

```bash
uv run --env-file .env.databricks credit-mlops train
```

```text
UserWarning: Hint: Inferred schema contains integer column(s). Integer columns in Python cannot represent missing values. ...
WARNING:databricks.sdk.mixins.files:Unexpected status resolving file endpoints (403, request ID fa9bb730-93cb-4779-a673-dfd9a6506813), using the default endpoints.
🏃 View run hpo-n100-d2 at: https://<tu-workspace>.cloud.databricks.com/ml/experiments/2375199926349171/runs/7a285add96d34fdca6ff1cc48749b40f
🧪 View experiment at: https://<tu-workspace>.cloud.databricks.com/ml/experiments/2375199926349171
...
🏃 View run hpo-n400-d4 at: https://<tu-workspace>.cloud.databricks.com/ml/experiments/2375199926349171/runs/bf835975cb4546ad98a8855da9af9e5c
...
Successfully registered model 'workspace.default.credit_default_pd'.
Uploading artifacts: 100%|██████████| 46/46 [00:16<00:00,  2.75it/s]
Created version '1' of model 'workspace.default.credit_default_pd'.
🏃 View run train-2023-01 at: https://<tu-workspace>.cloud.databricks.com/ml/experiments/2375199926349171/runs/54c0d32717f341e2940ecb76590a3a4e
🧪 View experiment at: https://<tu-workspace>.cloud.databricks.com/ml/experiments/2375199926349171
{"name":"workspace.default.credit_default_pd","version":"1","run_id":"54c0d32717f341e2940ecb76590a3a4e"}
```

(salida recortada). Es la misma secuencia de la guía 03: 9 corridas hijas del HPO, la corrida padre `train-2023-01` y la versión 1 con el alias `@challenger`. Los avisos `Unexpected status resolving file endpoints (403 ...)` y `Connection pool is full` son del SDK de Databricks al subir artefactos. La subida terminó bien.

En la Mac del instructor tardó 3 min 40 s, contra unos 5 s con el servidor local. El entrenamiento local sigue tomando segundos. El resto del tiempo se va en la red: cada llamada al workspace le pide al CLI un token nuevo (`databricks auth token --force-refresh`), y una corrida hace decenas de llamadas.

Comprueba que el modelo da las mismas métricas que en la guía 03:

```bash
uv run --env-file .env.databricks python -c "from mlflow import MlflowClient; c = MlflowClient('databricks', 'databricks-uc'); r = c.get_run('54c0d32717f341e2940ecb76590a3a4e'); print(round(r.data.metrics['oot_roc_auc'], 6), round(r.data.metrics['oot_calibration_ratio'], 6))"
```

Cambia el id por el `run_id` de tu salida.

```text
0.804272 0.973481
```

Son los mismos valores que imprime la guía 03 en el paso 9. Los datos, la semilla y el código son los mismos. Solo cambió dónde quedan guardados.

## Paso 4. Promueve

```bash
uv run --env-file .env.databricks credit-mlops promote
```

```text
PROMOTED: challenger v1 is now champion (previous: none)
  - no champion yet
```

(salida recortada: quitamos la barra de progreso `Downloading artifacts` y los mismos avisos del SDK del paso 3). Tardó 41 s. Igual que en la guía 03, no había champion y el gate aprueba sin comparar. El gate descargó el modelo desde Unity Catalog, lo evaluó en tu laptop sobre los meses OOT y movió el alias en Unity Catalog.

## Paso 5. Verifica desde la terminal

El CLI de Databricks lee Unity Catalog sin pasar por nuestro código:

```bash
databricks registered-models get workspace.default.credit_default_pd --include-aliases --profile <tu-perfil>
```

```text
{
  "aliases": [
    {
      "alias_name": "challenger",
      "id": "f3ed9753-4c39-4d90-a3e9-7be44fa36853",
      "version_num": 1
    },
    {
      "alias_name": "champion",
      "id": "e8bfabe3-42c9-4c61-9de2-d59ef0b017e9",
      "version_num": 1
    }
  ],
  "catalog_name": "workspace",
  ...
  "full_name": "workspace.default.credit_default_pd",
  ...
  "owner": "<tu-correo>",
  "schema_name": "default",
  ...
}
```

(salida recortada). Sin `--include-aliases`, la respuesta no trae los aliases.

Desde MLflow, la misma consulta que usa la API de la guía 04 para cargar el champion:

```bash
uv run --env-file .env.databricks python -c "from mlflow import MlflowClient; mv = MlflowClient('databricks', 'databricks-uc').get_model_version_by_alias('workspace.default.credit_default_pd', 'champion'); print(mv.version, mv.aliases, mv.run_id)"
```

```text
1 ['challenger', 'champion'] 54c0d32717f341e2940ecb76590a3a4e
```

## Paso 6. Dónde verlo en la interfaz

Abre tu workspace en el navegador. Los nombres de los menús pueden cambiar con las versiones de Databricks.

1. En la barra lateral, entra a **Experiments** y abre `credit-default-training`. También puedes abrir el enlace `View experiment` de la salida del paso 3. Ves `train-2023-01` con sus 9 corridas hijas, como en la guía 03.
2. Abre `train-2023-01`. Revisa los parámetros de la ventana, las métricas `oot_*`, los tags (`git_sha`, `data_version`, `gate`) y los artefactos `model_card.md`, `gate_decision.json` y `reference/`.
3. En la barra lateral, entra a **Catalog**, luego a `workspace`, a `default` y a la sección de modelos. Abre `credit_default_pd`. La versión 1 muestra los aliases `@champion` y `@challenger`, y enlaza a la corrida que la creó.

## Qué agregaría el serving administrado

No lo ejecutamos. Con Model Serving, Databricks publicaría `workspace.default.credit_default_pd@champion` como un endpoint REST. Ese endpoint reemplazaría la API con FastAPI y la imagen de Docker de la guía 04:

- Databricks construye el entorno del modelo. MLflow ya guardó lo necesario. Durante el `train` copió `uv.lock`, `pyproject.toml` y `.python-version` dentro del modelo y exportó 64 dependencias. El código del paquete viaja en `code_paths`.
- La plataforma opera el servidor, su escalado y sus credenciales. Tú ya no mantienes el contenedor.
- Las solicitudes y respuestas pueden quedar registradas en tablas para el monitoreo de la guía 06.
- Puedes repartir el tráfico entre dos versiones del modelo para comparar un challenger en producción.

Lo que no cambia es el ciclo. El gate decide el `@champion` y el endpoint sirve lo que diga el alias. Free Edition limita los endpoints de serving (sin GPU, entre otros límites), así que revisa la documentación antes de crear uno.

## Cuotas de Free Edition y el día de clase

- Free Edition tiene una cuota diaria de compute (primer 04). Esta demo no enciende compute, pero un ensayo largo con notebooks o endpoints sí puede gastarla.
- No ensayes el mismo día de la clase. El ensayo puede gastar la cuota o vencer el login justo antes de la demo.
- Cada `train` crea una versión nueva en Unity Catalog. Después del ensayo, tu workspace ya tiene la versión 1 como `@champion`. En clase puedes mostrarla tal cual (pasos 5 y 6). Si vuelves a entrenar, verás la versión 2 y un gate que la compara contra la 1, no la salida de esta guía.
- El login del CLI vence. Antes de la clase, ejecuta `databricks auth profiles` y confirma que tu profile dice `YES`. Si dice `NO`, repite `databricks auth login --profile <tu-perfil>`.

## Por qué así

**Solo variables de entorno.** El código recibe `Settings` y no sabe dónde corre. Mover el proyecto es cambiar valores de configuración, igual que el umbral de decisión en la guía 02.

**Un archivo aparte.** `.env.databricks` no reemplaza a `.env`. Sin `--env-file`, los comandos siguen usando tu servidor local, y no puedes registrar en Databricks por accidente.

**Una corrección pequeña en el código.** El código del curso ya la incluye. Unity Catalog avisa de un alias que no existe con otro texto que el servidor local de MLflow (ver "Si algo falla"). `alias_version` en `registry/loading.py` reconoce los dos textos. El flujo local no cambió.

## Si algo falla

**`train` muestra `WARNING mlflow.utils.databricks_utils: Failed to create databricks SDK workspace client, error: ValueError('default auth: databricks-cli: cannot get access token: Error: forced token refresh: cache update: exit status 45. ...` y termina con `RestException: 401: Credential was not sent or was of an unsupported type for this API.`** Pasó durante la preparación de esta guía, en macOS, con un `.env.databricks` sin la línea `MLFLOW_DISABLE_TELEMETRY=true`. La telemetría de MLflow pedía un token en otro hilo al mismo tiempo que el entrenamiento. En la Mac del instructor, el CLI guarda los tokens en el llavero de macOS, y falla con ese `exit status 45` cuando dos procesos lo refrescan a la vez. MLflow sigue sin credenciales y el workspace responde 401. Agrega la línea y repite. La corrida fallida queda en el experimento con estado `FAILED`; no hace falta borrarla.

**`promote` responde `RESOURCE_DOES_NOT_EXIST: Registered Model Alias 'champion' does not exist.`** Si ves este error, tu copia del repositorio es anterior a la corrección de `registry/loading.py` (ver "Por qué así"). Actualízala y repite.

## Para el instructor

15 minutos, solo si hay tiempo. En la Mac del instructor, `train` tardó 3 min 40 s y `promote` 41 s. Lanza `train` al empezar el bloque y explica la tabla del paso 1 mientras corre. Después muestra la corrida y el modelo en la interfaz (paso 6) y cierra con el serving administrado. El experimento del instructor guarda también las corridas fallidas de la preparación (dos `train-2023-01` con estado `FAILED` y una corrida hija que quedó en `RUNNING`). Si alguien pregunta, son los errores de "Si algo falla".
