# Sirviendo el modelo

**Objetivo.** Usar el modelo `@champion` de tres formas: scoring batch de un mes, una API HTTP con explicación por predicción y la misma API dentro de un contenedor Docker.
Al final una aplicación cualquiera puede pedirle una PD al modelo y recibir también las razones de la decisión.

**Problema del prototipo que resuelve.** 8 (imposible de programar o servir). También cierra el 6 del lado de la API. Cada solicitud se valida antes de llegar al modelo.

Necesitas haber terminado la [guía 03](03-ciclo-de-vida-mlflow.md), con el servidor de MLflow corriendo en la segunda terminal y la versión 1 como `@champion`. Si nunca usaste una API o Docker, lee antes los primers [API y HTTP](../primers/01-api.md) y [Docker](../primers/02-docker.md). Las salidas salen de la Mac del instructor.

## Paso 1. Scoring batch de un mes

Muchas veces basta con puntuar todas las solicitudes de un mes de una vez.

```bash
uv run credit-mlops score --month 2025-03
```

```text
Wrote data/predictions/batch/2025-03.parquet
  approve: 2173
  review: 327
```

(salida recortada: quitamos la barra de progreso `Downloading artifacts`). En Windows la ruta se imprime con `\`. El comando valida el mes con el contrato de la guía 02, carga `models:/credit_default_pd@champion` y escribe una fila por préstamo. Con el umbral de 0.15, 327 de las 2500 solicitudes van a revisión.

Mira lo que escribió:

```bash
uv run python -c "import pandas as pd; df = pd.read_parquet('data/predictions/batch/2025-03.parquet'); print(df.columns.tolist()); print(df[['loan_id', 'model_version', 'probability_default', 'decision']].head(3).to_string(index=False))"
```

```text
['loan_id', 'origination_month', 'model_version', 'scored_at', 'probability_default', 'decision', 'base_log_odds', 'top_factors']
     loan_id model_version  probability_default decision
202503-00000             1             0.033844  approve
202503-00001             1             0.032362  approve
202503-00002             1             0.031745  approve
```

Cada fila guarda la versión del modelo que la produjo y la hora. Cuando en la guía 06 lleguen los defaults reales de estos préstamos, sabremos qué modelo predijo qué.

## Paso 2. Servir sin escribir código con `mlflow models serve`

MLflow puede servir cualquier modelo registrado. Abre una **tercera terminal** en la raíz del repositorio. Primero dile a MLflow dónde está el servidor; esta línea sí cambia según el sistema.

macOS:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Windows (PowerShell):

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
```

Luego, en la misma terminal:

```bash
uv run mlflow models serve -m models:/credit_default_pd@champion -p 5001 --env-manager local
```

```text
INFO mlflow.pyfunc.backend: === Running command 'exec uvicorn --host 127.0.0.1 --port 5001 --workers 1 mlflow.pyfunc.scoring_server.app:app'
INFO:     Started server process [12732]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5001 (Press CTRL+C to quit)
```

(salida recortada). `--env-manager local` usa tu `.venv` en vez de crear un entorno nuevo. En Windows la línea `Running command` es distinta; lo importante es la última.

Desde la primera terminal, envía las dos solicitudes de ejemplo. MLflow espera el formato `{"dataframe_records": [...]}`, así que este comando lee `docs/guias/ejemplo_solicitud.json` y lo adapta (funciona igual en macOS y PowerShell):

```bash
uv run python -c "import json, urllib.request as u; apps = json.load(open('docs/guias/ejemplo_solicitud.json'))['applications']; body = json.dumps({'dataframe_records': apps}).encode(); req = u.Request('http://127.0.0.1:5001/invocations', data=body, headers={'Content-Type': 'application/json'}); print(u.urlopen(req).read().decode())"
```

```text
{"predictions": [{"probability_default": 0.011541670188307762, "decision": "approve", "base_log_odds": -2.2595245838165283, "top_factors": "[{\"feature\": \"bureau_score\", \"value\": 780.0, \"contribution\": -1.4395185708999634}, {\"feature\": \"debt_to_income\", \"value\": 0.18, \"contribution\": -0.23016135394573212}, {\"feature\": \"max_days_past_due_12m\", \"value\": 0.0, \"contribution\": -0.15825925767421722}]"}, {"probability_default": 0.7593980431556702, "decision": "review", "base_log_odds": -2.2595245838165283, "top_factors": "[{\"feature\": \"bureau_score\", \"value\": 510.0, \"contribution\": 1.6833338737487793}, {\"feature\": \"max_days_past_due_12m\", \"value\": 75.0, \"contribution\": 0.4425206184387207}, {\"feature\": \"months_at_job\", \"value\": 4.0, \"contribution\": 0.3897905945777893}]"}]}
```

La decisión y la explicación llegan sin que escribas una línea de código de servidor, porque viven dentro del modelo pyfunc. Le faltan cosas que un servicio real necesita: `top_factors` llega como texto JSON y no como objetos, no hay un endpoint que diga qué versión está cargada y no queda registro de las predicciones. Para eso escribimos nuestra API.

Detén este servidor con `Ctrl+C` en la tercera terminal.

## Paso 3. La API con FastAPI

En la tercera terminal:

```bash
uv run credit-mlops serve-api
```

```text
INFO:     Started server process [12814]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

(salida recortada: quitamos la barra de progreso `Downloading artifacts`). Al arrancar, la API carga el `@champion` desde MLflow una sola vez. Abre http://localhost:8000/docs en el navegador. Es la documentación interactiva que FastAPI genera a partir de `api/schemas.py`. Verás los endpoints `/health`, `/ready`, `/model` y `/predict`.

Desde la primera terminal (en Windows escribe `curl.exe` en vez de `curl`):

```bash
curl http://localhost:8000/ready
```

```json
{"status":"ready","model_version":"1"}
```

```bash
curl http://localhost:8000/model
```

```json
{"model_name":"credit_default_pd","alias":"champion","model_version":"1","training_window":{"train_start":"2019-01","train_end":"2021-06","oot_start":"2021-07","oot_end":"2021-12","as_of":"2023-01"}}
```

## Paso 4. Una predicción con explicación

macOS:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @docs/guias/ejemplo_solicitud.json
```

Windows (PowerShell):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/guias/ejemplo_solicitud.json"
```

La respuesta llega en una sola línea; aquí la mostramos ordenada:

```json
{"model_name":"credit_default_pd","model_version":"1","threshold":0.15,
 "predictions":[
  {"probability_default":0.011541670188307762,"decision":"approve",
   "explanation":{"base_log_odds":-2.2595245838165283,
    "top_factors":[
     {"feature":"bureau_score","value":780.0,"contribution":-1.4395185708999634},
     {"feature":"debt_to_income","value":0.18,"contribution":-0.23016135394573212},
     {"feature":"max_days_past_due_12m","value":0.0,"contribution":-0.15825925767421722}]}},
  {"probability_default":0.7593980431556702,"decision":"review",
   "explanation":{"base_log_odds":-2.2595245838165283,
    "top_factors":[
     {"feature":"bureau_score","value":510.0,"contribution":1.6833338737487793},
     {"feature":"max_days_past_due_12m","value":75.0,"contribution":0.4425206184387207},
     {"feature":"months_at_job","value":4.0,"contribution":0.3897905945777893}]}}]}
```

Son las mismas probabilidades que devolvió `mlflow models serve` en el paso 2, porque es el mismo modelo. Cada respuesta dice qué versión la produjo, y la API agrega cada predicción a `data/predictions/api/predictions.jsonl` con un id de solicitud y la hora.

### Cómo leer la explicación

Las contribuciones son valores SHAP exactos (TreeSHAP) en **log-odds**, la escala interna de XGBoost antes de convertirla en probabilidad:

- `base_log_odds` (-2.26) es el punto de partida, igual para todos los solicitantes. Es el log-odds promedio del modelo sobre los datos de entrenamiento, antes de mirar a la persona.
- Cada `contribution` suma o resta a ese punto de partida. Positiva sube el riesgo, negativa lo baja.
- La suma del punto de partida y las contribuciones de **todas** las variables da el log-odds de la PD. La API muestra solo las 3 de mayor valor absoluto (`explanation_top_k` en `config/settings.toml`).

El segundo solicitante sale a revisión con PD 0.76. Sus tres factores principales son todos positivos: un score de buró de 510 (+1.68), 75 días de atraso máximo en el último año (+0.44) y solo 4 meses en su trabajo (+0.39). En crédito estas son las **razones de la decisión** (reason codes) que el analista o el cliente pueden leer: "score de buró bajo, atrasos recientes, poca antigüedad laboral". Para el primer solicitante las tres contribuciones son negativas, y su score de 780 es lo que más baja su riesgo.

### Una solicitud inválida

En http://localhost:8000/docs abre `POST /predict`, pulsa **Try it out**, pega el contenido de `docs/guias/ejemplo_solicitud.json` en el cuerpo y cambia el primer `"monthly_income_pen": 6500.0` por `6.5` (el bug de unidades de la guía 02). Pulsa **Execute**. La respuesta es un 422:

```json
{"detail":[{"type":"greater_than_equal","loc":["body","applications",0,"monthly_income_pen"],"msg":"Input should be greater than or equal to 500","input":6.5,"ctx":{"ge":500.0}}]}
```

La solicitud no llega al modelo. Los límites salen de las mismas reglas que el contrato batch (`schema.py`), y una prueba (`tests/api/test_schema_sync.py`) verifica que no se separen.

## Paso 5. `/health` frente a `/ready`

Los dos endpoints responden preguntas distintas:

- `/health` (liveness): ¿el proceso está vivo? Responde 200 mientras el servidor corra.
- `/ready` (readiness): ¿puede predecir? Responde 200 solo si el modelo cargó.

Compruébalo. Detén la API (`Ctrl+C` en la tercera terminal) y el servidor de MLflow (`Ctrl+C` en la segunda). Vuelve a iniciar solo la API, en la tercera terminal:

```bash
uv run credit-mlops serve-api
```

Desde la primera terminal (`curl.exe` en Windows):

```bash
curl -i http://localhost:8000/health
```

```text
HTTP/1.1 200 OK
...
{"status":"ok"}
```

```bash
curl -i http://localhost:8000/ready
```

```text
HTTP/1.1 503 Service Unavailable
...
{"detail":"MLflow server at http://127.0.0.1:5000 is unreachable (<urlopen error [Errno 61] Connection refused>); start it with: credit-mlops mlflow-server"}
```

(salida recortada: quitamos las cabeceras `date`, `server`, `content-length` y `content-type`). En Windows el mensaje de error dice `WinError 10061` en vez de `Errno 61`.

El proceso está vivo pero no puede atender. Un orquestador (Docker, Kubernetes, un balanceador) usa `/health` para decidir si reinicia el contenedor y `/ready` para decidir si le manda tráfico. Si los mezclas, un servidor sin modelo recibe tráfico, o uno que solo espera a MLflow se reinicia sin parar.

Vuelve a iniciar MLflow en la segunda terminal (`uv run credit-mlops mlflow-server`) y pregunta otra vez:

```bash
curl http://localhost:8000/ready
```

```json
{"detail":"MLflow server at http://127.0.0.1:5000 is unreachable (<urlopen error [Errno 61] Connection refused>); start it with: credit-mlops mlflow-server"}
```

Sigue en 503 aunque MLflow ya responde, porque la API intenta cargar el modelo solo al arrancar y guarda el error de ese momento. Reinicia la API (`Ctrl+C` y `uv run credit-mlops serve-api`) y repite el comando:

```json
{"status":"ready","model_version":"1"}
```

Antes del paso 6, detén la API con `Ctrl+C`. El contenedor usará el mismo puerto 8000.

## Paso 6. La API en Docker

`Dockerfile` construye la imagen en dos etapas (multi-stage). La primera instala las dependencias con uv y la segunda copia solo el entorno listo sobre `python:3.13-slim`. `compose.yaml` la ejecuta con el puerto 8000 publicado y la apunta al MLflow de tu máquina (`host.docker.internal:5000`).

Con MLflow corriendo en la segunda terminal, en la tercera:

```bash
docker compose up --build
```

```text
...
 Image credit-mlops-api Built
 Container demo-mlops-api-1 Created
Attaching to api-1
api-1  | INFO:     Started server process [1]
api-1  | INFO:     Waiting for application startup.
api-1  | INFO:     Application startup complete.
api-1  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

(salida recortada: la construcción imprime muchas líneas `#n`). El prefijo del contenedor (`demo-mlops`) es el nombre de la carpeta de tu repositorio. La primera vez Docker descarga la imagen base y cerca de 1 GB de dependencias; las siguientes reutiliza la caché.

Desde la primera terminal, la misma solicitud de antes:

macOS:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @docs/guias/ejemplo_solicitud.json
```

Windows (PowerShell):

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/guias/ejemplo_solicitud.json"
```

```json
{"model_name":"credit_default_pd","model_version":"1","threshold":0.15,"predictions":[{"probability_default":0.011541670188307762,"decision":"approve", ... },{"probability_default":0.7593980431556702,"decision":"review", ... }]}
```

(salida recortada). Las mismas probabilidades, hasta el último decimal, ahora desde un contenedor Linux.

El contenedor escribe su registro de predicciones en la misma carpeta que la API local, porque `compose.yaml` monta `./data/predictions` como volumen:

```bash
uv run python -c "lines = open('data/predictions/api/predictions.jsonl').read().splitlines(); print(len(lines), 'predicciones registradas')"
```

```text
4 predicciones registradas
```

Dos de la API local (paso 4) y dos del contenedor.

**Windows.** Con Docker Desktop, `compose.yaml` usa `host.docker.internal` para llegar al MLflow de tu máquina, y los comandos de este paso deberían funcionar igual. Si `/ready` del contenedor responde 503 con un mensaje que menciona `host.docker.internal:5000` mientras MLflow sí corre en la segunda terminal, avísale al instructor y sigue con `uv run credit-mlops serve-api`.

**Linux nativo.** Esta guía se probó en macOS con Docker Desktop. Con Docker nativo en Linux hay dos diferencias:

1. Si `data/predictions` no existe, Docker la crea como `root` y el contenedor (usuario `app`, uid 1000) no puede escribir en ella. Créala antes con tu usuario (`mkdir -p data/predictions`) y comprueba que tu uid sea 1000, o cambia el bind mount por un volumen con nombre.
2. `host.docker.internal` apunta a la IP del bridge de Docker, y un MLflow escuchando en `127.0.0.1` rechaza esa conexión. Tendrías que iniciar MLflow en `0.0.0.0`, lo que lo expone a toda tu red sin autenticación. Hazlo solo en una red de confianza.

## Paso 7. El tamaño de la imagen

```bash
docker images credit-mlops-api
```

```text
IMAGE                     ID             DISK USAGE   CONTENT SIZE   EXTRA
credit-mlops-api:latest   3f7351f368f7       1.07GB             0B   U
```

El ID depende de tu máquina. Para comparar, `docker/Dockerfile.naive` construye la misma API de la forma ingenua, con la imagen `python:3.13` completa, una sola etapa y todas las dependencias (incluidas las de desarrollo). Construirla es opcional. Ocupa 2.5 GB más en tu disco y tardó casi 3 minutos en la Mac del instructor.

```bash
docker build -f docker/Dockerfile.naive -t credit-mlops-naive .
```

```bash
docker images credit-mlops-naive
```

```text
IMAGE                       ID             DISK USAGE   CONTENT SIZE   EXTRA
credit-mlops-naive:latest   53dd3a683473        2.5GB             0B
```

| Imagen | Archivo | Tamaño |
| --- | --- | --- |
| `python:3.13-slim`, multi-stage, solo dependencias de la API | `Dockerfile` | 1.07 GB |
| `python:3.13`, una etapa, todas las dependencias | `docker/Dockerfile.naive` | 2.5 GB |

La imagen slim pesa menos de la mitad, pero no es pequeña. Casi todo su tamaño es la pila de ML (pandas, pyarrow, scipy, scikit-learn, XGBoost). Borra la imagen ingenua:

```bash
docker rmi credit-mlops-naive
```

Para terminar, detén el contenedor con `Ctrl+C` en la tercera terminal y elimínalo:

```bash
docker compose down
```

```text
 Container demo-mlops-api-1 Stopping
 Container demo-mlops-api-1 Stopped
 Container demo-mlops-api-1 Removing
 Container demo-mlops-api-1 Removed
 Network demo-mlops_default Removing
 Network demo-mlops_default Removed
```

## Por qué así

**Scoring batch.** Muchas decisiones de crédito no necesitan respuesta en milisegundos. Una campaña o una revisión de cartera puntúa miles de filas de una vez. Es la forma más simple de servir un modelo y la primera que deberías considerar.

**`mlflow models serve`.** Sirve para probar un modelo en minutos. Lo mostramos porque deja claro que la explicación vive dentro del modelo, no en la API.

**FastAPI.** Genera la validación de entrada y la documentación (`/docs`) a partir de clases pydantic. Las reglas de `api/schemas.py` repiten las de `schema.py`, y una prueba las mantiene sincronizadas.

**Explicación por predicción.** En crédito, quien recibe una decisión tiene que saber por qué. Las contribuciones TreeSHAP son exactas para árboles y se suman al log-odds del modelo, así que la explicación es consistente con la probabilidad.

**Docker slim y multi-stage.** La imagen final no lleva compiladores, uv, pruebas ni notebooks. Usamos `python:3.13-slim` y no Alpine porque scikit-learn y XGBoost no publican wheels para musl (ver el primer de Docker). La imagen corre con un usuario sin privilegios y declara un `HEALTHCHECK` contra `/health`.

## Si algo falla

**`/ready` responde 503 con `MLflow server at http://127.0.0.1:5000 is unreachable`.** El servidor de MLflow no corría cuando arrancó la API. Inícialo (`uv run credit-mlops mlflow-server`) y reinicia la API, porque el modelo se carga solo al arrancar. Si el mensaje dice `host.docker.internal:5000`, quien responde es el contenedor y la solución es la misma. Inicia MLflow, detén el contenedor con `Ctrl+C` y vuelve a ejecutar `docker compose up`.

**Las respuestas a veces vienen de la API local y a veces del contenedor.** Pasó durante la preparación de esta guía. Dejamos `serve-api` corriendo y levantamos el contenedor. En macOS los dos procesos escuchan en el puerto 8000 a la vez (uno en `127.0.0.1`, el otro en todas las interfaces), y `localhost` puede llegar a cualquiera de los dos. Detén `serve-api` antes de `docker compose up`. Si `/ready` devuelve un error, el del contenedor menciona `host.docker.internal`.

## Para el instructor

45 minutos. Tiempos en la Mac del instructor: `score` entre 3 y 4 s, `mlflow models serve` listo en 3 a 5 s, `serve-api` listo en 1 a 3 s. `docker compose up --build` con la caché de Docker ya llena respondió `/health` en unos 9 s; una construcción sin caché de capas (pero con las imágenes base y la caché de uv ya descargadas) tardó 22 s. En una laptop que construye por primera vez cuenta varios minutos de descarga, así que pide que ejecuten `docker compose build` al empezar el bloque, antes del paso 1. Solo construye la imagen, no ocupa el puerto 8000, y en el paso 6 `docker compose up --build` reutiliza esa caché. Si vas tarde, recorta `mlflow models serve` (paso 2) y la comparación de tamaños (paso 7). Si Docker falla en una laptop, ese estudiante sigue con `uv run credit-mlops serve-api`, que es la misma API.
