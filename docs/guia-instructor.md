# Guía del instructor

Agenda, preparación previa, comandos por bloque, planes B y tiempos medidos de la sesión de MLOps (unas 4 horas en una sola sesión). Los tiempos son de la Mac del instructor (Apple M5 Pro, `loans_per_month=2500`, Python 3.13); en otras máquinas variarán.

## Agenda

El contenido central cabe en 3 h 50 min, con una pausa de 10 minutos. La última columna lista lo que puedes recortar si vas tarde, empezando por el epílogo en Databricks. Todo lo recortado queda documentado en su guía, y los estudiantes pueden hacerlo después por su cuenta.

| # | Bloque | Guía | Tiempo | Recortable el día de la clase |
| --- | --- | --- | --- | --- |
| 0 | Primers y preparación, verificada con `credit-mlops doctor` | [00-preparacion.md](guias/00-preparacion.md) | antes de clase | |
| 1 | Por qué MLOps; el notebook prototipo y sus problemas | [01-por-que-mlops.md](guias/01-por-que-mlops.md) | 20 min | |
| 2 | Del notebook al paquete: mapa, settings, CLI, contrato GX, DVC, `TrainingWindow`, `Pipeline`, HPO | [02-del-notebook-al-paquete.md](guias/02-del-notebook-al-paquete.md) | 40 min | |
| 3 | MLflow: pyfunc con firma y SHAP, trazabilidad, registry, aliases, gate de promoción | [03-ciclo-de-vida-mlflow.md](guias/03-ciclo-de-vida-mlflow.md) | 40 min | model card |
| | Pausa | | 10 min | |
| 4 | Servir: batch, FastAPI con explicaciones, Docker slim | [04-sirviendo-el-modelo.md](guias/04-sirviendo-el-modelo.md) | 45 min | `mlflow models serve`, comparación de tamaños de imagen |
| 5 | CI con GitHub Actions (lint y pruebas) | [05-ci.md](guias/05-ci.md) | 20 min | jobs `quality-gate` y `docker` (mostrados como corridas ya en verde) |
| 6 | Monitoreo: PSI, madurez de etiquetas, laboratorio de incidentes, reentrenamiento, gate y cambio de alias con el flujo de Prefect | [06-monitoreo-y-reentrenamiento.md](guias/06-monitoreo-y-reentrenamiento.md) | 45 min | programación con Prefect y su interfaz |
| 7 | Epílogo en Databricks | [07-epilogo-databricks.md](guias/07-epilogo-databricks.md) | 15 min si hay tiempo | todo el bloque |
| 8 | Checklist del capstone y cierre | [08-cierre.md](guias/08-cierre.md) | 10 min | |

## Antes de la clase

- [ ] Fusiona el trabajo en `main` de `AHuaman/demo-mlops`. "Use this template" copia solo la rama por defecto.
- [ ] Haz público el repositorio (**Settings**, **General**, **Danger Zone**, **Change visibility**) o da acceso de lectura a los estudiantes. Hoy es privado. Una plantilla privada solo la ve quien tiene acceso, y lo mismo pasa con los enlaces a las corridas de CI de la guía 05 y de esta guía.
- [ ] Marca `AHuaman/demo-mlops` como repositorio plantilla en GitHub (**Settings**, **General**, casilla **Template repository**). Sin esto los estudiantes no ven el botón **Use this template** (guía 00, paso 5).
- [ ] Envía los primers (`docs/primers/`) y `docs/guias/00-preparacion.md` con tiempo para que preparen sus laptops.
- [ ] Pide a cada estudiante que ejecute `uv run credit-mlops doctor` y te mande la salida. Ninguna línea debe tener `✗`.
- [ ] Recuerda a quienes usan macOS que ejecuten `brew install libomp`. Sin libomp, XGBoost no carga y `doctor` lo marca con `✗`.
- [ ] En una máquina con Windows, verifica los bloques de PowerShell de las guías y `docker compose up` de la guía 04, que llega a MLflow por `host.docker.internal`. Los bloques de PowerShell no se ejecutaron al escribir las guías.
- [ ] Confirma los datos de Databricks Free Edition que el primer 04 marca como no confirmados: si piden tarjeta, la forma de la URL del workspace, el valor de la cuota diaria, el nombre del catálogo por defecto y las funciones de MLflow y Unity Catalog incluidas. Confirma también la URL de registro (https://www.databricks.com/learn/free-edition).
- [ ] Si instalas el CLI de Databricks con Homebrew 6, puede que antes tengas que ejecutar `brew trust --formula databricks/tap/databricks`.
- [ ] Ensaya el epílogo de Databricks un día distinto al de la clase. La cuota diaria de Free Edition puede agotarse con el ensayo.
- [ ] Opcional: toma capturas de un check de CI en verde, de la interfaz de Prefect y, en Databricks, del experimento y de la versión 1 con `@champion` (guía 07, paso 6), por si la red falla en clase.
- [ ] Problema conocido: `credit-mlops train` y `credit-mlops promote` se quedan unos 4 minutos sin responder si el servidor de MLflow no está corriendo, y luego fallan con `Connection refused`. Pide que inicien `uv run credit-mlops mlflow-server` antes de entrenar (guía 03, paso 1).

## Comandos por bloque

Todos se ejecutan desde la raíz del repositorio. Las guías tienen la salida esperada de cada uno.

**Bloque 1.**

```bash
uv run credit-mlops simulate
uv run jupyter notebook notebooks/00_prototipo.ipynb
git restore notebooks/00_prototipo.ipynb
```

**Bloque 2.**

```bash
uv run credit-mlops --help
uv run dvc add data/raw
uv run dvc push
uv run dvc pull
uv run credit-mlops validate --month 2025-03
uv run credit-mlops simulate --incident income-bug
uv run credit-mlops validate --month 2025-03
uv run dvc checkout --force
```

**Bloque 3.** El servidor de MLflow queda corriendo en una segunda terminal hasta el bloque 6.

```bash
uv run credit-mlops mlflow-server
uv run credit-mlops train
uv run credit-mlops promote
```

**Bloque 4.**

```bash
uv run credit-mlops score --month 2025-03
uv run credit-mlops serve-api
docker compose up --build
docker compose down
```

**Bloque 5.**

```bash
uv run ruff check
uv run ruff format --check
uv run pytest
uv run pre-commit install
git switch -c demo/ci-rojo
git push -u origin demo/ci-rojo
```

**Bloque 6.**

```bash
uv run credit-mlops score --month 2024-07
uv run credit-mlops monitor --month 2024-07
uv run jupyter notebook notebooks/06_laboratorio_monitoreo.ipynb
uv run credit-mlops simulate-production --start 2023-01 --end 2023-03
uv run prefect server start
uv run credit-mlops serve-flow
```

**Bloque 7.** Los comandos están en la guía 07. Los de `credit-mlops` y Python llevan `uv run --env-file .env.databricks`, que carga las variables de Databricks. Crea `.env.databricks` antes de la clase (guía 07, paso 2).

## Planes B

**Docker no funciona en una laptop.** El estudiante sigue con `uv run credit-mlops serve-api`, que es la misma API sin contenedor (guía 04).

**`/ready` del contenedor responde 503 en Windows.** Si menciona `host.docker.internal:5000` y MLflow sí corre, sigue con `uv run credit-mlops serve-api` y anota el caso.

**MLflow no arranca o el puerto 5000 está ocupado.** `uv run credit-mlops doctor` dice si el servidor responde o si el puerto está libre. Si otro programa ocupa el puerto 5000, ciérralo antes de iniciar MLflow.

**CI tarda en empezar o los runners están en cola.** Muestra las corridas ya terminadas: la corrida verde de la plantilla (https://github.com/AHuaman/demo-mlops/actions/runs/43899602459) y las corridas roja y verde del ensayo de la guía 05 (enlaces en "Tiempos medidos"). Para el ejercicio basta con el job de Ubuntu.

**El notebook del laboratorio no corre.** El notebook está guardado con sus salidas. Ábrelo en GitHub o en Jupyter sin ejecutarlo y comenta las salidas. `uv run pytest -m slow tests/integration/test_notebooks.py` comprueba que corre de punta a punta.

**Prefect falla con `Failed to reach API`.** Hay un `PREFECT_API_URL` configurado y su servidor está detenido. `uv run prefect config unset PREFECT_API_URL` vuelve al servidor temporal.

**Un estudiante dejó el lago con el bug de ingresos.** `uv run dvc checkout --force` restaura la versión registrada (guía 02). Si no llegó a versionar con DVC, `uv run credit-mlops simulate` regenera el lago limpio.

**Vas tarde.** Recorta en este orden: el epílogo en Databricks, la interfaz de Prefect y `serve-flow`, `mlflow models serve` y la comparación de imágenes, la model card. Todo queda en las guías.

## Tiempos medidos

| Comando | Tiempo |
| --- | --- |
| Corrida completa del laboratorio (`uv run pytest -m slow tests/integration/test_lab.py`: lago 2019-2025, entrenamiento, 36 meses de producción y los dos incidentes) | 29 s (32 s de reloj con el arranque de `uv`) |
| `uv run pytest -m slow` (laboratorio más `test_flows.py`) | 49 s (52 s de reloj) |
| Notebook `06_laboratorio_monitoreo.ipynb` completo, ejecutado con `jupyter nbconvert --execute` | entre 42 y 52 s, en cinco ejecuciones |
| `uv run pytest -m slow tests/integration/test_notebooks.py` (los dos notebooks) | 47 s (50 s de reloj) |
| `uv run pytest` (suite por defecto: 172 pruebas) | 27 s (31 s de reloj) |
| `uv run credit-mlops score --month 2024-07` | 4 s de reloj |
| `uv run credit-mlops monitor --month 2024-07` | entre 3 y 6 s de reloj |
| `uv run credit-mlops simulate-production` de 3 meses, con el servidor temporal de Prefect | 12 s de reloj |
| Lo mismo, con `prefect server start` ya corriendo | 6 s de reloj |
| `uv run prefect server start` hasta responder | unos 3 s |
| CI en rojo de la guía 05 (corrida https://github.com/AHuaman/demo-mlops/actions/runs/43952332250) | `test (ubuntu-latest)` 1 min 20 s, `test (windows-latest)` 7 min 4 s |
| CI en verde de la guía 05 (corrida https://github.com/AHuaman/demo-mlops/actions/runs/43953724900) | `test (ubuntu-latest)` 1 min 34 s, `test (windows-latest)` 5 min 27 s, `quality-gate` 1 min 8 s, `docker` 1 min 45 s |
| Cada corrida de CI de la guía 05, de su creación al último job | unos 7 min (7 min 10 s la roja, 7 min 20 s la verde) |
| `uv run --env-file .env.databricks credit-mlops train` contra Databricks Free Edition (guía 07) | 3 min 40 s de reloj |
| `uv run --env-file .env.databricks credit-mlops promote` contra Databricks Free Edition (guía 07) | 41 s de reloj |

Los tiempos de los bloques 1 a 4 están en la sección "Para el instructor" de cada guía.
