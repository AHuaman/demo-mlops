# Checklist del capstone

Tu capstone debe cumplir los ítems **obligatorios**. Están escritos para cualquier problema supervisado, no solo para riesgo de crédito. Algunos empiezan con "Si..." porque dependen de tu problema. Si la condición no aplica, escribe en una línea por qué. Los ítems **opcionales** los vimos en la sesión y suman, pero no son requisito.

Para cada ítem, la columna "Cómo lo muestras" dice qué evidencia esperamos. La columna "En este repositorio" te dice dónde mirar un ejemplo que funciona. Las rutas de código son relativas a `src/credit_mlops/`.

## Código

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| El código vive en un paquete dentro de `src/`, no en notebooks | El árbol del repositorio | `src/credit_mlops/` |
| `uv sync` reproduce el entorno desde cero | `pyproject.toml` y `uv.lock` versionados en git | `pyproject.toml`, `uv.lock` |
| La configuración (rutas, semilla, umbrales, grillas) vive en un archivo y se lee con pydantic-settings; nada de números sueltos en el código | Un archivo de configuración y una clase `Settings` | `config/settings.toml`, `settings.py` |
| Cada paso (preparar datos, validar, entrenar, promover, puntuar, monitorear) es un comando de la CLI | La salida de `--help` | `cli.py` |
| Hay pruebas automáticas para la lógica que importa (contrato, ventanas, gate, métricas) | `uv run pytest` en verde | `tests/` |

Opcional: `ruff` y pre-commit (`.pre-commit-config.yaml`).

## Datos

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| Un contrato de datos con Great Expectations valida la entrada antes de entrenar o puntuar, y un fallo detiene el proceso | Un ejemplo de datos malos rechazados, con el mensaje | `data/contract.py`, guía 02, paso 6 |
| Los datos de entrenamiento están versionados con DVC | El archivo `.dvc` en git | `data/raw.dvc`, que creas en la guía 02 |
| Cada modelo registra el hash de los datos con los que se entrenó | El tag en la corrida de MLflow | tag `data_version`, guía 03 |

## Entrenamiento

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| El preprocesamiento y el modelo forman un solo `Pipeline` de scikit-learn, que se guarda completo | El código del pipeline | `features.py` |
| Si tus datos tienen una dimensión de tiempo, entrenas y evalúas en ventanas fuera de tiempo (OOT), pasadas como parámetros y no escritas en el código | Las fechas de la ventana como parámetros de la corrida | `TrainingWindow` en `settings.py`, guía 02, paso 7 |
| Si tus etiquetas llegan tarde, declaras la ventana de la etiqueta y solo usas ejemplos con la etiqueta ya madura | La regla escrita y su prueba | `attach_labels`, `label_window_months` |
| Una búsqueda de hiperparámetros pequeña y explícita (por ejemplo, 2 parámetros con 3 valores cada uno), validada sin mirar el conjunto de prueba | La grilla en la configuración y una corrida por combinación | `hpo_n_estimators`, `hpo_max_depth`, guía 03 |
| Si las salidas del modelo se usan como probabilidades, mides la calibración y no la deformas (por ejemplo, con pesos de clase) | La razón de calibración en la evaluación | guía 02, paso 9 |

## MLflow

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| El pipeline completo se registra como modelo pyfunc con firma (signature) y ejemplo de entrada | La pestaña del modelo en MLflow | `registry/pyfunc.py`, `registry/logging.py` |
| Cada corrida guarda tags de trazabilidad: SHA de git, versión de datos y configuración | Los tags de la corrida | `git_sha`, `data_version`, `settings_hash` |
| El modelo está en el registry con los aliases `@champion` y `@challenger` | La página del modelo registrado | guía 03 |
| Un gate escrito en código decide la promoción con reglas explícitas, y su decisión queda registrada | Una promoción aceptada y una rechazada, con sus razones | `registry/gate.py`, guía 03 |
| Las reglas del gate y del monitoreo revisan el segmento que importa en tu dominio (región, producto, tipo de cliente) | La regla por segmento en el código | `gate_max_region_auc_drop`, `region_min_loans` |

Opcional: una ficha del modelo (model card) generada en cada entrenamiento (`registry/model_card.py`).

## Servir el modelo

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| Scoring batch: un comando puntúa un lote completo y guarda la versión del modelo en cada fila | El archivo de salida | `scoring.py`, guía 04 |
| Una API con FastAPI devuelve la predicción, la decisión (si tu modelo alimenta una), la versión del modelo y una explicación por predicción | Una respuesta real de `/predict` | `api/app.py`, guía 04 |
| La explicación usa un método adecuado a tu modelo (TreeSHAP para árboles, coeficientes para un modelo lineal) | La explicación en la respuesta y una línea que justifique el método | `top_factors`, guía 04 |
| La API registra cada predicción con un id y la hora | El archivo de registro | `prediction_log.py` |
| Una imagen Docker slim y multi-stage ejecuta la API | El `Dockerfile` y `docker compose up` funcionando | `Dockerfile`, `compose.yaml` |

Opcional: servir sin código con `mlflow models serve`.

## Automatización

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| GitHub Actions ejecuta el lint y las pruebas en cada push y pull request | Un check en verde en tu repositorio | `.github/workflows/ci.yml`, guía 05 |

Opcional: un job de quality gate y uno que construye la imagen en CI; un ciclo mensual programado con Prefect (`flows.py`).

## Monitoreo

| Obligatorio | Cómo lo muestras | En este repositorio |
| --- | --- | --- |
| Drift de cada variable con PSI, contra una referencia guardada con el modelo (no contra el mes anterior) | El reporte de un período | `monitoring/psi.py`, `monitoring/report.py` |
| Drift de la predicción (PSI de la salida del modelo) | El mismo reporte | `prediction_psi` |
| Si las etiquetas llegan tarde, el monitoreo de performance usa solo etiquetas maduras, y mientras tanto vigilas el drift | La regla de madurez escrita en tu guía o README | guía 06, paso 3 |
| Un ejercicio de diagnóstico: al menos un incidente de datos inyectado y el detector que lo atrapa | El reporte del incidente | laboratorio de la guía 06 |
| El ciclo reentrenar, gate y cambio de alias funciona de punta a punta, y una alerta aislada no basta para reentrenar | Una corrida con alerta, reentrenamiento y decisión del gate | `flows.py`, `notebooks/06_laboratorio_monitoreo.ipynb` |
| Si haces muchas pruebas estadísticas por período (por segmento, por métrica), tus umbrales de alerta lo tienen en cuenta | El umbral y su justificación | `calibration_min_z`, guía 06 |

## Nube

Opcional: el mismo código registra el modelo en una plataforma gestionada, por ejemplo Databricks con Unity Catalog ([guía 07](guias/07-epilogo-databricks.md)).
