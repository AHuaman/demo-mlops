# credit-mlops

Repositorio de referencia para una sesión de MLOps de unas 4 horas, parte de un curso de posgrado en ciencia de datos. Lleva un modelo de riesgo de crédito (probabilidad de default a 12 meses) desde un notebook prototipo hasta un paquete de Python con datos versionados, registro de modelos, una API en Docker, CI y monitoreo con reentrenamiento.

Los datos son sintéticos. Un simulador con semilla fija genera 7 años de solicitudes de préstamos de consumo, con un shock regional y dos incidentes de datos que puedes inyectar para practicar el diagnóstico. El modelo es para enseñar, no para decidir créditos reales.

Herramientas: uv, pydantic-settings, Typer, Great Expectations, DVC, scikit-learn y XGBoost, MLflow, explicaciones TreeSHAP, FastAPI, Docker, GitHub Actions y Prefect, con un epílogo en Databricks.

## Inicio rápido

Sigue primero la [guía de preparación](docs/guias/00-preparacion.md): instala git, uv, Docker Desktop y, en macOS, libomp, y crea tu repositorio desde esta plantilla. Luego, en la raíz de tu repositorio:

```bash
uv sync --all-extras
uv run credit-mlops doctor
```

`doctor` revisa tu entorno y marca con `✗` lo que falta. Cuando no muestre ninguna `✗`, empieza por la guía 01.

## Guías de la sesión

| # | Guía |
| --- | --- |
| 00 | [Preparación del entorno](docs/guias/00-preparacion.md) (antes de clase) |
| 01 | [Por qué MLOps](docs/guias/01-por-que-mlops.md) |
| 02 | [Del notebook al paquete](docs/guias/02-del-notebook-al-paquete.md) |
| 03 | [Ciclo de vida con MLflow](docs/guias/03-ciclo-de-vida-mlflow.md) |
| 04 | [Sirviendo el modelo](docs/guias/04-sirviendo-el-modelo.md) |
| 05 | [CI con GitHub Actions](docs/guias/05-ci.md) |
| 06 | [Monitoreo y reentrenamiento](docs/guias/06-monitoreo-y-reentrenamiento.md) |
| 07 | [Epílogo en Databricks](docs/guias/07-epilogo-databricks.md) |
| 08 | [Checklist del capstone y cierre](docs/guias/08-cierre.md) |

## Lecturas previas (primers)

- [API y HTTP](docs/primers/01-api.md)
- [Docker](docs/primers/02-docker.md)
- [CI/CD](docs/primers/03-ci-cd.md)
- [La nube y Databricks](docs/primers/04-nube-y-databricks.md)

## Otros documentos

- [Del notebook al código](docs/mapa-notebook-a-codigo.md): qué archivo del paquete reemplaza cada celda del prototipo.
- [Checklist del capstone](docs/checklist-capstone.md): lo que tu proyecto final debe mostrar.
- [Guía del instructor](docs/guia-instructor.md): agenda, preparación previa, planes B y tiempos medidos.

## Notebooks

- `notebooks/00_prototipo.ipynb`: el prototipo con sus nueve problemas (guía 01).
- `notebooks/06_laboratorio_monitoreo.ipynb`: tres años de producción simulada, con alertas, un reentrenamiento y dos incidentes de datos (guía 06).

Los dos están guardados con sus salidas, así que puedes leerlos sin ejecutarlos.
