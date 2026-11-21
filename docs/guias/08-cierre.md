# Checklist del capstone y cierre

**Objetivo.** Repasar qué corrigió cada bloque del día y cómo llevarlo a tu propio proyecto con el [checklist del capstone](../checklist-capstone.md).
Al final sabes qué ítems del checklist son obligatorios en tu capstone y por dónde empezar.

**Problemas del prototipo que cierra.** Los nueve. Esta guía no agrega código; muestra dónde quedó resuelto cada uno.

## Lo que corrigió cada bloque

El día empezó con un notebook que funcionaba una vez, en una máquina, y no dejaba rastro. Cada bloque cambió una parte:

| Bloque | Guía | Qué tenía el prototipo | Qué tienes ahora |
| --- | --- | --- | --- |
| 1 | [Por qué MLOps](01-por-que-mlops.md) | Un AUC impreso y nueve problemas sin nombre | La lista de problemas y el bloque que corrige cada uno |
| 2 | [Del notebook al paquete](02-del-notebook-al-paquete.md) | Rutas fijas, etiquetas usadas antes de existir, split aleatorio, preprocesamiento fuera del modelo, sin validación (problemas 1, 2, 3, 6 y 7) | Un paquete con configuración, CLI, contrato de datos, datos versionados con DVC, ventanas fuera de tiempo y un `Pipeline` |
| 3 | [Ciclo de vida con MLflow](03-ciclo-de-vida-mlflow.md) | Parámetros sin registro, modelo sin guardar, sin rastro de los datos (problemas 4, 5 y 9) | Cada corrida con sus parámetros, métricas, SHA de git y hash de datos; un registry con `@champion` y un gate en código |
| 4 | [Sirviendo el modelo](04-sirviendo-el-modelo.md) | Imposible de programar o servir (problema 8) | Scoring batch, una API con explicación por predicción y la misma API en Docker |
| 5 | [CI con GitHub Actions](05-ci.md) | Nadie revisaba nada antes de cambiar el código | Lint y pruebas en Ubuntu y Windows en cada pull request |
| 6 | [Monitoreo y reentrenamiento](06-monitoreo-y-reentrenamiento.md) | El prototipo terminaba en el AUC | PSI contra la referencia del modelo, performance con etiquetas maduras, alertas con evidencia estadística y un ciclo que reentrena y pasa por el gate |
| 7 | [Epílogo en Databricks](07-epilogo-databricks.md) | Todo vivía en una laptop | Los mismos comandos registrando en Unity Catalog |

Las guías repiten tres reglas:

- **Todo es un comando.** Si un paso necesita que alguien abra un notebook y recuerde el orden de las celdas, no se puede repetir ni programar.
- **Todo deja rastro.** Para cada modelo en producción puedes decir con qué código, con qué datos y con qué configuración se entrenó, y por qué el gate lo aceptó.
- **Ningún cambio llega sin evidencia.** El contrato frena los datos malos, CI frena el código roto, el gate frena el modelo peor y la alerta de monitoreo necesita significancia estadística y una racha de meses.

## Cómo usar el checklist

El [checklist del capstone](../checklist-capstone.md) lista los ítems obligatorios por área (código, datos, entrenamiento, MLflow, servir, automatización, monitoreo) y los opcionales. Úsalo así:

1. **Elige el problema y responde las condiciones.** Algunos ítems empiezan con "Si...". ¿Tus datos tienen una dimensión de tiempo? ¿Las etiquetas llegan tarde? ¿Las salidas del modelo se usan como probabilidades? ¿Qué segmento importa en tu dominio? Escribe las respuestas en el README de tu capstone. Si una condición no aplica, dilo en una línea.
2. **Parte de este repositorio.** Crea tu capstone desde la plantilla, como hiciste en la preparación, y reemplaza en este orden. Las rutas siguientes cuelgan de `src/credit_mlops/`:
   - `schema.py`: tus variables, sus tipos y sus límites. El contrato de datos y la validación de la API salen de aquí.
   - `simulation/`: cámbialo por la carga de tus datos. Mantén la misma salida (un archivo por período) si tu problema tiene tiempo.
   - `settings.py` y `config/settings.toml`: la ventana de entrenamiento, la grilla del HPO, el umbral de decisión y los umbrales de monitoreo.
   - `registry/gate.py` y `monitoring/report.py`: las reglas por segmento de tu dominio.
3. **Marca cada ítem con su evidencia.** Copia las tablas del checklist a un issue o a tu README y, en cada ítem, enlaza la evidencia: un archivo, una corrida de MLflow, un check de CI o una salida pegada.
4. **Corre todo desde cero antes de entregar.** Clona tu repositorio en otra carpeta, ejecuta `uv sync --all-extras` y los comandos de tu README en orden. Si un paso falla en la copia limpia, también le fallará a quien lo revise.

## Antes de cerrar la laptop

Detén lo que quedó corriendo: el servidor de MLflow (`Ctrl+C` en la segunda terminal), la API o el contenedor si siguen arriba, y Jupyter. Si en la guía 06 configuraste Prefect con `prefect config set`, quita esa configuración como dice el final de su paso 6.

## Por qué así

**Un checklist general, no uno de crédito.** Tu capstone puede ser de churn, de demanda o de fraude. Los ítems piden la práctica (validar la entrada, evaluar como en producción, registrar, monitorear) y dejan que elijas cómo aplicarla a tu problema.

**Evidencia en vez de afirmaciones.** Igual que en las guías, donde cada número sale de un comando que puedes repetir, un ítem del checklist cuenta cuando enlazas lo que lo demuestra.

## Para el instructor

10 minutos. Recorre la tabla de bloques en voz alta, abre el checklist y resuelve las dudas sobre los ítems condicionales con ejemplos de los problemas que eligieron los estudiantes.
