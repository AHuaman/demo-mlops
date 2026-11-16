# Primer 4: nube y Databricks

En la sesión corres todo en tu laptop: un servidor MLflow local, un registry en un archivo SQLite y una API en un contenedor. En una empresa, ese trabajo suele vivir en una plataforma de ML administrada en la nube. Este primer explica qué cambia y cómo prepararte para probarlo.

## Qué agrega una plataforma administrada

- **Tracking administrado.** Alguien más opera el servidor de experimentos, sus copias de seguridad y sus accesos.
- **Registry con gobierno.** Los modelos tienen dueños, permisos y un historial de quién cambió qué.
- **Compute y serving administrados.** Entrenas en máquinas que la plataforma enciende y apaga, y publicas el modelo como un endpoint sin construir tu propio contenedor.

Lo que no cambia es el ciclo: entrenar, registrar, evaluar, promover, servir, monitorear.

## Cómo se mapea nuestro diseño

Nuestro código no sabe en qué plataforma corre. Lee dos variables de entorno (`config/settings.toml` lista los valores por defecto):

- `CREDIT_MLOPS_MLFLOW_TRACKING_URI`, la dirección del servidor de tracking.
- `CREDIT_MLOPS_MLFLOW_REGISTRY_URI`, dónde vive el registry (`databricks-uc` para Unity Catalog).

Cambiar esas variables mueve el mismo código a otra plataforma. Los equivalentes principales:

| Plataforma | Tracking y registry | Serving |
| --- | --- | --- |
| Databricks | MLflow administrado y Unity Catalog | Model Serving |
| Azure ML | Workspace de Azure ML (compatible con MLflow) | Managed online endpoints |
| AWS SageMaker | SageMaker con MLflow administrado | Endpoints de SageMaker |
| Google Vertex AI | Vertex AI Model Registry | Endpoints de Vertex AI |

La tabla da solo la idea general. Los nombres de producto cambian con frecuencia, así que revisa la documentación del proveedor antes de decidir algo.

## Databricks Free Edition

Databricks ofrece Free Edition, una versión gratuita para aprender. La documentación pública confirma que funciona solo con compute serverless y con cuotas limitadas, y que si superas la cuota diaria el compute se apaga hasta el día siguiente (tus datos se conservan). También indica límites en los endpoints de model serving (por ejemplo, sin endpoints con GPU).

Lo siguiente no lo pudimos confirmar en la documentación y el instructor lo confirmará en clase: si piden tarjeta de crédito (normalmente no), la forma exacta de la URL de tu workspace, el valor de la cuota diaria, el nombre del catálogo por defecto (normalmente `workspace`) y qué funciones de MLflow y Unity Catalog incluye.

### Crear la cuenta y el workspace

1. Entra a https://www.databricks.com/learn/free-edition (o busca "Databricks Free Edition") y elige un método de registro.
2. Databricks crea tu workspace. Anota su URL, que normalmente tiene la forma `https://<algo>.cloud.databricks.com`.
3. Abre la URL y confirma que ves la interfaz del workspace.

### Instalar el CLI

La documentación oficial está en https://docs.databricks.com/aws/en/dev-tools/cli/install. Según esa página:

macOS:

```bash
brew install databricks/tap/databricks
```

Windows (PowerShell):

```powershell
winget install Databricks.DatabricksCLI
```

Cierra y abre la terminal. Verifica con `databricks -v`.

### Autenticarte

```bash
databricks auth login --host <url-del-workspace> --profile <nombre>
```

Reemplaza `<url-del-workspace>` con tu URL y `<nombre>` con un nombre corto, por ejemplo `free`. El comando abre el navegador para que apruebes el acceso.

El **profile** es un nombre guardado en tu archivo `~/.databrickscfg` (en Windows, `%USERPROFILE%\.databrickscfg`) con la URL y las credenciales. Con varios workspaces, eliges cuál usar con `--profile <nombre>`.

### Unity Catalog y nombres de tres niveles

Unity Catalog organiza los objetos en tres niveles: `catalog.schema.model`. Por ejemplo, `workspace.credit.credit_default_pd` sería el modelo `credit_default_pd` en el esquema `credit` del catálogo `workspace`.

En un workspace nuevo, el catálogo por defecto normalmente se llama `workspace`. Confírmalo en la interfaz, en Catalog. Con `databricks-uc` como registry, el nombre del modelo debe tener los tres niveles. Nuestro `Settings` rechaza un nombre sin ese formato.

### Cuota diaria

Como el compute de Free Edition tiene una cuota diaria, una corrida larga puede detenerse cuando la agotas. No dejes tareas de prueba para el último minuto, y no repitas entrenamientos innecesarios.

## Verifica que funciona

Con el CLI instalado y tu profile creado:

```bash
databricks auth profiles
```

Debes ver tu `<nombre>` en la lista, con su URL y una columna que indica si el login sigue vigente. Ejemplo ilustrativo, no ejecutado (no usamos ningún workspace para escribir esta guía), y el formato exacto puede variar:

```text
Name  Host                                  Valid
free  https://<algo>.cloud.databricks.com   YES
```

Si la columna de validez dice `NO`, repite `databricks auth login`.

## Dónde lo usamos

Probamos este cambio de plataforma en el epílogo: [Epílogo Databricks](../guias/07-epilogo-databricks.md).
