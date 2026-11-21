# Primer 2: Docker

Docker empaqueta un programa con todo lo que necesita (Python, librerías, configuración) para que corra igual en tu laptop, en la de un compañero y en un servidor. Evita el clásico "en mi máquina funciona".

## Imagen y contenedor

- Una **imagen** es el paquete, de solo lectura. Se construye una vez.
- Un **contenedor** es una imagen en ejecución. Puedes iniciar varios contenedores desde la misma imagen.

La analogía de programación: la imagen es la clase, el contenedor es la instancia.

## Dockerfile, capas y caché

Un `Dockerfile` es la receta de la imagen. Cada instrucción (`FROM`, `COPY`, `RUN`) crea una **capa**. Docker guarda cada capa en caché y la reutiliza si nada cambió antes de ella.

Por eso nuestro `Dockerfile` copia primero `pyproject.toml` y `uv.lock`, instala las dependencias, y solo después copia `src`. Si cambias una línea de código, Docker reutiliza la capa de dependencias (la más lenta) y reconstruye solo lo que sigue.

## Slim, multi-stage y por qué no Alpine

Medimos dos imágenes de la misma API. Los tamaños salen de `docker images` en la máquina del instructor y variarán un poco en la tuya:

| Imagen | Archivo | Tamaño |
| --- | --- | --- |
| `python:3.13-slim`, multi-stage con uv | `Dockerfile` | 1.07 GB |
| `python:3.13`, un solo stage, con dependencias de desarrollo | `docker/Dockerfile.naive` | 2.49 GB |

**Multi-stage** significa dos `FROM` en el mismo archivo. El primer stage (builder) instala las dependencias con uv. El segundo copia solo la carpeta `.venv` ya lista. Las herramientas de construcción no llegan a la imagen final.

Alpine es una base mucho más pequeña, y por eso aparece en muchos tutoriales. No la usamos porque Alpine usa la librería musl, y los paquetes precompilados (wheels) para musl se llaman `musllinux`. Verificamos en PyPI si las versiones fijadas en `uv.lock` los publican. Este comando cuenta los wheels `musllinux` de un paquete (funciona igual en macOS y Windows PowerShell):

```bash
uv run python -c "import json,urllib.request as u; d=json.load(u.urlopen('https://pypi.org/pypi/xgboost/3.4.1/json')); print(sum('musllinux' in f['filename'] for f in d['urls']))"
```

Para cambiar de paquete, cambia el nombre y la versión en la URL. Salida real (consultado el 2026-11-16) con las versiones de `uv.lock`:

| Paquete | Versión | Wheels musllinux |
| --- | --- | --- |
| scikit-learn | 1.9.1 | 0 |
| xgboost | 3.4.1 | 0 |
| numpy | 2.5.5 | 12 |

numpy sirve de contraste. Otros paquetes del proyecto (pandas, pyarrow) también publican wheels musllinux. El problema son scikit-learn y xgboost.

scikit-learn y xgboost solo publican wheels `manylinux` (para glibc). En Alpine, pip tendría que compilarlos desde el código fuente, lo cual es más lento, más frágil y exige compiladores dentro de la imagen. `python:3.13-slim` usa Debian con glibc e instala los wheels directo.

## Compose

`compose.yaml` describe cómo ejecutar uno o varios contenedores con un solo comando, sin recordar los flags de `docker run`. El nuestro define el servicio `api`: construye la imagen, publica el puerto 8000 y apunta al servidor MLflow que corre en tu máquina (`host.docker.internal`).

```bash
docker compose up --build
```

## Docker Desktop y WSL2

Docker Desktop es la aplicación que instala el motor de Docker en macOS y Windows. En Windows corre sobre WSL2 (Windows Subsystem for Linux), un Linux liviano dentro de Windows. La guía `00-preparacion.md` explica cómo instalarlo.

## Verifica que funciona

```bash
docker run --rm hello-world
```

La primera vez, Docker descarga la imagen `hello-world`. Después de las líneas de descarga, debes ver:

```text
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

## Dónde lo usamos

Construimos y ejecutamos la API en un contenedor en [Servir el modelo](../guias/04-sirviendo-el-modelo.md).
