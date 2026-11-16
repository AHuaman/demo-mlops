# Preparación del entorno

Haz esta guía en tu laptop antes de la sesión. Toma unos 30 minutos, casi todo descargas. Al final, un comando (`doctor`) te dice si todo quedó listo.

Los comandos están en inglés tal como los escribirías. Donde macOS y Windows difieren, hay un bloque para cada uno. En Windows usa **PowerShell** (no CMD).

## Qué vas a instalar

| Herramienta | Para qué |
| --- | --- |
| git | Descargar el repositorio del curso |
| uv | Instala Python 3.13 y las dependencias del proyecto |
| Docker Desktop | Ejecutar la API en un contenedor (parte de la sesión) |
| libomp (solo macOS) | Librería que XGBoost necesita para cargar |

Si nunca has usado Docker o una API, lee antes los primers de `docs/primers/`. No hace falta dominarlos para esta guía.

## 1. git

macOS (Terminal). Si `git --version` ya responde, salta este paso.

```bash
xcode-select --install
```

Windows (PowerShell):

```powershell
winget install --id Git.Git -e
```

Cierra y vuelve a abrir la terminal. Verifica con `git --version`.

## 2. uv

uv descarga Python 3.13 por ti, así que no necesitas instalar Python aparte.

macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cierra y vuelve a abrir la terminal. Verifica con `uv --version`.

## 3. libomp (solo macOS)

XGBoost necesita el runtime OpenMP. Con Homebrew (https://brew.sh):

```bash
brew install libomp
```

Si no haces este paso, `doctor` marcará XGBoost con ✗ y te mostrará este mismo comando (ver "Si algo falla").

## 4. Docker Desktop

Descarga Docker Desktop desde https://www.docker.com/products/docker-desktop/ e instálalo. Ábrelo y espera a que diga "Engine running".

En Windows, Docker Desktop usa WSL2. Si el instalador te lo pide, ejecuta en PowerShell como administrador y reinicia:

```powershell
wsl --install
```

Verifica con `docker --version` y `docker info`. El segundo comando debe responder sin error.

## 5. El repositorio

Clona el repositorio del curso. El instructor te da acceso al repositorio de GitHub.

```bash
git clone https://github.com/AHuaman/demo-mlops.git
cd demo-mlops
```

## 6. Dependencias

```bash
uv sync --all-extras
```

uv crea la carpeta `.venv` con todas las dependencias. La primera vez descarga varios cientos de MB, así que usa una buena conexión.

## 7. Verifica con doctor

```bash
uv run credit-mlops doctor
```

Cada línea empieza con un símbolo. `✓` es correcto, `✗` es un problema que debes arreglar, `!` es un aviso de algo opcional. Salida de una máquina con todo listo (las rutas y la versión de Python cambian en la tuya):

```text
✓ Python 3.13: Python 3.13.13
✓ uv instalado: /opt/homebrew/bin/uv
✓ git instalado: /opt/homebrew/bin/git
✓ DVC disponible: dvc se puede importar
✓ XGBoost carga correctamente: import xgboost funciona
✓ Docker en ejecución: docker info responde
✓ MLflow accesible: sin servidor en http://127.0.0.1:5000; el puerto 5000 está libre para iniciarlo
```

Que MLflow diga "sin servidor" es normal. El servidor se inicia durante la sesión.

Docker es opcional para `doctor` (si falla verás `!`), pero lo necesitas para la parte de la API.

## Si algo falla

**XGBoost con ✗ en macOS.** Ejemplo de salida cuando falta libomp:

```text
✗ XGBoost carga correctamente: falta el runtime OpenMP: ejecuta brew install libomp
```

Ejecuta `brew install libomp` y repite `doctor`.

**XGBoost con "timed out after 10 seconds".** La primera vez que se importa XGBoost, el sistema puede tardar más de 10 segundos. Ejecuta `doctor` otra vez. La segunda vez responde en segundos.

**uv o git "no está en el PATH".** Cierra la terminal y ábrela de nuevo. Si sigue igual, repite la instalación de ese paso.

**Docker con `!`.** Abre Docker Desktop y espera a que termine de iniciar. En Windows, revisa que el backend WSL2 esté activo en Settings, General.

**`doctor` termina con código de salida 1.** Hay al menos un `✗` en una herramienta requerida. Arregla esa línea y repite.

Cuando `doctor` no muestre ningún `✗`, estás listo.
