"""Environment checks students run before the session (`credit-mlops doctor`)."""

import importlib.util
import shutil
import socket
import subprocess
import sys
from urllib.parse import urlparse

from pydantic import BaseModel

from credit_mlops.registry.health import is_http, unreachable_reason
from credit_mlops.settings import Settings

SUBPROCESS_TIMEOUT_SECONDS = 10
XGBOOST_IMPORT_TIMEOUT_SECONDS = 60  # a cold first import can take well over 10 s
MLFLOW_PORT = 5000
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class Check(BaseModel):
    name: str
    ok: bool
    required: bool
    detail: str


def _run(
    args: list[str], timeout: float = SUBPROCESS_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    """Run an argument list (no shell) with a timeout; callers handle OSError/timeouts."""
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _server_reachable(uri: str) -> bool:
    return unreachable_reason(uri) is None


def check_python() -> Check:
    version = sys.version_info
    found = f"{version.major}.{version.minor}.{version.micro}"
    ok = (version.major, version.minor) == (3, 13)
    detail = (
        f"Python {found}" if ok else f"Python {found}: se necesita 3.13 (uv python install 3.13)"
    )
    return Check(name="Python 3.13", ok=ok, required=True, detail=detail)


def check_uv() -> Check:
    path = shutil.which("uv")
    detail = path or "uv no está en el PATH (https://docs.astral.sh/uv/)"
    return Check(name="uv instalado", ok=path is not None, required=True, detail=detail)


def check_git() -> Check:
    path = shutil.which("git")
    detail = path or "git no está en el PATH (https://git-scm.com/downloads)"
    return Check(name="git instalado", ok=path is not None, required=True, detail=detail)


def check_dvc() -> Check:
    ok = importlib.util.find_spec("dvc") is not None
    detail = "dvc se puede importar" if ok else "dvc no está instalado: ejecuta uv sync"
    return Check(name="DVC disponible", ok=ok, required=True, detail=detail)


def check_xgboost() -> Check:
    name = "XGBoost carga correctamente"
    try:
        result = _run(
            [sys.executable, "-c", "import xgboost"], timeout=XGBOOST_IMPORT_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        detail = (
            f"import xgboost tardó más de {XGBOOST_IMPORT_TIMEOUT_SECONDS} s: "
            "vuelve a ejecutar doctor (la primera importación es lenta)"
        )
        return Check(name=name, ok=False, required=True, detail=detail)
    except OSError as error:
        return Check(name=name, ok=False, required=True, detail=f"no se pudo probar: {error}")
    if result.returncode == 0:
        return Check(name=name, ok=True, required=True, detail="import xgboost funciona")
    if sys.platform == "darwin":
        detail = "falta el runtime OpenMP: ejecuta brew install libomp"
    else:
        lines = result.stderr.strip().splitlines()
        detail = lines[-1] if lines else "import xgboost falló"
    return Check(name=name, ok=False, required=True, detail=detail)


def check_docker() -> Check:
    name = "Docker en ejecución"
    hint = (
        "instala Docker Desktop con WSL2 y ábrelo"
        if sys.platform == "win32"
        else "inicia Docker Desktop"
    )
    if shutil.which("docker") is None:
        return Check(name=name, ok=False, required=False, detail=f"docker no encontrado: {hint}")
    try:
        result = _run(["docker", "info"])
    except (OSError, subprocess.TimeoutExpired):
        return Check(name=name, ok=False, required=False, detail=f"docker no responde: {hint}")
    if result.returncode != 0:
        return Check(name=name, ok=False, required=False, detail=f"docker no responde: {hint}")
    return Check(name=name, ok=True, required=False, detail="docker info responde")


def check_mlflow(settings: Settings) -> Check:
    name = "MLflow accesible"
    uri = settings.mlflow_tracking_uri
    if not is_http(uri):
        return Check(
            name=name, ok=True, required=False, detail=f"tracking URI local ({uri}): sin servidor"
        )
    if _server_reachable(uri):
        return Check(name=name, ok=True, required=False, detail=f"servidor responde en {uri}")
    host = urlparse(uri).hostname
    if host in _LOCAL_HOSTS:
        if _port_is_free(MLFLOW_PORT):
            return Check(
                name=name,
                ok=True,
                required=False,
                detail=f"sin servidor en {uri}; el puerto {MLFLOW_PORT} está libre para iniciarlo",
            )
        return Check(
            name=name,
            ok=False,
            required=False,
            detail=f"el puerto {MLFLOW_PORT} está ocupado y {uri} no responde como MLflow",
        )
    return Check(name=name, ok=False, required=False, detail=f"{uri} no responde")


def run_checks(settings: Settings) -> list[Check]:
    return [
        check_python(),
        check_uv(),
        check_git(),
        check_dvc(),
        check_xgboost(),
        check_docker(),
        check_mlflow(settings),
    ]
