from pathlib import Path

import pytest
from pydantic import ValidationError

from credit_mlops.settings import Settings

EXAMPLE = Path(__file__).resolve().parents[1] / ".env.databricks.example"


@pytest.fixture
def filled_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in (
        "CREDIT_MLOPS_MLFLOW_TRACKING_URI",
        "CREDIT_MLOPS_MLFLOW_REGISTRY_URI",
        "CREDIT_MLOPS_REGISTERED_MODEL_NAME",
        "CREDIT_MLOPS_TRAINING_EXPERIMENT",
    ):
        monkeypatch.delenv(name, raising=False)
    text = EXAMPLE.read_text(encoding="utf-8")
    filled = text.replace("<tu-correo>", "ana@example.com").replace("<tu-perfil>", "free")
    path = tmp_path / ".env.databricks"
    path.write_text(filled, encoding="utf-8")
    return path


def test_databricks_example_passes_validation(filled_env: Path) -> None:
    s = Settings(_env_file=filled_env)
    assert s.mlflow_tracking_uri == "databricks"
    assert s.mlflow_registry_uri == "databricks-uc"
    assert s.registered_model_name == "workspace.default.credit_default_pd"
    assert s.training_experiment == "/Users/ana@example.com/credit-default-training"


def test_databricks_example_with_two_part_name_fails(filled_env: Path) -> None:
    text = filled_env.read_text(encoding="utf-8")
    filled_env.write_text(
        text.replace("workspace.default.credit_default_pd", "default.credit_default_pd"),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="catalog.schema.model"):
        Settings(_env_file=filled_env)
