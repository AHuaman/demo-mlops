from typer.testing import CliRunner

from credit_mlops.cli import app


def test_version_command_prints_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_simulate_income_bug_incident_is_wired(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(
        app, ["simulate", "--incident", "income-bug", "--loans-per-month", "20"]
    )
    assert result.exit_code == 0, result.output
    assert "Incidents applied: income-bug" in result.stdout
    apps = tmp_path / "raw" / "applications"
    assert pd.read_parquet(apps / "2025-03.parquet")["monthly_income_pen"].max() < 60
    assert pd.read_parquet(apps / "2025-02.parquet")["monthly_income_pen"].min() >= 500


def test_simulate_bureau_shift_incident_is_wired(tmp_path, monkeypatch):
    import pandas as pd

    def run(directory, *extra):
        monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(directory))
        result = CliRunner().invoke(app, ["simulate", "--loans-per-month", "30", *extra])
        assert result.exit_code == 0, result.output
        return result, directory / "raw" / "applications"

    result, shifted = run(tmp_path / "shifted", "--incident", "bureau-shift")
    _, clean = run(tmp_path / "clean")
    assert "Incidents applied: bureau-shift" in result.stdout

    may = "2025-05.parquet"
    june = "2025-06.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(shifted / may), pd.read_parquet(clean / may))
    s, c = pd.read_parquet(shifted / june), pd.read_parquet(clean / june)
    assert (s["bureau_score"] != c["bureau_score"]).any()
    assert (s["bureau_score"] == (c["bureau_score"] - 80).clip(300, 999)).all()


def _simulate_income_bug_lake(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(
        app, ["simulate", "--incident", "income-bug", "--loans-per-month", "20"]
    )
    assert result.exit_code == 0, result.output


def test_validate_passes_on_clean_month(tmp_path, monkeypatch):
    _simulate_income_bug_lake(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["validate", "--month", "2025-02"])
    assert result.exit_code == 0, result.output
    assert "contract OK" in result.stdout


def test_validate_fails_on_corrupted_month(tmp_path, monkeypatch):
    _simulate_income_bug_lake(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["validate", "--month", "2025-03"])
    assert result.exit_code == 1
    assert "contract FAILED" in result.stdout
    assert any(line.startswith("  - monthly_income_pen: ") for line in result.stdout.splitlines())


def test_validate_reports_missing_month(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["validate", "--month", "2025-03"])
    assert result.exit_code == 1
    assert "2025-03" in result.stdout


def test_train_requires_as_of_and_rolled_together():
    for args in (["train", "--rolled"], ["train", "--as-of", "2023-01"]):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 2
        assert "both --as-of and --rolled" in result.output


def test_mlflow_server_launches_mlflow_with_expected_arguments(monkeypatch):
    import subprocess
    import sys

    captured = {}

    def fake_run(command):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CliRunner().invoke(app, ["mlflow-server"])
    assert result.exit_code == 0
    command = captured["command"]
    assert command[:4] == [sys.executable, "-m", "mlflow", "server"]
    assert command[command.index("--backend-store-uri") + 1] == "sqlite:///mlflow.db"
    assert command[command.index("--artifacts-destination") + 1] == "./mlartifacts"
    assert "--allowed-hosts" in command


def test_serve_api_runs_uvicorn_with_defaults_and_overrides(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert CliRunner().invoke(app, ["serve-api"]).exit_code == 0
    assert (
        CliRunner().invoke(app, ["serve-api", "--host", "0.0.0.0", "--port", "9000"]).exit_code == 0
    )
    target = "credit_mlops.api.app:app"
    assert calls == [
        ((target,), {"host": "127.0.0.1", "port": 8000}),
        ((target,), {"host": "0.0.0.0", "port": 9000}),
    ]
