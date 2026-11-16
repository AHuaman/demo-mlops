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
