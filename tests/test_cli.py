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
