import subprocess

import pytest
from typer.testing import CliRunner

from credit_mlops import doctor
from credit_mlops.cli import app
from credit_mlops.settings import Settings


@pytest.fixture
def healthy(monkeypatch):
    """Every external probe succeeds, independent of this machine."""
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        doctor, "_run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", "")
    )
    monkeypatch.setattr(doctor, "_server_reachable", lambda uri: True)
    monkeypatch.setattr(doctor, "_port_is_free", lambda port: True)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())


def _by_name(checks, fragment):
    return next(c for c in checks if fragment in c.name)


def test_all_ok_exits_zero(healthy):
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "✗" not in result.output
    assert "!" not in result.output


def test_git_missing_fails_required_and_exits_one(healthy, monkeypatch):
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name: None if name == "git" else f"/usr/bin/{name}"
    )
    check = _by_name(doctor.run_checks(Settings()), "git")
    assert (check.ok, check.required) == (False, True)
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "✗ git instalado" in result.output


def test_docker_missing_is_a_warning_not_a_failure(healthy, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name: None if name == "docker" else f"/usr/bin/{name}"
    )
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "! Docker en ejecución" in result.output
    assert "WSL2" not in result.output
    assert "inicia Docker Desktop" in result.output


def test_docker_warning_mentions_wsl2_on_windows(healthy, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(
        doctor.shutil, "which", lambda name: None if name == "docker" else f"/usr/bin/{name}"
    )
    assert "WSL2" in _by_name(doctor.run_checks(Settings()), "Docker").detail


def test_xgboost_failure_on_macos_suggests_libomp(healthy, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "darwin")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "XGBoost Library"),
    )
    check = _by_name(doctor.run_checks(Settings()), "XGBoost")
    assert not check.ok and check.required
    assert "brew install libomp" in check.detail


def test_xgboost_failure_elsewhere_shows_error_line(healthy, monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "win32")
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "Traceback\nImportError: boom"
        ),
    )
    assert _by_name(doctor.run_checks(Settings()), "XGBoost").detail == "ImportError: boom"


def test_xgboost_timeout_asks_to_rerun(healthy, monkeypatch):
    def slow(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 10))

    monkeypatch.setattr(doctor, "_run", slow)
    check = _by_name(doctor.run_checks(Settings()), "XGBoost")
    assert not check.ok and check.required
    assert "60 s" in check.detail and "vuelve a ejecutar" in check.detail


def test_sqlite_tracking_uri_counts_as_ok(healthy, monkeypatch):
    monkeypatch.setenv("CREDIT_MLOPS_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    check = _by_name(doctor.run_checks(Settings()), "MLflow")
    assert check.ok and "sqlite" in check.detail


def test_mlflow_down_with_free_port_is_ok_and_busy_port_warns(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "_server_reachable", lambda uri: False)
    assert _by_name(doctor.run_checks(Settings()), "MLflow").ok
    monkeypatch.setattr(doctor, "_port_is_free", lambda port: False)
    assert not _by_name(doctor.run_checks(Settings()), "MLflow").ok


def test_status_symbols_fall_back_to_ascii_when_the_encoding_cannot_encode_them():
    assert doctor.status_symbols("utf-8") == ("✓", "✗", "!")
    assert doctor.status_symbols("cp1252") == ("OK", "X", "!")
    assert doctor.status_symbols(None) == ("OK", "X", "!")


def test_doctor_prints_ascii_markers_on_a_cp1252_stdout(healthy, monkeypatch):
    monkeypatch.setattr(doctor, "output_encoding", lambda: "cp1252")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "OK " in result.output
    assert "✓" not in result.output
