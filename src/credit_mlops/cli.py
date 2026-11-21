from enum import StrEnum
from typing import Annotated

import typer

from credit_mlops import __version__
from credit_mlops.months import month_label
from credit_mlops.settings import Settings
from credit_mlops.simulation.config import (
    LAB_BUREAU_SHIFT_MONTH,
    LAB_INCOME_BUG_MONTH,
    IncidentConfig,
    SimulationConfig,
)
from credit_mlops.simulation.generator import simulate, write_lake

app = typer.Typer(help="credit-mlops command line interface.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """credit-mlops command line interface."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


class Incident(StrEnum):
    income_bug = "income-bug"
    bureau_shift = "bureau-shift"


@app.command("validate")
def validate_command(
    month: Annotated[str, typer.Option(help="Month to validate, as YYYY-MM.")],
) -> None:
    """Check one month of applications against the data contract (exit 1 on failure)."""
    # Imported here so the CLI stays usable without the `train` extra for other commands.
    from credit_mlops.data.contract import validate_applications
    from credit_mlops.data.lake import load_applications
    from credit_mlops.months import parse_month

    settings = Settings()
    try:
        df = load_applications(settings, [parse_month(month)])
    except FileNotFoundError as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    result = validate_applications(df)
    if result.success:
        typer.echo(f"{month}: contract OK ({result.n_rows} rows)")
        return
    typer.echo(f"{month}: contract FAILED ({result.n_rows} rows)")
    for message in result.failed:
        typer.echo(f"  - {message}")
    raise typer.Exit(code=1)


@app.command("simulate")
def simulate_command(
    loans_per_month: Annotated[int, typer.Option(help="Loans originated per month.")] = 2500,
    incident: Annotated[
        list[Incident] | None, typer.Option(help="Data incident to inject (repeatable).")
    ] = None,
) -> None:
    """Generate the synthetic loan lake under `data_dir`."""
    chosen = set(incident or [])
    incidents = IncidentConfig(
        income_in_thousands_from=LAB_INCOME_BUG_MONTH if Incident.income_bug in chosen else None,
        bureau_shift_from=LAB_BUREAU_SHIFT_MONTH if Incident.bureau_shift in chosen else None,
    )
    cfg = SimulationConfig(loans_per_month=loans_per_month, incidents=incidents)
    settings = Settings()
    lake = simulate(cfg)
    write_lake(lake, settings.data_dir)
    n_months = lake.applications["origination_month"].nunique()
    typer.echo(
        f"Wrote {n_months} months ({len(lake.applications)} loans, "
        f"{month_label(cfg.start)} to {month_label(cfg.end)}) to {settings.data_dir / 'raw'}"
    )
    applied = ", ".join(sorted(i.value for i in chosen)) or "none"
    typer.echo(f"Incidents applied: {applied}")


@app.command("train")
def train_command(
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Reference month as YYYY-MM (with --rolled).")
    ] = None,
    rolled: Annotated[
        bool, typer.Option("--rolled", help="Use a rolled window ending at --as-of.")
    ] = False,
) -> None:
    """Train, log to MLflow and register the model as the `challenger`."""
    if rolled != (as_of is not None):
        typer.echo("Pass both --as-of and --rolled, or neither for the default window.")
        raise typer.Exit(code=2)

    # Imported here so the CLI stays usable without the `train` extra for other commands.
    from pathlib import Path

    from credit_mlops.data.contract import ContractViolation
    from credit_mlops.months import parse_month
    from credit_mlops.registry.health import MlflowUnavailableError, require_reachable
    from credit_mlops.registry.logging import log_and_register
    from credit_mlops.settings import DEFAULT_WINDOW, TrainingWindow
    from credit_mlops.training.train import train_model

    settings = Settings()
    if as_of is None:
        window = DEFAULT_WINDOW
    else:
        window = TrainingWindow.rolled(
            parse_month(as_of), half_life_months=settings.retrain_half_life_months
        )
    try:
        # Check the server first: training takes minutes and logging needs MLflow at the end.
        require_reachable(settings.mlflow_tracking_uri)
        result = train_model(settings, window)
        ref = log_and_register(settings, result, repo_root=Path.cwd())
    except (ContractViolation, MlflowUnavailableError, FileNotFoundError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(ref.model_dump_json())


@app.command("mlflow-server")
def mlflow_server_command() -> None:
    """Run a local MLflow tracking server (SQLite backend, proxied artifacts)."""
    import subprocess
    import sys

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--artifacts-destination",
        "./mlartifacts",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--allowed-hosts",
        "localhost:*,127.0.0.1:*,host.docker.internal:*",
    ]
    raise typer.Exit(code=subprocess.run(command).returncode)


@app.command("promote")
def promote_command() -> None:
    """Run the promotion gate on `@challenger` and move `@champion` if it passes."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from credit_mlops.registry.gate import alias_version, promote
    from credit_mlops.registry.health import MlflowUnavailableError, require_reachable

    settings = Settings()
    try:
        require_reachable(settings.mlflow_tracking_uri)
        client = MlflowClient(settings.mlflow_tracking_uri, settings.mlflow_registry_uri)
        name = settings.registered_model_name
        challenger = alias_version(client, name, "challenger")
        champion = alias_version(client, name, "champion")
        decision = promote(settings)
    except (ValueError, MlflowException, MlflowUnavailableError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    previous = f"v{champion}" if champion else "none"
    if decision.promote:
        typer.echo(f"PROMOTED: challenger v{challenger} is now champion (previous: {previous})")
    else:
        typer.echo(f"REJECTED: challenger v{challenger} vs champion v{champion}")
    for reason in decision.reasons:
        typer.echo(f"  - {reason}")


@app.command("rollback")
def rollback_command() -> None:
    """Point `@champion` back at `@previous_champion`."""
    from mlflow.exceptions import MlflowException

    from credit_mlops.registry.gate import rollback
    from credit_mlops.registry.health import MlflowUnavailableError

    settings = Settings()
    try:
        ref = rollback(settings)
    except (ValueError, MlflowException, MlflowUnavailableError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"champion is now {ref.name} v{ref.version}")


@app.command("score")
def score_command(
    month: Annotated[str, typer.Option(help="Month to score, as YYYY-MM.")],
) -> None:
    """Score one month of applications with `@champion` and write a parquet file."""
    import pandas as pd

    from credit_mlops.data.contract import ContractViolation
    from credit_mlops.months import parse_month
    from credit_mlops.registry.loading import ModelUnavailableError
    from credit_mlops.scoring import score_month

    settings = Settings()
    try:
        path = score_month(settings, parse_month(month))
    except (FileNotFoundError, ContractViolation, ModelUnavailableError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {path}")
    counts = pd.read_parquet(path, columns=["decision"])["decision"].value_counts()
    for decision, count in counts.items():
        typer.echo(f"  {decision}: {count}")


@app.command("monitor")
def monitor_command(
    month: Annotated[str, typer.Option(help="Production month to monitor, as YYYY-MM.")],
) -> None:
    """Monitor one production month: contract, drift and matured-label performance."""
    from credit_mlops.monitoring.report import monitor_month
    from credit_mlops.months import parse_month
    from credit_mlops.registry.loading import ModelUnavailableError

    settings = Settings()
    try:
        report = monitor_month(settings, parse_month(month))
    except (FileNotFoundError, ModelUnavailableError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    typer.echo(f"{month}: status {report.status} (model v{report.model_version})")
    for reason in report.reasons:
        typer.echo(f"  - {reason}")
    psis = dict(report.feature_psi)
    if report.prediction_psi is not None:
        psis["prediction"] = report.prediction_psi
    if psis:
        typer.echo("Top PSI:")
        for name, value in sorted(psis.items(), key=lambda item: -item[1])[:5]:
            typer.echo(f"  {name}: {value:.3f}")


@app.command("serve-api")
def serve_api_command(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
) -> None:
    """Serve `@champion` over HTTP with per-prediction explanations."""
    # Imported here: uvicorn belongs to the `api` extra.
    import uvicorn

    uvicorn.run("credit_mlops.api.app:app", host=host, port=port)


@app.command("simulate-production")
def simulate_production_command(
    start: Annotated[str, typer.Option(help="First production month, as YYYY-MM.")],
    end: Annotated[str, typer.Option(help="Last production month, as YYYY-MM.")],
) -> None:
    """Replay the monthly cycle (validate, score, monitor, retrain) over a range of months."""
    # Imported here: prefect belongs to the `flows` dependency set and is slow to import.
    from credit_mlops.flows import simulate_production
    from credit_mlops.registry.loading import ModelUnavailableError

    try:
        results = simulate_production(start, end)
    except (FileNotFoundError, ModelUnavailableError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=1) from error
    for result in results:
        line = f"{result.month}  {result.status:<11} scored={result.scored}"
        if result.retrained is not None and result.gate is not None:
            outcome = "passed" if result.gate.promote else "failed"
            line += f"  retrained v{result.retrained.version}, gate {outcome}"
        typer.echo(line)


@app.command("serve-flow")
def serve_flow_command() -> None:
    """Serve `monthly_cycle` on a cron (06:00 UTC on day 1) for the previous month.

    In served mode every run starts with an alert streak of 0.
    """
    from credit_mlops.flows import monthly_cycle

    monthly_cycle.serve(name="monthly-cycle", cron="0 6 1 * *")


@app.command()
def doctor() -> None:
    """Check this machine before the session (exit 1 if a required check fails)."""
    from credit_mlops.doctor import output_encoding, run_checks, status_symbols

    ok_mark, failed_mark, optional_mark = status_symbols(output_encoding())
    checks = run_checks(Settings())
    for check in checks:
        symbol = ok_mark if check.ok else (failed_mark if check.required else optional_mark)
        typer.echo(f"{symbol} {check.name}: {check.detail}")
    if any(c.required and not c.ok for c in checks):
        raise typer.Exit(code=1)
