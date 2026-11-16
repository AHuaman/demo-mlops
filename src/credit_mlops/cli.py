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

    from credit_mlops.months import parse_month
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
    ref = log_and_register(settings, train_model(settings, window), repo_root=Path.cwd())
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
