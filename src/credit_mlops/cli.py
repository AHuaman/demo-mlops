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
