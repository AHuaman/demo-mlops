import typer

from credit_mlops import __version__

app = typer.Typer(help="credit-mlops command line interface.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """credit-mlops command line interface."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
