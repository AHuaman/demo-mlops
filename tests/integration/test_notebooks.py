"""Execute the teaching notebooks end to end against a small simulated lake."""

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from credit_mlops.simulation.config import SimulationConfig
from credit_mlops.simulation.generator import simulate, write_lake

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]


def execute_notebook(
    name: str, workdir: Path, lake_config: SimulationConfig, timeout: int = 600
) -> nbformat.NotebookNode:
    """Run `notebooks/<name>` from a temp copy of the repo layout with a simulated lake.

    The notebook runs with cwd = `<workdir>/notebooks`, like it does for students, so its
    relative `../data/raw` paths resolve to the simulated lake.
    """
    notebooks = workdir / "notebooks"
    notebooks.mkdir(parents=True)
    shutil.copy(REPO_ROOT / "notebooks" / name, notebooks / name)
    write_lake(simulate(lake_config), workdir / "data")
    notebook = nbformat.read(notebooks / name, as_version=4)
    NotebookClient(
        notebook, timeout=timeout, resources={"metadata": {"path": str(notebooks)}}
    ).execute()
    return notebook


def stream_text(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "stream"
    )


def test_prototype_notebook_executes(tmp_path: Path) -> None:
    notebook = execute_notebook(
        "00_prototipo.ipynb", tmp_path, SimulationConfig(loans_per_month=400)
    )
    text = stream_text(notebook)
    assert "AUC split aleatorio:" in text
    assert "AUC fuera de tiempo:" in text
