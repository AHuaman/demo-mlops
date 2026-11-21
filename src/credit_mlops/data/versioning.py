"""Data and code version helpers, recorded as tags on every training run."""

import subprocess
from pathlib import Path

import yaml

UNKNOWN = "unknown"


def data_version(repo_root: Path, tracked: str = "data/raw") -> str:
    """Return the DVC md5 of the tracked path, or "untracked" if not DVC-tracked."""
    dvc_file = repo_root / f"{tracked}.dvc"
    if not dvc_file.is_file():
        return "untracked"
    content = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
    return str(content["outs"][0]["md5"])


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_info(repo_root: Path) -> dict[str, str]:
    """Return the current git SHA and whether the working tree is dirty."""
    sha = _git(repo_root, "rev-parse", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "git_sha": sha or UNKNOWN,
        "git_dirty": UNKNOWN if status is None else str(bool(status)).lower(),
    }
