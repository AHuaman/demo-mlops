from pathlib import Path

from credit_mlops.data.versioning import data_version, git_info


def test_data_version_untracked_when_no_dvc_file(tmp_path: Path) -> None:
    assert data_version(tmp_path) == "untracked"


def test_data_version_reads_md5_from_dvc_file(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "raw.dvc").write_text(
        "outs:\n- md5: abc123.dir\n  path: raw\n", encoding="utf-8"
    )
    assert data_version(tmp_path) == "abc123.dir"


def test_git_info_unknown_outside_repo(tmp_path: Path) -> None:
    assert git_info(tmp_path) == {"git_sha": "unknown", "git_dirty": "unknown"}


def test_git_info_in_repo_reports_sha_and_dirty() -> None:
    info = git_info(Path(__file__).resolve().parents[2])
    assert info["git_sha"] != "unknown"
    assert info["git_dirty"] in {"true", "false"}
