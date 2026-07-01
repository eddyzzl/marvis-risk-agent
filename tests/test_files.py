import json
from hashlib import sha256
from pathlib import Path

import pytest

from marvis.domain import FileRole
from marvis.files import (
    classify_file,
    scan_source_dir,
    write_json_atomic,
    write_text_atomic,
)


def test_classify_file_roles():
    assert classify_file(Path("model.ipynb")) == FileRole.NOTEBOOK
    assert classify_file(Path("sample.feather")) == FileRole.SAMPLE
    assert classify_file(Path("model_sample.xlsx")) == FileRole.SAMPLE
    assert classify_file(Path("data_dictionary.csv")) == FileRole.DATA_DICTIONARY
    assert classify_file(Path("fr_mob6_final.pmml")) == FileRole.MODEL_PMML
    assert classify_file(Path("数据字典.xlsx")) == FileRole.DATA_DICTIONARY
    assert classify_file(Path("~$04_验证数据汇总表.xlsx")) == FileRole.UNKNOWN
    assert classify_file(Path("~$验证文档.docx")) == FileRole.UNKNOWN


def test_scan_source_dir_ignores_lock_files_and_hashes_small_artifacts(tmp_path):
    notebook = tmp_path / "model.ipynb"
    checkpoint_dir = tmp_path / ".ipynb_checkpoints"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model-checkpoint.ipynb"
    sample = tmp_path / "sample.feather"
    hidden_sample = tmp_path / ".hidden.csv"
    lock_file = tmp_path / ".~04_验证数据汇总表.xlsx"
    office_workbook_lock_file = tmp_path / "~$04_验证数据汇总表.xlsx"
    office_docx_lock_file = tmp_path / "~$验证文档.docx"

    notebook.write_text("{}", encoding="utf-8")
    checkpoint.write_text("{}", encoding="utf-8")
    sample.write_bytes(b"sample data")
    hidden_sample.write_text("hidden", encoding="utf-8")
    lock_file.write_bytes(b"office lock")
    office_workbook_lock_file.write_bytes(b"office workbook lock")
    office_docx_lock_file.write_bytes(b"office docx lock")

    artifacts = scan_source_dir(tmp_path, hash_limit_bytes=sample.stat().st_size)

    by_name = {artifact.path.name: artifact for artifact in artifacts}
    assert ".~04_验证数据汇总表.xlsx" not in by_name
    assert "~$04_验证数据汇总表.xlsx" not in by_name
    assert "~$验证文档.docx" not in by_name
    assert ".hidden.csv" not in by_name
    assert "model-checkpoint.ipynb" not in by_name
    assert by_name["model.ipynb"].role == FileRole.NOTEBOOK
    assert by_name["sample.feather"].role == FileRole.SAMPLE
    assert by_name["sample.feather"].sha256 == sha256(b"sample data").hexdigest()
    assert by_name["model.ipynb"].sha256 == sha256(b"{}").hexdigest()


def test_scan_source_dir_raises_for_missing_source_dir(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_source_dir(missing_dir)


def test_scan_source_dir_raises_for_non_directory_path(tmp_path):
    source_file = tmp_path / "sample.feather"
    source_file.write_bytes(b"sample data")

    with pytest.raises(NotADirectoryError):
        scan_source_dir(source_file)


def test_scan_source_dir_rejects_too_many_files(tmp_path):
    for index in range(3):
        (tmp_path / f"sample-{index}.csv").write_text("x,y\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="too many files"):
        scan_source_dir(tmp_path, max_files=2)


def test_scan_source_dir_rejects_too_deep_paths(tmp_path):
    deep_dir = tmp_path / "a" / "b" / "c"
    deep_dir.mkdir(parents=True)
    (deep_dir / "sample.csv").write_text("x,y\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="too deep"):
        scan_source_dir(tmp_path, max_depth=2)


def test_scan_source_dir_ignores_symlinked_materials(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_sample = outside / "sample.csv"
    escaped_sample.write_text("x,y\n1,0\n", encoding="utf-8")

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.pmml").write_text("<PMML/>", encoding="utf-8")
    (source / "sample.csv").symlink_to(escaped_sample)

    artifacts = scan_source_dir(source)

    by_name = {artifact.path.name: artifact for artifact in artifacts}
    assert "sample.csv" not in by_name
    assert by_name["model.pmml"].role == FileRole.MODEL_PMML


def test_v2_scan_only_classifies_v2_roles(tmp_path):
    (tmp_path / "legacy_model.pkl").write_bytes(b"\x00\x00")
    (tmp_path / "old_report.docx").write_bytes(b"\x00\x00")
    (tmp_path / "model.pmml").write_text("<PMML/>", encoding="utf-8")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    artifacts = scan_source_dir(tmp_path)

    roles = {artifact.role.value for artifact in artifacts}
    assert roles == {"model_pmml", "sample"}


def test_write_text_atomic_replaces_existing_file(tmp_path: Path):
    target = tmp_path / "settings" / "state.txt"
    target.parent.mkdir()
    target.write_text("old", encoding="utf-8")

    result = write_text_atomic(target, "new")

    assert result == target
    assert target.read_text(encoding="utf-8") == "new"
    assert not list(target.parent.glob(".*.tmp"))


def test_write_json_atomic_writes_utf8_json(tmp_path: Path):
    target = tmp_path / "state.json"

    write_json_atomic(target, {"状态": "通过", "count": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"状态": "通过", "count": 2}


def test_write_text_atomic_keeps_existing_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "state.txt"
    target.write_text("old", encoding="utf-8")
    original_replace = Path.replace

    def fail_replace(self, destination):
        if Path(destination) == target:
            raise RuntimeError("replace failed")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        write_text_atomic(target, "new")

    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))
