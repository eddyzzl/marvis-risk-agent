from pathlib import Path

import pytest

from marvis.validation_materials import discover_validation_material_paths


def _write_validation_dir(root: Path, name: str) -> Path:
    source = root / name
    source.mkdir()
    (source / f"{name}.ipynb").write_text("{}", encoding="utf-8")
    (source / f"{name}.csv").write_text("y,split\n0,train\n", encoding="utf-8")
    (source / f"{name}.pmml").write_text("<PMML/>", encoding="utf-8")
    (source / "数据字典.xlsx").write_bytes(b"xlsx")
    return source


def test_discover_validation_material_paths_picks_unique_roles(tmp_path: Path):
    source = _write_validation_dir(tmp_path, "tcard")
    materials = discover_validation_material_paths(source)
    assert materials.notebook.name == "tcard.ipynb"
    assert materials.sample.name == "tcard.csv"
    assert materials.pmml.name == "tcard.pmml"
    assert materials.dictionary.name == "数据字典.xlsx"


def test_discover_validation_material_paths_rejects_missing_role(tmp_path: Path):
    source = tmp_path / "incomplete"
    source.mkdir()
    (source / "only.ipynb").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必需材料"):
        discover_validation_material_paths(source)
