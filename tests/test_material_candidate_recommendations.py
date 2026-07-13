from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import nbformat
import pandas as pd
import pytest

import marvis.validation.feature_metadata as feature_metadata
from marvis.api_scan_helpers import material_candidates_payload
from marvis.db import TaskRepository, init_db
from marvis.domain import FileRole, TaskCreate
from marvis.files import classify_file
from marvis.routers.scans import task_material_candidates
from marvis.validation.feature_metadata import (
    FeatureMetadataInspectionIncomplete,
    inspect_feature_metadata,
)
from marvis.validation.pmml_manifest import parse_pmml_input_manifest


@pytest.mark.parametrize(
    "name",
    [
        "特征元数据.xlsx",
        "metadata.csv",
        "feature_importance_best.csv",
        "feature importance.xlsx",
        "特征重要性.xlsx",
        "特征元数据.parquet",
    ],
)
def test_feature_metadata_names_are_classified_as_data_dictionary(name: str) -> None:
    assert classify_file(Path(name)) is FileRole.DATA_DICTIONARY


def test_unique_pmml_compatible_workbook_is_recommended(tmp_path: Path) -> None:
    source = _write_materials(tmp_path / "source")
    _write_complete_metadata(source / "03_模型评估材料.xlsx")
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert "03_模型评估材料.xlsx" in candidates
    assert candidates["03_模型评估材料.xlsx"]["metadata_compatibility"] == {
        "status": "compatible",
        "selection_count": 1,
        "blocking_errors": [],
    }
    assert candidates["03_模型评估材料.xlsx"]["recommended"] is True
    assert candidates["特征字典.xlsx"]["recommended"] is False
    assert candidates["特征字典.xlsx"]["metadata_compatibility"]["status"] == (
        "incompatible"
    )
    assert "importance column" in " ".join(
        candidates["特征字典.xlsx"]["metadata_compatibility"]["blocking_errors"]
    )
    assert payload["recommendation"] == {
        "dictionary_path": "03_模型评估材料.xlsx",
        "pmml_path": "model.pmml",
        "reason": "unique_pmml_compatible_feature_metadata",
    }
    assert payload["selection"]["dictionary_path"] == ""


def test_selected_pmml_enables_recommendation_when_multiple_pmml_files_exist(
    tmp_path: Path,
) -> None:
    source = _write_materials(tmp_path / "source", pmml_names=("model-a.pmml", "model-b.pmml"))
    _write_complete_metadata(source / "模型开发文档.xlsx")
    task = _create_task(tmp_path, source=source, pmml_path="model-b.pmml")

    payload = material_candidates_payload(task)

    assert payload["recommendation"] == {
        "dictionary_path": "模型开发文档.xlsx",
        "pmml_path": "model-b.pmml",
        "reason": "unique_pmml_compatible_feature_metadata",
    }


def test_neutral_named_table_is_evaluated_by_content(tmp_path: Path) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    ).to_csv(source / "变量清单.csv", index=False)
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidate = _dictionary_candidates_by_name(payload)["变量清单.csv"]
    assert candidate["metadata_compatibility"]["status"] == "compatible"
    assert candidate["recommended"] is True
    assert payload["recommendation"]["dictionary_path"] == "变量清单.csv"


def test_csv_without_required_aliases_stops_after_header(tmp_path: Path) -> None:
    metadata_path = tmp_path / "ordinary-sample.csv"
    metadata_path.write_text("customer_id,amount\n1,100\n", encoding="utf-8")
    manifest = parse_pmml_input_manifest(
        Path(__file__).parent / "fixtures" / "min_lr.pmml"
    )

    inspection = inspect_feature_metadata(metadata_path, manifest, max_rows=0)

    assert inspection.selections == ()
    assert any("missing feature column alias" in error for error in inspection.blocking_errors)


def test_explicit_dictionary_selection_is_never_overridden(tmp_path: Path) -> None:
    source = _write_materials(tmp_path / "source")
    _write_complete_metadata(source / "03_模型评估材料.xlsx")
    task = _create_task(
        tmp_path,
        source=source,
        pmml_path="model.pmml",
        dictionary_path="特征字典.xlsx",
    )

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert payload["selection"]["dictionary_path"] == "特征字典.xlsx"
    assert payload["recommendation"] is None
    assert candidates["03_模型评估材料.xlsx"]["recommended"] is False


def test_pmml_preview_ignores_persisted_dictionary_from_previous_pmml(
    tmp_path: Path,
) -> None:
    source = _write_materials(
        tmp_path / "source", pmml_names=("model-a.pmml", "model-b.pmml")
    )
    _write_complete_metadata(source / "模型开发文档.xlsx")
    task = _create_task(
        tmp_path,
        source=source,
        pmml_path="model-a.pmml",
        dictionary_path="特征字典.xlsx",
    )

    payload = material_candidates_payload(
        task, pmml_path_override="model-b.pmml"
    )

    assert payload["selection"]["dictionary_path"] == "特征字典.xlsx"
    assert payload["recommendation"]["dictionary_path"] == "模型开发文档.xlsx"
    assert payload["recommendation"]["pmml_path"] == "model-b.pmml"


def test_selected_sample_remains_selectable_when_name_looks_like_metadata(
    tmp_path: Path,
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "sample.csv").rename(source / "customer_metadata.csv")
    task = _create_task(
        tmp_path,
        source=source,
        sample_path="customer_metadata.csv",
    )

    payload = material_candidates_payload(task)

    assert payload["selection"]["sample_path"] == "customer_metadata.csv"
    assert "customer_metadata.csv" in {
        candidate["name"] for candidate in payload["candidates"][FileRole.SAMPLE.value]
    }


def test_multiple_compatible_workbooks_require_user_selection(tmp_path: Path) -> None:
    source = _write_materials(tmp_path / "source")
    _write_complete_metadata(source / "评估材料甲.xlsx")
    _write_complete_metadata(source / "评估材料乙.xlsx")
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert payload["recommendation"] is None
    assert candidates["评估材料甲.xlsx"]["metadata_compatibility"][
        "status"
    ] == "compatible"
    assert candidates["评估材料乙.xlsx"]["metadata_compatibility"][
        "status"
    ] == "compatible"
    assert candidates["评估材料甲.xlsx"]["recommended"] is False
    assert candidates["评估材料乙.xlsx"]["recommended"] is False


def test_multiple_unselected_pmml_files_disable_compatibility_recommendation(
    tmp_path: Path,
) -> None:
    source = _write_materials(
        tmp_path / "source", pmml_names=("model-a.pmml", "model-b.pmml")
    )
    _write_complete_metadata(source / "模型开发文档.xlsx")
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidate = _dictionary_candidates_by_name(payload)["模型开发文档.xlsx"]
    assert payload["recommendation"] is None
    assert "metadata_compatibility" not in candidate
    assert candidate["recommended"] is False


def test_pmml_preview_override_enables_recommendation_before_selection_is_saved(
    tmp_path: Path,
) -> None:
    source = _write_materials(
        tmp_path / "source", pmml_names=("model-a.pmml", "model-b.pmml")
    )
    _write_complete_metadata(source / "模型开发文档.xlsx")
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(
        task, pmml_path_override="model-b.pmml"
    )

    assert payload["recommendation"] == {
        "dictionary_path": "模型开发文档.xlsx",
        "pmml_path": "model-b.pmml",
        "reason": "unique_pmml_compatible_feature_metadata",
    }
    assert payload["selection"]["pmml_path"] == ""


def test_materials_endpoint_forwards_unsaved_pmml_preview_selection(
    tmp_path: Path,
) -> None:
    source = _write_materials(
        tmp_path / "source", pmml_names=("model-a.pmml", "model-b.pmml")
    )
    _write_complete_metadata(source / "模型开发文档.xlsx")
    task = _create_task(tmp_path, source=source)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                settings=SimpleNamespace(
                    db_path=tmp_path / "source-implicit.sqlite"
                )
            )
        )
    )

    payload = task_material_candidates(
        task.id, request, pmml_path="model-b.pmml"
    )

    assert payload["recommendation"]["dictionary_path"] == "模型开发文档.xlsx"
    assert payload["recommendation"]["pmml_path"] == "model-b.pmml"


def test_incomplete_bounded_evaluation_never_claims_unique_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    _write_complete_metadata(source / "评估材料甲.xlsx")
    _write_complete_metadata(source / "评估材料乙.xlsx")
    task = _create_task(tmp_path, source=source)
    monkeypatch.setattr(
        "marvis.api_scan_helpers.MAX_FEATURE_METADATA_CANDIDATES", 1
    )

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    statuses = {
        candidates["评估材料甲.xlsx"]["metadata_compatibility"]["status"],
        candidates["评估材料乙.xlsx"]["metadata_compatibility"]["status"],
    }
    assert statuses == {"compatible", "not_evaluated"}
    assert payload["recommendation"] is None
    assert candidates["评估材料甲.xlsx"]["recommended"] is False
    assert candidates["评估材料乙.xlsx"]["recommended"] is False


def test_inspector_resource_limit_never_claims_unique_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    _write_complete_metadata(source / "评估材料甲.xlsx")
    _write_complete_metadata(source / "评估材料乙.xlsx")
    task = _create_task(tmp_path, source=source)
    real_inspector = inspect_feature_metadata

    def bounded_inspector(path, manifest):
        if Path(path).name == "评估材料乙.xlsx":
            raise FeatureMetadataInspectionIncomplete(
                "feature metadata exceeds row limit"
            )
        return real_inspector(path, manifest)

    monkeypatch.setattr(
        "marvis.api_scan_helpers.inspect_feature_metadata", bounded_inspector
    )

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert candidates["评估材料甲.xlsx"]["metadata_compatibility"]["status"] == (
        "compatible"
    )
    assert candidates["评估材料乙.xlsx"]["metadata_compatibility"]["status"] == (
        "not_evaluated"
    )
    assert payload["recommendation"] is None
    assert candidates["评估材料甲.xlsx"]["recommended"] is False


def test_inspector_exposes_typed_incomplete_error_for_resource_limit(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    ).to_csv(metadata_path, index=False)
    manifest = parse_pmml_input_manifest(
        Path(__file__).parent / "fixtures" / "min_lr.pmml"
    )

    with pytest.raises(
        FeatureMetadataInspectionIncomplete,
        match="feature metadata exceeds row limit",
    ):
        inspect_feature_metadata(metadata_path, manifest, max_rows=1)


def test_mixed_sheet_resource_limit_cannot_hide_behind_valid_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    metadata_path = source / "metadata.xlsx"
    with pd.ExcelWriter(metadata_path) as writer:
        pd.DataFrame(
            {
                "feature": ["x1", "x2"],
                "category": ["内部", "征信"],
                "importance": [0.6, 0.4],
            }
        ).to_excel(writer, sheet_name="valid", index=False)
        pd.DataFrame(
            {
                "feature": ["x1", "x2", "x3"],
                "category": ["内部", "征信", "内部"],
                "importance": [0.6, 0.4, 0.1],
            }
        ).to_excel(writer, sheet_name="bounded", index=False)
    manifest = parse_pmml_input_manifest(
        Path(__file__).parent / "fixtures" / "min_lr.pmml"
    )

    with pytest.raises(
        FeatureMetadataInspectionIncomplete,
        match="feature metadata exceeds row limit",
    ):
        inspect_feature_metadata(metadata_path, manifest, max_rows=2)

    real_inspector = inspect_feature_metadata
    monkeypatch.setattr(
        "marvis.api_scan_helpers.inspect_feature_metadata",
        lambda path, candidate_manifest: real_inspector(
            path, candidate_manifest, max_rows=2
        ),
    )
    task = _create_task(tmp_path, source=source)
    payload = material_candidates_payload(task)
    candidate = _dictionary_candidates_by_name(payload)["metadata.xlsx"]

    assert candidate["metadata_compatibility"]["status"] == "not_evaluated"
    assert candidate["recommended"] is False
    assert payload["recommendation"] is None


def test_alias_work_budget_is_structured_as_incomplete_and_blocks_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    metadata_path = source / "metadata.csv"
    metadata_path.write_text(
        "feature,特征名,category,importance\n"
        "x1,x1,内部,0.6\n"
        "x2,x2,征信,0.4\n"
        "x3,x3,内部,0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(feature_metadata, "MAX_ALIAS_ROW_EVALUATIONS", 5)
    manifest = parse_pmml_input_manifest(
        Path(__file__).parent / "fixtures" / "min_lr.pmml"
    )

    inspection = inspect_feature_metadata(metadata_path, manifest)
    assert inspection.inspection_complete is False
    assert any(
        "alias row evaluation limit exceeded" in error
        for error in inspection.blocking_errors
    )

    task = _create_task(tmp_path, source=source)
    payload = material_candidates_payload(task)
    candidate = _dictionary_candidates_by_name(payload)["metadata.csv"]
    assert candidate["metadata_compatibility"]["status"] == "not_evaluated"
    assert candidate["recommended"] is False
    assert payload["recommendation"] is None


def test_oversized_candidate_never_creates_false_unique_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    small_metadata = source / "候选甲.csv"
    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    ).to_csv(small_metadata, index=False)
    oversized_metadata = source / "候选乙.xlsx"
    _write_complete_metadata(oversized_metadata)
    assert small_metadata.stat().st_size < oversized_metadata.stat().st_size
    monkeypatch.setattr(
        "marvis.api_scan_helpers.MAX_FEATURE_METADATA_CANDIDATE_FILE_BYTES",
        small_metadata.stat().st_size,
        raising=False,
    )
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert candidates["候选甲.csv"]["metadata_compatibility"]["status"] == (
        "compatible"
    )
    assert candidates["候选乙.xlsx"]["metadata_compatibility"]["status"] == (
        "not_evaluated"
    )
    assert payload["recommendation"] is None
    assert candidates["候选甲.csv"]["recommended"] is False
    assert candidates["候选乙.xlsx"]["recommended"] is False


def test_explicitly_named_oversized_sample_does_not_block_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_materials(tmp_path / "source")
    (source / "特征字典.xlsx").unlink()
    metadata_path = source / "候选甲.csv"
    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    ).to_csv(metadata_path, index=False)
    explicit_sample = source / "oot样本.csv"
    explicit_sample.write_text(
        "customer_id,value\n" + "1,2\n" * 100,
        encoding="utf-8",
    )
    assert metadata_path.stat().st_size < explicit_sample.stat().st_size
    monkeypatch.setattr(
        "marvis.api_scan_helpers.MAX_FEATURE_METADATA_CANDIDATE_FILE_BYTES",
        metadata_path.stat().st_size,
    )
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    candidates = _dictionary_candidates_by_name(payload)
    assert payload["recommendation"]["dictionary_path"] == "候选甲.csv"
    assert candidates["候选甲.csv"]["recommended"] is True
    assert "metadata_compatibility" not in candidates["oot样本.csv"]


def test_required_role_label_includes_feature_metadata(tmp_path: Path) -> None:
    source = _write_materials(tmp_path / "source")
    task = _create_task(tmp_path, source=source)

    payload = material_candidates_payload(task)

    dictionary_role = next(
        role
        for role in payload["required_roles"]
        if role["role"] == FileRole.DATA_DICTIONARY.value
    )
    assert dictionary_role["label"] == "数据字典/特征元数据"


def _write_materials(
    source: Path, *, pmml_names: tuple[str, ...] = ("model.pmml",)
) -> Path:
    source.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(cells=[]), source / "model.ipynb")
    pd.DataFrame({"x1": [0.0], "x2": [1.0], "y": [0]}).to_csv(
        source / "sample.csv", index=False
    )
    fixture = Path(__file__).parent / "fixtures" / "min_lr.pmml"
    for name in pmml_names:
        shutil.copy2(fixture, source / name)
    pd.DataFrame(
        {"特征名": ["x1", "x2"], "类别": ["内部", "征信"]}
    ).to_excel(source / "特征字典.xlsx", index=False)
    return source


def _write_complete_metadata(path: Path) -> None:
    pd.DataFrame(
        {
            "feature": ["x1", "x2"],
            "category": ["内部", "征信"],
            "importance": [0.6, 0.4],
        }
    ).to_excel(path, index=False)


def _create_task(
    tmp_path: Path,
    *,
    source: Path,
    sample_path: str = "sample.csv",
    pmml_path: str | None = None,
    dictionary_path: str | None = None,
):
    db_path = tmp_path / f"{source.name}-{pmml_path or 'implicit'}.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    return repo.create_task(
        TaskCreate(
            model_name="fixture",
            model_version="v2",
            validator="pytest",
            source_dir=str(source),
            notebook_path="model.ipynb",
            sample_path=sample_path,
            pmml_path=pmml_path,
            dictionary_path=dictionary_path,
        )
    )


def _dictionary_candidates_by_name(payload: dict) -> dict[str, dict]:
    return {
        candidate["name"]: candidate
        for candidate in payload["candidates"][FileRole.DATA_DICTIONARY.value]
    }
