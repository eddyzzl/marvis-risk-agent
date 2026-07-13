from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import marvis.validation.platform_metrics as platform_metrics
from marvis.validation.binning import compute_ks
from marvis.validation.config import ValidationConfig
from marvis.validation.effectiveness import run_effectiveness
from marvis.validation.input_contracts import (
    FeatureMetadataResolution,
    FeatureMetadataRow,
    MetadataCoverage,
    TransformationSpec,
)
from marvis.validation.pmml_score_artifacts import run_pmml_scoring
from marvis.validation.results import StressBaseline, StressTestResult
from marvis.validation.sample_schema import inspect_sample_schema
from marvis.validation.sample_stats import run_basic_info


PMML_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "min_lr.pmml"


class _ScoreFromX1:
    def score_chunk(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(frame["x1"].to_numpy(dtype=float), dtype="float64")


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _sample_frame(*, matching_pred: bool = False) -> pd.DataFrame:
    scores = [
        0.10,
        0.40,
        0.20,
        0.80,
        0.30,
        0.90,
        0.15,
        0.85,
        0.25,
        0.75,
        0.35,
        0.95,
    ]
    return pd.DataFrame(
        {
            "x1": scores,
            "x2": [0.0, 1.0] * 6,
            "pred": scores if matching_pred else [0.99] * 12,
            "y": [0, 1, 0, 1] * 3,
            "split": ["train"] * 4 + ["test"] * 4 + ["oot"] * 4,
            "apply_month": [
                "202601",
                "202601",
                "202602",
                "202602",
                "202603",
                "202603",
                "202604",
                "202604",
                "202605",
                "202605",
                "202606",
                "202606",
            ],
        }
    )


def _ready_for_sample(ready_contract, sample_path: Path, **changes):
    schema = inspect_sample_schema(sample_path)
    contract = replace(
        ready_contract,
        sample_schema=schema,
        material_hashes={
            **ready_contract.material_hashes,
            "sample": schema.sha256,
            "pmml": _digest(PMML_FIXTURE),
        },
    )
    return replace(contract, **changes) if changes else contract


def _score_sample(tmp_path: Path, contract, sample_path: Path):
    score_path = tmp_path / "pmml_scores.parquet"
    result = run_pmml_scoring(
        contract=contract,
        sample_path=sample_path,
        pmml_path=PMML_FIXTURE,
        score_path=score_path,
        chunk_size=3,
        scorer=_ScoreFromX1(),
    )
    return score_path, result


def _settings():
    return SimpleNamespace(bin_count=2, random_sample_size=1_000, random_seed=42)


def _task():
    return SimpleNamespace(model_name="fixture", model_version="v1")


def _stress() -> StressTestResult:
    return StressTestResult(
        baseline=StressBaseline(ks=1.0, sample_count=4, bin_table=[]),
        per_category=[],
    )


def test_platform_metrics_uses_verified_pmml_sidecar_not_sample_score(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    frame = _sample_frame(matching_pred=False)
    frame.to_parquet(sample_path, index=False, row_group_size=3)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)

    results = platform_metrics.compute_platform_validation_results(
        task=_task(),
        contract=contract,
        sample_path=sample_path,
        score_path=score_path,
        scoring_result=scoring,
        metadata_resolution=contract.require_feature_metadata(),
        stress_test=_stress(),
        settings=_settings(),
    )

    oot = frame[frame["split"] == "oot"]
    expected = compute_ks(oot["x1"].tolist(), oot["y"].tolist())
    actual = {row.split: row for row in results.effectiveness.overall}["oot"].ks
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(
        compute_ks(oot["pred"].tolist(), oot["y"].tolist())
    )
    assert results.schema_version == "marvis.validation_results.v2"
    assert results.pmml_scoring == scoring
    assert results.reproducibility is None


def test_pmml_metrics_match_every_legacy_basic_info_and_effectiveness_field(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    frame = _sample_frame(matching_pred=True)
    frame.to_parquet(sample_path, index=False, row_group_size=2)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    new = platform_metrics.compute_platform_validation_results(
        task=_task(),
        contract=contract,
        sample_path=sample_path,
        score_path=score_path,
        scoring_result=scoring,
        metadata_resolution=contract.require_feature_metadata(),
        stress_test=_stress(),
        settings=_settings(),
    )

    legacy_config = ValidationConfig(
        target_col="y",
        score_col="pred",
        split_col="split",
        time_col="apply_month",
        feature_columns=[],
        bin_count=2,
        random_sample_size=1_000,
        random_seed=42,
        score_decimal_places=6,
    )
    meta_path = tmp_path / "model_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "hyperparameters": {},
                "feature_importance": [
                    {
                        "feature": row.feature,
                        "category": row.category,
                        "importance": row.importance,
                    }
                    for row in contract.require_feature_metadata().rows
                    if row.in_pmml
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    legacy_basic = run_basic_info(
        sample=frame,
        config=legacy_config,
        model_meta_path=meta_path,
    )
    legacy_effectiveness = run_effectiveness(sample=frame, config=legacy_config)

    assert asdict(new.basic_info) == asdict(legacy_basic)
    assert asdict(new.effectiveness) == asdict(legacy_effectiveness)


def test_analysis_loads_only_control_transformation_closure(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "transformed.parquet"
    pd.DataFrame(
        {
            "x1": [0.10, 0.70, 0.20, 0.80, 0.30, 0.90],
            "x2": [0.0, 1.0] * 3,
            "raw_y": [0, 1] * 3,
            "raw_split": ["train", "train", "test", "test", "oot", "oot"],
            "raw_time": ["202601", "202601", "202602", "202602", "202603", "202603"],
            "poison": [object.__name__] * 6,
            "pred": [0.99] * 6,
        }
    ).to_parquet(sample_path, index=False)
    transformations = (
        TransformationSpec("copy", "y", ("raw_y",), {}),
        TransformationSpec("copy", "split", ("raw_split",), {}),
        TransformationSpec("copy", "apply_month", ("raw_time",), {}),
        TransformationSpec("copy", "unused", ("poison",), {}),
    )
    contract = _ready_for_sample(
        ready_contract,
        sample_path,
        transformations=transformations,
    )
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    selected: list[tuple[str, ...]] = []
    original = platform_metrics.read_selected_columns

    def record(path, *, columns, **kwargs):
        selected.append(columns)
        return original(path, columns=columns, **kwargs)

    monkeypatch.setattr(platform_metrics, "read_selected_columns", record)
    loaded = platform_metrics.load_pmml_analysis_frame(
        sample_path=sample_path,
        score_path=score_path,
        contract=contract,
        scoring_result=scoring,
    )

    assert selected == [("raw_y", "raw_split", "raw_time")]
    assert loaded.columns.tolist() == [
        "__target__",
        "__split__",
        "__time__",
        "__pmml_score__",
    ]


def test_split_mapping_is_type_stable_for_bool_int_and_string():
    values = pd.Series([True, 1, "1"], dtype="object")

    result = platform_metrics._canonical_split_series(
        values,
        {"train": True, "test": 1, "oot": "1"},
    )

    assert result.tolist() == ["train", "test", "oot"]


def test_non_pmml_metadata_is_excluded_from_importance(
    tmp_path: Path, ready_contract
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    base = ready_contract.require_feature_metadata()
    metadata = FeatureMetadataResolution(
        schema_version=base.schema_version,
        rows=base.rows + (FeatureMetadataRow("note", "其他", 9.0, "features", False),),
        coverage=MetadataCoverage(1.0, 1.0, 1.0, 1.0),
        per_category_raw_fields=base.per_category_raw_fields,
        extra_features=("note",),
        conflicts=(),
    )
    contract = _ready_for_sample(
        ready_contract,
        sample_path,
        feature_metadata=metadata,
    )
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)

    results = platform_metrics.compute_platform_validation_results(
        task=_task(),
        contract=contract,
        sample_path=sample_path,
        score_path=score_path,
        scoring_result=scoring,
        metadata_resolution=metadata,
        stress_test=_stress(),
        settings=_settings(),
    )

    assert [row.feature for row in results.basic_info.feature_importance] == ["x1", "x2"]


def test_metrics_rejects_sample_changed_after_scoring(tmp_path: Path, ready_contract):
    sample_path = tmp_path / "sample.parquet"
    frame = _sample_frame()
    frame.to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    frame.assign(x1=frame["x1"] + 1).to_parquet(sample_path, index=False)

    with pytest.raises(ValueError, match="confirmed SHA-256"):
        platform_metrics.load_pmml_analysis_frame(
            sample_path=sample_path,
            score_path=score_path,
            contract=contract,
            scoring_result=scoring,
        )


def test_metrics_rejects_tampered_score_sidecar(tmp_path: Path, ready_contract):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    scores = pd.read_parquet(score_path)
    scores.loc[0, "pmml_score"] = 0.123456
    scores.to_parquet(score_path, index=False)

    with pytest.raises(ValueError, match="hash mismatch"):
        platform_metrics.load_pmml_analysis_frame(
            sample_path=sample_path,
            score_path=score_path,
            contract=contract,
            scoring_result=scoring,
        )


def test_metrics_rejects_sidecar_replaced_after_verification(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    original = platform_metrics.validate_pmml_score_artifact

    def validate_then_replace(*args, **kwargs):
        result = original(*args, **kwargs)
        scores = pd.read_parquet(score_path)
        scores.loc[0, "pmml_score"] = float("inf")
        scores.to_parquet(score_path, index=False)
        return result

    monkeypatch.setattr(
        platform_metrics,
        "validate_pmml_score_artifact",
        validate_then_replace,
    )

    with pytest.raises(ValueError, match="non-finite score"):
        platform_metrics.load_pmml_analysis_frame(
            sample_path=sample_path,
            score_path=score_path,
            contract=contract,
            scoring_result=scoring,
        )


def test_metrics_rejects_finite_sidecar_replacement_by_fingerprint(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    original = platform_metrics.validate_pmml_score_artifact

    def validate_then_replace(*args, **kwargs):
        result = original(*args, **kwargs)
        scores = pd.read_parquet(score_path)
        scores.loc[0, "pmml_score"] = 0.123456
        scores.to_parquet(score_path, index=False)
        return result

    monkeypatch.setattr(
        platform_metrics,
        "validate_pmml_score_artifact",
        validate_then_replace,
    )

    with pytest.raises(ValueError, match="changed while metrics were loading"):
        platform_metrics.load_pmml_analysis_frame(
            sample_path=sample_path,
            score_path=score_path,
            contract=contract,
            scoring_result=scoring,
        )


def test_metrics_hashes_current_sample_once_then_uses_read_fingerprint(
    tmp_path: Path, ready_contract, monkeypatch
):
    sample_path = tmp_path / "sample.parquet"
    _sample_frame().to_parquet(sample_path, index=False)
    contract = _ready_for_sample(ready_contract, sample_path)
    score_path, scoring = _score_sample(tmp_path, contract, sample_path)
    original = platform_metrics.sha256_file_cancellable
    sample_hashes = 0

    def record(path, *args, **kwargs):
        nonlocal sample_hashes
        if Path(path) == sample_path:
            sample_hashes += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(platform_metrics, "sha256_file_cancellable", record)
    platform_metrics.load_pmml_analysis_frame(
        sample_path=sample_path,
        score_path=score_path,
        contract=contract,
        scoring_result=scoring,
    )

    assert sample_hashes == 1


def test_existing_effectiveness_rejects_missing_required_split():
    frame = pd.DataFrame(
        {
            "__target__": [0, 1, 0, 1],
            "__split__": ["train", "train", "test", "test"],
            "__time__": ["202601", "202601", "202602", "202602"],
            "__pmml_score__": [0.1, 0.9, 0.2, 0.8],
        }
    )
    config = ValidationConfig(
        target_col="__target__",
        score_col="__pmml_score__",
        split_col="__split__",
        time_col="__time__",
    )

    with pytest.raises(ValueError, match="oot"):
        platform_metrics.compute_existing_effectiveness(frame, config)
