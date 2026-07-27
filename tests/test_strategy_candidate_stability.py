from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import marvis.packs.strategy.candidate_stability as stability_module
from marvis.packs.strategy.candidate_stability import (
    CANDIDATE_STABILITY_MIN_MONTH_ROWS,
    CANDIDATE_STABILITY_SCHEMA_VERSION,
    CandidateStabilityError,
    build_candidate_stability_artifact,
    candidate_stability_artifact_content_hash,
    canonical_candidate_stability_artifact_json,
    validate_candidate_stability_artifact,
)
from marvis.validation.binning import compute_psi


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "month": [
                "202603",
                "202601",
                "202602",
                "202601",
                "202603",
                "202602",
                "202601",
                "202603",
                "202602",
                "202601",
            ],
            "bad": [0, 0, 0, 1, 1, 1, None, 1, None, 1],
        },
        index=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    )


def _hits() -> pd.Series:
    # 202601: 2/4, 202602: 1/3, 202603: 3/3; development: 6/10.
    return pd.Series(
        [True, False, True, True, True, False, True, True, False, False],
        index=_frame().index,
        dtype=bool,
    )


def _identity() -> dict:
    return {
        "task_id": "task-1",
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 2,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _asset_source() -> dict:
    return {
        "source_kind": "univariate_asset",
        "artifact_id": "artifact-1",
        "artifact_content_hash": HASH_D,
        "asset_id": "candidate-asset-1",
        "asset_hash": HASH_E,
        "rule_id": "candidate-rule-1",
    }


def _pool_source() -> dict:
    return {
        "source_kind": "pool_entry",
        "artifact_id": "pool-artifact-1",
        "artifact_content_hash": HASH_D,
        "pool_id": "strategy-pool-1",
        "revision": 4,
        "revision_id": "strategy-pool-revision-4",
        "snapshot_hash": HASH_E,
        "entry_id": "strategy-pool-entry-1",
        "rule_id": "candidate-rule-1",
    }


def _sample_ref() -> dict:
    return {
        "artifact_id": HASH_D,
        "artifact_content_hash": HASH_E,
        "sample_design_id": "strategy-sample-design-" + "1" * 24,
        "sample_design_content_hash": HASH_F,
        "partition": "development",
    }


def _build(**overrides) -> dict:
    values = {
        "frame": _frame(),
        "month_col": "month",
        "target_col": "bad",
        "hit_mask": _hits(),
        "basis": "asset_rule_hit",
        "identity": _identity(),
        "source_ref": _asset_source(),
        "sample_design_ref": _sample_ref(),
    }
    values.update(overrides)
    return build_candidate_stability_artifact(**values)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _rehash(document: dict) -> dict:
    body = {
        key: value
        for key, value in document.items()
        if key not in {"stability_id", "content_hash"}
    }
    document["stability_id"] = (
        "candidate-stability-"
        + hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()[:24]
    )
    without_hash = {
        key: value for key, value in document.items() if key != "content_hash"
    }
    document["content_hash"] = hashlib.sha256(
        _canonical(without_hash).encode("utf-8")
    ).hexdigest()
    return document


def test_builds_canonical_monthly_stability_against_full_development() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert first["stability_id"] == "candidate-stability-e70fc831907e714f4722fc5a"
    assert (
        first["content_hash"]
        == "e457d2bf9c98803aff3c1ea2f0731e5be17366d9c99485e71f67577a44003a9b"
    )
    assert first["schema_version"] == CANDIDATE_STABILITY_SCHEMA_VERSION
    assert first["basis"] == "asset_rule_hit"
    assert first["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    assert first["baseline"] == {
        "sample_count": 10,
        "hit_count": 6,
        "not_hit_count": 4,
        "hit_share": 0.6,
        "not_hit_share": 0.4,
        "labeled_count": 8,
        "label_coverage": 0.8,
        "hit_labeled_count": 5,
        "hit_bad_count": 3,
        "hit_bad_rate": 0.6,
        "psi_vs_development": 0.0,
    }
    assert [row["month"] for row in first["monthly"]] == [
        "202601",
        "202602",
        "202603",
    ]

    january, february, march = first["monthly"]
    assert (january["hit_count"], january["not_hit_count"]) == (2, 2)
    assert january["hit_bad_rate"] == 1.0
    assert (february["hit_count"], february["not_hit_count"]) == (1, 2)
    assert february["hit_bad_rate"] == 0.0
    assert (march["hit_count"], march["not_hit_count"]) == (3, 0)
    assert march["hit_bad_rate"] == pytest.approx(2 / 3)

    expected = np.asarray([0.4, 0.6])
    assert january["psi_vs_development"] == pytest.approx(
        compute_psi(expected, np.asarray([0.5, 0.5]))
    )
    assert february["psi_vs_development"] == pytest.approx(
        compute_psi(expected, np.asarray([2 / 3, 1 / 3]))
    )
    assert march["psi_vs_development"] == pytest.approx(
        compute_psi(expected, np.asarray([0.0, 1.0]))
    )
    assert first["summary"] == {
        "population_count": 10,
        "month_count": 3,
        "max_psi": march["psi_vs_development"],
        "max_psi_month": "202603",
        "insufficient_month_count": 3,
    }
    assert first["red_flags"] == [
        {
            "kind": "insufficient_month_rows",
            "month": month,
            "observed_rows": count,
            "minimum_rows": CANDIDATE_STABILITY_MIN_MONTH_ROWS,
        }
        for month, count in (("202601", 4), ("202602", 3), ("202603", 3))
    ]

    raw = canonical_candidate_stability_artifact_json(first)
    assert json.loads(raw) == first
    assert raw == canonical_candidate_stability_artifact_json(json.loads(raw))
    assert candidate_stability_artifact_content_hash(first) == first["content_hash"]
    assert hashlib.sha256(
        _canonical(
            {key: value for key, value in first.items() if key != "content_hash"}
        ).encode("utf-8")
    ).hexdigest() == first["content_hash"]


def test_pool_entry_basis_accepts_exact_pool_lineage() -> None:
    artifact = _build(
        basis="pool_entry_incremental_first_match",
        source_ref=_pool_source(),
    )

    assert artifact["basis"] == "pool_entry_incremental_first_match"
    assert artifact["source_ref"] == _pool_source()


def test_native_risk_development_reference_is_preserved_exactly() -> None:
    sample_ref = {
        **_sample_ref(),
        "partition": "risk/development",
    }

    artifact = _build(sample_design_ref=sample_ref)

    assert artifact["sample_design_ref"] == sample_ref


def test_missing_labels_only_reduce_label_metrics() -> None:
    frame = _frame()
    frame["bad"] = None
    artifact = _build(frame=frame)

    assert artifact["baseline"]["sample_count"] == 10
    assert artifact["baseline"]["hit_count"] == 6
    assert artifact["baseline"]["labeled_count"] == 0
    assert artifact["baseline"]["label_coverage"] == 0.0
    assert artifact["baseline"]["hit_labeled_count"] == 0
    assert artifact["baseline"]["hit_bad_count"] == 0
    assert artifact["baseline"]["hit_bad_rate"] is None
    assert all(row["hit_bad_rate"] is None for row in artifact["monthly"])


def test_zero_and_full_hit_distributions_keep_real_psi_values() -> None:
    no_hits = _build(
        hit_mask=pd.Series(False, index=_frame().index, dtype=bool)
    )
    all_hits = _build(
        hit_mask=pd.Series(True, index=_frame().index, dtype=bool)
    )

    assert no_hits["baseline"]["hit_share"] == 0.0
    assert all(row["psi_vs_development"] == 0.0 for row in no_hits["monthly"])
    assert all_hits["baseline"]["hit_share"] == 1.0
    assert all(row["psi_vs_development"] == 0.0 for row in all_hits["monthly"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frame": pd.DataFrame(columns=["month", "bad"])}, "must not be empty"),
        ({"month_col": "missing"}, "missing required columns"),
        ({"target_col": "missing"}, "missing required columns"),
        ({"target_col": "month"}, "must be different"),
        ({"hit_mask": [True]}, "length must match"),
        ({"hit_mask": [1] * 10}, "only booleans"),
        (
            {"hit_mask": pd.Series(_hits().to_numpy(), index=range(10))},
            "index must exactly match",
        ),
        ({"basis": "standalone"}, "basis must be"),
        (
            {"source_ref": _pool_source()},
            "source_kind does not match",
        ),
    ],
)
def test_rejects_invalid_population_mask_and_basis(
    overrides: dict,
    message: str,
) -> None:
    with pytest.raises(CandidateStabilityError, match=message):
        _build(**overrides)


@pytest.mark.parametrize("invalid", ["bad-date", None, "", "202613"])
def test_rejects_unparseable_month_values(invalid: object) -> None:
    frame = _frame()
    frame.iloc[0, frame.columns.get_loc("month")] = invalid

    with pytest.raises(CandidateStabilityError, match="month_col is invalid"):
        _build(frame=frame)


@pytest.mark.parametrize("invalid", [2, -1, "bad", float("inf")])
def test_rejects_non_binary_non_missing_targets(invalid: object) -> None:
    frame = _frame()
    if isinstance(invalid, str):
        frame["bad"] = frame["bad"].astype(object)
    frame.iloc[0, frame.columns.get_loc("bad")] = invalid

    with pytest.raises(CandidateStabilityError, match="binary 0/1"):
        _build(frame=frame)


def test_rejects_duplicate_columns_and_enforces_work_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = pd.DataFrame([[202601, 0]], columns=["month", "month"])
    with pytest.raises(CandidateStabilityError, match="columns must be unique"):
        _build(
            frame=duplicate,
            target_col="month",
            hit_mask=[True],
        )

    monkeypatch.setattr(stability_module, "CANDIDATE_STABILITY_MAX_ROWS", 9)
    with pytest.raises(CandidateStabilityError, match="row limit"):
        _build()
    monkeypatch.setattr(stability_module, "CANDIDATE_STABILITY_MAX_ROWS", 10)
    monkeypatch.setattr(stability_module, "CANDIDATE_STABILITY_MAX_MONTHS", 2)
    with pytest.raises(CandidateStabilityError, match="month limit"):
        _build()


def test_validator_rejects_rehashed_derived_drift_and_noncanonical_shape() -> None:
    artifact = _build()
    drifted = copy.deepcopy(artifact)
    drifted["monthly"][0]["hit_count"] = 3
    _rehash(drifted)
    with pytest.raises(CandidateStabilityError, match="do not cover"):
        validate_candidate_stability_artifact(drifted)

    drifted = copy.deepcopy(artifact)
    drifted["monthly"][0]["psi_vs_development"] = 0.0
    _rehash(drifted)
    with pytest.raises(CandidateStabilityError, match="does not match derived"):
        validate_candidate_stability_artifact(drifted)

    drifted = copy.deepcopy(artifact)
    drifted["summary"]["max_psi_month"] = "202601"
    _rehash(drifted)
    with pytest.raises(CandidateStabilityError, match="does not match monthly"):
        validate_candidate_stability_artifact(drifted)

    extra = copy.deepcopy(artifact)
    extra["unexpected"] = True
    with pytest.raises(CandidateStabilityError, match="unsupported fields"):
        validate_candidate_stability_artifact(extra)

    wrong_hash = copy.deepcopy(artifact)
    wrong_hash["content_hash"] = "0" * 64
    with pytest.raises(CandidateStabilityError, match="content_hash does not match"):
        validate_candidate_stability_artifact(wrong_hash)


def test_validator_rejects_fabricated_lifecycle_flags_even_when_rehashed() -> None:
    artifact = _build()
    artifact["lifecycle"]["not_adopted"] = False
    _rehash(artifact)

    with pytest.raises(CandidateStabilityError, match="non-mutating"):
        validate_candidate_stability_artifact(artifact)


def test_sample_reference_must_bind_development_partition() -> None:
    sample_ref = _sample_ref()
    sample_ref["partition"] = "validation"

    with pytest.raises(
        CandidateStabilityError,
        match="must be development or risk/development",
    ):
        _build(sample_design_ref=sample_ref)
