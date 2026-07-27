from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_fragment import build_verified_candidate_fragment
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import add_verified_candidate_fragment
from marvis.packs.strategy.pool_validation import (
    STRATEGY_POOL_VALIDATION_SCHEMA_VERSION,
    build_strategy_pool_validation_evidence,
    canonical_strategy_pool_validation_json,
    validate_strategy_pool_validation_evidence,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _action(action_type: str) -> dict:
    return {
        "type": action_type,
        "value": "approve" if action_type == "approval" else action_type,
        "reason_code": None if action_type == "approval" else "RISK",
        "stop": True,
    }


def _sample_identity() -> dict:
    return {
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _legacy_ref() -> dict:
    return {
        "artifact_id": "e" * 64,
        "artifact_content_hash": "f" * 64,
        "sample_design_id": "strategy-sample-design-" + "1" * 24,
        "sample_design_content_hash": "2" * 64,
        "partition": "development",
    }


def _condition(operator: str, value: int) -> dict:
    return {
        "op": "compare",
        "field": "score",
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }


def _fragment(index: int, condition: dict) -> dict:
    suffix = f"{index:064x}"
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{index}",
            "artifact_kind": "test_candidate_json",
            "artifact_schema_version": "test.candidate-artifact.v1",
            "artifact_content_hash": suffix,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "test.candidate.v1",
            "asset_id": f"candidate-asset-{index}",
            "asset_hash": suffix,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-{index}",
        condition=condition,
        requirements=[],
        effect_id=f"candidate-effect-{index}",
        evidence_id="candidate-evidence-1",
        evidence_hash=HASH_D,
        evidence_identity=_sample_identity(),
    )


def _pool() -> dict:
    result = None
    for index, condition in enumerate(
        (_condition("<", 5), _condition("<", 8)),
        start=1,
    ):
        result = add_verified_candidate_fragment(
            result,
            task_id="task-1",
            strategy_type="approval",
            default_action=_action("approval"),
            verified_candidate_fragment=_fragment(index, condition),
            action=_action("reject" if index == 1 else "review"),
        )
    assert result is not None
    return result


def _frame(*, target_bad_value: int = 1) -> pd.DataFrame:
    target = [1, 1, 0, 1, 0, 1, None, 0, 0, 1]
    if target_bad_value == 0:
        target = [None if value is None else 1 - value for value in target]
    return pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bad": target,
            "month": ["202601"] * 5 + ["202602"] * 5,
            "loan": [100, 100, 100, 100, None, 200, 200, 200, 200, 200],
            "overdue": [10, 0, 0, 5, 0, 20, None, 0, 0, 10],
        }
    )


def _pool_artifact_ref(pool: dict) -> dict:
    return {
        "artifact_id": "3" * 64,
        "artifact_content_hash": "4" * 64,
    }


def _sample_design_v2_ref(*, partition: str, count: int) -> dict:
    return {
        "membership_artifact_id": "5" * 64,
        "membership_artifact_content_hash": "6" * 64,
        "membership_id": "strategy-sample-membership-" + "7" * 24,
        "membership_content_hash": "8" * 64,
        "bundle_artifact_id": "9" * 64,
        "bundle_artifact_content_hash": "a" * 64,
        "bundle_id": "strategy-sample-design-v2-bundle-" + "b" * 24,
        "bundle_content_hash": "c" * 64,
        "sample_design_id": "strategy-sample-design-v2-" + "d" * 24,
        "sample_design_content_hash": "e" * 64,
        "partition_key": f"risk/{partition}",
        "partition_count": count,
        "analysis_universe_row_count": 30,
    }


def _dataset_binding() -> dict:
    return {
        "task_id": "task-1",
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "dataset_source_path": "task-1/sample.parquet",
        "dataset_registry_metadata_hash": "f" * 64,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
    }


def _build(
    *,
    partition: str = "validation",
    frame: pd.DataFrame | None = None,
    target_bad_value: int = 1,
    expected_partition_count: int | None = None,
    pool: dict | None = None,
    dataset_binding: dict | None = None,
) -> dict:
    selected = _frame(target_bad_value=target_bad_value) if frame is None else frame
    count = len(selected) if expected_partition_count is None else expected_partition_count
    current_pool = _pool() if pool is None else pool
    return build_strategy_pool_validation_evidence(
        pool=current_pool,
        frame=selected,
        pool_artifact_ref=_pool_artifact_ref(current_pool),
        sample_design_v2_ref=_sample_design_v2_ref(
            partition=partition,
            count=count,
        ),
        dataset_binding=(
            _dataset_binding() if dataset_binding is None else dataset_binding
        ),
        legacy_development_ref=_legacy_ref(),
        partition=partition,
        population="risk",
        comparison_mode="absolute",
        target_col="bad",
        target_bad_value=target_bad_value,
        month_col="month",
        loan_amount_col="loan",
        overdue_amount_col="overdue",
        development_rows_excluded=True,
    )


def _rehash(document: dict) -> dict:
    body = {
        key: value
        for key, value in document.items()
        if key not in {"evidence_id", "content_hash"}
    }
    encoded_body = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    document["evidence_id"] = (
        "strategy-pool-validation-"
        + hashlib.sha256(encoded_body.encode("utf-8")).hexdigest()[:24]
    )
    without_hash = {
        key: value for key, value in document.items() if key != "content_hash"
    }
    encoded = json.dumps(
        without_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    document["content_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return document


@pytest.mark.parametrize("partition", ["validation", "oot"])
def test_pool_validation_builds_independent_partition_evidence(
    partition: str,
) -> None:
    evidence = _build(partition=partition)

    assert evidence["schema_version"] == STRATEGY_POOL_VALIDATION_SCHEMA_VERSION
    assert evidence["partition"] == partition
    assert evidence["population"] == "risk"
    assert evidence["comparison_mode"] == "absolute"
    assert evidence["lifecycle"] == {
        "stage": partition,
        "validation_status": "independent_evidence",
        "mutates_pool": False,
        "creates_strategy": False,
        "adopts_strategy": False,
        "promotes_strategy": False,
        "deploys_strategy": False,
    }
    assert evidence["population_metrics"] == {
        "population_count": 10,
        "labelled_count": 9,
        "unlabelled_count": 1,
        "label_coverage": 0.9,
    }
    assert evidence["source_bindings"]["sample_design_v2"][
        "partition_key"
    ] == f"risk/{partition}"
    assert (
        evidence["source_bindings"]["development_lineage"][
            "legacy_development_ref"
        ]
        == _legacy_ref()
    )
    assert "sample_design_ref" not in evidence["source_bindings"]
    assert validate_strategy_pool_validation_evidence(evidence) == evidence
    assert json.loads(canonical_strategy_pool_validation_json(evidence)) == evidence


def test_pool_validation_uses_v2_target_polarity_and_retains_unlabelled_rows() -> None:
    bad_one = _build(target_bad_value=1)
    bad_zero = _build(target_bad_value=0)

    assert bad_zero["source_bindings"]["target"] == {
        "column": "bad",
        "good_value": 1,
        "bad_value": 0,
        "missing_policy": "retain_population_exclude_risk_denominator",
    }
    assert bad_zero["population_metrics"] == bad_one["population_metrics"]
    assert bad_zero["overall"] == bad_one["overall"]
    assert [
        (row["standalone"], row["incremental"], row["shadowed"])
        for row in bad_zero["waterfall"]
    ] == [
        (row["standalone"], row["incremental"], row["shadowed"])
        for row in bad_one["waterfall"]
    ]
    assert bad_zero["population_metrics"]["population_count"] == 10
    assert bad_zero["population_metrics"]["labelled_count"] == 9
    assert bad_zero["population_metrics"]["unlabelled_count"] == 1


def test_pool_validation_empty_partition_fails_clearly() -> None:
    with pytest.raises(StrategyError, match="validation partition is empty"):
        _build(
            frame=_frame().iloc[:0].copy(),
            expected_partition_count=0,
        )


def test_pool_validation_rejects_pool_and_v2_dataset_mismatch() -> None:
    drifted = _dataset_binding()
    drifted["dataset_content_hash"] = "0" * 64

    with pytest.raises(
        StrategyError,
        match="development lineage dataset_content_hash",
    ):
        _build(dataset_binding=drifted)


def test_pool_validation_waterfall_and_population_conserve_without_conditions() -> None:
    evidence = _build()
    first, second = evidence["waterfall"]

    assert first["standalone"]["population_count"] == 4
    assert first["incremental"]["population_count"] == 4
    assert first["shadowed"]["population_count"] == 0
    assert second["standalone"]["population_count"] == 7
    assert second["incremental"]["population_count"] == 3
    assert second["shadowed"]["population_count"] == 4
    assert evidence["default_unmatched"]["effect"]["population_count"] == 3
    assert (
        sum(
            row["incremental"]["population_count"]
            for row in evidence["waterfall"]
        )
        + evidence["default_unmatched"]["effect"]["population_count"]
        == evidence["population_metrics"]["population_count"]
    )
    encoded = canonical_strategy_pool_validation_json(evidence)
    assert '"condition"' not in encoded
    assert '"strategy_spec"' not in encoded


def test_pool_validation_canonical_hash_and_coherent_metric_tamper_fail_closed() -> None:
    evidence = _build()
    tampered_hash = copy.deepcopy(evidence)
    tampered_hash["population_metrics"]["population_count"] = 11
    with pytest.raises(StrategyError, match="content_hash"):
        validate_strategy_pool_validation_evidence(tampered_hash)

    coherently_rehashed = copy.deepcopy(evidence)
    coherently_rehashed["population_metrics"]["population_count"] = 11
    coherently_rehashed["source_bindings"]["sample_design_v2"][
        "partition_count"
    ] = 11
    _rehash(coherently_rehashed)
    with pytest.raises(StrategyError, match="overall effect|population"):
        validate_strategy_pool_validation_evidence(coherently_rehashed)
