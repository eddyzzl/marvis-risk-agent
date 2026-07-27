from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from marvis.feature.univariate import analyze_univariate
from marvis.packs.strategy.candidate_asset import (
    canonical_candidate_asset_json,
    refine_univariate_candidate,
)
from marvis.packs.strategy.candidate_evidence import (
    MetricObservation,
    build_candidate_evidence,
)
from marvis.packs.strategy.candidate_fragment import (
    VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION,
    CandidateFragmentError,
    build_verified_candidate_fragment,
    sample_context_hash_from_candidate_evidence,
    univariate_asset_to_verified_fragment,
    validate_verified_candidate_fragment,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _sample_design_ref(
    *,
    partition: str = "development",
) -> dict[str, str]:
    return {
        "artifact_id": "d" * 64,
        "artifact_content_hash": "e" * 64,
        "sample_design_id": "strategy-sample-design-test",
        "sample_design_content_hash": "f" * 64,
        "partition": partition,
    }


def _sample_design_source_token(
    *,
    kind: str = "strategy_sample_design",
    reference: dict[str, str] | None = None,
) -> str:
    payload = {
        "kind": kind,
        **(reference or _sample_design_ref()),
    }
    return "strategy-sample-design:" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [100, 130, 160, 190, 220, 250, 280, 310],
            "bad": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )


def _evidence(
    *,
    sample_design_ref: dict[str, str] | None = None,
    sample_design_kind: str = "strategy_sample_design",
    sample_design_token: str | None = None,
) -> dict:
    frame = _frame()
    reference = sample_design_ref or _sample_design_ref()
    analysis = analyze_univariate(
        frame,
        features=["score"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
    )
    return build_candidate_evidence(
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=2,
        semantic_mapping_hash=HASH_B,
        generation_parameters={
            "analysis_schema_version": analysis["schema_version"],
            "target_col": "bad",
            "drop_nan_labels": False,
            "nan_labels_dropped": 0,
            "features": ["score"],
            "methods": ["equal_width"],
            "loan_amount_col": None,
            "overdue_amount_col": None,
            "sample_design_ref": reference,
        },
        seed=0,
        budget=100_000,
        truncated=False,
        analysis=analysis,
        metrics=[
            MetricObservation("parent.iv", "count", "observed", 0.2),
            MetricObservation("parent.iv", "loan_amount", "unavailable", None),
            MetricObservation("parent.iv", "overdue_amount", "unavailable", None),
        ],
        source_refs=[
            "dataset:dataset-1",
            "analysis:univariate-1",
            sample_design_token
            or _sample_design_source_token(
                kind=sample_design_kind,
                reference=reference,
            ),
        ],
    )


def _asset_and_binding(
    *,
    sample_design_ref: dict[str, str] | None = None,
    sample_design_kind: str = "strategy_sample_design",
) -> tuple[dict, dict, dict]:
    evidence = _evidence(
        sample_design_ref=sample_design_ref,
        sample_design_kind=sample_design_kind,
    )
    source_bin = evidence["analysis"]["features"][0]["methods"][0]["bins"][0]
    asset = refine_univariate_candidate(
        evidence,
        _frame(),
        source_evidence={
            "artifact_id": "artifact-parent",
            "kind": "strategy_candidate_json",
            "content_hash": HASH_C,
        },
        feature="score",
        method="equal_width",
        merge_groups=[],
        selection={"source_bin_ids": [source_bin["id"]]},
    )
    identity = evidence["identity"]
    binding = {
        "artifact_id": "artifact-asset",
        "kind": "strategy_candidate_asset_json",
        "content_hash": HASH_C,
        "origin_tool": "strategy.refine_univariate_candidate",
        "artifact_schema_version": "strategy.candidate-asset-artifact.v1",
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": asset["asset_type"],
        "fragment_id": asset["rule"]["rule_id"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": asset["effect_stage"],
        "validation_status": asset["validation_status"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
        "evidence_identity": {
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
        },
    }
    return asset, binding, evidence


def test_univariate_adapter_preserves_asset_bytes_and_projects_strict_fragment() -> None:
    asset, binding, evidence = _asset_and_binding()
    before = canonical_candidate_asset_json(asset)
    assert hashlib.sha256(before.encode("utf-8")).hexdigest() == (
        "2828f2d494f1b5a18ef09ff290f378e4"
        "22032c96edbf50fafc3372d7c2bf703c"
    )
    asset_id = asset["asset_id"]
    asset_hash = asset["asset_hash"]

    fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=binding,
        candidate_evidence=evidence,
    )

    assert canonical_candidate_asset_json(asset) == before
    assert asset["asset_id"] == asset_id
    assert asset["asset_hash"] == asset_hash
    assert fragment["schema_version"] == VERIFIED_CANDIDATE_FRAGMENT_SCHEMA_VERSION
    assert fragment["asset"] == {
        "schema_version": "strategy.candidate-asset.v1",
        "asset_id": asset_id,
        "asset_hash": asset_hash,
        "asset_type": "univariate_refinement",
    }
    assert fragment["fragment"]["rule_id"] == asset["rule"]["rule_id"]
    assert fragment["fragment"]["fragment_id"] != asset["rule"]["rule_id"]
    assert fragment["candidate_stage"] == "development"
    assert fragment["observation_stage"] == "backtested"
    assert fragment["validation_status"] == "unvalidated"
    assert fragment["evidence"]["identity"]["sample_context_hash"] == (
        sample_context_hash_from_candidate_evidence(evidence)
    )
    assert validate_verified_candidate_fragment(fragment) == fragment


def test_sample_context_requires_and_hashes_exact_sample_design_ref() -> None:
    evidence = _evidence()
    original_hash = sample_context_hash_from_candidate_evidence(evidence)
    assert original_hash == (
        "675e33f6af577aa81bcacf853c2e6f5ee"
        "7267257c4026debad77c1055d11d5a0"
    )
    assert evidence["source_refs"][-1] == (
        "strategy-sample-design:"
        '{"artifact_content_hash":"'
        + "e" * 64
        + '","artifact_id":"'
        + "d" * 64
        + '","kind":"strategy_sample_design","partition":"development",'
        '"sample_design_content_hash":"'
        + "f" * 64
        + '","sample_design_id":"strategy-sample-design-test"}'
    )

    changed = deepcopy(evidence)
    changed["generation"]["parameters"]["sample_design_ref"] = {
        **_sample_design_ref(),
        "sample_design_content_hash": "0" * 64,
    }
    changed["source_refs"][-1] = _sample_design_source_token(
        reference=changed["generation"]["parameters"]["sample_design_ref"]
    )
    changed.pop("candidate_id")
    changed.pop("evidence_hash")
    changed = build_candidate_evidence(
        task_id=changed["identity"]["task_id"],
        dataset_id=changed["identity"]["dataset_id"],
        dataset_content_hash=changed["identity"]["dataset_content_hash"],
        workspace_revision=changed["identity"]["workspace_revision"],
        workspace_generation=changed["identity"]["workspace_generation"],
        semantic_mapping_hash=changed["identity"]["semantic_mapping_hash"],
        generation_parameters=changed["generation"]["parameters"],
        seed=changed["generation"]["seed"],
        budget=changed["generation"]["budget"],
        truncated=changed["generation"]["truncated"],
        analysis=changed["analysis"],
        metrics=changed["metrics"],
        source_refs=changed["source_refs"],
        red_flags=changed["red_flags"],
        producer_version=changed["producer_version"],
    )
    assert sample_context_hash_from_candidate_evidence(changed) != original_hash

    missing = _evidence()
    del missing["generation"]["parameters"]["sample_design_ref"]
    missing.pop("candidate_id")
    missing.pop("evidence_hash")
    missing = build_candidate_evidence(
        task_id=missing["identity"]["task_id"],
        dataset_id=missing["identity"]["dataset_id"],
        dataset_content_hash=missing["identity"]["dataset_content_hash"],
        workspace_revision=missing["identity"]["workspace_revision"],
        workspace_generation=missing["identity"]["workspace_generation"],
        semantic_mapping_hash=missing["identity"]["semantic_mapping_hash"],
        generation_parameters=missing["generation"]["parameters"],
        seed=missing["generation"]["seed"],
        budget=missing["generation"]["budget"],
        truncated=missing["generation"]["truncated"],
        analysis=missing["analysis"],
        metrics=missing["metrics"],
        source_refs=missing["source_refs"],
        red_flags=missing["red_flags"],
        producer_version=missing["producer_version"],
    )
    with pytest.raises(CandidateFragmentError, match="sample_design_ref"):
        sample_context_hash_from_candidate_evidence(missing)


def test_sample_context_accepts_exact_native_risk_development_lineage() -> None:
    native_ref = _sample_design_ref(partition="risk/development")
    evidence = _evidence(
        sample_design_ref=native_ref,
        sample_design_kind="strategy_sample_design_v2",
    )

    assert sample_context_hash_from_candidate_evidence(evidence) != (
        "675e33f6af577aa81bcacf853c2e6f5ee"
        "7267257c4026debad77c1055d11d5a0"
    )
    assert evidence["source_refs"][-1] == _sample_design_source_token(
        kind="strategy_sample_design_v2",
        reference=native_ref,
    )


def test_native_univariate_adapter_preserves_asset_bytes_and_projects_fragment() -> None:
    native_ref = _sample_design_ref(partition="risk/development")
    asset, binding, evidence = _asset_and_binding(
        sample_design_ref=native_ref,
        sample_design_kind="strategy_sample_design_v2",
    )
    before = canonical_candidate_asset_json(asset)

    fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=binding,
        candidate_evidence=evidence,
    )

    assert canonical_candidate_asset_json(asset) == before
    assert fragment["evidence"]["identity"]["sample_context_hash"] == (
        sample_context_hash_from_candidate_evidence(evidence)
    )
    assert validate_verified_candidate_fragment(fragment) == fragment


@pytest.mark.parametrize(
    ("kind", "partition"),
    [
        ("strategy_sample_design", "risk/development"),
        ("strategy_sample_design_v2", "development"),
        ("strategy_sample_design_v2", "risk/validation"),
        ("strategy_sample_design_future", "risk/development"),
    ],
)
def test_sample_context_rejects_unknown_or_crossed_sample_lineage(
    kind: str,
    partition: str,
) -> None:
    reference = _sample_design_ref(partition=partition)
    evidence = _evidence(
        sample_design_ref=reference,
        sample_design_kind=kind,
    )

    with pytest.raises(CandidateFragmentError, match="sample_design_ref"):
        sample_context_hash_from_candidate_evidence(evidence)


def test_sample_context_rejects_noncanonical_or_duplicate_sample_tokens() -> None:
    reference = _sample_design_ref()
    noncanonical = "strategy-sample-design:" + json.dumps(
        {"kind": "strategy_sample_design", **reference},
        ensure_ascii=False,
    )
    evidence = _evidence(
        sample_design_ref=reference,
        sample_design_token=noncanonical,
    )
    with pytest.raises(CandidateFragmentError, match="sample_design_ref"):
        sample_context_hash_from_candidate_evidence(evidence)

    mismatched_reference = {
        **reference,
        "sample_design_content_hash": "0" * 64,
    }
    mismatched = _evidence(
        sample_design_ref=reference,
        sample_design_token=_sample_design_source_token(
            reference=mismatched_reference,
        ),
    )
    with pytest.raises(CandidateFragmentError, match="sample_design_ref"):
        sample_context_hash_from_candidate_evidence(mismatched)

    duplicated_source = _evidence()
    second_token = _sample_design_source_token(
        kind="strategy_sample_design_v2",
        reference=_sample_design_ref(partition="risk/development"),
    )
    duplicated = build_candidate_evidence(
        task_id=duplicated_source["identity"]["task_id"],
        dataset_id=duplicated_source["identity"]["dataset_id"],
        dataset_content_hash=duplicated_source["identity"][
            "dataset_content_hash"
        ],
        workspace_revision=duplicated_source["identity"]["workspace_revision"],
        workspace_generation=duplicated_source["identity"][
            "workspace_generation"
        ],
        semantic_mapping_hash=duplicated_source["identity"][
            "semantic_mapping_hash"
        ],
        generation_parameters=duplicated_source["generation"]["parameters"],
        seed=duplicated_source["generation"]["seed"],
        budget=duplicated_source["generation"]["budget"],
        truncated=duplicated_source["generation"]["truncated"],
        analysis=duplicated_source["analysis"],
        metrics=duplicated_source["metrics"],
        source_refs=[*duplicated_source["source_refs"], second_token],
        red_flags=duplicated_source["red_flags"],
        producer_version=duplicated_source["producer_version"],
    )
    with pytest.raises(CandidateFragmentError, match="sample_design_ref"):
        sample_context_hash_from_candidate_evidence(duplicated)


def test_fragment_hash_and_nested_tampering_fail_closed() -> None:
    asset, binding, evidence = _asset_and_binding()
    fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=binding,
        candidate_evidence=evidence,
    )

    for mutation, message in (
        (("fragment", "rule_id", "forged-rule"), "fragment_hash"),
        (("fragment_hash", None, HASH_A), "fragment_hash"),
        (("observation_stage", None, "validated"), "backtested"),
    ):
        forged = deepcopy(fragment)
        top, child, value = mutation
        if child is None:
            forged[top] = value
        else:
            forged[top][child] = value
        with pytest.raises(CandidateFragmentError, match=message):
            validate_verified_candidate_fragment(forged)


def test_univariate_adapter_rejects_unknown_artifact_or_asset_contract() -> None:
    asset, binding, evidence = _asset_and_binding()
    unknown = deepcopy(binding)
    unknown["kind"] = "strategy_tree_candidate_json"
    with pytest.raises(CandidateFragmentError, match="strategy_candidate_asset_json"):
        univariate_asset_to_verified_fragment(
            asset,
            source_binding=unknown,
            candidate_evidence=evidence,
        )

    unknown_schema = deepcopy(binding)
    unknown_schema["artifact_schema_version"] = "strategy.future-artifact.v1"
    with pytest.raises(CandidateFragmentError, match="artifact_schema_version"):
        univariate_asset_to_verified_fragment(
            asset,
            source_binding=unknown_schema,
            candidate_evidence=evidence,
        )

    unknown_origin = deepcopy(binding)
    unknown_origin["origin_tool"] = "strategy.future_candidate"
    with pytest.raises(CandidateFragmentError, match="origin_tool"):
        univariate_asset_to_verified_fragment(
            asset,
            source_binding=unknown_origin,
            candidate_evidence=evidence,
        )

    forged = deepcopy(asset)
    forged["schema_version"] = "strategy.future-candidate-asset.v1"
    with pytest.raises(CandidateFragmentError, match="candidate asset"):
        univariate_asset_to_verified_fragment(
            forged,
            source_binding=binding,
            candidate_evidence=evidence,
        )


def test_builder_supports_typed_requirements_without_putting_them_in_identity() -> None:
    common = {
        "artifact": {
            "artifact_id": "artifact-a",
            "artifact_kind": "strategy_tree_candidate_json",
            "artifact_schema_version": "strategy.tree-candidate-artifact.v1",
            "artifact_content_hash": HASH_A,
            "origin_tool": "strategy.build_tree_candidate",
        },
        "asset": {
            "schema_version": "strategy.tree-candidate.v1",
            "asset_id": "tree-asset-a",
            "asset_hash": HASH_B,
            "asset_type": "decision_tree",
        },
        "fragment_type": "strategy_rule",
        "rule_id": "tree-rule-a",
        "condition": {
            "op": "compare",
            "field": "score",
            "operator": "<",
            "value": 500,
            "missing": "no_match",
        },
        "effect_id": "tree-effect-a",
        "evidence_id": "tree-evidence-a",
        "evidence_hash": HASH_C,
        "evidence_identity": {
            "dataset_id": "dataset-a",
            "dataset_content_hash": HASH_A,
            "workspace_revision": 1,
            "workspace_generation": 2,
            "semantic_mapping_hash": HASH_B,
            "sample_context_hash": HASH_C,
        },
    }
    first = build_verified_candidate_fragment(
        **common,
        requirements=[{"type": "sample_weight", "column": "weight_a"}],
    )
    second = build_verified_candidate_fragment(
        **common,
        requirements=[{"type": "sample_weight", "column": "weight_b"}],
    )

    assert first["evidence"]["identity"] == second["evidence"]["identity"]
    assert first["fragment"]["requirements"] != second["fragment"]["requirements"]
    assert first["fragment_hash"] != second["fragment_hash"]
