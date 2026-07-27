from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from marvis.packs.strategy import voting_candidate as voting_candidate_domain
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
)
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    set_pool_entry_action,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION,
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
    VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1,
    VotingCandidateAssetError,
    build_voting_candidate_asset,
    canonical_voting_candidate_asset_json,
    parse_voting_candidate_asset_json,
    validate_voting_candidate_asset,
    verify_voting_candidate_asset_against_pool,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
SAMPLE_DESIGN_REF = {
    "artifact_id": "1" * 64,
    "artifact_content_hash": "2" * 64,
    "sample_design_id": "sample-design-voting",
    "sample_design_content_hash": "3" * 64,
    "partition": "development",
}


def _approval(value: str = "approve") -> dict:
    return {
        "type": "approval",
        "value": value,
        "reason_code": None,
        "stop": True,
    }


def _reject(reason: str = "RISK") -> dict:
    return {
        "type": "reject",
        "value": "reject",
        "reason_code": reason,
        "stop": True,
    }


def _review() -> dict:
    return {
        "type": "review",
        "value": "review",
        "reason_code": "MANUAL",
        "stop": True,
    }


def _fragment(index: int) -> dict:
    suffix = str(index)
    hashes = (HASH_C, HASH_D, HASH_E)
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{suffix}",
            "artifact_kind": "strategy_test_candidate_json",
            "artifact_schema_version": "strategy.test-candidate-artifact.v1",
            "artifact_content_hash": hashes[index],
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "strategy.test-candidate.v1",
            "asset_id": f"candidate-asset-source-{suffix}",
            "asset_hash": hashes[index],
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-source-{suffix}",
        condition={
            "op": "compare",
            "field": f"score_{suffix}",
            "operator": ">=",
            "value": 600 + index * 10,
            "missing": "no_match",
        },
        requirements=[],
        effect_id=f"candidate-effect-source-{suffix}",
        evidence_id=f"candidate-evidence-source-{suffix}",
        evidence_hash=hashes[index],
        evidence_identity={
            "dataset_id": "dataset-1",
            "dataset_content_hash": HASH_A,
            "workspace_revision": 7,
            "workspace_generation": 3,
            "semantic_mapping_hash": HASH_B,
            "sample_context_hash": HASH_C,
        },
    )


def _pool() -> dict:
    pool = None
    for index in range(3):
        pool = add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action=_approval(),
            verified_candidate_fragment=_fragment(index),
            action=_reject(f"RISK_{index}"),
        )
    assert pool is not None
    return pool


def _effect() -> dict:
    return {
        "population_count": 100,
        "labeled_count": 100,
        "matched_count": 20,
        "matched_rate": 0.2,
        "matched_bad_count": 8,
        "matched_bad_rate": 0.4,
        "unmatched_count": 80,
        "unmatched_bad_count": 12,
        "unmatched_bad_rate": 0.15,
        "bad_capture_rate": 0.4,
        "lift": 2.0,
    }


def _asset(*, pool: dict | None = None, selected: list[str] | None = None) -> dict:
    current = _pool() if pool is None else pool
    ids = [entry["entry_id"] for entry in current["entries"]]
    return build_voting_candidate_asset(
        current,
        selected_entry_ids=ids if selected is None else selected,
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect=_effect(),
    )


def test_build_voting_candidate_is_canonical_self_authenticating_and_replayable() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]

    asset = _asset(pool=pool, selected=[ids[2], ids[0], ids[1]])

    assert asset["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION
    assert asset["asset_type"] == "voting_n_of_k"
    assert asset["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }
    assert asset["pool_ref"] == {
        "pool_id": pool["pool_id"],
        "task_id": pool["task_id"],
        "strategy_type": pool["strategy_type"],
        "revision": pool["revision"],
        "revision_id": pool["revision_id"],
        "snapshot_hash": pool["snapshot_hash"],
    }
    assert [ref["entry_id"] for ref in asset["selected_entries"]] == ids
    assert [ref["pool_position"] for ref in asset["selected_entries"]] == [0, 1, 2]
    assert asset["rule"]["condition"] == {
        "op": "n_of_k",
        "n": 2,
        "args": [entry["execution"]["condition"] for entry in pool["entries"]],
    }
    assert asset["fragment"]["condition"] == asset["rule"]["condition"]
    assert asset["fragment"]["effect_id"] == asset["effect"]["effect_id"]
    assert asset["fragment"]["requirements"] == []
    assert asset["measurement_context"] == {
        "target_col": "bad",
        "sample_context_hash": HASH_C,
    }
    assert asset["metrics"]["matched_bad_rate"] == 0.4
    assert asset["metrics"]["metrics_hash"]
    assert "action" not in canonical_voting_candidate_asset_json(asset)
    assert "adopt" not in canonical_voting_candidate_asset_json(asset)
    assert "deploy" not in canonical_voting_candidate_asset_json(asset)
    assert validate_voting_candidate_asset(asset) == asset
    assert parse_voting_candidate_asset_json(
        canonical_voting_candidate_asset_json(asset)
    ) == asset
    assert verify_voting_candidate_asset_against_pool(asset, pool) == asset


def test_native_risk_development_ref_is_canonical_without_changing_legacy_identity() -> None:
    legacy = _asset()
    native = build_voting_candidate_asset(
        _pool(),
        selected_entry_ids=[
            entry["entry_id"] for entry in _pool()["entries"]
        ],
        n=2,
        target_col="bad",
        sample_design_ref={
            **SAMPLE_DESIGN_REF,
            "partition": "risk/development",
        },
        effect=_effect(),
    )

    assert native["sample_design_ref"]["partition"] == "risk/development"
    assert validate_voting_candidate_asset(native) == native
    assert legacy["asset_id"] == "candidate-asset-d0d019535bb7f32bb37bd88fe74bf488"
    assert legacy["asset_hash"] == (
        "a0d6428f60781c85fe8599a3b8f84a1cba1461bebc2ba3abec9a359bcfdfa2a3"
    )
    assert hashlib.sha256(
        canonical_voting_candidate_asset_json(legacy).encode("utf-8")
    ).hexdigest() == (
        "9c01f9df81011450ffba08a24ccf7d40a33b594db7a48532f9a32587c0aa1b15"
    )


@pytest.mark.parametrize("partition", ["risk/validation", "unknown", ""])
def test_voting_candidate_rejects_unknown_sample_partition(partition: str) -> None:
    with pytest.raises(
        VotingCandidateAssetError,
        match="sample_design_ref must be one exact governed development reference",
    ):
        build_voting_candidate_asset(
            _pool(),
            selected_entry_ids=[
                entry["entry_id"] for entry in _pool()["entries"]
            ],
            n=2,
            target_col="bad",
            sample_design_ref={
                **SAMPLE_DESIGN_REF,
                "partition": partition,
            },
            effect=_effect(),
        )


def test_selected_entry_ids_are_an_unordered_set_canonicalized_by_pool_position() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]
    before = deepcopy(pool)

    forward = _asset(pool=pool, selected=ids)
    reverse = _asset(pool=pool, selected=list(reversed(ids)))

    assert reverse == forward
    assert pool == before


def test_legacy_v1_asset_remains_strictly_readable_but_new_build_is_v2() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]
    legacy = voting_candidate_domain._build_voting_candidate_asset(
        pool,
        selected_entry_ids=ids,
        n=2,
        target_col="bad",
        sample_design_ref=None,
        effect=_effect(),
        producer_version=VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1,
        schema_version=VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
    )

    assert legacy["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1
    assert "sample_design_ref" not in legacy
    assert validate_voting_candidate_asset(legacy) == legacy
    assert parse_voting_candidate_asset_json(
        canonical_voting_candidate_asset_json(legacy)
    ) == legacy
    assert verify_voting_candidate_asset_against_pool(legacy, pool) == legacy
    assert _asset(pool=pool)["schema_version"] == VOTING_CANDIDATE_ASSET_SCHEMA_VERSION


@pytest.mark.parametrize("n", [0, 4, True, 1.5])
def test_build_rejects_invalid_n(n: object) -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]

    with pytest.raises(VotingCandidateAssetError, match="n must be an integer between"):
        build_voting_candidate_asset(
            pool,
            selected_entry_ids=ids,
            n=n,  # type: ignore[arg-type]
            target_col="bad",
            sample_design_ref=SAMPLE_DESIGN_REF,
            effect=_effect(),
        )


def test_build_rejects_duplicate_unknown_and_non_voting_entry_sets() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]

    with pytest.raises(VotingCandidateAssetError, match="duplicate"):
        _asset(pool=pool, selected=[ids[0], ids[0]])
    with pytest.raises(VotingCandidateAssetError, match="unknown"):
        _asset(pool=pool, selected=[ids[0], "pool-entry-unknown"])
    with pytest.raises(VotingCandidateAssetError, match="between 2 and 50"):
        _asset(pool=pool, selected=[])
    with pytest.raises(VotingCandidateAssetError, match="between 2 and 50"):
        build_voting_candidate_asset(
            pool,
            selected_entry_ids=[ids[0]],
            n=1,
            target_col="bad",
            sample_design_ref=SAMPLE_DESIGN_REF,
            effect=_effect(),
        )


def test_effect_and_derived_metrics_are_strict_and_hash_authenticated() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]
    inconsistent = {**_effect(), "unmatched_count": 79}
    with pytest.raises(VotingCandidateAssetError, match="unmatched_count"):
        build_voting_candidate_asset(
            pool,
            selected_entry_ids=ids,
            n=2,
            target_col="bad",
            sample_design_ref=SAMPLE_DESIGN_REF,
            effect=inconsistent,
        )

    false_zero = {
        **_effect(),
        "matched_count": 0,
        "matched_rate": 0.0,
        "matched_bad_count": 0,
        "matched_bad_rate": 0.0,
        "unmatched_count": 100,
        "unmatched_bad_count": 20,
        "unmatched_bad_rate": 0.2,
        "bad_capture_rate": 0.0,
        "lift": None,
    }
    with pytest.raises(VotingCandidateAssetError, match="must be null"):
        build_voting_candidate_asset(
            pool,
            selected_entry_ids=ids,
            n=2,
            target_col="bad",
            sample_design_ref=SAMPLE_DESIGN_REF,
            effect=false_zero,
        )

    asset = _asset(pool=pool)
    drifted = deepcopy(asset)
    drifted["metrics"]["matched_bad_rate"] = 0.41
    with pytest.raises(VotingCandidateAssetError, match="metrics_hash"):
        validate_voting_candidate_asset(drifted)

    non_finite = {**_effect(), "lift": float("nan")}
    with pytest.raises(VotingCandidateAssetError, match="finite"):
        build_voting_candidate_asset(
            pool,
            selected_entry_ids=ids,
            n=2,
            target_col="bad",
            sample_design_ref=SAMPLE_DESIGN_REF,
            effect=non_finite,
        )


def test_empty_denominators_are_materialized_as_null_not_false_zero() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]
    asset = build_voting_candidate_asset(
        pool,
        selected_entry_ids=ids,
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect={
            "population_count": 100,
            "labeled_count": 100,
            "matched_count": 0,
            "matched_rate": 0.0,
            "matched_bad_count": 0,
            "matched_bad_rate": None,
            "unmatched_count": 100,
            "unmatched_bad_count": 0,
            "unmatched_bad_rate": 0.0,
            "bad_capture_rate": None,
            "lift": None,
        },
    )

    assert asset["effect"]["matched_bad_rate"] is None
    assert asset["effect"]["bad_capture_rate"] is None
    assert asset["effect"]["lift"] is None
    assert asset["metrics"]["matched_bad_rate"] is None


def test_validation_rejects_duplicate_refs_unknown_fields_and_hash_drift() -> None:
    asset = _asset()

    duplicate = deepcopy(asset)
    duplicate["selected_entries"][1] = deepcopy(duplicate["selected_entries"][0])
    with pytest.raises(VotingCandidateAssetError, match="duplicate entry_id"):
        validate_voting_candidate_asset(duplicate)

    extra = {**asset, "deployment_status": "active"}
    with pytest.raises(VotingCandidateAssetError, match="unsupported fields"):
        validate_voting_candidate_asset(extra)

    fragment_drift = deepcopy(asset)
    fragment_drift["selected_entries"][0]["source_fragment_hash"] = HASH_E
    with pytest.raises(VotingCandidateAssetError, match="source_hash"):
        validate_voting_candidate_asset(fragment_drift)

    sample_ref_drift = deepcopy(asset)
    sample_ref_drift["sample_design_ref"]["artifact_content_hash"] = "4" * 64
    with pytest.raises(VotingCandidateAssetError, match="candidate_id"):
        validate_voting_candidate_asset(sample_ref_drift)

    evidence_ref_drift = deepcopy(asset)
    evidence_ref_drift["candidate_evidence"]["sample_design_ref"][
        "artifact_content_hash"
    ] = "4" * 64
    with pytest.raises(VotingCandidateAssetError, match="sample_design_ref"):
        validate_voting_candidate_asset(evidence_ref_drift)


def test_verify_rejects_stale_pool_revision_or_snapshot() -> None:
    pool = _pool()
    asset = _asset(pool=pool)
    changed = set_pool_entry_action(
        pool,
        pool["entries"][0]["entry_id"],
        _review(),
        reason="manual review",
    )

    with pytest.raises(VotingCandidateAssetError, match="exact pool revision"):
        verify_voting_candidate_asset_against_pool(asset, changed)


def test_parser_rejects_duplicate_keys_and_non_object_json() -> None:
    asset = _asset()
    canonical = canonical_voting_candidate_asset_json(asset)
    duplicate_key = canonical[:-1] + ',"asset_hash":"' + HASH_A + '"}'

    with pytest.raises(VotingCandidateAssetError, match="duplicate JSON key"):
        parse_voting_candidate_asset_json(duplicate_key)
    with pytest.raises(VotingCandidateAssetError, match="must contain an object"):
        parse_voting_candidate_asset_json(json.dumps([asset]))


def test_effect_content_changes_all_dependent_ids_but_not_rule_identity() -> None:
    pool = _pool()
    ids = [entry["entry_id"] for entry in pool["entries"]]
    first = build_voting_candidate_asset(
        pool,
        selected_entry_ids=ids,
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect=_effect(),
    )
    second = _asset(pool=pool)
    assert first == second

    changed = build_voting_candidate_asset(
        pool,
        selected_entry_ids=ids,
        n=2,
        target_col="bad",
        sample_design_ref=SAMPLE_DESIGN_REF,
        effect={
            **_effect(),
            "matched_bad_count": 9,
            "matched_bad_rate": 0.45,
            "unmatched_bad_count": 11,
            "unmatched_bad_rate": 0.1375,
            "bad_capture_rate": 0.45,
            "lift": 2.25,
        },
    )
    assert changed["rule"]["rule_id"] == first["rule"]["rule_id"]
    assert changed["effect"]["effect_id"] != first["effect"]["effect_id"]
    assert changed["fragment"]["fragment_id"] != first["fragment"]["fragment_id"]
    assert changed["candidate_evidence"]["candidate_id"] != first[
        "candidate_evidence"
    ]["candidate_id"]
    assert changed["asset_id"] != first["asset_id"]
