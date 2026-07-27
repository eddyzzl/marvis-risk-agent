from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.packs.strategy.cross_rule_candidate import (
    CROSS_RULE_CANDIDATE_SCHEMA_VERSION,
    CrossRuleCandidateError,
    build_cross_rule_candidate,
    canonical_cross_rule_candidate_json,
    validate_cross_rule_candidate,
)
from tests.test_strategy_cross_rule_search import _request
from marvis.packs.strategy.cross_rule_search import (
    search_cross_threshold_rules,
)


def _asset(*, rule_index: int = 0) -> dict:
    search = search_cross_threshold_rules(_request())
    rule = search["rules"][rule_index]
    return build_cross_rule_candidate(
        search,
        search_artifact_ref={
            "artifact_id": "d" * 64,
            "artifact_content_hash": "e" * 64,
        },
        rule_id=rule["rule_id"],
    )


def test_exact_rule_pointer_materializes_self_authenticating_candidate_without_pool_claims() -> None:
    asset = _asset()

    assert asset["schema_version"] == CROSS_RULE_CANDIDATE_SCHEMA_VERSION
    assert asset["asset_type"] == "cross_threshold_rule"
    assert asset["effect_stage"] == "development"
    assert asset["validation_status"] == "unvalidated"
    assert asset["source_selection"]["eligible"] in {True, False}
    assert asset["dimension"] == 2
    assert asset["condition"]["op"] == "and"
    assert len(asset["condition"]["args"]) == 2
    assert asset["lifecycle"] == {
        "admitted": False,
        "applied": False,
        "adopted": False,
        "deployed": False,
    }
    assert validate_cross_rule_candidate(asset) == asset
    assert canonical_cross_rule_candidate_json(asset)


def test_reason_is_audited_but_does_not_change_semantic_asset_identity() -> None:
    search = search_cross_threshold_rules(_request())
    rule = search["rules"][0]
    ref = {
        "artifact_id": "d" * 64,
        "artifact_content_hash": "e" * 64,
    }
    first = build_cross_rule_candidate(
        search,
        search_artifact_ref=ref,
        rule_id=rule["rule_id"],
        selection_reason="用于验证渠道风险。",
    )
    second = build_cross_rule_candidate(
        search,
        search_artifact_ref=ref,
        rule_id=rule["rule_id"],
        selection_reason="业务确认后再考虑入池。",
    )

    assert first["asset_id"] == second["asset_id"]
    assert first["asset_hash"] == second["asset_hash"]
    assert first["selection_audit_hash"] != second["selection_audit_hash"]


def test_candidate_rejects_unknown_rule_and_tampered_search_or_metrics() -> None:
    search = search_cross_threshold_rules(_request())
    ref = {
        "artifact_id": "d" * 64,
        "artifact_content_hash": "e" * 64,
    }
    with pytest.raises(CrossRuleCandidateError, match="rule_id"):
        build_cross_rule_candidate(
            search,
            search_artifact_ref=ref,
            rule_id="cross-rule-" + "f" * 32,
        )

    asset = _asset()
    tampered = deepcopy(asset)
    tampered["metrics"]["lift"] += 0.1
    with pytest.raises(
        CrossRuleCandidateError,
        match="asset_id|asset_hash|metrics",
    ):
        validate_cross_rule_candidate(tampered)
