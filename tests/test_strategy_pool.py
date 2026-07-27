from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from marvis.feature.univariate import analyze_univariate
from marvis.packs.strategy import pool as pool_module
from marvis.packs.strategy.candidate_asset import refine_univariate_candidate
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
    univariate_asset_to_verified_fragment,
)
from marvis.packs.strategy.candidate_evidence import (
    MetricObservation,
    build_candidate_evidence,
)
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.pool import (
    CandidatePoolError,
    add_candidate,
    add_verified_candidate_fragment,
    canonical_strategy_pool_json,
    compile_strategy_pool,
    remove_pool_entry,
    reorder_strategy_pool,
    set_pool_entry_action,
    strategy_pool_snapshot_hash,
    validate_strategy_pool,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            "loan": [100.0] * 12,
            "overdue": [0, 0, 0, 5, 0, 10, 0, 15, 20, 25, 30, 40],
        }
    )


def _parent(frame: pd.DataFrame, *, workspace_revision: int = 3) -> dict:
    analysis = analyze_univariate(
        frame,
        features=["score"],
        target="bad",
        methods=["equal_width"],
        bin_count=3,
        loan_amount="loan",
        overdue_amount="overdue",
    )
    return build_candidate_evidence(
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash=HASH_A,
        workspace_revision=workspace_revision,
        workspace_generation=2,
        semantic_mapping_hash=HASH_B,
        generation_parameters={"features": ["score"], "bin_count": 3},
        seed=0,
        budget=100_000,
        truncated=False,
        analysis=analysis,
        metrics=[
            MetricObservation("parent.iv", "count", "observed", 0.2),
            MetricObservation("parent.iv", "loan_amount", "unavailable", None),
            MetricObservation("parent.iv", "overdue_amount", "unavailable", None),
        ],
        source_refs=["dataset:dataset-1", "analysis:univariate-1"],
        producer_version="strategy.univariate-candidate/1",
    )


def _asset(source_bin_index: int, *, workspace_revision: int = 3) -> dict:
    frame = _frame()
    evidence = _parent(frame, workspace_revision=workspace_revision)
    method = evidence["analysis"]["features"][0]["methods"][0]
    source_bin_id = method["bins"][source_bin_index]["id"]
    return refine_univariate_candidate(
        evidence,
        frame,
        source_evidence={
            "artifact_id": "artifact-parent",
            "kind": "strategy_candidate_json",
            "content_hash": HASH_C,
        },
        feature="score",
        method="equal_width",
        merge_groups=[],
        selection={"source_bin_ids": [source_bin_id]},
        selection_reason=f"select bin {source_bin_index}",
    )


def _binding(
    asset: dict,
    *,
    suffix: str,
    workspace_revision: int = 3,
) -> dict:
    identity = _parent(
        _frame(), workspace_revision=workspace_revision
    )["identity"]
    return {
        "artifact_id": f"artifact-asset-{suffix}",
        "kind": "strategy_candidate_asset_json",
        "content_hash": suffix * 64,
        "origin_tool": "strategy.refine_univariate_candidate",
        "artifact_schema_version": "strategy.candidate-asset-artifact.v1",
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "candidate_kind": "univariate_refinement",
        "fragment_id": asset["rule"]["rule_id"],
        "parent_candidate_id": asset["parent"]["candidate_id"],
        "parent_evidence_hash": asset["parent"]["evidence_hash"],
        "effect_id": asset["effect"]["effect_id"],
        "effect_stage": "development",
        "validation_status": "unvalidated",
        "evidence_identity": {
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
        },
    }


def _approval(value: str = "approve") -> dict:
    action_type = "approval" if value == "approve" else value
    return {"type": action_type, "value": value, "reason_code": None, "stop": True}


def _reject() -> dict:
    return {"type": "reject", "value": "reject", "reason_code": "RISK", "stop": True}


def _review() -> dict:
    return {"type": "review", "value": "review", "reason_code": "MANUAL", "stop": True}


def _initial_pool() -> tuple[dict, dict]:
    asset = _asset(0)
    pool = add_candidate(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        candidate_asset=asset,
        source_binding=_binding(asset, suffix="1"),
        action=_reject(),
    )
    return pool, asset


def test_add_candidate_creates_canonical_draft_and_compiles_same_dsl() -> None:
    pool, asset = _initial_pool()

    assert pool["revision"] == 1
    assert pool["status"] == "draft"
    assert pool["validation_status"] == "unvalidated"
    assert pool["entries"][0]["rule_id"] == asset["rule"]["rule_id"]
    assert pool["entries"][0]["execution"] == {
        "condition": asset["rule"]["condition"],
        "requirements": [],
    }
    assert canonical_strategy_pool_json(pool).endswith("}")
    assert validate_strategy_pool(pool) == pool

    selected = compile_strategy_pool(pool)
    assert selected["pool_ref"] == {
        "pool_id": pool["pool_id"],
        "task_id": "task-1",
        "strategy_type": "approval",
        "revision": 1,
        "revision_id": pool["revision_id"],
        "snapshot_hash": strategy_pool_snapshot_hash(pool),
    }
    assert selected["requirements"] == []
    assert selected["strategy_spec"]["rules"][0]["rule_id"] == asset["rule"]["rule_id"]
    assert selected["strategy_spec"]["rules"][0]["priority"] == 10
    assert len(selected["design_hash"]) == 64

    spec = parse_strategy_spec(selected["strategy_spec"])
    evaluated = evaluate_strategy_frame(_frame(), spec)
    expected = [
        bool(value)
        for value in _frame()["score"].between(100, 210, inclusive="left")
    ]
    assert evaluated.matched_rule_id.notna().tolist() == expected


def test_initial_workspace_revision_zero_is_valid_evidence_identity() -> None:
    asset = _asset(0, workspace_revision=0)
    pool = add_candidate(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        candidate_asset=asset,
        source_binding=_binding(
            asset,
            suffix="0",
            workspace_revision=0,
        ),
        action=_reject(),
    )

    assert pool["entries"][0]["source"]["evidence_identity"][
        "workspace_revision"
    ] == 0


def test_mutations_increment_revision_keep_entry_ids_and_compile_order() -> None:
    pool, first_asset = _initial_pool()
    first_entry_id = pool["entries"][0]["entry_id"]
    first_hash = strategy_pool_snapshot_hash(pool)
    second_asset = _asset(2)
    pool = add_candidate(
        pool,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        candidate_asset=second_asset,
        source_binding=_binding(second_asset, suffix="2"),
        action=_review(),
    )
    second_entry_id = pool["entries"][1]["entry_id"]
    assert pool["revision"] == 2
    assert strategy_pool_snapshot_hash(pool) != first_hash

    pool = reorder_strategy_pool(pool, [second_entry_id, first_entry_id])
    assert [row["entry_id"] for row in pool["entries"]] == [
        second_entry_id,
        first_entry_id,
    ]
    pool = set_pool_entry_action(pool, first_entry_id, _review())
    assert next(
        row for row in pool["entries"] if row["entry_id"] == first_entry_id
    )["action"] == _review()
    selected = compile_strategy_pool(pool)
    assert [row["rule_id"] for row in selected["strategy_spec"]["rules"]] == [
        second_asset["rule"]["rule_id"],
        first_asset["rule"]["rule_id"],
    ]
    assert [row["priority"] for row in selected["strategy_spec"]["rules"]] == [10, 20]

    removed = remove_pool_entry(pool, second_entry_id)
    assert removed["revision"] == pool["revision"] + 1
    assert [row["entry_id"] for row in removed["entries"]] == [first_entry_id]


def test_pool_rejects_duplicate_assets_rules_and_invalid_mutations() -> None:
    pool, asset = _initial_pool()
    entry_id = pool["entries"][0]["entry_id"]

    with pytest.raises(CandidatePoolError, match="duplicate asset"):
        add_candidate(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action=_approval(),
            candidate_asset=asset,
            source_binding=_binding(asset, suffix="1"),
            action=_reject(),
        )
    with pytest.raises(CandidatePoolError, match="unknown pool entry"):
        remove_pool_entry(pool, "pool-entry-missing")
    with pytest.raises(CandidatePoolError, match="unknown pool entry"):
        set_pool_entry_action(pool, "pool-entry-missing", _review())
    with pytest.raises(CandidatePoolError, match="complete"):
        reorder_strategy_pool(pool, [])
    with pytest.raises(CandidatePoolError, match="duplicate"):
        reorder_strategy_pool(pool, [entry_id, entry_id])
    with pytest.raises(CandidatePoolError, match="unknown"):
        reorder_strategy_pool(pool, ["pool-entry-missing"])

    other_asset = _asset(2)
    other_binding = _binding(other_asset, suffix="2")
    other_binding["evidence_identity"]["workspace_generation"] += 1
    with pytest.raises(CandidatePoolError, match="evidence identity"):
        add_candidate(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action=_approval(),
            candidate_asset=other_asset,
            source_binding=other_binding,
            action=_reject(),
        )


def test_pool_rejects_cross_task_type_action_and_lifecycle_claims() -> None:
    pool, asset = _initial_pool()
    other = _asset(1)

    with pytest.raises(CandidatePoolError, match="task_id"):
        add_candidate(
            pool,
            task_id="task-2",
            strategy_type="approval",
            default_action=_approval(),
            candidate_asset=other,
            source_binding=_binding(other, suffix="2"),
            action=_reject(),
        )
    with pytest.raises(CandidatePoolError, match="strategy_type"):
        add_candidate(
            pool,
            task_id="task-1",
            strategy_type="reject",
            default_action=_approval(),
            candidate_asset=other,
            source_binding=_binding(other, suffix="2"),
            action=_reject(),
        )
    with pytest.raises(CandidatePoolError, match="not allowed|incompatible"):
        set_pool_entry_action(
            pool,
            pool["entries"][0]["entry_id"],
            {"type": "limit", "value": 1000, "reason_code": None, "stop": True},
        )

    forged = deepcopy(pool)
    forged["status"] = "adopted"
    with pytest.raises(CandidatePoolError, match="draft"):
        validate_strategy_pool(forged)
    forged = deepcopy(pool)
    forged["validation_status"] = "validated"
    with pytest.raises(CandidatePoolError, match="unvalidated"):
        validate_strategy_pool(forged)
    forged = deepcopy(pool)
    forged["deployed"] = True
    with pytest.raises(CandidatePoolError, match="unsupported fields"):
        validate_strategy_pool(forged)


def test_compile_is_pure_and_design_hash_binds_order_and_action() -> None:
    pool, _ = _initial_pool()
    before = deepcopy(pool)
    first = compile_strategy_pool(pool)
    repeated = compile_strategy_pool(pool)
    assert first == repeated
    assert pool == before

    entry_id = pool["entries"][0]["entry_id"]
    changed = set_pool_entry_action(pool, entry_id, _review())
    assert compile_strategy_pool(changed)["design_hash"] != first["design_hash"]

    empty = remove_pool_entry(pool, entry_id)
    with pytest.raises(CandidatePoolError, match="empty"):
        compile_strategy_pool(empty)


def test_generic_pool_accepts_two_fragments_from_same_asset_and_rejects_pair() -> None:
    asset = _asset(0)
    first_fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=_binding(asset, suffix="1"),
    )
    identity = first_fragment["evidence"]["identity"]
    second_fragment = build_verified_candidate_fragment(
        artifact=first_fragment["artifact"],
        asset=first_fragment["asset"],
        fragment_type="strategy_rule",
        rule_id="candidate-rule-second-fragment",
        condition={
            "op": "compare",
            "field": "score",
            "operator": ">=",
            "value": 400,
            "missing": "no_match",
        },
        requirements=[{"type": "sample_weight", "column": "weight_b"}],
        effect_id="candidate-effect-second-fragment",
        evidence_id=first_fragment["evidence"]["evidence_id"],
        evidence_hash=first_fragment["evidence"]["evidence_hash"],
        evidence_identity=identity,
    )

    pool = add_verified_candidate_fragment(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        verified_candidate_fragment=first_fragment,
        action=_reject(),
    )
    pool = add_verified_candidate_fragment(
        pool,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        verified_candidate_fragment=second_fragment,
        action=_review(),
    )

    assert [entry["source"]["asset_id"] for entry in pool["entries"]] == [
        asset["asset_id"],
        asset["asset_id"],
    ]
    assert len({entry["source"]["fragment_id"] for entry in pool["entries"]}) == 2
    assert all(
        entry["rule_id"] != entry["source"]["fragment_id"]
        for entry in pool["entries"]
    )
    assert compile_strategy_pool(pool)["requirements"] == [
        {
            "rule_id": "candidate-rule-second-fragment",
            "fragment_id": second_fragment["fragment"]["fragment_id"],
            "requirement": {"type": "sample_weight", "column": "weight_b"},
        }
    ]

    same_pair_different_rule = build_verified_candidate_fragment(
        artifact=first_fragment["artifact"],
        asset=first_fragment["asset"],
        fragment_id=first_fragment["fragment"]["fragment_id"],
        fragment_type="strategy_rule",
        rule_id="candidate-rule-same-fragment-new-rule",
        condition=first_fragment["fragment"]["condition"],
        requirements=[],
        effect_id=first_fragment["fragment"]["effect_id"],
        evidence_id=first_fragment["evidence"]["evidence_id"],
        evidence_hash=first_fragment["evidence"]["evidence_hash"],
        evidence_identity=identity,
    )
    with pytest.raises(CandidatePoolError, match="duplicate asset fragment"):
        add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action=_approval(),
            verified_candidate_fragment=same_pair_different_rule,
            action=_reject(),
        )


def test_v2_snapshot_reader_keeps_preexisting_duplicate_conditions_compatible() -> None:
    asset = _asset(0)
    first_fragment = univariate_asset_to_verified_fragment(
        asset,
        source_binding=_binding(asset, suffix="1"),
    )
    second_fragment = build_verified_candidate_fragment(
        artifact=first_fragment["artifact"],
        asset=first_fragment["asset"],
        fragment_type="strategy_rule",
        rule_id="candidate-rule-legacy-duplicate-condition",
        condition=first_fragment["fragment"]["condition"],
        requirements=[],
        effect_id="candidate-effect-legacy-duplicate-condition",
        evidence_id=first_fragment["evidence"]["evidence_id"],
        evidence_hash=first_fragment["evidence"]["evidence_hash"],
        evidence_identity=first_fragment["evidence"]["identity"],
    )
    first_pool = add_verified_candidate_fragment(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        verified_candidate_fragment=first_fragment,
        action=_reject(),
    )
    second_pool = add_verified_candidate_fragment(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        verified_candidate_fragment=second_fragment,
        action=_review(),
    )
    second_entry = {**second_pool["entries"][0], "position": 1}

    historical = pool_module._snapshot(
        pool_id=first_pool["pool_id"],
        task_id="task-1",
        strategy_type="approval",
        revision=2,
        parent_revision_id=first_pool["revision_id"],
        operation_kind="add_candidate",
        reason=None,
        default_action=_approval(),
        entries=[first_pool["entries"][0], second_entry],
    )

    assert validate_strategy_pool(historical) == historical
    with pytest.raises(CandidatePoolError, match="historical Pool.*remove"):
        compile_strategy_pool(historical)
    repaired = remove_pool_entry(historical, second_entry["entry_id"])
    assert len(compile_strategy_pool(repaired)["strategy_spec"]["rules"]) == 1


def test_generic_pool_rejects_mixed_sample_context_without_changing_revision() -> None:
    asset = _asset(0)
    first = univariate_asset_to_verified_fragment(
        asset,
        source_binding=_binding(asset, suffix="1"),
    )
    pool = add_verified_candidate_fragment(
        None,
        task_id="task-1",
        strategy_type="approval",
        default_action=_approval(),
        verified_candidate_fragment=first,
        action=_reject(),
    )
    mixed_identity = deepcopy(first["evidence"]["identity"])
    mixed_identity["sample_context_hash"] = HASH_C
    mixed = build_verified_candidate_fragment(
        artifact={
            **first["artifact"],
            "artifact_id": "artifact-asset-other",
            "artifact_content_hash": HASH_C,
        },
        asset={
            **first["asset"],
            "asset_id": "candidate-asset-other",
        },
        fragment_type="strategy_rule",
        rule_id="candidate-rule-other",
        condition={
            "op": "compare",
            "field": "score",
            "operator": ">=",
            "value": 400,
            "missing": "no_match",
        },
        requirements=[],
        effect_id="candidate-effect-other",
        evidence_id="candidate-evidence-other",
        evidence_hash=HASH_C,
        evidence_identity=mixed_identity,
    )

    with pytest.raises(CandidatePoolError, match="evidence identity"):
        add_verified_candidate_fragment(
            pool,
            task_id="task-1",
            strategy_type="approval",
            default_action=_approval(),
            verified_candidate_fragment=mixed,
            action=_reject(),
        )
    assert pool["revision"] == 1
