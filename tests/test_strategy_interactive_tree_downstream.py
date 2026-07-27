from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvis.data.workspace import data_semantic_mapping_hash
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
)
from marvis.packs.strategy.candidate_stability_tools import (
    resolve_candidate_monthly_stability_inputs,
    run_measure_candidate_monthly_stability,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.interactive_tree_tools import (
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
)
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_validation_tools import (
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    run_materialize_sample_design_v2,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_sample_design_v2_tool import (
    _setup as _sample_v2_setup,
)


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _sample_v2_ref(fx: dict, output: dict) -> dict[str, str]:
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    membership = next(
        record
        for record in records
        if record["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle = next(
        record
        for record in records
        if record["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    return {
        "membership_artifact_id": membership["id"],
        "expected_membership_artifact_content_hash": membership[
            "content_hash"
        ],
        "bundle_artifact_id": bundle["id"],
        "expected_bundle_artifact_content_hash": bundle["content_hash"],
        "expected_bundle_id": output["bundle_id"],
        "expected_sample_design_id": output["sample_design_id"],
        "expected_sample_design_content_hash": output[
            "sample_design_content_hash"
        ],
    }


def _frontier_pool(tmp_path: Path) -> dict:
    fx = _sample_v2_setup(tmp_path)
    sample_v2 = run_materialize_sample_design_v2(
        fx["request"],
        fx["ctx"],
        fx["runtime"],
    )
    sample_v2_ref = _sample_v2_ref(fx, sample_v2)
    workspace = fx["workspace"]
    legacy_ref = fx["request"]["legacy_sample_design_ref"]
    strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "target_col": "bad",
            "sample_design_ref": legacy_ref,
            "features": ["legacy_score"],
            "drop_nan_labels": True,
            "sample_weight_col": "weight",
            "directions": {"legacy_score": "increasing"},
            "max_depth": 2,
            "min_leaf_count": 1,
            "min_weight_fraction_leaf": 0.0,
            "seed": 20260726,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "budgets": {
                "max_rows": 100,
                "max_features": 5,
                "max_cells": 500,
                "max_nodes": 31,
                "max_cutpoint_evaluations": 1_000,
            },
        },
        fx["ctx"],
    )
    repository = TaskArtifactRepository(fx["settings"].db_path)
    source_record = next(
        record
        for record in repository.list_for_task(fx["task"].id)
        if record["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    source_asset = json.loads(
        Path(source_record["path"]).read_text(encoding="utf-8")
    )
    split_id = next(
        node["node_id"]
        for node in reversed(source_asset["tree_result"]["tree"]["nodes"])
        if node["kind"] == "split"
    )
    revised = strategy_tools.tool_revise_interactive_tree(
        {
            "source_tree_id": source_asset["asset_id"],
            "node_id": split_id,
            "operation": "prune_subtree",
            "reason": "Reviewed for governed downstream replay.",
        },
        fx["ctx"],
    )
    revision_record = next(
        record
        for record in repository.list_for_task(fx["task"].id)
        if record["kind"] == INTERACTIVE_TREE_REVISION_ARTIFACT_KIND
        and record["provenance"]["revision_id"] == revised["revision_id"]
    )
    revision = json.loads(
        Path(revision_record["path"]).read_text(encoding="utf-8")
    )
    fragment = revision["fragments"][0]
    selection = (
        strategy_tools.tool_materialize_interactive_tree_frontier_selection(
            {
                "revision_id": revision["revision_id"],
                "source_node_id": fragment["source_node_id"],
                "selection_reason": "Send this reviewed frontier downstream.",
            },
            fx["ctx"],
        )
    )
    selection_artifact = selection["artifacts"][0]
    pool = strategy_tools.tool_add_candidate_to_pool(
        {
            "source_artifact_id": selection_artifact["artifact_id"],
            "expected_artifact_content_hash": selection_artifact[
                "content_hash"
            ],
            "expected_asset_id": selection["semantic_tree_id"],
            "expected_asset_hash": selection["tree_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", reason="INTERACTIVE_TREE_RISK"),
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        fx["ctx"],
    )
    [entry] = pool["entries"]
    pool_artifact = pool["artifacts"][0]
    pool_ref = {
        "artifact_id": pool_artifact["artifact_id"],
        "expected_artifact_content_hash": pool_artifact["content_hash"],
        "expected_pool_id": pool["pool_id"],
        "expected_revision": pool["revision"],
        "expected_revision_id": pool["pool"]["revision_id"],
        "expected_snapshot_hash": pool["snapshot_hash"],
    }
    return {
        **fx,
        "legacy_ref": legacy_ref,
        "sample_v2_ref": sample_v2_ref,
        "source_asset": source_asset,
        "revision": revision,
        "revision_record": revision_record,
        "fragment": fragment,
        "selection": selection,
        "pool": pool,
        "pool_ref": pool_ref,
        "entry": entry,
    }


def test_interactive_tree_frontier_pool_entry_replays_in_all_common_evidence_tools(
    tmp_path: Path,
) -> None:
    fx = _frontier_pool(tmp_path)
    entry = fx["entry"]
    fragment = fx["fragment"]

    assert entry["rule_id"] == fragment["rule_id"]
    assert entry["source"]["artifact_id"] == fx["selection"]["artifacts"][0][
        "artifact_id"
    ]
    assert entry["source"]["asset_id"] == fx["revision"]["semantic_tree_id"]
    assert entry["source"]["asset_hash"] == fx["revision"]["tree"]["tree_hash"]
    assert entry["source"]["fragment_id"] == fragment["fragment_id"]
    assert entry["source"]["effect_id"] == fragment["effect_id"]
    assert entry["execution"] == {
        "condition": fragment["condition"],
        "requirements": fragment["requirements"],
    }

    validation = run_measure_strategy_pool_validation(
        {
            "strategy_type": "approval",
            "pool_ref": fx["pool_ref"],
            "sample_design_ref": fx["sample_v2_ref"],
            "partition": "validation",
            "population": "risk",
            "comparison_mode": "absolute",
        },
        fx["ctx"],
        fx["runtime"],
    )
    impact = run_measure_strategy_impact_cube(
        {
            "strategy_type": "approval",
            "pool_ref": fx["pool_ref"],
            "sample_design_ref": fx["sample_v2_ref"],
            "partitions": ["development", "validation"],
            "population": "risk",
            "dimension_bindings": {
                "month_col": "apply_month",
                "group_col": "channel",
                "segment_col": "sample_split",
            },
            "current_strategy_ref": None,
            "economics_inputs": None,
        },
        fx["ctx"],
        fx["runtime"],
    )
    stability = run_measure_candidate_monthly_stability(
        resolve_candidate_monthly_stability_inputs(
            fx["runtime"],
            task_id=fx["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": entry["entry_id"],
            },
        ),
        fx["ctx"],
        fx["runtime"],
    )

    assert validation["pool_id"] == fx["pool"]["pool_id"]
    assert validation["pool_snapshot_hash"] == fx["pool"]["snapshot_hash"]
    assert (
        validation["evidence"]["source_bindings"]["development_lineage"][
            "legacy_development_ref"
        ]
        == fx["legacy_ref"]
    )
    [validation_entry] = validation["evidence"]["waterfall"]
    assert validation_entry["entry_id"] == entry["entry_id"]
    assert validation_entry["rule_id"] == entry["rule_id"]
    assert (
        validation_entry["source_ref"]["fragment_id"]
        == fragment["fragment_id"]
    )
    assert impact["pool_id"] == fx["pool"]["pool_id"]
    assert impact["cube"]["identity"]["snapshot_hash"] == fx["pool"][
        "snapshot_hash"
    ]
    assert (
        impact["cube"]["source_bindings"]["development_lineage"][
            "legacy_development_ref"
        ]
        == fx["legacy_ref"]
    )
    overall = next(
        row
        for row in impact["cube"]["slices"]
        if row["family"] == "overall"
        and row["population_role"] == "risk"
        and row["dimensions"]["partition"]["value"] == "validation"
    )
    [impact_entry] = overall["waterfall"]["value"]["entries"]
    assert impact_entry["entry_id"] == entry["entry_id"]
    assert impact_entry["rule_id"] == entry["rule_id"]
    assert impact_entry["source_ref"]["fragment_id"] == fragment["fragment_id"]
    assert stability["stability"]["source_ref"]["entry_id"] == entry["entry_id"]
    assert stability["stability"]["source_ref"]["rule_id"] == entry["rule_id"]
    assert (
        stability["stability"]["source_ref"]["snapshot_hash"]
        == fx["pool"]["snapshot_hash"]
    )
    assert stability["stability"]["sample_design_ref"] == fx["legacy_ref"]


@pytest.mark.parametrize("consumer", ["validation", "impact", "stability"])
def test_interactive_tree_downstream_rejects_revision_artifact_drift(
    tmp_path: Path,
    consumer: str,
) -> None:
    fx = _frontier_pool(tmp_path)
    revision_path = Path(fx["revision_record"]["path"])
    revision_path.write_bytes(revision_path.read_bytes() + b"\n")

    if consumer == "validation":
        invoke = lambda: run_measure_strategy_pool_validation(  # noqa: E731
            {
                "strategy_type": "approval",
                "pool_ref": fx["pool_ref"],
                "sample_design_ref": fx["sample_v2_ref"],
                "partition": "validation",
                "population": "risk",
                "comparison_mode": "absolute",
            },
            fx["ctx"],
            fx["runtime"],
        )
    elif consumer == "impact":
        invoke = lambda: run_measure_strategy_impact_cube(  # noqa: E731
            {
                "strategy_type": "approval",
                "pool_ref": fx["pool_ref"],
                "sample_design_ref": fx["sample_v2_ref"],
                "partitions": ["development", "validation"],
                "population": "risk",
                "dimension_bindings": {
                    "month_col": "apply_month",
                    "group_col": "channel",
                    "segment_col": "sample_split",
                },
                "current_strategy_ref": None,
                "economics_inputs": None,
            },
            fx["ctx"],
            fx["runtime"],
        )
    else:
        invoke = lambda: resolve_candidate_monthly_stability_inputs(  # noqa: E731
            fx["runtime"],
            task_id=fx["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": fx["entry"]["entry_id"],
            },
        )

    with pytest.raises(StrategyError, match="content hash|canonical|changed"):
        invoke()
