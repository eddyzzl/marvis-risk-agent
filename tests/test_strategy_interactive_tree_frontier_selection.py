from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import build_automatic_tree_asset
from marvis.packs.strategy.interactive_tree_frontier_selection import (
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION,
    INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION,
    InteractiveTreeFrontierSelectionError,
    build_interactive_tree_frontier_selection,
    canonical_interactive_tree_frontier_selection_json,
    interactive_tree_frontier_selection_to_verified_candidate_fragment,
    validate_interactive_tree_frontier_selection,
)
from marvis.packs.strategy.interactive_tree_revision import (
    build_interactive_tree_revision,
    canonical_interactive_tree_revision_json,
)
from marvis.packs.strategy.candidate_fragment import (
    validate_verified_candidate_fragment,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64

SAMPLE_DESIGN_REF = {
    "artifact_id": HASH_E,
    "artifact_content_hash": HASH_F,
    "sample_design_id": "strategy-sample-design-" + "1" * 32,
    "sample_design_content_hash": "2" * 64,
    "partition": "development",
}


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _automatic_asset() -> dict:
    frame = pd.DataFrame(
        {
            "x": list(range(16)),
            "z": [value % 4 for value in range(16)],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=["x", "z"],
        target_col="bad",
        max_depth=3,
        min_leaf_count=1,
    )
    sample_ref_token = "strategy-sample-design:" + _canonical_json(
        {"kind": "strategy_sample_design", **SAMPLE_DESIGN_REF}
    )
    return build_automatic_tree_asset(
        tree,
        task_id="task-interactive-frontier-selection",
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=[
            "workspace:task-interactive-frontier-selection:3",
            "dataset:dataset-labelled",
            sample_ref_token,
        ],
    )


def _revision(asset: dict) -> dict:
    split_nodes = [
        node
        for node in asset["tree_result"]["tree"]["nodes"]
        if node["kind"] == "split"
    ]
    return build_interactive_tree_revision(
        asset,
        node_id=split_nodes[-1]["node_id"],
        reason="Create a simpler reviewed frontier.",
    )


def _revision_binding(asset: dict, revision: dict) -> dict:
    canonical_bytes = canonical_interactive_tree_revision_json(
        revision,
        asset,
    ).encode("utf-8")
    provenance = {
        "schema_version": "strategy.interactive-tree-revision-artifact.v1",
        "producer_version": revision["producer_version"],
        "task_id": revision["identity"]["task_id"],
        "kind": "strategy_interactive_tree_revision_json",
        "format": "json",
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree"]["tree_hash"],
        "base_asset_id": revision["base_tree"]["asset_id"],
        "base_asset_hash": revision["base_tree"]["asset_hash"],
        "base_tree_result_hash": revision["base_tree"]["tree_result_hash"],
        "parent_revision_id": None,
        "source_tree_id": revision["base_tree"]["asset_id"],
        "edit_operation": revision["edit"]["operation"],
        "edit_node_id": revision["edit"]["node_id"],
        "sample_design_ref": SAMPLE_DESIGN_REF,
    }
    return {
        "artifact_id": "artifact-interactive-tree-revision",
        "task_id": revision["identity"]["task_id"],
        "kind": "strategy_interactive_tree_revision_json",
        "artifact_schema_version": (
            "strategy.interactive-tree-revision-artifact.v1"
        ),
        "content_hash": hashlib.sha256(canonical_bytes).hexdigest(),
        "origin_tool": "strategy.revise_interactive_tree",
        "path": (
            "/tasks/"
            + revision["identity"]["task_id"]
            + "/strategy_interactive_tree_revisions/"
            + revision["revision_id"]
            + ".json"
        ),
        "provenance": provenance,
        "canonical_bytes": canonical_bytes,
    }


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(child) for child in value))
    return set()


def _selection_provenance(selection: dict) -> dict:
    revision_artifact = selection["revision_artifact"]
    revision = selection["revision"]
    frontier = selection["frontier"]
    return {
        "schema_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": selection["producer_version"],
        "task_id": revision_artifact["task_id"],
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "format": "json",
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "revision_artifact_id": revision_artifact["artifact_id"],
        "revision_artifact_kind": revision_artifact["kind"],
        "revision_artifact_schema_version": (
            revision_artifact["artifact_schema_version"]
        ),
        "revision_artifact_content_hash": revision_artifact["content_hash"],
        "revision_artifact_origin_tool": revision_artifact["origin_tool"],
        "revision_artifact_path": revision_artifact["path"],
        "revision_artifact_provenance": revision_artifact["provenance"],
        "revision_schema_version": revision["schema_version"],
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree_hash"],
        "asset_type": revision["asset_type"],
        "source_node_id": frontier["source_node_id"],
        "leaf_id": frontier["leaf_id"],
        "fragment_id": frontier["fragment_id"],
        "fragment_hash": frontier["fragment_hash"],
        "rule_id": frontier["rule_id"],
        "effect_id": frontier["effect_id"],
    }


def _selection_binding(selection: dict) -> dict:
    canonical_bytes = canonical_interactive_tree_frontier_selection_json(
        selection
    ).encode("utf-8")
    return {
        "artifact_id": "artifact-interactive-tree-frontier-selection",
        "task_id": selection["revision_artifact"]["task_id"],
        "kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "content_hash": hashlib.sha256(canonical_bytes).hexdigest(),
        "origin_tool": INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
        "path": (
            "/tasks/"
            + selection["revision_artifact"]["task_id"]
            + "/strategy_interactive_tree_frontier_selections/"
            + selection["selection_id"]
            + ".json"
        ),
        "provenance": _selection_provenance(selection),
        "canonical_bytes": canonical_bytes,
    }


def test_single_frontier_selection_is_canonical_pointer_only_and_deterministic() -> (
    None
):
    asset = _automatic_asset()
    revision = _revision(asset)
    fragment = revision["fragments"][0]
    binding = _revision_binding(asset, revision)

    first = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=binding,
        source_node_id=fragment["source_node_id"],
        selection_reason="  Policy owner reviewed this frontier.  ",
    )
    second = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=binding,
        source_node_id=fragment["source_node_id"],
        selection_reason="Policy owner reviewed this frontier.",
    )

    assert first == second
    assert first["schema_version"] == (
        INTERACTIVE_TREE_FRONTIER_SELECTION_SCHEMA_VERSION
    )
    assert first["producer_version"] == (
        INTERACTIVE_TREE_FRONTIER_SELECTION_PRODUCER_VERSION
    )
    assert first["selection_reason"] == "Policy owner reviewed this frontier."
    assert first["selection_id"].startswith("interactive-tree-frontier-selection-")
    assert len(first["selection_hash"]) == 64
    assert first["revision"] == {
        "schema_version": revision["schema_version"],
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree"]["tree_hash"],
        "asset_type": revision["asset_type"],
    }
    assert first["frontier"] == {
        key: fragment[key]
        for key in (
            "source_node_id",
            "leaf_id",
            "fragment_id",
            "fragment_hash",
            "rule_id",
            "effect_id",
        )
    }
    assert not {
        "condition",
        "requirements",
        "metrics",
        "action",
        "candidate_stage",
        "observation_stage",
        "validation_status",
    } & _all_mapping_keys(first)
    assert json.loads(
        canonical_interactive_tree_frontier_selection_json(first)
    ) == first
    assert validate_interactive_tree_frontier_selection(first) == first

    detached = validate_interactive_tree_frontier_selection(first)
    detached["frontier"]["source_node_id"] = "changed"
    assert first["frontier"]["source_node_id"] != "changed"


def test_verified_adapter_replays_the_exact_revision_fragment_into_the_pool_seam() -> (
    None
):
    asset = _automatic_asset()
    revision = _revision(asset)
    fragment = revision["fragments"][-1]
    revision_binding = _revision_binding(asset, revision)
    selection = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=revision_binding,
        source_node_id=fragment["source_node_id"],
    )
    selection_binding = _selection_binding(selection)

    verified = interactive_tree_frontier_selection_to_verified_candidate_fragment(
        selection,
        revision,
        asset,
        selection_artifact_binding=selection_binding,
        revision_artifact_binding=revision_binding,
    )

    assert validate_verified_candidate_fragment(verified) == verified
    assert verified["artifact"] == {
        "artifact_id": selection_binding["artifact_id"],
        "artifact_kind": INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            INTERACTIVE_TREE_FRONTIER_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "artifact_content_hash": selection_binding["content_hash"],
        "origin_tool": INTERACTIVE_TREE_FRONTIER_SELECTION_ORIGIN_TOOL,
    }
    assert verified["asset"] == {
        "schema_version": revision["schema_version"],
        "asset_id": revision["semantic_tree_id"],
        "asset_hash": revision["tree"]["tree_hash"],
        "asset_type": "interactive_rule_tree",
    }
    assert verified["fragment"] == {
        "fragment_id": fragment["fragment_id"],
        "fragment_type": "strategy_rule",
        "rule_id": fragment["rule_id"],
        "condition": fragment["condition"],
        "requirements": fragment["requirements"],
        "effect_id": fragment["effect_id"],
    }
    assert verified["evidence"] == {
        "evidence_id": revision["candidate_evidence"]["candidate_id"],
        "evidence_hash": revision["candidate_evidence"]["evidence_hash"],
        "identity": {
            key: revision["identity"][key]
            for key in (
                "dataset_id",
                "dataset_content_hash",
                "workspace_revision",
                "workspace_generation",
                "semantic_mapping_hash",
                "sample_context_hash",
            )
        },
    }
    assert {
        key: verified[key]
        for key in (
            "candidate_stage",
            "observation_stage",
            "validation_status",
        )
    } == revision["lifecycle"]


def test_validator_rejects_noncanonical_reason_and_extra_pointer_fields() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    fragment = revision["fragments"][0]
    selection = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=_revision_binding(asset, revision),
        source_node_id=fragment["source_node_id"],
        selection_reason="Reviewed frontier.",
    )

    noncanonical_reason = deepcopy(selection)
    noncanonical_reason["selection_reason"] = "  Reviewed   frontier.  "
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="selection_reason",
    ):
        validate_interactive_tree_frontier_selection(noncanonical_reason)

    extra_pointer_field = deepcopy(selection)
    extra_pointer_field["revision_artifact"]["canonical_bytes"] = "not persisted"
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="revision_artifact",
    ):
        validate_interactive_tree_frontier_selection(extra_pointer_field)


def test_reason_changes_only_the_audit_selection_identity() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    fragment = revision["fragments"][0]
    revision_binding = _revision_binding(asset, revision)
    first = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=revision_binding,
        source_node_id=fragment["source_node_id"],
        selection_reason="First policy review.",
    )
    second = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=revision_binding,
        source_node_id=fragment["source_node_id"],
        selection_reason="Second policy review.",
    )

    assert first["revision"] == second["revision"]
    assert first["frontier"] == second["frontier"]
    assert first["selection_id"] != second["selection_id"]
    assert first["selection_hash"] != second["selection_hash"]

    first_verified = (
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            first,
            revision,
            asset,
            selection_artifact_binding=_selection_binding(first),
            revision_artifact_binding=revision_binding,
        )
    )
    second_verified = (
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            second,
            revision,
            asset,
            selection_artifact_binding=_selection_binding(second),
            revision_artifact_binding=revision_binding,
        )
    )
    for field in (
        "asset",
        "fragment",
        "evidence",
        "candidate_stage",
        "observation_stage",
        "validation_status",
    ):
        assert first_verified[field] == second_verified[field]
    assert first_verified["artifact"] != second_verified["artifact"]
    assert first_verified["fragment_hash"] != second_verified["fragment_hash"]


def test_builder_and_adapter_fail_closed_on_stale_or_tampered_bindings() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    revision_binding = _revision_binding(asset, revision)
    frontier_ids = set(revision["tree"]["frontier_node_ids"])
    hidden_node_id = next(
        node["node_id"]
        for node in asset["tree_result"]["tree"]["nodes"]
        if node["node_id"] not in frontier_ids
    )
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="current revision frontier node",
    ):
        build_interactive_tree_frontier_selection(
            revision,
            asset,
            revision_artifact_binding=revision_binding,
            source_node_id=hidden_node_id,
        )

    fragment = revision["fragments"][0]
    selection = build_interactive_tree_frontier_selection(
        revision,
        asset,
        revision_artifact_binding=revision_binding,
        source_node_id=fragment["source_node_id"],
    )
    selection_binding = _selection_binding(selection)

    tampered_selection = deepcopy(selection)
    tampered_selection["frontier"]["fragment_hash"] = "0" * 64
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="selection_id",
    ):
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            tampered_selection,
            revision,
            asset,
            selection_artifact_binding=selection_binding,
            revision_artifact_binding=revision_binding,
        )

    stale_selection_binding = deepcopy(selection_binding)
    stale_selection_binding["canonical_bytes"] += b" "
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="canonical bytes",
    ):
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            selection,
            revision,
            asset,
            selection_artifact_binding=stale_selection_binding,
            revision_artifact_binding=revision_binding,
        )

    tampered_provenance_binding = deepcopy(selection_binding)
    tampered_provenance_binding["provenance"]["fragment_hash"] = "0" * 64
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="provenance",
    ):
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            selection,
            revision,
            asset,
            selection_artifact_binding=tampered_provenance_binding,
            revision_artifact_binding=revision_binding,
        )

    stale_revision_binding = deepcopy(revision_binding)
    stale_revision_binding["canonical_bytes"] += b" "
    with pytest.raises(
        InteractiveTreeFrontierSelectionError,
        match="canonical bytes",
    ):
        interactive_tree_frontier_selection_to_verified_candidate_fragment(
            selection,
            revision,
            asset,
            selection_artifact_binding=selection_binding,
            revision_artifact_binding=stale_revision_binding,
        )
