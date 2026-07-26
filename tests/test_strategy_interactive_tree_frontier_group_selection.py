from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import build_automatic_tree_asset
from marvis.packs.strategy.interactive_tree_frontier_group_selection import (
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION,
    INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
    InteractiveTreeFrontierGroupSelectionError,
    build_interactive_tree_frontier_group_selection,
    canonical_interactive_tree_frontier_group_selection_json,
    interactive_tree_frontier_group_selection_to_verified_candidate_fragment,
    validate_interactive_tree_frontier_group_selection,
)
from marvis.packs.strategy.interactive_tree_frontier_group_tools import (
    interactive_tree_frontier_group_selection_provenance,
)
from marvis.packs.strategy.interactive_tree_revision import (
    build_interactive_tree_revision,
)
from tests.test_strategy_interactive_tree_frontier_selection import (
    HASH_A,
    HASH_B,
    HASH_C,
    HASH_D,
    SAMPLE_DESIGN_REF,
    _automatic_asset,
    _revision,
    _revision_binding,
)


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_recursive_keys(child) for child in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(child) for child in value), set())
    return set()


def _selection_binding(selection: dict) -> dict:
    canonical = canonical_interactive_tree_frontier_group_selection_json(
        selection
    ).encode("utf-8")
    return {
        "artifact_id": "artifact-interactive-tree-frontier-group-selection",
        "task_id": selection["revision_artifact"]["task_id"],
        "kind": INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_KIND,
        "artifact_schema_version": (
            INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ARTIFACT_SCHEMA_VERSION
        ),
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "origin_tool": INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ORIGIN_TOOL,
        "path": (
            "/tasks/"
            + selection["revision_artifact"]["task_id"]
            + "/strategy_interactive_tree_frontier_group_selections/"
            + selection["selection_id"]
            + ".json"
        ),
        "provenance": interactive_tree_frontier_group_selection_provenance(
            selection
        ),
        "canonical_bytes": canonical,
    }


def test_group_selection_canonicalizes_by_live_frontier_order_and_is_pointer_only() -> (
    None
):
    asset = _automatic_asset()
    revision = _revision(asset)
    frontier = revision["tree"]["frontier_node_ids"]
    assert len(frontier) >= 2

    selection = build_interactive_tree_frontier_group_selection(
        revision,
        asset,
        revision_artifact_binding=_revision_binding(asset, revision),
        source_node_ids=[frontier[1], frontier[0]],
        selection_reason="  Combine policy-equivalent branches.  ",
    )

    assert selection["source_node_ids"] == frontier[:2]
    assert selection["selection_reason"] == "Combine policy-equivalent branches."
    assert selection["group_id"].startswith("interactive-tree-frontier-group-")
    assert selection["selection_id"].startswith(
        "interactive-tree-frontier-group-selection-"
    )
    assert {
        "condition",
        "requirements",
        "metrics",
        "action",
        "fragment_id",
        "rule_id",
        "effect_id",
    }.isdisjoint(_recursive_keys(selection))
    assert validate_interactive_tree_frontier_group_selection(selection) == selection
    assert json.loads(
        canonical_interactive_tree_frontier_group_selection_json(selection)
    ) == selection

    replayed = (
        interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
            selection,
            revision,
            asset,
            selection_artifact_binding=_selection_binding(selection),
            revision_artifact_binding=_revision_binding(asset, revision),
        )
    )
    selected = revision["fragments"][:2]
    assert replayed["fragment"]["condition"] == {
        "op": "or",
        "args": [item["condition"] for item in selected],
    }
    assert replayed["fragment"]["requirements"] == []
    assert replayed["fragment"]["fragment_id"].startswith(
        "candidate-fragment-"
    )
    assert replayed["fragment"]["rule_id"].startswith("candidate-rule-")
    assert replayed["fragment"]["effect_id"].startswith("candidate-effect-")


def test_group_selection_reason_changes_only_audit_identity() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    source_node_ids = revision["tree"]["frontier_node_ids"][:2]
    binding = _revision_binding(asset, revision)

    first = build_interactive_tree_frontier_group_selection(
        revision,
        asset,
        revision_artifact_binding=binding,
        source_node_ids=source_node_ids,
        selection_reason="First audit reason.",
    )
    second = build_interactive_tree_frontier_group_selection(
        revision,
        asset,
        revision_artifact_binding=binding,
        source_node_ids=list(reversed(source_node_ids)),
        selection_reason="Second audit reason.",
    )

    assert first["group_id"] == second["group_id"]
    assert first["source_node_ids"] == second["source_node_ids"]
    assert first["selection_id"] != second["selection_id"]
    assert first["selection_hash"] != second["selection_hash"]
    first_fragment = (
        interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
            first,
            revision,
            asset,
            selection_artifact_binding=_selection_binding(first),
            revision_artifact_binding=binding,
        )
    )
    second_fragment = (
        interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
            second,
            revision,
            asset,
            selection_artifact_binding=_selection_binding(second),
            revision_artifact_binding=binding,
        )
    )
    assert first_fragment["asset"] == second_fragment["asset"]
    assert first_fragment["fragment"] == second_fragment["fragment"]
    assert first_fragment["evidence"] == second_fragment["evidence"]
    assert first_fragment["artifact"] != second_fragment["artifact"]
    assert first_fragment["fragment_hash"] != second_fragment["fragment_hash"]


@pytest.mark.parametrize(
    ("source_node_ids", "match"),
    [
        ([], "2 to 50"),
        (["node-" + "1" * 20], "2 to 50"),
        (
            ["node-" + f"{index:020x}" for index in range(51)],
            "2 to 50",
        ),
    ],
)
def test_group_selection_requires_two_to_fifty_nodes(
    source_node_ids: list[str],
    match: str,
) -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    with pytest.raises(InteractiveTreeFrontierGroupSelectionError, match=match):
        build_interactive_tree_frontier_group_selection(
            revision,
            asset,
            revision_artifact_binding=_revision_binding(asset, revision),
            source_node_ids=source_node_ids,
        )


def test_group_selection_rejects_duplicate_and_unknown_frontier_nodes() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    frontier = revision["tree"]["frontier_node_ids"]
    binding = _revision_binding(asset, revision)
    with pytest.raises(
        InteractiveTreeFrontierGroupSelectionError,
        match="duplicate",
    ):
        build_interactive_tree_frontier_group_selection(
            revision,
            asset,
            revision_artifact_binding=binding,
            source_node_ids=[frontier[0], frontier[0]],
        )

    unknown = next(
        node["node_id"]
        for node in asset["tree_result"]["tree"]["nodes"]
        if node["node_id"] not in set(frontier)
    )
    with pytest.raises(
        InteractiveTreeFrontierGroupSelectionError,
        match="current revision frontier",
    ):
        build_interactive_tree_frontier_group_selection(
            revision,
            asset,
            revision_artifact_binding=binding,
            source_node_ids=[frontier[0], unknown],
        )


def test_group_selection_validation_and_replay_reject_tampering() -> None:
    asset = _automatic_asset()
    revision = _revision(asset)
    binding = _revision_binding(asset, revision)
    selection = build_interactive_tree_frontier_group_selection(
        revision,
        asset,
        revision_artifact_binding=binding,
        source_node_ids=revision["tree"]["frontier_node_ids"][:2],
    )
    forged = deepcopy(selection)
    forged["source_node_ids"] = list(reversed(forged["source_node_ids"]))
    with pytest.raises(
        InteractiveTreeFrontierGroupSelectionError,
        match="group_id|canonical",
    ):
        validate_interactive_tree_frontier_group_selection(forged)

    stale_binding = _selection_binding(selection)
    stale_binding["canonical_bytes"] += b" "
    with pytest.raises(
        InteractiveTreeFrontierGroupSelectionError,
        match="canonical bytes",
    ):
        interactive_tree_frontier_group_selection_to_verified_candidate_fragment(
            selection,
            revision,
            asset,
            selection_artifact_binding=stale_binding,
            revision_artifact_binding=binding,
        )


def test_group_selection_accepts_exactly_fifty_current_frontier_nodes() -> None:
    rng = np.random.RandomState(42)
    row_count = 3000
    frame = pd.DataFrame(
        {
            **{
                f"x{index}": rng.normal(size=row_count)
                for index in range(10)
            },
            "bad": rng.randint(0, 2, size=row_count),
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=[f"x{index}" for index in range(10)],
        target_col="bad",
        max_depth=8,
        min_leaf_count=1,
    )
    sample_ref_token = "strategy-sample-design:" + json.dumps(
        {"kind": "strategy_sample_design", **SAMPLE_DESIGN_REF},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    asset = build_automatic_tree_asset(
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
    split_node = next(
        node
        for node in reversed(asset["tree_result"]["tree"]["nodes"])
        if node["kind"] == "split"
    )
    revision = build_interactive_tree_revision(
        asset,
        node_id=split_node["node_id"],
        reason="Create a large reviewed frontier.",
    )
    assert len(revision["tree"]["frontier_node_ids"]) >= 50

    selection = build_interactive_tree_frontier_group_selection(
        revision,
        asset,
        revision_artifact_binding=_revision_binding(asset, revision),
        source_node_ids=revision["tree"]["frontier_node_ids"][:50],
    )

    assert len(selection["source_node_ids"]) == 50
