from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import build_automatic_tree_asset
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_REVISION_SCHEMA_VERSION,
    InteractiveTreeRevisionError,
    build_interactive_tree_revision,
    canonical_interactive_tree_revision_json,
    validate_interactive_tree_revision,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


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
    return build_automatic_tree_asset(
        tree,
        task_id="task-interactive-tree",
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=[
            "workspace:task-interactive-tree:3",
            "dataset:dataset-labelled",
        ],
    )


def _nodes(source: dict) -> list[dict]:
    return source["tree_result"]["tree"]["nodes"]


def _split_nodes(source: dict) -> list[dict]:
    return [node for node in _nodes(source) if node["kind"] == "split"]


def _leaf_nodes(source: dict) -> list[dict]:
    return [node for node in _nodes(source) if node["kind"] == "leaf"]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _rehash_revision(revision: dict) -> None:
    body = {
        key: value
        for key, value in revision.items()
        if key not in {"revision_id", "revision_hash"}
    }
    revision["revision_id"] = f"interactive-tree-revision-{_digest(body)[:32]}"
    revision["revision_hash"] = _digest(
        {**body, "revision_id": revision["revision_id"]}
    )


def _rehash_semantic_revision(revision: dict) -> None:
    semantic_body = {
        "schema_version": "strategy.interactive-tree-semantic.v1",
        "base_tree": revision["base_tree"],
        "frontier_node_ids": revision["tree"]["frontier_node_ids"],
        "fragments": revision["fragments"],
    }
    semantic_hash = _digest(semantic_body)
    revision["semantic_tree_id"] = f"interactive-tree-{semantic_hash[:32]}"
    revision["tree"]["tree_hash"] = semantic_hash
    evidence_hash = _digest(
        {
            "schema_version": "strategy.interactive-tree-evidence.v1",
            "semantic_tree": semantic_body,
        }
    )
    revision["candidate_evidence"] = {
        "candidate_id": f"candidate-{evidence_hash[:32]}",
        "evidence_hash": evidence_hash,
    }
    _rehash_revision(revision)


def test_root_prune_builds_one_canonical_self_authenticating_revision() -> None:
    source = _automatic_asset()
    root_id = source["tree_result"]["tree"]["root_node_id"]

    revision = build_interactive_tree_revision(
        source,
        node_id=root_id,
        reason="  Entire tree is too granular.  ",
    )

    assert revision["schema_version"] == INTERACTIVE_TREE_REVISION_SCHEMA_VERSION
    assert revision["asset_type"] == "interactive_rule_tree"
    assert revision["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }
    assert revision["parent_revision"] is None
    assert revision["edit"] == {
        "operation": "prune_subtree",
        "node_id": root_id,
        "reason": "Entire tree is too granular.",
    }
    assert revision["tree"]["visible_node_ids"] == [root_id]
    assert revision["tree"]["frontier_node_ids"] == [root_id]
    assert len(revision["fragments"]) == 1
    assert revision["fragments"][0]["source_node_id"] == root_id
    assert revision["fragments"][0]["metrics"] == next(
        node["metrics"]
        for node in source["tree_result"]["tree"]["nodes"]
        if node["node_id"] == root_id
    )
    assert revision["checks"] == {
        "frontier_prefix_free": True,
        "all_base_leaves_covered_once": True,
        "fragment_source_matches": True,
        "metric_conservation": "passed",
    }
    assert revision["semantic_tree_id"].startswith("interactive-tree-")
    assert revision["revision_id"].startswith("interactive-tree-revision-")
    assert len(revision["revision_hash"]) == 64

    raw = canonical_interactive_tree_revision_json(revision, source)
    assert json.loads(raw) == revision
    assert raw == canonical_interactive_tree_revision_json(json.loads(raw), source)


def test_reason_changes_revision_identity_but_not_any_semantic_identity() -> None:
    source = _automatic_asset()
    node_id = _split_nodes(source)[-1]["node_id"]

    first = build_interactive_tree_revision(
        source,
        node_id=node_id,
        reason="Operational simplification.",
    )
    second = build_interactive_tree_revision(
        source,
        node_id=node_id,
        reason="Policy owner requested the merge.",
    )

    assert first["semantic_tree_id"] == second["semantic_tree_id"]
    assert first["tree"] == second["tree"]
    assert first["candidate_evidence"] == second["candidate_evidence"]
    assert first["fragments"] == second["fragments"]
    assert [
        (
            fragment["leaf_id"],
            fragment["fragment_id"],
            fragment["rule_id"],
            fragment["effect_id"],
        )
        for fragment in first["fragments"]
    ] == [
        (
            fragment["leaf_id"],
            fragment["fragment_id"],
            fragment["rule_id"],
            fragment["effect_id"],
        )
        for fragment in second["fragments"]
    ]
    assert first["revision_id"] != second["revision_id"]
    assert first["revision_hash"] != second["revision_hash"]


def test_internal_prunes_can_repeat_toward_root_and_branch_from_earlier_revision() -> None:
    source = _automatic_asset()
    root, middle, deepest = _split_nodes(source)
    first = build_interactive_tree_revision(
        source,
        node_id=deepest["node_id"],
        reason="First local merge.",
    )
    first_snapshot = copy.deepcopy(first)

    middle_branch = build_interactive_tree_revision(
        source,
        node_id=middle["node_id"],
        reason="Broaden the merged segment.",
        parent_revision=first,
    )
    root_branch = build_interactive_tree_revision(
        source,
        node_id=root["node_id"],
        reason="Use one catch-all frontier.",
        parent_revision=first,
    )

    assert first == first_snapshot
    assert middle_branch["parent_revision"]["revision_id"] == first["revision_id"]
    assert root_branch["parent_revision"]["revision_id"] == first["revision_id"]
    assert middle_branch["tree"]["frontier_node_ids"] != root_branch["tree"][
        "frontier_node_ids"
    ]
    assert middle_branch["tree"]["frontier_node_ids"] == [
        _leaf_nodes(source)[0]["node_id"],
        middle["node_id"],
    ]
    assert root_branch["tree"]["frontier_node_ids"] == [root["node_id"]]
    assert middle_branch["revision_id"] != root_branch["revision_id"]
    assert validate_interactive_tree_revision(
        middle_branch,
        source,
        parent_revision=first,
    ) == middle_branch
    assert validate_interactive_tree_revision(
        root_branch,
        source,
        parent_revision=first,
    ) == root_branch


def test_prune_rejects_leaf_hidden_and_already_frontier_nodes() -> None:
    source = _automatic_asset()
    _, middle, deepest = _split_nodes(source)
    leaf_id = _leaf_nodes(source)[0]["node_id"]
    with pytest.raises(InteractiveTreeRevisionError, match="split node"):
        build_interactive_tree_revision(source, node_id=leaf_id)

    first = build_interactive_tree_revision(source, node_id=deepest["node_id"])
    with pytest.raises(InteractiveTreeRevisionError, match="already a frontier"):
        build_interactive_tree_revision(
            source,
            node_id=deepest["node_id"],
            parent_revision=first,
        )

    middle_revision = build_interactive_tree_revision(
        source,
        node_id=middle["node_id"],
        parent_revision=first,
    )
    with pytest.raises(InteractiveTreeRevisionError, match="hidden"):
        build_interactive_tree_revision(
            source,
            node_id=deepest["node_id"],
            parent_revision=middle_revision,
            ancestor_revisions=(first,),
        )


def test_parent_from_another_automatic_tree_and_corrupt_parent_fail_closed() -> None:
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    parent = build_interactive_tree_revision(source, node_id=deepest_id)

    wrong_source = build_automatic_tree_asset(
        source["tree_result"],
        task_id="another-task",
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=["dataset:dataset-labelled", "workspace:another-task:3"],
    )
    with pytest.raises(InteractiveTreeRevisionError, match="base tree|identity"):
        build_interactive_tree_revision(
            wrong_source,
            node_id=_split_nodes(wrong_source)[0]["node_id"],
            parent_revision=parent,
        )

    corrupt = copy.deepcopy(parent)
    corrupt["revision_hash"] = "f" * 64
    with pytest.raises(InteractiveTreeRevisionError, match="revision_hash"):
        build_interactive_tree_revision(
            source,
            node_id=_split_nodes(source)[0]["node_id"],
            parent_revision=corrupt,
        )


def test_frontier_visible_and_fragment_indexes_are_always_in_base_preorder() -> None:
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    revision = build_interactive_tree_revision(source, node_id=deepest_id)
    source_order = {
        node["node_id"]: index for index, node in enumerate(_nodes(source))
    }

    assert revision["tree"]["visible_node_ids"] == sorted(
        revision["tree"]["visible_node_ids"],
        key=source_order.__getitem__,
    )
    assert revision["tree"]["frontier_node_ids"] == sorted(
        revision["tree"]["frontier_node_ids"],
        key=source_order.__getitem__,
    )
    assert [item["source_node_id"] for item in revision["fragments"]] == revision[
        "tree"
    ]["frontier_node_ids"]

    forged = copy.deepcopy(revision)
    forged["tree"]["frontier_node_ids"].reverse()
    forged["fragments"].reverse()
    _rehash_semantic_revision(forged)
    with pytest.raises(
        InteractiveTreeRevisionError,
        match="canonical|pre-order|frontier is inconsistent",
    ):
        validate_interactive_tree_revision(forged, source)


def test_frontier_is_prefix_free_covers_every_base_leaf_and_conserves_metrics() -> None:
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    revision = build_interactive_tree_revision(source, node_id=deepest_id)
    node_by_id = {node["node_id"]: node for node in _nodes(source)}
    frontier_paths = [
        tuple(node_by_id[node_id]["path"])
        for node_id in revision["tree"]["frontier_node_ids"]
    ]

    for left_index, left_path in enumerate(frontier_paths):
        for right_index, right_path in enumerate(frontier_paths):
            if left_index == right_index:
                continue
            assert right_path[: len(left_path)] != left_path
    for leaf in _leaf_nodes(source):
        leaf_path = tuple(leaf["path"])
        assert (
            sum(
                leaf_path[: len(frontier_path)] == frontier_path
                for frontier_path in frontier_paths
            )
            == 1
        )

    root_metrics = _nodes(source)[0]["metrics"]["unweighted"]
    for field in ("total", "good", "bad"):
        assert sum(
            fragment["metrics"]["unweighted"][field]
            for fragment in revision["fragments"]
        ) == root_metrics[field]

    middle_revision = build_interactive_tree_revision(
        source,
        node_id=_split_nodes(source)[1]["node_id"],
        parent_revision=revision,
    )
    forged = copy.deepcopy(middle_revision)
    forged["tree"]["frontier_node_ids"].append(deepest_id)
    deepest_fragment = next(
        item
        for item in revision["fragments"]
        if item["source_node_id"] == deepest_id
    )
    forged["fragments"].append(copy.deepcopy(deepest_fragment))
    _rehash_semantic_revision(forged)
    with pytest.raises(InteractiveTreeRevisionError, match="prefix-free|visible"):
        validate_interactive_tree_revision(
            forged,
            source,
            parent_revision=revision,
        )


def test_tampered_visible_frontier_fragment_edit_source_refs_and_checks_fail_closed() -> (
    None
):
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    original = build_interactive_tree_revision(source, node_id=deepest_id)

    visible = copy.deepcopy(original)
    visible["tree"]["visible_node_ids"].reverse()
    _rehash_revision(visible)
    with pytest.raises(InteractiveTreeRevisionError, match="visible node index"):
        validate_interactive_tree_revision(visible, source)

    frontier = copy.deepcopy(original)
    frontier["tree"]["frontier_node_ids"] = frontier["tree"][
        "frontier_node_ids"
    ][:-1]
    frontier["fragments"] = frontier["fragments"][:-1]
    _rehash_semantic_revision(frontier)
    with pytest.raises(InteractiveTreeRevisionError, match="cover|frontier"):
        validate_interactive_tree_revision(frontier, source)

    fragment = copy.deepcopy(original)
    fragment["fragments"][0]["metrics"]["unweighted"]["bad"] += 1
    _rehash_semantic_revision(fragment)
    with pytest.raises(InteractiveTreeRevisionError, match="fragments changed"):
        validate_interactive_tree_revision(fragment, source)

    edit = copy.deepcopy(original)
    edit["edit"]["operation"] = "grow_subtree"
    _rehash_revision(edit)
    with pytest.raises(InteractiveTreeRevisionError, match="operation"):
        validate_interactive_tree_revision(edit, source)

    source_refs = copy.deepcopy(original)
    source_refs["source_refs"].reverse()
    _rehash_revision(source_refs)
    with pytest.raises(InteractiveTreeRevisionError, match="source_refs"):
        validate_interactive_tree_revision(source_refs, source)

    checks = copy.deepcopy(original)
    checks["checks"]["metric_conservation"] = "failed"
    _rehash_revision(checks)
    with pytest.raises(InteractiveTreeRevisionError, match="checks"):
        validate_interactive_tree_revision(checks, source)


@pytest.mark.parametrize(
    ("field_path", "replacement", "match"),
    [
        (("revision_hash",), "f" * 64, "revision_hash"),
        (("tree", "tree_hash"), "f" * 64, "tree_hash"),
        (
            ("candidate_evidence", "evidence_hash"),
            "f" * 64,
            "candidate evidence",
        ),
        (("fragments", 0, "fragment_hash"), "f" * 64, "fragments changed"),
    ],
)
def test_every_persisted_hash_is_authenticated(
    field_path: tuple[str | int, ...],
    replacement: str,
    match: str,
) -> None:
    source = _automatic_asset()
    revision = build_interactive_tree_revision(
        source,
        node_id=_split_nodes(source)[-1]["node_id"],
    )
    target: object = revision
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[index]
    target[field_path[-1]] = replacement  # type: ignore[index]

    with pytest.raises(InteractiveTreeRevisionError, match=match):
        validate_interactive_tree_revision(revision, source)


@pytest.mark.parametrize(
    "reason",
    ["", "   ", "\x00unsafe", "x" * 501, 123],
)
def test_invalid_reason_is_rejected(reason: object) -> None:
    source = _automatic_asset()
    with pytest.raises(InteractiveTreeRevisionError, match="reason"):
        build_interactive_tree_revision(
            source,
            node_id=_split_nodes(source)[-1]["node_id"],
            reason=reason,  # type: ignore[arg-type]
        )


def test_self_consistent_noncanonical_text_and_unknown_fields_fail_closed() -> None:
    source = _automatic_asset()
    revision = build_interactive_tree_revision(
        source,
        node_id=_split_nodes(source)[-1]["node_id"],
        reason="Canonical reason.",
    )

    padded_reason = copy.deepcopy(revision)
    padded_reason["edit"]["reason"] = "  Canonical reason.  "
    _rehash_revision(padded_reason)
    with pytest.raises(InteractiveTreeRevisionError, match="canonical|reason"):
        validate_interactive_tree_revision(padded_reason, source)

    padded_node = copy.deepcopy(revision)
    padded_node["edit"]["node_id"] = f" {revision['edit']['node_id']} "
    _rehash_revision(padded_node)
    with pytest.raises(InteractiveTreeRevisionError, match="canonical|node_id"):
        validate_interactive_tree_revision(padded_node, source)

    padded_visible = copy.deepcopy(revision)
    padded_visible["tree"]["visible_node_ids"][0] = (
        f" {padded_visible['tree']['visible_node_ids'][0]} "
    )
    _rehash_revision(padded_visible)
    with pytest.raises(InteractiveTreeRevisionError, match="canonical|visible"):
        validate_interactive_tree_revision(padded_visible, source)

    unknown = copy.deepcopy(revision)
    unknown["caller_claim"] = True
    with pytest.raises(InteractiveTreeRevisionError, match="unexpected"):
        validate_interactive_tree_revision(unknown, source)


def test_child_requires_the_exact_parent_binding_not_a_semantically_equal_parent() -> (
    None
):
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    middle_id = _split_nodes(source)[1]["node_id"]
    selected_parent = build_interactive_tree_revision(
        source,
        node_id=deepest_id,
        reason="Selected audit trail.",
    )
    same_semantics_other_parent = build_interactive_tree_revision(
        source,
        node_id=deepest_id,
        reason="Different audit trail.",
    )
    assert (
        selected_parent["semantic_tree_id"]
        == same_semantics_other_parent["semantic_tree_id"]
    )
    assert selected_parent["revision_id"] != same_semantics_other_parent["revision_id"]

    child = build_interactive_tree_revision(
        source,
        node_id=middle_id,
        parent_revision=selected_parent,
    )
    with pytest.raises(InteractiveTreeRevisionError, match="parent revision binding"):
        validate_interactive_tree_revision(
            child,
            source,
            parent_revision=same_semantics_other_parent,
        )

    tampered_binding = copy.deepcopy(child)
    tampered_binding["parent_revision"]["revision_hash"] = "f" * 64
    tampered_binding["source_refs"][-1] = (
        "interactive-tree-revision:"
        f"{tampered_binding['parent_revision']['revision_id']}"
        "@sha256:"
        f"{tampered_binding['parent_revision']['revision_hash']}"
    )
    _rehash_revision(tampered_binding)
    with pytest.raises(InteractiveTreeRevisionError, match="parent revision binding"):
        validate_interactive_tree_revision(
            tampered_binding,
            source,
            parent_revision=selected_parent,
        )

    padded_binding = copy.deepcopy(child)
    padded_binding["parent_revision"]["revision_id"] = (
        f" {padded_binding['parent_revision']['revision_id']} "
    )
    _rehash_revision(padded_binding)
    with pytest.raises(InteractiveTreeRevisionError, match="canonical"):
        validate_interactive_tree_revision(
            padded_binding,
            source,
            parent_revision=selected_parent,
        )


def test_validator_rejects_a_self_consistent_noop_repeated_frontier_edit() -> None:
    source = _automatic_asset()
    deepest_id = _split_nodes(source)[-1]["node_id"]
    parent = build_interactive_tree_revision(source, node_id=deepest_id)
    forged_child = copy.deepcopy(parent)
    parent_ref = {
        "revision_id": parent["revision_id"],
        "revision_hash": parent["revision_hash"],
        "semantic_tree_id": parent["semantic_tree_id"],
        "tree_hash": parent["tree"]["tree_hash"],
    }
    forged_child["parent_revision"] = parent_ref
    forged_child["edit"] = {
        "operation": "prune_subtree",
        "node_id": deepest_id,
        "reason": "Pretend the same frontier was pruned again.",
    }
    forged_child["source_refs"].append(
        f"interactive-tree-revision:{parent['revision_id']}"
        f"@sha256:{parent['revision_hash']}"
    )
    _rehash_revision(forged_child)

    with pytest.raises(InteractiveTreeRevisionError, match="already.*frontier|no-op"):
        validate_interactive_tree_revision(
            forged_child,
            source,
            parent_revision=parent,
        )


def test_validator_requires_authenticated_parent_evidence_for_a_child() -> None:
    source = _automatic_asset()
    split_ids = [node["node_id"] for node in _split_nodes(source)]
    parent = build_interactive_tree_revision(source, node_id=split_ids[-1])
    child = build_interactive_tree_revision(
        source,
        node_id=split_ids[1],
        parent_revision=parent,
    )

    with pytest.raises(
        InteractiveTreeRevisionError,
        match="parent revision evidence is required",
    ):
        validate_interactive_tree_revision(child, source)


def test_validator_requires_and_accepts_the_complete_parent_chain() -> None:
    source = _automatic_asset()
    root, middle, deepest = _split_nodes(source)
    first = build_interactive_tree_revision(
        source,
        node_id=deepest["node_id"],
    )
    second = build_interactive_tree_revision(
        source,
        node_id=middle["node_id"],
        parent_revision=first,
    )
    third = build_interactive_tree_revision(
        source,
        node_id=root["node_id"],
        parent_revision=second,
        ancestor_revisions=(first,),
    )

    with pytest.raises(
        InteractiveTreeRevisionError,
        match="parent revision evidence is required",
    ):
        validate_interactive_tree_revision(
            third,
            source,
            parent_revision=second,
        )

    assert (
        validate_interactive_tree_revision(
            third,
            source,
            parent_revision=second,
            ancestor_revisions=(first,),
        )
        == third
    )
