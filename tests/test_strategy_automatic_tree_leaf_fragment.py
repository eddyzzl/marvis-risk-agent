from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import (
    AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION,
    AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
    AutomaticTreeLeafFragmentError,
    automatic_tree_leaf_fragment_content_hash,
    automatic_tree_leaf_fragment_to_verified_candidate_fragment,
    build_automatic_tree_leaf_fragment,
    canonical_automatic_tree_leaf_fragment_json,
    validate_automatic_tree_leaf_fragment,
)
from marvis.packs.strategy.candidate_fragment import (
    validate_verified_candidate_fragment,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )


def _asset(
    *,
    task_id: str = "task-automatic-tree",
    sample_context_hash: str = HASH_D,
    max_depth: int = 2,
) -> dict:
    tree = build_weighted_rule_tree(
        _frame(),
        feature_cols=["x", "z"],
        target_col="bad",
        max_depth=max_depth,
        min_leaf_count=1,
    )
    return build_automatic_tree_asset(
        tree,
        task_id=task_id,
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=sample_context_hash,
        source_refs=[
            "workspace:task-automatic-tree:3",
            "dataset:dataset-labelled",
        ],
    )


def _tree_binding(asset: dict) -> dict:
    canonical_bytes = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    return {
        "artifact_id": "artifact-automatic-tree",
        "task_id": asset["identity"]["task_id"],
        "kind": AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        "artifact_schema_version": AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
        "content_hash": hashlib.sha256(canonical_bytes).hexdigest(),
        "origin_tool": AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        "path": (
            "/tasks/"
            + asset["identity"]["task_id"]
            + "/strategy_automatic_trees/"
            + asset["asset_id"]
            + ".json"
        ),
        "provenance": {
            "schema_version": AUTOMATIC_TREE_SOURCE_ARTIFACT_SCHEMA_VERSION,
            "producer_version": asset["producer_version"],
            "task_id": asset["identity"]["task_id"],
            "kind": AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
            "format": "json",
            "asset_id": asset["asset_id"],
            "asset_hash": asset["asset_hash"],
            "tree_result_hash": asset["tree_result"]["result_hash"],
            "dataset_id": asset["identity"]["dataset_id"],
            "dataset_content_hash": asset["identity"]["dataset_content_hash"],
            "workspace_revision": asset["identity"]["workspace_revision"],
            "workspace_generation": asset["identity"]["workspace_generation"],
            "semantic_mapping_hash": asset["identity"]["semantic_mapping_hash"],
            "registry_metadata_hash": asset["identity"]["registry_metadata_hash"],
            "sample_context_hash": asset["identity"]["sample_context_hash"],
        },
        "canonical_bytes": canonical_bytes,
    }


def _selection(
    asset: dict,
    *,
    leaf_id: str | None = None,
    selection_reason: str | None = None,
) -> dict:
    selected_leaf = leaf_id or asset["fragments"][0]["leaf_id"]
    return build_automatic_tree_leaf_fragment(
        asset,
        tree_artifact_binding=_tree_binding(asset),
        leaf_id=selected_leaf,
        selection_reason=selection_reason,
    )


def _selection_binding(selection: dict) -> dict:
    tree_artifact = selection["tree_artifact"]
    tree_asset = selection["tree_asset"]
    leaf = selection["leaf"]
    return {
        "artifact_id": "artifact-automatic-tree-leaf",
        "task_id": tree_artifact["task_id"],
        "kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "content_hash": automatic_tree_leaf_fragment_content_hash(selection),
        "origin_tool": AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
        "artifact_schema_version": (
            AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION
        ),
        "producer_version": selection["producer_version"],
        "selection_id": selection["selection_id"],
        "selection_hash": selection["selection_hash"],
        "tree_artifact_id": tree_artifact["artifact_id"],
        "tree_artifact_kind": tree_artifact["kind"],
        "tree_artifact_schema_version": tree_artifact["artifact_schema_version"],
        "tree_artifact_content_hash": tree_artifact["content_hash"],
        "tree_artifact_origin_tool": tree_artifact["origin_tool"],
        "tree_artifact_path": tree_artifact["path"],
        "tree_artifact_provenance": tree_artifact["provenance"],
        "tree_asset_schema_version": tree_asset["schema_version"],
        "tree_asset_id": tree_asset["asset_id"],
        "tree_asset_hash": tree_asset["asset_hash"],
        "tree_result_hash": tree_asset["tree_result_hash"],
        "leaf_id": leaf["leaf_id"],
        "fragment_id": leaf["fragment_id"],
        "fragment_hash": leaf["fragment_hash"],
        "rule_id": leaf["rule_id"],
        "effect_id": leaf["effect_id"],
    }


def _replace(value: dict, path: tuple[str, ...], replacement: object) -> dict:
    changed = deepcopy(value)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return changed


def _rehash_selection(payload: dict) -> dict:
    changed = deepcopy(payload)
    body = {
        key: value
        for key, value in changed.items()
        if key not in {"selection_id", "selection_hash"}
    }
    changed["selection_id"] = (
        "automatic-tree-leaf-selection-" + _sha256(_canonical_json(body))[:32]
    )
    without_hash = {
        key: value for key, value in changed.items() if key != "selection_hash"
    }
    changed["selection_hash"] = _sha256(_canonical_json(without_hash))
    return validate_automatic_tree_leaf_fragment(changed)


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_mapping_keys(child) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(child) for child in value))
    return set()


def test_selection_is_deterministic_canonical_and_self_authenticating() -> None:
    asset = _asset()
    first = _selection(asset)
    second = _selection(asset)

    assert first == second
    assert first["schema_version"] == AUTOMATIC_TREE_LEAF_FRAGMENT_SCHEMA_VERSION
    assert first["producer_version"] == (AUTOMATIC_TREE_LEAF_FRAGMENT_PRODUCER_VERSION)
    assert first["selection_reason"] is None
    assert first["selection_id"].startswith("automatic-tree-leaf-selection-")
    assert len(first["selection_hash"]) == 64
    assert json.loads(canonical_automatic_tree_leaf_fragment_json(first)) == first
    assert validate_automatic_tree_leaf_fragment(first) == first

    detached = validate_automatic_tree_leaf_fragment(first)
    detached["leaf"]["leaf_id"] = "changed"
    assert first["leaf"]["leaf_id"] != "changed"


def test_tree_provenance_mapping_order_does_not_change_selection_hash() -> None:
    asset = _asset()
    original_binding = _tree_binding(asset)
    reordered_binding = {
        **original_binding,
        "provenance": dict(reversed(list(original_binding["provenance"].items()))),
    }

    original = build_automatic_tree_leaf_fragment(
        asset,
        tree_artifact_binding=original_binding,
        leaf_id=asset["fragments"][0]["leaf_id"],
    )
    reordered = build_automatic_tree_leaf_fragment(
        asset,
        tree_artifact_binding=reordered_binding,
        leaf_id=asset["fragments"][0]["leaf_id"],
    )

    assert reordered == original


def test_leaf_selection_rejects_unknown_sample_design_partition() -> None:
    selection = _selection(_asset())
    selection["tree_artifact"]["provenance"]["sample_design_ref"] = {
        "artifact_id": HASH_A,
        "artifact_content_hash": HASH_B,
        "sample_design_id": "strategy-sample-design-" + ("c" * 24),
        "sample_design_content_hash": HASH_D,
        "partition": "shadow/development",
    }

    with pytest.raises(
        AutomaticTreeLeafFragmentError,
        match="sample_design_ref.partition",
    ):
        _rehash_selection(selection)


def test_every_leaf_is_explicitly_selectable_and_has_distinct_identity() -> None:
    asset = _asset()
    selections = [
        _selection(asset, leaf_id=fragment["leaf_id"])
        for fragment in asset["fragments"]
    ]

    assert len(selections) == len(asset["fragments"])
    assert len({item["selection_id"] for item in selections}) == len(selections)
    assert len({item["selection_hash"] for item in selections}) == len(selections)
    for selection, fragment in zip(selections, asset["fragments"], strict=True):
        assert selection["leaf"] == {
            key: fragment[key]
            for key in (
                "leaf_id",
                "fragment_id",
                "fragment_hash",
                "rule_id",
                "effect_id",
            )
        }


def test_reason_changes_audit_event_identity_and_hash_not_rule_facts() -> None:
    asset = _asset()
    without_reason = _selection(asset)
    with_reason = _selection(asset, selection_reason="  Analyst\t review\npassed  ")

    assert with_reason["selection_reason"] == "Analyst review passed"
    assert with_reason["selection_id"] != without_reason["selection_id"]
    assert with_reason["selection_hash"] != without_reason["selection_hash"]
    assert with_reason["leaf"] == without_reason["leaf"]
    assert with_reason["tree_asset"] == without_reason["tree_asset"]
    assert with_reason["tree_artifact"] == without_reason["tree_artifact"]


def test_payload_is_pointer_only_and_omits_executable_or_measured_facts() -> None:
    selection = _selection(_asset(), selection_reason="chosen for review")
    keys = _all_mapping_keys(selection)

    assert {
        "tree",
        "tree_result",
        "nodes",
        "rules",
        "condition",
        "requirements",
        "metrics",
        "effect",
        "action",
        "business_action",
        "canonical_bytes",
    }.isdisjoint(keys)


@pytest.mark.parametrize(
    "path",
    [
        ("unknown",),
        ("tree_artifact", "unknown"),
        ("tree_asset", "unknown"),
        ("leaf", "unknown"),
    ],
)
def test_unknown_fields_fail_closed(path: tuple[str, ...]) -> None:
    selection = _selection(_asset(), selection_reason="reviewed")
    with pytest.raises(AutomaticTreeLeafFragmentError, match="unsupported fields"):
        validate_automatic_tree_leaf_fragment(_replace(selection, path, True))


def test_non_string_unknown_field_fails_with_contract_error() -> None:
    selection = _selection(_asset())
    selection["leaf"][1] = "unsupported"
    with pytest.raises(AutomaticTreeLeafFragmentError, match="unsupported fields"):
        validate_automatic_tree_leaf_fragment(selection)


@pytest.mark.parametrize(
    "path",
    [
        ("selection_reason",),
        ("tree_artifact", "artifact_id"),
        ("tree_asset", "asset_hash"),
        ("leaf", "effect_id"),
    ],
)
def test_missing_fields_fail_closed(path: tuple[str, ...]) -> None:
    selection = _selection(_asset())
    target = selection
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(AutomaticTreeLeafFragmentError, match="missing"):
        validate_automatic_tree_leaf_fragment(selection)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), "strategy.future.v1"),
        (("tree_artifact", "artifact_id"), "artifact-other"),
        (("tree_artifact", "task_id"), "task-other"),
        (("tree_artifact", "kind"), "strategy_other_json"),
        (("tree_artifact", "artifact_schema_version"), "strategy.other.v1"),
        (("tree_artifact", "content_hash"), HASH_E),
        (("tree_artifact", "origin_tool"), "strategy.other"),
        (("tree_artifact", "path"), "/forged/tree.json"),
        (
            ("tree_artifact", "provenance"),
            {"schema_version": "forged.v1"},
        ),
        (("tree_asset", "schema_version"), "strategy.future-tree.v1"),
        (("tree_asset", "asset_id"), "candidate-asset-" + "0" * 32),
        (("tree_asset", "asset_hash"), HASH_E),
        (("tree_asset", "tree_result_hash"), HASH_E),
        (("leaf", "leaf_id"), "leaf-other"),
        (("leaf", "fragment_id"), "candidate-fragment-" + "0" * 32),
        (("leaf", "fragment_hash"), HASH_E),
        (("leaf", "rule_id"), "rule-other"),
        (("leaf", "effect_id"), "candidate-effect-" + "0" * 32),
        (("selection_reason",), "different reason"),
        (("producer_version",), "strategy.automatic-tree-leaf-fragment/2"),
        (("selection_id",), "automatic-tree-leaf-selection-" + "0" * 32),
        (("selection_hash",), HASH_E),
    ],
)
def test_every_authenticated_field_tamper_fails_closed(
    path: tuple[str, ...], replacement: object
) -> None:
    selection = _selection(_asset(), selection_reason="reviewed")
    with pytest.raises(AutomaticTreeLeafFragmentError):
        validate_automatic_tree_leaf_fragment(_replace(selection, path, replacement))


@pytest.mark.parametrize("leaf_id", [None, "", "   ", "leaf-unknown"])
def test_builder_rejects_non_explicit_or_unknown_leaf(leaf_id: str | None) -> None:
    asset = _asset()
    with pytest.raises(AutomaticTreeLeafFragmentError, match="leaf_id"):
        build_automatic_tree_leaf_fragment(
            asset,
            tree_artifact_binding=_tree_binding(asset),
            leaf_id=leaf_id,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("artifact_id", "", "artifact_id"),
        ("task_id", "task-other", "task_id"),
        ("kind", "strategy_other_json", "kind"),
        ("artifact_schema_version", "strategy.other.v1", "schema"),
        ("content_hash", HASH_E, "content_hash"),
        ("origin_tool", "strategy.other", "origin_tool"),
    ],
)
def test_builder_rejects_wrong_full_tree_artifact_binding(
    field: str, replacement: str, message: str
) -> None:
    asset = _asset()
    binding = {**_tree_binding(asset), field: replacement}
    with pytest.raises(AutomaticTreeLeafFragmentError, match=message):
        build_automatic_tree_leaf_fragment(
            asset,
            tree_artifact_binding=binding,
            leaf_id=asset["fragments"][0]["leaf_id"],
        )


def test_builder_rejects_unknown_tree_binding_field() -> None:
    asset = _asset()
    binding = {**_tree_binding(asset), "registry_row_version": 1}
    with pytest.raises(AutomaticTreeLeafFragmentError, match="unsupported fields"):
        build_automatic_tree_leaf_fragment(
            asset,
            tree_artifact_binding=binding,
            leaf_id=asset["fragments"][0]["leaf_id"],
        )


def test_builder_rejects_provenance_that_copies_measured_content() -> None:
    asset = _asset()
    binding = _tree_binding(asset)
    binding["provenance"] = {
        **binding["provenance"],
        "metrics": {"bad_rate": 0.5},
    }

    with pytest.raises(AutomaticTreeLeafFragmentError, match="unsupported fields"):
        build_automatic_tree_leaf_fragment(
            asset,
            tree_artifact_binding=binding,
            leaf_id=asset["fragments"][0]["leaf_id"],
        )


def test_adapter_replays_all_references_and_projects_verified_fragment() -> None:
    asset = _asset()
    asset_before = canonical_automatic_tree_asset_json(asset)
    selection = _selection(asset, selection_reason="analyst selected this leaf")
    selection_before = canonical_automatic_tree_leaf_fragment_json(selection)
    binding = _selection_binding(selection)
    selected = asset["fragments"][0]

    fragment = automatic_tree_leaf_fragment_to_verified_candidate_fragment(
        selection,
        asset,
        selection_artifact_binding=binding,
        tree_artifact_binding=_tree_binding(asset),
    )

    assert validate_verified_candidate_fragment(fragment) == fragment
    assert fragment["artifact"] == {
        "artifact_id": binding["artifact_id"],
        "artifact_kind": AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
        "artifact_schema_version": (
            AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION
        ),
        "artifact_content_hash": binding["content_hash"],
        "origin_tool": AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    }
    assert fragment["asset"] == {
        "schema_version": asset["schema_version"],
        "asset_id": asset["asset_id"],
        "asset_hash": asset["asset_hash"],
        "asset_type": asset["asset_type"],
    }
    assert fragment["fragment"] == {
        "fragment_id": selected["fragment_id"],
        "fragment_type": "strategy_rule",
        "rule_id": selected["rule_id"],
        "condition": selected["condition"],
        "requirements": selected["requirements"],
        "effect_id": selected["effect_id"],
    }
    assert fragment["evidence"] == {
        "evidence_id": asset["candidate_evidence"]["candidate_id"],
        "evidence_hash": asset["candidate_evidence"]["evidence_hash"],
        "identity": {
            key: asset["identity"][key]
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
    assert fragment["candidate_stage"] == "development"
    assert fragment["observation_stage"] == "backtested"
    assert fragment["validation_status"] == "unvalidated"
    assert canonical_automatic_tree_asset_json(asset) == asset_before
    assert canonical_automatic_tree_leaf_fragment_json(selection) == selection_before


def test_adapter_rejects_forged_registry_id_against_live_tree_binding() -> None:
    asset = _asset()
    selection = _selection(asset)
    forged = _rehash_selection(
        _replace(
            selection,
            ("tree_artifact", "artifact_id"),
            "artifact-forged-but-self-consistent",
        )
    )

    with pytest.raises(AutomaticTreeLeafFragmentError, match="artifact_id"):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            forged,
            asset,
            selection_artifact_binding=_selection_binding(forged),
            tree_artifact_binding=_tree_binding(asset),
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("path", "/forged/tree.json", "path"),
        ("content_hash", HASH_E, "content_hash"),
        (
            "provenance",
            {"schema_version": "forged.v1"},
            "provenance",
        ),
        ("origin_tool", "strategy.forged", "origin_tool"),
        ("canonical_bytes", b"{}", "canonical bytes"),
    ],
)
def test_adapter_rejects_tree_registry_binding_drift(
    field: str,
    replacement: object,
    message: str,
) -> None:
    asset = _asset()
    selection = _selection(asset)
    tree_binding = {**_tree_binding(asset), field: replacement}

    with pytest.raises(AutomaticTreeLeafFragmentError, match=message):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            selection,
            asset,
            selection_artifact_binding=_selection_binding(selection),
            tree_artifact_binding=tree_binding,
        )


def test_adapter_rejects_self_consistent_provenance_not_matching_asset() -> None:
    asset = _asset()
    selection = _selection(asset)
    forged_provenance = {
        **selection["tree_artifact"]["provenance"],
        "asset_hash": HASH_E,
    }
    forged_selection = _rehash_selection(
        _replace(
            selection,
            ("tree_artifact", "provenance"),
            forged_provenance,
        )
    )
    forged_tree_binding = {
        **_tree_binding(asset),
        "provenance": forged_provenance,
    }

    with pytest.raises(AutomaticTreeLeafFragmentError, match="provenance"):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            forged_selection,
            asset,
            selection_artifact_binding=_selection_binding(forged_selection),
            tree_artifact_binding=forged_tree_binding,
        )


def test_adapter_rejects_self_consistent_unknown_or_mismatched_leaf() -> None:
    asset = _asset()
    selection = _selection(asset)

    for path, replacement, message in (
        (("leaf", "leaf_id"), "leaf-unknown", "leaf_id"),
        (("leaf", "rule_id"), "rule-other", "rule_id"),
        (("tree_asset", "tree_result_hash"), HASH_E, "tree_result_hash"),
    ):
        forged = _rehash_selection(_replace(selection, path, replacement))
        binding = _selection_binding(forged)
        with pytest.raises(AutomaticTreeLeafFragmentError, match=message):
            automatic_tree_leaf_fragment_to_verified_candidate_fragment(
                forged,
                asset,
                selection_artifact_binding=binding,
                tree_artifact_binding=_tree_binding(asset),
            )


def test_adapter_rejects_wrong_tree_asset_task_and_selection_binding() -> None:
    asset = _asset()
    selection = _selection(asset)
    binding = _selection_binding(selection)

    wrong_asset = _asset(sample_context_hash=HASH_E, max_depth=1)
    with pytest.raises(AutomaticTreeLeafFragmentError, match="tree asset"):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            selection,
            wrong_asset,
            selection_artifact_binding=binding,
            tree_artifact_binding=_tree_binding(asset),
        )

    wrong_task_asset = _asset(task_id="task-other")
    with pytest.raises(AutomaticTreeLeafFragmentError, match="task_id"):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            selection,
            wrong_task_asset,
            selection_artifact_binding=binding,
            tree_artifact_binding=_tree_binding(asset),
        )

    for field, replacement, message in (
        ("artifact_id", "", "artifact_id"),
        ("task_id", "task-other", "task_id"),
        ("kind", "strategy_other_json", "kind"),
        ("artifact_schema_version", "strategy.other.v1", "schema"),
        ("origin_tool", "strategy.other", "origin_tool"),
        ("content_hash", HASH_E, "content_hash"),
        (
            "producer_version",
            "strategy.automatic-tree-leaf-fragment/2",
            "producer_version",
        ),
        (
            "selection_id",
            "automatic-tree-leaf-selection-" + "0" * 32,
            "selection_id",
        ),
        ("selection_hash", HASH_E, "selection_hash"),
        ("tree_artifact_id", "artifact-other", "tree_artifact_id"),
        ("tree_artifact_kind", "strategy_other_json", "tree_artifact_kind"),
        (
            "tree_artifact_schema_version",
            "strategy.other.v1",
            "tree_artifact_schema_version",
        ),
        (
            "tree_artifact_content_hash",
            HASH_E,
            "tree_artifact_content_hash",
        ),
        (
            "tree_artifact_origin_tool",
            "strategy.other",
            "tree_artifact_origin_tool",
        ),
        (
            "tree_artifact_path",
            "/forged/tree.json",
            "tree_artifact_path",
        ),
        (
            "tree_artifact_provenance",
            {"schema_version": "forged.v1"},
            "tree_artifact_provenance",
        ),
        (
            "tree_asset_schema_version",
            "strategy.other.v1",
            "tree_asset_schema_version",
        ),
        ("tree_asset_id", "candidate-asset-" + "0" * 32, "tree_asset_id"),
        ("tree_asset_hash", HASH_E, "tree_asset_hash"),
        ("tree_result_hash", HASH_E, "tree_result_hash"),
        ("leaf_id", "leaf-other", "leaf_id"),
        (
            "fragment_id",
            "candidate-fragment-" + "0" * 32,
            "fragment_id",
        ),
        ("fragment_hash", HASH_E, "fragment_hash"),
        ("rule_id", "rule-other", "rule_id"),
        ("effect_id", "candidate-effect-" + "0" * 32, "effect_id"),
    ):
        forged_binding = {**binding, field: replacement}
        with pytest.raises(AutomaticTreeLeafFragmentError, match=message):
            automatic_tree_leaf_fragment_to_verified_candidate_fragment(
                selection,
                asset,
                selection_artifact_binding=forged_binding,
                tree_artifact_binding=_tree_binding(asset),
            )


def test_adapter_binding_rejects_unknown_field() -> None:
    asset = _asset()
    selection = _selection(asset)
    binding = {**_selection_binding(selection), "selection_reason": "duplicate"}

    with pytest.raises(AutomaticTreeLeafFragmentError, match="unsupported fields"):
        automatic_tree_leaf_fragment_to_verified_candidate_fragment(
            selection,
            asset,
            selection_artifact_binding=binding,
            tree_artifact_binding=_tree_binding(asset),
        )


def test_schema_reference_is_the_committed_full_tree_contract() -> None:
    selection = _selection(_asset())
    assert selection["tree_asset"]["schema_version"] == (
        AUTOMATIC_TREE_ASSET_SCHEMA_VERSION
    )
