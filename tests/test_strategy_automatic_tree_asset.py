from __future__ import annotations

import copy
import hashlib
import json

import pandas as pd
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import (
    AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
    AutomaticTreeAssetError,
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.errors import StrategyError


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )


def _tree(*, with_direction_violations: bool = False) -> dict:
    direction = "decreasing" if with_direction_violations else "unordered"
    return build_weighted_rule_tree(
        _frame(),
        feature_cols=["x", "z"],
        target_col="bad",
        directions={"x": direction, "z": direction},
        max_depth=2,
        min_leaf_count=1,
    )


def _asset(
    tree_result: dict | None = None,
    **overrides: object,
) -> dict:
    arguments: dict[str, object] = {
        "task_id": "task-automatic-tree",
        "dataset_id": "dataset-labelled",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 7,
        "semantic_mapping_hash": HASH_B,
        "registry_metadata_hash": HASH_C,
        "sample_context_hash": HASH_D,
        "source_refs": [
            "workspace:task-automatic-tree:3",
            "dataset:dataset-labelled",
        ],
    }
    arguments.update(overrides)
    return build_automatic_tree_asset(
        tree_result if tree_result is not None else _tree(),
        **arguments,
    )


def _rehash_tree(tree_result: dict) -> None:
    body = {key: value for key, value in tree_result.items() if key != "result_hash"}
    tree_result["result_hash"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_builds_deterministic_full_tree_asset_and_canonical_roundtrip() -> None:
    first = _asset()
    second = _asset()

    assert first == second
    assert first["schema_version"] == AUTOMATIC_TREE_ASSET_SCHEMA_VERSION
    assert first["asset_type"] == "automatic_rule_tree"
    assert first["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
    }
    assert first["identity"] == {
        "task_id": "task-automatic-tree",
        "dataset_id": "dataset-labelled",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 7,
        "semantic_mapping_hash": HASH_B,
        "registry_metadata_hash": HASH_C,
        "sample_context_hash": HASH_D,
    }
    assert first["tree_result"] == _tree()
    assert first["source_refs"] == sorted(first["source_refs"])
    assert first["producer_version"] == "strategy.automatic-tree-asset/1"
    assert first["candidate_evidence"].keys() == {
        "candidate_id",
        "evidence_hash",
    }
    assert first["candidate_evidence"]["candidate_id"].startswith("candidate-")
    assert len(first["candidate_evidence"]["evidence_hash"]) == 64
    assert first["asset_id"].startswith("candidate-asset-")
    assert len(first["asset_hash"]) == 64

    raw = canonical_automatic_tree_asset_json(first)
    assert json.loads(raw) == first
    assert raw == canonical_automatic_tree_asset_json(json.loads(raw))

    detached = validate_automatic_tree_asset(first)
    detached["identity"]["task_id"] = "changed"
    assert first["identity"]["task_id"] == "task-automatic-tree"


def test_fragment_index_is_one_exact_independent_fragment_per_tree_rule() -> None:
    asset = _asset()
    rules = asset["tree_result"]["rules"]
    fragments = asset["fragments"]

    assert len(fragments) == len(rules)
    assert [fragment["leaf_id"] for fragment in fragments] == [
        rule["leaf_id"] for rule in rules
    ]
    assert len({fragment["fragment_id"] for fragment in fragments}) == len(fragments)
    assert len({fragment["fragment_hash"] for fragment in fragments}) == len(fragments)
    assert len({fragment["effect_id"] for fragment in fragments}) == len(fragments)
    for fragment, rule in zip(fragments, rules, strict=True):
        assert fragment.keys() == {
            "leaf_id",
            "fragment_id",
            "fragment_hash",
            "rule_id",
            "condition",
            "requirements",
            "effect_id",
            "metrics",
        }
        assert fragment["fragment_id"].startswith("candidate-fragment-")
        assert fragment["fragment_id"] != fragment["rule_id"]
        assert fragment["leaf_id"] == rule["leaf_id"]
        assert fragment["rule_id"] == rule["rule_id"]
        assert fragment["condition"] == rule["condition"]
        assert fragment["requirements"] == []
        assert fragment["metrics"] == rule["metrics"]


def test_direction_violations_and_red_flags_are_exact_tree_derivatives() -> None:
    tree = _tree(with_direction_violations=True)
    asset = _asset(tree)
    split_violations = [
        node
        for node in tree["tree"]["nodes"]
        if node["kind"] == "split"
        and node["direction_diagnostic"]["status"] == "violation"
    ]

    violations = asset["diagnostics"]["direction_violations"]
    red_flags = asset["diagnostics"]["red_flags"]
    assert [row["node_id"] for row in violations] == [
        row["node_id"] for row in split_violations
    ]
    assert violations == [
        {
            "node_id": node["node_id"],
            "feature": node["feature"],
            "expected_direction": node["direction_diagnostic"]["expected_direction"],
            "basis": node["direction_diagnostic"]["basis"],
            "primary_bad_rate_delta": node["direction_diagnostic"][
                "primary_bad_rate_delta"
            ],
        }
        for node in split_violations
    ]
    assert red_flags == [
        {
            "code": "direction_violation",
            "node_id": row["node_id"],
            "feature": row["feature"],
            "expected_direction": row["expected_direction"],
        }
        for row in violations
    ]


def test_no_direction_violation_means_empty_derived_diagnostics() -> None:
    asset = _asset()
    assert asset["diagnostics"] == {
        "direction_violations": [],
        "red_flags": [],
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("top",), "unknown"),
        (("lifecycle", "top"), "unknown"),
        (("identity", "top"), "unknown"),
        (("candidate_evidence", "top"), "unknown"),
        (("fragments", 0, "top"), "unknown"),
        (("diagnostics", "top"), "unknown"),
    ],
)
def test_unknown_fields_fail_closed(path: tuple[object, ...], value: str) -> None:
    tampered = copy.deepcopy(_asset())
    target: object = tampered
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(AutomaticTreeAssetError):
        validate_automatic_tree_asset(tampered)


@pytest.mark.parametrize(
    "field",
    [
        "dataset_content_hash",
        "semantic_mapping_hash",
        "registry_metadata_hash",
        "sample_context_hash",
    ],
)
def test_identity_hashes_are_strict_lowercase_sha256(field: str) -> None:
    with pytest.raises(AutomaticTreeAssetError):
        _asset(**{field: HASH_E[:-1]})


@pytest.mark.parametrize("field", ["workspace_revision", "workspace_generation"])
def test_identity_revisions_are_non_negative_integers(field: str) -> None:
    with pytest.raises(AutomaticTreeAssetError):
        _asset(**{field: -1})
    with pytest.raises(AutomaticTreeAssetError):
        _asset(**{field: True})


def test_tree_topology_tamper_fails_even_after_tree_result_rehash() -> None:
    tree = copy.deepcopy(_tree())
    root = next(
        node
        for node in tree["tree"]["nodes"]
        if node["node_id"] == tree["tree"]["root_node_id"]
    )
    root["left_child_id"] = root["right_child_id"]
    _rehash_tree(tree)

    with pytest.raises(AutomaticTreeAssetError):
        _asset(tree)


def test_tree_metric_tamper_fails_even_after_tree_result_rehash() -> None:
    tree = copy.deepcopy(_tree())
    tree["rules"][0]["metrics"]["unweighted"]["bad"] += 1
    _rehash_tree(tree)

    with pytest.raises(AutomaticTreeAssetError):
        _asset(tree)


def test_tree_result_unknown_field_and_lifecycle_claim_fail_closed() -> None:
    unknown = copy.deepcopy(_tree())
    unknown["unknown"] = True
    _rehash_tree(unknown)
    with pytest.raises(AutomaticTreeAssetError):
        _asset(unknown)

    adopted = copy.deepcopy(_tree())
    adopted["lifecycle"]["stage"] = "production"
    _rehash_tree(adopted)
    with pytest.raises(AutomaticTreeAssetError):
        _asset(adopted)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["fragments"].reverse(),
        lambda value: value["fragments"].append(copy.deepcopy(value["fragments"][0])),
        lambda value: value["fragments"][0].__setitem__("rule_id", "rule-other"),
        lambda value: value["fragments"][0].__setitem__("effect_id", "effect-other"),
        lambda value: value["fragments"][0].__setitem__("requirements", [{}]),
        lambda value: value["fragments"][0]["metrics"]["unweighted"].__setitem__(
            "bad", 999
        ),
    ],
)
def test_fragment_order_duplicate_and_content_tamper_fail_closed(mutator) -> None:
    tampered = copy.deepcopy(_asset())
    mutator(tampered)
    with pytest.raises(AutomaticTreeAssetError):
        validate_automatic_tree_asset(tampered)


def test_direction_diagnostic_and_red_flag_tamper_fail_closed() -> None:
    tampered = copy.deepcopy(_asset(_tree(with_direction_violations=True)))
    tampered["diagnostics"]["direction_violations"][0]["primary_bad_rate_delta"] = 0.0
    with pytest.raises(AutomaticTreeAssetError):
        validate_automatic_tree_asset(tampered)

    tampered = copy.deepcopy(_asset(_tree(with_direction_violations=True)))
    tampered["diagnostics"]["red_flags"] = []
    with pytest.raises(AutomaticTreeAssetError):
        validate_automatic_tree_asset(tampered)


def test_candidate_evidence_and_asset_ids_and_hashes_are_all_verified() -> None:
    mutators = [
        lambda value: value["candidate_evidence"].__setitem__(
            "candidate_id", "candidate-" + "0" * 32
        ),
        lambda value: value["candidate_evidence"].__setitem__(
            "evidence_hash", "0" * 64
        ),
        lambda value: value.__setitem__("asset_id", "candidate-asset-" + "0" * 32),
        lambda value: value.__setitem__("asset_hash", "0" * 64),
    ]
    for mutator in mutators:
        tampered = copy.deepcopy(_asset())
        mutator(tampered)
        with pytest.raises(AutomaticTreeAssetError):
            validate_automatic_tree_asset(tampered)


def test_identity_and_tree_content_change_candidate_and_asset_identity() -> None:
    baseline = _asset()
    changed_identity = _asset(sample_context_hash=HASH_E)
    changed_tree = _asset(
        build_weighted_rule_tree(
            _frame(),
            feature_cols=["x", "z"],
            target_col="bad",
            max_depth=1,
            min_leaf_count=1,
        )
    )

    for changed in (changed_identity, changed_tree):
        assert changed["candidate_evidence"] != baseline["candidate_evidence"]
        assert changed["asset_id"] != baseline["asset_id"]
        assert changed["asset_hash"] != baseline["asset_hash"]


def test_source_refs_are_required_unique_and_canonical() -> None:
    assert _asset(source_refs=["z:source", "a:source"])["source_refs"] == [
        "a:source",
        "z:source",
    ]
    with pytest.raises(AutomaticTreeAssetError):
        _asset(source_refs=[])
    with pytest.raises(AutomaticTreeAssetError):
        _asset(source_refs=["same", "same"])
    with pytest.raises(AutomaticTreeAssetError):
        _asset(source_refs=[" padded "])


def test_producer_and_lifecycle_cannot_claim_a_later_stage() -> None:
    with pytest.raises(AutomaticTreeAssetError):
        _asset(producer_version="strategy.automatic-tree-asset/2")

    tampered = copy.deepcopy(_asset())
    tampered["lifecycle"]["validation_status"] = "validated"
    with pytest.raises(AutomaticTreeAssetError):
        validate_automatic_tree_asset(tampered)


def test_error_type_is_a_strategy_error() -> None:
    assert issubclass(AutomaticTreeAssetError, StrategyError)
