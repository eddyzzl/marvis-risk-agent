from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_split_search import (
    INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION,
    search_interactive_tree_split_candidates,
    validate_interactive_tree_split_search,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan, 11],
            "z": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _search(**overrides: object) -> dict:
    frame = _frame()
    inputs = {
        "frame": frame,
        "node_mask": np.ones(len(frame), dtype=bool),
        "node_id": "node-" + "a" * 20,
        "source_tree_id": "candidate-asset-" + "b" * 32,
        "features": ["x", "z"],
        "target": np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1]),
        "weights": None,
        "medians": {"x": 5.0, "z": 0.5},
        "directions": {"x": "increasing", "z": "unordered"},
        "min_leaf_count": 2,
        "max_thresholds_per_feature": 4,
        "max_row_evaluations": 1_000,
    }
    inputs.update(overrides)
    return search_interactive_tree_split_candidates(**inputs)


def test_search_is_deterministic_bounded_and_aggregate_only() -> None:
    first = _search()
    second = _search()

    assert first == second
    assert first["schema_version"] == (
        INTERACTIVE_TREE_SPLIT_SEARCH_SCHEMA_VERSION
    )
    assert first["search_id"].startswith("interactive-tree-split-search-")
    assert first["request"]["features"] == ["x", "z"]
    assert first["budget"]["evaluated_candidates"] <= 8
    assert first["budget"]["row_evaluations"] <= 1_000
    assert first["candidates"]
    assert any(candidate["eligible"] is True for candidate in first["candidates"])
    assert first["candidates"][0]["eligible"] is True
    assert all(
        candidate["candidate_id"].startswith("interactive-tree-split-candidate-")
        for candidate in first["candidates"]
    )
    assert all(
        set(candidate)
        == {
            "candidate_id",
            "rank",
            "feature",
            "threshold",
            "missing_child",
            "eligible",
            "failures",
            "gain",
            "parent",
            "left",
            "right",
            "direction",
        }
        for candidate in first["candidates"]
    )
    assert "winner" not in first
    assert "selected" not in first
    assert all(
        "records" not in candidate and "row_payload" not in candidate
        for candidate in first["candidates"]
    )
    assert validate_interactive_tree_split_search(first) == first


def test_search_supports_single_feature_candidate_exploration() -> None:
    result = _search(features=["x"], max_thresholds_per_feature=3)

    assert result["request"]["features"] == ["x"]
    assert {candidate["feature"] for candidate in result["candidates"]} == {"x"}
    assert result["budget"]["evaluated_candidates"] <= 3


def test_search_rejects_unbounded_or_invalid_inputs() -> None:
    with pytest.raises(StrategyError, match="row evaluation"):
        _search(max_row_evaluations=1)
    with pytest.raises(StrategyError, match="feature"):
        _search(features=["x", "x"])
    with pytest.raises(StrategyError, match="node mask"):
        _search(node_mask=np.array([True, False]))
    with pytest.raises(StrategyError, match="minimum leaf"):
        _search(min_leaf_count=7)


def test_search_validation_detects_metric_or_identity_tampering() -> None:
    result = _search()
    tampered = copy.deepcopy(result)
    tampered["candidates"][0]["gain"] += 0.01

    with pytest.raises(
        StrategyError,
        match="hash|identity|canonical|eligibility",
    ):
        validate_interactive_tree_split_search(tampered)
