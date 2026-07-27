from __future__ import annotations

import copy

import numpy as np
import pytest

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.interactive_tree_continuation import (
    continue_interactive_tree_subtree,
)
from marvis.packs.strategy.interactive_tree_replay import (
    replay_interactive_tree_split,
    replay_interactive_tree_threshold,
)
from marvis.packs.strategy.interactive_tree_revision import (
    build_interactive_tree_revision,
)
from marvis.packs.strategy.interactive_tree_revision_v2 import (
    build_adjusted_interactive_tree_revision_v2,
    build_continued_interactive_tree_revision_v2,
    build_pruned_interactive_tree_revision_v2,
    build_replaced_interactive_tree_split_revision_v2,
    validate_interactive_tree_revision_v2,
)
from marvis.packs.strategy.interactive_tree_split_search import (
    search_interactive_tree_split_candidates,
)


pytest_plugins = ("tests.test_strategy_interactive_tree_tool",)


def _continuation(scenario):
    source = scenario.source_asset
    frame = scenario.development_frame.reset_index(drop=True)
    target = frame["bad"].to_numpy(dtype=np.int8)
    weights = frame["weight"].to_numpy(dtype=float)
    root_id = source["tree_result"]["tree"]["root_node_id"]
    parent = build_interactive_tree_revision(source, node_id=root_id)
    search = search_interactive_tree_split_candidates(
        frame,
        node_mask=np.ones(len(frame), dtype=bool),
        node_id=root_id,
        source_tree_id=parent["revision_id"],
        features=["x", "z"],
        target=target,
        weights=weights,
        medians=source["tree_result"]["preprocessing"]["medians"],
        directions=source["tree_result"]["directions"],
        min_leaf_count=1,
        max_thresholds_per_feature=4,
        max_row_evaluations=10_000,
    )
    seed = next(item for item in search["candidates"] if item["eligible"])
    controls = {
        "features": ["x", "z"],
        "max_additional_depth": 3,
        "min_gini_gain": 0.0,
        "max_generated_nodes": 15,
        "max_thresholds_per_feature": 4,
        "max_row_evaluations": 100_000,
    }
    result = continue_interactive_tree_subtree(
        frame,
        source,
        source_tree_id=parent["revision_id"],
        node_id=root_id,
        seed_candidate=seed,
        features=controls["features"],
        target=target,
        weights=weights,
        loan_values=frame["loan_amount"].to_numpy(dtype=float),
        overdue_values=frame["overdue_amount"].to_numpy(dtype=float),
        parent_revision=parent,
        ancestor_revisions=(),
        max_additional_depth=controls["max_additional_depth"],
        min_gini_gain=controls["min_gini_gain"],
        max_generated_nodes=controls["max_generated_nodes"],
        max_thresholds_per_feature=controls[
            "max_thresholds_per_feature"
        ],
        max_row_evaluations=controls["max_row_evaluations"],
        objective="max_gini_gain",
        tie_break="eligible_gain_feature_threshold_candidate_id",
    )
    revision = build_continued_interactive_tree_revision_v2(
        source,
        node_id=root_id,
        search_id=search["search_id"],
        search_hash=search["search_hash"],
        candidate_id=seed["candidate_id"],
        feature=seed["feature"],
        threshold=seed["threshold"],
        missing_child=seed["missing_child"],
        controls=controls,
        reason="Continue the reviewed frontier within explicit limits.",
        continuation=result,
        parent_revision=parent,
    )
    return frame, target, weights, parent, search, result, revision


def test_continuation_is_bounded_deterministic_and_validated(scenario) -> None:
    (
        _frame,
        _target,
        _weights,
        parent,
        search,
        result,
        revision,
    ) = _continuation(scenario)

    assert result.replay["exactly_once"] is True
    assert result.replay["observed"]["generated_node_count"] <= 15
    assert result.replay["observed"]["row_evaluations"] <= 100_000
    assert revision["edit"]["search_id"] == search["search_id"]
    assert revision["edit"]["candidate_id"] == (
        search["candidates"][0]["candidate_id"]
    )
    assert revision["tree"]["frontier_node_ids"] == list(
        result.frontier_node_ids
    )
    assert validate_interactive_tree_revision_v2(
        revision,
        scenario.source_asset,
        parent_revision=parent,
    ) == revision
    assert all(
        "records" not in node and "row_payload" not in node
        for node in revision["tree"]["nodes"]
    )

    tampered = copy.deepcopy(revision)
    tampered["edit"]["controls"]["max_additional_depth"] = 2
    with pytest.raises(
        StrategyError,
        match="canonical|controls|hash|decision policy",
    ):
        validate_interactive_tree_revision_v2(
            tampered,
            scenario.source_asset,
            parent_revision=parent,
        )


def test_generated_split_supports_later_threshold_edit_and_prune(
    scenario,
) -> None:
    frame, target, weights, parent, _search, _result, revision = (
        _continuation(scenario)
    )
    generated_split = next(
        item
        for item in revision["tree"]["nodes"]
        if item["kind"] == "split"
        and item["node_id"] != revision["edit"]["node_id"]
    )
    threshold = generated_split["threshold"] + 0.1
    replay = replay_interactive_tree_threshold(
        frame,
        scenario.source_asset,
        node_id=generated_split["node_id"],
        threshold=threshold,
        target=target,
        weights=weights,
        loan_values=frame["loan_amount"].to_numpy(dtype=float),
        overdue_values=frame["overdue_amount"].to_numpy(dtype=float),
        parent_revision=revision,
    )
    adjusted = build_adjusted_interactive_tree_revision_v2(
        scenario.source_asset,
        node_id=generated_split["node_id"],
        threshold=threshold,
        reason="Review one generated split.",
        replay=replay,
        parent_revision=revision,
        ancestor_revisions=(parent,),
    )
    assert next(
        item
        for item in adjusted["tree"]["nodes"]
        if item["node_id"] == generated_split["node_id"]
    )["threshold"] == threshold

    pruned = build_pruned_interactive_tree_revision_v2(
        scenario.source_asset,
        node_id=generated_split["node_id"],
        reason="Stop the generated branch here.",
        parent_revision=adjusted,
        ancestor_revisions=(revision, parent),
    )
    assert generated_split["node_id"] in pruned["tree"]["frontier_node_ids"]
    assert validate_interactive_tree_revision_v2(
        pruned,
        scenario.source_asset,
        parent_revision=adjusted,
        ancestor_revisions=(revision, parent),
    ) == pruned


def test_generated_split_supports_later_feature_replacement(scenario) -> None:
    frame, target, weights, parent, _search, _result, revision = (
        _continuation(scenario)
    )
    generated_split = next(
        item
        for item in revision["tree"]["nodes"]
        if item["kind"] == "split"
        and item["node_id"] != revision["edit"]["node_id"]
    )
    replacement_feature = next(
        feature
        for feature in scenario.source_asset["tree_result"]["training"][
            "feature_order"
        ]
        if feature != generated_split["feature"]
    )
    threshold = float(frame[replacement_feature].median())
    replay = replay_interactive_tree_split(
        frame,
        scenario.source_asset,
        node_id=generated_split["node_id"],
        feature=replacement_feature,
        threshold=threshold,
        target=target,
        weights=weights,
        loan_values=frame["loan_amount"].to_numpy(dtype=float),
        overdue_values=frame["overdue_amount"].to_numpy(dtype=float),
        parent_revision=revision,
    )
    replaced = build_replaced_interactive_tree_split_revision_v2(
        scenario.source_asset,
        node_id=generated_split["node_id"],
        feature=replacement_feature,
        threshold=threshold,
        reason="Replace one generated split with a reviewed feature.",
        replay=replay,
        parent_revision=revision,
        ancestor_revisions=(parent,),
    )

    replaced_node = next(
        item
        for item in replaced["tree"]["nodes"]
        if item["node_id"] == generated_split["node_id"]
    )
    assert replaced_node["feature"] == replacement_feature
    assert replaced_node["threshold"] == threshold
    assert validate_interactive_tree_revision_v2(
        replaced,
        scenario.source_asset,
        parent_revision=revision,
        ancestor_revisions=(parent,),
    ) == replaced
