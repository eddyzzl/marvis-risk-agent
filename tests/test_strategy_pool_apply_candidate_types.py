from __future__ import annotations

from pathlib import Path

from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_apply_tools import run_apply_strategy_pool
from tests.test_strategy_pool_automatic_tree import (
    _add_inputs as _tree_add_inputs,
)
from tests.test_strategy_pool_automatic_tree import (
    _materialize as _materialize_tree,
)
from tests.test_strategy_pool_automatic_tree import _setup as _tree_setup
from tests.test_strategy_pool_cross_matrix import (
    _add_inputs as _cross_add_inputs,
)
from tests.test_strategy_pool_cross_matrix import (
    _materialize as _materialize_cross,
)
from tests.test_strategy_cross_matrix_cell_selection_tool import (
    _fixture as _cross_setup,
)


def _apply(pool: dict, *, ctx, runtime) -> dict:
    return run_apply_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
        },
        ctx,
        runtime,
    )


def _assert_assignment_conservation(result: dict) -> None:
    row_count = result["source"]["row_count"]
    assert result["result"]["row_count"] == row_count
    assert sum(result["action_counts"].values()) == row_count
    assert sum(result["rule_counts"].values()) + result["default_count"] == row_count
    assert (
        sum(result["entry_counts"].values()) + result["default_count"]
        == row_count
    )
    assert result["activated"] is False
    assert result["adopted"] is False
    assert result["deployed"] is False


def test_apply_current_pool_executes_automatic_tree_leaf(
    tmp_path: Path,
) -> None:
    fx = _tree_setup(tmp_path)
    selected = _materialize_tree(fx)
    added = strategy_tools.tool_add_candidate_to_pool(
        _tree_add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx["ctx"],
    )

    result = _apply(added["pool"], ctx=fx["ctx"], runtime=fx["runtime"])

    _assert_assignment_conservation(result)
    [entry] = added["entries"]
    assert result["rule_counts"][entry["rule_id"]] > 0
    assert result["entry_counts"][entry["entry_id"]] > 0


def test_apply_current_pool_executes_cross_matrix_cell_group(
    tmp_path: Path,
) -> None:
    fx = _cross_setup(tmp_path)
    selected = _materialize_cross(
        fx,
        [fx.populated[2]["cell_id"], fx.populated[0]["cell_id"]],
        reason="governed cross group",
    )
    added = strategy_tools.tool_add_candidate_to_pool(
        _cross_add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx.ctx,
    )

    result = _apply(added["pool"], ctx=fx.ctx, runtime=fx.runtime)

    _assert_assignment_conservation(result)
    [entry] = added["entries"]
    assert result["rule_counts"][entry["rule_id"]] > 0
    assert result["entry_counts"][entry["entry_id"]] > 0
