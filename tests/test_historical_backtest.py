"""Tests for the as-of historical replay kernel (historical_backtest.py)."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from marvis.packs.strategy.candidate_fragment import build_verified_candidate_fragment
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.historical_backtest import (
    EVIDENCE_STAGE_BACKTESTED,
    HISTORICAL_REPLAY_DECLARATION,
    VALIDATION_STATUS_UNVALIDATED,
    HistoricalBacktestError,
    MissingMonthSnapshotsError,
    canonical_historical_backtest_json,
    replay_historical_months,
    resolve_month_snapshots,
    resolve_month_snapshots_from_registry,
    resolve_strategy_spec,
)
from marvis.packs.strategy.pool import add_verified_candidate_fragment


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _condition(field: str, operator: str, value: int) -> dict:
    return {
        "op": "compare",
        "field": field,
        "operator": operator,
        "value": value,
        "missing": "no_match",
    }


def _action(action_type: str) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": None if action_type == "approval" else action_type.upper(),
        "stop": True,
    }


def _approval_spec() -> StrategySpec:
    """score < 3 -> reject; score >= 8 -> review; otherwise approve."""
    return StrategySpec(
        strategy_type="approval",
        default_action=StrategyAction(type="approval"),
        rules=(
            StrategyRuleSpec(
                rule_id="reject_low",
                priority=10,
                condition=_condition("score", "<", 3),
                action=StrategyAction(type="reject"),
            ),
            StrategyRuleSpec(
                rule_id="review_high",
                priority=20,
                condition=_condition("score", ">=", 8),
                action=StrategyAction(type="review"),
            ),
        ),
    )


def _spec_dict() -> dict:
    return _approval_spec().to_dict()


def _frame_202501() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [1, 2, 4, 9, 5],
            "target": [1, 0, 1, 1, 0],
            "amount": [100.0, 100.0, 100.0, 100.0, None],
        }
    )


def _frame_202502() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "score": [3, 6, 7, 1, 8],
            "target": [0, 1, 0, 1, 0],
            "amount": [200.0, 200.0, 200.0, 200.0, 200.0],
        }
    )


def _identity() -> dict:
    return {
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 1,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _fragment(index: int, condition: dict) -> dict:
    suffix = f"{index:064x}"
    return build_verified_candidate_fragment(
        artifact={
            "artifact_id": f"artifact-{index}",
            "artifact_kind": "test_candidate_json",
            "artifact_schema_version": "test.candidate-artifact.v1",
            "artifact_content_hash": suffix,
            "origin_tool": "strategy.test_candidate",
        },
        asset={
            "schema_version": "test.candidate.v1",
            "asset_id": f"candidate-asset-{index}",
            "asset_hash": suffix,
            "asset_type": "test_candidate",
        },
        fragment_type="strategy_rule",
        rule_id=f"candidate-rule-{index}",
        condition=condition,
        requirements=[],
        effect_id=f"candidate-effect-{index}",
        evidence_id="candidate-evidence-1",
        evidence_hash=HASH_D,
        evidence_identity=_identity(),
    )


def _build_pool() -> dict:
    default = _action("approval")
    conditions = (_condition("score", "<", 3), _condition("score", ">=", 8))
    actions = (_action("reject"), _action("review"))
    result = None
    for index, (condition, action) in enumerate(
        zip(conditions, actions, strict=True), start=1
    ):
        result = add_verified_candidate_fragment(
            result,
            task_id="task-1",
            strategy_type="approval",
            default_action=default,
            verified_candidate_fragment=_fragment(index, condition),
            action=action,
        )
    assert result is not None
    return result


def test_replay_historical_months_golden_numbers() -> None:
    result = replay_historical_months(
        _spec_dict(),
        [("202501", _frame_202501()), ("202502", _frame_202502())],
        target_col="target",
        amount_col="amount",
    )

    assert result.month_count == 2
    assert [month.month for month in result.months] == ["202501", "202502"]

    jan, feb = result.months
    # 202501: scores 1,2 reject; 4,5 approve; 9 review.
    assert jan.count == 5
    assert jan.pass_count == 2
    assert jan.pass_rate == 0.4
    assert jan.reject_count == 2
    assert jan.review_count == 1
    assert jan.labelled_pass_count == 2
    assert jan.label_coverage == 1.0
    assert jan.bad_count == 1  # passed target 1 (score 4), target 0 (score 5)
    assert jan.bad_rate == 0.5
    assert jan.amount is not None
    assert jan.amount.coverage_count == 1  # score 5 amount is missing
    assert jan.amount.coverage_rate == 0.5
    assert jan.amount.sum == 100.0

    # 202502: scores 3,6,7 approve; 1 reject; 8 review.
    assert feb.count == 5
    assert feb.pass_count == 3
    assert feb.pass_rate == 0.6
    assert feb.reject_count == 1
    assert feb.review_count == 1
    assert feb.labelled_pass_count == 3
    assert feb.bad_count == 1  # passed targets 0,1,0
    assert feb.bad_rate == 1 / 3
    assert feb.amount is not None
    assert feb.amount.coverage_count == 3
    assert feb.amount.coverage_rate == 1.0
    assert feb.amount.sum == 600.0

    assert result.total_count == 10
    assert result.total_pass_count == 5
    assert result.total_pass_rate == 0.5
    assert result.total_reject_count == 3
    assert result.total_review_count == 2
    assert result.total_labelled_pass_count == 5
    assert result.total_bad_count == 2
    assert result.total_bad_rate == 0.4


def test_result_carries_backtested_unvalidated_markers() -> None:
    result = replay_historical_months(
        _spec_dict(), [("202501", _frame_202501())], target_col="target"
    )

    assert result.evidence_stage == "backtested"
    assert result.evidence_stage == EVIDENCE_STAGE_BACKTESTED
    assert result.validation_status == "unvalidated"
    assert result.validation_status == VALIDATION_STATUS_UNVALIDATED
    assert result.declaration == HISTORICAL_REPLAY_DECLARATION

    payload = result.to_dict()
    assert payload["evidence_stage"] == "backtested"
    assert payload["validation_status"] == "unvalidated"
    assert "非真实业绩" in payload["declaration"]
    assert "backtested" in payload["declaration"]


def test_replay_does_not_mutate_inputs() -> None:
    spec = _spec_dict()
    spec_before = copy.deepcopy(spec)
    jan = _frame_202501()
    feb = _frame_202502()
    jan_before = jan.copy(deep=True)
    feb_before = feb.copy(deep=True)

    replay_historical_months(
        spec,
        [("202501", jan), ("202502", feb)],
        target_col="target",
        amount_col="amount",
    )

    assert spec == spec_before
    pd.testing.assert_frame_equal(jan, jan_before)
    pd.testing.assert_frame_equal(feb, feb_before)


def test_replay_is_deterministic() -> None:
    def run():
        return replay_historical_months(
            _spec_dict(),
            [("202501", _frame_202501()), ("202502", _frame_202502())],
            target_col="target",
            amount_col="amount",
        )

    first = run()
    second = run()

    assert first.content_hash == second.content_hash
    assert first.to_dict() == second.to_dict()
    assert canonical_historical_backtest_json(first) == canonical_historical_backtest_json(
        second
    )


def test_resolve_month_snapshots_splits_and_normalizes() -> None:
    frame = pd.concat(
        [
            _frame_202501().assign(month="202501"),
            _frame_202502().assign(month="202502"),
        ],
        ignore_index=True,
    )
    snapshots = resolve_month_snapshots(
        frame,
        month_col="month",
        requested_months=["202502", "202501", "202501"],
    )
    assert [snapshot.month for snapshot in snapshots] == ["202501", "202502"]
    assert len(snapshots[0].frame) == 5
    assert len(snapshots[1].frame) == 5


def test_resolve_month_snapshots_missing_month_fails_closed() -> None:
    frame = _frame_202501().assign(month="202501")
    with pytest.raises(MissingMonthSnapshotsError) as exc:
        resolve_month_snapshots(
            frame,
            month_col="month",
            requested_months=["202501", "202503", "202504"],
        )
    assert exc.value.missing_months == ("202503", "202504")
    assert "202503" in str(exc.value)
    assert "202504" in str(exc.value)


class _FakeRegistry:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        self.reads = 0

    def read_authenticated_parquet_snapshot(self, dataset_id: str) -> pd.DataFrame:
        self.reads += 1
        return self._frame


def test_resolve_month_snapshots_from_registry() -> None:
    frame = pd.concat(
        [
            _frame_202501().assign(month="202501"),
            _frame_202502().assign(month="202502"),
        ],
        ignore_index=True,
    )
    registry = _FakeRegistry(frame)
    snapshots = resolve_month_snapshots_from_registry(
        registry,
        "ds-1",
        month_col="month",
        requested_months=["202501", "202502"],
    )
    assert registry.reads == 1
    assert [snapshot.month for snapshot in snapshots] == ["202501", "202502"]


def test_resolve_month_snapshots_from_real_registry(tmp_path) -> None:
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.db import DatasetRepository, init_db

    db_path = tmp_path / "app.sqlite"
    datasets_root = tmp_path / "datasets"
    init_db(db_path)
    repo = DatasetRepository(db_path)
    registry = DatasetRegistry(repo, DataBackend(datasets_root), datasets_root)

    frame = pd.concat(
        [
            _frame_202501().assign(month="202501"),
            _frame_202502().assign(month="202502"),
        ],
        ignore_index=True,
    )
    source = tmp_path / "snapshots.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_existing(source, task_id="task-1", role="strategy_sample")
    assert dataset.content_hash is not None

    snapshots = resolve_month_snapshots_from_registry(
        registry,
        dataset.id,
        month_col="month",
        requested_months=["202501", "202502"],
    )
    assert [snapshot.month for snapshot in snapshots] == ["202501", "202502"]
    assert len(snapshots[0].frame) == 5
    assert len(snapshots[1].frame) == 5


def test_resolve_strategy_spec_from_pool() -> None:
    pool = _build_pool()
    resolved = resolve_strategy_spec(pool)
    assert resolved.ref["kind"] == "strategy_pool"
    assert resolved.ref["pool_ref"]["revision"] == pool["revision"]
    assert resolved.ref["pool_ref"]["snapshot_hash"] == pool["snapshot_hash"]
    assert resolved.spec.strategy_type == "approval"


def test_replay_historical_months_from_pool() -> None:
    result = replay_historical_months(
        _build_pool(),
        [("202501", _frame_202501()), ("202502", _frame_202502())],
        target_col="target",
    )
    assert result.strategy_ref["kind"] == "strategy_pool"
    assert result.total_pass_count == 5
    assert result.total_reject_count == 3
    assert result.total_review_count == 2
    assert result.months[0].pass_rate == 0.4
    assert result.months[1].bad_rate == 1 / 3


def test_replay_rejects_non_approval_strategy() -> None:
    spec = StrategySpec(
        strategy_type="limit",
        default_action=StrategyAction(type="limit", value=1000),
    )
    with pytest.raises(HistoricalBacktestError, match="approval/reject"):
        replay_historical_months(spec, [("202501", _frame_202501())], target_col="target")


def test_replay_enforces_row_budget() -> None:
    with pytest.raises(HistoricalBacktestError, match="row budget"):
        replay_historical_months(
            _spec_dict(),
            [("202501", _frame_202501()), ("202502", _frame_202502())],
            target_col="target",
            row_budget=5,
        )


def test_replay_rejects_non_binary_target() -> None:
    frame = _frame_202501()
    frame = frame.assign(target=[1, 0, 1, 2, 0])
    with pytest.raises(HistoricalBacktestError, match="target"):
        replay_historical_months(
            _spec_dict(), [("202501", frame)], target_col="target"
        )


def test_replay_rejects_boolean_amount() -> None:
    frame = _frame_202501()
    frame = frame.assign(amount=[True, False, True, False, True])
    with pytest.raises(HistoricalBacktestError, match="amount"):
        replay_historical_months(
            _spec_dict(),
            [("202501", frame)],
            target_col="target",
            amount_col="amount",
        )


def test_target_bad_value_zero_inverts_polarity() -> None:
    frame = pd.DataFrame({"score": [4, 4], "target": [0, 0]})
    default = replay_historical_months(
        _spec_dict(), [("202501", frame)], target_col="target"
    )
    inverted = replay_historical_months(
        _spec_dict(), [("202501", frame)], target_col="target", target_bad_value=0
    )
    assert default.months[0].bad_count == 0
    assert default.months[0].bad_rate == 0.0
    assert inverted.months[0].bad_count == 2
    assert inverted.months[0].bad_rate == 1.0
