import pytest

import marvis.db as db_module
from marvis.db import StrategyRepository, connect, init_db
from marvis.packs.strategy import BacktestResult, build_strategy


def _strategy():
    return build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="baseline cutoff",
    )


def _backtest_result(strategy_id: str) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id,
        approval_rate=0.7,
        approved_count=70,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.22,
        expected_profit=2300.0,
        swap_in_count=5,
        swap_out_count=8,
        swap_in_bad_rate=0.12,
        swap_out_bad_rate=0.01,
        by_segment=(
            {"decision": "approve", "count": 70, "bad_count": 3, "bad_rate": 0.04},
            {"decision": "reject", "count": 30, "bad_count": 7, "bad_rate": 0.22},
        ),
    )


def test_strategy_repository_round_trips_strategy_and_backtest(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    result = _backtest_result(strategy.id)

    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")
    repo.save_backtest(
        "backtest-1",
        strategy.id,
        "dataset-1",
        result,
        created_at="2026-06-19T00:01:00Z",
    )

    assert repo.get_strategy(strategy.id) == strategy
    assert repo.list_for_task("task-1") == [strategy]
    assert repo.get_backtest("backtest-1") == result
    assert repo.list_backtests(strategy.id) == [result]


def test_strategy_repository_creates_strategy_with_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()

    repo.create_strategy_with_audit(
        "task-1",
        strategy,
        audit={
            "kind": "strategy.create",
            "target_ref": strategy.id,
            "outcome": "succeeded",
            "detail": {"task_id": "task-1", "rule_count": len(strategy.rules)},
        },
        created_at="2026-06-19T00:00:00Z",
    )

    assert repo.get_strategy(strategy.id) == strategy
    audit = db_module.PluginRepository(db_path).list_audit(kind="strategy.create")[0]
    assert audit["target_ref"] == strategy.id
    assert audit["detail"]["rule_count"] == len(strategy.rules)


def test_strategy_repository_rolls_back_strategy_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.create_strategy_with_audit(
            "task-1",
            strategy,
            audit={
                "kind": "strategy.create",
                "target_ref": strategy.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_strategy(strategy.id) is None


def test_strategy_repository_saves_backtest_with_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    result = _backtest_result(strategy.id)
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    repo.save_backtest_with_audit(
        "backtest-1",
        strategy.id,
        "dataset-1",
        result,
        audit={
            "kind": "strategy.backtest",
            "target_ref": "backtest-1",
            "outcome": "succeeded",
            "detail": {"strategy_id": strategy.id, "dataset_id": "dataset-1"},
        },
        created_at="2026-06-19T00:01:00Z",
    )

    assert repo.get_backtest("backtest-1") == result
    audit = db_module.PluginRepository(db_path).list_audit(kind="strategy.backtest")[0]
    assert audit["target_ref"] == "backtest-1"
    assert audit["detail"]["strategy_id"] == strategy.id


def test_strategy_repository_rolls_back_backtest_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    result = _backtest_result(strategy.id)
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.save_backtest_with_audit(
            "backtest-1",
            strategy.id,
            "dataset-1",
            result,
            audit={
                "kind": "strategy.backtest",
                "target_ref": "backtest-1",
                "outcome": "succeeded",
            },
        )

    assert repo.get_backtest("backtest-1") is None


def test_strategy_backtests_cascade_when_strategy_is_deleted(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    result = _backtest_result(strategy.id)
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")
    repo.save_backtest(
        "backtest-1",
        strategy.id,
        "dataset-1",
        result,
        created_at="2026-06-19T00:01:00Z",
    )

    with connect(db_path) as conn:
        conn.execute("DELETE FROM strategies WHERE id = ?", (strategy.id,))

    assert repo.get_strategy(strategy.id) is None
    assert repo.get_backtest("backtest-1") is None
