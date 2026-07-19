from contextlib import contextmanager
import hashlib
import json

import pandas as pd
import pytest

import marvis.db as db_module
import marvis.repositories.strategy as strategy_repo_module
from marvis.db import StrategyRepository, connect, init_db
from marvis.packs.strategy import (
    BacktestResult,
    apply_strategy,
    build_strategy,
    build_strategy_from_spec,
)
from marvis.packs.strategy.profit import ProfitParams
from marvis.packs.strategy.typed_backtest import ApprovalProfitInputs, run_typed_backtest
from marvis.state_machine import ConflictError
from marvis.strategy_adoption import AdoptionReasonError
from marvis.strategy_lifecycle import StrategyLifecycleError


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


def test_strategy_repository_rejects_backtest_owned_by_another_strategy(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)

    with pytest.raises(ValueError, match="strategy_id does not match"):
        repo.save_backtest(
            "backtest-mismatched-owner",
            strategy.id,
            "dataset-1",
            _backtest_result("another-strategy"),
        )

    assert repo.get_backtest("backtest-mismatched-owner") is None


def test_strategy_repository_round_trips_typed_backtest_with_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    result = run_typed_backtest(
        pd.DataFrame(
            {
                "score": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                "target": [1, 1, 0, 1, None, 0, 0, 0, 1, None],
                "ead": [200.0] * 10,
                "pd": [0.0] * 10,
            }
        ),
        strategy.spec,
        target_col="target",
        strategy_id=strategy.id,
        approval_profit_inputs=ApprovalProfitInputs(
            params=ProfitParams(
                annual_rate=0.125,
                funding_rate=0.0,
                lgd=0.5,
                operating_cost_per_loan=0.0,
                term_months=12,
            ),
            ead_col="ead",
            pd_col="pd",
        ),
    )
    repo.create_strategy("task-1", strategy)

    repo.save_backtest_with_audit(
        "typed-backtest-1",
        strategy.id,
        "dataset-1",
        result,
        audit={
            "kind": "strategy.backtest",
            "target_ref": "typed-backtest-1",
            "outcome": "succeeded",
            "detail": {
                "strategy_id": strategy.id,
                "schema_version": result.schema_version,
            },
        },
    )

    assert repo.get_backtest("typed-backtest-1") == result
    assert repo.list_backtests(strategy.id) == [result]
    with connect(db_path) as conn:
        stored = json.loads(
            conn.execute(
                "SELECT result_json FROM backtests WHERE id = 'typed-backtest-1'"
            ).fetchone()["result_json"]
        )
    assert stored == result.to_dict()
    audit = db_module.PluginRepository(db_path).list_audit(
        kind="strategy.backtest"
    )[0]
    assert audit["target_ref"] == "typed-backtest-1"
    assert audit["detail"]["schema_version"] == "strategy.backtest.v2"


def test_strategy_repository_reads_legacy_backtest_without_rewriting_json(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    legacy_json = json.dumps(
        {
            "strategy_id": strategy.id,
            "approval_rate": 0.7,
            "approved_count": 70,
            "approved_bad_rate": 0.04,
            "rejected_bad_rate": 0.22,
            "expected_profit": 2300.0,
            "swap_in_count": 5,
            "swap_out_count": 8,
            "swap_in_bad_rate": 0.12,
            "swap_out_bad_rate": 0.01,
            "by_segment": [],
        },
        separators=(", ", ": "),
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtests(id, strategy_id, dataset_id, result_json, created_at)
            VALUES ('legacy-backtest-1', ?, 'dataset-1', ?, '2026-07-18T00:00:00Z')
            """,
            (strategy.id, legacy_json),
        )

    loaded = repo.get_backtest("legacy-backtest-1")

    assert loaded == BacktestResult(
        strategy_id=strategy.id,
        approval_rate=0.7,
        approved_count=70,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.22,
        expected_profit=2300.0,
        swap_in_count=5,
        swap_out_count=8,
        swap_in_bad_rate=0.12,
        swap_out_bad_rate=0.01,
        by_segment=(),
    )
    with connect(db_path) as conn:
        stored_json = conn.execute(
            "SELECT result_json FROM backtests WHERE id = 'legacy-backtest-1'"
        ).fetchone()["result_json"]
    assert stored_json == legacy_json


def test_strategy_repository_rejects_unknown_backtest_schema_version(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtests(id, strategy_id, dataset_id, result_json, created_at)
            VALUES (
                'future-backtest-1', ?, 'dataset-1',
                '{"schema_version":"strategy.backtest.v999"}',
                '2026-07-18T00:00:00Z'
            )
            """,
            (strategy.id,),
        )

    with pytest.raises(ValueError, match="unsupported strategy backtest schema_version"):
        repo.get_backtest("future-backtest-1")


def test_strategy_repository_persists_canonical_dsl_as_authoritative_spec(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()

    repo.create_strategy("task-1", strategy)

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json, dsl_schema_version, dsl_content_hash "
            "FROM strategies WHERE id = ?",
            (strategy.id,),
        ).fetchone()
    payload = json.loads(row["dsl_json"])
    loaded = repo.get_strategy(strategy.id)

    assert row["dsl_schema_version"] == "strategy.dsl.v1"
    assert row["dsl_content_hash"] == hashlib.sha256(
        row["dsl_json"].encode("utf-8")
    ).hexdigest()
    assert payload["rules"][0]["rule_id"] == strategy.rules[0].rule_id
    assert loaded.spec.to_dict() == strategy.spec.to_dict()
    assert apply_strategy(pd.DataFrame({"score": [599, 600]}), loaded).tolist() == [
        "reject",
        "approve",
    ]


def test_migration_018_backfills_existing_canonical_strategy_hash(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute("ALTER TABLE strategies DROP COLUMN dsl_content_hash")
        conn.execute("PRAGMA user_version = 17")

    init_db(db_path)

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json, dsl_content_hash FROM strategies WHERE id = ?",
            (strategy.id,),
        ).fetchone()
    assert row["dsl_content_hash"] == hashlib.sha256(
        row["dsl_json"].encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "cleared_column",
    ("dsl_json", "dsl_schema_version", "dsl_content_hash"),
)
def test_repository_rejects_incomplete_canonical_dsl_columns(
    tmp_path,
    cleared_column: str,
) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute(
            f"UPDATE strategies SET {cleared_column} = NULL WHERE id = ?",
            (strategy.id,),
        )

    with pytest.raises(ValueError, match="canonical DSL columns are incomplete"):
        repo.get_strategy(strategy.id)
    with pytest.raises(ValueError, match="canonical DSL columns are incomplete"):
        repo.get_strategy_spec_hash(strategy.id)


def test_strategy_repository_round_trips_value_action_output_aliases(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {
                "type": "limit",
                "value": 1000,
                "output_value": {"legacy": "fallback"},
            },
            "rules": [
                {
                    "rule_id": "positive",
                    "priority": 10,
                    "condition": {
                        "op": "compare",
                        "field": "x",
                        "operator": ">",
                        "value": 0,
                    },
                    "action": {
                        "type": "limit",
                        "value": 2000,
                        "output_value": 1900,
                    },
                }
            ],
        }
    )

    repo.create_strategy("task-1", strategy)
    loaded = repo.get_strategy(strategy.id)

    assert loaded.spec.to_dict() == strategy.spec.to_dict()
    assert loaded.default_decision == {"legacy": "fallback"}
    assert loaded.rules[0].value == 2000
    assert apply_strategy(pd.DataFrame({"x": [0, 1]}), loaded).tolist() == [
        {"legacy": "fallback"},
        1900,
    ]


def test_strategy_repository_adapts_legacy_null_dsl_without_rewriting_row(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO strategies(
                id, task_id, strategy_type, rules_json, score_col,
                default_decision_json, description, created_at,
                dsl_json, dsl_schema_version
            ) VALUES (
                'legacy-1', 'task-1', 'approval',
                '[{"condition":"score < 600","decision":"reject","value":null}]',
                'score', '"approve"', 'legacy', '2026-07-18T00:00:00Z', NULL, NULL
            )
            """
        )

    strategy = repo.get_strategy("legacy-1")

    assert strategy.spec.schema_version == "strategy.dsl.v1"
    assert apply_strategy(pd.DataFrame({"score": [500, 700]}), strategy).tolist() == [
        "reject",
        "approve",
    ]
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json, dsl_schema_version FROM strategies WHERE id = 'legacy-1'"
        ).fetchone()
    assert row["dsl_json"] is None
    assert row["dsl_schema_version"] is None


def test_legacy_custom_default_decision_remains_row_equivalent(tmp_path) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO strategies(
                id, task_id, strategy_type, rules_json, score_col,
                default_decision_json, description, created_at,
                dsl_json, dsl_schema_version
            ) VALUES (
                'legacy-pass', 'task-1', 'approval', '[]', 'score',
                '"pass"', 'legacy custom token', '2026-07-18T00:00:00Z', NULL, NULL
            )
            """
        )

    strategy = repo.get_strategy("legacy-pass")

    assert strategy is not None
    assert strategy.spec.default_action.type == "approval"
    assert strategy.spec.default_action.value == "approve"
    assert strategy.spec.default_action.output_value == "pass"
    assert apply_strategy(pd.DataFrame({"score": [500, 700]}), strategy).tolist() == [
        "pass",
        "pass",
    ]


def test_repository_fails_closed_when_compatibility_rules_drift_from_dsl(tmp_path) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE strategies SET rules_json = ? WHERE id = ?",
            (
                '[{"condition":"score < 100","decision":"reject",'
                '"value":null,"rule_id":"tampered","priority":10,'
                '"reason_code":null}]',
                strategy.id,
            ),
        )

    with pytest.raises(
        ValueError, match="compatibility fields do not match canonical DSL"
    ):
        repo.get_strategy(strategy.id)


def test_repository_fails_closed_when_default_action_value_drifts_from_output_alias(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {
                "type": "limit",
                "value": 1000,
                "output_value": 900,
            },
            "rules": [],
        }
    )
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json FROM strategies WHERE id = ?",
            (strategy.id,),
        ).fetchone()
        payload = json.loads(row["dsl_json"])
        payload["default_action"]["value"] = 1100
        conn.execute(
            "UPDATE strategies SET dsl_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), strategy.id),
        )

    with pytest.raises(
        ValueError, match="compatibility fields do not match canonical DSL"
    ):
        repo.get_strategy(strategy.id)


def test_repository_fails_closed_when_rule_output_alias_drifts_from_typed_value(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {"type": "limit", "value": 1000},
            "rules": [
                {
                    "rule_id": "positive",
                    "priority": 10,
                    "condition": {
                        "op": "compare",
                        "field": "x",
                        "operator": ">",
                        "value": 0,
                    },
                    "action": {
                        "type": "limit",
                        "value": 2000,
                        "output_value": 1900,
                    },
                }
            ],
        }
    )
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json FROM strategies WHERE id = ?",
            (strategy.id,),
        ).fetchone()
        payload = json.loads(row["dsl_json"])
        payload["rules"][0]["action"]["output_value"] = 1800
        conn.execute(
            "UPDATE strategies SET dsl_json = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False), strategy.id),
        )

    with pytest.raises(
        ValueError, match="compatibility fields do not match canonical DSL"
    ):
        repo.get_strategy(strategy.id)


def test_canonical_strategy_hash_ignores_display_description(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    before = repo.get_strategy_spec_hash(strategy.id)

    with connect(db_path) as conn:
        conn.execute(
            "UPDATE strategies SET description = 'renamed for display' WHERE id = ?",
            (strategy.id,),
        )

    assert repo.get_strategy_spec_hash(strategy.id) == before


def test_backtest_repository_round_trips_unavailable_profit_and_note(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    result = BacktestResult(
        strategy_id=strategy.id,
        approval_rate=0.5,
        approved_count=1,
        approved_bad_rate=0.0,
        rejected_bad_rate=1.0,
        expected_profit=None,
        swap_in_count=0,
        swap_out_count=0,
        swap_in_bad_rate=None,
        swap_out_bad_rate=None,
        by_segment=(),
        profit_note="缺少 pd_col，未计算预期利润",
    )

    repo.save_backtest("bt-null-profit", strategy.id, "dataset-1", result)

    assert repo.get_backtest("bt-null-profit") == result


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

    monkeypatch.setattr(strategy_repo_module, "_write_audit_row", fail_audit)

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

    monkeypatch.setattr(strategy_repo_module, "_write_audit_row", fail_audit)

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



def _adopt_audit(strategy_id: str) -> dict:
    return {
        "kind": "strategy.adopt",
        "target_ref": strategy_id,
        "outcome": "succeeded",
        "detail": {"strategy_id": strategy_id},
    }


def test_adopt_strategy_marks_adopted_with_metadata(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    result = repo.adopt_strategy_with_audit(
        strategy.id,
        reason="approved by committee",
        audit=_adopt_audit(strategy.id),
        adopted_at="2026-06-20T00:00:00Z",
    )

    assert result["version"] == 1
    assert result["retired_strategy_ids"] == []
    meta = repo.get_strategy_meta(strategy.id)
    assert meta["status"] == "adopted"
    assert meta["asset_status"] == "adopted_local"
    assert meta["adopted_at"] == "2026-06-20T00:00:00Z"
    assert meta["adoption_reason"] == "approved by committee"
    assert meta["parent_strategy_id"] is None
    assert meta["version"] == 1
    audit = db_module.PluginRepository(db_path).list_audit(kind="strategy.adopt")[0]
    assert audit["target_ref"] == strategy.id
    assert audit["detail"]["adoption_reason"] == "approved by committee"


@pytest.mark.parametrize(
    "reason",
    [
        "",
        "   ",
        "x",
        "（待采纳时确认）",
        "待确认后补充",
        "TODO later",
        "pending approval",
        "placeholder text",
    ],
)
def test_adopt_strategy_rejects_blank_or_placeholder_reason_without_mutation(
    tmp_path, reason
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    with pytest.raises(AdoptionReasonError, match="采纳理由"):
        repo.adopt_strategy_with_audit(
            strategy.id,
            reason=reason,
            audit=_adopt_audit(strategy.id),
        )

    assert repo.get_strategy_meta(strategy.id)["status"] == "draft"
    assert db_module.PluginRepository(db_path).list_audit(kind="strategy.adopt") == []


def test_adopt_strategy_normalizes_reason_before_persisting_and_auditing(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    repo.adopt_strategy_with_audit(
        strategy.id,
        reason="  委员会批准 Q3 上线  ",
        audit=_adopt_audit(strategy.id),
    )

    assert repo.get_strategy_meta(strategy.id)["adoption_reason"] == "委员会批准 Q3 上线"
    audit = db_module.PluginRepository(db_path).list_audit(kind="strategy.adopt")[0]
    assert audit["detail"]["adoption_reason"] == "委员会批准 Q3 上线"


def test_double_adopt_raises_conflict(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    repo.adopt_strategy_with_audit(
        strategy.id, reason="first", audit=_adopt_audit(strategy.id)
    )
    with pytest.raises(ConflictError):
        repo.adopt_strategy_with_audit(
            strategy.id, reason="again", audit=_adopt_audit(strategy.id)
        )


def test_adopting_new_strategy_retires_prior_adopted_sibling(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy_a = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="A",
    )
    strategy_b = build_strategy(
        "approval",
        [{"condition": "score < 650", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="B",
    )
    repo.create_strategy("task-1", strategy_a, created_at="2026-06-19T00:00:00Z")
    repo.create_strategy("task-1", strategy_b, created_at="2026-06-19T00:00:10Z")

    repo.adopt_strategy_with_audit(strategy_a.id, reason="A first", audit=_adopt_audit(strategy_a.id))
    result = repo.adopt_strategy_with_audit(
        strategy_b.id, reason="B replaces A", audit=_adopt_audit(strategy_b.id)
    )

    assert result["retired_strategy_ids"] == [strategy_a.id]
    assert repo.get_strategy_meta(strategy_a.id)["status"] == "retired"
    assert repo.get_strategy_meta(strategy_a.id)["asset_status"] == "retired"
    assert repo.get_strategy_meta(strategy_b.id)["status"] == "adopted"
    assert repo.get_strategy_meta(strategy_b.id)["asset_status"] == "adopted_local"
    retire_audit = db_module.PluginRepository(db_path).list_audit(kind="strategy.retire")[0]
    assert retire_audit["target_ref"] == strategy_a.id
    assert retire_audit["detail"]["superseded_by"] == strategy_b.id


def test_sibling_retire_conflict_rolls_back_atomically(tmp_path, monkeypatch):
    """TOCTOU: a sibling that is 'adopted' at the sibling-SELECT gets flipped out
    from under the transaction before the retire UPDATE runs. The rowcount==0
    guard must raise ConflictError AND the whole adopt transaction must roll back
    -- strategy_b stays draft (never adopted), preserving atomicity so we never
    end up with zero adopted strategies where there should be exactly one."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy_a = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="A",
    )
    strategy_b = build_strategy(
        "approval",
        [{"condition": "score < 650", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="B",
    )
    repo.create_strategy("task-1", strategy_a, created_at="2026-06-19T00:00:00Z")
    repo.create_strategy("task-1", strategy_b, created_at="2026-06-19T00:00:10Z")
    repo.adopt_strategy_with_audit(strategy_a.id, reason="A first", audit=_adopt_audit(strategy_a.id))

    real_connect = strategy_repo_module.connect

    class _RacingConn:
        # sqlite3.Connection.execute is read-only, so intercept via a proxy that
        # delegates everything except execute to the real connection.
        def __init__(self, conn):
            self._conn = conn
            self._state = {"injected": False, "reentrant": False}

        def execute(self, sql, *args, **kwargs):
            # Just before the retire UPDATE fires, flip strategy_a out of
            # 'adopted' on the same connection, simulating a concurrent
            # retirement landing in the SELECT->UPDATE window.
            if (
                sql.strip().startswith("UPDATE strategies SET status = 'retired'")
                and not self._state["injected"]
                and not self._state["reentrant"]
            ):
                self._state["injected"] = True
                self._state["reentrant"] = True
                try:
                    self._conn.execute(
                        "UPDATE strategies SET status = 'retired' WHERE id = ?",
                        (strategy_a.id,),
                    )
                finally:
                    self._state["reentrant"] = False
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextmanager
    def racing_connect(path):
        with real_connect(path) as conn:
            yield _RacingConn(conn)

    monkeypatch.setattr(strategy_repo_module, "connect", racing_connect)

    with pytest.raises(ConflictError, match="并发修改，请重试"):
        repo.adopt_strategy_with_audit(
            strategy_b.id, reason="B replaces A", audit=_adopt_audit(strategy_b.id)
        )

    monkeypatch.undo()
    # Atomicity: the main adopt UPDATE never took effect -> B is still draft.
    assert repo.get_strategy_meta(strategy_b.id)["status"] == "draft"
    # No stray retire audit row was committed for A (the whole txn rolled back).
    assert db_module.PluginRepository(db_path).list_audit(kind="strategy.retire") == []


def test_new_version_from_clones_lineage_and_bumps_version(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    child = repo.new_version_from(
        strategy.id,
        rules=[{"condition": "score < 620", "decision": "reject"}],
        description="tightened cutoff",
        created_at="2026-06-21T00:00:00Z",
    )

    child_meta = repo.get_strategy_meta(child.id)
    assert child_meta["version"] == 2
    assert child_meta["status"] == "draft"
    assert child_meta["asset_status"] == "draft"
    assert child_meta["parent_strategy_id"] == strategy.id
    assert child.description == "tightened cutoff"
    assert child.rules[0].condition == "score < 620"
    # A third version stacks on top of the max, not the parent's version.
    grandchild = repo.new_version_from(child.id, created_at="2026-06-22T00:00:00Z")
    assert repo.get_strategy_meta(grandchild.id)["version"] == 3


def test_new_version_from_accepts_canonical_nested_dsl_override(tmp_path) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    source = build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "rules": [
                {
                    "rule_id": "two-signals",
                    "priority": 10,
                    "condition": {
                        "op": "n_of_k",
                        "n": 2,
                        "args": [
                            {
                                "op": "compare",
                                "field": "late_count",
                                "operator": ">=",
                                "value": 2,
                            },
                            {"op": "is_null", "field": "income"},
                        ],
                    },
                    "action": {"type": "reject", "reason_code": "TWO_SIGNALS"},
                }
            ],
        }
    )
    repo.create_strategy("task-1", source)
    revised_spec = source.spec.to_dict()
    revised_spec["rules"][0]["condition"]["args"][0]["value"] = 3

    child = repo.new_version_from(
        source.id,
        strategy_spec=revised_spec,
        description="raise late threshold",
    )

    assert repo.get_strategy_meta(child.id)["parent_strategy_id"] == source.id
    assert child.spec.rules[0].condition["op"] == "n_of_k"
    assert child.spec.rules[0].condition["args"][0]["value"] == 3
    frame = pd.DataFrame(
        {"late_count": [2, 3, 3], "income": [None, 1000, None]}
    )
    assert apply_strategy(frame, child).tolist() == [
        "approve",
        "approve",
        "reject",
    ]


def test_new_version_from_rules_preserves_default_action_value_and_output_alias(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    source = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {
                "type": "limit",
                "value": 1000,
                "output_value": 900,
            },
            "rules": [],
        }
    )
    repo.create_strategy("task-1", source)

    child = repo.new_version_from(
        source.id,
        rules=[{"condition": "x > 0", "decision": "limit", "value": 2000}],
    )
    loaded = repo.get_strategy(child.id)

    assert loaded is not None
    assert (
        loaded.spec.default_action.to_dict()
        == source.spec.default_action.to_dict()
    )


def test_new_version_from_on_connection_rules_preserves_default_action_value_and_output_alias(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    source = build_strategy_from_spec(
        {
            "strategy_type": "limit",
            "default_action": {
                "type": "limit",
                "value": 1000,
                "output_value": 900,
            },
            "rules": [],
        }
    )
    repo.create_strategy("source-task", source)

    with repo.transaction() as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, task_type, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (
                'target-task', 'strategy', 'strategy task', 'v1', 'pytest',
                '/tmp/source', 'created', 'created',
                '2026-07-20T00:00:00Z', '2026-07-20T00:00:00Z'
            )
            """
        )
        child = repo.new_version_from_on_connection(
            conn,
            source.id,
            target_task_id="target-task",
            rules=[
                {"condition": "x > 0", "decision": "limit", "value": 2000}
            ],
        )

    loaded = repo.get_strategy(child.id)

    assert loaded is not None
    assert (
        loaded.spec.default_action.to_dict()
        == source.spec.default_action.to_dict()
    )


def test_strategy_artifacts_round_trip(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    art_id = repo.save_strategy_artifact(
        strategy.id,
        kind="decision_table_csv",
        path="workspace/tasks/task-1/strategy/decision.csv",
        created_at="2026-06-20T00:00:00Z",
    )
    artifacts = repo.list_strategy_artifacts(strategy.id)
    assert [a["id"] for a in artifacts] == [art_id]
    assert artifacts[0]["kind"] == "decision_table_csv"
    assert artifacts[0]["path"].endswith("decision.csv")


def test_task_scoped_artifact_queries_return_stored_canonical_asset_status(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    artifact_id = repo.save_strategy_artifact(
        strategy.id,
        kind="strategy_doc_md",
        path="strategy.md",
    )
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status = 'validated' WHERE id = ?",
            (strategy.id,),
        )

    listed = repo.list_strategy_artifacts_for_task("task-1")
    loaded = repo.get_strategy_artifact_for_task("task-1", artifact_id)

    assert listed[0]["status"] == "draft"
    assert listed[0]["asset_status"] == "validated"
    assert loaded is not None
    assert loaded["status"] == "draft"
    assert loaded["asset_status"] == "validated"


def test_strategy_artifacts_cascade_when_strategy_is_deleted(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")
    repo.save_strategy_artifact(
        strategy.id, kind="strategy_doc_md", path="doc.md", created_at="2026-06-20T00:00:00Z"
    )

    with connect(db_path) as conn:
        conn.execute("DELETE FROM strategies WHERE id = ?", (strategy.id,))

    assert repo.list_strategy_artifacts(strategy.id) == []


def test_existing_strategy_defaults_to_draft_version_one(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy, created_at="2026-06-19T00:00:00Z")

    meta = repo.get_strategy_meta(strategy.id)
    assert meta["version"] == 1
    assert meta["status"] == "draft"
    assert meta["asset_status"] == "draft"
    assert meta["adopted_at"] is None
    metas = repo.list_meta_for_task("task-1")
    assert [m["id"] for m in metas] == [strategy.id]
    assert [m["asset_status"] for m in metas] == ["draft"]


def test_validated_strategy_can_be_locally_adopted_with_dual_write(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status = 'validated' WHERE id = ?",
            (strategy.id,),
        )

    repo.adopt_strategy_with_audit(
        strategy.id,
        reason="validated locally",
        audit=_adopt_audit(strategy.id),
    )

    meta = repo.get_strategy_meta(strategy.id)
    assert meta["status"] == "adopted"
    assert meta["asset_status"] == "adopted_local"


def test_strategy_meta_rejects_dual_status_drift(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    strategy = _strategy()
    repo.create_strategy("task-1", strategy)
    with connect(db_path) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """
            UPDATE strategies
               SET status = 'adopted', asset_status = 'draft'
             WHERE id = ?
            """,
            (strategy.id,),
        )

    with pytest.raises(StrategyLifecycleError, match="lifecycle drift"):
        repo.get_strategy_meta(strategy.id)
