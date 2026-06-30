import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from marvis.db_schema import connect
from marvis.packs.strategy.contracts import (
    BacktestResult,
    Strategy,
    StrategyRule,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StrategyRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def create_strategy(
        self,
        task_id: str,
        strategy: Strategy,
        *,
        created_at: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_strategy_row(conn, task_id, strategy, created_at or _now())

    def create_strategy_with_audit(
        self,
        task_id: str,
        strategy: Strategy,
        *,
        audit: dict,
        created_at: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_strategy_row(conn, task_id, strategy, created_at or _now())
            _write_audit_row(conn, **audit)

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
        return None if row is None else _strategy_from_row(row)

    def list_for_task(self, task_id: str) -> list[Strategy]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at
                  FROM strategies
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_strategy_from_row(row) for row in rows]

    def save_backtest(
        self,
        backtest_id: str,
        strategy_id: str,
        dataset_id: str,
        result: BacktestResult,
        *,
        created_at: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_backtest_row(
                conn,
                backtest_id,
                strategy_id,
                dataset_id,
                result,
                created_at or _now(),
            )

    def save_backtest_with_audit(
        self,
        backtest_id: str,
        strategy_id: str,
        dataset_id: str,
        result: BacktestResult,
        *,
        audit: dict,
        created_at: str | None = None,
    ) -> None:
        with connect(self.db_path) as conn:
            _insert_backtest_row(
                conn,
                backtest_id,
                strategy_id,
                dataset_id,
                result,
                created_at or _now(),
            )
            _write_audit_row(conn, **audit)

    def get_backtest(self, backtest_id: str) -> BacktestResult | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, strategy_id, dataset_id, result_json, created_at
                  FROM backtests
                 WHERE id = ?
                """,
                (backtest_id,),
            ).fetchone()
        return None if row is None else _backtest_result_from_row(row)

    def list_backtests(self, strategy_id: str) -> list[BacktestResult]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, strategy_id, dataset_id, result_json, created_at
                  FROM backtests
                 WHERE strategy_id = ?
                 ORDER BY created_at, id
                """,
                (strategy_id,),
            ).fetchall()
        return [_backtest_result_from_row(row) for row in rows]


def _write_audit_row(
    conn: sqlite3.Connection,
    *,
    kind: str,
    target_ref: str,
    actor: str = "system",
    inputs_hash: str | None = None,
    outcome: str | None = None,
    detail: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO audit(
            id, kind, actor, target_ref, inputs_hash, outcome,
            detail_json, at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid.uuid4().hex,
            kind,
            actor,
            target_ref,
            inputs_hash,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
            _now(),
        ),
    )


def _strategy_insert_values(task_id: str, strategy: Strategy, created_at: str) -> tuple:
    return (
        strategy.id,
        task_id,
        strategy.strategy_type,
        _dump_json_any([_strategy_rule_to_dict(rule) for rule in strategy.rules]),
        strategy.score_col,
        _dump_json_any(strategy.default_decision),
        strategy.description,
        created_at,
    )


def _insert_strategy_row(
    conn: sqlite3.Connection,
    task_id: str,
    strategy: Strategy,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO strategies(
            id, task_id, strategy_type, rules_json, score_col,
            default_decision_json, description, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _strategy_insert_values(task_id, strategy, created_at),
    )


def _strategy_from_row(row: sqlite3.Row) -> Strategy:
    return Strategy(
        id=str(row["id"]),
        strategy_type=str(row["strategy_type"]),
        rules=tuple(
            _strategy_rule_from_dict(item)
            for item in _load_json_array(row["rules_json"])
        ),
        score_col=_optional_str(row["score_col"]),
        default_decision=json.loads(row["default_decision_json"]),
        description=str(row["description"]),
    )


def _strategy_rule_to_dict(rule: StrategyRule) -> dict:
    return asdict(rule)


def _strategy_rule_from_dict(payload: dict) -> StrategyRule:
    return StrategyRule(
        condition=str(payload["condition"]),
        decision=str(payload["decision"]),
        value=payload.get("value"),
    )


def _backtest_insert_values(
    backtest_id: str,
    strategy_id: str,
    dataset_id: str,
    result: BacktestResult,
    created_at: str,
) -> tuple:
    return (
        backtest_id,
        strategy_id,
        dataset_id,
        _dump_json_any(_backtest_result_to_dict(result)),
        created_at,
    )


def _insert_backtest_row(
    conn: sqlite3.Connection,
    backtest_id: str,
    strategy_id: str,
    dataset_id: str,
    result: BacktestResult,
    created_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO backtests(
            id, strategy_id, dataset_id, result_json, created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        _backtest_insert_values(
            backtest_id,
            strategy_id,
            dataset_id,
            result,
            created_at,
        ),
    )


def _backtest_result_to_dict(result: BacktestResult) -> dict:
    payload = asdict(result)
    payload["by_segment"] = list(result.by_segment)
    return payload


def _backtest_result_from_row(row: sqlite3.Row) -> BacktestResult:
    return _backtest_result_from_dict(_load_json_object(row["result_json"]))


def _backtest_result_from_dict(payload: dict) -> BacktestResult:
    return BacktestResult(
        strategy_id=str(payload["strategy_id"]),
        approval_rate=float(payload["approval_rate"]),
        approved_count=int(payload["approved_count"]),
        approved_bad_rate=float(payload["approved_bad_rate"]),
        rejected_bad_rate=float(payload["rejected_bad_rate"]),
        expected_profit=float(payload["expected_profit"]),
        swap_in_count=int(payload["swap_in_count"]),
        swap_out_count=int(payload["swap_out_count"]),
        swap_in_bad_rate=float(payload["swap_in_bad_rate"]),
        swap_out_bad_rate=float(payload["swap_out_bad_rate"]),
        by_segment=tuple(dict(item) for item in payload.get("by_segment") or ()),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized or None


def _dump_json_any(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _load_json_array(raw: str | None) -> list:
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _load_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
