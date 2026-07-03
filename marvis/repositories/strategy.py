import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from marvis.db_schema import connect
from marvis.packs.strategy.contracts import (
    BacktestResult,
    Strategy,
    StrategyRule,
)
from marvis.state_machine import ConflictError


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

    def get_strategy_meta(self, strategy_id: str) -> dict | None:
        """Lifecycle metadata (version/status/adopted_at/parent) for a strategy.

        Kept separate from get_strategy so the frozen Strategy dataclass (and its
        equality tests) stay untouched; callers that need the S2 versioning fields
        read them here."""
        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, strategy_type, version, status, adopted_at,
                       adoption_reason, parent_strategy_id, created_at
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
        return None if row is None else _strategy_meta_from_row(row)

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

    def list_meta_for_task(self, task_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, strategy_type, version, status, adopted_at,
                       adoption_reason, parent_strategy_id, created_at
                  FROM strategies
                 WHERE task_id = ?
                 ORDER BY created_at, id
                """,
                (task_id,),
            ).fetchall()
        return [_strategy_meta_from_row(row) for row in rows]

    def adopt_strategy_with_audit(
        self,
        strategy_id: str,
        *,
        reason: str,
        audit: dict,
        adopted_at: str | None = None,
    ) -> dict:
        """Atomically move a draft strategy to adopted, retiring any sibling
        adopted strategy (same task_id + strategy_type) in the same transaction.

        The status transition is a single guarded UPDATE (... WHERE id=? AND
        status='draft'); rowcount==0 -> ConflictError, so a concurrent or repeated
        adopt of the same strategy raises instead of silently double-adopting
        (the confirm_step compare-and-swap lesson, tests/test_concurrency.py).
        Returns {"version", "retired_strategy_ids"}."""
        stamp = adopted_at or _now()
        with connect(self.db_path) as conn:
            head = conn.execute(
                "SELECT task_id, strategy_type, version FROM strategies WHERE id = ?",
                (strategy_id,),
            ).fetchone()
            if head is None:
                raise KeyError(strategy_id)
            task_id = str(head["task_id"])
            strategy_type = str(head["strategy_type"])
            version = int(head["version"])
            # Retire in-role siblings first, in the same transaction, so the
            # "at most one adopted per (task, type)" invariant holds atomically.
            retired_rows = conn.execute(
                """
                SELECT id FROM strategies
                 WHERE task_id = ? AND strategy_type = ? AND status = 'adopted'
                   AND id <> ?
                 ORDER BY created_at, id
                """,
                (task_id, strategy_type, strategy_id),
            ).fetchall()
            retired_ids = [str(r["id"]) for r in retired_rows]
            for retired_id in retired_ids:
                # rowcount guard: the sibling was 'adopted' at the SELECT above,
                # but a concurrent adopt/retire can flip it in the window between
                # that SELECT and this UPDATE. Without the guard a rowcount==0
                # here silently no-ops yet still writes a retire audit row for a
                # retirement that never happened, breaking the "at most one
                # adopted per (task, type)" invariant. On rowcount==0 we abort
                # the whole transaction (the main adopt UPDATE below is never
                # reached, connect() rolls back), so adoption stays atomic:
                # either every sibling retires and this strategy adopts, or
                # nothing changes.
                retire_cursor = conn.execute(
                    "UPDATE strategies SET status = 'retired' WHERE id = ? AND status = 'adopted'",
                    (retired_id,),
                )
                if retire_cursor.rowcount == 0:
                    raise ConflictError("并发修改，请重试")
                _write_audit_row(
                    conn,
                    kind="strategy.retire",
                    target_ref=retired_id,
                    outcome="succeeded",
                    detail={
                        "task_id": task_id,
                        "strategy_type": strategy_type,
                        "superseded_by": strategy_id,
                    },
                )
            cursor = conn.execute(
                """
                UPDATE strategies
                   SET status = 'adopted', adopted_at = ?, adoption_reason = ?
                 WHERE id = ? AND status = 'draft'
                """,
                (stamp, reason, strategy_id),
            )
            if cursor.rowcount == 0:
                current = conn.execute(
                    "SELECT status FROM strategies WHERE id = ?",
                    (strategy_id,),
                ).fetchone()
                raise ConflictError(
                    f"strategy {strategy_id} is not draft: {current['status']}"
                )
            _write_audit_row(conn, **audit)
        return {"version": version, "retired_strategy_ids": retired_ids}

    def new_version_from(
        self,
        strategy_id: str,
        *,
        rules: list | None = None,
        description: str | None = None,
        new_strategy_id: str | None = None,
        created_at: str | None = None,
    ) -> Strategy:
        """Clone a strategy into a new draft at version=max(version)+1, with
        parent_strategy_id pointing back at the source. rules/description override
        the clone when supplied."""
        stamp = created_at or _now()
        with connect(self.db_path) as conn:
            src = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
            if src is None:
                raise KeyError(strategy_id)
            task_id = str(src["task_id"])
            max_version_row = conn.execute(
                """
                SELECT MAX(version) AS mx FROM strategies
                 WHERE task_id = ? AND strategy_type = ?
                """,
                (task_id, str(src["strategy_type"])),
            ).fetchone()
            next_version = int(max_version_row["mx"] or 0) + 1
            child_id = new_strategy_id or uuid.uuid4().hex
            child_rules = (
                _dump_json_any(
                    [_strategy_rule_to_dict(_coerce_rule(rule)) for rule in rules]
                )
                if rules is not None
                else str(src["rules_json"])
            )
            child_description = (
                str(description) if description is not None else str(src["description"])
            )
            conn.execute(
                """
                INSERT INTO strategies(
                    id, task_id, strategy_type, rules_json, score_col,
                    default_decision_json, description, created_at,
                    version, status, parent_strategy_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?)
                """,
                (
                    child_id,
                    task_id,
                    str(src["strategy_type"]),
                    child_rules,
                    _optional_str(src["score_col"]),
                    str(src["default_decision_json"]),
                    child_description,
                    stamp,
                    next_version,
                    strategy_id,
                ),
            )
            row = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at
                  FROM strategies
                 WHERE id = ?
                """,
                (child_id,),
            ).fetchone()
        return _strategy_from_row(row)

    def save_strategy_artifact(
        self,
        strategy_id: str,
        *,
        kind: str,
        path: str,
        created_at: str | None = None,
        artifact_id: str | None = None,
    ) -> str:
        new_id = artifact_id or uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO strategy_artifacts(id, strategy_id, kind, path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (new_id, strategy_id, str(kind), str(path), created_at or _now()),
            )
        return new_id

    def list_strategy_artifacts(self, strategy_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, strategy_id, kind, path, created_at
                  FROM strategy_artifacts
                 WHERE strategy_id = ?
                 ORDER BY created_at, id
                """,
                (strategy_id,),
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "strategy_id": str(row["strategy_id"]),
                "kind": str(row["kind"]),
                "path": str(row["path"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

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

    def list_monitoring_due(self, now: datetime | None = None) -> list[dict]:
        """S5: adopted strategies whose next monitoring run is due (overdue).

        Due date = (last_run_at or adopted_at) + cadence_days, read from each
        adopted strategy's latest monitoring_plan_json artifact. A strategy with
        no monitoring plan is skipped (nothing to be due against). All the SQL and
        the plan-JSON parsing lives here so callers get plain dicts. Returns only
        strategies that are currently overdue, most-overdue first."""
        reference = now or datetime.now(UTC)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.id AS strategy_id, s.adopted_at AS adopted_at,
                       (SELECT a.path FROM strategy_artifacts a
                         WHERE a.strategy_id = s.id AND a.kind = 'monitoring_plan_json'
                         ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS plan_path
                  FROM strategies s
                 WHERE s.status = 'adopted'
                 ORDER BY s.adopted_at, s.id
                """
            ).fetchall()
        due: list[dict] = []
        for row in rows:
            plan_path = _optional_str(row["plan_path"])
            if plan_path is None:
                continue
            plan = _read_monitoring_plan_fields(plan_path)
            if plan is None:
                continue
            anchor_ts = _parse_iso(plan.get("last_run_at")) or _parse_iso(row["adopted_at"])
            if anchor_ts is None:
                continue
            cadence_days = int(plan.get("cadence_days") or 30)
            due_at = anchor_ts + timedelta(days=cadence_days)
            overdue_seconds = (reference - due_at).total_seconds()
            if overdue_seconds <= 0:
                continue
            due.append(
                {
                    "strategy_id": str(row["strategy_id"]),
                    "due_at": due_at.isoformat(),
                    "overdue_days": overdue_seconds / 86400.0,
                    "last_run_at": _optional_str(plan.get("last_run_at")),
                    "cadence_days": cadence_days,
                }
            )
        due.sort(key=lambda item: (-item["overdue_days"], item["strategy_id"]))
        return due


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


def _strategy_meta_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "strategy_type": str(row["strategy_type"]),
        "version": int(row["version"]),
        "status": str(row["status"]),
        "adopted_at": _optional_str(row["adopted_at"]),
        "adoption_reason": _optional_str(row["adoption_reason"]),
        "parent_strategy_id": _optional_str(row["parent_strategy_id"]),
        "created_at": str(row["created_at"]),
    }


def _coerce_rule(rule) -> StrategyRule:
    if isinstance(rule, StrategyRule):
        return rule
    return _strategy_rule_from_dict(dict(rule))


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
        swap_in_bad_rate=_optional_float_field(payload["swap_in_bad_rate"]),
        swap_out_bad_rate=_optional_float_field(payload["swap_out_bad_rate"]),
        by_segment=tuple(dict(item) for item in payload.get("by_segment") or ()),
    )


def _optional_str(value) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized or None


def _optional_float_field(value) -> float | None:
    return None if value is None else float(value)


def _parse_iso(value) -> datetime | None:
    text = _optional_str(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _read_monitoring_plan_fields(plan_path: str) -> dict | None:
    """Read cadence_days/last_run_at from a monitoring plan file. Returns None if
    the file is missing or unparseable -- a broken plan file must not make a due
    sweep raise (it just drops that strategy from the due list)."""
    try:
        raw = Path(plan_path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


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
