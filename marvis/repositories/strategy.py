import hashlib
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
from marvis.packs.strategy.dsl import (
    StrategySpec,
    canonical_strategy_json,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
from marvis.state_machine import ConflictError
from marvis.strategy_adoption import normalize_adoption_reason


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _strategy_effect_conflict(reason: str) -> ConflictError:
    return ConflictError(f"策略效果授权无效：{reason}")


def _strategy_spec_hash_from_row(row: sqlite3.Row) -> str:
    """Hash the reviewed strategy definition, excluding lifecycle metadata."""

    if "dsl_json" in row.keys() and row["dsl_json"] is not None:
        spec = parse_strategy_spec(json.loads(str(row["dsl_json"])))
        stored_version = _optional_str(row["dsl_schema_version"])
        if stored_version != spec.schema_version:
            raise ValueError(
                "strategy dsl_schema_version does not match canonical dsl_json"
            )
        return strategy_spec_hash(spec)

    payload = {
        "strategy_type": str(row["strategy_type"]),
        "rules": json.loads(str(row["rules_json"])),
        "score_col": _optional_str(row["score_col"]),
        "default_decision": json.loads(str(row["default_decision_json"])),
        "description": str(row["description"]),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_strategy_effect_authorization(
    conn: sqlite3.Connection,
    *,
    effect_execution_id: str,
    runtime_generation: str,
    strategy_id: str,
    task_id: str,
    strategy_type: str,
    version: int,
    status: str,
    strategy_spec_hash: str,
) -> dict:
    """Validate the one-shot effect receipt and its frozen strategy target.

    The worker receives only an opaque execution id and a host generation.  It
    never receives an approval id, reservation id, or target binding as tool
    inputs.  Those values are loaded from the local governance ledger and
    cross-checked against current domain state while the caller holds the same
    SQLite writer transaction used for adoption.
    """
    row = conn.execute(
        """
        SELECT e.id AS effect_execution_id,
               e.approval_id,
               e.reservation_id AS effect_reservation_id,
               e.runtime_generation,
               e.status AS effect_status,
               e.released_at,
               e.detail_json,
               a.status AS approval_status,
               a.reservation_id AS approval_reservation_id,
               a.task_id AS approval_task_id,
               a.tool_ref,
               a.effect_target_json
          FROM effect_executions e
          JOIN approval_records a ON a.id = e.approval_id
         WHERE e.id = ?
        """,
        (effect_execution_id,),
    ).fetchone()
    if row is None:
        raise _strategy_effect_conflict("效果执行记录不存在")
    if str(row["effect_status"]) != "dispatched" or row["released_at"] is not None:
        raise _strategy_effect_conflict("效果执行记录不是可提交的 dispatched 状态")
    if str(row["approval_status"]) != "reserved":
        raise _strategy_effect_conflict("批准记录不是 reserved 状态")
    effect_reservation_id = str(row["effect_reservation_id"] or "")
    if not effect_reservation_id or effect_reservation_id != str(
        row["approval_reservation_id"] or ""
    ):
        raise _strategy_effect_conflict("reservation 绑定不一致")
    if str(row["runtime_generation"]) != runtime_generation:
        raise _strategy_effect_conflict("runtime generation 已被隔离")
    if str(row["approval_task_id"]) != task_id:
        raise _strategy_effect_conflict("批准任务与策略任务不一致")
    tool_ref = str(row["tool_ref"])
    if not tool_ref.startswith("strategy.adopt_strategy@"):
        raise _strategy_effect_conflict("批准工具不是 strategy.adopt_strategy")

    try:
        target = json.loads(str(row["effect_target_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _strategy_effect_conflict("effect target 不是有效 JSON") from exc
    if not isinstance(target, dict):
        raise _strategy_effect_conflict("effect target 必须是对象")

    expected_status = target.get("expected_status")
    if expected_status is None:
        expected_statuses = target.get("expected_statuses")
        if isinstance(expected_statuses, list) and len(expected_statuses) == 1:
            expected_status = expected_statuses[0]
    expected = {
        "kind": "strategy",
        "id": strategy_id,
        "expected_status": "draft",
        "result_status": "adopted",
        "version": version,
        "task_id": task_id,
        "strategy_type": strategy_type,
        "strategy_spec_hash": strategy_spec_hash,
    }
    actual = {
        "kind": target.get("kind"),
        "id": target.get("id"),
        "expected_status": expected_status,
        "result_status": target.get("result_status"),
        "version": target.get("version"),
        "task_id": target.get("task_id"),
        "strategy_type": target.get("strategy_type"),
        "strategy_spec_hash": target.get("strategy_spec_hash"),
    }
    if actual != expected or status != "draft":
        raise _strategy_effect_conflict("目标 id/status/version/task/type/spec 已漂移")

    champion_rows = conn.execute(
        """
        SELECT id
          FROM strategies
         WHERE task_id = ? AND strategy_type = ? AND status = 'adopted'
           AND id <> ?
         ORDER BY id
        """,
        (task_id, strategy_type, strategy_id),
    ).fetchall()
    current_champion_ids = sorted(str(item["id"]) for item in champion_rows)
    bound_champions = target.get("current_champion_ids")
    if bound_champions is None and "current_champion_id" in target:
        singular = target.get("current_champion_id")
        bound_champions = [] if singular in (None, "") else [singular]
    if not isinstance(bound_champions, list):
        raise _strategy_effect_conflict("授权未冻结 current champion 列表")
    normalized_bound_champions = sorted(str(item) for item in bound_champions)
    if normalized_bound_champions != current_champion_ids:
        raise _strategy_effect_conflict("current champion 已漂移")

    return {
        "effect_execution_id": effect_execution_id,
        "approval_id": str(row["approval_id"]),
        "reservation_id": effect_reservation_id,
        "runtime_generation": runtime_generation,
        "detail": _json_object_or_empty(row["detail_json"]),
    }


def _commit_strategy_effect_receipt(
    conn: sqlite3.Connection,
    *,
    receipt: dict,
    strategy_id: str,
    version: int,
    retired_strategy_ids: list[str],
) -> None:
    committed_at = _now()
    domain_receipt = {
        "kind": "strategy.adopt",
        "strategy_id": strategy_id,
        "version": version,
        "status": "adopted",
        "retired_strategy_ids": list(retired_strategy_ids),
    }
    result_hash = hashlib.sha256(
        json.dumps(
            domain_receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    detail = dict(receipt["detail"])
    detail["domain_receipt"] = domain_receipt
    effect_cursor = conn.execute(
        """
        UPDATE effect_executions
           SET status = 'committed', committed_at = ?, result_hash = ?,
               detail_json = ?
         WHERE id = ? AND status = 'dispatched' AND released_at IS NULL
           AND reservation_id = ? AND runtime_generation = ?
        """,
        (
            committed_at,
            result_hash,
            json.dumps(detail, ensure_ascii=False, sort_keys=True),
            receipt["effect_execution_id"],
            receipt["reservation_id"],
            receipt["runtime_generation"],
        ),
    )
    if effect_cursor.rowcount != 1:
        raise _strategy_effect_conflict("效果执行提交发生并发冲突")
    approval_cursor = conn.execute(
        """
        UPDATE approval_records
           SET status = 'consumed', consumed_at = ?
         WHERE id = ? AND status = 'reserved' AND reservation_id = ?
        """,
        (committed_at, receipt["approval_id"], receipt["reservation_id"]),
    )
    if approval_cursor.rowcount != 1:
        raise _strategy_effect_conflict("批准消费发生并发冲突")


def _json_object_or_empty(value) -> dict:
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


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
                       default_decision_json, description, created_at,
                       dsl_json, dsl_schema_version
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

    def get_strategy_spec_hash(self, strategy_id: str) -> str | None:
        """Return a canonical hash of the strategy definition under review."""

        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT strategy_type, rules_json, score_col,
                       default_decision_json, description,
                       dsl_json, dsl_schema_version
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
        return None if row is None else _strategy_spec_hash_from_row(row)

    def list_for_task(self, task_id: str) -> list[Strategy]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at,
                       dsl_json, dsl_schema_version
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
        effect_execution_id: str | None = None,
        runtime_generation: str | None = None,
    ) -> dict:
        """Atomically move a draft strategy to adopted, retiring any sibling
        adopted strategy (same task_id + strategy_type) in the same transaction.

        The status transition is a single guarded UPDATE (... WHERE id=? AND
        status='draft'); rowcount==0 -> ConflictError, so a concurrent or repeated
        adopt of the same strategy raises instead of silently double-adopting
        (the confirm_step compare-and-swap lesson, tests/test_concurrency.py).
        Returns {"version", "retired_strategy_ids"}."""
        governed_effect = effect_execution_id is not None or runtime_generation is not None
        if governed_effect and (
            not str(effect_execution_id or "").strip()
            or not str(runtime_generation or "").strip()
        ):
            raise ConflictError("策略效果授权无效：治理执行元数据不完整")

        normalized_reason = normalize_adoption_reason(reason)
        adoption_audit = dict(audit)
        adoption_audit["detail"] = {
            **dict(audit.get("detail") or {}),
            "strategy_id": strategy_id,
            "adoption_reason": normalized_reason,
        }
        stamp = adopted_at or _now()
        with connect(self.db_path) as conn:
            # A governed side effect must validate its immutable authorization
            # snapshot and mutate the domain under the same writer lock.  Without
            # BEGIN IMMEDIATE a different adoption could land after the champion
            # snapshot check but before our guarded writes, defeating the fence.
            if governed_effect:
                conn.execute("BEGIN IMMEDIATE")
            head = conn.execute(
                """
                SELECT task_id, strategy_type, version, status, rules_json,
                       score_col, default_decision_json, description,
                       dsl_json, dsl_schema_version
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
            if head is None:
                raise KeyError(strategy_id)
            task_id = str(head["task_id"])
            strategy_type = str(head["strategy_type"])
            version = int(head["version"])
            effect_receipt = None
            if governed_effect:
                effect_receipt = _validate_strategy_effect_authorization(
                    conn,
                    effect_execution_id=str(effect_execution_id),
                    runtime_generation=str(runtime_generation),
                    strategy_id=strategy_id,
                    task_id=task_id,
                    strategy_type=strategy_type,
                    version=version,
                    status=str(head["status"]),
                    strategy_spec_hash=_strategy_spec_hash_from_row(head),
                )
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
                (stamp, normalized_reason, strategy_id),
            )
            if cursor.rowcount == 0:
                current = conn.execute(
                    "SELECT status FROM strategies WHERE id = ?",
                    (strategy_id,),
                ).fetchone()
                raise ConflictError(
                    f"strategy {strategy_id} is not draft: {current['status']}"
                )
            _write_audit_row(conn, **adoption_audit)
            if effect_receipt is not None:
                _commit_strategy_effect_receipt(
                    conn,
                    receipt=effect_receipt,
                    strategy_id=strategy_id,
                    version=version,
                    retired_strategy_ids=retired_ids,
                )
        return {"version": version, "retired_strategy_ids": retired_ids}

    def new_version_from(
        self,
        strategy_id: str,
        *,
        rules: list | None = None,
        strategy_spec: StrategySpec | dict | None = None,
        description: str | None = None,
        new_strategy_id: str | None = None,
        created_at: str | None = None,
    ) -> Strategy:
        """Clone a strategy into a new draft at version=max(version)+1, with
        parent_strategy_id pointing back at the source. A canonical strategy_spec
        or legacy rules (mutually exclusive) may override the definition."""
        if rules is not None and strategy_spec is not None:
            raise ValueError("rules and strategy_spec are mutually exclusive")
        stamp = created_at or _now()
        with connect(self.db_path) as conn:
            src = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at,
                       dsl_json, dsl_schema_version
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
            child_description = (
                str(description) if description is not None else str(src["description"])
            )
            source_strategy = _strategy_from_row(src)
            if strategy_spec is not None:
                from marvis.packs.strategy.strategy import build_strategy_from_spec

                base_spec = parse_strategy_spec(strategy_spec)
                if base_spec.strategy_type != source_strategy.strategy_type:
                    raise ValueError(
                        "new strategy version cannot change strategy_type"
                    )
                built = build_strategy_from_spec(
                    base_spec,
                    score_col=source_strategy.score_col,
                    description=child_description,
                )
                child_rules = built.rules
            elif rules is not None:
                # Local import avoids making repository module import order part of
                # the pack's public surface while still using the one canonical
                # legacy->DSL normalization path.
                from marvis.packs.strategy.strategy import build_strategy

                built = build_strategy(
                    source_strategy.strategy_type,
                    [
                        _strategy_rule_to_dict(_coerce_rule(rule))
                        for rule in rules
                    ],
                    score_col=source_strategy.score_col,
                    default_decision=source_strategy.default_decision,
                    description=child_description,
                )
                child_rules = built.rules
                base_spec = built.spec
            else:
                base_spec = source_strategy.spec
                child_rules = _rules_with_spec_identity(
                    source_strategy.rules,
                    base_spec,
                )
            spec_payload = base_spec.to_dict()
            metadata = dict(spec_payload.get("metadata") or {})
            lineage = dict(metadata.get("lineage") or {})
            lineage.update(
                {
                    "source": "strategy_version",
                    "parent_strategy_id": strategy_id,
                }
            )
            metadata["lineage"] = lineage
            metadata["description"] = child_description
            spec_payload["metadata"] = metadata
            child_spec = parse_strategy_spec(spec_payload)
            conn.execute(
                """
                INSERT INTO strategies(
                    id, task_id, strategy_type, rules_json, score_col,
                    default_decision_json, description, created_at,
                    version, status, parent_strategy_id,
                    dsl_json, dsl_schema_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?)
                """,
                (
                    child_id,
                    task_id,
                    child_spec.strategy_type,
                    _dump_json_any(
                        [_strategy_rule_to_dict(rule) for rule in child_rules]
                    ),
                    _optional_str(src["score_col"]),
                    _dump_json_any(child_spec.default_action.decision_value),
                    child_description,
                    stamp,
                    next_version,
                    strategy_id,
                    canonical_strategy_json(child_spec),
                    child_spec.schema_version,
                ),
            )
            row = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at,
                       dsl_json, dsl_schema_version
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
    spec = strategy.spec or legacy_strategy_to_spec(strategy)
    _assert_strategy_matches_spec(strategy, spec)
    return (
        strategy.id,
        task_id,
        strategy.strategy_type,
        _dump_json_any([_strategy_rule_to_dict(rule) for rule in strategy.rules]),
        strategy.score_col,
        _dump_json_any(strategy.default_decision),
        strategy.description,
        created_at,
        canonical_strategy_json(spec),
        spec.schema_version,
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
            default_decision_json, description, created_at,
            dsl_json, dsl_schema_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _strategy_insert_values(task_id, strategy, created_at),
    )


def _strategy_from_row(row: sqlite3.Row) -> Strategy:
    strategy = Strategy(
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
    if "dsl_json" in row.keys() and row["dsl_json"] is not None:
        spec = parse_strategy_spec(json.loads(str(row["dsl_json"])))
        stored_version = _optional_str(row["dsl_schema_version"])
        if stored_version != spec.schema_version:
            raise ValueError(
                "strategy dsl_schema_version does not match canonical dsl_json"
            )
        _assert_strategy_matches_spec(strategy, spec)
        return Strategy(
            id=strategy.id,
            strategy_type=strategy.strategy_type,
            rules=strategy.rules,
            score_col=strategy.score_col,
            default_decision=strategy.default_decision,
            description=strategy.description,
            spec=spec,
        )
    else:
        spec = legacy_strategy_to_spec(strategy)
    normalized_rules = _rules_with_spec_identity(strategy.rules, spec)
    return Strategy(
        id=strategy.id,
        strategy_type=strategy.strategy_type,
        rules=normalized_rules,
        score_col=strategy.score_col,
        default_decision=strategy.default_decision,
        description=strategy.description,
        spec=spec,
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


def _rules_with_spec_identity(
    rules: tuple[StrategyRule, ...],
    spec: StrategySpec,
) -> tuple[StrategyRule, ...]:
    if len(rules) != len(spec.rules):
        raise ValueError("strategy compatibility rules do not match canonical DSL")
    legacy_by_priority = {
        (
            rule.priority
            if rule.priority is not None
            else (ordinal + 1) * 10
        ): rule
        for ordinal, rule in enumerate(rules)
    }
    return tuple(
        StrategyRule(
            condition=legacy.condition,
            decision=legacy.decision,
            value=legacy.value,
            rule_id=typed.rule_id,
            priority=typed.priority,
            reason_code=typed.action.reason_code,
        )
        for typed in spec.rules
        for legacy in (legacy_by_priority[typed.priority],)
    )


def _assert_strategy_matches_spec(strategy: Strategy, spec: StrategySpec) -> None:
    compatibility_spec = legacy_strategy_to_spec(strategy)
    if strategy_spec_hash(compatibility_spec) != strategy_spec_hash(spec):
        raise ValueError("strategy compatibility fields do not match canonical DSL")


def _strategy_rule_from_dict(payload: dict) -> StrategyRule:
    return StrategyRule(
        condition=str(payload["condition"]),
        decision=str(payload["decision"]),
        value=payload.get("value"),
        rule_id=_optional_str(payload.get("rule_id")),
        priority=(
            int(payload["priority"])
            if payload.get("priority") is not None
            else None
        ),
        reason_code=_optional_str(payload.get("reason_code")),
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
        expected_profit=_optional_float_field(payload["expected_profit"]),
        swap_in_count=int(payload["swap_in_count"]),
        swap_out_count=int(payload["swap_out_count"]),
        swap_in_bad_rate=_optional_float_field(payload["swap_in_bad_rate"]),
        swap_out_bad_rate=_optional_float_field(payload["swap_out_bad_rate"]),
        by_segment=tuple(dict(item) for item in payload.get("by_segment") or ()),
        profit_note=_optional_str(payload.get("profit_note")),
        rejected_count=int(payload.get("rejected_count", 0)),
        review_count=int(payload.get("review_count", 0)),
        review_rate=float(payload.get("review_rate", 0.0)),
        review_bad_rate=_optional_float_field(payload.get("review_bad_rate")),
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
