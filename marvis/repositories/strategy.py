import hashlib
import hmac
import json
import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeAlias

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
from marvis.packs.strategy.monitoring_plan import monitoring_plan_from_dict
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.typed_backtest import (
    STRATEGY_BACKTEST_SCHEMA_VERSION,
    StrategyBacktestResult,
)
from marvis.state_machine import ConflictError
from marvis.strategy_adoption import normalize_adoption_reason
from marvis.strategy_lifecycle import (
    ASSET_STATUS_ADOPTED_LOCAL,
    ASSET_STATUS_DRAFT,
    ASSET_STATUS_RETIRED,
    ASSET_STATUS_VALIDATED,
    LEGACY_STATUS_ADOPTED,
    LEGACY_STATUS_DRAFT,
    LEGACY_STATUS_RETIRED,
    StrategyLifecycleError,
    asset_status_from_legacy,
    is_locally_adopted,
    resolve_asset_status,
)


BacktestRecord: TypeAlias = BacktestResult | StrategyBacktestResult
_STRATEGY_ARTIFACT_IDENTITY_NAMESPACE = "marvis.strategy_artifact.v1"
POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION = "strategy.pool-materialization.v1"
POOL_MATERIALIZATION_PRODUCER_VERSION = "marvis.strategy.pool-materialization/1"
POOL_MATERIALIZATION_AUDIT_KIND = "strategy.materialize_pool"

_POOL_MATERIALIZATION_INPUT_FIELDS = frozenset(
    {
        "id",
        "task_id",
        "strategy_type",
        "strategy_id",
        "pool_id",
        "pool_revision_id",
        "pool_revision",
        "pool_snapshot_hash",
        "pool_artifact_id",
        "pool_artifact_content_hash",
        "selected_design_hash",
        "requirements",
        "requirements_hash",
        "strategy_spec_hash",
        "strategy_dsl_content_hash",
        "audit_id",
    }
)
_POOL_MATERIALIZATION_HASH_FIELDS = (
    "pool_snapshot_hash",
    "pool_artifact_content_hash",
    "selected_design_hash",
    "requirements_hash",
    "strategy_spec_hash",
    "strategy_dsl_content_hash",
)
_POOL_MATERIALIZATION_STRATEGY_TYPES = frozenset(
    {"approval", "reject", "limit", "pricing", "segmentation"}
)


class StrategyArtifactConflictError(RuntimeError):
    """A verified content identity already exists with different evidence."""


class StrategyArtifactDataError(ValueError):
    """Verified strategy artifact metadata violates its immutable contract."""


class StrategyPoolMaterializationError(StrategyError):
    """Pool materialization lineage or its persisted Strategy drifted."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _strategy_effect_conflict(reason: str) -> ConflictError:
    return ConflictError(f"策略效果授权无效：{reason}")


def _current_local_champion_ids(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    strategy_type: str,
    exclude_strategy_id: str,
) -> list[str]:
    """Return canonical local champions while rejecting any stored drift."""

    rows = conn.execute(
        """
        SELECT id, status, asset_status
          FROM strategies
         WHERE task_id = ? AND strategy_type = ? AND id <> ?
         ORDER BY id
        """,
        (task_id, strategy_type, exclude_strategy_id),
    ).fetchall()
    champions: list[str] = []
    for row in rows:
        try:
            adopted = is_locally_adopted(row["status"], row["asset_status"])
        except StrategyLifecycleError as exc:
            raise _strategy_effect_conflict(
                f"策略 {row['id']} 的生命周期状态漂移"
            ) from exc
        if adopted:
            champions.append(str(row["id"]))
    return champions


def _strategy_spec_hash_from_row(row: sqlite3.Row) -> str:
    """Hash the reviewed strategy definition, excluding lifecycle metadata."""

    dsl_state = _strategy_dsl_state(row)
    if dsl_state == "canonical":
        spec = _strategy_spec_from_row(row)
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
    asset_status: str,
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
    try:
        current_asset_status = resolve_asset_status(status, asset_status)
    except StrategyLifecycleError as exc:
        raise _strategy_effect_conflict("当前策略生命周期状态漂移") from exc

    has_canonical_target = (
        "expected_asset_status" in target or "result_asset_status" in target
    )
    if has_canonical_target:
        expected_asset_status = target.get("expected_asset_status")
        result_asset_status = target.get("result_asset_status")
    else:
        # Compatibility for ApprovalRecords issued before canonical lifecycle
        # fields existed.  Only exact known legacy tokens are mapped.
        try:
            expected_asset_status = asset_status_from_legacy(expected_status)
            result_asset_status = asset_status_from_legacy(
                target.get("result_status")
            )
        except StrategyLifecycleError as exc:
            raise _strategy_effect_conflict(
                "旧 effect target 的 lifecycle 状态无法映射"
            ) from exc

    expected = {
        "kind": "strategy",
        "id": strategy_id,
        "expected_status": LEGACY_STATUS_DRAFT,
        "result_status": LEGACY_STATUS_ADOPTED,
        "expected_asset_status": current_asset_status,
        "result_asset_status": ASSET_STATUS_ADOPTED_LOCAL,
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
        "expected_asset_status": expected_asset_status,
        "result_asset_status": result_asset_status,
        "version": target.get("version"),
        "task_id": target.get("task_id"),
        "strategy_type": target.get("strategy_type"),
        "strategy_spec_hash": target.get("strategy_spec_hash"),
    }
    if actual != expected or status != LEGACY_STATUS_DRAFT:
        raise _strategy_effect_conflict(
            "目标 id/status/asset_status/version/task/type/spec 已漂移"
        )

    current_champion_ids = _current_local_champion_ids(
        conn,
        task_id=task_id,
        strategy_type=strategy_type,
        exclude_strategy_id=strategy_id,
    )
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
        "status": LEGACY_STATUS_ADOPTED,
        "asset_status": ASSET_STATUS_ADOPTED_LOCAL,
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

    def transaction(self):
        return connect(self.db_path)

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
                       dsl_json, dsl_schema_version, dsl_content_hash
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
                SELECT id, task_id, strategy_type, version, status, asset_status,
                       adopted_at, adoption_reason, parent_strategy_id, created_at
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
                       dsl_json, dsl_schema_version, dsl_content_hash
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
        return None if row is None else _strategy_spec_hash_from_row(row)

    def get_strategy_snapshot(self, strategy_id: str) -> dict[str, Any] | None:
        """Read definition, lifecycle metadata, and hash from one database row.

        Consumers that bind a plan to a strategy must not combine three
        independently read snapshots: a concurrent replacement or reassignment
        could otherwise mix metadata from one state with the definition from
        another.
        """

        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, task_id, strategy_type, version, status, asset_status,
                       adopted_at, adoption_reason, parent_strategy_id, created_at,
                       rules_json, score_col, default_decision_json, description,
                       dsl_json, dsl_schema_version, dsl_content_hash
                  FROM strategies
                 WHERE id = ?
                """,
                (strategy_id,),
            ).fetchone()
            if row is None:
                return None
            strategy = _strategy_from_row(row)
            metadata = _strategy_meta_from_row(row)
            spec_hash = _strategy_spec_hash_from_row(row)
        return {
            "strategy": strategy,
            "metadata": metadata,
            "strategy_spec_hash": spec_hash,
        }

    def get_pool_materialization(
        self,
        materialization_id: str,
    ) -> dict[str, Any] | None:
        with connect(self.db_path) as conn:
            return self.get_pool_materialization_on_connection(
                conn,
                materialization_id,
            )

    def get_pool_materialization_on_connection(
        self,
        conn: sqlite3.Connection,
        materialization_id: str,
    ) -> dict[str, Any] | None:
        normalized_id = _pool_materialization_text(
            materialization_id,
            "materialization_id",
        )
        row = conn.execute(
            """
            SELECT *
              FROM strategy_pool_materializations
             WHERE id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        persisted = _pool_materialization_from_row(row)
        authenticated = _require_pool_materialization_on_connection(
            conn,
            expected={
                field: persisted[field]
                for field in _POOL_MATERIALIZATION_INPUT_FIELDS
            },
        )
        return authenticated["materialization"]

    def get_pool_materialization_for_pool_revision_on_connection(
        self,
        conn: sqlite3.Connection,
        pool_revision_id: str,
    ) -> dict[str, Any] | None:
        normalized_id = _pool_materialization_text(
            pool_revision_id,
            "pool_revision_id",
        )
        row = conn.execute(
            """
            SELECT *
              FROM strategy_pool_materializations
             WHERE pool_revision_id = ?
            """,
            (normalized_id,),
        ).fetchone()
        if row is None:
            return None
        persisted = _pool_materialization_from_row(row)
        authenticated = _require_pool_materialization_on_connection(
            conn,
            expected={
                field: persisted[field]
                for field in _POOL_MATERIALIZATION_INPUT_FIELDS
            },
        )
        return authenticated["materialization"]

    def require_pool_materialization_on_connection(
        self,
        conn: sqlite3.Connection,
        expected: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Re-authenticate ledger, canonical Strategy, and its one exact audit."""

        normalized = _normalize_pool_materialization(expected)
        return _require_pool_materialization_on_connection(
            conn,
            expected=normalized,
        )

    def materialize_pool_strategy_draft_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        strategy: Strategy,
        materialization: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Create one canonical root draft and immutable lineage in one transaction."""

        if not conn.in_transaction:
            raise StrategyPoolMaterializationError(
                "Pool materialization requires a caller-owned writer transaction"
            )
        normalized = _normalize_pool_materialization(materialization)
        if strategy.id != normalized["strategy_id"]:
            raise StrategyPoolMaterializationError(
                "materialized Strategy identity does not match the ledger"
            )
        if strategy.strategy_type != normalized["strategy_type"]:
            raise StrategyPoolMaterializationError(
                "materialized Strategy type does not match the Pool"
            )
        if strategy.spec is None:
            raise StrategyPoolMaterializationError(
                "Pool materialization requires a canonical Strategy DSL"
            )
        if not hmac.compare_digest(
            strategy_spec_hash(strategy.spec),
            normalized["strategy_spec_hash"],
        ):
            raise StrategyPoolMaterializationError(
                "materialized Strategy effect hash changed"
            )
        if not hmac.compare_digest(
            _strategy_dsl_content_hash(strategy.spec),
            normalized["strategy_dsl_content_hash"],
        ):
            raise StrategyPoolMaterializationError(
                "materialized Strategy DSL content hash changed"
            )

        collisions = conn.execute(
            """
            SELECT *
              FROM strategy_pool_materializations
             WHERE id = ? OR pool_revision_id = ? OR strategy_id = ?
             ORDER BY id
            """,
            (
                normalized["id"],
                normalized["pool_revision_id"],
                normalized["strategy_id"],
            ),
        ).fetchall()
        if collisions:
            if len(collisions) != 1:
                raise StrategyPoolMaterializationError(
                    "Pool materialization identity collision"
                )
            persisted = _pool_materialization_from_row(collisions[0])
            if not _same_pool_materialization(persisted, normalized):
                raise StrategyPoolMaterializationError(
                    "Pool materialization identity collision"
                )
            return _require_pool_materialization_on_connection(
                conn,
                expected=normalized,
            )

        orphan_strategy = conn.execute(
            "SELECT 1 FROM strategies WHERE id = ?",
            (normalized["strategy_id"],),
        ).fetchone()
        if orphan_strategy is not None:
            raise StrategyPoolMaterializationError(
                "materialized Strategy identity collision"
            )

        stamp = created_at or _now()
        _insert_strategy_row(
            conn,
            normalized["task_id"],
            strategy,
            stamp,
        )
        detail = _pool_materialization_audit_detail(normalized)
        conn.execute(
            """
            INSERT INTO audit(
                id, kind, actor, target_ref, inputs_hash, outcome,
                detail_json, at
            )
            VALUES (?, ?, 'system', ?, ?, 'succeeded', ?, ?)
            """,
            (
                normalized["audit_id"],
                POOL_MATERIALIZATION_AUDIT_KIND,
                normalized["strategy_id"],
                normalized["selected_design_hash"],
                _pool_materialization_canonical_json(detail),
                stamp,
            ),
        )
        conn.execute(
            """
            INSERT INTO strategy_pool_materializations(
                id, schema_version, producer_version, task_id, strategy_type,
                strategy_id, strategy_version, pool_id, pool_revision_id,
                pool_revision,
                pool_snapshot_hash, pool_artifact_id,
                pool_artifact_content_hash, selected_design_hash,
                requirements_json, requirements_hash, strategy_spec_hash,
                strategy_dsl_content_hash, audit_id, created_at
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                normalized["id"],
                POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION,
                POOL_MATERIALIZATION_PRODUCER_VERSION,
                normalized["task_id"],
                normalized["strategy_type"],
                normalized["strategy_id"],
                1,
                normalized["pool_id"],
                normalized["pool_revision_id"],
                normalized["pool_revision"],
                normalized["pool_snapshot_hash"],
                normalized["pool_artifact_id"],
                normalized["pool_artifact_content_hash"],
                normalized["selected_design_hash"],
                _pool_materialization_canonical_json(normalized["requirements"]),
                normalized["requirements_hash"],
                normalized["strategy_spec_hash"],
                normalized["strategy_dsl_content_hash"],
                normalized["audit_id"],
                stamp,
            ),
        )
        return _require_pool_materialization_on_connection(
            conn,
            expected=normalized,
        )

    def list_for_task(self, task_id: str) -> list[Strategy]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, task_id, strategy_type, rules_json, score_col,
                       default_decision_json, description, created_at,
                       dsl_json, dsl_schema_version, dsl_content_hash
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
                SELECT id, task_id, strategy_type, version, status, asset_status,
                       adopted_at, adoption_reason, parent_strategy_id, created_at
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
        with connect(self.db_path) as conn:
            return self.adopt_strategy_with_audit_on_connection(
                conn,
                strategy_id,
                reason=reason,
                audit=audit,
                adopted_at=adopted_at,
                effect_execution_id=effect_execution_id,
                runtime_generation=runtime_generation,
            )

    def adopt_strategy_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        strategy_id: str,
        *,
        reason: str,
        audit: dict,
        adopted_at: str | None = None,
        effect_execution_id: str | None = None,
        runtime_generation: str | None = None,
    ) -> dict:
        """Apply the adoption lifecycle and effect fence on a caller connection.

        The caller owns commit/rollback so adoption can share one SQLite
        transaction with its required artifact and audit writes.
        """
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
            "status": LEGACY_STATUS_ADOPTED,
            "asset_status": ASSET_STATUS_ADOPTED_LOCAL,
        }
        stamp = adopted_at or _now()
        # A governed side effect must validate its immutable authorization
        # snapshot and mutate the domain under the same writer lock.  Without
        # BEGIN IMMEDIATE a different adoption could land after the champion
        # snapshot check but before our guarded writes, defeating the fence.
        if governed_effect and not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        head = conn.execute(
            """
            SELECT task_id, strategy_type, version, status, asset_status, rules_json,
                   score_col, default_decision_json, description,
                   dsl_json, dsl_schema_version, dsl_content_hash
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
        current_asset_status = resolve_asset_status(
            str(head["status"]),
            str(head["asset_status"]),
        )
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
                asset_status=current_asset_status,
                strategy_spec_hash=_strategy_spec_hash_from_row(head),
            )
        if str(head["status"]) != LEGACY_STATUS_DRAFT or current_asset_status not in {
            ASSET_STATUS_DRAFT,
            ASSET_STATUS_VALIDATED,
        }:
            raise ConflictError(
                f"strategy {strategy_id} is not adoptable: "
                f"status={head['status']}, asset_status={current_asset_status}"
            )
        # Retire in-role siblings first, in the same transaction, so the
        # "at most one adopted per (task, type)" invariant holds atomically.
        retired_ids = _current_local_champion_ids(
            conn,
            task_id=task_id,
            strategy_type=strategy_type,
            exclude_strategy_id=strategy_id,
        )
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
                "UPDATE strategies SET status = 'retired', asset_status = 'retired' "
                "WHERE id = ? AND status = 'adopted' "
                "AND asset_status = 'adopted_local'",
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
                    "status": LEGACY_STATUS_RETIRED,
                    "asset_status": ASSET_STATUS_RETIRED,
                },
            )
        cursor = conn.execute(
            """
            UPDATE strategies
               SET status = ?, asset_status = ?, adopted_at = ?, adoption_reason = ?
             WHERE id = ? AND status = ? AND asset_status = ?
            """,
            (
                LEGACY_STATUS_ADOPTED,
                ASSET_STATUS_ADOPTED_LOCAL,
                stamp,
                normalized_reason,
                strategy_id,
                LEGACY_STATUS_DRAFT,
                current_asset_status,
            ),
        )
        if cursor.rowcount == 0:
            current = conn.execute(
                "SELECT status, asset_status FROM strategies WHERE id = ?",
                (strategy_id,),
            ).fetchone()
            raise ConflictError(
                f"strategy {strategy_id} is not adoptable: "
                f"status={current['status']}, asset_status={current['asset_status']}"
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
            src = _select_strategy_version_source(conn, strategy_id)
            task_id = str(src["task_id"])
            max_version_row = conn.execute(
                """
                SELECT MAX(version) AS mx FROM strategies
                 WHERE task_id = ? AND strategy_type = ?
                """,
                (task_id, str(src["strategy_type"])),
            ).fetchone()
            next_version = int(max_version_row["mx"] or 0) + 1
            return _insert_strategy_version_from_source(
                conn,
                src,
                target_task_id=task_id,
                next_version=next_version,
                rules=rules,
                strategy_spec=strategy_spec,
                description=description,
                new_strategy_id=new_strategy_id,
                created_at=stamp,
            )

    def new_version_from_on_connection(
        self,
        conn: sqlite3.Connection,
        parent_strategy_id: str,
        *,
        target_task_id: str,
        rules: list | None = None,
        strategy_spec: StrategySpec | dict | None = None,
        description: str | None = None,
        new_strategy_id: str | None = None,
        created_at: str | None = None,
    ) -> Strategy:
        """Create a governed draft version inside the caller's transaction.

        ``target_task_id`` is explicit so a red-monitoring handoff can create a
        fresh strategy task and then attach the child to that task without a
        second transaction. The caller must already hold a writer transaction;
        this method never commits or adopts the child.
        """

        if not conn.in_transaction:
            raise ValueError(
                "new_version_from_on_connection requires a caller-owned transaction"
            )
        if rules is not None and strategy_spec is not None:
            raise ValueError("rules and strategy_spec are mutually exclusive")
        target_id = str(target_task_id).strip()
        if not target_id:
            raise ValueError("target_task_id must be non-empty")
        target_task = conn.execute(
            "SELECT id, task_type FROM tasks WHERE id = ?",
            (target_id,),
        ).fetchone()
        if target_task is None:
            raise KeyError(target_id)
        if str(target_task["task_type"]) != "strategy":
            raise ValueError("target_task_id must identify a strategy task")

        src = _select_strategy_version_source(conn, parent_strategy_id)
        parent_version = int(src["version"])
        target_max_row = conn.execute(
            """
            SELECT MAX(version) AS mx FROM strategies
             WHERE task_id = ? AND strategy_type = ?
            """,
            (target_id, str(src["strategy_type"])),
        ).fetchone()
        # A newly-created handoff task has no strategies, so this is exactly
        # parent.version + 1. max(...) also keeps the method safe if a caller
        # deliberately targets a non-empty strategy task.
        next_version = max(
            parent_version + 1,
            int(target_max_row["mx"] or 0) + 1,
        )
        return _insert_strategy_version_from_source(
            conn,
            src,
            target_task_id=target_id,
            next_version=next_version,
            rules=rules,
            strategy_spec=strategy_spec,
            description=description,
            new_strategy_id=new_strategy_id,
            created_at=created_at or _now(),
        )

    def save_strategy_artifact(
        self,
        strategy_id: str,
        *,
        kind: str,
        path: str,
        created_at: str | None = None,
        artifact_id: str | None = None,
    ) -> str:
        with connect(self.db_path) as conn:
            return _insert_strategy_artifact_row(
                conn,
                strategy_id,
                kind=kind,
                path=path,
                created_at=created_at,
                artifact_id=artifact_id,
            )

    def register_verified_strategy_artifact(
        self,
        strategy_id: str,
        *,
        kind: str,
        path: str,
        content_hash: str,
        content_size: int,
        provenance: Mapping[str, Any],
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Register a content-addressed artifact or return its exact replay."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return _register_verified_strategy_artifact_row(
                conn,
                strategy_id,
                kind=kind,
                path=path,
                content_hash=content_hash,
                content_size=content_size,
                provenance=provenance,
                created_at=created_at,
            )

    def register_verified_strategy_artifact_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        strategy_id: str,
        *,
        kind: str,
        path: str,
        content_hash: str,
        content_size: int,
        provenance: Mapping[str, Any],
        audit: dict,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        """Register verified bytes and their first-write audit in one transaction."""

        record = _register_verified_strategy_artifact_row(
            conn,
            strategy_id,
            kind=kind,
            path=path,
            content_hash=content_hash,
            content_size=content_size,
            provenance=provenance,
            created_at=created_at,
        )
        if record["created"]:
            _write_audit_row(conn, **audit)
        return record

    def save_strategy_artifact_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        strategy_id: str,
        *,
        kind: str,
        path: str,
        audit: dict,
        created_at: str | None = None,
        artifact_id: str | None = None,
    ) -> str:
        new_id = _insert_strategy_artifact_row(
            conn,
            strategy_id,
            kind=kind,
            path=path,
            created_at=created_at,
            artifact_id=artifact_id,
        )
        _write_audit_row(conn, **audit)
        return new_id

    def list_strategy_artifacts(self, strategy_id: str) -> list[dict]:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, strategy_id, kind, path, created_at,
                       content_hash, content_size, provenance_json
                  FROM strategy_artifacts
                 WHERE strategy_id = ?
                 ORDER BY created_at, id
                """,
                (strategy_id,),
            ).fetchall()
        return [_strategy_artifact_record_from_row(row) for row in rows]

    def list_strategy_artifacts_for_task(self, task_id: str) -> list[dict]:
        """Return artifact rows joined to their task-owned strategy metadata."""

        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT a.id, a.strategy_id, a.kind, a.path, a.created_at,
                       a.content_hash, a.content_size, a.provenance_json,
                       s.strategy_type, s.version, s.status, s.asset_status
                  FROM strategy_artifacts a
                  JOIN strategies s ON s.id = a.strategy_id
                 WHERE s.task_id = ?
                 ORDER BY a.created_at, a.id
                """,
                (task_id,),
            ).fetchall()
        records = []
        for row in rows:
            record = _strategy_artifact_record_from_row(row)
            record.update(
                {
                "id": str(row["id"]),
                "strategy_id": str(row["strategy_id"]),
                "kind": str(row["kind"]),
                "path": str(row["path"]),
                "created_at": str(row["created_at"]),
                "strategy_type": str(row["strategy_type"]),
                "version": int(row["version"]),
                "status": str(row["status"]),
                "asset_status": resolve_asset_status(
                    str(row["status"]), str(row["asset_status"])
                ),
                }
            )
            records.append(record)
        return records

    def get_strategy_artifact_for_task(
        self,
        task_id: str,
        artifact_id: str,
    ) -> dict | None:
        """Load an artifact only when its strategy belongs to ``task_id``."""

        with connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT a.id, a.strategy_id, a.kind, a.path, a.created_at,
                       a.content_hash, a.content_size, a.provenance_json,
                       s.strategy_type, s.version, s.status, s.asset_status
                  FROM strategy_artifacts a
                  JOIN strategies s ON s.id = a.strategy_id
                 WHERE s.task_id = ? AND a.id = ?
                """,
                (task_id, artifact_id),
            ).fetchone()
        if row is None:
            return None
        status = str(row["status"])
        asset_status = resolve_asset_status(status, str(row["asset_status"]))
        record = _strategy_artifact_record_from_row(row)
        record.update({
            "id": str(row["id"]),
            "strategy_id": str(row["strategy_id"]),
            "kind": str(row["kind"]),
            "path": str(row["path"]),
            "created_at": str(row["created_at"]),
            "strategy_type": str(row["strategy_type"]),
            "version": int(row["version"]),
            "status": status,
            "asset_status": asset_status,
        })
        return record

    def save_backtest(
        self,
        backtest_id: str,
        strategy_id: str,
        dataset_id: str,
        result: BacktestRecord,
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
        result: BacktestRecord,
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

    def save_backtest_with_audit_on_connection(
        self,
        conn: sqlite3.Connection,
        backtest_id: str,
        strategy_id: str,
        dataset_id: str,
        result: BacktestRecord,
        *,
        audit: dict,
        created_at: str | None = None,
    ) -> None:
        """Persist backtest evidence on a caller-owned writer transaction."""

        _insert_backtest_row(
            conn,
            backtest_id,
            strategy_id,
            dataset_id,
            result,
            created_at or _now(),
        )
        _write_audit_row(conn, **audit)

    def get_backtest_on_connection(
        self,
        conn: sqlite3.Connection,
        backtest_id: str,
    ) -> BacktestRecord | None:
        row = conn.execute(
            """
            SELECT id, strategy_id, dataset_id, result_json, created_at
              FROM backtests
             WHERE id = ?
            """,
            (backtest_id,),
        ).fetchone()
        return None if row is None else _backtest_result_from_row(row)

    def get_backtest(self, backtest_id: str) -> BacktestRecord | None:
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

    def list_backtests(self, strategy_id: str) -> list[BacktestRecord]:
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
        """Adopted strategies whose next monitoring run is overdue.

        V2 derives cadence from the latest immutable monitoring-plan revision and
        anchors it at that plan's latest persisted run. A plan with no run gets a
        full cadence from the later of plan creation and strategy adoption, so a
        newly-created plan cannot be immediately reported as stale. Only a
        strategy with no ledger plan at all falls back to its V1 artifact.
        """
        reference = now or datetime.now(UTC)
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT s.id AS strategy_id, s.adopted_at AS adopted_at,
                       (SELECT a.path FROM strategy_artifacts a
                         WHERE a.strategy_id = s.id AND a.kind = 'monitoring_plan_json'
                         ORDER BY a.created_at DESC, a.id DESC LIMIT 1) AS plan_path
                  FROM strategies s
                 WHERE s.status = 'adopted' AND s.asset_status = 'adopted_local'
                 ORDER BY s.adopted_at, s.id
                """
            ).fetchall()
            due: list[dict] = []
            for row in rows:
                strategy_id = str(row["strategy_id"])
                ledger_plan = conn.execute(
                    """
                    SELECT id, payload_json, created_at
                      FROM strategy_monitoring_plans
                     WHERE strategy_id = ?
                     ORDER BY revision DESC, created_at DESC, id DESC
                     LIMIT 1
                    """,
                    (strategy_id,),
                ).fetchone()
                if ledger_plan is not None:
                    resolved = _ledger_monitoring_due_fields(
                        conn,
                        strategy_id=strategy_id,
                        adopted_at=row["adopted_at"],
                        plan_row=ledger_plan,
                    )
                else:
                    resolved = _legacy_monitoring_due_fields(
                        plan_path=row["plan_path"],
                        adopted_at=row["adopted_at"],
                    )
                if resolved is None:
                    continue
                anchor_ts, last_run_at, cadence_days = resolved
                due_at = anchor_ts + timedelta(days=cadence_days)
                overdue_seconds = (reference - due_at).total_seconds()
                if overdue_seconds <= 0:
                    continue
                due.append(
                    {
                        "strategy_id": strategy_id,
                        "due_at": due_at.isoformat(),
                        "overdue_days": overdue_seconds / 86400.0,
                        "last_run_at": last_run_at,
                        "cadence_days": cadence_days,
                    }
                )
        due.sort(key=lambda item: (-item["overdue_days"], item["strategy_id"]))
        return due


def _normalize_pool_materialization(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StrategyPoolMaterializationError(
            "Pool materialization must be an object"
        )
    missing = sorted(_POOL_MATERIALIZATION_INPUT_FIELDS - set(value))
    unexpected = sorted(set(value) - _POOL_MATERIALIZATION_INPUT_FIELDS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unsupported: " + ", ".join(unexpected))
        raise StrategyPoolMaterializationError(
            "Pool materialization fields are invalid (" + "; ".join(details) + ")"
        )
    strategy_type = _pool_materialization_text(
        value["strategy_type"],
        "strategy_type",
    )
    if strategy_type not in _POOL_MATERIALIZATION_STRATEGY_TYPES:
        raise StrategyPoolMaterializationError(
            "Pool materialization strategy_type is unsupported"
        )
    revision = value["pool_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise StrategyPoolMaterializationError(
            "Pool materialization pool_revision must be a positive integer"
        )
    raw_requirements = value["requirements"]
    if not isinstance(raw_requirements, list):
        raise StrategyPoolMaterializationError(
            "Pool materialization requirements must be an array"
        )
    requirements_json = _pool_materialization_canonical_json(raw_requirements)
    requirements = json.loads(requirements_json)
    normalized = {
        "id": _pool_materialization_text(value["id"], "materialization_id"),
        "task_id": _pool_materialization_text(value["task_id"], "task_id"),
        "strategy_type": strategy_type,
        "strategy_id": _pool_materialization_text(
            value["strategy_id"],
            "strategy_id",
        ),
        "pool_id": _pool_materialization_text(value["pool_id"], "pool_id"),
        "pool_revision_id": _pool_materialization_text(
            value["pool_revision_id"],
            "pool_revision_id",
        ),
        "pool_revision": revision,
        "pool_artifact_id": _pool_materialization_text(
            value["pool_artifact_id"],
            "pool_artifact_id",
        ),
        "requirements": requirements,
        "audit_id": _pool_materialization_text(value["audit_id"], "audit_id"),
    }
    for field in _POOL_MATERIALIZATION_HASH_FIELDS:
        normalized[field] = _pool_materialization_hash(value[field], field)
    computed_requirements_hash = hashlib.sha256(
        requirements_json.encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(
        computed_requirements_hash,
        normalized["requirements_hash"],
    ):
        raise StrategyPoolMaterializationError(
            "Pool materialization requirements hash changed"
        )
    return normalized


def _pool_materialization_from_row(row: sqlite3.Row) -> dict[str, Any]:
    if (
        str(row["schema_version"])
        != POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION
        or str(row["producer_version"]) != POOL_MATERIALIZATION_PRODUCER_VERSION
    ):
        raise StrategyPoolMaterializationError(
            "Pool materialization schema or producer version changed"
        )
    try:
        requirements = json.loads(str(row["requirements_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyPoolMaterializationError(
            "Pool materialization requirements are invalid"
        ) from exc
    normalized = _normalize_pool_materialization(
        {
            "id": row["id"],
            "task_id": row["task_id"],
            "strategy_type": row["strategy_type"],
            "strategy_id": row["strategy_id"],
            "pool_id": row["pool_id"],
            "pool_revision_id": row["pool_revision_id"],
            "pool_revision": row["pool_revision"],
            "pool_snapshot_hash": row["pool_snapshot_hash"],
            "pool_artifact_id": row["pool_artifact_id"],
            "pool_artifact_content_hash": row["pool_artifact_content_hash"],
            "selected_design_hash": row["selected_design_hash"],
            "requirements": requirements,
            "requirements_hash": row["requirements_hash"],
            "strategy_spec_hash": row["strategy_spec_hash"],
            "strategy_dsl_content_hash": row["strategy_dsl_content_hash"],
            "audit_id": row["audit_id"],
        }
    )
    canonical_requirements = _pool_materialization_canonical_json(
        normalized["requirements"]
    )
    strategy_version = row["strategy_version"]
    if (
        isinstance(strategy_version, bool)
        or not isinstance(strategy_version, int)
        or strategy_version != 1
    ):
        raise StrategyPoolMaterializationError(
            "Pool materialization Strategy version changed"
        )
    if not hmac.compare_digest(
        canonical_requirements,
        str(row["requirements_json"]),
    ):
        raise StrategyPoolMaterializationError(
            "Pool materialization requirements are not canonical"
        )
    created_at = _pool_materialization_text(row["created_at"], "created_at")
    return {
        **normalized,
        "schema_version": POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION,
        "producer_version": POOL_MATERIALIZATION_PRODUCER_VERSION,
        "strategy_version": strategy_version,
        "created_at": created_at,
    }


def _same_pool_materialization(
    persisted: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(
        persisted.get(field) == expected.get(field)
        for field in _POOL_MATERIALIZATION_INPUT_FIELDS
    )


def _require_pool_materialization_on_connection(
    conn: sqlite3.Connection,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM strategy_pool_materializations WHERE id = ?",
        (expected["id"],),
    ).fetchone()
    if row is None:
        raise StrategyPoolMaterializationError(
            "Pool materialization ledger disappeared"
        )
    persisted = _pool_materialization_from_row(row)
    if not _same_pool_materialization(persisted, expected):
        raise StrategyPoolMaterializationError(
            "Pool materialization ledger changed"
        )

    pool_revision = conn.execute(
        """
        SELECT id, pool_id, task_id, strategy_type, revision, snapshot_hash,
               artifact_id, artifact_content_hash
          FROM strategy_candidate_pool_revisions
         WHERE id = ?
        """,
        (persisted["pool_revision_id"],),
    ).fetchone()
    expected_pool_revision = {
        "id": persisted["pool_revision_id"],
        "pool_id": persisted["pool_id"],
        "task_id": persisted["task_id"],
        "strategy_type": persisted["strategy_type"],
        "revision": persisted["pool_revision"],
        "snapshot_hash": persisted["pool_snapshot_hash"],
        "artifact_id": persisted["pool_artifact_id"],
        "artifact_content_hash": persisted["pool_artifact_content_hash"],
    }
    if pool_revision is None or any(
        pool_revision[field] != expected_value
        for field, expected_value in expected_pool_revision.items()
    ):
        raise StrategyPoolMaterializationError(
            "Pool revision binding changed"
        )
    pool_artifact = conn.execute(
        """
        SELECT id, task_id, kind, content_hash
          FROM task_artifacts
         WHERE id = ?
        """,
        (persisted["pool_artifact_id"],),
    ).fetchone()
    expected_pool_artifact = {
        "id": persisted["pool_artifact_id"],
        "task_id": persisted["task_id"],
        "kind": "strategy_candidate_pool_json",
        "content_hash": persisted["pool_artifact_content_hash"],
    }
    if pool_artifact is None or any(
        pool_artifact[field] != expected_value
        for field, expected_value in expected_pool_artifact.items()
    ):
        raise StrategyPoolMaterializationError(
            "Pool artifact binding changed"
        )

    strategy_row = conn.execute(
        """
        SELECT id, task_id, strategy_type, version, status, asset_status,
               adopted_at, adoption_reason, parent_strategy_id, created_at,
               rules_json, score_col, default_decision_json, description,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (persisted["strategy_id"],),
    ).fetchone()
    if strategy_row is None:
        raise StrategyPoolMaterializationError(
            "materialized Strategy disappeared"
        )
    try:
        strategy = _strategy_from_row(strategy_row)
        metadata = _strategy_meta_from_row(strategy_row)
        effect_hash = _strategy_spec_hash_from_row(strategy_row)
    except (StrategyError, StrategyLifecycleError, TypeError, ValueError) as exc:
        raise StrategyPoolMaterializationError(
            "materialized Strategy is invalid"
        ) from exc
    if (
        strategy.id != persisted["strategy_id"]
        or strategy.strategy_type != persisted["strategy_type"]
        or metadata["task_id"] != persisted["task_id"]
        or metadata["version"] != persisted["strategy_version"]
        or metadata["parent_strategy_id"] is not None
        or metadata["created_at"] != persisted["created_at"]
        or not hmac.compare_digest(
            effect_hash,
            persisted["strategy_spec_hash"],
        )
        or not hmac.compare_digest(
            str(strategy_row["dsl_content_hash"]),
            persisted["strategy_dsl_content_hash"],
        )
    ):
        raise StrategyPoolMaterializationError(
            "materialized Strategy binding changed"
        )

    audit_detail = _pool_materialization_audit_detail(persisted)
    audit = conn.execute(
        """
        SELECT id, kind, actor, target_ref, inputs_hash, outcome,
               detail_json, at
          FROM audit
         WHERE id = ?
        """,
        (persisted["audit_id"],),
    ).fetchone()
    expected_audit = {
        "id": persisted["audit_id"],
        "kind": POOL_MATERIALIZATION_AUDIT_KIND,
        "actor": "system",
        "target_ref": persisted["strategy_id"],
        "inputs_hash": persisted["selected_design_hash"],
        "outcome": "succeeded",
        "detail_json": _pool_materialization_canonical_json(audit_detail),
        "at": persisted["created_at"],
    }
    if audit is None or any(
        str(audit[field]) != expected_value
        for field, expected_value in expected_audit.items()
    ):
        raise StrategyPoolMaterializationError(
            "Pool materialization audit changed"
        )
    audit_count = conn.execute(
        """
        SELECT COUNT(*)
          FROM audit
         WHERE kind = ? AND target_ref = ?
        """,
        (
            POOL_MATERIALIZATION_AUDIT_KIND,
            persisted["strategy_id"],
        ),
    ).fetchone()
    if audit_count is None or int(audit_count[0]) != 1:
        raise StrategyPoolMaterializationError(
            "Pool materialization must own exactly one creation audit"
        )
    return {
        "materialization": persisted,
        "strategy": strategy,
        "metadata": metadata,
    }


def _pool_materialization_audit_detail(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": POOL_MATERIALIZATION_LEDGER_SCHEMA_VERSION,
        "materialization_id": value["id"],
        "task_id": value["task_id"],
        "strategy_type": value["strategy_type"],
        "strategy_id": value["strategy_id"],
        "strategy_version": value.get("strategy_version", 1),
        "pool_id": value["pool_id"],
        "pool_revision_id": value["pool_revision_id"],
        "pool_revision": value["pool_revision"],
        "pool_snapshot_hash": value["pool_snapshot_hash"],
        "pool_artifact_id": value["pool_artifact_id"],
        "pool_artifact_content_hash": value["pool_artifact_content_hash"],
        "selected_design_hash": value["selected_design_hash"],
        "requirements_hash": value["requirements_hash"],
        "strategy_spec_hash": value["strategy_spec_hash"],
        "strategy_dsl_content_hash": value["strategy_dsl_content_hash"],
    }


def _pool_materialization_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise StrategyPoolMaterializationError(
            f"Pool materialization {field} must be canonical text"
        )
    return value


def _pool_materialization_hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StrategyPoolMaterializationError(
            f"Pool materialization {field} must be a lowercase SHA-256"
        )
    return value


def _pool_materialization_canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise StrategyPoolMaterializationError(
            "Pool materialization must be finite canonical JSON"
        ) from exc


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


def _insert_strategy_artifact_row(
    conn: sqlite3.Connection,
    strategy_id: str,
    *,
    kind: str,
    path: str,
    created_at: str | None = None,
    artifact_id: str | None = None,
) -> str:
    new_id = artifact_id or uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO strategy_artifacts(id, strategy_id, kind, path, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (new_id, strategy_id, str(kind), str(path), created_at or _now()),
    )
    return new_id


def _register_verified_strategy_artifact_row(
    conn: sqlite3.Connection,
    strategy_id: str,
    *,
    kind: str,
    path: str,
    content_hash: str,
    content_size: int,
    provenance: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    normalized_strategy_id = _required_artifact_text(
        strategy_id, field="strategy_id"
    )
    normalized_kind = _required_artifact_text(kind, field="kind")
    normalized_path = _required_artifact_text(path, field="path")
    normalized_hash = _required_artifact_hash(content_hash)
    if isinstance(content_size, bool) or not isinstance(content_size, int):
        raise StrategyArtifactDataError("content_size must be a non-negative integer")
    if content_size < 0:
        raise StrategyArtifactDataError("content_size must be a non-negative integer")
    normalized_provenance, provenance_json = _canonical_artifact_provenance(
        provenance
    )
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    strategy = conn.execute(
        "SELECT 1 FROM strategies WHERE id = ?", (normalized_strategy_id,)
    ).fetchone()
    if strategy is None:
        raise StrategyArtifactDataError(
            f"strategy not found: {normalized_strategy_id}"
        )
    existing = conn.execute(
        """
        SELECT id, strategy_id, kind, path, created_at,
               content_hash, content_size, provenance_json
          FROM strategy_artifacts
         WHERE strategy_id = ? AND kind = ? AND content_hash = ?
        """,
        (normalized_strategy_id, normalized_kind, normalized_hash),
    ).fetchone()
    if existing is not None:
        drifted = []
        if str(existing["path"]) != normalized_path:
            drifted.append("path")
        if int(existing["content_size"]) != content_size:
            drifted.append("content_size")
        if str(existing["provenance_json"]) != provenance_json:
            drifted.append("provenance")
        if drifted:
            raise StrategyArtifactConflictError(
                "verified strategy artifact content already exists with drift in "
                + ", ".join(drifted)
            )
        record = _strategy_artifact_record_from_row(existing)
        record["created"] = False
        return record

    artifact_id = _verified_strategy_artifact_id(
        strategy_id=normalized_strategy_id,
        kind=normalized_kind,
        content_hash=normalized_hash,
    )
    collision = conn.execute(
        "SELECT 1 FROM strategy_artifacts WHERE id = ?", (artifact_id,)
    ).fetchone()
    if collision is not None:
        raise StrategyArtifactConflictError(
            "verified strategy artifact id collided with another identity"
        )
    timestamp = created_at or _now()
    try:
        conn.execute(
            """
            INSERT INTO strategy_artifacts(
                id, strategy_id, kind, path, created_at,
                content_hash, content_size, provenance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                normalized_strategy_id,
                normalized_kind,
                normalized_path,
                timestamp,
                normalized_hash,
                content_size,
                provenance_json,
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise StrategyArtifactConflictError(
            "could not register verified strategy artifact"
        ) from exc
    return {
        "id": artifact_id,
        "strategy_id": normalized_strategy_id,
        "kind": normalized_kind,
        "path": normalized_path,
        "created_at": timestamp,
        "content_hash": normalized_hash,
        "content_size": content_size,
        "provenance": normalized_provenance,
        "integrity_status": "verified",
        "created": True,
    }


def _strategy_artifact_record_from_row(row: sqlite3.Row) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": str(row["id"]),
        "strategy_id": str(row["strategy_id"]),
        "kind": str(row["kind"]),
        "path": str(row["path"]),
        "created_at": str(row["created_at"]),
    }
    integrity_values = (
        row["content_hash"],
        row["content_size"],
        row["provenance_json"],
    )
    if all(value is None for value in integrity_values):
        return record
    if any(value is None for value in integrity_values):
        raise StrategyArtifactDataError(
            f"strategy artifact {record['id']} has partial integrity metadata"
        )
    content_hash = _required_artifact_hash(row["content_hash"])
    content_size = row["content_size"]
    if isinstance(content_size, bool) or not isinstance(content_size, int):
        raise StrategyArtifactDataError(
            f"strategy artifact {record['id']} has invalid content_size"
        )
    try:
        provenance = json.loads(str(row["provenance_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyArtifactDataError(
            f"strategy artifact {record['id']} has invalid provenance"
        ) from exc
    normalized_provenance, canonical = _canonical_artifact_provenance(provenance)
    if not hmac.compare_digest(canonical, str(row["provenance_json"])):
        raise StrategyArtifactDataError(
            f"strategy artifact {record['id']} provenance is not canonical"
        )
    record.update(
        {
            "content_hash": content_hash,
            "content_size": content_size,
            "provenance": normalized_provenance,
            "integrity_status": "verified",
        }
    )
    return record


def _required_artifact_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise StrategyArtifactDataError(f"{field} must be a non-empty string")
    return value.strip()


def _required_artifact_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise StrategyArtifactDataError("content_hash must be a SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise StrategyArtifactDataError("content_hash must be a SHA-256 digest")
    return normalized


def _canonical_artifact_provenance(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise StrategyArtifactDataError("provenance must be a JSON object")
    try:
        payload = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise StrategyArtifactDataError("provenance must be a JSON object") from exc
    if not isinstance(normalized, dict):
        raise StrategyArtifactDataError("provenance must be a JSON object")
    return normalized, payload


def _verified_strategy_artifact_id(
    *, strategy_id: str, kind: str, content_hash: str
) -> str:
    identity = json.dumps(
        [strategy_id, kind, content_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"{_STRATEGY_ARTIFACT_IDENTITY_NAMESPACE}:{identity}".encode("utf-8")
    ).hexdigest()


def _select_strategy_version_source(
    conn: sqlite3.Connection,
    strategy_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, rules_json, score_col,
               default_decision_json, description, created_at, version,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (strategy_id,),
    ).fetchone()
    if row is None:
        raise KeyError(strategy_id)
    return row


def _insert_strategy_version_from_source(
    conn: sqlite3.Connection,
    src: sqlite3.Row,
    *,
    target_task_id: str,
    next_version: int,
    rules: list | None,
    strategy_spec: StrategySpec | dict | None,
    description: str | None,
    new_strategy_id: str | None,
    created_at: str,
) -> Strategy:
    parent_strategy_id = str(src["id"])
    child_id = new_strategy_id or uuid.uuid4().hex
    child_description = (
        str(description) if description is not None else str(src["description"])
    )
    source_strategy = _strategy_from_row(src)
    if strategy_spec is not None:
        from marvis.packs.strategy.strategy import build_strategy_from_spec

        base_spec = parse_strategy_spec(strategy_spec)
        if base_spec.strategy_type != source_strategy.strategy_type:
            raise ValueError("new strategy version cannot change strategy_type")
        built = build_strategy_from_spec(
            base_spec,
            score_col=source_strategy.score_col,
            description=child_description,
        )
        child_rules = built.rules
    elif rules is not None:
        # Local import avoids making repository module import order part of the
        # pack's public surface while still using the canonical legacy->DSL path.
        from marvis.packs.strategy.strategy import build_strategy

        built = build_strategy(
            source_strategy.strategy_type,
            [_strategy_rule_to_dict(_coerce_rule(rule)) for rule in rules],
            score_col=source_strategy.score_col,
            default_decision=source_strategy.spec.default_action.value,
            description=child_description,
        )
        child_rules = built.rules
        spec_payload = built.spec.to_dict()
        spec_payload["default_action"] = source_strategy.spec.default_action.to_dict()
        base_spec = parse_strategy_spec(spec_payload)
    else:
        base_spec = source_strategy.spec
        if base_spec is None:  # pragma: no cover - _strategy_from_row always supplies it
            raise ValueError("source strategy has no canonical DSL")
        child_rules = _rules_with_spec_identity(source_strategy.rules, base_spec)

    spec_payload = base_spec.to_dict()
    metadata = dict(spec_payload.get("metadata") or {})
    lineage = dict(metadata.get("lineage") or {})
    lineage.update(
        {
            "source": "strategy_version",
            "parent_strategy_id": parent_strategy_id,
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
            version, status, asset_status, parent_strategy_id,
            dsl_json, dsl_schema_version, dsl_content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            child_id,
            target_task_id,
            child_spec.strategy_type,
            _dump_json_any([_strategy_rule_to_dict(rule) for rule in child_rules]),
            _optional_str(src["score_col"]),
            _dump_json_any(child_spec.default_action.decision_value),
            child_description,
            created_at,
            next_version,
            LEGACY_STATUS_DRAFT,
            ASSET_STATUS_DRAFT,
            parent_strategy_id,
            canonical_strategy_json(child_spec),
            child_spec.schema_version,
            _strategy_dsl_content_hash(child_spec),
        ),
    )
    row = conn.execute(
        """
        SELECT id, task_id, strategy_type, rules_json, score_col,
               default_decision_json, description, created_at,
               dsl_json, dsl_schema_version, dsl_content_hash
          FROM strategies
         WHERE id = ?
        """,
        (child_id,),
    ).fetchone()
    assert row is not None
    return _strategy_from_row(row)


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
        _strategy_dsl_content_hash(spec),
    )


def _insert_strategy_row(
    conn: sqlite3.Connection,
    task_id: str,
    strategy: Strategy,
    created_at: str,
) -> None:
    values = _strategy_insert_values(task_id, strategy, created_at)
    conn.execute(
        """
        INSERT INTO strategies(
            id, task_id, strategy_type, rules_json, score_col,
            default_decision_json, description, created_at,
            status, asset_status,
            dsl_json, dsl_schema_version, dsl_content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values[:8]
        + (LEGACY_STATUS_DRAFT, ASSET_STATUS_DRAFT)
        + values[8:],
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
    dsl_state = _strategy_dsl_state(row)
    if dsl_state == "canonical":
        spec = _strategy_spec_from_row(row)
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
    status = str(row["status"])
    asset_status = resolve_asset_status(status, row["asset_status"])
    return {
        "id": str(row["id"]),
        "task_id": str(row["task_id"]),
        "strategy_type": str(row["strategy_type"]),
        "version": int(row["version"]),
        "status": status,
        "asset_status": asset_status,
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
    compatibility_source = strategy
    if spec.default_action.type in {"limit", "pricing", "segment"}:
        compatibility_source = Strategy(
            id=strategy.id,
            strategy_type=strategy.strategy_type,
            rules=strategy.rules,
            score_col=strategy.score_col,
            default_decision=spec.default_action.value,
            description=strategy.description,
        )
    try:
        compatibility_spec = legacy_strategy_to_spec(compatibility_source)
    except StrategyError as exc:
        raise ValueError(
            "strategy compatibility fields do not match canonical DSL"
        ) from exc
    if (
        strategy.strategy_type != spec.strategy_type
        or _dump_json_any(strategy.default_decision)
        != _dump_json_any(spec.default_action.decision_value)
        or len(compatibility_spec.rules) != len(spec.rules)
        or any(
            _compatibility_rule_payload(legacy_rule)
            != _compatibility_rule_payload(typed_rule)
            for legacy_rule, typed_rule in zip(
                compatibility_spec.rules,
                spec.rules,
                strict=True,
            )
        )
    ):
        raise ValueError("strategy compatibility fields do not match canonical DSL")


def _compatibility_rule_payload(rule) -> dict[str, Any]:
    payload = rule.to_dict()
    action = payload["action"]
    if action["type"] in {"limit", "pricing", "segment"}:
        action.pop("output_value", None)
    return payload


def _strategy_dsl_content_hash(spec: StrategySpec) -> str:
    return hashlib.sha256(canonical_strategy_json(spec).encode("utf-8")).hexdigest()


def _strategy_dsl_state(row: sqlite3.Row) -> str:
    values = tuple(
        row[field] if field in row.keys() else None
        for field in ("dsl_json", "dsl_schema_version", "dsl_content_hash")
    )
    populated = tuple(value is not None for value in values)
    if all(populated):
        return "canonical"
    if not any(populated):
        return "legacy"
    raise ValueError("strategy canonical DSL columns are incomplete")


def _strategy_spec_from_row(row: sqlite3.Row) -> StrategySpec:
    spec = parse_strategy_spec(json.loads(str(row["dsl_json"])))
    stored_version = _optional_str(row["dsl_schema_version"])
    if stored_version != spec.schema_version:
        raise ValueError(
            "strategy dsl_schema_version does not match canonical dsl_json"
        )
    stored_hash = _optional_str(row["dsl_content_hash"])
    if stored_hash is None:
        raise ValueError("strategy canonical DSL columns are incomplete")
    if (
        len(stored_hash) != 64
        or stored_hash != stored_hash.lower()
        or any(character not in "0123456789abcdef" for character in stored_hash)
        or not hmac.compare_digest(stored_hash, _strategy_dsl_content_hash(spec))
    ):
        raise ValueError(
            "strategy compatibility fields do not match canonical DSL: "
            "dsl_content_hash drifted"
        )
    return spec


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
    result: BacktestRecord,
    created_at: str,
) -> tuple:
    if str(result.strategy_id) != str(strategy_id):
        raise ValueError(
            "backtest result strategy_id does not match persisted strategy_id"
        )
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
    result: BacktestRecord,
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


def _backtest_result_to_dict(result: BacktestRecord) -> dict:
    if isinstance(result, StrategyBacktestResult):
        return result.to_dict()
    payload = asdict(result)
    payload["by_segment"] = list(result.by_segment)
    return payload


def _backtest_result_from_row(row: sqlite3.Row) -> BacktestRecord:
    return _backtest_result_from_dict(_load_json_object(row["result_json"]))


def _backtest_result_from_dict(payload: dict) -> BacktestRecord:
    if "schema_version" in payload:
        schema_version = payload["schema_version"]
        if schema_version == STRATEGY_BACKTEST_SCHEMA_VERSION:
            return StrategyBacktestResult.from_dict(payload)
        raise ValueError(
            f"unsupported strategy backtest schema_version: {schema_version!r}"
        )
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


def _ledger_monitoring_due_fields(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    adopted_at,
    plan_row: sqlite3.Row,
) -> tuple[datetime, str | None, int] | None:
    """Resolve V2 cadence and anchor without consulting mutable artifacts."""

    try:
        payload = json.loads(str(plan_row["payload_json"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        plan = monitoring_plan_from_dict(
            payload,
            source=f"database:{plan_row['id']}",
        )
    except (TypeError, ValueError, StrategyError):
        return None
    if plan.strategy_id != strategy_id or plan.monitoring_plan_id != str(plan_row["id"]):
        return None

    # The ledger stores only completed monitoring results; failed executions do
    # not create strategy_monitoring_runs rows, so the newest row is the newest
    # successful run for this exact plan revision.
    run = conn.execute(
        """
        SELECT created_at
          FROM strategy_monitoring_runs
         WHERE strategy_id = ? AND monitoring_plan_id = ?
         ORDER BY created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id, str(plan_row["id"])),
    ).fetchone()
    if run is not None:
        last_run_at = _optional_str(run["created_at"])
        anchor_ts = _parse_iso(last_run_at)
        if anchor_ts is None:
            return None
    else:
        last_run_at = None
        anchors = [
            timestamp
            for timestamp in (
                _parse_iso(plan_row["created_at"]),
                _parse_iso(adopted_at),
            )
            if timestamp is not None
        ]
        if not anchors:
            return None
        anchor_ts = max(anchors)
    return anchor_ts, last_run_at, int(plan.cadence_days or 30)


def _legacy_monitoring_due_fields(
    *,
    plan_path,
    adopted_at,
) -> tuple[datetime, str | None, int] | None:
    """Preserve the artifact-backed V1 due calculation unchanged."""

    resolved_path = _optional_str(plan_path)
    if resolved_path is None:
        return None
    plan = _read_monitoring_plan_fields(resolved_path)
    if plan is None:
        return None
    last_run_at = _optional_str(plan.get("last_run_at"))
    anchor_ts = _parse_iso(last_run_at) or _parse_iso(adopted_at)
    if anchor_ts is None:
        return None
    try:
        cadence_days = int(plan.get("cadence_days") or 30)
    except (TypeError, ValueError):
        return None
    return anchor_ts, last_run_at, cadence_days


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
