"""Atomic red-monitoring handoff into a fresh strategy task.

This is a repository/service boundary, not an API or Agent policy layer.  It
accepts only persisted monitoring evidence, validates the latest red state and
reuses the immutable registered dataset file by reference.  No sample rows are
copied and the child strategy always remains a draft.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from marvis.data.contracts import Dataset
from marvis.db_schema import connect
from marvis.domain import TASK_TYPE_STRATEGY, StrategyTaskInput, TaskCreate
from marvis.files import sha256_file
from marvis.repositories.audit import _write_audit_row
from marvis.repositories.datasets import _dataset_from_row, _insert_dataset_row
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_monitoring import (
    StrategyMonitoringDataError,
    validate_monitoring_run_result,
)
from marvis.repositories.tasks import (
    _insert_task_record_row,
    _row_to_task,
    _task_record_from_create,
)
from marvis.state_machine import ConflictError
from marvis.strategy_lifecycle import StrategyLifecycleError, is_locally_adopted


_HANDOFF_AUDIT_KIND = "strategy.monitoring.new_version_handoff"


class StrategyHandoffRepository:
    """Creates a new governed strategy task from one persisted red run."""

    def __init__(self, db_path: Path, datasets_root: Path):
        self.db_path = Path(db_path)
        self.datasets_root = Path(datasets_root)
        self._strategy_repo = StrategyRepository(self.db_path)

    def create_new_version_from_red_run(
        self,
        *,
        source_task_id: str,
        parent_strategy_id: str,
        monitoring_run_id: str,
        new_task_id: str | None = None,
        new_strategy_id: str | None = None,
        new_dataset_id: str | None = None,
        actor: str = "system",
        created_at: str | None = None,
    ) -> dict[str, str]:
        """Run the complete handoff under one ``BEGIN IMMEDIATE`` transaction."""

        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.create_new_version_from_red_run_on_connection(
                conn,
                source_task_id=source_task_id,
                parent_strategy_id=parent_strategy_id,
                monitoring_run_id=monitoring_run_id,
                new_task_id=new_task_id,
                new_strategy_id=new_strategy_id,
                new_dataset_id=new_dataset_id,
                actor=actor,
                created_at=created_at,
            )

    def create_new_version_from_red_run_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        source_task_id: str,
        parent_strategy_id: str,
        monitoring_run_id: str,
        new_task_id: str | None = None,
        new_strategy_id: str | None = None,
        new_dataset_id: str | None = None,
        actor: str = "system",
        created_at: str | None = None,
    ) -> dict[str, str]:
        """Connection-scoped handoff; caller owns the writer transaction.

        The method intentionally does not begin, commit, or roll back.  This lets
        a later governed disposition tool include its own effect receipt in the
        exact same transaction.
        """

        if not conn.in_transaction:
            raise ValueError(
                "create_new_version_from_red_run_on_connection requires "
                "a caller-owned BEGIN IMMEDIATE transaction"
            )
        source_id = _required_identifier(source_task_id, field="source_task_id")
        parent_id = _required_identifier(parent_strategy_id, field="parent_strategy_id")
        run_id = _required_identifier(monitoring_run_id, field="monitoring_run_id")
        task_id = _optional_identifier(new_task_id) or uuid.uuid4().hex
        strategy_id = _optional_identifier(new_strategy_id) or uuid.uuid4().hex
        dataset_id = _optional_identifier(new_dataset_id) or uuid.uuid4().hex
        timestamp = _optional_identifier(created_at) or _utc_now()
        normalized_actor = _optional_identifier(actor) or "system"

        replay = conn.execute(
            "SELECT id FROM audit WHERE kind = ? AND target_ref = ? LIMIT 1",
            (_HANDOFF_AUDIT_KIND, run_id),
        ).fetchone()
        if replay is not None:
            raise ConflictError(
                f"monitoring run {run_id} already created a new strategy version"
            )

        source_row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (source_id,),
        ).fetchone()
        if source_row is None:
            raise KeyError(source_id)
        source_task = _row_to_task(source_row)
        if source_task.task_type != TASK_TYPE_STRATEGY:
            raise ConflictError("source task must be a strategy task")

        parent = conn.execute(
            """
            SELECT id, task_id, strategy_type, version, status, asset_status
              FROM strategies
             WHERE id = ?
            """,
            (parent_id,),
        ).fetchone()
        if parent is None:
            raise KeyError(parent_id)
        if str(parent["task_id"]) != source_id:
            raise ConflictError("parent strategy does not belong to source task")
        try:
            parent_is_adopted = is_locally_adopted(
                parent["status"], parent["asset_status"]
            )
        except StrategyLifecycleError as exc:
            raise ConflictError("parent strategy lifecycle state is invalid") from exc
        if not parent_is_adopted:
            raise ConflictError(
                "parent strategy must be adopted_local before handoff; "
                "local adoption is not production deployment"
            )
        parent_type = str(parent["strategy_type"])
        parent_version = int(parent["version"])

        run = conn.execute(
            """
            SELECT id, strategy_id, monitoring_plan_id, dataset_id,
                   dataset_content_hash, result_json, result_hash,
                   overall_level, created_at
              FROM strategy_monitoring_runs
             WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if run is None:
            raise KeyError(run_id)
        if str(run["strategy_id"]) != parent_id:
            raise ConflictError("monitoring run does not belong to parent strategy")
        if str(run["overall_level"]) != "red":
            raise ConflictError("monitoring run must be red to create a new version")
        _validate_run_result_integrity(run)

        latest_run = conn.execute(
            """
            SELECT id FROM strategy_monitoring_runs
             WHERE strategy_id = ?
             ORDER BY created_at DESC, id DESC
             LIMIT 1
            """,
            (parent_id,),
        ).fetchone()
        if latest_run is None or str(latest_run["id"]) != run_id:
            raise ConflictError("handoff requires the latest monitoring run")

        plan_id = str(run["monitoring_plan_id"])
        plan = conn.execute(
            """
            SELECT id, strategy_id, strategy_version, revision,
                   payload_json, payload_hash
              FROM strategy_monitoring_plans
             WHERE id = ?
            """,
            (plan_id,),
        ).fetchone()
        if plan is None or str(plan["strategy_id"]) != parent_id:
            raise ConflictError("monitoring plan does not belong to parent strategy")
        if int(plan["strategy_version"]) != parent_version:
            raise ConflictError(
                "monitoring plan does not bind the parent strategy version"
            )
        _validate_plan_payload_integrity(plan)
        latest_plan = conn.execute(
            """
            SELECT id FROM strategy_monitoring_plans
             WHERE strategy_id = ?
             ORDER BY revision DESC, created_at DESC, id DESC
             LIMIT 1
            """,
            (parent_id,),
        ).fetchone()
        if latest_plan is None or str(latest_plan["id"]) != plan_id:
            raise ConflictError("handoff requires the latest monitoring plan")

        source_dataset_row = conn.execute(
            """
            SELECT id, task_id, role, source_path, format, sheet, row_count,
                   columns_json, has_target, target_col, created_at, content_hash
              FROM datasets
             WHERE id = ?
            """,
            (str(run["dataset_id"]),),
        ).fetchone()
        if source_dataset_row is None:
            raise KeyError(str(run["dataset_id"]))
        if str(source_dataset_row["task_id"]) != source_id:
            raise ConflictError("monitoring dataset does not belong to source task")
        source_dataset = _dataset_from_row(source_dataset_row)
        stored_hash = _required_sha256(
            source_dataset.content_hash,
            field="registered dataset content_hash",
        )
        run_hash = _required_sha256(
            run["dataset_content_hash"],
            field="monitoring run dataset_content_hash",
        )
        if not hmac.compare_digest(stored_hash, run_hash):
            raise ConflictError(
                "monitoring run dataset hash does not match the registered dataset"
            )
        live_path = _resolve_registered_dataset_path(
            self.datasets_root,
            source_dataset.source_path,
        )
        live_hash = sha256_file(live_path)
        if not hmac.compare_digest(live_hash, stored_hash):
            raise ConflictError(
                "registered dataset live content hash no longer matches stored evidence"
            )
        shared_hash_rows = conn.execute(
            "SELECT DISTINCT content_hash FROM datasets WHERE source_path = ?",
            (source_dataset.source_path,),
        ).fetchall()
        if any(
            row["content_hash"] is None
            or not hmac.compare_digest(str(row["content_hash"]), stored_hash)
            for row in shared_hash_rows
        ):
            raise ConflictError(
                "registered dataset source has inconsistent content-hash references"
            )

        strategy_input = _child_strategy_input(
            source_task.strategy_input,
            parent_strategy_id=parent_id,
            parent_strategy_type=parent_type,
        )
        task_payload = TaskCreate(
            task_type=TASK_TYPE_STRATEGY,
            model_name=source_task.model_name,
            model_version=source_task.model_version,
            validator=source_task.validator,
            source_dir=source_task.source_dir,
            algorithm=source_task.algorithm,
            run_mode=source_task.run_mode,
            target_col=source_task.target_col,
            score_col=source_task.score_col,
            split_col=source_task.split_col,
            time_col=source_task.time_col,
            feature_columns=list(source_task.feature_columns),
            target_type=source_task.target_type,
            recipes=list(source_task.recipes),
            sample_weight_col=source_task.sample_weight_col,
            oot_ks_min=source_task.oot_ks_min,
            strategy_input=strategy_input,
            metrics=(
                None if source_task.metrics is None else list(source_task.metrics)
            ),
            capability_tier=source_task.capability_tier,
            # Material paths and report values are deliberately not inherited:
            # they are task-local artifacts, not safe strategy metadata.
        )
        child_task = replace(
            _task_record_from_create(task_payload),
            id=task_id,
            created_at=timestamp,
            updated_at=timestamp,
        )
        _insert_task_record_row(conn, child_task, report_values={})

        child_dataset = Dataset(
            id=dataset_id,
            task_id=task_id,
            role="strategy.new_version_source",
            source_path=source_dataset.source_path,
            format=source_dataset.format,
            sheet=source_dataset.sheet,
            row_count=source_dataset.row_count,
            columns=source_dataset.columns,
            has_target=source_dataset.has_target,
            target_col=source_dataset.target_col,
            created_at=timestamp,
            content_hash=stored_hash,
        )
        _insert_dataset_row(conn, child_dataset)

        child_strategy = self._strategy_repo.new_version_from_on_connection(
            conn,
            parent_id,
            target_task_id=task_id,
            new_strategy_id=strategy_id,
            created_at=timestamp,
        )
        child_meta = conn.execute(
            "SELECT version, status, asset_status FROM strategies WHERE id = ?",
            (child_strategy.id,),
        ).fetchone()
        if (
            child_meta is None
            or str(child_meta["status"]) != "draft"
            or str(child_meta["asset_status"]) != "draft"
        ):
            raise ConflictError("new strategy version was not created as a draft")
        if int(child_meta["version"]) != parent_version + 1:
            raise ConflictError(
                "new strategy version does not follow its parent version"
            )

        result = {
            "source_task_id": source_id,
            "new_task_id": task_id,
            "parent_strategy_id": parent_id,
            "new_strategy_id": child_strategy.id,
            "monitoring_run_id": run_id,
            "monitoring_plan_id": plan_id,
            "source_dataset_id": source_dataset.id,
            "new_dataset_id": child_dataset.id,
            "dataset_content_hash": stored_hash,
        }
        audit_detail = {
            **result,
            "parent_strategy_version": parent_version,
            "new_strategy_version": int(child_meta["version"]),
            "monitoring_plan_revision": int(plan["revision"]),
            "monitoring_plan_payload_hash": str(plan["payload_hash"]),
            "monitoring_run_result_hash": str(run["result_hash"]),
            "dataset_source_path": source_dataset.source_path,
        }
        _write_audit_row(
            conn,
            kind=_HANDOFF_AUDIT_KIND,
            target_ref=run_id,
            actor=normalized_actor,
            inputs_hash=_canonical_hash(
                {
                    "source_task_id": source_id,
                    "parent_strategy_id": parent_id,
                    "monitoring_run_id": run_id,
                    "monitoring_run_result_hash": str(run["result_hash"]),
                    "monitoring_plan_id": plan_id,
                    "monitoring_plan_payload_hash": str(plan["payload_hash"]),
                    "dataset_content_hash": stored_hash,
                }
            ),
            outcome="succeeded",
            detail=audit_detail,
        )
        return result


def _child_strategy_input(
    source: StrategyTaskInput | None,
    *,
    parent_strategy_id: str,
    parent_strategy_type: str,
) -> StrategyTaskInput:
    if source is None:
        return StrategyTaskInput(
            strategy_type=parent_strategy_type,
            baseline_strategy_id=parent_strategy_id,
        )
    if source.strategy_type != parent_strategy_type:
        raise ConflictError(
            "source task strategy contract does not match parent strategy type"
        )
    return replace(source, baseline_strategy_id=parent_strategy_id)


def _validate_run_result_integrity(row: sqlite3.Row) -> None:
    result_json = str(row["result_json"])
    try:
        result = json.loads(result_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConflictError("monitoring run result JSON is invalid") from exc
    if not isinstance(result, dict):
        raise ConflictError("monitoring run result must be a JSON object")
    canonical = _canonical_json(result)
    stored_hash = _required_sha256(row["result_hash"], field="monitoring result_hash")
    if not hmac.compare_digest(
        hashlib.sha256(canonical.encode()).hexdigest(), stored_hash
    ):
        raise ConflictError("monitoring run result hash is invalid")
    try:
        validate_monitoring_run_result(
            result,
            overall_level=row["overall_level"],
        )
    except StrategyMonitoringDataError as exc:
        raise ConflictError(
            "monitoring run result violates the semantic contract"
        ) from exc


def _validate_plan_payload_integrity(row: sqlite3.Row) -> None:
    payload_json = str(row["payload_json"])
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConflictError("monitoring plan payload JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ConflictError("monitoring plan payload must be a JSON object")
    canonical = _canonical_json(payload)
    stored_hash = _required_sha256(row["payload_hash"], field="monitoring payload_hash")
    if not hmac.compare_digest(
        hashlib.sha256(canonical.encode()).hexdigest(), stored_hash
    ):
        raise ConflictError("monitoring plan payload hash is invalid")


def _resolve_registered_dataset_path(root: Path, source_path: str) -> Path:
    resolved_root = Path(root).resolve()
    relative = Path(source_path)
    if relative.is_absolute():
        raise ConflictError("registered dataset source_path must be relative")
    resolved_path = (resolved_root / relative).resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ConflictError("registered dataset source_path escapes datasets root")
    if not resolved_path.is_file():
        raise ConflictError("registered dataset source file is missing")
    return resolved_path


def _required_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConflictError(f"{field} must be a sha256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConflictError(f"{field} must be a sha256 hash") from exc
    return value.lower()


def _required_identifier(value: object, *, field: str) -> str:
    normalized = _optional_identifier(value)
    if normalized is None:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def _optional_identifier(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("identifier must be a non-empty string")
    return value.strip()


def _canonical_json(value: dict) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ConflictError("monitoring evidence is not canonical JSON") from exc


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
