"""Immutable strategy monitoring plan and run ledgers.

This repository deliberately contains no monitoring business logic. It freezes
the exact plan revision and evidence hashes that a later governed disposition
tool will execute against. Plan changes are append-only and use optimistic CAS;
runs are immutable and reject replay of the same plan/evidence tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import hmac
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
import uuid

from marvis.db_schema import connect
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    canonical_economics_bindings_hash,
    canonical_monitoring_plan_hash,
    canonical_monitoring_plan_json,
    monitoring_plan_from_dict,
)


_LEVELS = frozenset({"green", "amber", "red", "n/a"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


class StrategyMonitoringNotFoundError(KeyError):
    """A requested plan, run, strategy, or dataset does not exist."""


class StrategyMonitoringConflictError(RuntimeError):
    """The caller's expected latest plan revision/hash is stale."""


class StrategyMonitoringDuplicateError(StrategyMonitoringConflictError):
    """An immutable plan id or monitoring evidence tuple was already recorded."""


class StrategyMonitoringDataError(ValueError):
    """Supplied or persisted monitoring evidence violates the ledger contract."""


def validate_monitoring_run_result(
    result: object,
    *,
    overall_level: object,
) -> None:
    """Validate the semantic relationship between checks and the run level.

    Hash verification proves that persisted bytes have not changed unnoticed;
    this contract additionally proves that those bytes describe a coherent
    monitoring result.
    """

    if not isinstance(result, Mapping):
        raise StrategyMonitoringDataError("monitoring result must be an object")

    run_level = _monitoring_level(overall_level, field="run overall_level")
    if "overall_level" not in result:
        raise StrategyMonitoringDataError(
            "monitoring result overall_level must be present"
        )
    result_level = _monitoring_level(
        result["overall_level"], field="monitoring result overall_level"
    )
    if result_level != run_level:
        raise StrategyMonitoringDataError(
            "monitoring result overall_level does not match run overall_level"
        )

    checks = result.get("checks")
    if not isinstance(checks, list):
        raise StrategyMonitoringDataError("monitoring result checks must be a list")

    check_levels: list[str] = []
    check_ids: set[str] = set()
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise StrategyMonitoringDataError(
                f"monitoring result checks[{index}] must be an object"
            )
        check_id = _required_identifier(check.get("id"), field=f"checks[{index}].id")
        if check_id in check_ids:
            raise StrategyMonitoringDataError(
                f"monitoring result contains duplicate check id: {check_id}"
            )
        check_ids.add(check_id)
        check_levels.append(
            _monitoring_level(check.get("level"), field=f"checks[{index}].level")
        )

    calculated_level = _overall_level(check_levels)
    if result_level != calculated_level:
        raise StrategyMonitoringDataError(
            "monitoring result overall_level does not match check levels: "
            f"expected {calculated_level}, found {result_level}"
        )


@dataclass(frozen=True)
class MonitoringPlanRecord:
    id: str
    strategy_id: str
    strategy_version: int
    revision: int
    schema_version: str
    plan: MonitoringPlan
    payload_hash: str
    supersedes_plan_id: str | None
    created_at: str


@dataclass(frozen=True)
class MonitoringRunRecord:
    id: str
    strategy_id: str
    monitoring_plan_id: str
    dataset_id: str
    dataset_content_hash: str
    strategy_effect_hash: str
    economics_binding_hash: str
    result: dict[str, Any]
    result_hash: str
    overall_level: str
    created_at: str


class StrategyMonitoringRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def create_plan(
        self,
        plan: MonitoringPlan,
        *,
        expected_revision: int,
        expected_payload_hash: str | None = None,
        plan_id: str | None = None,
        created_at: str | None = None,
    ) -> MonitoringPlanRecord:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.create_plan_on_connection(
                conn,
                plan,
                expected_revision=expected_revision,
                expected_payload_hash=expected_payload_hash,
                plan_id=plan_id,
                created_at=created_at,
            )

    def create_plan_on_connection(
        self,
        conn: sqlite3.Connection,
        plan: MonitoringPlan,
        *,
        expected_revision: int,
        expected_payload_hash: str | None = None,
        plan_id: str | None = None,
        created_at: str | None = None,
    ) -> MonitoringPlanRecord:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        if not isinstance(plan, MonitoringPlan):
            raise StrategyMonitoringDataError("plan must be a MonitoringPlan")
        expected = _non_negative_int(expected_revision, field="expected_revision")
        expected_hash = _optional_sha256(
            expected_payload_hash, field="expected_payload_hash"
        )
        resolved_id = (
            _optional_identifier(plan_id, field="plan_id")
            or plan.monitoring_plan_id
            or uuid.uuid4().hex
        )
        if plan.monitoring_plan_id not in (None, resolved_id):
            raise StrategyMonitoringDataError(
                "plan monitoring_plan_id does not match requested plan_id"
            )
        if plan.last_run_at is not None:
            raise StrategyMonitoringDataError(
                "immutable monitoring ledger plans must keep last_run_at empty"
            )

        duplicate = conn.execute(
            "SELECT 1 FROM strategy_monitoring_plans WHERE id = ?", (resolved_id,)
        ).fetchone()
        if duplicate is not None:
            raise StrategyMonitoringDuplicateError(
                f"duplicate monitoring plan id: {resolved_id}"
            )

        strategy = conn.execute(
            "SELECT id, version FROM strategies WHERE id = ?", (plan.strategy_id,)
        ).fetchone()
        if strategy is None:
            raise StrategyMonitoringNotFoundError(
                f"strategy not found: {plan.strategy_id}"
            )
        actual_strategy_version = int(strategy["version"])
        if actual_strategy_version != plan.version:
            raise StrategyMonitoringDataError(
                "monitoring plan strategy version does not match current strategy version"
            )

        latest_row = _select_latest_plan_row(conn, plan.strategy_id)
        latest = None if latest_row is None else _plan_record_from_row(latest_row)
        actual_revision = 0 if latest is None else latest.revision
        if actual_revision != expected:
            raise StrategyMonitoringConflictError(
                f"stale monitoring plan revision: expected {expected}, found {actual_revision}"
            )
        if plan.revision != expected + 1:
            raise StrategyMonitoringDataError(
                f"new monitoring plan revision must be {expected + 1}"
            )
        if latest is None:
            if expected_hash is not None:
                raise StrategyMonitoringConflictError(
                    "initial monitoring plan must not supply an expected payload hash"
                )
            if plan.supersedes_plan_id is not None:
                raise StrategyMonitoringDataError(
                    "initial monitoring plan cannot supersede another plan"
                )
        else:
            if expected_hash is None or not hmac.compare_digest(
                latest.payload_hash, expected_hash
            ):
                raise StrategyMonitoringConflictError(
                    "monitoring plan expected payload hash does not match latest plan"
                )
            if plan.supersedes_plan_id != latest.id:
                raise StrategyMonitoringDataError(
                    "monitoring plan supersedes_plan_id must identify the latest plan"
                )

        stored_plan = replace(plan, monitoring_plan_id=resolved_id)
        payload_json = canonical_monitoring_plan_json(stored_plan)
        payload_hash = canonical_monitoring_plan_hash(stored_plan)
        schema_version = f"strategy.monitoring_plan.v{stored_plan.plan_version}"
        timestamp = _optional_identifier(created_at, field="created_at") or _now()
        try:
            conn.execute(
                """
                INSERT INTO strategy_monitoring_plans(
                    id, strategy_id, strategy_version, revision, schema_version,
                    payload_json, payload_hash, supersedes_plan_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_id,
                    stored_plan.strategy_id,
                    stored_plan.version,
                    stored_plan.revision,
                    schema_version,
                    payload_json,
                    payload_hash,
                    stored_plan.supersedes_plan_id,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StrategyMonitoringDuplicateError(
                "duplicate monitoring plan revision or payload"
            ) from exc
        row = conn.execute(
            "SELECT * FROM strategy_monitoring_plans WHERE id = ?", (resolved_id,)
        ).fetchone()
        assert row is not None
        return _plan_record_from_row(row)

    def get_plan(self, plan_id: str) -> MonitoringPlanRecord | None:
        normalized_id = _required_identifier(plan_id, field="plan_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM strategy_monitoring_plans WHERE id = ?",
                (normalized_id,),
            ).fetchone()
        return None if row is None else _plan_record_from_row(row)

    def latest_plan(self, strategy_id: str) -> MonitoringPlanRecord | None:
        normalized_id = _required_identifier(strategy_id, field="strategy_id")
        with connect(self.db_path) as conn:
            row = _select_latest_plan_row(conn, normalized_id)
        return None if row is None else _plan_record_from_row(row)

    def list_plans(self, strategy_id: str) -> list[MonitoringPlanRecord]:
        normalized_id = _required_identifier(strategy_id, field="strategy_id")
        with connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM strategy_monitoring_plans
                 WHERE strategy_id = ?
                 ORDER BY revision, created_at, id
                """,
                (normalized_id,),
            ).fetchall()
        return [_plan_record_from_row(row) for row in rows]

    def create_run(
        self,
        *,
        strategy_id: str,
        monitoring_plan_id: str,
        expected_plan_revision: int,
        expected_plan_payload_hash: str,
        dataset_id: str,
        dataset_content_hash: str,
        strategy_effect_hash: str,
        economics_binding_hash: str,
        result: Mapping[str, Any],
        overall_level: str,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> MonitoringRunRecord:
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            return self.create_run_on_connection(
                conn,
                strategy_id=strategy_id,
                monitoring_plan_id=monitoring_plan_id,
                expected_plan_revision=expected_plan_revision,
                expected_plan_payload_hash=expected_plan_payload_hash,
                dataset_id=dataset_id,
                dataset_content_hash=dataset_content_hash,
                strategy_effect_hash=strategy_effect_hash,
                economics_binding_hash=economics_binding_hash,
                result=result,
                overall_level=overall_level,
                run_id=run_id,
                created_at=created_at,
            )

    def create_run_on_connection(
        self,
        conn: sqlite3.Connection,
        *,
        strategy_id: str,
        monitoring_plan_id: str,
        expected_plan_revision: int,
        expected_plan_payload_hash: str,
        dataset_id: str,
        dataset_content_hash: str,
        strategy_effect_hash: str,
        economics_binding_hash: str,
        result: Mapping[str, Any],
        overall_level: str,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> MonitoringRunRecord:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        normalized_strategy_id = _required_identifier(strategy_id, field="strategy_id")
        normalized_plan_id = _required_identifier(
            monitoring_plan_id, field="monitoring_plan_id"
        )
        normalized_dataset_id = _required_identifier(dataset_id, field="dataset_id")
        expected_revision = _non_negative_int(
            expected_plan_revision, field="expected_plan_revision"
        )
        expected_plan_hash = _required_sha256(
            expected_plan_payload_hash, field="expected_plan_payload_hash"
        )
        dataset_hash = _required_sha256(
            dataset_content_hash, field="dataset_content_hash"
        )
        effect_hash = _required_sha256(
            strategy_effect_hash, field="strategy_effect_hash"
        )
        economics_hash = _required_sha256(
            economics_binding_hash, field="economics_binding_hash"
        )
        level = _monitoring_level(overall_level, field="overall_level")
        resolved_run_id = _optional_identifier(run_id, field="run_id") or uuid.uuid4().hex
        duplicate_id = conn.execute(
            "SELECT 1 FROM strategy_monitoring_runs WHERE id = ?", (resolved_run_id,)
        ).fetchone()
        if duplicate_id is not None:
            raise StrategyMonitoringDuplicateError(
                f"duplicate monitoring run id: {resolved_run_id}"
            )

        strategy = conn.execute(
            "SELECT id, task_id FROM strategies WHERE id = ?",
            (normalized_strategy_id,),
        ).fetchone()
        if strategy is None:
            raise StrategyMonitoringNotFoundError(
                f"strategy not found: {normalized_strategy_id}"
            )
        plan_row = conn.execute(
            "SELECT * FROM strategy_monitoring_plans WHERE id = ? AND strategy_id = ?",
            (normalized_plan_id, normalized_strategy_id),
        ).fetchone()
        if plan_row is None:
            raise StrategyMonitoringNotFoundError(
                f"monitoring plan not found: {normalized_plan_id}"
            )
        plan_record = _plan_record_from_row(plan_row)
        latest_row = _select_latest_plan_row(conn, normalized_strategy_id)
        assert latest_row is not None
        if str(latest_row["id"]) != plan_record.id:
            raise StrategyMonitoringConflictError(
                "monitoring run must bind the latest monitoring plan"
            )
        if plan_record.revision != expected_revision:
            raise StrategyMonitoringConflictError(
                "monitoring run expected plan revision is stale"
            )
        if not hmac.compare_digest(plan_record.payload_hash, expected_plan_hash):
            raise StrategyMonitoringConflictError(
                "monitoring run expected plan payload hash does not match"
            )

        dataset = conn.execute(
            "SELECT id, task_id, content_hash FROM datasets WHERE id = ?",
            (normalized_dataset_id,),
        ).fetchone()
        if dataset is None or str(dataset["task_id"]) != str(strategy["task_id"]):
            raise StrategyMonitoringNotFoundError(
                f"dataset not found in strategy task: {normalized_dataset_id}"
            )
        stored_dataset_hash = _optional_sha256(
            dataset["content_hash"], field="registered dataset content_hash"
        )
        if stored_dataset_hash is None or not hmac.compare_digest(
            stored_dataset_hash, dataset_hash
        ):
            raise StrategyMonitoringDataError(
                "monitoring run dataset content hash does not match registered dataset"
            )
        expected_economics_hash = canonical_economics_bindings_hash(
            plan_record.plan.economics_bindings
        )
        if not hmac.compare_digest(expected_economics_hash, economics_hash):
            raise StrategyMonitoringDataError(
                "monitoring run economics binding hash does not match monitoring plan"
            )
        baseline_effect_hash = plan_record.plan.expectation_baseline.get(
            "strategy_effect_hash"
        )
        if baseline_effect_hash is not None:
            normalized_baseline_hash = _required_sha256(
                baseline_effect_hash,
                field="expectation_baseline.strategy_effect_hash",
            )
            if not hmac.compare_digest(normalized_baseline_hash, effect_hash):
                raise StrategyMonitoringDataError(
                    "monitoring run strategy effect hash does not match monitoring plan"
                )

        result_json = _canonical_json_object(result, field="result")
        result_payload = json.loads(result_json)
        validate_monitoring_run_result(result_payload, overall_level=level)
        result_hash = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        timestamp = _optional_identifier(created_at, field="created_at") or _now()
        try:
            conn.execute(
                """
                INSERT INTO strategy_monitoring_runs(
                    id, strategy_id, monitoring_plan_id, dataset_id,
                    dataset_content_hash, strategy_effect_hash,
                    economics_binding_hash, result_json, result_hash,
                    overall_level, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_run_id,
                    normalized_strategy_id,
                    normalized_plan_id,
                    normalized_dataset_id,
                    dataset_hash,
                    effect_hash,
                    economics_hash,
                    result_json,
                    result_hash,
                    level,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StrategyMonitoringDuplicateError(
                "duplicate monitoring run evidence"
            ) from exc
        row = conn.execute(
            "SELECT * FROM strategy_monitoring_runs WHERE id = ?", (resolved_run_id,)
        ).fetchone()
        assert row is not None
        return _run_record_from_row(row)

    def get_run(self, run_id: str) -> MonitoringRunRecord | None:
        normalized_id = _required_identifier(run_id, field="run_id")
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM strategy_monitoring_runs WHERE id = ?",
                (normalized_id,),
            ).fetchone()
        return None if row is None else _run_record_from_row(row)

    def list_runs(
        self,
        strategy_id: str,
        *,
        monitoring_plan_id: str | None = None,
    ) -> list[MonitoringRunRecord]:
        normalized_strategy_id = _required_identifier(
            strategy_id, field="strategy_id"
        )
        normalized_plan_id = _optional_identifier(
            monitoring_plan_id, field="monitoring_plan_id"
        )
        query = (
            "SELECT * FROM strategy_monitoring_runs WHERE strategy_id = ?"
            if normalized_plan_id is None
            else (
                "SELECT * FROM strategy_monitoring_runs "
                "WHERE strategy_id = ? AND monitoring_plan_id = ?"
            )
        )
        params = (
            (normalized_strategy_id,)
            if normalized_plan_id is None
            else (normalized_strategy_id, normalized_plan_id)
        )
        with connect(self.db_path) as conn:
            rows = conn.execute(
                f"{query} ORDER BY created_at, id", params
            ).fetchall()
        return [_run_record_from_row(row) for row in rows]


def _select_latest_plan_row(
    conn: sqlite3.Connection, strategy_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM strategy_monitoring_plans
         WHERE strategy_id = ?
         ORDER BY revision DESC, created_at DESC, id DESC
         LIMIT 1
        """,
        (strategy_id,),
    ).fetchone()


def _plan_record_from_row(row: sqlite3.Row) -> MonitoringPlanRecord:
    try:
        payload = json.loads(str(row["payload_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyMonitoringDataError(
            f"stored monitoring plan {row['id']} has invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise StrategyMonitoringDataError(
            f"stored monitoring plan {row['id']} payload is not an object"
        )
    try:
        plan = monitoring_plan_from_dict(payload, source=f"database:{row['id']}")
    except ValueError as exc:
        raise StrategyMonitoringDataError(
            f"stored monitoring plan {row['id']} violates the plan contract"
        ) from exc
    calculated_hash = canonical_monitoring_plan_hash(plan)
    stored_hash = _required_sha256(row["payload_hash"], field="stored payload_hash")
    if not hmac.compare_digest(calculated_hash, stored_hash):
        raise StrategyMonitoringDataError(
            f"stored monitoring plan {row['id']} payload hash does not match"
        )
    record = MonitoringPlanRecord(
        id=str(row["id"]),
        strategy_id=str(row["strategy_id"]),
        strategy_version=int(row["strategy_version"]),
        revision=int(row["revision"]),
        schema_version=str(row["schema_version"]),
        plan=plan,
        payload_hash=stored_hash,
        supersedes_plan_id=(
            str(row["supersedes_plan_id"])
            if row["supersedes_plan_id"] is not None
            else None
        ),
        created_at=str(row["created_at"]),
    )
    if (
        plan.monitoring_plan_id != record.id
        or plan.strategy_id != record.strategy_id
        or plan.version != record.strategy_version
        or plan.revision != record.revision
        or plan.supersedes_plan_id != record.supersedes_plan_id
        or record.schema_version != f"strategy.monitoring_plan.v{plan.plan_version}"
    ):
        raise StrategyMonitoringDataError(
            f"stored monitoring plan {record.id} columns do not match its payload"
        )
    return record


def _run_record_from_row(row: sqlite3.Row) -> MonitoringRunRecord:
    try:
        result = json.loads(str(row["result_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StrategyMonitoringDataError(
            f"stored monitoring run {row['id']} has invalid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise StrategyMonitoringDataError(
            f"stored monitoring run {row['id']} result is not an object"
        )
    result_hash = _required_sha256(row["result_hash"], field="stored result_hash")
    calculated_hash = hashlib.sha256(
        _canonical_json_object(result, field="stored result").encode("utf-8")
    ).hexdigest()
    if not hmac.compare_digest(result_hash, calculated_hash):
        raise StrategyMonitoringDataError(
            f"stored monitoring run {row['id']} result hash does not match"
        )
    validate_monitoring_run_result(result, overall_level=row["overall_level"])
    return MonitoringRunRecord(
        id=str(row["id"]),
        strategy_id=str(row["strategy_id"]),
        monitoring_plan_id=str(row["monitoring_plan_id"]),
        dataset_id=str(row["dataset_id"]),
        dataset_content_hash=_required_sha256(
            row["dataset_content_hash"], field="stored dataset_content_hash"
        ),
        strategy_effect_hash=_required_sha256(
            row["strategy_effect_hash"], field="stored strategy_effect_hash"
        ),
        economics_binding_hash=_required_sha256(
            row["economics_binding_hash"], field="stored economics_binding_hash"
        ),
        result=result,
        result_hash=result_hash,
        overall_level=str(row["overall_level"]),
        created_at=str(row["created_at"]),
    )


def _canonical_json_object(value: Mapping[str, Any], *, field: str) -> str:
    if not isinstance(value, Mapping):
        raise StrategyMonitoringDataError(f"{field} must be an object")
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StrategyMonitoringDataError(
            f"{field} must contain canonical JSON values"
        ) from exc


def _monitoring_level(value: object, *, field: str) -> str:
    if not isinstance(value, str) or value not in _LEVELS:
        raise StrategyMonitoringDataError(
            f"{field} must be one of: amber, green, n/a, red"
        )
    return value


def _overall_level(levels: list[str]) -> str:
    material_levels = {level for level in levels if level != "n/a"}
    if "red" in material_levels:
        return "red"
    if "amber" in material_levels:
        return "amber"
    if "green" in material_levels:
        return "green"
    return "n/a"


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StrategyMonitoringDataError(f"{field} must be a non-negative integer")
    return value


def _required_identifier(value: object, *, field: str) -> str:
    normalized = _optional_identifier(value, field=field)
    if normalized is None:
        raise StrategyMonitoringDataError(f"{field} must be non-empty")
    return normalized


def _optional_identifier(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise StrategyMonitoringDataError(f"{field} must be non-empty")
    return value.strip()


def _required_sha256(value: object, *, field: str) -> str:
    normalized = _optional_sha256(value, field=field)
    if normalized is None:
        raise StrategyMonitoringDataError(f"{field} must be a sha256 hash")
    return normalized


def _optional_sha256(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise StrategyMonitoringDataError(f"{field} must be a sha256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise StrategyMonitoringDataError(f"{field} must be a sha256 hash") from exc
    return value.lower()


__all__ = [
    "MonitoringPlanRecord",
    "MonitoringRunRecord",
    "StrategyMonitoringConflictError",
    "StrategyMonitoringDataError",
    "StrategyMonitoringDuplicateError",
    "StrategyMonitoringNotFoundError",
    "StrategyMonitoringRepository",
    "validate_monitoring_run_result",
]
