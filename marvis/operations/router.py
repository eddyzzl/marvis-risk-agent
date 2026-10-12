"""Local, production-role-gated operations API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from marvis.operations.contracts import (
    FixedIntervalCalendar,
    RetryPolicy,
    ScheduleContract,
)
from marvis.operations.integration import (
    OperationsRuntime,
    RecoveryReport,
    UnknownMonitoringReference,
)
from marvis.operations.repository import (
    EventRecord,
    NotificationRecord,
    OutcomeRecord,
    PeriodRecord,
    RunRecord,
    ScheduleRecord,
    ScheduleRevisionConflict,
)
from marvis.operations.scheduler import TickReport
from marvis.operations.bindings import MonitoringBinding
from marvis.operations.monitoring import read_monitoring_evidence
from marvis.operations.http_schemas import MonthlyCalendarPayload, SchedulePayload
from marvis.operations.month_calendar import CalendarMonthCalendar
from marvis.operations.diagnostics import period_diagnostic
from marvis.operations.rechecks import request_recheck
from marvis.operations.access import contract_access, visible_contract
from marvis.db_schema import connect
from marvis.production_governance.errors import GovernanceNotFound
from marvis.production_governance.repository import ProductionGovernanceRepository
from marvis.repositories.tasks import TaskRepository


router = APIRouter(prefix="/api/operations", tags=["operations"])


class PublishScheduleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_revision: StrictInt = Field(ge=0)
    schedule: SchedulePayload


class RecheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    period_key: str = Field(min_length=1, max_length=500)
    expected_revision: StrictInt = Field(ge=1)
    idempotency_key: str = Field(
        min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    monitoring_binding: MonitoringBinding


class TickRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catch_up_budget: StrictInt = Field(ge=1, le=100)
    notification_budget: StrictInt = Field(default=0, ge=0, le=100)


class DatasetSourceGcSweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: StrictInt = Field(default=100, ge=1, le=500)
    force: StrictBool = False


class TaskFilesystemGcSweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: StrictInt = Field(default=100, ge=1, le=500)
    force: StrictBool = False


class TaskFilesystemGcRequeueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: Literal[
        "task_dir",
        "dataset_identity_dir",
        "dataset_task_dir",
        "risk_intake_dir",
        "validation_batch_source_dir",
    ]
    relative_path: str = Field(min_length=1, max_length=1024)


def _runtime(request: Request) -> OperationsRuntime:
    runtime = getattr(request.app.state, "operations_runtime", None)
    if not isinstance(runtime, OperationsRuntime):
        raise HTTPException(status_code=503, detail="operations runtime is unavailable")
    return runtime


def _current_principal(request: Request) -> dict:
    local = getattr(request.state, "local_principal", None)
    principal_id = str(getattr(local, "id", "")).strip()
    if not principal_id:
        raise HTTPException(status_code=401, detail="server session principal required")
    try:
        return ProductionGovernanceRepository(
            request.app.state.settings.db_path
        ).get_principal(principal_id)
    except GovernanceNotFound as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _require_role(
    request: Request, *allowed: Literal["maker", "checker", "admin"]
) -> dict:
    principal = _current_principal(request)
    if principal.get("role") not in allowed:
        roles = ", ".join(allowed)
        raise HTTPException(status_code=403, detail=f"operation requires role: {roles}")
    return principal


def _dataset_source_gc(
    request: Request,
    *,
    method: Literal["status", "sweep"],
):
    collector = getattr(request.app.state, "dataset_source_gc", None)
    if not callable(getattr(collector, method, None)):
        raise HTTPException(
            status_code=503,
            detail="dataset source garbage collector is unavailable",
        )
    return collector


def _task_filesystem_gc(
    request: Request,
    *,
    method: Literal["status", "sweep", "requeue_quarantined"],
):
    collector = getattr(request.app.state, "task_filesystem_gc", None)
    if not callable(getattr(collector, method, None)):
        raise HTTPException(
            status_code=503,
            detail="task filesystem garbage collector is unavailable",
        )
    return collector


def _schedule_access(request, schedule_id, action="read", *, revision=None):
    record = _runtime(request).store.get_schedule(schedule_id, revision=revision)
    if record is None:
        raise HTTPException(404, detail="operations schedule not found")
    contract_access(request.app.state.settings.db_path, record.contract, action)
    return record


def _period_access(request, schedule_id, period_key, action="read"):
    period = _runtime(request).store.get_period(schedule_id, period_key)
    if period is None:
        raise HTTPException(404, detail="operations period not found")
    return _schedule_access(
        request, schedule_id, action, revision=period.schedule_revision
    )


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    _require_role(request, "maker", "checker", "admin")
    runtime = _runtime(request)
    return {
        "monitoring_refs": list(runtime.allowed_monitoring_refs),
        "monitoring_binding_schema": MonitoringBinding.model_json_schema(),
        "notification_target": "local_application_inbox",
        "runtime": runtime.loop.status() if runtime.loop else None,
    }


@router.get("/inbox")
def inbox(
    request: Request, limit: StrictInt = Query(default=100, ge=1, le=500)
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    adapter = _runtime(request).local_inbox
    items = [] if adapter is None else adapter.list(limit=limit)
    visible = []
    for item in items:
        period = _runtime(request).store.get_period(
            item["schedule_id"], item["period_key"]
        )
        record = (
            _runtime(request).store.get_schedule(
                item["schedule_id"], revision=period.schedule_revision
            )
            if period
            else None
        )
        if record and visible_contract(
            request.app.state.settings.db_path, record.contract
        ):
            visible.append(item)
    return {"notifications": visible}


@router.post("/inbox/{notification_id}/acknowledge")
def acknowledge_notification(notification_id: str, request: Request) -> dict:
    principal = _require_role(request, "maker", "checker", "admin")
    adapter = _runtime(request).local_inbox
    try:
        if adapter is None:
            raise KeyError(notification_id)
        with connect(request.app.state.settings.db_path) as conn:
            row = conn.execute(
                "SELECT o.schedule_id,o.period_key FROM operations_local_inbox i JOIN operations_notification_outbox o ON o.notification_id=i.notification_id WHERE i.notification_id=?",
                (notification_id,),
            ).fetchone()
        if row is None:
            raise KeyError(notification_id)
        _period_access(request, row["schedule_id"], row["period_key"])
        adapter.acknowledge(notification_id, str(principal["id"]))
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="inbox notification not found"
        ) from exc
    return {"acknowledged": True, "automatic_action_permitted": False}


@router.get("/schedules/{schedule_id}/evidence")
def monitoring_evidence(schedule_id: str, period_key: str, request: Request) -> dict:
    _require_role(request, "maker", "checker", "admin")
    _period_access(request, schedule_id, period_key)
    outcome = _runtime(request).store.get_outcome(schedule_id, period_key)
    if outcome is None:
        raise HTTPException(status_code=404, detail="monitoring outcome not found")
    try:
        return read_monitoring_evidence(
            request.app.state.settings.workspace, outcome.outcome
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=409, detail="monitoring evidence unavailable or changed"
        ) from exc


@router.post("/schedules", status_code=201)
def publish_schedule(payload: PublishScheduleRequest, request: Request) -> dict:
    from marvis.institution.policy import require_human_principal
    from dataclasses import replace

    require_human_principal(request)
    principal = _require_role(request, "maker", "admin")
    try:
        contract = replace(
            _schedule_contract(payload.schedule),
            execution_actor_id=str(principal["id"]),
        )
        contract_access(
            request.app.state.settings.db_path,
            contract,
            "execute",
            actor_id=principal["id"],
        )
        old = _runtime(request).store.get_schedule(contract.schedule_id)
        if old:
            from marvis.operations.schedule_proposals import schedule_task

            if any(
                (
                    old.contract.monitoring_binding,
                    old.contract.portfolio_binding,
                    old.contract.feedback_binding,
                )
            ) and schedule_task(old.contract) != schedule_task(contract):
                raise ScheduleRevisionConflict(
                    "schedule task identity cannot be rebound"
                )
            contract_access(
                request.app.state.settings.db_path,
                old.contract,
                "execute",
                actor_id=principal["id"],
            )
        record = _runtime(request).publish_schedule(
            contract,
            expected_revision=payload.expected_revision,
        )
    except ScheduleRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (UnknownMonitoringReference, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _schedule_record(record)


@router.get("/schedules")
def list_active_schedules(request: Request, include_disabled: bool = False) -> dict:
    _require_role(request, "maker", "checker", "admin")
    store = _runtime(request).store
    records = (
        store.list_schedules() if include_disabled else store.list_active_schedules()
    )
    schedules = [
        _schedule_record(record)
        for record in records
        if visible_contract(request.app.state.settings.db_path, record.contract)
    ]
    return {"schedules": schedules, "count": len(schedules)}


@router.get("/schedules/{schedule_id}")
def get_schedule(
    schedule_id: str,
    request: Request,
    revision: StrictInt | None = Query(default=None, ge=1),
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    try:
        record = _runtime(request).store.get_schedule(schedule_id, revision=revision)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="operations schedule not found")
    contract_access(request.app.state.settings.db_path, record.contract)
    return _schedule_record(record)


@router.get("/schedules/{schedule_id}/revisions")
def list_schedule_revisions(schedule_id: str, request: Request) -> dict:
    _require_role(request, "maker", "checker", "admin")
    try:
        records = _runtime(request).store.list_schedule_revisions(schedule_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "schedules": [
            _schedule_record(record)
            for record in records
            if visible_contract(request.app.state.settings.db_path, record.contract)
        ]
    }


@router.get("/schedules/{schedule_id}/periods")
def list_periods(
    schedule_id: str,
    request: Request,
    limit: StrictInt = Query(default=100, ge=1, le=500),
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    _schedule_access(request, schedule_id)
    store = _runtime(request).store
    periods = []
    for period in store.list_periods(schedule_id, limit=limit):
        record = store.get_schedule(schedule_id, revision=period.schedule_revision)
        if not record or not visible_contract(
            request.app.state.settings.db_path, record.contract
        ):
            continue
        outcome = store.get_outcome(schedule_id, period.period.key)
        events = store.list_events(schedule_id, period.period.key)
        periods.append(
            {
                "period": _period_record(period),
                "diagnostic": period_diagnostic(period, outcome, events),
            }
        )
    return {"periods": periods}


@router.post("/schedules/{schedule_id}/rechecks", status_code=201)
def recheck(schedule_id: str, payload: RecheckRequest, request: Request) -> dict:
    from marvis.institution.policy import require_human_principal
    from marvis.operations.schedule_proposals import schedule_task

    require_human_principal(request)
    _require_role(request, "maker", "admin")
    source = _period_access(request, schedule_id, payload.period_key, "execute")
    if schedule_task(source.contract) != payload.monitoring_binding.task_id:
        raise HTTPException(422, detail="recheck task owner differs")
    try:
        record = request_recheck(
            _runtime(request),
            schedule_id,
            period_key=payload.period_key,
            expected_revision=payload.expected_revision,
            idempotency_key=payload.idempotency_key,
            binding=payload.monitoring_binding,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="source schedule or period not found"
        ) from exc
    except ScheduleRevisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _schedule_record(record)


@router.get("/schedules/{schedule_id}/period")
def get_period(
    schedule_id: str,
    request: Request,
    period_key: str = Query(min_length=1, max_length=500),
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    _period_access(request, schedule_id, period_key)
    store = _runtime(request).store
    try:
        period = store.get_period(schedule_id, period_key)
        if period is None:
            raise HTTPException(status_code=404, detail="operations period not found")
        outcome = store.get_outcome(schedule_id, period_key)
        runs = store.list_runs(schedule_id, period_key)
        events = store.list_events(schedule_id, period_key)
        notifications = store.list_notifications(schedule_id, period_key)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "period": _period_record(period),
        "diagnostic": period_diagnostic(period, outcome, events),
        "outcome": None if outcome is None else _outcome_record(outcome),
        "runs": [_run_record(record) for record in runs],
        "events": [_event_record(record) for record in events],
        "notifications": [_notification_record(record) for record in notifications],
    }


@router.post("/tick")
def tick(payload: TickRequest, request: Request) -> dict:
    _require_role(request, "admin")
    try:
        report = _runtime(request).tick(
            catch_up_budget=payload.catch_up_budget,
            notification_budget=payload.notification_budget,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _tick_report(report)


@router.post("/recover")
def recover(request: Request) -> dict:
    _require_role(request, "admin")
    return _recovery_report(_runtime(request).recover())


@router.get("/dataset-source-gc")
def dataset_source_gc_status(
    request: Request,
    limit: StrictInt = Query(default=500, ge=1, le=5000),
    offset: StrictInt = Query(default=0, ge=0),
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    try:
        status = _dataset_source_gc(request, method="status").status(
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total_count = status.pending_count + status.quarantined_count
    entries = [_dataset_source_gc_entry(entry) for entry in status.entries]
    return {
        "pending_count": status.pending_count,
        "quarantined_count": status.quarantined_count,
        "offset": offset,
        "returned_count": len(entries),
        "has_more": offset + len(entries) < total_count,
        "entries": entries,
        "watchdog": _dataset_source_gc_watchdog_status(request),
        "startup": {
            "report": _gc_sweep_report(
                request.app.state.dataset_source_gc_startup_report
            ),
            "error": getattr(
                request.app.state,
                "dataset_source_gc_startup_error",
                None,
            ),
        },
    }


@router.post("/dataset-source-gc/sweep")
def sweep_dataset_source_gc(
    payload: DatasetSourceGcSweepRequest,
    request: Request,
) -> dict:
    principal = _require_role(request, "admin")
    audit_repo = TaskRepository(request.app.state.settings.db_path)
    audit_detail = {"limit": payload.limit, "force": payload.force}
    audit_repo.write_audit(
        kind="dataset_source_gc.manual_sweep",
        target_ref="dataset-source-gc",
        actor=str(principal["id"]),
        outcome="started",
        detail=audit_detail,
    )
    try:
        report = _dataset_source_gc(request, method="sweep").sweep(
            limit=payload.limit,
            force=payload.force,
        )
    except ValueError as exc:
        audit_repo.write_audit(
            kind="dataset_source_gc.manual_sweep",
            target_ref="dataset-source-gc",
            actor=str(principal["id"]),
            outcome="rejected",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        audit_repo.write_audit(
            kind="dataset_source_gc.manual_sweep",
            target_ref="dataset-source-gc",
            actor=str(principal["id"]),
            outcome="failed",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="dataset source garbage collection failed",
        ) from exc
    report_payload = _gc_sweep_report(report)
    audit_repo.write_audit(
        kind="dataset_source_gc.manual_sweep",
        target_ref="dataset-source-gc",
        actor=str(principal["id"]),
        outcome="succeeded",
        detail={**audit_detail, "report": report_payload},
    )
    return report_payload


@router.get("/task-filesystem-gc")
def task_filesystem_gc_status(
    request: Request,
    limit: StrictInt = Query(default=500, ge=1, le=5000),
    offset: StrictInt = Query(default=0, ge=0),
) -> dict:
    _require_role(request, "maker", "checker", "admin")
    try:
        status = _task_filesystem_gc(request, method="status").status(
            limit=limit,
            offset=offset,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    total_count = status.pending_count + status.quarantined_count
    entries = [_task_filesystem_gc_entry(entry) for entry in status.entries]
    return {
        "pending_count": status.pending_count,
        "quarantined_count": status.quarantined_count,
        "offset": offset,
        "returned_count": len(entries),
        "has_more": offset + len(entries) < total_count,
        "entries": entries,
        "watchdog": _task_filesystem_gc_watchdog_status(request),
        "startup": {
            "report": _gc_sweep_report(
                request.app.state.task_filesystem_gc_startup_report
            ),
            "error": getattr(
                request.app.state,
                "task_filesystem_gc_startup_error",
                None,
            ),
        },
    }


@router.post("/task-filesystem-gc/sweep")
def sweep_task_filesystem_gc(
    payload: TaskFilesystemGcSweepRequest,
    request: Request,
) -> dict:
    principal = _require_role(request, "admin")
    audit_repo = TaskRepository(request.app.state.settings.db_path)
    audit_detail = {"limit": payload.limit, "force": payload.force}
    audit_repo.write_audit(
        kind="task_filesystem_gc.manual_sweep",
        target_ref="task-filesystem-gc",
        actor=str(principal["id"]),
        outcome="started",
        detail=audit_detail,
    )
    try:
        report = _task_filesystem_gc(request, method="sweep").sweep(
            limit=payload.limit,
            force=payload.force,
        )
    except ValueError as exc:
        audit_repo.write_audit(
            kind="task_filesystem_gc.manual_sweep",
            target_ref="task-filesystem-gc",
            actor=str(principal["id"]),
            outcome="rejected",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        audit_repo.write_audit(
            kind="task_filesystem_gc.manual_sweep",
            target_ref="task-filesystem-gc",
            actor=str(principal["id"]),
            outcome="failed",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="task filesystem garbage collection failed",
        ) from exc
    report_payload = _gc_sweep_report(report)
    audit_repo.write_audit(
        kind="task_filesystem_gc.manual_sweep",
        target_ref="task-filesystem-gc",
        actor=str(principal["id"]),
        outcome="succeeded",
        detail={**audit_detail, "report": report_payload},
    )
    return report_payload


@router.post("/task-filesystem-gc/requeue")
def requeue_task_filesystem_gc(
    payload: TaskFilesystemGcRequeueRequest,
    request: Request,
) -> dict:
    """Explicitly release one still-safe quarantine after operator review."""

    principal = _require_role(request, "admin")
    audit_repo = TaskRepository(request.app.state.settings.db_path)
    audit_detail = {
        "target_type": payload.target_type,
        "relative_path": payload.relative_path,
    }
    target_ref = f"task-filesystem-gc:{payload.target_type}:{payload.relative_path}"
    audit_repo.write_audit(
        kind="task_filesystem_gc.manual_requeue",
        target_ref=target_ref,
        actor=str(principal["id"]),
        outcome="started",
        detail=audit_detail,
    )
    collector = _task_filesystem_gc(request, method="requeue_quarantined")
    try:
        requeued = collector.requeue_quarantined(
            payload.target_type,
            payload.relative_path,
        )
    except ValueError as exc:
        audit_repo.write_audit(
            kind="task_filesystem_gc.manual_requeue",
            target_ref=target_ref,
            actor=str(principal["id"]),
            outcome="rejected",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        audit_repo.write_audit(
            kind="task_filesystem_gc.manual_requeue",
            target_ref=target_ref,
            actor=str(principal["id"]),
            outcome="failed",
            detail={**audit_detail, "error": exc.__class__.__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="task filesystem quarantine requeue failed",
        ) from exc
    if not requeued:
        audit_repo.write_audit(
            kind="task_filesystem_gc.manual_requeue",
            target_ref=target_ref,
            actor=str(principal["id"]),
            outcome="rejected",
            detail={**audit_detail, "error": "not_quarantined"},
        )
        raise HTTPException(
            status_code=409,
            detail="task filesystem target is not quarantined",
        )
    audit_repo.write_audit(
        kind="task_filesystem_gc.manual_requeue",
        target_ref=target_ref,
        actor=str(principal["id"]),
        outcome="succeeded",
        detail={**audit_detail, "state": "pending"},
    )
    return {**audit_detail, "state": "pending"}


def _schedule_contract(payload: SchedulePayload) -> ScheduleContract:
    return ScheduleContract(
        schedule_id=payload.schedule_id,
        revision=payload.revision,
        monitoring_ref=payload.monitoring_ref,
        active_from=payload.active_from,
        calendar=CalendarMonthCalendar(
            payload.calendar.anchor_at,
            payload.calendar.interval_months,
            payload.calendar.timezone,
        )
        if isinstance(payload.calendar, MonthlyCalendarPayload)
        else FixedIntervalCalendar(
            anchor_at=payload.calendar.anchor_at,
            interval_seconds=payload.calendar.interval_seconds,
        ),
        catch_up_budget=payload.catch_up_budget,
        lease_seconds=payload.lease_seconds,
        retry_policy=RetryPolicy(
            max_attempts=payload.retry_policy.max_attempts,
            initial_backoff_seconds=payload.retry_policy.initial_backoff_seconds,
            multiplier=payload.retry_policy.multiplier,
            max_backoff_seconds=payload.retry_policy.max_backoff_seconds,
        ),
        enabled=payload.enabled,
        monitoring_binding=payload.monitoring_binding,
        portfolio_binding=payload.portfolio_binding,
        feedback_binding=payload.feedback_binding,
        active_until=payload.active_until,
    )


def _schedule_record(record: ScheduleRecord) -> dict:
    return {
        "contract": record.contract.to_dict(),
        "contract_hash": record.contract_hash,
        "created_at": _iso(record.created_at),
    }


def _period_record(record: PeriodRecord) -> dict:
    return {
        "schedule_id": record.schedule_id,
        "period": {
            "key": record.period.key,
            "index": record.period.index,
            "starts_at": _iso(record.period.starts_at),
            "ends_at": _iso(record.period.ends_at),
        },
        "schedule_revision": record.schedule_revision,
        "state": record.state,
        "attempt_count": record.attempt_count,
        "lease_owner": record.lease_owner,
        "lease_expires_at": _optional_iso(record.lease_expires_at),
        "next_attempt_at": _optional_iso(record.next_attempt_at),
        "outcome_id": record.outcome_id,
        "updated_at": _iso(record.updated_at),
    }


def _outcome_record(record: OutcomeRecord) -> dict:
    return {
        "outcome_id": record.outcome_id,
        "run_id": record.run_id,
        "schedule_id": record.schedule_id,
        "period_key": record.period_key,
        "outcome": record.outcome.to_dict(),
        "outcome_hash": record.outcome_hash,
        "recorded_at": _iso(record.recorded_at),
    }


def _run_record(record: RunRecord) -> dict:
    return {
        "run_id": record.run_id,
        "schedule_id": record.schedule_id,
        "period_key": record.period_key,
        "schedule_revision": record.schedule_revision,
        "attempt_number": record.attempt_number,
        "owner_id": record.owner_id,
        "claimed_at": _iso(record.claimed_at),
        "lease_expires_at": _iso(record.lease_expires_at),
    }


def _event_record(record: EventRecord) -> dict:
    return {
        "sequence": record.sequence,
        "event_id": record.event_id,
        "schedule_id": record.schedule_id,
        "period_key": record.period_key,
        "run_id": record.run_id,
        "event_type": record.event_type,
        "payload": record.payload,
        "created_at": _iso(record.created_at),
    }


def _notification_record(record: NotificationRecord) -> dict:
    return {
        "notification_id": record.notification_id,
        "outcome_id": record.outcome_id,
        "schedule_id": record.schedule_id,
        "period_key": record.period_key,
        "run_id": record.run_id,
        "notification": record.notification.to_dict(),
        "payload_hash": record.payload_hash,
        "status": record.status,
        "attempt_count": record.attempt_count,
        "max_attempts": record.max_attempts,
        "next_attempt_at": _optional_iso(record.next_attempt_at),
        "lease_owner": record.lease_owner,
        "lease_expires_at": _optional_iso(record.lease_expires_at),
        "created_at": _iso(record.created_at),
        "updated_at": _iso(record.updated_at),
    }


def _tick_report(report: TickReport) -> dict:
    return {
        "attempted": report.attempted,
        "succeeded": report.succeeded,
        "failed": report.failed,
        "recovered": report.recovered,
        "claimed_period_keys": list(report.claimed_period_keys),
        "notifications_attempted": report.notifications_attempted,
        "notifications_sent": report.notifications_sent,
        "notifications_failed": report.notifications_failed,
        "notifications_recovered": report.notifications_recovered,
    }


def _recovery_report(report: RecoveryReport) -> dict:
    return {
        "period_leases_recovered": report.period_leases_recovered,
        "notification_leases_recovered": report.notification_leases_recovered,
        "period_keys": list(report.period_keys),
        "notification_ids": list(report.notification_ids),
    }


def _dataset_source_gc_entry(entry) -> dict:
    return {
        "source_path": entry.source_path,
        "state": entry.state,
        "attempt_count": entry.attempt_count,
        "next_attempt_at": _optional_utc_iso(entry.next_attempt_at),
        "last_error_code": entry.last_error_code,
        "last_error_message": entry.last_error_message,
        "origin_task_id": entry.origin_task_id,
        "enqueued_at": _optional_utc_iso(entry.enqueued_at),
        "updated_at": _optional_utc_iso(entry.updated_at),
    }


def _gc_sweep_report(report) -> dict:
    return {
        "examined": report.examined,
        "deleted": report.deleted,
        "cancelled_referenced": report.cancelled_referenced,
        "deferred": report.deferred,
        "quarantined": report.quarantined,
    }


def _dataset_source_gc_watchdog_status(request: Request) -> dict:
    watchdog = getattr(request.app.state, "dataset_source_gc_watchdog", None)
    if not callable(getattr(watchdog, "status", None)):
        return {
            "alive": False,
            "last_started_at": None,
            "last_finished_at": None,
            "last_report": None,
            "last_error": "watchdog unavailable",
        }
    status = watchdog.status()
    return {
        "alive": status.alive,
        "last_started_at": _optional_utc_iso(status.last_started_at),
        "last_finished_at": _optional_utc_iso(status.last_finished_at),
        "last_report": (
            None if status.last_report is None else _gc_sweep_report(status.last_report)
        ),
        "last_error": status.last_error,
    }


def _task_filesystem_gc_entry(entry) -> dict:
    return {
        "target_type": entry.target_type,
        "relative_path": entry.relative_path,
        "state": entry.state,
        "attempt_count": entry.attempt_count,
        "next_attempt_at": _optional_utc_iso(entry.next_attempt_at),
        "last_error_code": entry.last_error_code,
        "last_error_message": entry.last_error_message,
        "origin_task_id": entry.origin_task_id,
        "origin_task_created_at": entry.origin_task_created_at,
        "enqueued_at": _optional_utc_iso(entry.enqueued_at),
        "updated_at": _optional_utc_iso(entry.updated_at),
    }


def _task_filesystem_gc_watchdog_status(request: Request) -> dict:
    watchdog = getattr(request.app.state, "task_filesystem_gc_watchdog", None)
    if not callable(getattr(watchdog, "status", None)):
        return {
            "alive": False,
            "last_started_at": None,
            "last_finished_at": None,
            "last_report": None,
            "last_error": "watchdog unavailable",
        }
    status = watchdog.status()
    return {
        "alive": status.alive,
        "last_started_at": _optional_utc_iso(status.last_started_at),
        "last_finished_at": _optional_utc_iso(status.last_finished_at),
        "last_report": (
            None if status.last_report is None else _gc_sweep_report(status.last_report)
        ),
        "last_error": status.last_error,
    }


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _optional_utc_iso(value: datetime | None) -> str | None:
    return None if value is None else _utc_iso(value)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _optional_iso(value: datetime | None) -> str | None:
    return None if value is None else _iso(value)


def _mount_extensions():
    from marvis.operations.extensions_router import router as extensions_router

    router.include_router(extensions_router)


_mount_extensions()
