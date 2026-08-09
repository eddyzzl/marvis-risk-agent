"""Application assembly for the local operations scheduler.

The runtime deliberately receives already-bound deterministic executors from
server code.  A schedule can select only an exact reference present when the
runtime was constructed; references are never interpreted as imports, module
paths, or user-supplied callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Callable, Mapping
import uuid

from marvis.operations.contracts import ScheduleContract
from marvis.operations.notifications import NotificationAdapter, NotificationRedactor
from marvis.operations.repository import (
    NotificationRecord,
    OperationsStore,
    PeriodRecord,
    ScheduleRecord,
)
from marvis.operations.scheduler import (
    LocalScheduler,
    MonitoringExecutor,
    ScheduledRunRequest,
    TickReport,
)
from marvis.settings import Settings


class UnknownMonitoringReference(ValueError):
    """A schedule named an executor not bound by trusted server assembly."""


@dataclass(frozen=True)
class RecoveryReport:
    """Result of an explicit, side-effect-free lease recovery pass."""

    period_leases_recovered: int
    notification_leases_recovered: int
    period_keys: tuple[str, ...]
    notification_ids: tuple[str, ...]


class OperationsRuntime:
    """Workspace-scoped operations store and synchronous scheduler."""

    def __init__(
        self,
        settings: Settings,
        *,
        executor_allowlist: Mapping[str, MonitoringExecutor],
        owner_id: str | None = None,
        clock: Callable[[], datetime] | None = None,
        notification_adapter: NotificationAdapter | None = None,
        notification_redactor: NotificationRedactor | None = None,
    ) -> None:
        if not isinstance(settings, Settings):
            raise ValueError("settings must be a Settings instance")
        if not isinstance(executor_allowlist, Mapping):
            raise ValueError("executor_allowlist must be a mapping")
        bound: dict[str, MonitoringExecutor] = {}
        for raw_ref, executor in executor_allowlist.items():
            ref = _monitoring_ref(raw_ref)
            if not callable(executor):
                raise ValueError(f"executor for {ref!r} must be callable")
            bound[ref] = executor
        resolved_owner = owner_id or f"marvis-operations-{uuid.uuid4().hex}"
        if not isinstance(resolved_owner, str) or not resolved_owner.strip():
            raise ValueError("owner_id must be non-empty text")

        self.settings = settings
        self.store = OperationsStore(settings.db_path, clock=clock)
        self._executors = MappingProxyType(bound)
        self._scheduler = LocalScheduler(
            self.store,
            executor=self._execute,
            owner_id=resolved_owner,
            clock=clock,
            notification_adapter=notification_adapter,
            notification_redactor=notification_redactor,
        )

    @property
    def allowed_monitoring_refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._executors))

    def publish_schedule(
        self,
        contract: ScheduleContract,
        *,
        expected_revision: int,
    ) -> ScheduleRecord:
        if not isinstance(contract, ScheduleContract):
            raise ValueError("contract must be a ScheduleContract")
        self._require_executor(contract.monitoring_ref)
        return self.store.publish_schedule(
            contract,
            expected_revision=expected_revision,
        )

    def tick(
        self,
        *,
        catch_up_budget: int,
        notification_budget: int = 0,
    ) -> TickReport:
        return self._scheduler.tick(
            catch_up_budget=catch_up_budget,
            notification_budget=notification_budget,
        )

    def recover(self) -> RecoveryReport:
        periods: tuple[PeriodRecord, ...] = self.store.recover_expired_leases()
        notifications: tuple[NotificationRecord, ...] = (
            self.store.recover_expired_notification_leases()
        )
        return RecoveryReport(
            period_leases_recovered=len(periods),
            notification_leases_recovered=len(notifications),
            period_keys=tuple(period.period.key for period in periods),
            notification_ids=tuple(
                notification.notification_id for notification in notifications
            ),
        )

    def _execute(self, request: ScheduledRunRequest):
        executor = self._require_executor(request.monitoring_ref)
        return executor(request)

    def _require_executor(self, monitoring_ref: str) -> MonitoringExecutor:
        ref = _monitoring_ref(monitoring_ref)
        try:
            return self._executors[ref]
        except KeyError as exc:
            raise UnknownMonitoringReference(
                f"monitoring_ref is not in the server allowlist: {ref}"
            ) from exc


def build_operations_runtime(
    settings: Settings,
    *,
    executor_allowlist: Mapping[str, MonitoringExecutor] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> OperationsRuntime:
    """Install operations on the workspace DB with no implicit side effects.

    Production monitoring bindings must be supplied by trusted server assembly.
    The default app runtime therefore starts with an empty allowlist and no
    notification adapter or background cadence.
    """

    return OperationsRuntime(
        settings,
        executor_allowlist=(
            {} if executor_allowlist is None else executor_allowlist
        ),
        clock=clock,
    )


def _monitoring_ref(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise ValueError("monitoring_ref must be non-empty text up to 200 characters")
    return value.strip()
