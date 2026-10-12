"""Public local scheduler contracts with lazy runtime imports.

Schema migrations can import the leaf DDL without loading analysis, strategy,
model delivery, or application executors. Existing public imports stay intact.
"""

from importlib import import_module

_EXPORTS = {
    "CALENDAR_SCHEMA_VERSION": "marvis.operations.contracts",
    "HUMAN_ESCALATION_SCHEMA_VERSION": "marvis.operations.contracts",
    "MAX_CATCH_UP_BUDGET": "marvis.operations.contracts",
    "MAX_RETRY_ATTEMPTS": "marvis.operations.contracts",
    "MONITORING_OUTCOME_SCHEMA_VERSION": "marvis.operations.contracts",
    "RETRY_SCHEMA_VERSION": "marvis.operations.contracts",
    "SCHEDULE_SCHEMA_VERSION": "marvis.operations.contracts",
    "CalendarPeriod": "marvis.operations.contracts",
    "FixedIntervalCalendar": "marvis.operations.contracts",
    "HumanEscalationSuggestion": "marvis.operations.contracts",
    "MonitoringOutcome": "marvis.operations.contracts",
    "RetryPolicy": "marvis.operations.contracts",
    "ScheduleContract": "marvis.operations.contracts",
    "EventRecord": "marvis.operations.repository",
    "NotificationClaim": "marvis.operations.repository",
    "NotificationRecord": "marvis.operations.repository",
    "OutcomeRecord": "marvis.operations.repository",
    "OperationsStore": "marvis.operations.repository",
    "PeriodRecord": "marvis.operations.repository",
    "RunClaim": "marvis.operations.repository",
    "RunRecord": "marvis.operations.repository",
    "ScheduleRecord": "marvis.operations.repository",
    "ScheduleRevisionConflict": "marvis.operations.repository",
    "StaleNotificationLeaseError": "marvis.operations.repository",
    "StaleRunLeaseError": "marvis.operations.repository",
    "REDACTED_NOTIFICATION_SCHEMA_VERSION": "marvis.operations.notifications",
    "NotificationAdapter": "marvis.operations.notifications",
    "NotificationRedactor": "marvis.operations.notifications",
    "RedactedNotification": "marvis.operations.notifications",
    "OPERATIONS_SCHEMA_VERSION": "marvis.operations.schema",
    "OperationsSchemaVersionError": "marvis.operations.schema",
    "install_operations_schema": "marvis.operations.schema",
    "LocalScheduler": "marvis.operations.scheduler",
    "MonitoringExecutor": "marvis.operations.scheduler",
    "ScheduledRunRequest": "marvis.operations.scheduler",
    "TickReport": "marvis.operations.scheduler",
    "OperationsRuntime": "marvis.operations.integration",
    "RecoveryReport": "marvis.operations.integration",
    "UnknownMonitoringReference": "marvis.operations.integration",
    "build_operations_runtime": "marvis.operations.integration",
}

__all__ = [
    "CALENDAR_SCHEMA_VERSION",
    "HUMAN_ESCALATION_SCHEMA_VERSION",
    "MAX_CATCH_UP_BUDGET",
    "MAX_RETRY_ATTEMPTS",
    "MONITORING_OUTCOME_SCHEMA_VERSION",
    "RETRY_SCHEMA_VERSION",
    "SCHEDULE_SCHEMA_VERSION",
    "CalendarPeriod",
    "FixedIntervalCalendar",
    "HumanEscalationSuggestion",
    "MonitoringOutcome",
    "MonitoringExecutor",
    "NotificationAdapter",
    "NotificationClaim",
    "NotificationRecord",
    "NotificationRedactor",
    "EventRecord",
    "OutcomeRecord",
    "OPERATIONS_SCHEMA_VERSION",
    "OperationsSchemaVersionError",
    "OperationsStore",
    "OperationsRuntime",
    "PeriodRecord",
    "REDACTED_NOTIFICATION_SCHEMA_VERSION",
    "RecoveryReport",
    "RedactedNotification",
    "RetryPolicy",
    "RunClaim",
    "RunRecord",
    "LocalScheduler",
    "ScheduleContract",
    "ScheduleRecord",
    "ScheduleRevisionConflict",
    "StaleNotificationLeaseError",
    "StaleRunLeaseError",
    "ScheduledRunRequest",
    "TickReport",
    "UnknownMonitoringReference",
    "build_operations_runtime",
    "install_operations_schema",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
