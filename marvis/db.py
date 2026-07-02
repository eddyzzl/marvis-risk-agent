from datetime import UTC, datetime

from marvis.db_schema import (
    _ensure_column as _ensure_column,
    connect as connect,
    init_db as init_db,
    sqlite_health as sqlite_health,
)
from marvis.repositories.audit import (
    _list_audit_rows as _list_audit_rows,
    _write_audit_row as _write_audit_row,
)
from marvis.repositories.datasets import DatasetRepository as DatasetRepository  # noqa: F401
from marvis.repositories.llm_calls import (
    llm_usage_summary as llm_usage_summary,
    record_llm_call as record_llm_call,
)
from marvis.repositories.drafts import DraftRepository as DraftRepository  # noqa: F401
from marvis.repositories.modeling import ModelingRepository as ModelingRepository  # noqa: F401
from marvis.repositories.plans import PlanRepository as PlanRepository  # noqa: F401
from marvis.repositories.plugins import PluginRepository as PluginRepository  # noqa: F401
from marvis.repositories.strategy import StrategyRepository as StrategyRepository  # noqa: F401
from marvis.repositories.tasks import (
    AGENT_REPORT_CONCLUSION_KEYS as AGENT_REPORT_CONCLUSION_KEYS,
    TaskRepository as TaskRepository,  # noqa: F401
    _normalize_task_type as _normalize_task_type,  # noqa: F401
)


def _now() -> str:
    return datetime.now(UTC).isoformat()
