"""Compatibility facade over ``marvis.db_schema`` and ``marvis.repositories``.

Every name here is a re-export that tests or callers still import from
``marvis.db``; ownership, users and removal conditions are registered in
``docs/compat-symbols.md``. New code imports from the defining module.
"""
from marvis.db_schema import (
    SCHEMA_VERSION as SCHEMA_VERSION,
    _ensure_column as _ensure_column,
    connect as connect,
    init_db as init_db,
    sqlite_health as sqlite_health,
)
from marvis.repositories.audit import (
    _list_audit_rows as _list_audit_rows,
)
from marvis.repositories.datasets import DatasetRepository as DatasetRepository  # noqa: F401
from marvis.repositories.data_workspace import (  # noqa: F401
    DataWorkspaceRepository as DataWorkspaceRepository,
)
from marvis.repositories.llm_calls import (
    llm_usage_summary as llm_usage_summary,
    record_llm_call as record_llm_call,
)
from marvis.repositories.drafts import DraftRepository as DraftRepository  # noqa: F401
from marvis.repositories.modeling import ModelingRepository as ModelingRepository  # noqa: F401
from marvis.repositories.plans import PlanRepository as PlanRepository  # noqa: F401
from marvis.repositories.plugins import PluginRepository as PluginRepository  # noqa: F401
from marvis.repositories.strategy import StrategyRepository as StrategyRepository  # noqa: F401
from marvis.repositories.task_artifacts import (
    TaskArtifactRepository as TaskArtifactRepository,  # noqa: F401
)
from marvis.repositories.tasks import (
    TaskRepository as TaskRepository,  # noqa: F401
    _normalize_task_type as _normalize_task_type,  # noqa: F401
)
from marvis.repositories.validation_contracts import (
    ValidationContractRepository as ValidationContractRepository,  # noqa: F401
)
