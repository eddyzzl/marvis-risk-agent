"""Contracts for governed Agent turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from marvis.domain import TASK_TYPE_DATA_JOIN
from marvis.domain import TASK_TYPE_FEATURE_ANALYSIS
from marvis.domain import TASK_TYPE_MODELING
from marvis.domain import TASK_TYPE_PORTFOLIO
from marvis.domain import TASK_TYPE_STRATEGY
from marvis.domain import TASK_TYPE_VINTAGE
from marvis.domain import TaskRecord
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.orchestrator.executor import PlanExecutor
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.validator import PlanValidator
from marvis.repositories.plans import PlanRepository
from marvis.repositories.tasks import TaskRepository
from marvis.settings import Settings
from typing import Callable


DRIVER_AGENT_TASK_TYPES = frozenset(
    {
        TASK_TYPE_DATA_JOIN,
        TASK_TYPE_FEATURE_ANALYSIS,
        TASK_TYPE_MODELING,
        TASK_TYPE_STRATEGY,
        TASK_TYPE_VINTAGE,
        TASK_TYPE_PORTFOLIO,
    }
)

AGENT_MAX_GATES = 8

_TERMINAL_PLAN_STATUS_VALUES = frozenset({"done", "failed", "cancelled"})


@dataclass(frozen=True)
class DriverTurnRuntime:
    settings: Settings
    plan_repo: PlanRepository
    plan_executor: PlanExecutor
    planner: Planner
    plan_validator: PlanValidator
    llm_client: OpenAICompatibleLLMClient | None
    tier: str
    allow_manual_gate_adapters: bool = True
    require_semantic_text_authorization: bool = False
    ui_action: str | None = None
    semantic_intent: str | None = None
    workflow_intake_route: Mapping[str, object] | None = None
    governance_service: object | None = None
    local_principal: object | None = None
    recovery_responder: Callable[..., tuple[str, dict]] | None = None
    hook_dispatcher: object | None = None
    cancellation_check: Callable[[], None] | None = None


@dataclass(frozen=True)
class _TurnHandlerSpec:
    # Metadata `intent` tag stamped on the logged user-turn message.
    intent: str
    # Exception type(s) from this type's *_setup module that map to a plain
    # chat error message (as opposed to DriverError, which always re-raises).
    setup_error_types: tuple[type[Exception], ...]
    # Human label used in the generic `except Exception` fallback message,
    # e.g. "数据拼接出错：{exc}".
    error_label: str
    # Setup callback run only when there is no active plan for the task. It
    # performs this type's proposal-building (and, for join/modeling, the C1
    # file-role gate sub-flow) and returns either:
    #   - a dict: an early-exit turn response (a gate pause, a skip
    #     confirmation, or a setup error) that should be returned as-is; or
    #   - a tuple (template_id, slots, start_kwargs): the driver.start(...)
    #     call to make once the pre-start assistant message has already been
    #     appended by the callback itself.
    run_setup: Callable[
        [DriverTurnRuntime, TaskRepository, TaskRecord, str | None, str],
        dict | tuple,
    ]
    # join/modeling display "已确认文件角色与目标列。" instead of the raw
    # [C1]-prefixed payload text when logging the user turn; the other three
    # types always log user_text verbatim.
    format_user_display: Callable[[str], str]
    # Optional per-type success_criteria builder threaded into start_kwargs
    # (mirrors _modeling_success_criteria); None means this type never injects
    # a deterministic criterion.
    success_criteria: Callable[[TaskRecord], list[dict] | None] | None = None


_SPECIALIZED_WORKFLOW_INTAKE_TYPES = frozenset(
    {TASK_TYPE_MODELING, TASK_TYPE_FEATURE_ANALYSIS}
)

_MODELING_INTAKE_PARAM_NAMES = frozenset(
    {
        "target_type",
        "recipes",
        "split_config",
        "n_trials",
        "sample_weight_col",
    }
)
