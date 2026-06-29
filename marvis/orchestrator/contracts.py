from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from marvis.plugins.manifest import ToolRef


class PlanStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    CONFIRMED = "confirmed"
    RUNNING = "running"
    AWAITING_CONFIRM = "awaiting_confirm"
    REVIEW = "review"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    BLOCKED = "blocked"
    AWAITING_CONFIRM = "awaiting_confirm"
    RUNNING = "running"
    CHECKING = "checking"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentStatus(str, Enum):
    SPAWNED = "spawned"
    RUNNING = "running"
    RETURNED = "returned"
    FAILED = "failed"
    KILLED = "killed"


@dataclass(frozen=True)
class PostCheck:
    kind: str
    spec: dict[str, Any]


@dataclass(frozen=True)
class ReviewVerdict:
    reviewer: str
    passed: bool
    reasons: list[str]
    at: str


@dataclass
class PlanStep:
    id: str
    plan_id: str
    index: int
    title: str
    tool_ref: ToolRef
    inputs: dict[str, Any]
    depends_on: list[str]
    post_checks: list[PostCheck]
    needs_confirmation: bool = False
    decision_point: bool = False
    sub_agent_scope: str | None = None
    granted_tools: list[ToolRef] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    sub_agent_id: str | None = None
    output_ref: str | None = None
    review_verdicts: list[ReviewVerdict] = field(default_factory=list)
    error: str | None = None
    # Display-only grouping label (e.g. "数据准备"/"特征"/"建模"/"报告"); does not
    # affect execution semantics. Populated by templates so the right-rail can
    # fold a flat step DAG into big-step phases. None for ungrouped steps.
    phase: str | None = None


@dataclass
class Plan:
    id: str
    task_id: str
    goal: str
    source: str
    template_id: str | None
    steps: list[PlanStep]
    autonomy_level: int
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str = ""
    updated_at: str = ""
    novel_mode: str = "plan_ahead"
    tier: str = "balanced"
    replan_count: int = 0
    loop_events: list["LoopEvent"] = field(default_factory=list)
    success_criteria: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class LoopEvent:
    type: str
    reason: str
    at: str
    trigger_step_id: str | None = None
    instruction: str | None = None  # free-text user constraint for user_instruction replans
    tool_ref: str | None = None


@dataclass
class ExploreCursor:
    plan_id: str
    segment_index: int
    open_goal: str
    done: bool


@dataclass
class SubAgent:
    id: str
    parent_task_id: str
    parent_step_id: str | None
    scope: str
    granted_tools: list[ToolRef]
    context_budget: int
    status: AgentStatus = AgentStatus.SPAWNED
    result_ref: str | None = None


@dataclass(frozen=True)
class OutputRef:
    kind: str
    value: str


OUTPUT_REF_KINDS = frozenset({"dataset", "metrics", "artifact", "value"})


def parse_output_ref(raw: str) -> OutputRef:
    if not isinstance(raw, str) or ":" not in raw:
        raise ValueError("output_ref must use '<kind>:<value>'")
    kind, value = raw.split(":", 1)
    if kind not in OUTPUT_REF_KINDS or not value:
        raise ValueError("output_ref kind or value is invalid")
    return OutputRef(kind=kind, value=value)


def format_output_ref(kind: str, value: str) -> str:
    return f"{parse_output_ref(f'{kind}:{value}').kind}:{value}"


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "id": plan.id,
        "task_id": plan.task_id,
        "goal": plan.goal,
        "source": plan.source,
        "template_id": plan.template_id,
        "steps": [_step_to_dict(step) for step in plan.steps],
        "autonomy_level": plan.autonomy_level,
        "status": plan.status.value,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
        "novel_mode": plan.novel_mode,
        "tier": plan.tier,
        "replan_count": plan.replan_count,
        "loop_events": [_loop_event_to_dict(event) for event in plan.loop_events],
        "success_criteria": [dict(item) for item in plan.success_criteria],
    }


def plan_from_dict(payload: dict[str, Any]) -> Plan:
    return Plan(
        id=str(payload["id"]),
        task_id=str(payload["task_id"]),
        goal=str(payload["goal"]),
        source=str(payload["source"]),
        template_id=_optional_str(payload.get("template_id")),
        steps=[_step_from_dict(item) for item in payload.get("steps") or []],
        autonomy_level=int(payload["autonomy_level"]),
        status=PlanStatus(payload.get("status", PlanStatus.DRAFT.value)),
        created_at=str(payload.get("created_at") or ""),
        updated_at=str(payload.get("updated_at") or ""),
        novel_mode=str(payload.get("novel_mode") or "plan_ahead"),
        tier=str(payload.get("tier") or "balanced"),
        replan_count=int(payload.get("replan_count") or 0),
        loop_events=[
            _loop_event_from_dict(item)
            for item in payload.get("loop_events") or []
        ],
        success_criteria=[
            dict(item)
            for item in payload.get("success_criteria") or []
            if isinstance(item, dict)
        ],
    )


def _step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "id": step.id,
        "plan_id": step.plan_id,
        "index": step.index,
        "title": step.title,
        "tool_ref": _tool_ref_to_dict(step.tool_ref),
        "inputs": step.inputs,
        "depends_on": list(step.depends_on),
        "post_checks": [_post_check_to_dict(check) for check in step.post_checks],
        "needs_confirmation": step.needs_confirmation,
        "decision_point": step.decision_point,
        "sub_agent_scope": step.sub_agent_scope,
        "granted_tools": [_tool_ref_to_dict(ref) for ref in step.granted_tools],
        "status": step.status.value,
        "sub_agent_id": step.sub_agent_id,
        "output_ref": step.output_ref,
        "review_verdicts": [_review_verdict_to_dict(verdict) for verdict in step.review_verdicts],
        "error": step.error,
        "phase": step.phase,
    }


def _step_from_dict(payload: dict[str, Any]) -> PlanStep:
    return PlanStep(
        id=str(payload["id"]),
        plan_id=str(payload["plan_id"]),
        index=int(payload["index"]),
        title=str(payload["title"]),
        tool_ref=_tool_ref_from_dict(payload["tool_ref"]),
        inputs=dict(payload.get("inputs") or {}),
        depends_on=[str(item) for item in payload.get("depends_on") or []],
        post_checks=[
            _post_check_from_dict(item)
            for item in payload.get("post_checks") or []
        ],
        needs_confirmation=bool(payload.get("needs_confirmation", False)),
        decision_point=bool(payload.get("decision_point", False)),
        sub_agent_scope=_optional_str(payload.get("sub_agent_scope")),
        granted_tools=[
            _tool_ref_from_dict(item)
            for item in payload.get("granted_tools") or []
        ],
        status=StepStatus(payload.get("status", StepStatus.PENDING.value)),
        sub_agent_id=_optional_str(payload.get("sub_agent_id")),
        output_ref=_optional_str(payload.get("output_ref")),
        review_verdicts=[
            _review_verdict_from_dict(item)
            for item in payload.get("review_verdicts") or []
        ],
        error=_optional_str(payload.get("error")),
        phase=_optional_str(payload.get("phase")),
    )


def _tool_ref_to_dict(ref: ToolRef) -> dict[str, str]:
    return {"plugin": ref.plugin, "tool": ref.tool, "version": ref.version}


def _tool_ref_from_dict(payload: dict[str, Any]) -> ToolRef:
    return ToolRef(
        plugin=str(payload["plugin"]),
        tool=str(payload["tool"]),
        version=str(payload.get("version") or ""),
    )


def _post_check_to_dict(check: PostCheck) -> dict[str, Any]:
    return {"kind": check.kind, "spec": check.spec}


def _post_check_from_dict(payload: dict[str, Any]) -> PostCheck:
    return PostCheck(kind=str(payload["kind"]), spec=dict(payload.get("spec") or {}))


def _review_verdict_to_dict(verdict: ReviewVerdict) -> dict[str, Any]:
    return {
        "reviewer": verdict.reviewer,
        "passed": verdict.passed,
        "reasons": list(verdict.reasons),
        "at": verdict.at,
    }


def _review_verdict_from_dict(payload: dict[str, Any]) -> ReviewVerdict:
    return ReviewVerdict(
        reviewer=str(payload["reviewer"]),
        passed=bool(payload["passed"]),
        reasons=[str(item) for item in payload.get("reasons") or []],
        at=str(payload["at"]),
    )


def _loop_event_to_dict(event: LoopEvent) -> dict[str, Any]:
    payload = {
        "type": event.type,
        "reason": event.reason,
        "at": event.at,
    }
    if event.trigger_step_id is not None:
        payload["trigger_step_id"] = event.trigger_step_id
    if event.instruction is not None:
        payload["instruction"] = event.instruction
    if event.tool_ref is not None:
        payload["tool_ref"] = event.tool_ref
    return payload


def _loop_event_from_dict(payload: dict[str, Any]) -> LoopEvent:
    return LoopEvent(
        type=str(payload["type"]),
        reason=str(payload.get("reason") or ""),
        at=str(payload.get("at") or ""),
        trigger_step_id=_optional_str(payload.get("trigger_step_id")),
        instruction=_optional_str(payload.get("instruction")),
        tool_ref=_optional_str(payload.get("tool_ref")),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
