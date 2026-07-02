from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marvis.orchestrator.contracts import Plan


@dataclass(frozen=True)
class EvalCase:
    id: str
    goal: str
    task_context: dict[str, Any]
    kind: str
    expected: dict[str, Any]
    fixtures: dict[str, Any]
    # Set when this case documents a *known, currently-unsafe* touchpoint
    # behavior rather than a behavior the platform actually guarantees today.
    # ``score_case``/regression tooling must not treat an expected_failure
    # case's non-pass as a regression; it is a tracked gap, recorded here so
    # the eval corpus stays honest about what is and is not defended today.
    expected_failure: str = ""


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    model_id: str
    tier: str
    passed: bool
    metrics: dict[str, float]
    transcript_ref: str


@dataclass(frozen=True)
class PlanRunTrace:
    plan: Plan | None
    tools: tuple[str, ...] = ()
    final_status: str = ""
    plan_valid: bool = False
    replan_count: int = 0
    segments: int = 0
    guardrail_hits: tuple[str, ...] = ()
    invented_numbers: bool = False
    transcript_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
