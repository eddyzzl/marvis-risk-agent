from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marvis.orchestrator.contracts import Plan

EVALUATION_MODES = frozenset({"blind", "contract_regression"})
EXECUTION_MODES = frozenset({"fixture_simulation", "real_execution"})
MODEL_SOURCES = frozenset({"real_model", "fixture_model", "unknown"})


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
    # Only explicit contract_regression runs may exclude a declared known
    # gap from their scoring denominator. Blind runs retain every case,
    # including these failures; the marker is then informational only.
    expected_failure: str = ""


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    model_id: str
    tier: str
    passed: bool
    metrics: dict[str, float]
    transcript_ref: str
    final_status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_mode: str = "unknown"
    execution_mode: str = "unknown"
    model_source: str = "unknown"
    executor_invoked: bool = False


@dataclass(frozen=True)
class PlanRunTrace:
    plan: Plan | None
    tools: tuple[str, ...] = ()
    final_status: str = ""
    plan_valid: bool = False
    replan_count: int = 0
    segments: int = 0
    guardrail_hits: tuple[str, ...] = ()
    guardrail_outcome: str = ""
    intervention_source: str = ""
    invented_numbers: bool = False
    transcript_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Missing provenance on historical/custom traces is never evidence of a
    # blind or real execution run. The fixture runner sets these explicitly.
    evaluation_mode: str = "unknown"
    execution_mode: str = "unknown"
    model_source: str = "unknown"
    executor_invoked: bool = False
