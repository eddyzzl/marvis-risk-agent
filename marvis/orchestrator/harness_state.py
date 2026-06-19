from __future__ import annotations

from marvis.orchestrator.contracts import PlanStatus, StepStatus
from marvis.orchestrator.errors import IllegalPlanTransition, IllegalStepTransition


PLAN_TRANSITIONS: dict[PlanStatus, frozenset[PlanStatus]] = {
    PlanStatus.DRAFT: frozenset({
        PlanStatus.VALIDATED,
        PlanStatus.FAILED,
        PlanStatus.CANCELLED,
    }),
    PlanStatus.VALIDATED: frozenset({
        PlanStatus.CONFIRMED,
        PlanStatus.FAILED,
        PlanStatus.CANCELLED,
    }),
    PlanStatus.CONFIRMED: frozenset({PlanStatus.RUNNING, PlanStatus.CANCELLED}),
    PlanStatus.RUNNING: frozenset({
        PlanStatus.AWAITING_CONFIRM,
        PlanStatus.REVIEW,
        PlanStatus.FAILED,
        PlanStatus.CANCELLED,
    }),
    PlanStatus.AWAITING_CONFIRM: frozenset({
        PlanStatus.RUNNING,
        PlanStatus.CANCELLED,
    }),
    PlanStatus.REVIEW: frozenset({PlanStatus.DONE, PlanStatus.FAILED}),
    PlanStatus.DONE: frozenset(),
    PlanStatus.FAILED: frozenset(),
    PlanStatus.CANCELLED: frozenset(),
}

STEP_TRANSITIONS: dict[StepStatus, frozenset[StepStatus]] = {
    StepStatus.PENDING: frozenset({
        StepStatus.BLOCKED,
        StepStatus.AWAITING_CONFIRM,
        StepStatus.RUNNING,
        StepStatus.SKIPPED,
    }),
    StepStatus.BLOCKED: frozenset({
        StepStatus.PENDING,
        StepStatus.AWAITING_CONFIRM,
        StepStatus.RUNNING,
        StepStatus.SKIPPED,
    }),
    StepStatus.AWAITING_CONFIRM: frozenset({
        StepStatus.RUNNING,
        StepStatus.SKIPPED,
    }),
    StepStatus.RUNNING: frozenset({StepStatus.CHECKING, StepStatus.FAILED}),
    StepStatus.CHECKING: frozenset({StepStatus.DONE, StepStatus.FAILED}),
    StepStatus.DONE: frozenset(),
    StepStatus.FAILED: frozenset({StepStatus.PENDING}),
    StepStatus.SKIPPED: frozenset(),
}


def assert_plan_transition(current: PlanStatus, target: PlanStatus) -> None:
    if target not in PLAN_TRANSITIONS.get(current, frozenset()):
        raise IllegalPlanTransition(current, target)


def assert_step_transition(current: StepStatus, target: StepStatus) -> None:
    if target not in STEP_TRANSITIONS.get(current, frozenset()):
        raise IllegalStepTransition(current, target)


class HarnessState:
    def __init__(self, plan_repo=None):
        self.plan_repo = plan_repo

    def assert_plan_transition(self, current: PlanStatus, target: PlanStatus) -> None:
        assert_plan_transition(current, target)

    def assert_step_transition(self, current: StepStatus, target: StepStatus) -> None:
        assert_step_transition(current, target)
