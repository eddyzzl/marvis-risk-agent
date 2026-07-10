from __future__ import annotations


class OrchestratorError(Exception):
    """Base class for V2 orchestration failures."""


class IllegalPlanTransition(OrchestratorError):
    def __init__(self, current, target) -> None:
        super().__init__(f"illegal plan transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


class IllegalStepTransition(OrchestratorError):
    def __init__(self, current, target) -> None:
        super().__init__(f"illegal step transition: {current.value} -> {target.value}")
        self.current = current
        self.target = target


class PlanNotFoundError(OrchestratorError):
    pass


class RefResolutionError(OrchestratorError):
    """A plan input reference could not be resolved to an upstream value."""

    def __init__(self, ref: str, reason: str) -> None:
        super().__init__(f"could not resolve ref {ref}: {reason}")
        self.ref = ref
        self.reason = reason
