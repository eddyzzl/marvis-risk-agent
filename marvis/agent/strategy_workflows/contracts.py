from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class StrategyWorkflowValidationError(ValueError):
    """Typed, user-correctable failure at the strategy workflow boundary."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_strategy_request",
        fields: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(dict.fromkeys(str(field) for field in fields))


class StrategyWorkflowResolutionMode(StrEnum):
    """Select the trust boundary used to normalize a workflow request."""

    FRESH = "fresh"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class StrategyWorkflowRequirements:
    dataset: bool | None
    target: bool | None
    complete_labels: bool | None


@dataclass(frozen=True, slots=True)
class StrategyWorkflowResolutionContext:
    allowed_columns: tuple[str, ...]
    target_col: str | None


@dataclass(frozen=True, slots=True)
class StrategyWorkflowPreparationContext:
    dataset_id: str | None
    drop_nan_labels: bool = False
    bind_sample_design: Callable[[bool], Mapping[str, object]] | None = None
    validate_strategy_ref: (
        Callable[[str, frozenset[str]], None] | None
    ) = None
    bind_model_score_comparison: (
        Callable[[str, str], Mapping[str, object]] | None
    ) = None
    bind_workflow_evidence: (
        Callable[[str, Mapping[str, Any]], Mapping[str, object]] | None
    ) = None


@dataclass(frozen=True, slots=True)
class ResolvedStrategyWorkflow:
    workflow_id: str
    workflow_inputs: Mapping[str, Any]
    confirmation: str
    requirements: StrategyWorkflowRequirements


@dataclass(frozen=True, slots=True)
class PreparedStrategyPlan:
    workflow_id: str
    template_id: str
    slots: Mapping[str, Any]
    success_criteria: tuple[Mapping[str, Any], ...] = ()

    def to_runtime_slots(self) -> dict[str, Any]:
        """Return JSON-native mutable slots at the execution boundary."""

        slots = deep_thaw(self.slots)
        if not isinstance(slots, dict):  # defensive: ``slots`` is typed Mapping
            raise TypeError("prepared strategy slots must thaw to an object")
        return slots


Validator = Callable[
    [Mapping[str, Any], StrategyWorkflowResolutionContext],
    dict[str, Any],
]
Confirmation = Callable[[Mapping[str, Any]], str]
Preparer = Callable[
    [Mapping[str, Any], StrategyWorkflowPreparationContext],
    PreparedStrategyPlan,
]
RequirementResolver = Callable[
    [Mapping[str, Any]],
    StrategyWorkflowRequirements,
]


@dataclass(frozen=True, slots=True)
class StrategyWorkflowSpec:
    """Canonical metadata plus adapters for one governed built-in workflow."""

    workflow_id: str
    fresh: bool
    replayable: bool
    manual: bool
    requirements: StrategyWorkflowRequirements | RequirementResolver
    template_ids: tuple[str, ...] = ()
    presenter_refs: tuple[str, ...] = ()
    validator: Validator | None = None
    replay_validator: Validator | None = None
    confirmation: Confirmation | None = None
    preparer: Preparer | None = None

    def __post_init__(self) -> None:
        adapters = (self.validator, self.confirmation, self.preparer)
        if any(adapter is not None for adapter in adapters) and not all(
            adapter is not None for adapter in adapters
        ):
            raise ValueError(
                "migrated strategy workflows must define all three adapters"
            )
        if self.replay_validator is not None and self.validator is None:
            raise ValueError(
                "strategy replay validator requires a fresh validator"
            )
        if len(self.presenter_refs) != len(set(self.presenter_refs)):
            raise ValueError("strategy workflow presenter refs must be unique")
        if any(not ref.strip() for ref in self.presenter_refs):
            raise ValueError("strategy workflow presenter refs must be non-empty")

    @property
    def migrated(self) -> bool:
        return all(
            adapter is not None
            for adapter in (self.validator, self.confirmation, self.preparer)
        )


def deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(
        value, str | bytes | bytearray
    ):
        return tuple(deep_freeze(item) for item in value)
    return value


def deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


__all__ = [
    "Confirmation",
    "PreparedStrategyPlan",
    "Preparer",
    "RequirementResolver",
    "ResolvedStrategyWorkflow",
    "StrategyWorkflowPreparationContext",
    "StrategyWorkflowRequirements",
    "StrategyWorkflowResolutionContext",
    "StrategyWorkflowResolutionMode",
    "StrategyWorkflowSpec",
    "StrategyWorkflowValidationError",
    "Validator",
    "deep_freeze",
    "deep_thaw",
]
