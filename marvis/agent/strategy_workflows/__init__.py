"""Canonical built-in strategy workflow boundary.

The catalog owns workflow identity and exposure.  Migrated families also own
validation, deterministic confirmation, and plan preparation.  Execution stays
behind PlanDriver/PlanValidator; this module never invokes a Tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._catalog import build_catalog
from .contracts import (
    PreparedStrategyPlan,
    ResolvedStrategyWorkflow,
    StrategyWorkflowPreparationContext,
    StrategyWorkflowRequirements,
    StrategyWorkflowResolutionContext,
    StrategyWorkflowResolutionMode,
    StrategyWorkflowSpec,
    StrategyWorkflowValidationError,
    deep_freeze,
)


_CATALOG = build_catalog()
_BY_ID = {spec.workflow_id: spec for spec in _CATALOG}
if len(_BY_ID) != len(_CATALOG):  # pragma: no cover - import-time invariant
    raise RuntimeError("duplicate canonical strategy workflow id")

FRESH_STANDARD_STRATEGY_WORKFLOWS = tuple(
    spec.workflow_id for spec in _CATALOG if spec.fresh
)
LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS = tuple(
    spec.workflow_id for spec in _CATALOG if spec.replayable and not spec.fresh
)
REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS = tuple(
    spec.workflow_id for spec in _CATALOG if spec.replayable
)
MANUAL_STANDARD_STRATEGY_WORKFLOWS = tuple(
    spec.workflow_id for spec in _CATALOG if spec.manual
)


def resolve_strategy_request(
    workflow_id: str,
    workflow_inputs: Mapping[str, Any],
    *,
    context: StrategyWorkflowResolutionContext,
    mode: StrategyWorkflowResolutionMode = StrategyWorkflowResolutionMode.FRESH,
) -> ResolvedStrategyWorkflow:
    """Validate one migrated structured request and deterministically echo it."""

    spec = _migrated_spec(workflow_id)
    if mode is StrategyWorkflowResolutionMode.FRESH:
        enabled = spec.fresh
        validator = spec.validator
    elif mode is StrategyWorkflowResolutionMode.REPLAY:
        enabled = spec.replayable
        validator = spec.replay_validator or spec.validator
    else:
        raise ValueError(f"unsupported strategy workflow resolution mode: {mode}")
    if not enabled:
        raise StrategyWorkflowValidationError(
            f"{workflow_id} is unavailable for {mode.value} resolution",
            code=f"strategy_workflow_{mode.value}_unsupported",
        )
    assert validator is not None
    assert spec.confirmation is not None
    normalized = validator(workflow_inputs, context)
    frozen = deep_freeze(normalized)
    requirements = _resolved_requirements(spec, frozen)
    return ResolvedStrategyWorkflow(
        workflow_id=workflow_id,
        workflow_inputs=frozen,
        confirmation=spec.confirmation(frozen),
        requirements=requirements,
    )


def prepare_strategy_plan(
    workflow_id: str,
    workflow_inputs: Mapping[str, Any],
    *,
    context: StrategyWorkflowPreparationContext,
) -> PreparedStrategyPlan:
    """Bind platform-owned evidence for one migrated workflow without execution."""

    spec = _migrated_spec(workflow_id)
    assert spec.preparer is not None
    prepared = spec.preparer(workflow_inputs, context)
    if prepared.workflow_id != workflow_id:
        raise RuntimeError(
            f"{workflow_id} prepared mismatched workflow {prepared.workflow_id}"
        )
    if prepared.template_id not in spec.template_ids:
        raise RuntimeError(
            f"{workflow_id} prepared undeclared template {prepared.template_id}"
        )
    return prepared


def migrated_workflow_requirements(
    workflow_id: str,
    workflow_inputs: Mapping[str, Any] | None = None,
) -> tuple[bool, bool, bool] | None:
    spec = _BY_ID.get(workflow_id)
    if spec is None or not spec.migrated:
        return None
    if callable(spec.requirements) and workflow_inputs is None:
        raise RuntimeError(
            f"{workflow_id} requires normalized workflow_inputs to resolve requirements"
        )
    requirements = _resolved_requirements(
        spec,
        {} if workflow_inputs is None else workflow_inputs,
    )
    if not all(
        isinstance(value, bool)
        for value in (
            requirements.dataset,
            requirements.target,
            requirements.complete_labels,
        )
    ):
        raise RuntimeError(f"{workflow_id} migrated with unknown requirements")
    return requirements.dataset, requirements.target, requirements.complete_labels


def _resolved_requirements(
    spec: StrategyWorkflowSpec,
    workflow_inputs: Mapping[str, Any],
) -> StrategyWorkflowRequirements:
    requirements = (
        spec.requirements(workflow_inputs)
        if callable(spec.requirements)
        else spec.requirements
    )
    if not isinstance(requirements, StrategyWorkflowRequirements):
        raise RuntimeError(
            f"{spec.workflow_id} requirement resolver returned an invalid contract"
        )
    if spec.migrated and not all(
        isinstance(value, bool)
        for value in (
            requirements.dataset,
            requirements.target,
            requirements.complete_labels,
        )
    ):
        raise RuntimeError(f"{spec.workflow_id} migrated with unknown requirements")
    return requirements


def migrated_workflow_confirmation(
    workflow_id: str, workflow_inputs: Mapping[str, Any]
) -> str | None:
    spec = _BY_ID.get(workflow_id)
    if spec is None or not spec.migrated:
        return None
    assert spec.confirmation is not None
    return spec.confirmation(workflow_inputs)


def is_migrated_workflow(workflow_id: str) -> bool:
    spec = _BY_ID.get(workflow_id)
    return bool(spec is not None and spec.migrated)


def _migrated_spec(workflow_id: str) -> StrategyWorkflowSpec:
    spec = _BY_ID.get(workflow_id)
    if spec is None or not spec.migrated:
        raise KeyError(workflow_id)
    return spec


__all__ = [
    "FRESH_STANDARD_STRATEGY_WORKFLOWS",
    "LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS",
    "MANUAL_STANDARD_STRATEGY_WORKFLOWS",
    "PreparedStrategyPlan",
    "REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS",
    "ResolvedStrategyWorkflow",
    "StrategyWorkflowPreparationContext",
    "StrategyWorkflowRequirements",
    "StrategyWorkflowResolutionContext",
    "StrategyWorkflowResolutionMode",
    "StrategyWorkflowSpec",
    "StrategyWorkflowValidationError",
    "is_migrated_workflow",
    "migrated_workflow_confirmation",
    "migrated_workflow_requirements",
    "prepare_strategy_plan",
    "resolve_strategy_request",
]
