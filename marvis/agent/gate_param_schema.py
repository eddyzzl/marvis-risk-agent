"""Adjustable-parameter schema summary for a gate (AGT-5).

``route_instruction`` previously saw only the gate title, so the routing LLM had
to blind-guess ``adjust`` parameter names/values from free text alone — a wrong
guess meant ``apply_adjust`` matched nothing and the user just saw "没有识别到可
调整的参数" (gate_execution_adapter.py). This module assembles the same
parameter names ``apply_adjust`` would actually accept — one entry per key in
each dependency step's ``inputs`` — plus type/current-value/bounds (from
``adjust_specs`` where the key is a recognised typed control), so the router can
be told up front which keys exist instead of discovering it after a failed
attempt.

Pure + deterministic: no LLM call, just reads ``plan``/``gate`` state.
"""

from __future__ import annotations

from marvis.agent.adjust_specs import (
    NONNEGATIVE_INT_ADJUST_PARAMS,
    POSITIVE_INT_ADJUST_PARAMS,
    UNIT_INTERVAL_ADJUST_PARAMS,
)
from marvis.agent.plan_utils import find_step
from marvis.orchestrator.contracts import Plan, PlanStep

_UNIT_INTERVAL_BOUNDS = {"min": 0, "max": 1}
_POSITIVE_INT_BOUNDS = {"min": 1}
_NONNEGATIVE_INT_BOUNDS = {"min": 0}


def gate_param_schema(plan: Plan, gate: PlanStep | None) -> list[dict]:
    """Adjustable-parameter summary for ``gate``'s dependency step(s).

    Returns a list of ``{"name", "type", "current", "bounds"}`` dicts (bounds
    omitted when unknown), one per input key across every dependency step —
    exactly the key set ``GateExecutionAdapter.apply_adjust`` matches ``params``
    against. Deterministic ordering (dependency order, then input-key sort) so
    prompts stay stable across otherwise-identical calls."""
    if gate is None:
        return []
    seen: set[str] = set()
    schema: list[dict] = []
    for dep_id in gate.depends_on or []:
        dep = find_step(plan, dep_id)
        if dep is None:
            continue
        for key in sorted((dep.inputs or {}).keys()):
            if key in seen:
                continue
            seen.add(key)
            value = dep.inputs[key]
            entry = {"name": key, "type": _type_name(value), "current": value}
            bounds = _bounds_for(key)
            if bounds:
                entry["bounds"] = bounds
            schema.append(entry)
    return schema


def _type_name(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _bounds_for(key: str) -> dict | None:
    if key in UNIT_INTERVAL_ADJUST_PARAMS:
        return dict(_UNIT_INTERVAL_BOUNDS)
    if key in POSITIVE_INT_ADJUST_PARAMS:
        return dict(_POSITIVE_INT_BOUNDS)
    if key in NONNEGATIVE_INT_ADJUST_PARAMS:
        return dict(_NONNEGATIVE_INT_BOUNDS)
    return None


__all__ = ["gate_param_schema"]
