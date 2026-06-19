from __future__ import annotations

from copy import deepcopy
import re

from marvis.orchestrator.contracts import Plan, PlanStep
from marvis.plugins.errors import (
    PluginNotFoundError,
    SchemaValidationError,
    ToolNotFoundError,
)
from marvis.plugins.schema_validation import validate_against_schema


METRIC_FIELDS = frozenset({"ks", "auc", "psi", "iv", "lift", "gini"})
POST_CHECK_KINDS = frozenset({
    "schema",
    "range",
    "rowcount",
    "invariant",
    "nonempty",
    "match_rate",
})
_SLOT_PLACEHOLDER_RE = re.compile(r"^\{slot:[A-Za-z_][A-Za-z0-9_:-]*\}$")


class PlanValidator:
    def __init__(self, tool_registry):
        self._tools = tool_registry

    def validate(self, plan: Plan) -> list[str]:
        problems: list[str] = []
        problems.extend(self._check_tools_exist(plan))
        problems.extend(self._check_inputs_schema(plan))
        problems.extend(self._check_dag(plan))
        problems.extend(self._check_ref_compatibility(plan))
        problems.extend(self._check_post_check_kinds(plan))
        problems.extend(self._check_join_gates(plan))
        problems.extend(self._check_determinism_checks(plan))
        problems.extend(self._check_subagent_grants(plan))
        problems.extend(self._check_decision_points(plan))
        return problems

    def _check_tools_exist(self, plan: Plan) -> list[str]:
        problems = []
        for step in plan.steps:
            try:
                self._tools.resolve(step.tool_ref)
            except (PluginNotFoundError, ToolNotFoundError) as exc:
                problems.append(f"step {step.title}: {exc}")
        return problems

    def _check_inputs_schema(self, plan: Plan) -> list[str]:
        problems = []
        for step in plan.steps:
            tool = self._resolve_step_tool(step)
            if tool is None:
                continue
            literal_inputs = {
                key: value
                for key, value in step.inputs.items()
                if not _is_deferred_input(value)
            }
            schema = _relax_required(tool.input_schema, step.inputs)
            try:
                validate_against_schema(literal_inputs, schema, label=f"inputs:{step.id}")
            except SchemaValidationError as exc:
                problems.append(f"step {step.title}: {exc}")
        return problems

    def _check_dag(self, plan: Plan) -> list[str]:
        problems = []
        ids = [step.id for step in plan.steps]
        id_set = set(ids)
        if len(id_set) != len(ids):
            problems.append("duplicate step id detected")
        for step in plan.steps:
            for dependency in step.depends_on:
                if dependency not in id_set:
                    problems.append(f"step {step.title}: dangling dependency {dependency}")
        if _has_cycle(plan.steps):
            problems.append("dependency cycle detected")
        return problems

    def _check_ref_compatibility(self, plan: Plan) -> list[str]:
        problems = []
        by_id = {step.id: step for step in plan.steps}
        for step in plan.steps:
            for value in step.inputs.values():
                if not _is_ref(value):
                    continue
                try:
                    upstream_id, field = _parse_ref(value)
                except ValueError as exc:
                    problems.append(f"step {step.title}: {exc}")
                    continue
                upstream = by_id.get(upstream_id)
                if upstream is None:
                    problems.append(f"step {step.title}: ref to unknown step {upstream_id}")
                    continue
                if upstream_id not in step.depends_on:
                    problems.append(
                        f"step {step.title}: ref to {upstream_id} lacks dependency edge"
                    )
                    continue
                upstream_tool = self._resolve_step_tool(upstream)
                if upstream_tool is None:
                    continue
                if field and field not in _schema_fields(upstream_tool.output_schema):
                    problems.append(
                        f"step {step.title}: ref field {field} not in upstream output"
                    )
        return problems

    def _check_join_gates(self, plan: Plan) -> list[str]:
        return [
            f"join step {step.title} must require confirmation (INV-3)"
            for step in plan.steps
            if step.tool_ref.tool == "execute_join" and not step.needs_confirmation
        ]

    def _check_post_check_kinds(self, plan: Plan) -> list[str]:
        problems = []
        for step in plan.steps:
            for check in step.post_checks:
                if check.kind not in POST_CHECK_KINDS:
                    problems.append(
                        f"step {step.title}: unknown post_check kind {check.kind}"
                    )
        return problems

    def _check_determinism_checks(self, plan: Plan) -> list[str]:
        problems = []
        for step in plan.steps:
            tool = self._resolve_step_tool(step)
            if tool is None:
                continue
            metric_fields = _metric_fields_in(tool.output_schema)
            checked = {
                check.spec.get("field")
                for check in step.post_checks
                if check.kind == "range"
            }
            for field in sorted(metric_fields - checked):
                problems.append(
                    f"step {step.title}: metric {field} lacks range post_check (INV-1)"
                )
        return problems

    def _check_subagent_grants(self, plan: Plan) -> list[str]:
        problems = []
        for step in plan.steps:
            if not step.sub_agent_scope:
                continue
            if not step.granted_tools:
                problems.append(f"sub-agent step {step.title} has empty granted_tools")
                continue
            for ref in step.granted_tools:
                try:
                    self._tools.resolve(ref)
                except (PluginNotFoundError, ToolNotFoundError) as exc:
                    problems.append(
                        f"sub-agent step {step.title}: granted tool {ref.label()} {exc}"
                    )
        return problems

    def _check_decision_points(self, plan: Plan) -> list[str]:
        return [
            f"decision_point is not allowed on safety step {step.title}"
            for step in plan.steps
            if step.decision_point and _is_safety_step(step)
        ]

    def _resolve_step_tool(self, step: PlanStep):
        try:
            return self._tools.resolve(step.tool_ref)
        except (PluginNotFoundError, ToolNotFoundError):
            return None


def _is_slot_placeholder(value) -> bool:
    return isinstance(value, str) and bool(_SLOT_PLACEHOLDER_RE.fullmatch(value))


def _is_ref(value) -> bool:
    return isinstance(value, str) and value.startswith("$ref:")


def _is_deferred_input(value) -> bool:
    return _is_ref(value) or _is_slot_placeholder(value)


def _relax_required(input_schema: dict, step_inputs: dict) -> dict:
    relaxed = deepcopy(input_schema)
    required = relaxed.get("required")
    if not isinstance(required, list):
        return relaxed
    deferred_keys = {
        key for key, value in step_inputs.items() if _is_deferred_input(value)
    }
    relaxed["required"] = [key for key in required if key not in deferred_keys]
    return relaxed


def _parse_ref(value: str) -> tuple[str, str]:
    raw = value[len("$ref:"):]
    marker = ".output"
    if marker not in raw:
        raise ValueError(f"invalid ref {value}")
    step_id, tail = raw.split(marker, 1)
    if not step_id:
        raise ValueError(f"invalid ref {value}")
    if not tail:
        return step_id, ""
    if not tail.startswith(".") or tail == ".":
        raise ValueError(f"invalid ref {value}")
    return step_id, tail[1:]


def _schema_fields(schema: dict) -> set[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return set()
    return set(properties)


def _metric_fields_in(schema: dict) -> set[str]:
    return _schema_fields(schema) & METRIC_FIELDS


def _is_safety_step(step: PlanStep) -> bool:
    if step.tool_ref.tool == "execute_join":
        return True
    return any(check.kind == "range" for check in step.post_checks)


def _has_cycle(steps: list[PlanStep]) -> bool:
    graph = {step.id: list(step.depends_on) for step in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(step_id: str) -> bool:
        if step_id in visited:
            return False
        if step_id in visiting:
            return True
        visiting.add(step_id)
        for dependency in graph.get(step_id, []):
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(step_id)
        visited.add(step_id)
        return False

    return any(visit(step_id) for step_id in graph)
