from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

from marvis.orchestrator.contracts import Plan, PlanStep
from marvis.orchestrator.references import parse_step_output_ref
from marvis.orchestrator.safety import (
    METRIC_FIELDS,
    is_draft_run_step,
    is_safety_step,
    literal_metric_claims,
)
from marvis.plugins.errors import (
    PluginNotFoundError,
    SchemaValidationError,
    ToolNotFoundError,
)
from marvis.plugins.schema_validation import validate_against_schema


POST_CHECK_KINDS = frozenset({
    "schema",
    "range",
    "rowcount",
    "invariant",
    "nonempty",
    "match_rate",
    "one_of",
})
_SLOT_PLACEHOLDER_RE = re.compile(r"^\{slot:[A-Za-z_][A-Za-z0-9_:-]*\}$")

JOIN_CONFIRMATION_REQUIRED = "inv3.join_confirmation_required"
JOIN_ROWCOUNT_REQUIRED = "inv2.join_rowcount_required"
JOIN_INVARIANT_REQUIRED = "inv2.join_invariant_required"
METRIC_RANGE_REQUIRED = "inv1.metric_range_required"
METRIC_LITERAL_UNBACKED = "inv1.metric_literal_unbacked"
JOIN_SAFELY_GATED = "inv3.join_safely_gated"
METRIC_TOOL_BACKED = "inv1.metric_tool_backed"


@dataclass(frozen=True)
class PlanValidationProblem:
    """A validator-owned problem whose identity cannot be forged by plan text."""

    code: str
    message: str
    step_id: str | None = None
    field: str | None = None
    source: str = "plan_validator"


class PlanValidator:
    def __init__(self, tool_registry):
        self._tools = tool_registry

    def validate(self, plan: Plan) -> list[str]:
        return [problem.message for problem in self.validate_problems(plan)]

    def validate_problems(self, plan: Plan) -> list[PlanValidationProblem]:
        """Return typed validation provenance while preserving legacy messages."""

        problems: list[PlanValidationProblem] = []
        problems.extend(_plain_problems("tool_catalog", self._check_tools_exist(plan)))
        problems.extend(_plain_problems("input_schema", self._check_inputs_schema(plan)))
        problems.extend(_plain_problems("dag", self._check_dag(plan)))
        problems.extend(_plain_problems("reference", self._check_ref_compatibility(plan)))
        problems.extend(
            _plain_problems("post_check_kind", self._check_post_check_kinds(plan))
        )
        problems.extend(self._join_gate_problems(plan))
        problems.extend(
            _plain_problems("draft_confirmation", self._check_draft_run_gates(plan))
        )
        problems.extend(self._determinism_problems(plan))
        problems.extend(self.unbacked_metric_literal_validation_problems(plan))
        problems.extend(
            _plain_problems("subagent_grant", self._check_subagent_grants(plan))
        )
        problems.extend(
            _plain_problems("decision_point", self._check_decision_points(plan))
        )
        problems.extend(
            _plain_problems("governance_policy", self._check_governance_policies(plan))
        )
        return problems

    def verified_safety_invariants(self, plan: Plan) -> frozenset[str]:
        """Return positive safety evidence derived from resolved plan structure."""

        verified: set[str] = set()
        join_steps = [
            step
            for step in plan.steps
            if step.tool_ref.tool == "execute_join"
            and self._resolve_step_tool(step) is not None
        ]
        join_codes = {
            problem.code for problem in self._join_gate_problems(plan)
        }
        if join_steps and not join_codes.intersection({
            JOIN_CONFIRMATION_REQUIRED,
            JOIN_ROWCOUNT_REQUIRED,
            JOIN_INVARIANT_REQUIRED,
        }):
            verified.add(JOIN_SAFELY_GATED)

        if self.verified_metric_fields(plan):
            verified.add(METRIC_TOOL_BACKED)
        return frozenset(verified)

    def verified_metric_fields(self, plan: Plan) -> frozenset[str]:
        """Metric leaf names backed by a resolved Tool and explicit bounds."""

        if self.unbacked_metric_literal_validation_problems(plan):
            return frozenset()
        verified: set[str] = set()
        for step in plan.steps:
            tool = self._resolve_step_tool(step)
            if tool is None:
                continue
            metric_fields = _metric_fields_in(tool.output_schema)
            checked_fields = {
                str(check.spec.get("field") or "")
                for check in step.post_checks
                if check.kind == "range"
            }
            if metric_fields and metric_fields.issubset(checked_fields):
                verified.update(
                    field.rsplit(".", 1)[-1].lower()
                    for field in metric_fields
                )
        return frozenset(verified)

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
            gate_deferred_keys = {
                key
                for key, value in step.inputs.items()
                if step.needs_confirmation and value is None
            }
            literal_inputs = {
                key: value
                for key, value in step.inputs.items()
                if not _is_deferred_input(value) and key not in gate_deferred_keys
            }
            schema = _relax_required(
                tool.input_schema,
                step.inputs,
                extra_deferred_keys=gate_deferred_keys,
            )
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
            for value in _iter_refs(step.inputs):
                try:
                    upstream_id, field = parse_step_output_ref(value)
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
                if field and not _schema_has_path(upstream_tool.output_schema, field):
                    problems.append(
                        f"step {step.title}: ref field {field} not in upstream output"
                    )
        return problems

    def _join_gate_problems(self, plan: Plan) -> list[PlanValidationProblem]:
        problems: list[PlanValidationProblem] = []
        for step in plan.steps:
            if step.tool_ref.tool != "execute_join":
                continue
            # The invariant only belongs to a catalog-authenticated join Tool.
            # An unknown plugin may choose the same LLM-controlled tool label;
            # catalog failure remains generic planning failure, not INV-3 proof.
            if self._resolve_step_tool(step) is None:
                continue
            if not step.needs_confirmation:
                problems.append(
                    PlanValidationProblem(
                        code=JOIN_CONFIRMATION_REQUIRED,
                        message=(
                            f"join step {step.title} must require confirmation (INV-3)"
                        ),
                        step_id=step.id,
                    )
                )
            check_kinds = {check.kind for check in step.post_checks}
            if "rowcount" not in check_kinds:
                problems.append(
                    PlanValidationProblem(
                        code=JOIN_ROWCOUNT_REQUIRED,
                        message=(
                            f"join step {step.title} must include rowcount "
                            "post_check (INV-2)"
                        ),
                        step_id=step.id,
                    )
                )
            invariant_rules = {
                str(check.spec.get("rule") or "").replace(" ", "")
                for check in step.post_checks
                if check.kind == "invariant"
            }
            if "joined_rows<=anchor_rows" not in invariant_rules:
                problems.append(
                    PlanValidationProblem(
                        code=JOIN_INVARIANT_REQUIRED,
                        message=(
                            f"join step {step.title} must include "
                            "joined_rows<=anchor_rows invariant (INV-2)"
                        ),
                        step_id=step.id,
                    )
                )
        return problems

    def _check_draft_run_gates(self, plan: Plan) -> list[str]:
        return [
            f"draft run step {step.title} must require confirmation"
            for step in plan.steps
            if is_draft_run_step(step) and not step.needs_confirmation
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

    def _determinism_problems(self, plan: Plan) -> list[PlanValidationProblem]:
        problems: list[PlanValidationProblem] = []
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
                    PlanValidationProblem(
                        code=METRIC_RANGE_REQUIRED,
                        message=(
                            f"step {step.title}: metric {field} lacks "
                            "range post_check (INV-1)"
                        ),
                        step_id=step.id,
                        field=field,
                    )
                )
        return problems

    def unbacked_metric_literal_problems(self, plan: Plan) -> list[str]:
        """Reject claimed metric results that are not produced by the step's tool.

        Numeric thresholds remain valid configuration. This guard only recognizes
        explicit result-shaped literals such as ``ks=0.42`` or ``{"auc": 0.78}``;
        refs/slots are evidence bindings and are therefore not treated as literals.
        """

        return [
            problem.message
            for problem in self.unbacked_metric_literal_validation_problems(plan)
        ]

    def unbacked_metric_literal_validation_problems(
        self,
        plan: Plan,
    ) -> list[PlanValidationProblem]:
        problems: list[PlanValidationProblem] = []
        for step in plan.steps:
            tool = self._resolve_step_tool(step)
            if tool is None:
                continue
            backed_fields = {
                path.rsplit(".", 1)[-1].lower()
                for path in _metric_fields_in(tool.output_schema)
            }
            for field in sorted(literal_metric_claims(step.inputs) - backed_fields):
                problems.append(
                    PlanValidationProblem(
                        code=METRIC_LITERAL_UNBACKED,
                        message=(
                            f"step {step.title}: metric {field} literal lacks "
                            "tool-backed output (INV-1)"
                        ),
                        step_id=step.id,
                        field=field,
                    )
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
            if step.decision_point and is_safety_step(step)
        ]

    def _check_governance_policies(self, plan: Plan) -> list[str]:
        """A plan may strengthen a manifest policy but may never weaken it."""

        problems: list[str] = []
        for step in plan.steps:
            tool = self._resolve_step_tool(step)
            if tool is None:
                continue
            required = tool.policy
            actual = step.policy
            if (
                required.human_decision_gate == "required"
                and actual.human_decision_gate != "required"
            ):
                problems.append(
                    f"step {step.title}: human_decision_gate cannot be lower than tool policy"
                )
            if (
                required.effect_authorization == "required"
                and actual.effect_authorization != "required"
            ):
                problems.append(
                    f"step {step.title}: effect_authorization cannot be lower than tool policy"
                )
            if (
                required.effect_authorization == "required"
                and actual.effect_authorization == "required"
                and actual.effect_target != required.effect_target
            ):
                problems.append(
                    f"step {step.title}: effect_authorization target must match tool policy"
                )
            if (
                actual.human_decision_gate == "required"
                and not step.needs_confirmation
            ):
                problems.append(
                    f"step {step.title}: human_decision_gate=required needs confirmation"
                )

            for ref in step.granted_tools:
                try:
                    granted = self._tools.resolve(ref)
                except (PluginNotFoundError, ToolNotFoundError):
                    continue
                if granted.policy.effect_authorization == "required":
                    problems.append(
                        f"sub-agent step {step.title}: effect-authorized tool "
                        f"{ref.label()} cannot be granted to a sub-agent"
                    )
                elif granted.policy.human_decision_gate == "required":
                    problems.append(
                        f"sub-agent step {step.title}: human-decision-gated tool "
                        f"{ref.label()} cannot be granted to a sub-agent"
                    )
        return problems

    def _resolve_step_tool(self, step: PlanStep):
        try:
            return self._tools.resolve(step.tool_ref)
        except (PluginNotFoundError, ToolNotFoundError):
            return None


def _plain_problems(code: str, messages: list[str]) -> list[PlanValidationProblem]:
    return [
        PlanValidationProblem(code=code, message=message)
        for message in messages
    ]


def _is_slot_placeholder(value) -> bool:
    return isinstance(value, str) and bool(_SLOT_PLACEHOLDER_RE.fullmatch(value))


def _is_ref(value) -> bool:
    return isinstance(value, str) and value.startswith("$ref:")


def _walk_values(value):
    yield value
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)


def _iter_refs(value):
    return (item for item in _walk_values(value) if _is_ref(item))


def _is_deferred_input(value) -> bool:
    # A container (e.g. FS-5's features union list `[{slot:feature_cols}, $ref:...]`) is
    # deferred if any element is — its final shape is only known after ref/slot resolution,
    # so it cannot be schema-validated at plan time. Use the same recursive walk as ref
    # compatibility checks so container shapes cannot diverge between the two passes.
    return any(
        _is_ref(item) or _is_slot_placeholder(item)
        for item in _walk_values(value)
    )


def _relax_required(
    input_schema: dict,
    step_inputs: dict,
    *,
    extra_deferred_keys: set[str] | None = None,
) -> dict:
    relaxed = deepcopy(input_schema)
    deferred_keys = {
        key for key, value in step_inputs.items() if _is_deferred_input(value)
    }
    deferred_keys.update(extra_deferred_keys or ())
    _relax_required_combinators(relaxed, deferred_keys)
    return relaxed


def _relax_required_combinators(schema: dict, deferred_keys: set[str]) -> None:
    """Relax deferred top-level inputs inside JSON-Schema branch combinators.

    Plans may carry a ``$ref`` for one branch of a ``oneOf``. The concrete value
    is intentionally absent from plan-time literal validation, so every required
    list governing that same top-level branch must ignore the deferred key. Runtime
    validation receives resolved inputs and the original manifest schema.
    """

    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [
            key for key in required if key not in deferred_keys
        ]
    for combinator in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(combinator)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    _relax_required_combinators(variant, deferred_keys)
    for conditional in ("if", "then", "else"):
        branch = schema.get(conditional)
        if isinstance(branch, dict):
            _relax_required_combinators(branch, deferred_keys)
    negated = schema.get("not")
    if isinstance(negated, dict):
        _relax_required_combinators(negated, deferred_keys)


def _schema_has_path(schema: dict, path: str) -> bool:
    return _schema_has_parts(schema, path.split("."))


def _schema_has_parts(schema: dict, parts: list[str]) -> bool:
    if not isinstance(schema, dict) or any(not part for part in parts):
        return False
    if not parts:
        return True
    for combinator in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(combinator)
        if isinstance(variants, list) and any(
            _schema_has_parts(variant, parts)
            for variant in variants
            if isinstance(variant, dict)
        ):
            return True
    part, *tail = parts
    properties = schema.get("properties")
    if isinstance(properties, dict) and part in properties:
        return _schema_has_parts(properties[part], tail)
    items = schema.get("items")
    if part.isdigit() and isinstance(items, dict):
        return _schema_has_parts(items, tail)
    return False


def _metric_fields_in(schema: dict) -> set[str]:
    return _metric_fields_in_schema(schema)


def _metric_fields_in_schema(schema: dict, *, prefix: str = "") -> set[str]:
    if not isinstance(schema, dict):
        return set()
    fields: set[str] = set()
    schema_type = schema.get("type")
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, child in properties.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            if str(name) in METRIC_FIELDS:
                fields.add(path)
            fields.update(_metric_fields_in_schema(child, prefix=path))
    if schema_type == "array" or "items" in schema:
        item_schema = schema.get("items")
        item_prefix = f"{prefix}.0" if prefix else "0"
        fields.update(_metric_fields_in_schema(item_schema, prefix=item_prefix))
    for combinator in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(combinator)
        if isinstance(variants, list):
            for variant in variants:
                fields.update(_metric_fields_in_schema(variant, prefix=prefix))
    return fields


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
