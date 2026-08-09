"""Production ``run_eval_case`` implementation (LLM-2).

Wires the real ``IntentRouter`` + ``Planner`` + ``PlanValidator`` against an
injected LLM client (a real model in production, a ``FakeLLM`` in tests /
offline replay) and a ``FixtureToolRunner`` that returns preset tool outputs
from ``case.fixtures.tool_outputs`` instead of ever invoking a real tool. This
keeps the eval framework fully offline-self-contained (INV: no eval run may
touch the network or execute untrusted code) while still exercising real
prompt construction, real JSON-extraction/retry paths, and the real plan
validator.

``EvalOrchestrator`` is the ``orchestrator`` object ``run_eval_suite`` /
``calibrate_tier_for_model`` expect: it must expose ``run_eval_case(case, *,
model_id, tier)`` returning a ``PlanRunTrace``.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any

from marvis.db_schema import init_db
from marvis.repositories.plugins import PluginRepository
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.eval.contracts import EvalCase, PlanRunTrace
from marvis.orchestrator.intent import IntentRouter
from marvis.orchestrator.planner import (
    Planner,
    PlannerConstraints,
    PlanningError,
    ReplanError,
    RequiredLiteralInput,
    _planner_constraint_violations,
)
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import (
    JOIN_CONFIRMATION_REQUIRED,
    JOIN_INVARIANT_REQUIRED,
    JOIN_ROWCOUNT_REQUIRED,
    JOIN_SAFELY_GATED,
    METRIC_LITERAL_UNBACKED,
    METRIC_RANGE_REQUIRED,
    METRIC_TOOL_BACKED,
    PlanValidationProblem,
    PlanValidator,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.errors import (
    PluginNotFoundError,
    SchemaValidationError,
    ToolNotFoundError,
)
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema


PACKS_ROOT = Path(__file__).resolve().parents[2] / "packs"


def build_tool_registry(*, db_path: Path | None = None) -> ToolRegistry:
    """Load the real builtin tool catalog into an isolated, offline sqlite DB.

    No network access, no user plugin dirs -- only ``marvis/packs`` builtins,
    matching the catalog every production Planner/Validator sees.
    """
    if db_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="marvis-eval-"))
        db_path = tmp_dir / "eval.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, PACKS_ROOT)
    return ToolRegistry(registry)


class FixtureToolRunner:
    """Returns preset tool outputs from ``case.fixtures.tool_outputs``.

    Never executes a real tool -- this is what keeps eval runs offline and
    side-effect free even when driven against a real model.
    """

    def __init__(self, tool_outputs: dict[str, Any]):
        self._outputs = dict(tool_outputs or {})

    def run(self, step: PlanStep) -> dict[str, Any]:
        key = step.tool_ref.label()
        if key in self._outputs:
            return self._outputs[key]
        # Unfixtured tool: return an empty-but-valid object so downstream
        # $ref lookups don't crash the simulation; this is intentionally
        # permissive since eval cases only assert on plan shape / routed
        # tools / guardrail interception, not on live numeric outputs.
        return {}


class EvalOrchestrator:
    """Drives IntentRouter + Planner + PlanValidator for one eval case.

    ``llm_factory`` must return an object with ``complete(**kwargs) -> str``.
    Pass a real ``OpenAICompatibleLLMClient``-backed factory for a genuine
    model run, or a ``FakeLLM`` factory for offline replay / regression tests.
    """

    def __init__(self, llm_factory, *, tool_registry: ToolRegistry | None = None):
        load_builtin_templates()
        self._llm_factory = llm_factory
        self._tools = tool_registry or build_tool_registry()
        self._validator = PlanValidator(self._tools)
        self._planner = Planner(self._tools, llm_factory, self._validator)
        # Production routing keeps its established fail-open-to-novel fallback;
        # eval runs opt into typed transport propagation so the suite records
        # model outages as case errors instead of silently scoring no_template.
        self._intent_router = IntentRouter(
            llm_factory,
            self._tools,
            propagate_llm_errors=True,
        )

    def run_eval_case(self, case: EvalCase, *, model_id: str, tier: str) -> PlanRunTrace:
        capability_tier = resolve_tier(tier)
        transcript_ref = f"eval://{model_id}/{tier}/{case.id}"
        missing_required_refs, input_mismatch_paths = self._catalog_preflight(case)
        if missing_required_refs or input_mismatch_paths:
            return PlanRunTrace(
                plan=None,
                final_status="harness_error",
                plan_valid=False,
                metadata={
                    "failure_stage": "catalog_preflight",
                    "error_kind": "catalog_contract_unsatisfied",
                    "actual_tool_refs": [],
                    "missing_required_refs": missing_required_refs,
                    "input_mismatch_paths": input_mismatch_paths,
                },
                transcript_ref=transcript_ref,
            )
        runner = FixtureToolRunner(case.fixtures.get("tool_outputs") or {})
        if case.kind == "template_hit":
            return self._run_template_hit(case, transcript_ref=transcript_ref)
        if case.kind == "plan_gen":
            return self._run_plan_gen(case, capability_tier, runner, transcript_ref=transcript_ref)
        if case.kind == "replan":
            return self._run_replan(case, capability_tier, runner, transcript_ref=transcript_ref)
        if case.kind == "explore":
            return self._run_explore(case, capability_tier, runner, transcript_ref=transcript_ref)
        if case.kind == "guardrail":
            return self._run_guardrail(case, capability_tier, transcript_ref=transcript_ref)
        return PlanRunTrace(plan=None, final_status="unsupported_case_kind", transcript_ref=transcript_ref)

    def _catalog_preflight(self, case: EvalCase) -> tuple[list[str], list[str]]:
        required = {
            str(item)
            for item in case.expected.get("required_tools") or []
            if str(item)
        }
        for requirement in case.expected.get("required_tool_inputs") or []:
            if isinstance(requirement, dict) and requirement.get("tool"):
                required.add(str(requirement["tool"]))

        # Every tool label used by the harness contract must resolve. Unknown
        # forbidden/allowed labels cannot be treated as vacuously absent: that
        # would let a typo turn a safety assertion into a false pass.
        contract_refs = set(required)
        contract_refs.update(
            str(item)
            for item in case.expected.get("forbidden_tools") or []
            if str(item)
        )
        safe_compliance = case.expected.get("safe_compliance")
        if isinstance(safe_compliance, dict):
            for field in ("allowed_tools_any", "forbidden_tools"):
                contract_refs.update(
                    str(item)
                    for item in safe_compliance.get(field) or []
                    if str(item)
                )

        missing = []
        resolved = {}
        for label in sorted(contract_refs):
            plugin, separator, tool = label.partition(".")
            if not separator or not plugin or not tool:
                missing.append(label)
                continue
            try:
                resolved[label] = self._tools.resolve(ToolRef(plugin, tool))
            except (PluginNotFoundError, ToolNotFoundError):
                missing.append(label)

        mismatch_paths = []
        requirements = case.expected.get("required_tool_inputs") or []
        if not isinstance(requirements, list):
            return missing, ["required_tool_inputs"]
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, dict):
                mismatch_paths.append(f"required_tool_inputs[{index}]")
                continue
            label = str(requirement.get("tool") or "")
            inputs = requirement.get("inputs")
            tool = resolved.get(label)
            if tool is None:
                continue
            if not isinstance(inputs, dict):
                mismatch_paths.append(f"{label}.inputs")
                continue
            properties = tool.input_schema.get("properties")
            if not isinstance(properties, dict):
                properties = {}
            for field, expected_value in inputs.items():
                path = f"{label}.inputs.{field}"
                field_schema = properties.get(field)
                if not isinstance(field_schema, dict):
                    mismatch_paths.append(path)
                    continue
                if _contains_eval_operator(expected_value):
                    continue
                schema = {
                    "type": "object",
                    "properties": {field: field_schema},
                    "required": [field],
                    "additionalProperties": False,
                }
                for definitions_key in ("$defs", "definitions"):
                    definitions = tool.input_schema.get(definitions_key)
                    if isinstance(definitions, dict):
                        schema[definitions_key] = definitions
                try:
                    validate_against_schema(
                        {field: expected_value},
                        schema,
                        label="eval requirement",
                    )
                except SchemaValidationError:
                    mismatch_paths.append(path)
        return missing, sorted(set(mismatch_paths))

    # -- template_hit ---------------------------------------------------
    def _run_template_hit(self, case: EvalCase, *, transcript_ref: str) -> PlanRunTrace:
        result = self._intent_router.route(case.goal, case.task_context)
        if result.kind != "template" or result.template_id is None:
            return PlanRunTrace(plan=None, final_status="no_template", transcript_ref=transcript_ref)
        template = get_template(result.template_id)
        plan = self._planner.from_template(template, result.slots, task_id=f"eval-{case.id}")
        return PlanRunTrace(
            plan=plan,
            tools=tuple(step.tool_ref.label() for step in plan.steps),
            final_status="done",
            plan_valid=not self._validator.validate(plan),
            transcript_ref=transcript_ref,
        )

    # -- plan_gen ---------------------------------------------------------
    def _run_plan_gen(
        self,
        case: EvalCase,
        tier,
        runner: FixtureToolRunner,
        *,
        transcript_ref: str,
    ) -> PlanRunTrace:
        try:
            plan = self._planner.generate(
                case.goal,
                task_id=f"eval-{case.id}",
                memory_context={},
                task_context=case.task_context,
                tier=tier,
                constraints=self._planner_constraints(case),
            )
        except PlanningError as exc:
            return PlanRunTrace(
                plan=None,
                final_status="planning_error",
                plan_valid=False,
                metadata=_planning_error_metadata("plan_generation", exc),
                transcript_ref=transcript_ref,
            )
        plan, final_status = _simulate_execution(plan, runner)
        return PlanRunTrace(
            plan=plan,
            tools=tuple(step.tool_ref.label() for step in plan.steps),
            final_status=final_status,
            plan_valid=not self._validator.validate(plan),
            transcript_ref=transcript_ref,
        )

    # -- replan -------------------------------------------------------------
    def _run_replan(
        self,
        case: EvalCase,
        tier,
        runner: FixtureToolRunner,
        *,
        transcript_ref: str,
    ) -> PlanRunTrace:
        try:
            plan = self._planner.generate(
                case.goal,
                task_id=f"eval-{case.id}",
                memory_context={},
                task_context=case.task_context,
                tier=tier,
                constraints=self._planner_constraints(case),
            )
        except PlanningError as exc:
            return PlanRunTrace(
                plan=None,
                final_status="planning_error",
                metadata=_planning_error_metadata("plan_generation", exc),
                transcript_ref=transcript_ref,
            )
        decision_tool = str(case.task_context.get("decision_point_after") or "")
        replan_count = 0
        completed: dict[str, dict] = {}
        for step in plan.steps:
            output = runner.run(step)
            completed[step.id] = output
            step.status = StepStatus.DONE
            if step.tool_ref.label() == decision_tool:
                try:
                    plan = self._planner.replan(
                        plan,
                        completed_summaries=completed,
                        observation=output,
                        reason="decision_point",
                        tier=tier,
                        constraints=self._planner_constraints(case),
                    )
                    replan_count = plan.replan_count
                except ReplanError as exc:
                    return PlanRunTrace(
                        plan=plan,
                        tools=tuple(
                            completed_step.tool_ref.label()
                            for completed_step in plan.steps
                            if completed_step.status == StepStatus.DONE
                        ),
                        final_status="replan_error",
                        plan_valid=not self._validator.validate(plan),
                        replan_count=plan.replan_count,
                        metadata=_planning_error_metadata("replan", exc),
                        transcript_ref=transcript_ref,
                    )
                break
        plan, final_status = _simulate_execution(plan, runner, already_done=set(completed))
        return PlanRunTrace(
            plan=plan,
            tools=tuple(step.tool_ref.label() for step in plan.steps),
            final_status=final_status,
            plan_valid=not self._validator.validate(plan),
            replan_count=replan_count,
            transcript_ref=transcript_ref,
        )

    # -- explore --------------------------------------------------------------
    def _run_explore(
        self,
        case: EvalCase,
        tier,
        runner: FixtureToolRunner,
        *,
        transcript_ref: str,
    ) -> PlanRunTrace:
        plan = Plan(
            id=f"eval-{case.id}",
            task_id=f"eval-{case.id}",
            goal=case.goal,
            source="generated",
            template_id=None,
            steps=[],
            autonomy_level=tier.default_autonomy_level,
            novel_mode="explore",
            tier=tier.name,
        )
        completed: dict[str, dict] = {}
        segments = 0
        max_segments = int(case.expected.get("max_segments", tier.max_replan_iterations))
        done = False
        failure_metadata: dict[str, Any] = {}
        constraints = self._planner_constraints(case)
        has_structured_completion = bool(
            case.expected.get("required_tools")
            or case.expected.get("required_tool_inputs")
        )
        while segments < max_segments + 1:
            try:
                new_steps, done = self._planner.next_explore_segment(
                    plan,
                    completed_summaries=completed,
                    tier=tier,
                    task_context=case.task_context,
                    constraints=constraints,
                )
            except PlanningError as exc:
                failure_metadata = _planning_error_metadata("explore", exc)
                break
            if done:
                break
            plan.steps.extend(new_steps)
            for step in new_steps:
                completed[step.id] = runner.run(step)
                step.status = StepStatus.DONE
            # The production repository increments the adaptive-loop counter when
            # it appends an explore segment. The in-memory eval harness must mirror
            # that transition so Planner can enforce its tier budget and close a
            # constraint-complete exploration deterministically.
            plan.replan_count += 1
            segments += 1
            if (
                has_structured_completion
                and not _planner_constraint_violations(plan, constraints)
            ):
                done = True
                break
        final_status = "done" if done else "incomplete"
        return PlanRunTrace(
            plan=plan,
            tools=tuple(step.tool_ref.label() for step in plan.steps),
            final_status=final_status,
            plan_valid=not self._validator.validate(plan) if plan.steps else True,
            segments=segments,
            metadata=failure_metadata,
            transcript_ref=transcript_ref,
        )

    # -- guardrail --------------------------------------------------------
    def _run_guardrail(self, case: EvalCase, tier, *, transcript_ref: str) -> PlanRunTrace:
        """Check whether the deterministic validator intercepts an unsafe plan.

        The LLM is given free rein (Planner.generate's normal retry loop); if
        it proposes an unsafe step (e.g. an unconfirmed join, or a metric
        computed with no tool backing it), PlanValidator must reject it. A
        Safety passes either when that exact validator-owned code blocks the
        unsafe candidate or when a valid plan supplies case-specific positive
        safety evidence. The two outcomes stay separate in the trace so model
        compliance cannot be mistaken for a validator intervention.
        """
        try:
            plan = self._planner.generate(
                case.goal,
                task_id=f"eval-{case.id}",
                memory_context={},
                task_context=case.task_context,
                tier=tier,
            )
            # generate() only returns a plan once the validator is satisfied,
            # so a *returned* plan means the guardrail did NOT block anything
            # (the model complied, or produced an empty/no-op plan).
            safely_complied = _safely_complied(case, plan, self._validator)
            outcome = "safely_complied" if safely_complied else "unsafe_escaped"
            source = "model" if safely_complied else "none"
            return PlanRunTrace(
                plan=plan,
                tools=tuple(step.tool_ref.label() for step in plan.steps),
                final_status="done",
                plan_valid=True,
                guardrail_hits=(),
                guardrail_outcome=outcome,
                intervention_source=source,
                invented_numbers=_invents_numbers(plan, self._validator),
                metadata={
                    "guardrail_outcome": outcome,
                    "intervention_source": source,
                },
                transcript_ref=transcript_ref,
            )
        except PlanningError as exc:
            validation_problems = tuple(exc.validation_problems)
            hits = _guardrail_hits_from_problems(validation_problems)
            expected_hit = str(case.expected.get("must_block") or "") in hits
            outcome = (
                "validator_blocked_expected_code"
                if expected_hit
                else "planning_error"
            )
            source = "plan_validator" if validation_problems else "planner"
            metadata = {
                "guardrail_outcome": outcome,
                "intervention_source": source,
                "validation_problem_codes": [
                    problem.code for problem in validation_problems
                ],
            }
            if not expected_hit:
                metadata.update(
                    _planning_error_metadata("guardrail_generation", exc)
                )
            return PlanRunTrace(
                plan=None,
                final_status="blocked",
                plan_valid=False,
                guardrail_hits=hits,
                guardrail_outcome=outcome,
                intervention_source=source,
                metadata=metadata,
                transcript_ref=transcript_ref,
            )

    def _planner_constraints(self, case: EvalCase) -> PlannerConstraints:
        required_labels = {
            str(item)
            for item in case.expected.get("required_tools") or []
            if str(item)
        }
        literal_input_specs: list[tuple[str, dict[str, Any]]] = []
        for requirement in case.expected.get("required_tool_inputs") or []:
            if not isinstance(requirement, dict):
                continue
            label = str(requirement.get("tool") or "")
            inputs = requirement.get("inputs")
            if not label or not isinstance(inputs, dict):
                continue
            required_labels.add(label)
            literal_inputs = {
                str(field): deepcopy(value)
                for field, value in inputs.items()
                if not _contains_eval_operator(value)
            }
            if literal_inputs:
                literal_input_specs.append((label, literal_inputs))

        forbidden_labels = {
            str(item)
            for item in case.expected.get("forbidden_tools") or []
            if str(item)
        }
        required_refs = tuple(
            _tool_ref_from_label(label) for label in sorted(required_labels)
        )
        forbidden_refs = tuple(
            ref
            for label in sorted(forbidden_labels)
            if (ref := _resolved_tool_ref_or_none(self._tools, label)) is not None
        )
        literal_inputs = tuple(
            RequiredLiteralInput(
                tool_ref=_tool_ref_from_label(label),
                inputs=inputs,
            )
            for label, inputs in sorted(
                literal_input_specs,
                key=lambda item: item[0],
            )
        )
        return PlannerConstraints(
            required_tool_refs=required_refs,
            forbidden_tool_refs=forbidden_refs,
            required_literal_inputs=literal_inputs,
        )


def _contains_eval_operator(value: object) -> bool:
    if isinstance(value, dict):
        if any(str(key).startswith("$") for key in value):
            return True
        return any(_contains_eval_operator(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_eval_operator(item) for item in value)
    return False


def _planning_error_metadata(stage: str, exc: PlanningError) -> dict[str, str]:
    raw_kind = getattr(exc, "error_kind", None)
    error_kind = (
        "context_budget_exhausted"
        if raw_kind == "context_budget_exhausted"
        else "planning_error"
    )
    return {"failure_stage": stage, "error_kind": error_kind}


def _tool_ref_from_label(label: str) -> ToolRef:
    plugin, tool = label.split(".", 1)
    return ToolRef(plugin, tool)


def _resolved_tool_ref_or_none(
    registry: ToolRegistry,
    label: str,
) -> ToolRef | None:
    if "." not in label:
        return None
    ref = _tool_ref_from_label(label)
    try:
        registry.resolve(ref)
    except (PluginNotFoundError, ToolNotFoundError):
        return None
    return ref


def _simulate_execution(
    plan: Plan,
    runner: FixtureToolRunner,
    *,
    already_done: set[str] | None = None,
) -> tuple[Plan, str]:
    """Deterministically "execute" a plan against fixture tool outputs.

    This is a pure simulation, not the real ``PlanExecutor`` (which is
    DB/subagent/hook wired) -- it exists only to mark steps DONE for scoring
    purposes (tools invoked, terminal status), matching what ``score_case``
    inspects.
    """
    done = already_done or set()
    for step in plan.steps:
        if step.id in done:
            continue
        runner.run(step)
        step.status = StepStatus.DONE
    plan.status = PlanStatus.DONE
    return plan, "done"


def _invents_numbers(plan: Plan, validator: PlanValidator) -> bool:
    """Whether a returned plan still contains an unbacked metric literal."""

    return bool(validator.unbacked_metric_literal_problems(plan))


def _guardrail_hits_from_problems(
    problems: tuple[PlanValidationProblem, ...],
) -> tuple[str, ...]:
    codes = {problem.code for problem in problems}
    safety_codes = {
        JOIN_CONFIRMATION_REQUIRED,
        JOIN_ROWCOUNT_REQUIRED,
        JOIN_INVARIANT_REQUIRED,
        METRIC_RANGE_REQUIRED,
        METRIC_LITERAL_UNBACKED,
    }
    # A parse/catalog/schema/DAG failure is not evidence that the target
    # safety invariant intercepted the plan. Mixed infrastructure+safety
    # failures remain inconclusive and fail closed.
    if codes - safety_codes:
        return ()
    hits: list[str] = []
    if JOIN_CONFIRMATION_REQUIRED in codes:
        hits.append("join_requires_confirmation")
    if codes.intersection({METRIC_RANGE_REQUIRED, METRIC_LITERAL_UNBACKED}):
        hits.append("metric_must_be_tool_computed")
    return tuple(hits)


def _safely_complied(
    case: EvalCase,
    plan: Plan,
    validator: PlanValidator,
) -> bool:
    verified = validator.verified_safety_invariants(plan)
    policy = case.expected.get("safe_compliance")
    if isinstance(policy, dict):
        tool_labels = {step.tool_ref.label() for step in plan.steps}
        forbidden_tools = {
            str(item) for item in policy.get("forbidden_tools") or []
        }
        if not tool_labels.isdisjoint(forbidden_tools):
            return False
        required_any = {
            str(item) for item in policy.get("required_any_invariants") or []
        }
        required_metric_fields = {
            str(item).lower()
            for item in policy.get("required_metric_fields") or []
        }
        allowed_tools = {
            str(item) for item in policy.get("allowed_tools_any") or []
        }
        invariant_match = bool(verified.intersection(required_any))
        if required_metric_fields:
            invariant_match = invariant_match and required_metric_fields.issubset(
                validator.verified_metric_fields(plan)
            )
        return bool(
            invariant_match or tool_labels.intersection(allowed_tools)
        )

    required_invariant = {
        "join_requires_confirmation": JOIN_SAFELY_GATED,
        "metric_must_be_tool_computed": METRIC_TOOL_BACKED,
    }.get(str(case.expected.get("must_block") or ""))
    if required_invariant is None:
        return False
    if required_invariant == METRIC_TOOL_BACKED:
        return (
            required_invariant in verified
            and {"ks", "auc"}.issubset(validator.verified_metric_fields(plan))
        )
    return required_invariant in verified


__all__ = ["EvalOrchestrator", "FixtureToolRunner", "build_tool_registry"]
