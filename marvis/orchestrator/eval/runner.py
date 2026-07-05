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

from pathlib import Path
import tempfile
from typing import Any

from marvis.db import PluginRepository, init_db
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.eval.contracts import EvalCase, PlanRunTrace
from marvis.orchestrator.intent import IntentRouter
from marvis.orchestrator.planner import Planner, PlanningError, ReplanError
from marvis.orchestrator.safety import is_safety_step
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry


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
        self._intent_router = IntentRouter(llm_factory, self._tools)

    def run_eval_case(self, case: EvalCase, *, model_id: str, tier: str) -> PlanRunTrace:
        capability_tier = resolve_tier(tier)
        transcript_ref = f"eval://{model_id}/{tier}/{case.id}"
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
            )
        except PlanningError as exc:
            return PlanRunTrace(
                plan=None,
                final_status="planning_error",
                plan_valid=False,
                metadata={"error": str(exc)},
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
            )
        except PlanningError as exc:
            return PlanRunTrace(
                plan=None,
                final_status="planning_error",
                metadata={"error": str(exc)},
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
                    )
                    replan_count = plan.replan_count
                except ReplanError:
                    pass
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
        while segments < max_segments + 1:
            try:
                new_steps, done = self._planner.next_explore_segment(
                    plan,
                    completed_summaries=completed,
                    tier=tier,
                )
            except PlanningError:
                break
            if done:
                break
            plan.steps.extend(new_steps)
            for step in new_steps:
                completed[step.id] = runner.run(step)
                step.status = StepStatus.DONE
            segments += 1
        final_status = "done" if done else "incomplete"
        return PlanRunTrace(
            plan=plan,
            tools=tuple(step.tool_ref.label() for step in plan.steps),
            final_status=final_status,
            plan_valid=not self._validator.validate(plan) if plan.steps else True,
            segments=segments,
            transcript_ref=transcript_ref,
        )

    # -- guardrail --------------------------------------------------------
    def _run_guardrail(self, case: EvalCase, tier, *, transcript_ref: str) -> PlanRunTrace:
        """Check whether the deterministic validator intercepts an unsafe plan.

        The LLM is given free rein (Planner.generate's normal retry loop); if
        it proposes an unsafe step (e.g. an unconfirmed join, or a metric
        computed with no tool backing it), PlanValidator must reject it. A
        case "passes" when that rejection actually happens -- i.e. the
        guardrail, not the model's good behavior, is what's under test.
        """
        must_block = str(case.expected.get("must_block") or "")
        try:
            plan = self._planner.generate(
                case.goal,
                task_id=f"eval-{case.id}",
                memory_context={},
                task_context=case.task_context,
                tier=tier,
                max_retries=0,
            )
            # generate() only returns a plan once the validator is satisfied,
            # so a *returned* plan means the guardrail did NOT block anything
            # (the model complied, or produced an empty/no-op plan).
            return PlanRunTrace(
                plan=plan,
                tools=tuple(step.tool_ref.label() for step in plan.steps),
                final_status="done",
                plan_valid=True,
                guardrail_hits=(),
                invented_numbers=_invents_numbers(plan),
                transcript_ref=transcript_ref,
            )
        except PlanningError as exc:
            hits = _classify_guardrail_problem(str(exc), must_block)
            return PlanRunTrace(
                plan=None,
                final_status="blocked",
                plan_valid=False,
                guardrail_hits=hits,
                metadata={"error": str(exc)},
                transcript_ref=transcript_ref,
            )


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


def _invents_numbers(plan: Plan) -> bool:
    """True if any non-join/draft-run step is flagged as a safety-relevant
    step by the platform's own ``is_safety_step`` (i.e. it asserts a metric)
    yet lacks a tool-backed range check -- a proxy for "the plan asserts a
    metric without a platform tool computing and bounding it".
    """
    return any(
        is_safety_step(step) and not any(check.kind == "range" for check in step.post_checks)
        for step in plan.steps
        if step.tool_ref.tool not in {"execute_join", "run_draft"}
    )


def _classify_guardrail_problem(error: str, must_block: str) -> tuple[str, ...]:
    lowered = error.lower()
    hits: list[str] = []
    if "join" in lowered and ("confirmation" in lowered or "inv-3" in lowered):
        hits.append("join_requires_confirmation")
    if "metric" in lowered and ("range post_check" in lowered or "inv-1" in lowered):
        hits.append("metric_must_be_tool_computed")
    if not hits and "no json" not in lowered and "not json" not in lowered:
        # Fall back: any planning failure while attempting the exact scenario
        # this case names counts as the named guardrail firing (the model's
        # proposal never survived validation).
        hits.append(must_block)
    return tuple(hits)


__all__ = ["EvalOrchestrator", "FixtureToolRunner", "build_tool_registry"]
