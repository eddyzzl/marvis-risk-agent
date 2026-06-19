from __future__ import annotations

import time
import uuid

from marvis.orchestrator.contracts import AgentStatus, PlanStatus, PlanStep, SubAgent
from marvis.orchestrator.templates import get_template
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.errors import ToolNotFoundError
from marvis.plugins.manifest import ToolRef
from marvis.plugins.runner import ToolResult


DEFAULT_SUBAGENT_BUDGET = 8192


class RestrictedToolRegistry:
    def __init__(self, base_registry, granted_tools: list[ToolRef]):
        self._base = base_registry
        self._grants = tuple(granted_tools)

    def resolve(self, ref: ToolRef):
        _manifest, tool = self.resolve_with_manifest(ref)
        return tool

    def resolve_with_manifest(self, ref: ToolRef):
        grant = self._matching_grant(ref)
        if grant is None:
            raise ToolNotFoundError(f"{ref.label()} is not granted to sub-agent")
        manifest, tool = self._base.resolve_with_manifest(ref)
        if grant.version and manifest.version != grant.version:
            raise ToolNotFoundError(
                f"{ref.label()} version {manifest.version} is not granted to sub-agent"
            )
        return manifest, tool

    def catalog_for_planner(self) -> list[dict]:
        catalog = []
        for item in self._base.catalog_for_planner():
            if self._matching_catalog_item(item) is not None:
                catalog.append(item)
        return catalog

    def _matching_grant(self, ref: ToolRef) -> ToolRef | None:
        for grant in self._grants:
            if grant.plugin != ref.plugin or grant.tool != ref.tool:
                continue
            if grant.version and ref.version and grant.version != ref.version:
                continue
            return grant
        return None

    def _matching_catalog_item(self, item: dict) -> ToolRef | None:
        for grant in self._grants:
            if grant.plugin != item.get("plugin") or grant.tool != item.get("tool"):
                continue
            if grant.version and grant.version != item.get("version"):
                continue
            return grant
        return None


class SubAgentDispatcher:
    def __init__(
        self,
        plan_repo,
        planner,
        executor_factory,
        tool_registry,
        intent_router,
        planner_factory=None,
    ):
        self._repo = plan_repo
        self._planner = planner
        self._executor_factory = executor_factory
        self._tools = tool_registry
        self._intent_router = intent_router
        self._planner_factory = planner_factory

    def spawn(self, step: PlanStep, *, parent_task_id: str) -> SubAgent:
        if not step.sub_agent_scope:
            raise ValueError("sub_agent_scope is required")
        if not step.granted_tools:
            raise ValueError("granted_tools must not be empty")

        sub = SubAgent(
            id=uuid.uuid4().hex,
            parent_task_id=parent_task_id,
            parent_step_id=step.id,
            scope=step.sub_agent_scope,
            granted_tools=list(step.granted_tools),
            context_budget=DEFAULT_SUBAGENT_BUDGET,
            status=AgentStatus.SPAWNED,
        )
        self._repo.upsert_sub_agent(sub)
        self._repo.write_audit(
            kind="subagent.spawn",
            target_ref=sub.id,
            outcome="succeeded",
            detail={
                "parent_step_id": step.id,
                "scope": sub.scope,
                "tools": [ref.label() for ref in sub.granted_tools],
            },
        )
        return sub

    def run(self, sub: SubAgent, *, goal_inputs: dict) -> ToolResult:
        started = time.monotonic()
        try:
            self._repo.set_sub_agent_status(sub.id, AgentStatus.RUNNING)
            restricted = RestrictedToolRegistry(self._tools, sub.granted_tools)
            planner = self._planner_factory(restricted) if self._planner_factory else self._planner
            intent = self._intent_router.route(sub.scope, goal_inputs)
            if intent.kind == "template":
                template = get_template(intent.template_id)
                mini_plan = planner.from_template(
                    template,
                    intent.slots,
                    sub.parent_task_id,
                )
            else:
                mini_plan = planner.generate(
                    sub.scope,
                    sub.parent_task_id,
                    memory_context={},
                    task_context=goal_inputs,
                )

            problems = PlanValidator(restricted).validate(mini_plan)
            if problems:
                raise RuntimeError("; ".join(problems))

            mini_plan.status = PlanStatus.CONFIRMED
            self._repo.create_plan(mini_plan)
            execution = self._executor_factory(restricted).run(mini_plan.id)
            result_ref = execution.summary_ref
            self._repo.set_sub_agent_status(
                sub.id,
                AgentStatus.RETURNED,
                result_ref=result_ref,
            )
            self._repo.write_audit(
                kind="subagent.run",
                target_ref=sub.id,
                outcome="succeeded",
                detail={"plan_id": mini_plan.id, "result_ref": result_ref},
            )
            return ToolResult(
                ok=True,
                output={"result_ref": result_ref},
                error=None,
                error_kind=None,
                duration_ms=_duration_ms(started),
            )
        except Exception as exc:
            self._repo.set_sub_agent_status(sub.id, AgentStatus.FAILED)
            self._repo.write_audit(
                kind="subagent.run",
                target_ref=sub.id,
                outcome="failed",
                detail={"error": str(exc)},
            )
            return ToolResult(
                ok=False,
                output=None,
                error=str(exc),
                error_kind="execution",
                duration_ms=_duration_ms(started),
            )


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
