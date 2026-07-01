"""Generic plan-conversation driver — one driver for all V2 task types.

See docs/plans/v2-plan-driver-spec.md. Given a task's template + filled slots,
the driver builds a plan, runs it on the real PlanExecutor, and at each
``needs_confirmation`` gate turns the *just-computed prior-step output* into an
append-only assistant message (with inline rich tables). The executor pauses
BEFORE the gate step, so what the user confirms is exactly what just ran.
Confirm resumes execution; task differences live in the template + the
tool->table registry below, not in the driver. This replaces the bespoke
``ModelingSession`` / ``modeling_agent`` prototype (decision #9 / #4).

The driver is deliberately pure-ish: it mutates plan state through the repo and
the executor, but it *returns* the assistant messages rather than persisting
them, so the API/job layer owns ``agent_messages`` and the driver stays unit
testable offline.
"""

from __future__ import annotations

import re

from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.agent.gate_execution_adapter import GateExecutionAdapter
from marvis.agent.gate_response_adapter import GateControlValidationError, validate_gate_control
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.templates import get_template

# A reply counts as confirmation of the current gate.
_CONFIRM = re.compile(
    r"(确认|确定|没问题|可以|就这样|同意|好的|继续|开始|对的?|ok|yes|go|proceed|looks good|sounds good)",
    re.IGNORECASE,
)
_NEGATED_CONFIRM = re.compile(
    r"(先别|别执行|别继续|别开始|不要|不用|不需要|不执行|不继续|先不|暂不|暂停|停止|取消|"
    r"不开始|不确认|不可以|hold on|do\s*not|don't|dont|not\s+(start|continue|proceed|go)|stop|cancel|wait)",
    re.IGNORECASE,
)
def is_confirm(text: str) -> bool:
    raw = text or ""
    if _NEGATED_CONFIRM.search(raw):
        return False
    if not _CONFIRM.search(raw):
        return False
    compact = re.sub(r"\s+", "", raw)
    return not _NEGATED_CONFIRM.search(compact)


def _parse_dedup_instruction(text: str) -> str | None:
    """Parse a manual-mode dedup reply at a join gate → "first"/"last"/None.

    Recognised only when the text actually mentions de-duplication (去重/dedup/策略/保留)
    so an unrelated instruction isn't misread as a strategy. first = keep the first row per
    key, last = keep the last (spec §6 conflict resolution)."""
    low = (text or "").lower()
    if re.search(r"(别|不要|不用|无需|不需要|勿|取消|暂停|停止|do\s*not|don't|dont|not\s+use)", text or "", re.IGNORECASE):
        return None
    if not any(token in low for token in ("去重", "dedup", "策略", "保留", "重复")):
        return None
    if "first" in low or "首" in text or "第一" in text or "前" in text:
        return "first"
    if "last" in low or "末" in text or "最后" in text or "最新" in text or "后" in text:
        return "last"
    return None


class DriverError(Exception):
    pass


class PlanDriver:
    def __init__(self, plan_repo, executor, *, planner=None, validator=None, llm_client=None):
        self._repo = plan_repo
        self._executor = executor
        self._planner = planner
        self._validator = validator
        # Optional LLM for agent-mode free-text gate instructions (adjust / replan).
        # None in manual mode — non-confirm replies then show the canned hint.
        self._llm = llm_client
        self._composer = PlanMessageComposer(
            load_output=self._safe_output,
            latest_failed_step_run_error_kind=self._latest_failed_step_run_error_kind,
        )
        self._gate_execution = GateExecutionAdapter(
            self._repo,
            self._executor,
            safe_output=self._safe_output,
            run_and_handle=self._run_and_handle,
            plan_overview_message=self._composer.plan_overview_message,
        )

    # -- entry points ---------------------------------------------------------
    def start(self, *, task_id, template_id, slots, autonomy=None, tier=None, run_seq=0) -> DriverTurn:
        """Build the plan and show its overview, then PAUSE at the plan-level 开始 gate.

        Spec §9 #2 (已锁): both modes first show the whole plan and only run after the
        user confirms 「开始」. The plan is left VALIDATED — nothing executes until
        resume() receives the 开始 confirm (the agent auto-driver feeds it in AUTO
        mode). This is what makes the first analysis step never run unprompted.
        """
        plan = self.build_plan(
            task_id=task_id, template_id=template_id, slots=slots, autonomy=autonomy, tier=tier
        )
        return DriverTurn(plan.id, plan.status.value, [self._composer.plan_overview_message(plan)])

    def resume(
        self,
        *,
        plan_id,
        user_text,
        run_seq=0,
        selection=None,
        dedup_strategies=None,
        adjust_params=None,
        expected_step_id=None,
    ) -> DriverTurn:
        """Advance the plan given a user reply. Two gate kinds are handled: the
        plan-level overview gate (plan not yet started) and per-step gates.

        ``selection`` (optional): the user's edited feature set from the §4 interactive
        screening table. When confirming a gate that depends on a ``screen_features``
        step, it overrides that step's proposed ``selected`` so downstream steps
        (``$ref:...output.selected``) train on exactly the features the user chose.

        ``dedup_strategies`` (optional): the user's per-feature dedup strategy map from
        the §4 join dedup picker. At a join gate it re-confirms the ``confirm_join``
        dependency with those strategies (resolving non-unique-key conflicts) and
        re-pauses at the gate, now clear, for the final execute confirm.

        ``adjust_params`` (optional): structured manual control overrides. Unlike
        free-text instructions, these do not require an LLM router.
        """
        plan = self._repo.load_plan(plan_id)
        # Plan-level overview gate: nothing has run yet → 「开始」 begins execution.
        if plan.status == PlanStatus.VALIDATED:
            if is_confirm(user_text):
                self._repo.confirm_plan(plan_id)  # VALIDATED -> CONFIRMED so the executor runs
                return self._run_and_handle(plan_id, run_seq=run_seq)
            return self._handle_instruction(plan, None, user_text, run_seq)
        # Per-step needs_confirmation gate.
        gate = self._awaiting_step(plan)
        try:
            validate_gate_control(
                plan,
                gate,
                expected_step_id=expected_step_id,
                selection=selection,
                dedup_strategies=dedup_strategies,
                adjust_params=adjust_params,
            )
        except GateControlValidationError as exc:
            raise DriverError(str(exc)) from exc
        # Join dedup picker: re-confirm with the chosen strategies, then re-pause at the
        # (now conflict-free) gate — do NOT confirm-execute yet; the user confirms after.
        if dedup_strategies and gate is not None:
            self._gate_execution.apply_dedup_strategies(plan, gate, dedup_strategies)
            return self._run_and_handle(plan_id, run_seq=run_seq)
        if adjust_params and gate is not None:
            return self._gate_execution.apply_adjust(plan, gate, adjust_params, run_seq)
        if is_confirm(user_text):
            if gate is not None:
                if selection is not None:
                    self._gate_execution.apply_screen_selection(plan, gate, selection)
                self._repo.confirm_step(gate.id)
            return self._run_and_handle(plan_id, run_seq=run_seq)
        # Manual-mode TEXT resolution of a same-key dedup conflict (no §4 picker available):
        # the user replies e.g. 「去重 first」/「用 last 去重」 → apply that strategy to every
        # feature confirm_join flagged as needs_dedup, then re-pause at the cleared gate.
        if gate is not None:
            strategy = _parse_dedup_instruction(user_text)
            if strategy:
                pending = self._gate_execution.needs_dedup_features(plan, gate)
                if pending:
                    self._gate_execution.apply_dedup_strategies(plan, gate, {fid: strategy for fid in pending})
                    return self._run_and_handle(plan_id, run_seq=run_seq)
        return self._handle_instruction(plan, gate, user_text, run_seq)

    def _handle_instruction(self, plan, gate, user_text, run_seq) -> DriverTurn:
        """Route a non-confirm reply. Manual mode (no LLM) shows the canned hint;
        agent mode classifies the instruction into confirm / adjust / replan / clarify
        and acts on it (spec §3 提指令→调整/重规划)."""
        if self._llm is None:
            return self._adjust_placeholder(plan.id, gate, run_seq)
        context = gate.title if gate is not None else "计划总览(尚未开始执行)"
        route = route_instruction(self._llm, gate_context=context, instruction=user_text)
        action = route["action"]
        if action == "confirm":
            if plan.status == PlanStatus.VALIDATED:
                self._repo.confirm_plan(plan.id)
            elif gate is not None:
                self._repo.confirm_step(gate.id)
            return self._run_and_handle(plan.id, run_seq=run_seq)
        if action == "adjust" and gate is not None and gate.depends_on:
            return self._gate_execution.apply_adjust(plan, gate, route["params"], run_seq)
        if action == "replan":
            return self._gate_execution.apply_replan(plan, gate, user_text, run_seq)
        return DriverTurn(
            plan.id,
            plan.status.value,
            [
                self._composer.instruction_message(
                    plan,
                    gate,
                    run_seq=run_seq,
                    text=route.get("reason") or "请明确指令:回复「确认」继续,或说明要调整的参数。",
                )
            ],
        )

    def _adjust_placeholder(self, plan_id, gate, run_seq) -> DriverTurn:
        # Manual mode (no LLM): non-confirm free text can only show the canned hint.
        plan = self._repo.load_plan(plan_id)
        return DriverTurn(
            plan_id,
            plan.status.value,
            [self._composer.manual_adjust_placeholder_message(plan, gate, run_seq=run_seq)],
        )

    # -- plan build -----------------------------------------------------------
    def build_plan(self, *, task_id, template_id, slots, autonomy=None, tier=None) -> Plan:
        if self._planner is None:
            raise DriverError("driver has no planner to build plans")
        plan = self._planner.from_template(
            get_template(template_id), dict(slots), task_id, autonomy=autonomy
        )
        if tier:
            plan.tier = tier
        if self._validator is not None:
            problems = self._validator.validate(plan)
            if problems:
                raise DriverError(f"plan failed validation: {problems}")
        plan.status = PlanStatus.VALIDATED
        self._repo.create_plan(plan)
        return plan

    # -- core loop ------------------------------------------------------------
    def _run_and_handle(self, plan_id, *, run_seq) -> DriverTurn:
        result = self._executor.run(plan_id)
        plan = self._repo.load_plan(plan_id)
        status = result.status
        if status == PlanStatus.AWAITING_CONFIRM:
            gate = self._awaiting_step(plan)
            return DriverTurn(plan_id, status.value, [self._composer.gate_message(plan, gate, run_seq=run_seq)])
        if status == PlanStatus.DONE:
            return DriverTurn(plan_id, status.value, [self._composer.done_message(plan, run_seq=run_seq)])
        if status == PlanStatus.REVIEW:
            return DriverTurn(plan_id, status.value, [self._composer.review_message(plan, run_seq=run_seq)])
        return DriverTurn(plan_id, status.value, [self._composer.failed_message(plan, run_seq=run_seq)])

    @staticmethod
    def _awaiting_step(plan: Plan) -> PlanStep | None:
        for step in sorted(plan.steps, key=lambda s: (s.index, s.id)):
            if step.status == StepStatus.AWAITING_CONFIRM:
                return step
        return None

    def _safe_output(self, step_id: str):
        try:
            return self._repo.load_step_output(step_id)
        except KeyError:
            return None

    def _latest_failed_step_run_error_kind(self, step_id: str) -> str | None:
        latest_error_kind = getattr(self._repo, "latest_failed_step_run_error_kind", None)
        if callable(latest_error_kind):
            return latest_error_kind(step_id)
        return None
__all__ = [
    "PlanDriver",
    "DriverMessage",
    "DriverTurn",
    "DriverError",
    "is_confirm",
    "render_tool_output",
]
