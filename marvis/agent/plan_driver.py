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
from marvis.agent.gates import build_failure_envelope, extract_gate_envelope
from marvis.agent.gate_adapters import render_gate_dependencies
from marvis.agent.gate_execution_adapter import GateExecutionAdapter
from marvis.agent.gate_payloads import build_model_delivery_payload
from marvis.agent.gate_response_adapter import GateControlValidationError, validate_gate_control
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_utils import downstream_step_ids as _downstream_step_ids
from marvis.agent.plan_utils import find_step as _find_step
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
        self._gate_execution = GateExecutionAdapter(
            self._repo,
            self._executor,
            safe_output=self._safe_output,
            run_and_handle=self._run_and_handle,
            plan_overview_message=self._plan_overview_message,
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
        return DriverTurn(plan.id, plan.status.value, [self._plan_overview_message(plan)])

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
        return self._instruction_message(
            plan, gate, run_seq,
            route.get("reason") or "请明确指令:回复「确认」继续,或说明要调整的参数。",
        )

    def _instruction_message(self, plan, gate, run_seq, text) -> DriverTurn:
        return DriverTurn(
            plan.id,
            plan.status.value,
            [DriverMessage("gate", text, {"plan_id": plan.id, "step_id": gate.id if gate else None, "run_seq": run_seq})],
        )

    def _adjust_placeholder(self, plan_id, gate, run_seq) -> DriverTurn:
        # Manual mode (no LLM): non-confirm free text can only show the canned hint.
        plan = self._repo.load_plan(plan_id)
        return DriverTurn(
            plan_id,
            plan.status.value,
            [
                DriverMessage(
                    "gate",
                    "收到。确认当前结果请回复「确认」继续。",
                    {"plan_id": plan_id, "step_id": gate.id if gate else None, "run_seq": run_seq},
                )
            ],
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
            return DriverTurn(plan_id, status.value, [self._compose_gate_message(plan, gate, run_seq=run_seq)])
        if status == PlanStatus.DONE:
            return DriverTurn(plan_id, status.value, [self._compose_done_message(plan, run_seq=run_seq)])
        if status == PlanStatus.REVIEW:
            return DriverTurn(plan_id, status.value, [self._compose_review_message(plan, run_seq=run_seq)])
        return DriverTurn(plan_id, status.value, [self._compose_failed_message(plan, run_seq=run_seq)])

    @staticmethod
    def _awaiting_step(plan: Plan) -> PlanStep | None:
        for step in sorted(plan.steps, key=lambda s: (s.index, s.id)):
            if step.status == StepStatus.AWAITING_CONFIRM:
                return step
        return None

    # -- message composition --------------------------------------------------
    def _plan_overview_message(self, plan: Plan) -> DriverMessage:
        order: list[str] = []
        by_phase: dict[str, list[str]] = {}
        for step in plan.steps:
            phase = step.phase or "步骤"
            if phase not in by_phase:
                by_phase[phase] = []
                order.append(phase)
            by_phase[phase].append(step.title)
        lines = ["我已生成执行计划,会在每个关键节点停下与你确认:"]
        for phase in order:
            lines.append(f"**{phase}**:{' → '.join(by_phase[phase])}")
        lines.append("确认「开始」后按计划执行。")
        meta = {"plan_id": plan.id, "kind": "plan_overview"}
        meta["gate_envelope"] = extract_gate_envelope({"metadata": meta}).to_dict()
        return DriverMessage(
            "plan_overview", "\n".join(lines), meta
        )

    def _compose_gate_message(self, plan: Plan, gate: PlanStep | None, *, run_seq) -> DriverMessage:
        rendered = render_gate_dependencies(plan, gate, self._safe_output)
        parts = rendered.parts
        if not parts:
            parts.append("上一步已完成。")
        parts.append("确认请回复「确认」继续;要调整可直接说明。")
        meta = {
            "plan_id": plan.id,
            "step_id": gate.id if gate else None,
            "run_seq": run_seq,
            "tables": rendered.tables,
            "kind": "gate",  # marks a needs-confirmation gate (manual-mode confirm control)
        }
        if rendered.output_refs:
            meta["output_refs"] = rendered.output_refs
        if rendered.screen is not None:
            meta["screen"] = rendered.screen
        if rendered.dedup is not None:
            meta["dedup"] = rendered.dedup
        if rendered.modeling_setup is not None:
            meta["modeling_setup"] = rendered.modeling_setup
        if rendered.model_delivery is not None:
            meta["model_delivery"] = rendered.model_delivery
        meta["gate_envelope"] = extract_gate_envelope({"metadata": meta}).to_dict()
        return DriverMessage("gate", "\n\n".join(parts), meta)

    def _compose_done_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        terminal = max(
            (s for s in plan.steps if s.status == StepStatus.DONE and s.output_ref),
            key=lambda s: s.index,
            default=None,
        )
        parts = ["✅ 计划已全部完成。"]
        tables: list[dict] = []
        output = None
        if terminal is not None:
            output = self._safe_output(terminal.id)
            if output is not None:
                text, tables = render_tool_output(terminal.tool_ref.tool, output)
                if text:
                    parts.append(text)
        meta = {"plan_id": plan.id, "run_seq": run_seq, "tables": tables}
        if terminal is not None and output is not None:
            report_output, report_step = self._report_dependency_output(plan, terminal)
            delivery = build_model_delivery_payload(
                output,
                terminal,
                report_output=report_output,
                report_step=report_step,
            )
            if delivery is not None:
                meta["model_delivery"] = delivery
        return DriverMessage("done", "\n\n".join(parts), meta)

    def _report_dependency_output(self, plan: Plan, step: PlanStep) -> tuple[dict | None, PlanStep | None]:
        for dep_id in step.depends_on or []:
            dep = _find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "generate_model_report":
                continue
            output = self._safe_output(dep.id)
            return (output if isinstance(output, dict) else None), dep
        return None, None

    def _compose_review_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        return DriverMessage(
            "review",
            "计划已执行完,但结果需要你复核一下再定论。",
            {"plan_id": plan.id, "run_seq": run_seq},
        )

    def _compose_failed_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        failed = next((s for s in plan.steps if s.status == StepStatus.FAILED), None)
        detail = f"「{failed.title}」失败:{failed.error}" if failed and failed.error else "执行中断。"
        meta = {"plan_id": plan.id, "step_id": failed.id if failed else None, "run_seq": run_seq}
        reset_steps: tuple[str, ...] = ()
        if failed is not None:
            downstream = _downstream_step_ids(plan, [failed.id])
            reset_steps = tuple(
                step.id
                for step in sorted(plan.steps, key=lambda item: (item.index, item.id))
                if step.id == failed.id or step.id in downstream
            )
        meta["failure_envelope"] = build_failure_envelope(
            plan_id=plan.id,
            step_id=failed.id if failed else None,
            run_seq=run_seq,
            message=detail,
            step_inputs=failed.inputs if failed else None,
            downstream_reset_steps=reset_steps,
            retryable=failed is not None,
        ).to_dict()
        return DriverMessage(
            "error",
            f"❌ {detail}",
            meta,
        )

    def _safe_output(self, step_id: str):
        try:
            return self._repo.load_step_output(step_id)
        except KeyError:
            return None
__all__ = [
    "PlanDriver",
    "DriverMessage",
    "DriverTurn",
    "DriverError",
    "is_confirm",
    "render_tool_output",
]
