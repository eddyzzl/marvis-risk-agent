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
from marvis.agent.gate_param_schema import gate_param_schema
from marvis.agent.gate_response_adapter import GateControlValidationError, validate_gate_control
from marvis.agent.gates.adapters import (
    GateReplyContext,
    gate_editable_input_schema,
    get_gate_adapter,
    monitoring_plain_confirm_error,
    parse_dedup_instruction as _parse_dedup_instruction,
    parse_monitoring_disposition as _parse_monitoring_disposition,
    parse_rule_selection_instruction as _parse_rule_selection_instruction,
)
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.renderers import render_tool_output
from marvis.governance.errors import AuthorizationError
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.templates import get_template
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason

# A reply counts as confirmation of the current gate only when, after stripping
# whitespace/punctuation, the *entire* remaining text is made up of short affirmative
# tokens (see _CONFIRM_TOKEN below). This full-string anchoring — rather than a
# substring `.search` over the raw reply — is what stops questions (“这样可以吗？”)
# and embedded/contrasting affirmatives (“结果不是很好的”, “好的地方是命中率，但…”)
# from being misread as confirmation (AGT-1 / H4).
_CONFIRM_TOKEN = r"(?:好的|好|可以|确认|确定|没问题|同意|就这样|继续|开始|对的|对|行|ok|okay|yes|y|go|proceed)"
_CONFIRM_FULLMATCH = re.compile(rf"(?:{_CONFIRM_TOKEN})+", re.IGNORECASE)
_CONFIRM_DIRECT_COMMANDS = {
    "开始数据处理",
    "开始特征分析",
    "开始风险分析",
    "开始建模",
    "开始模型开发",
    "开始模型验证",
    "开始策略开发",
    "确认采纳",
    "确认导出",
    "确认并导出",
    "接受矩阵",
    "接受并导出",
    "导出矩阵",
}
# Interrogative guard: any hard question mark/particle disqualifies a reply from
# being read as confirmation even if it also contains an affirmative token (e.g.
# “这样可以吗？”, “KS高吗，可以到0.3吗”, “行不行”). A trailing “吧” is handled
# after direct-command normalization so “请开始建模吧” can still confirm while
# “这样可以吧” stays chat.
_QUESTION = re.compile(
    r"[?？]|吗|行不行|可不可以|能不能|好不好|对不对|是不是|呢$",
    re.IGNORECASE,
)
_NEGATED_CONFIRM = re.compile(
    r"(先别|别执行|别继续|别开始|不要|不用|不需要|不执行|不继续|先不|暂不|暂停|停止|取消|"
    r"不开始|不确认|不可以|hold on|do\s*not|don't|dont|not\s+(start|continue|proceed|go)|stop|cancel|wait)",
    re.IGNORECASE,
)
_STRIP_PUNCT = re.compile(
    "[\\s" + "，。.!！~～、·；;:：" + chr(39) + chr(34)
    + "“”‘’()（）" + "\\-]+"
)
_CONFIRM_DIRECT_PREFIXES = ("好的", "好", "那", "请", "麻烦", "帮我", "先", "可以", "确认")
_CONFIRM_DIRECT_SUFFIXES = ("一下", "下", "吧", "了")
CONFIRMATION_SOURCE_HUMAN = "human"
CONFIRMATION_SOURCE_AUTO = "auto"
_CONFIRMATION_SOURCES = frozenset({
    CONFIRMATION_SOURCE_HUMAN,
    CONFIRMATION_SOURCE_AUTO,
})


def _strip_direct_confirm_affixes(value: str) -> str:
    content = value
    changed = True
    while changed:
        changed = False
        for prefix in _CONFIRM_DIRECT_PREFIXES:
            if content.startswith(prefix) and len(content) > len(prefix):
                content = content[len(prefix):]
                changed = True
        for suffix in _CONFIRM_DIRECT_SUFFIXES:
            if content.endswith(suffix) and len(content) > len(suffix):
                content = content[:-len(suffix)]
                changed = True
    return content


def is_confirm(text: str) -> bool:
    raw = text or ""
    compact = _STRIP_PUNCT.sub("", raw)
    direct_confirm = compact in _CONFIRM_DIRECT_COMMANDS or _strip_direct_confirm_affixes(compact) in _CONFIRM_DIRECT_COMMANDS
    if _QUESTION.search(raw):
        return False
    if compact.endswith("吧") and not direct_confirm:
        return False
    if _NEGATED_CONFIRM.search(raw):
        return False
    if not compact:
        return False
    if _NEGATED_CONFIRM.search(compact):
        return False
    if direct_confirm:
        return True
    return bool(_CONFIRM_FULLMATCH.fullmatch(compact))


def _has_adoption_reason_adjust(adjust_params) -> bool:
    return isinstance(adjust_params, dict) and "adoption_reason" in adjust_params


def _is_adoption_gate(gate: PlanStep | None) -> bool:
    return bool(
        gate is not None
        and gate.tool_ref is not None
        and gate.tool_ref.tool == "adopt_strategy"
    )


class DriverError(Exception):
    pass


def _normalize_confirmation_source(value: str) -> str:
    source = str(value or "").strip().lower()
    if source not in _CONFIRMATION_SOURCES:
        raise DriverError("确认来源无效，必须由平台标记为 human 或 auto。")
    return source


def _assert_source_may_operate_gate(gate: PlanStep | None, source: str) -> None:
    if source != CONFIRMATION_SOURCE_AUTO or gate is None:
        return
    policy = getattr(gate, "policy", None)
    if getattr(policy, "human_decision_gate", "none") == "required":
        raise DriverError("AUTO 不得操作或确认强制人工业务决策节点，请由人工继续。")


class PlanDriver:
    def __init__(
        self,
        plan_repo,
        executor,
        *,
        planner=None,
        validator=None,
        llm_client=None,
        governance_service=None,
        local_principal=None,
    ):
        self._repo = plan_repo
        self._executor = executor
        self._planner = planner
        self._validator = validator
        # Optional LLM for agent-mode free-text gate instructions (adjust / replan).
        # None in manual mode — non-confirm replies then show the canned hint.
        self._llm = llm_client
        self._governance = governance_service
        self._principal = local_principal
        artifact_repo = (
            TaskArtifactRepository(self._repo.db_path)
            if getattr(self._repo, "db_path", None) is not None
            else None
        )
        self._composer = PlanMessageComposer(
            load_output=self._safe_output,
            load_task_artifact=(
                artifact_repo.get_for_task if artifact_repo is not None else None
            ),
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
    def start(
        self,
        *,
        task_id,
        template_id,
        slots,
        autonomy=None,
        tier=None,
        run_seq=0,
        success_criteria=None,
    ) -> DriverTurn:
        """Build the plan and show its overview, then PAUSE at the plan-level 开始 gate.

        Spec §9 #2 (已锁): both modes first show the whole plan and only run after the
        user confirms 「开始」. The plan is left VALIDATED — nothing executes until
        resume() receives the 开始 confirm (the agent auto-driver feeds it in AUTO
        mode). This is what makes the first analysis step never run unprompted.

        ``success_criteria`` (optional, AGT-4): user/AUTO-supplied deterministic
        thresholds (e.g. [{"metric": "oot_ks", "min": 0.3, ...}]) layered on top of
        the template's own success_criteria (empty for the built-in modeling
        templates today). Only final_review's deterministic evaluation reads this —
        never a hard-coded platform default.
        """
        plan = self.build_plan(
            task_id=task_id,
            template_id=template_id,
            slots=slots,
            autonomy=autonomy,
            tier=tier,
            success_criteria=success_criteria,
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
        confirmation_source=CONFIRMATION_SOURCE_HUMAN,
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
        confirmation_source = _normalize_confirmation_source(confirmation_source)
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
        # Defense in depth: AUTO's decision layer normally halts before this
        # call.  The driver still owns the final confirmation boundary so a
        # direct/internal caller cannot bypass the canonical step policy.
        _assert_source_may_operate_gate(gate, confirmation_source)
        if (
            confirmation_source == CONFIRMATION_SOURCE_AUTO
            and gate is not None
            and self._requires_governed_human_decision(gate)
        ):
            raise DriverError("AUTO 不得操作或确认强制人工业务决策节点，请由人工继续。")
        # Join dedup picker: re-confirm with the chosen strategies, then re-pause at the
        # (now conflict-free) gate — do NOT confirm-execute yet; the user confirms after.
        if dedup_strategies and gate is not None:
            self._gate_execution.apply_dedup_strategies(plan, gate, dedup_strategies)
            return self._run_and_handle(plan_id, run_seq=run_seq)
        if _has_adoption_reason_adjust(adjust_params) and gate is not None:
            if not is_confirm(user_text):
                raise DriverError("提交采纳理由时必须同时确认采纳。")
            adoption_reason = self._require_adoption_reason(
                (adjust_params or {}).get("adoption_reason")
            )
            self._confirm_gate(
                plan,
                gate,
                reason=adoption_reason,
                input_updates={"adoption_reason": adoption_reason},
            )
            return self._run_and_handle(plan_id, run_seq=run_seq)
        if adjust_params and gate is not None:
            return self._gate_execution.apply_adjust(plan, gate, adjust_params, run_seq)
        if is_confirm(user_text):
            monitoring_error = monitoring_plain_confirm_error(
                plan,
                gate,
                self._safe_output,
            )
            if monitoring_error:
                return DriverTurn(
                    plan.id,
                    plan.status.value,
                    [
                        self._composer.instruction_message(
                            plan,
                            gate,
                            run_seq=run_seq,
                            text=monitoring_error,
                        )
                    ],
                )
            if gate is not None:
                if selection is not None:
                    self._gate_execution.apply_screen_selection(plan, gate, selection)
                if _is_adoption_gate(gate):
                    adoption_reason = self._require_adoption_reason(
                        (gate.inputs or {}).get("adoption_reason")
                    )
                    self._confirm_gate(
                        plan,
                        gate,
                        reason=adoption_reason,
                        input_updates={"adoption_reason": adoption_reason},
                    )
                else:
                    monitoring_updates = None
                    if (
                        gate.tool_ref.tool == "apply_monitoring_disposition"
                        and not str((gate.inputs or {}).get("reason") or "").strip()
                    ):
                        monitoring_updates = {
                            "reason": str(user_text or "人工确认本次监控结果")
                        }
                    self._confirm_gate(
                        plan,
                        gate,
                        reason=str(user_text or "人工确认当前业务决策"),
                        input_updates=monitoring_updates,
                    )
            return self._run_and_handle(plan_id, run_seq=run_seq)
        # Manual-mode TEXT gate reply, dispatched through the per-tool gate adapter
        # registry (marvis/agent/gates/adapters.py) instead of an inline per-tool
        # if-chain. Each adapter parses its own reply shape and applies it:
        #   * confirm_join      -- 「去重 first」/「用 last 去重」  (§6 same-key conflict)
        #   * select_rule_set   -- 「选 1,3,5」/「去掉 2」/「全选」 (§3 rule-set selection)
        #   * apply_monitoring_disposition -- 观察 / 调阈值 / 起新版本 (S5 red-light disposition)
        # A None from parse_reply (not this adapter's shape) or apply (a no-op, e.g.
        # a dedup instruction at a gate with no pending conflicts) falls through to
        # the generic confirm / LLM-router path unchanged.
        adapter = get_gate_adapter(gate)
        if adapter is not None:
            parsed = adapter.parse_reply(user_text, self._gate_reply_context(plan, gate))
            if parsed is not None:
                turn = adapter.apply(self, plan, gate, parsed, run_seq=run_seq)
                if turn is not None:
                    return turn
        return self._handle_instruction(plan, gate, user_text, run_seq)

    def replan_structured(
        self,
        *,
        plan_id,
        goal: str,
        expected_step_id=None,
        run_seq=0,
        confirmation_source=CONFIRMATION_SOURCE_HUMAN,
    ) -> DriverTurn:
        """Structural replan driven by an already-decided goal (AGT-8).

        Unlike ``resume(user_text=...)``, this does NOT feed ``goal`` back through
        ``is_confirm``/``route_instruction`` — it goes straight to
        ``GateExecutionAdapter.apply_replan`` (the same structured path
        ``_handle_instruction``'s ``action == "replan"`` branch already uses for a
        user-typed instruction). This is for callers that already hold a
        *structured* replan decision (AUTO's ``decide_gate``) and would otherwise
        have to round-trip it back through the free-text router — risking
        ``is_confirm`` misreading a phrase like "……并继续调参" as a plain confirm,
        or a second LLM classification pass misjudging it as ``clarify`` and
        silently dropping the replan intent (both routes never reach
        ``apply_replan`` in that case).
        """
        confirmation_source = _normalize_confirmation_source(confirmation_source)
        plan = self._repo.load_plan(plan_id)
        gate = None if plan.status == PlanStatus.VALIDATED else self._awaiting_step(plan)
        if expected_step_id and (gate is None or gate.id != str(expected_step_id)):
            raise DriverError("当前待确认步骤已变化，请刷新后重试。")
        _assert_source_may_operate_gate(gate, confirmation_source)
        if (
            confirmation_source == CONFIRMATION_SOURCE_AUTO
            and gate is not None
            and self._requires_governed_human_decision(gate)
        ):
            raise DriverError("AUTO 不得操作或确认强制人工业务决策节点，请由人工继续。")
        return self._gate_execution.apply_replan(plan, gate, goal, run_seq)

    def _handle_instruction(self, plan, gate, user_text, run_seq) -> DriverTurn:
        """Route a non-confirm reply. Manual mode (no LLM) shows the canned hint;
        agent mode classifies the instruction into confirm / adjust / replan / clarify
        and acts on it (spec §3 提指令→调整/重规划)."""
        if self._llm is None:
            return self._adjust_placeholder(plan.id, gate, run_seq)
        context = gate.title if gate is not None else "计划总览(尚未开始执行)"
        # AGT-5: tell the router which parameters this gate's dependency step(s)
        # actually declare (name/type/current value/bounds) instead of leaving it
        # to blind-guess key names from free text — a wrong guess previously only
        # surfaced as "没有识别到可调整的参数" after apply_adjust already failed.
        editable_schema = gate_editable_input_schema(
            plan,
            gate,
            self._safe_output,
        )
        param_schema = gate_param_schema(
            plan,
            gate,
            editable_input_schema=editable_schema,
        )
        route = route_instruction(
            self._llm, gate_context=context, instruction=user_text, param_schema=param_schema
        )
        action = route["action"]
        if action == "confirm":
            # This branch is reached only after ``is_confirm`` rejected the
            # user's text.  An LLM classification is therefore interpretation,
            # not an explicit human action.  It may neither start a validated
            # plan nor confirm any step (governed or otherwise).
            governed_gate = (
                gate is not None and self._requires_governed_human_decision(gate)
            )
            text = (
                "当前节点必须由你明确回复「确认」；Agent 不能代替你作出强制业务决策或签发副作用授权。"
                if governed_gate
                else "请由你明确回复「确认」后继续；Agent 不能根据语义猜测代替你启动计划或确认步骤。"
            )
            return DriverTurn(
                plan.id,
                plan.status.value,
                [
                    self._composer.instruction_message(
                        plan,
                        gate,
                        run_seq=run_seq,
                        text=text,
                    )
                ],
            )
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
                    text=route.get("reason") or "请明确指令:回复「确认」继续，或说明要调整的参数。",
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

    @staticmethod
    def _require_adoption_reason(value) -> str:
        try:
            return normalize_adoption_reason(value)
        except AdoptionReasonError as exc:
            raise DriverError(str(exc)) from exc

    # -- plan build -----------------------------------------------------------
    def build_plan(
        self,
        *,
        task_id,
        template_id,
        slots,
        autonomy=None,
        tier=None,
        success_criteria=None,
    ) -> Plan:
        if self._planner is None:
            raise DriverError("driver has no planner to build plans")
        plan = self._planner.from_template(
            get_template(template_id), dict(slots), task_id, autonomy=autonomy
        )
        if tier:
            plan.tier = tier
        if success_criteria:
            # AGT-4: layer user/AUTO-supplied criteria on top of the template's own
            # (empty for the built-in modeling templates today) rather than replacing
            # it, so a future template with real defaults still gets to keep them.
            plan.success_criteria = [*plan.success_criteria, *success_criteria]
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

    def _gate_reply_context(self, plan: Plan, gate: PlanStep) -> GateReplyContext:
        """The adapter-agnostic context a gate reply parser derives its needs from
        (the current plan + this driver's output loader). No adapter-specific detail
        lives here: e.g. the rule-set adapter reads its own mine_rules dependency's
        candidate count off this context, so the driver stays free of it."""
        return GateReplyContext(plan=plan, gate=gate, load_output=self._safe_output)

    def _apply_monitoring_disposition(
        self,
        gate: PlanStep,
        disposition: str,
        *,
        reason: str,
    ) -> None:
        """Bind an explicit alarm decision to the evidence-bound effect gate.

        The step remains ``AWAITING_CONFIRM`` while inputs are updated. Observe
        and new-version are immediately confirmed by the adapter; threshold
        adjustment remains paused until a concrete patch is supplied.
        """
        gate.inputs = {
            **(gate.inputs or {}),
            "disposition": disposition,
            "reason": reason,
        }
        self._repo.update_step(gate)

    def _confirm_gate(
        self,
        plan: Plan,
        gate: PlanStep,
        *,
        reason: str,
        input_updates: dict | None = None,
    ) -> None:
        if not self._requires_governed_human_decision(gate):
            if input_updates:
                self._repo.confirm_step_with_inputs(
                    gate.id,
                    input_updates=input_updates,
                )
            else:
                self._repo.confirm_step(gate.id)
            return
        if self._governance is None or self._principal is None:
            raise DriverError(
                "当前步骤需要平台记录人工决策，但本次请求没有有效的本地身份。"
            )
        try:
            self._governance.authorize_step(
                plan_id=plan.id,
                step_id=gate.id,
                principal=self._principal,
                reason=str(reason or "人工确认当前业务决策"),
                expected_plan_revision=int(plan.replan_count),
                input_updates=input_updates,
            )
        except (AuthorizationError, TypeError, ValueError) as exc:
            raise DriverError(str(exc)) from exc

    def _requires_governed_human_decision(self, gate: PlanStep) -> bool:
        policy = getattr(gate, "policy", None)
        if getattr(policy, "human_decision_gate", "none") == "required":
            return True
        resolver = getattr(self._governance, "requires_human_decision", None)
        if not callable(resolver):
            return False
        try:
            return bool(resolver(gate))
        except (AuthorizationError, TypeError, ValueError) as exc:
            raise DriverError(str(exc)) from exc

    def _latest_failed_step_run_error_kind(self, step_id: str) -> str | None:
        latest_error_kind = getattr(self._repo, "latest_failed_step_run_error_kind", None)
        if callable(latest_error_kind):
            return latest_error_kind(step_id)
        return None
__all__ = [
    "CONFIRMATION_SOURCE_AUTO",
    "CONFIRMATION_SOURCE_HUMAN",
    "PlanDriver",
    "DriverMessage",
    "DriverTurn",
    "DriverError",
    "is_confirm",
    "render_tool_output",
    # Backward-compat re-exports: the gate reply parsers moved to
    # marvis.agent.gates.adapters (LT-3) but tests + any external caller still
    # import them from here under their historical private names.
    "_parse_dedup_instruction",
    "_parse_monitoring_disposition",
    "_parse_rule_selection_instruction",
]
