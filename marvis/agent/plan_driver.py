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
from marvis.agent.instruction_router import route_instruction
from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.plan_utils import find_step
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.templates import get_template

# A reply counts as confirmation of the current gate only when, after stripping
# whitespace/punctuation, the *entire* remaining text is made up of short affirmative
# tokens (see _CONFIRM_TOKEN below). This full-string anchoring — rather than a
# substring `.search` over the raw reply — is what stops questions (“这样可以吗？”)
# and embedded/contrasting affirmatives (“结果不是很好的”, “好的地方是命中率，但…”)
# from being misread as confirmation (AGT-1 / H4).
_CONFIRM_TOKEN = r"(?:好的|好|可以|确认|确定|没问题|同意|就这样|继续|开始|对的|对|行|ok|okay|yes|y|go|proceed)"
_CONFIRM_FULLMATCH = re.compile(rf"(?:{_CONFIRM_TOKEN})+", re.IGNORECASE)
# Interrogative guard: any question mark, or a trailing/whole-string question particle,
# disqualifies a reply from being read as confirmation even if it also contains an
# affirmative token (e.g. “这样可以吗？”, “KS高吗，可以到0.3吗”, “行不行”).
_QUESTION = re.compile(
    r"[?？]|吗|吧$|行不行|可不可以|能不能|好不好|对不对|是不是|呢$",
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


def is_confirm(text: str) -> bool:
    raw = text or ""
    if _QUESTION.search(raw):
        return False
    if _NEGATED_CONFIRM.search(raw):
        return False
    compact = _STRIP_PUNCT.sub("", raw)
    if not compact:
        return False
    if _NEGATED_CONFIRM.search(compact):
        return False
    return bool(_CONFIRM_FULLMATCH.fullmatch(compact))


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


_SELECT_ALL = re.compile(r"(全选|都要|全部|全都|all)", re.IGNORECASE)
_DROP_PREFIX = re.compile(r"(去掉|去除|删掉|删除|移除|排除|不要|drop|remove|exclude)", re.IGNORECASE)
_KEEP_PREFIX = re.compile(r"(只?选|保留|选中|选择|要|keep|select|pick|use)", re.IGNORECASE)
_INDEX_TOKEN = re.compile(r"\d+")
_RULE_MENTION = re.compile(r"(规则|规则集|rule|条|第)", re.IGNORECASE)


def _parse_rule_selection_instruction(text: str, candidate_count: int) -> list[int] | None:
    """Parse a rule-set gate reply into an ordered list of 1-based indices.

    Recognises three shapes (spec §3, parallel to _parse_dedup_instruction):
      * 「全选」/「都要」/「all」                → keep every candidate, in order;
      * 「去掉 2」/「去除 2 4」/「drop 2」        → all candidates except those indices;
      * 「选 1,3,5」/「保留 1 3 5」/「pick 1 3」  → exactly those indices, in the
        order the user wrote them (so the user can also reorder).

    Returns None when the reply is not a rule-selection instruction (no keyword
    and no bare index list, or it looks like a question/negated-confirm) so an
    unrelated instruction falls through to the LLM router unchanged. Indices out
    of ``[1, candidate_count]`` are dropped defensively; an empty result returns
    None (nothing actionable) rather than an empty selection.
    """
    raw = text or ""
    if _QUESTION.search(raw):
        return None
    if candidate_count <= 0:
        return None
    all_indices = list(range(1, candidate_count + 1))
    if _SELECT_ALL.search(raw):
        return all_indices
    indices = _ordered_unique_indices(_INDEX_TOKEN.findall(raw), candidate_count)
    is_drop = bool(_DROP_PREFIX.search(raw))
    is_keep = bool(_KEEP_PREFIX.search(raw))
    if is_drop and not is_keep:
        if not indices:
            return None
        dropped = set(indices)
        kept = [index for index in all_indices if index not in dropped]
        return kept or None
    if (is_keep or _RULE_MENTION.search(raw)) and indices:
        return indices
    # A bare index list with no keyword ("1 3 5") is still a keep instruction.
    if indices and _looks_like_bare_index_list(raw):
        return indices
    return None


def _ordered_unique_indices(tokens: list[str], candidate_count: int) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            index = int(token)
        except ValueError:
            continue
        if 1 <= index <= candidate_count and index not in seen:
            seen.add(index)
            ordered.append(index)
    return ordered


def _looks_like_bare_index_list(text: str) -> bool:
    """True when text is essentially just numbers + separators (1,3,5 / 1 3 5),
    so a plain index list is treated as a keep-selection without a keyword."""
    stripped = re.sub(r"[\s,，、和及\-到~]+", "", text or "")
    return bool(stripped) and bool(re.fullmatch(r"\d+", stripped))


_MONITORING_NEW_VERSION = re.compile(r"(起新版本|新版本|新起一版|起一版|重起|new\s*version|new\s*strategy)", re.IGNORECASE)
_MONITORING_ADJUST = re.compile(r"(调阈值|改阈值|调整阈值|调门槛|调参|adjust\s*threshold|retune|re-?run)", re.IGNORECASE)
_MONITORING_OBSERVE = re.compile(r"(维持|观察|保持|继续观察|再看看|keep\s*watch|observe|hold)", re.IGNORECASE)


def _parse_monitoring_disposition(text: str) -> str | None:
    """Parse a strategy-monitoring alarm-gate reply into a disposition keyword.

    Recognises the three red-light checklist choices (spec S5), most-specific
    first so 「起新版本」 wins over a co-occurring 「观察」:
      * 「起新版本」/「新版本」/「new version」        -> "new_version"
      * 「调阈值」/「调整阈值」/「adjust threshold」    -> "adjust_threshold"
      * 「维持」/「观察」/「保持」/「observe」          -> "observe"

    Returns None when the reply names no disposition (a plain 「确认」 or an
    unrelated instruction), so it falls through to the normal confirm/route path.
    """
    raw = text or ""
    if _MONITORING_NEW_VERSION.search(raw):
        return "new_version"
    if _MONITORING_ADJUST.search(raw):
        return "adjust_threshold"
    if _MONITORING_OBSERVE.search(raw):
        return "observe"
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
        # Manual-mode TEXT rule-set selection at a select_rule_set gate (no §4
        # picker): the user replies e.g. 「选 1,3,5」/「去掉 2」/「全选」 → parse into
        # a 1-based index selection and push it through the SAME generic
        # apply_adjust override channel band_edges uses (the gate step's own
        # `selection` input, default None, is overwritten and the gate re-armed).
        if gate is not None and gate.tool_ref.tool == "select_rule_set":
            selection = _parse_rule_selection_instruction(user_text, self._rule_candidate_count(gate))
            if selection is not None:
                return self._gate_execution.apply_adjust(plan, gate, {"selection": selection}, run_seq)
        # S5 strategy-monitoring alarm gate: the report step is the gate (it renders
        # its run_strategy_monitoring dependency's verdict + red-light checklist). A
        # red-light reply naming one of the three dispositions (观察 / 调阈值 /
        # 起新版本) is recorded onto the report gate's own `disposition` input (so the
        # report surfaces next_action -- for 起新版本 a STRATEGY_DEVELOPMENT follow-up
        # prompt, never an auto-created task) before confirming the gate to proceed.
        if gate is not None and gate.tool_ref.tool == "render_monitoring_report":
            disposition = _parse_monitoring_disposition(user_text)
            if disposition is not None:
                self._apply_monitoring_disposition(gate, disposition)
                self._repo.confirm_step(gate.id)
                return self._run_and_handle(plan_id, run_seq=run_seq)
        return self._handle_instruction(plan, gate, user_text, run_seq)

    def replan_structured(
        self,
        *,
        plan_id,
        goal: str,
        expected_step_id=None,
        run_seq=0,
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
        plan = self._repo.load_plan(plan_id)
        gate = None if plan.status == PlanStatus.VALIDATED else self._awaiting_step(plan)
        if expected_step_id and (gate is None or gate.id != str(expected_step_id)):
            raise DriverError("当前待确认步骤已变化，请刷新后重试。")
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
        param_schema = gate_param_schema(plan, gate)
        route = route_instruction(
            self._llm, gate_context=context, instruction=user_text, param_schema=param_schema
        )
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

    def _rule_candidate_count(self, gate: PlanStep) -> int:
        """How many mined candidate rules the select_rule_set gate is choosing
        from -- read from its mine_rules dependency's persisted output so the
        text selection parser can bound/validate 1-based indices. Falls back to
        the gate's own resolved candidate_rules input, then 0 if neither is
        available yet."""
        plan = self._repo.load_plan(gate.plan_id)
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "mine_rules":
                continue
            output = self._safe_output(dep.id)
            if isinstance(output, dict) and isinstance(output.get("candidate_rules"), list):
                return len(output["candidate_rules"])
        candidates = (gate.inputs or {}).get("candidate_rules")
        return len(candidates) if isinstance(candidates, list) else 0

    def _apply_monitoring_disposition(self, gate: PlanStep, disposition: str) -> None:
        """Record the parsed red-light disposition onto the report gate's own
        `disposition` input (literal-None default in the template) so its output
        surfaces the right next_action. Persists the input change via update_step
        (preserving the step's AWAITING_CONFIRM status -- the apply_screen_selection
        precedent), so the following confirm_step + run executes the report tool with
        the chosen disposition. reset_step is deliberately NOT used here: it would
        clear the confirmation and re-arm the gate to pause a second time."""
        gate.inputs = {**(gate.inputs or {}), "disposition": disposition}
        self._repo.update_step(gate)

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
