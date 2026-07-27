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

from pathlib import Path
import re

from marvis.agent.adjust_specs import (
    adjust_param_error,
    has_feature_binning_adjust,
    has_special_value_adjust,
    normalize_adjust_params,
)
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
from marvis.agent.plan_utils import find_step
from marvis.agent.renderers import render_tool_output
from marvis.governance.errors import AuthorizationError
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.templates import get_template
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.state_machine import ConflictError
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
_EXPLICIT_CONFIRM_STATEMENT = re.compile(
    r"^(?:我\s*)?确认(?:无误|当前(?:结果|方案|设置|规格)|上述(?:结果|方案|设置|规格))?"
    r"(?:[\s，,。.!！；;:：]|$)",
    re.IGNORECASE,
)
_CONFIRM_STATEMENT_BLOCKER = re.compile(
    r"(?:但|但是|不过|然而|如果|假如|除非|改成|修改|调整|去掉|删除|增加|新增|换成|切换|"
    r"不要|别|暂停|停止|取消|先不|暂不|等一下|稍后|重新)",
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
    # Agent mode accepts a human's explicit confirmation sentence, not only a
    # one-token reply.  Keep this deliberately narrower than semantic intent:
    # the sentence must begin with "确认" (or "我确认"), and any contrast,
    # condition, cancellation, or parameter-adjustment wording still routes to
    # the instruction parser instead of releasing the gate.
    explicit = _EXPLICIT_CONFIRM_STATEMENT.match(raw.strip())
    if explicit and not _CONFIRM_STATEMENT_BLOCKER.search(raw[explicit.end():]):
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
        cancellation_check=None,
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
        self._cancellation_check = cancellation_check
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
            tasks_root=(
                Path(self._repo.db_path).parent / "tasks"
                if artifact_repo is not None
                else None
            ),
            db_path=(
                Path(self._repo.db_path)
                if artifact_repo is not None
                else None
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
        if has_feature_binning_adjust(adjust_params) and gate is not None:
            if not is_confirm(user_text):
                raise DriverError("提交分箱设置时必须同时确认。")
            params = {
                "features": list((adjust_params or {}).get("features") or []),
                "bins": (adjust_params or {}).get("bins", 10),
            }
            error = adjust_param_error(params) or self._feature_binning_adjust_error(
                plan, gate, params["features"]
            )
            if error:
                raise DriverError(error)
            self._confirm_gate(
                plan,
                gate,
                reason=(
                    f"人工选择 {len(params['features'])} 个特征进行 {int(params['bins'])} 箱分析"
                    if params["features"]
                    else "人工选择跳过可选分箱分析"
                ),
                input_updates={"features": params["features"], "bins": int(params["bins"])},
            )
            return self._run_and_handle(plan_id, run_seq=run_seq)
        if has_special_value_adjust(adjust_params) and gate is not None:
            if not is_confirm(user_text):
                raise DriverError("提交特殊值治理策略时必须同时确认。")
            raw_decisions = (adjust_params or {}).get("decisions")
            error = adjust_param_error({"decisions": raw_decisions})
            if error:
                raise DriverError(error)
            decisions = dict(raw_decisions)
            error = self._special_value_adjust_error(
                plan,
                gate,
                decisions,
                selection=selection,
            )
            if error:
                raise DriverError(error)
            if selection is not None:
                self._gate_execution.apply_screen_selection(plan, gate, selection)
            self._confirm_gate(
                plan,
                gate,
                reason="人工确认特殊值治理策略",
                input_updates={"decisions": decisions},
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
                if gate.tool_ref is not None and gate.tool_ref.tool == "resolve_special_values":
                    raw_decisions = (gate.inputs or {}).get("decisions")
                    decision_error = self._special_value_adjust_error(
                        plan,
                        gate,
                        dict(raw_decisions) if isinstance(raw_decisions, dict) else {},
                        selection=selection,
                    )
                    if decision_error:
                        raise DriverError(decision_error)
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

    def retry_failed_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        run_seq: int = 0,
        inputs: dict | None = None,
        preserve_target_confirmation: bool = False,
    ) -> DriverTurn:
        """Resume the same plan from its failed step, preserving prior outputs."""

        plan = self._repo.load_plan(plan_id)
        failed_step = find_step(plan, step_id)
        if failed_step is None:
            raise DriverError("失败步骤不存在，请刷新任务后重试。")
        retry_inputs = normalize_adjust_params({**(failed_step.inputs or {}), **(inputs or {})})
        # Persisted template inputs may keep recipes as a ``$ref:...`` until
        # execution resolves the upstream configure-tuning output.  Validate
        # concrete recipe lists (including legacy aliases), but do not treat a
        # valid unresolved reference as a malformed user adjustment.
        if isinstance(retry_inputs.get("recipes"), list):
            recipe_error = adjust_param_error({"recipes": retry_inputs["recipes"]})
            if recipe_error:
                raise DriverError(recipe_error)
        elif "recipes" in (inputs or {}) and not str(retry_inputs.get("recipes") or "").startswith(
            "$ref:"
        ):
            recipe_error = adjust_param_error({"recipes": retry_inputs.get("recipes")})
            if recipe_error:
                raise DriverError(recipe_error)
        inputs_unchanged = retry_inputs == dict(failed_step.inputs or {})
        was_confirmed = self._repo.is_step_confirmed(step_id)
        reauthorize_governed_target = bool(
            preserve_target_confirmation
            and was_confirmed
            and inputs_unchanged
            and self._governance is not None
            and self._principal is not None
            and self._requires_governed_human_decision(failed_step)
        )
        self._repo.retry_failed_step(
            plan_id,
            step_id,
            inputs=retry_inputs,
            # A governed step needs a fresh immutable decision bound to the
            # current manifest/input/evidence hashes.  Keeping only the old
            # ``confirmed`` bit can leave execution with no live governance
            # context after a restart or platform fix.  Non-governed gates may
            # still reuse the prior bit when the repository proves inputs are
            # unchanged.
            preserve_target_confirmation=(
                preserve_target_confirmation and not reauthorize_governed_target
            ),
        )
        if reauthorize_governed_target:
            paused = self._run_and_handle(plan_id, run_seq=run_seq)
            paused_plan = self._repo.load_plan(plan_id)
            gate = self._awaiting_step(paused_plan)
            if paused.status != PlanStatus.AWAITING_CONFIRM.value or gate is None:
                return paused
            if gate.id != step_id:
                raise DriverError("失败步骤重试时待确认节点已变化，请刷新后重试。")
            self._confirm_gate(
                paused_plan,
                gate,
                reason=f"人工明确授权从失败步骤重试：{failed_step.title}",
            )
        return self._run_and_handle(plan_id, run_seq=run_seq)

    def rollback_failed_plan_to_feature_screen(
        self,
        plan_id: str,
        failed_step_id: str,
        *,
        excluded_features: list[str],
        run_seq: int = 0,
    ) -> DriverTurn:
        """Revise the feature universe and resume the same plan from screening.

        This path is deliberately distinct from a failed-step retry.  It keeps
        the completed split/spec prefix, replaces the screen step's feature
        input with a concrete filtered list, and atomically invalidates that
        step plus every transitive descendant.  The screen gate is then shown
        again for fresh human confirmation.
        """

        plan = self._repo.load_plan(plan_id)
        if plan.status != PlanStatus.FAILED:
            raise DriverError("当前计划不是失败状态，不能执行上游回退。")
        failed_step = find_step(plan, failed_step_id)
        if failed_step is None or failed_step.status != StepStatus.FAILED:
            raise DriverError("当前失败步骤已变化，请刷新后重试。")

        ancestor_ids = _ancestor_step_ids(plan, failed_step_id)
        roots = [
            step
            for step in plan.steps
            if step.id in ancestor_ids
            and step.tool_ref.plugin == "modeling"
            and step.tool_ref.tool == "screen_features"
        ]
        if len(roots) != 1:
            raise DriverError("无法唯一定位已完成的模型特征筛选步骤，计划未修改。")
        root = roots[0]
        if root.status != StepStatus.DONE or not root.output_ref:
            raise DriverError("特征筛选步骤尚未完成，不能作为安全回退点。")
        if not root.needs_confirmation:
            raise DriverError("特征筛选步骤缺少人工确认门，拒绝自动回退。")

        normalized_exclusions: list[str] = []
        for item in excluded_features:
            name = str(item).strip()
            if name and name not in normalized_exclusions:
                normalized_exclusions.append(name)
        if not normalized_exclusions:
            raise DriverError("必须明确至少一个要排除的特征，计划未修改。")

        current_features = _resolve_revision_input(self._repo, (root.inputs or {}).get("features"))
        if not isinstance(current_features, list):
            current_features = _latest_ancestor_feature_cols(
                self._repo,
                plan,
                ancestor_ids,
                before_index=root.index,
            )
        feature_universe = _normalized_feature_list(current_features)
        if not feature_universe:
            raise DriverError("无法从已完成的建模规格解析特征集合，计划未修改。")

        protected_names = {
            str(value).strip()
            for value in (
                _resolve_revision_input(self._repo, (root.inputs or {}).get("target_col")),
                _resolve_revision_input(self._repo, (root.inputs or {}).get("split_col")),
            )
            if isinstance(value, str) and str(value).strip()
        }
        protected = [name for name in normalized_exclusions if name in protected_names]
        if protected:
            raise DriverError(
                "目标列或切分列不能作为普通特征排除：" + "、".join(protected) + "。"
            )
        unknown = [name for name in normalized_exclusions if name not in feature_universe]
        if unknown:
            raise DriverError(
                "以下列不在当前已确认的特征集合中，计划未修改：" + "、".join(unknown) + "。"
            )
        excluded_set = set(normalized_exclusions)
        remaining = [name for name in feature_universe if name not in excluded_set]
        if not remaining:
            raise DriverError("排除后特征集合为空，计划未修改。")

        revised_inputs = {**(root.inputs or {}), "features": remaining}
        try:
            self._repo.rollback_failed_plan_from_step(
                plan_id,
                root.id,
                failed_step_id,
                root_inputs=revised_inputs,
                excluded_features=normalized_exclusions,
                expected_plan_revision=int(plan.replan_count),
                expected_root_output_ref=str(root.output_ref),
            )
        except (ConflictError, KeyError, ValueError) as exc:
            raise DriverError(f"计划状态已变化，未执行上游回退：{exc}") from exc
        return self._run_and_handle(plan_id, run_seq=run_seq)

    def rollback_failed_plan_to_tuning_config(
        self,
        plan_id: str,
        failed_step_id: str,
        *,
        default_n_trials: int | None,
        n_trials_by_recipe: dict[str, int],
        run_seq: int = 0,
    ) -> DriverTurn:
        """Revise tuning budgets without invalidating split or feature work.

        The completed ``configure_tuning`` ancestor is the only truthful
        rollback root: changing the failed tune step directly would leave the
        configuration card showing stale budgets, while changing modeling spec
        would unnecessarily invalidate feature screening.  The repository
        resets configuration and every descendant atomically; execution then
        pauses at the fresh configuration confirmation gate before any trial is
        run.
        """

        plan = self._repo.load_plan(plan_id)
        if plan.status != PlanStatus.FAILED:
            raise DriverError("当前计划不是失败状态，不能修改调参预算。")
        failed_step = find_step(plan, failed_step_id)
        if failed_step is None or failed_step.status != StepStatus.FAILED:
            raise DriverError("当前失败步骤已变化，请刷新后重试。")

        ancestor_ids = _ancestor_step_ids(plan, failed_step_id)
        roots = [
            step
            for step in plan.steps
            if step.id in ancestor_ids
            and step.tool_ref.plugin == "modeling"
            and step.tool_ref.tool == "configure_tuning"
        ]
        if len(roots) != 1:
            raise DriverError("无法唯一定位已完成的配置调参步骤，计划未修改。")
        root = roots[0]
        if root.status != StepStatus.DONE or not root.output_ref:
            raise DriverError("配置调参步骤尚未完成，不能作为安全回退点。")
        if not root.needs_confirmation:
            raise DriverError("配置调参步骤缺少人工确认门，拒绝自动修改。")

        try:
            current_output = self._repo.load_step_output(root.id)
        except KeyError as exc:
            raise DriverError("配置调参输出不存在，计划未修改。") from exc
        recipes = _normalized_feature_list(current_output.get("recipes"))
        if not recipes:
            resolved_recipes = _resolve_revision_input(
                self._repo, (root.inputs or {}).get("recipes")
            )
            recipes = _normalized_feature_list(resolved_recipes)
        if not recipes:
            raise DriverError("无法从已完成配置解析候选算法，计划未修改。")

        requested: dict[str, int] = {}
        for raw_recipe, raw_count in dict(n_trials_by_recipe or {}).items():
            recipe = str(raw_recipe).strip()
            if (
                not recipe
                or isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 1
                or raw_count > 200
            ):
                raise DriverError("每种算法的调参预算必须是 1 到 200 的整数。")
            requested[recipe] = raw_count
        if default_n_trials is not None:
            if (
                isinstance(default_n_trials, bool)
                or not isinstance(default_n_trials, int)
                or default_n_trials < 1
                or default_n_trials > 200
            ):
                raise DriverError("统一调参预算必须是 1 到 200 的整数。")
            for recipe in recipes:
                requested.setdefault(recipe, default_n_trials)
        if not requested:
            raise DriverError("必须明确新的调参预算，计划未修改。")
        unknown = [recipe for recipe in requested if recipe not in recipes]
        if unknown:
            raise DriverError(
                "以下算法不在当前配置中，计划未修改：" + "、".join(unknown) + "。"
            )

        current_budgets = current_output.get("n_trials_by_recipe")
        revised_budgets = {
            recipe: int(
                (current_budgets or {}).get(
                    recipe,
                    current_output.get("n_trials") or 1,
                )
            )
            for recipe in recipes
        }
        revised_budgets.update(requested)
        revised_inputs = {
            **(root.inputs or {}),
            "n_trials_by_recipe": revised_budgets,
        }
        try:
            self._repo.rollback_failed_plan_from_step(
                plan_id,
                root.id,
                failed_step_id,
                root_inputs=revised_inputs,
                tuning_budgets=revised_budgets,
                expected_plan_revision=int(plan.replan_count),
                expected_root_output_ref=str(root.output_ref),
            )
        except (ConflictError, KeyError, ValueError) as exc:
            raise DriverError(f"计划状态已变化，未修改调参预算：{exc}") from exc
        return self._run_and_handle(plan_id, run_seq=run_seq)

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
        if self._cancellation_check is None:
            result = self._executor.run(plan_id)
        else:
            result = self._executor.run(
                plan_id,
                cancellation_check=self._cancellation_check,
            )
        plan = self._repo.load_plan(plan_id)
        status = result.status
        if status == PlanStatus.AWAITING_CONFIRM:
            gate = self._awaiting_step(plan)
            return DriverTurn(plan_id, status.value, [self._composer.gate_message(plan, gate, run_seq=run_seq)])
        if status == PlanStatus.DONE:
            return DriverTurn(plan_id, status.value, [self._composer.done_message(plan, run_seq=run_seq)])
        if status == PlanStatus.REVIEW:
            return DriverTurn(plan_id, status.value, [self._composer.review_message(plan, run_seq=run_seq)])
        if status == PlanStatus.CANCELLED:
            return DriverTurn(
                plan_id,
                status.value,
                [self._composer.cancelled_message(plan, run_seq=run_seq)],
            )
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

    def _feature_binning_adjust_error(
        self,
        plan: Plan,
        gate: PlanStep,
        features: list,
    ) -> str | None:
        allowed: set[str] = set()
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "compute_feature_metrics":
                continue
            output = self._safe_output(dep.id)
            for metric in (output.get("metrics") or []) if isinstance(output, dict) else []:
                if isinstance(metric, dict) and str(metric.get("feature") or "").strip():
                    allowed.add(str(metric["feature"]).strip())
        unknown = sorted({str(item).strip() for item in features} - allowed)
        if unknown:
            return f"分箱特征不在本次单变量分析结果中: {', '.join(unknown)}。"
        return None

    def _special_value_adjust_error(
        self,
        plan: Plan,
        gate: PlanStep,
        decisions: dict,
        *,
        selection=None,
    ) -> str | None:
        """Validate complete decisions against persisted screen evidence.

        This runs before mutating either the screen selection or the gate input,
        so a stale/partial UI submission cannot leave half-applied state.
        """
        selected: list[str] = []
        sentinel_columns: dict[str, object] = {}
        screen_found = False
        for dep_id in gate.depends_on or []:
            dep = find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "screen_features":
                continue
            screen_found = True
            output = self._safe_output(dep.id)
            if not isinstance(output, dict):
                return "缺少特征筛选输出，无法确认特殊值治理策略。"
            raw_selected = output.get("selected")
            if not isinstance(raw_selected, list):
                return "特征筛选输出缺少有效的已选特征列表，无法确认特殊值治理策略。"
            selected = [
                str(item).strip()
                for item in (
                    selection if selection is not None else raw_selected
                )
                if str(item).strip()
            ]
            raw_columns = output.get("sentinel_columns")
            if not isinstance(raw_columns, dict):
                return "特征筛选输出缺少有效的特殊值检测证据，无法确认治理策略。"
            sentinel_columns = dict(raw_columns)
            break
        if not screen_found:
            return "特殊值治理步骤未绑定特征筛选证据，无法确认。"
        relevant = [
            column
            for column in selected
            if column in sentinel_columns and sentinel_columns.get(column)
        ]
        if not relevant:
            return None
        decision_names = {str(column) for column in decisions}
        missing = [column for column in relevant if column not in decision_names]
        if missing:
            return (
                "以下已选特征检测到特殊值，必须逐列选择转空、保留或删除后再继续："
                + "、".join(missing)
                + "。"
            )
        unrelated = sorted(decision_names - set(relevant))
        if unrelated:
            return "治理决策包含当前未选或未检测到特殊值的特征：" + "、".join(unrelated) + "。"
        return adjust_param_error({"decisions": decisions})

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
def _ancestor_step_ids(plan: Plan, step_id: str) -> set[str]:
    by_id = {step.id: step for step in plan.steps}
    result: set[str] = set()
    pending = list((by_id.get(step_id).depends_on or []) if by_id.get(step_id) else [])
    while pending:
        candidate = str(pending.pop())
        if candidate in result:
            continue
        result.add(candidate)
        parent = by_id.get(candidate)
        if parent is not None:
            pending.extend(parent.depends_on or [])
    return result


def _resolve_revision_input(repo, value):
    if not (isinstance(value, str) and value.startswith("$ref:")):
        return list(value) if isinstance(value, list) else value
    match = re.fullmatch(r"\$ref:(?P<step>.+?)\.output(?:\.(?P<field>.+))?", value)
    if match is None:
        return None
    try:
        current = repo.load_step_output(match.group("step"))
    except KeyError:
        return None
    field = match.group("field")
    if not field:
        return current
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _latest_ancestor_feature_cols(
    repo,
    plan: Plan,
    ancestor_ids: set[str],
    *,
    before_index: int,
):
    candidates = sorted(
        (
            step
            for step in plan.steps
            if step.id in ancestor_ids and step.index < before_index and step.output_ref
        ),
        key=lambda step: (step.index, step.id),
        reverse=True,
    )
    for step in candidates:
        try:
            output = repo.load_step_output(step.id)
        except KeyError:
            continue
        feature_cols = output.get("feature_cols") if isinstance(output, dict) else None
        if isinstance(feature_cols, list):
            return feature_cols
    return None


def _normalized_feature_list(value) -> list[str]:
    if not isinstance(value, list) or not value:
        return []
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return []
        name = item.strip()
        if name not in result:
            result.append(name)
    return result


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
