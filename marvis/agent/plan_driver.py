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
from dataclasses import dataclass, field
from typing import Any

from marvis.agent.instruction_router import route_instruction
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.templates import get_template

# A reply counts as confirmation of the current gate.
_CONFIRM = re.compile(
    r"(确认|确定|没问题|可以|就这样|同意|好的|继续|开始|对的?|ok|yes|go|proceed|looks good|sounds good)",
    re.IGNORECASE,
)


def is_confirm(text: str) -> bool:
    return bool(_CONFIRM.search(text or ""))


def _parse_dedup_instruction(text: str) -> str | None:
    """Parse a manual-mode dedup reply at a join gate → "first"/"last"/None.

    Recognised only when the text actually mentions de-duplication (去重/dedup/策略/保留)
    so an unrelated instruction isn't misread as a strategy. first = keep the first row per
    key, last = keep the last (spec §6 conflict resolution)."""
    low = (text or "").lower()
    if not any(token in low for token in ("去重", "dedup", "策略", "保留", "重复")):
        return None
    if "first" in low or "首" in text or "第一" in text or "前" in text:
        return "first"
    if "last" in low or "末" in text or "最后" in text or "最新" in text or "后" in text:
        return "last"
    return None


@dataclass
class DriverMessage:
    """One append-only assistant message. ``metadata`` carries the locator
    ``{plan_id, step_id, run_seq}`` plus any inline ``tables``."""

    stage: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DriverTurn:
    plan_id: str
    status: str  # PlanStatus value
    messages: list[DriverMessage] = field(default_factory=list)


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

    def resume(self, *, plan_id, user_text, run_seq=0, selection=None, dedup_strategies=None) -> DriverTurn:
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
        # Join dedup picker: re-confirm with the chosen strategies, then re-pause at the
        # (now conflict-free) gate — do NOT confirm-execute yet; the user confirms after.
        if dedup_strategies and gate is not None:
            self._apply_dedup_strategies(plan, gate, dedup_strategies)
            return self._run_and_handle(plan_id, run_seq=run_seq)
        if is_confirm(user_text):
            if gate is not None:
                if selection is not None:
                    self._apply_screen_selection(plan, gate, selection)
                self._repo.confirm_step(gate.id)
            return self._run_and_handle(plan_id, run_seq=run_seq)
        # Manual-mode TEXT resolution of a same-key dedup conflict (no §4 picker available):
        # the user replies e.g. 「去重 first」/「用 last 去重」 → apply that strategy to every
        # feature confirm_join flagged as needs_dedup, then re-pause at the cleared gate.
        if gate is not None:
            strategy = _parse_dedup_instruction(user_text)
            if strategy:
                pending = self._needs_dedup_features(plan, gate)
                if pending:
                    self._apply_dedup_strategies(plan, gate, {fid: strategy for fid in pending})
                    return self._run_and_handle(plan_id, run_seq=run_seq)
        return self._handle_instruction(plan, gate, user_text, run_seq)

    def _needs_dedup_features(self, plan, gate) -> list[str]:
        """Feature ids the gate's confirm_join dependency flagged as needing a dedup
        strategy (same-key conflict). Empty when there is nothing to resolve."""
        if gate is None:
            return []
        for dep_id in gate.depends_on or []:
            dep = _find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "confirm_join":
                continue
            output = self._repo.load_step_output(dep.id) or {}
            pending = output.get("needs_dedup") or []
            return [str(f) for f in pending]
        return []

    def _apply_dedup_strategies(self, plan, gate, dedup_strategies) -> None:
        """Re-confirm the gate's ``confirm_join`` dependency with the user's per-feature
        dedup strategy map (§4 dedup picker). Only the confirm step and the gate are
        reset (the propose step's diagnostics + join plan are kept), so confirm_join
        re-runs against the same join plan with strategies applied — resolving same-key
        conflicts — and the executor re-pauses at the (now clear) execute gate. A
        structured manual-mode override that doesn't need the LLM adjust router."""
        if gate is None or not isinstance(dedup_strategies, dict) or not dedup_strategies:
            return
        clean = {str(k): str(v) for k, v in dedup_strategies.items() if str(v).strip()}
        if not clean:
            return
        reset_any = False
        for dep_id in gate.depends_on or []:
            dep = _find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "confirm_join":
                continue
            self._repo.reset_step(dep.id, inputs={**(dep.inputs or {}), "dedup_strategies": clean})
            reset_any = True
        if reset_any:
            self._repo.reset_step(gate.id)

    def _apply_screen_selection(self, plan, gate, selection) -> None:
        """Override a screening gate's proposed ``selected`` with the user's edited set.

        The chosen features are constrained to what the screen actually saw (any
        scored/bucketed column — including leakage/suspected ones the user may
        deliberately *force-select*), so an edited selection can narrow or re-pick
        among real screened features but can never smuggle in a column the screen
        never validated. A selection that resolves to nothing is ignored (keep the
        proposed set) rather than training on zero features.
        """
        if gate is None:
            return
        sel = [str(f) for f in (selection or []) if str(f).strip()]
        if not sel:
            return
        for dep_id in gate.depends_on or []:
            dep = _find_step(plan, dep_id)
            if dep is None or dep.tool_ref.tool != "screen_features":
                continue
            output = self._safe_output(dep_id)
            if not isinstance(output, dict):
                continue
            known = _screen_known_features(output)
            chosen = [f for f in dict.fromkeys(sel) if not known or f in known]
            if not chosen:
                continue
            self._repo.store_step_output(dep_id, {**output, "selected": chosen})

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
            return self._apply_adjust(plan, gate, route["params"], run_seq)
        if action == "replan":
            return self._apply_replan(plan, gate, user_text, run_seq)
        return self._instruction_message(
            plan, gate, run_seq,
            route.get("reason") or "请明确指令:回复「确认」继续,或说明要调整的参数。",
        )

    def _apply_replan(self, plan, gate, instruction, run_seq) -> DriverTurn:
        """Structural replan (spec §3 提指令→重规划): regenerate the remaining steps to
        satisfy the instruction. Before the plan starts, show the revised overview and
        await 开始 again; mid-execution, run the revised remaining steps to the next gate."""
        replan = getattr(self._executor, "replan_from_instruction", None)
        if replan is None or not replan(plan.id, instruction):
            return self._instruction_message(
                plan, gate, run_seq,
                "重规划未成功(重规划预算用尽或指令无法执行);可改为在节点处「调参重算」,"
                "或重新创建任务调整配置。",
            )
        revised = self._repo.load_plan(plan.id)
        if revised.status == PlanStatus.VALIDATED:
            # Not started yet → present the new plan and pause at the 开始 gate again.
            return DriverTurn(revised.id, revised.status.value, [
                DriverMessage("chat", "已按指令重规划,请查看新计划。",
                              {"plan_id": revised.id, "run_seq": run_seq}),
                self._plan_overview_message(revised),
            ])
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage("chat", "已按指令重规划并继续执行。",
                          {"plan_id": plan.id, "run_seq": run_seq}),
        )
        return turn

    def _apply_adjust(self, plan, gate, params, run_seq) -> DriverTurn:
        """Re-run ALL of the gate's analysis dependencies with overridden parameters, then
        re-pause at the gate showing the recomputed result. Each override is applied only
        to a dependency whose inputs declare that key (so a param meant for one step isn't
        forced onto another); every dependency is reset so the recompute is consistent."""
        deps = [step for step in (_find_step(plan, dep_id) for dep_id in (gate.depends_on or [])) if step is not None]
        if not deps:
            return self._instruction_message(plan, gate, run_seq, "没找到可调整的上一步,请重新确认。")
        params = params or {}
        # Apply each override only to a dep that ALREADY declares that input key, and to
        # EVERY such dep (per-key fan-out). This never injects a schema-forbidden key (the
        # tools use additionalProperties:false, so an undeclared key would fail validation
        # and FAIL the plan), and keeps sibling deps that share a param consistent.
        primary = None
        for dep in deps:
            overrides = {key: value for key, value in params.items() if key in (dep.inputs or {})}
            self._repo.reset_step(dep.id, inputs={**(dep.inputs or {}), **overrides})
            if overrides and primary is None:
                primary = dep
        primary = primary or deps[0]
        self._repo.reset_step(gate.id)
        turn = self._run_and_handle(plan.id, run_seq=run_seq)
        turn.messages.insert(
            0,
            DriverMessage(
                "chat",
                f"已按指令调整参数 {params} 并重算「{primary.title}」。",
                {"plan_id": plan.id, "step_id": primary.id, "run_seq": run_seq},
            ),
        )
        return turn

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
        return DriverMessage(
            "plan_overview", "\n".join(lines), {"plan_id": plan.id, "kind": "plan_overview"}
        )

    def _compose_gate_message(self, plan: Plan, gate: PlanStep | None, *, run_seq) -> DriverMessage:
        parts: list[str] = []
        tables: list[dict] = []
        screen: dict | None = None
        confirm_join_o: dict | None = None
        propose_join_o: dict | None = None
        for dep_id in gate.depends_on if gate else []:
            dep = _find_step(plan, dep_id)
            if dep is None:
                continue
            output = self._safe_output(dep_id)
            if output is None:
                continue
            text, dep_tables = render_tool_output(dep.tool_ref.tool, output)
            if text:
                parts.append(text)
            tables.extend(dep_tables)
            # Surface the structured screening result so the frontend can render the
            # §4 interactive selection table (rows=features, metric cols, checkboxes).
            # Pass-through of the tool output + the screen step id (so an edited
            # selection can be confirmed back against it) + the gating thresholds.
            if dep.tool_ref.tool == "screen_features":
                screen = _screen_payload(output, dep)
            elif dep.tool_ref.tool == "confirm_join":
                confirm_join_o = output
            elif dep.tool_ref.tool == "propose_join":
                propose_join_o = output
        if not parts:
            parts.append("上一步已完成。")
        parts.append("确认请回复「确认」继续;要调整可直接说明。")
        meta = {
            "plan_id": plan.id,
            "step_id": gate.id if gate else None,
            "run_seq": run_seq,
            "tables": tables,
            "kind": "gate",  # marks a needs-confirmation gate (manual-mode confirm control)
        }
        if screen is not None:
            meta["screen"] = screen
        # §4 join dedup picker: when confirm_join left features awaiting a dedup strategy
        # (non-unique keys), surface them + their conflict counts so the frontend can
        # render per-feature first/last pickers the user resolves before executing.
        dedup = _dedup_payload(confirm_join_o, propose_join_o)
        if dedup is not None:
            meta["dedup"] = dedup
        return DriverMessage("gate", "\n\n".join(parts), meta)

    def _compose_done_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        terminal = max(
            (s for s in plan.steps if s.status == StepStatus.DONE and s.output_ref),
            key=lambda s: s.index,
            default=None,
        )
        parts = ["✅ 计划已全部完成。"]
        tables: list[dict] = []
        if terminal is not None:
            output = self._safe_output(terminal.id)
            if output is not None:
                text, tables = render_tool_output(terminal.tool_ref.tool, output)
                if text:
                    parts.append(text)
        return DriverMessage("done", "\n\n".join(parts), {"plan_id": plan.id, "run_seq": run_seq, "tables": tables})

    def _compose_review_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        return DriverMessage(
            "review",
            "计划已执行完,但结果需要你复核一下再定论。",
            {"plan_id": plan.id, "run_seq": run_seq},
        )

    def _compose_failed_message(self, plan: Plan, *, run_seq) -> DriverMessage:
        failed = next((s for s in plan.steps if s.status == StepStatus.FAILED), None)
        detail = f"「{failed.title}」失败:{failed.error}" if failed and failed.error else "执行中断。"
        return DriverMessage(
            "error",
            f"❌ {detail}",
            {"plan_id": plan.id, "step_id": failed.id if failed else None, "run_seq": run_seq},
        )

    def _safe_output(self, step_id: str):
        try:
            return self._repo.load_step_output(step_id)
        except KeyError:
            return None


def _find_step(plan: Plan, step_id: str) -> PlanStep | None:
    for step in plan.steps:
        if step.id == step_id:
            return step
    return None


# ---------------------------------------------------------------------------
# tool -> table registry (decision #4 in the driver spec)
# Each renderer turns a tool's raw output into (markdown text, [table dicts]).
# Task differences land HERE; the driver loop above stays task-agnostic. A table
# dict is {title, columns, rows} — the frontend maps it onto renderMetricTableSection.
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _names(items) -> list[str]:
    out = []
    for item in items or []:
        out.append(item[0] if isinstance(item, (list, tuple)) and item else item)
    return [str(x) for x in out]


def _pct(value):
    """Missing-rate as a percentage string; ``None`` → n/a."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _triple(item):
    """A (feature, ks, reason) row from a leakage/suspected entry, tolerant of shape."""
    if isinstance(item, (list, tuple)):
        feat = str(item[0]) if len(item) > 0 else ""
        ks = item[1] if len(item) > 1 else None
        reason = str(item[2]) if len(item) > 2 else ""
        return feat, ks, reason
    return str(item), None, ""


def _render_screen(o: dict):
    selected = o.get("selected") or []
    leak = o.get("leakage") or []
    susp = o.get("suspected") or []
    unusable = o.get("unusable") or []
    scores = o.get("scores") if isinstance(o.get("scores"), dict) else {}
    leak_names = _names(leak)
    susp_names = _names(susp)
    n = o.get("n_screened") or o.get("n") or (len(selected) + len(leak) + len(susp))
    text = (
        f"**特征筛选完成**:从 {n} 个候选中提议保留 **{len(selected)}** 个特征。\n"
        f"- 剔除疑似**泄漏** {len(leak_names)} 个" + (f"(如 {leak_names[:3]})" if leak_names else "") + "\n"
        f"- 疑似**模型输出/评分**列 {len(susp_names)} 个" + (f"(如 {susp_names[:5]})" if susp_names else "") + "\n"
        f"- 剔除**不可用**(常量/稀疏) {len(unusable)} 个"
    )
    tables = []
    if selected:
        rows = []
        for feat in selected[:20]:
            s = scores.get(feat) if isinstance(scores.get(feat), dict) else {}
            rows.append([feat, _num(s.get("ks")), _num(s.get("iv")), _pct(s.get("missing_rate"))])
        tables.append({"title": "入选特征(前20)", "columns": ["特征", "KS", "IV", "缺失率"], "rows": rows})
    if leak:
        tables.append({
            "title": f"疑似泄漏(KS≥阈值,共{len(leak)})",
            "columns": ["特征", "KS", "原因"],
            "rows": [[f, _num(k), r] for f, k, r in (_triple(i) for i in leak[:20])],
        })
    if susp:
        tables.append({
            "title": f"疑似模型输出/评分列(共{len(susp)})",
            "columns": ["特征", "KS", "原因"],
            "rows": [[f, _num(k), r] for f, k, r in (_triple(i) for i in susp[:20])],
        })
    if unusable:
        rows = []
        for item in unusable[:20]:
            if isinstance(item, (list, tuple)):
                rows.append([str(item[0]) if item else "", str(item[1]) if len(item) > 1 else ""])
            else:
                rows.append([str(item), ""])
        tables.append({"title": f"剔除·不可用(常量/稀疏,共{len(unusable)})", "columns": ["特征", "原因"], "rows": rows})
    return text, tables


def _dedup_payload(confirm_o: dict | None, propose_o: dict | None) -> dict | None:
    """Per-feature dedup picker payload for a join gate (§4). Returns None unless
    ``confirm_join`` left features awaiting a dedup strategy. For each such feature,
    attach the conflict-key count + conflicting columns from the propose-step
    diagnostics so the picker shows *why* a strategy is needed."""
    confirm = confirm_o if isinstance(confirm_o, dict) else {}
    needs = [str(f) for f in (confirm.get("needs_dedup") or [])]
    if not needs:
        return None
    info: dict[str, dict] = {}
    propose = propose_o if isinstance(propose_o, dict) else {}
    for join in propose.get("joins") or []:
        if not isinstance(join, dict):
            continue
        fid = str(join.get("feature_id"))
        diag = join.get("diagnostics") if isinstance(join.get("diagnostics"), dict) else {}
        report = diag.get("conflict_report") if isinstance(diag.get("conflict_report"), dict) else {}
        info[fid] = {
            "conflict_keys": int(report.get("n_conflict_keys") or 0),
            "conflict_columns": [str(c) for c in (report.get("conflict_columns") or [])],
        }
    features = [
        {"feature_id": fid, **info.get(fid, {"conflict_keys": 0, "conflict_columns": []})}
        for fid in needs
    ]
    return {"needs_dedup": needs, "features": features, "strategies": ["first", "last"]}


def _screen_known_features(output: dict) -> set:
    """Every feature the screen actually saw — scored, ranked, or bucketed into
    leakage/suspected/unusable. Used to constrain an edited selection so it can only
    re-pick among validated columns (force-selecting a flagged one is allowed)."""
    known: set = set()
    o = output if isinstance(output, dict) else {}
    scores = o.get("scores")
    if isinstance(scores, dict):
        known.update(str(k) for k in scores)
    for key in ("ranked", "leakage", "suspected", "unusable"):
        for item in o.get(key) or []:
            if isinstance(item, (list, tuple)) and item:
                known.add(str(item[0]))
            elif isinstance(item, str):
                known.add(item)
    known.update(str(f) for f in (o.get("selected") or []))
    return known


def _screen_payload(output: dict, dep) -> dict:
    """Structured screening result for the frontend §4 interactive selection table.

    A pass-through of the screen tool output (ranked KS, per-feature scores, the
    leakage/suspected/unusable buckets with reasons) plus (a) the screen step id —
    so an edited selection can be confirmed back against that exact step — and (b)
    the gating thresholds the screen used, so the table's sliders default to them.
    """
    o = output if isinstance(output, dict) else {}
    inputs = getattr(dep, "inputs", None) or {}

    def _flt(key, default):
        try:
            return float(inputs.get(key, default))
        except (TypeError, ValueError):
            return default

    return {
        "step_id": getattr(dep, "id", None),
        "step_title": getattr(dep, "title", None),
        "selected": list(o.get("selected") or []),
        "ranked": o.get("ranked") or [],
        "leakage": o.get("leakage") or [],
        "suspected": o.get("suspected") or [],
        "unusable": o.get("unusable") or [],
        "scores": o.get("scores") if isinstance(o.get("scores"), dict) else {},
        "n_screened": o.get("n_screened") or 0,
        "thresholds": {
            "leakage_ks": _flt("leakage_ks", 0.40),
            "max_missing_rate": _flt("max_missing_rate", 0.95),
        },
    }


def _render_tune(o: dict):
    best_params = o.get("best_params") or {}
    best_metrics = o.get("best_metrics") or {}
    trials = [t for t in (o.get("trials") or []) if isinstance(t, dict)]
    text = f"**调参完成**:{o.get('n_trials', '?')} 轮搜索,选出最优超参组合。"
    tables = []
    if trials:
        # trials leaderboard (G4): each trial's train/test/oot KS + overfit gap,
        # ranked by the in-time selection score (OOT is the unbiased final metric).
        ranked = sorted(
            trials,
            key=lambda t: t.get("score") if isinstance(t.get("score"), (int, float)) else float("-inf"),
            reverse=True,
        )
        rows = []
        for rank, trial in enumerate(ranked[:15], start=1):
            train_ks, test_ks = trial.get("train_ks"), trial.get("test_ks")
            # overfit gaps: prefer stored values, fall back to deriving train-test.
            gap_tt = trial.get("overfit_gap_tt")
            if gap_tt is None and isinstance(train_ks, (int, float)) and isinstance(test_ks, (int, float)):
                gap_tt = train_ks - test_ks
            rows.append([
                str(rank), _num(train_ks), _num(test_ks), _num(trial.get("oot_ks")),
                _num(trial.get("test_auc")), _num(trial.get("oot_auc")),
                _num(trial.get("lift_head_5")), _num(trial.get("lift_head_10")),
                _num(trial.get("lift_tail_5")), _num(trial.get("lift_tail_10")),
                _num(gap_tt), _num(trial.get("overfit_gap_to")),
            ])
        tables.append({
            "title": "trials 排行(按 in-time 选优;前15)",
            "columns": [
                "#", "train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc",
                "头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%",
                "过拟合gap(tt)", "过拟合gap(to)",
            ],
            "rows": rows,
        })
    if best_metrics:
        tables.append({"title": "最优 trial 指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in best_metrics.items()]})
    if best_params:
        tables.append({"title": "最优超参", "columns": ["参数", "值"], "rows": [[k, _fmt(v)] for k, v in best_params.items()]})
    return text, tables


def _render_train(o: dict):
    metrics = o.get("metrics") or {}
    text = "**训练完成**。"
    tables = []
    if metrics:
        scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
        if scalar:
            tables.append({"title": "模型指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in scalar.items()]})
    importance = o.get("feature_importance") or []
    rows = []
    for item in importance[:15]:
        if isinstance(item, (list, tuple)) and item:
            rows.append([str(item[0]), _fmt(item[1]) if len(item) > 1 else ""])
    if rows:
        tables.append({"title": "特征重要性(前15)", "columns": ["特征", "重要性"], "rows": rows})
    return text, tables


def _render_train_models(o: dict):
    experiments = [e for e in (o.get("experiments") or []) if isinstance(e, dict)]
    best_id = o.get("best_experiment_id")
    best_recipe = o.get("best_recipe")
    target_type = str(o.get("target_type") or "binary")
    tables = []
    rows = []
    best_metrics: dict = {}
    if target_type == "continuous":
        metric_columns = ["train_rmse", "test_rmse", "oot_rmse", "test_mae", "oot_mae", "test_r2", "oot_r2"]
        selector_label = "按 OOT RMSE"
    elif target_type == "multiclass":
        metric_columns = ["train_macro_auc", "test_macro_auc", "oot_macro_auc", "test_logloss", "oot_logloss", "test_accuracy", "oot_accuracy"]
        selector_label = "按 OOT macro-AUC"
    else:
        metric_columns = ["train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc"]
        selector_label = "按 OOT KS"
    for exp in experiments:
        metrics = exp.get("metrics") or {}
        is_best = exp.get("experiment_id") == best_id
        if is_best:
            best_metrics = metrics
        rows.append(
            [str(exp.get("recipe", "?")) + (" ★" if is_best else "")]
            + [_num(metrics.get(column)) for column in metric_columns]
        )
    if len(experiments) > 1:
        text = f"**训练完成**:对比 {len(experiments)} 个算法,最优 **{best_recipe}**(★;{selector_label})。"
        tables.append({
            "title": "候选模型对比",
            "columns": ["算法", *metric_columns],
            "rows": rows,
        })
    else:
        text = "**训练完成**。"
    # the best model's full metrics (mirrors the single-model 模型指标 table)
    scalar = {k: v for k, v in best_metrics.items() if isinstance(v, (int, float, str, bool))}
    if scalar:
        tables.append({"title": "模型指标", "columns": ["指标", "值"], "rows": [[k, _fmt(v)] for k, v in scalar.items()]})
    return text, tables


def _render_compare(o: dict):
    experiments = o.get("experiments") or []
    rows = []
    for exp in experiments:
        if not isinstance(exp, dict):
            continue
        caps = exp.get("capabilities") or {}
        rows.append([
            exp.get("recipe") or "?",
            "是" if caps.get("pmml_supported") else "否",
            "是" if caps.get("handoff_supported") else "否",
            "是" if caps.get("native_model_supported") else "否",
            caps.get("reason") or "",
        ])
    tables = []
    if rows:
        tables.append({
            "title": "训练后动作能力",
            "columns": ["算法", "PMML", "移交验证", "原生模型", "说明"],
            "rows": rows,
        })
    return f"**实验对比完成**:共 {len(experiments)} 个实验候选。", tables


def _render_report(o: dict):
    path = o.get("report_path") or ""
    sections = o.get("section_status") or []
    return f"**模型开发报告已生成**:`{path}`(共 {len(sections)} 个 sheet,可在右栏下载)。", []


def _num(value):
    return "n/a" if value is None else _fmt(value)


def _render_feature_metrics(o: dict):
    metrics = [metric for metric in (o.get("metrics") or []) if isinstance(metric, dict)]
    # The risk-aware head/tail lift columns show only when that metric was selected
    # (absent keys → not computed); base columns are always present.
    has_head_tail = any("lift_head_5" in metric for metric in metrics)
    has_importance = any("importance" in metric for metric in metrics)
    columns = ["特征", "IV", "KS", "AUC", "PSI", "缺失率", "头部lift"]
    if has_head_tail:
        columns += ["头部lift5%", "头部lift10%", "尾部lift5%", "尾部lift10%"]
    if has_importance:
        columns += ["重要性"]
    rows = []
    for metric in metrics:
        row = [
            str(metric.get("feature", "?")),
            _num(metric.get("iv")),
            _num(metric.get("ks")),
            _num(metric.get("auc")),
            _num(metric.get("psi")),
            _num(metric.get("missing_rate")),
            _num(metric.get("lift_top_bin")),
        ]
        if has_head_tail:
            row += [
                _num(metric.get("lift_head_5")),
                _num(metric.get("lift_head_10")),
                _num(metric.get("lift_tail_5")),
                _num(metric.get("lift_tail_10")),
            ]
        if has_importance:
            row += [_num(metric.get("importance"))]
        rows.append(row)
    text = (
        f"**特征分析完成**:{len(rows)} 个特征的指标如下"
        "(IV/KS/AUC 越高区分力越强;PSI/缺失率越低越稳)。可在右栏下载分析报告。"
    )
    tables = []
    if rows:
        tables.append({
            "title": "特征指标",
            "columns": columns,
            "rows": rows,
        })
    # Optional collinear / VIF section (computed only when the metric was selected).
    collinear = o.get("collinear")
    if isinstance(collinear, dict):
        vif = collinear.get("vif") or {}
        if vif:
            tables.append({
                "title": "VIF(共线性)",
                "columns": ["特征", "VIF"],
                "rows": [[str(feat), _num(value)] for feat, value in vif.items()],
            })
        pairs = [p for p in (collinear.get("collinear_pairs") or []) if isinstance(p, (list, tuple)) and len(p) >= 3]
        if pairs:
            tables.append({
                "title": "高相关特征对",
                "columns": ["特征A", "特征B", "相关系数"],
                "rows": [[str(p[0]), str(p[1]), _num(p[2])] for p in pairs],
            })
    return text, tables


def _render_feature_report(o: dict):
    # Reuse the metrics wide table (the tool echoes metrics) and append the report link.
    text, tables = _render_feature_metrics(o)
    path = o.get("report_path") or ""
    if path:
        text += f"\n\n**特征分析报告已生成**:`{path}`(可在右栏下载)。"
    return text, tables


def _render_propose_join(o: dict):
    joins = o.get("joins") or []
    rows = []
    relax_rows = []
    any_conflict = False
    any_fp_mismatch = False
    for j in joins:
        diag = j.get("diagnostics") or {}
        match_rate = diag.get("match_rate")
        unique = diag.get("feature_key_unique")
        fan_out = diag.get("fan_out_detected", diag.get("fan_out"))
        # Prefer the friendly file name (features.parquet) over the raw ds_<hash> id.
        fname = str(j.get("feature_name") or j.get("feature_id", "?"))
        key_pairs = j.get("key_pairs") or []
        keys = ", ".join(f"{p.get('anchor_col')}={p.get('feature_col')}" for p in key_pairs) or "?"
        # Dynamic key relaxation proposals (spec §4/§5): low-match keys may match better with
        # one element dropped — surface as suggestions (the user confirms; never auto-applied).
        for alt in (diag.get("key_alternatives") or []):
            if not isinstance(alt, dict):
                continue
            alt_keys = ", ".join(f"{a}={f}" for a, f in (alt.get("key_pairs") or []))
            relax_rows.append([
                fname,
                _fmt(match_rate) if match_rate is not None else "n/a",
                f"减「{alt.get('dropped', '?')}」→ {alt_keys}",
                _fmt(alt.get("match_rate")) if alt.get("match_rate") is not None else "n/a",
                "是" if alt.get("feature_key_unique") else "否",
                "⚠️是" if alt.get("fan_out_detected") else "否",
            ])
        # Fingerprint consistency (spec §5 C2 "指纹 raw=md5? ✓/✗"): transform_side == "both"
        # means anchor and feature key share format (both raw or both md5); anything else
        # means one side is raw and the other md5 (键格式不一致), joinable only via a hash
        # transform — surfaced so the user can sanity-check the key before执行.
        fp_consistent = all((p.get("transform_side") or "both") == "both" for p in key_pairs) if key_pairs else True
        if not fp_consistent:
            any_fp_mismatch = True
        fp_cell = "✓" if fp_consistent else "✗ raw≠md5"
        # Two-level dedup breakdown (spec §6): safe whole-row dups vs same-key conflicts.
        report = diag.get("conflict_report") or {}
        conflict_keys = int(report.get("n_conflict_keys") or 0)
        safe_dropped = int(report.get("safe_dropped") or 0)
        if conflict_keys:
            any_conflict = True
        dedup_cell = "-" if unique else f"安全{safe_dropped}/⚠️冲突{conflict_keys}"
        rows.append([
            fname,
            keys,
            fp_cell,
            _fmt(match_rate) if match_rate is not None else "n/a",
            "是" if unique else "否",
            "⚠️是" if fan_out else "否",
            dedup_cell,
        ])
    text = (
        f"**拼接诊断完成**:{len(joins)} 张特征表待左连接到锚样本(锚行数 **1:1 保留**)。\n"
        "请核对每张表的命中率/键唯一性/是否膨胀。键不唯一的特征需选去重策略;确认后才会真正执行拼接。"
    )
    if any_conflict:
        text += (
            "\n\n⚠️ 检测到**同键值冲突**(同一键多行但特征值不一致):这类**不会自动删除**,"
            "请先确认去重策略或清洗数据后再拼接。"
        )
    if any_fp_mismatch:
        text += (
            "\n\n⚠️ 检测到**键指纹不一致**(`✗ raw≠md5`:锚/特征侧一为原文、一为 md5):"
            "系统会自动对齐哈希后再连接,但请确认这是同一标识(避免误配)。"
        )
    if relax_rows:
        text += (
            "\n\n💡 部分特征表命中率偏低,**减一个识别要素**可提高命中(见下「择键建议」):"
            "系统只提议、不会自动改键;若减后**膨胀**需配合去重策略。请确认后再选用。"
        )
    tables = []
    if rows:
        tables.append({
            "title": "拼接诊断(逐特征表)",
            "columns": ["特征表", "匹配键", "指纹(raw=md5?)", "命中率", "键唯一", "膨胀", "去重(安全/冲突键)"],
            "rows": rows,
        })
    if relax_rows:
        tables.append({
            "title": "择键建议(减要素换更高命中)",
            "columns": ["特征表", "当前命中率", "建议键", "减后命中率", "减后唯一", "减后膨胀"],
            "rows": relax_rows,
        })
    return text, tables


def _render_confirm_join(o: dict):
    # Internal plumbing step (marks engine specs confirmed). It is a dependency of
    # the execute_join gate, but its summary would show "已确认…" before the human
    # actually confirms, which is confusing — so render nothing at the gate…
    needs = o.get("needs_dedup") or []
    if needs:
        # …UNLESS a feature has a same-key conflict (spec §6): surface it so the user knows
        # the join can't execute until they pick a dedup strategy (or exclude the feature).
        labels = o.get("needs_dedup_labels") or {}
        listed = "、".join(f"`{labels.get(f, f)}`" for f in needs)
        return (
            f"⚠️ 特征 {listed} 存在**同键冲突**(同一键多行、特征值不一致),"
            "需先定去重策略才能拼接。回复「去重 first」(保留首条)或「去重 last」(保留末条)解决;"
            "或排除这些特征后重试。"
        ), []
    return "", []


def _render_execute_join(o: dict):
    anchor_rows = o.get("anchor_rows")
    joined_rows = o.get("joined_rows")
    ok = anchor_rows == joined_rows
    text = (
        f"**拼接执行完成**:结果数据集 `{o.get('result_dataset_id', '')}`,"
        f"锚行 {anchor_rows} → 拼接后 {joined_rows} 行"
        + ("(1:1 保持 ✓)" if ok else "(⚠️ 行数发生变化,请检查膨胀)")
    )
    warnings = o.get("warnings") or []
    if warnings:
        text += "\n警告:" + "; ".join(str(w) for w in warnings)
    # §8 per-table contribution summary from real diagnostics.
    tables = []
    per_table = [row for row in (o.get("per_table") or []) if isinstance(row, dict)]
    if per_table:
        tables.append({
            "title": "各特征表贡献",
            "columns": ["特征表", "命中率", "新增列", "新列缺失率", "去重策略"],
            "rows": [
                [
                    str(row.get("feature_id", "?")),
                    _num(row.get("match_rate")),
                    str(row.get("new_columns", "")),
                    _num(row.get("new_columns_null_rate")),
                    str(row.get("dedup_strategy", "无")),
                ]
                for row in per_table
            ],
        })
    return text, tables


def _render_make_split(o: dict):
    """G1 split gate: surface the train/test/oot counts + per month/channel distribution so
    the user can sanity-check the split (proportions, OOT-by-time, no cross-group leakage)
    before spending compute on screening/training."""
    analysis = o.get("sample_analysis") or {}
    counts = analysis.get("split_counts") or {}
    total = analysis.get("total_rows")
    rows = [
        [str(split), int(n), _fmt(n / total) if total else "n/a"]
        for split, n in counts.items()
    ]
    text = (
        f"**样本切分完成**:共 {total} 行。请核对 train/test/oot 划分"
        "(占比是否合理、OOT 是否按时间、分组是否防泄漏)后再继续。"
    )
    tables = []
    if rows:
        tables.append({
            "title": "切分计数(train/test/oot)",
            "columns": ["划分", "行数", "占比"],
            "rows": rows,
        })
    for group_col, dist in (analysis.get("group_distributions") or {}).items():
        if not isinstance(dist, dict):
            continue
        group_values = sorted({gv for per in dist.values() if isinstance(per, dict) for gv in per})
        grows = [
            [str(split)] + [int(per.get(gv, 0)) for gv in group_values]
            for split, per in dist.items() if isinstance(per, dict)
        ]
        if grows:
            tables.append({
                "title": f"按「{group_col}」分布(逐划分)",
                "columns": ["划分", *[str(gv) for gv in group_values]],
                "rows": grows,
            })
    return text, tables


_RENDERERS = {
    "make_split": _render_make_split,
    "screen_features": _render_screen,
    "tune_hyperparameters": _render_tune,
    "train_model": _render_train,
    "train_models": _render_train_models,
    "compare_experiments": _render_compare,
    "generate_model_report": _render_report,
    "propose_join": _render_propose_join,
    "confirm_join": _render_confirm_join,
    "execute_join": _render_execute_join,
    "compute_feature_metrics": _render_feature_metrics,
    "generate_feature_report": _render_feature_report,
}


def _render_generic(o: dict):
    if not isinstance(o, dict) or not o:
        return "已完成。", []
    scalar = {k: v for k, v in o.items() if isinstance(v, (str, int, float, bool))}
    if scalar:
        head = ", ".join(f"{k}={_fmt(v)}" for k, v in list(scalar.items())[:6])
        return f"已完成:{head}", []
    return "已完成。", []


def render_tool_output(tool: str, output: dict):
    """Render a tool's raw output to (text, tables); falls back to generic."""
    renderer = _RENDERERS.get(tool, _render_generic)
    try:
        return renderer(output or {})
    except Exception:
        return _render_generic(output or {})


__all__ = [
    "PlanDriver",
    "DriverMessage",
    "DriverTurn",
    "DriverError",
    "is_confirm",
    "render_tool_output",
]
