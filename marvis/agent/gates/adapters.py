"""Per-tool gate reply-adapter registry for PlanDriver (LT-3).

Renderers are already a ``tool -> renderer`` registry (``marvis/agent/renderers.py``);
gate *reply* parsing was not. The driver carried a handful of task-specific
"门回复解析器" inline -- the join dedup instruction, the rule-set text selection,
and the strategy-monitoring red-light disposition -- plus an ``if
gate.tool_ref.tool == ...`` dispatch chain in ``resume()``.

This module collects each of those into a small adapter, keyed by the gate step's
own source tool, so ``PlanDriver`` only depends on a registry lookup and no longer
imports task-specific parsing/dispatch details. Every adapter is a *mechanical*
move of the existing logic -- behaviour is unchanged; only the location moved.

Each adapter exposes three things:

* ``parse_reply(text, ctx) -> object | None`` -- turn a free-text gate reply into a
  structured instruction, or ``None`` when the reply is not for this adapter (so
  the driver falls through to the generic confirm / LLM-router path).
* ``apply(driver, plan, gate, parsed, *, run_seq) -> DriverTurn | None`` -- act on
  the parsed instruction against the current gate, returning the driver turn (or
  ``None`` when it turned out to be a no-op, e.g. a dedup instruction at a gate with
  no pending conflicts, so the driver falls back to the generic path).
* ``adjust_schema(driver, plan, gate) -> dict`` -- declare this adapter's adjustable
  parameters as a JSON schema (``{"type": "object", "properties": {...}}``),
  surfaced onto the gate payload as ``editable_input_schema`` (aligned with the
  LT-4 retry form's key) so the frontend has a real schema for the gate's controls
  rather than only the type-inferred controls.

``GateReplyContext`` carries the small pieces of driver state an adapter's parser
needs (currently just the candidate count for the rule-set adapter); the driver
builds it lazily so parsers that do not need it pay nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from marvis.agent.driver_turn import DriverMessage, DriverTurn
from marvis.agent.plan_utils import find_step
from marvis.orchestrator.contracts import Plan, PlanStep
from marvis.strategy_adoption import ADOPTION_REASON_MIN_LENGTH


def _pending_dedup_features(plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> list[str]:
    """Feature ids a join confirmation dependency still needs a dedup strategy for --
    the same read GateExecutionAdapter.needs_dedup_features does, reimplemented here
    so adjust_schema stays decoupled from the driver."""
    for dep_id in gate.depends_on or []:
        dep = find_step(plan, dep_id)
        if dep is None or dep.tool_ref.tool != "confirm_join":
            continue
        output = load_output(dep.id)
        if not isinstance(output, dict):
            return []
        return [str(feature) for feature in (output.get("needs_dedup") or [])]
    return []


def _rule_candidate_count(plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> int:
    """How many mined candidate rules a select_rule_set gate is choosing from --
    the same read PlanDriver._rule_candidate_count does (mine_rules dependency
    output, then the gate's own resolved candidate_rules input, then 0)."""
    for dep_id in gate.depends_on or []:
        dep = find_step(plan, dep_id)
        if dep is None or dep.tool_ref.tool != "mine_rules":
            continue
        output = load_output(dep.id)
        if isinstance(output, dict) and isinstance(output.get("candidate_rules"), list):
            return len(output["candidate_rules"])
    candidates = (gate.inputs or {}).get("candidate_rules")
    return len(candidates) if isinstance(candidates, list) else 0


def _monitoring_run_output(
    plan: Plan,
    gate: PlanStep,
    load_output: Callable[[str], Any],
) -> dict:
    """Return the evidence output reviewed by a monitoring disposition gate."""

    for dep_id in gate.depends_on or []:
        dep = find_step(plan, dep_id)
        if dep is None or dep.tool_ref.tool != "run_strategy_monitoring":
            continue
        output = load_output(dep.id)
        return dict(output) if isinstance(output, dict) else {}
    return {}


@dataclass(frozen=True)
class GateReplyContext:
    """Driver-side state a gate reply parser may need, adapter-agnostic.

    The driver builds this once per gate with the current ``plan`` and its
    ``load_output(step_id)`` callback; each adapter's parser derives whatever it
    needs from them (e.g. the rule-set adapter reads its mine_rules dependency's
    candidate count) so the driver stays free of any adapter-specific knowledge.
    """

    plan: Plan
    gate: PlanStep
    load_output: Callable[[str], Any]

    def rule_candidate_count(self) -> int:
        return _rule_candidate_count(self.plan, self.gate, self.load_output)

    def feature_metric_candidates(self) -> list[str]:
        for dep_id in self.gate.depends_on or []:
            dep = find_step(self.plan, dep_id)
            if dep is None or dep.tool_ref.tool != "compute_feature_metrics":
                continue
            output = self.load_output(dep.id)
            if not isinstance(output, dict):
                return []
            return [
                str(item.get("feature")).strip()
                for item in (output.get("metrics") or [])
                if isinstance(item, dict) and str(item.get("feature") or "").strip()
            ]
        return []

    def special_value_context(self) -> tuple[list[str], dict[str, list]]:
        """Return the current screen selection and its detected sentinel rows.

        ``resolve_special_values`` is deliberately a gate *after*
        ``screen_features``.  Keeping this lookup in the adapter context lets
        both natural-language parsing and JSON-schema rendering consume the
        same persisted screen evidence without teaching PlanDriver how to
        interpret a specific tool's output.
        """
        for dep_id in self.gate.depends_on or []:
            dep = find_step(self.plan, dep_id)
            if dep is None or dep.tool_ref.tool != "screen_features":
                continue
            output = self.load_output(dep.id)
            if not isinstance(output, dict):
                return [], {}
            selected = [
                str(item).strip()
                for item in (output.get("selected") or [])
                if str(item).strip()
            ]
            raw_columns = output.get("sentinel_columns")
            sentinel_columns = {
                str(column): list(rows)
                for column, rows in (raw_columns.items() if isinstance(raw_columns, dict) else [])
                if isinstance(rows, (list, tuple)) and rows
            }
            return selected, sentinel_columns
        return [], {}


class GateReplyAdapter(Protocol):
    """The minimal interface PlanDriver dispatches through."""

    tool: str

    def parse_reply(self, text: str, ctx: GateReplyContext) -> Any | None: ...

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: Any, *, run_seq) -> DriverTurn | None: ...

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict: ...


# ---------------------------------------------------------------------------
# join dedup adapter (confirm_join gate)
# ---------------------------------------------------------------------------
def parse_dedup_instruction(text: str) -> str | None:
    """Parse a manual-mode dedup reply at a join gate -> "first"/"last"/None.

    Recognised only when the text actually mentions de-duplication (去重/dedup/策略/保留)
    so an unrelated instruction isn't misread as a strategy. first = keep the first row per
    key, last = keep the last (spec section 6 conflict resolution)."""
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


class _JoinDedupAdapter:
    """execute_join gate: a text dedup instruction ("去重 first"/"用 last 去重")
    applies that strategy to every feature the gate's confirm_join DEPENDENCY
    flagged as needs_dedup, then re-pauses at the cleared gate. Keyed on the gate
    step's own tool (execute_join); confirm_join is that gate's dependency, which
    is where needs_dedup lives. Mirrors the join dedup picker but without the
    structured picker payload. A no-op (returns None) at any execute_join gate
    with no pending conflicts, so a non-dedup instruction there is unaffected."""

    tool = "execute_join"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> str | None:
        return parse_dedup_instruction(text)

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: str, *, run_seq) -> DriverTurn | None:
        pending = driver._gate_execution.needs_dedup_features(plan, gate)
        if not pending:
            return None
        driver._gate_execution.apply_dedup_strategies(plan, gate, {fid: parsed for fid in pending})
        return driver._run_and_handle(plan.id, run_seq=run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        pending = _pending_dedup_features(plan, gate, load_output)
        if not pending:
            return {}
        return {
            "type": "object",
            "properties": {
                "dedup_strategies": {
                    "type": "object",
                    "title": "去重策略（逐特征）",
                    "propertyNames": {"enum": [str(fid) for fid in pending]},
                    "additionalProperties": {"type": "string", "enum": ["first", "last"]},
                }
            },
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# rule-set selection adapter (select_rule_set gate)
# ---------------------------------------------------------------------------
_SELECT_ALL = re.compile(r"(全选|都要|全部|全都|all)", re.IGNORECASE)
_DROP_PREFIX = re.compile(r"(去掉|去除|删掉|删除|移除|排除|不要|drop|remove|exclude)", re.IGNORECASE)
_KEEP_PREFIX = re.compile(r"(只?选|保留|选中|选择|要|keep|select|pick|use)", re.IGNORECASE)
_INDEX_TOKEN = re.compile(r"\d+")
_RULE_MENTION = re.compile(r"(规则|规则集|rule|条|第)", re.IGNORECASE)
_QUESTION = re.compile(
    r"[?？]|吗|吧$|行不行|可不可以|能不能|好不好|对不对|是不是|呢$",
    re.IGNORECASE,
)


def parse_rule_selection_instruction(text: str, candidate_count: int) -> list[int] | None:
    """Parse a rule-set gate reply into an ordered list of 1-based indices.

    Recognises three shapes (spec section 3, parallel to parse_dedup_instruction):
      * 「全选」/「都要」/「all」                -> keep every candidate, in order;
      * 「去掉 2」/「去除 2 4」/「drop 2」        -> all candidates except those indices;
      * 「选 1,3,5」/「保留 1 3 5」/「pick 1 3」  -> exactly those indices, in the
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


class _RuleSelectionAdapter:
    """select_rule_set gate: a text rule-set selection ("选 1,3,5"/"去掉 2"/"全选")
    is parsed into a 1-based index list and pushed through the SAME generic
    apply_adjust override channel band_edges uses (the gate step's own `selection`
    input, default None, is overwritten and the gate re-armed)."""

    tool = "select_rule_set"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> list[int] | None:
        return parse_rule_selection_instruction(text, ctx.rule_candidate_count())

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: list[int], *, run_seq) -> DriverTurn:
        return driver._gate_execution.apply_adjust(plan, gate, {"selection": parsed}, run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        candidate_count = _rule_candidate_count(plan, gate, load_output)
        if candidate_count <= 0:
            return {}
        return {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "array",
                    "title": "规则集选择（1-based 序号，按命中顺序）",
                    "items": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": candidate_count,
                    },
                }
            },
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# strategy-monitoring disposition adapter (apply_monitoring_disposition gate)
# ---------------------------------------------------------------------------
_MONITORING_NEW_VERSION = re.compile(
    r"(?:起新版本|新版本|新起一版|起一版|重起)|\bnew\s+(?:version|strategy)\b",
    re.IGNORECASE,
)
_MONITORING_ADJUST = re.compile(
    r"(?:调阈值|改阈值|调整阈值|调门槛)|\badjust\s+threshold\b",
    re.IGNORECASE,
)
_MONITORING_OBSERVE = re.compile(
    r"观察|\b(?:keep\s+watch(?:ing)?|observe)\b",
    re.IGNORECASE,
)
_MONITORING_QUESTION = re.compile(
    r"[?？]|(?:吗|呢)\s*[。.!！]?$|是否|是不是|要不要|该不该|能不能|可不可以|还是",
    re.IGNORECASE,
)
_MONITORING_NEGATION = re.compile(
    r"(?:不|没|无|勿|别|反对|拒绝|取消|暂停|停止|暂缓)|"
    r"(?:(?:不建议|没有必要|没必要|不认为|不主张|不应|不该)"
    r"[^，。；;！？!?\n]{0,12}(?:观察|阈值|版本))|"
    r"(?:不要|不用|无需|不需要|勿|别|取消|暂停|停止|暂不|先不|"
    r"不(?:观察|调|调整|改|起|开|做|维持|保持|继续|采用|选择))|"
    r"\b(?:do\s+not|don't|dont|not|never|without)\b",
    re.IGNORECASE,
)

_MONITORING_DISPOSITIONS = ("observe", "adjust_threshold", "new_version")


def parse_monitoring_disposition(text: str) -> str | None:
    """Parse a strategy-monitoring alarm-gate reply into a disposition keyword.

    Recognises exactly one explicit red-light checklist choice (spec S5):
      * 「起新版本」/「新版本」/「new version」        -> "new_version"
      * 「调阈值」/「调整阈值」/「adjust threshold」    -> "adjust_threshold"
      * 「观察」/「observe」/「keep watch」             -> "observe"

    Questions, negations, generic wording such as 「保持」, and replies naming
    multiple choices return None so they fall through to the normal router.
    """
    raw = text or ""
    if _MONITORING_QUESTION.search(raw) or _MONITORING_NEGATION.search(raw):
        return None
    matches = [
        disposition
        for disposition, pattern in (
            ("new_version", _MONITORING_NEW_VERSION),
            ("adjust_threshold", _MONITORING_ADJUST),
            ("observe", _MONITORING_OBSERVE),
        )
        if pattern.search(raw)
    ]
    return matches[0] if len(matches) == 1 else None


class _MonitoringDispositionAdapter:
    """Govern the real effect of a strategy-monitoring alarm decision.

    The gate is bound to the immutable plan/run receipt rendered from its
    ``run_strategy_monitoring`` dependency. Observe and new-version are complete
    explicit decisions and execute immediately. Threshold adjustment is a
    two-part decision: choosing it records the disposition, then the user must
    provide a concrete threshold patch and explicitly confirm that patch.
    """

    tool = "apply_monitoring_disposition"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> str | None:
        return parse_monitoring_disposition(text)

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: str, *, run_seq) -> DriverTurn:
        reason = f"人工选择监控处置：{parsed}"
        driver._apply_monitoring_disposition(gate, parsed, reason=reason)
        if parsed == "adjust_threshold" and not isinstance(
            (gate.inputs or {}).get("threshold_patch"), dict
        ):
            return DriverTurn(
                plan.id,
                plan.status.value,
                [
                    DriverMessage(
                        "gate",
                        "已选择「调阈值重跑」。请说明要调整的监控项及 warn/fail 数值；"
                        "Agent 会生成结构化阈值补丁，展示后仍需你明确确认才会追加计划版本并重跑。",
                        {
                            "plan_id": plan.id,
                            "step_id": gate.id,
                            "run_seq": run_seq,
                        },
                    )
                ],
            )
        driver._confirm_gate(
            plan,
            gate,
            reason=reason,
        )
        return driver._run_and_handle(plan.id, run_seq=run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        output = _monitoring_run_output(plan, gate, load_output)
        raw_threshold_ids = output.get("adjustable_threshold_ids")
        threshold_ids = (
            list(
                dict.fromkeys(
                    value.strip()
                    for value in raw_threshold_ids
                    if isinstance(value, str) and value.strip()
                )
            )
            if isinstance(raw_threshold_ids, list)
            else []
        )
        patch_schema = {
            "type": "object",
            "title": "阈值补丁",
            "description": "只填写要调整的现有监控项及 warn/fail；其余计划证据保持不变。",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "warn": {"type": "number"},
                    "fail": {"type": "number"},
                },
                "minProperties": 1,
                "additionalProperties": False,
            },
        }
        if threshold_ids:
            patch_schema["propertyNames"] = {"enum": threshold_ids}
        else:
            # Fail closed when the immutable run receipt does not declare the
            # exact monitoring-plan keys that may be changed.  Check ``metric``
            # aliases are presentation data and must never become patch ids.
            patch_schema["maxProperties"] = 0
        return {
            "type": "object",
            "properties": {
                "disposition": {
                    "type": "string",
                    "title": "红灯处置（三选一）",
                    "enum": list(_MONITORING_DISPOSITIONS),
                },
                "reason": {
                    "type": "string",
                    "title": "处置理由",
                    "minLength": 4,
                },
                "threshold_patch": patch_schema,
            },
            "additionalProperties": False,
        }


def monitoring_plain_confirm_error(
    plan: Plan,
    gate: PlanStep | None,
    load_output: Callable[[str], Any],
) -> str | None:
    """Explain why a bare confirmation cannot execute a red-light gate.

    Green/amber runs may be acknowledged with a plain confirmation. A red run
    must name one of the three dispositions; threshold adjustment additionally
    needs a non-empty patch. This guard prevents ``确认`` from silently becoming
    ``observe`` or from executing an empty threshold revision.
    """

    if gate is None or gate.tool_ref.tool != "apply_monitoring_disposition":
        return None
    output = _monitoring_run_output(plan, gate, load_output)
    if str(output.get("overall_level") or "") != "red":
        return None
    inputs = gate.inputs or {}
    disposition = inputs.get("disposition")
    if disposition not in _MONITORING_DISPOSITIONS:
        return "本次监控为红灯，不能只回复「确认」。请明确选择「观察」「调阈值」或「起新版本」。"
    if disposition == "adjust_threshold" and not inputs.get("threshold_patch"):
        return "已选择调阈值，但还没有具体 warn/fail 补丁；请先说明要调整的监控项和数值。"
    return None


# ---------------------------------------------------------------------------
# strategy adoption reason adapter (adopt_strategy gate)
# ---------------------------------------------------------------------------
class _AdoptionReasonAdapter:
    """Declare the gate-time business reason required by ``adopt_strategy``.

    The value is submitted through the structured ``adjust_params`` channel and
    consumed atomically by :class:`PlanDriver`; this adapter intentionally does
    not guess a reason from arbitrary free text.
    """

    tool = "adopt_strategy"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> None:
        return None

    def apply(
        self,
        driver,
        plan: Plan,
        gate: PlanStep,
        parsed: Any,
        *,
        run_seq,
    ) -> None:
        return None

    def adjust_schema(
        self,
        plan: Plan,
        gate: PlanStep,
        load_output: Callable[[str], Any],
    ) -> dict:
        return {
            "type": "object",
            "properties": {
                "adoption_reason": {
                    "type": "string",
                    "title": "采纳理由",
                    "description": "说明基于当前策略与回测证据采纳该版本的业务理由。",
                    "minLength": ADOPTION_REASON_MIN_LENGTH,
                }
            },
            "required": ["adoption_reason"],
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# optional feature-binning adapter (analyze_feature_bins gate)
# ---------------------------------------------------------------------------
_BINNING_SKIP = re.compile(r"(跳过|不做|不用|无需|不需要).{0,8}(分箱)?|直接.{0,6}报告", re.IGNORECASE)
_BINNING_MENTION = re.compile(r"(分箱|箱分析|binning|bins?)", re.IGNORECASE)
_BIN_COUNT = re.compile(r"(?:分|做|按)?\s*(\d{1,2})\s*箱|(?:bins?|箱数)\s*[=:：]?\s*(\d{1,2})", re.IGNORECASE)


class _FeatureBinningAdapter:
    tool = "analyze_feature_bins"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> dict | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        candidates = ctx.feature_metric_candidates()
        count_match = _BIN_COUNT.search(raw)
        bins = int(next(group for group in count_match.groups() if group)) if count_match else 10
        if _BINNING_SKIP.search(raw):
            return {"features": [], "bins": bins}
        selected = [
            feature for feature in candidates
            if re.search(rf"(?<![\w]){re.escape(feature)}(?![\w])", raw, re.IGNORECASE)
        ]
        if selected and (_BINNING_MENTION.search(raw) or count_match):
            return {"features": selected, "bins": bins}
        return None

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: dict, *, run_seq) -> DriverTurn:
        error = driver._feature_binning_adjust_error(plan, gate, parsed["features"])
        if not 3 <= int(parsed["bins"]) <= 20:
            error = "分箱数必须是 3 到 20 之间的整数。"
        if error:
            return DriverTurn(
                plan.id,
                plan.status.value,
                [driver._composer.instruction_message(plan, gate, run_seq=run_seq, text=error)],
            )
        driver._confirm_gate(
            plan,
            gate,
            reason=(
                f"用户通过自然语言选择 {len(parsed['features'])} 个特征进行 {parsed['bins']} 箱分析"
                if parsed["features"]
                else "用户通过自然语言选择跳过可选分箱分析"
            ),
            input_updates={"features": parsed["features"], "bins": parsed["bins"]},
        )
        return driver._run_and_handle(plan.id, run_seq=run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        ctx = GateReplyContext(plan=plan, gate=gate, load_output=load_output)
        candidates = ctx.feature_metric_candidates()
        return {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "title": "需要分箱分析的特征（可为空）",
                    "items": {"type": "string", "enum": candidates},
                    "uniqueItems": True,
                },
                "bins": {
                    "type": "integer",
                    "title": "分箱数",
                    "minimum": 3,
                    "maximum": 20,
                    "default": 10,
                },
            },
            "required": ["features", "bins"],
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# special-value governance adapter (resolve_special_values gate)
# ---------------------------------------------------------------------------
_SPECIAL_MASK = re.compile(
    r"(?:转(?:为)?(?:空值?|缺失值?)|置空|设为(?:空|缺失)|按缺失处理|替换为\s*(?:nan|na)|\bmask\b|\bnan\b)",
    re.IGNORECASE,
)
_SPECIAL_DROP = re.compile(r"(?:删除|剔除|去掉|移除|不用|不使用|\bdrop\b|\bremove\b)", re.IGNORECASE)
_SPECIAL_RETAIN = re.compile(r"(?:保留|原样|照常使用|\bretain\b|\bkeep\b)", re.IGNORECASE)
_SPECIAL_ALL = re.compile(r"(?:全部|所有|都|all)", re.IGNORECASE)
_SPECIAL_REASON = re.compile(r"(?:原因|理由)\s*[:：]\s*(.+)$", re.IGNORECASE)
_SPECIAL_NEGATED_ACTION = re.compile(
    r"(?:不|不要|别|勿|无需|不需要)\s*"
    r"(?:转(?:为)?(?:空值?|缺失值?)|置空|设为(?:空|缺失)|按缺失处理|"
    r"删除|剔除|去掉|移除|保留|原样|照常使用|mask|drop|remove|retain|keep)"
    r"|(?:do\s+not|don't|dont)\s+(?:mask|drop|remove|retain|keep)",
    re.IGNORECASE,
)


def _special_action(text: str) -> str | None:
    if _SPECIAL_NEGATED_ACTION.search(text):
        return None
    matches = [
        action
        for action, pattern in (
            ("mask", _SPECIAL_MASK),
            ("drop", _SPECIAL_DROP),
            ("retain", _SPECIAL_RETAIN),
        )
        if pattern.search(text)
    ]
    return matches[0] if len(matches) == 1 else None


def _special_reason(text: str) -> str:
    match = _SPECIAL_REASON.search(text)
    return str(match.group(1) if match else "").strip(" \t，,。；;")


def _special_unknown_column_hint(clause: str) -> str:
    """Return a stable label for an action clause naming no known column.

    Unknown clauses must survive parsing so the driver can reject them against
    persisted screen evidence. Silently dropping ``ghost 删除`` would otherwise
    let a message that also covered every real column advance the gate.
    """

    candidate = _SPECIAL_REASON.sub("", str(clause or ""))
    for pattern in (_SPECIAL_MASK, _SPECIAL_DROP, _SPECIAL_RETAIN):
        candidate = pattern.sub("", candidate)
    candidate = re.sub(
        r"(?:请|把|将|对|字段|特征|变量|这一?列|该列)",
        " ",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = candidate.strip(" \t，,。；;:：")
    return candidate[:128] or "__unknown_instruction__"


def _special_named_columns(text: str, columns: list[str]) -> list[str]:
    """Find exact column mentions without treating ``x1`` as part of ``x10``."""

    named: list[str] = []
    for column in columns:
        escaped = re.escape(column)
        prefix = r"(?<![A-Za-z0-9_])" if column[:1].isalnum() or column.startswith("_") else ""
        suffix = r"(?![A-Za-z0-9_])" if column[-1:].isalnum() or column.endswith("_") else ""
        if re.search(prefix + escaped + suffix, text):
            named.append(column)
    return named


def parse_special_value_instruction(
    text: str,
    *,
    selected: list[str],
    sentinel_columns: dict[str, list],
) -> dict[str, dict] | None:
    """Compile a concise natural-language policy into tool decisions.

    Supported examples include ``全部转空`` and
    ``x1 转空；x2 删除；x3 保留，原因：业务约定值``.  Ambiguous clauses or
    incomplete coverage return ``None``/a partial mapping so the adapter can
    ask for a precise decision instead of guessing.
    """
    raw = str(text or "").strip()
    relevant = [column for column in selected if column in sentinel_columns]
    if not raw or not relevant:
        return None

    decisions: dict[str, dict] = {}
    explicitly_named = _special_named_columns(raw, relevant)
    global_action = (
        _special_action(raw)
        if _SPECIAL_ALL.search(raw) and not explicitly_named
        else None
    )
    if global_action:
        reason = _special_reason(raw)
        if global_action == "retain" and not reason:
            return None
        for column in relevant:
            decision = {
                "action": global_action,
                "values": [
                    row[0] if isinstance(row, (list, tuple)) and row else row
                    for row in sentinel_columns[column]
                ],
            }
            if global_action == "retain":
                decision.update({"confirmed": True, "reason": reason})
            decisions[column] = decision
        return decisions

    clauses = [
        clause.strip()
        for clause in re.split(r"[\n；;]+", raw)
        if clause.strip()
    ]
    for clause in clauses:
        named = _special_named_columns(clause, relevant)
        action = _special_action(clause)
        if action is None:
            continue
        if not named:
            decisions[_special_unknown_column_hint(clause)] = {"action": action}
            continue
        reason = _special_reason(clause)
        if action == "retain" and not reason:
            continue
        for column in named:
            decision = {
                "action": action,
                "values": [
                    row[0] if isinstance(row, (list, tuple)) and row else row
                    for row in sentinel_columns[column]
                ],
            }
            if action == "retain":
                decision.update({"confirmed": True, "reason": reason})
            decisions[column] = decision
    return decisions or None


class _SpecialValueAdapter:
    tool = "resolve_special_values"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> dict | None:
        selected, sentinel_columns = ctx.special_value_context()
        return parse_special_value_instruction(
            text,
            selected=selected,
            sentinel_columns=sentinel_columns,
        )

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: dict, *, run_seq) -> DriverTurn:
        error = driver._special_value_adjust_error(plan, gate, parsed)
        if error:
            return DriverTurn(
                plan.id,
                plan.status.value,
                [driver._composer.instruction_message(plan, gate, run_seq=run_seq, text=error)],
            )
        driver._confirm_gate(
            plan,
            gate,
            reason="用户通过自然语言确认特殊值治理策略",
            input_updates={"decisions": parsed},
        )
        return driver._run_and_handle(plan.id, run_seq=run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        ctx = GateReplyContext(plan=plan, gate=gate, load_output=load_output)
        selected, sentinel_columns = ctx.special_value_context()
        relevant = [column for column in selected if column in sentinel_columns]
        decision_properties = {
            column: {
                "type": "object",
                "title": f"{column} 特殊值策略",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["mask", "retain", "drop"],
                    },
                    "confirmed": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            }
            for column in relevant
        }
        return {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "object",
                    "title": "特殊值治理决策",
                    "properties": decision_properties,
                    "required": relevant,
                    "additionalProperties": False,
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_ADAPTERS: dict[str, GateReplyAdapter] = {
    adapter.tool: adapter
    for adapter in (
        _JoinDedupAdapter(),
        _RuleSelectionAdapter(),
        _MonitoringDispositionAdapter(),
        _AdoptionReasonAdapter(),
        _FeatureBinningAdapter(),
        _SpecialValueAdapter(),
    )
}


def get_gate_adapter(gate: PlanStep | None) -> GateReplyAdapter | None:
    """The reply adapter for ``gate``'s source tool, or None when the gate has no
    task-specific reply parser (so the driver uses the generic confirm/route path)."""
    if gate is None or gate.tool_ref is None:
        return None
    return _ADAPTERS.get(gate.tool_ref.tool)


def gate_editable_input_schema(
    plan: Plan, gate: PlanStep | None, load_output: Callable[[str], Any]
) -> dict:
    """The adapter-declared adjustable-param JSON schema for ``gate`` (A.3), or an
    empty dict when the gate has no reply adapter or nothing adjustable right now.
    Surfaced onto the gate payload as ``editable_input_schema`` (LT-4 key).
    ``load_output(step_id)`` reads persisted dependency outputs (same callback the
    message composer already holds), so this stays decoupled from PlanDriver."""
    adapter = get_gate_adapter(gate)
    if adapter is None:
        return {}
    return adapter.adjust_schema(plan, gate, load_output)


__all__ = [
    "GateReplyAdapter",
    "GateReplyContext",
    "gate_editable_input_schema",
    "get_gate_adapter",
    "monitoring_plain_confirm_error",
    "parse_dedup_instruction",
    "parse_special_value_instruction",
    "parse_monitoring_disposition",
    "parse_rule_selection_instruction",
]
