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

from marvis.agent.driver_turn import DriverTurn
from marvis.agent.plan_utils import find_step
from marvis.orchestrator.contracts import Plan, PlanStep


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
# strategy-monitoring disposition adapter (render_monitoring_report gate)
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
    """render_monitoring_report gate (the report step IS the gate, rendering its
    run_strategy_monitoring dependency's verdict + red-light checklist). A
    red-light reply naming one of the three dispositions (观察 / 调阈值 / 起新版本)
    is recorded onto the report gate's own `disposition` input before confirming
    the gate to proceed (so the report surfaces next_action)."""

    tool = "render_monitoring_report"

    def parse_reply(self, text: str, ctx: GateReplyContext) -> str | None:
        return parse_monitoring_disposition(text)

    def apply(self, driver, plan: Plan, gate: PlanStep, parsed: str, *, run_seq) -> DriverTurn:
        driver._apply_monitoring_disposition(gate, parsed)
        driver._repo.confirm_step(gate.id)
        return driver._run_and_handle(plan.id, run_seq=run_seq)

    def adjust_schema(self, plan: Plan, gate: PlanStep, load_output: Callable[[str], Any]) -> dict:
        return {
            "type": "object",
            "properties": {
                "disposition": {
                    "type": "string",
                    "title": "红灯处置（三选一）",
                    "enum": list(_MONITORING_DISPOSITIONS),
                }
            },
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
    "parse_dedup_instruction",
    "parse_monitoring_disposition",
    "parse_rule_selection_instruction",
]
