"""Deterministic business slots for approval/reject strategy development.

The LLM compiler is the general grammar, but a development request only needs
four business facts: strategy type, objective, and at least one approval
constraint.  This module reads exactly those facts from the user's own words
so a clarification answer ("准入就是审批策略", "坏账不超过 5% 的前提下尽量提高
通过率") is never lost between turns.  It never supplies a default: a value is
either quoted from the utterance or left missing for the caller to ask about.
"""

from __future__ import annotations

import calendar
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
import re

STRATEGY_DEVELOPMENT_SLOT_NAMES = (
    "strategy_type",
    "objective",
    "max_bad_rate",
    "min_approval_rate",
    "baseline",
    "baseline_strategy_id",
)

_NEGATION = r"(?:不是|并非|而非|不做|不要|不用|别做|非)"

_APPROVAL_RE = re.compile(r"准入|审批|授信策略|approval", re.IGNORECASE)
_REJECT_RE = re.compile(r"拒绝(?!率)|拒单|reject", re.IGNORECASE)
_NEGATED_APPROVAL_RE = re.compile(_NEGATION + r"\s*(?:准入|审批|approval)", re.IGNORECASE)
_NEGATED_REJECT_RE = re.compile(_NEGATION + r"\s*(?:拒绝|拒单|reject)", re.IGNORECASE)

_MAX_PROFIT_RE = re.compile(
    r"max_profit|(?:利润|收益|盈利)\s*(?:最大|最高|最优|最多|尽量高|越高越好)|"
    r"(?:最大化|尽量提高|尽可能提高|提高|提升|拉高|追求)\s*(?:利润|收益|盈利)",
    re.IGNORECASE,
)
_MAX_APPROVAL_RE = re.compile(
    r"max_approval|通过率\s*(?:最大|最高|尽量高|越高越好|尽可能高)|"
    r"(?:最大化|尽量提高|尽可能提高|尽量提升|尽量拉高|尽量多|尽可能多|提高|提升|拉高)\s*(?:通过率|通过量|通过)",
    re.IGNORECASE,
)

_NUMBER = r"(\d+(?:\.\d+)?)\s*(%|％|个点)?"
_BAD_SUBJECT = r"(?:坏账率?|坏率|坏样本率|不良率?|逾期率|违约率|bad\s*rate)"
_BAD_CAP_RES = (
    re.compile(
        _BAD_SUBJECT
        + r"[^，。；;,\d]{0,12}?(?:不超过|不高于|不大于|低于|小于|控制在|压在|上限(?:是|为)?|<=|≤|<)\s*"
        + _NUMBER,
        re.IGNORECASE,
    ),
    re.compile(_BAD_SUBJECT + r"[^，。；;,\d]{0,8}?" + _NUMBER + r"\s*(?:以内|以下)", re.IGNORECASE),
)
_APPROVAL_SUBJECT = r"(?:通过率|审批通过率|批核率|approval\s*rate)"
_APPROVAL_FLOOR_RES = (
    re.compile(
        _APPROVAL_SUBJECT
        + r"[^，。；;,\d]{0,8}?(?:不低于|不少于|不小于|至少|最少|高于|大于|保持在|下限(?:是|为)?|>=|≥|>)\s*"
        + _NUMBER,
        re.IGNORECASE,
    ),
    re.compile(_APPROVAL_SUBJECT + r"[^，。；;,\d]{0,8}?" + _NUMBER + r"\s*(?:以上)", re.IGNORECASE),
)

_NO_BASELINE_RE = re.compile(
    r"(?:没有|无|不需要|不用|不要|没)\s*(?:已有的?|现有的?|历史的?)?\s*基线|"
    r"(?:没有|无)\s*(?:已有|现有|历史|线上)\s*策略|从零开始|从头开始|从零做|从头做|全新(?:开发|做)",
)
_BASELINE_MENTION_RE = re.compile(
    r"基线|(?:对比|比较|相比|跑赢)\s*(?:现有|当前|线上|已有)\s*策略|"
    r"(?:现有|当前|线上|已有)\s*策略\s*(?:对比|比较)",
)

_DEVELOPMENT_RE = re.compile(
    r"(?:开发|设计|制定|搭建|构建|新做|重做|做|出)\s*(?:一套|一版|一个|一份|个|套)?\s*"
    r"[^，。；;,]{0,12}?策略|策略\s*开发|开始策略开发",
)
_NON_DEVELOPMENT_RE = re.compile(
    r"回测|监控|挖掘|采纳|上线|部署|对比已有|报告|导出|应用|写回|样本设计|分箱|自动树|决策树|"
    r"投票|voting|交叉|策略池|规则池|项目上下文|额度|定价|利率|分群|分层|催收|滚动率|迁徙",
    re.IGNORECASE,
)

_QUESTION_RE = re.compile(
    r"[?？]|为什么|为何|什么是|是什么|什么意思|有什么区别|区别是|怎么选|选哪个|哪个好|是否需要|要不要"
)
_CONDITIONAL_RE = re.compile(r"如果|假如|要是|若是|万一|的话")
_CANCEL_RE = re.compile(
    r"^(?:算了|先不做了?|不做了|先不开始|暂不开始|取消(?:策略开发)?|先取消|停止|暂停|先别做了?|不用了|先放一放)"
    r"[。.!！]?$"
)
_WITHHOLD_RE = re.compile(r"先别|先不|暂不|暂时不|等一下|稍后再|晚点再|别开始|不要开始|先停")
_START_RE = re.compile(r"^(?:好的?[，,]?\s*)?(?:开始|开始吧|开始策略开发|开始开发|可以开始了?|继续|就这样|按这个来)[。.!！]?$")


def parse_development_slots(text: str | None) -> dict[str, object]:
    """Return only the slots the utterance states explicitly."""

    raw = str(text or "")
    slots: dict[str, object] = {}
    strategy_type = _strategy_type(raw)
    if strategy_type is not None:
        slots["strategy_type"] = strategy_type
    objective = _objective(raw)
    if objective is not None:
        slots["objective"] = objective
    max_bad_rate = _first_ratio(_BAD_CAP_RES, raw)
    if max_bad_rate is not None:
        slots["max_bad_rate"] = max_bad_rate
    min_approval_rate = _first_ratio(_APPROVAL_FLOOR_RES, raw)
    if min_approval_rate is not None:
        slots["min_approval_rate"] = min_approval_rate
    if _NO_BASELINE_RE.search(raw):
        slots["baseline"] = "none"
    elif _BASELINE_MENTION_RE.search(raw):
        slots["baseline"] = "mentioned"
    return slots


def merge_development_slots(*layers: Mapping[str, object] | None) -> dict[str, object]:
    """Merge slot layers; later layers override earlier ones."""

    merged: dict[str, object] = {}
    for layer in layers:
        for name, value in (layer or {}).items():
            if name in STRATEGY_DEVELOPMENT_SLOT_NAMES and value is not None:
                merged[name] = value
    return merged


def missing_development_slots(slots: Mapping[str, object]) -> list[str]:
    missing: list[str] = []
    if slots.get("strategy_type") not in {"approval", "reject"}:
        missing.append("strategy_type")
    if slots.get("objective") not in {"max_approval", "max_profit"}:
        missing.append("objective")
    if slots.get("max_bad_rate") is None and slots.get("min_approval_rate") is None:
        missing.append("constraint")
    if slots.get("baseline") == "mentioned" and not slots.get("baseline_strategy_id"):
        missing.append("baseline")
    return missing


def is_strategy_development_request(text: str | None) -> bool:
    """A positive request to develop an approval/reject strategy from data."""

    raw = str(text or "")
    if not _DEVELOPMENT_RE.search(raw) or _NON_DEVELOPMENT_RE.search(raw):
        return False
    return not utterance_is_question(raw)


def utterance_targets_other_strategy_operation(text: str | None) -> bool:
    """Backtest, monitoring, pools, typed strategies ... are not slot answers."""

    return bool(_NON_DEVELOPMENT_RE.search(str(text or "")))


def utterance_is_question(text: str | None) -> bool:
    return bool(_QUESTION_RE.search(str(text or "")))


def utterance_is_conditional(text: str | None) -> bool:
    return bool(_CONDITIONAL_RE.search(str(text or "")))


def utterance_cancels(text: str | None) -> bool:
    return bool(_CANCEL_RE.match(_compact(text)))


def utterance_withholds(text: str | None) -> bool:
    raw = _NO_BASELINE_RE.sub("", str(text or ""))
    return bool(_WITHHOLD_RE.search(raw))


def utterance_starts(text: str | None) -> bool:
    return bool(_START_RE.match(_compact(text)))


def development_slot_summary(slots: Mapping[str, object]) -> list[str]:
    """Business-Chinese echo of the known slots (no internal names)."""

    lines: list[str] = []
    strategy_type = slots.get("strategy_type")
    if strategy_type == "approval":
        lines.append("审批（准入）策略")
    elif strategy_type == "reject":
        lines.append("拒绝策略")
    objective = slots.get("objective")
    if objective == "max_approval":
        lines.append("目标：在约束内尽量提高通过率")
    elif objective == "max_profit":
        lines.append("目标：在约束内尽量提高收益")
    if slots.get("min_approval_rate") is not None:
        lines.append(f"通过率不低于 {_percent(slots['min_approval_rate'])}")
    if slots.get("max_bad_rate") is not None:
        lines.append(f"通过客群坏账率不超过 {_percent(slots['max_bad_rate'])}")
    if slots.get("baseline") == "none":
        lines.append("没有基线策略，从零开发")
    return lines


def _strategy_type(raw: str) -> str | None:
    approval = bool(_APPROVAL_RE.search(_NEGATED_APPROVAL_RE.sub("", raw)))
    reject = bool(_REJECT_RE.search(_NEGATED_REJECT_RE.sub("", raw)))
    if approval == reject:
        return None
    return "approval" if approval else "reject"


def _objective(raw: str) -> str | None:
    profit = bool(_MAX_PROFIT_RE.search(raw))
    approval = bool(_MAX_APPROVAL_RE.search(raw))
    if profit == approval:
        return None
    return "max_profit" if profit else "max_approval"


def _first_ratio(patterns: Iterable[re.Pattern[str]], raw: str) -> float | None:
    for pattern in patterns:
        match = pattern.search(raw)
        if match is None:
            continue
        value = float(match.group(1))
        unit = match.group(2)
        if unit or value > 1:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return round(value, 6)
    return None


def _percent(value: object) -> str:
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}%"


def _compact(text: str | None) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


# ---------------------------------------------------------------------------
# Sample-design proposal for development (2026-10-21, eval S8 follow-up).
#
# Development binds a governed, mature sample design.  When the active sample
# has none, the Agent proposes one read deterministically from the sample's own
# columns and asks only for what data cannot show: the performance window, the
# date the labels were observed up to, and whether every row has matured.  The
# confirmed reply becomes one complete sample-design command that must pass the
# same validation and word-by-word grounding as a typed request; nothing here
# supplies a business value the user did not state or the data does not show.
# ---------------------------------------------------------------------------

_SPLIT_ROLE_VALUES = {
    "development": {"train", "training", "dev", "develop", "development", "开发", "训练"},
    "validation": {"test", "testing", "valid", "validation", "val", "验证", "测试"},
    "oot": {"oot", "out_of_time", "时间外"},
}
_ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MONTH_VALUE_RE = re.compile(r"^\d{4}-\d{2}$")
_ENTITY_NAME_RE = re.compile(r"(?:^|_)id$|^id_|_id_|客户号|客户编号", re.I)
_OPTIONAL_FIELD_NAMES = {
    "loan_amount_field": {"loan_amount", "loan_amt"},
    "weight_field": {"weight", "sample_weight"},
    "overdue_amount_field": {"overdue_amount", "overdue_amt"},
}
_FIELD_ROLES = {
    "entity_field": "id",
    "time_field": "date",
    "month_field": "month",
    "loan_amount_field": "loan_amount",
    "weight_field": "weight",
    "overdue_amount_field": "overdue_amount",
}
_HIGHER_RISKIER = r"越高(?:越)?(?:风险(?:越)?(?:高|大)|越差|越坏)|高分高风险"
_LOWER_RISKIER = (
    r"越低(?:越)?(?:风险(?:越)?(?:高|大)|越差|越坏)|越高(?:越好|风险越低)|低分高风险"
)

_MATURED_RE = re.compile(r"(?:已经|已|确认(?:为)?|都已|均已)\s*(?:完全)?成熟")
_NOT_MATURED_RE = re.compile(
    r"(?:尚未|还没有?|没有|未|不)(?:完全)?成熟|"
    r"(?:未|不|没有?)(?:能)?确认(?:已经|已)?成熟|(?:不是|并非)(?:已经|已)?成熟"
)
_MATURITY_UNKNOWN_RE = re.compile(
    r"(?:不确定|不知道|不清楚|说不准).{0,8}(?:成熟|表现期)|"
    r"(?:成熟|表现期).{0,8}(?:不确定|不知道|不清楚)"
)
_WINDOW_DAYS_RE = re.compile(
    r"表现(?:期|窗)[^\d，,；;。]{0,6}(\d{1,4})\s*天|(\d{1,4})\s*天\s*(?:的)?表现(?:期|窗)"
)
_ANSWER_DATE_RE = re.compile(
    r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})\s*日?"
)
_KEEP_NAN_RE = re.compile(
    r"(?:不丢弃|不排除|保留).{0,12}(?:空标签|缺失标签)|"
    r"(?:空标签|缺失标签).{0,12}(?:保留|不丢弃|不排除)"
)
_DROP_NAN_RE = re.compile(
    r"(?:丢弃|排除|剔除|删除|去掉).{0,12}(?:空标签|缺失标签)|"
    r"(?:空标签|缺失标签).{0,12}(?:丢弃|排除|剔除|删除|去掉)"
)


def sample_design_profile_columns(
    column_dtypes: Mapping[str, str],
    *,
    target_col: str,
    field_roles: Mapping[str, str],
) -> list[str]:
    """Columns worth reading to propose a sample design (never every feature)."""

    wanted_names = set().union(*_OPTIONAL_FIELD_NAMES.values())
    wanted_roles = set(_FIELD_ROLES.values())
    selected = [target_col]
    for column, dtype in column_dtypes.items():
        if column == target_col:
            continue
        numeric = str(dtype).lower().startswith(("int", "uint", "float"))
        if (
            not numeric
            or field_roles.get(column) in wanted_roles
            or column in wanted_names
            or _ENTITY_NAME_RE.search(column)
        ):
            selected.append(column)
    return selected


def propose_sample_design(
    frame,
    *,
    target_col: str,
    columns: Iterable[str],
    field_roles: Mapping[str, str],
    request_text: str = "",
) -> dict[str, object]:
    """Read a sample-design proposal from the active sample's own values.

    ``frame`` holds the columns chosen by :func:`sample_design_profile_columns`.
    A role is proposed only when exactly one column fits it; anything else is
    left empty for the user, never guessed.
    """

    import pandas as pd

    target = frame[target_col]
    proposal: dict[str, object] = {
        "target": {
            "column": target_col,
            "rows": int(len(target)),
            "bad": int((target == 1).sum()),
            "good": int((target == 0).sum()),
            "missing": int(target.isna().sum()),
        }
    }
    others = [column for column in frame.columns if column != target_col]

    def complete(column: str) -> bool:
        return not bool(frame[column].isna().any())

    def texts(column: str) -> list[str]:
        return [str(value).strip() for value in frame[column].tolist()]

    def only(candidates: list[str], role: str) -> str | None:
        by_role = [column for column in others if field_roles.get(column) == role]
        pool = by_role or candidates
        return pool[0] if len(pool) == 1 else None

    splits: list[dict[str, object]] = []
    for column in others:
        if not complete(column) or pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = sorted(set(texts(column)))
        if len(values) != 3:
            continue
        mapped: dict[str, str] = {}
        for value in values:
            roles = [
                role
                for role, names in _SPLIT_ROLE_VALUES.items()
                if value.casefold() in names
            ]
            if len(roles) == 1 and roles[0] not in mapped:
                mapped[roles[0]] = value
        if len(mapped) == 3:
            counts: dict[str, int] = {}
            for value in texts(column):
                counts[value] = counts.get(value, 0) + 1
            splits.append(
                {
                    "column": column,
                    **mapped,
                    "counts": {role: counts[value] for role, value in mapped.items()},
                }
            )
    proposal["split"] = splits[0] if len(splits) == 1 else None

    days: list[str] = []
    months: list[str] = []
    for column in others:
        if not complete(column):
            continue
        series = frame[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            days.append(column)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = texts(column)
        if values and all(_ISO_DAY_RE.match(text) and _is_iso_day(text) for text in values):
            days.append(column)
        elif values and all(_is_month_value(text) for text in values):
            months.append(column)
    time_field = only(days, "date")
    proposal["time_field"] = None
    # A date role confirmed on a column whose values are not days (e.g. a
    # month column fixed 按月口径 earlier) is read by its values below.
    if time_field is not None and time_field in days:
        dates = sorted(_as_day(value) for value in frame[time_field].tolist())
        proposal["time_field"] = {
            "column": time_field,
            "start": dates[0].isoformat(),
            "end": dates[-1].isoformat(),
        }
    month_field = only(months, "month")
    proposal["month_field"] = None
    if month_field is not None:
        values = sorted(texts(month_field))
        proposal["month_field"] = {
            "column": month_field,
            "start": values[0],
            "end": values[-1],
        }
        if proposal["time_field"] is None and not days and month_field in months:
            # D-2 按月口径: whole months enter the observation window and a
            # month is mature only once its month end has a full window.
            proposal["time_field"] = {
                "column": month_field,
                "start": _month_first_day(values[0]).isoformat(),
                "end": _month_last_day(values[-1]).isoformat(),
                "granularity": "month",
                "start_month": values[0],
                "end_month": values[-1],
            }

    entities = [
        column
        for column in others
        if _ENTITY_NAME_RE.search(column)
        and complete(column)
        and bool(frame[column].is_unique)
    ]
    proposal["entity_field"] = only(entities, "id")
    for field, names in _OPTIONAL_FIELD_NAMES.items():
        proposal[field] = only(
            [column for column in others if column in names], _FIELD_ROLES[field]
        )
    proposal["historical_score"] = _stated_score_direction(
        request_text, [column for column in columns if column != target_col]
    )
    return proposal


def sample_design_blockers(proposal: Mapping[str, object]) -> list[str]:
    """Business reasons the proposal cannot become a development sample yet."""

    blockers: list[str] = []
    if proposal.get("split") is None:
        blockers.append(
            "没有找到可以直接沿用的切分列（取值为 train / test / oot 一类，"
            "分别对应开发、验证、时间外样本）"
        )
    if proposal.get("time_field") is None:
        blockers.append(
            "没有找到唯一的申请日期列（格式如 2025-10-15）或申请月份列（格式如 2025-10）"
            "；平台要逐笔核对每个样本是否已过完整表现期，需要申请时间"
        )
    return blockers


def sample_design_proposal_lines(proposal: Mapping[str, object]) -> list[str]:
    """Business-Chinese description of what was read from the sample."""

    target = proposal["target"]
    assert isinstance(target, Mapping)
    missing = int(target["missing"])
    lines = [
        f"坏样本标签：{target['column']}，取值 1 的有 {target['bad']} 行、0 的有 "
        f"{target['good']} 行，按 1 代表坏样本、0 代表好样本；"
        + (f"另有 {missing} 行空标签。" if missing else "没有空标签。")
    ]
    split = proposal.get("split")
    if isinstance(split, Mapping):
        counts = split["counts"]
        lines.append(
            f"样本切分：沿用已有切分列 {split['column']}，{split['development']} 作开发样本"
            f"（{counts['development']} 行）、{split['validation']} 作验证样本"
            f"（{counts['validation']} 行）、{split['oot']} 作时间外样本"
            f"（{counts['oot']} 行）。"
        )
    time_field = proposal.get("time_field")
    month = proposal.get("month_field")
    if _is_month_time_field(time_field):
        lines.append(
            f"申请月份：{time_field['column']}，{time_field['start_month']} 至 "
            f"{time_field['end_month']}，作为观察期。样本只有月份，按月口径处理："
            f"观察窗按整月计入（{time_field['start']} 至 {time_field['end']}），"
            "每个月按月末判断标签是否已满表现期。"
        )
    elif isinstance(time_field, Mapping):
        lines.append(
            f"申请日期：{time_field['column']}，{time_field['start']} 至 "
            f"{time_field['end']}，作为观察期。"
        )
    if isinstance(month, Mapping) and not _is_month_time_field(time_field):
        lines.append(
            f"申请月份：{month['column']}，{month['start']} 至 {month['end']}。"
        )
    fields = [
        f"{label} {proposal[field]}"
        for field, label in (
            ("entity_field", "客户编号"),
            ("loan_amount_field", "放款金额"),
            ("weight_field", "样本权重"),
            ("overdue_amount_field", "逾期金额"),
        )
        if proposal.get(field)
    ]
    score = proposal.get("historical_score")
    if isinstance(score, Mapping):
        words = (
            "越高风险越高" if score["direction"] == "higher_is_riskier" else "越低风险越高"
        )
        fields.append(f"历史评分 {score['column']}（按你说的{words}）")
    if fields:
        lines.append("其他字段：" + "；".join(fields) + "。")
    lines.append("审批人群和风险人群都用全表同一批样本，不额外筛选。")
    return lines


def parse_sample_design_answer(text: str | None) -> dict[str, object]:
    """Return only the sample-design facts the reply states explicitly."""

    raw = str(text or "")
    answer: dict[str, object] = {}
    if _MATURITY_UNKNOWN_RE.search(raw):
        answer["maturity"] = "unknown"
    elif _NOT_MATURED_RE.search(raw):
        answer["maturity"] = "not_matured"
    elif _MATURED_RE.search(raw):
        answer["maturity"] = "confirmed_matured"
    windows = {
        int(match.group(1) or match.group(2))
        for match in _WINDOW_DAYS_RE.finditer(raw)
    }
    if len(windows) == 1 and min(windows) > 0:
        answer["performance_window_days"] = min(windows)
    dates: set[str] = set()
    for match in _ANSWER_DATE_RE.finditer(raw):
        try:
            dates.add(date(*(int(part) for part in match.groups())).isoformat())
        except ValueError:
            continue
    if len(dates) == 1:
        answer["cutoff_date"] = min(dates)
    if _KEEP_NAN_RE.search(raw):
        answer["drop_nan_labels"] = False
    elif _DROP_NAN_RE.search(raw):
        answer["drop_nan_labels"] = True
    return answer


def missing_sample_design_answer(
    proposal: Mapping[str, object], answer: Mapping[str, object]
) -> list[str]:
    """Business labels of the facts still needed before the design is fixed."""

    missing: list[str] = []
    if answer.get("performance_window_days") is None:
        missing.append("表现期是多少天")
    if answer.get("cutoff_date") is None:
        missing.append("标签统计到哪一天（表现观察截止日）")
    if answer.get("maturity") != "confirmed_matured":
        missing.append("是否每笔样本都已过完整表现期（标签已成熟）")
    target = proposal.get("target")
    if (
        isinstance(target, Mapping)
        and int(target.get("missing") or 0) > 0
        and answer.get("drop_nan_labels") is None
    ):
        missing.append("空标签是保留，还是只从风险分母排除")
    return missing


def sample_design_maturity_gap(
    proposal: Mapping[str, object], answer: Mapping[str, object]
) -> str | None:
    """Platform check that the latest application has a full window by the cutoff."""

    time_field = proposal.get("time_field")
    if not isinstance(time_field, Mapping):
        return None
    latest = date.fromisoformat(str(time_field["end"]))
    window = int(answer["performance_window_days"])  # type: ignore[call-overload]
    cutoff = date.fromisoformat(str(answer["cutoff_date"]))
    matured_by = latest + timedelta(days=window)
    if matured_by <= cutoff:
        return None
    if _is_month_time_field(time_field):
        return (
            f"按月口径、表现期 {window} 天、标签统计到 {cutoff.isoformat()} 计算，"
            f"月末晚于 {(cutoff - timedelta(days=window)).isoformat()} 的申请月份还没满表现期"
            f"（最晚的申请月份 {time_field['end_month']} 按月末 {latest.isoformat()} 计，"
            f"要到 {matured_by.isoformat()} 才满），不能按“标签已成熟”处理。"
            "请核对表现期和截止日。"
        )
    return (
        f"按表现期 {window} 天、标签统计到 {cutoff.isoformat()} 计算，"
        f"{(cutoff - timedelta(days=window)).isoformat()} 之后申请的样本还没满表现期"
        f"（最晚一笔申请在 {latest.isoformat()}，要到 {matured_by.isoformat()} "
        "才满），不能按“标签已成熟”处理。请核对表现期和截止日。"
    )


def sample_design_command(
    proposal: Mapping[str, object], answer: Mapping[str, object]
) -> tuple[str, dict[str, object]]:
    """Compose the complete sample-design command and its workflow inputs.

    Every control in the inputs is stated word for word in the command, so the
    pair passes the same grounding a typed sample-design request must pass.
    """

    split = proposal["split"]
    time_field = proposal["time_field"]
    assert isinstance(split, Mapping) and isinstance(time_field, Mapping)
    days = int(answer["performance_window_days"])  # type: ignore[call-overload]
    cutoff = str(answer["cutoff_date"])
    drop = bool(answer.get("drop_nan_labels", False))
    month = proposal.get("month_field")
    bindings = {
        "entity_field": proposal.get("entity_field"),
        "time_field": time_field["column"],
        "group_field": None,
        "month_field": month["column"] if isinstance(month, Mapping) else None,
        "weight_field": proposal.get("weight_field"),
        "loan_amount_field": proposal.get("loan_amount_field"),
        "overdue_amount_field": proposal.get("overdue_amount_field"),
    }
    parts = [
        "固化策略样本设计",
        "1 代表坏样本",
        "空标签仅从风险分母排除" if drop else "不丢弃缺失标签",
        "审批总体与风险总体是同一批样本的嵌套关系",
        "审批总体无纳排条件",
        "风险总体无纳排条件",
        f"切分列 {split['column']}",
        f"开发值 {split['development']}",
        f"验证值 {split['validation']}",
        f"OOT 值 {split['oot']}",
        f"表现窗 {days} 天",
        f"观察窗 {time_field['start']} 至 {time_field['end']}",
        "成熟度已确认成熟",
        f"成熟表现窗 {days} 天",
        f"成熟度截止日 {cutoff}",
    ]
    for field, label in (
        ("entity_field", "客户字段"),
        ("time_field", "时间字段"),
        ("group_field", "分组字段"),
        ("month_field", "月份字段"),
        ("weight_field", "权重字段"),
        ("loan_amount_field", "放款金额字段"),
        ("overdue_amount_field", "逾期金额字段"),
    ):
        value = bindings[field]
        parts.append(f"{label} {value}" if value else f"{label}暂无")
    score = proposal.get("historical_score")
    if isinstance(score, Mapping):
        words = "越高越风险" if score["direction"] == "higher_is_riskier" else "越低越风险"
        parts.append(f"历史分 {score['column']}，{words}")
        historical: dict[str, object] = {
            "status": "available",
            "column": score["column"],
            "direction": score["direction"],
            "reason": None,
        }
    else:
        parts.append("历史分暂无，原因 未说明评分方向")
        historical = {
            "status": "unavailable",
            "column": None,
            "direction": None,
            "reason": "未说明评分方向",
        }
    inputs: dict[str, object] = {
        "target_bad_value": 1,
        "drop_nan_labels": drop,
        "relationship": "nested_same_cohort",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                role: {
                    "op": "eq",
                    "left": {"column": split["column"]},
                    "right": {"literal": split[role]},
                }
                for role in ("development", "validation", "oot")
            },
        },
        "maturity": {
            "status": "confirmed_matured",
            "performance_window_days": days,
            "cutoff_date": cutoff,
            "reason": None,
        },
        "performance_window": {"status": "provided", "days": days},
        "observation_window": {
            "status": "provided",
            "start": time_field["start"],
            "end": time_field["end"],
        },
        "field_bindings": bindings,
        "historical_score": historical,
    }
    return "；".join(parts) + "。", inputs


def _stated_score_direction(
    request_text: str, columns: Iterable[str]
) -> dict[str, str] | None:
    """A score column and its direction quoted from the user's own request."""

    found: set[tuple[str, str]] = set()
    for column in columns:
        name = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        for direction, words in (
            ("higher_is_riskier", _HIGHER_RISKIER),
            ("lower_is_riskier", _LOWER_RISKIER),
        ):
            if re.search(rf"{name}[^，,；;。]{{0,8}}?(?:{words})", request_text):
                found.add((column, direction))
    if len(found) != 1:
        return None
    column, direction = next(iter(found))
    return {"column": column, "direction": direction}


# D-3: score direction authority for strategy development.  The tool enum is
# ``higher_is_riskier`` / ``higher_is_better``; business words follow it.
_TOOL_SCORE_DIRECTIONS = {
    "higher_is_riskier": "higher_is_riskier",
    "lower_is_riskier": "higher_is_better",
}
_DIRECTION_WORDS = {
    "higher_is_riskier": "越高风险越高",
    "higher_is_better": "越低风险越高",
}
_DIRECTION_SOURCE_WORDS = {
    "request": "你在本次请求里说明",
    "sample_contract": "已固化的样本设计里的历史分方向",
    "workspace": "数据工作区已确认的字段方向",
}


def stated_development_score_direction(text: str | None, score_col: str) -> str | None:
    """Source 1: the direction the user's own words state for ``score_col``."""

    stated = _stated_score_direction(str(text or ""), [score_col])
    return None if stated is None else _TOOL_SCORE_DIRECTIONS[stated["direction"]]


def workspace_score_direction(field_direction: str | None) -> str | None:
    """Source 2: a confirmed V2.7 field direction, as the tool enum (else None)."""

    return _TOOL_SCORE_DIRECTIONS.get(str(field_direction or ""))


def resolve_development_score_direction(
    *,
    score_col: str,
    request_direction: str | None,
    sample_contract_direction: str | None,
    workspace_direction: str | None,
    workspace_non_monotonic: bool = False,
) -> dict[str, object]:
    """Pick the authoritative direction: request, sample contract, workspace.

    Returns ``{"status": "resolved", "direction", "source"}`` or a stop
    (``missing`` / ``contradiction``) with a business-Chinese ``message``.
    Data never decides; it is only checked afterwards
    (:func:`score_direction_conflict_message`).
    """

    if (
        request_direction is not None
        and sample_contract_direction is not None
        and request_direction != sample_contract_direction
    ):
        return {
            "status": "contradiction",
            "message": (
                f"你这次说评分列 {score_col} {_DIRECTION_WORDS[request_direction]}，"
                f"但已固化的样本设计里记录的是{_DIRECTION_WORDS[sample_contract_direction]}。"
                "两者相反，本次没有开始策略开发；请先按正确方向重新固化样本设计。"
            ),
        }
    if (
        request_direction is None
        and sample_contract_direction is not None
        and workspace_direction is not None
        and sample_contract_direction != workspace_direction
    ):
        # Two recorded user facts disagree; neither silently wins.
        return {
            "status": "contradiction",
            "message": (
                f"已固化的样本设计里记录评分列 {score_col} "
                f"{_DIRECTION_WORDS[sample_contract_direction]}，"
                f"但数据工作区确认的是{_DIRECTION_WORDS[workspace_direction]}。"
                "两者相反，本次没有开始策略开发；"
                f"请说明「{score_col} 越高风险越高」还是「{score_col} 越低风险越高」，"
                "并按正确方向更正样本设计或数据工作区。"
            ),
        }
    for source, direction in (
        ("request", request_direction),
        ("sample_contract", sample_contract_direction),
        ("workspace", workspace_direction),
    ):
        if direction is not None:
            return {"status": "resolved", "direction": direction, "source": source}
    reason = (
        f"数据工作区把评分列 {score_col} 确认为非单调字段，不能直接按分数高低切 cutoff。"
        if workspace_non_monotonic
        else f"评分列 {score_col} 的分数方向还没有确认。"
    )
    return {
        "status": "missing",
        "message": (
            reason
            + "方向错了策略会整体反过来，平台不按默认方向或样本数据推断。"
            f"请说明「{score_col} 越高风险越高」还是「{score_col} 越低风险越高」，"
            "或在数据工作区确认该字段的风险方向；本次还没有创建计划。"
        ),
    }


def score_direction_summary(score_col: str, direction: str, source: str) -> str:
    return (
        f"分数方向：{score_col} {_DIRECTION_WORDS[direction]}"
        f"（来源：{_DIRECTION_SOURCE_WORDS[source]}）。"
    )


def score_direction_conflict_message(
    score_col: str, direction: str, implied: str, *, corr: float, labeled: int
) -> str:
    """Both directions, shown at the confirmation gate before anything runs."""

    return (
        f"注意：按{score_col} {_DIRECTION_WORDS[direction]}开发，但样本数据显示 "
        f"{score_col} {_DIRECTION_WORDS[implied]}（{labeled} 笔有标签样本，"
        f"相关系数 {corr:.3f}）。两者不一致，计划已创建但不会自动开始；"
        f"确认仍按「{score_col} {_DIRECTION_WORDS[direction]}」开发请点「开始」，"
        "方向说反了请取消后重新说明分数方向。"
    )


def score_direction_quick_commands(score_col: str) -> tuple[str, str]:
    return (
        f"开始策略开发，{score_col} 越高风险越高",
        f"开始策略开发，{score_col} 越低风险越高",
    )


def _is_month_value(text: str) -> bool:
    return bool(_MONTH_VALUE_RE.match(text)) and 1 <= int(text[5:7]) <= 12


def _is_month_time_field(time_field: object) -> bool:
    return isinstance(time_field, Mapping) and time_field.get("granularity") == "month"


def _month_first_day(text: str) -> date:
    year, month = (int(part) for part in text.split("-"))
    return date(year, month, 1)


def _month_last_day(text: str) -> date:
    year, month = (int(part) for part in text.split("-"))
    return date(year, month, calendar.monthrange(year, month)[1])


def _is_iso_day(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _as_day(value: object) -> date:
    day = getattr(value, "date", None)
    if callable(day):
        return day()
    return date.fromisoformat(str(value).strip())


__all__ = [
    "STRATEGY_DEVELOPMENT_SLOT_NAMES",
    "development_slot_summary",
    "is_strategy_development_request",
    "merge_development_slots",
    "missing_development_slots",
    "missing_sample_design_answer",
    "parse_development_slots",
    "parse_sample_design_answer",
    "propose_sample_design",
    "resolve_development_score_direction",
    "sample_design_blockers",
    "sample_design_command",
    "sample_design_maturity_gap",
    "sample_design_profile_columns",
    "sample_design_proposal_lines",
    "score_direction_conflict_message",
    "score_direction_quick_commands",
    "score_direction_summary",
    "stated_development_score_direction",
    "utterance_cancels",
    "utterance_is_conditional",
    "utterance_is_question",
    "utterance_starts",
    "utterance_targets_other_strategy_operation",
    "utterance_withholds",
    "workspace_score_direction",
]
