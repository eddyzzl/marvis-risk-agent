"""S6 ad-hoc natural-language slice/aggregate turn wiring.

Ad-hoc "问数" (「按渠道看 5 月坏率」) has no workflow template: the LLM only
parses the utterance into a *structured* SliceSpec (it never computes a number —
INV-1), the platform validates every column against the dataset profile whitelist
and the op against the fixed operator set, and only after a 口径确认门 (a plain-
Chinese echo of exactly what will be grouped/measured/filtered) does the single-
step slice_aggregate plan run.

Parse failure or any hallucinated column produces a Chinese clarification question
(never a guess). The question-intent detector is deliberately conservative: a turn
that does not clearly read as a data question defaults back to the normal flow
(the caller keeps its original branch), so this never hijacks an unrelated turn.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

from marvis.llm_prompts import SLICE_SPEC_SYS

# The whitelisted aggregate operators, mirrored from
# marvis.packs.data_ops.tools._metric_expr so the platform rejects a hallucinated
# op here (before ever building a plan) rather than deep inside the tool.
_ALLOWED_OPS = frozenset(
    {"count", "sum", "mean", "min", "max", "bad_rate", "approval_rate", "distinct"}
)
_OPS_NEEDING_COL = frozenset(
    {"sum", "mean", "min", "max", "bad_rate", "approval_rate", "distinct"}
)
_ALLOWED_FILTER_OPS = frozenset({"==", "!=", ">", ">=", "<", "<=", "in", "between"})
_MAX_GROUP_BY = 3
_MAX_FILTERS = 8

# Conservative Chinese/English question-intent 启发词. A turn matches only when it
# reads as a data question; anything else defaults to the caller's original branch
# (问数意图分支防御式默认走原路).
_QUESTION_HINTS = (
    "看一下",
    "看下",
    "统计",
    "分布",
    "多少",
    "坏率",
    "通过率",
    "占比",
    "平均",
    "按",
    "各",
    "分组",
    "count",
    "average",
    "distribution",
    "how many",
)


def detect_question_intent(utterance: str | None) -> bool:
    """True when the utterance clearly reads as an ad-hoc data question. Kept
    conservative on purpose: a non-match means the caller keeps its normal flow."""
    if not utterance:
        return False
    text = str(utterance).strip().lower()
    if not text:
        return False
    return any(hint.lower() in text for hint in _QUESTION_HINTS)


@dataclass(frozen=True)
class SliceMetric:
    op: str
    col: str | None = None

    def as_dict(self) -> dict:
        payload: dict = {"op": self.op}
        if self.col is not None:
            payload["col"] = self.col
        return payload


@dataclass(frozen=True)
class SliceFilter:
    col: str
    op: str
    value: object

    def as_dict(self) -> dict:
        return {"col": self.col, "op": self.op, "value": self.value}


@dataclass(frozen=True)
class SliceSpec:
    group_by: tuple[str, ...] = ()
    metrics: tuple[SliceMetric, ...] = ()
    filters: tuple[SliceFilter, ...] = ()
    month_col: str | None = None
    months: tuple[str, ...] = ()
    sort_by: str | None = None

    def tool_inputs(self, dataset_id: str) -> dict:
        """The slice_aggregate tool inputs for a confirmed spec (single-step plan)."""
        inputs: dict = {
            "dataset_id": dataset_id,
            "metrics": [metric.as_dict() for metric in self.metrics],
        }
        if self.group_by:
            inputs["group_by"] = list(self.group_by)
        if self.filters:
            inputs["filters"] = [f.as_dict() for f in self.filters]
        if self.month_col and self.months:
            inputs["month_col"] = self.month_col
            inputs["months"] = list(self.months)
        if self.sort_by:
            inputs["sort_by"] = self.sort_by
        return inputs


@dataclass(frozen=True)
class SliceSpecResult:
    """Either a validated spec that still needs the 口径确认门 (``clarify`` is None),
    or a Chinese clarification question (``spec`` is None)."""

    spec: SliceSpec | None = None
    clarify: str | None = None
    confirmation_text: str | None = None

    @property
    def needs_clarification(self) -> bool:
        return self.spec is None


def _clarify(message: str) -> SliceSpecResult:
    return SliceSpecResult(spec=None, clarify=message, confirmation_text=None)


def build_slice_spec_from_utterance(
    utterance: str,
    dataset_profile,
    llm,
    *,
    caller: str = "adhoc_analysis",
) -> SliceSpecResult:
    """Parse an utterance into a validated SliceSpec + confirmation text, or a
    Chinese clarification. ``dataset_profile`` is any iterable of column names (the
    whitelist). ``llm`` exposes ``.complete(system_prompt=, user_prompt=, ...)`` and
    returns a JSON string; the platform (not the LLM) validates columns/ops."""
    allowed_columns = _column_whitelist(dataset_profile)
    if not allowed_columns:
        return _clarify("当前数据集没有可用列，无法解析问数请求。")

    raw = _invoke_llm(utterance, allowed_columns, llm, caller=caller)
    if raw is None:
        return _clarify("没能理解这个问题，请换一种说法，或指明要按哪列分组、看什么指标。")
    parsed = _parse_json_object(raw)
    if parsed is None:
        return _clarify("没能理解这个问题，请换一种说法，或指明要按哪列分组、看什么指标。")
    if isinstance(parsed.get("clarify"), str) and parsed["clarify"].strip():
        return _clarify(parsed["clarify"].strip())

    return validate_slice_spec(parsed, allowed_columns)


def validate_slice_spec(parsed: dict, allowed_columns) -> SliceSpecResult:
    """Platform-side validation of an LLM-produced spec against the column
    whitelist + fixed operator set. Any unknown column / bad op -> a Chinese
    clarification (never a guess, never a silent drop)."""
    whitelist = _column_whitelist(allowed_columns)

    group_by = [str(col) for col in _as_list(parsed.get("group_by"))]
    if len(group_by) > _MAX_GROUP_BY:
        return _clarify(f"最多支持按 {_MAX_GROUP_BY} 列分组，请减少分组维度。")
    for col in group_by:
        if col not in whitelist:
            return _clarify(f"没有找到列「{col}」，请从现有列里选择分组列。")

    raw_metrics = _as_list(parsed.get("metrics"))
    if not raw_metrics:
        return _clarify("没识别到要统计的指标，请说明要看数量、坏率还是均值等。")
    metrics: list[SliceMetric] = []
    for metric in raw_metrics:
        if not isinstance(metric, dict):
            return _clarify("指标格式无法识别，请重新描述要统计的指标。")
        op = str(metric.get("op") or "")
        if op not in _ALLOWED_OPS:
            return _clarify(f"暂不支持算子「{op}」，可用：数量/求和/均值/最小/最大/坏率/通过率/去重计数。")
        col = _optional_str(metric.get("col"))
        if op in _OPS_NEEDING_COL:
            if not col:
                return _clarify(f"算子「{op}」需要指定一列，请说明对哪列计算。")
            if col not in whitelist:
                return _clarify(f"没有找到列「{col}」，请从现有列里选择要统计的列。")
        metrics.append(SliceMetric(op=op, col=col if op in _OPS_NEEDING_COL else None))

    raw_filters = _as_list(parsed.get("filters"))
    if len(raw_filters) > _MAX_FILTERS:
        return _clarify(f"最多支持 {_MAX_FILTERS} 个筛选条件，请精简筛选。")
    filters: list[SliceFilter] = []
    for f in raw_filters:
        if not isinstance(f, dict):
            return _clarify("筛选条件格式无法识别，请重新描述筛选。")
        col = _optional_str(f.get("col"))
        op = str(f.get("op") or "")
        if not col or col not in whitelist:
            return _clarify(f"筛选用到的列「{col}」不存在，请从现有列里选择。")
        if op not in _ALLOWED_FILTER_OPS:
            return _clarify(f"暂不支持筛选比较符「{op}」。")
        filters.append(SliceFilter(col=col, op=op, value=f.get("value")))

    month_col = _optional_str(parsed.get("month_col"))
    months = tuple(str(month) for month in _as_list(parsed.get("months")))
    if month_col and month_col not in whitelist:
        return _clarify(f"没有找到时间列「{month_col}」，请指明按哪列筛月份。")
    if month_col and not months:
        return _clarify("指定了时间列但没有月份范围，请说明要看哪些月份。")

    sort_by = _optional_str(parsed.get("sort_by"))
    metric_labels = {_metric_label(m.op, m.col) for m in metrics}
    if sort_by and sort_by not in whitelist and sort_by not in metric_labels:
        return _clarify(f"排序依据「{sort_by}」既不是现有列也不是所选指标。")

    spec = SliceSpec(
        group_by=tuple(group_by),
        metrics=tuple(metrics),
        filters=tuple(filters),
        month_col=month_col if (month_col and months) else None,
        months=months if (month_col and months) else (),
        sort_by=sort_by,
    )
    return SliceSpecResult(
        spec=spec,
        clarify=None,
        confirmation_text=slice_spec_confirmation_text(spec),
    )


def slice_spec_confirmation_text(spec: SliceSpec) -> str:
    """The 口径确认门 copy: a plain-Chinese echo of exactly what will run, so the
    user confirms the口径 before any aggregate executes (确认门先行)."""
    group_text = "、".join(spec.group_by) if spec.group_by else "全体样本"
    metric_text = "、".join(_metric_display(m) for m in spec.metrics)
    parts = [f"将按〔{group_text}〕统计〔{metric_text}〕"]
    if spec.month_col and spec.months:
        parts.append(f"，时间范围〔{'、'.join(spec.months)}〕")
    if spec.filters:
        filter_text = "、".join(f"{f.col}{f.op}{f.value}" for f in spec.filters)
        parts.append(f"，筛选〔{filter_text}〕")
    parts.append("，确认？")
    return "".join(parts)


_METRIC_DISPLAY = {
    "count": "数量",
    "sum": "求和",
    "mean": "均值",
    "min": "最小值",
    "max": "最大值",
    "bad_rate": "坏率",
    "approval_rate": "通过率",
    "distinct": "去重计数",
}


def _metric_display(metric: SliceMetric) -> str:
    label = _METRIC_DISPLAY.get(metric.op, metric.op)
    return label if not metric.col else f"{metric.col} 的{label}"


def _metric_label(op: str, col: str | None) -> str:
    # Mirror marvis.packs.data_ops.tools._metric_label so sort_by can name a
    # metric output label consistently across the platform boundary.
    return op if op == "count" or not col else f"{op}_{col}"


def _invoke_llm(utterance: str, allowed_columns, llm, *, caller: str) -> str | None:
    user_prompt = (
        f"数据集可用列白名单：{list(allowed_columns)}\n"
        f"用户问题：{utterance}\n"
        "请输出结构化 JSON 规格（或 clarify）。"
    )
    try:
        return llm.complete(
            system_prompt=SLICE_SPEC_SYS.text,
            user_prompt=user_prompt,
            temperature=0.0,
            stream=False,
            caller=caller,
            prompt_name=SLICE_SPEC_SYS.name,
            prompt_version=SLICE_SPEC_SYS.version,
        )
    except Exception:
        return None


def _parse_json_object(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    # Tolerate a fenced ```json block or leading prose before the JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _column_whitelist(dataset_profile) -> tuple[str, ...]:
    if dataset_profile is None:
        return ()
    columns: list[str] = []
    for column in dataset_profile:
        name = str(column)
        if name and name not in columns:
            columns.append(name)
    return tuple(columns)


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


__all__ = [
    "SliceFilter",
    "SliceMetric",
    "SliceSpec",
    "SliceSpecResult",
    "build_slice_spec_from_utterance",
    "detect_question_intent",
    "slice_spec_confirmation_text",
    "validate_slice_spec",
]
