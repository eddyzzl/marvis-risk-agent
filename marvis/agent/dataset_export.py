"""Natural-language compiler for governed, task-owned dataset exports.

This module only recognizes explicit exports of the current dataset/sample to
CSV or Excel.  Reports, strategies and rules are intentionally outside this
route so an Agent cannot accidentally substitute a raw-data export for a
domain report workflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


_DATASET_SCOPE = re.compile(
    r"(?:当前|这份|本次)(?:的)?(?:数据集|数据|样本)"
    r"|(?:current|this)\s+(?:dataset|data|sample)\b",
    re.IGNORECASE,
)
_EXPORT_ACTION = re.compile(r"(?:导出|下载|\bexport\b|\bdownload\b)", re.IGNORECASE)
_CSV_FORMAT = re.compile(r"(?<![A-Za-z0-9_])csv(?![A-Za-z0-9_])", re.IGNORECASE)
_XLSX_FORMAT = re.compile(
    r"(?<![A-Za-z0-9_])(?:xlsx|excel)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_NON_DATA_OBJECT = re.compile(
    r"(?:策略|规则|报告|文档|模板|分析结果|缺失(?:值)?分析|统计(?:结果|分析|概览)?|"
    r"数据概览|样本概览|数据字典|字段字典|"
    r"strategy|rules?|reports?|documents?|templates?|analysis\s+results?|"
    r"missing(?:ness)?\s+analysis|statistics?|data\s+(?:summary|profile|dictionary))",
    re.IGNORECASE,
)
_TEXT_MODE = re.compile(
    r"(?:这些)?(?:列|字段|变量)?\s*(?:按|以)(?:纯)?文本(?:格式)?(?:导出|写入)?"
    r"|(?:treat|write|export)?\s*(?:these\s+)?(?:columns?|fields?)?\s*as\s+text\b",
    re.IGNORECASE,
)
_FIELD_SPLIT = re.compile(r"\s*(?:、|，|,|和|及|与|\band\b|\+)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetExportRequest:
    format: str
    text_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetExportRequestResult:
    request: DatasetExportRequest | None = None
    clarification: str | None = None


class _NeedClarification(ValueError):
    pass


class _ColumnResolver:
    def __init__(
        self,
        columns: Sequence[str],
        business_names: Mapping[str, str],
    ) -> None:
        ordered = tuple(dict.fromkeys(str(column).strip() for column in columns))
        if not ordered or any(not column for column in ordered):
            raise _NeedClarification("当前数据集没有可导出的有效字段。")
        aliases: dict[str, list[str]] = {}
        for column in ordered:
            aliases.setdefault(column.casefold(), []).append(column)
            business_name = str(business_names.get(column) or "").strip()
            if business_name:
                aliases.setdefault(business_name.casefold(), []).append(column)
        self._aliases = aliases

    def resolve(self, value: str) -> str:
        cleaned = _clean_field(value)
        matches = tuple(dict.fromkeys(self._aliases.get(cleaned.casefold(), ())))
        if not matches:
            raise _NeedClarification(
                f"当前数据集中没有字段「{cleaned or value.strip()}」，"
                "请使用现有字段名或业务名称。"
            )
        if len(matches) != 1:
            raise _NeedClarification(
                f"字段名称「{cleaned}」对应多个字段，请改用原始字段名。"
            )
        return matches[0]


def detect_dataset_export_intent(utterance: str | None) -> bool:
    """Detect explicit current-dataset exports, including format questions.

    A missing or conflicting supported format is still an export intent: the
    turn handler owns the clarification instead of letting the request fall
    through to a generic chat path.
    """

    text = str(utterance or "").strip()
    if not text:
        return False
    if not (_DATASET_SCOPE.search(text) and _EXPORT_ACTION.search(text)):
        return False
    return not _export_object_is_non_data(text)


def build_dataset_export_request(
    utterance: str,
    *,
    columns: Sequence[str],
    business_names: Mapping[str, str],
) -> DatasetExportRequestResult:
    """Compile an export utterance into the closed Tool input subset."""

    text = str(utterance or "").strip()
    try:
        if not text:
            raise _NeedClarification("请说明要导出当前数据，并选择 CSV 或 Excel 格式。")
        if not (_DATASET_SCOPE.search(text) and _EXPORT_ACTION.search(text)):
            if _NON_DATA_OBJECT.search(text):
                raise _NeedClarification(
                    "这里仅导出当前数据集（原始明细）；分析结果、数据字典、策略、规则或报告"
                    "请使用对应的分析或报告导出功能。"
                )
            raise _NeedClarification("请明确说明要导出当前数据集或当前样本。")
        if _export_object_is_non_data(text):
            raise _NeedClarification(
                "这里仅导出当前数据集（原始明细）；分析结果、数据字典、策略、规则或报告"
                "请使用对应的分析或报告导出功能。"
            )

        has_csv = bool(_CSV_FORMAT.search(text))
        has_xlsx = bool(_XLSX_FORMAT.search(text))
        if has_csv and has_xlsx:
            raise _NeedClarification("一次请选择 CSV 或 Excel 其中一种导出格式。")
        if not has_csv and not has_xlsx:
            raise _NeedClarification("请选择 CSV 或 Excel 导出格式。")

        resolver = _ColumnResolver(columns, business_names)
        text_columns = _parse_text_columns(text, resolver)
        return DatasetExportRequestResult(
            request=DatasetExportRequest(
                format="csv" if has_csv else "xlsx",
                text_columns=text_columns,
            )
        )
    except _NeedClarification as exc:
        return DatasetExportRequestResult(clarification=str(exc))


def _parse_text_columns(text: str, resolver: _ColumnResolver) -> tuple[str, ...]:
    marker = _TEXT_MODE.search(text)
    if marker is None:
        return ()

    prefix = text[: marker.start()].strip()
    # Text-mode fields are expected in the final comma/semicolon-delimited
    # clause immediately before "按文本" / "as text".
    segment = re.split(r"[，,；;。.]", prefix)[-1].strip()
    if segment == prefix:
        formats = [
            match
            for pattern in (_CSV_FORMAT, _XLSX_FORMAT)
            for match in pattern.finditer(prefix)
        ]
        if formats:
            segment = prefix[max(formats, key=lambda item: item.end()).end() :]
    segment = segment.strip(" ，,；;：:")
    segment = re.sub(
        r"^(?:把|将|并把|并将|and\s+|export\s+|write\s+|treat\s+)",
        "",
        segment,
        flags=re.I,
    )
    segment = re.sub(
        r"^(?:当前|这份|本次)(?:的)?(?:数据集|数据|样本)"
        r"(?:中(?:的)?|里(?:的)?|内(?:的)?)?\s*",
        "",
        segment,
    )
    segment = re.sub(r"(?:这些)?(?:列|字段|变量)\s*$", "", segment).strip()
    if not segment:
        raise _NeedClarification("请明确哪些字段需要按文本导出。")

    tokens = [item for item in _FIELD_SPLIT.split(segment) if item.strip()]
    if not tokens:
        raise _NeedClarification("请明确哪些字段需要按文本导出。")
    resolved = tuple(dict.fromkeys(resolver.resolve(item) for item in tokens))
    return resolved


def _export_object_is_non_data(text: str) -> bool:
    """Reject a nearer report/rule object even when data is mentioned as context."""

    action = _EXPORT_ACTION.search(text)
    if action is None:
        return False
    formats = [
        match
        for pattern in (_CSV_FORMAT, _XLSX_FORMAT)
        for match in pattern.finditer(text, action.end())
    ]
    if formats:
        nearest_format = min(formats, key=lambda item: item.start())
        if _NON_DATA_OBJECT.search(text[action.end() : nearest_format.start()]):
            return True
    else:
        # With no format token, inspect the object after the action so
        # "导出当前数据的分析结果" remains owned by the analysis/report routes.
        # Purpose clauses do not redefine the exported object, e.g.
        # "导出当前数据用于后续报告" is still a raw-dataset export.
        suffix = text[action.end() :]
        object_clause = re.split(
            r"(?:用于|用来|以便|以用于|\b(?:for|to)\s+(?:a\s+)?(?:later|subsequent)\b)",
            suffix,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        if _NON_DATA_OBJECT.search(object_clause):
            return True

    clause_start = max(
        text.rfind("，", 0, action.start()),
        text.rfind(",", 0, action.start()),
        text.rfind("；", 0, action.start()),
        text.rfind(";", 0, action.start()),
        text.rfind("。", 0, action.start()),
    )
    prefix = text[clause_start + 1 : action.start()]
    dataset_matches = list(_DATASET_SCOPE.finditer(prefix))
    non_data_matches = list(_NON_DATA_OBJECT.finditer(prefix))
    return bool(
        non_data_matches
        and (
            not dataset_matches
            or non_data_matches[-1].start() > dataset_matches[-1].start()
        )
    )


def _clean_field(value: str) -> str:
    cleaned = str(value).strip().strip("`'\"")
    return re.sub(r"\s*(?:这些)?(?:列|字段|变量)\s*$", "", cleaned).strip()


__all__ = [
    "DatasetExportRequest",
    "DatasetExportRequestResult",
    "build_dataset_export_request",
    "detect_dataset_export_intent",
]
