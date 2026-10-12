"""Confirmed workspace field labels; physical schema remains authoritative."""

from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata

from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_instructions import column_identifiers


class FieldAliasError(ValueError):
    pass


def field_alias_bindings(columns, business_names: Mapping) -> tuple[tuple[str, str], ...]:
    """Only confirmed names of existing fields may become schema metadata.

    Unsafe labels stay local. They cannot certify prose, paths or record values.
    Duplicate labels are preserved here so a referenced ambiguity fails closed.
    """
    if not isinstance(business_names, Mapping) or len(business_names) > 10000:
        raise FieldAliasError("字段别名目录无效，请重新确认数据语义。")
    schema = set(columns)
    bindings = []
    for physical, label in business_names.items():
        if physical not in schema or not isinstance(label, str):
            raise FieldAliasError("字段别名不属于当前数据集，请重新确认数据语义。")
        try:
            column_identifiers([physical, label])
        except OutboundContextError:
            continue
        if label != physical:
            bindings.append((physical, label))
    return tuple(sorted(bindings))


def alias_grounding(utterance: str, columns, bindings) -> tuple[str, list[dict]]:
    """Admit metadata and ambiguities; preserve every control character verbatim."""
    physical_columns = set(columns)
    normalized_physical = {}
    for physical in physical_columns:
        normalized_physical.setdefault(unicodedata.normalize("NFKC", physical).casefold(), set()).add(physical)
    labels: dict[str, set[str]] = {}
    exported = []
    for physical, label in bindings:
        if physical not in physical_columns:
            raise FieldAliasError("字段别名绑定已失效，请重新选择数据集。")
        column_identifiers([physical, label])
        normalized = unicodedata.normalize("NFKC", label).casefold()
        labels.setdefault(normalized, set()).add(physical)
        exported.append({"physical": physical, "business_name": label})
    if not labels:
        return utterance, []
    normalized_instruction = unicodedata.normalize("NFKC", utterance)
    for label, physical in labels.items():
        choices = physical | normalized_physical.get(label, set())
        referenced = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])"
            if label.isascii() else re.escape(label), normalized_instruction, re.I,
        )
        if referenced and len(choices) != 1:
            raise FieldAliasError("指令引用的业务字段名对应多个物理字段，请明确使用物理字段名或修正别名。")

    return utterance, exported


class GroundingField(str):
    """A canonical field carrying local labels; never control-text substitution."""

    def __new__(cls, physical, labels):
        value = super().__new__(cls, physical)
        value.business_aliases = tuple(labels)
        return value


def grounding_fields(columns, bindings):
    by_physical = {}
    for physical, label in bindings:
        by_physical.setdefault(physical, []).append(label)
    return tuple(GroundingField(column, by_physical.get(column, ())) for column in columns)


def explicit_alias_span(utterance, start, end):
    """Only a syntactic field position or a quoted label grants a field span."""
    before, after = utterance[max(0, start - 128):start], utterance[end:end + 64]
    if before and after and (before[-1], after[0]) in {('"', '"'), ("'", "'"), ('“', '”'), ('「', '」'), ('`', '`')}:
        return True
    prefix = r"(?:字段|特征|列|(?:客户|账户|借据)\s*(?:ID|id|编号)|时间|日期|月份|月龄|MOB|状态|余额权重|金额权重|[a-z][a-z0-9_]*(?:col|column|field|feature)|features|columns)\s*(?:为|是|=|:|：)?\s*$"
    suffix = r"^\s*(?:(?:作为|用作|是|为)\s*(?:字段|列|特征|时间|日期|月份|状态|余额(?:权重)?|金额权重|客户\s*ID)|(?:>=|<=|>|<|=|大于|小于|不超过|至少)\s*-?\d)"
    return bool(re.search(prefix, before, re.I) or re.search(suffix, after, re.I))
