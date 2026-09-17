"""impact request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping
import math
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _POOL_IMPACT_NEGATED_RE
    from . import _POOL_IMPACT_NONCOMMAND_RE
    from . import _POOL_IMPACT_POSITIVE_INTENT_RE
    from . import _POOL_IMPACT_REPORT_ONLY_RE
    from . import _POOL_IMPACT_SECOND_OPERATION_RE
    from . import _POOL_IMPACT_STRATEGY_ID_RE
    from . import _POOL_IMPACT_TARGET_RE
    from . import _POOL_STRATEGY_TYPE_GROUNDING
    from . import _clarification
    from . import _pool_impact_span_is_negated
    from . import _pool_impact_token_is_negated
    from . import _pool_impact_tokens_are_alternatives
    from . import _utterance_contains_token
    from . import _voting_strategy_type_mentions
    from . import utterance_targets_strategy_pool_stability

_IMPACT_CUBE_EXPLICIT_TARGET_RE = re.compile(
    r"(?<![A-Za-z0-9_])impact(?:\s|-|_)*cube(?![A-Za-z0-9_])|"
    r"(?:统一|五类)[^；;。.!?？\n]{0,16}(?:策略)?(?:影响|效果|测算)|"
    r"(?:策略)?(?:影响|效果|测算)[^；;。.!?？\n]{0,16}(?:统一|五类)",
    re.IGNORECASE,
)

_IMPACT_CUBE_PARTITION_ORDER = (
    "development",
    "validation",
    "oot",
)

_IMPACT_CUBE_PARTITION_GROUNDING = {
    "development": re.compile(
        r"(?<![A-Za-z0-9_])development(?![A-Za-z0-9_])|"
        r"(?:开发|训练)(?:集|样本|分区)",
        re.IGNORECASE,
    ),
    "validation": re.compile(
        r"(?<![A-Za-z0-9_])validation(?![A-Za-z0-9_])|"
        r"(?:验证)(?:集|样本|分区)",
        re.IGNORECASE,
    ),
    "oot": re.compile(
        r"(?<![A-Za-z0-9_])oot(?![A-Za-z0-9_])|"
        r"(?:时间外|跨期|样本外)(?:集|样本|分区)?",
        re.IGNORECASE,
    ),
}

_IMPACT_CUBE_ALL_PARTITIONS_RE = re.compile(
    r"(?:全部|所有|完整|全量)(?:三个|三类|3个|3类)?(?:样本)?分区|"
    r"(?<![A-Za-z0-9_])all\s+(?:three\s+)?partitions(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_IMPACT_CUBE_ECONOMICS_GROUNDING = {
    "ead": re.compile(
        r"(?<![A-Za-z0-9_])ead(?![A-Za-z0-9_])|(?:风险暴露|敞口|放款金额)",
        re.IGNORECASE,
    ),
    "pd": re.compile(
        r"(?<![A-Za-z0-9_])pd(?![A-Za-z0-9_])|(?:违约概率|坏账概率)",
        re.IGNORECASE,
    ),
    "annual_rate": re.compile(
        r"(?<![A-Za-z0-9_])annual_rate(?![A-Za-z0-9_])|年利率",
        re.IGNORECASE,
    ),
    "funding_rate": re.compile(
        r"(?<![A-Za-z0-9_])funding_rate(?![A-Za-z0-9_])|资金成本率",
        re.IGNORECASE,
    ),
    "lgd": re.compile(
        r"(?<![A-Za-z0-9_])lgd(?![A-Za-z0-9_])|(?:违约损失率|损失率)",
        re.IGNORECASE,
    ),
    "operating_cost_per_loan": re.compile(
        r"(?<![A-Za-z0-9_])operating_cost_per_loan(?![A-Za-z0-9_])|"
        r"(?:单笔)?运营成本",
        re.IGNORECASE,
    ),
    "term_months": re.compile(
        r"(?<![A-Za-z0-9_])term_months(?![A-Za-z0-9_])|期限月数",
        re.IGNORECASE,
    ),
    "utilization": re.compile(
        r"(?<![A-Za-z0-9_])utilization(?![A-Za-z0-9_])|(?:额度)?使用率",
        re.IGNORECASE,
    ),
}

def _impact_cube_strategy_type_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    """Ignore dimension words such as ``分群列`` when selecting Pool type."""

    mentions = []
    for strategy_type, start, end in _voting_strategy_type_mentions(
        utterance
    ):
        matched = utterance[start:end]
        suffix = utterance[end : end + 24]
        if (
            re.search(r"(?:池|pool|strategy)", matched, re.IGNORECASE)
            or re.match(
                r"\s*(?:策略池|策略|strategy(?:\s|-|_)*pool|\bpool\b)",
                suffix,
                re.IGNORECASE,
            )
        ):
            mentions.append((strategy_type, start, end))
    return tuple(mentions)

def utterance_targets_strategy_impact_cube(utterance: str) -> bool:
    """Reserve only a positive, executable unified/non-binary impact clause."""

    if utterance_targets_strategy_pool_stability(utterance):
        return False
    if _POOL_IMPACT_TARGET_RE.search(utterance) is None:
        return False
    if (
        _POOL_IMPACT_REPORT_ONLY_RE.search(utterance) is not None
        or _POOL_IMPACT_POSITIVE_INTENT_RE.search(utterance) is None
    ):
        return False
    explicit = tuple(_IMPACT_CUBE_EXPLICIT_TARGET_RE.finditer(utterance))
    if any(
        not _pool_impact_span_is_negated(utterance, start=match.start())
        for match in explicit
    ):
        return True
    mentioned_types = {
        strategy_type
        for strategy_type, start, _end in _impact_cube_strategy_type_mentions(
            utterance
        )
        if not _pool_impact_span_is_negated(utterance, start=start)
    }
    return bool(mentioned_types & {"limit", "pricing", "segmentation"})

def _ground_strategy_impact_cube_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Prove every unified ImpactCube control came from this utterance."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _POOL_IMPACT_NEGATED_RE.search(utterance)
        or _POOL_IMPACT_NONCOMMAND_RE.search(utterance)
        or _POOL_IMPACT_REPORT_ONLY_RE.search(utterance)
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求统一 Strategy ImpactCube；"
            "否定、问句、历史/未来描述或仅报告不会执行测算。",
            code="strategy_impact_cube_positive_command_required",
            fields=("measurement_intent",),
        )
    if _POOL_IMPACT_POSITIVE_INTENT_RE.search(utterance) is None:
        return _clarification(
            "原话没有明确授权执行统一 Strategy ImpactCube。请明确说出要测算的"
            " Pool 类型；本 Workflow 只生成可逆的只读证据。",
            code="strategy_impact_cube_positive_command_required",
            fields=("measurement_intent",),
        )
    if _POOL_IMPACT_SECOND_OPERATION_RE.search(utterance):
        return _clarification(
            "统一 Strategy ImpactCube 必须是当前轮唯一操作；Pool 修改、"
            "创建策略、写回、报告、采纳、晋级或部署必须拆成后续请求。",
            code="strategy_impact_cube_single_operation_required",
            fields=("workflow",),
        )

    missing_controls: list[str] = []
    strategy_type = str(inputs.get("strategy_type") or "")
    strategy_type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    strategy_type_mentions = _impact_cube_strategy_type_mentions(utterance)
    mentioned_strategy_types = {item[0] for item in strategy_type_mentions}
    selected_type_is_negated = any(
        item[0] == strategy_type
        and _pool_impact_span_is_negated(utterance, start=item[1])
        for item in strategy_type_mentions
    )
    if (
        strategy_type_pattern is None
        or strategy_type_pattern.search(utterance) is None
        or mentioned_strategy_types != {strategy_type}
        or selected_type_is_negated
    ):
        missing_controls.append(f"strategy_type {strategy_type or 'unknown'}")

    mentioned_partitions = _impact_cube_partition_mentions(utterance)
    negated_partitions = _impact_cube_negated_partition_mentions(utterance)
    requested_partitions = inputs.get("partitions")
    if requested_partitions is not None:
        if (
            set(requested_partitions) != mentioned_partitions
            or set(requested_partitions) & negated_partitions
        ):
            missing_controls.append("partitions")
    elif mentioned_partitions or negated_partitions:
        missing_controls.append("partitions")

    explicit_columns = _impact_cube_explicit_column_bindings(
        utterance,
        whitelist,
    )
    mentioned_columns = tuple(
        column
        for column in whitelist
        if _utterance_contains_token(utterance, column)
    )
    for field in ("month_col", "group_col", "segment_col"):
        selected = inputs.get(field)
        values = explicit_columns.get(field, set())
        if selected is None:
            if values:
                missing_controls.append(
                    f"{field} {'/'.join(sorted(values))}"
                )
            continue
        if (
            values != {selected}
            or not _utterance_contains_token(utterance, str(selected))
            or _pool_impact_token_is_negated(utterance, str(selected))
            or any(
                other != selected
                and _pool_impact_tokens_are_alternatives(
                    utterance,
                    str(selected),
                    other,
                )
                for other in mentioned_columns
            )
        ):
            missing_controls.append(f"{field} {selected}")

    current_strategy_id = inputs.get("current_strategy_id")
    strategy_id_matches = tuple(
        _POOL_IMPACT_STRATEGY_ID_RE.finditer(utterance)
    )
    positive_strategy_ids = {
        match.group(0).casefold()
        for match in strategy_id_matches
        if not _pool_impact_span_is_negated(
            utterance,
            start=match.start(),
        )
    }
    if current_strategy_id is not None:
        selected_id = str(current_strategy_id).casefold()
        if (
            positive_strategy_ids != {selected_id}
            or not _utterance_contains_token(
                utterance,
                str(current_strategy_id),
            )
        ):
            missing_controls.append(str(current_strategy_id))
    elif positive_strategy_ids:
        missing_controls.append("current_strategy_id")

    raw_economics = inputs.get("economics_inputs")
    mentioned_components = _impact_cube_explicit_economics_components(
        utterance
    )
    if isinstance(raw_economics, Mapping):
        for component, binding in raw_economics.items():
            pattern = _IMPACT_CUBE_ECONOMICS_GROUNDING.get(component)
            grounded = pattern is not None and pattern.search(utterance) is not None
            if grounded and binding["kind"] == "column":
                grounded = _impact_cube_economics_value_is_grounded(
                    utterance,
                    component=component,
                    value=str(binding["column"]),
                    is_column=True,
                )
            elif grounded:
                grounded = _impact_cube_economics_value_is_grounded(
                    utterance,
                    component=component,
                    value=binding["value"],
                    is_column=False,
                )
            if not grounded:
                missing_controls.append(f"economics_inputs.{component}")
        if mentioned_components != set(raw_economics):
            missing_controls.append("economics_inputs")
    elif mentioned_components:
        missing_controls.append("economics_inputs")

    if missing_controls:
        rendered = "、".join(dict.fromkeys(missing_controls))
        return _clarification(
            "统一 Strategy ImpactCube 只能使用用户原话中的 Pool 类型、分区、"
            "精确维度列、当前策略 ID 和 typed economics_inputs；当前无法核对："
            f"{rendered}。平台不会采用 LLM 猜测的引用、列、数字或指标。",
            code="strategy_impact_cube_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing_controls)),
        )
    return result

def _impact_cube_partition_mentions(utterance: str) -> set[str]:
    negated = _impact_cube_negated_partition_mentions(utterance)
    if any(
        not _pool_impact_span_is_negated(
            utterance,
            start=match.start(),
        )
        for match in _IMPACT_CUBE_ALL_PARTITIONS_RE.finditer(utterance)
    ):
        return set(_IMPACT_CUBE_PARTITION_ORDER) - negated
    return {
        partition
        for partition, pattern in _IMPACT_CUBE_PARTITION_GROUNDING.items()
        if any(
            not _pool_impact_span_is_negated(
                utterance,
                start=match.start(),
            )
            for match in pattern.finditer(utterance)
        )
    }

def _impact_cube_negated_partition_mentions(utterance: str) -> set[str]:
    negated = {
        partition
        for partition, pattern in _IMPACT_CUBE_PARTITION_GROUNDING.items()
        if any(
            _pool_impact_span_is_negated(
                utterance,
                start=match.start(),
            )
            for match in pattern.finditer(utterance)
        )
    }
    if any(
        _pool_impact_span_is_negated(
            utterance,
            start=match.start(),
        )
        for match in _IMPACT_CUBE_ALL_PARTITIONS_RE.finditer(utterance)
    ):
        negated.update(_IMPACT_CUBE_PARTITION_ORDER)
    return negated

def _impact_cube_explicit_column_bindings(
    utterance: str,
    whitelist: tuple[str, ...],
) -> dict[str, set[str]]:
    labels = {
        "month_col": (
            r"(?:月份|月度|申请月|观察月)(?:字段|列)|"
            r"(?<![A-Za-z0-9_])month(?:_col|\s+column)(?![A-Za-z0-9_])"
        ),
        "group_col": (
            r"(?:分组|组别|渠道)(?:字段|列)|"
            r"(?<![A-Za-z0-9_])group(?:_col|\s+column)(?![A-Za-z0-9_])"
        ),
        "segment_col": (
            r"(?:分群|客群|分层)(?:字段|列)|"
            r"(?<![A-Za-z0-9_])segment(?:_col|\s+column)(?![A-Za-z0-9_])"
        ),
    }
    bindings = {field: set() for field in labels}
    for column in sorted(whitelist, key=len, reverse=True):
        token = (
            rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
        )
        for field, label in labels.items():
            before = re.compile(
                rf"(?:{label})\s*(?:(?:为|是|用|使用|选择|指定)\s*)?"
                rf"(?:=|:|：)?\s*{token}",
                re.IGNORECASE,
            )
            after = re.compile(
                rf"{token}\s*(?:(?:作为|用作|是|为)\s*)?(?:{label})",
                re.IGNORECASE,
            )
            if before.search(utterance) or after.search(utterance):
                bindings[field].add(column)
    return {field: values for field, values in bindings.items() if values}

def _impact_cube_economics_value_is_grounded(
    utterance: str,
    *,
    component: str,
    value: str | int | float,
    is_column: bool,
) -> bool:
    """Require each economics value to be locally bound to its own component."""

    component_pattern = _IMPACT_CUBE_ECONOMICS_GROUNDING.get(component)
    if component_pattern is None:
        return False
    if is_column:
        token = (
            rf"(?<![A-Za-z0-9_]){re.escape(str(value))}"
            rf"(?![A-Za-z0-9_])"
        )
        marker = r"(?:列|字段|column)"
        connector = r"(?:绑定|使用|采用|选择|指定|为|是|=|:|：)?"
        forward = re.compile(
            rf"(?:{component_pattern.pattern})\s*{marker}\s*"
            rf"{connector}\s*{token}",
            re.IGNORECASE,
        )
        backward = re.compile(
            rf"{token}\s*(?:作为|用作|绑定为|是|为)?\s*"
            rf"(?:{component_pattern.pattern})\s*{marker}",
            re.IGNORECASE,
        )
        return any(
            not _pool_impact_span_is_negated(
                utterance,
                start=match.start(),
            )
            for pattern in (forward, backward)
            for match in pattern.finditer(utterance)
        )

    value_patterns: list[re.Pattern[str]] = []
    candidates = {str(value)}
    if isinstance(value, float):
        candidates.add(format(value, ".15g"))
        percent = value * 100.0
        if math.isfinite(percent):
            candidates.add(format(percent, ".15g") + "%")
    value_patterns.extend(
        re.compile(
            rf"(?<![A-Za-z0-9_.]){re.escape(candidate)}"
            rf"(?![A-Za-z0-9_.])",
            re.IGNORECASE,
        )
        for candidate in sorted(candidates, key=len, reverse=True)
    )

    value_spans = {
        (match.start(), match.end())
        for pattern in value_patterns
        for match in pattern.finditer(utterance)
        if not _pool_impact_span_is_negated(
            utterance,
            start=match.start(),
        )
    }
    if not value_spans:
        return False
    component_spans = [
        (name, match.start(), match.end())
        for name, pattern in _IMPACT_CUBE_ECONOMICS_GROUNDING.items()
        for match in pattern.finditer(utterance)
        if not _pool_impact_span_is_negated(
            utterance,
            start=match.start(),
        )
    ]
    separators = re.compile(r"[、，,；;。.!?？\n]")
    for value_start, value_end in value_spans:
        nearby: list[tuple[int, str]] = []
        for name, start, end in component_spans:
            between = (
                utterance[end:value_start]
                if end <= value_start
                else utterance[value_end:start]
                if value_end <= start
                else ""
            )
            if separators.search(between):
                continue
            distance = (
                value_start - end
                if end <= value_start
                else start - value_end
                if value_end <= start
                else 0
            )
            if distance <= 32:
                nearby.append((distance, name))
        if not nearby:
            continue
        nearest_distance = min(distance for distance, _name in nearby)
        nearest = {
            name for distance, name in nearby if distance == nearest_distance
        }
        if nearest == {component}:
            return True
    return False

def _impact_cube_explicit_economics_components(
    utterance: str,
) -> set[str]:
    """Find only typed economics controls, not columns that share a name.

    A dataset can legitimately contain a column named ``pd`` and use it as a
    grouping dimension.  The bare token therefore cannot prove that the user
    requested an economics binding.  Requiring either a column marker or an
    explicit scalar assignment keeps omitted-control detection precise.
    """

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?"
    result: set[str] = set()
    for component, pattern in _IMPACT_CUBE_ECONOMICS_GROUNDING.items():
        column_binding = re.compile(
            rf"(?:{pattern.pattern})\s*(?:列|字段|column)",
            re.IGNORECASE,
        )
        scalar_binding = re.compile(
            rf"(?:{pattern.pattern})\s*(?:=|:|：|为|是|is)\s*{number}",
            re.IGNORECASE,
        )
        if any(
            not _pool_impact_span_is_negated(
                utterance,
                start=match.start(),
            )
            for binding in (column_binding, scalar_binding)
            for match in binding.finditer(utterance)
        ):
            result.add(component)
    return result
