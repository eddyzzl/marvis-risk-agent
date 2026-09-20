"""Shared text grounding primitives; no workflow routing or compilation."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _utterance_contains_token(utterance: str, token: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    return re.search(pattern, utterance) is not None


def _ratio_token_value(value: str) -> float:
    normalized = re.sub(r"\s+", "", value)
    if normalized.startswith("百分之"):
        return float(normalized[len("百分之") :]) / 100.0
    if normalized.endswith("%"):
        return float(normalized[:-1]) / 100.0
    return float(normalized)


def _cross_mention_is_within(
    start: int,
    end: int,
    command_span: tuple[int, int],
) -> bool:
    return command_span[0] <= start and end <= command_span[1]


def _impact_cube_strategy_type_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    """Ignore dimension words such as ``分群列`` when selecting Pool type."""

    mentions = []
    for strategy_type, start, end in _voting_strategy_type_mentions(utterance):
        matched = utterance[start:end]
        suffix = utterance[end : end + 24]
        if re.search(r"(?:池|pool|strategy)", matched, re.IGNORECASE) or re.match(
            r"\s*(?:策略池|策略|strategy(?:\s|-|_)*pool|\bpool\b)",
            suffix,
            re.IGNORECASE,
        ):
            mentions.append((strategy_type, start, end))
    return tuple(mentions)


def _has_positive_chained_operation(
    utterance: str,
    *,
    operation_re: re.Pattern[str],
) -> bool:
    """Return true only when a clause positively requests a chained operation."""

    boundaries = "；;。.!?？\n，,、/"
    for match in operation_re.finditer(utterance):
        left = max(utterance.rfind(mark, 0, match.start()) for mark in boundaries) + 1
        prefix = utterance[left : match.start()]
        if re.search(
            r"(?:不需要|不用|暂不|先不|不要|无需|不再|不做|"
            r"别|禁止|不会|未|没有|并非|而非|不(?!只|仅))"
            r"\s*(?:再|进行|做|生成|形成|输出|进入|开展|执行|"
            r"训练|构建|建立|创建|采纳|采用|部署|投产|上线)?\s*$|"
            r"(?:(?:do\s+not\s+need\s+to|don't\s+need\s+to|"
            r"do\s+not|don't|never|without|no)\s+)"
            r"(?:(?:further\s+)?(?:do|generate|create|run)\s+)?$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


_POOL_STRATEGY_TYPE_GROUNDING = {
    "approval": re.compile(
        r"(?:审批|准入|approval(?=.{0,12}(?:策略池|pool|strategy)))",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"(?:拒绝(?:策略|规则)?池|拒绝策略|reject(?=.{0,12}(?:策略池|pool|strategy)))",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"(?:额度|授信|limit(?=.{0,12}(?:策略池|pool|strategy)))",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"(?:定价|利率|pricing(?=.{0,12}(?:策略池|pool|strategy)))",
        re.IGNORECASE,
    ),
    "segmentation": re.compile(
        r"(?:分群|分层|segment(?:ation)?(?=.{0,12}(?:策略池|pool|strategy)))",
        re.IGNORECASE,
    ),
}


def _sample_design_clauses(utterance: str) -> tuple[str, ...]:
    return tuple(
        clause.strip()
        for clause in re.split(r"[；;。.!?？\n]+", utterance)
        if clause.strip()
    )


_SCORECARD_SECOND_OPERATION_RE = re.compile(
    r"(?:加入|放入|写入|纳入)[^，,；;。\n]{0,20}(?:策略池|规则池|Pool)|"
    r"(?:入池|应用|写回|回写|采纳|采用|部署|上线|投产|生成报告|出报告)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"apply|write[-\s]*back|adopt|deploy|go[-\s]?live|"
    r"generate\s+(?:a\s+)?report)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _automatic_tree_segment(
    utterance: str,
    *,
    start: int,
    end: int,
    separators: Sequence[str],
) -> tuple[str, int, int]:
    left = max(utterance.rfind(separator, 0, start) for separator in separators) + 1
    right_candidates = [
        position
        for separator in separators
        if (position := utterance.find(separator, end)) >= 0
    ]
    right = min(right_candidates, default=len(utterance))
    return utterance[left:right], left, right


def _automatic_tree_column_mentions(
    utterance: str,
    whitelist: Sequence[str],
) -> tuple[tuple[int, int, str], ...]:
    mentions, _ = _automatic_tree_column_mention_resolution(utterance, whitelist)
    return mentions


def _automatic_tree_column_mention_resolution(
    utterance: str,
    whitelist: Sequence[str],
) -> tuple[tuple[tuple[int, int, str], ...], tuple[str, ...]]:
    """Resolve contained names and fail closed on genuinely ambiguous overlaps."""

    candidates: list[tuple[int, int, str, int, bool]] = []
    for order, column in enumerate(whitelist):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        candidates.extend(
            (
                match.start(),
                match.end(),
                column,
                order,
                match.group(0) == column,
            )
            for match in pattern.finditer(utterance)
        )

    components: list[list[tuple[int, int, str, int, bool]]] = []
    component_end = -1
    for candidate in sorted(candidates, key=lambda item: (item[0], item[1], item[3])):
        if not components or candidate[0] >= component_end:
            components.append([candidate])
            component_end = candidate[1]
            continue
        components[-1].append(candidate)
        component_end = max(component_end, candidate[1])

    accepted: list[tuple[int, int, str]] = []
    ambiguous: set[str] = set()
    for component in components:
        spans = {(start, end) for start, end, *_ in component}
        if len(spans) == 1:
            chosen_span = next(iter(spans))
        else:
            containers = [
                (start, end)
                for start, end in spans
                if all(
                    start <= other_start and other_end <= end
                    for other_start, other_end in spans
                )
            ]
            if len(containers) != 1:
                ambiguous.update(candidate[2] for candidate in component)
                continue
            chosen_span = containers[0]

        choices = [candidate for candidate in component if candidate[:2] == chosen_span]
        exact_choices = [candidate for candidate in choices if candidate[4]]
        if len(exact_choices) == 1:
            chosen = exact_choices[0]
        elif len(choices) == 1:
            chosen = choices[0]
        else:
            ambiguous.update(candidate[2] for candidate in choices)
            continue
        accepted.append((chosen[0], chosen[1], chosen[2]))

    ordered_ambiguities = tuple(column for column in whitelist if column in ambiguous)
    return (
        tuple(sorted(accepted, key=lambda item: (item[0], item[1], item[2]))),
        ordered_ambiguities,
    )


def _automatic_tree_span_is_negated(
    utterance: str,
    *,
    start: int,
    end: int,
) -> bool:
    segment, left, right = _automatic_tree_segment(
        utterance,
        start=start,
        end=end,
        separators=("，", ",", "、", "；", ";", "。", "\n"),
    )
    local_start = start - left
    local_end = end - left
    prefix = segment[max(0, local_start - 24) : local_start]
    suffix = segment[local_end : min(len(segment), local_end + 24)]
    negative_prefix = re.compile(
        r"(?:不要|无需|不用|不使用|不选|别|禁止|排除|剔除|去掉|"
        r"不是|并非|而非|不)\s*"
        r"(?:再|用|使用|选择|选|包含|加入|设置|设为|作为)?\s*$",
        re.IGNORECASE,
    )
    negative_suffix = re.compile(
        r"^\s*(?:不要|无需|不用|不使用|不选|不作为|别|禁止|排除|"
        r"剔除|去掉|不是|并非|而非)",
        re.IGNORECASE,
    )
    return (
        negative_prefix.search(prefix) is not None
        or negative_suffix.search(suffix) is not None
        or right < end
    )


def _voting_strategy_type_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (strategy_type, match.start(), match.end())
        for strategy_type, pattern in _POOL_STRATEGY_TYPE_GROUNDING.items()
        for match in pattern.finditer(utterance)
    )
