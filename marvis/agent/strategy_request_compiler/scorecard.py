"""scorecard request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
import json
import math
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _CANDIDATE_STABILITY_ACTION_RE
    from . import _CANDIDATE_STABILITY_ASSET_ID_TOKEN_RE
    from . import _CANDIDATE_STABILITY_MEASUREMENT_RE
    from . import _CANDIDATE_STABILITY_NOT_AUTHORIZED_RE
    from . import _CANDIDATE_STABILITY_PLATFORM_CONTROL_RE
    from . import _CANDIDATE_STABILITY_POOL_ENTRY_ID_TOKEN_RE
    from . import _CANDIDATE_STABILITY_SECOND_OPERATION_RE
    from . import _CANDIDATE_STABILITY_SUBJECT_RE
    from . import _POOL_STRATEGY_TYPE_GROUNDING
    from . import _automatic_tree_span_is_negated
    from . import _clarification
    from . import _cross_mention_is_within
    from . import _voting_strategy_type_mentions

_SCORECARD_SUBJECT_RE = re.compile(
    r"(?:Scorecard|评分卡).{0,20}"
    r"(?:分数带|评分带|分档|档位|分带|分成\s*\d+\s*档|cutoff|通过线)|"
    r"(?:分数带|评分带|分档|档位|分带|cutoff|通过线)"
    r".{0,20}(?:Scorecard|评分卡)",
    re.IGNORECASE,
)

_SCORECARD_BUILD_ACTION_RE = re.compile(
    r"(?:构建|生成|创建|计算|设计|物化|分成|划分(?:为|成)?)|"
    r"(?<![A-Za-z0-9_])(?:build|create|generate|compute|design|materialize)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SCORECARD_SELECTION_ACTION_RE = re.compile(
    r"(?:选择|选取|物化)|"
    r"(?<![A-Za-z0-9_])(?:select|choose|materialize)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SCORECARD_NOT_AUTHORIZED_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|先不|暂不|取消|撤销|禁止|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|假设|假如|如果|"
    r"以后|未来|将来|稍后|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|how\s+to|what\s+if|later|previously|"
    r"in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SCORECARD_HEURISTIC_SELECTION_RE = re.compile(
    r"(?:最好|最优|最佳|最差|风险最高|坏率最高|自动(?:选择|挑选|推荐)|"
    r"按(?:坏率|通过率|KS|AUC|Lift|收益|利润).{0,16}(?:选择|推荐))|"
    r"(?<![A-Za-z0-9_])(?:best|worst|top[- ]?\d*|highest[- ]risk|"
    r"automatically\s+(?:select|choose|recommend)|recommend)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SCORECARD_SECOND_OPERATION_RE = re.compile(
    r"(?:加入|放入|写入|纳入)[^，,；;。\n]{0,20}(?:策略池|规则池|Pool)|"
    r"(?:入池|应用|写回|回写|采纳|采用|部署|上线|投产|生成报告|出报告)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"apply|write[-\s]*back|adopt|deploy|go[-\s]?live|"
    r"generate\s+(?:a\s+)?report)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_SCORECARD_BIN_COUNT_RE = re.compile(
    r"(?:等频\s*)?(?P<count>\d+)\s*(?:档|带|bands?)",
    re.IGNORECASE,
)

_SCORECARD_RAW_PD_EDGES_RE = re.compile(
    r"(?:raw\s*pd|原始\s*PD|原始坏账概率)"
    r"(?:\s*(?:分带)?(?:边界|切点|edges?))?\s*(?:为|是|=|:|：)?\s*"
    r"[\[【](?P<body>[^\]】]{1,500})[\]】]",
    re.IGNORECASE,
)

_SCORECARD_SELECTION_REASON_RE = re.compile(
    r"(?:选择理由|理由|原因|说明|reason)\s*(?:为|是|=|:|：)\s*"
    r"(?P<reason>[^；;。.!?？\n]{1,500})",
    re.IGNORECASE,
)

_SCORECARD_BAND_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-band-asset-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_SCORECARD_CUTOFF_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-cutoff-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-cutoff-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

def _explicit_manual_breakpoint_bindings(
    utterance: str,
    *,
    whitelist: Sequence[str],
    command_span: tuple[int, int] | None = None,
) -> tuple[dict[str, list[float]], bool]:
    """Parse only explicit ``feature manual 切点 [..]`` controls."""

    bindings: dict[str, list[float]] = {}
    ambiguous = False
    for column in sorted(whitelist, key=len, reverse=True):
        token = (
            rf"(?<![A-Za-z0-9_]){re.escape(column)}"
            rf"(?![A-Za-z0-9_])"
        )
        pattern = re.compile(
            rf"{token}\s*(?:轴\s*)?(?:(?:使用|用|按|采用)\s*)?"
            rf"(?:手工|人工|manual)\s*(?:分箱\s*)?"
            rf"(?:切点|断点|breakpoints?)\s*(?:(?:为|是)\s*)?"
            rf"(?:=|:|：)?\s*\[(?P<points>[^\[\]]*)\]",
            re.IGNORECASE,
        )
        for match in pattern.finditer(utterance):
            if command_span is not None and not _cross_mention_is_within(
                match.start(),
                match.end(),
                command_span,
            ):
                ambiguous = True
                continue
            if _automatic_tree_span_is_negated(
                utterance,
                start=match.start(),
                end=match.end(),
            ):
                ambiguous = True
                continue
            raw = (
                "["
                + match.group("points").replace("、", ",").replace("，", ",")
                + "]"
            )
            try:
                values = json.loads(raw)
            except json.JSONDecodeError:
                ambiguous = True
                continue
            if (
                not isinstance(values, list)
                or not 1 <= len(values) <= 19
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int | float)
                    or (
                        isinstance(item, int)
                        and abs(item) > 2**53 - 1
                    )
                    for item in values
                )
            ):
                ambiguous = True
                continue
            points = [float(item) for item in values]
            if (
                any(not math.isfinite(item) for item in points)
                or any(
                    left >= right
                    for left, right in zip(points, points[1:], strict=False)
                )
                or column in bindings
            ):
                ambiguous = True
                continue
            bindings[column] = points
    return bindings, ambiguous

def _ground_univariate_candidate_analysis(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Keep user-owned manual cutpoints byte-for-byte grounded in this turn."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    observed, ambiguous = _explicit_manual_breakpoint_bindings(
        utterance,
        whitelist=whitelist,
    )
    expected = inputs.get("manual_breakpoints", {})
    if not ambiguous and observed == expected:
        return result
    return _clarification(
        "manual 分箱必须用“字段名 manual 切点 [值1, 值2]”明确写出"
        "每个字段的严格递增切点；平台不会让模型补写、改序或把其他数字当切点。",
        code="univariate_manual_breakpoints_not_grounded",
        fields=("manual_breakpoints",),
    )

def utterance_targets_candidate_monthly_stability(utterance: str) -> bool:
    """Reserve candidate/Pool-entry monthly PSI for its governed Workflow."""

    return bool(
        _CANDIDATE_STABILITY_SUBJECT_RE.search(utterance)
        and _CANDIDATE_STABILITY_MEASUREMENT_RE.search(utterance)
    )

def utterance_targets_scorecard_cutoff_selection(utterance: str) -> bool:
    """Reserve explicit scorecard cutoff materialization for its pointer Workflow."""

    scorecard_context = bool(
        _SCORECARD_SUBJECT_RE.search(utterance)
        or _SCORECARD_BAND_ASSET_ID_TOKEN_RE.search(utterance)
    )
    return bool(
        scorecard_context
        and re.search(r"(?:cutoff|通过线|分数线)", utterance, re.IGNORECASE)
        and _SCORECARD_SELECTION_ACTION_RE.search(utterance)
    )

def utterance_targets_scorecard_band_build(utterance: str) -> bool:
    """Reserve complete scorecard-band generation without cutoff selection."""

    return bool(
        not utterance_targets_scorecard_cutoff_selection(utterance)
        and _SCORECARD_SUBJECT_RE.search(utterance)
        and _SCORECARD_BUILD_ACTION_RE.search(utterance)
    )

def _scorecard_raw_pd_edge_mentions(
    utterance: str,
) -> tuple[tuple[float, ...], ...] | None:
    """Parse only explicitly labelled raw-PD arrays; malformed arrays fail closed."""

    mentions: list[tuple[float, ...]] = []
    for match in _SCORECARD_RAW_PD_EDGES_RE.finditer(utterance):
        tokens = [
            token.strip()
            for token in re.split(r"[,，]", match.group("body"))
        ]
        if not tokens or any(not token for token in tokens):
            return None
        values: list[float] = []
        for token in tokens:
            try:
                value = float(Decimal(token))
            except (InvalidOperation, OverflowError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            values.append(value)
        mentions.append(tuple(values))
    return tuple(mentions)

def _ground_scorecard_band_build(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _SCORECARD_HEURISTIC_SELECTION_RE.search(
        utterance
    ) or _SCORECARD_SECOND_OPERATION_RE.search(utterance):
        return _clarification(
            "Scorecard 分数带构建必须是单独一步；自动选择/排名 cutoff、"
            "入池、应用、写回、报告、采纳或部署必须拆成后续请求。",
            code="scorecard_band_single_step_required",
            fields=("workflow",),
        )
    if (
        _SCORECARD_NOT_AUTHORIZED_RE.search(utterance)
        or _SCORECARD_BUILD_ACTION_RE.search(utterance) is None
    ):
        return _clarification(
            "请用当前轮、肯定式命令明确要求构建 Scorecard 完整分数带。",
            code="scorecard_band_positive_command_required",
            fields=("build_intent",),
        )
    bin_counts = tuple(
        int(match.group("count"))
        for match in _SCORECARD_BIN_COUNT_RE.finditer(utterance)
    )
    raw_edges = _scorecard_raw_pd_edge_mentions(utterance)
    expected_count = inputs.get("bin_count")
    expected_edges = inputs.get("raw_pd_band_edges")
    if (
        raw_edges is None
        or (expected_count is not None and bin_counts != (expected_count,))
        or (expected_count is None and bin_counts)
        or (
            expected_edges is not None
            and raw_edges != (tuple(float(value) for value in expected_edges),)
        )
        or (expected_edges is None and raw_edges)
    ):
        return _clarification(
            "bin_count 或 raw_pd_band_edges 只能逐字采用本轮唯一显式值；"
            "两者均未提供时才使用 Tool 默认等频 10 档。",
            code="scorecard_band_controls_not_grounded",
            fields=("bin_count", "raw_pd_band_edges"),
        )
    return result

def _ground_scorecard_cutoff_selection(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    assets = tuple(
        match.group(0)
        for match in _SCORECARD_BAND_ASSET_ID_TOKEN_RE.finditer(utterance)
    )
    cutoffs = tuple(
        match.group(0)
        for match in _SCORECARD_CUTOFF_ID_TOKEN_RE.finditer(utterance)
    )
    if len(assets) != 1 or len(cutoffs) != 1:
        return _clarification(
            "Scorecard cutoff 选择必须逐字提供且只提供一个完整 "
            "scorecard-band-asset ID 与一个完整 scorecard-cutoff ID；"
            "不能按最好、坏率、排名或推荐自动挑选。",
            code="scorecard_cutoff_explicit_id_required",
            fields=("asset_id", "cutoff_id"),
        )
    if _SCORECARD_SECOND_OPERATION_RE.search(utterance):
        return _clarification(
            "Scorecard cutoff 选择必须是单独一步；入池、应用、写回、"
            "报告、采纳或部署必须拆成后续请求。",
            code="scorecard_cutoff_single_step_required",
            fields=("workflow",),
        )
    if (
        _SCORECARD_NOT_AUTHORIZED_RE.search(utterance)
        or _SCORECARD_SELECTION_ACTION_RE.search(utterance) is None
    ):
        return _clarification(
            "请用当前轮、肯定式命令明确选择一个 Scorecard cutoff。",
            code="scorecard_cutoff_positive_command_required",
            fields=("selection_intent",),
        )
    if (
        _SCORECARD_HEURISTIC_SELECTION_RE.search(utterance)
        or assets != (inputs["asset_id"],)
        or cutoffs != (inputs["cutoff_id"],)
    ):
        return _clarification(
            "Scorecard asset/cutoff 必须与用户原话中的唯一完整 pointer "
            "逐字一致；平台不会替换、补全、排名或推荐。",
            code="scorecard_cutoff_controls_not_grounded",
            fields=("asset_id", "cutoff_id"),
        )
    reasons = tuple(
        match.group("reason").strip()
        for match in _SCORECARD_SELECTION_REASON_RE.finditer(utterance)
    )
    reason = inputs.get("reason")
    if bool(reasons or reason is not None) and (
        len(reasons) != 1
        or not isinstance(reason, str)
        or reasons[0] != reason
    ):
        return _clarification(
            "可选 reason 必须与用户以“选择理由/理由/原因/说明”明确标注的"
            "唯一文本逐字一致；未标注时必须省略。",
            code="scorecard_cutoff_reason_not_grounded",
            fields=("reason",),
        )
    return result

def _ground_candidate_monthly_stability_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not utterance_targets_candidate_monthly_stability(utterance):
        return _clarification(
            "原话没有同时明确候选/Pool 条目和逐月稳定性或 PSI；"
            "本 Workflow 不会替代 Pool 影响、通用监控或其他候选操作。",
            code="candidate_monthly_stability_measurement_required",
            fields=("measurement_intent",),
        )
    if (
        _CANDIDATE_STABILITY_NOT_AUTHORIZED_RE.search(utterance)
        or _CANDIDATE_STABILITY_ACTION_RE.search(utterance) is None
    ):
        return _clarification(
            "请用当前轮、肯定式命令明确要求计算一个已有单变量候选资产，"
            "或当前 Strategy Pool 某条目的逐月稳定性/PSI。",
            code="candidate_monthly_stability_positive_command_required",
            fields=("measurement_intent",),
        )
    if _CANDIDATE_STABILITY_SECOND_OPERATION_RE.search(utterance):
        return _clarification(
            "候选逐月稳定性必须是当前轮唯一操作；入池、删改、重排、编译、"
            "写回、报告、采纳或部署请拆成后续请求。",
            code="candidate_monthly_stability_single_operation_required",
            fields=("workflow",),
        )
    if _CANDIDATE_STABILITY_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "候选逐月稳定性的 artifact/hash、Pool revision、活动 workspace、"
            "SampleDesign 与月份字段只能由平台恢复，请不要在请求中指定。",
            code="candidate_monthly_stability_platform_binding_forbidden",
            fields=("platform_binding",),
        )

    asset_ids = tuple(
        match.group(0)
        for match in _CANDIDATE_STABILITY_ASSET_ID_TOKEN_RE.finditer(utterance)
    )
    entry_ids = tuple(
        match.group(0)
        for match in _CANDIDATE_STABILITY_POOL_ENTRY_ID_TOKEN_RE.finditer(
            utterance
        )
    )
    if "asset_id" in inputs:
        expected = str(inputs["asset_id"])
        if (
            len(asset_ids) != 1
            or asset_ids[0] != expected
            or entry_ids
        ):
            return _clarification(
                "请逐字提供且只提供一个完整的单变量 candidate-asset ID；"
                "代词、缺失、多个 ID 或同时出现 Pool entry 时平台不会猜测。",
                code="candidate_monthly_stability_source_not_grounded",
                fields=("asset_id",),
            )
        return result

    expected_entry = str(inputs.get("entry_id") or "")
    strategy_type = str(inputs.get("strategy_type") or "")
    mentioned_types = {
        item[0] for item in _voting_strategy_type_mentions(utterance)
    }
    type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if (
        asset_ids
        or len(entry_ids) != 1
        or entry_ids[0] != expected_entry
        or type_pattern is None
        or type_pattern.search(utterance) is None
        or mentioned_types != {strategy_type}
    ):
        return _clarification(
            "Pool 条目逐月稳定性需要在同一请求中明确且唯一提供 Strategy Pool "
            "类型与一个完整 pool-entry ID；平台不会从动作、历史或其他 Pool 猜测。",
            code="candidate_monthly_stability_source_not_grounded",
            fields=("strategy_type", "entry_id"),
        )
    return result
