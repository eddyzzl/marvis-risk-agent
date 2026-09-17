"""model_evidence request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _clarification
    from . import _sample_design_clauses

_MODEL_EVIDENCE_SUBJECT_RE = re.compile(
    r"(?:Model\s*Evidence|模型证据|单变量(?:候选)?证据(?:包|汇总)?|"
    r"认证单变量(?:候选)?(?:证据|结果))",
    re.IGNORECASE,
)

_MODEL_EVIDENCE_ACTION_RE = re.compile(
    r"(?:汇总|归集|物化|固化|生成|创建|整理|materialize|aggregate|collect|build|create)",
    re.IGNORECASE,
)

_MODEL_EVIDENCE_CHAIN_RE = re.compile(
    r"(?:训练(?:模型)?|建模|模型对比|比较模型|模型比较|逐月|月度|OOT|时间外|"
    r"验证模型|模型验证|生成报告|形成报告|出报告|部署|投产|上线|采纳)|"
    r"(?<![A-Za-z0-9_])(?:train(?:ing)?|model\s+comparison|compare\s+models?|"
    r"monthly|out[-_\s]*of[-_\s]*time|validation|report|deploy|production|adopt)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_MODEL_EVIDENCE_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|有没有|有无|可以吗|能不能|如何|怎么|怎样|假设|假如|如果|"
    r"未来|将来|以后|稍后|明天|下周|下月)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|what\s+if|"
    r"how\s+to|in\s+the\s+future|later|tomorrow|next\s+(?:week|month))"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_MODEL_EVIDENCE_PLATFORM_CONTROL_RE = re.compile(
    r"\b(?:artifact_id|candidate_id|evidence_hash|content_hash|sample_design_ref|"
    r"membership_artifact_id|bundle_artifact_id|expected_[a-z0-9_]*(?:hash|id))\b|"
    r"(?:工件|候选|证据|样本设计|bundle|membership)\s*(?:ID|id|hash|哈希|引用)",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_SUBJECT_RE = re.compile(
    r"(?:模型(?:评分|分数)(?:证据)?(?:对比|比较)|"
    r"(?:对比|比较)(?:当前任务(?:中)?|已有|已认证)?(?:的)?(?:两个|多个|至少两个)?模型(?:评分|分数)(?:证据)?|"
    r"model[-_\s]*score(?:\s+evidence)?\s+comparison|compare\s+(?:the\s+)?(?:model\s+)?scores?)",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_ACTION_RE = re.compile(
    r"(?:物化|固化|生成|创建|构建|比较|对比|materialize|build|create|compare)",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|如果|未来|以后)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|what\s+if|how\s+to|later)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_PLATFORM_CONTROL_RE = re.compile(
    r"\b(?:artifact(?:_id|_ids)?|dataset(?:_id|_ref)?|sample_design_ref|"
    r"model_score_evidence_refs?|evidence_artifact_id|score_vector_artifact_id|"
    r"expected_[a-z0-9_]+|registry_token|cas|content_hash|"
    r"selected_model_evidence_ref)\b|"
    r"(?:工件|产物|数据集|样本设计|评分证据|分数向量)\s*(?:ID|id|hash|哈希|引用)|"
    r"(?:CAS|registry)\s*(?:token|令牌)",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_SELECTION_RE = re.compile(
    r"(?:选择|挑选|推荐|确定)(?:冠军|胜者|最佳|最优)?|冠军模型|"
    r"(?:采纳|采用|部署|投产|上线)|"
    r"(?<![A-Za-z0-9_])(?:select|recommend|choose|pick|adopt|deploy|go[-\s]?live)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_MODEL_SCORE_COMPARISON_POPULATIONS = {
    "approval": re.compile(
        r"(?<![A-Za-z0-9_])approval(?![A-Za-z0-9_])|审批总体|申请总体",
        re.IGNORECASE,
    ),
    "risk": re.compile(
        r"(?<![A-Za-z0-9_])risk(?![A-Za-z0-9_])|风险总体",
        re.IGNORECASE,
    ),
}

_MODEL_SCORE_COMPARISON_PARTITIONS = {
    "overall": re.compile(
        r"(?<![A-Za-z0-9_])overall(?![A-Za-z0-9_])|整体分区|全量分区",
        re.IGNORECASE,
    ),
    "development": re.compile(
        r"(?<![A-Za-z0-9_])development(?![A-Za-z0-9_])|开发分区|开发样本",
        re.IGNORECASE,
    ),
    "validation": re.compile(
        r"(?<![A-Za-z0-9_])validation(?![A-Za-z0-9_])|验证分区|验证样本",
        re.IGNORECASE,
    ),
    "oot": re.compile(
        r"(?<![A-Za-z0-9_])oot(?![A-Za-z0-9_])|时间外分区|时间外样本",
        re.IGNORECASE,
    ),
}

def utterance_targets_model_score_comparison_v2(utterance: str) -> bool:
    """Reserve explicit model-score comparison for its non-selecting Workflow."""

    return bool(
        _MODEL_SCORE_COMPARISON_SUBJECT_RE.search(utterance)
        and _MODEL_SCORE_COMPARISON_ACTION_RE.search(utterance)
    )

def _utterance_targets_strategy_model_evidence_v2(utterance: str) -> bool:
    historical_or_negated_action = re.compile(
        r"(?:昨天|之前|此前|过去|上次|历史上|曾|曾经|已经|已|"
        r"未|没有|有没有|不要|不用|无需|别|禁止|取消|暂不|先不)\s*$|"
        r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|already|"
        r"do\s+not|don't|never|cancel)\s*$",
        re.I,
    )
    for clause in _sample_design_clauses(utterance):
        subjects = tuple(_MODEL_EVIDENCE_SUBJECT_RE.finditer(clause))
        actions = tuple(_MODEL_EVIDENCE_ACTION_RE.finditer(clause))
        for action in actions:
            prefix = clause[max(0, action.start() - 24) : action.start()]
            if historical_or_negated_action.search(prefix):
                continue
            if any(
                (
                    action.end() <= subject.start()
                    and subject.start() - action.end() <= 48
                )
                or (
                    subject.end() <= action.start()
                    and action.start() - subject.end() <= 32
                )
                for subject in subjects
            ):
                return True
    return False

def _ground_model_score_comparison_v2_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    if _MODEL_SCORE_COMPARISON_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "模型评分比较的 SampleDesign、评分证据、分数向量、artifact id/hash "
            "与 registry CAS 全部由当前 task 自动发现并复核，不能由自然语言注入。",
            code="strategy_model_score_comparison_v2_platform_binding_forbidden",
            fields=("task_context",),
        )
    if (
        not utterance_targets_model_score_comparison_v2(utterance)
        or _MODEL_SCORE_COMPARISON_NONCOMMAND_RE.search(utterance)
    ):
        return _clarification(
            "请明确发出一条当前、肯定式的模型评分比较证据物化命令。",
            code="strategy_model_score_comparison_v2_positive_command_required",
            fields=("build_intent",),
        )
    if _has_positive_chained_operation(
        utterance,
        operation_re=_MODEL_SCORE_COMPARISON_SELECTION_RE,
    ):
        return _clarification(
            "本 Workflow 只物化比较证据并固定为 no_selection；冠军选择、"
            "采纳和部署必须作为后续独立治理动作。",
            code="strategy_model_score_comparison_v2_selection_forbidden",
            fields=("selection",),
        )
    inputs = draft.workflow_inputs
    populations = {
        value
        for value, pattern in _MODEL_SCORE_COMPARISON_POPULATIONS.items()
        if pattern.search(utterance)
    }
    partitions = {
        value
        for value, pattern in _MODEL_SCORE_COMPARISON_PARTITIONS.items()
        if pattern.search(utterance)
    }
    missing: list[str] = []
    if populations != {inputs["population"]}:
        missing.append("population")
    if partitions != {inputs["partition"]}:
        missing.append("partition")
    if missing:
        return _clarification(
            "模型评分比较必须逐字明确唯一 population 与 partition；"
            "平台不会让模型补默认值或改写业务切片。",
            code="strategy_model_score_comparison_v2_dimensions_not_grounded",
            fields=missing,
        )
    return result

def _ground_strategy_model_evidence_v2_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    if _model_evidence_v2_has_positive_chain(utterance):
        return _clarification(
            "当前 ModelEvidence V2 只归集当前 task 中已有的认证单变量候选；"
            "训练、模型比较、月度/OOT/验证模型、报告、采纳和部署必须拆分并等待对应认证证据。",
            code="strategy_model_evidence_v2_univariate_only",
            fields=("requested_evidence",),
        )
    if _MODEL_EVIDENCE_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "ModelEvidence 的 SampleDesign、candidate 与 artifact id/hash 全部由当前 task 发现并复核，"
            "不能由自然语言注入。",
            code="strategy_model_evidence_v2_platform_binding_forbidden",
            fields=("task_context",),
        )
    if (
        not _utterance_targets_strategy_model_evidence_v2(utterance)
        or _MODEL_EVIDENCE_NONCOMMAND_RE.search(utterance)
    ):
        return _clarification(
            "请明确发出一条当前、肯定式命令，只归集已有认证单变量候选为 ModelEvidence V2。",
            code="strategy_model_evidence_v2_positive_command_required",
            fields=("build_intent",),
        )
    return result

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

def _model_evidence_v2_has_positive_chain(utterance: str) -> bool:
    """Return true only for a positively requested downstream operation."""

    return _has_positive_chained_operation(
        utterance,
        operation_re=_MODEL_EVIDENCE_CHAIN_RE,
    )
