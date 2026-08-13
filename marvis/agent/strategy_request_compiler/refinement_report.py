"""refinement_report request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Sequence
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _POOL_MUTATION_WORKFLOWS
    from . import _REFINEMENT_MERGE_ACTION_RE
    from . import _REFINEMENT_SELECTION_ACTION_RE
    from . import _STRATEGY_POOL_APPLY_WORKFLOWS
    from . import _STRATEGY_POOL_MATERIALIZE_WORKFLOWS
    from . import _STRATEGY_POOL_MEASUREMENT_WORKFLOWS
    from . import _STRATEGY_POOL_VALIDATION_WORKFLOWS
    from . import _STRATEGY_POOL_WORKFLOWS
    from . import _clarification
    from . import _explicit_manual_breakpoint_bindings
    from . import _ground_automatic_tree_apply
    from . import _ground_automatic_tree_candidate_build
    from . import _ground_automatic_tree_leaf_materialization
    from . import _ground_candidate_monthly_stability_request
    from . import _ground_cross_matrix_analysis
    from . import _ground_cross_matrix_candidate_build_from_search
    from . import _ground_cross_matrix_candidate_search
    from . import _ground_cross_matrix_cell_selection
    from . import _ground_cross_rule_candidate_build
    from . import _ground_cross_rule_search
    from . import _ground_interactive_tree_auto_continuation
    from . import _ground_interactive_tree_frontier_group_materialization
    from . import _ground_interactive_tree_frontier_materialization
    from . import _ground_interactive_tree_revision
    from . import _ground_interactive_tree_split_search
    from . import _ground_model_score_comparison_v2_request
    from . import _ground_roll_rate_column_bindings
    from . import _ground_scorecard_band_build
    from . import _ground_scorecard_cutoff_selection
    from . import _ground_strategy_impact_cube_request
    from . import _ground_strategy_model_evidence_v2_request
    from . import _ground_strategy_pool_apply_request
    from . import _ground_strategy_pool_impact_request
    from . import _ground_strategy_pool_materialize_request
    from . import _ground_strategy_pool_request
    from . import _ground_strategy_pool_stability_request
    from . import _ground_strategy_pool_validation_request
    from . import _ground_strategy_sample_design_v2_request
    from . import _ground_univariate_candidate_analysis
    from . import _ground_voting_candidate_build
    from . import _ground_voting_candidate_build_from_search
    from . import _ground_voting_candidate_search
    from . import _is_canonical_stored_strategy_report_request
    from . import _utterance_contains_token
    from . import _utterance_supports_risk_threshold
    from . import _utterance_targets_automatic_tree_apply
    from . import _utterance_targets_cross_candidate_search
    from . import _utterance_targets_cross_matrix
    from . import _utterance_targets_cross_matrix_cell_selection
    from . import _utterance_targets_cross_rule_search
    from . import _utterance_targets_cross_rule_selection
    from . import _utterance_targets_cross_search_selection
    from . import _utterance_targets_strategy_model_evidence_v2
    from . import _utterance_targets_strategy_pool_apply
    from . import _utterance_targets_strategy_pool_impact
    from . import _utterance_targets_strategy_pool_validation
    from . import _utterance_targets_voting_candidate
    from . import _utterance_targets_voting_candidate_search
    from . import _utterance_targets_voting_search_selection
    from . import utterance_targets_candidate_monthly_stability
    from . import utterance_targets_interactive_tree_frontier_group_materialization
    from . import utterance_targets_interactive_tree_frontier_materialization
    from . import utterance_targets_model_score_comparison_v2
    from . import utterance_targets_scorecard_band_build
    from . import utterance_targets_scorecard_cutoff_selection
    from . import utterance_targets_strategy_impact_cube
    from . import utterance_targets_strategy_pool_materialize
    from . import utterance_targets_strategy_pool_stability
    from . import utterance_targets_strategy_sample_design

_PROJECT_CONTEXT_SUBJECT_RE = re.compile(
    r"(?:策略)?项目(?:上下文|现状|背景|情况)|当前项目(?:现状|情况)|"
    r"历史(?:版本)?策略(?:效果|复盘|回顾)?|project\s+context|"
    r"current\s+project\s+(?:status|context)|historical\s+strateg(?:y|ies)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_ACTION_RE = re.compile(
    r"(?:整理|梳理|汇总|收集|建立|创建|生成|固化|刷新|更新|补充|记录|盘点|复盘|"
    r"materialize|collect|build|create|refresh|update|record|review)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|先不|暂不|取消|假设|如果|以后|稍后|未来)|"
    r"(?:do\s+not|don't|never|cancel|what\s+if|later|in\s+the\s+future)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_CHAINED_ACTION_RE = re.compile(
    r"(?:然后|接着|随后|并且|并|再).{0,24}"
    r"(?:样本设计|单变量|建模|建树|决策树|入池|策略开发|影响测算|报告|采纳|部署|上线)",
    re.IGNORECASE,
)

_STRATEGY_REPORT_DEFAULT_TITLE = "策略迭代评审报告"

_STRATEGY_REPORT_DEFAULT_STATUS = "partial"

_STRATEGY_REPORT_SUBJECT_RE = re.compile(
    r"(?:策略(?:迭代|开发|分析|项目)?评审报告|"
    r"受治理(?:的)?策略(?:迭代)?(?:评审)?报告|"
    r"StrategyReportBundle(?:\s*V2)?|"
    r"governed\s+strategy\s+report(?:\s+bundle)?|"
    r"strategy\s+(?:iteration|development|review)\s+report(?:\s+bundle)?|"
    r"report\s+bundle\s+(?:for|on)\s+(?:the\s+)?(?:current\s+)?strategy)",
    re.IGNORECASE,
)

_STRATEGY_REPORT_STORED_STRATEGY_RE = re.compile(
    r"(?:已有|已保存|已创建|现有)"
    r"[^；;。.!?？\n]{0,24}策略(?:评审)?报告|"
    r"(?<![A-Za-z0-9_])(?:existing|saved|stored)"
    r"[^;.!?\n]{0,32}\bstrategy(?:\s+review)?\s+report\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_ACTION_RE = re.compile(
    r"(?:生成|创建|制作|编制|形成|出一份|出个|给我|导出|构建)"
    r"[^；;。.!?？\n]{0,80}(?:报告|Report)|"
    r"(?<![A-Za-z0-9_])(?:generate|create|build|produce|prepare|render|export)"
    r"[^;.!?\n]{0,80}\breport(?:\s+bundle)?\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_NEGATED_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止|别|不(?!是))"
    r"[^；;。.!?？\n]{0,48}(?:生成|创建|制作|编制|形成|导出|报告)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|cancel|stop)"
    r"[^;.!?\n]{0,48}(?:generate|create|build|produce|prepare|render|export)"
    r"[^;.!?\n]{0,32}\breport\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能|要不要|会不会|如何|怎么|怎样|假设|假如|如果|"
    r"若|演示|示范|举例|教程|说明一下|解释一下)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|"
    r"should\s+(?:i|we)|is\s+it\s+possible|what\s+if|suppose|assuming|"
    r"how\s+to|hypothetical(?:ly)?|example|demo|test|tutorial)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_PAST_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史上)|"
    r"(?:已经|已)\s*(?:生成|创建|制作|编制|形成|导出)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historically|"
    r"last\s+time|in\s+the\s+past|already\s+(?:generated|created|"
    r"built|produced|prepared|rendered|exported))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_CURRENT_RE = re.compile(
    r"(?:现在|本次|这次|重新|再生成|立即|马上)|"
    r"(?<![A-Za-z0-9_])(?:now|currently|this\s+time|again|regenerate)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_RETRIEVAL_RE = re.compile(
    r"(?:查看|看一下|看下|打开|获取|调出|检索|回看|下载)"
    r"[^；;。.!?？\n]{0,80}(?:报告|Report)|"
    r"(?<![A-Za-z0-9_])(?:view|open|retrieve|get|fetch|show|download)"
    r"[^;.!?\n]{0,80}\breport(?:\s+bundle)?\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_CHAINED_OPERATION_RE = re.compile(
    r"(?:训练(?:模型)?|建模|(?:模型|数据)?评分|打分|"
    r"(?:生成|构建|开发|筛选|分析)(?:策略)?候选|"
    r"(?:测算|计算|评估|回测)(?:当前)?(?:策略池|Pool)?(?:的)?影响|"
    r"影响测算|采纳|采用|部署|上线|投产)|"
    r"(?<![A-Za-z0-9_])(?:train(?:ing)?(?:\s+(?:a\s+)?model)?|"
    r"score(?:\s+(?:the\s+)?(?:model|data|dataset))|"
    r"(?:build|create|generate|develop|select|analy[sz]e)\s+"
    r"(?:a\s+)?(?:strategy\s+)?candidate|"
    r"(?:measure|calculate|assess|backtest)(?:\s+(?:the\s+)?)?"
    r"(?:strategy\s+pool\s+)?impact|adopt|deploy|go[-\s]?live|"
    r"put\s+into\s+production)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_CHAIN_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|不再|并未|未|不|禁止|避免)\s*$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without)\s*$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:project_context_ref|sample_design_ref|candidate_pool_ref|"
    r"candidate_stability_ref|pool_impact_ref|impact_cube_ref|"
    r"strategy_identity|model_evidence_ref|"
    r"training_evidence_ref|score_evidence_ref|report_revision|"
    r"previous_report_id|previous_report_content_hash|generated_at|"
    r"strategy_id|strategy_version|artifact_id|content_hash|"
    r"expected_[a-z0-9_]+|cas|metrics?)(?![A-Za-z0-9_])|"
    r"(?:项目上下文|样本设计|策略池|影响测算|模型证据|训练证据|评分证据)"
    r"\s*(?:artifact|工件|产物)?\s*(?:ID|id|hash|哈希|引用)|"
    r"(?:报告|report)\s*(?:revision|版本)\s*(?:=|:|：)\s*\d+|"
    r"(?:生成时间|generated\s+at)\s*(?:=|:|：)|"
    r"(?:通过率|审批率|准入率|坏账率|风险率|逾期率|"
    r"KS|AUC|PSI|收益|利润|损失)\s*(?:=|:|：|为)\s*[-+]?\d",
    re.IGNORECASE,
)

_STRATEGY_REPORT_TITLE_RE = re.compile(
    r"(?:报告标题|标题|report\s+title|title)\s*"
    r"(?:为|是|叫|is|=|:|：)\s*"
    r"(?:[“\"'《](?P<quoted>[^”\"'》\n]{1,200})[”\"'》]|"
    r"(?P<plain>[^，,；;。.!?？\n]{1,200}))",
    re.IGNORECASE,
)

_STRATEGY_REPORT_STATUS_VALUE_PATTERNS = {
    "draft": re.compile(
        r"(?:草稿|草案)|(?<![A-Za-z0-9_])draft(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "partial": re.compile(
        r"(?:阶段性|部分|中间)|"
        r"(?<![A-Za-z0-9_])partial(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "final": re.compile(
        r"(?:最终|终稿|定稿)|"
        r"(?<![A-Za-z0-9_])final(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}

_STRATEGY_REPORT_STATUS_LABEL_RE = re.compile(
    r"(?:报告)?状态|(?<![A-Za-z0-9_])status(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_STATUS_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|禁止|排除|而非|不是|并非|不使用)"
    r"[^，,；;。.!?？\n]{0,20}$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without|exclude)"
    r"[^,;.!?\n]{0,20}$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_STATUS_HISTORY_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史|已归档|已生成)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historical|"
    r"last\s+time|archived|already\s+generated)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_FORMAT_SEQUENCE = (
    r"(?:Python|SQL|JSON)"
    r"(?:(?:\s*(?:[、,/+]|和|及|与|and)\s*|\s+)"
    r"(?:Python|SQL|JSON)){1,2}"
)

_STRATEGY_DSL_DELIVERY_SUBJECT_RE = re.compile(
    r"(?:策略(?:DSL|代码|交付包|交付文件)|"
    r"策略[^；;。.!?？\n]{0,48}(?:DSL|代码|交付包|交付文件|"
    + _STRATEGY_DSL_DELIVERY_FORMAT_SEQUENCE
    + r")|"
    + _STRATEGY_DSL_DELIVERY_FORMAT_SEQUENCE
    + r"[^；;。.!?？\n]{0,40}(?:策略|代码|交付)|"
    r"\bstrategy\b[^;.!?\n]{0,64}(?:DSL|code|delivery|delivery\s+bundle|"
    + _STRATEGY_DSL_DELIVERY_FORMAT_SEQUENCE
    + r")\b)",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_ACTION_RE = re.compile(
    r"(?:导出|生成|创建|构建|打包|交付|下载)"
    r"[^；;。.!?？\n]{0,80}(?:策略(?:DSL|代码|交付)|Python|SQL|JSON)|"
    r"(?<![A-Za-z0-9_])(?:export|generate|create|build|package|deliver|download)"
    r"[^;.!?\n]{0,80}\b(?:strategy\s+)?"
    r"(?:DSL|code|delivery|Python|SQL|JSON)\b",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_NEGATED_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止|别|不(?!是))"
    r"[^；;。.!?？\n]{0,48}(?:导出|生成|创建|构建|打包|交付|下载)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|cancel|stop)"
    r"[^;.!?\n]{0,48}(?:export|generate|create|build|package|deliver|download)",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能|要不要|会不会|如何|怎么|怎样|"
    r"假设|假如|如果|若|演示|示范|举例|教程|说明一下|解释一下)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|"
    r"should\s+(?:i|we)|is\s+it\s+possible|what\s+if|suppose|assuming|"
    r"how\s+to|hypothetical(?:ly)?|example|demo|test|tutorial)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_PAST_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史上)|"
    r"(?:已经|已)\s*(?:导出|生成|创建|构建|打包|交付|下载)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historically|"
    r"last\s+time|in\s+the\s+past|already\s+(?:exported|generated|created|"
    r"built|packaged|delivered|downloaded))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_CURRENT_RE = re.compile(
    r"(?:现在|本次|这次|重新|再导出|立即|马上)|"
    r"(?<![A-Za-z0-9_])(?:now|currently|this\s+time|again|re-export)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_CHAIN_RE = re.compile(
    r"(?:应用|写回|回写|采纳|采用|部署|上线|投产|晋级|生成报告|创建报告|"
    r"影响测算|训练(?:模型)?|建模|(?:模型|数据)?评分)|"
    r"(?<![A-Za-z0-9_])(?:apply|write\s*back|adopt|deploy|go[-\s]?live|"
    r"put\s+into\s+production|promote|generate\s+(?:a\s+)?report|"
    r"measure\s+impact|train(?:\s+(?:a\s+)?model)?|score)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_CHAIN_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|不再|并未|未|不|禁止|避免)\s*$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without)\s*$",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_NEGATED_CHAIN_LIST_RE = re.compile(
    r"(?:不要|不用|无需|不再|禁止|避免|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|never|without))\s*"
    r"(?:应用|写回|回写|采纳|采用|部署|上线|投产|晋级|生成报告|创建报告|"
    r"影响测算|训练(?:模型)?|建模|(?:模型|数据)?评分|"
    r"apply|write\s*back|adopt|deploy|go[-\s]?live|promote|"
    r"generate\s+(?:a\s+)?report|measure\s+impact|"
    r"train(?:\s+(?:a\s+)?model)?|score)"
    r"(?:\s*(?:、|,|，|或|和|及|与|/|\band\b|\bor\b)\s*"
    r"(?:应用|写回|回写|采纳|采用|部署|上线|投产|晋级|生成报告|创建报告|"
    r"影响测算|训练(?:模型)?|建模|(?:模型|数据)?评分|"
    r"apply|write\s*back|adopt|deploy|go[-\s]?live|promote|"
    r"generate\s+(?:a\s+)?report|measure\s+impact|"
    r"train(?:\s+(?:a\s+)?model)?|score))*",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_PLATFORM_CONTROL_RE = re.compile(
    r"\b(?:strategy_ref|dataset_ref|workspace_ref|workspace_revision|"
    r"analysis_generation|semantic_mapping_hash|expected_strategy_type|expected_version|"
    r"expected_spec_hash|expected_content_hash|maximum_equivalence_rows|"
    r"source_row_count|sample_count|sample_hash|result_hashes|content_hash|"
    r"artifact_id|delivery_id|equivalence_id)\b|"
    r"(?:策略(?:版本|类型|哈希)|样本(?:数据)?(?:ID|哈希)|"
    r"数据集(?:ID|哈希)|数据哈希|等价(?:校验)?(?:样本)?(?:上限|行数)|"
    r"产物ID|工件ID)\s*"
    r"(?:(?:设置|设|指定|调整|改)?(?:为|成)|使用|采用|取|=|:|：)?\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_.:-]*)|"
    r"(?:用|使用|采用|指定)\s*(?:数据集|样本数据)\s*"
    r"(?:(?:为|是)|=|:|：)?\s*[A-Za-z0-9][A-Za-z0-9_.:-]*|"
    r"(?:审批|准入|拒绝|额度|限额|授信|定价|利率|分群|分层)\s*策略|"
    r"版本\s*(?:(?:为|是)|=|:|：)?\s*\d+(?!\d)|"
    r"(?:v(?:ersion)?\s*\d+)[^，,；;。.!?？\n]{0,12}策略|"
    r"策略[^，,；;。.!?？\n]{0,12}(?:v(?:ersion)?\s*\d+)|"
    r"\d+\s*行[^，,；;。.!?？\n]{0,12}等价|"
    r"等价[^，,；;。.!?？\n]{0,12}\d+\s*行|"
    r"\b(?:strategy\s+(?:version|type|hash)|"
    r"dataset(?:\s+(?:id|hash))?|data\s+hash|"
    r"workspace\s+(?:revision|generation)|semantic\s+mapping\s+hash|"
    r"artifact\s+id)\s*(?:(?:is|to|as)|=|:)?\s*"
    r"(?:[-+]?\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_.:-]*)\b|"
    r"\b(?:approval|admission|reject(?:ion)?|limit|pricing|segmentation)"
    r"\s+strategy\b|"
    r"\b(?:v(?:ersion)?\s*\d+)[^;,.!?\n]{0,20}\bstrategy\b|"
    r"\bstrategy\b[^;,.!?\n]{0,20}\b(?:v(?:ersion)?\s*\d+)\b|"
    r"\b(?:use|using|with|select|choose)\s+(?:the\s+)?dataset\s+"
    r"[A-Za-z0-9][A-Za-z0-9_.:-]*\b|"
    r"\b\d+\s*[- ]?\s*rows?\b[^;,.!?\n]{0,24}\bequivalence\b|"
    r"\bequivalence\b[^;,.!?\n]{0,24}\b\d+\s*[- ]?\s*rows?\b|"
    r"\b(?:maximum|max(?:imum)?|limit(?:ed)?\s+to)\s+\d+\s+"
    r"(?:equivalence(?:\s+(?:sample|check))?\s+rows?|"
    r"rows?\s+for\s+equivalence)\b|"
    r"\bequivalence(?:\s+(?:sample|check))?\s+(?:limit|rows?)\s*"
    r"(?:(?:is|to)|=|:)?\s*\d+\b",
    re.IGNORECASE,
)

_STRATEGY_DSL_DELIVERY_STRATEGY_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])strategy-[A-Za-z0-9][A-Za-z0-9_-]*"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

def _ground_refinement_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
) -> StrategyRequestCompilation:
    draft = result.draft
    if _utterance_targets_voting_search_selection(utterance):
        if not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "voting_candidate_build_from_search"
        ):
            return _clarification(
                "原话明确要求从一个 Voting 搜索结果的完整 search_id 与 combo_id "
                "构建候选，只能编译为 voting_candidate_build_from_search；不能改路由为"
                "重新搜索、自由 rule ID 构建、通用策略生命周期或其他 Workflow。",
                code="voting_search_selection_workflow_required",
                fields=("workflow",),
            )
        return _ground_voting_candidate_build_from_search(utterance, result)
    if utterance_targets_strategy_dsl_delivery(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_dsl_delivery"
    ):
        return _clarification(
            "原话明确要求导出离线策略代码和等价证据，只能编译为 "
            "strategy_dsl_delivery；不能改路由到通用策略应用、报告、"
            "采纳或部署。",
            code="strategy_dsl_delivery_workflow_required",
            fields=("workflow",),
        )
    if (
        utterance_targets_strategy_report_bundle_v2(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "strategy_report_bundle_v2"
        )
        and not _is_canonical_stored_strategy_report_request(draft)
    ):
        return _clarification(
            "原话明确要求生成受治理策略评审报告，只能编译为 "
            "strategy_report_bundle_v2；不能改路由到通用策略报告、训练、"
            "评分、候选、影响测算、采纳或部署。",
            code="strategy_report_bundle_v2_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_strategy_project_context(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_project_context"
    ):
        return _clarification(
            "原话明确要求整理当前项目现状或历史策略，只能编译为 "
            "strategy_project_context；不能改路由到样本、候选分析、报告或通用策略生命周期。",
            code="strategy_project_context_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_strategy_sample_design(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_sample_design_v2"
    ):
        return _clarification(
            "原话明确要求固化策略样本设计，只能编译为 strategy_sample_design_v2；"
            "不能改路由到建模、建树、Strategy Pool、报告或通用策略生命周期。",
            code="strategy_sample_design_v2_workflow_required",
            fields=("workflow",),
        )
    if _utterance_targets_strategy_model_evidence_v2(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_model_evidence_v2"
    ):
        return _clarification(
            "原话明确要求归集已有认证单变量证据，只能编译为 "
            "strategy_model_evidence_v2；不能改路由到训练、模型比较、报告或部署。",
            code="strategy_model_evidence_v2_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_model_score_comparison_v2(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_model_score_comparison_v2"
    ):
        return _clarification(
            "原话明确要求物化模型评分比较证据，只能编译为 "
            "strategy_model_score_comparison_v2；不能改路由到训练、"
            "冠军选择、采纳或部署。",
            code="strategy_model_score_comparison_v2_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_candidate_monthly_stability(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "candidate_monthly_stability"
    ):
        return _clarification(
            "原话明确要求候选资产或 Strategy Pool 条目的逐月稳定性/PSI，"
            "只能编译为 candidate_monthly_stability；不能改路由到通用监控、"
            "Pool 影响测算或其他 Workflow。",
            code="candidate_monthly_stability_workflow_required",
            fields=("workflow",),
        )
    if (
        utterance_targets_scorecard_cutoff_selection(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            in {"scorecard_band_build", "scorecard_cutoff_selection"}
        )
    ):
        return _clarification(
            "原话明确要求从完整 Scorecard 分数带中精确选择一个 cutoff，"
            "只能编译为 scorecard_cutoff_selection；不能改路由到分数带构建、"
            "自动推荐、Strategy Pool、采纳或部署。",
            code="scorecard_cutoff_selection_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_scorecard_band_build(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "scorecard_band_build"
    ):
        return _clarification(
            "原话明确要求构建完整 Scorecard 分数带，只能编译为 "
            "scorecard_band_build；不能改路由到 cutoff 选择、自动推荐、"
            "Strategy Pool、采纳或部署。",
            code="scorecard_band_build_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_interactive_tree_frontier_group_materialization(
        utterance
    ) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow
        == "interactive_tree_frontier_group_materialization"
    ):
        return _clarification(
            "原话明确要求从一个交互树 revision 精确物化多个 frontier "
            "node/leaf 的 OR 分组，只能编译为 "
            "interactive_tree_frontier_group_materialization；不能改路由到"
            " singleton、修剪、Strategy Pool 或其他 Workflow。",
            code="interactive_tree_frontier_group_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_interactive_tree_frontier_materialization(
        utterance
    ) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "interactive_tree_frontier_materialization"
    ):
        return _clarification(
            "原话明确要求从一个交互树 revision 精确物化 frontier node/leaf，"
            "只能编译为 interactive_tree_frontier_materialization；不能改路由"
            "到修剪、自动树叶选择、Strategy Pool 或其他 Workflow。",
            code="interactive_tree_frontier_workflow_required",
            fields=("workflow",),
        )
    if _utterance_targets_automatic_tree_apply(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow in {"automatic_tree_apply", "interactive_tree_revision"}
    ):
        return _clarification(
            "原话明确要求把完整自动树写回当前样本，只能编译为 "
            "automatic_tree_apply；不能改路由到通用策略应用、建树、叶节点"
            "物化、入池或其他 Workflow。",
            code="automatic_tree_apply_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_strategy_pool_materialize(utterance):
        if not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "strategy_pool_materialize"
        ):
            return _clarification(
                "原话明确要求把当前 Strategy Pool 物化为持久化 draft Strategy，"
                "只能编译为 strategy_pool_materialize；不能改路由到 Pool 编译预览、"
                "已有策略 build、采纳、部署或其他 Workflow。",
                code="strategy_pool_materialize_workflow_required",
                fields=("workflow",),
            )
        # This dedicated command owns the whole utterance. Ground it now so
        # words such as "backtest/report" in a chained follow-up cannot be
        # mistaken for a different Pool workflow before the single-operation
        # guard reports the precise materialization error.
        return _ground_strategy_pool_materialize_request(utterance, result)
    if _utterance_targets_strategy_pool_apply(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and (
            draft.workflow == "strategy_pool_apply"
            or draft.workflow == "automatic_tree_apply"
            or draft.workflow in _POOL_MUTATION_WORKFLOWS
        )
    ):
        return _clarification(
            "原话明确要求把当前 Strategy Pool 应用或写回当前样本，只能编译为 "
            "strategy_pool_apply；不能改路由到 Pool 编译预览、通用已有策略应用、"
            "影响测算、采纳、部署或其他 Workflow。",
            code="strategy_pool_apply_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_strategy_pool_stability(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_stability"
    ):
        return _clarification(
            "原话明确要求测量当前 Strategy Pool 的跨分区分布稳定性，只能编译为 "
            "strategy_pool_stability；不能改路由到 ImpactCube、独立效果验证、"
            "报告或生命周期操作。",
            code="strategy_pool_stability_workflow_required",
            fields=("workflow",),
        )
    if _utterance_targets_strategy_pool_validation(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_validation"
    ):
        return _clarification(
            "原话明确要求对当前 Strategy Pool 执行 validation/OOT 独立样本"
            "回放验证，只能编译为 strategy_pool_validation；不能改路由到"
            " Pool 影响、逐月稳定性、编译、应用、报告或生命周期操作。",
            code="strategy_pool_validation_workflow_required",
            fields=("workflow",),
        )
    if utterance_targets_strategy_impact_cube(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_impact_cube"
    ):
        return _clarification(
            "原话明确要求五类统一 Strategy ImpactCube，只能编译为 "
            "strategy_impact_cube；不能降级到 approval/reject 旧影响口径、"
            "Pool 修改、报告、采纳或部署。",
            code="strategy_impact_cube_workflow_required",
            fields=("workflow",),
        )
    if _utterance_targets_strategy_pool_impact(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and (
            draft.workflow in {"strategy_pool_impact", "strategy_impact_cube"}
            or (
                draft.workflow == "candidate_monthly_stability"
                and utterance_targets_candidate_monthly_stability(utterance)
            )
        )
    ):
        return _clarification(
            "原话明确要求 Strategy Pool 影响测算，只能编译为 strategy_pool_impact；"
            "不能改路由到 Pool 修改、通用策略生命周期、报告或其他 Workflow。",
            code="strategy_pool_impact_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_rule_selection(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            == "cross_rule_candidate_build_from_search"
        )
    ):
        return _clarification(
            "原话明确提供 Cross rule search_id 与 rule_id 并要求精确构建"
            "候选，只能编译为 cross_rule_candidate_build_from_search；"
            "不能按排名选择、重新搜索或改路由到 Cross Matrix。",
            code="cross_rule_selection_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_rule_search(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "cross_rule_search"
        )
    ):
        return _clarification(
            "原话明确要求搜索 2D/3D Cross 阈值规则，只能编译为 "
            "cross_rule_search；不能改路由到 Cross Matrix 字段对搜索、"
            "显式双轴构建或通用策略生命周期。",
            code="cross_rule_search_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_search_selection(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            == "cross_matrix_candidate_build_from_search"
        )
    ):
        return _clarification(
            "原话明确提供 Cross search_id 与 pair_id 并要求精确构建候选，"
            "只能编译为 cross_matrix_candidate_build_from_search；"
            "不能重新搜索、按排名选择或改路由到其他 Workflow。",
            code="cross_search_selection_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_candidate_search(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "cross_matrix_candidate_search"
        )
    ):
        return _clarification(
            "原话明确要求搜索 Cross Matrix 特征组合，只能编译为 "
            "cross_matrix_candidate_search；不能改路由为显式双轴构建、"
            "通用策略生命周期或其他 Workflow。",
            code="cross_candidate_search_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_matrix_cell_selection(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "cross_matrix_cell_selection"
        )
    ):
        return _clarification(
            "原话明确要求从 Cross Matrix 精确选择单元格，只能编译为 "
            "cross_matrix_cell_selection；不能改路由到矩阵构建、通用策略生命周期"
            "或其他 Workflow。",
            code="cross_matrix_cell_selection_workflow_required",
            fields=("workflow",),
        )
    if (
        _utterance_targets_cross_matrix(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow
            in {
                "cross_matrix_analysis",
                "cross_matrix_cell_selection",
                "cross_matrix_candidate_search",
                "cross_matrix_candidate_build_from_search",
            }
        )
    ):
        return _clarification(
            "原话明确要求二维 Cross Matrix，只能编译为 cross_matrix_analysis；"
            "通用策略生命周期或其他 Workflow 不能消费这两个交叉轴。",
            code="cross_matrix_workflow_required",
            fields=("workflow",),
        )
    if (
        draft is not None
        and _utterance_targets_voting_candidate_search(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "voting_candidate_search"
        )
    ):
        return _clarification(
            "原话明确要求搜索、查找或优化 Voting 组合，只能编译为 "
            "voting_candidate_search；不能改路由为显式成员构建、通用策略"
            "生命周期或其他 Workflow。",
            code="voting_candidate_search_workflow_required",
            fields=("workflow",),
        )
    if (
        draft is not None
        and _utterance_targets_voting_candidate(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow in {"voting_candidate_search", "voting_candidate_build"}
        )
    ):
        return _clarification(
            "原话明确点名 Voting / n-of-k 和多个完整 candidate-rule ID，"
            "只能编译为 voting_candidate_build；通用策略生命周期或其他 "
            "Workflow 不能消费这些控制。",
            code="voting_candidate_workflow_required",
            fields=("workflow",),
        )
    if not isinstance(draft, StandardWorkflowRequestDraft):
        return result
    if draft.workflow == "roll_rate_matrix":
        return _ground_roll_rate_column_bindings(
            utterance,
            result,
            whitelist=whitelist,
            target_col=target_col,
        )
    if draft.workflow == "strategy_project_context":
        return _ground_strategy_project_context_request(utterance, result)
    if draft.workflow == "strategy_sample_design_v2":
        return _ground_strategy_sample_design_v2_request(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "strategy_model_evidence_v2":
        return _ground_strategy_model_evidence_v2_request(utterance, result)
    if draft.workflow == "strategy_model_score_comparison_v2":
        return _ground_model_score_comparison_v2_request(utterance, result)
    if draft.workflow == "candidate_monthly_stability":
        return _ground_candidate_monthly_stability_request(utterance, result)
    if draft.workflow == "scorecard_band_build":
        return _ground_scorecard_band_build(utterance, result)
    if draft.workflow == "scorecard_cutoff_selection":
        return _ground_scorecard_cutoff_selection(utterance, result)
    if draft.workflow == "strategy_dsl_delivery":
        return _ground_strategy_dsl_delivery_request(utterance, result)
    if draft.workflow == "strategy_report_bundle_v2":
        return _ground_strategy_report_bundle_v2_request(utterance, result)
    if draft.workflow == "strategy_pool_stability":
        return _ground_strategy_pool_stability_request(utterance, result)
    if draft.workflow == "strategy_impact_cube":
        return _ground_strategy_impact_cube_request(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow in _STRATEGY_POOL_APPLY_WORKFLOWS:
        return _ground_strategy_pool_apply_request(utterance, result)
    if draft.workflow in _STRATEGY_POOL_MATERIALIZE_WORKFLOWS:
        return _ground_strategy_pool_materialize_request(utterance, result)
    if draft.workflow in _STRATEGY_POOL_VALIDATION_WORKFLOWS:
        return _ground_strategy_pool_validation_request(utterance, result)
    if draft.workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS:
        return _ground_strategy_pool_impact_request(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow in _STRATEGY_POOL_WORKFLOWS:
        return _ground_strategy_pool_request(utterance, result)
    if draft.workflow == "automatic_tree_candidate_build":
        return _ground_automatic_tree_candidate_build(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "automatic_tree_apply":
        return _ground_automatic_tree_apply(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "automatic_tree_leaf_materialization":
        return _ground_automatic_tree_leaf_materialization(utterance, result)
    if draft.workflow == "interactive_tree_split_search":
        return _ground_interactive_tree_split_search(utterance, result)
    if draft.workflow == "interactive_tree_auto_continuation":
        return _ground_interactive_tree_auto_continuation(utterance, result)
    if draft.workflow == "interactive_tree_revision":
        return _ground_interactive_tree_revision(utterance, result)
    if draft.workflow == "interactive_tree_frontier_group_materialization":
        return _ground_interactive_tree_frontier_group_materialization(
            utterance,
            result,
        )
    if draft.workflow == "interactive_tree_frontier_materialization":
        return _ground_interactive_tree_frontier_materialization(
            utterance,
            result,
        )
    if draft.workflow == "voting_candidate_search":
        return _ground_voting_candidate_search(utterance, result)
    if draft.workflow == "voting_candidate_build_from_search":
        return _ground_voting_candidate_build_from_search(utterance, result)
    if draft.workflow == "voting_candidate_build":
        return _ground_voting_candidate_build(utterance, result)
    if draft.workflow == "cross_rule_search":
        return _ground_cross_rule_search(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "cross_rule_candidate_build_from_search":
        return _ground_cross_rule_candidate_build(utterance, result)
    if draft.workflow == "cross_matrix_candidate_search":
        return _ground_cross_matrix_candidate_search(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "cross_matrix_candidate_build_from_search":
        return _ground_cross_matrix_candidate_build_from_search(
            utterance,
            result,
        )
    if draft.workflow == "cross_matrix_cell_selection":
        return _ground_cross_matrix_cell_selection(utterance, result)
    if draft.workflow == "cross_matrix_analysis":
        return _ground_cross_matrix_analysis(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow == "univariate_candidate_analysis":
        return _ground_univariate_candidate_analysis(
            utterance,
            result,
            whitelist=whitelist,
        )
    if draft.workflow != "univariate_candidate_refinement":
        return result
    inputs = draft.to_dict()["workflow_inputs"]
    missing_controls: list[str] = []
    source_candidate_id = inputs.get("source_candidate_id")
    if source_candidate_id is not None and not _utterance_contains_token(
        utterance, source_candidate_id
    ):
        missing_controls.append("source_candidate_id")
    if source_candidate_id is None:
        observed_breakpoints, breakpoint_syntax_ambiguous = (
            _explicit_manual_breakpoint_bindings(
                utterance,
                whitelist=whitelist,
            )
        )
        if (
            breakpoint_syntax_ambiguous
            or observed_breakpoints != inputs.get("manual_breakpoints", {})
        ):
            missing_controls.append("manual_breakpoints")

    selection = inputs["selection"]
    if "source_bin_ids" in selection:
        if not _REFINEMENT_SELECTION_ACTION_RE.search(utterance):
            missing_controls.append("选择动作")
        missing_controls.extend(
            bin_id
            for bin_id in selection["source_bin_ids"]
            if not _utterance_contains_token(utterance, bin_id)
        )
    else:
        threshold = selection["risk_threshold"]
        if not _utterance_supports_risk_threshold(
            utterance,
            operator=threshold["operator"],
            value=threshold["value"],
        ):
            missing_controls.append("明确的观测坏率门槛")

    merge_groups = inputs["merge_groups"]
    if merge_groups:
        if not _REFINEMENT_MERGE_ACTION_RE.search(utterance):
            missing_controls.append("合并动作")
        missing_controls.extend(
            bin_id
            for group in merge_groups
            for bin_id in group
            if not _utterance_contains_token(utterance, bin_id)
        )
    if not missing_controls:
        return result
    return _clarification(
        "请明确提供要选择的 source bin id，或给出可核对的观测坏率门槛；"
        "合并/选择已有箱时还需引用分析结果中展示的完整 candidate ID。"
        "我不会根据“最好”等模糊表述自行生成门槛或重绑分箱。",
        code="strategy_refinement_controls_not_grounded",
        fields=tuple(dict.fromkeys(missing_controls)),
    )

def _ground_strategy_project_context_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        not utterance_targets_strategy_project_context(utterance)
        or _PROJECT_CONTEXT_NONCOMMAND_RE.search(utterance)
    ):
        return _clarification(
            "请单独发出一次立即整理项目现状/历史策略上下文的肯定命令；"
            "问句、否定、假设或未来描述不会刷新项目证据。",
            code="strategy_project_context_positive_command_required",
            fields=("materialize_intent",),
        )
    if _PROJECT_CONTEXT_CHAINED_ACTION_RE.search(utterance):
        return _clarification(
            "本轮只固化项目现状和历史证据；样本设计、候选分析、影响测算、"
            "报告、采纳或部署必须在后续受治理步骤中执行。",
            code="strategy_project_context_single_step_required",
            fields=("next_action",),
        )

    missing: list[str] = []
    if not _project_context_date_is_grounded(utterance, inputs["as_of"]):
        missing.append("as_of")
    scope = inputs.get("scope")
    if isinstance(scope, str) and scope not in utterance:
        missing.append("scope")
    for field_path, value in inputs["business_context"].items():
        if isinstance(value, str):
            if value not in utterance:
                missing.append(f"business_context.{field_path}")
        elif not _project_context_unavailable_is_grounded(utterance, field_path):
            missing.append(f"business_context.{field_path}")
    for field_path in inputs["explicit_unavailable"]:
        if not _project_context_unavailable_is_grounded(utterance, field_path):
            missing.append(f"explicit_unavailable.{field_path}")
    for filename in inputs["external_report_filenames"]:
        # Preserve the exact relative path the user supplied.  Accepting only
        # its basename would let an LLM silently select a different same-name
        # file from a subdirectory of the task source boundary.
        if filename not in utterance:
            missing.append(f"external_report_filenames.{filename}")
    if missing:
        return _clarification(
            "截止日期、项目文字、明确不可用字段和外部报告文件名只能采用用户原话；"
            "平台不会让模型补写背景、缺失状态或证据文件。请补充或删除不在原话中的字段。",
            code="strategy_project_context_controls_not_grounded",
            fields=tuple(missing),
        )
    return result

def _project_context_date_is_grounded(utterance: str, iso_date: str) -> bool:
    if iso_date in utterance:
        return True
    year, month, day = (int(part) for part in iso_date.split("-"))
    return re.search(
        rf"(?<!\d){year}\s*年\s*0?{month}\s*月\s*0?{day}\s*日(?!\d)",
        utterance,
    ) is not None

_PROJECT_CONTEXT_UNAVAILABLE_RE = re.compile(
    r"(?:暂时没有|暂缺|暂无|没有|未提供|不可用|不知道|未知|待补充|"
    r"unavailable|not\s+available|unknown|missing)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_FIELD_LABELS = {
    "approval": re.compile(r"通过率|审批率|准入率|approval", re.IGNORECASE),
    "risk": re.compile(r"坏账率|风险率|逾期率|risk|bad\s+rate", re.IGNORECASE),
    "volume": re.compile(r"申请量|进件量|放款量|业务量|规模|volume", re.IGNORECASE),
    "economics": re.compile(r"收益|利润|成本|经济|economics|profit", re.IGNORECASE),
    "background": re.compile(r"背景|background", re.IGNORECASE),
    "scope": re.compile(r"范围|客群|渠道|产品|scope", re.IGNORECASE),
    "history": re.compile(r"历史|旧版|上一版|history|historical", re.IGNORECASE),
    "historical_strategy_reviews": re.compile(
        r"历史(?:版本)?策略|历史材料|旧版策略|history|historical",
        re.IGNORECASE,
    ),
    "sample": re.compile(r"样本|sample", re.IGNORECASE),
}

def _project_context_unavailable_is_grounded(
    utterance: str,
    field_path: str,
) -> bool:
    if _PROJECT_CONTEXT_UNAVAILABLE_RE.search(utterance) is None:
        return False
    if field_path in utterance:
        return True
    components = tuple(reversed(field_path.split(".")))
    return any(
        label.search(utterance) is not None
        for component in components
        if (label := _PROJECT_CONTEXT_FIELD_LABELS.get(component)) is not None
    )

def utterance_targets_strategy_project_context(utterance: str) -> bool:
    """Recognize an explicit project-context materialization request."""

    return bool(
        _PROJECT_CONTEXT_SUBJECT_RE.search(utterance)
        and _PROJECT_CONTEXT_ACTION_RE.search(utterance)
    )

def utterance_targets_strategy_dsl_delivery(utterance: str) -> bool:
    """Recognize an explicit offline Strategy DSL delivery request."""

    return bool(
        _STRATEGY_DSL_DELIVERY_SUBJECT_RE.search(utterance)
        and _STRATEGY_DSL_DELIVERY_ACTION_RE.search(utterance)
    )

def _ground_strategy_dsl_delivery_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]

    if _STRATEGY_DSL_DELIVERY_NEGATED_RE.search(utterance):
        return _clarification(
            "否定的策略代码导出请求不会创建或执行交付计划；"
            "请在需要执行时单独发出肯定命令。",
            code="strategy_dsl_delivery_intent_negated",
            fields=("delivery_intent",),
        )
    if (
        not utterance_targets_strategy_dsl_delivery(utterance)
        or _STRATEGY_DSL_DELIVERY_NONCOMMAND_RE.search(utterance)
        or (
            _STRATEGY_DSL_DELIVERY_PAST_RE.search(utterance)
            and _STRATEGY_DSL_DELIVERY_CURRENT_RE.search(utterance) is None
        )
    ):
        return _clarification(
            "请单独发出一次立即导出当前策略 Python、SQL、JSON 与等价证据的"
            "肯定命令；问句、假设、演示或仅历史描述不会创建交付。",
            code="strategy_dsl_delivery_positive_command_required",
            fields=("delivery_intent",),
        )
    if _strategy_dsl_delivery_has_positive_chained_operation(utterance):
        return _clarification(
            "本轮只能导出离线策略代码与等价证据；应用、写回、报告、影响测算、"
            "训练、评分、采纳、晋级或部署必须作为后续独立受治理请求。",
            code="strategy_dsl_delivery_single_operation_required",
            fields=("next_action",),
        )
    if _STRATEGY_DSL_DELIVERY_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "策略交付只允许用户提供 strategy_id；策略类型、version/spec hash、"
            "活动数据集及 hash、等价样本预算、artifact id/hash 和结果均由平台绑定。",
            code="strategy_dsl_delivery_platform_binding_forbidden",
            fields=("platform_bindings",),
        )

    mentioned_ids = tuple(
        dict.fromkeys(
            match.group(0)
            for match in _STRATEGY_DSL_DELIVERY_STRATEGY_ID_RE.finditer(
                utterance
            )
        )
    )
    selected_id = inputs.get("strategy_id")
    if selected_id is None:
        if mentioned_ids:
            return _clarification(
                "原话中的完整 strategy_id 必须逐字进入交付请求；平台不会忽略"
                "已点名策略并改用其他策略。",
                code="strategy_dsl_delivery_controls_not_grounded",
                fields=("strategy_id",),
            )
    elif mentioned_ids != (selected_id,):
        return _clarification(
            "策略交付只能逐字使用原话中唯一完整的 strategy_id；多个 ID、"
            "遗漏或模型替换都不会执行。",
            code="strategy_dsl_delivery_controls_not_grounded",
            fields=("strategy_id",),
        )
    return result

def _strategy_dsl_delivery_has_positive_chained_operation(
    utterance: str,
) -> bool:
    active_text = _STRATEGY_DSL_DELIVERY_NEGATED_CHAIN_LIST_RE.sub(
        " ",
        utterance,
    )
    for match in _STRATEGY_DSL_DELIVERY_CHAIN_RE.finditer(active_text):
        prefix = active_text[max(0, match.start() - 16) : match.start()]
        if _STRATEGY_DSL_DELIVERY_CHAIN_NEGATION_RE.search(prefix) is None:
            return True
    return False

def utterance_targets_strategy_report_bundle_v2(utterance: str) -> bool:
    """Recognize a command-shaped governed strategy-report request."""

    return bool(
        _STRATEGY_REPORT_STORED_STRATEGY_RE.search(utterance) is None
        and _STRATEGY_REPORT_RETRIEVAL_RE.search(utterance) is None
        and _STRATEGY_REPORT_SUBJECT_RE.search(utterance)
        and _STRATEGY_REPORT_ACTION_RE.search(utterance)
    )

def _ground_strategy_report_bundle_v2_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]

    if _STRATEGY_REPORT_NEGATED_RE.search(utterance):
        return _clarification(
            "否定的报告请求不会创建或执行报告计划；请在需要执行时单独发出肯定命令。",
            code="strategy_report_bundle_v2_intent_negated",
            fields=("report_intent",),
        )
    if (
        not utterance_targets_strategy_report_bundle_v2(utterance)
        or _STRATEGY_REPORT_NONCOMMAND_RE.search(utterance)
        or (
            _STRATEGY_REPORT_PAST_RE.search(utterance)
            and not _STRATEGY_REPORT_CURRENT_RE.search(utterance)
        )
    ):
        return _clarification(
            "请单独发出一次立即生成受治理策略评审报告的肯定命令；"
            "问句、假设、演示或仅历史描述不会创建报告。",
            code="strategy_report_bundle_v2_positive_command_required",
            fields=("report_intent",),
        )
    if _strategy_report_has_positive_chained_operation(utterance):
        return _clarification(
            "本轮只能生成报告；训练、评分、候选构建/分析、影响测算、"
            "采纳、部署或上线必须作为后续独立受治理请求。",
            code="strategy_report_bundle_v2_single_operation_required",
            fields=("next_action",),
        )
    if _STRATEGY_REPORT_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "报告只允许用户提供 title/status；ProjectContext、SampleDesign、"
            "Pool、ImpactCube/兼容 PoolImpact、模型证据、策略身份、"
            "revision/CAS、generated_at、artifact id/hash 和指标均由平台绑定。",
            code="strategy_report_bundle_v2_platform_binding_forbidden",
            fields=("platform_bindings",),
        )

    missing: list[str] = []
    title_mentions = _strategy_report_title_mentions(utterance)
    if len(title_mentions) > 1:
        missing.append("title")
    elif title_mentions:
        if inputs["title"] != title_mentions[0]:
            missing.append("title")
    elif inputs["title"] != _STRATEGY_REPORT_DEFAULT_TITLE:
        missing.append("title")

    positive_statuses, negated_statuses = _strategy_report_status_mentions(
        utterance
    )
    status_mentions = set(positive_statuses)
    negated_status_mentions = set(negated_statuses)
    if inputs["status"] in negated_status_mentions:
        missing.append("status")
    elif len(status_mentions) > 1:
        missing.append("status")
    elif status_mentions:
        if inputs["status"] != next(iter(status_mentions)):
            missing.append("status")
    elif inputs["status"] != _STRATEGY_REPORT_DEFAULT_STATUS:
        missing.append("status")

    if missing:
        return _clarification(
            "报告标题和状态只能逐字采用用户原话；未提供时平台固定使用"
            "「策略迭代评审报告」与 partial，不允许模型补写或覆盖。",
            code="strategy_report_bundle_v2_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing)),
        )
    return result

def _strategy_report_has_positive_chained_operation(utterance: str) -> bool:
    for match in _STRATEGY_REPORT_CHAINED_OPERATION_RE.finditer(utterance):
        prefix = utterance[max(0, match.start() - 16) : match.start()]
        if _STRATEGY_REPORT_CHAIN_NEGATION_RE.search(prefix) is None:
            return True
    return False

def _strategy_report_title_mentions(utterance: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _STRATEGY_REPORT_TITLE_RE.finditer(utterance):
        value = (match.group("quoted") or match.group("plain") or "").strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)

def _strategy_report_status_mentions(
    utterance: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return positive and prohibited status controls from the report command."""

    masked = list(utterance)
    for match in _STRATEGY_REPORT_TITLE_RE.finditer(utterance):
        masked[match.start() : match.end()] = " " * (
            match.end() - match.start()
        )
    command_text = "".join(masked)
    action_matches = tuple(_STRATEGY_REPORT_ACTION_RE.finditer(command_text))
    if not action_matches:
        return (), ()

    positive: list[str] = []
    negated: list[str] = []
    for status, value_pattern in _STRATEGY_REPORT_STATUS_VALUE_PATTERNS.items():
        for value_match in value_pattern.finditer(command_text):
            if (
                _STRATEGY_REPORT_STATUS_HISTORY_RE.search(
                    _strategy_report_control_clause(
                        command_text,
                        start=value_match.start(),
                        end=value_match.end(),
                    )
                )
                or not _strategy_report_status_shares_command(
                    command_text,
                    value_start=value_match.start(),
                    value_end=value_match.end(),
                    action_matches=action_matches,
                )
            ):
                continue
            if _strategy_report_status_span_is_negated(
                command_text,
                start=value_match.start(),
            ):
                negated.append(status)
                break
            if _strategy_report_status_has_positive_assignment(
                command_text,
                value_start=value_match.start(),
                value_end=value_match.end(),
            ) or any(
                action.start() <= value_match.start()
                and value_match.end() <= action.end()
                for action in action_matches
            ):
                positive.append(status)
                break
    return tuple(positive), tuple(negated)

def _strategy_report_control_clause(
    utterance: str,
    *,
    start: int,
    end: int,
) -> str:
    separators = ("，", ",", "；", ";", "。", ".", "！", "!", "？", "?", "\n")
    clause_start = max(
        utterance.rfind(separator, 0, start) for separator in separators
    )
    clause_end_candidates = [
        position
        for separator in separators
        if (position := utterance.find(separator, end)) >= 0
    ]
    clause_end = (
        min(clause_end_candidates)
        if clause_end_candidates
        else len(utterance)
    )
    return utterance[clause_start + 1 : clause_end]

def _strategy_report_status_has_positive_assignment(
    utterance: str,
    *,
    value_start: int,
    value_end: int,
) -> bool:
    before = utterance[max(0, value_start - 32) : value_start]
    if re.search(
        r"(?:设为|设置为|改为|采用|使用|用|选择|指定为|而是)\s*$|"
        r"(?<![A-Za-z0-9_])(?:set\s+to|use|as|instead)(?![A-Za-z0-9_])\s*$",
        before,
        re.IGNORECASE,
    ):
        return True
    label_matches = tuple(_STRATEGY_REPORT_STATUS_LABEL_RE.finditer(before))
    if label_matches:
        tail = before[label_matches[-1].end() :]
        if re.fullmatch(
            r"\s*(?:(?:设置|设定|指定|设|定)\s*)?"
            r"(?:(?:为|是|用|采用|设为|设置为|=|:|：|is)\s*)?",
            tail,
            re.IGNORECASE,
        ):
            return True

    after = utterance[value_end : value_end + 24]
    return re.match(
        r"\s*(?:版|报告)?\s*(?:(?:作为|设为|设置为|is)\s*)?"
        r"(?:(?:报告)?状态|(?<![A-Za-z0-9_])status(?![A-Za-z0-9_]))",
        after,
        re.IGNORECASE,
    ) is not None

def _strategy_report_status_shares_command(
    utterance: str,
    *,
    value_start: int,
    value_end: int,
    action_matches: Sequence[re.Match[str]],
) -> bool:
    for action in action_matches:
        between = utterance[
            min(action.start(), value_start) : max(action.end(), value_end)
        ]
        if re.search(r"[；;。.!?！？\n]", between) is None:
            return True
    return False

def _strategy_report_status_span_is_negated(
    utterance: str,
    *,
    start: int,
) -> bool:
    prefix = utterance[max(0, start - 32) : start]
    return _STRATEGY_REPORT_STATUS_NEGATION_RE.search(prefix) is not None

