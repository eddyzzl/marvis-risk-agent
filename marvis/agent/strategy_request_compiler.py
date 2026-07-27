"""Pure natural-language compiler for governed strategy requests.

The LLM only translates an utterance into a draft.  This module then validates
that draft against fixed operation/type vocabularies, the Strategy DSL and the
dataset column whitelist.  It never executes a tool and never accepts calculated
metrics from the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import math
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from marvis.agent.json_reply import load_json_object
from marvis.data.predicate_ast import PredicateAstError, canonicalize_predicate
from marvis.llm_prompts import STRATEGY_REQUEST_COMPILER_SYS
from marvis.packs.strategy.candidate_design import (
    CANDIDATE_DESIGN_SCHEMA_VERSION,
    CandidateDesignError,
    normalize_candidate_design,
    normalize_candidate_economics_inputs,
)
from marvis.packs.strategy.dsl import StrategyAction, parse_strategy_spec
from marvis.packs.strategy.errors import StrategyError
from marvis.strategy_adoption import AdoptionReasonError, normalize_adoption_reason


_SYSTEM = STRATEGY_REQUEST_COMPILER_SYS.text


STRATEGY_OPERATIONS = (
    "develop",
    "analyze",
    "backtest",
    "apply",
    "compare",
    "adopt",
    "report",
    "monitor",
    "mine_rules",
)
STRATEGY_TYPES = (
    "approval",
    "reject",
    "limit",
    "pricing",
    "segmentation",
)
STRATEGY_REQUEST_KINDS = (
    "strategy_lifecycle",
    "standard_workflow",
)
FRESH_STANDARD_STRATEGY_WORKFLOWS = (
    "strategy_project_context",
    "strategy_sample_design_v2",
    "strategy_model_evidence_v2",
    "profit_calc",
    "roll_rate_matrix",
    "limit_pricing_matrix",
    "univariate_candidate_analysis",
    "univariate_candidate_refinement",
    "candidate_monthly_stability",
    "scorecard_band_build",
    "scorecard_cutoff_selection",
    "automatic_tree_candidate_build",
    "automatic_tree_apply",
    "automatic_tree_leaf_materialization",
    "interactive_tree_revision",
    "interactive_tree_frontier_group_materialization",
    "interactive_tree_frontier_materialization",
    "voting_candidate_search",
    "voting_candidate_build_from_search",
    "voting_candidate_build",
    "cross_matrix_candidate_search",
    "cross_matrix_candidate_build_from_search",
    "cross_rule_search",
    "cross_rule_candidate_build_from_search",
    "cross_matrix_analysis",
    "cross_matrix_cell_selection",
    "strategy_pool_add_candidate",
    "strategy_pool_remove_entry",
    "strategy_pool_set_action",
    "strategy_pool_reorder",
    "strategy_pool_compile",
    "strategy_pool_materialize",
    "strategy_pool_apply",
    "strategy_pool_validation",
    "strategy_pool_impact",
    "strategy_impact_cube",
    "strategy_pool_stability",
    "strategy_dsl_delivery",
    "strategy_report_bundle_v2",
)
LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS = (
    "strategy_sample_design",
)
REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS = (
    *FRESH_STANDARD_STRATEGY_WORKFLOWS,
    *LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS,
)
# Backward-compatible public name: this is intentionally the fresh-emittable
# surface.  Persisted legacy requests must opt into the replay enum explicitly.
STANDARD_STRATEGY_WORKFLOWS = FRESH_STANDARD_STRATEGY_WORKFLOWS

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
_PROJECT_CONTEXT_FIELD_PATH_RE = re.compile(
    r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){0,7}$"
)

_SAMPLE_DESIGN_STATUS_VALUES = {
    "performance_window_status": frozenset({"provided", "unavailable"}),
    "observation_window_status": frozenset({"provided", "unavailable"}),
    "maturity_status": frozenset(
        {"confirmed_matured", "not_matured", "unknown"}
    ),
}
_SAMPLE_DESIGN_SUBJECT_RE = re.compile(
    r"(?:策略)?样本(?:设计|边界|方案)|sample(?:\s|-|_)*design|"
    r"performance\s+window|表现(?:窗|期).{0,20}(?:成熟|观察|样本)",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_ACTION_RE = re.compile(
    r"(?:创建|生成|构建|设计|固化|冻结|物化|计算|分析|探索|先做)|"
    r"(?<![A-Za-z0-9_])(?:create|build|design|freeze|materialize|"
    r"compute|analy[sz]e|explore)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_NEGATED_ACTION_RE = re.compile(
    r"(?:不要|不用|无需|别|禁止|取消|暂不|先不)\s*"
    r"(?:创建|生成|构建|设计|固化|冻结|物化|计算|分析|探索)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel)\s+"
    r"(?:create|build|design|freeze|materialize|compute|analy[sz]e|explore)",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|假如|如果|"
    r"昨天|之前|此前|过去|上次|历史上|未来|将来|以后|稍后|明天|下周|下月)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|what\s+if|"
    r"how\s+to|yesterday|previously|earlier|in\s+the\s+future|later|"
    r"tomorrow|next\s+(?:week|month))(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_EXPLORATION_RE = re.compile(
    r"(?:先|仅|只).{0,12}(?:探索|分析|看|检查)|探索(?:性|阶段)?|"
    r"explor(?:e|ation|atory)|analysis\s+only",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_PERFORMANCE_UNAVAILABLE_RE = re.compile(
    r"(?:表现(?:窗|期)|performance\s+window).{0,18}"
    r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|unavailable|not\s+available|unknown)|"
    r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|unavailable|not\s+available|unknown)"
    r".{0,18}(?:表现(?:窗|期)|performance\s+window)",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_OBSERVATION_UNAVAILABLE_RE = re.compile(
    r"(?:观察(?:窗|期)|observation\s+window).{0,18}"
    r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|unavailable|not\s+available|unknown)|"
    r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|unavailable|not\s+available|unknown)"
    r".{0,18}(?:观察(?:窗|期)|observation\s+window)",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_MATURITY_PATTERNS = {
    "confirmed_matured": re.compile(
        r"(?:确认(?:为)?|已经|已|明确(?:为)?)(?:完全)?成熟|"
        r"成熟度.{0,10}(?:确认(?:为)?(?:已)?成熟|已成熟)|"
        r"(?:confirmed|fully)\s+matured",
        re.IGNORECASE,
    ),
    "not_matured": re.compile(
        r"(?:尚未|还没|未|不)(?:完全)?成熟|not\s+(?:yet\s+)?matured?",
        re.IGNORECASE,
    ),
    "unknown": re.compile(
        r"(?:成熟度|maturity).{0,12}(?:未知|不确定|不知道|unknown)|"
        r"(?:未知|不确定|不知道|unknown).{0,12}(?:成熟度|maturity)",
        re.IGNORECASE,
    ),
}
_SAMPLE_DESIGN_PLATFORM_CONTROL_RE = re.compile(
    r"\b(?:dataset_id|expected_(?:content_)?hash|workspace_(?:revision|generation)|"
    r"analysis_generation|semantic_mapping_hash|target_col|sample_context_hash)\b|"
    r"(?:数据集|样本)\s*(?:ID|id|hash|哈希)|工作区\s*(?:revision|版本)|"
    r"语义(?:映射)?\s*(?:hash|哈希)|目标列",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_V2_PLATFORM_CONTROL_RE = re.compile(
    r"\b(?:legacy_sample_design_ref|scope|policy|dataset_id|"
    r"expected_[a-z0-9_]*hash|workspace_(?:revision|generation)|"
    r"analysis_generation|semantic_mapping_hash|target_col|artifact(?:_id)?|"
    r"sample_design_(?:id|ref)|membership_(?:id|ref)|bundle_(?:id|ref)|"
    r"content_hash|request_hash)\b|"
    r"(?:数据集|样本|工件|产物|bundle|membership)\s*(?:ID|id|hash|哈希|引用)|"
    r"工作区\s*(?:revision|generation|版本|代次)|策略样本(?:设计)?\s*(?:ID|id|引用)|"
    r"(?:诊断)?策略\s*(?:policy|政策)|(?:分析)?范围\s*(?:scope|策略)",
    re.IGNORECASE,
)
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
_STRATEGY_REPORT_DEFAULT_TITLE = "策略迭代评审报告"
_STRATEGY_REPORT_DEFAULT_STATUS = "partial"
_STRATEGY_REPORT_STATUSES = frozenset({"draft", "partial", "final"})
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
_SAMPLE_V2_POPULATION_ROLE_RE = re.compile(
    r"(?:审批(?:总体|样本)|approval\s+population|"
    r"风险(?:总体|样本)|risk\s+population)",
    re.IGNORECASE,
)
_SAMPLE_V2_POPULATION_DIRECTION_RE = re.compile(
    r"(?P<exclusion>排除|剔除|不纳入|exclusion|exclude)"
    r"|(?P<inclusion>(?<!不)纳入|包含|保留|inclusion|include)",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_OTHER_OPERATION_RE = re.compile(
    r"(?:建模|训练模型|评分卡|自动树|决策树|叶节点|策略池|入池|采纳|采用|部署|"
    r"投产|上线|生成报告|形成报告|出报告|模型报告|过滤|筛选|纳入|排除|"
    r"仅保留|只保留|删(?:除)?行|删除(?:异常)?行|清洗|派生(?:字段|列|变量)|"
    r"新增(?:字段|列|变量))|"
    r"(?<![A-Za-z0-9_])(?:model(?:ing)?|scorecard|automatic\s+tree|decision\s+tree|"
    r"strategy\s+pool|adopt|deploy|production|report|filter|clean|derive|"
    r"drop\s+rows?|keep\s+only)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_SAMPLE_DESIGN_V2_CHAIN_RE = re.compile(
    r"(?:建模|训练模型|评分卡|自动树|决策树|叶节点|策略池|入池|采纳|采用|部署|"
    r"投产|上线|生成报告|形成报告|出报告|模型报告|清洗|派生(?:字段|列|变量)|"
    r"新增(?:字段|列|变量))|"
    r"(?<![A-Za-z0-9_])(?:model(?:ing)?|scorecard|automatic\s+tree|decision\s+tree|"
    r"strategy\s+pool|adopt|deploy|production|report|clean|derive)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
UNIVARIATE_BINNING_METHODS = (
    "equal_frequency",
    "equal_width",
    "chimerge",
    "tree",
    "manual",
)
UNIVARIATE_REFINEMENT_METHODS = (*UNIVARIATE_BINNING_METHODS, "categorical")
_CROSS_MATRIX_TARGET_RE = re.compile(
    r"(?:二维|2\s*[dD])[^，,；;。\n]{0,24}(?:交叉|cross)|"
    r"(?:交叉|cross)[^，,；;。\n]{0,24}(?:矩阵|matrix)",
    re.IGNORECASE,
)
_CROSS_MATRIX_BUILD_RE = re.compile(
    r"(?:构建|生成|创建|计算|分析|制作|做)|"
    r"(?<![A-Za-z0-9_])(?:build|create|generate|compute|analy[sz]e|make)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_NEGATED_BUILD_RE = re.compile(
    r"(?:不要|不再|无需|不用|别|禁止)\s*(?:构建|生成|创建|计算|分析|制作|做)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+"
    r"(?:build|create|generate|compute|analy[sz]e|make)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_NONCOMMAND_RE = re.compile(
    r"[?？]|"
    r"(?:能否|可否|是否|可以吗|能不能|要不要|会不会|如何|怎么|怎样|"
    r"假设|假如|如果|若|万一|演示|示范|测试|举例|说明|解释|介绍|"
    r"描述|告诉我|展示)"
    r"[^；;。\n]{0,220}(?:二维|2\s*[dD]|交叉|cross|matrix)|"
    r"(?:昨天|昨日|之前|此前|过去|上次|前次|早些时候|曾经|历史上|"
    r"文档|报告|示例|例子|原文|材料|未来|将来|以后|稍后|晚点|"
    r"回头|明天|后天|下周|下月|下个月|月底|届时)"
    r"[^；;。\n]{0,220}(?:构建|生成|创建|计算|分析|二维|交叉|cross|matrix)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|"
    r"is\s+it\s+possible|what\s+if|suppose|assuming|hypothetically|"
    r"how\s+to|demonstrate|demo|test|example|yesterday|previously|"
    r"earlier|last\s+time|in\s+the\s+future|later|tomorrow|"
    r"next\s+(?:week|month)|when|once|after)"
    r"[^;.!?\n]{0,220}(?:build|create|generate|compute|analy[sz]e|"
    r"cross|matrix)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_POSTPONED_CANCELLATION_RE = re.compile(
    r"(?:^|[，,；;。.!?？！]\s*)(?:等等|等一下|算了|作罢|反悔了|"
    r"取消(?:吧|了)?|撤回|撤销|停止|先不做(?:了)?|暂不做(?:了)?|"
    r"别做(?:了)?|不要做(?:了)?|不执行(?:了)?)(?:[，,。.!！?？]?\s*)$|"
    r"(?:^|[,;.!?]\s*)(?:never\s+mind|forget\s+it|scratch\s+that|"
    r"cancel|abort|withdraw|stop|do(?:n't|\s+not)\s+(?:do|execute)\s+it)"
    r"(?:[,!.?]?\s*)$",
    re.IGNORECASE,
)
_CROSS_MATRIX_CONTROL_REWRITE_RE = re.compile(
    r"(?:不要|不用|别用|不使用|排除|剔除|去掉)\s*(?:用|使用)?"
    r"[^，,；;。\n]{0,32}(?:等频|等数量|分位数|等距|等宽|卡方|"
    r"决策树|类别(?:等值)?箱|quantile|equal[-_\s]*(?:frequency|width)|"
    r"chi[-_\s]*merge|chimerge|tree|categorical)|"
    r"(?:改用|改成|改为|换成|而不是)|"
    r"(?<![A-Za-z0-9_])(?:instead\s+of|switch\s+to|change\s+to)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_COMMAND_CLAUSE_RE = re.compile(r"[^；;。.!！?？\n]+")
_CROSS_MATRIX_BIN_COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<count>\d{1,2})\s*(?:个)?(?:箱|bins?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_MIN_BIN_PCT_RE = re.compile(
    r"(?:最小箱占比|min[_\s-]*bin[_\s-]*(?:pct|share))\s*"
    r"(?:=|:|：|为)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<pct>%?)",
    re.IGNORECASE,
)
_CROSS_MATRIX_SENTINEL_LABEL_RE = re.compile(
    r"(?:哨兵(?:值)?|特殊值|sentinel(?:[_\s-]*values?)?)",
    re.IGNORECASE,
)
_CROSS_MATRIX_SENTINEL_STOP_RE = re.compile(
    r"[，,]\s*(?=(?:最小箱占比|min[_\s-]*bin|放款金额|授信金额|借款金额|"
    r"逾期金额|坏账金额|损失金额|loan[_\s-]*amount|"
    r"overdue[_\s-]*amount|[xXyY]\s*轴|两个轴|每(?:个)?轴|目标箱数))",
    re.IGNORECASE,
)
_CROSS_MATRIX_SENTINEL_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_CROSS_MATRIX_FOLLOW_UP_RE = re.compile(
    r"(?:选(?:择|中)?格|格子入池|加入策略池|入池|采纳|部署|上线|投产|"
    r"写回|回写|生成(?:Python|SQL|代码)|"
    r"(?<![A-Za-z0-9_])(?:select\s+cells?|add\s+to\s+(?:strategy\s+)?pool|"
    r"adopt|deploy|write[-\s]*back|generate\s+(?:python|sql|code))"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_CROSS_SEARCH_ID_RE = re.compile(r"^cross-search-[0-9a-f]{32}$")
_CROSS_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-search-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_CROSS_PAIR_ID_RE = re.compile(r"^cross-pair-[0-9a-f]{32}$")
_CROSS_PAIR_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-pair-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_CROSS_SEARCH_INTENT_RE = re.compile(
    r"(?:搜索|查找|寻找|检索|枚举|筛选|比较)"
    r"[^，,；;。\n]{0,64}(?:Cross|交叉)(?:\s*Matrix|矩阵)?"
    r"[^，,；;。\n]{0,32}(?:组合|候选|特征对|pair)?|"
    r"(?:Cross|交叉)(?:\s*Matrix|矩阵)?"
    r"[^，,；;。\n]{0,48}(?:组合|候选|特征对|pair)"
    r"[^，,；;。\n]{0,32}(?:搜索|查找|检索|枚举|筛选|比较)|"
    r"(?<![A-Za-z0-9_])(?:search|find|enumerate|screen|compare)"
    r"[^;.!?\n]{0,64}cross(?:\s+matrix)?(?:\s+(?:candidate|feature))?"
    r"(?:\s+pairs?)?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_SELECTION_INTENT_RE = re.compile(
    r"(?:构建|物化|生成|创建)[^；;。\n]{0,80}"
    r"(?:Cross|交叉|搜索(?:结果|证据)|组合|候选)|"
    r"(?:Cross|交叉|搜索(?:结果|证据)|组合)"
    r"[^；;。\n]{0,80}(?:构建|物化|生成|创建)|"
    r"(?<![A-Za-z0-9_])(?:build|materialize|create|generate)"
    r"[^;.!?\n]{0,80}(?:cross|search\s+(?:result|evidence)|candidate)|"
    r"(?<![A-Za-z0-9_])(?:cross|search\s+(?:result|evidence))"
    r"[^;.!?\n]{0,80}(?:build|materialize|create|generate)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_FEATURES_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:features?|feature[_\s-]*list|特征(?:列表)?)"
    r"\s*(?:=|:|：|为)\s*\[(?P<value>[^\]\n]{1,1000})\]",
    re.IGNORECASE,
)
_CROSS_SEARCH_MAX_PAIRS_RE = re.compile(
    r"(?<![A-Za-z0-9_])max[_\s-]*pairs?\s*(?:=|:|：|为)?\s*"
    r"(?P<value>\d{1,3})(?![A-Za-z0-9_])|"
    r"最多\s*(?:评估|搜索|比较|枚举)?\s*(?P<zh_value>\d{1,3})\s*"
    r"(?:个|组)?(?:组合|特征对|pairs?)",
    re.IGNORECASE,
)
_CROSS_SEARCH_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:x_method|y_method|axis_methods?|methods?|"
    r"source_artifact_id|expected_artifact_content_hash|"
    r"expected_candidate_id|expected_evidence_hash|dataset_id|target_col|"
    r"candidate_asset|asset_hash|evidence_hash|pair_id|rank|winner|champion)"
    r"\s*(?:=|:|：)|"
    r"(?:轴|分箱)\s*(?:方法|method)\s*(?:=|:|：|为)|"
    r"(?:工件|资产|证据|数据集|目标列)\s*(?:ID|id|hash|哈希)\s*(?:=|:|：|为)",
    re.IGNORECASE,
)
_CROSS_SEARCH_SELECTION_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:artifact_id|artifact_hash|asset_id|asset_hash|"
    r"content_hash|source_artifact_id|expected_[a-z0-9_]+|candidate_id|"
    r"evidence_hash|dataset_id|target_col|x_feature|x_method|y_feature|"
    r"y_method|axis_methods?|features?|max_pairs|rank|winner|champion)"
    r"\s*(?:=|:|：)|"
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])|"
    r"(?:工件|资产|证据|数据集|目标列|轴|分箱方法|排名)"
    r"\s*(?:ID|id|hash|哈希|=|:|：|为)",
    re.IGNORECASE,
)
_CROSS_SEARCH_SELECTION_HEURISTIC_RE = re.compile(
    r"(?:第[一二三四五六七八九十百\d]+名|第一(?:个|名)|最好(?:的)?|"
    r"最优|最佳|冠军|Top\s*[-#]?\s*\d+|排名|名次|刚才(?:那个|这个|的)?|"
    r"上述|这个组合|那个组合)|"
    r"(?<![A-Za-z0-9_])(?:winner|champion|first|best|top\s*[-#]?\s*\d+|"
    r"rank(?:ing)?|previous|that\s+one|this\s+one)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_RESEARCH_RE = re.compile(
    r"(?:重新|再次|再|同时)\s*(?:搜索|查找|检索|枚举|筛选|比较)"
    r"[^，,；;。\n]{0,48}(?:Cross|交叉)|"
    r"(?<![A-Za-z0-9_])(?:re-?search|search|find|enumerate|screen|compare)"
    r"[^,;.!?\n]{0,48}(?:again[^,;.!?\n]{0,16})?cross"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_FOLLOW_UP_RE = re.compile(
    r"(?:构建|物化|生成|创建|选择|选中|入池|加入|放入|写入|纳入|"
    r"设置动作|应用|采纳|部署|上线|投产|写回|回写)|"
    r"(?<![A-Za-z0-9_])(?:build|materialize|create|generate|select|choose|"
    r"add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|set\s+action|apply|"
    r"adopt|deploy|publish|write[- ]?back)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_SELECTION_FOLLOW_UP_RE = re.compile(
    r"(?:入池|加入|放入|写入|纳入|修改(?:策略池|规则池|Pool)|设置动作|"
    r"应用|采纳|部署|上线|投产|写回|回写)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"modify\s+(?:the\s+)?(?:strategy\s+)?pool|set\s+action|apply|adopt|"
    r"deploy|publish|write[- ]?back)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_SEARCH_NEGATION_PREFIX_RE = re.compile(
    r"(?:不|不要|不用|无需|不需要|先不|暂不|不会|不再|别|禁止)\s*$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|never|without)\s*$",
    re.IGNORECASE,
)
_CROSS_RULE_SUBJECT_RE = re.compile(
    r"(?:2|3)\s*[dD][^，,；;。\n]{0,24}(?:Cross|交叉)"
    r"[^，,；;。\n]{0,24}(?:阈值)?规则|"
    r"(?:Cross|交叉)[^，,；;。\n]{0,24}(?:阈值|threshold)"
    r"[^，,；;。\n]{0,16}(?:规则|rules?)|"
    r"(?:Cross|交叉)[^，,；;。\n]{0,16}(?:规则|rules?)",
    re.IGNORECASE,
)
_CROSS_RULE_SEARCH_INTENT_RE = re.compile(
    r"(?:搜索|查找|挖掘|枚举|筛选|探索)|"
    r"(?<![A-Za-z0-9_])(?:search|find|mine|enumerate|screen|explore)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_SEARCH_ID_RE = re.compile(
    r"^cross-rule-search-[0-9a-f]{32}$"
)
_CROSS_RULE_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-rule-search-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_CROSS_RULE_ID_RE = re.compile(r"^cross-rule-[0-9a-f]{32}$")
_CROSS_RULE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-rule-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_CROSS_RULE_SELECTION_INTENT_RE = re.compile(
    r"(?:构建|物化|生成|创建)[^；;。\n]{0,80}(?:Cross|交叉|规则|候选)|"
    r"(?:Cross|交叉|规则|搜索结果)[^；;。\n]{0,80}(?:构建|物化|生成|创建)|"
    r"(?<![A-Za-z0-9_])(?:build|materialize|create|generate)"
    r"[^;.!?\n]{0,80}(?:cross|rule|candidate)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_DIMENSION_RE = re.compile(
    r"(?<![A-Za-z0-9_])dimension\s*(?:=|:|：|为)?\s*(?P<value>[23])"
    r"(?![A-Za-z0-9_])|"
    r"(?P<zh_value>[23])\s*[dD维]",
    re.IGNORECASE,
)
_CROSS_RULE_MIN_LIFT_RE = re.compile(
    r"(?<![A-Za-z0-9_])min[_\s-]*lift\s*(?:=|:|：|为)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_MIN_BAD_COUNT_RE = re.compile(
    r"(?<![A-Za-z0-9_])min[_\s-]*bad[_\s-]*count"
    r"\s*(?:=|:|：|为)?\s*(?P<value>\d+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_MAX_HIT_SHARE_RE = re.compile(
    r"(?<![A-Za-z0-9_])max[_\s-]*hit[_\s-]*share"
    r"\s*(?:=|:|：|为)?\s*(?P<value>\d+(?:\.\d+)?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_MIN_AMOUNT_LIFT_RE = re.compile(
    r"(?<![A-Za-z0-9_])min[_\s-]*amount[_\s-]*lift"
    r"\s*(?:=|:|：|为)?\s*(?P<value>null|none|\d+(?:\.\d+)?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_MAX_TRIALS_RE = re.compile(
    r"(?<![A-Za-z0-9_])max[_\s-]*trials?"
    r"\s*(?:=|:|：|为)?\s*(?P<value>\d{1,5})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_RULE_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:source_artifact_id|"
    r"expected_artifact_content_hash|expected_candidate_id|"
    r"expected_evidence_hash|dataset_id|target_col|thresholds?|directions?|"
    r"rule_id|rank|winner|champion|content_hash|artifact_id)"
    r"\s*(?:=|:|：)",
    re.IGNORECASE,
)
_CROSS_RULE_SELECTION_HEURISTIC_RE = re.compile(
    r"(?:第[一二三四五六七八九十百\d]+名|第一(?:个|名|条)|最好(?:的)?|"
    r"最优|最佳|冠军|Top\s*[-#]?\s*\d+|排名|刚才(?:那个|这个|的)?|"
    r"上述|这个规则|那个规则)|"
    r"(?<![A-Za-z0-9_])(?:winner|champion|first|best|top\s*[-#]?\s*\d+|"
    r"rank(?:ing)?|previous|that\s+one|this\s+one)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_METHOD_GROUNDING = {
    "equal_frequency": re.compile(
        r"(?:等频|等数量|分位数|quantile|equal[-_\s]*frequency)",
        re.IGNORECASE,
    ),
    "equal_width": re.compile(
        r"(?:等距|等宽|equal[-_\s]*width)",
        re.IGNORECASE,
    ),
    "chimerge": re.compile(r"(?:卡方|chi[-_\s]*merge|chimerge)", re.IGNORECASE),
    "tree": re.compile(r"(?:决策树|tree)", re.IGNORECASE),
    "manual": re.compile(
        r"(?:手工|人工|manual)\s*(?:分箱|切点|断点|breakpoints?)?",
        re.IGNORECASE,
    ),
    "categorical": re.compile(
        r"(?:类别等值箱|类别箱|等值箱|categorical)",
        re.IGNORECASE,
    ),
}
AUTOMATIC_TREE_DIRECTIONS = (
    "increasing",
    "decreasing",
    "unordered",
)
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{32}$")
_CANDIDATE_ASSET_ID_RE = re.compile(r"^candidate-asset-[0-9a-f]{32}$")
_POOL_ENTRY_ID_RE = re.compile(r"^pool-entry-[0-9a-f]{32}$")
_CANDIDATE_STABILITY_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_CANDIDATE_STABILITY_POOL_ENTRY_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])pool-entry-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_CANDIDATE_STABILITY_SUBJECT_RE = re.compile(
    r"(?:候选(?:资产|规则)?|策略池(?:条目|规则)|Pool\s*(?:entry|条目)|"
    r"candidate(?:\s+asset)?|candidate-asset-|pool-entry-)",
    re.IGNORECASE,
)
_CANDIDATE_STABILITY_MEASUREMENT_RE = re.compile(
    r"(?:逐月|按月|月度|跨月)[^；;。.!?？\n]{0,40}"
    r"(?:稳定性|分布稳定|PSI)|"
    r"(?:稳定性|分布稳定|PSI)[^；;。.!?？\n]{0,40}"
    r"(?:逐月|按月|月度|跨月)|"
    r"(?<![A-Za-z0-9_])monthly[^;.!?\n]{0,40}"
    r"(?:stability|PSI)|"
    r"(?<![A-Za-z0-9_])(?:stability|PSI)[^;.!?\n]{0,40}"
    r"monthly(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CANDIDATE_STABILITY_ACTION_RE = re.compile(
    r"(?:做|计算|测算|分析|评估|检查|生成|查看)|"
    r"(?<![A-Za-z0-9_])(?:compute|calculate|measure|analy[sz]e|"
    r"assess|evaluate|check|build|show)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CANDIDATE_STABILITY_NOT_AUTHORIZED_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|别|禁止|取消|先不|暂不|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|假如|如果|"
    r"以后|未来|将来|稍后|明天|下周|下月|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|would\s+you|how\s+to|what\s+if|later|tomorrow|"
    r"previously|in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CANDIDATE_STABILITY_SECOND_OPERATION_RE = re.compile(
    r"(?:入池|加入(?:策略)?池|删除|移除|改动作|重排|编译|"
    r"写回|回写|生成报告|形成报告|出报告|采纳|采用|部署|上线|投产)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"remove|delete|reorder|compile|write[-\s]*back|"
    r"generate\s+(?:a\s+)?report|adopt|deploy|go[-\s]?live)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CANDIDATE_STABILITY_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:source_kind|source_artifact_id|"
    r"expected_(?:artifact_)?content_hash|expected_asset_(?:id|hash)|"
    r"expected_pool_(?:revision|snapshot_hash)|dataset_id|"
    r"expected_dataset_content_hash|workspace_(?:revision|generation)|"
    r"analysis_generation|semantic_mapping_hash|sample_design_ref|"
    r"target_col|month_col)(?![A-Za-z0-9_])|"
    r"(?:artifact|数据集|workspace|工作区|样本设计|月份列)"
    r"\s*(?:ID|id|hash|哈希|revision|版本|字段|列)\s*(?:=|:|：)",
    re.IGNORECASE,
)
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
_AUTOMATIC_TREE_LEAF_SELECTION_ID_RE = re.compile(
    r"^automatic-tree-leaf-selection-[0-9a-f]{32}$"
)
_AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])automatic-tree-leaf-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_CROSS_MATRIX_CELL_SELECTION_ID_RE = re.compile(
    r"^cross-matrix-cell-selection-[0-9a-f]{32}$"
)
_CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-matrix-cell-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_SCORECARD_BAND_ASSET_ID_RE = re.compile(
    r"^scorecard-band-asset-[0-9a-f]{32}$"
)
_SCORECARD_BAND_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-band-asset-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_SCORECARD_CUTOFF_ID_RE = re.compile(
    r"^scorecard-cutoff-[0-9a-f]{32}$"
)
_SCORECARD_CUTOFF_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-cutoff-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_SCORECARD_CUTOFF_SELECTION_ID_RE = re.compile(
    r"^scorecard-cutoff-selection-[0-9a-f]{32}$"
)
_SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])scorecard-cutoff-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_FRONTIER_SELECTION_ID_RE = re.compile(
    r"^interactive-tree-frontier-selection-[0-9a-f]{32}$"
)
_INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-frontier-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_RE = re.compile(
    r"^interactive-tree-frontier-group-selection-[0-9a-f]{32}$"
)
_INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"interactive-tree-frontier-group-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_POOL_SOURCE_LIKE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:candidate-asset|automatic-tree-leaf-selection|"
    r"interactive-tree-frontier-group-selection|"
    r"interactive-tree-frontier-selection|cross-matrix-cell-selection|"
    r"scorecard-cutoff-selection)-"
    r"[A-Za-z0-9_-]+(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_POOL_SOURCE_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:candidate-asset|automatic-tree-leaf-selection|"
    r"interactive-tree-frontier-group-selection|"
    r"interactive-tree-frontier-selection|cross-matrix-cell-selection|"
    r"scorecard-cutoff-selection)-",
    re.IGNORECASE,
)
_POOL_SOURCE_CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "\u200b": None,
        "\u200c": None,
        "\u200d": None,
        "\u2060": None,
        "\ufeff": None,
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "﹘": "-",
        "﹣": "-",
        "－": "-",
    }
)
_POOL_MAX_CONTROL_VALUE_CHARS = 4096
_POOL_MAX_UTTERANCE_CHARS = 8192
_POOL_MAX_CONTROL_LABEL_MATCHES = 32
_POOL_UNPARSEABLE_VALUE = object()
_STRATEGY_REPLY_MAX_CHARS = 100_000
_STRATEGY_REPLY_MAX_DEPTH = 64
_STRATEGY_REPLY_MAX_NODES = 10_000
_AUTOMATIC_TREE_LEAF_ID_RE = re.compile(r"^leaf-[0-9a-f]{20}$")
_CROSS_MATRIX_CELL_ID_RE = re.compile(r"^cross-cell-[0-9a-f]{32}$")
_CROSS_MATRIX_CELL_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-cell-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_AUTOMATIC_TREE_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_SOURCE_ID_RE = re.compile(
    r"^(?:candidate-asset|interactive-tree-revision)-[0-9a-f]{32}$"
)
_INTERACTIVE_TREE_SOURCE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:candidate-asset|interactive-tree-revision)-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_NODE_ID_RE = re.compile(r"^node-[0-9a-f]{20}$")
_INTERACTIVE_TREE_NODE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])node-[0-9a-f]{20}(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_REVISION_ID_RE = re.compile(
    r"^interactive-tree-revision-[0-9a-f]{32}$"
)
_INTERACTIVE_TREE_REVISION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-revision-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_FRONTIER_NODE_ID_RE = re.compile(
    r"^(?:node|leaf)-[0-9a-f]{20}$"
)
_INTERACTIVE_TREE_FRONTIER_NODE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:node|leaf)-[0-9a-f]{20}"
    r"(?![A-Za-z0-9_-])"
)
_INTERACTIVE_TREE_FRONTIER_SUBJECT_RE = re.compile(
    r"(?:交互(?:式)?树|树修订|修订树|前沿(?:节点)?|"
    r"interactive[-\s]*tree|tree\s+revision|frontier(?:\s+node)?)",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_ACTION_RE = re.compile(
    r"(?:物化|固化|创建(?:选择|指针)|选中)|"
    r"(?<![A-Za-z0-9_])(?:materialize|persist|select)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_GROUP_SEMANTICS_RE = re.compile(
    r"(?<![A-Za-z0-9_])OR(?![A-Za-z0-9_])|"
    r"(?:逻辑或|或关系|任一(?:节点|成员)?命中|"
    r"(?:按|以|用)\s*或\s*(?:关系|条件|逻辑)?(?:组合|分组))",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_GROUP_INTENT_RE = re.compile(
    r"(?:组合|分组|成组)|"
    r"(?<![A-Za-z0-9_])group(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_GROUP_AMBIGUOUS_SELECTION_RE = re.compile(
    r"(?:全部|所有|整组|一组|若干|多个|这些|上述|刚才(?:那些)?|"
    r"最好|最优|最佳|最差|最坏|风险最高|坏率最高|"
    r"自动(?:选择|挑选|推荐))"
    r"[^，,；;。\n]{0,32}(?:前沿|节点|叶(?:子|节点)?)|"
    r"(?<![A-Za-z0-9_])(?:all|every|some|several|these|those|"
    r"best|worst|highest[-\s]+risk|automatically\s+(?:select|pick|recommend))"
    r"[^,;.!?\n]{0,32}(?:frontier|nodes?|leaves)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_GROUP_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:group_id|selection_id|selection_hash)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_AMBIGUOUS_SELECTION_RE = re.compile(
    r"(?:最好|最优|最佳|最差|最坏|风险最高|坏率最高|"
    r"自动(?:选择|挑选|推荐))"
    r"[^，,；;。\n]{0,24}(?:前沿|节点|叶(?:子|节点)?)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|highest[-\s]+risk|"
    r"automatically\s+(?:select|pick|recommend))"
    r"[^,;.!?\n]{0,24}(?:frontier|node|leaf)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_NEGATED_OR_NONCURRENT_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|先不|暂不|取消|撤销|禁止|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|假设|假如|如果|"
    r"以后|未来|将来|稍后|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|how\s+to|what\s+if|later|previously|"
    r"in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_FRONTIER_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:source_artifact_id|artifact_(?:id|hash)|"
    r"expected_[A-Za-z0-9_]*(?:hash|id)|revision_hash|semantic_tree_id|"
    r"tree_hash|fragment_(?:id|hash)|rule_id|effect_id|condition|metrics|"
    r"dataset_id|workspace_(?:revision|generation)|sample_design_ref)"
    r"(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:artifact|content|selection|tree|fragment)\s+hash"
    r"(?![A-Za-z0-9_])|(?:工件|产物|内容|选择|树|片段)\s*(?:hash|哈希)",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_PRUNE_ACTION_RE = re.compile(
    r"(?:修剪|剪枝|删除(?:该|这个|指定)?(?:节点|子树)|合并(?:该|这个|指定)?子树)|"
    r"(?<![A-Za-z0-9_])prune_subtree(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:prune|remove|delete)\s+"
    r"(?:the\s+)?(?:node|subtree)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_THRESHOLD_ACTION_RE = re.compile(
    r"(?:调整|调节|修改|更改|设置|设定|改动)"
    r"[^，,；;。.!?！？\n]{0,180}(?:分裂|切分)?阈值|"
    r"(?:分裂|切分)?阈值"
    r"[^，,；;。.!?！？\n]{0,180}(?:调整|调节|修改|更改|设置|设定|改为|改成)|"
    r"(?<![A-Za-z0-9_])adjust_split_threshold(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:adjust|change|set)\s+(?:the\s+)?"
    r"(?:split\s+)?threshold(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_THRESHOLD_AMBIGUOUS_RE = re.compile(
    r"(?:调好一点|调(?:整|节)?一点|优化(?:一下)?|自动(?:调整|调节|优化|选择|"
    r"推荐)|最佳阈值|最优阈值|最合适阈值|全部节点|所有节点|每个节点)|"
    r"(?<![A-Za-z0-9_])(?:slightly|best|optimal|automatically\s+"
    r"(?:adjust|optimi[sz]e|select)|all\s+nodes?|every\s+node)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_THRESHOLD_VALUE_RE = re.compile(
    r"(?:新\s*)?(?:分裂|切分)?阈值\s*"
    r"(?:调整|调节|修改|更改|设置|设定|改)?\s*"
    r"(?:为|成|到|=|:|：)\s*"
    r"(?P<zh_value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"(?<![A-Za-z0-9_])(?:adjust|change|set)\s+(?:the\s+)?"
    r"(?:new\s+)?(?:split\s+)?threshold\s+(?:to|=|:)\s*"
    r"(?P<en_value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_AMBIGUOUS_NODE_RE = re.compile(
    r"(?:最好|最优|最佳|最差|风险最高|坏率最高|自动(?:选择|挑选)|"
    r"表现不好|不稳定)"
    r"[^，,；;。\n]{0,20}(?:节点|子树)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|highest[- ]risk|unstable)"
    r"\s+(?:node|subtree)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_NEGATED_OR_NONCURRENT_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|先不|暂不|取消|撤销|禁止|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|假设|假如|如果|"
    r"以后|未来|将来|稍后|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|how\s+to|what\s+if|later|previously|"
    r"in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_INTERACTIVE_TREE_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:source_artifact_id|expected_[A-Za-z0-9_]*hash|"
    r"dataset_id|workspace_(?:revision|generation)|sample_design_ref|"
    r"frontier_node_ids|visible_node_ids|metrics|condition|tree_json)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_TARGET_RE = re.compile(
    r"(?:自动(?:决策)?树|决策树|完整树|"
    r"candidate-asset-[0-9a-f]{32}|"
    r"(?<![A-Za-z0-9_])(?:automatic|decision)\s+tree(?![A-Za-z0-9_]))"
    r"[^，,；;。.!?！？\n]{0,100}"
    r"(?:应用|执行|写回|回写|回填|打标|"
    r"(?<![A-Za-z0-9_])(?:apply|write[-\s]*back|assign)(?![A-Za-z0-9_]))|"
    r"(?:应用|执行|写回|回写|回填|打标|"
    r"(?<![A-Za-z0-9_])(?:apply|write[-\s]*back|assign)(?![A-Za-z0-9_]))"
    r"[^，,；;。.!?！？\n]{0,100}"
    r"(?:自动(?:决策)?树|决策树|完整树|"
    r"candidate-asset-[0-9a-f]{32}|"
    r"(?<![A-Za-z0-9_])(?:automatic|decision)\s+tree(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_ACTION_RE = re.compile(
    r"(?:应用|执行|写回|回写|回填|打标)|"
    r"(?<![A-Za-z0-9_])(?:apply|write[-\s]*back|assign)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_NOT_AUTHORIZED_RE = re.compile(
    r"[?？]|"
    r"(?:不要|不用|无需|别|禁止|取消|先不|暂不|未授权|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|假如|如果|"
    r"以后|未来|将来|稍后|晚点|明天|下周|下月|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|would\s+you|how\s+to|what\s+if|later|tomorrow|"
    r"previously|in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_FOLLOW_UP_RE = re.compile(
    r"(?:策略池|规则池|入池|加入[^，,；;。\n]{0,12}(?:池|Pool)|"
    r"采纳|采用|部署|上线|投产|发布到?生产|生成报告|形成报告|出报告|"
    r"物化[^，,；;。\n]{0,16}(?:叶|leaf)|选择[^，,；;。\n]{0,16}(?:叶|leaf)|"
    r"(?:拒绝|审批|通过|复核)[^，,；;。\n]{0,20}(?:客户|命中)|"
    r"(?<![A-Za-z0-9_])(?:strategy\s+pool|add\s+to\s+(?:the\s+)?pool|"
    r"adopt|deploy|production|go[-\s]+live|generate\s+(?:a\s+)?report|"
    r"materialize\s+(?:a\s+)?leaf|select\s+(?:a\s+)?leaf)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:source_artifact_id|expected_(?:artifact_)?content_hash|"
    r"expected_asset_(?:id|hash)|expected_tree_result_hash|dataset_id|"
    r"workspace_revision|analysis_generation|semantic_mapping_hash|activate_result)"
    r"(?![A-Za-z0-9_])|"
    r"(?:artifact|资产|数据集|workspace|工作区|语义映射)\s*(?:hash|哈希|revision|版本)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_OUTPUT_COLUMN_RE = re.compile(
    r"(?P<label>叶节点|叶子|leaf(?:\s*id)?|规则|rule(?:\s*id)?)\s*"
    r"(?:的)?\s*(?:输出)?\s*(?:字段|列)(?:名)?\s*"
    r"(?:为|是|叫|设为|设置为|=|:|：)?\s*"
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]{0,63})",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_NAMED_OUTPUT_COLUMN_RE = re.compile(
    r"(?P<field>leaf_id_column|rule_id_column)\s*(?:=|:|：)\s*"
    r"(?P<column>[A-Za-z_][A-Za-z0-9_]{0,63})",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_GENERIC_OUTPUT_COLUMN_RE = re.compile(
    r"(?:输出|结果)\s*(?:字段|列)(?:名)?\s*"
    r"(?:为|是|叫|设为|设置为|=|:|：)?\s*"
    r"[A-Za-z_][A-Za-z0-9_]{0,63}",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_APPLY_OUTPUT_COLUMN_NAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"
)
_AUTOMATIC_TREE_LEAF_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])leaf-[0-9a-f]{20}(?![A-Za-z0-9_-])"
)
_AUTOMATIC_TREE_LEAF_AMBIGUOUS_SELECTION_RE = re.compile(
    r"(?:最好|最优|最佳|最差|最坏|"
    r"(?:坏账率|坏率|风险|捕获率|通过率|收益)\s*"
    r"(?:最高|最低|最大|最小)|"
    r"(?:最高|最低|最大|最小)\s*"
    r"(?:坏账率|坏率|风险|捕获率|通过率|收益)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|"
    r"(?:highest|lowest|maximum|minimum)[-\s]+(?:bad[-\s]+rate|risk|lift|"
    r"capture[-\s]+rate|approval[-\s]+rate|profit))(?![A-Za-z0-9_]))"
    r"[^，,；;。\n]{0,20}(?:叶(?:子|节点)?|(?<![A-Za-z0-9_])leaf(?![A-Za-z0-9_]))|"
    r"(?:叶(?:子|节点)?|(?<![A-Za-z0-9_])leaf(?![A-Za-z0-9_]))"
    r"[^，,；;。\n]{0,20}(?:最好|最优|最佳|最差|最坏|"
    r"(?:坏账率|坏率|风险|捕获率|通过率|收益)\s*"
    r"(?:最高|最低|最大|最小)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|"
    r"(?:highest|lowest|maximum|minimum)[-\s]+(?:bad[-\s]+rate|risk|lift|"
    r"capture[-\s]+rate|approval[-\s]+rate|profit))(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_MATERIALIZATION_ACTION_RE = re.compile(
    r"(?:物化|固化|选中|(?<!候)选择)|"
    r"(?<![A-Za-z0-9_])(?:materialize|select|pick)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REASON_RE = re.compile(
    r"(?:(?:选择)?理由|原因|说明)\s*(?:是|为|[:：])\s*"
    r"(?P<zh>(?:(?!(?:但(?:是)?|不过|可是|然而|却|而(?:是)?))[^，,；;。])+)|"
    r"(?<![A-Za-z0-9_])(?:selection\s+reason|reason|rationale)"
    r"\s*(?::|is)\s*"
    r"(?P<en>(?:(?!(?<![A-Za-z0-9_])(?:but|yet|however|instead)"
    r"(?![A-Za-z0-9_]))[^,;.!?])+)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REASON_NEGATION_RE = re.compile(
    r"(?:不要|不|无需|不需要|别|禁止)\s*(?:使用|填写|记录|保留|采用)?\s*$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+(?:use|record|keep)?\s*$",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REASON_REPLACEMENT_RE = re.compile(
    r"(?:(?:选择)?理由|原因|说明)\s*(?:是|为|[:：])|"
    r"(?:改为|改成|换成|替换为)|"
    r"(?<![A-Za-z0-9_])(?:(?:reason|rationale)\s*(?::|is)|instead|"
    r"rather\s+than)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_NEGATED_REASON_CLAUSE_RE = re.compile(
    r"(?:不要|不|无需|不需要|别|禁止)\s*(?:使用|填写|记录|保留|采用)?\s*"
    r"(?:(?:选择)?理由|原因|说明)\s*(?:是|为|[:：])\s*"
    r"(?:(?!(?:但(?:是)?|不过|可是|然而|却|而(?:是)?))[^，,；;。])+|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+"
    r"(?:use|record|keep)?\s*(?:selection\s+reason|reason|rationale)"
    r"\s*(?::|is)\s*(?:(?!(?<![A-Za-z0-9_])(?:but|yet|however|instead)"
    r"(?![A-Za-z0-9_]))[^,;.!?])+",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REASON_FORBIDDEN_OPERATION_RE = re.compile(
    r"(?:策略池|规则池|(?<![A-Za-z0-9_])(?:strategy\s+)?pool(?![A-Za-z0-9_])|"
    r"采纳|部署|上线|投产|投入(?:生产|使用)|发布到?生产|发布|启用|生效|"
    r"激活|落地|执行|应用|使用|运行|拒绝|通过审批|审批|"
    r"写回|回写|回填|"
    r"(?<![A-Za-z0-9_])(?:adopt(?:s|ed|ing)?|deploy(?:s|ed|ing)?|"
    r"promot(?:e|es|ed|ing)|activat(?:e|es|ed|ing)|enabl(?:e|es|ed|ing)|"
    r"effective|publish(?:es|ed|ing)?|releas(?:e|es|ed|ing)|"
    r"launch(?:es|ed|ing)?|production|execut(?:e|es|ed|ing)|"
    r"appl(?:y|ies|ied|ying)|us(?:e|es|ed|ing)|run(?:s|ning)?|"
    r"reject(?:s|ed|ing)?|approv(?:e|es|ed|ing)|rout(?:e|es|ed|ing)|"
    r"go[-\s]+live|roll[-\s]+out|"
    r"write[-\s]*back)"
    r"(?![A-Za-z0-9_])|"
    r"(?:动作|action)\s*(?:改成|设为|设置为|[:=])|"
    r"(?:拒绝|通过|审批|人工复核|复核)[^，,；;。\n]{0,16}(?:客户|命中|叶)|"
    r"(?:客户|命中|叶)[^，,；;。\n]{0,16}(?:拒绝|通过|审批|人工复核|复核)|"
    r"(?<![A-Za-z0-9_])(?:reject|approve|review|route)"
    r"[^,;.!?\n]{0,32}(?:match(?:ing|ed)?|customers?|leaves?|leaf)|"
    r"(?<![A-Za-z0-9_])(?:match(?:ing|ed)?|customers?|leaves?|leaf)"
    r"[^,;.!?\n]{0,32}(?:reject|approve|review|route)(?![A-Za-z0-9_])|"
    r"(?:随后|然后|接着|同时|直接|"
    r"(?<![A-Za-z0-9_])(?:and\s+then|then|afterwards)(?![A-Za-z0-9_])))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE = re.compile(
    r"(?=[^，,；;。\n]{0,120}(?:所有|全部|其他|其余|"
    r"(?<![A-Za-z0-9_])(?:all|every|any)\s+other(?![A-Za-z0-9_])))"
    r"(?=[^，,；;。\n]{0,120}(?:高于|低于|大于|小于|优于|差于|"
    r"(?<![A-Za-z0-9_])(?:higher|lower|greater|less|better|worse)"
    r"(?![A-Za-z0-9_])))|"
    r"(?:最高|最低|最大|最小|最好|最优|最差|最坏|最危险|第一|首位|排名|排行)|"
    r"第\s*(?:\d+|[零一二两三四五六七八九十百]+)\s*(?:名|位)?|"
    r"(?:次高|次低|居首|垫底|末位)|(?:NO\.?\s*1|#\s*1|前\s*\d+\s*名)|"
    r"(?:高于|低于|大于|小于|优于|差于)[^，,；;。\n]{0,24}"
    r"(?:所有|全部|其他|其余)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|top(?:[-\s]*\d+)?|most|least|highest|"
    r"lowest|maximum|minimum|largest|smallest|greatest|fewest|riskiest|safest|"
    r"optimal|leading|trailing|rank(?:ed|ing)?|number\s+(?:one|two|three|\d+)|"
    r"first|second|third|fourth|\d+(?:st|nd|rd|th)|no\.?\s*1|#\s*1|"
    r"(?:higher|lower|greater|less|better|worse)[^,;.!?\n]{0,16}\s+than\s+"
    r"(?:all|every|any)\s+other)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_RATIONALE_START_RE = re.compile(
    r"^(?:"
    r"(?:(?:人工|业务|风险|合规|监管|专家|样本|数据|模型|项目|候选)\s*)?"
    r"(?:确认|复核|评审|审核|验证|分析|判断|讨论|记录|审计|测试|研究|要求|依据)|"
    r"(?:用于|供|后续由)\s*[^，,；;。\n]{0,20}"
    r"(?:确认|复核|评审|审核|验证|分析|判断|讨论|记录|审计|测试|研究)|"
    r"(?:(?:manual|business|risk|compliance|regulatory|expert|sample|data|"
    r"model|project)\s+)?(?:confirmation|review|assessment|validation|analysis|"
    r"judgment|discussion|audit|testing|research|requirement|evidence)|"
    r"(?:for|to\s+support)\s+[^,;.!?\n]{0,24}(?:review|assessment|validation|"
    r"analysis|audit|testing|research)"
    r")",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_RATIONALE_TOKEN_RE = re.compile(
    r"(?:人工|业务|风险|合规|监管|专家|样本|数据|模型|项目|候选|该|这个|本次|"
    r"本轮|下一轮|后续|未来|阶段|叶节点|叶子|叶|用于|供|由|作为|待|再次|"
    r"确认|复核|评审|审核|验证|分析|判断|讨论|记录|审计|测试|研究|要求|依据|说明|"
    r"(?<![A-Za-z0-9_])(?i:manual|business|risk|compliance|regulatory|expert|"
    r"sample|data|model|project|candidate|this|current|next|later|future|phase|"
    r"leaf|for|to|support|confirmation|review|assessment|validation|analysis|"
    r"judgment|discussion|audit|testing|research|requirement|evidence)"
    r"(?![A-Za-z0-9_])|[A-Z0-9][A-Z0-9._-]*|[\u00c0-\u024f]+)"
)
_AUTOMATIC_TREE_LEAF_RATIONALE_PUNCTUATION_RE = re.compile(
    r"[\s，,；;。:：、.!?！？()（）\[\]{}\-_/]+"
)
_AUTOMATIC_TREE_LEAF_RATIONALE_DECISION_SUBJECT_RE = re.compile(
    r"(?:命中|客户|申请人|借款人|用户|业务动作|策略池|规则池|生产|投产)|"
    r"(?<![A-Za-z0-9_])(?:match(?:ing|ed)?|customers?|applicants?|borrowers?|"
    r"actions?|pool|production)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_ALLOWED_REQUEST_TOKEN_RE = re.compile(
    r"(?:请|帮我|麻烦|从|在|把|将|只|仅|也|和|以及|或|但(?:是)?|不过|可是|"
    r"然而|却|而(?:是)?|一个|这个|该|指定|精确|"
    r"完整|自动树|候选树|决策树|树资产|候选资产|资产|树|中|里的|里|的|"
    r"叶节点|叶子|叶|节点|物化|固化|选中|(?<!候)选择|指针|引用|"
    r"再次|确认|是|ID|id|"
    r"(?<![A-Za-z0-9_])(?:please|from|in|the|a|an|this|that|exact|specified|"
    r"automatic|decision|candidate|tree|asset|leaf|node|materialize|select|pick|"
    r"pointer|reference|confirm|again|only|and|or|but|yet|however|instead)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_REQUEST_PUNCTUATION_RE = re.compile(
    r"[\s，,；;。:：、.!?！？()（）\[\]{}\-_/]+"
)
_AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE = re.compile(
    r"(?:加入|写入|放入|添加到).{0,16}(?:策略池|strategy\s*pool)|入池|"
    r"(?<![A-Za-z0-9_])add\b.{0,24}\b(?:strategy\s*)?pool\b",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE = re.compile(
    r"(?:并|并且|然后|随后|再|且|以及|但(?:是)?|可是|不过|"
    r"and(?:\s+then)?|but|yet)"
    r"[^，,；;。\n]{0,48}"
    r"(?:设置为?[^，,；;。\n]{0,12}(?:动作|action)|拒绝|通过审批|人工复核|"
    r"(?<![A-Za-z0-9_])(?:set\s+(?:the\s+)?action|reject|approve|review)"
    r"(?![A-Za-z0-9_]))|"
    r"(?:作为|设为|设置为|转为|执行)"
    r"[^，,；;。\n]{0,20}"
    r"(?:拒绝|通过|审批|人工复核|动作|"
    r"(?<![A-Za-z0-9_])(?:action|reject|approve|review)(?![A-Za-z0-9_]))|"
    r"(?:^|[，,；;])\s*(?:直接|立即)?"
    r"(?:拒绝|让[^，,；;。\n]{0,20}通过审批|转[^，,；;。\n]{0,8}人工复核|"
    r"(?<![A-Za-z0-9_])action\s*[:=]\s*(?:reject|approve|review)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE = re.compile(
    r"(?:并|并且|然后|随后|再|且|以及|但(?:是)?|可是|不过|"
    r"and(?:\s+then)?|but|yet)"
    r"[^，,；;。\n]{0,40}(?:采纳|采用这(?:条|个)|部署|上线|"
    r"(?<![A-Za-z0-9_])(?:adopt|deploy)(?![A-Za-z0-9_]))|"
    r"(?:^|[，,；;])\s*(?:直接|立即)?(?:采纳|采用|部署|上线|"
    r"(?<![A-Za-z0-9_])(?:adopt|deploy)(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE = re.compile(
    r"(?:写回|回写|write[-\s]*back)"
    r"[^，,；;。\n]{0,24}(?:叶(?:子|节点)?\s*(?:id|ID)?|leaf|数据集|dataset)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_NEGATED_CLAUSE_RE = re.compile(
    r"(?:也\s*)?(?:不要|不再|无需|不需要|别|禁止)\s*(?:"
    r"(?:自动\s*)?(?:选择|挑选|推荐|找出)\s*"
    r"(?:最好|最优|最佳|最差|最坏|风险最高|坏率最高)?\s*叶(?:子|节点)?|"
    r"(?:加入|写入|放入|加到)\s*(?:策略池|规则池|pool)|"
    r"(?:采纳|部署|上线)(?:\s*或\s*(?:采纳|部署|上线))*"
    r")|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't)\s+(?:"
    r"(?:automatically\s+)?(?:select|pick)\s+(?:the\s+)?"
    r"(?:best|worst|highest[-\s]+risk)?\s*leaf|"
    r"add\s+(?:it\s+)?to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"(?:adopt|deploy)(?:\s+it)?(?:\s+or\s+(?:adopt|deploy)(?:\s+it)?)*"
    r")(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])without\s+adding\s+(?:it\s+)?to\s+"
    r"(?:the\s+)?(?:strategy\s+)?pool(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_SELECTION_ACTION_RE = re.compile(
    r"(?:物化|固化|选中|选择|提取|引用)"
    r"[^，,；;。\n]{0,24}(?:格子|单元格|cell)|"
    r"(?:格子|单元格|cell)"
    r"[^，,；;。\n]{0,24}(?:物化|固化|选中|选择|提取|引用)|"
    r"(?<![A-Za-z0-9_])(?:materialize|select|pick|extract|reference)"
    r"(?:\s+the)?\s+(?:exact\s+)?cells?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_SELECTION_VERB_RE = re.compile(
    r"(?:物化|固化|选中|选择|提取|引用)|"
    r"(?<![A-Za-z0-9_])(?:materialize|select|pick|extract|reference)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_SELECTION_NEGATED_RE = re.compile(
    r"(?:不要|不再|无需|不用|别|禁止|未|没有)\s*"
    r"[^，,；;。\n]{0,160}(?:物化|固化|选中|选择|提取|引用)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+"
    r"(?:materialize|select|pick|extract|reference)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_AMBIGUOUS_SELECTION_RE = re.compile(
    r"(?:最好|最优|最佳|最差|最坏|高风险|低风险|风险最高|风险最低|"
    r"坏账率最高|坏率最高|lift最高|woe最高|iv最高|前\s*\d+|排名|排行|"
    r"(?<![A-Za-z0-9_])(?:best|worst|top[-\s]*\d+|highest|lowest|"
    r"riskiest|safest|rank(?:ed|ing)?)(?![A-Za-z0-9_]))"
    r"[^，,；;。\n]{0,32}(?:格子|单元格|cells?)|"
    r"(?:格子|单元格|cells?)[^，,；;。\n]{0,32}"
    r"(?:最好|最优|最佳|最差|最坏|高风险|低风险|最高|最低|排名|排行|"
    r"(?<![A-Za-z0-9_])(?:best|worst|top|highest|lowest|riskiest|safest|"
    r"rank(?:ed|ing)?)(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_HEURISTIC_CONTROL_RE = re.compile(
    r"(?:坏账率|坏率|风险|lift|woe|iv|占比|样本量|count|share|bad[-_\s]*rate)"
    r"[^，,；;。\n]{0,24}(?:>=|<=|>|<|高于|低于|大于|小于|不少于|不超过|阈值|门槛)|"
    r"(?:>=|<=|>|<|高于|低于|大于|小于|不少于|不超过|阈值|门槛)"
    r"[^，,；;。\n]{0,24}(?:坏账率|坏率|风险|lift|woe|iv|占比|样本量|count|"
    r"share|bad[-_\s]*rate)",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_NEGATED_FOLLOW_UP_RE = re.compile(
    r"(?:也\s*)?(?:不要|不再|无需|不需要|不|别|禁止)\s*(?:"
    r"(?:加入|写入|放入|加到)\s*(?:策略池|规则池|pool)|入池|"
    r"设置?[^，,；;。\n]{0,12}(?:动作|action)|"
    r"(?:采纳|部署|上线|投产|写回|回写)"
    r"(?:(?:也|或|、|和)(?:不|不要)?(?:采纳|部署|上线|投产|写回|回写))*)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+(?:"
    r"add\s+(?:them?\s+)?to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"set\s+(?:the\s+)?action|adopt|deploy|write[-\s]*back)"
    r"(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])without\s+(?:adding|adopting|deploying)"
    r"[^，,；;。\n]*(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CROSS_MATRIX_CELL_ALLOWED_REQUEST_TOKEN_RE = re.compile(
    r"(?:请|帮我|麻烦|从|在|把|将|只|仅|也|和|以及|但(?:是)?|不过|"
    r"一个|这些|以下|指定|精确|完整|二维|交叉|矩阵|候选|资产|结果|中|里的|"
    r"格子|单元格|格|物化|固化|选中|选择|提取|引用|指针|是|ID|id|"
    r"(?<![A-Za-z0-9_])(?:please|from|in|the|a|an|these|following|exact|"
    r"specified|cross|matrix|candidate|asset|result|cell|cells|materialize|"
    r"select|pick|extract|reference|pointer|only|and|but)(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_POOL_ITEM_ID_RE = re.compile(r"^(?:candidate-rule|pool-entry)-[0-9a-f]{32}$")
_VOTING_RULE_ID_RE = re.compile(r"^candidate-rule-[0-9a-f]{32}$")
_VOTING_RULE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-rule-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_VOTING_SEARCH_ID_RE = re.compile(r"^voting-search-[0-9a-f]{32}$")
_VOTING_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])voting-search-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_VOTING_COMBO_ID_RE = re.compile(r"^voting-combo-[0-9a-f]{32}$")
_VOTING_COMBO_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])voting-combo-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)
_VOTING_SEARCH_SELECTION_INTENT_RE = re.compile(
    r"(?:构建|物化|生成|创建)[^；;。\n]{0,80}"
    r"(?:Voting|投票|n[-_ ]?of[-_ ]?k|候选)|"
    r"(?:Voting|投票|n[-_ ]?of[-_ ]?k|搜索(?:结果|证据)|组合)"
    r"[^；;。\n]{0,80}(?:构建|物化|生成|创建)|"
    r"(?<![A-Za-z0-9_])(?:build|materialize|create|generate)"
    r"[^;.!?\n]{0,80}(?:voting|n[-_ ]?of[-_ ]?k|candidate)|"
    r"(?<![A-Za-z0-9_])(?:voting|n[-_ ]?of[-_ ]?k|search\s+(?:result|evidence))"
    r"[^;.!?\n]{0,80}(?:build|materialize|create|generate)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_SELECTION_HEURISTIC_RE = re.compile(
    r"(?:第[一二三四五六七八九十百\d]+名|第一(?:个|名)|最好(?:的)?|最优|"
    r"最佳|冠军|Top\s*[-#]?\s*\d+|排名|名次|刚才(?:那个|这个|的)?|"
    r"上述|这个组合|那个组合)|"
    r"(?<![A-Za-z0-9_])(?:winner|champion|first|best|top\s*[-#]?\s*\d+|"
    r"rank(?:ing)?|previous|that\s+one|this\s+one)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_SELECTION_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:artifact_id|artifact_hash|artifact_content_hash|"
    r"expected_artifact_content_hash|search_content_hash|"
    r"expected_search_content_hash|expected_content_hash|content_hash|"
    r"(?:expected_)?pool_(?:revision|snapshot_hash|id)|"
    r"(?:expected_)?revision_id|pool_ref|dataset_id|dataset_binding|"
    r"(?:expected_)?dataset_content_hash|target_(?:col|polarity|semantics)|"
    r"target_binding|polarity|"
    r"sample_design_(?:ref|id|(?:content_)?hash|partition)|partition|"
    r"workspace_(?:revision|generation)|semantic_mapping(?:_hash)?|"
    r"requirement_bindings?|observation_bindings|provenance|rule_ids|"
    r"member_rule_ids|member_ids|entry_ids|selected_entry_ids|n|rank)"
    r"\s*(?:=|:|：)|"
    r"(?<![A-Za-z0-9_])(?:pool\s+(?:revision|snapshot\s+hash|id)|"
    r"revision\s+id|dataset\s+(?:id|content\s+hash)|content\s+hash|"
    r"target\s+(?:column|col|polarity|semantics)|polarity|"
    r"sample\s+design\s+(?:reference|ref|id|hash|partition)|"
    r"workspace\s+(?:revision|generation)|semantic\s+mapping(?:\s+hash)?|"
    r"requirement\s+bindings?)\s*(?:=|:|：)|"
    r"(?<![A-Za-z0-9_-])(?:candidate-rule|pool-entry)-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])|"
    r"(?:artifact|工件)\s*(?:id|hash|哈希|引用)|"
    r"(?:(?:(?:策略|规则)?池|(?:Strategy\s+)?Pool)\s*"
    r"(?:版本|修订|快照(?:哈希|hash)|ID|id)|"
    r"(?:版本|修订)ID|(?:数据集|数据)(?:ID|id|内容(?:哈希|hash))|"
    r"(?:目标|标签|坏标签)(?:列|字段|极性|语义|方向|取值)|"
    r"样本设计(?:引用|ID|id|(?:内容)?(?:哈希|hash)|分区)|"
    r"工作区(?:版本|修订|代次|revision|generation)|"
    r"语义映射(?:(?:哈希|hash))?|(?:规则)?需求绑定)\s*(?:=|:|：|为)",
    re.IGNORECASE,
)
_VOTING_SEARCH_SELECTION_FOLLOW_UP_RE = re.compile(
    r"(?:加入|放入|写入|纳入|添加|加到)"
    r"[^，,；;。\n]{0,24}(?:策略池|规则池|Pool)|"
    r"(?:修改|调整|变更|编辑)[^，,；;。\n]{0,20}"
    r"(?:策略池|规则池|Pool)|"
    r"(?:设置|设为|改为)[^，,；;。\n]{0,20}(?:拒绝|审批|复核|动作)|"
    r"(?:入池|设置动作|应用|套用|执行|采纳|部署|上线|投产|写回|回写)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"modify\s+(?:the\s+)?(?:strategy\s+)?pool|set\s+action|apply|adopt|"
    r"deploy|publish|write[- ]?back)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_SELECTION_RESEARCH_RE = re.compile(
    r"(?:重新|再次|再)?(?:搜索|查找|寻找|检索|枚举|优化|筛选)"
    r"[^，,；;。\n]{0,48}(?:投票|Voting|n[-_ ]?of[-_ ]?k)(?:组合|候选)?|"
    r"(?:搜索|查找|寻找|找|检索|枚举|优化|筛选)"
    r"[^，,；;。\n]{0,16}(?:一遍|一次|更好(?:的)?(?:组合|候选))|"
    r"(?<![A-Za-z0-9_])(?:re-?search|search|find|enumerate|optimi[sz]e)"
    r"[^,;.!?\n]{0,48}(?:voting|n[-_ ]?of[-_ ]?k)(?:\s+combinations?)?"
    r"(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:re-?search|search(?:ing)?|find|enumerate|optimi[sz]e)"
    r"[^,;.!?\n]{0,32}(?:again|better\s+(?:combination|candidate))"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_SELECTION_NEGATED_RESEARCH_RE = re.compile(
    r"(?:不|不要|不用|无需|不需要|先不|暂不|别|禁止)"
    r"[^，,；;。\n]{0,20}(?:重新|再次|再)?"
    r"(?:搜索|查找|寻找|找|检索|枚举|优化|筛选)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|never|without)"
    r"[^,;.!?\n]{0,24}(?:re-?search|search(?:ing)?|find|enumerate|optimi[sz]e)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SUBJECT_RE = re.compile(
    r"(?:投票|(?<![A-Za-z0-9_])(?:Voting|n[-_ ]?of[-_ ]?k)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_VOTING_SEARCH_INTENT_RE = re.compile(
    r"(?:搜索|查找|寻找|检索|枚举|优化|筛选)"
    r"[^，,；;。\n]{0,48}(?:投票|Voting|n[-_ ]?of[-_ ]?k)(?:组合|候选)?|"
    r"(?:投票|Voting|n[-_ ]?of[-_ ]?k)(?:组合|候选)?"
    r"[^，,；;。\n]{0,48}(?:搜索|查找|寻找|检索|枚举|优化|筛选)|"
    r"(?<![A-Za-z0-9_])(?:search|find|enumerate|optimi[sz]e|screen)"
    r"[^,;.!?\n]{0,48}(?:voting|n[-_ ]?of[-_ ]?k)(?:\s+combinations?)?|"
    r"(?<![A-Za-z0-9_])(?:voting|n[-_ ]?of[-_ ]?k)(?:\s+combinations?)?"
    r"[^,;.!?\n]{0,48}(?:search|find|enumerate|optimi[sz]e|screen)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_NEGATED_RE = re.compile(
    r"(?:不要|不用|无需|不需要|先不|暂不|取消|停止)"
    r"[^，,；;。\n]{0,32}(?:搜索|查找|寻找|检索|枚举|优化|筛选|比较|"
    r"投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|cancel|stop)"
    r"[^,;.!?\n]{0,32}(?:search|find|enumerate|optimi[sz]e|screen|compare|"
    r"voting|n[-_ ]?of[-_ ]?k)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_RESULT_REFERENCE = (
    r"(?:组合|候选|结果|第[一二三四五六七八九十百\d]+名|"
    r"Top\s*[-#]?\s*\d*)"
)
_VOTING_SEARCH_POOL_REFERENCE = (
    r"(?:策略池|规则池|(?<![A-Za-z0-9_])Pool(?![A-Za-z0-9_]))"
)
_VOTING_SEARCH_FOLLOW_UP_OPERATION_RE = re.compile(
    rf"(?:构建|生成|创建|物化)[^，,；;。\n]{{0,32}}"
    rf"(?:{_VOTING_SEARCH_RESULT_REFERENCE}|Voting|n[-_ ]?of[-_ ]?k)|"
    rf"(?:选择|选中|选取|挑选|采用|使用)[^，,；;。\n]{{0,24}}"
    rf"{_VOTING_SEARCH_RESULT_REFERENCE}|"
    rf"(?:应用|套用|执行)[^，,；;。\n]{{0,32}}"
    rf"(?:{_VOTING_SEARCH_RESULT_REFERENCE}|当前样本)|"
    rf"{_VOTING_SEARCH_RESULT_REFERENCE}[^，,；;。\n]{{0,24}}"
    r"(?:应用|套用|执行)|"
    r"(?:加入|放入|写入|纳入|添加|加到)"
    rf"[^，,；;。\n]{{0,24}}{_VOTING_SEARCH_POOL_REFERENCE}|"
    rf"(?:修改|调整|变更|编辑)[^，,；;。\n]{{0,16}}"
    rf"{_VOTING_SEARCH_POOL_REFERENCE}|"
    rf"{_VOTING_SEARCH_RESULT_REFERENCE}[^，,；;。\n]{{0,24}}"
    r"(?:设置|设为|改为)[^，,；;。\n]{0,16}(?:拒绝|审批|复核|动作)|"
    r"(?:入池|设置动作|采纳|部署|上线|投产|写回|回写)|"
    r"(?<![A-Za-z0-9_])(?:add|put|write|insert)"
    r"[^,;.!?\n]{0,32}(?:to|into)\s+(?:the\s+)?"
    r"(?:strategy\s+|rule\s+)?pool(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:use|adopt|apply)"
    r"[^,;.!?\n]{0,24}(?:top|first|second|third|result|combination|candidate)"
    r"(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:modify|change|update|edit)"
    r"[^,;.!?\n]{0,20}(?:strategy\s+|rule\s+)?pool(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])set[^,;.!?\n]{0,32}"
    r"(?:action|reject|approve|review)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:build|create|materialize|select|choose|admit|"
    r"add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|set\s+action|adopt|"
    r"apply|deploy|publish|write[- ]?back)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_NEGATED_FOLLOW_UP_RE = re.compile(
    r"(?:不要|不用|无需|不需要|先不|暂不|不会|不再|不|别|禁止)"
    r"[^，,；;。\n]{0,16}(?:构建|生成|创建|物化|选择|选中|选取|挑选|"
    r"采用|使用|应用|套用|执行|入池|加入|放入|写入|纳入|添加|加到|"
    r"修改|调整|变更|编辑|设置|设为|改为|采纳|部署|上线|投产|写回|回写)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|never|without)"
    r"[^,;.!?\n]{0,20}(?:build|create|materialize|select|choose|admit|"
    r"add|put|insert|use|apply|modify|change|update|edit|set|adopt|"
    r"deploy|publish|write[- ]?back)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:pool_ref|dataset_id|target_col|hit_matrix|"
    r"weights|amounts|search_result|artifact_id|content_hash)"
    r"(?![A-Za-z0-9_])|"
    r"(?:数据集|目标列|命中矩阵|样本权重|金额向量|工件)"
    r"\s*(?:ID|id|hash|哈希|引用)",
    re.IGNORECASE,
)
_VOTING_SEARCH_MEMBER_COUNT_PATTERNS = (
    re.compile(
        r"(?<![A-Za-z0-9_])(?:K|member[_ -]?count)"
        r"\s*(?:=|:|：|为)?\s*(?P<k>\d+)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<k>\d+)\s*(?:个|条|项)?\s*选\s*(?P<n>\d+)"),
    re.compile(
        r"(?P<n>\d+)\s*(?:-|/|\s)\s*(?:of|OF)\s*(?P<k>\d+)",
        re.IGNORECASE,
    ),
)
_VOTING_SEARCH_METRIC_ALIASES = {
    "hit_count": "hit_count",
    "命中样本数": "hit_count",
    "命中数": "hit_count",
    "hit_share": "hit_share",
    "命中样本占比": "hit_share",
    "命中占比": "hit_share",
    "命中率": "hit_share",
    "good_count": "good_count",
    "好样本数": "good_count",
    "bad_count": "bad_count",
    "坏样本数": "bad_count",
    "bad_rate": "bad_rate",
    "坏样本率": "bad_rate",
    "坏率": "bad_rate",
    "坏账率": "bad_rate",
    "lift": "lift",
    "提升度": "lift",
    "bad_capture_rate": "bad_capture_rate",
    "坏样本捕获率": "bad_capture_rate",
    "坏样本召回率": "bad_capture_rate",
    "weighted_hit_total": "weighted_hit_total",
    "加权命中总量": "weighted_hit_total",
    "weighted_hit_share": "weighted_hit_share",
    "加权命中占比": "weighted_hit_share",
    "weighted_good_total": "weighted_good_total",
    "加权好样本总量": "weighted_good_total",
    "weighted_bad_total": "weighted_bad_total",
    "加权坏样本总量": "weighted_bad_total",
    "weighted_bad_rate": "weighted_bad_rate",
    "加权坏样本率": "weighted_bad_rate",
    "加权坏率": "weighted_bad_rate",
    "weighted_bad_capture_rate": "weighted_bad_capture_rate",
    "加权坏样本捕获率": "weighted_bad_capture_rate",
    "hit_amount": "hit_amount",
    "命中金额": "hit_amount",
    "hit_amount_share": "hit_amount_share",
    "命中金额占比": "hit_amount_share",
    "good_amount": "good_amount",
    "好样本金额": "good_amount",
    "bad_amount": "bad_amount",
    "坏样本金额": "bad_amount",
    "bad_amount_rate": "bad_amount_rate",
    "坏样本金额率": "bad_amount_rate",
    "bad_amount_capture_rate": "bad_amount_capture_rate",
    "坏样本金额捕获率": "bad_amount_capture_rate",
}
_VOTING_SEARCH_METRIC_TOKEN = "|".join(
    re.escape(alias)
    for alias in sorted(
        _VOTING_SEARCH_METRIC_ALIASES,
        key=lambda value: (-len(value), value),
    )
)
_VOTING_SEARCH_OBJECTIVE_PATTERNS = (
    re.compile(
        r"(?:目标|objective)\s*(?:为|是|=|:|：)?\s*"
        r"(?P<direction>最大化|最小化|maximi[sz]e|minimi[sz]e)\s*"
        rf"(?P<metric>{_VOTING_SEARCH_METRIC_TOKEN})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:目标|objective)\s*(?:为|是|=|:|：)?\s*"
        rf"(?P<metric>{_VOTING_SEARCH_METRIC_TOKEN})\s*"
        r"(?P<direction>最大化|最小化|maximi[sz]e|minimi[sz]e)",
        re.IGNORECASE,
    ),
)
_VOTING_SEARCH_CONSTRAINT_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<metric>{_VOTING_SEARCH_METRIC_TOKEN})\s*"
    r"(?P<operator>>=|<=|gte|lte|至少|不少于|不低于|至多|不超过|不高于)"
    r"\s*(?P<value>\d+(?:\.\d+)?%?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_SEARCH_INCLUDE_LABEL_RE = re.compile(
    r"(?:必须包含|包含|纳入|include(?:_rule_ids)?)"
    r"\s*(?:规则|rule(?:s|_ids)?)?\s*(?:=|:|：)?",
    re.IGNORECASE,
)
_VOTING_SEARCH_EXCLUDE_LABEL_RE = re.compile(
    r"(?:排除|剔除|去掉|exclude(?:_rule_ids)?)"
    r"\s*(?:规则|rule(?:s|_ids)?)?\s*(?:=|:|：)?",
    re.IGNORECASE,
)
_VOTING_SEARCH_NEGATED_LABEL_PREFIX_RE = re.compile(
    r"(?:不要|不用|无需|不需要|先不|暂不|别|禁止|不)\s*$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|must\s+not|never)\s*$",
    re.IGNORECASE,
)
_VOTING_SEARCH_MAX_COMBINATIONS_RE = re.compile(
    r"(?<![A-Za-z0-9_])max[_ -]?combinations"
    r"\s*(?:=|:|：|为)?\s*(?P<value>\d+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_BUILD_INTENT_RE = re.compile(
    r"(?:构建|生成|创建|测算|分析|评估|做)"
    r"[^，,；;。\n]{0,40}(?:投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?:投票|Voting|n[-_ ]?of[-_ ]?k)"
    r"[^，,；;。\n]{0,40}(?:构建|生成|创建|测算|分析|评估)|"
    r"(?<![A-Za-z0-9_])(?:build|create|generate|evaluate|analy[sz]e)"
    r"[^,;.!?\n]{0,40}(?:voting|n[-_ ]?of[-_ ]?k)"
    r"(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:voting|n[-_ ]?of[-_ ]?k)"
    r"[^,;.!?\n]{0,40}(?:build|create|generate|evaluate|analy[sz]e)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_NEGATED_BUILD_RE = re.compile(
    r"(?:不要|不用|无需|不需要|先不|暂不|取消|停止)"
    r"[^，,；;。\n]{0,24}(?:构建|生成|创建|测算|分析|评估|投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|cancel|stop)"
    r"[^,;.!?\n]{0,24}(?:build|create|generate|evaluate|voting|n[-_ ]?of[-_ ]?k)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_NONCOMMAND_RE = re.compile(
    r"[?？]|"
    r"(?:能否|可否|是否|可以吗|能不能|要不要|会不会|如何|怎么|怎样|"
    r"假设|假如|如果|若|万一|演示|示范|测试|举例|说明一下|解释|"
    r"介绍|描述|告诉我|展示)"
    r"[^；;。\n]{0,220}(?:构建|生成|创建|测算|分析|评估|投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?:昨天|昨日|之前|此前|过去|上次|前次|早些时候|曾经|历史上|"
    r"文档|报告|示例|例子|原文|材料)"
    r"[^；;。\n]{0,220}(?:构建|生成|创建|测算|投票|Voting|"
    r"n[-_ ]?of[-_ ]?k|candidate-rule-[0-9a-f]{32})|"
    r"(?:未来|将来|以后|稍后|晚点|回头|明天|后天|下周|下月|下个月|"
    r"月底|届时|[一二两三四五六七八九十百0-9]+天后)"
    r"[^；;。\n]{0,220}(?:构建|生成|创建|测算|投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?:等|待)[^；;。\n]{0,100}(?:后|之后|再|才)"
    r"[^；;。\n]{0,140}(?:构建|生成|创建|测算|投票|Voting|n[-_ ]?of[-_ ]?k)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|"
    r"is\s+it\s+possible|what\s+if|suppose|assuming|hypothetically|"
    r"how\s+to|demonstrate|demo|test|example|yesterday|previously|"
    r"earlier|last\s+time|in\s+the\s+future|later|tomorrow|"
    r"next\s+(?:week|month)|when|once|after)"
    r"[^;.!?\n]{0,220}(?:build|create|generate|evaluate|analy[sz]e|"
    r"voting|n[-_ ]?of[-_ ]?k)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_POSTPOSED_CANCELLATION_RE = re.compile(
    r"(?:^|[，,；;。.!?？！]\s*)(?:等等|等一下|算了|作罢|反悔了|"
    r"取消(?:吧|了)?|撤回|撤销|停止|先不做(?:了)?|暂不做(?:了)?|"
    r"别做(?:了)?|不要做(?:了)?|不执行(?:了)?)(?:[，,。.!！?？]?\s*)$|"
    r"(?:^|[,;.!?]\s*)(?:never\s+mind|forget\s+it|scratch\s+that|"
    r"cancel|abort|withdraw|stop|do(?:n't|\s+not)\s+(?:do|execute)\s+it)"
    r"(?:[,!.?]?\s*)$",
    re.IGNORECASE,
)
_VOTING_HEURISTIC_SELECTION_RE = re.compile(
    r"(?:最好|最优|最佳|最差|最坏|风险最高|坏率最高|表现最好|"
    r"自动(?:选择|挑选|推荐)|刚才(?:那些|这些|的)?|上述|这些规则|那些规则)|"
    r"(?<![A-Za-z0-9_])(?:best|worst|top[- ]?\d*|highest[- ]risk|"
    r"automatically\s+(?:select|pick|recommend)|those|these|previous)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_FOLLOW_UP_RE = re.compile(
    r"(?:加入|放入|写入)[^，,；;。\n]{0,16}(?:策略池|规则池|Pool)|"
    r"(?:入池|设置(?:业务)?动作|采纳|部署|上线|投产|写回|回写)|"
    r"(?:并|并且|然后|随后|再|同时|接着|直接)"
    r"(?![^，,；;。\n]{0,24}(?:比较|对比))"
    r"[^，,；;。\n]{0,40}(?:拒绝|审批|通过|复核)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"set\s+action|adopt|deploy|publish|write[- ]?back)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:and(?:\s+then)?|then|also)"
    r"(?![^,;.!?\n]{0,24}(?:compare|comparison))"
    r"[^,;.!?\n]{0,40}(?:reject|approve|review)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_OTHER_POOL_OPERATION_RE = re.compile(
    r"(?:删除|移除|重排|重新排序|排序|编译|预览)"
    r"[^，,；;。\n]{0,32}(?:策略池|规则池|pool|pool-entry-|candidate-rule-)|"
    r"(?:策略池|规则池|pool)"
    r"[^，,；;。\n]{0,32}(?:删除|移除|重排|重新排序|排序|编译|预览)|"
    r"(?<![A-Za-z0-9_])(?:remove|delete|reorder|sort|compile|preview)"
    r"[^,;.!?\n]{0,32}(?:strategy\s+pool|rule\s+pool|pool-entry-|candidate-rule-)|"
    r"(?<![A-Za-z0-9_])(?:strategy\s+pool|rule\s+pool|pool)"
    r"[^,;.!?\n]{0,32}(?:remove|delete|reorder|sort|compile|preview)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_NEGATED_CONTROL_RE = re.compile(
    r"(?:不要|不用|别用|不选|别选|排除|剔除|去掉|忽略|删除|移除)"
    r"[^，,；;。\n]{0,24}(?:candidate-rule-[0-9a-f]{32}|"
    r"n\s*(?:=|:|：|为)?\s*\d+)|"
    r"(?:candidate-rule-[0-9a-f]{32}|n\s*(?:=|:|：|为)?\s*\d+)"
    r"[^，,；;。\n]{0,16}(?:不要|不用|不选|排除|剔除|去掉|忽略)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|exclude|omit|remove|delete)"
    r"[^,;.!?\n]{0,24}(?:candidate-rule-[0-9a-f]{32}|"
    r"n\s*(?:=|:)?\s*\d+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VOTING_COMMAND_CLAUSE_RE = re.compile(r"[^；;。.!！?？\n]+")
_VOTING_COMMAND_RESET_RE = re.compile(
    r"(?:现在|本次|这次|当前|接下来|立即|马上|请|再|然后|随后)\s*$"
)
_VOTING_N_PATTERNS = (
    re.compile(
        r"(?:n|min[_ -]?hits?|阈值|最少命中数|至少命中|命中至少)\s*"
        r"(?:=|:|：|为)?\s*(?P<n>\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<n>\d+)\s*(?:-|/|\s)\s*(?:of|OF)\s*(?P<k>\d+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?P<k>\d+)\s*(?:个)?\s*选\s*(?P<n>\d+)"),
    re.compile(
        r"(?:至少|最少)\s*(?:命中)?\s*(?P<n>\d+)\s*(?:个|条|项)?",
        re.IGNORECASE,
    ),
)
_VOTING_SEARCH_METRICS = frozenset(_VOTING_SEARCH_METRIC_ALIASES.values())
_VOTING_SEARCH_RATE_METRICS = frozenset(
    {
        "hit_share",
        "bad_rate",
        "bad_capture_rate",
        "weighted_hit_share",
        "weighted_bad_rate",
        "weighted_bad_capture_rate",
        "hit_amount_share",
        "bad_amount_rate",
        "bad_amount_capture_rate",
    }
)
_VOTING_SEARCH_REQUIRED_MINIMUM_SHARE = {
    "bad_rate": "hit_share",
    "weighted_bad_rate": "weighted_hit_share",
    "bad_amount_rate": "hit_amount_share",
}
_STRATEGY_POOL_WORKFLOWS = frozenset(
    {
        "strategy_pool_add_candidate",
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
        "strategy_pool_reorder",
        "strategy_pool_compile",
    }
)
_STRATEGY_POOL_MEASUREMENT_WORKFLOWS = frozenset({"strategy_pool_impact"})
_STRATEGY_POOL_APPLY_WORKFLOWS = frozenset({"strategy_pool_apply"})
_STRATEGY_POOL_MATERIALIZE_WORKFLOWS = frozenset(
    {"strategy_pool_materialize"}
)
_STRATEGY_POOL_VALIDATION_WORKFLOWS = frozenset(
    {"strategy_pool_validation"}
)
_STRATEGY_POOL_STABILITY_WORKFLOWS = frozenset(
    {"strategy_pool_stability"}
)
_POOL_MUTATION_WORKFLOWS = _STRATEGY_POOL_WORKFLOWS - {"strategy_pool_compile"}
_POOL_ACTION_TYPES = {
    "approval": frozenset({"approval", "reject", "review"}),
    "reject": frozenset({"approval", "reject", "review"}),
    "limit": frozenset({"limit"}),
    "pricing": frozenset({"pricing"}),
    "segmentation": frozenset({"segment"}),
}
_POOL_ADD_PLACEMENT_MODES = frozenset(
    {"before_selected_members", "replace_selected_members"}
)
_POOL_ACTION_GROUNDING = {
    "approval": re.compile(
        r"(?:通过|批准|准入)|(?<![A-Za-z0-9_])(?:approve|approval)"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"拒绝|(?<![A-Za-z0-9_])reject(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "review": re.compile(
        r"(?:人工复核|人工审核|复核|审核)|"
        r"(?<![A-Za-z0-9_])review(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"(?:额度|授信)|(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"(?:定价|利率)|(?<![A-Za-z0-9_])(?:pricing|price)"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "segment": re.compile(
        r"(?:分群|分层)|(?<![A-Za-z0-9_])segment(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}
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
_POOL_STRATEGY_TYPE_VALUE_GROUNDING = {
    "approval": re.compile(r"(?<![A-Za-z0-9_])approval(?![A-Za-z0-9_])|(?:审批|准入)"),
    "reject": re.compile(r"(?<![A-Za-z0-9_])reject(?![A-Za-z0-9_])|拒绝"),
    "limit": re.compile(r"(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])|(?:额度|授信)"),
    "pricing": re.compile(r"(?<![A-Za-z0-9_])pricing(?![A-Za-z0-9_])|(?:定价|利率)"),
    "segmentation": re.compile(
        r"(?<![A-Za-z0-9_])segmentation(?![A-Za-z0-9_])|(?:分群|分层)"
    ),
}
_POOL_APPLY_TARGET_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:当前样本|当前数据|current\s+(?:sample|dataset)))"
    r"(?=.*(?:应用|写回|回写|回填|打标|"
    r"(?<![A-Za-z0-9_])(?:apply|write[-\s]*back|assign)(?![A-Za-z0-9_])))",
    re.IGNORECASE,
)
_POOL_APPLY_POSITIVE_INTENT_RE = re.compile(
    r"(?:应用|写回|回写|回填|打标)"
    r"[^；;。.!?？\n]{0,180}(?:策略池|规则池|当前样本|当前数据)|"
    r"(?:策略池|规则池)[^；;。.!?？\n]{0,180}"
    r"(?:应用|写回|回写|回填|打标)|"
    r"(?<![A-Za-z0-9_])(?:apply|write[-\s]*back|assign)"
    r"[^;.!?\n]{0,180}(?:pool|current\s+(?:sample|dataset))"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_APPLY_NONCURRENT_RE = re.compile(
    r"[?？]|(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止|"
    r"能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|假如|如果|"
    r"以后|未来|将来|稍后|之前|此前|过去|上次)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+you|"
    r"could\s+you|how\s+to|what\s+if|later|previously|"
    r"in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_APPLY_SECOND_OPERATION_RE = re.compile(
    r"(?:采纳|采用|部署|上线|投产|生效|激活|切换|导出|下载|"
    r"加入|添加|入池|删除|移除|改动作|修改动作|重排|排序|编译|"
    r"修改策略池|改(?:一下)?(?:策略池|规则池)|生成报告|形成报告|出报告)|"
    r"(?<![A-Za-z0-9_])(?:adopt|deploy|promote|activate|switch|export|"
    r"download|add|insert|remove|delete|reorder|compile|modify|"
    r"generate\s+(?:a\s+)?report)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_APPLY_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:expected_pool_revision|"
    r"expected_pool_snapshot_hash|pool_(?:id|artifact_id)|"
    r"artifact_(?:id|hash)|dataset_(?:id|content_hash)|sample_design_ref|"
    r"requirements(?:_hash)?|strategy_spec|design_hash|action_counts|"
    r"activated|adopted|deployed)(?![A-Za-z0-9_])|"
    r"(?:Pool|策略池|数据集|dataset|artifact|工件|产物)\s*(?:hash|哈希|revision|版本)",
    re.IGNORECASE,
)
_POOL_MATERIALIZE_TARGET_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:物化|固化|创建|生成|"
    r"(?<![A-Za-z0-9_])(?:materialize|create)(?![A-Za-z0-9_])))"
    r"(?=.*(?:草案策略|策略草案|draft\s+strategy|strategy\s+draft))",
    re.IGNORECASE,
)
_POOL_MATERIALIZE_POSITIVE_INTENT_RE = re.compile(
    r"(?:物化|固化|创建|生成)[^；;。.!?？\n]{0,180}"
    r"(?:草案策略|策略草案|draft\s+strategy|strategy\s+draft)|"
    r"(?<![A-Za-z0-9_])(?:materialize|create)[^;.!?\n]{0,180}"
    r"(?:draft\s+strategy|strategy\s+draft)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_MATERIALIZE_NONCURRENT_RE = _POOL_APPLY_NONCURRENT_RE
_POOL_MATERIALIZE_SECOND_OPERATION_RE = re.compile(
    r"(?:采纳|采用|部署|上线|投产|生效|回测|测试|验证|应用|写回|"
    r"生成报告|形成报告|出报告|监控|漂移|导出|下载)|"
    r"(?<![A-Za-z0-9_])(?:adopt|deploy|promote|backtest|test|validate|"
    r"apply|write[-\s]*back|report|monitor|export|download)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_MATERIALIZE_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:expected_pool_revision|"
    r"expected_pool_snapshot_hash|expected_pool_artifact_id|"
    r"expected_pool_artifact_content_hash|expected_design_hash|"
    r"pool_(?:id|artifact_id)|artifact_(?:id|hash)|strategy_spec|"
    r"requirements?|metrics?|design_hash)(?![A-Za-z0-9_])|"
    r"(?:Pool|策略池|artifact|工件|产物|design|设计)\s*"
    r"(?:hash|哈希|revision|版本)",
    re.IGNORECASE,
)
_POOL_MATERIALIZE_NEGATED_LIFECYCLE_DISCLAIMER_RE = re.compile(
    r"(?:，|,|；|;)\s*(?:"
    r"(?:不要|不用|无需|不需要|不得|不会)\s*"
    r"(?:采纳|采用|部署|上线|投产)"
    r"(?:\s*(?:或|和|、|以及|并且?)\s*(?:采纳|采用|部署|上线|投产))*"
    r"|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|without)\s+"
    r"(?:adopt|deploy|promote)"
    r"(?:\s*(?:or|and|,)\s*(?:adopt|deploy|promote))*"
    r")\s*[。.!]?\s*$",
    re.IGNORECASE,
)
_POOL_APPLY_OUTPUT_PREFIX_LABEL_RE = re.compile(
    r"(?:(?:输出|字段|列名)\s*前缀|output_prefix|output\s+prefix|prefix)"
    r"\s*(?:为|是|设为|设置为|=|:|：)?",
    re.IGNORECASE,
)
_POOL_APPLY_OUTPUT_PREFIX_RE = re.compile(
    _POOL_APPLY_OUTPUT_PREFIX_LABEL_RE.pattern
    + r"\s*(?P<prefix>[A-Za-z_][A-Za-z0-9_]{0,47})"
    r"(?![A-Za-z0-9_./-])",
    re.IGNORECASE,
)
_POOL_APPLY_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,47}$")
_POOL_VALIDATION_TARGET_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?:(?=.*(?:独立样本|独立回放|回放验证|独立验证|"
    r"independent(?:ly)?\s+(?:(?:sample\s+)?replay|validat(?:e|ed|ion))|"
    r"replay\s+validation))|"
    r"(?=.*(?:验证集|验证样本|验证分区|时间外样本|时间外验证|时间外分区|"
    r"(?<![A-Za-z0-9_])(?:validation|oot)(?![A-Za-z0-9_])))"
    r"(?=.*(?:验证|回放|(?<![A-Za-z0-9_])(?:validate|replay)"
    r"(?![A-Za-z0-9_]))))",
    re.IGNORECASE,
)
_POOL_VALIDATION_POSITIVE_INTENT_RE = re.compile(
    r"(?:执行|运行|开展|进行|做|验证|回放)"
    r"[^；;。.!?？\n]{0,160}(?:独立样本|独立回放|回放验证|策略池|规则池)|"
    r"(?:独立样本|独立回放|回放验证|策略池|规则池)"
    r"[^；;。.!?？\n]{0,160}(?:执行|运行|开展|进行|验证|回放)|"
    r"(?<![A-Za-z0-9_])(?:run|perform|execute|validate|replay)"
    r"[^;.!?\n]{0,160}(?:independent|replay|validation|oot|pool)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_VALIDATION_NONCURRENT_RE = re.compile(
    r"[?？]|(?:吗|呢)\s*$|(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止|"
    r"能否|可否|是否|可以吗|能不能|可不可以|该不该|要不要|需不需要|"
    r"如何|怎么|怎样|假设|假如|如果|"
    r"以后|未来|将来|稍后|晚点|回头|等会儿|待会儿|一会儿|"
    r"明天|明早|今晚|后天|下次|下周|下月|下个月|月底|届时|"
    r"之前|此前|过去|上次|上一版|上个版本|历史上|曾经|曾|做过|"
    r"已完成)|"
    r"(?:执行|运行|开展|进行|做|验证|回放|完成)(?:了|过)|"
    r"(?:等|待|样本|数据|材料|审核|审批|评审|确认)"
    r"[^；;。.!?？\n]{0,24}(?:后|之后)|"
    r"已经[^；;。.!?？\n]{0,80}(?:完成|做过|验证过|回放过)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+(?:you|we)|"
    r"could\s+(?:you|we)|would\s+you|should\s+(?:we|i)|"
    r"do\s+we\s+need(?:\s+to)?|"
    r"is\s+it\s+possible|"
    r"tell\s+me\s+whether|how\s+to|what\s+if|later|previously|"
    r"in\s+the\s+future|tomorrow|next\s+time)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:was|has\s+been)\b"
    r"[^;.!?\n]{0,80}\b(?:validated|replayed|completed)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_VALIDATION_SECOND_OPERATION_RE = re.compile(
    r"(?:加入|添加|入池|删除|移除|改动作|修改动作|重排|排序|编译|"
    r"修改策略池|应用|写回|回写|回填|打标|生成报告|形成报告|出报告|"
    r"采纳|采用|晋级|提升为|部署|上线|投产|生效|激活|切换)|"
    r"(?<![A-Za-z0-9_])(?:add|insert|remove|delete|reorder|compile|"
    r"apply|write[-\s]*back|report|adopt|promote|deploy|activate|switch)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_VALIDATION_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:pool_ref|sample_design_ref|population|"
    r"comparison_mode|expected_pool_revision|expected_pool_snapshot_hash|"
    r"pool_(?:id|artifact_id)|artifact_(?:id|hash)|"
    r"dataset_(?:id|content_hash)|workspace_revision|target_col|"
    r"requirements(?:_hash)?|metrics?|validation_status)"
    r"(?![A-Za-z0-9_])|"
    r"(?:Pool|策略池|数据集|dataset|artifact|工件|产物)"
    r"\s*(?:hash|哈希|revision|版本)",
    re.IGNORECASE,
)
_POOL_VALIDATION_EVIDENCE_SCOPE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:psi|stability|drift)(?![A-Za-z0-9_])|"
    r"(?:稳定性|漂移)",
    re.IGNORECASE,
)
_POOL_VALIDATION_PARTITION_GROUNDING = {
    "validation": re.compile(
        r"(?:验证集|验证样本|验证分区)|"
        r"(?:(?<![A-Za-z0-9_])(?:on|in)\s+validation"
        r"(?![A-Za-z0-9_])|"
        r"(?<![A-Za-z0-9_])validation"
        r"(?=\s*(?:上|中|里|partition|sample|set|独立样本|独立回放)))",
        re.IGNORECASE,
    ),
    "oot": re.compile(
        r"(?:时间外样本|时间外验证|时间外分区)|"
        r"(?<![A-Za-z0-9_])oot(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "development": re.compile(
        r"(?:开发集|开发样本|开发分区)|"
        r"(?<![A-Za-z0-9_])development(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}
_POOL_STABILITY_TARGET_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:跨(?:分区|样本)(?:分布)?(?:稳定性|漂移)?|"
    r"分布(?:稳定性|漂移)|稳定性|漂移|"
    r"(?<![A-Za-z0-9_])psi(?![A-Za-z0-9_])|"
    r"cross[-\s]*partition\s+stability))",
    re.IGNORECASE,
)
_POOL_STABILITY_POSITIVE_INTENT_RE = re.compile(
    r"(?:测量|测算|分析|计算|评估|检查)"
    r"[^；;。.!?？\n]{0,160}(?:稳定性|漂移|PSI|策略池|规则池)|"
    r"(?:策略池|规则池)[^；;。.!?？\n]{0,160}"
    r"(?:测量|测算|分析|计算|评估|检查)|"
    r"(?<![A-Za-z0-9_])(?:measure|calculate|analy[sz]e|assess|evaluate|check)"
    r"[^;.!?\n]{0,160}(?:stability|drift|psi|pool)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_STABILITY_NONCURRENT_RE = re.compile(
    r"[?？]|(?:吗|呢)\s*$|(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止|"
    r"能否|可否|是否|可以吗|能不能|可不可以|该不该|要不要|需不需要|"
    r"如何|怎么|怎样|假设|假如|如果|以后|未来|将来|稍后|"
    r"明天|下次|之前|此前|过去|上次|上一版|历史上|曾经|昨天)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|can\s+(?:you|we)|"
    r"could\s+(?:you|we)|would\s+you|should\s+(?:we|i)|how\s+to|"
    r"what\s+if|later|previously|historically|yesterday|tomorrow|"
    r"in\s+the\s+future)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_STABILITY_SECOND_OPERATION_RE = re.compile(
    r"(?:加入|添加|入池|删除|移除|改动作|修改动作|重排|排序|编译|"
    r"应用|写回|回写|回填|打标|生成报告|形成报告|出报告|"
    r"创建策略|采纳|采用|晋级|提升为|部署|上线|投产|生效|激活|切换)|"
    r"(?<![A-Za-z0-9_])(?:add|insert|remove|delete|reorder|compile|"
    r"apply|write[-\s]*back|report|create\s+strategy|adopt|promote|"
    r"deploy|activate|switch)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_STABILITY_PLATFORM_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:impact_cube_ref|pool_stability_ref|pool_ref|"
    r"sample_design_ref|artifact_id|content_hash|expected_[a-z0-9_]+|"
    r"pool_(?:id|revision|snapshot_hash)|dataset_(?:id|content_hash)|"
    r"workspace_revision|target_col|metrics?|psi_threshold|thresholds?)"
    r"(?![A-Za-z0-9_])|"
    r"(?:Pool|策略池|数据集|dataset|artifact|工件|产物)"
    r"\s*(?:hash|哈希|revision|版本)|"
    r"(?:PSI|稳定性|漂移)\s*(?:阈值|threshold)\s*(?:=|:|：)?\s*[-+]?\d",
    re.IGNORECASE,
)
_POOL_IMPACT_TARGET_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:影响|效果|瀑布|逐月|通过率|坏账率|风险率|测算|评估|回测|"
    r"impact|effect|waterfall|monthly|approval\s+rate|bad\s+rate|risk\s+rate|"
    r"measure|assess|evaluat|backtest))",
    re.IGNORECASE,
)
_POOL_IMPACT_POSITIVE_INTENT_RE = re.compile(
    r"(?:测算|评估|分析|回测|计算|查看|看一下|看下)"
    r"[^；;。.!?？\n]{0,160}?(?:影响|效果|瀑布|逐月|策略池|规则池)|"
    r"(?:策略池|规则池)[^；;。.!?？\n]{0,160}?"
    r"(?:测算|评估|分析|回测|计算|查看|看一下|看下)|"
    r"(?<![A-Za-z0-9_])(?:measure|assess|evaluate|analy[sz]e|calculate|backtest)"
    r"[^;.!?\n]{0,160}?(?:impact|effect|waterfall|monthly|pool)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_NEGATED_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|取消|停止|禁止)"
    r"[^；;。.!?？\n]{0,64}(?:测算|评估|分析|回测|影响|效果)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cancel|stop)"
    r"[^;.!?\n]{0,64}(?:measure|assess|evaluate|analy[sz]e|calculate|backtest)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能|如何|怎么|怎样|假设|假如|"
    r"如果|若|演示|示范|举例|说明|解释|介绍|昨天|之前|此前|过去|上次|"
    r"曾经|历史上|未来|以后|稍后|明天|下周|下月)"
    r"[^；;。\n]{0,180}(?:策略池|规则池|影响|效果|测算|评估|回测)|"
    r"(?<![A-Za-z0-9_])(?:can\s+you|could\s+you|would\s+you|what\s+if|"
    r"how\s+to|example|demo|previously|yesterday|historically|later|tomorrow)"
    r"[^;.!?\n]{0,180}(?:pool|impact|effect|measure|assess|backtest)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_REPORT_ONLY_RE = re.compile(
    r"(?:只|仅)\s*(?:生成|出|写|整理|汇总)?\s*(?:报告|文档|汇报|总结)|"
    r"(?:生成|出|写|整理|汇总|制作|导出|下载)"
    r"[^；;。.!?？\n]{0,32}(?:报告|文档|汇报|总结)|"
    r"(?:报告|文档|汇报|总结)"
    r"[^；;。.!?？\n]{0,24}(?:生成|制作|导出|下载)|"
    r"(?:报告|文档|汇报|总结)\s*(?:即可|就行|only)|"
    r"(?<![A-Za-z0-9_])(?:generate|create|write|export|download)"
    r"[^;.!?\n]{0,32}(?:report|document|summary)|"
    r"(?<![A-Za-z0-9_])(?:report|document|summary)\s+only"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_SECOND_OPERATION_RE = re.compile(
    r"(?:加入|添加|入池|删除|移除|改动作|修改动作|重排|排序|编译|预览|"
    r"采纳|采用|部署|上线|投产|生效|写回|回写|创建策略|生成策略|"
    r"Vintage|迁徙率|迁徙矩阵|利润|收益|导出|下载)|"
    r"(?<![A-Za-z0-9_])(?:add|insert|remove|delete|reorder|compile|preview|"
    r"adopt|deploy|promote|activate|write[-\s]*back|create\s+strategy|"
    r"vintage|roll[-\s]*rate|profit|export|download)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_BASELINE_MODE_RE = re.compile(
    r"(?:相对|相较(?:于)?|相比(?:于)?|对比|比较| versus |\bvs\.?\b)"
    r"[^；;。.!?？\n]{0,80}(?:基线|baseline|strategy[-_A-Za-z0-9]+)|"
    r"(?:基线|baseline)[^；;。.!?？\n]{0,80}(?:对比|比较|影响|效果|vs)",
    re.IGNORECASE,
)
_POOL_IMPACT_ABSOLUTE_MODE_RE = re.compile(
    r"(?:绝对(?:效果|影响|口径)?|不(?:做|要|用)?(?:基线)?(?:对比|比较)|"
    r"无需(?:基线)?(?:对比|比较))|"
    r"(?<![A-Za-z0-9_])absolute(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:without|no)\s+baseline(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_STRATEGY_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])strategy-[A-Za-z0-9][A-Za-z0-9_-]*"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_DROP_NAN_TRUE_RE = re.compile(
    r"(?:明确|确认|允许|同意)?\s*(?:丢弃|排除|剔除|删除)"
    r"[^；;。.!?？\n]{0,24}(?:NaN|nan|空标签|缺失标签|无效标签)|"
    r"drop[_\s-]*nan[_\s-]*labels?\s*(?:=|:)?\s*true|"
    r"(?<![A-Za-z0-9_])(?:drop|exclude)[^;.!?\n]{0,24}"
    r"(?:nan|missing)\s+labels?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_IMPACT_DROP_NAN_NEGATED_RE = re.compile(
    r"(?:不要|不|禁止|拒绝|不同意|未授权|不能)\s*"
    r"(?:允许|确认|同意)?\s*(?:丢弃|排除|剔除|删除)"
    r"[^；;。.!?？\n]{0,24}(?:NaN|nan|空标签|缺失标签|无效标签)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+(?:drop|exclude)"
    r"[^;.!?\n]{0,24}(?:nan|missing)\s+labels?(?![A-Za-z0-9_])|"
    r"drop[_\s-]*nan[_\s-]*labels?\s*(?:=|:)?\s*false",
    re.IGNORECASE,
)
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
_IMPACT_CUBE_ECONOMICS_COMPONENTS = {
    "approval": frozenset(
        {
            "ead",
            "pd",
            "annual_rate",
            "funding_rate",
            "lgd",
            "operating_cost_per_loan",
            "term_months",
        }
    ),
    "reject": frozenset(
        {
            "ead",
            "pd",
            "annual_rate",
            "funding_rate",
            "lgd",
            "operating_cost_per_loan",
            "term_months",
        }
    ),
    "limit": frozenset({"pd", "lgd", "utilization"}),
    "pricing": frozenset(
        {
            "ead",
            "pd",
            "lgd",
            "funding_rate",
            "term_months",
            "operating_cost_per_loan",
        }
    ),
    "segmentation": frozenset(),
}
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
_POOL_PARTIAL_REORDER_RE = re.compile(
    r"(?:放|移|挪|排|调)(?:到|至|在)?(?:前面|后面|最前|末尾|最后|"
    r"第[一二三四五六七八九十百0-9]+(?:位|个|条))|"
    r"(?:上移|下移)[一二三四五六七八九十百0-9]+(?:位|个|条)|"
    r"置顶|提前|排(?:在)?(?:最前|第一|末尾|最后)|优先放|"
    r"(?:交换|互换)[^；;。\n]{0,200}(?:顺序|位置)|"
    r"move\s+.*\s+(?:first|last|(?:to\s+)?(?:position\s+\d+|"
    r"(?:second|third|fourth)\s+(?:place|position)))|"
    r"swap\s+.*(?:order|position)",
    re.IGNORECASE,
)
_POOL_HEURISTIC_REORDER_RE = re.compile(
    r"(?:按.{0,12}(?:效果|坏率|lift|最好|最优|风险).{0,8}(?:排序|重排)|"
    r"(?:自动|智能).{0,8}(?:排序|重排)|sort.{0,12}(?:best|effect|risk|lift))",
    re.IGNORECASE,
)
_POOL_ADD_INTENT_RE = re.compile(
    r"(?:加入|添加到?|写入|放入|加到|放进|纳入|写到|新增到)"
    r"[^，,；;。\n]{0,160}(?:策略池|规则池|(?<![A-Za-z0-9_])"
    r"(?:strategy\s+)?pool(?![A-Za-z0-9_]))|"
    r"(?:入池)(?!理由|原因|说明)|"
    r"(?<![A-Za-z0-9_])(?:add|append|insert|write)\b"
    r"[^,;.!?\n]{0,160}\bto\s+(?:the\s+)?"
    r"(?:(?:approval|reject|limit|pricing|segmentation)\s+)?"
    r"(?:(?:strategy|rule)\s+)?pool\b|"
    r"(?<![A-Za-z0-9_])put\b[^,;.!?\n]{0,160}\binto\s+(?:the\s+)?"
    r"(?:(?:approval|reject|limit|pricing|segmentation)\s+)?"
    r"(?:(?:strategy|rule)\s+)?pool\b",
    re.IGNORECASE,
)
_POOL_ADD_HYPOTHETICAL_RE = re.compile(
    r"[?？]|"
    r"(?:假设|假如|如果|若)\s*[^；;。\n]{0,180}(?:加入|添加|放进|纳入|入池)|"
    r"(?:如何|怎么|怎样|请说明|说明一下|演示一下|示范一下|测试一下|举例)"
    r"[^；;。\n]{0,180}(?:加入|添加|放进|纳入|入池)|"
    r"(?:文档|说明|示例|例子|原文|材料|报告)[^；;。\n]{0,80}"
    r"(?:写着|提到|说|包含|展示)[^；;。\n]{0,180}"
    r"(?:加入|添加|放进|纳入|入池)|"
    r"(?:文档|说明|示例|例子|原文|材料|报告)[^；;。\n]{0,120}"
    r"(?:默认动作|命中动作|default\s+action|hit\s+action)|"
    r"[“\"'‘][^”\"'’；;。\n]{0,240}(?:加入|添加|放进|纳入|入池)"
    r"[^”\"'’；;。\n]{0,240}[”\"'’]|"
    r"[“\"'‘][^”\"'’；;。\n]{0,240}"
    r"(?:默认动作|命中动作|default\s+action|hit\s+action)"
    r"[^”\"'’；;。\n]{0,240}[”\"'’]|"
    r"(?:不要|不用|别|请勿|排除|忽略)[^；;。\n]{0,48}"
    r"(?:(?:这个|该)\s*(?:source|ID|id|来源)|来源)|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never)\s+use\s+"
    r"(?:this\s+|that\s+)?(?:source|id)(?![A-Za-z0-9_])|"
    r"(?:只|仅)\s*(?:告诉|说明|解释|展示|描述)[^；;。\n]{0,180}"
    r"(?:加入|入池|策略池)|"
    r"(?:评估|分析|了解|看看|查看|解释|模拟(?:一下)?)[^；;。\n]{0,180}"
    r"(?:加入|入池|策略池)|"
    r"(?:能否|可否|是否可以|可以吗|能不能)[^；;。\n]{0,180}"
    r"(?:加入|入池|策略池)|"
    r"(?:昨天|昨日|之前|此前|过去|上次|前次|早些时候|曾经|已经)"
    r"[^；;。\n]{0,180}(?:加入|添加|放进|纳入|入池)|"
    r"(?:改写|重写|润色|翻译|复述)[^；;。\n]{0,120}"
    r"(?:这句|这句话|下句|以下|文本|文案|内容)|"
    r"(?:改写|重写|润色|翻译|复述)(?:得|成|为|一下)|"
    r"(?:无法|未能|没能)[^；;。\n]{0,80}(?:加入|添加|放进|纳入|入池)|"
    r"(?:加入|添加|放进|纳入|入池)(?:不了|失败|不进去|不上)|"
    r"(?:未来|将来|以后|之后|稍后|晚点|回头|明天|明早|今晚|后天|"
    r"下周|下月|下个月|月底|届时|[一二两三四五六七八九十百0-9]+天后)"
    r"[^；;。\n]{0,180}"
    r"(?:加入|入池|策略池)|"
    r"(?:等|待)?(?:审批|审核|评审|批准|确认)"
    r"(?:通过|完成|同意|批准)?(?:后|之后|就|再|才)"
    r"[^；;。\n]{0,180}(?:加入|添加|放进|纳入|入池)|"
    r"(?:等|待)[^；;。\n]{0,100}(?:再|才)[^；;。\n]{0,100}"
    r"(?:加入|入池|策略池)|"
    r"(?:加入|添加|放进|纳入|入池)[^；;。\n]{0,100}"
    r"(?:不允许|禁止|不可|不能执行)|"
    r"(?:会发生什么|会怎样|将会怎样)|"
    r"(?<![A-Za-z0-9_])(?:what\s+if|suppose|assuming|hypothetically|"
    r"can\s+you|could\s+you|would\s+you|is\s+it\s+possible|"
    r"evaluate|analy[sz]e|explain|how\s+to|demonstrate|demo|test|"
    r"show\s+me\s+what\s+happens|documentation\s+says|example|"
    r"yesterday|previously|earlier|last\s+time|failed\s+to|"
    r"unable\s+to|could\s+not|couldn't|rewrite|rephrase|translate|"
    r"in\s+the\s+future|later|tomorrow|next\s+(?:week|month)|"
    r"after\s+approval|when|once)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_POSTPOSED_CANCELLATION_RE = re.compile(
    r"(?:^|[，,；;。.!?？！]\s*)(?:等等|等一下|停一下|先等等)"
    r"[，,]?\s*(?:还是\s*)?(?:(?:先\s*)?(?:不要了|不加(?:了)?|"
    r"不入(?:了)?|别加(?:了)?|别入(?:了)?))"
    r"(?:[，,。.!！?？]?\s*(?:谢谢(?:你)?|多谢|辛苦了))?\s*$|"
    r"(?:^|[，,；;。.!?？！]\s*)(?:我\s*)?反悔了"
    r"(?:[，,。.!！?？]?\s*(?:谢谢(?:你)?|多谢|辛苦了))?\s*$|"
    r"(?:^|[，,；;。.!?？！]\s*)(?:等等|等一下|先等等|不[，,]?\s*)?"
    r"(?:(?:刚才|前面)(?:那句|请求|操作)?\s*)?"
    r"(?:算了|作废|撤回|不加了|不入了|不做了|别做了|先不弄了|"
    r"取消(?:入池|操作|执行)?(?:吧)?|撤销(?:入池|操作|执行)?|"
    r"不要(?:入池|执行|操作|做了)|先别(?:入池|执行|操作))"
    r"(?:[，,]\s*(?:这次|本次)?\s*(?:先不做了|别做了|不执行了))?"
    r"(?:[，,。.!！?？]?\s*(?:谢谢(?:你)?|多谢|辛苦了|麻烦(?:你)?了))?\s*$|"
    r"(?:^|[,;.!?]\s*)(?:actually\s+)?(?:no|never\s+mind|forget\s+it|"
    r"scratch\s+that|abort|withdraw|stop|cancel\s+(?:that|it)|"
    r"do(?:n't|\s+not)\s+(?:do|execute)\s+(?:that|it))"
    r"(?:[,!.?]?\s*(?:thanks(?:\s+a\s+lot)?|thank\s+you))?\s*$",
    re.IGNORECASE,
)
_POOL_ADD_LIFECYCLE_RE = re.compile(
    r"(?:采纳|采用|部署|上线|投产|投入生产|上生产|投用|发布到?生产|"
    r"发布到?线上|推到?线上|推生产|正式运行|落地执行|立即执行|执行它?|"
    r"投入使用|开始使用|启用|生效|激活)|"
    r"(?<![A-Za-z0-9_])(?:adopt(?:s|ed|ing)?|deploy(?:s|ed|ing)?|"
    r"promot(?:e|es|ed|ing)|activat(?:e|es|ed|ing)|"
    r"enabl(?:e|es|ed|ing)|ship(?:s|ped|ping)?|"
    r"push(?:es|ed|ing)?|releas(?:e|es|ed|ing)|"
    r"publish(?:es|ed|ing)?|launch(?:es|ed|ing)?|"
    r"productioniz(?:e|es|ed|ing)|execut(?:e|es|ed|ing)|run(?:s|ning)?|"
    r"use(?:s|d|ing)?[^;.!?\n]{0,32}(?:in|on)[-\s]+prod(?:uction)?|"
    r"put[^;.!?\n]{0,32}into[-\s]+prod(?:uction)?|"
    r"enter(?:s|ed|ing)?[-\s]+prod(?:uction)?|take[^;.!?\n]{0,20}live|"
    r"(?:go(?:es|ing)?|went)[-\s]+live|roll(?:s|ed|ing)?[-\s]+out)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_SECOND_OPERATION_RE = re.compile(
    r"(?:删除|删掉|删去|删了|移除|撤下|撤掉|撤回|去掉|去除|清空|"
    r"踢出|拿掉|剔除)"
    r"[^；;。\n]{0,100}"
    r"(?:pool-entry|candidate-rule|"
    r"策略池|规则池|条目|规则|它|这个|该项)|"
    r"(?:pool-entry|candidate-rule|策略池|规则池|条目|规则|它|这个|该项)"
    r"[^；;。\n]{0,100}(?:删除|删掉|删去|删了|移除|撤下|撤掉|撤回|"
    r"去掉|去除|清空|踢出|拿掉|剔除)|"
    r"(?:动作)[^；;。\n]{0,32}(?:改为|改成|设置为|设置成|设为|设成|"
    r"置为|置成|切换为|切换成|修改为)|"
    r"(?:pool-entry|candidate-rule|条目|规则|它|这个|该项)[^；;。\n]{0,80}"
    r"(?:改为|改成|设置为|设置成|设为|设成|置为|置成|切换为|切换成|"
    r"修改为)|"
    r"(?:重新|再次|然后|随后|接着|再)\s*(?:把[^；;。\n]{0,48})?"
    r"(?:改为|改成|设置为|设置成|设为|设成|置为|置成|切换为|切换成)"
    r"\s*(?:approval|reject|review|limit|pricing|segment|通过|拒绝|复核)|"
    r"(?:调整|修改|变更)[^；;。\n]{0,80}(?:为|成)\s*"
    r"(?:approval|reject|review|limit|pricing|segment|通过|拒绝|复核)|"
    r"(?:完整)?(?:重排|排序)[^；;。\n]{0,100}(?:策略池|规则池)|"
    r"(?:编译|预览)[^；;。\n]{0,80}(?:策略池|规则池)|"
    r"(?:策略池|规则池)[^；;。\n]{0,80}(?:编译|预览)|"
    r"(?:重新|再次|然后|随后|再)\s*(?:编译|预览)|"
    r"(?:^|[，,；;。.!?？！]\s*)(?:(?:然后|随后|接着|再|并且|同时|"
    r"完成后)\s*)?(?:立即\s*)?(?:回测|测算(?:效果|影响)?|"
    r"应用到?(?:当前)?样本|生成(?:效果|策略|分析)?报告|形成文档|"
    r"提交(?:审批|审核|评审)|发起(?:审批|审核|评审)|送审)|"
    r"(?:^|[,;.!?]\s*)(?:(?:then|next|afterwards|and)\s+)?"
    r"(?:immediately\s+)?(?:backtest|apply[^;.!?\n]{0,40}(?:sample|dataset)|"
    r"generate[^;.!?\n]{0,40}report|submit[^;.!?\n]{0,40}(?:approval|review))|"
    r"(?<![A-Za-z0-9_])(?:remove|delete)\b[^;.!?\n]{0,100}"
    r"(?:pool-entry|candidate-rule|pool|entry|rule)|"
    r"(?<![A-Za-z0-9_])(?:set|change|update|make)\b[^;.!?\n]{0,80}"
    r"(?:\baction\b|\b(?:approval|reject|review|limit|pricing|segment)\b)|"
    r"(?<![A-Za-z0-9_])(?:reorder|sort|compile|preview)\b"
    r"[^;.!?\n]{0,100}\bpool\b",
    re.IGNORECASE,
)
_POOL_MUTATION_INTENT_PATTERNS = {
    "strategy_pool_remove_entry": re.compile(
        r"(?:删除|删掉|删去|移除|撤下|撤掉|去掉|去除|踢出|拿掉|剔除)|"
        r"(?<![A-Za-z0-9_])(?:remove|delete)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "strategy_pool_set_action": re.compile(
        r"(?:动作[^；;。\n]{0,32})?(?:改为|改成|设置为|设置成|设为|设成|"
        r"置为|置成|切换为|切换成|修改为)|"
        r"(?<![A-Za-z0-9_])(?:set|change|update)\b[^;.!?\n]{0,80}\baction\b",
        re.IGNORECASE,
    ),
    "strategy_pool_reorder": re.compile(
        r"(?:(?:按)?完整(?:顺序)?\s*)?(?:重排|排序)|"
        r"(?<![A-Za-z0-9_])(?:reorder|sort)(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}
_POOL_MUTATION_NONCOMMAND_RE = re.compile(
    r"[?？]|(?:能否|可否|是否|可以吗|能不能)|"
    r"(?:昨天|昨日|之前|此前|过去|上次|前次|早些时候|曾经|已经)"
    r"[^；;。\n]{0,180}(?:删除|移除|改成|设置|重排|排序)|"
    r"(?:改写|重写|润色|翻译|复述)[^；;。\n]{0,120}"
    r"(?:这句|这句话|下句|以下|文本|文案|内容)|"
    r"(?<![A-Za-z0-9_])(?:can|could|would)\s+you\b|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|last\s+time|"
    r"rewrite|rephrase|translate)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_REASON_CANCELLATION_RE = re.compile(
    r"(?:算了|作罢|反悔|暂停|停一下|放一放|不要了|先别|取消|撤销|撤回)|"
    r"(?<![A-Za-z0-9_])(?:never\s+mind|forget\s+it|scratch\s+that|"
    r"hold\s+on|cancel|abort|withdraw|stop)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_STRATEGY_TYPE_LABEL_RE = re.compile(
    r"(?:策略池类型|Pool\s*类型|strategy\s+pool\s+type|pool\s+type)"
    r"\s*(?:[:：=]|是|为)",
    re.IGNORECASE,
)
_POOL_ADD_DEFAULT_ACTION_LABEL_RE = re.compile(
    r"(?:(?:Pool|策略池)\s*)?默认动作|"
    r"(?<![A-Za-z0-9_])default\s+(?:pool\s+)?action"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_HIT_ACTION_LABEL_RE = re.compile(
    r"(?:规则)?命中动作|(?<![A-Za-z0-9_])(?:hit|match(?:ed)?)"
    r"\s+action(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_DEFAULT_REASON_CODE_LABEL_RE = re.compile(
    r"(?:(?:Pool|策略池)\s*)?默认(?:动作)?原因码|"
    r"(?<![A-Za-z0-9_])default(?:\s+action)?\s+"
    r"reason(?:\s+|[-_])code(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_HIT_REASON_CODE_LABEL_RE = re.compile(
    r"(?:规则)?命中(?:动作)?原因码|"
    r"(?<![A-Za-z0-9_])(?:hit|match(?:ed)?)(?:\s+action)?\s+"
    r"reason(?:\s+|[-_])code(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_DEFAULT_OUTPUT_VALUE_LABEL_RE = re.compile(
    r"(?:(?:Pool|策略池)\s*)?默认(?:动作)?输出值|"
    r"(?<![A-Za-z0-9_])default(?:\s+action)?\s+"
    r"output(?:\s+|[-_])value(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_HIT_OUTPUT_VALUE_LABEL_RE = re.compile(
    r"(?:规则)?命中(?:动作)?输出值|"
    r"(?<![A-Za-z0-9_])(?:hit|match(?:ed)?)(?:\s+action)?\s+"
    r"output(?:\s+|[-_])value(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_POOL_ADD_PLACEMENT_MODE_LABEL_RE = re.compile(
    r"(?:放置方式|placement\s+mode)\s*(?:[:：=]|是|为)",
    re.IGNORECASE,
)
_POOL_ADD_BEFORE_SELECTED_MEMBERS_RE = re.compile(
    r"保留(?:原|所选|这些)?成员作为回退"
    r"\s*(?:，|,|并且|并|且)?\s*"
    r"(?:将|把)?\s*(?:Voting|投票(?:候选)?)?\s*"
    r"放在(?:原|所选|这些)?成员前(?:面)?",
    re.IGNORECASE,
)
_POOL_ADD_REPLACE_SELECTED_MEMBERS_RE = re.compile(
    r"由\s*(?:Voting|投票(?:候选)?)\s*"
    r"(?:替代|替换|取代)(?:原|所选|这些)?成员",
    re.IGNORECASE,
)
_POOL_ADD_BEFORE_SELECTED_MEMBERS_EXPLANATION_RE = re.compile(
    r"保留(?:原|所选|这些)?成员作为未达\s*n\s*时的后续规则",
    re.IGNORECASE,
)
_POOL_ADD_REASON_LABEL_RE = re.compile(
    r"(?:入池|添加|操作)?理由\s*(?:[:：=]|是|为)\s*"
    r"(?P<zh>[^，,；;。\n]+)|"
    r"(?<![A-Za-z0-9_])(?:pool\s+reason|reason|rationale)"
    r"\s*(?::|=|is)\s*(?P<en>[^,;.!?\n]+)",
    re.IGNORECASE,
)
_POOL_ADD_STRATEGY_TYPE_NOUN_PATTERNS = {
    "approval": re.compile(
        r"(?:审批|准入)(?:策略|规则)?池|"
        r"(?:审批|准入)\s*(?:(?:Strategy|Rule)\s*)?Pool|"
        r"(?<![A-Za-z0-9_])approval\s+(?:(?:strategy|rule)\s+)?pool"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"拒绝(?:策略|规则)?池|拒绝\s*(?:(?:Strategy|Rule)\s*)?Pool|"
        r"(?<![A-Za-z0-9_])reject\s+(?:(?:strategy|rule)\s+)?pool"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"(?:额度|授信)(?:策略|规则)?池|"
        r"(?:额度|授信)\s*(?:(?:Strategy|Rule)\s*)?Pool|"
        r"(?<![A-Za-z0-9_])limit\s+(?:(?:strategy|rule)\s+)?pool"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"(?:定价|利率)(?:策略|规则)?池|"
        r"(?:定价|利率)\s*(?:(?:Strategy|Rule)\s*)?Pool|"
        r"(?<![A-Za-z0-9_])pricing\s+(?:(?:strategy|rule)\s+)?pool"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "segmentation": re.compile(
        r"(?:分群|分层)(?:策略|规则)?池|"
        r"(?:分群|分层)\s*(?:(?:Strategy|Rule)\s*)?Pool|"
        r"(?<![A-Za-z0-9_])segmentation\s+(?:(?:strategy|rule)\s+)?pool"
        r"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}
_REFINEMENT_SELECTION_ACTION_RE = re.compile(
    r"(?:选择|选中|保留|筛选|作为|select|keep|retain)", re.IGNORECASE
)
_REFINEMENT_MERGE_ACTION_RE = re.compile(r"(?:合并|并箱|merge|combine)", re.IGNORECASE)
_AUTOMATIC_TREE_NODE_TOKEN_PATTERN = (
    r"(?:"
    r"(?:(?:坏率|风险)\s*最高(?:的)?|高风险|终端|末端)\s*(?:叶子|叶节点|节点)|"
    r"叶(?:子|节点)?(?!权重|样本|数)|"
    r"(?<![A-Za-z0-9_])(?:leaf|leaves)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:high(?:est)?[-\s]+risk|terminal|end)[-\s]+nodes?"
    r"(?![A-Za-z0-9_])"
    r")"
)
_AUTOMATIC_TREE_DECISION_EFFECT_PATTERN = (
    r"(?:拒绝(?!率|数量|数|占比)|通过(?!率|数量|数|占比)|"
    r"审批(?!率|数量|数|占比)|复核(?!率|数量|数|占比)|"
    r"额度|定价|分群|动作|策略|"
    r"(?<![A-Za-z0-9_])(?:action|reject|approve|review|limit|pricing|segment|strategy)"
    r"(?:s|d|ed|ing)?(?![A-Za-z0-9_]))"
)
_AUTOMATIC_TREE_MULTI_STEP_RE = re.compile(
    r"(?:然后|随后|之后|再|同时|并(?:且)?|and\s+then).{0,60}"
    r"(?:自动\s*)?(?:选择|挑选|推荐|找出|(?<!候)选|select|pick|identify|materialize|加入|add).{0,24}"
    r"(?:叶(?:子|节点)?(?!权重|样本|数)|leaf|策略池|strategy\s*pool|pool)",
    re.IGNORECASE | re.DOTALL,
)
_AUTOMATIC_TREE_BEST_LEAF_RE = re.compile(
    r"(?:自动\s*)?(?:选择|挑选|推荐|找出|(?<!候)选|select|pick|identify).{0,12}"
    r"(?:最好|最优|最佳|best).{0,8}(?:叶(?:子|节点)?(?!权重|样本|数)|leaf)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_REVERSED_BEST_LEAF_RE = re.compile(
    r"(?:最好|最优|最佳|best).{0,8}(?:叶(?:子|节点)?(?!权重|样本|数)|leaf).{0,12}"
    r"(?:自动\s*)?(?:选择|挑选|推荐|找出|(?<!候)选|select|pick|identify)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_FOLLOW_UP_RE = re.compile(
    r"(?:选择|挑选|固化|物化|(?<!候)选|select|pick|materialize).{0,16}"
    r"(?:叶(?:子|节点)?(?!权重|样本|数)|leaf)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_POOL_FOLLOW_UP_RE = re.compile(
    r"(?:加入|写入|放入|add).{0,16}(?:策略池|strategy\s*pool|pool)|入池",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_DECISION_FOLLOW_UP_RE = re.compile(
    r"(?:"
    r"(?:把|将)?\s*(?:任何|任一|某个|该|这个)?\s*"
    r"(?:叶(?:子|节点)?(?!权重|样本|数)|leaf)"
    r"[^，,；;。\n]{0,24}(?:作为|设为|设置为|配置为|用作|转为|为)"
    rf"[^，,；;。\n]{{0,16}}{_AUTOMATIC_TREE_DECISION_EFFECT_PATTERN}|"
    r"(?:设置|配置|采用|使用|把|将)"
    r"[^，,；;。\n]{0,20}(?:叶(?:子|节点)?(?!权重|样本|数)|leaf)"
    rf"[^，,；;。\n]{{0,20}}{_AUTOMATIC_TREE_DECISION_EFFECT_PATTERN}"
    r")",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_HEURISTIC_LEAF_FOLLOW_UP_RE = re.compile(
    r"(?:采用|使用|选用|选择|挑选|推荐|pick|select|use)"
    r"[^，,；;。\n]{0,20}(?:坏率|风险|lift|捕获率|通过率|收益|profit)?"
    r"[^，,；;。\n]{0,10}(?:最高|最低|最大|最小|最好|最优|最佳|best|highest|lowest)"
    r"[^，,；;。\n]{0,10}(?:叶(?:子|节点)?(?!权重|样本|数)|leaf)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_NODE_RANK_FOLLOW_UP_RE = re.compile(
    rf"(?:排名|排序|rank|sort)[^，,；;。\n]{{0,32}}{_AUTOMATIC_TREE_NODE_TOKEN_PATTERN}|"
    rf"{_AUTOMATIC_TREE_NODE_TOKEN_PATTERN}[^，,；;。\n]{{0,32}}(?:排名|排序|rank|sort)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_NODE_SELECT_FOLLOW_UP_RE = re.compile(
    rf"(?:选择|挑选|保留|采用|选用|select|pick|retain|keep|use)"
    rf"[^，,；;。\n]{{0,24}}{_AUTOMATIC_TREE_NODE_TOKEN_PATTERN}|"
    rf"{_AUTOMATIC_TREE_NODE_TOKEN_PATTERN}[^，,；;。\n]{{0,24}}"
    r"(?:选择|挑选|保留|select|pick|retain|keep)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_NODE_EXTRACT_FOLLOW_UP_RE = re.compile(
    rf"(?:提取|extract)\s*(?!(?:完整|全部|所有|complete|all))"
    rf"[^，,；;。\n]{{0,24}}{_AUTOMATIC_TREE_NODE_TOKEN_PATTERN}",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LIFECYCLE_FOLLOW_UP_RE = re.compile(
    r"(?:采纳|采用)\s*(?:这棵|该|当前|整个)\s*(?:树|决策树|模型|结果)|"
    r"(?:部署|上线|发布到?生产|提升为生产|投入生产)|"
    r"(?<![A-Za-z0-9_])(?:adopt\s+(?:it|this\s+tree|the\s+tree)|"
    r"deploy|promote(?:\s+(?:it|this\s+tree|the\s+tree))?)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_ID_WRITEBACK_RE = re.compile(
    r"(?:写回|回写|持久化|保存|write\s*back|persist|store)"
    r"[^，,；;。\n]{0,20}(?:叶(?:子|节点)?\s*(?:ID|id|编号)|leaf[-_\s]*id)|"
    r"(?:叶(?:子|节点)?\s*(?:ID|id|编号)|leaf[-_\s]*id)"
    r"[^，,；;。\n]{0,20}(?:写回|回写|持久化|保存|write\s*back|persist|store)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_DECISION_ARTIFACT_RE = re.compile(
    r"(?:生成|形成|制定|配置|执行|create|generate|form|formulate|define|configure|execute)"
    rf"[^，,；;。\n]{{0,24}}{_AUTOMATIC_TREE_DECISION_EFFECT_PATTERN}",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_FOLLOW_UP_CLAUSE_BOUNDARY_RE = re.compile(
    r"[，,；;。\n]+|"
    r"(?:但(?:是)?|可是|不过|却|而(?:是)?|并且|同时|然后|随后|之后|接着)|"
    r"后(?=\s*(?:直接|接着|然后|再|把|将|对|让|依据|执行|设置|采用|使用|"
    r"选择|拒绝|通过|给出|加入|写入))|"
    r"(?<!不)(?<!不要)(?<!无需)(?<!不用)(?<!不必)(?<!禁止)(?<!别)"
    r"再(?=\s*(?:把|将|对|让|依据|直接|执行|设置|采用|使用|选择|拒绝|通过|"
    r"给出|加入|写入))|"
    r"且|"
    r"并(?=(?:把|将|对|让|依据|执行|设置|采用|使用|选择|拒绝|通过|给出|加入|写入))|"
    r"(?<![A-Za-z0-9_])(?:but|however|yet|and|then|afterwards|after\s+that)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_FOLLOW_UP_NEGATION_RE = re.compile(
    r"(?:不要|无需|不用|不必|别|禁止|不再|"
    r"不(?=\s*(?:把|将|让|对|依据|直接|自动|采用|使用|选择|挑选|推荐|"
    r"物化|固化|选中|设置|执行|拒绝|通过|审批|复核|给出|作为|加入|写入|放入))|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don['’]t|never|not)(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LEAF_TOKEN_RE = re.compile(
    _AUTOMATIC_TREE_NODE_TOKEN_PATTERN,
    re.IGNORECASE,
)
_AUTOMATIC_TREE_DECISION_EFFECT_RE = re.compile(
    _AUTOMATIC_TREE_DECISION_EFFECT_PATTERN,
    re.IGNORECASE,
)
_AUTOMATIC_TREE_FOLLOW_UP_ACTION_ANCHOR_RE = re.compile(
    r"(?:选择|挑选|推荐|找出|(?<!候)选|加入|写入|放入|入池|作为|设为|"
    r"设置|设置为|配置为|用作|转为|执行|给出|采用|使用|拒绝|通过|审批|复核|"
    r"排名|排序|保留|提取|采纳|部署|上线|写回|回写|持久化|保存|生成|形成|"
    r"制定|配置|"
    r"(?<![A-Za-z0-9_])(?:select|pick|identify|materialize|add|use|route|set|"
    r"make|reject|approve|review|rank|sort|retain|extract|adopt|deploy|promote|"
    r"persist|store|create|generate|form|formulate|define|configure|execute)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_NEGATED_FOLLOW_UP_PREFIX_RE = re.compile(
    r"^(?:\s+|[-/:]|"
    r"(?:再|自动|直接|仅|只|把|将|让|对|给|依据|根据|按|以|任何|任一|"
    r"某个|该|这个|这些|这棵|整个|全部|所有|高风险|低风险|风险|坏率|给|"
    r"风险最高|风险最低|坏率最高|坏率最低|最好|最优|最佳|完整|终端|末端|"
    r"叶节点|叶子|叶|节点|树|决策树|模型|结果|走|"
    r"转为|作为|设为|设置|设置为|配置|配置为|用作|执行|进行|给出|采用|使用|"
    r"选择|挑选|推荐|找出|选用|加入|写入|放入|入池|人工|拒绝|通过|"
    r"审批|复核|额度|定价|分群|规则|动作|策略|策略池|排名|排序|保留|"
    r"提取|采纳|部署|上线|写回|回写|持久化|保存|生成|形成|制定)|"
    r"(?<![A-Za-z0-9_])(?:auto|automatically|directly|the|any|a|an|all|some|"
    r"high(?:est)?[-\s]+risk|best|worst|leaf(?:[-_][A-Za-z0-9.]+)?|leaves|"
    r"to|as|manual|use|route|pick|select|set|make|turn|into|for|reject|"
    r"approve|review|rule|action|add|materialize|strategy|pool|rank|sort|"
    r"retain|extract|adopt|deploy|promote|persist|store|create|generate|form|"
    r"formulate|define|configure|execute|it|this|tree)"
    r"(?![A-Za-z0-9_]))*$",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_NEGATED_BUILD_RE = re.compile(
    r"(?:不要|无需|不用|不必|别|禁止|不想|不需要)"
    r"[^，,；;。\n]{0,40}(?:建(?:一棵)?(?:自动)?(?:决策)?树|"
    r"(?:构建|训练|创建)[^，,；;。\n]{0,12}(?:树|决策树))|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don['’]t|never|not)"
    r"[^，,；;。\n]{0,32}(?:build|train|create)"
    r"[^，,；;。\n]{0,12}(?:tree)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_DATASET_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:dataset(?:_id)?|sample(?:_id)?)(?![A-Za-z0-9_])"
    r"\s*(?:[:：=]|为|是)\s*[^\s，,；;。]+|"
    r"(?:用|使用|改用|切换(?:到)?|换成)\s*(?:另一个|其他|新的|指定的?)?\s*"
    r"(?:数据集|样本)(?!权重|数)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_TARGET_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:target(?:_col|\s+column)?|label(?:_col|\s+column)?)"
    r"(?![A-Za-z0-9_])\s*(?:[:：=]|为|是)\s*[^\s，,；;。]+|"
    r"(?:目标|标签)(?:列|字段)\s*(?:[:：=]|为|是|改为|切换为)\s*"
    r"[^\s，,；;。]+",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_LABEL_POLICY_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:drop|keep)[_\s-]*(?:nan|null|missing)[_\s-]*labels?"
    r"(?![A-Za-z0-9_])|"
    r"(?:删除|丢弃|保留|填充|忽略)[^，,；;。\n]{0,12}"
    r"(?:空|缺失|NULL|NaN)[^，,；;。\n]{0,6}(?:标签|目标值)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_BUDGET_CONTROL_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:budgets?\s*[.:]\s*)?max[_\s-]*"
    r"(?:rows|features|cells|nodes|cutpoints?)(?![A-Za-z0-9_])|"
    r"(?:最多|至多|不超过|仅|只|限制)[^，,；;。\n]{0,16}"
    r"[0-9]+[^，,；;。\n]{0,8}(?:行|特征|变量|单元格|节点|切点)",
    re.IGNORECASE,
)
_AUTOMATIC_TREE_DIRECTION_GROUNDING = {
    "increasing": (
        r"(?:单调\s*)?(?:递增|上升)|正向|"
        r"(?<![A-Za-z0-9_])increasing(?![A-Za-z0-9_])"
    ),
    "decreasing": (
        r"(?:单调\s*)?(?:递减|下降)|负向|"
        r"(?<![A-Za-z0-9_])decreasing(?![A-Za-z0-9_])"
    ),
    "unordered": (
        r"无序|不约束(?:方向)?|不限方向|"
        r"(?<![A-Za-z0-9_])unordered(?![A-Za-z0-9_])"
    ),
}
_AUTOMATIC_TREE_NUMBER_LABELS = {
    "max_depth": (
        r"(?<![A-Za-z0-9_])max[_\s-]*depth(?![A-Za-z0-9_])|"
        r"最大(?:树)?深度|树深"
    ),
    "min_leaf_count": (
        r"(?<![A-Za-z0-9_])min[_\s-]*leaf[_\s-]*count(?![A-Za-z0-9_])|"
        r"最小(?:叶(?:子|节点)?)(?:样本)?(?:数|量)?|"
        r"叶(?:子|节点)?最少(?:样本)?(?:数|量)?"
    ),
    "min_weight_fraction_leaf": (
        r"(?<![A-Za-z0-9_])min[_\s-]*weight[_\s-]*fraction[_\s-]*leaf"
        r"(?![A-Za-z0-9_])|"
        r"最小(?:叶(?:子|节点)?)?权重(?:占比|比例)|"
        r"叶(?:子|节点)?最小权重(?:占比|比例)"
    ),
    "seed": r"(?:随机)?种子|(?<![A-Za-z0-9_])seed(?![A-Za-z0-9_])",
}
_AUTOMATIC_TREE_COLUMN_ROLE_LABELS = {
    "sample_weight_col": (
        r"sample[_\s-]*weight(?:[_\s-]*col)?|"
        r"样本权重(?:列|字段)?|权重(?:列|字段)"
    ),
    "loan_amount_col": (
        r"loan[_\s-]*amount(?:[_\s-]*col)?|"
        r"放款金额(?:列|字段)?|贷款金额(?:列|字段)?"
    ),
    "overdue_amount_col": (r"overdue[_\s-]*amount(?:[_\s-]*col)?|逾期金额(?:列|字段)?"),
}
_RISK_THRESHOLD_EXPRESSION_RE = re.compile(
    r"(?:观测)?(?:坏率|坏账率|风险率|bad\s*rate|risk\s*rate)"
    r"\s*(?:为|是|需|需要|应|must\s+be|is)?\s*"
    r"(?P<operator>大于等于|不低于|至少|不少于|达到|>=|≥|"
    r"小于等于|不高于|至多|最多|<=|≤|"
    r"大于|高于|超过|>|小于|低于|少于|<|"
    r"greater\s+than\s+or\s+equal(?:\s+to)?|at\s+least|"
    r"less\s+than\s+or\s+equal(?:\s+to)?|at\s+most|"
    r"more\s+than|greater\s+than|less\s+than)"
    r"\s*(?P<value>百分之\s*[0-9]+(?:\.[0-9]+)?|"
    r"[0-9]+(?:\.[0-9]+)?\s*%|"
    r"(?:0(?:\.\d+)?|1(?:\.0+)?))",
    re.IGNORECASE,
)

_OPTIONAL_DRAFT_FIELDS = {
    "objective",
    "max_bad_rate",
    "min_approval_rate",
    "baseline_strategy_id",
    "strategy_id",
    "adoption_reason",
    "profit",
    "economics_inputs",
    "candidate_design",
    "strategy_spec",
}
_DRAFT_FIELDS = {"operation", "strategy_type"} | _OPTIONAL_DRAFT_FIELDS
_LIFECYCLE_DRAFT_FIELDS = _DRAFT_FIELDS | {"request_kind"}
_STANDARD_WORKFLOW_DRAFT_FIELDS = {
    "request_kind",
    "workflow",
    "workflow_inputs",
}
_PROFIT_FIELDS = {
    "ead_col",
    "pd_col",
    "annual_rate",
    "funding_rate",
    "lgd",
    "operating_cost_per_loan",
    "term_months",
}
_PROFIT_PARAMETER_FIELDS = _PROFIT_FIELDS - {"ead_col", "pd_col"}
_LIMIT_ECONOMICS_NAMES = ("pd", "lgd", "utilization")
_PRICING_ECONOMICS_NAMES = (
    "ead",
    "pd",
    "lgd",
    "funding_rate",
    "term_months",
    "operating_cost_per_loan",
)
_ECONOMICS_VALUE_MAXIMUMS = {
    "pd": 1.0,
    "lgd": 1.0,
    "utilization": 1.0,
    "funding_rate": 1.0,
}
_ECONOMICS_LABELS = {
    "ead": "EAD",
    "pd": "PD",
    "lgd": "LGD",
    "utilization": "额度使用率",
    "funding_rate": "资金成本率",
    "term_months": "期限",
    "operating_cost_per_loan": "单笔运营成本",
}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_COLLECTION_STRATEGY_RE = re.compile(
    r"(?:催收|\bcollection(?:s)?(?:\s+|[-_])"
    r"(?:strategy|actions?|allocation|policy|workflow|campaign|frequency)\b)",
    re.IGNORECASE,
)
_NON_REPAIRABLE_CLARIFICATION_CODES = frozenset(
    {
        "candidate_economics_ambiguous",
        "candidate_economics_incomplete",
        "candidate_requires_observed_economics",
        "strategy_report_bundle_v2_platform_binding_forbidden",
        "strategy_dsl_delivery_platform_binding_forbidden",
        "strategy_request_too_complex",
    }
)


_CANDIDATE_DESIGN_JSON_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "score_band_limit"},
                "score_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "limit_grid": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "uniqueItems": True,
                    "items": {"type": "number", "exclusiveMinimum": 0},
                },
                "max_expected_loss_per_account": {
                    "type": "number",
                    "minimum": 0,
                },
                "missing_policy": {"const": "zero_limit"},
            },
            "required": [
                "method",
                "score_col",
                "limit_grid",
                "max_expected_loss_per_account",
            ],
            "additionalProperties": False,
        },
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "score_band_pricing"},
                "score_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "rate_grid": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 50,
                    "uniqueItems": True,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "min_roa": {"type": "number", "minimum": 0, "maximum": 1},
                "missing_policy": {"const": "highest_risk_rate"},
            },
            "required": ["method", "score_col", "rate_grid"],
            "additionalProperties": False,
        },
        {
            "properties": {
                "schema_version": {"const": CANDIDATE_DESIGN_SCHEMA_VERSION},
                "method": {"const": "single_variable_segmentation"},
                "feature_col": {"type": "string", "minLength": 1},
                "n_bands": {"type": "integer", "minimum": 2, "maximum": 20},
                "missing_policy": {"const": "separate_segment"},
            },
            "required": ["method", "feature_col"],
            "additionalProperties": False,
        },
    ],
}


STRATEGY_REQUEST_JSON_SCHEMA = {
    "name": "strategy_request_draft",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "request_kind": {
                "type": "string",
                "enum": list(STRATEGY_REQUEST_KINDS),
            },
            "operation": {"type": "string", "enum": list(STRATEGY_OPERATIONS)},
            "strategy_type": {"type": "string", "enum": list(STRATEGY_TYPES)},
            "objective": {"type": "string", "minLength": 1},
            "max_bad_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "min_approval_rate": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "baseline_strategy_id": {"type": "string", "minLength": 1},
            "strategy_id": {"type": "string", "minLength": 1},
            "adoption_reason": {"type": "string", "minLength": 1},
            "profit": {
                "type": "object",
                "properties": {
                    "ead_col": {"type": "string", "minLength": 1},
                    "pd_col": {"type": "string", "minLength": 1},
                    "annual_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "funding_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "lgd": {"type": "number", "minimum": 0, "maximum": 1},
                    "operating_cost_per_loan": {
                        "type": "number",
                        "minimum": 0,
                    },
                    "term_months": {"type": "integer", "minimum": 1},
                },
                "required": sorted(_PROFIT_FIELDS),
                "additionalProperties": False,
            },
            "economics_inputs": {
                "type": "object",
                "properties": {
                    "ead_col": {"type": "string", "minLength": 1},
                    "ead_value": {"type": "number", "minimum": 0},
                    "pd_col": {"type": "string", "minLength": 1},
                    "pd_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "lgd_col": {"type": "string", "minLength": 1},
                    "lgd_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "utilization_col": {"type": "string", "minLength": 1},
                    "utilization_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "funding_rate_col": {"type": "string", "minLength": 1},
                    "funding_rate_value": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "term_months_col": {"type": "string", "minLength": 1},
                    "term_months_value": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                    },
                    "operating_cost_per_loan_col": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "operating_cost_per_loan_value": {
                        "type": "number",
                        "minimum": 0,
                    },
                },
                "oneOf": [
                    {
                        "allOf": [
                            {
                                "oneOf": [
                                    {"required": ["pd_col"]},
                                    {"required": ["pd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["lgd_col"]},
                                    {"required": ["lgd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["utilization_col"]},
                                    {"required": ["utilization_value"]},
                                ]
                            },
                            {
                                "not": {
                                    "anyOf": [
                                        {"required": ["ead_col"]},
                                        {"required": ["ead_value"]},
                                        {"required": ["funding_rate_col"]},
                                        {"required": ["funding_rate_value"]},
                                        {"required": ["term_months_col"]},
                                        {"required": ["term_months_value"]},
                                        {"required": ["operating_cost_per_loan_col"]},
                                        {"required": ["operating_cost_per_loan_value"]},
                                    ]
                                }
                            },
                        ]
                    },
                    {
                        "allOf": [
                            {
                                "oneOf": [
                                    {"required": ["ead_col"]},
                                    {"required": ["ead_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["pd_col"]},
                                    {"required": ["pd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["lgd_col"]},
                                    {"required": ["lgd_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["funding_rate_col"]},
                                    {"required": ["funding_rate_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["term_months_col"]},
                                    {"required": ["term_months_value"]},
                                ]
                            },
                            {
                                "oneOf": [
                                    {"required": ["operating_cost_per_loan_col"]},
                                    {"required": ["operating_cost_per_loan_value"]},
                                ]
                            },
                            {
                                "not": {
                                    "anyOf": [
                                        {"required": ["utilization_col"]},
                                        {"required": ["utilization_value"]},
                                    ]
                                }
                            },
                        ]
                    },
                ],
                "additionalProperties": False,
            },
            "candidate_design": _CANDIDATE_DESIGN_JSON_SCHEMA,
            "strategy_spec": {"type": "object"},
            "workflow": {
                "type": "string",
                "enum": list(STANDARD_STRATEGY_WORKFLOWS),
            },
            "workflow_inputs": {"type": "object"},
            "clarification": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
        "oneOf": [
            {"required": ["operation", "strategy_type"]},
            {
                "properties": {
                    "request_kind": {"const": "standard_workflow"},
                },
                "required": ["request_kind", "workflow", "workflow_inputs"],
            },
            {"required": ["clarification"]},
        ],
    },
}


@dataclass(frozen=True)
class StrategyRequestDraft(Mapping[str, Any]):
    """Canonical, platform-validated strategy request draft."""

    operation: str
    strategy_type: str
    objective: str | None = None
    max_bad_rate: float | None = None
    min_approval_rate: float | None = None
    baseline_strategy_id: str | None = None
    strategy_id: str | None = None
    adoption_reason: str | None = None
    profit: Mapping[str, Any] | None = None
    economics_inputs: Mapping[str, Any] | None = None
    candidate_design: Mapping[str, Any] | None = None
    strategy_spec: Mapping[str, Any] | None = None

    @property
    def request_kind(self) -> str:
        return "strategy_lifecycle"

    def __post_init__(self) -> None:
        if self.profit is not None:
            object.__setattr__(self, "profit", _deep_freeze(self.profit))
        if self.economics_inputs is not None:
            object.__setattr__(
                self,
                "economics_inputs",
                _deep_freeze(self.economics_inputs),
            )
        if self.candidate_design is not None:
            object.__setattr__(
                self,
                "candidate_design",
                _deep_freeze(self.candidate_design),
            )
        if self.strategy_spec is not None:
            object.__setattr__(
                self,
                "strategy_spec",
                _deep_freeze(self.strategy_spec),
            )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "strategy_type": self.strategy_type,
        }
        for field_name in (
            "objective",
            "max_bad_rate",
            "min_approval_rate",
            "baseline_strategy_id",
            "strategy_id",
            "adoption_reason",
            "profit",
            "economics_inputs",
            "candidate_design",
            "strategy_spec",
        ):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = _deep_thaw(value)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class StandardWorkflowRequestDraft(Mapping[str, Any]):
    """Canonical request for a built-in, deterministic strategy analysis."""

    workflow: str
    workflow_inputs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "workflow_inputs",
            _deep_freeze(self.workflow_inputs),
        )

    @property
    def request_kind(self) -> str:
        return "standard_workflow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_kind": self.request_kind,
            "workflow": self.workflow,
            "workflow_inputs": _deep_thaw(self.workflow_inputs),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


CompiledStrategyRequestDraft = StrategyRequestDraft | StandardWorkflowRequestDraft


@dataclass(frozen=True)
class StrategyRequestCompilation:
    """A validated draft awaiting confirmation, or a Chinese clarification."""

    draft: CompiledStrategyRequestDraft | None
    clarification: str | None
    confirmation: str | None
    clarification_code: str | None = None
    clarification_fields: tuple[str, ...] = ()

    @property
    def validated_draft(self) -> CompiledStrategyRequestDraft | None:
        return self.draft

    @property
    def clarify(self) -> str | None:
        return self.clarification

    @property
    def confirmation_text(self) -> str | None:
        return self.confirmation

    @property
    def needs_clarification(self) -> bool:
        return self.draft is None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "draft": None if self.draft is None else self.draft.to_dict(),
            "clarification": self.clarification,
            "confirmation": self.confirmation,
        }
        if self.clarification_code is not None:
            payload["clarification_code"] = self.clarification_code
        if self.clarification_fields:
            payload["clarification_fields"] = list(self.clarification_fields)
        return payload


@dataclass(frozen=True)
class _ValidationOutcome:
    result: StrategyRequestCompilation
    accepted: bool
    error: str | None = None


def compile_strategy_request(
    utterance: str,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
    llm,
    caller: str = "strategy_request_compiler",
) -> StrategyRequestCompilation:
    """Compile one utterance with at most one LLM-format repair attempt."""

    if not isinstance(utterance, str) or not utterance.strip():
        return _clarification("请说明希望执行的策略操作和策略类型。")
    normalized_utterance = utterance.strip()
    if _COLLECTION_STRATEGY_RE.search(normalized_utterance):
        return _clarification(
            "催收动作策略尚无已评审的动作、成本、产能和回收口径，"
            "当前不能映射为审批、拒绝或分群策略。请说明是否只需要风险分层分析。",
            code="collection_strategy_unsupported",
            fields=("strategy_type", "collection_action_contract"),
        )
    whitelist = _column_whitelist(allowed_columns)
    observed_target = _normalized_target_col(target_col)
    prompt = _user_prompt(normalized_utterance, whitelist, target_col=observed_target)
    try:
        raw = _complete(llm, prompt=prompt, caller=caller)
    except Exception:
        return _clarification(
            "当前暂时无法解析策略请求，请稍后重试或直接说明操作、策略类型和策略对象。"
        )
    outcome = _validate_reply(raw, whitelist, target_col=observed_target)
    if outcome.accepted:
        return _ground_refinement_request(
            normalized_utterance,
            outcome.result,
            whitelist=whitelist,
        )
    if outcome.result.clarification_code in _NON_REPAIRABLE_CLARIFICATION_CODES:
        # These are platform-derived business-contract gaps, not JSON-format
        # mistakes. A second LLM pass cannot supply missing economics safely and
        # must not downgrade typed code/fields into a generic clarification.
        return outcome.result

    repair_prompt = _repair_prompt(
        prompt,
        raw=raw,
        error=outcome.error or "输出格式无效",
    )
    try:
        repaired = _complete(llm, prompt=repair_prompt, caller=caller)
    except Exception:
        return outcome.result
    repaired_outcome = _validate_reply(
        repaired,
        whitelist,
        target_col=observed_target,
    )
    if repaired_outcome.accepted:
        return _ground_refinement_request(
            normalized_utterance,
            repaired_outcome.result,
            whitelist=whitelist,
        )
    return repaired_outcome.result


def validate_strategy_request(
    payload: object,
    *,
    allowed_columns: Iterable[str] | None,
    target_col: str | None = None,
    allow_legacy_replay: bool = False,
) -> StrategyRequestCompilation:
    """Validate an already parsed request without invoking an LLM.

    Fresh callers intentionally cannot emit the retired V1 sample workflow.
    Only a persistence/recovery boundary may opt into ``allow_legacy_replay``.
    """

    return _validate_payload(
        payload,
        _column_whitelist(allowed_columns),
        target_col=_normalized_target_col(target_col),
        allow_legacy_replay=allow_legacy_replay,
    ).result


def strategy_request_confirmation_text(
    draft: CompiledStrategyRequestDraft,
) -> str:
    """Render a plain-Chinese echo of the request before any workflow runs."""

    if isinstance(draft, StandardWorkflowRequestDraft):
        return _standard_workflow_confirmation_text(draft)

    operation = _OPERATION_LABELS[draft.operation]
    strategy_type = _TYPE_LABELS[draft.strategy_type]
    details = [f"已识别为〔{strategy_type}〕的〔{operation}〕请求"]
    if draft.strategy_id:
        details.append(f"策略 ID：{draft.strategy_id}")
    if draft.baseline_strategy_id:
        details.append(f"基线策略 ID：{draft.baseline_strategy_id}")
    if draft.objective:
        details.append(f"业务目标：{draft.objective}")
    constraints: list[str] = []
    if draft.max_bad_rate is not None:
        constraints.append(f"最大坏账率 {draft.max_bad_rate:.2%}")
    if draft.min_approval_rate is not None:
        constraints.append(f"最低通过率 {draft.min_approval_rate:.2%}")
    if constraints:
        details.append("业务约束：" + "、".join(constraints))
    if draft.profit is not None:
        details.append(
            "利润口径："
            f"EAD 列 {draft.profit['ead_col']}，PD 列 {draft.profit['pd_col']}，"
            f"年利率 {draft.profit['annual_rate']:.2%}，"
            f"资金成本率 {draft.profit['funding_rate']:.2%}，"
            f"LGD {draft.profit['lgd']:.2%}，"
            f"单笔运营成本 {draft.profit['operating_cost_per_loan']:g}，"
            f"期限 {draft.profit['term_months']} 个月"
        )
    if draft.economics_inputs is not None:
        details.append(_economics_confirmation(draft))
    if draft.candidate_design is not None:
        details.append(_candidate_design_confirmation(draft))
    if draft.strategy_spec is not None:
        details.append(_strategy_spec_confirmation(draft.strategy_spec))
    if draft.adoption_reason:
        details.append(f"采纳理由：{draft.adoption_reason}")
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；"
        "所有指标由平台确定性计算，采纳等治理动作仍需相应人工确认。"
    )
    return "；".join(details)


def _strategy_spec_confirmation(strategy_spec: Mapping[str, Any]) -> str:
    """Echo every executable rule/action so confirmation is not blind.

    The request row remains the authoritative canonical payload.  This text is
    a deterministic, human-reviewable projection: it contains no calculated
    metrics and does not ask the LLM to explain its own draft.
    """

    rules = list(strategy_spec.get("rules") or [])
    default_action = _compact_json(strategy_spec.get("default_action") or {})
    rendered = [
        f"规则草案：{len(rules)} 条规则，匹配方式 first_match，默认动作 {default_action}"
    ]
    for index, rule in enumerate(rules, start=1):
        rule_id = str(rule.get("rule_id") or f"rule-{index}")
        priority = rule.get("priority")
        condition = _compact_json(rule.get("condition") or {})
        action = _compact_json(rule.get("action") or {})
        rendered.append(
            f"规则 {index} [{rule_id}]（优先级 {priority}）：IF {condition} THEN {action}"
        )
    return "；".join(rendered)


def _compact_json(value: object) -> str:
    return json.dumps(
        _deep_thaw(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _complete(llm, *, prompt: str, caller: str):
    return llm.complete(
        system_prompt=_SYSTEM,
        user_prompt=prompt,
        temperature=0.0,
        response_format={"type": "json_object"},
        json_schema=STRATEGY_REQUEST_JSON_SCHEMA,
        stream=False,
        caller=caller,
        prompt_name=STRATEGY_REQUEST_COMPILER_SYS.name,
        prompt_version=STRATEGY_REQUEST_COMPILER_SYS.version,
    )


def _strategy_payload_within_limits(payload: object) -> bool:
    stack: list[tuple[object, int]] = [(payload, 0)]
    seen_containers: set[int] = set()
    node_count = 0
    while stack:
        value, depth = stack.pop()
        node_count += 1
        if node_count > _STRATEGY_REPLY_MAX_NODES or depth > _STRATEGY_REPLY_MAX_DEPTH:
            return False
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            identity = id(value)
            if identity in seen_containers:
                return False
            seen_containers.add(identity)
            stack.extend((item, depth + 1) for item in value)
    return True


def _validate_reply(
    raw: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> _ValidationOutcome:
    if isinstance(raw, str) and len(raw) > _STRATEGY_REPLY_MAX_CHARS:
        return _invalid(
            "模型返回的策略草案过长，请缩小为一个明确的策略操作。",
            code="strategy_request_too_complex",
            fields=("reply",),
        )
    payload, error = load_json_object(raw)
    if payload is None:
        message = "模型返回的策略草案不是有效 JSON 对象，请重新说明策略请求。"
        return _ValidationOutcome(_clarification(message), False, error or message)
    return _validate_payload(
        payload,
        whitelist,
        target_col=target_col,
        allow_legacy_replay=False,
    )


def _validate_payload(
    payload: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    allow_legacy_replay: bool,
) -> _ValidationOutcome:
    if not isinstance(payload, Mapping):
        message = "策略请求必须是 JSON 对象，请重新说明。"
        return _invalid(message)
    if not _strategy_payload_within_limits(payload):
        return _invalid(
            "策略草案嵌套过深或字段过多，请缩小为一个明确的策略操作。",
            code="strategy_request_too_complex",
            fields=("payload",),
        )
    if any(not isinstance(key, str) for key in payload):
        return _invalid("策略请求的字段名必须是文本，请重新说明。")

    if "clarification" in payload:
        if set(payload) != {"clarification"}:
            return _invalid("澄清问题不能和策略草案字段同时出现，请重新选择一种输出。")
        value = payload["clarification"]
        if not isinstance(value, str) or not value.strip():
            return _invalid("澄清问题必须是非空文本。")
        return _ValidationOutcome(
            _clarification(_chinese_clarification(value)),
            True,
        )

    request_kind = payload.get("request_kind", "strategy_lifecycle")
    if not isinstance(request_kind, str) or request_kind not in STRATEGY_REQUEST_KINDS:
        return _invalid(
            "不支持的 request_kind；只能是 strategy_lifecycle 或 standard_workflow。"
        )
    if request_kind == "standard_workflow":
        unexpected = sorted(set(payload) - _STANDARD_WORKFLOW_DRAFT_FIELDS)
        if unexpected:
            rendered = "、".join(f"「{field}」" for field in unexpected)
            return _invalid(
                f"标准 Workflow 请求包含不支持的字段 {rendered}，请删除后重新说明。"
            )
        return _validate_standard_workflow_payload(
            payload,
            whitelist,
            target_col=target_col,
            allow_legacy_replay=allow_legacy_replay,
        )

    unexpected = sorted(set(payload) - _LIFECYCLE_DRAFT_FIELDS)
    if unexpected:
        rendered = "、".join(f"「{field}」" for field in unexpected)
        return _invalid(f"策略请求包含不支持的字段 {rendered}，请删除后重新说明。")
    if payload.get("request_kind") not in (None, "strategy_lifecycle"):
        return _invalid("策略生命周期请求的 request_kind 必须是 strategy_lifecycle。")
    missing = [
        field for field in ("operation", "strategy_type") if field not in payload
    ]
    if missing:
        rendered = "、".join(missing)
        return _invalid(f"没有识别到必需字段 {rendered}，请补充策略操作和策略类型。")

    operation = payload["operation"]
    if not isinstance(operation, str) or operation not in STRATEGY_OPERATIONS:
        return _invalid(
            "不支持的策略操作；可选操作为：" + "、".join(STRATEGY_OPERATIONS) + "。"
        )
    strategy_type = payload["strategy_type"]
    if not isinstance(strategy_type, str) or strategy_type not in STRATEGY_TYPES:
        return _invalid(
            "不支持的策略类型；可选类型为：" + "、".join(STRATEGY_TYPES) + "。"
        )

    try:
        _validate_economics_field_ownership(payload, strategy_type=strategy_type)
        _validate_candidate_field_ownership(
            payload,
            operation=operation,
            strategy_type=strategy_type,
        )
        objective = _optional_text(payload, "objective")
        max_bad_rate = _optional_ratio(payload, "max_bad_rate")
        min_approval_rate = _optional_ratio(payload, "min_approval_rate")
        baseline_strategy_id = _optional_text(payload, "baseline_strategy_id")
        strategy_id = _optional_text(payload, "strategy_id")
        adoption_reason = _optional_adoption_reason(payload)
        profit = _optional_profit(payload, whitelist)
        economics_inputs = _optional_economics_inputs(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
        candidate_design = _optional_candidate_design(
            payload,
            operation=operation,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
        if candidate_design is not None:
            try:
                economics_inputs = normalize_candidate_economics_inputs(
                    strategy_type,
                    economics_inputs,
                    allowed_columns=whitelist,
                )
            except CandidateDesignError as exc:
                raise _DraftValidationError(
                    str(exc),
                    code=exc.code,
                    fields=exc.fields,
                ) from exc
        strategy_spec = _optional_strategy_spec(
            payload,
            strategy_type=strategy_type,
            whitelist=whitelist,
        )
    except _DraftValidationError as exc:
        return _invalid(str(exc), code=exc.code, fields=exc.fields)

    draft = StrategyRequestDraft(
        operation=operation,
        strategy_type=strategy_type,
        objective=objective,
        max_bad_rate=max_bad_rate,
        min_approval_rate=min_approval_rate,
        baseline_strategy_id=baseline_strategy_id,
        strategy_id=strategy_id,
        adoption_reason=adoption_reason,
        profit=profit,
        economics_inputs=economics_inputs,
        candidate_design=candidate_design,
        strategy_spec=strategy_spec,
    )
    result = StrategyRequestCompilation(
        draft=draft,
        clarification=None,
        confirmation=strategy_request_confirmation_text(draft),
    )
    return _ValidationOutcome(result, True)


def _validate_standard_workflow_payload(
    payload: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    allow_legacy_replay: bool,
) -> _ValidationOutcome:
    missing = [
        field for field in ("workflow", "workflow_inputs") if field not in payload
    ]
    if missing:
        return _invalid("标准 Workflow 请求缺少字段：" + "、".join(missing) + "。")
    workflow = payload["workflow"]
    allowed_workflows = (
        REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS
        if allow_legacy_replay
        else FRESH_STANDARD_STRATEGY_WORKFLOWS
    )
    if not isinstance(workflow, str) or workflow not in allowed_workflows:
        return _invalid(
            "不支持的标准 Workflow；可选值为："
            + "、".join(allowed_workflows)
            + "。"
        )
    raw_inputs = payload["workflow_inputs"]
    if not isinstance(raw_inputs, Mapping):
        return _invalid("workflow_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        return _invalid("workflow_inputs 的字段名必须是文本。")
    try:
        if workflow == "strategy_project_context":
            normalized = _validate_strategy_project_context_inputs(raw_inputs)
        elif workflow == "strategy_sample_design_v2":
            normalized = _validate_strategy_sample_design_v2_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
                allow_raw_population_ast=allow_legacy_replay,
            )
        elif workflow == "strategy_model_evidence_v2":
            normalized = _validate_strategy_model_evidence_v2_inputs(raw_inputs)
        elif workflow == "strategy_report_bundle_v2":
            normalized = _validate_strategy_report_bundle_v2_inputs(raw_inputs)
        elif workflow == "strategy_dsl_delivery":
            normalized = _validate_strategy_dsl_delivery_inputs(raw_inputs)
        elif workflow == "strategy_sample_design":
            normalized = _validate_strategy_sample_design_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "profit_calc":
            normalized = _validate_profit_workflow_inputs(raw_inputs, whitelist)
        elif workflow == "roll_rate_matrix":
            normalized = _validate_roll_rate_workflow_inputs(raw_inputs, whitelist)
        elif workflow == "limit_pricing_matrix":
            normalized = _validate_pricing_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "univariate_candidate_analysis":
            normalized = _validate_univariate_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "univariate_candidate_refinement":
            normalized = _validate_univariate_refinement_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "candidate_monthly_stability":
            normalized = _validate_candidate_monthly_stability_inputs(raw_inputs)
        elif workflow == "scorecard_band_build":
            normalized = _validate_scorecard_band_build_inputs(raw_inputs)
        elif workflow == "scorecard_cutoff_selection":
            normalized = _validate_scorecard_cutoff_selection_inputs(raw_inputs)
        elif workflow == "automatic_tree_candidate_build":
            normalized = _validate_automatic_tree_candidate_build_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "automatic_tree_apply":
            normalized = _validate_automatic_tree_apply_inputs(raw_inputs)
        elif workflow == "automatic_tree_leaf_materialization":
            normalized = _validate_automatic_tree_leaf_materialization_inputs(
                raw_inputs
            )
        elif workflow == "interactive_tree_revision":
            normalized = _validate_interactive_tree_revision_inputs(raw_inputs)
        elif workflow == "interactive_tree_frontier_group_materialization":
            normalized = (
                _validate_interactive_tree_frontier_group_materialization_inputs(
                    raw_inputs
                )
            )
        elif workflow == "interactive_tree_frontier_materialization":
            normalized = _validate_interactive_tree_frontier_materialization_inputs(
                raw_inputs
            )
        elif workflow == "voting_candidate_search":
            normalized = _validate_voting_candidate_search_inputs(raw_inputs)
        elif workflow == "voting_candidate_build_from_search":
            normalized = _validate_voting_candidate_build_from_search_inputs(
                raw_inputs
            )
        elif workflow == "voting_candidate_build":
            normalized = _validate_voting_candidate_build_inputs(raw_inputs)
        elif workflow == "cross_matrix_candidate_search":
            normalized = _validate_cross_matrix_candidate_search_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "cross_matrix_candidate_build_from_search":
            normalized = (
                _validate_cross_matrix_candidate_build_from_search_inputs(
                    raw_inputs
                )
            )
        elif workflow == "cross_rule_search":
            normalized = _validate_cross_rule_search_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "cross_rule_candidate_build_from_search":
            normalized = _validate_cross_rule_candidate_build_inputs(
                raw_inputs
            )
        elif workflow == "cross_matrix_analysis":
            normalized = _validate_cross_matrix_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "cross_matrix_cell_selection":
            normalized = _validate_cross_matrix_cell_selection_inputs(raw_inputs)
        elif workflow == "strategy_pool_stability":
            normalized = _validate_strategy_pool_stability_inputs(raw_inputs)
        elif workflow == "strategy_impact_cube":
            normalized = _validate_strategy_impact_cube_inputs(
                raw_inputs,
                whitelist,
            )
        elif workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS:
            normalized = _validate_strategy_pool_impact_inputs(
                raw_inputs,
                whitelist,
            )
        elif workflow in _STRATEGY_POOL_APPLY_WORKFLOWS:
            normalized = _validate_strategy_pool_apply_inputs(raw_inputs)
        elif workflow in _STRATEGY_POOL_MATERIALIZE_WORKFLOWS:
            normalized = _validate_strategy_pool_materialize_inputs(raw_inputs)
        elif workflow in _STRATEGY_POOL_VALIDATION_WORKFLOWS:
            normalized = _validate_strategy_pool_validation_inputs(raw_inputs)
        elif workflow in _STRATEGY_POOL_WORKFLOWS:
            normalized = _validate_strategy_pool_workflow_inputs(
                workflow,
                raw_inputs,
            )
        else:  # pragma: no cover - guarded by STANDARD_STRATEGY_WORKFLOWS
            raise _DraftValidationError(f"不支持的标准 Workflow：{workflow}。")
    except _DraftValidationError as exc:
        return _invalid(str(exc), code=exc.code, fields=exc.fields)

    draft = StandardWorkflowRequestDraft(
        workflow=workflow,
        workflow_inputs=normalized,
    )
    return _ValidationOutcome(
        StrategyRequestCompilation(
            draft=draft,
            clarification=None,
            confirmation=strategy_request_confirmation_text(draft),
        ),
        True,
    )


def _validate_strategy_project_context_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only user-owned project-context facts.

    Repository heads, message identities, evidence references, metrics and
    availability conclusions remain platform-owned and are injected later by
    the Agent/Tool boundary.
    """

    workflow = "strategy_project_context"
    allowed = {
        "as_of",
        "scope",
        "business_context",
        "explicit_unavailable",
        "external_report_filenames",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if "as_of" not in inputs:
        raise _DraftValidationError(
            f"{workflow} 缺少 as_of；请明确本次项目现状的截止日期。"
        )
    as_of = _required_text(inputs["as_of"], name=f"{workflow} as_of")
    try:
        parsed_as_of = date.fromisoformat(as_of)
    except ValueError as exc:
        raise _DraftValidationError(
            f"{workflow} as_of 必须是 YYYY-MM-DD ISO 日期。"
        ) from exc
    if parsed_as_of.isoformat() != as_of:
        raise _DraftValidationError(
            f"{workflow} as_of 必须是 YYYY-MM-DD ISO 日期。"
        )

    normalized: dict[str, Any] = {"as_of": as_of}
    if "scope" in inputs:
        raw_scope = inputs["scope"]
        if raw_scope is None:
            normalized["scope"] = None
        else:
            scope = _required_text(raw_scope, name=f"{workflow} scope")
            if len(scope) > 4000:
                raise _DraftValidationError(f"{workflow} scope 最多 4000 个字符。")
            normalized["scope"] = scope

    raw_context = inputs.get("business_context", {})
    if not isinstance(raw_context, Mapping) or len(raw_context) > 50:
        raise _DraftValidationError(
            f"{workflow} business_context 必须是最多 50 个字段的对象。"
        )
    context: dict[str, str | None] = {}
    for raw_path, raw_value in raw_context.items():
        if not isinstance(raw_path, str):
            raise _DraftValidationError(
                f"{workflow} business_context 字段名必须是文本路径。"
            )
        field_path = raw_path.strip()
        if (
            not _PROJECT_CONTEXT_FIELD_PATH_RE.fullmatch(field_path)
            or len(field_path) > 256
        ):
            raise _DraftValidationError(
                f"{workflow} business_context 字段路径无效：{raw_path!r}。"
            )
        if raw_value is None:
            context[field_path] = None
            continue
        value = _required_text(
            raw_value,
            name=f"{workflow} business_context.{field_path}",
        )
        if len(value) > 4000:
            raise _DraftValidationError(
                f"{workflow} business_context.{field_path} 最多 4000 个字符。"
            )
        context[field_path] = value
    normalized["business_context"] = context

    normalized["explicit_unavailable"] = _project_context_text_list(
        inputs.get("explicit_unavailable", []),
        name=f"{workflow} explicit_unavailable",
        maximum=100,
        field_paths=True,
    )
    normalized["external_report_filenames"] = _project_context_text_list(
        inputs.get("external_report_filenames", []),
        name=f"{workflow} external_report_filenames",
        maximum=20,
        field_paths=False,
    )
    return normalized


def _validate_strategy_report_bundle_v2_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the only two caller-owned report controls."""

    workflow = "strategy_report_bundle_v2"
    allowed = {"title", "status"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise _DraftValidationError(
            f"{workflow} workflow_inputs 只允许 title/status；平台字段 "
            + "、".join(unexpected)
            + " 由 Agent 在计划创建时绑定。",
            code="strategy_report_bundle_v2_platform_binding_forbidden",
            fields=tuple(unexpected),
        )

    title = inputs.get("title", _STRATEGY_REPORT_DEFAULT_TITLE)
    if not isinstance(title, str) or not title.strip() or "\x00" in title:
        raise _DraftValidationError(
            "strategy_report_bundle_v2 title 必须是非空文本。",
            code="strategy_report_bundle_v2_title_invalid",
            fields=("title",),
        )
    title = title.strip()
    if len(title) > 200:
        raise _DraftValidationError(
            "strategy_report_bundle_v2 title 最多 200 个字符。",
            code="strategy_report_bundle_v2_title_invalid",
            fields=("title",),
        )

    status = inputs.get("status", _STRATEGY_REPORT_DEFAULT_STATUS)
    if not isinstance(status, str) or status not in _STRATEGY_REPORT_STATUSES:
        raise _DraftValidationError(
            "strategy_report_bundle_v2 status 只能是 draft、partial 或 final。",
            code="strategy_report_bundle_v2_status_invalid",
            fields=("status",),
        )
    return {"title": title, "status": status}


def _validate_strategy_dsl_delivery_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, str]:
    """Accept only an optional user-owned strategy identifier."""

    workflow = "strategy_dsl_delivery"
    allowed = {"strategy_id"}
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise _DraftValidationError(
            f"{workflow} workflow_inputs 只允许 strategy_id；平台字段 "
            + "、".join(unexpected)
            + " 由 Agent 在计划创建时绑定。",
            code="strategy_dsl_delivery_platform_binding_forbidden",
            fields=tuple(unexpected),
        )
    if "strategy_id" not in inputs:
        return {}
    strategy_id = _required_text(
        inputs["strategy_id"],
        name="strategy_dsl_delivery strategy_id",
    )
    if (
        len(strategy_id) > 128
        or _STRATEGY_DSL_DELIVERY_STRATEGY_ID_RE.fullmatch(strategy_id) is None
    ):
        raise _DraftValidationError(
            "strategy_dsl_delivery strategy_id 必须是完整的 strategy-* ID。",
            code="strategy_dsl_delivery_strategy_id_invalid",
            fields=("strategy_id",),
        )
    return {"strategy_id": strategy_id}


def _project_context_text_list(
    value: object,
    *,
    name: str,
    maximum: int,
    field_paths: bool,
) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise _DraftValidationError(f"{name} 必须是数组。")
    if len(value) > maximum:
        raise _DraftValidationError(f"{name} 最多包含 {maximum} 项。")
    normalized: list[str] = []
    for raw_item in value:
        item = _required_text(raw_item, name=name)
        if len(item) > 512:
            raise _DraftValidationError(f"{name} 每项最多 512 个字符。")
        if field_paths:
            if (
                len(item) > 256
                or not _PROJECT_CONTEXT_FIELD_PATH_RE.fullmatch(item)
            ):
                raise _DraftValidationError(f"{name} 包含无效字段路径：{item!r}。")
        else:
            normalized_path = item.replace("\\", "/")
            parts = normalized_path.split("/")
            if (
                normalized_path.startswith("/")
                or any(part in {"", ".", ".."} for part in parts)
                or "\x00" in item
            ):
                raise _DraftValidationError(f"{name} 包含不安全的相对文件名。")
            item = normalized_path
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise _DraftValidationError(f"{name} 不能包含重复项。")
    return normalized


def _validate_strategy_sample_design_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    """Validate only the sample-boundary facts explicitly owned by the user."""

    workflow = "strategy_sample_design"
    allowed = {
        "performance_window_status",
        "performance_window_days",
        "observation_window_status",
        "observation_start",
        "observation_end",
        "maturity_status",
        "target_bad_value",
        "split_col",
        "development_values",
        "validation_values",
        "oot_values",
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    required_controls = {*_SAMPLE_DESIGN_STATUS_VALUES, "target_bad_value"}
    missing_controls = sorted(required_controls - set(inputs))
    if missing_controls:
        raise _DraftValidationError(
            f"{workflow} 缺少必需口径：" + "、".join(missing_controls) + "。"
        )

    normalized: dict[str, Any] = {}
    for field, values in _SAMPLE_DESIGN_STATUS_VALUES.items():
        value = inputs[field]
        if not isinstance(value, str) or value not in values:
            raise _DraftValidationError(
                f"{workflow} {field} 只能是：" + "、".join(sorted(values)) + "。"
            )
        normalized[field] = value

    target_bad_value = inputs["target_bad_value"]
    if (
        isinstance(target_bad_value, bool)
        or not isinstance(target_bad_value, (int, float))
        or not math.isfinite(float(target_bad_value))
        or float(target_bad_value) not in {0.0, 1.0}
    ):
        raise _DraftValidationError(
            f"{workflow} target_bad_value 必须是整数 0 或 1，不能是布尔值。"
        )
    normalized["target_bad_value"] = int(target_bad_value)

    performance_status = normalized["performance_window_status"]
    if performance_status == "provided":
        days = inputs.get("performance_window_days")
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise _DraftValidationError(
                f"{workflow} performance_window_days 必须是正整数。"
            )
        normalized["performance_window_days"] = days
    elif "performance_window_days" in inputs:
        raise _DraftValidationError(
            f"{workflow} 表现窗 unavailable 时不能填写 performance_window_days。"
        )

    observation_status = normalized["observation_window_status"]
    observation_fields = {"observation_start", "observation_end"}
    if observation_status == "provided":
        missing = sorted(observation_fields - set(inputs))
        if missing:
            raise _DraftValidationError(
                f"{workflow} 观察窗 provided 时缺少：" + "、".join(missing) + "。"
            )
        parsed_dates: dict[str, date] = {}
        for field in sorted(observation_fields):
            value = inputs[field]
            if not isinstance(value, str):
                raise _DraftValidationError(f"{workflow} {field} 必须是 ISO 日期。")
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise _DraftValidationError(
                    f"{workflow} {field} 必须是 YYYY-MM-DD ISO 日期。"
                ) from exc
            if parsed.isoformat() != value:
                raise _DraftValidationError(
                    f"{workflow} {field} 必须是 YYYY-MM-DD ISO 日期。"
                )
            parsed_dates[field] = parsed
            normalized[field] = value
        if parsed_dates["observation_start"] > parsed_dates["observation_end"]:
            raise _DraftValidationError(
                f"{workflow} observation_start 不能晚于 observation_end。"
            )
    else:
        unexpected = sorted(observation_fields & set(inputs))
        if unexpected:
            raise _DraftValidationError(
                f"{workflow} 观察窗 unavailable 时不能填写："
                + "、".join(unexpected)
                + "。"
            )

    split_fields = {
        "split_col",
        "development_values",
        "validation_values",
        "oot_values",
    }
    supplied_split_fields = split_fields & set(inputs)
    if supplied_split_fields and supplied_split_fields != split_fields:
        missing = sorted(split_fields - supplied_split_fields)
        raise _DraftValidationError(
            f"{workflow} 指定切分时必须同时提供 split_col 与三组值；缺少："
            + "、".join(missing)
            + "。"
        )
    if supplied_split_fields:
        split_col = _workflow_column(
            inputs["split_col"],
            name=f"{workflow} split_col",
            whitelist=whitelist,
        )
        if target_col is not None and split_col == target_col:
            raise _DraftValidationError(f"{workflow} split_col 不能使用目标列。")
        normalized["split_col"] = split_col
        value_sets: dict[str, set[tuple[str, object]]] = {}
        for field in (
            "development_values",
            "validation_values",
            "oot_values",
        ):
            values = _sample_design_value_sequence(
                inputs[field],
                name=field,
                minimum_items=1 if field == "development_values" else 0,
            )
            normalized[field] = values
            value_sets[field] = {_sample_design_value_identity(item) for item in values}
        overlaps: list[str] = []
        for left, right in (
            ("development_values", "validation_values"),
            ("development_values", "oot_values"),
            ("validation_values", "oot_values"),
        ):
            if value_sets[left] & value_sets[right]:
                overlaps.append(f"{left}/{right}")
        if overlaps:
            raise _DraftValidationError(
                f"{workflow} 三组 split values 必须互不重叠："
                + "、".join(overlaps)
                + "。"
            )

    for field in (
        "month_col",
        "weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        if field not in inputs:
            continue
        column = _workflow_column(
            inputs[field],
            name=f"{workflow} {field}",
            whitelist=whitelist,
        )
        if target_col is not None and column == target_col:
            raise _DraftValidationError(f"{workflow} {field} 不能使用目标列。")
        normalized[field] = column

    bound_columns = [
        normalized[field]
        for field in (
            "split_col",
            "month_col",
            "weight_col",
            "loan_amount_col",
            "overdue_amount_col",
        )
        if field in normalized
    ]
    if len(bound_columns) != len(set(bound_columns)):
        raise _DraftValidationError(
            f"{workflow} 切分、月份、权重和金额字段必须彼此不同。"
        )

    if "drop_nan_labels" in inputs:
        value = inputs["drop_nan_labels"]
        if not isinstance(value, bool):
            raise _DraftValidationError(
                f"{workflow} drop_nan_labels 必须是布尔值。"
            )
        normalized["drop_nan_labels"] = value
    return normalized


def _validate_strategy_sample_design_v2_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    allow_raw_population_ast: bool = False,
) -> dict[str, Any]:
    """Validate the user-owned portion of the V2 dual-population request.

    Relationship is business semantics and must be explicit. Runtime identity,
    compatibility references, scope and diagnostic policy are injected later
    from task context. Fresh requests use a deliberately small population DTO;
    persisted replay may opt into the historical canonical predicate AST.
    """

    workflow = "strategy_sample_design_v2"
    allowed = {
        "target_bad_value",
        "drop_nan_labels",
        "relationship",
        "approval_population",
        "risk_population",
        "partitioning",
        "maturity",
        "performance_window",
        "observation_window",
        "field_bindings",
        "historical_score",
    }
    unexpected = tuple(sorted(set(inputs) - allowed))
    if unexpected:
        raise _DraftValidationError(
            f"{workflow} workflow_inputs 只能包含用户拥有的样本口径；平台字段不允许："
            + "、".join(unexpected)
            + "。",
            code="strategy_sample_design_v2_platform_binding_forbidden",
            fields=unexpected,
        )
    missing = tuple(sorted(allowed - set(inputs)))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少必需口径：" + "、".join(missing) + "。",
            fields=missing,
        )

    bad_value = inputs["target_bad_value"]
    if isinstance(bad_value, bool) or not isinstance(bad_value, int) or bad_value not in {0, 1}:
        raise _DraftValidationError(
            f"{workflow} target_bad_value 必须是整数 0 或 1。",
            fields=("target_bad_value",),
        )
    drop_missing = inputs["drop_nan_labels"]
    if not isinstance(drop_missing, bool):
        raise _DraftValidationError(
            f"{workflow} drop_nan_labels 必须是布尔值。",
            fields=("drop_nan_labels",),
        )
    relationship = inputs["relationship"]
    if relationship not in {"nested_same_cohort", "parallel_time_cohorts"}:
        raise _DraftValidationError(
            f"{workflow} relationship 只能是 nested_same_cohort 或 "
            "parallel_time_cohorts。",
            fields=("relationship",),
        )

    approval = _validate_sample_v2_population(
        inputs["approval_population"],
        name="approval_population",
        whitelist=whitelist,
        target_col=target_col,
        allow_raw_ast=allow_raw_population_ast,
    )
    risk = _validate_sample_v2_population(
        inputs["risk_population"],
        name="risk_population",
        whitelist=whitelist,
        target_col=target_col,
        allow_raw_ast=allow_raw_population_ast,
    )

    partitioning = _validate_sample_v2_partitioning(
        inputs["partitioning"],
        whitelist=whitelist,
        target_col=target_col,
        allow_recursive_ast=allow_raw_population_ast,
    )
    maturity = _validate_sample_v2_maturity(inputs["maturity"])
    performance = _validate_sample_v2_performance_window(
        inputs["performance_window"]
    )
    observation = _validate_sample_v2_observation_window(
        inputs["observation_window"]
    )
    if maturity["status"] in {"confirmed_matured", "not_matured"}:
        if performance["status"] != "provided":
            raise _DraftValidationError(
                f"{workflow} 已评估成熟度必须同时提供表现窗。",
                fields=("maturity", "performance_window"),
            )
        if maturity["performance_window_days"] != performance["days"]:
            raise _DraftValidationError(
                f"{workflow} maturity 与 performance_window 天数必须一致。",
                fields=("maturity.performance_window_days", "performance_window.days"),
            )
    fields = _validate_sample_v2_field_bindings(
        inputs["field_bindings"],
        whitelist=whitelist,
        target_col=target_col,
    )
    if (
        partitioning["method"] == "time_ranges"
        and (
            fields["time_field"] is None
            or partitioning["column"] != fields["time_field"]
        )
    ):
        raise _DraftValidationError(
            f"{workflow} time_ranges.column 必须与非空的 "
            "field_bindings.time_field 完全相同。",
            fields=("partitioning.column", "field_bindings.time_field"),
        )
    if maturity["status"] in {"confirmed_matured", "not_matured"} and fields["time_field"] is None:
        raise _DraftValidationError(
            f"{workflow} 已评估成熟度需要 time_field。",
            fields=("field_bindings.time_field",),
        )
    if observation["status"] == "provided" and fields["time_field"] is None:
        raise _DraftValidationError(
            f"{workflow} 已提供观察窗需要 time_field。",
            fields=("field_bindings.time_field",),
        )
    historical = _validate_sample_v2_historical_score(
        inputs["historical_score"],
        whitelist=whitelist,
        target_col=target_col,
    )
    return {
        "target_bad_value": bad_value,
        "drop_nan_labels": drop_missing,
        "relationship": relationship,
        "approval_population": approval,
        "risk_population": risk,
        "partitioning": partitioning,
        "maturity": maturity,
        "performance_window": performance,
        "observation_window": observation,
        "field_bindings": fields,
        "historical_score": historical,
    }


def _validate_strategy_model_evidence_v2_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    if inputs:
        fields = tuple(sorted(inputs))
        raise _DraftValidationError(
            "strategy_model_evidence_v2 workflow_inputs 必须是空对象；"
            "SampleDesign 与认证单变量候选引用全部由当前 task 绑定。",
            code="strategy_model_evidence_v2_platform_binding_forbidden",
            fields=fields,
        )
    return {}


def _validate_sample_v2_population(
    value: object,
    *,
    name: str,
    whitelist: tuple[str, ...],
    target_col: str | None,
    allow_raw_ast: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"inclusion", "exclusion"}:
        raise _DraftValidationError(
            f"{name} 必须且只能包含 inclusion 与 exclusion。",
            fields=(name,),
        )
    normalized: dict[str, Any] = {}
    for field in ("inclusion", "exclusion"):
        predicate = value[field]
        if predicate is None:
            normalized[field] = None
            continue
        if not allow_raw_ast:
            predicate = _sample_v2_population_filter_to_ast(
                predicate,
                name=f"{name}.{field}",
                whitelist=whitelist,
                target_col=target_col,
            )
        try:
            canonical = canonicalize_predicate(
                predicate,
                whitelist,
                max_nodes=256,
                max_depth=12,
            )
        except PredicateAstError as exc:
            raise _DraftValidationError(
                f"{name}.{field} 不是受支持的严格 predicate AST：{exc}",
                fields=(f"{name}.{field}",),
            ) from exc
        if target_col is not None and target_col in canonical.required_columns:
            raise _DraftValidationError(
                f"{name}.{field} 不能用目标列定义样本总体。",
                fields=(f"{name}.{field}",),
            )
        normalized[field] = canonical.canonical
    return normalized


def _sample_v2_population_filter_to_ast(
    value: object,
    *,
    name: str,
    whitelist: tuple[str, ...],
    target_col: str | None,
) -> dict[str, Any]:
    """Compile the bounded fresh-request population DTO to canonical AST data."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"match", "conditions"}
        or value.get("match") not in {"all", "any"}
    ):
        raise _DraftValidationError(
            f"{name} 必须是只含 match 与 conditions 的受限条件对象；"
            "fresh 请求不能直接提交 predicate AST。",
            code="strategy_sample_design_v2_population_dto_required",
            fields=(name,),
        )
    conditions = value["conditions"]
    if (
        not isinstance(conditions, Sequence)
        or isinstance(conditions, str | bytes)
        or not 1 <= len(conditions) <= 8
    ):
        raise _DraftValidationError(
            f"{name}.conditions 必须包含 1 到 8 个简单条件。",
            code="strategy_sample_design_v2_population_dto_required",
            fields=(name,),
        )

    compiled: list[dict[str, Any]] = []
    identities: set[str] = set()
    comparison_ops = {"eq", "ne", "gt", "gte", "lt", "lte"}
    null_ops = {"is_null", "is_not_null"}
    for index, condition in enumerate(conditions):
        condition_name = f"{name}.conditions[{index}]"
        if not isinstance(condition, Mapping):
            raise _DraftValidationError(
                f"{condition_name} 必须是简单条件对象。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        operator = condition.get("operator")
        expected = (
            {"column", "operator", "value"}
            if operator in comparison_ops
            else {"column", "operator"}
            if operator in null_ops
            else set()
        )
        if not expected or set(condition) != expected:
            raise _DraftValidationError(
                f"{condition_name} 只支持简单比较或空值判断。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        column = _workflow_column(
            condition["column"],
            name=f"{condition_name}.column",
            whitelist=whitelist,
        )
        if target_col is not None and column == target_col:
            raise _DraftValidationError(
                f"{condition_name} 不能使用目标列定义样本总体。",
                fields=(name,),
            )
        if operator in comparison_ops:
            literal = condition["value"]
            if (
                literal is None
                or not isinstance(literal, str | int | float | bool)
                or isinstance(literal, float) and not math.isfinite(literal)
                or isinstance(literal, str) and not literal.strip()
            ):
                raise _DraftValidationError(
                    f"{condition_name}.value 必须是有限的非空字符串、数字或布尔值。",
                    code="strategy_sample_design_v2_population_dto_required",
                    fields=(name,),
                )
            node = {
                "op": operator,
                "left": {"column": column},
                "right": {"literal": literal},
            }
        else:
            node = {"op": operator, "arg": {"column": column}}
        identity = json.dumps(
            node,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity in identities:
            raise _DraftValidationError(
                f"{name}.conditions 不能包含重复条件。",
                code="strategy_sample_design_v2_population_dto_required",
                fields=(name,),
            )
        identities.add(identity)
        compiled.append(node)
    if len(compiled) == 1:
        return compiled[0]
    return {
        "op": "and" if value["match"] == "all" else "or",
        "args": compiled,
    }


def _validate_sample_v2_partitioning(
    value: object,
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
    allow_recursive_ast: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _DraftValidationError("partitioning 必须是对象。", fields=("partitioning",))
    method = value.get("method")
    if method == "time_ranges":
        return _validate_sample_v2_time_ranges(
            value,
            whitelist=whitelist,
            target_col=target_col,
        )
    if method != "predicate_ast" or set(value) != {"method", "selectors"}:
        raise _DraftValidationError(
            "partitioning 必须是严格 predicate_ast 或 time_ranges 对象。",
            fields=("partitioning",),
        )
    selectors = value["selectors"]
    partition_names = ("development", "validation", "oot")
    if not isinstance(selectors, Mapping) or set(selectors) != set(partition_names):
        raise _DraftValidationError(
            "partitioning.selectors 必须完整包含 development、validation、oot。",
            fields=("partitioning.selectors",),
        )
    normalized: dict[str, Any] = {}
    for partition in partition_names:
        if (
            not allow_recursive_ast
            and not _sample_v2_fresh_partition_selector_shape(
                selectors[partition]
            )
        ):
            raise _DraftValidationError(
                f"partitioning.selectors.{partition} 只能是单个简单条件，"
                "或由同一种 and/or 连接的单层 2 到 8 个简单条件；"
                "fresh 请求禁止嵌套逻辑、not 和列间比较。",
                fields=(f"partitioning.selectors.{partition}",),
            )
        try:
            canonical = canonicalize_predicate(
                selectors[partition],
                whitelist,
                max_nodes=256,
                max_depth=12,
            )
        except PredicateAstError as exc:
            raise _DraftValidationError(
                f"partitioning.selectors.{partition} 不是严格 predicate AST：{exc}",
                fields=(f"partitioning.selectors.{partition}",),
            ) from exc
        if (
            target_col is not None
            and target_col in canonical.required_columns
        ):
            raise _DraftValidationError(
                "partitioning 不能使用目标列。",
                fields=(f"partitioning.selectors.{partition}",),
            )
        normalized[partition] = canonical.canonical
    return {"method": "predicate_ast", "selectors": normalized}


def _sample_v2_fresh_partition_selector_shape(value: object) -> bool:
    """Accept only an unambiguous fresh selector surface.

    Historical persisted requests may still replay the complete recursive
    predicate AST.  Fresh requests deliberately expose one leaf or one flat
    logical row so the user's words can be grounded without inferring
    parentheses or operator precedence.
    """

    if _sample_v2_fresh_partition_leaf_shape(value):
        return True
    if not isinstance(value, Mapping) or set(value) != {"op", "args"}:
        return False
    if value.get("op") not in {"and", "or"}:
        return False
    args = value.get("args")
    return (
        isinstance(args, Sequence)
        and not isinstance(args, str | bytes | bytearray)
        and 2 <= len(args) <= 8
        and all(_sample_v2_fresh_partition_leaf_shape(arg) for arg in args)
    )


def _sample_v2_fresh_partition_leaf_shape(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    op = value.get("op")
    if op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        return (
            set(value) == {"op", "left", "right"}
            and isinstance(value.get("left"), Mapping)
            and set(value["left"]) == {"column"}
            and isinstance(value.get("right"), Mapping)
            and set(value["right"]) == {"literal"}
        )
    if op in {"is_null", "is_not_null"}:
        return (
            set(value) == {"op", "arg"}
            and isinstance(value.get("arg"), Mapping)
            and set(value["arg"]) == {"column"}
        )
    return False


def _simple_partition_equality(
    predicate: Mapping[str, Any],
    *,
    name: str,
) -> tuple[str, object]:
    if (
        set(predicate) != {"op", "left", "right"}
        or predicate.get("op") != "eq"
        or not isinstance(predicate.get("left"), Mapping)
        or set(predicate["left"]) != {"column"}
        or not isinstance(predicate.get("right"), Mapping)
        or set(predicate["right"]) != {"literal"}
    ):
        raise _DraftValidationError(
            f"{name} 当前必须是 column == literal 的简单等值 predicate。",
            code="strategy_sample_design_v2_native_bootstrap_required",
            fields=(name,),
        )
    literal = predicate["right"]["literal"]
    if literal is None or isinstance(literal, Mapping | Sequence) and not isinstance(literal, str):
        raise _DraftValidationError(
            f"{name} literal 必须是非空标量。",
            fields=(name,),
        )
    return str(predicate["left"]["column"]), literal


def _validate_sample_v2_time_ranges(
    value: Mapping[str, Any],
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
) -> dict[str, Any]:
    if set(value) != {"method", "column", "ranges"}:
        raise _DraftValidationError("time_ranges 字段不完整。", fields=("partitioning",))
    column = _workflow_column(value["column"], name="partitioning.column", whitelist=whitelist)
    if target_col is not None and column == target_col:
        raise _DraftValidationError("partitioning 不能使用目标列。", fields=("partitioning.column",))
    ranges = value["ranges"]
    if not isinstance(ranges, Mapping) or set(ranges) != {"development", "validation", "oot"}:
        raise _DraftValidationError("time_ranges.ranges 必须完整包含三组。", fields=("partitioning.ranges",))
    normalized_ranges: dict[str, dict[str, str | None]] = {}
    for name, raw in ranges.items():
        if not isinstance(raw, Mapping) or set(raw) != {"start", "end"}:
            raise _DraftValidationError(f"partitioning.ranges.{name} 字段无效。")
        bounds = [_strict_optional_iso_date(raw[field], f"partitioning.ranges.{name}.{field}") for field in ("start", "end")]
        if bounds == [None, None] or bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
            raise _DraftValidationError(f"partitioning.ranges.{name} 日期范围无效。")
        normalized_ranges[str(name)] = {"start": bounds[0], "end": bounds[1]}
    return {
        "method": "time_ranges",
        "column": column,
        "ranges": normalized_ranges,
    }


def _validate_sample_v2_maturity(value: object) -> dict[str, Any]:
    fields = {"status", "performance_window_days", "cutoff_date", "reason"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _DraftValidationError("maturity 字段必须完整且精确。", fields=("maturity",))
    status = value["status"]
    if status not in {"confirmed_matured", "not_matured", "unknown", "unavailable"}:
        raise _DraftValidationError("maturity.status 不受支持。", fields=("maturity.status",))
    if status in {"confirmed_matured", "not_matured"}:
        days = value["performance_window_days"]
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise _DraftValidationError("maturity.performance_window_days 必须是正整数。")
        cutoff = _strict_iso_date(value["cutoff_date"], "maturity.cutoff_date")
        if status == "confirmed_matured":
            if value["reason"] is not None:
                raise _DraftValidationError("confirmed_matured 的 reason 必须为 null。")
            reason = None
        else:
            reason = _required_text(value["reason"], name="maturity.reason")
        return {"status": status, "performance_window_days": days, "cutoff_date": cutoff, "reason": reason}
    if value["performance_window_days"] is not None or value["cutoff_date"] is not None:
        raise _DraftValidationError("unknown/unavailable maturity 的天数与截止日必须为 null。")
    return {"status": status, "performance_window_days": None, "cutoff_date": None, "reason": _required_text(value["reason"], name="maturity.reason")}


def _validate_sample_v2_performance_window(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "days"}:
        raise _DraftValidationError("performance_window 字段必须完整且精确。")
    status = value["status"]
    if status not in {"provided", "unavailable"}:
        raise _DraftValidationError("performance_window.status 不受支持。")
    days = value["days"]
    if status == "provided":
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise _DraftValidationError("performance_window.days 必须是正整数。")
    elif days is not None:
        raise _DraftValidationError("unavailable performance_window.days 必须为 null。")
    return {"status": status, "days": days}


def _validate_sample_v2_observation_window(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "start", "end"}:
        raise _DraftValidationError("observation_window 字段必须完整且精确。")
    status = value["status"]
    if status not in {"provided", "unavailable"}:
        raise _DraftValidationError("observation_window.status 不受支持。")
    if status == "provided":
        start = _strict_iso_date(value["start"], "observation_window.start")
        end = _strict_iso_date(value["end"], "observation_window.end")
        if start > end:
            raise _DraftValidationError("observation_window.start 不能晚于 end。")
    else:
        if value["start"] is not None or value["end"] is not None:
            raise _DraftValidationError("unavailable observation_window 边界必须为 null。")
        start = end = None
    return {"status": status, "start": start, "end": end}


def _validate_sample_v2_field_bindings(
    value: object,
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
) -> dict[str, str | None]:
    fields = {
        "entity_field", "time_field", "group_field", "month_field",
        "weight_field", "loan_amount_field", "overdue_amount_field",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise _DraftValidationError("field_bindings 字段必须完整且精确。", fields=("field_bindings",))
    normalized: dict[str, str | None] = {}
    for field in sorted(fields):
        raw = value[field]
        if raw is None:
            normalized[field] = None
            continue
        column = _workflow_column(raw, name=f"field_bindings.{field}", whitelist=whitelist)
        if target_col is not None and column == target_col:
            raise _DraftValidationError(f"field_bindings.{field} 不能使用目标列。")
        normalized[field] = column
    return normalized


def _validate_sample_v2_historical_score(
    value: object,
    *,
    whitelist: tuple[str, ...],
    target_col: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"status", "column", "direction", "reason"}:
        raise _DraftValidationError("historical_score 字段必须完整且精确。")
    status = value["status"]
    if status not in {"available", "unavailable", "not_applicable"}:
        raise _DraftValidationError("historical_score.status 不受支持。")
    if status == "available":
        column = _workflow_column(value["column"], name="historical_score.column", whitelist=whitelist)
        if target_col is not None and column == target_col:
            raise _DraftValidationError("historical_score.column 不能使用目标列。")
        direction = value["direction"]
        if direction not in {"higher_is_riskier", "lower_is_riskier"}:
            raise _DraftValidationError("historical_score.direction 不受支持。")
        if value["reason"] is not None:
            raise _DraftValidationError("available historical_score.reason 必须为 null。")
        reason = None
    else:
        if value["column"] is not None or value["direction"] is not None:
            raise _DraftValidationError("非 available historical_score 的 column/direction 必须为 null。")
        column = direction = None
        reason = _required_text(value["reason"], name="historical_score.reason")
    return {"status": status, "column": column, "direction": direction, "reason": reason}


def _strict_iso_date(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise _DraftValidationError(f"{name} 必须是 YYYY-MM-DD ISO 日期。")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _DraftValidationError(f"{name} 必须是 YYYY-MM-DD ISO 日期。") from exc
    if parsed.isoformat() != value:
        raise _DraftValidationError(f"{name} 必须是 YYYY-MM-DD ISO 日期。")
    return value


def _strict_optional_iso_date(value: object, name: str) -> str | None:
    return None if value is None else _strict_iso_date(value, name)


def _sample_design_value_sequence(
    value: object,
    *,
    name: str,
    minimum_items: int,
) -> list[object]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not minimum_items <= len(value) <= 100
    ):
        raise _DraftValidationError(
            f"{name} 必须是包含 {minimum_items} 到 100 个标量值的数组。"
        )
    normalized: list[object] = []
    identities: set[tuple[str, object]] = set()
    for item in value:
        if item is None or not isinstance(item, str | int | float | bool):
            raise _DraftValidationError(f"{name} 只能包含文本、布尔值或有限数字。")
        if isinstance(item, float) and not math.isfinite(item):
            raise _DraftValidationError(f"{name} 只能包含文本、布尔值或有限数字。")
        if isinstance(item, int) and not isinstance(item, bool) and abs(item) > 2**53 - 1:
            raise _DraftValidationError(f"{name} 中的整数超出精确 JSON 范围。")
        identity = _sample_design_value_identity(item)
        if identity in identities:
            raise _DraftValidationError(f"{name} 不能包含重复值。")
        identities.add(identity)
        normalized.append(item)
    return normalized


def _sample_design_value_identity(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        return ("number", Decimal(str(value)).normalize())
    return ("string", value)


def _validate_profit_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    allowed = {"segment_col", "ead_col", "pd_col", "profit_params"}
    _reject_workflow_fields(inputs, allowed, workflow="profit_calc")
    missing = sorted({"ead_col", "pd_col", "profit_params"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            "profit_calc 缺少字段：" + "、".join(missing) + "。"
        )
    params = inputs["profit_params"]
    if not isinstance(params, Mapping):
        raise _DraftValidationError("profit_calc 的 profit_params 必须是对象。")
    if any(not isinstance(key, str) for key in params):
        raise _DraftValidationError("profit_calc 的 profit_params 字段名必须是文本。")
    missing_params = sorted(_PROFIT_PARAMETER_FIELDS - set(params))
    unexpected_params = sorted(set(params) - _PROFIT_PARAMETER_FIELDS)
    if missing_params:
        raise _DraftValidationError(
            "profit_calc 的 profit_params 缺少字段：" + "、".join(missing_params) + "。"
        )
    if unexpected_params:
        raise _DraftValidationError(
            "profit_calc 的 profit_params 包含不支持的字段："
            + "、".join(unexpected_params)
            + "。"
        )
    profit = _optional_profit(
        {
            "profit": {
                "ead_col": inputs["ead_col"],
                "pd_col": inputs["pd_col"],
                **dict(params),
            }
        },
        whitelist,
    )
    assert profit is not None
    normalized: dict[str, Any] = {
        "ead_col": profit.pop("ead_col"),
        "pd_col": profit.pop("pd_col"),
        "profit_params": profit,
    }
    if "segment_col" in inputs:
        normalized["segment_col"] = _workflow_column(
            inputs["segment_col"],
            name="profit_calc segment_col",
            whitelist=whitelist,
        )
    return normalized


def _validate_roll_rate_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    allowed = {
        "id_col",
        "time_col",
        "status_col",
        "states",
        "balance_col",
        "observation_semantics",
    }
    _reject_workflow_fields(inputs, allowed, workflow="roll_rate_matrix")
    required = {"id_col", "time_col", "status_col", "states"}
    missing = sorted(required - set(inputs))
    if missing:
        raise _DraftValidationError(
            "roll_rate_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    normalized = {
        key: _workflow_column(
            inputs[key], name=f"roll_rate_matrix {key}", whitelist=whitelist
        )
        for key in ("id_col", "time_col", "status_col")
    }
    if len(set(normalized.values())) != len(normalized):
        raise _DraftValidationError(
            "roll_rate_matrix 的 id_col、time_col、status_col 必须互不相同。"
        )
    states = inputs["states"]
    if (
        not isinstance(states, Sequence)
        or isinstance(states, str | bytes | bytearray)
        or not 2 <= len(states) <= 50
    ):
        raise _DraftValidationError(
            "roll_rate_matrix states 必须是包含 2 到 50 个状态的有序数组。"
        )
    normalized_states = [
        _required_text(state, name="roll_rate_matrix states 状态") for state in states
    ]
    if len(set(normalized_states)) != len(normalized_states):
        raise _DraftValidationError("roll_rate_matrix states 不能包含重复状态。")
    normalized["states"] = normalized_states
    semantics = inputs.get("observation_semantics", "adjacent_observation")
    if semantics != "adjacent_observation":
        raise _DraftValidationError(
            "roll_rate_matrix observation_semantics 只能是 adjacent_observation；"
            "固定月末快照迁徙应使用 portfolio Workflow。"
        )
    normalized["observation_semantics"] = semantics
    if "balance_col" in inputs:
        balance_col = _workflow_column(
            inputs["balance_col"],
            name="roll_rate_matrix balance_col",
            whitelist=whitelist,
        )
        if balance_col in {
            normalized["id_col"],
            normalized["time_col"],
            normalized["status_col"],
        }:
            raise _DraftValidationError(
                "roll_rate_matrix balance_col 不能复用 ID、时间或状态列。"
            )
        normalized["balance_col"] = balance_col
    return normalized


def _validate_pricing_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    allowed = {
        "score_col",
        "pd_col",
        "target_col",
        "band_edges",
        "n_bands",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
        "strategy_id",
        "drop_nan_labels",
    }
    _reject_workflow_fields(inputs, allowed, workflow="limit_pricing_matrix")
    required = {
        "score_col",
        "limit_grid",
        "rate_grid",
        "lgd",
        "funding_rate",
        "term_months",
        "cost_per_loan",
        "el_ead_max",
    }
    missing = sorted(required - set(inputs))
    if missing:
        raise _DraftValidationError(
            "limit_pricing_matrix 缺少字段：" + "、".join(missing) + "。"
        )
    has_pd = "pd_col" in inputs
    has_target = "target_col" in inputs
    if has_pd == has_target:
        raise _DraftValidationError(
            "limit_pricing_matrix 的 pd_col 与 target_col 必须且只能二选一。"
        )
    has_edges = "band_edges" in inputs
    has_band_count = "n_bands" in inputs
    if has_edges == has_band_count:
        raise _DraftValidationError(
            "limit_pricing_matrix 的 band_edges 与 n_bands 必须且只能二选一。"
        )

    normalized: dict[str, Any] = {
        "score_col": _workflow_column(
            inputs["score_col"],
            name="limit_pricing_matrix score_col",
            whitelist=whitelist,
        ),
        "limit_grid": _number_sequence(
            inputs["limit_grid"],
            name="limit_pricing_matrix limit_grid",
            minimum=0,
            exclusive_minimum=True,
            maximum_items=50,
        ),
        "rate_grid": _number_sequence(
            inputs["rate_grid"],
            name="limit_pricing_matrix rate_grid",
            minimum=0,
            maximum=1,
            maximum_items=50,
        ),
        "lgd": _bounded_number(
            inputs["lgd"], name="limit_pricing_matrix lgd", maximum=1
        ),
        "funding_rate": _bounded_number(
            inputs["funding_rate"],
            name="limit_pricing_matrix funding_rate",
            maximum=1,
        ),
        "cost_per_loan": _bounded_number(
            inputs["cost_per_loan"],
            name="limit_pricing_matrix cost_per_loan",
        ),
        "el_ead_max": _bounded_number(
            inputs["el_ead_max"],
            name="limit_pricing_matrix el_ead_max",
            maximum=1,
        ),
    }
    term_months = inputs["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or not 1 <= term_months <= 600
    ):
        raise _DraftValidationError(
            "limit_pricing_matrix term_months 必须是 1 到 600 的整数。"
        )
    normalized["term_months"] = term_months

    if has_pd:
        normalized["pd_col"] = _workflow_column(
            inputs["pd_col"],
            name="limit_pricing_matrix pd_col",
            whitelist=whitelist,
        )
    else:
        requested_target = _required_text(
            inputs["target_col"],
            name="limit_pricing_matrix target_col",
        )
        if target_col is None or requested_target != target_col:
            raise _DraftValidationError(
                "limit_pricing_matrix target_col 必须与任务当前确认的目标列一致。"
            )
        normalized["target_col"] = requested_target

    if has_edges:
        edges = _number_sequence(
            inputs["band_edges"],
            name="limit_pricing_matrix band_edges",
            minimum=None,
            maximum_items=51,
            minimum_items=2,
        )
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise _DraftValidationError(
                "limit_pricing_matrix band_edges 必须严格递增。"
            )
        normalized["band_edges"] = edges
        band_count = len(edges) - 1
    else:
        n_bands = inputs["n_bands"]
        if (
            isinstance(n_bands, bool)
            or not isinstance(n_bands, int)
            or not 1 <= n_bands <= 20
        ):
            raise _DraftValidationError(
                "limit_pricing_matrix n_bands 必须是 1 到 20 的整数。"
            )
        normalized["n_bands"] = n_bands
        band_count = n_bands
    if band_count * len(normalized["limit_grid"]) * len(normalized["rate_grid"]) > 2000:
        raise _DraftValidationError("limit_pricing_matrix 网格最多允许 2000 个组合。")
    if "strategy_id" in inputs:
        normalized["strategy_id"] = _required_text(
            inputs["strategy_id"],
            name="limit_pricing_matrix strategy_id",
        )
    if "drop_nan_labels" in inputs:
        if not isinstance(inputs["drop_nan_labels"], bool):
            raise _DraftValidationError(
                "limit_pricing_matrix drop_nan_labels 必须是布尔值。"
            )
        if has_pd:
            raise _DraftValidationError(
                "limit_pricing_matrix 使用 pd_col 时不会读取标签，"
                "请删除未使用的 drop_nan_labels。"
            )
        normalized["drop_nan_labels"] = inputs["drop_nan_labels"]
    return normalized


def _validate_univariate_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
    manual_features: Sequence[str] | None = None,
) -> dict[str, Any]:
    allowed = {
        "features",
        "methods",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    workflow = "univariate_candidate_analysis"
    _reject_workflow_fields(inputs, allowed, workflow=workflow)

    raw_features = inputs.get("features", [])
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or len(raw_features) > 50
    ):
        raise _DraftValidationError(
            f"{workflow} features 必须是最多 50 个字段的有序数组。"
        )
    features = [
        _workflow_column(
            value,
            name=f"{workflow} features",
            whitelist=whitelist,
        )
        for value in raw_features
    ]
    if len(features) != len(set(features)):
        raise _DraftValidationError(f"{workflow} features 不能包含重复字段。")
    if target_col is not None and target_col in features:
        raise _DraftValidationError(
            f"{workflow} features 不能包含目标列 {target_col}。"
        )

    methods_supplied = "methods" in inputs
    raw_methods = inputs.get("methods", [])
    if (
        not isinstance(raw_methods, Sequence)
        or isinstance(raw_methods, str | bytes | bytearray)
        or (
            methods_supplied
            and not 1 <= len(raw_methods) <= len(UNIVARIATE_BINNING_METHODS)
        )
    ):
        raise _DraftValidationError(
            f"{workflow} methods 必须包含 1 到 {len(UNIVARIATE_BINNING_METHODS)} 个分箱方法。"
        )
    methods = [
        _required_text(value, name=f"{workflow} methods") for value in raw_methods
    ]
    unknown_methods = sorted(set(methods) - set(UNIVARIATE_BINNING_METHODS))
    if unknown_methods:
        raise _DraftValidationError(
            f"{workflow} 不支持分箱方法：" + "、".join(unknown_methods) + "。"
        )
    if len(methods) != len(set(methods)):
        raise _DraftValidationError(f"{workflow} methods 不能包含重复方法。")

    bin_count = inputs.get("bin_count", 10)
    if (
        isinstance(bin_count, bool)
        or not isinstance(bin_count, int)
        or not 3 <= bin_count <= 20
    ):
        raise _DraftValidationError(f"{workflow} bin_count 必须是 3 到 20 的整数。")
    min_bin_pct = _bounded_number(
        inputs.get("min_bin_pct", 0.02),
        name=f"{workflow} min_bin_pct",
        maximum=0.5,
    )
    normalized: dict[str, Any] = {
        "features": features,
        "methods": methods,
        "bin_count": bin_count,
        "min_bin_pct": min_bin_pct,
        "sentinel_values": _sentinel_sequence(
            inputs.get("sentinel_values", []),
            name=f"{workflow} sentinel_values",
        ),
    }
    expected_manual_features = (
        features if manual_features is None else list(manual_features)
    )
    manual_breakpoints = _validate_manual_breakpoint_mapping(
        inputs.get("manual_breakpoints"),
        manual_requested="manual" in methods,
        expected_features=expected_manual_features,
        workflow=workflow,
    )
    if manual_breakpoints:
        normalized["manual_breakpoints"] = manual_breakpoints
    for field in ("loan_amount_col", "overdue_amount_col"):
        if field in inputs:
            normalized[field] = _workflow_column(
                inputs[field],
                name=f"{workflow} {field}",
                whitelist=whitelist,
            )
            if target_col is not None and normalized[field] == target_col:
                raise _DraftValidationError(f"{workflow} {field} 不能使用目标列。")
    if normalized.get("loan_amount_col") is not None and normalized.get(
        "loan_amount_col"
    ) == normalized.get("overdue_amount_col"):
        raise _DraftValidationError(
            f"{workflow} loan_amount_col 与 overdue_amount_col 必须是不同字段。"
        )
    return normalized


def _validate_cross_matrix_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    """Validate only user-owned controls for one explicit 2D matrix build."""

    workflow = "cross_matrix_analysis"
    axis_fields = {"x_feature", "x_method", "y_feature", "y_method"}
    derived_fields = {"features", "methods"}
    analysis_fields = {
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    _reject_workflow_fields(
        inputs,
        axis_fields | analysis_fields | derived_fields,
        workflow=workflow,
    )
    missing = sorted(axis_fields - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    x_feature = _workflow_column(
        inputs["x_feature"],
        name=f"{workflow} x_feature",
        whitelist=whitelist,
    )
    y_feature = _workflow_column(
        inputs["y_feature"],
        name=f"{workflow} y_feature",
        whitelist=whitelist,
    )
    if x_feature == y_feature:
        raise _DraftValidationError(f"{workflow} 两个轴必须使用不同字段。")
    if target_col is not None and target_col in {x_feature, y_feature}:
        raise _DraftValidationError(f"{workflow} 交叉轴不能使用目标列。")

    def axis_method(field: str) -> str:
        method = _required_text(inputs[field], name=f"{workflow} {field}")
        if method not in UNIVARIATE_REFINEMENT_METHODS:
            raise _DraftValidationError(
                f"{workflow} {field} 只能是："
                + "、".join(UNIVARIATE_REFINEMENT_METHODS)
                + "。"
            )
        return method

    x_method = axis_method("x_method")
    y_method = axis_method("y_method")
    numeric_methods = list(
        dict.fromkeys(
            method
            for method in (x_method, y_method)
            if method != "categorical"
        )
    )
    if "features" in inputs and inputs["features"] != [x_feature, y_feature]:
        raise _DraftValidationError(
            f"{workflow} features 只能是平台派生的有序轴字段。"
        )
    if "methods" in inputs and inputs["methods"] != numeric_methods:
        raise _DraftValidationError(
            f"{workflow} methods 只能是平台派生的数值轴方法。"
        )
    analysis_inputs: dict[str, Any] = {
        "features": [x_feature, y_feature],
        **{field: inputs[field] for field in analysis_fields if field in inputs},
    }
    if numeric_methods:
        analysis_inputs["methods"] = numeric_methods
    normalized = _validate_univariate_workflow_inputs(
        analysis_inputs,
        whitelist,
        target_col=target_col,
        manual_features=[
            feature
            for feature, method in (
                (x_feature, x_method),
                (y_feature, y_method),
            )
            if method == "manual"
        ],
    )
    return {
        **normalized,
        "x_feature": x_feature,
        "x_method": x_method,
        "y_feature": y_feature,
        "y_method": y_method,
    }


def _validate_manual_breakpoint_mapping(
    value: object,
    *,
    manual_requested: bool,
    expected_features: Sequence[str],
    workflow: str,
) -> dict[str, list[float]]:
    if value is None:
        if manual_requested:
            raise _DraftValidationError(
                f"{workflow} manual 分箱必须提供 manual_breakpoints。"
            )
        return {}
    if not manual_requested:
        raise _DraftValidationError(
            f"{workflow} 只有选择 manual 分箱时才能提供 manual_breakpoints。"
        )
    if not isinstance(value, Mapping) or not value:
        raise _DraftValidationError(
            f"{workflow} manual_breakpoints 必须是非空字段到切点数组映射。"
        )
    expected = tuple(expected_features)
    if not expected or set(value) != set(expected) or len(value) != len(expected):
        raise _DraftValidationError(
            f"{workflow} manual_breakpoints 必须且只能覆盖 manual 轴/字段："
            + "、".join(expected)
            + "。"
        )
    normalized: dict[str, list[float]] = {}
    for feature in expected:
        raw_points = value[feature]
        if (
            not isinstance(raw_points, Sequence)
            or isinstance(raw_points, str | bytes | bytearray)
            or not 1 <= len(raw_points) <= 19
        ):
            raise _DraftValidationError(
                f"{workflow} manual_breakpoints.{feature} 必须包含 1 到 19 个切点。"
            )
        points: list[float] = []
        for item in raw_points:
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise _DraftValidationError(
                    f"{workflow} manual_breakpoints.{feature} 只能包含有限数字。"
                )
            if isinstance(item, int) and abs(item) > 2**53 - 1:
                raise _DraftValidationError(
                    f"{workflow} manual_breakpoints.{feature} 超出精确 JSON 范围。"
                )
            number = float(item)
            if not math.isfinite(number):
                raise _DraftValidationError(
                    f"{workflow} manual_breakpoints.{feature} 只能包含有限数字。"
                )
            points.append(number)
        if any(left >= right for left, right in zip(points, points[1:])):
            raise _DraftValidationError(
                f"{workflow} manual_breakpoints.{feature} 必须严格递增且不重复。"
            )
        normalized[feature] = points
    return normalized


def _sentinel_sequence(value: object, *, name: str) -> list[str | int | float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 20
    ):
        raise _DraftValidationError(f"{name} 必须是最多 20 个文本或有限数字的数组。")
    normalized: list[str | int | float] = []
    identities: set[str] = set()
    for item in value:
        if isinstance(item, bool) or not isinstance(item, str | int | float):
            raise _DraftValidationError(f"{name} 只能包含文本或有限数字。")
        if isinstance(item, float) and not math.isfinite(item):
            raise _DraftValidationError(f"{name} 只能包含文本或有限数字。")
        if isinstance(item, int) and abs(item) > 2**53 - 1:
            raise _DraftValidationError(f"{name} 中的整数超出精确 JSON 范围。")
        identity = json.dumps(
            [type(item).__name__, item],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if identity in identities:
            raise _DraftValidationError(f"{name} 不能包含重复值。")
        identities.add(identity)
        normalized.append(item)
    return normalized


def _validate_automatic_tree_candidate_build_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    """Validate only controls the user owns for one automatic-tree build.

    Dataset/workspace identity, target selection, label policy, execution
    budgets and every result field are deliberately absent.  The trusted
    template binds those values after confirmation.
    """

    workflow = "automatic_tree_candidate_build"
    allowed = {
        "features",
        "sample_weight_col",
        "directions",
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "seed",
        "loan_amount_col",
        "overdue_amount_col",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if "features" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少必需字段 features。")
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 1 <= len(raw_features) <= 50
    ):
        raise _DraftValidationError(
            f"{workflow} features 必须是包含 1 到 50 个字段的有序数组。"
        )
    features = [
        _workflow_column(
            value,
            name=f"{workflow} features",
            whitelist=whitelist,
        )
        for value in raw_features
    ]
    if len(features) != len(set(features)):
        raise _DraftValidationError(f"{workflow} features 不能包含重复字段。")
    if target_col is not None and target_col in features:
        raise _DraftValidationError(
            f"{workflow} features 不能包含目标列 {target_col}。"
        )

    normalized: dict[str, Any] = {"features": features}
    for field in (
        "sample_weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        if field not in inputs:
            continue
        column = _workflow_column(
            inputs[field],
            name=f"{workflow} {field}",
            whitelist=whitelist,
        )
        if target_col is not None and column == target_col:
            raise _DraftValidationError(f"{workflow} {field} 不能使用目标列。")
        normalized[field] = column

    if "directions" in inputs:
        raw_directions = inputs["directions"]
        if not isinstance(raw_directions, Mapping) or not raw_directions:
            raise _DraftValidationError(
                f"{workflow} directions 必须是至少包含一个特征方向的对象；"
                "没有风险方向诊断期望或检查时请省略该字段。"
            )
        if any(not isinstance(key, str) for key in raw_directions):
            raise _DraftValidationError(f"{workflow} directions 的字段名必须是文本。")
        unexpected_features = sorted(set(raw_directions) - set(features))
        if unexpected_features:
            raise _DraftValidationError(
                f"{workflow} directions 引用了未选择的特征："
                + "、".join(unexpected_features)
                + "。"
            )
        directions: dict[str, str] = {}
        for feature, value in raw_directions.items():
            if not isinstance(value, str) or value not in AUTOMATIC_TREE_DIRECTIONS:
                raise _DraftValidationError(
                    f"{workflow} directions.{feature} 只能是 increasing、"
                    "decreasing 或 unordered。"
                )
            directions[feature] = value
        normalized["directions"] = directions

    if "max_depth" in inputs:
        max_depth = inputs["max_depth"]
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or not 1 <= max_depth <= 8
        ):
            raise _DraftValidationError(f"{workflow} max_depth 必须是 1 到 8 的整数。")
        normalized["max_depth"] = max_depth
    if "min_leaf_count" in inputs:
        min_leaf_count = inputs["min_leaf_count"]
        if (
            isinstance(min_leaf_count, bool)
            or not isinstance(min_leaf_count, int)
            or min_leaf_count <= 0
        ):
            raise _DraftValidationError(f"{workflow} min_leaf_count 必须是正整数。")
        normalized["min_leaf_count"] = min_leaf_count
    if "min_weight_fraction_leaf" in inputs:
        normalized["min_weight_fraction_leaf"] = _bounded_number(
            inputs["min_weight_fraction_leaf"],
            name=f"{workflow} min_weight_fraction_leaf",
            maximum=0.5,
        )
    if "seed" in inputs:
        seed = inputs["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 4_294_967_295
        ):
            raise _DraftValidationError(
                f"{workflow} seed 必须是 0 到 4294967295 的整数。"
            )
        normalized["seed"] = seed

    assigned_columns = [
        normalized[field]
        for field in (
            "sample_weight_col",
            "loan_amount_col",
            "overdue_amount_col",
        )
        if field in normalized
    ]
    duplicate_roles = {
        column for column in assigned_columns if assigned_columns.count(column) > 1
    }
    feature_conflicts = set(features) & set(assigned_columns)
    if duplicate_roles or feature_conflicts:
        conflicts = sorted(duplicate_roles | feature_conflicts)
        raise _DraftValidationError(
            f"{workflow} features、sample_weight_col、loan_amount_col 与 "
            "overdue_amount_col 必须使用不同字段：" + "、".join(conflicts) + "。"
        )
    return normalized


def _validate_automatic_tree_leaf_materialization_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the two explicit pointers and optional user-owned rationale.

    The source TaskArtifact and every integrity hash are deliberately resolved
    later by the platform.  The compiler may never accept a copied rule,
    metric, effect, fragment or business action from the LLM.
    """

    workflow = "automatic_tree_leaf_materialization"
    allowed = {"tree_asset_id", "leaf_id", "selection_reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"tree_asset_id", "leaf_id"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    tree_asset_id = _required_text(
        inputs["tree_asset_id"],
        name=f"{workflow} tree_asset_id",
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(tree_asset_id) is None:
        raise _DraftValidationError(
            f"{workflow} tree_asset_id 必须是完整的自动树 candidate asset id。"
        )
    leaf_id = _required_text(
        inputs["leaf_id"],
        name=f"{workflow} leaf_id",
    )
    if _AUTOMATIC_TREE_LEAF_ID_RE.fullmatch(leaf_id) is None:
        raise _DraftValidationError(
            f"{workflow} leaf_id 必须是 leaf- 后接 20 位小写十六进制字符。"
        )

    normalized = {"tree_asset_id": tree_asset_id, "leaf_id": leaf_id}
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _automatic_tree_selection_reason(
            inputs["selection_reason"]
        )
    return normalized


def _validate_interactive_tree_revision_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one explicit immutable edit over a visible split pointer.

    Artifact rows, hashes, topology, frontier, conditions, metrics, dataset and
    SampleDesign bindings are platform-owned and cannot enter the draft.
    """

    workflow = "interactive_tree_revision"
    allowed = {
        "source_tree_id",
        "node_id",
        "operation",
        "threshold",
        "reason",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"source_tree_id", "node_id", "operation"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    source_tree_id = _required_text(
        inputs["source_tree_id"],
        name=f"{workflow} source_tree_id",
    )
    if _INTERACTIVE_TREE_SOURCE_ID_RE.fullmatch(source_tree_id) is None:
        raise _DraftValidationError(
            f"{workflow} source_tree_id 必须是完整 automatic-tree asset "
            "或 interactive-tree revision ID。"
        )
    node_id = _required_text(
        inputs["node_id"],
        name=f"{workflow} node_id",
    )
    if _INTERACTIVE_TREE_NODE_ID_RE.fullmatch(node_id) is None:
        raise _DraftValidationError(
            f"{workflow} node_id 必须是 node- 后接 20 位小写十六进制字符。"
        )
    operation = _required_text(
        inputs["operation"],
        name=f"{workflow} operation",
    )
    if operation not in {"prune_subtree", "adjust_split_threshold"}:
        raise _DraftValidationError(
            f"{workflow} operation 只允许 prune_subtree 或 "
            "adjust_split_threshold。"
        )
    has_threshold = "threshold" in inputs
    if operation == "adjust_split_threshold" and not has_threshold:
        raise _DraftValidationError(
            f"{workflow} adjust_split_threshold 必须提供 threshold。"
        )
    if operation == "prune_subtree" and has_threshold:
        raise _DraftValidationError(
            f"{workflow} prune_subtree 不能提供 threshold。"
        )
    normalized: dict[str, Any] = {
        "source_tree_id": source_tree_id,
        "node_id": node_id,
        "operation": operation,
    }
    if has_threshold:
        threshold = inputs["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not math.isfinite(float(threshold))
        ):
            raise _DraftValidationError(
                f"{workflow} threshold 必须是 finite number。"
            )
        if isinstance(threshold, int) and abs(threshold) > 2**53 - 1:
            raise _DraftValidationError(
                f"{workflow} threshold 超出精确 JSON number 范围。"
            )
        normalized["threshold"] = float(threshold)
    if "reason" in inputs:
        reason = inputs["reason"]
        if reason is None:
            normalized["reason"] = None
        else:
            normalized["reason"] = _interactive_tree_revision_reason(reason)
    return normalized


def _validate_interactive_tree_frontier_materialization_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only an explicit revision/frontier pointer.

    The selected revision artifact, its recursively authenticated ancestry,
    tree identity and candidate fragment are all restored by the platform.
    """

    workflow = "interactive_tree_frontier_materialization"
    allowed = {"revision_id", "source_node_id", "selection_reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"revision_id", "source_node_id"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    revision_id = _required_text(
        inputs["revision_id"],
        name=f"{workflow} revision_id",
    )
    if _INTERACTIVE_TREE_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise _DraftValidationError(
            f"{workflow} revision_id 必须是 interactive-tree-revision- "
            "后接 32 位小写十六进制字符。"
        )
    source_node_id = _required_text(
        inputs["source_node_id"],
        name=f"{workflow} source_node_id",
    )
    if _INTERACTIVE_TREE_FRONTIER_NODE_ID_RE.fullmatch(source_node_id) is None:
        raise _DraftValidationError(
            f"{workflow} source_node_id 必须是 node- 或 leaf- "
            "后接 20 位小写十六进制字符。"
        )
    normalized: dict[str, Any] = {
        "revision_id": revision_id,
        "source_node_id": source_node_id,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _interactive_tree_frontier_reason(
            inputs["selection_reason"]
        )
    return normalized


def _validate_interactive_tree_frontier_group_materialization_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only an explicit revision/frontier OR-group pointer."""

    workflow = "interactive_tree_frontier_group_materialization"
    allowed = {"revision_id", "source_node_ids", "selection_reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"revision_id", "source_node_ids"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    revision_id = _required_text(
        inputs["revision_id"],
        name=f"{workflow} revision_id",
    )
    if _INTERACTIVE_TREE_REVISION_ID_RE.fullmatch(revision_id) is None:
        raise _DraftValidationError(
            f"{workflow} revision_id 必须是 interactive-tree-revision- "
            "后接 32 位小写十六进制字符。"
        )
    raw_node_ids = inputs["source_node_ids"]
    if not isinstance(raw_node_ids, list):
        raise _DraftValidationError(
            f"{workflow} source_node_ids 必须是数组。"
        )
    if not 2 <= len(raw_node_ids) <= 50:
        raise _DraftValidationError(
            f"{workflow} source_node_ids 必须包含 2 到 50 个完整节点 ID。"
        )
    source_node_ids: list[str] = []
    for index, value in enumerate(raw_node_ids):
        source_node_id = _required_text(
            value,
            name=f"{workflow} source_node_ids[{index}]",
        )
        if (
            _INTERACTIVE_TREE_FRONTIER_NODE_ID_RE.fullmatch(source_node_id)
            is None
        ):
            raise _DraftValidationError(
                f"{workflow} source_node_ids[{index}] 必须是 node- 或 leaf- "
                "后接 20 位小写十六进制字符。"
            )
        source_node_ids.append(source_node_id)
    if len(source_node_ids) != len(set(source_node_ids)):
        raise _DraftValidationError(
            f"{workflow} source_node_ids 不能包含重复节点 ID。"
        )
    normalized: dict[str, Any] = {
        "revision_id": revision_id,
        "source_node_ids": source_node_ids,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _interactive_tree_frontier_group_reason(
            inputs["selection_reason"]
        )
    return normalized


def _interactive_tree_frontier_group_reason(value: object) -> str:
    if not isinstance(value, str):
        raise _DraftValidationError(
            "interactive_tree_frontier_group_materialization "
            "selection_reason 必须是文本。"
        )
    if "\x00" in value:
        raise _DraftValidationError(
            "interactive_tree_frontier_group_materialization "
            "selection_reason 不能包含 NUL。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise _DraftValidationError(
            "interactive_tree_frontier_group_materialization "
            "selection_reason 必须是 1 到 500 字符的非空文本。"
        )
    return canonical


def _interactive_tree_frontier_reason(value: object) -> str:
    if not isinstance(value, str):
        raise _DraftValidationError(
            "interactive_tree_frontier_materialization selection_reason 必须是文本。"
        )
    if "\x00" in value:
        raise _DraftValidationError(
            "interactive_tree_frontier_materialization selection_reason 不能包含 NUL。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise _DraftValidationError(
            "interactive_tree_frontier_materialization selection_reason "
            "必须是 1 到 500 字符的非空文本。"
        )
    return canonical


def _interactive_tree_revision_reason(value: object) -> str:
    if not isinstance(value, str):
        raise _DraftValidationError(
            "interactive_tree_revision reason 必须是文本。"
        )
    if "\x00" in value:
        raise _DraftValidationError(
            "interactive_tree_revision reason 不能包含 NUL。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise _DraftValidationError(
            "interactive_tree_revision reason 必须是 1 到 500 字符的非空文本。"
        )
    return canonical


def _validate_automatic_tree_apply_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only the explicit full-tree pointer and output column names.

    The source artifact, all hashes, source dataset, workspace lineage and
    activation policy are resolved by the platform after compilation.
    """

    workflow = "automatic_tree_apply"
    allowed = {"tree_asset_id", "leaf_id_column", "rule_id_column"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if "tree_asset_id" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少字段：tree_asset_id。")

    tree_asset_id = _required_text(
        inputs["tree_asset_id"],
        name=f"{workflow} tree_asset_id",
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(tree_asset_id) is None:
        raise _DraftValidationError(
            f"{workflow} tree_asset_id 必须是完整的自动树 candidate asset id。"
        )

    normalized: dict[str, Any] = {"tree_asset_id": tree_asset_id}
    for field in ("leaf_id_column", "rule_id_column"):
        if field not in inputs:
            continue
        value = _required_text(inputs[field], name=f"{workflow} {field}")
        if _AUTOMATIC_TREE_APPLY_OUTPUT_COLUMN_NAME_RE.fullmatch(value) is None:
            raise _DraftValidationError(
                f"{workflow} {field} 必须是最多 64 位的 ASCII 标识符。"
            )
        normalized[field] = value

    if (
        "leaf_id_column" in normalized
        and "rule_id_column" in normalized
        and normalized["leaf_id_column"].casefold()
        == normalized["rule_id_column"].casefold()
    ):
        raise _DraftValidationError(
            f"{workflow} leaf_id_column 与 rule_id_column 必须不同（忽略大小写）。"
        )
    return normalized


def _validate_cross_matrix_cell_selection_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only exact user-owned Cross asset and cell pointers."""

    workflow = "cross_matrix_cell_selection"
    allowed = {"cross_asset_id", "cell_ids", "selection_reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"cross_asset_id", "cell_ids"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    cross_asset_id = _required_text(
        inputs["cross_asset_id"],
        name=f"{workflow} cross_asset_id",
    )
    if _CANDIDATE_ASSET_ID_RE.fullmatch(cross_asset_id) is None:
        raise _DraftValidationError(
            f"{workflow} cross_asset_id 必须是完整的 Cross candidate asset id。"
        )

    raw_cell_ids = inputs["cell_ids"]
    if (
        not isinstance(raw_cell_ids, Sequence)
        or isinstance(raw_cell_ids, str | bytes | bytearray)
        or not 1 <= len(raw_cell_ids) <= 400
    ):
        raise _DraftValidationError(
            f"{workflow} cell_ids 必须是 1 到 400 个完整 cross-cell ID 的数组。"
        )
    cell_ids: list[str] = []
    for value in raw_cell_ids:
        cell_id = _required_text(value, name=f"{workflow} cell_ids")
        if _CROSS_MATRIX_CELL_ID_RE.fullmatch(cell_id) is None:
            raise _DraftValidationError(
                f"{workflow} cell_ids 必须是 cross-cell- 后接 32 位小写十六进制字符。"
            )
        cell_ids.append(cell_id)
    if len(set(cell_ids)) != len(cell_ids):
        raise _DraftValidationError(f"{workflow} cell_ids 不能包含重复 ID。")

    normalized: dict[str, Any] = {
        "cross_asset_id": cross_asset_id,
        "cell_ids": cell_ids,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _cross_matrix_cell_selection_reason(
            inputs["selection_reason"]
        )
    return normalized


def _validate_voting_candidate_build_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only the explicit human controls for one n-of-k candidate.

    Pool revision/hash, entry ids, executable conditions, sample bindings and
    every measured value remain platform-owned and are resolved after the
    natural-language request has passed grounding.
    """

    workflow = "voting_candidate_build"
    allowed = {"strategy_type", "rule_ids", "n"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    strategy_type = _required_text(
        inputs["strategy_type"], name=f"{workflow} strategy_type"
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    raw_rule_ids = inputs["rule_ids"]
    if (
        not isinstance(raw_rule_ids, Sequence)
        or isinstance(raw_rule_ids, str | bytes | bytearray)
        or not 2 <= len(raw_rule_ids) <= 50
    ):
        raise _DraftValidationError(
            f"{workflow} rule_ids 必须是 2 到 50 个完整 rule_id 的数组。"
        )
    rule_ids: list[str] = []
    for value in raw_rule_ids:
        rule_id = _required_text(value, name=f"{workflow} rule_ids")
        if _VOTING_RULE_ID_RE.fullmatch(rule_id) is None:
            raise _DraftValidationError(
                f"{workflow} rule_ids 必须是完整的 candidate-rule ID。"
            )
        rule_ids.append(rule_id)
    if len(set(rule_ids)) != len(rule_ids):
        raise _DraftValidationError(f"{workflow} rule_ids 不能包含重复 ID。")
    n = inputs["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(rule_ids):
        raise _DraftValidationError(
            f"{workflow} n 必须是 1 到规则数 {len(rule_ids)} 的整数。"
        )
    return {"strategy_type": strategy_type, "rule_ids": rule_ids, "n": n}


def _validate_voting_candidate_build_from_search_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the two exact search pointers and optional Pool type only."""

    workflow = "voting_candidate_build_from_search"
    allowed = {"search_id", "combo_id", "strategy_type"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"search_id", "combo_id"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    search_id = _required_text(inputs["search_id"], name=f"{workflow} search_id")
    if _VOTING_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise _DraftValidationError(
            f"{workflow} search_id 必须是完整的 voting-search ID。"
        )
    combo_id = _required_text(inputs["combo_id"], name=f"{workflow} combo_id")
    if _VOTING_COMBO_ID_RE.fullmatch(combo_id) is None:
        raise _DraftValidationError(
            f"{workflow} combo_id 必须是完整的 voting-combo ID。"
        )

    normalized = {"search_id": search_id, "combo_id": combo_id}
    if "strategy_type" in inputs:
        strategy_type = _required_text(
            inputs["strategy_type"],
            name=f"{workflow} strategy_type",
        )
        if strategy_type not in STRATEGY_TYPES:
            raise _DraftValidationError(
                f"{workflow} strategy_type 只能是："
                + "、".join(STRATEGY_TYPES)
                + "。"
            )
        normalized["strategy_type"] = strategy_type
    return normalized


def _validate_cross_matrix_candidate_search_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    """Validate only explicit feature names and the bounded pair budget."""

    workflow = "cross_matrix_candidate_search"
    allowed = {"features", "max_pairs"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 2 <= len(raw_features) <= 20
    ):
        raise _DraftValidationError(
            f"{workflow} features 必须是 2 到 20 个明确字段的数组。"
        )
    features = [
        _workflow_column(
            value,
            name=f"{workflow} features",
            whitelist=whitelist,
        )
        for value in raw_features
    ]
    if len(set(features)) != len(features):
        raise _DraftValidationError(f"{workflow} features 不能包含重复字段。")
    if target_col is not None and target_col in features:
        raise _DraftValidationError(
            f"{workflow} features 不能包含当前目标列「{target_col}」。"
        )
    max_pairs = inputs["max_pairs"]
    if (
        isinstance(max_pairs, bool)
        or not isinstance(max_pairs, int)
        or not 1 <= max_pairs <= 190
    ):
        raise _DraftValidationError(
            f"{workflow} max_pairs 必须是 1 到 190 的整数。"
        )
    return {"features": features, "max_pairs": max_pairs}


def _validate_cross_matrix_candidate_build_from_search_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, str]:
    """Validate only the two exact authenticated search pointers."""

    workflow = "cross_matrix_candidate_build_from_search"
    allowed = {"search_id", "pair_id"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    search_id = _required_text(
        inputs["search_id"],
        name=f"{workflow} search_id",
    )
    pair_id = _required_text(
        inputs["pair_id"],
        name=f"{workflow} pair_id",
    )
    if _CROSS_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise _DraftValidationError(
            f"{workflow} search_id 必须是完整的 cross-search ID。"
        )
    if _CROSS_PAIR_ID_RE.fullmatch(pair_id) is None:
        raise _DraftValidationError(
            f"{workflow} pair_id 必须是完整的 cross-pair ID。"
        )
    return {"search_id": search_id, "pair_id": pair_id}


def _validate_cross_rule_search_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    """Validate only the human-owned rule-search universe and hard budget."""

    workflow = "cross_rule_search"
    allowed = {"features", "dimension", "constraints", "max_trials"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted(allowed - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    raw_features = inputs["features"]
    if (
        not isinstance(raw_features, Sequence)
        or isinstance(raw_features, str | bytes | bytearray)
        or not 2 <= len(raw_features) <= 12
    ):
        raise _DraftValidationError(
            f"{workflow} features 必须是 2 到 12 个明确字段的数组。"
        )
    features = [
        _workflow_column(
            value,
            name=f"{workflow} features",
            whitelist=whitelist,
        )
        for value in raw_features
    ]
    if len(set(features)) != len(features):
        raise _DraftValidationError(f"{workflow} features 不能包含重复字段。")
    if target_col is not None and target_col in features:
        raise _DraftValidationError(
            f"{workflow} features 不能包含当前目标列「{target_col}」。"
        )
    dimension = inputs["dimension"]
    if isinstance(dimension, bool) or dimension not in {2, 3}:
        raise _DraftValidationError(
            f"{workflow} dimension 只能是整数 2 或 3。"
        )
    raw_constraints = inputs["constraints"]
    constraint_fields = {
        "min_lift",
        "min_bad_count",
        "max_hit_share",
        "min_amount_lift",
    }
    if not isinstance(raw_constraints, Mapping):
        raise _DraftValidationError(
            f"{workflow} constraints 必须是对象。"
        )
    unknown = sorted(set(raw_constraints) - constraint_fields)
    missing_constraints = sorted(constraint_fields - set(raw_constraints))
    if unknown or missing_constraints:
        details = []
        if unknown:
            details.append("不支持 " + "、".join(unknown))
        if missing_constraints:
            details.append("缺少 " + "、".join(missing_constraints))
        raise _DraftValidationError(
            f"{workflow} constraints 字段无效：" + "；".join(details) + "。"
        )

    def finite_constraint(
        name: str,
        *,
        minimum: float,
        maximum: float,
        optional: bool = False,
    ) -> float | None:
        value = raw_constraints[name]
        if optional and value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise _DraftValidationError(
                f"{workflow} constraints.{name} 必须是 "
                f"{minimum:g} 到 {maximum:g} 的有限数值"
                + ("或 null。" if optional else "。")
            )
        return float(value)

    min_bad_count = raw_constraints["min_bad_count"]
    if (
        isinstance(min_bad_count, bool)
        or not isinstance(min_bad_count, int)
        or min_bad_count < 0
    ):
        raise _DraftValidationError(
            f"{workflow} constraints.min_bad_count 必须是非负整数。"
        )
    constraints = {
        "min_lift": finite_constraint(
            "min_lift",
            minimum=0.0,
            maximum=1_000.0,
        ),
        "min_bad_count": min_bad_count,
        "max_hit_share": finite_constraint(
            "max_hit_share",
            minimum=0.0,
            maximum=1.0,
        ),
        "min_amount_lift": finite_constraint(
            "min_amount_lift",
            minimum=0.0,
            maximum=1_000.0,
            optional=True,
        ),
    }
    max_trials = inputs["max_trials"]
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or not 1 <= max_trials <= 5_000
    ):
        raise _DraftValidationError(
            f"{workflow} max_trials 必须是 1 到 5000 的整数。"
        )
    return {
        "features": features,
        "dimension": dimension,
        "constraints": constraints,
        "max_trials": max_trials,
    }


def _validate_cross_rule_candidate_build_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact rule pointer and an optional human audit reason."""

    workflow = "cross_rule_candidate_build_from_search"
    allowed = {"search_id", "rule_id", "selection_reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"search_id", "rule_id"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    search_id = _required_text(
        inputs["search_id"],
        name=f"{workflow} search_id",
    )
    rule_id = _required_text(
        inputs["rule_id"],
        name=f"{workflow} rule_id",
    )
    if _CROSS_RULE_SEARCH_ID_RE.fullmatch(search_id) is None:
        raise _DraftValidationError(
            f"{workflow} search_id 必须是完整的 cross-rule-search ID。"
        )
    if _CROSS_RULE_ID_RE.fullmatch(rule_id) is None:
        raise _DraftValidationError(
            f"{workflow} rule_id 必须是完整的 cross-rule ID。"
        )
    normalized: dict[str, Any] = {
        "search_id": search_id,
        "rule_id": rule_id,
    }
    if "selection_reason" in inputs:
        normalized["selection_reason"] = _cross_rule_selection_reason(
            inputs["selection_reason"]
        )
    return normalized


def _cross_rule_selection_reason(value: object) -> str:
    if not isinstance(value, str) or "\x00" in value:
        raise _DraftValidationError(
            "cross_rule_candidate_build_from_search selection_reason "
            "必须是文本。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical or len(canonical) > 500:
        raise _DraftValidationError(
            "cross_rule_candidate_build_from_search selection_reason "
            "必须是 1 到 500 个字符。"
        )
    return canonical


def _validate_voting_candidate_search_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate human search controls and inject only documented safe defaults."""

    workflow = "voting_candidate_search"
    required = {"strategy_type", "member_count", "n", "objective"}
    allowed = {
        *required,
        "constraints",
        "include_rule_ids",
        "exclude_rule_ids",
        "max_combinations",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted(required - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是：" + "、".join(STRATEGY_TYPES) + "。"
        )
    member_count = inputs["member_count"]
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or not 2 <= member_count <= 50
    ):
        raise _DraftValidationError(f"{workflow} member_count 必须是 2 到 50 的整数。")
    n = inputs["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= member_count:
        raise _DraftValidationError(f"{workflow} n 必须是 1 到 member_count 的整数。")

    objective_raw = inputs["objective"]
    if not isinstance(objective_raw, Mapping):
        raise _DraftValidationError(f"{workflow} objective 必须是对象。")
    objective_fields = {"metric", "direction"}
    if set(objective_raw) != objective_fields:
        raise _DraftValidationError(
            f"{workflow} objective 必须且只能包含 metric、direction。"
        )
    objective_metric = _required_text(
        objective_raw["metric"],
        name=f"{workflow} objective.metric",
    )
    if objective_metric not in _VOTING_SEARCH_METRICS:
        raise _DraftValidationError(f"{workflow} objective.metric 不受支持。")
    objective_direction = _required_text(
        objective_raw["direction"],
        name=f"{workflow} objective.direction",
    )
    if objective_direction not in {"maximize", "minimize"}:
        raise _DraftValidationError(
            f"{workflow} objective.direction 只能是 maximize 或 minimize。"
        )
    objective = {
        "metric": objective_metric,
        "direction": objective_direction,
    }

    constraints_raw = inputs.get("constraints", [])
    if (
        not isinstance(constraints_raw, Sequence)
        or isinstance(constraints_raw, str | bytes | bytearray)
        or len(constraints_raw) > 32
    ):
        raise _DraftValidationError(f"{workflow} constraints 必须是最多 32 项的数组。")
    constraints: list[dict[str, Any]] = []
    constraint_identities: set[tuple[str, str]] = set()
    for index, value in enumerate(constraints_raw):
        if not isinstance(value, Mapping) or set(value) != {
            "metric",
            "operator",
            "value",
        }:
            raise _DraftValidationError(
                f"{workflow} constraints[{index}] 必须且只能包含 "
                "metric、operator、value。"
            )
        metric = _required_text(
            value["metric"],
            name=f"{workflow} constraints[{index}].metric",
        )
        if metric not in _VOTING_SEARCH_METRICS:
            raise _DraftValidationError(
                f"{workflow} constraints[{index}].metric 不受支持。"
            )
        operator = _required_text(
            value["operator"],
            name=f"{workflow} constraints[{index}].operator",
        )
        if operator not in {"gte", "lte"}:
            raise _DraftValidationError(
                f"{workflow} constraints[{index}].operator 只能是 gte 或 lte。"
            )
        number = _bounded_number(
            value["value"],
            name=f"{workflow} constraints[{index}].value",
            maximum=1.0 if metric in _VOTING_SEARCH_RATE_METRICS else None,
        )
        identity = (metric, operator)
        if identity in constraint_identities:
            raise _DraftValidationError(
                f"{workflow} constraints 不能重复 metric/operator。"
            )
        constraint_identities.add(identity)
        constraints.append({"metric": metric, "operator": operator, "value": number})
    constraints.sort(key=lambda item: (item["metric"], item["operator"], item["value"]))

    include_rule_ids = _voting_search_rule_id_array(
        inputs.get("include_rule_ids", []),
        name=f"{workflow} include_rule_ids",
    )
    exclude_rule_ids = _voting_search_rule_id_array(
        inputs.get("exclude_rule_ids", []),
        name=f"{workflow} exclude_rule_ids",
    )
    overlap = sorted(set(include_rule_ids) & set(exclude_rule_ids))
    if overlap:
        raise _DraftValidationError(
            f"{workflow} include_rule_ids 与 exclude_rule_ids 不能重叠。"
        )
    if len(include_rule_ids) > member_count:
        raise _DraftValidationError(
            f"{workflow} include_rule_ids 数量不能超过 member_count。"
        )

    max_combinations = inputs.get("max_combinations", 10_000)
    if (
        isinstance(max_combinations, bool)
        or not isinstance(max_combinations, int)
        or not 1 <= max_combinations <= 10_000
    ):
        raise _DraftValidationError(
            f"{workflow} max_combinations 必须是 1 到 10000 的整数。"
        )

    required_share = _VOTING_SEARCH_REQUIRED_MINIMUM_SHARE.get(objective_metric)
    if objective_direction == "minimize" and required_share is not None:
        has_positive_share = any(
            item["metric"] == required_share
            and item["operator"] == "gte"
            and item["value"] > 0
            for item in constraints
        )
        if not has_positive_share:
            raise _DraftValidationError(
                f"最小化 {objective_metric} 必须提供正数 {required_share} gte "
                "约束，绝对命中量不能替代占比下限。",
                code="voting_search_minimum_share_required",
                fields=("constraints", required_share),
            )

    return {
        "strategy_type": strategy_type,
        "member_count": member_count,
        "n": n,
        "objective": objective,
        "constraints": constraints,
        "include_rule_ids": include_rule_ids,
        "exclude_rule_ids": exclude_rule_ids,
        "max_combinations": max_combinations,
    }


def _voting_search_rule_id_array(
    value: object,
    *,
    name: str,
) -> list[str]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 50
    ):
        raise _DraftValidationError(f"{name} 必须是最多 50 个完整 rule_id 的数组。")
    normalized: list[str] = []
    for raw in value:
        rule_id = _required_text(raw, name=name)
        if _VOTING_RULE_ID_RE.fullmatch(rule_id) is None:
            raise _DraftValidationError(f"{name} 必须只包含完整的 candidate-rule ID。")
        normalized.append(rule_id)
    if len(set(normalized)) != len(normalized):
        raise _DraftValidationError(f"{name} 不能包含重复 ID。")
    return sorted(normalized)


def _automatic_tree_selection_reason(value: object) -> str:
    """Use the leaf-fragment contract's NFC and canonical-whitespace rules."""

    if not isinstance(value, str):
        raise _DraftValidationError(
            "automatic_tree_leaf_materialization selection_reason 必须是文本。"
        )
    if "\x00" in value:
        raise _DraftValidationError(
            "automatic_tree_leaf_materialization selection_reason 不能包含 NUL。"
        )
    normalized = unicodedata.normalize("NFC", value)
    canonical = " ".join(normalized.split())
    if not canonical:
        raise _DraftValidationError(
            "automatic_tree_leaf_materialization selection_reason 必须是非空文本。"
        )
    return canonical


def _cross_matrix_cell_selection_reason(value: object) -> str:
    if not isinstance(value, str):
        raise _DraftValidationError(
            "cross_matrix_cell_selection selection_reason 必须是文本。"
        )
    if "\x00" in value:
        raise _DraftValidationError(
            "cross_matrix_cell_selection selection_reason 不能包含 NUL。"
        )
    canonical = " ".join(unicodedata.normalize("NFC", value).split())
    if not canonical:
        raise _DraftValidationError(
            "cross_matrix_cell_selection selection_reason 必须是非空文本。"
        )
    if len(canonical) > 500:
        raise _DraftValidationError(
            "cross_matrix_cell_selection selection_reason 最多 500 个字符。"
        )
    return canonical


def _validate_univariate_refinement_workflow_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> dict[str, Any]:
    workflow = "univariate_candidate_refinement"
    analysis_fields = {
        "features",
        "methods",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
        "manual_breakpoints",
    }
    allowed = analysis_fields | {
        "feature",
        "method",
        "merge_groups",
        "selection",
        "selection_reason",
        "source_candidate_id",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"feature", "method", "selection"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )

    source_candidate_id = None
    if "source_candidate_id" in inputs:
        source_candidate_id = _required_text(
            inputs["source_candidate_id"],
            name=f"{workflow} source_candidate_id",
        )
        if _CANDIDATE_ID_RE.fullmatch(source_candidate_id) is None:
            raise _DraftValidationError(
                f"{workflow} source_candidate_id 必须是完整 candidate id。"
            )
        ignored_analysis_fields = sorted(set(inputs) & analysis_fields)
        if ignored_analysis_fields:
            raise _DraftValidationError(
                f"{workflow} 已绑定已有 candidate 时不能重设分析参数："
                + "、".join(ignored_analysis_fields)
                + "。"
            )

    feature = (
        _required_text(inputs["feature"], name=f"{workflow} feature")
        if source_candidate_id is not None
        else _workflow_column(
            inputs["feature"],
            name=f"{workflow} feature",
            whitelist=whitelist,
        )
    )
    if target_col is not None and feature == target_col:
        raise _DraftValidationError(f"{workflow} feature 不能使用目标列。")
    method = _required_text(inputs["method"], name=f"{workflow} method")
    if method not in UNIVARIATE_REFINEMENT_METHODS:
        raise _DraftValidationError(
            f"{workflow} 不支持分箱方法 {method}；可选值为："
            + "、".join(UNIVARIATE_REFINEMENT_METHODS)
            + "。"
        )

    analysis: dict[str, Any] = {}
    if source_candidate_id is None:
        analysis_inputs = {
            field: inputs[field] for field in analysis_fields if field in inputs
        }
        if "features" not in analysis_inputs:
            analysis_inputs["features"] = [feature]
        if "methods" not in analysis_inputs and method != "categorical":
            analysis_inputs["methods"] = [method]
        analysis = _validate_univariate_workflow_inputs(
            analysis_inputs,
            whitelist,
            target_col=target_col,
        )
        if feature not in analysis["features"]:
            raise _DraftValidationError(
                f"{workflow} feature 必须包含在本次候选字段 features 中。"
            )
        if method != "categorical" and method not in analysis["methods"]:
            raise _DraftValidationError(
                f"{workflow} method 必须包含在本次数值分箱方法 methods 中。"
            )

    merge_groups = _candidate_merge_groups(
        inputs.get("merge_groups", []),
        name=f"{workflow} merge_groups",
    )
    selection = _candidate_selection(
        inputs["selection"],
        name=f"{workflow} selection",
    )
    uses_source_bin_ids = "source_bin_ids" in selection or bool(merge_groups)
    if uses_source_bin_ids and source_candidate_id is None:
        raise _DraftValidationError(
            f"{workflow} 使用 source bin id 时必须提供用户已查看证据的 "
            "source_candidate_id，不能重新分析后猜测绑定。"
        )
    normalized = {
        **analysis,
        "feature": feature,
        "method": method,
        "merge_groups": merge_groups,
        "selection": selection,
    }
    if source_candidate_id is not None:
        normalized["source_candidate_id"] = source_candidate_id
    if "selection_reason" in inputs:
        reason = _required_text(
            inputs["selection_reason"],
            name=f"{workflow} selection_reason",
        )
        if len(reason) > 500:
            raise _DraftValidationError(
                f"{workflow} selection_reason 最多 500 个字符。"
            )
        normalized["selection_reason"] = reason
    return normalized


def _validate_candidate_monthly_stability_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only the user's immutable source pointer.

    Artifact identity, Pool CAS, active workspace, SampleDesign and month
    semantics are deliberately absent.  The Agent preflight rehydrates those
    platform-owned controls immediately before creating the plan.
    """

    workflow = "candidate_monthly_stability"
    if any(not isinstance(key, str) for key in inputs):
        raise _DraftValidationError(f"{workflow} 字段名必须是文本。")
    fields = set(inputs)
    if fields == {"asset_id"}:
        asset_id = _required_text(
            inputs["asset_id"],
            name=f"{workflow} asset_id",
        )
        if _CANDIDATE_ASSET_ID_RE.fullmatch(asset_id) is None:
            raise _DraftValidationError(
                f"{workflow} asset_id 必须是 candidate-asset- 后接 "
                "32 位小写十六进制字符。"
            )
        return {"asset_id": asset_id}
    if fields == {"strategy_type", "entry_id"}:
        strategy_type = _required_text(
            inputs["strategy_type"],
            name=f"{workflow} strategy_type",
        )
        if strategy_type not in STRATEGY_TYPES:
            raise _DraftValidationError(
                f"{workflow} strategy_type 只能是："
                + "、".join(STRATEGY_TYPES)
                + "。"
            )
        entry_id = _required_text(
            inputs["entry_id"],
            name=f"{workflow} entry_id",
        )
        if _POOL_ENTRY_ID_RE.fullmatch(entry_id) is None:
            raise _DraftValidationError(
                f"{workflow} entry_id 必须是 pool-entry- 后接 "
                "32 位小写十六进制字符。"
            )
        return {
            "strategy_type": strategy_type,
            "entry_id": entry_id,
        }
    platform_fields = sorted(
        fields
        & {
            "source_kind",
            "source_artifact_id",
            "expected_artifact_content_hash",
            "expected_asset_id",
            "expected_asset_hash",
            "expected_pool_revision",
            "expected_pool_snapshot_hash",
            "dataset_id",
            "expected_dataset_content_hash",
            "workspace_revision",
            "workspace_generation",
            "analysis_generation",
            "semantic_mapping_hash",
            "sample_design_ref",
            "target_col",
            "month_col",
            "metrics",
            "psi",
        }
    )
    if platform_fields:
        raise _DraftValidationError(
            f"{workflow} 包含平台拥有的字段："
            + "、".join(platform_fields)
            + "；这些字段必须由 preflight 恢复。",
            code="candidate_monthly_stability_platform_binding_forbidden",
            fields=platform_fields,
        )
    raise _DraftValidationError(
        f"{workflow} 必须且只能提供一个完整 asset_id，"
        "或同时提供明确 strategy_type 与一个完整 entry_id。",
        code="candidate_monthly_stability_source_required",
        fields=("asset_id", "strategy_type", "entry_id"),
    )


def _validate_scorecard_band_build_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only user-owned scorecard banding controls."""

    workflow = "scorecard_band_build"
    allowed = {"bin_count", "raw_pd_band_edges"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if set(inputs) == allowed:
        raise _DraftValidationError(
            f"{workflow} bin_count 与 raw_pd_band_edges 必须二选一；"
            "也可均省略以使用平台默认等频 10 档。"
        )
    if "bin_count" in inputs:
        value = inputs["bin_count"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 2 <= value <= 20
        ):
            raise _DraftValidationError(
                f"{workflow} bin_count 必须是 2 到 20 的整数。"
            )
        return {"bin_count": value}
    if "raw_pd_band_edges" not in inputs:
        return {}
    raw = inputs["raw_pd_band_edges"]
    if (
        not isinstance(raw, Sequence)
        or isinstance(raw, str | bytes | bytearray)
        or not 3 <= len(raw) <= 21
    ):
        raise _DraftValidationError(
            f"{workflow} raw_pd_band_edges 必须包含 3 到 21 个数字。"
        )
    edges: list[float] = []
    for value in raw:
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            raise _DraftValidationError(
                f"{workflow} raw_pd_band_edges 只能包含有限数字。"
            )
        edges.append(float(value))
    if edges[0] != 0.0 or edges[-1] != 1.0:
        raise _DraftValidationError(
            f"{workflow} raw_pd_band_edges 必须从 0.0 开始并以 1.0 结束。"
        )
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise _DraftValidationError(
            f"{workflow} raw_pd_band_edges 必须严格递增。"
        )
    return {"raw_pd_band_edges": edges}


def _validate_scorecard_cutoff_selection_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one exact human-selected cutoff pointer."""

    workflow = "scorecard_cutoff_selection"
    allowed = {"asset_id", "cutoff_id", "reason"}
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    missing = sorted({"asset_id", "cutoff_id"} - set(inputs))
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少字段：" + "、".join(missing) + "。"
        )
    asset_id = _required_text(
        inputs["asset_id"],
        name=f"{workflow} asset_id",
    )
    cutoff_id = _required_text(
        inputs["cutoff_id"],
        name=f"{workflow} cutoff_id",
    )
    if _SCORECARD_BAND_ASSET_ID_RE.fullmatch(asset_id) is None:
        raise _DraftValidationError(
            f"{workflow} asset_id 必须是 scorecard-band-asset- 后接 "
            "32 位小写十六进制字符。"
        )
    if _SCORECARD_CUTOFF_ID_RE.fullmatch(cutoff_id) is None:
        raise _DraftValidationError(
            f"{workflow} cutoff_id 必须是 scorecard-cutoff- 后接 "
            "32 位小写十六进制字符。"
        )
    normalized = {"asset_id": asset_id, "cutoff_id": cutoff_id}
    if "reason" in inputs:
        reason = _required_text(
            inputs["reason"],
            name=f"{workflow} reason",
        )
        if len(reason) > 500:
            raise _DraftValidationError(
                f"{workflow} reason 最多 500 个字符。"
            )
        normalized["reason"] = reason
    return normalized


def _candidate_merge_groups(value: object, *, name: str) -> list[list[str]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or len(value) > 20
    ):
        raise _DraftValidationError(f"{name} 必须是最多 20 个合并组的数组。")
    normalized: list[list[str]] = []
    seen: set[str] = set()
    for group_index, raw_group in enumerate(value):
        if (
            not isinstance(raw_group, Sequence)
            or isinstance(raw_group, str | bytes | bytearray)
            or not 2 <= len(raw_group) <= 20
        ):
            raise _DraftValidationError(
                f"{name}[{group_index}] 必须包含 2 到 20 个 source bin id。"
            )
        group: list[str] = []
        for raw_bin_id in raw_group:
            bin_id = _required_text(raw_bin_id, name=f"{name}[{group_index}]")
            if len(bin_id) > 128:
                raise _DraftValidationError(f"{name} 中的 bin id 最多 128 个字符。")
            if bin_id in seen:
                raise _DraftValidationError(f"{name} 不能重复使用 bin id {bin_id}。")
            seen.add(bin_id)
            group.append(bin_id)
        normalized.append(group)
    return normalized


def _candidate_selection(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise _DraftValidationError(f"{name} 必须是对象。")
    keys = set(value)
    if keys not in ({"source_bin_ids"}, {"risk_threshold"}):
        raise _DraftValidationError(
            f"{name} 必须在 source_bin_ids 与 risk_threshold 中严格二选一。"
        )
    if "source_bin_ids" in value:
        raw_ids = value["source_bin_ids"]
        if (
            not isinstance(raw_ids, Sequence)
            or isinstance(raw_ids, str | bytes | bytearray)
            or not 1 <= len(raw_ids) <= 50
        ):
            raise _DraftValidationError(
                f"{name}.source_bin_ids 必须包含 1 到 50 个 source bin id。"
            )
        bin_ids = [
            _required_text(item, name=f"{name}.source_bin_ids") for item in raw_ids
        ]
        if any(len(bin_id) > 128 for bin_id in bin_ids):
            raise _DraftValidationError(
                f"{name}.source_bin_ids 中每个值最多 128 个字符。"
            )
        if len(bin_ids) != len(set(bin_ids)):
            raise _DraftValidationError(f"{name}.source_bin_ids 不能包含重复值。")
        return {"source_bin_ids": bin_ids}

    threshold = value["risk_threshold"]
    if not isinstance(threshold, Mapping) or any(
        not isinstance(key, str) for key in threshold
    ):
        raise _DraftValidationError(f"{name}.risk_threshold 必须是对象。")
    if set(threshold) != {"operator", "value"}:
        raise _DraftValidationError(
            f"{name}.risk_threshold 只能包含 operator 和 value。"
        )
    operator = _required_text(
        threshold["operator"], name=f"{name}.risk_threshold.operator"
    )
    if operator not in {">=", ">", "<=", "<"}:
        raise _DraftValidationError(
            f"{name}.risk_threshold.operator 只能是 >=、>、<=、<。"
        )
    risk_value = _bounded_number(
        threshold["value"],
        name=f"{name}.risk_threshold.value",
        maximum=1.0,
    )
    return {"risk_threshold": {"operator": operator, "value": risk_value}}


def _validate_strategy_pool_workflow_inputs(
    workflow: str,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate user-owned Pool controls before platform bindings are added.

    Artifact hashes and the current Pool revision/hash are deliberately absent
    here.  The turn handler resolves those values from task-owned persistence at
    plan creation time; an LLM is never allowed to author integrity metadata.
    """

    common = {"strategy_type", "reason"}
    allowed_by_workflow = {
        "strategy_pool_add_candidate": common
        | {
            "candidate_asset_id",
            "selection_id",
            "default_action",
            "action",
            "placement_mode",
        },
        "strategy_pool_remove_entry": common | {"rule_id", "entry_id"},
        "strategy_pool_set_action": common | {"rule_id", "entry_id", "action"},
        "strategy_pool_reorder": common | {"ordered_ids"},
        "strategy_pool_compile": {"strategy_type"},
    }
    _reject_workflow_fields(inputs, allowed_by_workflow[workflow], workflow=workflow)
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"], name=f"{workflow} strategy_type"
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是：" + "、".join(STRATEGY_TYPES) + "。"
        )
    normalized: dict[str, Any] = {"strategy_type": strategy_type}

    if workflow == "strategy_pool_add_candidate":
        missing = sorted({"default_action", "action"} - set(inputs))
        if missing:
            raise _DraftValidationError(
                f"{workflow} 缺少字段：" + "、".join(missing) + "。"
            )
        source_fields = tuple(
            name for name in ("candidate_asset_id", "selection_id") if name in inputs
        )
        if len(source_fields) != 1:
            raise _DraftValidationError(
                f"{workflow} 必须且只能在 candidate_asset_id 与 selection_id 中二选一。"
            )
        source_field = source_fields[0]
        source_id = _required_text(
            inputs[source_field],
            name=f"{workflow} {source_field}",
        )
        if (
            source_field == "candidate_asset_id"
            and _CANDIDATE_ASSET_ID_RE.fullmatch(source_id) is None
        ):
            raise _DraftValidationError(
                f"{workflow} candidate_asset_id 必须是完整 candidate-asset id。"
            )
        if (
            source_field == "selection_id"
            and _AUTOMATIC_TREE_LEAF_SELECTION_ID_RE.fullmatch(source_id) is None
            and _INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_RE.fullmatch(
                source_id
            )
            is None
            and _INTERACTIVE_TREE_FRONTIER_SELECTION_ID_RE.fullmatch(source_id)
            is None
            and _CROSS_MATRIX_CELL_SELECTION_ID_RE.fullmatch(source_id) is None
            and _SCORECARD_CUTOFF_SELECTION_ID_RE.fullmatch(source_id) is None
        ):
            raise _DraftValidationError(
                f"{workflow} selection_id 必须是 automatic-tree-leaf-selection- "
                "、interactive-tree-frontier-group-selection-、"
                "interactive-tree-frontier-selection-、"
                "cross-matrix-cell-selection- 或 "
                "scorecard-cutoff-selection- 后接 32 位小写十六进制字符。"
            )
        normalized.update(
            {
                source_field: source_id,
                "default_action": _strategy_pool_action(
                    inputs["default_action"],
                    strategy_type=strategy_type,
                    name=f"{workflow} default_action",
                ),
                "action": _strategy_pool_action(
                    inputs["action"],
                    strategy_type=strategy_type,
                    name=f"{workflow} action",
                ),
            }
        )
        if "placement_mode" in inputs:
            placement_mode = _required_text(
                inputs["placement_mode"],
                name=f"{workflow} placement_mode",
            )
            if placement_mode not in _POOL_ADD_PLACEMENT_MODES:
                raise _DraftValidationError(
                    f"{workflow} placement_mode 只能是 "
                    "before_selected_members 或 replace_selected_members。"
                )
            normalized["placement_mode"] = placement_mode
    elif workflow in {"strategy_pool_remove_entry", "strategy_pool_set_action"}:
        identifiers = [name for name in ("rule_id", "entry_id") if name in inputs]
        if len(identifiers) != 1:
            raise _DraftValidationError(
                f"{workflow} 必须且只能提供 rule_id 或 entry_id 之一。"
            )
        identifier_name = identifiers[0]
        normalized[identifier_name] = _strategy_pool_identifier(
            inputs[identifier_name],
            name=f"{workflow} {identifier_name}",
        )
        if workflow == "strategy_pool_set_action":
            if "action" not in inputs:
                raise _DraftValidationError(f"{workflow} 缺少 action。")
            normalized["action"] = _strategy_pool_action(
                inputs["action"],
                strategy_type=strategy_type,
                name=f"{workflow} action",
            )
    elif workflow == "strategy_pool_reorder":
        if "ordered_ids" not in inputs:
            raise _DraftValidationError(f"{workflow} 缺少 ordered_ids。")
        raw_order = inputs["ordered_ids"]
        if (
            not isinstance(raw_order, Sequence)
            or isinstance(raw_order, str | bytes | bytearray)
            or not 1 <= len(raw_order) <= 200
        ):
            raise _DraftValidationError(
                f"{workflow} ordered_ids 必须是 1 到 200 个完整 rule_id/entry_id。"
            )
        ordered_ids = [
            _strategy_pool_identifier(item, name=f"{workflow} ordered_ids")
            for item in raw_order
        ]
        if len(set(ordered_ids)) != len(ordered_ids):
            raise _DraftValidationError(f"{workflow} ordered_ids 不能包含重复 ID。")
        normalized["ordered_ids"] = ordered_ids

    if workflow in _POOL_MUTATION_WORKFLOWS and "reason" in inputs:
        reason = _required_text(inputs["reason"], name=f"{workflow} reason")
        if len(reason) > 500:
            raise _DraftValidationError(f"{workflow} reason 最多 500 个字符。")
        normalized["reason"] = reason
    return normalized


def _validate_strategy_pool_apply_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the only two user-owned controls for Pool writeback."""

    workflow = "strategy_pool_apply"
    _reject_workflow_fields(
        inputs,
        {"strategy_type", "output_prefix"},
        workflow=workflow,
    )
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是：" + "、".join(STRATEGY_TYPES) + "。"
        )
    normalized: dict[str, Any] = {"strategy_type": strategy_type}
    if "output_prefix" in inputs:
        output_prefix = _required_text(
            inputs["output_prefix"],
            name=f"{workflow} output_prefix",
        )
        if _POOL_APPLY_SAFE_PREFIX_RE.fullmatch(output_prefix) is None:
            raise _DraftValidationError(
                f"{workflow} output_prefix 必须是最长 48 字符的 ASCII identifier "
                "prefix，且不能以数字开头。"
            )
        normalized["output_prefix"] = output_prefix
    return normalized


def _validate_strategy_pool_materialize_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the single user-owned Pool-to-draft control."""

    workflow = "strategy_pool_materialize"
    _reject_workflow_fields(inputs, {"strategy_type"}, workflow=workflow)
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是：" + "、".join(STRATEGY_TYPES) + "。"
        )
    return {"strategy_type": strategy_type}


def _validate_strategy_pool_validation_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the only two user-owned independent replay controls."""

    workflow = "strategy_pool_validation"
    _reject_workflow_fields(
        inputs,
        {"strategy_type", "partition"},
        workflow=workflow,
    )
    missing = [
        field
        for field in ("strategy_type", "partition")
        if field not in inputs
    ]
    if missing:
        raise _DraftValidationError(
            f"{workflow} 缺少 " + "、".join(missing) + "。"
        )
    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    partition = _required_text(
        inputs["partition"],
        name=f"{workflow} partition",
    )
    if partition not in {"validation", "oot"}:
        raise _DraftValidationError(
            f"{workflow} partition 只能是 validation 或 oot；"
            "development 不是独立样本回放验证。"
        )
    return {
        "strategy_type": strategy_type,
        "partition": partition,
    }


def _validate_strategy_pool_stability_inputs(
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sole user-owned cross-partition stability control."""

    workflow = "strategy_pool_stability"
    _reject_workflow_fields(
        inputs,
        {"strategy_type"},
        workflow=workflow,
    )
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    return {"strategy_type": strategy_type}


def _validate_strategy_pool_impact_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    """Validate only user-owned controls for read-only Pool impact evidence."""

    workflow = "strategy_pool_impact"
    allowed = {
        "strategy_type",
        "comparison_mode",
        "baseline_strategy_id",
        "month_col",
        "loan_amount_col",
        "overdue_amount_col",
        "drop_nan_labels",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"], name=f"{workflow} strategy_type"
    )
    if strategy_type not in {"approval", "reject"}:
        raise _DraftValidationError(
            "strategy_pool_impact 首个 V2 纵切只支持 approval 或 reject Pool；"
            "limit、pricing 与 segmentation 的专属影响口径将在 V2 后续纵切交付，"
            "当前不会套用准入/拒绝口径。"
        )

    comparison_mode = inputs.get("comparison_mode", "absolute")
    if not isinstance(comparison_mode, str) or comparison_mode not in {
        "absolute",
        "vs_baseline",
    }:
        raise _DraftValidationError(
            f"{workflow} comparison_mode 只能是 absolute 或 vs_baseline。"
        )
    baseline_strategy_id = None
    if "baseline_strategy_id" in inputs:
        baseline_strategy_id = _required_text(
            inputs["baseline_strategy_id"],
            name=f"{workflow} baseline_strategy_id",
        )
    if comparison_mode == "vs_baseline" and baseline_strategy_id is None:
        raise _DraftValidationError(
            f"{workflow} 使用 vs_baseline 时必须提供 baseline_strategy_id。"
        )
    if comparison_mode == "absolute" and baseline_strategy_id is not None:
        raise _DraftValidationError(
            f"{workflow} 使用 absolute 时禁止提供 baseline_strategy_id。"
        )

    normalized: dict[str, Any] = {
        "strategy_type": strategy_type,
        "comparison_mode": comparison_mode,
    }
    if baseline_strategy_id is not None:
        normalized["baseline_strategy_id"] = baseline_strategy_id
    for field in ("month_col", "loan_amount_col", "overdue_amount_col"):
        if field in inputs:
            normalized[field] = _workflow_column(
                inputs[field],
                name=f"{workflow} {field}",
                whitelist=whitelist,
            )
    drop_nan_labels = inputs.get("drop_nan_labels", False)
    if not isinstance(drop_nan_labels, bool):
        raise _DraftValidationError(
            f"{workflow} drop_nan_labels 必须是布尔值。"
        )
    normalized["drop_nan_labels"] = drop_nan_labels
    return normalized


def _validate_strategy_impact_cube_inputs(
    inputs: Mapping[str, Any],
    whitelist: tuple[str, ...],
) -> dict[str, Any]:
    """Validate only user-owned controls for the unified read-only ImpactCube."""

    workflow = "strategy_impact_cube"
    allowed = {
        "strategy_type",
        "partitions",
        "month_col",
        "group_col",
        "segment_col",
        "current_strategy_id",
        "economics_inputs",
    }
    _reject_workflow_fields(inputs, allowed, workflow=workflow)
    if "strategy_type" not in inputs:
        raise _DraftValidationError(f"{workflow} 缺少 strategy_type。")
    strategy_type = _required_text(
        inputs["strategy_type"],
        name=f"{workflow} strategy_type",
    )
    if strategy_type not in STRATEGY_TYPES:
        raise _DraftValidationError(
            f"{workflow} strategy_type 只能是："
            + "、".join(STRATEGY_TYPES)
            + "。"
        )
    normalized: dict[str, Any] = {"strategy_type": strategy_type}

    if "partitions" in inputs:
        raw_partitions = inputs["partitions"]
        if (
            not isinstance(raw_partitions, Sequence)
            or isinstance(raw_partitions, str | bytes | bytearray)
            or not 1 <= len(raw_partitions) <= len(_IMPACT_CUBE_PARTITION_ORDER)
        ):
            raise _DraftValidationError(
                f"{workflow} partitions 必须是 1 到 3 个明确分区。"
            )
        partitions = [
            _required_text(
                item,
                name=f"{workflow} partitions",
            )
            for item in raw_partitions
        ]
        if (
            len(set(partitions)) != len(partitions)
            or any(item not in _IMPACT_CUBE_PARTITION_ORDER for item in partitions)
        ):
            raise _DraftValidationError(
                f"{workflow} partitions 只能无重复地使用 "
                "development、validation、oot。"
            )
        normalized["partitions"] = [
            item
            for item in _IMPACT_CUBE_PARTITION_ORDER
            if item in partitions
        ]

    selected_dimension_columns: list[str] = []
    for field in ("month_col", "group_col", "segment_col"):
        if field not in inputs:
            continue
        column = _workflow_column(
            inputs[field],
            name=f"{workflow} {field}",
            whitelist=whitelist,
        )
        normalized[field] = column
        selected_dimension_columns.append(column)
    if len(set(selected_dimension_columns)) != len(selected_dimension_columns):
        raise _DraftValidationError(
            f"{workflow} month/group/segment 必须绑定不同列。"
        )

    if "current_strategy_id" in inputs:
        normalized["current_strategy_id"] = _required_text(
            inputs["current_strategy_id"],
            name=f"{workflow} current_strategy_id",
        )

    if "economics_inputs" in inputs and inputs["economics_inputs"] is not None:
        raw_economics = inputs["economics_inputs"]
        if (
            not isinstance(raw_economics, Mapping)
            or not 1 <= len(raw_economics) <= 16
            or any(not isinstance(key, str) for key in raw_economics)
        ):
            raise _DraftValidationError(
                f"{workflow} economics_inputs 必须是 1 到 16 个 typed bindings。"
            )
        allowed_components = _IMPACT_CUBE_ECONOMICS_COMPONENTS[strategy_type]
        unsupported = sorted(set(raw_economics) - allowed_components)
        if unsupported:
            raise _DraftValidationError(
                f"{workflow} {strategy_type} 不支持 economics_inputs："
                + "、".join(unsupported)
                + "。"
            )
        economics: dict[str, dict[str, Any]] = {}
        for component in sorted(raw_economics):
            binding = raw_economics[component]
            if not isinstance(binding, Mapping):
                raise _DraftValidationError(
                    f"{workflow} economics_inputs.{component} 必须是 typed binding。"
                )
            kind = binding.get("kind")
            if kind == "column" and set(binding) == {"kind", "column"}:
                economics[component] = {
                    "kind": "column",
                    "column": _workflow_column(
                        binding["column"],
                        name=(
                            f"{workflow} economics_inputs.{component}.column"
                        ),
                        whitelist=whitelist,
                    ),
                }
            elif kind == "scalar" and set(binding) == {"kind", "value"}:
                value = binding["value"]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    or (
                        isinstance(value, int)
                        and abs(value) > 2**53 - 1
                    )
                ):
                    raise _DraftValidationError(
                        f"{workflow} economics_inputs.{component}.value "
                        "必须是有限且可精确表示的数字。"
                    )
                if component == "term_months" and (
                    not isinstance(value, int) or value < 1
                ):
                    raise _DraftValidationError(
                        f"{workflow} economics_inputs.term_months "
                        "必须是正整数月数。"
                    )
                economics[component] = {
                    "kind": "scalar",
                    "value": value,
                }
            else:
                raise _DraftValidationError(
                    f"{workflow} economics_inputs.{component} "
                    "只能是 column 或 scalar binding。"
                )
        normalized["economics_inputs"] = economics
    return normalized


def _strategy_pool_action(
    value: object,
    *,
    strategy_type: str,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _DraftValidationError(f"{name} 必须是 typed StrategyAction 对象。")
    try:
        action = StrategyAction.from_dict(value)
    except StrategyError as exc:
        raise _DraftValidationError(f"{name} 无效：{exc}") from exc
    if action.type not in _POOL_ACTION_TYPES[strategy_type]:
        raise _DraftValidationError(
            f"{name} 的 {action.type} 动作不适用于 {strategy_type} 策略。"
        )
    return action.to_dict()


def _strategy_pool_identifier(value: object, *, name: str) -> str:
    identifier = _required_text(value, name=name)
    if _POOL_ITEM_ID_RE.fullmatch(identifier) is None:
        raise _DraftValidationError(f"{name} 必须是完整、安全的 rule_id 或 entry_id。")
    return identifier


def _utterance_targets_voting_candidate_search(utterance: str) -> bool:
    """Reserve explicit Voting combination search before exact-member build."""

    return (
        not _utterance_targets_voting_search_selection(utterance)
        and _VOTING_SUBJECT_RE.search(utterance) is not None
        and _VOTING_SEARCH_INTENT_RE.search(utterance) is not None
    )


def _utterance_targets_cross_candidate_search(utterance: str) -> bool:
    """Reserve bounded Cross feature-pair search before explicit matrix build."""

    return (
        not _utterance_targets_cross_rule_search(utterance)
        and not _utterance_targets_cross_rule_selection(utterance)
        and not _utterance_targets_cross_search_selection(utterance)
        and _CROSS_SEARCH_INTENT_RE.search(utterance) is not None
    )


def _utterance_targets_cross_rule_search(utterance: str) -> bool:
    """Reserve bounded threshold-rule mining before Matrix pair routing."""

    return (
        not _utterance_targets_cross_rule_selection(utterance)
        and _CROSS_RULE_SUBJECT_RE.search(utterance) is not None
        and _CROSS_RULE_SEARCH_INTENT_RE.search(utterance) is not None
    )


def _utterance_targets_cross_rule_selection(utterance: str) -> bool:
    """Reserve one exact rule materialization before every Cross route."""

    return (
        _CROSS_RULE_SEARCH_ID_TOKEN_RE.search(utterance) is not None
        and _CROSS_RULE_ID_TOKEN_RE.search(utterance) is not None
        and _CROSS_RULE_SELECTION_INTENT_RE.search(utterance) is not None
    )


def _utterance_targets_cross_search_selection(utterance: str) -> bool:
    """Reserve exact search-pair materialization before search/build routes."""

    return (
        _CROSS_SEARCH_ID_TOKEN_RE.search(utterance) is not None
        and _CROSS_PAIR_ID_TOKEN_RE.search(utterance) is not None
        and _CROSS_SEARCH_SELECTION_INTENT_RE.search(utterance) is not None
    )


def _cross_search_has_positive_follow_up(utterance: str) -> bool:
    """Ignore explicit negative disclaimers while rejecting chained actions."""

    return _cross_search_pattern_has_positive(
        utterance,
        _CROSS_SEARCH_FOLLOW_UP_RE,
    )


def _cross_search_pattern_has_positive(
    utterance: str,
    pattern: re.Pattern[str],
) -> bool:
    for match in pattern.finditer(utterance):
        prefix = utterance[max(0, match.start() - 20) : match.start()]
        local_start = max(
            prefix.rfind(separator)
            for separator in ("，", ",", "；", ";", "。", ".", "!", "！", "?", "？")
        )
        if _CROSS_SEARCH_NEGATION_PREFIX_RE.search(
            prefix[local_start + 1 :]
        ) is None:
            return True
    return False


def _cross_search_selection_has_positive_research(utterance: str) -> bool:
    """Search-like substrings inside exact pointer ids are not new commands."""

    scrubbed = _CROSS_SEARCH_ID_TOKEN_RE.sub(
        lambda match: " " * len(match.group(0)),
        utterance,
    )
    scrubbed = _CROSS_PAIR_ID_TOKEN_RE.sub(
        lambda match: " " * len(match.group(0)),
        scrubbed,
    )
    return _CROSS_SEARCH_RESEARCH_RE.search(scrubbed) is not None


def _utterance_targets_voting_search_selection(utterance: str) -> bool:
    """Reserve an exact search-result materialization before search/build."""

    return (
        _VOTING_SEARCH_ID_TOKEN_RE.search(utterance) is not None
        and _VOTING_COMBO_ID_TOKEN_RE.search(utterance) is not None
        and _VOTING_SEARCH_SELECTION_INTENT_RE.search(utterance) is not None
    )


def _utterance_chains_voting_search_operation(utterance: str) -> bool:
    """Detect a positive lifecycle follow-up even without a connector word."""

    search_seen = False
    for clause_match in _VOTING_COMMAND_CLAUSE_RE.finditer(utterance):
        clause = clause_match.group(0)
        if not search_seen:
            search_match = _VOTING_SEARCH_INTENT_RE.search(clause)
            search_seen = (
                _VOTING_SUBJECT_RE.search(clause) is not None
                and search_match is not None
            )
            if search_seen and search_match is not None:
                if _voting_search_text_has_positive_follow_up(
                    clause[: search_match.start()]
                ) or _voting_search_text_has_positive_follow_up(
                    clause[search_match.end() :]
                ):
                    return True
            continue
        if _voting_search_text_has_positive_follow_up(clause):
            return True
    return False


def _voting_search_text_has_positive_follow_up(text: str) -> bool:
    """Evaluate each follow-up operation against only its local polarity."""

    previous_end = 0
    for match in _VOTING_SEARCH_FOLLOW_UP_OPERATION_RE.finditer(text):
        local_start = max(
            previous_end,
            *(
                text.rfind(separator, previous_end, match.start()) + 1
                for separator in (
                    "，",
                    ",",
                    "；",
                    ";",
                    "。",
                    ".",
                    "!",
                    "！",
                    "?",
                    "？",
                )
            ),
        )
        fragment = text[local_start : match.end()]
        if _VOTING_SEARCH_NEGATED_FOLLOW_UP_RE.search(fragment) is None:
            return True
        previous_end = match.end()
    return False


def _voting_search_selection_has_positive_follow_up(text: str) -> bool:
    """Reject lifecycle chaining while allowing explicit negative disclaimers."""

    previous_end = 0
    for match in _VOTING_SEARCH_SELECTION_FOLLOW_UP_RE.finditer(text):
        clause_start = max(
            text.rfind(separator, 0, match.start()) + 1
            for separator in ("；", ";", "。", ".", "!", "！", "?", "？")
        )
        clause_prefix = text[clause_start : match.start()]
        english_negation = tuple(
            re.finditer(
                r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|never|without)"
                r"(?![A-Za-z0-9_])",
                clause_prefix,
                re.IGNORECASE,
            )
        )
        if english_negation:
            after_negation = clause_prefix[english_negation[-1].end() :]
            if re.search(
                r"(?<![A-Za-z0-9_])(?:but|however|then)(?![A-Za-z0-9_])",
                after_negation,
                re.IGNORECASE,
            ) is None:
                previous_end = match.end()
                continue
        local_start = max(
            previous_end,
            *(
                text.rfind(separator, previous_end, match.start()) + 1
                for separator in (
                    "，",
                    ",",
                    "；",
                    ";",
                    "。",
                    ".",
                    "!",
                    "！",
                    "?",
                    "？",
                )
            ),
        )
        fragment = text[local_start : match.end()]
        if _VOTING_SEARCH_NEGATED_FOLLOW_UP_RE.search(fragment) is None:
            return True
        previous_end = match.end()
    return False


def _voting_search_selection_has_positive_research(text: str) -> bool:
    """Detect a second search command, excluding an explicitly negated one."""

    scrubbed = _VOTING_SEARCH_ID_TOKEN_RE.sub(
        lambda match: " " * len(match.group(0)),
        text,
    )
    scrubbed = _VOTING_COMBO_ID_TOKEN_RE.sub(
        lambda match: " " * len(match.group(0)),
        scrubbed,
    )
    for match in _VOTING_SEARCH_SELECTION_RESEARCH_RE.finditer(scrubbed):
        local_start = max(
            scrubbed.rfind(separator, 0, match.start()) + 1
            for separator in ("，", ",", "；", ";", "。", ".", "!", "！", "?", "？")
        )
        if _VOTING_SEARCH_SELECTION_NEGATED_RESEARCH_RE.search(
            scrubbed[local_start : match.end()]
        ) is None:
            return True
    return False


def _utterance_targets_voting_candidate(utterance: str) -> bool:
    """Keep an explicit Voting request out of generic lifecycle/workflow routes."""

    return (
        _VOTING_SUBJECT_RE.search(utterance) is not None
        and len(tuple(_VOTING_RULE_ID_TOKEN_RE.finditer(utterance))) >= 2
    )


def _voting_positive_command_clause_spans(
    utterance: str,
) -> tuple[tuple[int, int], ...]:
    """Return one span per positive Voting command, preserving duplicates."""

    spans: list[tuple[int, int]] = []
    for clause_match in _VOTING_COMMAND_CLAUSE_RE.finditer(utterance):
        clause = clause_match.group(0)
        for command_match in _VOTING_BUILD_INTENT_RE.finditer(clause):
            prefix = clause[: command_match.start()]
            comma = max(prefix.rfind("，"), prefix.rfind(","))
            local_start = comma + 1
            reset = _VOTING_COMMAND_RESET_RE.search(prefix)
            if reset is not None:
                local_start = max(local_start, reset.start())
            spans.append(
                (clause_match.start() + local_start, clause_match.end())
            )
    return tuple(spans)


def _utterance_targets_cross_matrix(utterance: str) -> bool:
    if (
        _utterance_targets_cross_rule_search(utterance)
        or _utterance_targets_cross_rule_selection(utterance)
    ):
        return False
    without_selection_ids = _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE.sub(
        " ", utterance
    )
    return _CROSS_MATRIX_TARGET_RE.search(without_selection_ids) is not None


def _utterance_targets_cross_matrix_cell_selection(utterance: str) -> bool:
    has_pointer_ids = (
        _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.search(utterance) is not None
        and _CROSS_MATRIX_CELL_ID_TOKEN_RE.search(utterance) is not None
    )
    if (
        _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE.search(utterance) is not None
        and not has_pointer_ids
    ):
        return False
    explicit_cell_action = (
        _CROSS_MATRIX_CELL_SELECTION_ACTION_RE.search(utterance) is not None
    )
    return (explicit_cell_action and _utterance_targets_cross_matrix(utterance)) or (
        has_pointer_ids
        and _CROSS_MATRIX_CELL_SELECTION_VERB_RE.search(utterance) is not None
    )


def _cross_positive_command_clause_spans(
    utterance: str,
) -> tuple[tuple[int, int], ...]:
    """Return clauses that contain one positive Cross build request."""

    spans: list[tuple[int, int]] = []
    for clause_match in _CROSS_MATRIX_COMMAND_CLAUSE_RE.finditer(utterance):
        clause = clause_match.group(0)
        if (
            _CROSS_MATRIX_TARGET_RE.search(clause) is not None
            and _CROSS_MATRIX_BUILD_RE.search(clause) is not None
            and _CROSS_MATRIX_NEGATED_BUILD_RE.search(clause) is None
        ):
            spans.append(clause_match.span())
    return tuple(spans)


def _cross_mention_is_within(
    start: int,
    end: int,
    command_span: tuple[int, int],
) -> bool:
    return command_span[0] <= start and end <= command_span[1]


def _cross_spans_are_near(
    utterance: str,
    first: tuple[int, int],
    second: tuple[int, int],
    *,
    maximum_gap: int = 32,
) -> bool:
    first_start, first_end = first
    second_start, second_end = second
    if first_end <= second_start:
        gap_start, gap_end = first_end, second_start
    elif second_end <= first_start:
        gap_start, gap_end = second_end, first_start
    else:
        gap_start = gap_end = max(first_start, second_start)
    return (
        gap_end - gap_start <= maximum_gap
        and not any(
            separator in utterance[gap_start:gap_end]
            for separator in ("；", ";", "。", "\n")
        )
    )


def _cross_method_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        sorted(
            (
                (method, match.start(), match.end())
                for method, pattern in _CROSS_METHOD_GROUNDING.items()
                for match in pattern.finditer(utterance)
            ),
            key=lambda item: (item[1], item[2], item[0]),
        )
    )


def _cross_axis_method_is_grounded(
    utterance: str,
    *,
    feature: str,
    method: str,
    whitelist: Sequence[str],
    shared_method: bool,
    command_span: tuple[int, int],
) -> bool:
    feature_spans = [
        (start, end)
        for start, end, column in _automatic_tree_column_mentions(
            utterance,
            whitelist,
        )
        if column == feature
        and _cross_mention_is_within(start, end, command_span)
        and not _automatic_tree_span_is_negated(
            utterance,
            start=start,
            end=end,
        )
    ]
    method_spans = [
        (start, end)
        for observed_method, start, end in _cross_method_mentions(utterance)
        if observed_method == method
        and _cross_mention_is_within(start, end, command_span)
        and not _automatic_tree_span_is_negated(
            utterance,
            start=start,
            end=end,
        )
    ]
    if shared_method:
        return bool(feature_spans and method_spans)
    return any(
        _cross_spans_are_near(
            utterance,
            (feature_start, feature_end),
            (method_start, method_end),
        )
        for feature_start, feature_end in feature_spans
        for method_start, method_end in method_spans
    )


def _cross_amount_column_is_grounded(
    utterance: str,
    *,
    column: str,
    field: str,
    whitelist: Sequence[str],
    command_span: tuple[int, int],
) -> bool:
    label_pattern = (
        re.compile(r"(?:放款|授信|借款)金额|loan[_\s-]*amount", re.IGNORECASE)
        if field == "loan_amount_col"
        else re.compile(r"(?:逾期|坏账|损失)金额|overdue[_\s-]*amount", re.IGNORECASE)
    )
    column_spans = [
        (start, end)
        for start, end, observed in _automatic_tree_column_mentions(
            utterance,
            whitelist,
        )
        if observed == column and _cross_mention_is_within(start, end, command_span)
    ]
    label_spans = [
        match.span()
        for match in label_pattern.finditer(utterance)
        if _cross_mention_is_within(match.start(), match.end(), command_span)
    ]
    return any(
        _cross_spans_are_near(utterance, column_span, label_span, maximum_gap=48)
        for column_span in column_spans
        for label_span in label_spans
    )


def _cross_sentinel_literal(token: str) -> str | int | float | None:
    """Parse one explicitly written sentinel without guessing its JSON type."""

    value = token.strip()
    if not value:
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        if value[0] == '"':
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, str) else None
        inner = value[1:-1]
        return inner if "\\" not in inner else None
    if _CROSS_MATRIX_SENTINEL_NUMBER_RE.fullmatch(value) is not None:
        try:
            parsed = float(value) if any(mark in value.lower() for mark in (".", "e")) else int(value)
        except ValueError:
            return None
        if isinstance(parsed, float) and not math.isfinite(parsed):
            return None
        return parsed
    if re.fullmatch(r"[^\s，,、/;；。.!！?？]+", value) is not None:
        return value
    return None


def _cross_explicit_sentinel_values(
    utterance: str,
    *,
    command_span: tuple[int, int],
) -> tuple[tuple[str | int | float, ...] | None, bool]:
    """Return the exact sentinel sequence named in the positive command.

    ``None`` means no sentinel control was present. The boolean marks syntax that
    cannot be interpreted without guessing, which must fail closed.
    """

    command = utterance[command_span[0] : command_span[1]]
    labels = tuple(_CROSS_MATRIX_SENTINEL_LABEL_RE.finditer(command))
    if not labels:
        return None, False
    if len(labels) != 1:
        return (), True
    label = labels[0]
    prefix = command[: label.start()]
    reverse = re.search(r"(?:作为|当作|视为|按)\s*$", prefix)
    if reverse is not None:
        start = max(
            prefix.rfind("，", 0, reverse.start()),
            prefix.rfind(",", 0, reverse.start()),
        )
        body = prefix[start + 1 : reverse.start()]
    else:
        body = command[label.end() :]
        body = re.sub(
            r"^\s*(?:=|:|：|为|是|包括|包含|采用|使用|用)\s*",
            "",
            body,
        )
        stop = _CROSS_MATRIX_SENTINEL_STOP_RE.search(body)
        if stop is not None:
            body = body[: stop.start()]
    body = body.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1].strip()
    if not body:
        return (), True
    raw_tokens = re.split(r"\s*(?:、|，|,|/|和|与|及|\band\b)\s*", body)
    if not raw_tokens or any(not token for token in raw_tokens):
        return (), True
    values: list[str | int | float] = []
    identities: set[str] = set()
    for token in raw_tokens:
        value = _cross_sentinel_literal(token)
        if value is None:
            return (), True
        identity = json.dumps(
            [type(value).__name__, value],
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        if identity in identities:
            return (), True
        identities.add(identity)
        values.append(value)
    return tuple(values), False


def _cross_analysis_controls_not_grounded(
    utterance: str,
    *,
    inputs: Mapping[str, Any],
    whitelist: Sequence[str],
    command_span: tuple[int, int],
) -> tuple[str, ...]:
    missing: list[str] = []
    bin_mentions = tuple(_CROSS_MATRIX_BIN_COUNT_RE.finditer(utterance))
    if any(
        not _cross_mention_is_within(match.start(), match.end(), command_span)
        for match in bin_mentions
    ):
        missing.append("bin_count")
    observed_bin_counts = {int(match.group("count")) for match in bin_mentions}
    if observed_bin_counts:
        if observed_bin_counts != {inputs["bin_count"]}:
            missing.append("bin_count")
    elif inputs["bin_count"] != 10:
        missing.append("bin_count")

    min_pct_mentions = tuple(_CROSS_MATRIX_MIN_BIN_PCT_RE.finditer(utterance))
    if any(
        not _cross_mention_is_within(match.start(), match.end(), command_span)
        for match in min_pct_mentions
    ):
        missing.append("min_bin_pct")
    observed_min_pcts = {
        float(match.group("value")) / (100.0 if match.group("pct") else 1.0)
        for match in min_pct_mentions
    }
    if observed_min_pcts:
        if len(observed_min_pcts) != 1 or not math.isclose(
            next(iter(observed_min_pcts)),
            float(inputs["min_bin_pct"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            missing.append("min_bin_pct")
    elif not math.isclose(
        float(inputs["min_bin_pct"]),
        0.02,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        missing.append("min_bin_pct")

    for field in ("loan_amount_col", "overdue_amount_col"):
        if field in inputs and not _cross_amount_column_is_grounded(
            utterance,
            column=str(inputs[field]),
            field=field,
            whitelist=whitelist,
            command_span=command_span,
        ):
            missing.append(field)
    observed_sentinels, sentinel_syntax_ambiguous = _cross_explicit_sentinel_values(
        utterance,
        command_span=command_span,
    )
    expected_sentinels = tuple(inputs["sentinel_values"])
    if observed_sentinels is None:
        if expected_sentinels:
            missing.append("sentinel_values")
    else:
        observed_identities = {
            json.dumps(
                [type(value).__name__, value],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in observed_sentinels
        }
        expected_identities = {
            json.dumps(
                [type(value).__name__, value],
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in expected_sentinels
        }
        if sentinel_syntax_ambiguous or observed_identities != expected_identities:
            missing.append("sentinel_values")
    observed_breakpoints, breakpoint_syntax_ambiguous = (
        _explicit_manual_breakpoint_bindings(
            utterance,
            whitelist=whitelist,
            command_span=command_span,
        )
    )
    expected_breakpoints = inputs.get("manual_breakpoints", {})
    if (
        breakpoint_syntax_ambiguous
        or observed_breakpoints != expected_breakpoints
    ):
        missing.append("manual_breakpoints")
    return tuple(dict.fromkeys(missing))


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
                    for left, right in zip(points, points[1:])
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


def _ground_cross_matrix_analysis(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Require an explicit positive 2D matrix command and two grounded axes."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _CROSS_MATRIX_NEGATED_BUILD_RE.search(utterance) is not None:
        return _clarification(
            "原话否定了二维 Cross Matrix 构建，因此本次不会执行。",
            code="cross_matrix_build_intent_negated",
            fields=("build_intent",),
        )
    if (
        _CROSS_MATRIX_NONCOMMAND_RE.search(utterance) is not None
        or _CROSS_MATRIX_POSTPONED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "当前原话是问句、假设/未来/历史描述、演示性文本或已在句尾撤销，"
            "不能视为立即执行二维 Cross Matrix 的唯一正向命令。请单独重述本次"
            "要构建的两个有序轴、各自分箱方法和明确分析参数。",
            code="cross_matrix_positive_command_required",
            fields=("build_intent",),
        )
    command_spans = _cross_positive_command_clause_spans(utterance)
    if not command_spans:
        return _clarification(
            "请明确发出一次正向的二维 Cross Matrix 构建命令；查看、说明或"
            "假设性请求不会创建候选资产。",
            code="cross_matrix_build_intent_required",
            fields=("build_intent",),
        )
    if len(command_spans) != 1:
        return _clarification(
            "一次请求只能包含一个立即执行的二维 Cross Matrix 构建子句；"
            "请把不同轴组合拆成独立请求。",
            code="cross_matrix_single_command_required",
            fields=("build_intent",),
        )
    command_span = command_spans[0]
    if (
        len(
            tuple(
                _CROSS_MATRIX_TARGET_RE.finditer(
                    utterance[command_span[0] : command_span[1]]
                )
            )
        )
        != 1
    ):
        return _clarification(
            "一次请求只能构建一个二维 Cross Matrix；请把多个矩阵拆开。",
            code="cross_matrix_single_command_required",
            fields=("build_intent",),
        )
    if _CROSS_MATRIX_FOLLOW_UP_RE.search(utterance) is not None:
        return _clarification(
            "本轮只能生成二维 Cross Matrix 及 development evidence。"
            "选格、入池、代码、写回、采纳或部署必须拆成后续请求。",
            code="cross_matrix_single_step_required",
            fields=("next_action",),
        )
    if _CROSS_MATRIX_CONTROL_REWRITE_RE.search(utterance) is not None:
        return _clarification(
            "原话包含被否定或随后改写的轴/分箱控制。请只保留最终的一组"
            "有序轴和分箱方法后重新发送，平台不会替你选择新旧值。",
            code="cross_matrix_controls_rewritten",
            fields=("x_feature", "x_method", "y_feature", "y_method"),
        )

    mentions, ambiguous = _automatic_tree_column_mention_resolution(
        utterance,
        whitelist,
    )
    if ambiguous:
        return _clarification(
            "交叉轴字段在原话中存在重叠或大小写歧义，请用分隔符写出两个"
            "准确列名：" + "、".join(ambiguous) + "。",
            code="cross_matrix_axes_ambiguous",
            fields=ambiguous,
        )
    if any(
        not _cross_mention_is_within(start, end, command_span)
        for start, end, _column in mentions
    ):
        return _clarification(
            "二维 Cross Matrix 的字段和分析列必须全部位于唯一正向构建子句中；"
            "历史、引用、否定或其他子句中的列不会被消费。",
            code="cross_matrix_controls_outside_command",
            fields=("x_feature", "y_feature"),
        )
    if any(
        _automatic_tree_span_is_negated(
            utterance,
            start=start,
            end=end,
        )
        for start, end, _column in mentions
    ):
        return _clarification(
            "原话包含被否定的字段控制。请只保留最终要使用的两个有序轴。",
            code="cross_matrix_controls_rewritten",
            fields=("x_feature", "y_feature"),
        )

    positive_mentions = [
        (start, end, column)
        for start, end, column in mentions
    ]
    expected_columns = {inputs["x_feature"], inputs["y_feature"]}
    expected_columns.update(
        inputs[field]
        for field in ("loan_amount_col", "overdue_amount_col")
        if field in inputs
    )
    observed_columns = {column for _start, _end, column in positive_mentions}
    if not {inputs["x_feature"], inputs["y_feature"]} <= observed_columns:
        return _clarification(
            "请在原话中明确写出两个不同的交叉轴字段；平台不会从列白名单"
            "补齐或猜测第二个轴。",
            code="cross_matrix_axes_not_grounded",
            fields=("x_feature", "y_feature"),
        )
    if observed_columns != expected_columns:
        return _clarification(
            "请在唯一构建子句中只写出一个明确轴对及已声明的金额列；"
            "平台不会从额外字段中挑选两个轴，也不会遗漏用户点名的字段。",
            code="cross_matrix_axes_not_unique",
            fields=("x_feature", "y_feature"),
        )

    axis_order: list[str] = []
    for _start, _end, column in positive_mentions:
        if (
            column in {inputs["x_feature"], inputs["y_feature"]}
            and column not in axis_order
        ):
            axis_order.append(column)
    if axis_order != [inputs["x_feature"], inputs["y_feature"]]:
        return _clarification(
            "矩阵 X/Y 方向必须与原话中两个轴的首次出现顺序一致；"
            "平台不会让模型任意转置后生成不同 asset hash。",
            code="cross_matrix_axis_order_not_grounded",
            fields=("x_feature", "y_feature"),
        )

    method_mentions = _cross_method_mentions(utterance)
    if any(
        not _cross_mention_is_within(start, end, command_span)
        for _method, start, end in method_mentions
    ):
        return _clarification(
            "两个轴的分箱方法必须全部位于唯一正向构建子句中。",
            code="cross_matrix_controls_outside_command",
            fields=("x_method", "y_method"),
        )
    if any(
        _automatic_tree_span_is_negated(
            utterance,
            start=start,
            end=end,
        )
        for _method, start, end in method_mentions
    ):
        return _clarification(
            "原话包含被否定的分箱方法。请只保留最终使用的方法。",
            code="cross_matrix_controls_rewritten",
            fields=("x_method", "y_method"),
        )
    if {method for method, _start, _end in method_mentions} != {
        inputs["x_method"],
        inputs["y_method"],
    }:
        return _clarification(
            "原话中的分箱方法与结构化草案不唯一或不一致；"
            "平台不会补全、替换或遗漏方法。",
            code="cross_matrix_methods_not_grounded",
            fields=("x_method", "y_method"),
        )

    shared_method = inputs["x_method"] == inputs["y_method"]
    missing_methods = [
        field
        for field, feature_field in (
            ("x_method", "x_feature"),
            ("y_method", "y_feature"),
        )
        if not _cross_axis_method_is_grounded(
            utterance,
            feature=inputs[feature_field],
            method=inputs[field],
            whitelist=whitelist,
            shared_method=shared_method,
            command_span=command_span,
        )
    ]
    if missing_methods:
        return _clarification(
            "请明确两个轴各自使用的分箱方法；相同方法可说明一次，混合方法"
            "必须分别紧邻对应字段。平台不会替你选择方法。",
            code="cross_matrix_methods_not_grounded",
            fields=tuple(missing_methods),
        )
    missing_analysis_controls = _cross_analysis_controls_not_grounded(
        utterance,
        inputs=inputs,
        whitelist=whitelist,
        command_span=command_span,
    )
    if missing_analysis_controls:
        return _clarification(
            "目标箱数、最小箱占比、金额列和哨兵值只能采用原话明确值；"
            "未写明时只能使用平台默认值，不能由模型另选。",
            code="cross_matrix_analysis_controls_not_grounded",
            fields=missing_analysis_controls,
        )
    return result


def _voting_strategy_type_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (strategy_type, match.start(), match.end())
        for strategy_type, pattern in _POOL_STRATEGY_TYPE_GROUNDING.items()
        for match in pattern.finditer(utterance)
    )


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


def _voting_n_mentions(
    utterance: str,
) -> tuple[tuple[int, int | None, int, int], ...]:
    mentions: list[tuple[int, int | None, int, int]] = []
    for pattern in _VOTING_N_PATTERNS:
        for match in pattern.finditer(utterance):
            k_token = match.groupdict().get("k")
            mentions.append(
                (
                    int(match.group("n")),
                    None if k_token is None else int(k_token),
                    match.start(),
                    match.end(),
                )
            )
    return tuple(mentions)


def _voting_mention_is_within(
    mention_start: int,
    mention_end: int,
    command_span: tuple[int, int],
) -> bool:
    return (
        command_span[0] <= mention_start
        and mention_end <= command_span[1]
    )


def _is_canonical_stored_strategy_report_request(
    draft: CompiledStrategyRequestDraft | None,
) -> bool:
    """Keep a fully identified stored-strategy report on its legacy route."""

    return bool(
        isinstance(draft, StrategyRequestDraft)
        and draft.operation == "report"
        and draft.strategy_spec is None
        and draft.strategy_id
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


def _ground_refinement_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
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


def utterance_targets_strategy_sample_design(utterance: str) -> bool:
    """Return true only when a build verb targets the sample-design subject.

    References such as ``基于已固化的样本设计构建自动树`` must remain with
    their downstream workflow, so co-occurrence anywhere in a clause is not
    sufficient authorization to materialize a new sample design.
    """

    before_subject_actions = {
        "创建",
        "生成",
        "构建",
        "设计",
        "固化",
        "冻结",
        "物化",
        "create",
        "build",
        "design",
        "freeze",
        "materialize",
    }
    after_subject_actions = {
        "创建",
        "生成",
        "固化",
        "冻结",
        "物化",
        "create",
        "freeze",
        "materialize",
    }
    reference_prefix = re.compile(
        r"(?:基于|使用|参考|依据|按照|依赖|沿用|using|based\s+on|refer(?:ring)?\s+to)\s*$",
        re.I,
    )
    past_prefix = re.compile(
        r"(?:已|已经|曾|曾经|此前|历史|already|previously)\s*$",
        re.I,
    )
    for clause in _sample_design_clauses(utterance):
        subjects = tuple(_SAMPLE_DESIGN_SUBJECT_RE.finditer(clause))
        actions = tuple(_SAMPLE_DESIGN_ACTION_RE.finditer(clause))
        for subject in subjects:
            subject_prefix = clause[max(0, subject.start() - 14) : subject.start()]
            subject_is_reference = reference_prefix.search(subject_prefix) is not None
            for action in actions:
                if action.start() < subject.end() and subject.start() < action.end():
                    continue
                token = action.group(0).lower()
                action_prefix = clause[max(0, action.start() - 12) : action.start()]
                if _SAMPLE_DESIGN_NEGATED_ACTION_RE.search(
                    clause[max(0, action.start() - 12) : action.end()]
                ):
                    continue
                if action.end() <= subject.start():
                    gap = clause[action.end() : subject.start()]
                    if (
                        token in before_subject_actions
                        and len(gap) <= 24
                        and "的" not in gap
                        and past_prefix.search(action_prefix) is None
                    ):
                        return True
                elif subject.end() <= action.start():
                    gap = clause[subject.end() : action.start()]
                    if (
                        not subject_is_reference
                        and token in after_subject_actions
                        and len(gap) <= 8
                    ):
                        return True
    return False


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


def _sample_design_build_intent_negated(utterance: str) -> bool:
    subject = r"(?:(?:策略)?样本(?:设计|边界|方案)|sample(?:\s|-|_)*design)"
    action = (
        r"(?:创建|生成|构建|设计|固化|冻结|物化|计算|分析|探索|"
        r"create|build|design|freeze|materialize|compute|analy[sz]e|explore)"
    )
    negation = (
        r"(?:不要|不用|无需|别|禁止|取消|暂不|先不|"
        r"do\s+not|don't|never|cancel)"
    )
    clauses = re.split(r"[；;。.!?？\n，,、/]+", utterance)
    return any(
        re.search(rf"{negation}\s*{action}.{{0,16}}{subject}", clause, re.I)
        or re.search(rf"{negation}\s*{subject}.{{0,12}}{action}", clause, re.I)
        for clause in clauses
    )


def _ground_strategy_sample_design_v2_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Ground every user-owned V2 control before a compatibility plan exists."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _sample_design_build_intent_negated(utterance):
        return _clarification(
            "原话否定或取消了 V2 样本设计固化，本轮不会创建计划。",
            code="strategy_sample_design_v2_intent_negated",
            fields=("build_intent",),
        )
    if (
        not utterance_targets_strategy_sample_design(utterance)
        or _SAMPLE_DESIGN_NONCOMMAND_RE.search(utterance)
    ):
        return _clarification(
            "请用一条当前、肯定式命令说明要固化 V2 策略样本设计；"
            "问句、假设、历史或未来描述不会被当成立即执行授权。",
            code="strategy_sample_design_v2_positive_command_required",
            fields=("build_intent",),
        )
    if _sample_design_v2_has_chained_operation(utterance):
        return _clarification(
            "本轮只能生成兼容锚点并固化 V2 双总体样本证据；建模、比较、报告、"
            "Strategy Pool、采纳和部署必须拆成后续请求。",
            code="strategy_sample_design_v2_single_step_required",
            fields=("next_action",),
        )
    if _SAMPLE_DESIGN_V2_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "legacy ref、scope、policy、数据身份、workspace 和所有"
            " artifact/id/hash 均由当前 task 绑定，不能由自然语言注入。",
            code="strategy_sample_design_v2_platform_binding_forbidden",
            fields=("platform_binding",),
        )

    missing: list[str] = []
    performance = inputs["performance_window"]
    if not _sample_design_performance_grounded(
        utterance,
        status=performance["status"],
        days=performance["days"],
    ):
        missing.append("performance_window")
    observation = inputs["observation_window"]
    if not _sample_design_observation_grounded(
        utterance,
        status=observation["status"],
        start=observation["start"],
        end=observation["end"],
    ):
        missing.append("observation_window")
    maturity = inputs["maturity"]
    maturity_status_grounded = (
        _sample_design_maturity_grounded(utterance, maturity["status"])
        if maturity["status"] != "unavailable"
        else re.search(
            r"(?:成熟度|maturity).{0,12}(?:暂时没有|暂无|未提供|不可用|unavailable)|"
            r"(?:暂时没有|暂无|未提供|不可用|unavailable).{0,12}(?:成熟度|maturity)",
            utterance,
            re.I,
        )
        is not None
    )
    if not maturity_status_grounded:
        missing.append("maturity.status")
    maturity_days = maturity["performance_window_days"]
    if maturity_days is not None and not _sample_v2_maturity_days_grounded(
        utterance,
        maturity_days,
    ):
        missing.append("maturity.performance_window_days")
    maturity_cutoff = maturity["cutoff_date"]
    if maturity_cutoff is not None and not _sample_v2_maturity_cutoff_grounded(
        utterance,
        maturity_cutoff,
    ):
        missing.append("maturity.cutoff_date")
    maturity_reason = maturity["reason"]
    if maturity_reason is not None and not _sample_v2_maturity_reason_grounded(
        utterance,
        maturity_reason,
    ):
        missing.append("maturity.reason")
    if not _sample_design_target_bad_value_grounded(
        utterance,
        inputs["target_bad_value"],
    ):
        missing.append("target_bad_value")
    if not _sample_v2_drop_policy_grounded(utterance, inputs["drop_nan_labels"]):
        missing.append("drop_nan_labels")
    if not _sample_v2_relationship_grounded(
        utterance,
        inputs["relationship"],
    ):
        missing.append("relationship")

    for role, labels in (
        ("approval_population", ("审批总体", "审批样本", "approval population")),
        ("risk_population", ("风险总体", "风险样本", "risk population")),
    ):
        population = inputs[role]
        if population["inclusion"] is None and population["exclusion"] is None:
            if not _sample_v2_no_population_filters_grounded(utterance, labels):
                missing.append(role)
        else:
            for field in ("inclusion", "exclusion"):
                predicate = population[field]
                if predicate is not None and not _sample_v2_predicate_grounded(
                    utterance,
                    predicate,
                    role_labels=labels,
                    direction=field,
                ):
                    missing.append(f"{role}.{field}")

    partitioning = inputs["partitioning"]
    partition_labels = (
        ("development", ("开发", "development")),
        ("validation", ("验证", "validation")),
        ("oot", ("OOT", "时间外")),
    )
    if partitioning["method"] == "time_ranges":
        if not _sample_design_column_role_grounded(
            utterance,
            column=partitioning["column"],
            labels=(
                "时间切分列",
                "时间拆分列",
                "time partition column",
            ),
        ):
            missing.append("partitioning.column")
        for partition, labels in partition_labels:
            bounds = partitioning["ranges"][partition]
            if not _sample_v2_partition_time_range_grounded(
                utterance,
                start=bounds["start"],
                end=bounds["end"],
                labels=labels,
            ):
                missing.append(f"partitioning.ranges.{partition}")
    else:
        selectors = partitioning["selectors"]
        simple: dict[str, tuple[str, object]] = {}
        for partition, _labels in partition_labels:
            try:
                simple[partition] = _simple_partition_equality(
                    selectors[partition],
                    name=f"partitioning.selectors.{partition}",
                )
            except _DraftValidationError:
                simple = {}
                break
        simple_columns = {column for column, _value in simple.values()}
        if len(simple) == 3 and len(simple_columns) == 1:
            split_column = next(iter(simple_columns))
            for partition, labels in partition_labels:
                _column, value = simple[partition]
                if not _sample_v2_partition_equality_grounded(
                    utterance,
                    values=[value],
                    labels=labels,
                ):
                    missing.append(f"partitioning.selectors.{partition}")
            if not _sample_design_column_role_grounded(
                utterance,
                column=split_column,
                labels=("切分列", "拆分列", "split column", "split_col"),
            ):
                missing.append("partitioning.column")
        else:
            for partition, labels in partition_labels:
                if not _sample_v2_partition_predicate_grounded(
                    utterance,
                    selectors[partition],
                    labels=labels,
                ):
                    missing.append(f"partitioning.selectors.{partition}")

    binding_labels = {
        "entity_field": ("实体字段", "客户字段", "entity field"),
        "time_field": ("时间字段", "日期字段", "time field"),
        "group_field": ("分组字段", "群组字段", "group field"),
        "month_field": ("月份字段", "月度字段", "month field"),
        "weight_field": ("权重字段", "weight field"),
        "loan_amount_field": ("放款金额字段", "贷款金额字段", "loan amount field"),
        "overdue_amount_field": ("逾期金额字段", "overdue amount field"),
    }
    for field, labels in binding_labels.items():
        value = inputs["field_bindings"][field]
        if value is None:
            if not _sample_v2_unavailable_role_grounded(utterance, labels):
                missing.append(f"field_bindings.{field}")
        elif not _sample_design_column_role_grounded(
            utterance,
            column=value,
            labels=labels,
        ):
            missing.append(f"field_bindings.{field}")
        for candidate in whitelist:
            if _sample_design_column_role_grounded(
                utterance,
                column=candidate,
                labels=labels,
            ) and value != candidate:
                missing.append(f"field_bindings.{field}={candidate}")

    historical = inputs["historical_score"]
    if not _sample_v2_historical_score_grounded(
        utterance,
        historical,
        whitelist=whitelist,
    ):
        missing.append("historical_score")
    if missing:
        fields = tuple(dict.fromkeys(missing))
        return _clarification(
            "V2 样本设计草案存在无法逐字与原话核对的控制项："
            + "、".join(fields)
            + "。平台不会猜测、补写或静默降级。",
            code="strategy_sample_design_v2_controls_not_grounded",
            fields=fields,
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


def _model_evidence_v2_has_positive_chain(utterance: str) -> bool:
    """Return true only for a positively requested downstream operation."""

    boundaries = "；;。.!?？\n，,、/"
    for match in _MODEL_EVIDENCE_CHAIN_RE.finditer(utterance):
        left = max(utterance.rfind(mark, 0, match.start()) for mark in boundaries) + 1
        prefix = utterance[left : match.start()]
        if re.search(
            r"(?:不需要|不用|暂不|先不|不要|无需|不再|不做|"
            r"别|禁止|不会|未|没有|并非|而非|不)"
            r"\s*(?:再|进行|做|生成|形成|输出)?\s*$|"
            r"(?:(?:do\s+not\s+need\s+to|don't\s+need\s+to|"
            r"do\s+not|don't|never|without|no)\s+)"
            r"(?:(?:further\s+)?(?:do|generate|create|run)\s+)?$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def _sample_design_v2_has_chained_operation(utterance: str) -> bool:
    for match in _SAMPLE_DESIGN_V2_CHAIN_RE.finditer(utterance):
        prefix = utterance[max(0, match.start() - 10) : match.start()]
        if re.search(r"(?:不|不要|无需|不再|别|禁止|not\s+|do\s+not\s+|don't\s+)$", prefix, re.I):
            continue
        return True
    return False


def _sample_v2_value_grounded(utterance: str, value: object) -> bool:
    if isinstance(value, str):
        return _utterance_contains_token(utterance, value) or value in utterance
    return re.search(rf"(?<![0-9.]){re.escape(str(value))}(?![0-9.])", utterance) is not None


def _sample_v2_drop_policy_grounded(utterance: str, expected: bool) -> bool:
    true_match = re.search(
        r"(?:丢弃|排除|剔除|删除).{0,12}(?:NaN|nan|空标签|缺失标签)|"
        r"(?:drop|exclude).{0,12}(?:nan|missing)\s+labels?",
        utterance,
        re.I,
    )
    false_match = re.search(
        r"(?:不丢弃|不排除|保留).{0,12}(?:NaN|nan|空标签|缺失标签)|"
        r"(?:do\s+not|don't)\s+(?:drop|exclude).{0,12}(?:nan|missing)\s+labels?",
        utterance,
        re.I,
    )
    return false_match is not None if expected is False else true_match is not None and false_match is None


def _sample_v2_relationship_grounded(
    utterance: str,
    relationship: object,
) -> bool:
    """Require the user to state the two-population relationship explicitly."""

    if relationship not in {"nested_same_cohort", "parallel_time_cohorts"}:
        return False
    observed: set[str] = set()
    for clause in _sample_design_clauses(utterance):
        has_both_roles = (
            re.search(r"(?:审批总体|审批样本|approval\s+population)", clause, re.I)
            is not None
            and re.search(r"(?:风险总体|风险样本|risk\s+population)", clause, re.I)
            is not None
        )
        if not has_both_roles:
            continue
        if re.search(
            r"(?:nested[_\s-]*same[_\s-]*cohort|"
            r"(?:同批|同一|相同).{0,8}(?:cohort|队列|样本).{0,16}"
            r"(?:嵌套|包含|子集)|"
            r"(?:嵌套|包含|子集).{0,16}(?:同批|同一|相同).{0,8}"
            r"(?:cohort|队列|样本))",
            clause,
            re.I,
        ):
            observed.add("nested_same_cohort")
        if re.search(
            r"(?:parallel[_\s-]*time[_\s-]*cohorts?|"
            r"(?:平行|并行|独立).{0,10}(?:时间|时点|月份).{0,8}"
            r"(?:cohort|队列|样本)|"
            r"(?:时间|时点|月份).{0,8}(?:cohort|队列|样本).{0,10}"
            r"(?:平行|并行|独立))",
            clause,
            re.I,
        ):
            observed.add("parallel_time_cohorts")
    return observed == {relationship}


def _sample_v2_no_population_filters_grounded(
    utterance: str,
    labels: Sequence[str],
) -> bool:
    no_filter = re.compile(
        r"(?:无|没有|不设|不设置|不使用|无需)\s*(?:任何)?\s*"
        r"(?:(?:纳排|纳入\s*(?:和|及|或|/)?\s*排除|筛选|过滤)(?:条件)?|"
        r"(?:额外|附加)?条件)|"
        r"(?:(?:no|without)\s+(?:population\s+)?filters?|"
        r"inclusion\s*(?:=|:)?\s*(?:none|null)\s*(?:and|,|，|、|/)\s*"
        r"exclusion\s*(?:=|:)?\s*(?:none|null))",
        re.I,
    )
    positive_filter = re.compile(
        r"(?:纳入|包含|保留|排除|剔除|不纳入|inclusion|include|"
        r"exclusion|exclude).{0,40}"
        r"(?:不等于|等于|不为|大于等于|小于等于|大于|小于|"
        r"!=|<>|>=|<=|(?<![<>!=])=(?!=)|\b(?:eq|ne|gt|gte|lt|lte)\b)",
        re.I,
    )
    return any(
        no_filter.search(segment) is not None
        and positive_filter.search(segment) is None
        for segment in _sample_v2_role_owned_segments(utterance, labels)
    )


def _sample_v2_predicate_grounded(
    utterance: str,
    predicate: object,
    *,
    role_labels: Sequence[str],
    direction: str,
) -> bool:
    if direction not in {"inclusion", "exclusion"}:
        return False
    for role_segment in _sample_v2_role_owned_segments(utterance, role_labels):
        markers = tuple(_SAMPLE_V2_POPULATION_DIRECTION_RE.finditer(role_segment))
        for index, marker in enumerate(markers):
            if marker.lastgroup != direction:
                continue
            end = (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(role_segment)
            )
            local = role_segment[marker.start() : end]
            if _sample_v2_predicate_semantics_grounded(local, predicate):
                return True
    return False


def _sample_v2_role_owned_segments(
    utterance: str,
    labels: Sequence[str],
) -> tuple[str, ...]:
    expected = re.compile(
        "|".join(
            re.escape(label)
            for label in sorted(labels, key=len, reverse=True)
        ),
        re.I,
    )
    segments: list[str] = []
    for clause in _sample_design_clauses(utterance):
        roles = tuple(_SAMPLE_V2_POPULATION_ROLE_RE.finditer(clause))
        for index, role in enumerate(roles):
            if expected.search(role.group(0)) is None:
                continue
            start = 0 if index == 0 else role.start()
            end = roles[index + 1].start() if index + 1 < len(roles) else len(clause)
            segments.append(clause[start:end].strip())
    return tuple(segments)


def _sample_v2_predicate_semantics_grounded(
    text: str,
    predicate: object,
) -> bool:
    if not _sample_v2_fresh_partition_selector_shape(predicate):
        return False
    assert isinstance(predicate, Mapping)
    op = predicate.get("op")
    if op not in {"and", "or"}:
        leaf_pattern = _sample_v2_predicate_leaf_grounding_pattern(predicate)
        return leaf_pattern is not None and re.search(
            leaf_pattern,
            text,
            re.I,
        ) is not None
    if op in {"and", "or"}:
        args = predicate.get("args")
        assert isinstance(args, Sequence)
        patterns = [
            _sample_v2_predicate_leaf_grounding_pattern(item)
            for item in args
        ]
        if any(pattern is None for pattern in patterns):
            return False
        connector = (
            r"(?:且|并且|同时|(?<![A-Za-z0-9_])and(?![A-Za-z0-9_]))"
            if op == "and"
            else r"(?:或|或者|(?<![A-Za-z0-9_])or(?![A-Za-z0-9_]))"
        )
        opposite = (
            r"(?:或|或者|(?<![A-Za-z0-9_])or(?![A-Za-z0-9_]))"
            if op == "and"
            else r"(?:且|并且|同时|(?<![A-Za-z0-9_])and(?![A-Za-z0-9_]))"
        )
        joined = patterns[0] + "".join(
            rf"\s*(?:，|,)?\s*(?:{connector})\s*{pattern}"
            for pattern in patterns[1:]
        )
        match = re.search(joined, text, re.I)
        return match is not None and re.search(
            opposite,
            match.group(0),
            re.I,
        ) is None
    return False


def _sample_v2_predicate_leaf_grounding_pattern(
    predicate: object,
) -> str | None:
    if not _sample_v2_fresh_partition_leaf_shape(predicate):
        return None
    assert isinstance(predicate, Mapping)
    op = predicate.get("op")
    if op in {"eq", "ne", "gt", "gte", "lt", "lte"}:
        left = _sample_v2_operand_pattern(predicate.get("left"))
        right = _sample_v2_operand_pattern(predicate.get("right"))
        operator = _sample_v2_operator_pattern(str(op))
        if left is None or right is None or operator is None:
            return None
        return rf"{left}.{{0,24}}?(?:{operator}).{{0,24}}?{right}"
    arg = _sample_v2_operand_pattern(predicate.get("arg"))
    if arg is None:
        return None
    null_operator = (
        r"(?:(?<!不)为空|(?<!不)是空值|is\s+null)"
        if op == "is_null"
        else r"(?:不为空|不是空值|非空|is\s+not\s+null)"
    )
    return (
        rf"(?:{arg}.{{0,16}}?(?:{null_operator})|"
        rf"(?:{null_operator}).{{0,16}}?{arg})"
    )


def _sample_v2_operand_pattern(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if set(value) == {"column"}:
        return _sample_v2_token_pattern(value["column"])
    if set(value) == {"literal"}:
        return _sample_v2_token_pattern(value["literal"])
    return None


def _sample_v2_token_pattern(value: object) -> str:
    text = _sample_design_scalar_text(value)
    escaped = re.escape(text)
    if re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        return rf"(?<![A-Za-z0-9_.-]){escaped}(?![A-Za-z0-9_.-])"
    return escaped


def _sample_v2_operator_pattern(operator: str) -> str | None:
    return {
        "eq": (
            r"(?:(?<!不)(?<!大于)(?<!小于)等于|(?<!不)为|(?<!不)是|"
            r"(?<![<>!=])=(?!=)|(?<![A-Za-z0-9_])"
            r"(?:eq|equals?|equal\s+to)(?![A-Za-z0-9_]))"
        ),
        "ne": (
            r"(?:不等于|不为|不是|!=|<>|(?<![A-Za-z0-9_])"
            r"(?:ne|not\s+equal(?:\s+to)?)(?![A-Za-z0-9_]))"
        ),
        "gt": (
            r"(?:大于(?!等于)|高于|>(?!=)|(?<![A-Za-z0-9_])"
            r"(?:gt|greater\s+than)(?![A-Za-z0-9_]))"
        ),
        "gte": (
            r"(?:大于等于|不小于|至少|>=|(?<![A-Za-z0-9_])"
            r"(?:gte|greater\s+than\s+or\s+equal(?:\s+to)?|at\s+least)"
            r"(?![A-Za-z0-9_]))"
        ),
        "lt": (
            r"(?:小于(?!等于)|低于|<(?!=)|(?<![A-Za-z0-9_])"
            r"(?:lt|less\s+than)(?![A-Za-z0-9_]))"
        ),
        "lte": (
            r"(?:小于等于|不大于|至多|<=|(?<![A-Za-z0-9_])"
            r"(?:lte|less\s+than\s+or\s+equal(?:\s+to)?|at\s+most)"
            r"(?![A-Za-z0-9_]))"
        ),
    }.get(operator)


def _sample_v2_maturity_days_grounded(
    utterance: str,
    days: object,
) -> bool:
    if isinstance(days, bool) or not isinstance(days, int):
        return False
    contexts = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if re.search(
            r"(?:成熟(?:度)?(?:表现)?窗|成熟表现期|"
            r"maturity.{0,12}performance\s+window)",
            clause,
            re.I,
        )
    )
    observed = {
        int(value)
        for clause in contexts
        for value in re.findall(r"(?<!\d)(\d{1,6})\s*(?:天|days?)(?!\w)", clause, re.I)
    }
    return observed == {days}


def _sample_v2_maturity_cutoff_grounded(
    utterance: str,
    cutoff: object,
) -> bool:
    if not isinstance(cutoff, str):
        return False
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    maturity_label = r"(?:成熟(?:度)?|maturity)"
    cutoff_label = r"(?:截止日|截止日期|cutoff(?:\s+date)?)"
    contexts = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if re.search(
            rf"{maturity_label}.{{0,16}}{cutoff_label}|"
            rf"{cutoff_label}.{{0,16}}{maturity_label}",
            clause,
            re.I,
        )
    )
    dates = {
        value
        for clause in contexts
        for value in re.findall(date_pattern, clause)
    }
    return dates == {cutoff}


def _sample_v2_maturity_reason_grounded(
    utterance: str,
    reason: object,
) -> bool:
    if not isinstance(reason, str) or not reason:
        return False
    return any(
        re.search(r"(?:成熟(?:度)?(?:原因|理由)|maturity\s+reason)", clause, re.I)
        is not None
        and _sample_v2_value_grounded(clause, reason)
        for clause in _sample_design_clauses(utterance)
    )


def _sample_v2_partition_equality_grounded(
    utterance: str,
    *,
    values: Sequence[object],
    labels: Sequence[str],
) -> bool:
    label_pattern = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    matches = [
        match
        for clause in _sample_design_clauses(utterance)
        if (match := re.search(rf"(?:{label_pattern})", clause, re.I))
    ]
    if len(matches) != 1:
        return False
    clause_match = matches[0]
    clause = clause_match.string
    remainder = clause[clause_match.end() :]
    if re.match(
        r"^\s*(?:样本)?\s*(?:取?值|values?)\s*"
        r"(?:不等于|不为|不是|!=|<>|大于|小于|高于|低于|>=|<=|>|<)",
        remainder,
        re.I,
    ):
        return False
    if re.match(
        r"^\s*(?:样本)?\s*(?:(?:取?值|values?)\s*(?:为|是|等于|=)?|"
        r"(?:为|是|等于|=))",
        remainder,
        re.I,
    ) is None:
        return False
    return _sample_design_split_values_grounded(
        utterance,
        values=values,
        labels=labels,
    )


def _sample_v2_partition_predicate_grounded(
    utterance: str,
    predicate: object,
    *,
    labels: Sequence[str],
) -> bool:
    role = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    clauses = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if re.search(
            rf"(?:{role}).{{0,10}}(?:条件|谓词|selector|predicate)",
            clause,
            re.I,
        )
    )
    return (
        len(clauses) == 1
        and _sample_v2_predicate_semantics_grounded(clauses[0], predicate)
    )


def _sample_v2_partition_time_range_grounded(
    utterance: str,
    *,
    start: object,
    end: object,
    labels: Sequence[str],
) -> bool:
    role = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    clauses = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if re.search(
            rf"(?:{role}).{{0,10}}(?:时间范围|日期范围|time\s+range)",
            clause,
            re.I,
        )
    )
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    expected = {
        value for value in (start, end) if isinstance(value, str)
    }
    observed = set(re.findall(date_pattern, clause))
    if expected != observed:
        return False
    if isinstance(start, str) and isinstance(end, str):
        return (
            re.search(
                rf"{re.escape(start)}\s*(?:至|到|~|—|–|to|through)\s*"
                rf"{re.escape(end)}",
                clause,
                re.I,
            )
            is not None
        )
    if start is None:
        return re.search(
            r"(?:起始|开始|start).{0,8}(?:无|暂无|开放|none|null)",
            clause,
            re.I,
        ) is not None
    return re.search(
        r"(?:结束|截止|end).{0,8}(?:无|暂无|开放|none|null)",
        clause,
        re.I,
    ) is not None


def _sample_v2_unavailable_role_grounded(
    utterance: str,
    labels: Sequence[str],
) -> bool:
    role = "(?:" + "|".join(re.escape(label) for label in labels) + ")"
    unavailable = r"(?:暂时没有|暂无|没有|未提供|不可用|不适用|unavailable|not\s+available|none|null)"
    return (
        re.search(
            rf"{role}.{{0,12}}(?:{unavailable})|(?:{unavailable}).{{0,12}}{role}",
            utterance,
            re.I,
        )
        is not None
    )


def _sample_v2_historical_score_grounded(
    utterance: str,
    historical: Mapping[str, Any],
    *,
    whitelist: Sequence[str],
) -> bool:
    subject = r"(?:历史分|历史评分|历史模型分|historical\s+score)"
    status = historical["status"]
    if status == "available":
        column = historical["column"]
        direction = historical["direction"]
        owned_clauses = tuple(
            clause
            for clause in _sample_design_clauses(utterance)
            if re.search(subject, clause, re.I)
            and _sample_v2_value_grounded(clause, column)
        )
        if len(owned_clauses) != 1:
            return False
        clause = owned_clauses[0]
        if any(
            candidate != column
            and _sample_v2_value_grounded(clause, candidate)
            for candidate in whitelist
        ):
            return False
        direction_patterns = {
            "higher_is_riskier": (
                r"(?:越高越风险|高分高风险|higher[_\s-]*is[_\s-]*riskier)"
            ),
            "lower_is_riskier": (
                r"(?:越低越风险|低分高风险|lower[_\s-]*is[_\s-]*riskier)"
            ),
        }
        observed = {
            candidate
            for candidate, pattern in direction_patterns.items()
            if re.search(pattern, clause, re.I)
        }
        return observed == {direction}
    status_pattern = (
        r"(?:暂时没有|暂无|没有|未提供|不可用|unavailable)"
        if status == "unavailable"
        else r"(?:不适用|not[_\s-]*applicable)"
    )
    return any(
        re.search(
            rf"{subject}.{{0,16}}(?:{status_pattern})|"
            rf"(?:{status_pattern}).{{0,16}}{subject}",
            clause,
            re.I,
        )
        and _sample_v2_value_grounded(clause, historical["reason"])
        for clause in _sample_design_clauses(utterance)
    )


def _ground_strategy_sample_design_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Ground the small, user-owned sample-boundary contract in one command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _sample_design_build_intent_negated(utterance):
        return _clarification(
            "原话否定或取消了样本设计固化，本轮不会创建计划。",
            code="strategy_sample_design_intent_negated",
            fields=("build_intent",),
        )
    if (
        not utterance_targets_strategy_sample_design(utterance)
        or _SAMPLE_DESIGN_NONCOMMAND_RE.search(utterance)
    ):
        return _clarification(
            "请用一条当前、肯定式命令说明要固化策略样本设计；问句、假设、历史或"
            "未来描述不会被当成立即执行授权。",
            code="strategy_sample_design_positive_command_required",
            fields=("build_intent",),
        )
    if _sample_design_has_forbidden_operation(utterance):
        return _clarification(
            "本轮只能冻结当前活动样本边界并计算样本证据；建模、建树、入池、"
            "采纳、部署和报告必须拆成后续请求。",
            code="strategy_sample_design_single_step_required",
            fields=("next_action",),
        )
    if _SAMPLE_DESIGN_PLATFORM_CONTROL_RE.search(utterance):
        return _clarification(
            "数据集/hash、workspace、语义映射和目标列都由平台从当前活动 "
            "DataWorkspace 绑定，不能由自然语言注入。",
            code="strategy_sample_design_platform_binding_forbidden",
            fields=("dataset_binding",),
        )

    missing: list[str] = []
    performance_status = inputs["performance_window_status"]
    if not _sample_design_performance_grounded(
        utterance,
        status=performance_status,
        days=inputs.get("performance_window_days"),
    ):
        missing.append(
            f"performance_window_days={inputs['performance_window_days']}"
            if performance_status == "provided"
            else "performance_window_status=unavailable"
        )

    observation_status = inputs["observation_window_status"]
    if not _sample_design_observation_grounded(
        utterance,
        status=observation_status,
        start=inputs.get("observation_start"),
        end=inputs.get("observation_end"),
    ):
        if observation_status == "provided":
            missing.extend(
                (
                    f"observation_start={inputs['observation_start']}",
                    f"observation_end={inputs['observation_end']}",
                )
            )
        else:
            missing.append("observation_window_status=unavailable")

    maturity_status = inputs["maturity_status"]
    if not _sample_design_maturity_grounded(utterance, maturity_status):
        missing.append(f"maturity_status={maturity_status}")
    if (
        performance_status == "unavailable"
        or maturity_status == "unknown"
    ) and _SAMPLE_DESIGN_EXPLORATION_RE.search(utterance) is None:
        missing.append("exploration_only")

    target_bad_value = inputs["target_bad_value"]
    if not _sample_design_target_bad_value_grounded(
        utterance,
        target_bad_value,
    ):
        missing.append(f"target_bad_value={target_bad_value}")

    column_roles = {
        "split_col": ("切分列", "拆分列", "split_col", "split column"),
        "month_col": ("月份列", "月度列", "month_col", "month column"),
        "weight_col": ("权重列", "weight_col", "weight column"),
        "loan_amount_col": (
            "放款金额列",
            "贷款金额列",
            "loan_amount_col",
            "loan amount column",
        ),
        "overdue_amount_col": (
            "逾期金额列",
            "overdue_amount_col",
            "overdue amount column",
        ),
    }
    for field, labels in column_roles.items():
        supplied = inputs.get(field)
        if isinstance(supplied, str) and not _sample_design_column_role_grounded(
            utterance,
            column=supplied,
            labels=labels,
        ):
            missing.append(f"{field}={supplied}")
        for candidate in whitelist:
            if _sample_design_column_role_grounded(
                utterance,
                column=candidate,
                labels=labels,
            ) and supplied != candidate:
                missing.append(f"{field}={candidate}")

    if "split_col" in inputs:
        split_labels = {
            "development_values": ("开发", "development", "dev"),
            "validation_values": ("验证", "validation", "valid"),
            "oot_values": ("oot", "时间外"),
        }
        for field, labels in split_labels.items():
            if not _sample_design_split_values_grounded(
                utterance,
                values=inputs[field],
                labels=labels,
            ):
                missing.append(field)

    drop_true = re.search(
        r"(?:丢弃|排除|剔除|删除).{0,12}(?:NaN|nan|空标签|缺失标签)|"
        r"(?:drop|exclude).{0,12}(?:nan|missing)\s+labels?",
        utterance,
        re.I,
    )
    drop_false = re.search(
        r"(?:不丢弃|不排除|保留).{0,12}(?:NaN|nan|空标签|缺失标签)|"
        r"(?:do\s+not|don't)\s+(?:drop|exclude).{0,12}"
        r"(?:nan|missing)\s+labels?",
        utterance,
        re.I,
    )
    if drop_false is not None:
        drop_true = None
    if drop_true is not None and inputs.get("drop_nan_labels") is not True:
        missing.append("drop_nan_labels=true")
    if drop_false is not None and inputs.get("drop_nan_labels") is not False:
        missing.append("drop_nan_labels=false")
    if "drop_nan_labels" in inputs and drop_true is None and drop_false is None:
        missing.append(f"drop_nan_labels={str(inputs['drop_nan_labels']).lower()}")

    if missing:
        unique = tuple(dict.fromkeys(missing))
        return _clarification(
            "样本设计草案中的表现窗、观察窗、成熟度、标签语义、切分或可选列无法逐字与"
            "原话核对：" + "、".join(unique) + "。平台不会猜测或补写口径。",
            code="strategy_sample_design_controls_not_grounded",
            fields=unique,
        )
    return result


def _sample_design_clauses(utterance: str) -> tuple[str, ...]:
    return tuple(
        clause.strip()
        for clause in re.split(r"[；;。.!?？\n]+", utterance)
        if clause.strip()
    )


def _sample_design_has_forbidden_operation(utterance: str) -> bool:
    """Reject every second operation unless that operation is explicitly negated."""

    boundaries = "；;。.!?？\n，,、/"
    for match in _SAMPLE_DESIGN_OTHER_OPERATION_RE.finditer(utterance):
        left = max(utterance.rfind(mark, 0, match.start()) for mark in boundaries) + 1
        right_candidates = [
            position
            for mark in boundaries
            if (position := utterance.find(mark, match.end())) >= 0
        ]
        right = min(right_candidates) if right_candidates else len(utterance)
        prefix = utterance[left : match.start()]
        suffix = utterance[match.end() : right]
        # Explicit target-null handling is a supported sample-design control,
        # not a user-authored filter or row-mutation operation.
        if match.group(0) == "排除" and re.search(
            r"(?:NaN|nan|空标签|缺失标签)", suffix, re.I
        ):
            continue
        if re.search(
            r"(?:不|不要|不再|不做|无需|不需要|别|禁止|不会|未|没有)\s*$|"
            r"(?:\bdo\s+not|\bdon't|\bwithout|\bno)\s*$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def _sample_design_performance_grounded(
    utterance: str,
    *,
    status: str,
    days: object,
) -> bool:
    labels = re.compile(r"表现(?:窗|期)|performance\s+window", re.I)
    maturity_labels = re.compile(
        r"成熟(?:度)?(?:表现)?(?:窗|期)|"
        r"maturity.{0,12}performance\s+window",
        re.I,
    )
    clauses = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if labels.search(clause) and maturity_labels.search(clause) is None
    )
    if not clauses:
        return False
    unavailable = any(
        re.search(
            r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|"
            r"unavailable|not\s+available|unknown)",
            clause,
            re.I,
        )
        for clause in clauses
    )
    explicit_days = {
        int(match.group(1))
        for clause in clauses
        for match in re.finditer(r"(?<!\d)(\d{1,6})\s*(?:天|days?)(?!\w)", clause, re.I)
    }
    negated_days = any(
        re.search(
            r"(?:不是|并非|不为|not)\s*\d{1,6}\s*(?:天|days?)",
            clause,
            re.I,
        )
        for clause in clauses
    )
    if status == "unavailable":
        return unavailable and not explicit_days and not negated_days
    return (
        isinstance(days, int)
        and not isinstance(days, bool)
        and not unavailable
        and not negated_days
        and explicit_days == {days}
    )


def _sample_design_observation_grounded(
    utterance: str,
    *,
    status: str,
    start: object,
    end: object,
) -> bool:
    label = r"(?:观察(?:窗|期)|observation\s+window)"
    clauses = tuple(
        clause
        for clause in _sample_design_clauses(utterance)
        if re.search(label, clause, re.I)
    )
    if not clauses:
        return False
    unavailable = any(
        re.search(
            r"(?:暂时没有|暂无|没有|未提供|不可用|不知道|未知|"
            r"unavailable|not\s+available|unknown)",
            clause,
            re.I,
        )
        for clause in clauses
    )
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    dates = {
        token
        for clause in clauses
        for token in re.findall(date_pattern, clause)
    }
    if status == "unavailable":
        return unavailable and not dates
    if not isinstance(start, str) or not isinstance(end, str) or unavailable:
        return False

    pairs: set[tuple[str, str]] = set()
    for clause in clauses:
        range_patterns = (
            rf"{label}.{{0,32}}?({date_pattern})\s*(?:至|到|~|—|–|to|through)\s*({date_pattern})",
            rf"({date_pattern})\s*(?:至|到|~|—|–|to|through)\s*({date_pattern}).{{0,20}}?{label}",
        )
        for pattern in range_patterns:
            pairs.update(re.findall(pattern, clause, re.I))
        starts = re.findall(
            rf"(?:开始|起始|start)\s*(?:为|是|=|:|：)?\s*({date_pattern})",
            clause,
            re.I,
        )
        ends = re.findall(
            rf"(?:结束|截止|end)\s*(?:为|是|=|:|：)?\s*({date_pattern})",
            clause,
            re.I,
        )
        if len(starts) == 1 and len(ends) == 1:
            pairs.add((starts[0], ends[0]))
    return pairs == {(start, end)} and dates == {start, end}


def _sample_design_maturity_grounded(utterance: str, status: str) -> bool:
    if re.search(
        r"(?:成熟度|maturity).{0,16}(?:未|不)(?:能)?确认(?:已经|已)?成熟|"
        r"(?:未|不)(?:能)?确认(?:已经|已)?成熟|"
        r"(?:不是|并非)(?:已经|已)?成熟|not\s+confirmed\s+matured",
        utterance,
        re.I,
    ):
        return False
    observed = {
        candidate
        for candidate, pattern in _SAMPLE_DESIGN_MATURITY_PATTERNS.items()
        if pattern.search(utterance) is not None
    }
    return observed == {status}


def _sample_design_target_bad_value_grounded(
    utterance: str,
    target_bad_value: int,
) -> bool:
    bad_values = _sample_design_binary_role_values(
        utterance,
        label=r"(?:坏(?:样本|客户|标签|类)?|坏账|bad(?:\s+(?:sample|label|class))?)",
    )
    bad_values.update(
        int(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9_])target_bad_value\s*(?:=|:|：)\s*([01])(?!\d)",
            utterance,
            re.I,
        )
    )
    good_values = _sample_design_binary_role_values(
        utterance,
        label=r"(?:好(?:样本|客户|标签|类)?|正常样本|good(?:\s+(?:sample|label|class))?)",
    )
    return bad_values == {target_bad_value} and (
        not good_values or good_values == {1 - target_bad_value}
    )


def _sample_design_binary_role_values(
    utterance: str,
    *,
    label: str,
) -> set[int]:
    values: set[int] = set()
    patterns = (
        rf"(?<!\d)([01])(?!\d)\s*(?:也\s*)?(?:是|为|代表|表示|标记为|编码为)\s*{label}",
        rf"{label}\s*(?:值|标签值|编码)?\s*(?:是|为|=|:|：)\s*([01])(?!\d)",
    )
    for pattern in patterns:
        values.update(int(value) for value in re.findall(pattern, utterance, re.I))
    return values


def _sample_design_column_role_grounded(
    utterance: str,
    *,
    column: str,
    labels: Sequence[str],
) -> bool:
    column_pattern = re.escape(column)
    label_pattern = "|".join(re.escape(label) for label in labels)
    for match in re.finditer(
        rf"(?<![A-Za-z0-9_]){column_pattern}(?![A-Za-z0-9_])",
        utterance,
        re.I,
    ):
        left = max(
            utterance.rfind(separator, 0, match.start())
            for separator in ("；", ";", "。", "\n")
        ) + 1
        right_candidates = [
            position
            for separator in ("；", ";", "。", "\n")
            if (position := utterance.find(separator, match.end())) >= 0
        ]
        right = min(right_candidates) if right_candidates else len(utterance)
        if re.search(label_pattern, utterance[left:right], re.I):
            return True
    return False


def _sample_design_split_values_grounded(
    utterance: str,
    *,
    values: Sequence[object],
    labels: Sequence[str],
) -> bool:
    label_pattern = "|".join(
        re.escape(label) for label in sorted(labels, key=len, reverse=True)
    )
    matches: list[tuple[str, re.Match[str]]] = []
    for clause in _sample_design_clauses(utterance):
        match = re.search(rf"(?:{label_pattern})", clause, re.I)
        if match is not None:
            matches.append((clause, match))
    if len(matches) != 1:
        return False
    clause, role_match = matches[0]
    remainder = clause[role_match.end() :]
    remainder = re.sub(
        r"^\s*(?:样本)?\s*(?:取?值|values?)?\s*(?:为|是|=|:|：)?\s*",
        "",
        remainder,
        flags=re.I,
    ).strip()
    if not values:
        return bool(
            re.search(
                r"(?:暂无|没有|未提供|不可用|为空|空数组|unavailable|none|empty)",
                remainder,
                re.I,
            )
        )
    if re.search(
        r"(?:暂无|没有|未提供|不可用|为空|空数组|unavailable|none|empty)",
        remainder,
        re.I,
    ):
        return False
    raw_tokens = re.split(r"\s*(?:、|，|,|和|及|与)\s*", remainder)
    tokens = {
        token.strip().strip("[]()（）{}\"'` ").casefold()
        for token in raw_tokens
        if token.strip().strip("[]()（）{}\"'` ")
    }
    expected = {
        _sample_design_scalar_text(value).casefold()
        for value in values
    }
    return tokens == expected


def _sample_design_scalar_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ground_cross_rule_search(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Ground every bounded rule-search control in the current command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_cross_rule_search(utterance):
        return _clarification(
            "请明确要求搜索 2D/3D Cross 阈值规则，并在当前请求中提供 "
            "features、dimension、四项 constraints 与 max_trials。",
            code="cross_rule_search_intent_required",
            fields=("search_intent",),
        )
    if (
        re.search(
            r"(?:不要|不用|无需|先不|暂不|取消|停止)"
            r"[^，,；;。\n]{0,32}(?:搜索|查找|挖掘|枚举|筛选)|"
            r"(?<![A-Za-z0-9_])(?:do\s+not|don't|cancel|stop)"
            r"[^,;.!?\n]{0,32}(?:search|find|mine|enumerate|screen)",
            utterance,
            re.IGNORECASE,
        )
        is not None
        or _CROSS_MATRIX_NONCOMMAND_RE.search(utterance) is not None
        or _CROSS_MATRIX_POSTPONED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Cross 阈值规则搜索必须是当前轮立即执行的肯定式命令。",
            code="cross_rule_search_positive_command_required",
            fields=("search_intent",),
        )
    if _cross_search_pattern_has_positive(
        utterance,
        _CROSS_SEARCH_FOLLOW_UP_RE,
    ):
        return _clarification(
            "本轮只搜索 Cross 阈值规则；构建候选、入池、应用、采纳和"
            "部署必须另发请求。",
            code="cross_rule_search_single_step_required",
            fields=("next_action",),
        )
    if _CROSS_RULE_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Cross 阈值规则搜索只接受字段、维度、四项业务约束和试验预算；"
            "阈值、方向、artifact/hash、rule/rank/winner 均由平台恢复或计算。",
            code="cross_rule_search_platform_binding_forbidden",
            fields=("platform_binding",),
        )

    bindings = tuple(_CROSS_SEARCH_FEATURES_RE.finditer(utterance))
    if len(bindings) != 1:
        return _clarification(
            "请且只请用 features=[字段1, 字段2, ...] 给出 2 到 12 个"
            "候选字段。",
            code="cross_rule_search_controls_not_grounded",
            fields=("features",),
        )
    observed_features = [
        token.strip().strip("'\"`")
        for token in re.split(r"[,，]", bindings[0].group("value"))
        if token.strip()
    ]
    if (
        observed_features != list(inputs["features"])
        or len(set(observed_features)) != len(observed_features)
        or any(feature not in whitelist for feature in observed_features)
    ):
        return _clarification(
            "features 必须逐字等于当前命令中的唯一白名单字段列表；"
            "模型不得补写、删减或改序。",
            code="cross_rule_search_controls_not_grounded",
            fields=("features",),
        )

    dimensions = {
        int(match.group("value") or match.group("zh_value"))
        for match in _CROSS_RULE_DIMENSION_RE.finditer(utterance)
    }
    constraints = inputs["constraints"]
    min_lifts = {
        float(match.group("value"))
        for match in _CROSS_RULE_MIN_LIFT_RE.finditer(utterance)
    }
    min_bad_counts = {
        int(match.group("value"))
        for match in _CROSS_RULE_MIN_BAD_COUNT_RE.finditer(utterance)
    }
    max_hit_shares = {
        float(match.group("value"))
        for match in _CROSS_RULE_MAX_HIT_SHARE_RE.finditer(utterance)
    }
    raw_amount_lifts = {
        match.group("value").casefold()
        for match in _CROSS_RULE_MIN_AMOUNT_LIFT_RE.finditer(utterance)
    }
    amount_lifts = {
        None if value in {"null", "none"} else float(value)
        for value in raw_amount_lifts
    }
    max_trials = {
        int(match.group("value"))
        for match in _CROSS_RULE_MAX_TRIALS_RE.finditer(utterance)
    }
    if (
        dimensions != {inputs["dimension"]}
        or min_lifts != {float(constraints["min_lift"])}
        or min_bad_counts != {constraints["min_bad_count"]}
        or max_hit_shares != {float(constraints["max_hit_share"])}
        or amount_lifts != {constraints["min_amount_lift"]}
        or max_trials != {inputs["max_trials"]}
    ):
        return _clarification(
            "dimension、min_lift、min_bad_count、max_hit_share、"
            "min_amount_lift 与 max_trials 必须在当前命令中逐项明确且唯一；"
            "模型不得补默认值或改写约束。",
            code="cross_rule_search_controls_not_grounded",
            fields=(
                "dimension",
                "constraints",
                "max_trials",
            ),
        )
    return result


def _ground_cross_rule_candidate_build(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one exact search/rule pointer without heuristic selection."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_cross_rule_selection(utterance):
        return _clarification(
            "请在独立请求中提供一个完整 cross-rule-search ID、一个完整"
            " cross-rule ID，并明确要求构建候选。",
            code="cross_rule_selection_intent_required",
            fields=("build_intent", "search_id", "rule_id"),
        )
    if _CROSS_RULE_SELECTION_HEURISTIC_RE.search(utterance) is not None:
        return _clarification(
            "请逐字点名完整 search_id 与 rule_id；平台不会消费第一名、"
            "最好、冠军、Top N、排名或‘刚才那个’。",
            code="cross_rule_selection_explicit_ids_required",
            fields=("search_id", "rule_id"),
        )
    if (
        _CROSS_MATRIX_NEGATED_BUILD_RE.search(utterance) is not None
        or _CROSS_MATRIX_NONCOMMAND_RE.search(utterance) is not None
        or _CROSS_MATRIX_POSTPONED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Cross 规则候选构建必须是当前轮立即执行的肯定式单步命令。",
            code="cross_rule_selection_positive_command_required",
            fields=("build_intent",),
        )
    if _cross_search_pattern_has_positive(
        utterance,
        _CROSS_SEARCH_SELECTION_FOLLOW_UP_RE,
    ):
        return _clarification(
            "本轮只能构建一个精确 Cross 规则候选；入池、设置动作、应用、"
            "采纳和部署必须另发请求。",
            code="cross_rule_selection_single_step_required",
            fields=("next_action",),
        )
    search_ids = tuple(
        match.group(0)
        for match in _CROSS_RULE_SEARCH_ID_TOKEN_RE.finditer(utterance)
    )
    rule_ids = tuple(
        match.group(0)
        for match in _CROSS_RULE_ID_TOKEN_RE.finditer(utterance)
    )
    if (
        search_ids != (inputs["search_id"],)
        or rule_ids != (inputs["rule_id"],)
    ):
        return _clarification(
            "Cross 规则候选构建必须逐字提供且只提供一个完整 search_id "
            "与一个完整 rule_id。",
            code="cross_rule_selection_ids_not_grounded",
            fields=("search_id", "rule_id"),
        )
    reason = inputs.get("selection_reason")
    if reason is not None:
        labeled = re.search(
            r"(?:选择理由|理由|原因|说明|selection[_\s-]*reason)"
            r"\s*(?:=|:|：|为)\s*(?P<reason>[^；;。\n]{1,500})",
            utterance,
            re.IGNORECASE,
        )
        if labeled is None or " ".join(
            unicodedata.normalize("NFC", labeled.group("reason")).split()
        ) != reason:
            return _clarification(
                "selection_reason 仅在当前命令显式标注时逐字抄录。",
                code="cross_rule_selection_reason_not_grounded",
                fields=("selection_reason",),
            )
    return result


def _ground_cross_matrix_candidate_search(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Prove the bounded feature universe and pair budget came from this turn."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_cross_candidate_search(utterance):
        return _clarification(
            "请明确要求搜索 Cross Matrix 特征组合，并在当前请求中提供 "
            "features=[...] 与 max_pairs。",
            code="cross_search_intent_required",
            fields=("search_intent",),
        )
    if (
        re.search(
            r"(?:不要|不用|无需|先不|暂不|取消|停止)"
            r"[^，,；;。\n]{0,32}(?:搜索|查找|检索|枚举|筛选|比较)|"
            r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|cancel|stop)"
            r"[^,;.!?\n]{0,32}(?:search|find|enumerate|screen|compare)",
            utterance,
            re.IGNORECASE,
        )
        is not None
        or _CROSS_MATRIX_NONCOMMAND_RE.search(utterance) is not None
        or _CROSS_MATRIX_POSTPONED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Cross Matrix 自动组合搜索必须是当前轮立即执行的肯定式命令；"
            "问句、否定、假设、历史/未来描述或句尾撤销不会启动搜索。",
            code="cross_search_positive_command_required",
            fields=("search_intent",),
        )
    if _cross_search_has_positive_follow_up(utterance):
        return _clarification(
            "本轮只搜索 Cross Matrix 特征组合；构建或选择候选、入池、"
            "应用、采纳和部署必须另发请求。",
            code="cross_search_single_step_required",
            fields=("next_action",),
        )
    if _CROSS_SEARCH_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Cross 自动搜索只接受 features 与 max_pairs；轴方法、候选资产、"
            "artifact/hash、pair/rank/winner 均由平台恢复或计算，不能注入。",
            code="cross_search_platform_binding_forbidden",
            fields=("platform_binding",),
        )

    bindings = tuple(_CROSS_SEARCH_FEATURES_RE.finditer(utterance))
    if len(bindings) != 1:
        return _clarification(
            "请且只请用 features=[字段1, 字段2, ...] 明确给出 2 到 20 个"
            "候选字段；平台不会从上下文补全或替你选字段。",
            code="cross_search_controls_not_grounded",
            fields=("features",),
        )
    raw_tokens = re.split(r"[,，]", bindings[0].group("value"))
    observed_features = [
        token.strip().strip("'\"`")
        for token in raw_tokens
        if token.strip()
    ]
    expected_features = list(inputs["features"])
    if (
        len(observed_features) != len(expected_features)
        or observed_features != expected_features
        or len(set(observed_features)) != len(observed_features)
        or any(feature not in whitelist for feature in observed_features)
    ):
        return _clarification(
            "features 必须逐字等于当前命令中唯一列表里的 2 到 20 个互不重复"
            "白名单字段；模型不得补写、删减、改序或使用目标列。",
            code="cross_search_controls_not_grounded",
            fields=("features",),
        )

    observed_max = {
        int(match.group("value") or match.group("zh_value"))
        for match in _CROSS_SEARCH_MAX_PAIRS_RE.finditer(utterance)
    }
    if observed_max != {int(inputs["max_pairs"])}:
        return _clarification(
            "max_pairs 必须在当前命令中明确且唯一写为 1 到 190 的整数；"
            "平台不会让模型补默认预算。",
            code="cross_search_controls_not_grounded",
            fields=("max_pairs",),
        )
    return result


def _ground_cross_matrix_candidate_build_from_search(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one exact search/pair pointer pair in an independent turn."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_cross_search_selection(utterance):
        return _clarification(
            "请在后续独立请求中提供一个完整 Cross search_id 和一个完整 "
            "pair_id，并明确要求构建候选。",
            code="cross_search_selection_intent_required",
            fields=("build_intent", "search_id", "pair_id"),
        )
    if (
        _CROSS_MATRIX_NEGATED_BUILD_RE.search(utterance) is not None
        or _CROSS_MATRIX_NONCOMMAND_RE.search(utterance) is not None
        or _CROSS_MATRIX_POSTPONED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Cross 搜索结果构建必须是当前轮立即执行的肯定式单步命令；"
            "问句、否定、假设、历史/未来描述或句尾撤销不会构建候选。",
            code="cross_search_selection_positive_command_required",
            fields=("build_intent",),
        )
    if (
        _cross_search_selection_has_positive_research(utterance)
        or _cross_search_pattern_has_positive(
            utterance,
            _CROSS_SEARCH_SELECTION_FOLLOW_UP_RE,
        )
    ):
        return _clarification(
            "本轮只能从精确 search_id/pair_id 构建一个 Cross 候选；"
            "重新搜索、入池、设置动作、应用、采纳、部署或写回必须另发请求。",
            code="cross_search_selection_single_step_required",
            fields=("next_action",),
        )
    if _CROSS_SEARCH_SELECTION_HEURISTIC_RE.search(utterance) is not None:
        return _clarification(
            "请逐字点名完整 search_id 与 pair_id；即使同时提供 pointer，"
            "平台也不会消费第一名、最好、冠军、Top N、排名或‘刚才那个’。",
            code="cross_search_selection_explicit_ids_required",
            fields=("search_id", "pair_id"),
        )
    if _CROSS_SEARCH_SELECTION_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Cross 搜索结果构建只接受 search_id 与 pair_id；artifact/hash、"
            "轴字段和方法、asset、rank/winner 均由平台重新认证和恢复。",
            code="cross_search_selection_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    search_ids = tuple(
        match.group(0)
        for match in _CROSS_SEARCH_ID_TOKEN_RE.finditer(utterance)
    )
    pair_ids = tuple(
        match.group(0)
        for match in _CROSS_PAIR_ID_TOKEN_RE.finditer(utterance)
    )
    if search_ids != (inputs["search_id"],) or pair_ids != (inputs["pair_id"],):
        return _clarification(
            "Cross 搜索结果构建必须逐字提供且只提供一个完整 search_id 与"
            "一个完整 pair_id；平台不会补全、替换、按排名选择或消费代词。",
            code="cross_search_selection_controls_not_grounded",
            fields=("search_id", "pair_id"),
        )
    return result


def _ground_voting_candidate_search(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove every search control came from this immediate user request."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_voting_candidate_search(utterance):
        return _clarification(
            "请明确要求搜索、查找或优化当前 Strategy Pool 的 Voting / n-of-k 组合。",
            code="voting_search_intent_required",
            fields=("search_intent",),
        )
    if (
        _VOTING_SEARCH_NEGATED_RE.search(utterance) is not None
        or _VOTING_NONCOMMAND_RE.search(utterance) is not None
        or _VOTING_POSTPOSED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Voting 组合搜索必须是当前轮立即执行的肯定式命令；问句、"
            "否定、假设、历史/未来描述或句尾撤销不会启动搜索。",
            code="voting_search_positive_command_required",
            fields=("search_intent",),
        )
    if _utterance_chains_voting_search_operation(utterance):
        return _clarification(
            "本轮只搜索 Voting 组合；构建候选、选择组合、修改或加入 "
            "Strategy Pool、应用、采纳和部署必须另发请求。",
            code="voting_search_single_step_required",
            fields=("next_action",),
        )
    if _VOTING_SEARCH_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Voting 搜索的 Pool ref、dataset/target、逐行命中矩阵、权重、"
            "金额向量和 artifact 身份只能由平台绑定，请删除这些控制。",
            code="voting_search_platform_binding_forbidden",
            fields=("platform_binding",),
        )

    expected_type = str(inputs["strategy_type"])
    observed_types = {
        value for value, _start, _end in _voting_strategy_type_mentions(utterance)
    }
    if observed_types != {expected_type}:
        return _clarification(
            "请在当前搜索命令中明确且唯一标注 Strategy Pool 类型；"
            "平台不会替用户选择策略类型。",
            code="voting_search_strategy_type_not_grounded",
            fields=("strategy_type",),
        )

    expected_member_count = int(inputs["member_count"])
    observed_member_counts = _voting_search_member_counts(utterance)
    if observed_member_counts != {expected_member_count}:
        return _clarification(
            "请明确且唯一给出每个 Voting 组合的 K/member_count（2 到 50）。",
            code="voting_search_member_count_not_grounded",
            fields=("member_count",),
        )

    expected_n = int(inputs["n"])
    n_bindings = _voting_n_bindings(utterance)
    observed_ns = {value for value, _k in n_bindings}
    explicit_ks = {value for _n, value in n_bindings if value is not None}
    if observed_ns != {expected_n} or (
        explicit_ks and explicit_ks != {expected_member_count}
    ):
        return _clarification(
            "请明确且唯一给出 n，并保证显式 n-of-k/“K 选 n”中的 K "
            "与 member_count 一致。",
            code="voting_search_n_not_grounded",
            fields=("n",),
        )

    observed_objectives = _voting_search_objectives(utterance)
    expected_objective = (
        str(inputs["objective"]["metric"]),
        str(inputs["objective"]["direction"]),
    )
    if observed_objectives != {expected_objective}:
        return _clarification(
            "请在“目标”标签后明确且唯一写出 objective metric 与 "
            "maximize/minimize 方向。",
            code="voting_search_objective_not_grounded",
            fields=("objective",),
        )

    observed_constraints = _voting_search_constraints(utterance)
    expected_constraints = {
        (
            str(item["metric"]),
            str(item["operator"]),
            float(item["value"]),
        )
        for item in inputs["constraints"]
    }
    if observed_constraints != expected_constraints:
        return _clarification(
            "Voting 搜索 constraints 只能逐项采用当前原话明确的 "
            "metric、gte/lte 与数值；未提供时固定为空。",
            code="voting_search_constraints_not_grounded",
            fields=("constraints",),
        )

    include_ids, exclude_ids, all_labeled_ids = _voting_search_rule_controls(utterance)
    all_rule_ids = {
        match.group(0) for match in _VOTING_RULE_ID_TOKEN_RE.finditer(utterance)
    }
    expected_include = set(inputs["include_rule_ids"])
    expected_exclude = set(inputs["exclude_rule_ids"])
    if (
        include_ids != expected_include
        or exclude_ids != expected_exclude
        or all_rule_ids != all_labeled_ids
    ):
        return _clarification(
            "include/exclude 只接受当前请求在对应标签后逐字给出的完整 "
            "candidate-rule ID；代词、未标注 ID、遗漏或补写均不会消费。",
            code="voting_search_rule_controls_not_grounded",
            fields=("include_rule_ids", "exclude_rule_ids"),
        )

    observed_max = {
        int(match.group("value"))
        for match in _VOTING_SEARCH_MAX_COMBINATIONS_RE.finditer(utterance)
    }
    expected_max = int(inputs["max_combinations"])
    if (observed_max and observed_max != {expected_max}) or (
        not observed_max and expected_max != 10_000
    ):
        return _clarification(
            "max_combinations 只能采用当前原话唯一明确的 1..10000 整数；"
            "未提供时固定为 10000。",
            code="voting_search_budget_not_grounded",
            fields=("max_combinations",),
        )
    return result


def _voting_search_member_counts(utterance: str) -> set[int]:
    values: set[int] = set()
    for pattern in _VOTING_SEARCH_MEMBER_COUNT_PATTERNS:
        for match in pattern.finditer(utterance):
            values.add(int(match.group("k")))
    return values


def _voting_search_objectives(utterance: str) -> set[tuple[str, str]]:
    values: set[tuple[str, str]] = set()
    for pattern in _VOTING_SEARCH_OBJECTIVE_PATTERNS:
        for match in pattern.finditer(utterance):
            raw_direction = match.group("direction").lower()
            direction = (
                "maximize"
                if raw_direction in {"最大化", "maximize", "maximise"}
                else "minimize"
            )
            metric = _VOTING_SEARCH_METRIC_ALIASES[
                match.group("metric").casefold()
            ]
            values.add((metric, direction))
    return values


def _voting_search_constraints(
    utterance: str,
) -> set[tuple[str, str, float]]:
    values: set[tuple[str, str, float]] = set()
    for match in _VOTING_SEARCH_CONSTRAINT_RE.finditer(utterance):
        operator = match.group("operator").lower()
        normalized_operator = (
            "gte" if operator in {">=", "gte", "至少", "不少于", "不低于"} else "lte"
        )
        token = match.group("value")
        number = float(token[:-1]) / 100.0 if token.endswith("%") else float(token)
        metric = _VOTING_SEARCH_METRIC_ALIASES[
            match.group("metric").casefold()
        ]
        values.add((metric, normalized_operator, number))
    return values


def _voting_search_rule_controls(
    utterance: str,
) -> tuple[set[str], set[str], set[str]]:
    labels: list[tuple[int, int, str]] = []
    labels.extend(
        (match.start(), match.end(), "include")
        for match in _VOTING_SEARCH_INCLUDE_LABEL_RE.finditer(utterance)
    )
    labels.extend(
        (match.start(), match.end(), "exclude")
        for match in _VOTING_SEARCH_EXCLUDE_LABEL_RE.finditer(utterance)
    )
    labels.sort(key=lambda item: (item[0], item[1], item[2]))
    include: set[str] = set()
    exclude: set[str] = set()
    labeled: set[str] = set()
    for index, (start, end, kind) in enumerate(labels):
        clause_start = max(
            utterance.rfind(separator, 0, start)
            for separator in ("；", ";", "。", ".", "!", "！", "?", "？", "\n")
        ) + 1
        clause_end_candidates = [
            position
            for separator in ("；", ";", "。", ".", "!", "！", "?", "？", "\n")
            if (position := utterance.find(separator, end)) >= 0
        ]
        clause_end = (
            min(clause_end_candidates)
            if clause_end_candidates
            else len(utterance)
        )
        next_start = (
            labels[index + 1][0]
            if index + 1 < len(labels)
            and labels[index + 1][0] < clause_end
            else clause_end
        )
        if _VOTING_SEARCH_NEGATED_LABEL_PREFIX_RE.search(
            utterance[clause_start:start]
        ):
            continue
        for match in _VOTING_RULE_ID_TOKEN_RE.finditer(
            utterance,
            end,
            next_start,
        ):
            local_start = (
                max(
                    utterance.rfind("，", end, match.start()),
                    utterance.rfind(",", end, match.start()),
                    end - 1,
                )
                + 1
            )
            local_end_candidates = [
                position
                for separator in ("，", ",")
                if (
                    position := utterance.find(
                        separator,
                        match.end(),
                        next_start,
                    )
                )
                >= 0
            ]
            local_end = (
                min(local_end_candidates)
                if local_end_candidates
                else next_start
            )
            if _VOTING_NEGATED_CONTROL_RE.search(
                utterance[local_start:local_end]
            ):
                continue
            rule_id = match.group(0)
            labeled.add(rule_id)
            if kind == "include":
                include.add(rule_id)
            else:
                exclude.add(rule_id)
    return include, exclude, labeled


def _ground_voting_candidate_build_from_search(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one exact search/combo pointer pair in the current command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if not _utterance_targets_voting_search_selection(utterance):
        return _clarification(
            "请明确要求从一个完整 Voting search_id 和一个完整 combo_id "
            "构建或物化候选。",
            code="voting_search_selection_intent_required",
            fields=("build_intent", "search_id", "combo_id"),
        )
    if (
        _VOTING_NEGATED_BUILD_RE.search(utterance) is not None
        or _VOTING_NONCOMMAND_RE.search(utterance) is not None
        or _VOTING_POSTPOSED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "Voting 搜索结果构建必须是当前轮立即执行的肯定式单步命令；问句、"
            "否定、假设、历史/未来描述或句尾撤销不会构建候选。",
            code="voting_search_selection_positive_command_required",
            fields=("build_intent",),
        )
    if (
        _voting_search_selection_has_positive_research(utterance)
        or _voting_search_selection_has_positive_follow_up(utterance)
    ):
        return _clarification(
            "本轮只从精确 search_id/combo_id 构建 Voting 候选；加入或修改 "
            "Strategy Pool、设置动作、应用、采纳、部署和写回必须另发请求。",
            code="voting_search_selection_single_step_required",
            fields=("next_action",),
        )
    if _VOTING_SEARCH_SELECTION_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Voting 搜索结果构建只接受 search_id、combo_id 与可选 strategy_type；"
            "artifact/hash、rule/entry/member IDs、n 和 rank 均由平台重新恢复，"
            "不能由自然语言注入。",
            code="voting_search_selection_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _VOTING_SEARCH_SELECTION_HEURISTIC_RE.search(utterance) is not None:
        return _clarification(
            "请逐字点名完整 search_id 与 combo_id；即使同时提供 pointer，平台也"
            "不会消费第一名、最好、冠军、Top N、排名或‘刚才那个’等启发式选择。",
            code="voting_search_selection_explicit_ids_required",
            fields=("search_id", "combo_id"),
        )
    search_ids = tuple(
        match.group(0) for match in _VOTING_SEARCH_ID_TOKEN_RE.finditer(utterance)
    )
    combo_ids = tuple(
        match.group(0) for match in _VOTING_COMBO_ID_TOKEN_RE.finditer(utterance)
    )
    if search_ids != (inputs["search_id"],) or combo_ids != (inputs["combo_id"],):
        return _clarification(
            "Voting 搜索结果构建必须逐字提供且只提供一个完整 search_id 与一个"
            "完整 combo_id；平台不会补全、替换、按排名选择或消费代词。",
            code="voting_search_selection_controls_not_grounded",
            fields=("search_id", "combo_id"),
        )
    observed_types = {
        value for value, _start, _end in _voting_strategy_type_mentions(utterance)
    }
    expected_type = inputs.get("strategy_type")
    if (
        expected_type is not None
        and observed_types != {expected_type}
    ) or (expected_type is None and observed_types):
        return _clarification(
            "可选 strategy_type 只能逐字采用当前请求中唯一明确的 Strategy Pool "
            "类型；未明确时模型必须省略并由平台唯一解析。",
            code="voting_search_selection_strategy_type_not_grounded",
            fields=("strategy_type",),
        )
    return result


def _ground_voting_candidate_build(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove the exact rule set and n came from one positive user command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _VOTING_NEGATED_BUILD_RE.search(utterance) is not None:
        return _clarification(
            "原话否定了 Voting 候选构建，本轮不会生成候选。"
            "如需继续，请重新给出一条明确的正向构建请求。",
            code="voting_candidate_build_intent_negated",
            fields=("build_intent",),
        )
    if (
        _VOTING_NONCOMMAND_RE.search(utterance) is not None
        or _VOTING_POSTPOSED_CANCELLATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "当前原话是问句、假设/未来/历史描述、演示性文本或已在句尾撤销，"
            "不能视为立即执行 Voting 构建的唯一正向命令。请单独重述本次要构建的"
            "策略池类型、完整 rule_id 列表和唯一 n-of-k 阈值。",
            code="voting_candidate_positive_command_required",
            fields=("build_intent",),
        )
    command_spans = _voting_positive_command_clause_spans(utterance)
    if not command_spans:
        return _clarification(
            "请明确说出要构建或测算一个 Voting / n-of-k 候选，并在同一条"
            "请求中给出策略池类型、完整 rule_id 列表和 n。",
            code="voting_candidate_build_intent_required",
            fields=("build_intent",),
        )
    if len(command_spans) != 1:
        return _clarification(
            "一次请求只能包含一个立即执行的 Voting 构建/评估子句；"
            "请把每组 rule_id 与 n-of-k 控制拆成独立请求。",
            code="voting_candidate_single_command_required",
            fields=("build_intent",),
        )
    command_span = command_spans[0]
    if _VOTING_HEURISTIC_SELECTION_RE.search(utterance) is not None:
        return _clarification(
            "Voting 构建必须逐字点名当前 Strategy Pool 中的完整 rule_id；"
            "不能让模型按最好、风险最高、刚才那些等表述自动选择规则。",
            code="voting_candidate_explicit_rules_required",
            fields=("rule_ids",),
        )
    if (
        _VOTING_FOLLOW_UP_RE.search(utterance) is not None
        or _VOTING_OTHER_POOL_OPERATION_RE.search(utterance) is not None
    ):
        return _clarification(
            "本轮只生成并测算 Voting 候选；删除、重排、编译、加入 "
            "Strategy Pool、设置业务动作、采纳、部署或写回必须另发请求。",
            code="voting_candidate_single_step_required",
            fields=("next_action",),
        )
    if _VOTING_NEGATED_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "本轮 Voting 控制中含有被否定、排除或随后改写的 rule_id/n；"
            "请重新给出不含历史值和否定值的一组完整 rule_id 与唯一 n。",
            code="voting_candidate_negated_control",
            fields=("rule_ids", "n"),
        )

    rule_matches = tuple(_VOTING_RULE_ID_TOKEN_RE.finditer(utterance))
    strategy_type_mentions = _voting_strategy_type_mentions(utterance)
    n_mentions = _voting_n_mentions(utterance)
    if (
        any(
            not _voting_mention_is_within(match.start(), match.end(), command_span)
            for match in rule_matches
        )
        or any(
            not _voting_mention_is_within(start, end, command_span)
            for _strategy_type, start, end in strategy_type_mentions
        )
        or any(
            not _voting_mention_is_within(start, end, command_span)
            for _n, _k, start, end in n_mentions
        )
    ):
        return _clarification(
            "Voting 的策略池类型、完整 rule_id 与 n-of-k 必须全部位于唯一"
            "正向构建子句中；历史、引用、否定或其他子句中的控制不会被消费。",
            code="voting_candidate_controls_outside_command",
            fields=("strategy_type", "rule_ids", "n"),
        )

    observed_rule_ids = [match.group(0) for match in rule_matches]
    expected_rule_ids = list(inputs["rule_ids"])
    if (
        len(observed_rule_ids) != len(set(observed_rule_ids))
        or set(observed_rule_ids) != set(expected_rule_ids)
        or len(observed_rule_ids) != len(expected_rule_ids)
    ):
        return _clarification(
            "请逐字提供 2 到 50 个互不重复的完整 candidate-rule ID；"
            "模型不能补全、替换、遗漏或从代词推断规则。",
            code="voting_candidate_rules_not_grounded",
            fields=("rule_ids",),
        )

    strategy_type = str(inputs["strategy_type"])
    observed_strategy_types = {
        candidate for candidate, _start, _end in strategy_type_mentions
    }
    if observed_strategy_types != {strategy_type}:
        return _clarification(
            "请显式且唯一标注 Voting 来源 Strategy Pool 的类型；存在缺失、多个"
            "类型或与结构化草案不一致时，平台不会替用户选择 approval/reject/"
            "limit/pricing/segmentation。",
            code="voting_candidate_strategy_type_not_grounded",
            fields=("strategy_type",),
        )

    n_bindings = _voting_n_bindings(
        utterance[command_span[0] : command_span[1]]
    )
    if (
        not n_bindings
        or {value for value, _k in n_bindings} != {inputs["n"]}
        or any(
            supplied_k is not None and supplied_k != len(expected_rule_ids)
            for _value, supplied_k in n_bindings
        )
    ):
        return _clarification(
            "请明确且唯一给出与规则数量一致的 n-of-k 命中阈值，例如“n=2”"
            "或“3 选 2”；多个阈值、错误的 k 或草案不一致时平台不会替用户选择。",
            code="voting_candidate_n_not_grounded",
            fields=("n",),
        )
    return result


def utterance_targets_strategy_pool_stability(utterance: str) -> bool:
    """Reserve a positive current-Pool distribution-stability command."""

    if utterance_targets_candidate_monthly_stability(utterance):
        return False
    if _POOL_VALIDATION_TARGET_RE.search(utterance) is not None:
        return False
    if (
        _POOL_STABILITY_TARGET_RE.search(utterance) is None
        or _POOL_STABILITY_POSITIVE_INTENT_RE.search(utterance) is None
        or _POOL_IMPACT_REPORT_ONLY_RE.search(utterance) is not None
    ):
        return False
    return True


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


def _utterance_targets_strategy_pool_apply(utterance: str) -> bool:
    """Reserve any explicit current-Pool-to-current-sample application clause."""

    return _POOL_APPLY_TARGET_RE.search(utterance) is not None


def utterance_targets_strategy_pool_materialize(utterance: str) -> bool:
    """Reserve an explicit current-Pool-to-draft-Strategy command."""

    return _POOL_MATERIALIZE_TARGET_RE.search(utterance) is not None


def _utterance_targets_strategy_pool_validation(utterance: str) -> bool:
    """Reserve explicit independent validation/OOT Pool replay clauses."""

    return _POOL_VALIDATION_TARGET_RE.search(utterance) is not None


def _utterance_targets_strategy_pool_impact(utterance: str) -> bool:
    if utterance_targets_strategy_pool_stability(utterance):
        return False
    if _POOL_IMPACT_TARGET_RE.search(utterance) is None:
        return False
    if (
        _POOL_IMPACT_REPORT_ONLY_RE.search(utterance) is not None
        and _POOL_IMPACT_POSITIVE_INTENT_RE.search(utterance) is None
        and _POOL_IMPACT_NEGATED_RE.search(utterance) is None
        and _POOL_IMPACT_NONCOMMAND_RE.search(utterance) is None
    ):
        # "加入 Pool，然后生成效果报告" contains both Pool and effect, but
        # its report clause does not authorize an impact measurement.
        return False
    signals = tuple(
        match
        for pattern in (
            _POOL_IMPACT_POSITIVE_INTENT_RE,
            _POOL_IMPACT_NEGATED_RE,
            _POOL_IMPACT_NONCOMMAND_RE,
            _POOL_IMPACT_REPORT_ONLY_RE,
        )
        for match in pattern.finditer(utterance)
    )
    if not signals:
        return False
    other_operations = tuple(_POOL_IMPACT_SECOND_OPERATION_RE.finditer(utterance))
    if not other_operations:
        return True
    if _POOL_IMPACT_NEGATED_RE.search(utterance) is not None:
        # A negated impact clause followed by a separate positive operation
        # (for example "不要回测，只编译") must not hijack that operation.
        return False
    # "评估把 X 加入策略池的影响" describes one hypothetical add: its impact
    # phrase encloses the add verb and must stay on the existing add guardrail.
    # A standalone "测算 Pool 影响，然后编译/部署" span does not overlap the
    # second operation and must still force the impact-specific clarification.
    return any(
        all(
            signal.end() <= operation.start()
            or operation.end() <= signal.start()
            for operation in other_operations
        )
        for signal in signals
    )


def _ground_strategy_pool_apply_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove the one Pool type and optional output prefix came from this command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _POOL_APPLY_NONCURRENT_RE.search(utterance) is not None
        or _POOL_APPLY_POSITIVE_INTENT_RE.search(utterance) is None
        or _POOL_APPLY_TARGET_RE.search(utterance) is None
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求把一个指定类型的当前 "
            "Strategy Pool 应用或写回当前样本；否定、问句、历史/未来或假设"
            "描述不会创建派生数据集。",
            code="strategy_pool_apply_positive_command_required",
            fields=("apply_intent",),
        )
    if _POOL_APPLY_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Pool revision/hash、artifact、数据集、SampleDesign、requirements、"
            "StrategySpec 和生命周期状态只能由平台恢复；请求中只能提供 Pool "
            "类型与可选 output_prefix。",
            code="strategy_pool_apply_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _POOL_APPLY_SECOND_OPERATION_RE.search(utterance) is not None:
        return _clarification(
            "Strategy Pool 应用必须是当前轮唯一操作；Pool 修改、采纳、激活、"
            "部署、上线、导出或报告必须拆成后续请求。派生数据集默认不激活。",
            code="strategy_pool_apply_single_operation_required",
            fields=("workflow",),
        )

    missing_controls: list[str] = []
    strategy_type = str(inputs.get("strategy_type") or "")
    mentioned_types = {
        item[0] for item in _voting_strategy_type_mentions(utterance)
    }
    pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if (
        pattern is None
        or pattern.search(utterance) is None
        or mentioned_types != {strategy_type}
    ):
        missing_controls.append(f"strategy_type {strategy_type or 'unknown'}")

    prefix_labels = tuple(_POOL_APPLY_OUTPUT_PREFIX_LABEL_RE.finditer(utterance))
    prefix_matches = tuple(_POOL_APPLY_OUTPUT_PREFIX_RE.finditer(utterance))
    prefix_mentions = tuple(match.group("prefix") for match in prefix_matches)
    output_prefix = inputs.get("output_prefix")
    if len(prefix_labels) != len(prefix_matches):
        missing_controls.append("output_prefix")
    elif output_prefix is None:
        if prefix_mentions:
            missing_controls.append("output_prefix")
    elif prefix_mentions != (output_prefix,):
        missing_controls.append(f"output_prefix {output_prefix}")

    if missing_controls:
        missing_controls = list(dict.fromkeys(missing_controls))
        return _clarification(
            "Strategy Pool 应用只能采用原话中唯一明确的 Pool 类型与可选 ASCII "
            "输出前缀；当前无法核对："
            + "、".join(missing_controls)
            + "。平台不会替用户猜测 Pool、前缀或任何数据/证据绑定。",
            code="strategy_pool_apply_controls_not_grounded",
            fields=tuple(missing_controls),
        )
    return result


def _ground_strategy_pool_materialize_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove one current Pool type and a draft-only materialization command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    intent_text = _POOL_MATERIALIZE_NEGATED_LIFECYCLE_DISCLAIMER_RE.sub(
        "",
        utterance,
    )
    if (
        _POOL_MATERIALIZE_NONCURRENT_RE.search(intent_text) is not None
        or _POOL_MATERIALIZE_POSITIVE_INTENT_RE.search(intent_text) is None
        or _POOL_MATERIALIZE_TARGET_RE.search(intent_text) is None
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求把一个指定类型的当前 "
            "Strategy Pool 物化为 draft Strategy；否定、问句、历史/未来或"
            "假设描述不会创建策略。",
            code="strategy_pool_materialize_positive_command_required",
            fields=("materialize_intent",),
        )
    if _POOL_MATERIALIZE_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Pool revision/hash、artifact、design hash、StrategySpec、"
            "requirements 和指标只能由平台恢复；请求中只能提供 Pool 类型。",
            code="strategy_pool_materialize_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _POOL_MATERIALIZE_SECOND_OPERATION_RE.search(intent_text) is not None:
        return _clarification(
            "Strategy Pool 物化必须是当前轮唯一操作；采纳、部署、回测、应用、"
            "报告、监控和 DSL 导出必须拆成后续请求。本步骤只创建 draft Strategy。",
            code="strategy_pool_materialize_single_operation_required",
            fields=("workflow",),
        )

    strategy_type = str(inputs.get("strategy_type") or "")
    mentioned_types = {
        item[0] for item in _voting_strategy_type_mentions(utterance)
    }
    pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if (
        pattern is None
        or pattern.search(utterance) is None
        or mentioned_types != {strategy_type}
    ):
        return _clarification(
            "Strategy Pool 物化只能采用原话中唯一明确的 Pool 类型；平台不会"
            "替用户猜测 Pool、hash、StrategySpec、requirements 或指标。",
            code="strategy_pool_materialize_controls_not_grounded",
            fields=("strategy_type",),
        )
    return result


def _ground_strategy_pool_validation_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove Pool type and one independent partition came from this command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _POOL_VALIDATION_NONCURRENT_RE.search(utterance) is not None
        or _POOL_VALIDATION_POSITIVE_INTENT_RE.search(utterance) is None
        or _POOL_VALIDATION_TARGET_RE.search(utterance) is None
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求对一个 approval/reject "
            "Strategy Pool 执行 validation 或 OOT 独立样本回放验证；"
            "否定、问句、历史/未来或假设描述不会执行。",
            code="strategy_pool_validation_positive_command_required",
            fields=("validation_intent",),
        )
    if _POOL_VALIDATION_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "Pool/SampleDesign artifact、revision/hash、dataset/workspace、"
            "target、requirements、population、comparison_mode、指标与状态"
            "只能由平台恢复；请求中只能提供 Pool 类型和 validation/OOT 分区。",
            code="strategy_pool_validation_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _POOL_VALIDATION_EVIDENCE_SCOPE_RE.search(utterance) is not None:
        return _clarification(
            "独立样本回放验证只发布实际 validation/OOT 动作、风险、金额和逐月"
            "回放证据，不计算或声称 PSI、稳定性或漂移；这些必须使用单独的"
            "稳定性 Workflow。",
            code="strategy_pool_validation_evidence_scope_forbidden",
            fields=("evidence_scope",),
        )
    if _POOL_VALIDATION_SECOND_OPERATION_RE.search(utterance) is not None:
        return _clarification(
            "Strategy Pool 独立样本回放验证必须是当前轮唯一操作；改 Pool、"
            "应用写回、报告、晋级、采纳或部署必须拆成后续请求。",
            code="strategy_pool_validation_single_operation_required",
            fields=("workflow",),
        )

    missing_controls: list[str] = []
    strategy_type = str(inputs.get("strategy_type") or "")
    mentioned_types = {
        item[0] for item in _voting_strategy_type_mentions(utterance)
    }
    type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if (
        type_pattern is None
        or type_pattern.search(utterance) is None
        or mentioned_types != {strategy_type}
    ):
        missing_controls.append(f"strategy_type {strategy_type or 'unknown'}")

    partition = str(inputs.get("partition") or "")
    mentioned_partitions = {
        name
        for name, pattern in _POOL_VALIDATION_PARTITION_GROUNDING.items()
        if pattern.search(utterance) is not None
    }
    partition_pattern = _POOL_VALIDATION_PARTITION_GROUNDING.get(partition)
    if (
        partition_pattern is None
        or partition_pattern.search(utterance) is None
        or mentioned_partitions != {partition}
    ):
        missing_controls.append(f"partition {partition or 'unknown'}")

    if missing_controls:
        return _clarification(
            "独立样本回放验证只能采用原话中唯一明确的 approval/reject Pool "
            "类型和一个 validation/OOT 分区；当前无法核对："
            + "、".join(dict.fromkeys(missing_controls))
            + "。平台不会猜测类型、分区或证据绑定。",
            code="strategy_pool_validation_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing_controls)),
        )
    return result


def _ground_strategy_pool_stability_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove the one current Pool type came from this stability command."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _POOL_STABILITY_NONCURRENT_RE.search(utterance) is not None
        or _POOL_STABILITY_POSITIVE_INTENT_RE.search(utterance) is None
        or _POOL_STABILITY_TARGET_RE.search(utterance) is None
        or _POOL_IMPACT_REPORT_ONLY_RE.search(utterance) is not None
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求测量一个当前 Strategy Pool "
            "的跨分区 PSI 稳定性；否定、问句、历史/未来、假设或仅生成报告"
            "不会执行测量。",
            code="strategy_pool_stability_positive_command_required",
            fields=("stability_intent",),
        )
    if _POOL_STABILITY_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "ImpactCube/Pool/SampleDesign artifact、revision/hash、dataset、"
            "阈值、指标与结果只能由平台冻结或计算；请求中只能提供五类 Pool "
            "之一的 strategy_type。",
            code="strategy_pool_stability_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _POOL_STABILITY_SECOND_OPERATION_RE.search(utterance) is not None:
        return _clarification(
            "Strategy Pool 跨分区稳定性测量必须是当前轮唯一操作；Pool 修改、"
            "应用写回、报告、创建、采纳、晋级或部署必须拆成后续请求。",
            code="strategy_pool_stability_single_operation_required",
            fields=("workflow",),
        )

    strategy_type = str(inputs.get("strategy_type") or "")
    mentioned_types = {
        item[0] for item in _impact_cube_strategy_type_mentions(utterance)
    }
    type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if (
        type_pattern is None
        or type_pattern.search(utterance) is None
        or mentioned_types != {strategy_type}
    ):
        return _clarification(
            "跨分区稳定性只能采用原话中唯一明确的 approval、reject、limit、"
            "pricing 或 segmentation Pool 类型；平台不会从动作、指标或历史"
            "证据猜测类型。",
            code="strategy_pool_stability_controls_not_grounded",
            fields=(f"strategy_type {strategy_type or 'unknown'}",),
        )
    return result


def _ground_strategy_pool_impact_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Prove every executable measurement control came from this utterance."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _POOL_IMPACT_NEGATED_RE.search(utterance)
        or _POOL_IMPACT_NONCOMMAND_RE.search(utterance)
        or _POOL_IMPACT_REPORT_ONLY_RE.search(utterance)
    ):
        return _clarification(
            "请用当前轮、肯定式的单一命令明确要求 Strategy Pool 影响测算；"
            "否定、问句、历史/未来描述或仅生成报告不会执行测算。",
            code="strategy_pool_impact_positive_command_required",
            fields=("measurement_intent",),
        )
    if _POOL_IMPACT_POSITIVE_INTENT_RE.search(utterance) is None:
        return _clarification(
            "原话没有明确授权执行 Strategy Pool 影响测算。请明确说出要测算的"
            " approval 或 reject Pool；本 Workflow 只生成只读证据。",
            code="strategy_pool_impact_positive_command_required",
            fields=("measurement_intent",),
        )
    if _POOL_IMPACT_SECOND_OPERATION_RE.search(utterance):
        return _clarification(
            "Strategy Pool 影响测算必须是当前轮唯一操作；入池、删除、改动作、重排、"
            "编译、创建策略、写回、采纳或部署必须拆成后续请求。",
            code="strategy_pool_impact_single_operation_required",
            fields=("workflow",),
        )

    missing_controls: list[str] = []
    strategy_type = str(inputs.get("strategy_type") or "")
    strategy_type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    strategy_type_mentions = _voting_strategy_type_mentions(utterance)
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

    comparison_mode = str(inputs.get("comparison_mode") or "absolute")
    mentions_baseline_comparison = (
        _POOL_IMPACT_BASELINE_MODE_RE.search(utterance) is not None
    )
    mentions_absolute = _POOL_IMPACT_ABSOLUTE_MODE_RE.search(utterance) is not None
    if comparison_mode == "vs_baseline":
        if not mentions_baseline_comparison or mentions_absolute:
            missing_controls.append("comparison_mode vs_baseline")
        baseline_strategy_id = str(inputs.get("baseline_strategy_id") or "")
        baseline_id_mentions = tuple(
            _POOL_IMPACT_STRATEGY_ID_RE.finditer(utterance)
        )
        positively_mentioned_ids = {
            match.group(0).casefold()
            for match in baseline_id_mentions
            if not _pool_impact_span_is_negated(utterance, start=match.start())
        }
        negated_ids = {
            match.group(0).casefold()
            for match in baseline_id_mentions
            if _pool_impact_span_is_negated(utterance, start=match.start())
        }
        selected_id = baseline_strategy_id.casefold()
        if (
            not baseline_strategy_id
            or not _utterance_contains_token(utterance, baseline_strategy_id)
            or positively_mentioned_ids != {selected_id}
            or selected_id in negated_ids
        ):
            missing_controls.append(
                baseline_strategy_id or "baseline_strategy_id"
            )
    elif mentions_baseline_comparison or (
        _POOL_IMPACT_STRATEGY_ID_RE.search(utterance) is not None
    ):
        missing_controls.append("comparison_mode vs_baseline")

    mentioned_columns = tuple(
        column for column in whitelist if _utterance_contains_token(utterance, column)
    )
    explicit_column_bindings = _pool_impact_explicit_column_bindings(
        utterance,
        whitelist,
    )
    for field, values in explicit_column_bindings.items():
        selected = inputs.get(field)
        if len(values) != 1 or selected not in values:
            expected = "/".join(sorted(values)) or field
            missing_controls.append(f"{field} {expected}")
    for field in ("month_col", "loan_amount_col", "overdue_amount_col"):
        value = inputs.get(field)
        if isinstance(value, str):
            if (
                not _utterance_contains_token(utterance, value)
                or _pool_impact_token_is_negated(utterance, value)
                or any(
                    other != value
                    and _pool_impact_tokens_are_alternatives(
                        utterance, value, other
                    )
                    for other in mentioned_columns
                )
            ):
                missing_controls.append(f"{field} {value}")
    if inputs.get("drop_nan_labels") is True and (
        _POOL_IMPACT_DROP_NAN_TRUE_RE.search(utterance) is None
        or _POOL_IMPACT_DROP_NAN_NEGATED_RE.search(utterance) is not None
    ):
        missing_controls.append("drop_nan_labels=true")
    if missing_controls:
        rendered = "、".join(dict.fromkeys(missing_controls))
        return _clarification(
            "Strategy Pool 影响测算只能使用用户原话中的 Pool 类型、基线模式/完整 ID、"
            "精确列名和空标签授权；当前无法核对："
            f"{rendered}。平台不会采用 LLM 猜测的数据绑定、列、hash、指标或策略。",
            code="strategy_pool_impact_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing_controls)),
        )
    return result


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


def _pool_impact_span_is_negated(utterance: str, *, start: int) -> bool:
    prefix = utterance[max(0, start - 32) : start]
    return re.search(
        r"(?:不要|别|不用|不使用|禁止|排除|剔除|而非|不是|并非)"
        r"[^，,；;。.!?？\n]{0,16}$|"
        r"(?<![A-Za-z0-9_])(?:do\s+not|don't|not|exclude|without)"
        r"[^,;.!?\n]{0,16}$",
        prefix,
        re.IGNORECASE,
    ) is not None


def _pool_impact_token_is_negated(utterance: str, token: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )
    return any(
        _pool_impact_span_is_negated(utterance, start=match.start())
        for match in pattern.finditer(utterance)
    )


def _pool_impact_explicit_column_bindings(
    utterance: str,
    whitelist: tuple[str, ...],
) -> dict[str, set[str]]:
    labels = {
        "month_col": (
            r"(?:月份|月度|申请月|观察月)(?:字段|列)|"
            r"(?<![A-Za-z0-9_])month(?:_col|\s+column)(?![A-Za-z0-9_])"
        ),
        "loan_amount_col": (
            r"(?:放款|贷款|借款)金额(?:字段|列)|"
            r"(?<![A-Za-z0-9_])loan(?:_amount_col|\s+amount\s+column)"
            r"(?![A-Za-z0-9_])"
        ),
        "overdue_amount_col": (
            r"逾期金额(?:字段|列)|"
            r"(?<![A-Za-z0-9_])overdue(?:_amount_col|\s+amount\s+column)"
            r"(?![A-Za-z0-9_])"
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


def _pool_impact_tokens_are_alternatives(
    utterance: str,
    left_token: str,
    right_token: str,
) -> bool:
    token_patterns = (
        re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for token in (left_token, right_token)
    )
    left_matches, right_matches = (tuple(pattern.finditer(utterance)) for pattern in token_patterns)
    for left in left_matches:
        for right in right_matches:
            first, second = sorted((left, right), key=lambda item: item.start())
            between = utterance[first.end() : second.start()]
            if len(between) <= 24 and re.search(
                r"(?:或者|或是|还是|或|/|\bor\b)", between, re.IGNORECASE
            ):
                return True
    return False


def _voting_n_bindings(utterance: str) -> tuple[tuple[int, int | None], ...]:
    bindings: list[tuple[int, int | None]] = []
    for n, k, _start, _end in _voting_n_mentions(utterance):
        binding = (n, k)
        if binding not in bindings:
            bindings.append(binding)
    return tuple(bindings)


def _automatic_tree_platform_control_clarification(
    utterance: str,
) -> StrategyRequestCompilation | None:
    if _AUTOMATIC_TREE_DATASET_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "自动树绑定当前任务的 dataset 与 workspace，本请求不能切换样本。"
            "请先切换 workspace 或创建使用目标样本的新任务，再发起建树。",
            code="automatic_tree_build_dataset_context_required",
            fields=("dataset_id", "workspace_id"),
        )
    if _AUTOMATIC_TREE_TARGET_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "自动树目标列由当前任务上下文绑定，本请求不能覆盖 target_col。"
            "请先在任务中确认或切换标签列，再发起建树。",
            code="automatic_tree_build_target_context_required",
            fields=("target_col",),
        )
    if _AUTOMATIC_TREE_LABEL_POLICY_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "空标签处理策略由平台任务契约绑定，不能在本次自动树请求中覆盖。"
            "请先确认任务的标签清洗口径。",
            code="automatic_tree_build_label_policy_not_overridable",
            fields=("drop_nan_labels",),
        )
    if _AUTOMATIC_TREE_BUDGET_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "自动树执行预算及其默认值由平台治理，不能在本次请求中覆盖；"
            "询问预算默认值也不会创建 build。请使用当前平台预算，或先调整治理配置。",
            code="automatic_tree_build_platform_budget_not_overridable",
            fields=(
                "budgets",
                "max_rows",
                "max_features",
                "max_cells",
                "max_nodes",
                "max_cutpoint",
            ),
        )
    return None


def _automatic_tree_leaf_has_positive_materialization_intent(utterance: str) -> bool:
    """Return true only for an explicit, non-negated pointer operation."""

    operation_text = _AUTOMATIC_TREE_LEAF_REASON_RE.sub(" ", utterance)
    for clause in _automatic_tree_follow_up_clauses(operation_text):
        for match in _AUTOMATIC_TREE_LEAF_MATERIALIZATION_ACTION_RE.finditer(clause):
            if re.search(r"(?:不|未|没(?:有)?)\s*$", clause[: match.start()]):
                continue
            if not _automatic_tree_follow_up_action_is_negated(
                clause,
                action_start=match.start(),
            ):
                return True
    return False


def _automatic_tree_leaf_explicit_reasons(utterance: str) -> tuple[str, ...]:
    reasons: list[str] = []
    for match in _AUTOMATIC_TREE_LEAF_REASON_RE.finditer(utterance):
        left = max(
            utterance.rfind(separator, 0, match.start())
            for separator in ("，", ",", "；", ";", "。", "\n")
        )
        prefix = utterance[left + 1 : match.start()]
        if _AUTOMATIC_TREE_LEAF_REASON_NEGATION_RE.search(prefix) is not None:
            continue
        value = match.group("zh") or match.group("en") or ""
        canonical = " ".join(unicodedata.normalize("NFC", value).split())
        if canonical:
            reasons.append(canonical)
    return tuple(reasons)


def _automatic_tree_leaf_all_reason_values(utterance: str) -> tuple[str, ...]:
    return tuple(
        " ".join(
            unicodedata.normalize(
                "NFC",
                match.group("zh") or match.group("en") or "",
            ).split()
        )
        for match in _AUTOMATIC_TREE_LEAF_REASON_RE.finditer(utterance)
    )


def _automatic_tree_leaf_rationale_is_allowed(reason: str) -> bool:
    if _AUTOMATIC_TREE_LEAF_RATIONALE_START_RE.search(reason) is None:
        return False
    remaining = _AUTOMATIC_TREE_LEAF_RATIONALE_TOKEN_RE.sub(" ", reason)
    remaining = _AUTOMATIC_TREE_LEAF_RATIONALE_PUNCTUATION_RE.sub(" ", remaining)
    return not remaining.strip()


def _cross_matrix_cell_rationale_is_allowed(reason: str) -> bool:
    without_cell_terms = re.sub(
        r"(?:二维|交叉|Cross\s+Matrix|matrix|这(?:些|两个)?|这些|两个|多个|"
        r"格子|单元格|cells?)",
        " ",
        reason,
        flags=re.IGNORECASE,
    )
    return _automatic_tree_leaf_rationale_is_allowed(without_cell_terms)


def _automatic_tree_leaf_unconsumed_request_text(utterance: str) -> str:
    """Remove the one allowed pointer operation and return every other demand.

    The grammar is intentionally narrow. New natural-language synonyms do not
    silently become executable multi-step operations; they require a safe
    clarification until the platform assigns them an explicit contract.
    """

    remaining = unicodedata.normalize("NFC", utterance)
    remaining = _AUTOMATIC_TREE_LEAF_NEGATED_REASON_CLAUSE_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_REASON_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_NEGATED_CLAUSE_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_ID_TOKEN_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_ALLOWED_REQUEST_TOKEN_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_REQUEST_PUNCTUATION_RE.sub(" ", remaining)
    return " ".join(remaining.split())


def _cross_matrix_cell_has_positive_selection_intent(utterance: str) -> bool:
    operation_text = _AUTOMATIC_TREE_LEAF_REASON_RE.sub(" ", utterance)
    operation_text = _CROSS_MATRIX_CELL_NEGATED_FOLLOW_UP_RE.sub(" ", operation_text)
    for clause in _automatic_tree_follow_up_clauses(operation_text):
        for match in _CROSS_MATRIX_CELL_SELECTION_VERB_RE.finditer(clause):
            prefix = clause[: match.start()]
            if re.search(r"(?:不|未|没(?:有)?)\s*$", prefix):
                continue
            if not _automatic_tree_follow_up_action_is_negated(
                clause,
                action_start=match.start(),
            ):
                return True
    return False


def _cross_matrix_cell_unconsumed_request_text(utterance: str) -> str:
    remaining = unicodedata.normalize("NFC", utterance)
    remaining = _AUTOMATIC_TREE_LEAF_NEGATED_REASON_CLAUSE_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_REASON_RE.sub(" ", remaining)
    remaining = _CROSS_MATRIX_CELL_NEGATED_FOLLOW_UP_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.sub(" ", remaining)
    remaining = _CROSS_MATRIX_CELL_ID_TOKEN_RE.sub(" ", remaining)
    remaining = _CROSS_MATRIX_CELL_ALLOWED_REQUEST_TOKEN_RE.sub(" ", remaining)
    remaining = _AUTOMATIC_TREE_LEAF_REQUEST_PUNCTUATION_RE.sub(" ", remaining)
    return " ".join(remaining.split())


def _ground_cross_matrix_cell_selection(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Bind an exact Cross asset and explicit cell set to one pointer operation."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    positive_operation_text = _CROSS_MATRIX_CELL_NEGATED_FOLLOW_UP_RE.sub(
        " ", utterance
    )
    positive_operation_text = _AUTOMATIC_TREE_LEAF_REASON_RE.sub(
        " ", positive_operation_text
    )

    if (
        _CROSS_MATRIX_CELL_AMBIGUOUS_SELECTION_RE.search(positive_operation_text)
        is not None
        or _CROSS_MATRIX_CELL_HEURISTIC_CONTROL_RE.search(positive_operation_text)
        is not None
    ):
        return _clarification(
            "请从完整 Cross Matrix 结果中复制明确的 cell ID；不能按排名、"
            "极值、风险描述或指标阈值替你选择格子。",
            code="cross_matrix_cell_selection_ambiguous",
            fields=("cell_ids",),
        )
    if (
        _CROSS_MATRIX_CELL_SELECTION_NEGATED_RE.search(positive_operation_text)
        is not None
        or not _cross_matrix_cell_has_positive_selection_intent(utterance)
    ):
        return _clarification(
            "原话没有明确授权一次正向的 Cross Matrix 单元格选择；否定式或仅"
            "描述 ID 的请求不会创建 pointer。请明确说出完整 Cross asset ID 和"
            "要选择的全部 cell ID。",
            code="cross_matrix_cell_intent_negated",
            fields=("selection_intent",),
        )

    reason_values = _automatic_tree_leaf_all_reason_values(utterance)
    explicit_reasons = _automatic_tree_leaf_explicit_reasons(utterance)
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_REPLACEMENT_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "一条请求只能给出一个最终 selection_reason；理由中不能嵌套理由"
            "字段或替换指令。",
            code="cross_matrix_cell_reason_not_grounded",
            fields=("selection_reason",),
        )
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE.search(reason) is not None
        or _CROSS_MATRIX_CELL_HEURISTIC_CONTROL_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "selection_reason 不能包含指标极值、排名或阈值选格语义。请只保留"
            "人工明确选择依据。",
            code="cross_matrix_cell_selection_ambiguous",
            fields=("cell_ids", "selection_reason"),
        )
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_FORBIDDEN_OPERATION_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "selection_reason 不能藏入 Strategy Pool、业务动作、采纳、部署或"
            "写回请求；这些操作必须拆成后续请求。",
            code="cross_matrix_cell_single_step_required",
            fields=("selection_reason", "next_action"),
        )
    if any(
        not _cross_matrix_cell_rationale_is_allowed(reason)
        or _AUTOMATIC_TREE_LEAF_RATIONALE_DECISION_SUBJECT_RE.search(reason)
        is not None
        for reason in explicit_reasons
    ):
        return _clarification(
            "selection_reason 必须是人工/业务/风险/合规/样本评审依据类短说明，"
            "不能包含命中客户、业务动作、策略池或生产操作。",
            code="cross_matrix_cell_reason_not_grounded",
            fields=("selection_reason",),
        )

    active_follow_up_text = _CROSS_MATRIX_CELL_NEGATED_FOLLOW_UP_RE.sub(
        " ", positive_operation_text
    )
    if any(
        pattern.search(active_follow_up_text) is not None
        for pattern in (
            _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE,
        )
    ):
        return _clarification(
            "本轮只创建 Cross Matrix 单元格选择 pointer；加入 Strategy Pool、"
            "设置业务动作、采纳、部署或写回必须分别发起后续请求。",
            code="cross_matrix_cell_single_step_required",
            fields=("next_action",),
        )

    asset_matches = tuple(_AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.finditer(utterance))
    cell_matches = tuple(_CROSS_MATRIX_CELL_ID_TOKEN_RE.finditer(utterance))
    asset_ids = frozenset(match.group(0) for match in asset_matches)
    cell_ids = frozenset(match.group(0) for match in cell_matches)
    ambiguous_fields: list[str] = []
    if len(asset_matches) != 1 or len(asset_ids) != 1:
        ambiguous_fields.append("cross_asset_id")
    if (
        not 1 <= len(cell_matches) <= 400
        or len(cell_matches) != len(cell_ids)
    ):
        ambiguous_fields.append("cell_ids")
    if ambiguous_fields:
        return _clarification(
            "请在同一条请求中逐字提供且只提供一个完整 Cross candidate asset ID"
            "（candidate-asset- 后接 32 位小写十六进制），以及 1 到 400 个"
            "互不重复的完整 cell ID（cross-cell- 后接 32 位小写十六进制）；"
            "不能使用‘刚才那些’‘这些格子’等代词。",
            code="cross_matrix_cell_explicit_ids_required",
            fields=tuple(ambiguous_fields),
        )

    ungrounded: list[str] = []
    if asset_ids != {inputs["cross_asset_id"]}:
        ungrounded.append("cross_asset_id")
    if cell_ids != set(inputs["cell_ids"]):
        ungrounded.append("cell_ids")
    if ungrounded:
        return _clarification(
            "模型草案中的 Cross asset 或 cell ID 与用户原话不一致。平台不会"
            "替换、补全、排序选择或猜测 ID。",
            code="cross_matrix_cell_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    selection_reason = inputs.get("selection_reason")
    if bool(explicit_reasons or selection_reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(selection_reason, str)
        or selection_reason != explicit_reasons[0]
    ):
        return _clarification(
            "selection_reason 必须与用户以‘选择理由/理由/原因/说明’显式给出的"
            "唯一理由完全一致；未给理由时模型必须省略。",
            code="cross_matrix_cell_reason_not_grounded",
            fields=("selection_reason",),
        )

    if _cross_matrix_cell_unconsumed_request_text(utterance):
        return _clarification(
            "本轮只接受一次明确的 Cross Matrix 单元格 pointer 选择；请求中还有"
            "无法按该单步契约解释的内容。请把入池、动作、采纳、部署或写回拆成"
            "后续请求。",
            code="cross_matrix_cell_single_step_required",
            fields=("next_action",),
        )
    return result


def _ground_automatic_tree_leaf_materialization(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Fail closed unless one exact full-tree asset and leaf were named.

    This stage creates only an immutable pointer. It cannot rank/select on
    measured outcomes or smuggle a later Pool, action, lifecycle or writeback
    operation into the same confirmation.
    """

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    positive_operation_text = _AUTOMATIC_TREE_LEAF_NEGATED_CLAUSE_RE.sub(
        "",
        utterance,
    )

    if (
        _AUTOMATIC_TREE_LEAF_AMBIGUOUS_SELECTION_RE.search(positive_operation_text)
        is not None
    ):
        return _clarification(
            "请从完整候选树结果中复制一个明确的 leaf ID；不能按“最好”或"
            "“风险最高”等指标描述替你选择叶节点。",
            code="automatic_tree_leaf_selection_ambiguous",
            fields=("leaf_id",),
        )
    if not _automatic_tree_leaf_has_positive_materialization_intent(utterance):
        return _clarification(
            "原话没有明确授权一次正向的叶节点物化；否定式或仅描述 ID 的请求"
            "不会创建 pointer。如需继续，请重新明确说出要物化的完整资产 ID 和"
            "叶节点 ID。",
            code="automatic_tree_leaf_intent_negated",
            fields=("materialization_intent",),
        )
    reason_values = _automatic_tree_leaf_all_reason_values(utterance)
    explicit_reasons = _automatic_tree_leaf_explicit_reasons(utterance)
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_REPLACEMENT_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "一条请求只能给出一个最终 selection_reason；理由内容中不能再次嵌套"
            "理由字段或改为/替换指令。请只保留最终理由后重新确认。",
            code="automatic_tree_leaf_reason_not_grounded",
            fields=("selection_reason",),
        )
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "选择理由也不能包含按指标极值、排名或“最好/最差”替用户选择叶节点"
            "的语义。请从完整候选树结果中复制一个人工明确确认的 leaf ID。",
            code="automatic_tree_leaf_selection_ambiguous",
            fields=("leaf_id", "selection_reason"),
        )
    if any(
        _AUTOMATIC_TREE_LEAF_REASON_FORBIDDEN_OPERATION_RE.search(reason) is not None
        for reason in reason_values
    ):
        return _clarification(
            "selection_reason 只能记录本次人工选择说明，不能藏入随后入池、"
            "业务动作、采纳、部署、投产或写回请求。请把这些操作拆成后续请求。",
            code="automatic_tree_leaf_single_step_required",
            fields=("selection_reason", "next_action"),
        )
    if any(
        not _automatic_tree_leaf_rationale_is_allowed(reason)
        or _AUTOMATIC_TREE_LEAF_RATIONALE_DECISION_SUBJECT_RE.search(reason) is not None
        for reason in explicit_reasons
    ):
        return _clarification(
            "selection_reason 必须是人工/业务/风险/合规/样本评审依据类短说明，"
            "不能包含命中客户、业务动作、策略池或生产操作。请只保留本次人工"
            "选择依据，其他动作另发请求。",
            code="automatic_tree_leaf_reason_not_grounded",
            fields=("selection_reason",),
        )
    if any(
        pattern.search(positive_operation_text) is not None
        for pattern in (
            _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE,
        )
    ):
        return _clarification(
            "本轮只创建叶节点指针；加入 Strategy Pool、设置业务动作、采纳、"
            "部署或把叶 ID 写回数据集必须分别发起后续请求。",
            code="automatic_tree_leaf_single_step_required",
            fields=("next_action",),
        )

    asset_ids = frozenset(
        match.group(0)
        for match in _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.finditer(utterance)
    )
    leaf_ids = frozenset(
        match.group(0) for match in _AUTOMATIC_TREE_LEAF_ID_TOKEN_RE.finditer(utterance)
    )
    ambiguous_fields: list[str] = []
    if len(asset_ids) != 1:
        ambiguous_fields.append("tree_asset_id")
    if len(leaf_ids) != 1:
        ambiguous_fields.append("leaf_id")
    if ambiguous_fields:
        return _clarification(
            "请在同一条请求中逐字提供且只提供一个完整自动树 candidate asset ID"
            "（candidate-asset- 后接 32 位小写十六进制）和一个完整 leaf ID"
            "（leaf- 后接 20 位小写十六进制）；不能使用“刚才那棵树”或"
            "“这个叶子”等代词。",
            code="automatic_tree_leaf_explicit_ids_required",
            fields=tuple(ambiguous_fields),
        )

    ungrounded: list[str] = []
    if asset_ids != {inputs["tree_asset_id"]}:
        ungrounded.append("tree_asset_id")
    if leaf_ids != {inputs["leaf_id"]}:
        ungrounded.append("leaf_id")
    if ungrounded:
        return _clarification(
            "模型草案中的自动树资产或叶节点 ID 与用户原话不一致。请重新复制"
            "完整 tree asset ID 和 leaf ID；平台不会替换、补全或猜测 ID。",
            code="automatic_tree_leaf_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    selection_reason = inputs.get("selection_reason")
    reason_mismatch = bool(explicit_reasons or selection_reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(selection_reason, str)
        or selection_reason != explicit_reasons[0]
    )
    if reason_mismatch:
        return _clarification(
            "selection_reason 必须与用户以“选择理由/理由/原因/说明”"
            "显式给出的唯一理由完全一致；用户未给理由时模型也必须"
            "省略该字段。平台不会改写、补充或推断选择理由。",
            code="automatic_tree_leaf_reason_not_grounded",
            fields=("selection_reason",),
        )

    if _automatic_tree_leaf_unconsumed_request_text(utterance):
        return _clarification(
            "本轮只接受一次明确的叶节点 pointer 物化；请求中还有无法按该"
            "单步契约解释的内容。请把加入规则/策略池、业务动作、采纳、投产或"
            "写回等操作拆成后续请求。",
            code="automatic_tree_leaf_single_step_required",
            fields=("next_action",),
        )
    return result


def _ground_interactive_tree_revision(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one current edit over exact tree, split and threshold controls."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    threshold_action = (
        _INTERACTIVE_TREE_THRESHOLD_ACTION_RE.search(utterance) is not None
    )
    prune_action = _INTERACTIVE_TREE_PRUNE_ACTION_RE.search(utterance) is not None
    if _INTERACTIVE_TREE_AMBIGUOUS_NODE_RE.search(utterance) is not None:
        return _clarification(
            "请从认证树拓扑中明确复制一个当前可见的完整 split node ID；平台不会"
            "按“最好”“风险最高”“不稳定”或代词替你选择节点。",
            code="interactive_tree_revision_node_selection_ambiguous",
            fields=("node_id",),
        )
    if _INTERACTIVE_TREE_THRESHOLD_AMBIGUOUS_RE.search(utterance) is not None:
        fields = (
            ("node_id", "threshold")
            if re.search(
                r"(?:全部节点|所有节点|每个节点|all\s+nodes?|every\s+node)",
                utterance,
                re.IGNORECASE,
            )
            else ("threshold",)
        )
        return _clarification(
            "阈值调整必须点名一个当前可见 split node 并给出一个有限的新阈值；"
            "平台不会按“调好一点”“最佳阈值”“自动优化”或“全部节点”"
            "替用户搜索、推荐或批量修改。",
            code="interactive_tree_revision_threshold_ambiguous",
            fields=fields,
        )
    if (
        _INTERACTIVE_TREE_NEGATED_OR_NONCURRENT_RE.search(utterance) is not None
        or not (prune_action or threshold_action)
    ):
        return _clarification(
            "原话必须是当前、肯定的一次修剪或阈值调整命令；问句、否定、假设、未来或"
            "历史描述不会创建交互树修订。",
            code="interactive_tree_revision_intent_negated",
            fields=("edit_intent",),
        )
    if _INTERACTIVE_TREE_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "树制品、hash、frontier、condition、metrics、数据集与样本绑定由"
            "平台恢复，不能由本次自然语言请求覆盖。",
            code="interactive_tree_revision_platform_controls_forbidden",
            fields=("platform_bindings",),
        )
    if (
        prune_action
        and threshold_action
        or any(
        pattern.search(utterance) is not None
        for pattern in (
            _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE,
        )
        )
        or re.search(
        r"(?:生成报告|出报告|应用整棵树|继续自动分裂|自动继续|"
        r"物化[^，,；;。\n]{0,24}(?:前沿|frontier)|"
        r"(?<![A-Za-z0-9_])(?:generate\s+(?:a\s+)?report|apply\s+tree|"
        r"apply\s+(?:it|the\s+tree)\s+to\s+(?:the\s+)?dataset|"
        r"materiali[sz]e\s+(?:the\s+)?frontier|"
        r"auto[- ]?continue)(?![A-Za-z0-9_]))",
        utterance,
        re.IGNORECASE,
        )
    ):
        return _clarification(
            "本轮只允许一次 prune_subtree 或 adjust_split_threshold；"
            "前沿物化、入池、业务动作、整树应用、继续分裂、报告、采纳、"
            "部署或写回必须拆成后续请求。",
            code="interactive_tree_revision_single_step_required",
            fields=("next_action",),
        )

    source_matches = tuple(
        _INTERACTIVE_TREE_SOURCE_ID_TOKEN_RE.finditer(utterance)
    )
    node_matches = tuple(_INTERACTIVE_TREE_NODE_ID_TOKEN_RE.finditer(utterance))
    source_ids = frozenset(match.group(0) for match in source_matches)
    node_ids = frozenset(match.group(0) for match in node_matches)
    missing_or_ambiguous: list[str] = []
    if len(source_matches) != 1 or len(source_ids) != 1:
        missing_or_ambiguous.append("source_tree_id")
    if len(node_matches) != 1 or len(node_ids) != 1:
        missing_or_ambiguous.append("node_id")
    if missing_or_ambiguous:
        return _clarification(
            "请在同一条命令中逐字提供且只提供一个完整 automatic-tree asset "
            "或 interactive-tree revision ID，以及一个完整 split node ID；"
            "不能使用“刚才那棵树”“那个节点”等代词。",
            code="interactive_tree_revision_explicit_ids_required",
            fields=tuple(missing_or_ambiguous),
        )
    threshold_values = _interactive_tree_threshold_values(utterance)
    expected_operation = (
        "adjust_split_threshold" if threshold_action else "prune_subtree"
    )
    if expected_operation == "adjust_split_threshold" and len(threshold_values) != 1:
        return _clarification(
            "阈值调整必须在同一条命令中明确且只给出一个有限的新 threshold "
            "数值；平台不会从描述、指标或历史树中推断。",
            code="interactive_tree_revision_explicit_threshold_required",
            fields=("threshold",),
        )

    ungrounded: list[str] = []
    if source_ids != {inputs["source_tree_id"]}:
        ungrounded.append("source_tree_id")
    if node_ids != {inputs["node_id"]}:
        ungrounded.append("node_id")
    if inputs["operation"] != expected_operation:
        ungrounded.append("operation")
    if expected_operation == "adjust_split_threshold":
        supplied_threshold = inputs.get("threshold")
        if (
            isinstance(supplied_threshold, bool)
            or not isinstance(supplied_threshold, int | float)
            or float(supplied_threshold) != threshold_values[0]
        ):
            ungrounded.append("threshold")
    elif "threshold" in inputs:
        ungrounded.append("threshold")
    if ungrounded:
        return _clarification(
            "模型草案中的来源树、节点、操作或新阈值与用户原话不一致；"
            "平台不会替换、补全、猜测、优化或改选控制值。",
            code="interactive_tree_revision_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    explicit_reasons = _automatic_tree_leaf_explicit_reasons(utterance)
    supplied_reason = inputs.get("reason")
    if bool(explicit_reasons or supplied_reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(supplied_reason, str)
        or supplied_reason != explicit_reasons[0]
    ):
        return _clarification(
            "reason 只有在用户以“理由/原因/说明/reason”显式标注时才能逐字"
            "抄录；未提供时模型必须省略，平台不会代写。",
            code="interactive_tree_revision_reason_not_grounded",
            fields=("reason",),
        )
    return result


def _interactive_tree_threshold_values(utterance: str) -> tuple[float, ...]:
    values: list[float] = []
    for match in _INTERACTIVE_TREE_THRESHOLD_VALUE_RE.finditer(utterance):
        token = match.group("zh_value") or match.group("en_value")
        try:
            value = float(token)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value):
            values.append(value)
    return tuple(values)


def utterance_targets_interactive_tree_frontier_group_materialization(
    utterance: str,
) -> bool:
    """Recognize an explicit interactive-tree frontier OR-group action."""

    return bool(
        _INTERACTIVE_TREE_FRONTIER_SUBJECT_RE.search(utterance)
        and _INTERACTIVE_TREE_FRONTIER_ACTION_RE.search(utterance)
        and (
            _INTERACTIVE_TREE_FRONTIER_GROUP_SEMANTICS_RE.search(utterance)
            or _INTERACTIVE_TREE_FRONTIER_GROUP_INTENT_RE.search(utterance)
        )
        and (
            _INTERACTIVE_TREE_REVISION_ID_TOKEN_RE.search(utterance)
            or re.search(
                r"(?:交互(?:式)?树|树修订|tree\s+revision).{0,80}"
                r"(?:前沿|frontier)",
                utterance,
                re.IGNORECASE,
            )
        )
    )


def _ground_interactive_tree_frontier_group_materialization(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Require one explicit 2..50-member OR pointer over one exact revision."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _INTERACTIVE_TREE_FRONTIER_GROUP_AMBIGUOUS_SELECTION_RE.search(utterance)
        is not None
        or _AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE.search(utterance) is not None
        or _INTERACTIVE_TREE_FRONTIER_GROUP_SEMANTICS_RE.search(utterance)
        is None
    ):
        return _clarification(
            "请从交互树 revision 的完整 frontier 清单中复制 2 到 50 个"
            "明确 node/leaf ID，并明确它们按 OR 组合；平台不会按全部、最好、"
            "最差、风险或指标排名替你选节点。",
            code="interactive_tree_frontier_group_selection_ambiguous",
            fields=("source_node_ids", "or_semantics"),
        )
    if (
        _INTERACTIVE_TREE_FRONTIER_NEGATED_OR_NONCURRENT_RE.search(utterance)
        is not None
        or _INTERACTIVE_TREE_FRONTIER_ACTION_RE.search(utterance) is None
    ):
        return _clarification(
            "原话必须是当前、肯定的一次交互树前沿 OR 分组物化命令；"
            "问句、否定、假设、历史或未来描述不会创建 group pointer。",
            code="interactive_tree_frontier_group_intent_negated",
            fields=("materialization_intent",),
        )
    if (
        _INTERACTIVE_TREE_FRONTIER_PLATFORM_CONTROL_RE.search(utterance)
        is not None
        or _INTERACTIVE_TREE_FRONTIER_GROUP_PLATFORM_CONTROL_RE.search(utterance)
        is not None
    ):
        return _clarification(
            "selection/group/revision artifact、hash、tree、fragment、condition、"
            "metrics、数据集与 workspace 绑定由平台恢复，不能由自然语言指定"
            "或覆盖。",
            code="interactive_tree_frontier_group_platform_controls_forbidden",
            fields=("platform_bindings",),
        )
    if any(
        pattern.search(utterance) is not None
        for pattern in (
            _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE,
            _SCORECARD_SECOND_OPERATION_RE,
        )
    ):
        return _clarification(
            "本轮只创建一个交互树 frontier OR group pointer；加入 Strategy "
            "Pool、设置业务动作、应用、采纳、部署或写回必须分别发起后续请求。",
            code="interactive_tree_frontier_group_single_step_required",
            fields=("next_action",),
        )

    revision_matches = tuple(
        _INTERACTIVE_TREE_REVISION_ID_TOKEN_RE.finditer(utterance)
    )
    node_matches = tuple(
        _INTERACTIVE_TREE_FRONTIER_NODE_ID_TOKEN_RE.finditer(utterance)
    )
    revision_ids = frozenset(match.group(0) for match in revision_matches)
    observed_node_ids = tuple(match.group(0) for match in node_matches)
    node_ids = frozenset(observed_node_ids)
    missing_or_ambiguous: list[str] = []
    if len(revision_matches) != 1 or len(revision_ids) != 1:
        missing_or_ambiguous.append("revision_id")
    if (
        not 2 <= len(node_matches) <= 50
        or len(node_ids) != len(node_matches)
    ):
        missing_or_ambiguous.append("source_node_ids")
    if missing_or_ambiguous:
        return _clarification(
            "请在同一条命令中逐字提供且只提供一个完整 interactive-tree "
            "revision ID，以及 2 到 50 个互不重复的完整 frontier node/leaf "
            "ID；不能使用代词、截断 ID 或重复 ID。",
            code="interactive_tree_frontier_group_explicit_ids_required",
            fields=tuple(missing_or_ambiguous),
        )

    ungrounded: list[str] = []
    if revision_ids != {inputs["revision_id"]}:
        ungrounded.append("revision_id")
    supplied_node_ids = inputs.get("source_node_ids")
    if (
        not isinstance(supplied_node_ids, list)
        or len(supplied_node_ids) != len(node_ids)
        or frozenset(supplied_node_ids) != node_ids
    ):
        ungrounded.append("source_node_ids")
    if ungrounded:
        return _clarification(
            "模型草案中的 revision 或 frontier node/leaf ID 集合与用户原话"
            "不一致；平台不会替换、补全、猜测、新增或删除节点。成员输入顺序"
            "不具有语义，最终顺序由 revision frontier 规范化。",
            code="interactive_tree_frontier_group_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    explicit_reasons = _automatic_tree_leaf_explicit_reasons(utterance)
    supplied_reason = inputs.get("selection_reason")
    if bool(explicit_reasons or supplied_reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(supplied_reason, str)
        or supplied_reason != explicit_reasons[0]
    ):
        return _clarification(
            "selection_reason 只有在用户以“选择理由/理由/原因/说明/reason”"
            "显式标注时才能逐字抄录；未提供时模型必须省略。",
            code="interactive_tree_frontier_group_reason_not_grounded",
            fields=("selection_reason",),
        )
    return result


def utterance_targets_interactive_tree_frontier_materialization(
    utterance: str,
) -> bool:
    """Recognize only an explicit interactive-tree revision frontier action."""

    return bool(
        not utterance_targets_interactive_tree_frontier_group_materialization(
            utterance
        )
        and
        _INTERACTIVE_TREE_FRONTIER_SUBJECT_RE.search(utterance)
        and _INTERACTIVE_TREE_FRONTIER_ACTION_RE.search(utterance)
        and (
            _INTERACTIVE_TREE_REVISION_ID_TOKEN_RE.search(utterance)
            or re.search(
                r"(?:交互(?:式)?树|树修订|tree\s+revision).{0,80}"
                r"(?:前沿|frontier)",
                utterance,
                re.IGNORECASE,
            )
        )
    )


def _ground_interactive_tree_frontier_materialization(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Require one current singleton pointer over an exact revision frontier."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if (
        _INTERACTIVE_TREE_FRONTIER_AMBIGUOUS_SELECTION_RE.search(utterance)
        is not None
        or _AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE.search(utterance) is not None
    ):
        return _clarification(
            "请从交互树 revision 的完整 frontier 清单中复制一个明确的 node/leaf "
            "ID；平台不会按最好、最差、风险或指标排名替你选择。",
            code="interactive_tree_frontier_selection_ambiguous",
            fields=("source_node_id",),
        )
    if (
        _INTERACTIVE_TREE_FRONTIER_NEGATED_OR_NONCURRENT_RE.search(utterance)
        is not None
        or _INTERACTIVE_TREE_FRONTIER_ACTION_RE.search(utterance) is None
    ):
        return _clarification(
            "原话必须是当前、肯定的一次交互树前沿物化命令；问句、否定、"
            "假设、历史或未来描述不会创建 selection pointer。",
            code="interactive_tree_frontier_intent_negated",
            fields=("materialization_intent",),
        )
    if _INTERACTIVE_TREE_FRONTIER_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "selection/revision artifact、hash、tree、fragment、condition、metrics、"
            "数据集与 workspace 绑定由平台恢复，不能由自然语言指定或覆盖。",
            code="interactive_tree_frontier_platform_controls_forbidden",
            fields=("platform_bindings",),
        )
    if any(
        pattern.search(utterance) is not None
        for pattern in (
            _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE,
            _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE,
        )
    ):
        return _clarification(
            "本轮只创建一个交互树 frontier pointer；加入 Strategy Pool、"
            "设置业务动作、采纳、部署或写回必须分别发起后续请求。",
            code="interactive_tree_frontier_single_step_required",
            fields=("next_action",),
        )

    revision_matches = tuple(
        _INTERACTIVE_TREE_REVISION_ID_TOKEN_RE.finditer(utterance)
    )
    node_matches = tuple(
        _INTERACTIVE_TREE_FRONTIER_NODE_ID_TOKEN_RE.finditer(utterance)
    )
    revision_ids = frozenset(match.group(0) for match in revision_matches)
    node_ids = frozenset(match.group(0) for match in node_matches)
    missing_or_ambiguous: list[str] = []
    if len(revision_matches) != 1 or len(revision_ids) != 1:
        missing_or_ambiguous.append("revision_id")
    if len(node_matches) != 1 or len(node_ids) != 1:
        missing_or_ambiguous.append("source_node_id")
    if missing_or_ambiguous:
        return _clarification(
            "请在同一条命令中逐字提供且只提供一个完整 interactive-tree "
            "revision ID，以及一个完整 frontier node/leaf ID；不能使用"
            "“刚才的修订”“这个前沿节点”等代词。",
            code="interactive_tree_frontier_explicit_ids_required",
            fields=tuple(missing_or_ambiguous),
        )

    ungrounded: list[str] = []
    if revision_ids != {inputs["revision_id"]}:
        ungrounded.append("revision_id")
    if node_ids != {inputs["source_node_id"]}:
        ungrounded.append("source_node_id")
    if ungrounded:
        return _clarification(
            "模型草案中的 revision 或 frontier node/leaf ID 与用户原话不一致；"
            "平台不会替换、补全、猜测或改选节点。",
            code="interactive_tree_frontier_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    explicit_reasons = _automatic_tree_leaf_explicit_reasons(utterance)
    supplied_reason = inputs.get("selection_reason")
    if bool(explicit_reasons or supplied_reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(supplied_reason, str)
        or supplied_reason != explicit_reasons[0]
    ):
        return _clarification(
            "selection_reason 只有在用户以“选择理由/理由/原因/说明/reason”"
            "显式标注时才能逐字抄录；未提供时模型必须省略。",
            code="interactive_tree_frontier_reason_not_grounded",
            fields=("selection_reason",),
        )
    return result


def _utterance_targets_automatic_tree_apply(utterance: str) -> bool:
    """Recognize full-tree dataset writeback without stealing build/leaf turns."""

    return _AUTOMATIC_TREE_APPLY_TARGET_RE.search(utterance) is not None


def _automatic_tree_apply_explicit_columns(
    utterance: str,
) -> tuple[dict[str, frozenset[str]], tuple[tuple[int, int], ...]]:
    values: dict[str, set[str]] = {
        "leaf_id_column": set(),
        "rule_id_column": set(),
    }
    spans: list[tuple[int, int]] = []
    for match in _AUTOMATIC_TREE_APPLY_OUTPUT_COLUMN_RE.finditer(utterance):
        label = match.group("label").casefold()
        field = (
            "leaf_id_column"
            if ("叶" in label or "leaf" in label)
            else "rule_id_column"
        )
        values[field].add(match.group("column"))
        spans.append(match.span())
    for match in _AUTOMATIC_TREE_APPLY_NAMED_OUTPUT_COLUMN_RE.finditer(utterance):
        field = match.group("field").casefold()
        values[field].add(match.group("column"))
        spans.append(match.span())
    return (
        {field: frozenset(columns) for field, columns in values.items()},
        tuple(spans),
    )


def _automatic_tree_apply_has_unlabeled_output_column(
    utterance: str,
    labeled_spans: Sequence[tuple[int, int]],
) -> bool:
    for match in _AUTOMATIC_TREE_APPLY_GENERIC_OUTPUT_COLUMN_RE.finditer(utterance):
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in labeled_spans
        ):
            return True
    return False


def _ground_automatic_tree_apply(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Bind one affirmative command to one exact tree and optional columns."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]

    if (
        _AUTOMATIC_TREE_APPLY_ACTION_RE.search(utterance) is None
        or _AUTOMATIC_TREE_APPLY_NOT_AUTHORIZED_RE.search(utterance) is not None
    ):
        return _clarification(
            "原话没有授权一次立即、肯定的自动树全量写回。否定、问句、"
            "假设、历史或未来描述都不会创建派生数据集；请重新发出单独的"
            "执行命令。",
            code="automatic_tree_apply_intent_not_authorized",
            fields=("apply_intent",),
        )
    if _AUTOMATIC_TREE_APPLY_PLATFORM_CONTROL_RE.search(utterance) is not None:
        return _clarification(
            "自动树 artifact/hash、数据集与 workspace lineage 必须由平台从"
            "当前任务重新校验并绑定，不能接受自然语言指定或覆盖。",
            code="automatic_tree_apply_platform_binding_forbidden",
            fields=("platform_binding",),
        )
    if _AUTOMATIC_TREE_APPLY_FOLLOW_UP_RE.search(utterance) is not None:
        return _clarification(
            "本轮只把一棵完整自动树确定性写入一个不可变派生数据集；入池、"
            "叶节点选择、业务动作、报告、采纳和部署必须拆成后续请求。",
            code="automatic_tree_apply_single_step_required",
            fields=("next_action",),
        )

    asset_mentions = tuple(
        match.group(0)
        for match in _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.finditer(utterance)
    )
    if len(asset_mentions) != 1:
        return _clarification(
            "请在同一条写回命令中逐字提供且只提供一个完整自动树 asset ID"
            "（candidate-asset- 后接 32 位小写十六进制）；不能使用“刚才"
            "那棵树”等代词。",
            code="automatic_tree_apply_explicit_asset_required",
            fields=("tree_asset_id",),
        )
    if asset_mentions[0] != inputs["tree_asset_id"]:
        return _clarification(
            "模型草案中的自动树 asset ID 与用户原话不一致；平台不会替换、"
            "补全或猜测完整 tree asset ID。",
            code="automatic_tree_apply_controls_not_grounded",
            fields=("tree_asset_id",),
        )

    explicit_columns, labeled_spans = _automatic_tree_apply_explicit_columns(
        utterance
    )
    if _automatic_tree_apply_has_unlabeled_output_column(
        utterance,
        labeled_spans,
    ) or any(len(values) > 1 for values in explicit_columns.values()):
        return _clarification(
            "输出列必须明确标注为叶节点列或规则列，且每种角色最多一个最终"
            "列名；仅说“输出列”不能判断要覆盖哪一种结果。",
            code="automatic_tree_apply_output_column_ambiguous",
            fields=("leaf_id_column", "rule_id_column"),
        )

    ungrounded: list[str] = []
    for field, values in explicit_columns.items():
        explicit = next(iter(values)) if values else None
        if inputs.get(field) != explicit:
            ungrounded.append(field)
    if ungrounded:
        return _clarification(
            "模型草案中的叶节点/规则输出列必须与用户显式标注的列名逐字"
            "一致；用户未提供时必须省略并由 Tool 使用受控默认值。",
            code="automatic_tree_apply_controls_not_grounded",
            fields=tuple(ungrounded),
        )

    source_columns = {column.casefold() for column in whitelist}
    collisions = [
        field
        for field in ("leaf_id_column", "rule_id_column")
        if isinstance(inputs.get(field), str)
        and inputs[field].casefold() in source_columns
    ]
    if collisions:
        return _clarification(
            "自动树写回输出列不能覆盖当前样本已有字段，请为叶节点列和规则列"
            "选择新的列名。",
            code="automatic_tree_apply_output_column_conflict",
            fields=tuple(collisions),
        )
    return result


def _ground_automatic_tree_candidate_build(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    """Prove every tree-build control came from the user's original text."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if _AUTOMATIC_TREE_NEGATED_BUILD_RE.search(utterance) is not None:
        return _clarification(
            "原话明确否定了自动树构建，因此本次不会创建或执行 build。"
            "如需建树，请重新给出一条明确的正向构建请求。",
            code="automatic_tree_build_intent_negated",
            fields=("build_intent",),
        )
    platform_control_clarification = _automatic_tree_platform_control_clarification(
        utterance
    )
    if platform_control_clarification is not None:
        return platform_control_clarification
    if _utterance_requests_automatic_tree_follow_up(utterance):
        return _clarification(
            "自动树需要按可审计步骤逐次确认：本次只能单独完成候选树构建。"
            "构建完成后，请查看平台叶子证据并在下一条请求中引用明确的 leaf；"
            "平台不会让 LLM 自动选择“最好叶子”或直接写入 Strategy Pool。",
            code="automatic_tree_build_single_step_required",
            fields=("workflow_step", "leaf_id"),
        )

    column_mentions, ambiguous_columns = _automatic_tree_column_mention_resolution(
        utterance,
        whitelist,
    )
    if ambiguous_columns:
        return _clarification(
            "自动树字段名在原话中存在交叉重叠或大小写歧义，请用分隔符逐个写出"
            "准确列名："
            + "、".join(ambiguous_columns)
            + "。平台不会按白名单顺序猜测。",
            code="automatic_tree_build_column_mention_ambiguous",
            fields=ambiguous_columns,
        )
    column_spans = tuple((start, end) for start, end, _ in column_mentions)
    missing_controls: list[str] = []
    missing_controls.extend(
        feature
        for feature in inputs["features"]
        if not _utterance_supports_automatic_tree_feature(
            utterance,
            feature,
            whitelist=whitelist,
        )
    )
    explicit_features = tuple(
        column
        for column in whitelist
        if _utterance_supports_automatic_tree_feature(
            utterance,
            column,
            whitelist=whitelist,
        )
    )
    missing_controls.extend(
        f"features includes {feature}"
        for feature in explicit_features
        if feature not in inputs["features"]
    )
    for field in (
        "sample_weight_col",
        "loan_amount_col",
        "overdue_amount_col",
    ):
        column = inputs.get(field)
        if isinstance(
            column, str
        ) and not _utterance_supports_automatic_tree_column_role(
            utterance,
            field=field,
            column=column,
            whitelist=whitelist,
        ):
            missing_controls.append(f"{field}={column}")
        explicit_columns = tuple(
            candidate
            for candidate in whitelist
            if _utterance_supports_automatic_tree_column_role(
                utterance,
                field=field,
                column=candidate,
                whitelist=whitelist,
            )
        )
        missing_controls.extend(
            f"{field}={candidate}"
            for candidate in explicit_columns
            if column != candidate
        )

    for feature, direction in inputs.get("directions", {}).items():
        if not _utterance_supports_automatic_tree_direction(
            utterance,
            feature=feature,
            direction=direction,
            column_spans=column_spans,
            whitelist=whitelist,
        ):
            missing_controls.append(f"{feature}={direction}")
    direction_features = tuple(dict.fromkeys((*explicit_features, *inputs["features"])))
    supplied_directions = inputs.get("directions", {})
    for feature in direction_features:
        explicit_directions = tuple(
            direction
            for direction in AUTOMATIC_TREE_DIRECTIONS
            if _utterance_supports_automatic_tree_direction(
                utterance,
                feature=feature,
                direction=direction,
                column_spans=column_spans,
                whitelist=whitelist,
            )
        )
        missing_controls.extend(
            f"directions.{feature}={direction}"
            for direction in explicit_directions
            if supplied_directions.get(feature) != direction
        )
    for field in (
        "max_depth",
        "min_leaf_count",
        "min_weight_fraction_leaf",
        "seed",
    ):
        if field in inputs and not _utterance_supports_automatic_tree_number(
            utterance,
            field=field,
            value=inputs[field],
            column_spans=column_spans,
        ):
            missing_controls.append(f"{field}={inputs[field]}")
        explicit_values = _automatic_tree_number_values(
            utterance,
            field=field,
            column_spans=column_spans,
        )
        supplied_value = inputs.get(field)
        missing_controls.extend(
            f"{field}={_automatic_tree_number_text(field, explicit_value)}"
            for explicit_value in explicit_values
            if supplied_value is None
            or not math.isclose(
                float(supplied_value),
                explicit_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    if not missing_controls:
        return result
    unique_missing = tuple(dict.fromkeys(missing_controls))
    return _clarification(
        "请在原话中明确列出自动树候选的全部特征，以及实际需要覆盖的权重列、"
        "金额列、方向或树参数；当前无法核对："
        + "、".join(unique_missing)
        + "。平台不会采用 LLM 猜测的列、参数、默认值、结果或推荐。",
        code="automatic_tree_build_controls_not_grounded",
        fields=unique_missing,
    )


def _utterance_requests_automatic_tree_follow_up(utterance: str) -> bool:
    follow_up_patterns = (
        _AUTOMATIC_TREE_MULTI_STEP_RE,
        _AUTOMATIC_TREE_BEST_LEAF_RE,
        _AUTOMATIC_TREE_REVERSED_BEST_LEAF_RE,
        _AUTOMATIC_TREE_LEAF_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_POOL_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_LEAF_DECISION_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_HEURISTIC_LEAF_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_NODE_RANK_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_NODE_SELECT_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_NODE_EXTRACT_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_LIFECYCLE_FOLLOW_UP_RE,
        _AUTOMATIC_TREE_LEAF_ID_WRITEBACK_RE,
        _AUTOMATIC_TREE_DECISION_ARTIFACT_RE,
    )
    for clause in _automatic_tree_follow_up_clauses(utterance):
        leaf_matches = tuple(_AUTOMATIC_TREE_LEAF_TOKEN_RE.finditer(clause))
        effect_matches = tuple(_AUTOMATIC_TREE_DECISION_EFFECT_RE.finditer(clause))
        for leaf_match in leaf_matches:
            for effect_match in effect_matches:
                if not _automatic_tree_follow_up_action_is_negated(
                    clause,
                    action_start=effect_match.start(),
                ):
                    return True
        for pattern in follow_up_patterns:
            for match in pattern.finditer(clause):
                anchor = _AUTOMATIC_TREE_FOLLOW_UP_ACTION_ANCHOR_RE.search(
                    clause,
                    match.start(),
                    match.end(),
                )
                action_start = anchor.start() if anchor is not None else match.start()
                if not _automatic_tree_follow_up_action_is_negated(
                    clause,
                    action_start=action_start,
                ):
                    return True
    return False


def _automatic_tree_follow_up_clauses(utterance: str) -> tuple[str, ...]:
    """Split follow-up semantics so negation cannot hide a later positive action."""

    clauses = tuple(
        clause.strip()
        for clause in _AUTOMATIC_TREE_FOLLOW_UP_CLAUSE_BOUNDARY_RE.split(utterance)
        if clause.strip()
    )
    return clauses or (utterance,)


def _automatic_tree_follow_up_action_is_negated(
    clause: str,
    *,
    action_start: int,
) -> bool:
    """Accept negation only when it strictly scopes the leaf follow-up action."""

    negations = tuple(
        match
        for match in _AUTOMATIC_TREE_FOLLOW_UP_NEGATION_RE.finditer(
            clause,
            0,
            action_start,
        )
    )
    if not negations:
        return False
    closest = negations[-1]
    between = clause[closest.end() : action_start]
    return _AUTOMATIC_TREE_NEGATED_FOLLOW_UP_PREFIX_RE.fullmatch(between) is not None


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


def _automatic_tree_column_spans(
    utterance: str,
    whitelist: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (start, end)
                for start, end, _ in _automatic_tree_column_mentions(
                    utterance,
                    whitelist,
                )
            }
        )
    )


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


def _automatic_tree_feature_span_is_negated(
    utterance: str,
    *,
    start: int,
    end: int,
) -> bool:
    """Extend local negation across an explicitly excluded feature list."""

    if _automatic_tree_span_is_negated(utterance, start=start, end=end):
        return True
    segment, left, _ = _automatic_tree_segment(
        utterance,
        start=start,
        end=end,
        separators=("，", ",", "；", ";", "。", "\n"),
    )
    local_start = start - left
    prefix = segment[:local_start]
    scoped = re.search(
        r"(?P<cue>不要(?:使用|选择|选)?|无需(?:使用|选择|选)?|"
        r"不用|不使用|不选|禁止|排除|剔除|去掉|除去|除了|除)"
        r"\s*(?:特征|候选变量|入模变量|自变量)?\s*(?P<body>.*)$",
        prefix,
        re.IGNORECASE,
    )
    if scoped is None:
        return False
    cue = scoped.group("cue")
    following_sentence = utterance[
        end : utterance.find("。", end) if "。" in utterance[end:] else len(utterance)
    ]
    if cue in {"除", "除了"} and re.search(
        r"(?:还|也|另外|再加|并且)", following_sentence
    ):
        # Chinese “除了 A，还用 B” is additive rather than exclusionary.
        return False
    return (
        re.search(
            r"(?:但(?:是)?|而(?:是)?|改为|改用|转而)\s*"
            r"(?:使用|选择|选用|保留|加入)?",
            scoped.group("body"),
            re.IGNORECASE,
        )
        is None
    )


def _automatic_tree_span_overlaps_columns(
    start: int,
    end: int,
    column_spans: Sequence[tuple[int, int]],
) -> bool:
    return any(
        start < column_end and column_start < end
        for column_start, column_end in column_spans
    )


def _utterance_supports_automatic_tree_feature(
    utterance: str,
    feature: str,
    *,
    whitelist: Sequence[str],
) -> bool:
    if any(
        _utterance_supports_automatic_tree_column_role(
            utterance,
            field=field,
            column=feature,
            whitelist=whitelist,
        )
        for field in _AUTOMATIC_TREE_COLUMN_ROLE_LABELS
    ):
        return False
    mentions = tuple(
        (start, end)
        for start, end, column in _automatic_tree_column_mentions(
            utterance,
            whitelist,
        )
        if column == feature
    )
    cue_pattern = re.compile(
        r"特征|候选变量|入模变量|自变量|features?|构建|"
        r"建(?:一棵)?(?:自动)?(?:决策)?树|build|tree",
        re.IGNORECASE,
    )
    blocker_pattern = re.compile(
        "|".join(
            f"(?:{pattern})"
            for pattern in (
                *_AUTOMATIC_TREE_COLUMN_ROLE_LABELS.values(),
                *_AUTOMATIC_TREE_NUMBER_LABELS.values(),
            )
        ),
        re.IGNORECASE,
    )
    for start, end in mentions:
        if _automatic_tree_feature_span_is_negated(
            utterance,
            start=start,
            end=end,
        ):
            continue
        segment, _, _ = _automatic_tree_segment(
            utterance,
            start=start,
            end=end,
            separators=("，", ",", "；", ";", "。", "\n"),
        )
        if cue_pattern.search(segment) is not None:
            return True
        sentence, sentence_left, _ = _automatic_tree_segment(
            utterance,
            start=start,
            end=end,
            separators=("；", ";", "。", "\n"),
        )
        feature_start = start - sentence_left
        feature_end = end - sentence_left
        for cue in cue_pattern.finditer(sentence):
            between = (
                sentence[cue.end() : feature_start]
                if cue.end() <= feature_start
                else sentence[feature_end : cue.start()]
            )
            if blocker_pattern.search(between) is None:
                return True
    return False


def _utterance_supports_automatic_tree_column_role(
    utterance: str,
    *,
    field: str,
    column: str,
    whitelist: Sequence[str],
) -> bool:
    label = _AUTOMATIC_TREE_COLUMN_ROLE_LABELS[field]
    resolved_mentions = tuple(
        (start, end)
        for start, end, resolved_column in _automatic_tree_column_mentions(
            utterance,
            whitelist,
        )
        if resolved_column == column
    )
    if not resolved_mentions:
        return False
    column_pattern = rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])"
    paired = re.compile(
        rf"(?:(?:{label})\s*(?:[:：=]|为|是|使用|用|取|设为)?\s*"
        rf"{column_pattern}|"
        rf"{column_pattern}\s*(?:作为|是|为|用作|设为)\s*(?:{label}))",
        re.IGNORECASE,
    )
    for match in paired.finditer(utterance):
        if not any(
            match.start() <= start and end <= match.end()
            for start, end in resolved_mentions
        ):
            continue
        if not _automatic_tree_span_is_negated(
            utterance,
            start=match.start(),
            end=match.end(),
        ) and not _automatic_tree_value_is_replaced(utterance, end=match.end()):
            return True
    replacement = re.compile(
        rf"(?:{label})\s*(?:[:：=]|为|是|使用|用|取|设为)?\s*"
        r"(?:从|由)?\s*[^\s，,；;。]+\s*"
        r"(?:改为|改成|调整为|替换为|而非|不是而是)\s*"
        rf"{column_pattern}",
        re.IGNORECASE,
    )
    return any(
        any(
            match.start() <= start and end <= match.end()
            for start, end in resolved_mentions
        )
        and not _automatic_tree_span_is_negated(
            utterance,
            start=match.start(),
            end=match.end(),
        )
        for match in replacement.finditer(utterance)
    )


def _automatic_tree_value_is_replaced(utterance: str, *, end: int) -> bool:
    return (
        re.match(
            r"\s*(?:改为|改成|调整为|替换为|而非|不是而是)",
            utterance[end:],
        )
        is not None
    )


def _utterance_supports_automatic_tree_direction(
    utterance: str,
    *,
    feature: str,
    direction: str,
    column_spans: Sequence[tuple[int, int]],
    whitelist: Sequence[str],
) -> bool:
    feature_mentions = tuple(
        (start, end)
        for start, end, column in _automatic_tree_column_mentions(
            utterance,
            whitelist,
        )
        if column == feature
    )
    for feature_start, feature_end in feature_mentions:
        segment, left, _ = _automatic_tree_segment(
            utterance,
            start=feature_start,
            end=feature_end,
            separators=("，", ",", "、", "；", ";", "。", "\n"),
        )
        feature_center = (feature_start + feature_end) / 2 - left
        candidates: list[tuple[float, str]] = []
        for candidate_direction, pattern in _AUTOMATIC_TREE_DIRECTION_GROUNDING.items():
            for direction_match in re.finditer(pattern, segment, re.IGNORECASE):
                absolute_start = left + direction_match.start()
                absolute_end = left + direction_match.end()
                if _automatic_tree_span_overlaps_columns(
                    absolute_start,
                    absolute_end,
                    column_spans,
                ) or _automatic_tree_span_is_negated(
                    utterance,
                    start=absolute_start,
                    end=absolute_end,
                ):
                    continue
                replacement = segment[
                    direction_match.end() : direction_match.end() + 16
                ]
                if re.match(r"\s*(?:改为|改成|调整为|而非|不是而是)", replacement):
                    continue
                direction_center = (direction_match.start() + direction_match.end()) / 2
                candidates.append(
                    (abs(direction_center - feature_center), candidate_direction)
                )
        if not candidates:
            continue
        nearest_distance = min(distance for distance, _ in candidates)
        nearest = {
            candidate_direction
            for distance, candidate_direction in candidates
            if math.isclose(
                distance,
                nearest_distance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        }
        if nearest == {direction}:
            return True
    return False


def _utterance_supports_automatic_tree_number(
    utterance: str,
    *,
    field: str,
    value: object,
    column_spans: Sequence[tuple[int, int]],
) -> bool:
    expected = float(value)
    return any(
        math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)
        for observed in _automatic_tree_number_values(
            utterance,
            field=field,
            column_spans=column_spans,
        )
    )


def _automatic_tree_number_values(
    utterance: str,
    *,
    field: str,
    column_spans: Sequence[tuple[int, int]],
) -> tuple[float, ...]:
    label = _AUTOMATIC_TREE_NUMBER_LABELS[field]
    expression = re.compile(
        rf"(?:{label})\s*(?:[:：=]|为|设为|设置为|设成|设置成)?\s*"
        r"(?P<value>百分之\s*[0-9]+(?:\.[0-9]+)?|"
        r"[0-9]+(?:\.[0-9]+)?\s*%)?"
        r"(?P<number>[0-9]+(?:\.[0-9]+)?)?",
        re.IGNORECASE,
    )
    observed_values: list[float] = []
    for match in expression.finditer(utterance):
        if _automatic_tree_span_overlaps_columns(
            match.start(),
            match.end(),
            column_spans,
        ) or _automatic_tree_span_is_negated(
            utterance,
            start=match.start(),
            end=match.end(),
        ):
            continue
        token = match.group("value") or match.group("number")
        if token is None:
            continue
        replacement = re.match(
            r"\s*(?:改为|改成|调整为|替换为|而非|不是而是)\s*"
            r"(?P<value>百分之\s*[0-9]+(?:\.[0-9]+)?|"
            r"[0-9]+(?:\.[0-9]+)?\s*%|[0-9]+(?:\.[0-9]+)?)",
            utterance[match.end() :],
        )
        if replacement is not None:
            replacement_token = replacement.group("value")
            replacement_value = _automatic_tree_number_token_value(
                field,
                replacement_token,
            )
            if (
                replacement_value is not None
                and replacement_value not in observed_values
            ):
                observed_values.append(replacement_value)
            continue
        observed = _automatic_tree_number_token_value(field, token)
        if observed is None:
            continue
        if observed not in observed_values:
            observed_values.append(observed)
    return tuple(observed_values)


def _automatic_tree_number_token_value(field: str, token: str) -> float | None:
    if field == "min_weight_fraction_leaf":
        return _ratio_token_value(token)
    if "百分之" in token or "%" in token:
        return None
    return float(token)


def _automatic_tree_number_text(field: str, value: float) -> str:
    if field in {"max_depth", "min_leaf_count", "seed"}:
        return str(int(value))
    return format(value, ".15g")


def _pool_clause_prefix(
    utterance: str,
    *,
    start: int,
) -> str:
    left = max(
        utterance.rfind(separator, 0, start)
        for separator in ("，", ",", "；", ";", "。", ".", "\n")
    )
    return utterance[left + 1 : start]


def _pool_operation_is_negated(utterance: str, *, start: int) -> bool:
    prefix = _pool_clause_prefix(utterance, start=start)
    negations = tuple(
        re.finditer(
            r"(?:不要|不用|不再|无需|无须|不需要|不能|不可|不允许|不想|"
            r"不打算|暂不|先不|别|请勿|切勿|勿|禁止|严禁|不得|拒绝|取消|"
            r"撤销|停止|放弃|暂缓|不|未|无|没)|"
            r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cannot|can't|won't|"
            r"must\s+not|mustn't|should\s+not|shouldn't|not|without|cancel|"
            r"stop|avoid|refrain)(?![A-Za-z0-9_])",
            prefix,
            re.IGNORECASE,
        )
    )
    if not negations:
        return False
    negation = negations[-1]
    return (
        re.search(
            r"(?:但(?:是)?|而(?:是)?|改为|改成|转而)|"
            r"(?<![A-Za-z0-9_])(?:but|instead|rather\s+than)(?![A-Za-z0-9_])",
            prefix[negation.end() :],
            re.IGNORECASE,
        )
        is None
    )


def _pool_add_intent_state(utterance: str) -> tuple[bool, bool]:
    reason_spans = _pool_add_reason_spans(utterance)
    matches = tuple(
        match
        for match in _POOL_ADD_INTENT_RE.finditer(utterance)
        if not any(left <= match.start() < right for left, right in reason_spans)
    )
    states = tuple(
        not _pool_operation_is_negated(utterance, start=match.start())
        for match in matches
    )
    return (
        bool(matches),
        bool(matches)
        and any(states)
        and all(states)
        and _POOL_ADD_HYPOTHETICAL_RE.search(utterance) is None
        and _POOL_ADD_POSTPOSED_CANCELLATION_RE.search(utterance) is None,
    )


def _pool_mutation_has_positive_intent(
    utterance: str,
    workflow: str,
    inputs: Mapping[str, Any],
) -> bool:
    pattern = _POOL_MUTATION_INTENT_PATTERNS.get(workflow)
    if pattern is None:
        return False
    reason_spans = _pool_add_reason_spans(utterance)
    matches = tuple(
        match
        for match in pattern.finditer(utterance)
        if not any(left <= match.start() < right for left, right in reason_spans)
    )
    states = tuple(
        not _pool_operation_is_negated(utterance, start=match.start())
        for match in matches
    )
    return (
        len(matches) == 1
        and any(states)
        and all(states)
        and _POOL_MUTATION_NONCOMMAND_RE.search(utterance) is None
        and not _pool_mutation_unconsumed_text(
            utterance,
            workflow=workflow,
            inputs=inputs,
            intent_match=matches[0] if len(matches) == 1 else None,
        )
    )


def _pool_source_prefix_count(utterance: str) -> int:
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFKC", utterance)
        if unicodedata.category(character) not in {"Cf", "Mn", "Me"}
    ).translate(
        _POOL_SOURCE_CONFUSABLE_TRANSLATION
        | str.maketrans(
            {
                "а": "a",
                "А": "A",
                "е": "e",
                "Е": "E",
                "о": "o",
                "О": "O",
                "с": "c",
                "С": "C",
                "х": "x",
                "Х": "X",
            }
        )
    )
    return sum(1 for _ in _POOL_SOURCE_PREFIX_RE.finditer(normalized))


_POOL_COMMAND_GLUE_RE = re.compile(
    r"(?:请你|麻烦你|麻烦|请|帮我|帮忙|替我|给我|我要|我想要|我希望|"
    r"现在|立即|直接|本次|这次|当前|先|就|把|将|从|在|到|至|"
    r"这个|这条|该|上述|以下|选择结果|选中结果|候选资产|候选规则|"
    r"候选|资产|叶节点|叶子|结果|规则池|策略池|规则|策略|池|条目|"
    r"一条|一个|中|里|内|的|动作|按|完整|全部|所有|顺序|依次|"
    r"和|及|与)|"
    r"(?<![A-Za-z0-9_])(?:please|kindly|i\s+want\s+to|"
    r"i\s+would\s+like\s+to|help\s+me|for\s+me|now|immediately|"
    r"directly|this|that|the|selected|selection|candidate|asset|rule|"
    r"leaf|result|from|in|inside|of|action|complete|full|order|all|"
    r"strategy|pool|entry|and|to)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _pool_strip_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    characters = list(text)
    for left, right in spans:
        for index in range(max(left, 0), min(right, len(characters))):
            characters[index] = " "
    return "".join(characters)


def _pool_command_residual(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _POOL_COMMAND_GLUE_RE.sub(" ", text)
    return re.sub(r"[\s，,；;。.!?？！:：=、'\"“”‘’（）()【】\[\]{}]+", "", text)


def _pool_mutation_unconsumed_text(
    utterance: str,
    *,
    workflow: str,
    inputs: Mapping[str, Any],
    intent_match: re.Match[str] | None,
) -> str:
    if intent_match is None:
        return utterance
    spans: list[tuple[int, int]] = [intent_match.span()]
    identifiers: list[str] = []
    if workflow in {"strategy_pool_remove_entry", "strategy_pool_set_action"}:
        for field in ("rule_id", "entry_id"):
            value = inputs.get(field)
            if isinstance(value, str):
                identifiers.append(value)
    elif workflow == "strategy_pool_reorder":
        ordered_ids = inputs.get("ordered_ids")
        if isinstance(ordered_ids, Sequence) and not isinstance(
            ordered_ids, str | bytes | bytearray
        ):
            identifiers.extend(value for value in ordered_ids if isinstance(value, str))
    for identifier in identifiers:
        spans.extend(match.span() for match in re.finditer(re.escape(identifier), utterance))
    strategy_type = str(inputs.get("strategy_type") or "")
    strategy_type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if strategy_type_pattern is not None:
        spans.extend(
            match.span() for match in strategy_type_pattern.finditer(utterance)
        )
    if workflow == "strategy_pool_set_action":
        action = inputs.get("action")
        action_type = str(action.get("type") or "") if isinstance(action, Mapping) else ""
        action_pattern = _POOL_ACTION_GROUNDING.get(action_type)
        if action_pattern is not None:
            spans.extend(match.span() for match in action_pattern.finditer(utterance))
    spans.extend(_pool_add_reason_spans(utterance))
    reason = inputs.get("reason")
    if isinstance(reason, str) and reason:
        spans.extend(match.span() for match in re.finditer(re.escape(reason), utterance))
    return _pool_command_residual(_pool_strip_spans(utterance, spans))


def _pool_add_unconsumed_text(utterance: str) -> str:
    spans: list[tuple[int, int]] = []
    spans.extend(match.span() for match in _POOL_ADD_INTENT_RE.finditer(utterance))
    spans.extend(
        match.span()
        for pattern in (
            _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE,
            _AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE,
            _INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE,
            _INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE,
            _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE,
            _SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE,
        )
        for match in pattern.finditer(utterance)
    )
    for pattern in (
        _POOL_ADD_STRATEGY_TYPE_LABEL_RE,
        _POOL_ADD_DEFAULT_ACTION_LABEL_RE,
        _POOL_ADD_HIT_ACTION_LABEL_RE,
        _POOL_ADD_DEFAULT_REASON_CODE_LABEL_RE,
        _POOL_ADD_HIT_REASON_CODE_LABEL_RE,
        _POOL_ADD_DEFAULT_OUTPUT_VALUE_LABEL_RE,
        _POOL_ADD_HIT_OUTPUT_VALUE_LABEL_RE,
        _POOL_ADD_PLACEMENT_MODE_LABEL_RE,
    ):
        spans.extend(
            (match.start(), _pool_add_clause_end(utterance, start=match.end()))
            for match in pattern.finditer(utterance)
        )
    for pattern in (
        _POOL_ADD_BEFORE_SELECTED_MEMBERS_RE,
        _POOL_ADD_REPLACE_SELECTED_MEMBERS_RE,
        _POOL_ADD_BEFORE_SELECTED_MEMBERS_EXPLANATION_RE,
    ):
        spans.extend(match.span() for match in pattern.finditer(utterance))
    spans.extend(_pool_add_negated_follow_up_spans(utterance))
    spans.extend(_pool_add_reason_spans(utterance))
    return _pool_command_residual(_pool_strip_spans(utterance, spans))


def _pool_add_negated_follow_up_spans(
    utterance: str,
) -> tuple[tuple[int, int], ...]:
    clause_spans: set[tuple[int, int]] = set()
    for pattern in (
        _POOL_ADD_LIFECYCLE_RE,
        _POOL_ADD_SECOND_OPERATION_RE,
        _POOL_PARTIAL_REORDER_RE,
        _POOL_HEURISTIC_REORDER_RE,
    ):
        for match in pattern.finditer(utterance):
            negated = (
                _pool_lifecycle_operation_is_negated(
                    utterance,
                    start=match.start(),
                )
                if pattern is _POOL_ADD_LIFECYCLE_RE
                else _pool_operation_is_negated(utterance, start=match.start())
            )
            if not negated:
                continue
            left = max(
                utterance.rfind(separator, 0, match.start())
                for separator in ("，", ",", "；", ";", "。", ".", "\n")
            )
            right = _pool_add_clause_end(utterance, start=match.end())
            if _POOL_ADD_INTENT_RE.search(utterance[left + 1 : right]) is None:
                clause_spans.add((left + 1, right))
    return tuple(sorted(clause_spans))


def _pool_lifecycle_operation_is_negated(utterance: str, *, start: int) -> bool:
    prefix = _pool_clause_prefix(utterance, start=start)
    return (
        re.search(
            r"(?:不要|不用|不再|无需|无须|不需要|不能|不可|不允许|暂不|"
            r"先不|别|请勿|切勿|勿|禁止|严禁|不得|不|未|无|没(?:有)?)"
            r"\s*(?:再|进行|立即|直接)?\s*"
            r"(?:(?:采纳|采用|部署|上线|投产|投入生产|上生产|投用|"
            r"发布到?生产|发布到?线上|推到?线上|推生产|正式运行|落地执行|"
            r"立即执行|执行它?|投入使用|开始使用|启用|生效|激活)"
            r"\s*(?:或|和|及|以及|、)?\s*)*$|"
            r"(?<![A-Za-z0-9_])(?:do\s+not|don't|never|cannot|can't|won't|"
            r"must\s+not|mustn't|should\s+not|shouldn't|not|without)\s+"
            r"(?:(?:adopt(?:s|ed|ing)?|deploy(?:s|ed|ing)?|"
            r"promot(?:e|es|ed|ing)|activat(?:e|es|ed|ing)|"
            r"enabl(?:e|es|ed|ing)|ship(?:s|ped|ping)?|"
            r"push(?:es|ed|ing)?|releas(?:e|es|ed|ing)|"
            r"publish(?:es|ed|ing)?|launch(?:es|ed|ing)?|"
            r"productioniz(?:e|es|ed|ing)|execut(?:e|es|ed|ing)|run(?:s|ning)?|"
            r"use(?:s|d|ing)?[^;.!?\n]{0,32}(?:in|on)[-\s]+prod(?:uction)?|"
            r"put[^;.!?\n]{0,32}into[-\s]+prod(?:uction)?|"
            r"enter(?:s|ed|ing)?[-\s]+prod(?:uction)?|"
            r"take[^;.!?\n]{0,20}live|"
            r"(?:go(?:es|ing)?|went)[-\s]+live|"
            r"roll(?:s|ed|ing)?[-\s]+out)\s*(?:or|and)?\s*)*$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _pool_add_has_positive_lifecycle_follow_up(utterance: str) -> bool:
    positive_lifecycle = any(
        not _pool_lifecycle_operation_is_negated(utterance, start=match.start())
        for match in _POOL_ADD_LIFECYCLE_RE.finditer(utterance)
    )
    positive_second_operation = any(
        not _pool_operation_is_negated(utterance, start=match.start())
        for pattern in (
            _POOL_ADD_SECOND_OPERATION_RE,
            _POOL_PARTIAL_REORDER_RE,
            _POOL_HEURISTIC_REORDER_RE,
        )
        for match in pattern.finditer(utterance)
    )
    return positive_lifecycle or positive_second_operation


def _pool_add_strategy_types(utterance: str) -> tuple[frozenset[str], bool]:
    reason_spans = _pool_add_reason_spans(utterance)
    add_target_matches = tuple(
        match
        for match in _POOL_ADD_INTENT_RE.finditer(utterance)
        if not _pool_operation_is_negated(utterance, start=match.start())
        and not any(left <= match.start() < right for left, right in reason_spans)
    )
    observed: set[str] = set()
    add_targets_valid = True
    for match in add_target_matches:
        add_target = match.group(0)
        if _pool_add_body_is_negated(add_target):
            add_targets_valid = False
            continue
        observed.update(
            strategy_type
            for strategy_type, pattern in _POOL_ADD_STRATEGY_TYPE_NOUN_PATTERNS.items()
            if pattern.search(add_target) is not None
        )
    label_bodies = _pool_add_label_bodies(
        utterance,
        _POOL_ADD_STRATEGY_TYPE_LABEL_RE,
    )
    for label_value in label_bodies:
        observed.update(
            strategy_type
            for strategy_type, pattern in _POOL_STRATEGY_TYPE_VALUE_GROUNDING.items()
            if pattern.search(label_value) is not None
        )
    raw_label_count = _pool_add_label_match_count(
        utterance,
        _POOL_ADD_STRATEGY_TYPE_LABEL_RE,
    )
    return (
        frozenset(observed),
        add_targets_valid
        and raw_label_count == len(label_bodies)
        and raw_label_count <= 1,
    )


def _pool_add_reason_value_spans(utterance: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    for match in _POOL_ADD_REASON_LABEL_RE.finditer(utterance):
        group = "zh" if match.group("zh") is not None else "en"
        spans.append(match.span(group))
    return tuple(spans)


def _pool_add_reason_spans(utterance: str) -> tuple[tuple[int, int], ...]:
    return tuple(
        match.span() for match in _POOL_ADD_REASON_LABEL_RE.finditer(utterance)
    )


def _pool_add_label_match_count(utterance: str, pattern: re.Pattern[str]) -> int:
    reason_spans = _pool_add_reason_spans(utterance)
    return sum(
        1
        for match in pattern.finditer(utterance)
        if not any(left <= match.start() < right for left, right in reason_spans)
    )


def _pool_add_clause_end(utterance: str, *, start: int) -> int:
    """Find the next top-level clause boundary without splitting data values."""

    opening = {"[": "]", "{": "}", "(": ")", "（": "）", "【": "】"}
    closing = frozenset(opening.values())
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for index in range(start, len(utterance)):
        char = utterance[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char in opening:
            stack.append(opening[char])
            continue
        if char in closing:
            if stack and stack[-1] == char:
                stack.pop()
            continue
        if stack:
            continue
        if char in {"；", ";", "。", "\n", "?", "？", "!", "！", "，"}:
            return index
        if char == ",":
            before = utterance[index - 1] if index > 0 else ""
            after = utterance[index + 1] if index + 1 < len(utterance) else ""
            if before.isdigit() and after.isdigit():
                continue
            return index
        if char == ".":
            before = utterance[index - 1] if index > 0 else ""
            after = utterance[index + 1] if index + 1 < len(utterance) else ""
            if before.isdigit() and after.isdigit():
                continue
            if after and not after.isspace() and after not in "，,；;。.!?？！":
                continue
            return index
    return len(utterance)


def _pool_add_authorized_clause_spans(
    utterance: str,
) -> tuple[tuple[int, int], ...]:
    """Return positive add clauses; IDs elsewhere cannot authorize a mutation."""

    reason_spans = _pool_add_reason_value_spans(utterance)
    spans: list[tuple[int, int]] = []
    for match in _POOL_ADD_INTENT_RE.finditer(utterance):
        if any(left <= match.start() < right for left, right in reason_spans):
            continue
        if _pool_operation_is_negated(utterance, start=match.start()):
            continue
        left = max(
            utterance.rfind(separator, 0, match.start())
            for separator in ("，", ",", "；", ";", "。", ".", "\n")
        )
        prefix = utterance[left + 1 : match.start()]
        contrasts = tuple(
            re.finditer(
                r"(?:但(?:是)?|不过|而是|转而)|"
                r"(?<![A-Za-z0-9_])(?:but|instead|rather\s+than)"
                r"(?![A-Za-z0-9_])",
                prefix,
                re.IGNORECASE,
            )
        )
        if contrasts:
            left += contrasts[-1].end()
        right = _pool_add_clause_end(utterance, start=match.end())
        spans.append((left + 1, right))
    return tuple(dict.fromkeys(spans))


def _pool_add_label_is_negated(utterance: str, *, start: int) -> bool:
    prefix = _pool_clause_prefix(utterance, start=start)
    return (
        re.search(
            r"(?:不要|不用|不应|不能|不可|并非|不是|非|未|别|禁止|取消)"
            r"\s*(?:使用|设置|选择|采用)?\s*(?:这|该|一个|the)?\s*$|"
            r"(?<![A-Za-z0-9_])(?:not|non|never)[-\s]*$|"
            r"(?<![A-Za-z0-9_])(?:do\s+not|don't|cannot|can't|must\s+not|"
            r"should\s+not)\s+(?:use|set|choose)\s*$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _pool_add_body_is_negated(body: str) -> bool:
    return (
        re.search(
            r"(?:^|\s|但|但是|却|而|,|，)(?:请\s*)?"
            r"(?:不要|不用|不应|不能|不可|并非|并不是|绝非|绝不是|不是|"
            r"非|未|别|禁止|取消|"
            r"排除|剔除|忽略|除外|不包含|不包括)|"
            r"(?:非|排除|剔除|忽略|除外|不包含|不包括)"
            r"(?=审批|准入|拒绝|额度|授信|定价|利率|分群|分层|"
            r"approval|reject|review|limit|pricing|segment)|"
            r"(?:除了|除开|不含)\s*(?:审批|准入|拒绝|额度|授信|定价|"
            r"利率|分群|分层|approval|reject|review|limit|pricing|segment)|"
            r"(?:审批|准入|拒绝|额度|授信|定价|利率|分群|分层|"
            r"approval|reject|review|limit|pricing|segment)"
            r"[^；;。\n]{0,24}(?:以外|之外|除外|排除|剔除|忽略|不包含|不包括)|"
            r"(?:^|\s|but\s+)(?<![A-Za-z0-9_])(?:do\s+not|don't|cannot|can't|"
            r"must\s+not|should\s+not|not|never|without|avoid|exclude(?:d|s|ing)?|"
            r"except(?:ed|ing)?|omit(?:ted|ting)?|ignor(?:e|ed|ing))"
            r"(?![A-Za-z0-9_])|"
            r"(?<![A-Za-z0-9_])(?:approval|reject|review|limit|pricing|segment)"
            r"\s+(?:is\s+)?(?:excluded|excepted|omitted|ignored)"
            r"(?![A-Za-z0-9_])|"
            r"(?<![A-Za-z0-9_])anything\s+but\s+"
            r"(?:approval|reject|review|limit|pricing|segment)"
            r"(?![A-Za-z0-9_])|"
            r"(?<![A-Za-z0-9_])(?:anything\s+)?other\s+than\s+"
            r"(?:approval|reject|review|limit|pricing|segment)"
            r"(?![A-Za-z0-9_])",
            body,
            re.IGNORECASE,
        )
        is not None
    )


def _pool_add_label_bodies(utterance: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    bodies: list[str] = []
    reason_spans = _pool_add_reason_value_spans(utterance)
    matches: list[re.Match[str]] = []
    for match in pattern.finditer(utterance):
        matches.append(match)
        if len(matches) > _POOL_MAX_CONTROL_LABEL_MATCHES:
            return ()
    for match in matches:
        if any(left <= match.start() < right for left, right in reason_spans):
            continue
        if _pool_add_label_is_negated(utterance, start=match.start()):
            continue
        right = _pool_add_clause_end(utterance, start=match.end())
        body = utterance[match.end() : right]
        body = re.sub(r"^\s*(?:[:：=]|是|为)?\s*", "", body)
        body = body.strip()
        if body and not _pool_add_body_is_negated(body):
            bodies.append(body)
    return tuple(bodies)


def _pool_add_labeled_action_types(
    utterance: str,
    *,
    pattern: re.Pattern[str],
) -> tuple[frozenset[str], ...]:
    return tuple(
        frozenset(
            action_type
            for action_type, grounding in _POOL_ACTION_GROUNDING.items()
            if grounding.search(body) is not None
        )
        for body in _pool_add_label_bodies(utterance, pattern)
    )


def _pool_add_action_body_residual(
    body: str,
    action: Mapping[str, Any],
) -> str:
    action_type = str(action.get("type") or "")
    pattern = _POOL_ACTION_GROUNDING.get(action_type)
    spans: list[tuple[int, int]] = []
    if pattern is not None:
        spans.extend(match.span() for match in pattern.finditer(body))
    if action_type in {"limit", "pricing", "segment"}:
        value = action.get("value")
        candidates = {str(value)}
        try:
            candidates.add(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError):
            pass
        for candidate in candidates:
            spans.extend(
                match.span()
                for match in re.finditer(re.escape(candidate), body, re.IGNORECASE)
            )
    return _pool_command_residual(_pool_strip_spans(body, spans))


def _pool_add_strategy_type_body_residual(body: str, strategy_type: str) -> str:
    pattern = _POOL_STRATEGY_TYPE_VALUE_GROUNDING.get(strategy_type)
    spans = tuple(match.span() for match in pattern.finditer(body)) if pattern else ()
    return _pool_command_residual(_pool_strip_spans(body, spans))


def _pool_reason_has_active_language(reason: str) -> bool:
    if _POOL_REASON_CANCELLATION_RE.search(reason) is not None:
        return True
    return any(
        pattern.search(reason) is not None
        for pattern in (
            _POOL_ADD_INTENT_RE,
            _POOL_ADD_LIFECYCLE_RE,
            _POOL_ADD_SECOND_OPERATION_RE,
            _POOL_PARTIAL_REORDER_RE,
            _POOL_HEURISTIC_REORDER_RE,
            *_POOL_MUTATION_INTENT_PATTERNS.values(),
        )
    )


def _pool_add_parse_complete_value(body: str) -> object:
    text = body.strip()
    if not text or len(text) > _POOL_MAX_CONTROL_VALUE_CHARS:
        return _POOL_UNPARSEABLE_VALUE
    try:
        return json.loads(
            text,
            object_pairs_hook=_pool_add_unique_json_object,
            parse_constant=_pool_add_reject_json_constant,
            parse_float=Decimal,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        pass
    if text.startswith(("[", "{")):
        return _POOL_UNPARSEABLE_VALUE
    if text.casefold() in {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }:
        return _POOL_UNPARSEABLE_VALUE
    number = re.fullmatch(
        r"(?P<number>[-+]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
        r"(?:\.[0-9]+)?)(?P<percent>\s*%)?",
        text,
    )
    if number is not None:
        token = number.group("number").replace(",", "")
        unsigned = token.lstrip("+-")
        integer_part = unsigned.split(".", 1)[0]
        if len(integer_part) > 1 and integer_part.startswith("0"):
            return _POOL_UNPARSEABLE_VALUE
        try:
            value: object = Decimal(token) if "." in token else int(token)
        except (InvalidOperation, ValueError):
            return _POOL_UNPARSEABLE_VALUE
        return value / 100 if number.group("percent") else value
    if text[0] in {"'", '"'} or text[-1] in {"'", '"'}:
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            return text[1:-1]
        return _POOL_UNPARSEABLE_VALUE
    return unicodedata.normalize("NFC", text)


def _pool_add_unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _pool_add_reject_json_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON constant: {token}")


def _pool_add_values_equal(observed: object, expected: object) -> bool:
    try:
        return _pool_add_comparison_value(observed) == _pool_add_comparison_value(
            expected
        )
    except (InvalidOperation, RecursionError, TypeError, ValueError):
        return False


def _pool_add_comparison_value(value: object) -> object:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return (type(value).__name__, value)
    if isinstance(value, int | float | Decimal):
        numeric = value if isinstance(value, Decimal) else Decimal(str(value))
        if not numeric.is_finite():
            raise ValueError("non-finite numeric value")
        return ("number", numeric)
    if isinstance(value, list):
        return ("array", tuple(_pool_add_comparison_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                (key, _pool_add_comparison_value(item))
                for key, item in sorted(value.items())
            ),
        )
    raise TypeError("unsupported comparison value")


def _pool_add_body_matches_action_value(
    body: str,
    *,
    action_type: str,
    expected: object,
) -> bool:
    grounding = _POOL_ACTION_GROUNDING.get(action_type)
    if grounding is None:
        return False
    match = grounding.search(body)
    if match is None:
        return False
    value_text = re.sub(
        r"^\s*(?:[:：=]|是|为)?\s*",
        "",
        body[match.end() :],
    )
    return _pool_add_values_equal(
        _pool_add_parse_complete_value(value_text),
        expected,
    )


def _pool_add_body_matches_complete_value(body: str, expected: object) -> bool:
    return _pool_add_values_equal(
        _pool_add_parse_complete_value(body),
        expected,
    )


def _pool_add_action_payload_controls(
    utterance: str,
    action: Mapping[str, Any],
    *,
    action_bodies: Sequence[str],
    reason_code_label: re.Pattern[str],
    output_value_label: re.Pattern[str],
) -> tuple[str, ...]:
    missing: list[str] = []
    reason_code_bodies = _pool_add_label_bodies(utterance, reason_code_label)
    reason_code_label_count = _pool_add_label_match_count(
        utterance,
        reason_code_label,
    )
    reason_code = action.get("reason_code")
    if bool(reason_code_label_count or reason_code is not None) and (
        reason_code_label_count != len(reason_code_bodies)
        or reason_code_label_count != 1
        or len(reason_code_bodies) != 1
        or not isinstance(reason_code, str)
        or reason_code_bodies[0] != reason_code
    ):
        missing.append("reason_code")
    action_type = str(action.get("type") or "")
    if action_type in {"limit", "pricing", "segment"} and (
        len(action_bodies) != 1
        or not _pool_add_body_matches_action_value(
            action_bodies[0],
            action_type=action_type,
            expected=action.get("value"),
        )
    ):
        missing.append("value")
    output_value = action.get("output_value")
    output_value_bodies = _pool_add_label_bodies(utterance, output_value_label)
    output_value_label_count = _pool_add_label_match_count(
        utterance,
        output_value_label,
    )
    if bool(output_value_label_count or output_value is not None) and (
        output_value_label_count != len(output_value_bodies)
        or output_value_label_count != 1
        or len(output_value_bodies) != 1
        or output_value is None
        or not _pool_add_body_matches_complete_value(
            output_value_bodies[0],
            output_value,
        )
    ):
        missing.append("output_value")
    return tuple(missing)


def _pool_add_placement_modes(
    utterance: str,
) -> tuple[frozenset[str], bool, bool]:
    """Read only an exact label or one of the two reviewed Chinese semantics."""

    reason_spans = _pool_add_reason_spans(utterance)
    label_bodies = _pool_add_label_bodies(
        utterance,
        _POOL_ADD_PLACEMENT_MODE_LABEL_RE,
    )
    label_count = _pool_add_label_match_count(
        utterance,
        _POOL_ADD_PLACEMENT_MODE_LABEL_RE,
    )
    observed: set[str] = set()
    label_values_valid = label_count == len(label_bodies) and label_count <= 1
    for body in label_bodies:
        if body in _POOL_ADD_PLACEMENT_MODES:
            observed.add(body)
            continue
        body_matches = {
            mode
            for mode, pattern in (
                (
                    "before_selected_members",
                    _POOL_ADD_BEFORE_SELECTED_MEMBERS_RE,
                ),
                (
                    "replace_selected_members",
                    _POOL_ADD_REPLACE_SELECTED_MEMBERS_RE,
                ),
            )
            if (
                (match := pattern.fullmatch(body)) is not None
                and match.start() == 0
            )
        }
        if len(body_matches) != 1:
            label_values_valid = False
        observed.update(body_matches)

    phrase_matches: list[tuple[str, re.Match[str]]] = []
    for mode, pattern in (
        ("before_selected_members", _POOL_ADD_BEFORE_SELECTED_MEMBERS_RE),
        ("replace_selected_members", _POOL_ADD_REPLACE_SELECTED_MEMBERS_RE),
    ):
        phrase_matches.extend((mode, match) for match in pattern.finditer(utterance))
    phrase_values_valid = True
    for mode, match in phrase_matches:
        if any(left <= match.start() < right for left, right in reason_spans):
            continue
        if _pool_operation_is_negated(utterance, start=match.start()):
            phrase_values_valid = False
            continue
        observed.add(mode)

    explicit = label_count > 0 or any(
        not any(left <= match.start() < right for left, right in reason_spans)
        for _mode, match in phrase_matches
    )
    return (
        frozenset(observed),
        label_values_valid and phrase_values_valid,
        explicit,
    )


def _pool_add_explicit_reasons(utterance: str) -> tuple[str, ...]:
    return tuple(
        (match.group("zh") or match.group("en")).strip()
        for match in _POOL_ADD_REASON_LABEL_RE.finditer(utterance)
    )


def _ground_strategy_pool_add_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Bind one explicit selection/asset and three independently labeled controls."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]

    if len(utterance) > _POOL_MAX_UTTERANCE_CHARS:
        return _clarification(
            "单次 Strategy Pool 入池指令过长；请只保留一个 source ID、"
            "策略池类型、默认动作、命中动作和可选理由。",
            code="strategy_pool_add_request_too_large",
            fields=("utterance",),
        )

    has_intent, has_positive_intent = _pool_add_intent_state(utterance)
    if not has_positive_intent:
        code = (
            "strategy_pool_add_intent_negated"
            if has_intent
            else "strategy_pool_add_intent_required"
        )
        return _clarification(
            "原话没有明确授权一次正向的 Strategy Pool 入池；否定式请求不会"
            "创建 Pool revision。请明确说出要加入的完整 source ID。",
            code=code,
            fields=("pool_add_intent",),
        )
    if _pool_add_has_positive_lifecycle_follow_up(utterance):
        return _clarification(
            "本轮只能把一个明确候选写入可逆 draft Strategy Pool；采纳、部署、"
            "上线或投产必须在后续请求中单独发起。",
            code="strategy_pool_add_single_step_required",
            fields=("next_action",),
        )

    candidate_matches = tuple(_AUTOMATIC_TREE_ASSET_ID_TOKEN_RE.finditer(utterance))
    selection_matches = tuple(
        match
        for pattern in (
            _AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE,
            _INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE,
            _INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE,
            _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE,
            _SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE,
        )
        for match in pattern.finditer(utterance)
    )
    candidate_ids = frozenset(match.group(0) for match in candidate_matches)
    selection_ids = frozenset(match.group(0) for match in selection_matches)
    source_like_matches = tuple(_POOL_SOURCE_LIKE_TOKEN_RE.finditer(utterance))
    source_like_ids = frozenset(match.group(0) for match in source_like_matches)
    source_prefix_count = _pool_source_prefix_count(utterance)
    canonical_source_ids = candidate_ids | selection_ids
    source_count = len(candidate_matches) + len(selection_matches)
    authorized_spans = _pool_add_authorized_clause_spans(utterance)
    authorized_candidate_matches = tuple(
        match
        for match in candidate_matches
        if any(
            left <= match.start() and match.end() <= right
            for left, right in authorized_spans
        )
    )
    authorized_selection_matches = tuple(
        match
        for match in selection_matches
        if any(
            left <= match.start() and match.end() <= right
            for left, right in authorized_spans
        )
    )
    authorized_source_like_matches = tuple(
        match
        for match in source_like_matches
        if any(
            left <= match.start() and match.end() <= right
            for left, right in authorized_spans
        )
    )
    authorized_canonical_matches = (
        authorized_candidate_matches + authorized_selection_matches
    )
    authorized_source_like_ids = frozenset(
        match.group(0) for match in authorized_source_like_matches
    )
    authorized_canonical_ids = frozenset(
        match.group(0) for match in authorized_canonical_matches
    )
    if (
        source_count != 1
        or len(source_like_matches) != 1
        or source_like_ids != canonical_source_ids
        or source_prefix_count != len(source_like_matches)
        or len(authorized_spans) != 1
        or len(authorized_canonical_matches) != 1
        or len(authorized_source_like_matches) != 1
        or authorized_source_like_ids != authorized_canonical_ids
    ):
        legacy_asset_id = inputs.get("candidate_asset_id")
        if (
            source_count == 0
            and not source_like_matches
            and source_prefix_count == 0
            and isinstance(legacy_asset_id, str)
        ):
            return _clarification(
                "请在原话中明确提供 Strategy Pool 的策略类型、完整 ID 和 typed "
                f"action；当前无法核对：{legacy_asset_id}。平台不会采用 LLM "
                "猜测的 ID、动作、顺序、hash 或指标。",
                code="strategy_pool_controls_not_grounded",
                fields=(legacy_asset_id,),
            )
        return _clarification(
            "请逐字提供且只提供一个完整 candidate_asset_id 或 selection_id；"
            "selection_id 必须是 automatic-tree-leaf-selection-、"
            "interactive-tree-frontier-selection-、cross-matrix-cell-selection- "
            "或 scorecard-cutoff-selection- "
            "后接 32 位小写十六进制字符，不能同时给出两类来源。",
            code="strategy_pool_add_source_required",
            fields=("candidate_asset_id", "selection_id"),
        )
    expected_source_field = (
        "selection_id" if authorized_selection_matches else "candidate_asset_id"
    )
    observed_source_id = authorized_canonical_matches[0].group(0)
    if (
        set(inputs) & {"candidate_asset_id", "selection_id"} != {expected_source_field}
        or inputs.get(expected_source_field) != observed_source_id
    ):
        return _clarification(
            "模型草案中的入池来源与用户原话不一致；平台不会替换、补全或"
            "猜测 candidate_asset_id/selection_id。",
            code="strategy_pool_add_source_not_grounded",
            fields=(expected_source_field,),
        )

    missing_controls: list[str] = []
    observed_strategy_types, strategy_type_labels_valid = _pool_add_strategy_types(
        utterance
    )
    strategy_type_bodies = _pool_add_label_bodies(
        utterance,
        _POOL_ADD_STRATEGY_TYPE_LABEL_RE,
    )
    if (
        not strategy_type_labels_valid
        or observed_strategy_types != {inputs["strategy_type"]}
        or any(
            _pool_add_strategy_type_body_residual(body, inputs["strategy_type"])
            for body in strategy_type_bodies
        )
    ):
        missing_controls.append("strategy_type")
    for field, pattern, reason_code_label, output_value_label in (
        (
            "default_action",
            _POOL_ADD_DEFAULT_ACTION_LABEL_RE,
            _POOL_ADD_DEFAULT_REASON_CODE_LABEL_RE,
            _POOL_ADD_DEFAULT_OUTPUT_VALUE_LABEL_RE,
        ),
        (
            "action",
            _POOL_ADD_HIT_ACTION_LABEL_RE,
            _POOL_ADD_HIT_REASON_CODE_LABEL_RE,
            _POOL_ADD_HIT_OUTPUT_VALUE_LABEL_RE,
        ),
    ):
        action_bodies = _pool_add_label_bodies(utterance, pattern)
        action_label_count = _pool_add_label_match_count(utterance, pattern)
        labeled_types = _pool_add_labeled_action_types(utterance, pattern=pattern)
        expected_type = inputs[field]["type"]
        if (
            action_label_count != 1
            or action_label_count != len(action_bodies)
            or len(labeled_types) != 1
            or labeled_types[0] != {expected_type}
            or (
                len(action_bodies) == 1
                and _pool_add_action_body_residual(action_bodies[0], inputs[field])
            )
        ):
            missing_controls.append(field)
        payload_controls = _pool_add_action_payload_controls(
            utterance,
            inputs[field],
            action_bodies=action_bodies,
            reason_code_label=reason_code_label,
            output_value_label=output_value_label,
        )
        if payload_controls:
            missing_controls.append(field)
    if missing_controls:
        return _clarification(
            "请分别显式标注策略池类型、Pool 默认动作和命中动作；三者是"
            "独立控制，平台不会从动作词推断 Pool 类型，也不会对调两个动作。"
            "当前无法核对：" + "、".join(dict.fromkeys(missing_controls)) + "。",
            code="strategy_pool_add_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing_controls)),
        )

    observed_placement_modes, placement_values_valid, placement_is_explicit = (
        _pool_add_placement_modes(utterance)
    )
    placement_mode = inputs.get("placement_mode")
    if (
        placement_mode is not None
        and (
            not placement_values_valid
            or observed_placement_modes != {placement_mode}
        )
    ) or (placement_mode is None and placement_is_explicit):
        return _clarification(
            "可选 placement_mode 只能由“放置方式: "
            "before_selected_members/replace_selected_members”或清晰中文"
            "“保留成员作为回退并放在成员前/由 Voting 替代成员”落地；"
            "缺失、模糊、冲突或与草案不一致时不会猜测。",
            code="strategy_pool_add_placement_mode_not_grounded",
            fields=("placement_mode",),
        )

    explicit_reasons = _pool_add_explicit_reasons(utterance)
    reason = inputs.get("reason")
    if bool(explicit_reasons or reason is not None) and (
        len(explicit_reasons) != 1
        or not isinstance(reason, str)
        or reason != explicit_reasons[0]
    ):
        return _clarification(
            "可选 reason 必须与用户以“入池理由/理由/reason”显式标注的"
            "唯一文本逐字一致；未显式给出时模型必须省略。",
            code="strategy_pool_add_reason_not_grounded",
            fields=("reason",),
        )
    if isinstance(reason, str) and _pool_reason_has_active_language(reason):
        return _clarification(
            "reason 只能说明被动业务依据，不能承载入池、删除、改动作、"
            "重排、撤销或其他操作指令。",
            code="strategy_pool_reason_not_passive",
            fields=("reason",),
        )
    if residual := _pool_add_unconsumed_text(utterance):
        return _clarification(
            "Strategy Pool 入池只能包含一个明确命令子句和已知的显式控制标签；"
            "历史叙述、转述、考虑中描述、撤销语句或其他未消费操作不会执行。",
            code="strategy_pool_add_command_not_explicit",
            fields=(residual[:80],),
        )
    return result


def _ground_strategy_pool_request(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Prove that every executable Pool control came from the user text."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    workflow = draft.workflow

    if workflow == "strategy_pool_add_candidate":
        return _ground_strategy_pool_add_request(utterance, result)

    if workflow == "strategy_pool_reorder" and (
        _POOL_PARTIAL_REORDER_RE.search(utterance)
        or _POOL_HEURISTIC_REORDER_RE.search(utterance)
    ):
        return _clarification(
            "Strategy Pool 重排必须提供当前池全部 rule_id/entry_id 的完整、无重复顺序；"
            "不能只说把某条放前面，也不能按效果、坏率或推荐自动排序。",
            code="strategy_pool_full_order_required",
            fields=("ordered_ids",),
        )

    missing_controls: list[str] = []
    strategy_type = str(inputs.get("strategy_type") or "")
    strategy_type_pattern = _POOL_STRATEGY_TYPE_GROUNDING.get(strategy_type)
    if strategy_type_pattern is None or strategy_type_pattern.search(utterance) is None:
        missing_controls.append(f"strategy_type {strategy_type or 'unknown'}")
    if workflow in {"strategy_pool_remove_entry", "strategy_pool_set_action"}:
        identifier_name = "rule_id" if "rule_id" in inputs else "entry_id"
        identifier = inputs[identifier_name]
        if not _utterance_contains_token(utterance, identifier):
            missing_controls.append(identifier)
        if workflow == "strategy_pool_set_action":
            missing_controls.extend(
                _ungrounded_pool_actions(utterance, inputs["action"])
            )
    elif workflow == "strategy_pool_reorder":
        ordered_ids = inputs["ordered_ids"]
        positions = [utterance.find(identifier) for identifier in ordered_ids]
        missing_controls.extend(
            identifier
            for identifier, position in zip(ordered_ids, positions, strict=True)
            if position < 0
        )
        observed_positions = [position for position in positions if position >= 0]
        if observed_positions != sorted(observed_positions):
            missing_controls.append("用户原话中的完整顺序")
    elif workflow == "strategy_pool_compile":
        pass

    reason = inputs.get("reason")
    if isinstance(reason, str) and reason.casefold() not in utterance.casefold():
        missing_controls.append(reason)

    if missing_controls:
        rendered = "、".join(dict.fromkeys(missing_controls))
        return _clarification(
            "请在原话中明确提供 Strategy Pool 的策略类型、完整 ID 和 typed action；"
            f"当前无法核对：{rendered}。平台不会采用 LLM 猜测的 ID、动作、顺序、hash 或指标。",
            code="strategy_pool_controls_not_grounded",
            fields=tuple(dict.fromkeys(missing_controls)),
        )
    if isinstance(reason, str) and _pool_reason_has_active_language(reason):
        return _clarification(
            "reason 只能说明被动业务依据，不能承载入池、删除、改动作、"
            "重排、撤销或其他操作指令。",
            code="strategy_pool_reason_not_passive",
            fields=("reason",),
        )
    if workflow in _POOL_MUTATION_INTENT_PATTERNS and not (
        _pool_mutation_has_positive_intent(utterance, workflow, inputs)
    ):
        return _clarification(
            "原话没有明确授权当前轮执行一次正向 Strategy Pool 修改；否定、"
            "问句、历史描述、失败态、撤销或其他未消费操作不会创建新的 Pool revision。",
            code="strategy_pool_mutation_intent_required",
            fields=("pool_mutation_intent",),
        )
    return result


def _ungrounded_pool_actions(
    utterance: str,
    *actions: Mapping[str, Any],
) -> list[str]:
    missing: list[str] = []
    for action in actions:
        action_type = str(action.get("type") or "")
        pattern = _POOL_ACTION_GROUNDING.get(action_type)
        if pattern is None or pattern.search(utterance) is None:
            missing.append(f"typed action {action_type or 'unknown'}")
        reason_code = action.get("reason_code")
        if isinstance(reason_code, str) and not _utterance_contains_token(
            utterance, reason_code
        ):
            missing.append(reason_code)
        if action_type in {"limit", "pricing", "segment"} and not (
            _utterance_contains_pool_action_value(utterance, action.get("value"))
        ):
            missing.append(f"typed action value {action.get('value')}")
        output_value = action.get("output_value")
        if output_value is not None and not _utterance_contains_pool_action_value(
            utterance, output_value
        ):
            missing.append(f"typed action output_value {output_value}")
    return missing


def _utterance_contains_pool_action_value(utterance: str, value: object) -> bool:
    candidates = {str(value)}
    try:
        candidates.add(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        return False
    folded = utterance.casefold()
    return any(candidate.casefold() in folded for candidate in candidates)


def _utterance_supports_risk_threshold(
    utterance: str,
    *,
    operator: str,
    value: float,
) -> bool:
    return any(
        candidate_operator == operator
        and math.isclose(candidate_value, value, rel_tol=0.0, abs_tol=1e-12)
        for candidate_operator, candidate_value in _risk_threshold_expressions(
            utterance
        )
    )


def _utterance_contains_token(utterance: str, token: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])"
    return re.search(pattern, utterance) is not None


def _risk_threshold_expressions(utterance: str) -> tuple[tuple[str, float], ...]:
    expressions: list[tuple[str, float]] = []
    for match in _RISK_THRESHOLD_EXPRESSION_RE.finditer(utterance):
        expressions.append(
            (
                _normalized_threshold_operator(match.group("operator")),
                _ratio_token_value(match.group("value")),
            )
        )
    return tuple(expressions)


def _normalized_threshold_operator(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    if normalized in {
        "大于等于",
        "不低于",
        "至少",
        "不少于",
        "达到",
        ">=",
        "≥",
        "greater than or equal",
        "greater than or equal to",
        "at least",
    }:
        return ">="
    if normalized in {
        "小于等于",
        "不高于",
        "至多",
        "最多",
        "<=",
        "≤",
        "less than or equal",
        "less than or equal to",
        "at most",
    }:
        return "<="
    if normalized in {"大于", "高于", "超过", ">", "more than", "greater than"}:
        return ">"
    return "<"


def _ratio_token_value(value: str) -> float:
    normalized = re.sub(r"\s+", "", value)
    if normalized.startswith("百分之"):
        return float(normalized[len("百分之") :]) / 100.0
    if normalized.endswith("%"):
        return float(normalized[:-1]) / 100.0
    return float(normalized)


def _reject_workflow_fields(
    inputs: Mapping[str, Any],
    allowed: set[str],
    *,
    workflow: str,
) -> None:
    unexpected = sorted(set(inputs) - allowed)
    if unexpected:
        raise _DraftValidationError(
            f"{workflow} workflow_inputs 包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )


def _workflow_column(
    value: object,
    *,
    name: str,
    whitelist: tuple[str, ...],
) -> str:
    column = _required_text(value, name=name)
    if column not in whitelist:
        raise _DraftValidationError(f"{name} 使用了数据集中不存在的列「{column}」。")
    return column


def _number_sequence(
    value: object,
    *,
    name: str,
    minimum: float | None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    minimum_items: int = 1,
    maximum_items: int,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or not minimum_items <= len(value) <= maximum_items
    ):
        raise _DraftValidationError(
            f"{name} 必须是包含 {minimum_items} 到 {maximum_items} 个有限数字的数组。"
        )
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise _DraftValidationError(f"{name} 只能包含有限数字。")
        number = float(item)
        if not math.isfinite(number):
            raise _DraftValidationError(f"{name} 只能包含有限数字。")
        if minimum is not None and (
            number < minimum or (exclusive_minimum and number == minimum)
        ):
            relation = "大于" if exclusive_minimum else "大于等于"
            raise _DraftValidationError(
                f"{name} 中每个值都必须{relation} {minimum:g}。"
            )
        if maximum is not None and number > maximum:
            raise _DraftValidationError(f"{name} 中每个值都必须小于等于 {maximum:g}。")
        numbers.append(number)
    if len(set(numbers)) != len(numbers):
        raise _DraftValidationError(f"{name} 不能包含重复值。")
    return numbers


def _standard_workflow_confirmation_text(
    draft: StandardWorkflowRequestDraft,
) -> str:
    inputs = draft.workflow_inputs
    if draft.workflow == "strategy_project_context":
        details = [
            "已识别为〔策略项目上下文 Workflow〕",
            f"现状截止日：{inputs['as_of']}",
            "本步骤只收集并绑定项目现状、历史策略和缺失信息，不计算或复制业务指标",
        ]
        if inputs.get("scope"):
            details.append(f"分析范围：{inputs['scope']}")
        if inputs["business_context"]:
            details.append(
                "用户提供字段：" + "、".join(inputs["business_context"].keys())
            )
        if inputs["explicit_unavailable"]:
            details.append(
                "明确暂缺字段：" + "、".join(inputs["explicit_unavailable"])
            )
        if inputs["external_report_filenames"]:
            details.append(
                "外部报告仅作为不透明证据复制并绑定："
                + "、".join(inputs["external_report_filenames"])
            )
        details.extend(
            [
                "当前样本、Pool、回测、监控与内部历史由平台自动发现并校验",
                "revision/CAS、消息哈希、artifact id、来源引用和所有指标由平台拥有",
            ]
        )
    elif draft.workflow == "strategy_sample_design_v2":
        performance = inputs["performance_window"]
        observation = inputs["observation_window"]
        maturity = inputs["maturity"]
        historical_score = inputs["historical_score"]
        details = [
            "已识别为〔V2 双总体策略样本设计 Workflow〕",
            f"总体关系：{inputs['relationship']}",
            (
                "总体纳排：两者均无纳排"
                if all(
                    population[field] is None
                    for population in (
                        inputs["approval_population"],
                        inputs["risk_population"],
                    )
                    for field in ("inclusion", "exclusion")
                )
                else "总体纳排：已按用户提供的受限条件固化"
            ),
            (
                f"表现窗：{performance['days']} 天"
                if performance["status"] == "provided"
                else "表现窗：unavailable"
            ),
            (
                f"观察窗：{observation['start']} 至 {observation['end']}"
                if observation["status"] == "provided"
                else "观察窗：unavailable"
            ),
            f"成熟度：{maturity['status']}；坏样本值：{inputs['target_bad_value']}",
            (
                f"历史分：{historical_score['status']}；"
                f"字段：{historical_score['column'] or 'null'}；"
                f"方向：{historical_score['direction'] or 'null'}"
            ),
            "approval/risk 总体与 development/validation/OOT 三分区将分别固化",
            "平台将按请求语义选择无损 compatibility 链或原生 V2 执行；"
            "不会删减总体纳排、时间切分或复杂 selector",
            "legacy ref、scope、policy、数据/workspace 身份和所有 id/hash 由平台绑定",
        ]
    elif draft.workflow == "strategy_model_evidence_v2":
        details = [
            "已识别为〔Strategy ModelEvidence V2 Workflow〕",
            "workflow_inputs 为空；平台从当前 task 绑定已认证 SampleDesign V2 与单变量候选",
            "当前只归集 univariate evidence，不训练模型、不比较模型、不生成月度/OOT 模型证据",
            "不报告、不采纳、不部署",
        ]
    elif draft.workflow == "strategy_dsl_delivery":
        details = [
            "已识别为〔离线 Strategy DSL 交付 Workflow〕",
            (
                f"策略 ID：{inputs['strategy_id']}"
                if "strategy_id" in inputs
                else "策略 ID：由平台仅在当前任务恰有一个可交付策略时唯一绑定"
            ),
            "平台将原子绑定策略类型、version/spec hash 与当前活动 task-owned "
            "dataset id/content hash",
            "将生成 Python、DuckDB SQL、canonical JSON 和等价证据四个"
            "content-hash 固定下载文件",
            "等价证据会明确显示校验样本数、源数据行数及是否为最多 4096 行的"
            "受治理有界样本，不会把抽样校验称为全量校验",
            "本步骤只生成离线代码；不会应用、写回、采纳、晋级或部署策略",
        ]
    elif draft.workflow == "strategy_report_bundle_v2":
        details = [
            "已识别为〔StrategyReportBundle V2 Workflow〕",
            f"报告标题：{inputs['title']}",
            f"报告状态：{inputs['status']}",
            "平台将绑定当前认证 ProjectContext、最新精确 SampleDesign V2、"
            "当前非空 approval/reject Pool 及其最新精确 PoolImpact",
            "只有与同一 SampleDesign/模型链完全兼容的最新认证模型证据才会纳入；"
            "不兼容证据保持缺省，损坏的最新证据不会回退到旧版本",
            "策略身份、report head revision/CAS、generated_at、artifact id/hash "
            "和所有指标均由平台拥有",
            "本步骤只生成 JSON、Markdown、XLSX 三种受治理报告；"
            "不会创建策略，也不代表采纳或部署",
        ]
    elif draft.workflow == "strategy_sample_design":
        performance = (
            f"已提供 {inputs['performance_window_days']} 天"
            if inputs["performance_window_status"] == "provided"
            else "unavailable"
        )
        observation = (
            f"{inputs['observation_start']} 至 {inputs['observation_end']}"
            if inputs["observation_window_status"] == "provided"
            else "unavailable"
        )
        maturity_labels = {
            "confirmed_matured": "已确认成熟",
            "not_matured": "尚未成熟",
            "unknown": "未知",
        }
        details = [
            "已识别为〔策略样本设计 Workflow〕",
            f"表现窗：{performance}",
            f"观察窗：{observation}",
            f"成熟度：{maturity_labels[inputs['maturity_status']]}",
            f"坏样本值：{inputs['target_bad_value']}；"
            f"好样本值：{1 - inputs['target_bad_value']}",
        ]
        if "split_col" in inputs:
            validation_values = (
                "、".join(str(value) for value in inputs["validation_values"])
                if inputs["validation_values"]
                else "unavailable"
            )
            oot_values = (
                "、".join(str(value) for value in inputs["oot_values"])
                if inputs["oot_values"]
                else "unavailable"
            )
            details.append(
                f"切分列 {inputs['split_col']}；开发样本值 "
                + "、".join(str(value) for value in inputs["development_values"])
                + f"；验证样本值 {validation_values}；OOT 样本值 {oot_values}"
            )
        else:
            details.append("未声明开发/验证/OOT 切分，平台只冻结 overall 样本边界")
        for field, label in (
            ("month_col", "月份列"),
            ("weight_col", "权重列"),
            ("loan_amount_col", "放款金额列"),
            ("overdue_amount_col", "逾期金额列"),
        ):
            if field in inputs:
                details.append(f"{label}：{inputs[field]}")
        if inputs.get("drop_nan_labels") is True:
            details.append(
                "标签缺失处理：保留总体样本行，仅从好坏/风险分母排除空标签"
            )
        elif inputs.get("drop_nan_labels") is False:
            details.append("标签缺失处理：保留 NaN 标签行，遇到缺失时不会自动排除")
        if (
            inputs["performance_window_status"] == "unavailable"
            or inputs["observation_window_status"] == "unavailable"
            or inputs["maturity_status"] != "confirmed_matured"
        ):
            details.append(
                "本次仅按 exploration-only 降级固化；不能据此声称样本已成熟、"
                "完成独立验证或晋级"
            )
        details.extend(
            [
                "数据集/hash、workspace、语义映射和目标列由平台绑定",
                "本步骤只冻结活动样本/派生数据边界并计算样本证据；"
                "不做自由文本过滤或派生，不建模、不建树、不入池、不采纳、不部署、"
                "不生成最终报告",
            ]
        )
    elif draft.workflow == "profit_calc":
        params = inputs["profit_params"]
        details = [
            "已识别为〔标准利润分析 Workflow〕",
            f"EAD 列 {inputs['ead_col']}，PD 列 {inputs['pd_col']}",
            "分析范围："
            + (
                f"按 {inputs['segment_col']} 分组"
                if "segment_col" in inputs
                else "全样本"
            ),
            (
                f"年利率 {params['annual_rate']:.2%}，资金成本率 {params['funding_rate']:.2%}，"
                f"LGD {params['lgd']:.2%}，单笔成本 {params['operating_cost_per_loan']:g}，"
                f"期限 {params['term_months']} 个月"
            ),
        ]
    elif draft.workflow == "roll_rate_matrix":
        details = [
            "已识别为〔标准滚动率矩阵 Workflow〕",
            (
                f"客户 ID 列 {inputs['id_col']}，时间列 {inputs['time_col']}，"
                f"状态列 {inputs['status_col']}"
            ),
            "状态顺序：" + " → ".join(inputs["states"]),
            "观测口径：相邻观测记录，不等同于固定月末快照迁徙",
        ]
        if "balance_col" in inputs:
            details.append(f"余额加权列：{inputs['balance_col']}")
    elif draft.workflow == "limit_pricing_matrix":
        risk_source = (
            f"PD 列 {inputs['pd_col']}"
            if "pd_col" in inputs
            else f"目标列 {inputs['target_col']}"
        )
        banding = (
            "分箱边界 " + "、".join(f"{value:g}" for value in inputs["band_edges"])
            if "band_edges" in inputs
            else f"等频分为 {inputs['n_bands']} 档"
        )
        details = [
            "已识别为〔标准额度定价矩阵 Workflow〕",
            f"评分列 {inputs['score_col']}，风险来源 {risk_source}，{banding}",
            "额度网格："
            + "、".join(f"{value:,.12g}" for value in inputs["limit_grid"]),
            "利率网格：" + "、".join(f"{value:.2%}" for value in inputs["rate_grid"]),
            (
                f"LGD {inputs['lgd']:.2%}，资金成本率 {inputs['funding_rate']:.2%}，"
                f"期限 {inputs['term_months']} 个月，单笔成本 {inputs['cost_per_loan']:g}，"
                f"EL/EAD 上限 {inputs['el_ead_max']:.2%}"
            ),
        ]
        if "target_col" in inputs:
            details.append(
                "标签缺失处理："
                + (
                    "按明确授权丢弃 NaN 标签行"
                    if inputs.get("drop_nan_labels")
                    else "不自动丢弃 NaN 标签行"
                )
            )
        if "strategy_id" in inputs:
            details.append(f"关联策略 ID：{inputs['strategy_id']}")
        details.append("平台先计算完整矩阵；接受或导出矩阵仍需第二次明确确认")
    elif draft.workflow == "univariate_candidate_analysis":
        feature_text = (
            "、".join(inputs["features"])
            if inputs["features"]
            else "当前语义映射中的全部候选字段"
        )
        method_text = (
            "数值字段自动比较等频、等距、ChiMerge、决策树；类别字段使用等值箱"
            if not inputs["methods"]
            else "、".join(inputs["methods"]) + "；类别字段仍使用等值箱"
        )
        details = [
            "已识别为〔单变量候选分析 Workflow〕",
            f"候选字段：{feature_text}",
            "分箱方法：" + method_text,
            (f"目标箱数 {inputs['bin_count']}，最小箱占比 {inputs['min_bin_pct']:.2%}"),
        ]
        if "loan_amount_col" in inputs:
            details.append(f"放款金额列：{inputs['loan_amount_col']}")
        if "overdue_amount_col" in inputs:
            details.append(f"逾期金额列：{inputs['overdue_amount_col']}")
        if inputs["sentinel_values"]:
            details.append(
                "独立哨兵值："
                + "、".join(str(value) for value in inputs["sentinel_values"])
            )
        if "manual_breakpoints" in inputs:
            details.append(
                "手工切点："
                + "；".join(
                    feature
                    + "=["
                    + "、".join(f"{value:g}" for value in points)
                    + "]"
                    for feature, points in inputs["manual_breakpoints"].items()
                )
            )
        details.append("只生成 development/unvalidated 候选证据，不冒充独立验证结果")
    elif draft.workflow == "univariate_candidate_refinement":
        merge_text = (
            "；".join(" + ".join(group) for group in inputs["merge_groups"])
            if inputs["merge_groups"]
            else "不合并，保留原分箱"
        )
        if "source_bin_ids" in inputs["selection"]:
            selection_text = "显式选择 " + "、".join(
                inputs["selection"]["source_bin_ids"]
            )
        else:
            threshold = inputs["selection"]["risk_threshold"]
            selection_text = f"按观测坏率 {threshold['operator']} {threshold['value']:.2%} 确定性选择"
        details = [
            "已识别为〔单变量候选选择与合并 Workflow〕",
            f"候选字段与方法：{inputs['feature']} / {inputs['method']}",
            f"分箱合并：{merge_text}",
            f"候选选择：{selection_text}",
            "平台会从任务自有证据重放样本并重新计算全部指标",
            "只生成 development/unvalidated 候选资产，不代表独立验证、采纳或上线",
        ]
        if "selection_reason" in inputs:
            details.append(f"选择说明：{inputs['selection_reason']}")
        if "manual_breakpoints" in inputs:
            details.append(
                "手工切点："
                + "；".join(
                    feature
                    + "=["
                    + "、".join(f"{value:g}" for value in points)
                    + "]"
                    for feature, points in inputs["manual_breakpoints"].items()
                )
            )
    elif draft.workflow == "candidate_monthly_stability":
        details = [
            "已识别为〔候选逐月稳定性 Workflow〕",
        ]
        if "asset_id" in inputs:
            details.extend(
                [
                    f"来源：已有单变量候选资产 {inputs['asset_id']}",
                    "统计口径：该候选规则命中/未命中的逐月分布与 PSI",
                ]
            )
        else:
            details.extend(
                [
                    (
                        "来源：当前 "
                        f"{inputs['strategy_type']} Strategy Pool 条目 "
                        f"{inputs['entry_id']}"
                    ),
                    "统计口径：该条目按当前 Pool 精确顺序的增量首次命中/未命中分布与 PSI",
                ]
            )
        details.extend(
            [
                "平台将在计划创建前恢复并认证 artifact/Pool CAS、活动 workspace、"
                "成熟 development SampleDesign 与月份字段",
                "基准固定为完整 development 样本；每个 YYYYMM 与同一基准比较，"
                "不会滚动改基准",
                "本步骤只生成只读稳定性证据；不会修改 Pool、入池、采纳或部署",
            ]
        )
    elif draft.workflow == "scorecard_band_build":
        if "bin_count" in inputs:
            banding = f"等频 {inputs['bin_count']} 档"
        elif "raw_pd_band_edges" in inputs:
            banding = (
                "raw PD 边界 ["
                + "、".join(f"{value:g}" for value in inputs["raw_pd_band_edges"])
                + "]"
            )
        else:
            banding = "省略用户分带参数，由受控 Tool 使用默认等频 10 档"
        details = [
            "已识别为〔Scorecard 完整分数带 Workflow〕",
            f"分带方式：{banding}",
            "平台将绑定最新精确兼容且完整认证的 ScoreEvidence、原始分数向量"
            "和 StrategySampleDesign V2；损坏的最新证据不会回退",
            "本步骤只生成全部分带及全部可选 cutoff 证据；不会自动选择、"
            "排名或推荐 cutoff",
            "不会入池、应用、采纳或部署",
        ]
    elif draft.workflow == "scorecard_cutoff_selection":
        details = [
            "已识别为〔Scorecard cutoff 精确选择 Workflow〕",
            f"完整分数带资产 pointer：{inputs['asset_id']}",
            f"精确 cutoff pointer：{inputs['cutoff_id']}",
            "平台将从当前任务严格恢复 source artifact/hash 与完整分数带；"
            "不会自动排名或推荐",
            "本步骤只物化 pointer，不复制全部分带；不会入池、应用、采纳或部署",
        ]
        if "reason" in inputs:
            details.append(f"用户原话选择说明：{inputs['reason']}")
    elif draft.workflow == "cross_matrix_candidate_search":
        details = [
            "已识别为〔Cross Matrix 自动组合搜索 Workflow〕",
            "显式候选字段：" + "、".join(inputs["features"]),
            f"确定性评估预算：最多 {inputs['max_pairs']} 个特征对",
            "平台将在计划创建时绑定最新精确单变量候选证据，并仅使用其"
            " risk/development 样本；每个字段的轴方法由受控 Tool 从父证据中"
            "选择最高排名的可用方法",
            "本步骤只发布聚合搜索证据；不会构建候选或选择候选，不会入池、"
            "应用、采纳或部署",
        ]
    elif draft.workflow == "cross_matrix_candidate_build_from_search":
        details = [
            "已识别为〔Cross 搜索结果精确构建 Workflow〕",
            f"搜索证据 pointer：{inputs['search_id']}",
            f"特征对 pointer：{inputs['pair_id']}",
            "平台将在计划开始前重新认证完整搜索证据、父候选、数据与"
            " risk/development 样本，并重新计算精确 Cross 资产",
            "不会采用排名、最好或冠军等启发式选择；本步骤只构建一个 "
            "development/backtested/unvalidated Cross 候选",
            "不会加入或修改 Pool，不会设置动作、应用、采纳或部署",
        ]
    elif draft.workflow == "cross_rule_search":
        constraints = inputs["constraints"]
        amount_lift = constraints["min_amount_lift"]
        details = [
            "已识别为〔2D/3D Cross 阈值规则搜索 Workflow〕",
            "显式候选字段：" + "、".join(inputs["features"]),
            f"组合维度：{inputs['dimension']}D；最多评估 "
            f"{inputs['max_trials']} 条确定性试验",
            "约束："
            f"min_lift={constraints['min_lift']:g}，"
            f"min_bad_count={constraints['min_bad_count']}，"
            f"max_hit_share={constraints['max_hit_share']:g}，"
            "min_amount_lift="
            + ("null" if amount_lift is None else f"{amount_lift:g}"),
            "平台将绑定最新精确单变量证据与 risk/development 样本，"
            "从认证分箱边界和风险方向生成有预算的 2D/3D 阈值组合",
            "本步骤只发布全部已评估规则的聚合证据与排序；不会自动选择、"
            "构建候选、入池、应用、采纳或部署",
        ]
    elif draft.workflow == "cross_rule_candidate_build_from_search":
        details = [
            "已识别为〔Cross 阈值规则精确候选构建 Workflow〕",
            f"搜索证据 pointer：{inputs['search_id']}",
            f"规则 pointer：{inputs['rule_id']}",
            "平台将重新认证并完整重放搜索、数据、样本和规则条件；"
            "不会采用排名、最好或冠军等启发式选择",
            "本步骤只构建一个 development/unvalidated 候选；不会自动入池、"
            "应用、采纳或部署",
        ]
        if "selection_reason" in inputs:
            details.append(
                f"用户原话选择说明：{inputs['selection_reason']}"
            )
    elif draft.workflow == "cross_matrix_analysis":
        details = [
            "已识别为〔二维 Cross Matrix 候选分析 Workflow〕",
            (
                f"X 轴：{inputs['x_feature']} / {inputs['x_method']}；"
                f"Y 轴：{inputs['y_feature']} / {inputs['y_method']}"
            ),
            (
                f"数值目标箱数 {inputs['bin_count']}，"
                f"最小箱占比 {inputs['min_bin_pct']:.2%}"
            ),
            "平台会先生成两个轴的不可变单变量证据，再逐行重放完整二维矩阵",
            "只生成 development/backtested/unvalidated Cross evidence；"
            "不会选择格子、入池、采纳或部署",
        ]
        if "loan_amount_col" in inputs:
            details.append(f"放款金额列：{inputs['loan_amount_col']}")
        if "overdue_amount_col" in inputs:
            details.append(f"逾期金额列：{inputs['overdue_amount_col']}")
        if inputs["sentinel_values"]:
            details.append(
                "独立哨兵值："
                + "、".join(str(value) for value in inputs["sentinel_values"])
            )
        if "manual_breakpoints" in inputs:
            details.append(
                "手工轴切点："
                + "；".join(
                    feature
                    + "=["
                    + "、".join(f"{value:g}" for value in points)
                    + "]"
                    for feature, points in inputs["manual_breakpoints"].items()
                )
            )
    elif draft.workflow == "cross_matrix_cell_selection":
        details = [
            "已识别为〔Cross Matrix 精确单元格选择 Workflow〕",
            f"完整 Cross 候选资产 pointer：{inputs['cross_asset_id']}",
            "精确 cell pointers：" + "、".join(inputs["cell_ids"]),
            "多个 cell 按确定性 OR 语义组成一个不可变选择；平台按源矩阵顺序归一化",
            "本步骤不排名、不推荐、不生成业务动作，也不会入池、采纳或部署",
        ]
        if "selection_reason" in inputs:
            details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    elif draft.workflow == "automatic_tree_candidate_build":
        direction_labels = {
            "increasing": "递增",
            "decreasing": "递减",
            "unordered": "无序",
        }
        details = [
            "已识别为〔自动决策树候选构建 Workflow〕",
            "候选特征：" + "、".join(inputs["features"]),
        ]
        if "sample_weight_col" in inputs:
            details.append(f"样本权重列 {inputs['sample_weight_col']}")
        if "directions" in inputs:
            details.append(
                "风险方向诊断期望："
                + "、".join(
                    f"{feature}={direction_labels[direction]}"
                    for feature, direction in inputs["directions"].items()
                )
            )
        if "max_depth" in inputs:
            details.append(f"最大深度 {inputs['max_depth']}")
        if "min_leaf_count" in inputs:
            details.append(f"最小叶样本数 {inputs['min_leaf_count']}")
        if "min_weight_fraction_leaf" in inputs:
            details.append(f"最小叶权重占比 {inputs['min_weight_fraction_leaf']:.2%}")
        if "seed" in inputs:
            details.append(f"随机种子 {inputs['seed']}")
        if "loan_amount_col" in inputs:
            details.append(f"放款金额列 {inputs['loan_amount_col']}")
        if "overdue_amount_col" in inputs:
            details.append(f"逾期金额列 {inputs['overdue_amount_col']}")
        details.extend(
            [
                "数据集、hash、workspace、目标列、标签处理和执行预算由平台绑定，"
                "LLM 不得填写",
                "本步骤只构建完整候选树及确定性证据；不会自动选择叶子、写入 "
                "Strategy Pool、采纳或部署",
                "平台不会给叶子生成“最佳”自动排名；后续操作必须由用户引用明确 leaf",
            ]
        )
    elif draft.workflow == "automatic_tree_apply":
        details = [
            "已识别为〔自动树全量写回 Workflow〕",
            f"完整树候选资产 pointer：{inputs['tree_asset_id']}",
        ]
        if "leaf_id_column" in inputs:
            details.append(f"叶节点输出列：{inputs['leaf_id_column']}")
        else:
            details.append("叶节点输出列：由受控 Tool 使用默认列名")
        if "rule_id_column" in inputs:
            details.append(f"规则输出列：{inputs['rule_id_column']}")
        else:
            details.append("规则输出列：由受控 Tool 使用默认列名")
        details.extend(
            [
                "平台将从当前任务重新校验 source artifact/hash、asset hash、"
                "原始数据集与 workspace lineage，LLM 不得填写或覆盖",
                "本步骤创建不可变派生数据集，但不会激活或替换当前 workspace",
                "结果仍是 development / unvalidated；不会入池、采纳或部署，"
                "也不生成业务动作",
            ]
        )
    elif draft.workflow == "automatic_tree_leaf_materialization":
        details = [
            "已识别为〔自动树精确叶节点物化 Workflow〕",
            f"完整树候选资产 pointer：{inputs['tree_asset_id']}",
            f"精确叶节点 pointer：{inputs['leaf_id']}",
            "本步骤只创建指向完整树中该叶节点的不可变 pointer；"
            "不复制规则、条件、指标或业务动作",
            "不会加入 Strategy Pool，也不会采纳或部署策略",
        ]
        if "selection_reason" in inputs:
            details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    elif draft.workflow == "interactive_tree_revision":
        operation = inputs["operation"]
        details = [
            (
                "已识别为〔交互式树阈值调整修订 Workflow〕"
                if operation == "adjust_split_threshold"
                else "已识别为〔交互式树修剪修订 Workflow〕"
            ),
            f"来源树 pointer：{inputs['source_tree_id']}",
            f"精确 split node pointer：{inputs['node_id']}",
            f"操作：{operation}",
            "平台将恢复并认证完整自动树、父 revision、数据集、workspace 与"
            "SampleDesign，并在 development 样本逐行重放新 frontier",
            "每次编辑发布一个不可变 revision；不会修改原树，也不会加入 "
            "Strategy Pool；不会采纳或部署，也不会设置业务动作或写回",
        ]
        if operation == "adjust_split_threshold":
            details.insert(4, f"用户明确的新阈值：{inputs['threshold']}")
        if "reason" in inputs and inputs["reason"] is not None:
            details.append(f"用户原话编辑说明：{inputs['reason']}")
    elif draft.workflow == "interactive_tree_frontier_group_materialization":
        details = [
            "已识别为〔交互树前沿显式 OR 分组物化 Workflow〕",
            f"不可变 revision pointer：{inputs['revision_id']}",
            (
                "精确 frontier node/leaf pointers："
                + "、".join(inputs["source_node_ids"])
            ),
            "组合语义：任一成员命中（OR）；成员顺序不具有语义，平台会按"
            " revision frontier 顺序规范化",
            "平台将从当前 task 唯一恢复并认证 revision artifact、完整父链、"
            "原始自动树与所有成员 candidate fragment",
            "本步骤只创建 pointer-only OR group，不复制 condition、metrics "
            "或业务动作；不会加入 Strategy Pool，也不会应用、采纳、部署或写回",
        ]
        if "selection_reason" in inputs:
            details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    elif draft.workflow == "interactive_tree_frontier_materialization":
        details = [
            "已识别为〔交互树前沿精确物化 Workflow〕",
            f"不可变 revision pointer：{inputs['revision_id']}",
            f"精确 frontier node/leaf pointer：{inputs['source_node_id']}",
            "平台将从当前 task 唯一恢复并认证 revision artifact、完整父链、"
            "原始自动树与确定性 candidate fragment",
            "本步骤只创建 singleton pointer，不复制 condition、metrics 或业务动作；"
            "不会加入 Strategy Pool，也不会采纳、部署或写回",
        ]
        if "selection_reason" in inputs:
            details.append(f"用户原话选择说明：{inputs['selection_reason']}")
    elif draft.workflow == "voting_candidate_search":
        objective = inputs["objective"]
        details = [
            "已识别为〔Voting 组合搜索 Workflow〕",
            f"来源 Strategy Pool 类型：{inputs['strategy_type']}",
            f"组合参数：K={inputs['member_count']}，n={inputs['n']}",
            (f"排序目标：{objective['metric']} / {objective['direction']}"),
            f"确定性评估预算：最多 {inputs['max_combinations']:,} 个组合",
        ]
        if inputs["constraints"]:
            details.append(
                "资格约束："
                + "、".join(
                    f"{item['metric']} {item['operator']} {item['value']:g}"
                    for item in inputs["constraints"]
                )
            )
        else:
            details.append("资格约束：无")
        if inputs["include_rule_ids"]:
            details.append("必须包含：" + "、".join(inputs["include_rule_ids"]))
        if inputs["exclude_rule_ids"]:
            details.append("排除：" + "、".join(inputs["exclude_rule_ids"]))
        details.extend(
            [
                "平台将在计划创建前绑定当前 Pool 身份与受治理 development 样本；"
                "无需用户或模型提供 dataset pointer",
                "本步骤只搜索并发布聚合证据；不会构建候选或选择冠军，"
                "不会修改 Pool、入池、应用、采纳或部署",
            ]
        )
    elif draft.workflow == "voting_candidate_build_from_search":
        details = [
            "已识别为〔Voting 搜索结果精确构建 Workflow〕",
            f"搜索证据 pointer：{inputs['search_id']}",
            f"组合 pointer：{inputs['combo_id']}",
        ]
        if "strategy_type" in inputs:
            details.append(f"来源 Strategy Pool 类型：{inputs['strategy_type']}")
        details.extend(
            [
                "平台将在计划开始前重新校验搜索证据、组合与当前 Pool；"
                "不会采用排名、最好或冠军等启发式选择",
                "本步骤只构建 development/backtested/unvalidated Voting 候选；"
                "不会加入或修改 Pool，不会设置动作、应用、采纳或部署",
            ]
        )
    elif draft.workflow == "voting_candidate_build":
        details = [
            "已识别为〔Voting / n-of-k 候选构建 Workflow〕",
            f"来源 Strategy Pool 类型：{inputs['strategy_type']}",
            "精确成员规则：" + "、".join(inputs["rule_ids"]),
            f"组合条件：{len(inputs['rule_ids'])} 条规则中至少命中 {inputs['n']} 条",
            "平台将绑定当前 Pool revision/hash 和原始样本，逐行计算命中数与风险效果",
            "本步骤只生成 development/backtested/unvalidated 候选；"
            "不会入池、设置业务动作、采纳或部署",
        ]
    elif draft.workflow == "strategy_pool_add_candidate":
        source_field = (
            "selection_id" if "selection_id" in inputs else "candidate_asset_id"
        )
        source_label = (
            "精确选择结果" if source_field == "selection_id" else "候选资产"
        )
        details = [
            "已识别为〔Strategy Pool 添加候选 Workflow〕",
            f"{source_label}：{inputs[source_field]}",
            f"策略类型：{inputs['strategy_type']}",
            "默认动作：" + _compact_json(inputs["default_action"]),
            "命中动作：" + _compact_json(inputs["action"]),
            "平台将从当前 task 的不可变 artifact 绑定 hash、rule/effect/metrics，"
            "并展示 development / unvalidated 证据后自动写入可逆 draft Pool revision",
            "本操作只修改 draft Pool，不会采纳或部署策略",
        ]
        if "reason" in inputs:
            details.append(f"操作说明：{inputs['reason']}")
        if "placement_mode" in inputs:
            details.append(f"Voting 成员放置方式：{inputs['placement_mode']}")
    elif draft.workflow in {
        "strategy_pool_remove_entry",
        "strategy_pool_set_action",
    }:
        identifier_name = "rule_id" if "rule_id" in inputs else "entry_id"
        operation = (
            "删除条目" if draft.workflow == "strategy_pool_remove_entry" else "修改动作"
        )
        details = [
            f"已识别为〔Strategy Pool {operation} Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            f"目标 {identifier_name}：{inputs[identifier_name]}",
            "平台将从 task 当前 Pool 解析完整条目并绑定 revision/hash，旧 revision 保持不可变",
            "本操作不会采纳或部署策略",
        ]
        if "action" in inputs:
            details.append("新动作：" + _compact_json(inputs["action"]))
        if "reason" in inputs:
            details.append(f"操作说明：{inputs['reason']}")
    elif draft.workflow == "strategy_pool_reorder":
        details = [
            "已识别为〔Strategy Pool 完整重排 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "完整顺序：" + " → ".join(inputs["ordered_ids"]),
            "平台会核对该列表与当前 Pool 是同一组无重复条目；遗漏不会被当作删除",
            "旧 revision 保持不可变，本操作不会采纳或部署策略",
        ]
        if "reason" in inputs:
            details.append(f"操作说明：{inputs['reason']}")
    elif draft.workflow == "strategy_pool_compile":
        details = [
            "已识别为〔Strategy Pool 编译预览 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "平台只读编译当前 task Pool 为 canonical StrategySpec 并计算 design hash",
            "结果只是草案预览，不会创建已采纳策略，也不会采纳或部署",
        ]
    elif draft.workflow == "strategy_pool_materialize":
        details = [
            "已识别为〔Strategy Pool 物化 draft Strategy Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "平台将在计划创建时完整认证当前非空 Pool，并冻结 revision、snapshot、"
            "artifact 与 design hash",
            "本步骤只创建或精确复用持久化 draft Strategy；不采纳、不部署，"
            "后续回测、DSL 交付、采纳和监控仍受各自证据与生命周期门禁约束",
        ]
    elif draft.workflow == "strategy_pool_apply":
        details = [
            "已识别为〔Strategy Pool 应用写回 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "平台将在执行时认证当前 task 内指定类型的唯一非空 Pool、CAS revision/hash、"
            "来源样本与 requirements，再确定性逐行应用",
            "本步骤只创建保留原始行的不可变派生数据集；不激活或替换当前 workspace，"
            "不采纳、不部署，也不改变 Pool",
        ]
        details.append(
            "输出前缀："
            + (
                str(inputs["output_prefix"])
                if "output_prefix" in inputs
                else "由受控 Tool 使用默认前缀"
            )
        )
    elif draft.workflow == "strategy_pool_validation":
        details = [
            "已识别为〔Strategy Pool 独立样本回放验证 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            f"独立样本分区：{inputs['partition']}",
            "平台将在计划创建时恢复当前非空 Pool、精确且成熟的 "
            "StrategySampleDesign V2 membership/bundle、数据/目标/语义与"
            " requirements，并由确定性 Tool 重新认证",
            "结果是 independent replay evidence：审批/拒绝展示动作、风险、"
            "金额和逐月证据，额度/定价/分群保留原生数值或分群分布；"
            "不声称 PSI、稳定性或漂移",
            "本步骤不会修改 Pool，不创建策略，也不晋级、不采纳、不部署",
        ]
    elif draft.workflow == "strategy_pool_stability":
        details = [
            "已识别为〔Strategy Pool 跨分区稳定性 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "平台将在计划创建时冻结当前非空 Pool、最新认证且风险结果成熟的 "
            "StrategySampleDesign V2，以及 development 和全部可用非空 "
            "validation/OOT 分区",
            "Agent 会先生成 exact ImpactCube，再把该步骤的四个精确输出引用"
            "直接交给确定性 Tool 计算 waterfall/action 分布 PSI",
            "结果只表示跨分区分布稳定性，不等于独立效果验证；不会修改 Pool、"
            "创建策略，也不采纳、不晋级、不部署",
        ]
    elif draft.workflow == "strategy_impact_cube":
        details = [
            "已识别为〔统一 Strategy ImpactCube Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            "平台将绑定当前非空 Pool、最新认证 StrategySampleDesign V2、"
            "数据/语义版本和用户明确控制，确定性计算五类类型化影响",
            (
                "分区："
                + (
                    "、".join(inputs["partitions"])
                    if "partitions" in inputs
                    else "全部可用且非空的 development/validation/OOT"
                )
            ),
            "逐月/分组/分群列未显式提供时，只会采用唯一确认的语义角色；"
            "缺失时对应切片 unavailable，冲突时先澄清",
            "结果只发布聚合、可下载、可复核证据；"
            "不会修改 Pool、创建、采纳、晋级或部署策略",
        ]
        for field, label in (
            ("month_col", "月份列"),
            ("group_col", "分组列"),
            ("segment_col", "分群列"),
        ):
            if field in inputs:
                details.append(f"{label}：{inputs[field]}")
        if "current_strategy_id" in inputs:
            details.append(f"当前策略 ID：{inputs['current_strategy_id']}")
        if "economics_inputs" in inputs:
            details.append(
                "经济口径：" + _compact_json(inputs["economics_inputs"])
            )
    elif draft.workflow == "strategy_pool_impact":
        mode_label = (
            "相对基线"
            if inputs["comparison_mode"] == "vs_baseline"
            else "绝对效果"
        )
        details = [
            "已识别为〔Strategy Pool 影响测算 Workflow〕",
            f"策略类型：{inputs['strategy_type']}",
            f"比较口径：{mode_label}",
            "平台将绑定当前非空 Pool、活动 DataWorkspace、数据 hash、"
            "确认的目标列和语义版本，确定性计算级联 waterfall 与风险影响",
            "月份/放款金额/逾期金额列未显式提供时只会采用唯一确认的语义角色；"
            "没有角色则对应结果 unavailable，多个角色会先澄清",
            "本步骤只生成可下载的只读影响证据；不会创建、修改、采纳或部署策略",
        ]
        if "baseline_strategy_id" in inputs:
            details.append(f"基线策略 ID：{inputs['baseline_strategy_id']}")
        for field, label in (
            ("month_col", "月份列"),
            ("loan_amount_col", "放款金额列"),
            ("overdue_amount_col", "逾期金额列"),
        ):
            if field in inputs:
                details.append(f"{label}：{inputs[field]}")
        details.append(
            "空标签处理："
            + (
                "用户明确允许保留样本行、仅从风险分母排除"
                if inputs["drop_nan_labels"]
                else "不默认从风险分母排除"
            )
        )
    else:  # pragma: no cover - validated workflow exhaustiveness
        raise ValueError(f"unsupported standard workflow {draft.workflow}")
    details.append(
        "请确认以上口径。确认后 Agent 只编排受信任工具；所有数字由平台确定性计算。"
    )
    return "；".join(details)


class _DraftValidationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_strategy_request",
        fields: Iterable[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(dict.fromkeys(str(field) for field in fields))


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise _DraftValidationError(f"{key} 必须是非空文本，请重新说明。")
    return value.strip()


def _optional_ratio(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise _DraftValidationError(f"{key} 必须是 0 到 1 之间的有限数字。")
    return number


def _optional_adoption_reason(payload: Mapping[str, Any]) -> str | None:
    if "adoption_reason" not in payload:
        return None
    try:
        return normalize_adoption_reason(payload["adoption_reason"])
    except AdoptionReasonError as exc:
        raise _DraftValidationError(str(exc)) from exc


def _optional_profit(
    payload: Mapping[str, Any], whitelist: tuple[str, ...]
) -> dict[str, Any] | None:
    if "profit" not in payload:
        return None
    profit = payload["profit"]
    if not isinstance(profit, Mapping):
        raise _DraftValidationError("利润参数 profit 必须是一个对象。")
    if any(not isinstance(key, str) for key in profit):
        raise _DraftValidationError("利润参数字段名必须是文本。")
    missing = sorted(_PROFIT_FIELDS - set(profit))
    unexpected = sorted(set(profit) - _PROFIT_FIELDS)
    if missing:
        raise _DraftValidationError("利润参数缺少字段：" + "、".join(missing) + "。")
    if unexpected:
        raise _DraftValidationError(
            "利润参数包含不支持的字段：" + "、".join(unexpected) + "。"
        )
    ead_col = _required_text(profit["ead_col"], name="利润 EAD 列 ead_col")
    pd_col = _required_text(profit["pd_col"], name="利润 PD 列 pd_col")
    for name, column in (("ead_col", ead_col), ("pd_col", pd_col)):
        if column not in whitelist:
            raise _DraftValidationError(
                f"利润参数 {name} 使用了数据集中不存在的列「{column}」，请从列白名单选择。"
            )
    annual_rate = _bounded_number(
        profit["annual_rate"], name="利润 annual_rate", maximum=1
    )
    funding_rate = _bounded_number(
        profit["funding_rate"], name="利润 funding_rate", maximum=1
    )
    lgd = _bounded_number(profit["lgd"], name="利润 lgd", maximum=1)
    operating_cost = _bounded_number(
        profit["operating_cost_per_loan"],
        name="利润 operating_cost_per_loan",
    )
    term_months = profit["term_months"]
    if (
        isinstance(term_months, bool)
        or not isinstance(term_months, int)
        or term_months < 1
    ):
        raise _DraftValidationError("利润 term_months 必须是大于等于 1 的整数。")
    return {
        "ead_col": ead_col,
        "pd_col": pd_col,
        "annual_rate": annual_rate,
        "funding_rate": funding_rate,
        "lgd": lgd,
        "operating_cost_per_loan": operating_cost,
        "term_months": term_months,
    }


def _validate_economics_field_ownership(
    payload: Mapping[str, Any], *, strategy_type: str
) -> None:
    if "profit" in payload and strategy_type not in {"approval", "reject"}:
        raise _DraftValidationError(
            "profit 只适用于审批或拒绝策略；额度和定价策略请使用 economics_inputs，"
            "分群策略不接受经济参数。"
        )
    if "economics_inputs" in payload and strategy_type not in {"limit", "pricing"}:
        raise _DraftValidationError(
            "economics_inputs 只适用于额度或定价策略；审批和拒绝策略请使用 profit，"
            "分群策略不接受经济参数。"
        )


def _validate_candidate_field_ownership(
    payload: Mapping[str, Any],
    *,
    operation: str,
    strategy_type: str,
) -> None:
    has_candidate = "candidate_design" in payload
    has_spec = "strategy_spec" in payload
    if has_candidate and has_spec:
        raise _DraftValidationError(
            "candidate_design 与 strategy_spec 必须二选一；LLM 不得同时提交候选输入和规则结果。",
            code="candidate_spec_mutually_exclusive",
            fields=("candidate_design", "strategy_spec"),
        )
    if has_candidate and (
        operation != "develop"
        or strategy_type not in {"limit", "pricing", "segmentation"}
    ):
        raise _DraftValidationError(
            "candidate_design 只适用于 limit、pricing、segmentation 的 develop 请求。",
            code="candidate_design_not_allowed",
            fields=("candidate_design",),
        )
    if has_spec and strategy_type in {"limit", "pricing", "segmentation"}:
        raise _DraftValidationError(
            "非审批策略的 Strategy DSL 必须由平台候选设计工具确定性生成；"
            "LLM 不得提交 strategy_spec、动作值或推荐结果。",
            code="llm_strategy_spec_forbidden",
            fields=("strategy_spec",),
        )
    if (
        operation == "develop"
        and strategy_type in {"limit", "pricing", "segmentation"}
        and not has_candidate
    ):
        raise _DraftValidationError(
            f"开发{_TYPE_LABELS[strategy_type]}需要 candidate_design；"
            "请补充候选列、候选网格和必要业务约束，平台再确定性生成规则。",
            code="candidate_design_required",
            fields=("candidate_design",),
        )


def _optional_economics_inputs(
    payload: Mapping[str, Any],
    *,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "economics_inputs" not in payload:
        return None
    raw_inputs = payload["economics_inputs"]
    if not isinstance(raw_inputs, Mapping):
        raise _DraftValidationError("经济参数 economics_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        raise _DraftValidationError("经济参数 economics_inputs 的字段名必须是文本。")

    names = (
        _LIMIT_ECONOMICS_NAMES if strategy_type == "limit" else _PRICING_ECONOMICS_NAMES
    )
    allowed_fields = {key for name in names for key in (f"{name}_col", f"{name}_value")}
    unexpected = sorted(set(raw_inputs) - allowed_fields)
    if unexpected:
        raise _DraftValidationError(
            f"{_TYPE_LABELS[strategy_type]}经济参数包含不支持的字段："
            + "、".join(unexpected)
            + "。"
        )

    normalized: dict[str, Any] = {}
    missing: list[str] = []
    for name in names:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        has_column = column_key in raw_inputs
        has_value = value_key in raw_inputs
        if has_column and has_value:
            raise _DraftValidationError(
                f"经济参数 {name} 必须在 {column_key} 和 {value_key} 中二选一，不能同时提供。",
                code="candidate_economics_ambiguous",
                fields=(column_key, value_key),
            )
        if not has_column and not has_value:
            missing.append(f"{column_key}/{value_key}")
            continue
        if has_column:
            column = _required_text(
                raw_inputs[column_key],
                name=f"经济参数 {column_key}",
            )
            if column not in whitelist:
                raise _DraftValidationError(
                    f"经济参数 {column_key} 使用了数据集中不存在或不可用于策略的列"
                    f"「{column}」，请从列白名单选择。"
                )
            normalized[column_key] = column
            continue
        normalized[value_key] = _economics_value(name, raw_inputs[value_key])

    if missing:
        raise _DraftValidationError(
            f"{_TYPE_LABELS[strategy_type]}经济参数不完整，缺少："
            + "、".join(missing)
            + "。",
            code="candidate_economics_incomplete",
            fields=missing,
        )
    return normalized


def _economics_value(name: str, value: object) -> float:
    label = _ECONOMICS_LABELS[name]
    if name == "term_months":
        number = _bounded_number(value, name=f"经济参数 {label}")
        if number <= 0:
            raise _DraftValidationError(f"经济参数 {label} 必须是大于 0 的有限数字。")
        return number
    return _bounded_number(
        value,
        name=f"经济参数 {label}",
        maximum=_ECONOMICS_VALUE_MAXIMUMS.get(name),
    )


def _economics_confirmation(draft: StrategyRequestDraft) -> str:
    assert draft.economics_inputs is not None
    names = (
        _LIMIT_ECONOMICS_NAMES
        if draft.strategy_type == "limit"
        else _PRICING_ECONOMICS_NAMES
    )
    items: list[str] = []
    for name in names:
        column_key = f"{name}_col"
        value_key = f"{name}_value"
        label = _ECONOMICS_LABELS[name]
        if column_key in draft.economics_inputs:
            items.append(f"{label} 取数据列 {draft.economics_inputs[column_key]}")
        else:
            value = draft.economics_inputs[value_key]
            if name in _ECONOMICS_VALUE_MAXIMUMS:
                items.append(f"{label} 取固定值 {value:.2%}")
            elif name == "term_months":
                items.append(f"{label} 取固定值 {value:g} 个月")
            else:
                items.append(f"{label} 取固定值 {value:g}")
    return f"{_TYPE_LABELS[draft.strategy_type]}经济参数：" + "，".join(items)


def _candidate_design_confirmation(draft: StrategyRequestDraft) -> str:
    assert draft.candidate_design is not None
    design = draft.candidate_design
    if draft.strategy_type == "limit":
        details = (
            f"评分列 {design['score_col']}，固定等频 {design['n_bands']} 箱，"
            "候选额度 "
            + "、".join(f"{value:g}" for value in design["limit_grid"])
            + f"，单户预期损失预算 {design['max_expected_loss_per_account']:g}"
        )
    elif draft.strategy_type == "pricing":
        details = (
            f"评分列 {design['score_col']}，固定等频 {design['n_bands']} 箱，"
            "候选年利率 "
            + "、".join(f"{value:.2%}" for value in design["rate_grid"])
            + f"，最小 ROA {design['min_roa']:.2%}"
        )
    else:
        details = (
            f"单变量列 {design['feature_col']}，固定等频 {design['n_bands']} 箱，"
            "风险标签由平台按样本坏率稳定生成"
        )
    return (
        f"候选设计输入：{details}；缺失策略 {design['missing_policy']}。"
        "此处只确认搜索空间和业务口径，推荐动作、规则与指标尚未生成，"
        "将由平台确定性计算。"
    )


def _optional_candidate_design(
    payload: Mapping[str, Any],
    *,
    operation: str,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "candidate_design" not in payload:
        return None
    if operation != "develop":
        raise _DraftValidationError(
            "candidate_design 只适用于 develop 请求。",
            code="candidate_design_not_allowed",
            fields=("candidate_design",),
        )
    try:
        return normalize_candidate_design(
            strategy_type,
            payload["candidate_design"],
            allowed_columns=whitelist,
        )
    except CandidateDesignError as exc:
        raise _DraftValidationError(
            str(exc),
            code=exc.code,
            fields=exc.fields,
        ) from exc


def _optional_strategy_spec(
    payload: Mapping[str, Any],
    *,
    strategy_type: str,
    whitelist: tuple[str, ...],
) -> dict[str, Any] | None:
    if "strategy_spec" not in payload:
        return None
    if strategy_type not in {"approval", "reject"}:
        raise _DraftValidationError(
            "非审批策略的 strategy_spec 必须由平台确定性生成，LLM 不得提交。",
            code="llm_strategy_spec_forbidden",
            fields=("strategy_spec",),
        )
    raw_spec = payload["strategy_spec"]
    if not isinstance(raw_spec, Mapping):
        raise _DraftValidationError("策略规则草案 strategy_spec 必须是一个对象。")
    raw_metadata = raw_spec.get("metadata", {})
    if raw_metadata not in ({}, {"lineage": {}}):
        raise _DraftValidationError(
            "策略规则草案 metadata 由平台生成，LLM 不得写入指标结果或其他元数据。"
        )
    try:
        parsed = parse_strategy_spec(raw_spec)
    except (StrategyError, TypeError, ValueError) as exc:
        raise _DraftValidationError(
            "策略规则草案格式或取值无效，请检查规则条件、优先级和动作。"
        ) from exc
    if parsed.strategy_type != strategy_type:
        raise _DraftValidationError(
            "strategy_spec 的 strategy_type 必须与请求中的策略类型一致。"
        )
    unknown_columns = sorted(
        {
            field
            for rule in parsed.rules
            for field in _condition_fields(rule.condition)
            if field not in whitelist
        }
    )
    if unknown_columns:
        rendered = "、".join(f"「{column}」" for column in unknown_columns)
        raise _DraftValidationError(
            f"策略条件使用了数据集中不存在的列 {rendered}，请从列白名单选择。"
        )
    return parsed.to_dict()


def _condition_fields(condition: Mapping[str, Any]) -> tuple[str, ...]:
    op = condition["op"]
    if op in {"compare", "between", "is_null", "is_not_null"}:
        return (condition["field"],)
    if op in {"and", "or", "n_of_k"}:
        return tuple(
            field
            for argument in condition["args"]
            for field in _condition_fields(argument)
        )
    if op == "not":
        return _condition_fields(condition["arg"])
    return ()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _DraftValidationError(f"{name} 必须是非空文本。")
    return value.strip()


def _bounded_number(
    value: object,
    *,
    name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _DraftValidationError(f"{name} 必须是有限数字。")
    number = float(value)
    if (
        not math.isfinite(number)
        or number < 0
        or (maximum is not None and number > maximum)
    ):
        if maximum is None:
            raise _DraftValidationError(f"{name} 必须是大于等于 0 的有限数字。")
        raise _DraftValidationError(f"{name} 必须是 0 到 {maximum:g} 之间的有限数字。")
    return number


def _column_whitelist(
    allowed_columns: Iterable[str] | None,
) -> tuple[str, ...]:
    if allowed_columns is None:
        return ()
    if isinstance(allowed_columns, str):
        values = (allowed_columns,)
    else:
        try:
            values = tuple(allowed_columns)
        except TypeError:
            return ()
    return tuple(sorted({column for column in values if isinstance(column, str)}))


def _normalized_target_col(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _user_prompt(
    utterance: str,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
) -> str:
    return (
        "【数据集列白名单】\n"
        f"{json.dumps(list(whitelist), ensure_ascii=False)}\n"
        "【任务当前目标列（仅可作为 limit_pricing_matrix 的 target_col 风险来源，"
        "禁止用于策略规则）】\n"
        f"{json.dumps(target_col, ensure_ascii=False)}\n"
        "【用户策略请求】\n"
        f"{utterance}\n"
        "只输出结构化策略草案或一个中文 clarification，不要输出任何指标结果。"
        "对于 limit/pricing/segmentation 的 develop 请求，只能抽取 candidate_design "
        "搜索空间与用户明确给出的 economics_inputs；禁止输出 strategy_spec、规则、"
        "动作、默认动作、推荐值或计算指标。缺少必要经济口径时只返回 clarification。"
        "对于 strategy_project_context，只能逐字抄录用户明确提供的截止日 as_of、"
        "可选 scope、business_context 文本/null、明确暂缺字段路径和任务 source_dir 下"
        "用户点名的外部报告相对文件名。禁止输出 revision/CAS、message id/hash、"
        "dataset/Pool/backtest/monitoring 引用、artifact id/hash、来源引用、可用性结论或指标。"
        "一次只刷新项目上下文，不得串联样本、候选、影响、报告、采纳或部署；没有 as_of"
        "必须 clarification，不能默认今天。"
        "对于 strategy_sample_design_v2，workflow_inputs 必须精确包含用户明确提供的"
        " target_bad_value、drop_nan_labels、relationship、approval_population、"
        "risk_population、partitioning、maturity、performance_window、observation_window、"
        "field_bindings、historical_score。relationship 必须由用户明确说明为"
        " nested_same_cohort 或 parallel_time_cohorts。所有嵌套字段、null、状态、列、值、"
        "方向与 reason 都必须能逐字回到原话。fresh population 的非 null inclusion/exclusion"
        " 只能输出 match=all/any 和 1 到 8 个简单 conditions；condition 只含"
        " column/operator/value，is_null/is_not_null 不含 value。禁止直接输出 population"
        " predicate AST。approval/risk、inclusion/exclusion 与 operator/column/value"
        " 必须在各自局部语境中逐项对应；表现窗、观察窗和 maturity cutoff 日期不得互相借用。"
        "普通表现窗不能只借用成熟表现窗；maturity cutoff 必须在同一子句带成熟度限定；"
        "historical_score 的字段和方向必须在同一局部子句绑定，不能借用其他字段的方向。"
        "平台会把 nested_same_cohort、双总体均无纳排且同列三个互异简单等值 selector"
        " 路由到 compatibility 链；parallel_time_cohorts、time_ranges、任一总体纳排或"
        "复杂 selector 路由到原生 V2，禁止删减或降级。"
        "scope/policy、legacy ref、dataset/workspace/target、membership/bundle/"
        "artifact id/hash 全部禁止填写，也不得串联建模、模型比较、报告、Strategy Pool、"
        "采纳或部署。"
        "对于 strategy_model_evidence_v2，workflow_inputs 必须是空对象；只汇总当前 task"
        " 已有认证单变量候选。所有 SampleDesign/candidate/artifact 引用由平台绑定；训练、"
        "模型对比、月度/OOT/验证模型、报告、采纳或部署的正向请求必须 clarification；"
        "明确说不做这些后续动作可以接受；“此前/已有/已认证”只有位于当前归集动作之后时"
        "才视为来源状态；“此前未汇总”“没有汇总”“有没有汇总”必须 clarification。"
        "对于 strategy_report_bundle_v2，workflow_inputs 只能包含用户明确提供的 title "
        "和 status；status 仅允许 draft/partial/final。用户未提供时必须使用固定默认值"
        " title=策略迭代评审报告、status=partial。ProjectContext、SampleDesign、Pool、"
        "ImpactCube/兼容 PoolImpact、ModelEvidence/training/score、strategy identity、"
        "report revision/previous head CAS、generated_at、artifact id/hash、来源引用和"
        "所有指标必须省略，由平台在计划创建时精确绑定。报告可在原话中点名 approval/"
        "reject/limit/pricing/segmentation，但 strategy_type 也必须省略。报告请求必须是"
        "当前、肯定、单步骤命令；问句、否定、"
        "假设、演示、仅历史描述，或同轮串联训练、评分、候选、影响测算、采纳、部署、"
        "上线时必须 clarification。"
        "对于 strategy_dsl_delivery，workflow_inputs 只能包含用户原话中唯一完整的"
        "可选 strategy_id；没有 ID 时必须省略，由平台仅在当前任务只有一个可交付策略时"
        "唯一绑定。strategy_ref、策略类型/version/spec hash、dataset_ref、数据 hash、"
        "workspace_ref/revision/generation/semantic hash、"
        "maximum_equivalence_rows、artifact id/hash、等价结果和所有指标均由平台绑定，"
        "禁止输出。请求必须是当前、肯定、单步骤导出命令；问句、否定、假设、演示、"
        "仅历史描述，或同轮串联应用、写回、报告、影响测算、训练、评分、采纳、晋级、"
        "部署时必须 clarification。"
        "对于 automatic_tree_candidate_build，只能抄录用户明确提供的 features、"
        "权重/金额字段、方向和树参数；不得填写平台拥有的数据绑定、目标列、标签策略、"
        "预算、结果、叶子、动作、排名或推荐，也不得串联选叶或 Strategy Pool。"
        "对于 automatic_tree_apply，只能逐字抄录用户原话中唯一完整的 tree_asset_id；"
        "leaf_id_column 和 rule_id_column 只有在用户分别明确标注叶节点列/规则列时才能"
        "抄录，未提供时必须省略并由受控 Tool 使用默认值。不得填写 source artifact、"
        "artifact hash、asset hash、tree result hash、dataset/hash、workspace lineage、"
        "activate_result、结果或指标。它只创建 development / unvalidated 的不可变派生"
        "数据集，不激活当前 workspace；不得串联选叶、Strategy Pool、业务动作、报告、"
        "采纳或部署。"
        "对于 automatic_tree_leaf_materialization，只能逐字抄录用户原话中唯一的"
        "完整 tree_asset_id、唯一的完整 leaf_id，以及用户用“选择理由/理由/原因/说明”"
        "显式标注时的逐字 selection_reason；未显式标注时必须省略。它只创建"
        "pointer，不得复制规则、条件、指标、动作或平台 artifact/hash，也不得串联"
        "Strategy Pool、业务动作、采纳、部署或 leaf ID 写回。selection_reason 中也"
        "不得藏入理由替换、后续动作、生命周期操作或极值/排名选叶语义；它只接受"
        "人工/业务/风险/合规/样本评审依据类短说明。"
        "对于 interactive_tree_revision，只能逐字抄录用户当前肯定命令中唯一完整的"
        " source_tree_id（candidate-asset- 或 interactive-tree-revision- 后接 32 位"
        "小写十六进制）、唯一完整的 split node_id（node- 后接 20 位小写十六进制）、"
        "唯一 operation=prune_subtree 或 adjust_split_threshold，以及用户显式标注时"
        "逐字一致的 reason。adjust_split_threshold 还必须逐字抄录用户明确给出的唯一"
        "有限数值 threshold；prune_subtree 必须省略 threshold。“调好一点”“最佳阈值”"
        "“自动优化”“全部节点”等模糊、推荐或批量修改请求必须 clarification。不得输出"
        "artifact/hash、父链、tree/frontier/condition/metrics、dataset/workspace/"
        "SampleDesign 或重放结果，这些均由平台恢复。不得按最好、风险最高、不稳定或"
        "代词替用户选节点，也不得同轮串联另一种树编辑、前沿物化、入池、业务动作、"
        "自动继续、整树应用、报告、采纳、部署或写回。"
        "对于 interactive_tree_frontier_group_materialization，只能逐字抄录用户"
        "当前肯定命令中唯一完整的 revision_id（interactive-tree-revision- 后接 "
        "32 位小写十六进制）、2 到 50 个互不重复的完整 source_node_ids（node- "
        "或 leaf- 后接 20 位小写十六进制），以及用户显式标注时逐字一致的 "
        "selection_reason。用户必须明确 OR/逻辑或/任一成员命中语义；成员输入顺序"
        "不具有语义，平台按 revision frontier 顺序规范化。不得输出 selection/group/"
        "revision artifact、hash、父链、semantic tree、fragment/rule/effect、condition/"
        "metrics、dataset/workspace/SampleDesign 或动作。不得用全部、最好、最差、"
        "风险最高、自动排名或代词替用户选节点，也不得同轮串联 Strategy Pool、"
        "业务动作、应用、采纳、部署或写回。"
        "对于 interactive_tree_frontier_materialization，只能逐字抄录用户当前肯定"
        "命令中唯一完整的 revision_id（interactive-tree-revision- 后接 32 位小写"
        "十六进制）、唯一完整的 source_node_id（node- 或 leaf- 后接 20 位小写"
        "十六进制），以及用户显式标注时逐字一致的 selection_reason。不得输出"
        "selection/revision artifact、hash、父链、semantic tree、fragment/rule/effect、"
        "condition/metrics、dataset/workspace/SampleDesign 或动作，这些均由平台恢复。"
        "不得按最好、最差、风险最高、自动排名或代词替用户选节点，也不得同轮串联"
        "Strategy Pool、业务动作、采纳、部署或写回。"
        "对于 voting_candidate_search，必须逐字抄录用户明确提供的 strategy_type、"
        "member_count/K、n 和 objective metric+direction；中文指标只能采用 system "
        "prompt 明列的别名并输出对应 canonical metric，禁止自行扩展近义词；"
        "constraints、include_rule_ids、"
        "exclude_rule_ids 未提供时固定为空数组，max_combinations 未提供时固定为 10000。"
        "include/exclude 只能采用当前句在对应标签后完整给出的 candidate-rule ID。"
        "最小化 bad_rate/weighted_bad_rate/bad_amount_rate 时，必须分别提供正数 "
        "hit_share/weighted_hit_share/hit_amount_share 的 gte 约束，绝对命中量不能"
        "替代。不得输出 Pool ref/revision/hash、dataset/target、逐行矩阵/target/weights/"
        "amounts、artifact、result 或已计算排名。该步骤只搜索，不构建、不选择、不修改"
        "或加入 Pool、不应用、不采纳、不部署；搜索/查找/优化 Voting 组合的原话必须优先"
        "路由到本 Workflow 或 clarification。"
        "对于 voting_candidate_build_from_search，只能逐字抄录用户当前请求中唯一完整"
        "的 search_id、唯一完整的 combo_id 和可选的唯一 strategy_type；strategy_type"
        " 未明确时必须省略。不得输出 artifact/hash、rule/entry/member IDs、n、rank、"
        "winner/champion、指标、结果或 Pool 身份，也不得按第一名、最好、Top N、"
        "刚才那个等表述选择组合。该步骤只构建候选；同轮串联入池、修改 Pool、设置"
        "动作、应用、采纳、部署或写回时必须 clarification。它必须优先于重新搜索和"
        "自由 rule ID Voting 构建路由。"
        "对于 voting_candidate_build，只能逐字抄录用户明确标注的 strategy_type、"
        "2 到 50 个完整 candidate-rule ID 和整数 n；不得输出 entry_id、Pool revision/hash、"
        "condition、指标、动作、推荐或平台数据绑定。规则集合必须全部来自同一条正向"
        "Voting/n-of-k 构建命令；‘最好规则’‘刚才那些’等启发式引用，或同一句串联入池、"
        "动作、采纳、部署、写回时必须澄清。问句、假设/未来/历史描述、演示文本、句尾"
        "撤销以及多个 strategy_type/n 候选也必须澄清；显式 k 必须与 rule_ids 数量一致。"
        "对于 candidate_monthly_stability，workflow_inputs 必须严格二选一：只抄录"
        "用户原话中唯一完整的 asset_id，或只抄录用户明确的 strategy_type 与唯一完整"
        "entry_id。禁止输出 source_kind、source artifact/hash、asset hash、Pool "
        "revision/hash、dataset/workspace/semantic、SampleDesign、target、month_col、"
        "基准、指标或结果；这些值由平台在 preflight 恢复。代词、多个 pointer、"
        "问句、否定、历史/未来/假设描述，或同轮串联入池、删改、重排、编译、写回、"
        "报告、采纳、部署时必须 clarification。"
        "对于 cross_matrix_candidate_search，只能逐字抄录用户唯一 "
        "features=[...] 列表中的 2 到 20 个白名单字段和明确的 1..190 "
        "max_pairs；两者都必须显式提供。不得填写轴方法、source artifact/"
        "candidate/evidence hash、dataset/target/sample、candidate asset、"
        "pair/rank/winner/champion、指标或结果。平台绑定精确父单变量证据和"
        " risk/development 样本，并由 Tool 为每个字段选择父证据中最高排名"
        "的可用方法。本步骤只搜索，不构建、不选择、不入池、不应用、不采纳、"
        "不部署；同轮后续动作必须 clarification。"
        "对于 cross_matrix_candidate_build_from_search，只能逐字抄录当前"
        "请求中唯一完整的 cross-search ID 与唯一完整的 cross-pair ID。不得"
        "输出 artifact/hash、轴字段/方法、asset fingerprint、rank/winner/"
        "champion、指标或结果，也不得按第一名、最好、Top N、刚才那个或代词"
        "选择。它必须是后续独立的单步构建请求；同轮重新搜索、入池、设置动作、"
        "应用、采纳、部署或写回必须 clarification。"
        "对于 cross_rule_search，只能逐字抄录用户唯一 features=[...] 列表中"
        "的 2 到 12 个白名单字段、dimension=2/3、完整 constraints"
        "（min_lift、min_bad_count、max_hit_share、min_amount_lift）和 1..5000 "
        "max_trials；所有控制都必须在当前请求显式提供。不得填写 source "
        "artifact/hash、dataset/target/sample、阈值、方向、rule/rank/winner/"
        "champion、指标或结果。平台从最新认证单变量证据恢复阈值和风险方向，"
        "并在 risk/development 样本做有预算的 2D/3D 聚合搜索。本步骤只搜索和"
        "排序全部已评估证据，不自动选择、不构建、不入池、不应用、不采纳、不部署。"
        "对于 cross_rule_candidate_build_from_search，只能逐字抄录当前请求中"
        "唯一完整的 cross-rule-search ID、唯一完整的 cross-rule ID，以及用户"
        "显式标注时的 selection_reason；未显式标注必须省略。不得输出 artifact/"
        "hash、条件、阈值、方向、rank/winner/champion、指标或结果，也不得按"
        "第一名、最好、Top N 或代词选择。它只精确构建一个未验证候选；入池、"
        "动作、应用、采纳或部署必须另发请求。"
        "对于 cross_matrix_analysis，只能抄录两个明确轴字段、各自方法及用户明确给出的"
        "单变量分析参数；不得输出平台数据绑定、目标列、预算、边界、cell、condition、"
        "指标、artifact/asset/effect/rule id、动作或推荐。它只构建二维矩阵证据，不能"
        "串联选格、入池、代码、写回、采纳或部署；明确的二维 Cross Matrix 请求不能"
        "改路由到其他 Workflow。"
        "对于 cross_matrix_cell_selection，只能逐字抄录用户原话中唯一完整的"
        "cross_asset_id、1 到 400 个互不重复的完整 cross-cell ID，以及显式标注时"
        "逐字一致的 selection_reason。禁止代词、排名、Top N、风险/指标极值或阈值"
        "选格。多个 cell 是集合语义并由平台按源矩阵顺序归一化为确定性 OR；不得输出"
        "condition、rule、effect、metrics、action 或任何 artifact/hash 平台绑定。它只"
        "创建 pointer，不得串联 Strategy Pool、业务动作、采纳、部署、投产或写回。"
        "对于 strategy_pool_add_candidate，candidate_asset_id 与 selection_id "
        "严格二选一且必须逐字抄录唯一完整 ID；必须分别抄录显式的策略池类型、"
        "Pool 默认动作和命中动作标签，不能对调或从动作反推 Pool 类型。reason 仅在"
        "显式标注时逐字抄录，未标注时省略；默认/命中 reason_code、output_value 和"
        "value 也必须逐字归属各自标签，不得省略或对调。可选 placement_mode 只能"
        "逐字抄录 before_selected_members/replace_selected_members，或从“保留成员"
        "作为回退并放在成员前/由 Voting 替代成员”二选一映射；用户未提供时省略。"
        "否定入池或串联采纳/部署时"
        "必须澄清。selection_id 只允许 automatic-tree-leaf-selection-、"
        "interactive-tree-frontier-group-selection-、"
        "interactive-tree-frontier-selection-、cross-matrix-cell-selection- 后接 "
        "32 位小写十六进制；完整 Cross Matrix asset"
        "不能直接入池。source ID 必须与唯一正向入池命令位于同一子句，不能从否定子句、"
        "reason、引用或代词上下文借用；未来/条件指令、问句、how-to、演示和测试也"
        "必须澄清。"
        "对于 strategy_pool_stability，只能逐字抄录用户当前肯定命令中唯一明确"
        "的五类 strategy_type。partitions、exact ImpactCube/Pool/SampleDesign "
        "artifact、revision/hash、dataset/workspace/target、阈值、PSI、分布、"
        "指标与结果全部禁止填写，由平台在计划创建和两步 Workflow 执行时冻结或"
        "确定性计算。否定、问句、历史/未来/假设、仅报告，或同轮修改 Pool、应用、"
        "创建、采纳、晋级、部署时必须 clarification。结果只是跨分区分布稳定性，"
        "不是独立效果验证，也不会修改 Pool 或进入策略生命周期。"
        "对于 strategy_impact_cube，只能抄录用户明确的五类 strategy_type、"
        "可选 development/validation/oot partitions、精确 month_col/group_col/"
        "segment_col、完整 current_strategy_id，以及 typed economics_inputs"
        "（每项仅 column 或有限 scalar）。不得输出 Pool/SampleDesign artifact、"
        "revision/hash、population、target、metrics、condition 或 strategy_spec。"
        "用户未指定分区时省略，由平台选择最新样本设计中全部非空可用分区；"
        "用户未指定维度列时省略，由平台仅绑定唯一确认语义角色。任何原话控制被遗漏、"
        "替换、否定，或同轮串联写回、报告、采纳、晋级、部署时必须澄清。"
        "对于 strategy_dsl_delivery，只能逐字抄录用户原话中唯一完整的可选"
        " strategy_id；未点名时必须省略。不得输出 strategy_ref、策略类型/version/"
        "spec hash、dataset_ref/hash、workspace_ref/revision/generation/"
        "semantic hash、等价样本预算、artifact id/hash、代码内容、"
        "等价结果或指标。它只导出离线 Python/SQL/JSON 与明确标注范围的等价证据，"
        "不得串联应用、写回、报告、影响测算、训练、评分、采纳、晋级或部署。"
        "对于 strategy_pool_impact，只能抄录用户明确的 approval/reject Pool 类型，"
        "可选 absolute/vs_baseline 比较模式、完整 baseline_strategy_id、精确 month_col/"
        "loan_amount_col/overdue_amount_col 和明确的 drop_nan_labels 布尔授权。普通肯定式"
        "请求可默认 absolute；vs_baseline 必须同时有原话中的比较表达和完整基线 ID。"
        "禁止输出 dataset/target、Pool revision/hash、workspace、sample binding、semantic hash、"
        "metrics、conditions 或 strategy_spec。用户未指定月份/金额列时必须省略，平台只会"
        "使用唯一确认的语义角色；没有角色则 unavailable，多个角色则澄清，Agent 不得猜列。"
        "limit/pricing/segmentation、否定/问句/历史/仅报告或同轮修改/采纳/部署必须澄清。"
        "对于 strategy_pool_validation，只能逐字抄录用户当前肯定命令中唯一"
        "明确的五类 strategy_type 和 validation/oot partition。"
        "Pool ref/revision/hash/artifact、SampleDesign membership/bundle/ref、"
        "dataset/workspace/target、requirements、population、comparison_mode、"
        "指标、月份、状态与结果全部禁止填写，由平台在计划创建和 Tool 执行时恢复。"
        "development、缺少或多个类型/分区、问句、"
        "否定、历史/未来/假设，或同轮修改 Pool、应用、报告、晋级、采纳、部署"
        "必须 clarification。它只发布 native typed independent replay evidence，"
        "不得声称 "
        "PSI、stability 或 drift，也不会修改 Pool、创建、晋级、采纳或部署策略。"
        "对于 strategy_pool_apply，只能抄录用户当前肯定命令中唯一明确的五类 "
        "strategy_type，以及用户以“输出前缀/output_prefix/output prefix/prefix”"
        "显式标注的可选 ASCII identifier output_prefix；未提供时必须省略并由 Tool"
        " 使用默认值。expected Pool revision/snapshot hash、Pool/artifact、dataset、"
        "SampleDesign、requirements、StrategySpec、指标和生命周期状态全部禁止填写，"
        "由平台在计划创建与执行时恢复。请求必须明确把当前 Pool 应用或写回当前样本，"
        "且必须是当前、肯定、单步骤命令；否定、问句、历史/未来/假设、模糊或多 Pool，"
        "或同轮串联 Pool 修改、采纳、激活、部署、上线、导出或报告必须 clarification。"
        "结果只创建不可变派生数据集，不激活当前 workspace，不采纳、不部署。"
        "对于 strategy_pool_materialize，只能抄录用户当前肯定命令中唯一明确的"
        "五类 strategy_type。Pool revision/snapshot hash、Pool artifact id/content "
        "hash、design hash、StrategySpec、requirements、指标和 lifecycle 全部禁止"
        "填写，由平台在计划创建和 Tool 执行时恢复。请求必须明确把当前 Pool 物化/"
        "固化/创建为 draft Strategy，且必须是当前、肯定、单步骤命令；否定、问句、"
        "历史/未来/假设、模糊或多 Pool，或同轮串联采纳、部署、回测、应用、报告、"
        "监控或 DSL 导出必须 clarification。本步骤只创建 draft Strategy，不采纳、"
        "不部署，也不声称后续 readiness。"
    )


def _repair_prompt(prompt: str, *, raw: object, error: str) -> str:
    if isinstance(raw, Mapping):
        raw_text = json.dumps(raw, ensure_ascii=False, default=str)
    else:
        raw_text = str(raw)
    raw_text = raw_text[:4000]
    return (
        f"{prompt}\n\n"
        "【上一次输出未通过平台校验】\n"
        f"错误：{error}\n"
        f"上一次输出：{raw_text}\n"
        "这是唯一一次修复机会。请删除未知字段、修正类型/范围/列名；"
        "不能确定时只返回中文 clarification。仍然禁止输出任何指标结果。"
    )


def _invalid(
    message: str,
    *,
    code: str = "invalid_strategy_request",
    fields: Iterable[str] = (),
) -> _ValidationOutcome:
    return _ValidationOutcome(
        _clarification(message, code=code, fields=fields),
        False,
        message,
    )


def _clarification(
    message: str,
    *,
    code: str = "clarification_required",
    fields: Iterable[str] = (),
) -> StrategyRequestCompilation:
    return StrategyRequestCompilation(
        draft=None,
        clarification=message,
        confirmation=None,
        clarification_code=code,
        clarification_fields=tuple(dict.fromkeys(str(field) for field in fields)),
    )


def _chinese_clarification(message: str) -> str:
    normalized = message.strip()
    if _CJK_RE.search(normalized):
        return normalized
    return "请补充更明确的策略操作、策略类型和相关策略对象。"


_OPERATION_LABELS = {
    "develop": "开发",
    "analyze": "分析",
    "backtest": "回测",
    "apply": "应用",
    "compare": "对比",
    "adopt": "采纳",
    "report": "生成报告",
    "monitor": "监控",
    "mine_rules": "规则挖掘",
}
_TYPE_LABELS = {
    "approval": "审批策略",
    "reject": "拒绝策略",
    "limit": "额度策略",
    "pricing": "定价策略",
    "segmentation": "分群策略",
}


__all__ = [
    "CompiledStrategyRequestDraft",
    "FRESH_STANDARD_STRATEGY_WORKFLOWS",
    "LEGACY_REPLAY_STANDARD_STRATEGY_WORKFLOWS",
    "REPLAYABLE_STANDARD_STRATEGY_WORKFLOWS",
    "STANDARD_STRATEGY_WORKFLOWS",
    "UNIVARIATE_BINNING_METHODS",
    "UNIVARIATE_REFINEMENT_METHODS",
    "STRATEGY_REQUEST_KINDS",
    "STRATEGY_OPERATIONS",
    "STRATEGY_REQUEST_JSON_SCHEMA",
    "STRATEGY_TYPES",
    "StrategyRequestCompilation",
    "StrategyRequestDraft",
    "StandardWorkflowRequestDraft",
    "compile_strategy_request",
    "strategy_request_confirmation_text",
    "utterance_targets_candidate_monthly_stability",
    "utterance_targets_interactive_tree_frontier_group_materialization",
    "utterance_targets_interactive_tree_frontier_materialization",
    "utterance_targets_scorecard_band_build",
    "utterance_targets_scorecard_cutoff_selection",
    "utterance_targets_strategy_dsl_delivery",
    "utterance_targets_strategy_pool_materialize",
    "utterance_targets_strategy_pool_stability",
    "utterance_targets_strategy_project_context",
    "utterance_targets_strategy_report_bundle_v2",
    "utterance_targets_strategy_sample_design",
    "validate_strategy_request",
]
