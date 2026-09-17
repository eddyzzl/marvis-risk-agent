"""cross request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import json
import math
import re
from typing import Any
import unicodedata

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE
    from . import _AUTOMATIC_TREE_LEAF_ACTION_CHAIN_RE
    from . import _AUTOMATIC_TREE_LEAF_LIFECYCLE_CHAIN_RE
    from . import _AUTOMATIC_TREE_LEAF_NEGATED_REASON_CLAUSE_RE
    from . import _AUTOMATIC_TREE_LEAF_POOL_CHAIN_RE
    from . import _AUTOMATIC_TREE_LEAF_RATIONALE_DECISION_SUBJECT_RE
    from . import _AUTOMATIC_TREE_LEAF_REASON_EXTREME_RE
    from . import _AUTOMATIC_TREE_LEAF_REASON_FORBIDDEN_OPERATION_RE
    from . import _AUTOMATIC_TREE_LEAF_REASON_RE
    from . import _AUTOMATIC_TREE_LEAF_REASON_REPLACEMENT_RE
    from . import _AUTOMATIC_TREE_LEAF_REQUEST_PUNCTUATION_RE
    from . import _AUTOMATIC_TREE_LEAF_WRITEBACK_CHAIN_RE
    from . import _automatic_tree_column_mention_resolution
    from . import _automatic_tree_column_mentions
    from . import _automatic_tree_follow_up_action_is_negated
    from . import _automatic_tree_follow_up_clauses
    from . import _automatic_tree_leaf_all_reason_values
    from . import _automatic_tree_leaf_explicit_reasons
    from . import _automatic_tree_leaf_rationale_is_allowed
    from . import _automatic_tree_span_is_negated
    from . import _clarification
    from . import _explicit_manual_breakpoint_bindings

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

_CROSS_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-search-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

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

_CROSS_RULE_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-rule-search-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

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

_CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-matrix-cell-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_CROSS_MATRIX_CELL_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])cross-cell-[0-9a-f]{32}(?![A-Za-z0-9_-])"
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

def _cross_matrix_cell_rationale_is_allowed(reason: str) -> bool:
    without_cell_terms = re.sub(
        r"(?:二维|交叉|Cross\s+Matrix|matrix|这(?:些|两个)?|这些|两个|多个|"
        r"格子|单元格|cells?)",
        " ",
        reason,
        flags=re.IGNORECASE,
    )
    return _automatic_tree_leaf_rationale_is_allowed(without_cell_terms)

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
