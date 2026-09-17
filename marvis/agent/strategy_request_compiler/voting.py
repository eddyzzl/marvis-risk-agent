"""voting request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _POOL_STRATEGY_TYPE_GROUNDING
    from . import _clarification
    from . import _utterance_chains_voting_search_operation

_VOTING_RULE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-rule-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

_VOTING_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])voting-search-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

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

def _utterance_targets_voting_candidate_search(utterance: str) -> bool:
    """Reserve explicit Voting combination search before exact-member build."""

    return (
        not _utterance_targets_voting_search_selection(utterance)
        and _VOTING_SUBJECT_RE.search(utterance) is not None
        and _VOTING_SEARCH_INTENT_RE.search(utterance) is not None
    )

def _utterance_targets_voting_search_selection(utterance: str) -> bool:
    """Reserve an exact search-result materialization before search/build."""

    return (
        _VOTING_SEARCH_ID_TOKEN_RE.search(utterance) is not None
        and _VOTING_COMBO_ID_TOKEN_RE.search(utterance) is not None
        and _VOTING_SEARCH_SELECTION_INTENT_RE.search(utterance) is not None
    )

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

def _voting_strategy_type_mentions(
    utterance: str,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (strategy_type, match.start(), match.end())
        for strategy_type, pattern in _POOL_STRATEGY_TYPE_GROUNDING.items()
        for match in pattern.finditer(utterance)
    )

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

def _voting_n_bindings(utterance: str) -> tuple[tuple[int, int | None], ...]:
    bindings: list[tuple[int, int | None]] = []
    for n, k, _start, _end in _voting_n_mentions(utterance):
        binding = (n, k)
        if binding not in bindings:
            bindings.append(binding)
    return tuple(bindings)
