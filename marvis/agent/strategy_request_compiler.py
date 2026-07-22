"""Pure natural-language compiler for governed strategy requests.

The LLM only translates an utterance into a draft.  This module then validates
that draft against fixed operation/type vocabularies, the Strategy DSL and the
dataset column whitelist.  It never executes a tool and never accepts calculated
metrics from the model.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from marvis.agent.json_reply import load_json_object
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
STANDARD_STRATEGY_WORKFLOWS = (
    "profit_calc",
    "roll_rate_matrix",
    "limit_pricing_matrix",
    "univariate_candidate_analysis",
    "univariate_candidate_refinement",
    "automatic_tree_candidate_build",
    "automatic_tree_leaf_materialization",
    "voting_candidate_build",
    "cross_matrix_analysis",
    "cross_matrix_cell_selection",
    "strategy_pool_add_candidate",
    "strategy_pool_remove_entry",
    "strategy_pool_set_action",
    "strategy_pool_reorder",
    "strategy_pool_compile",
    "strategy_pool_impact",
)
UNIVARIATE_BINNING_METHODS = (
    "equal_frequency",
    "equal_width",
    "chimerge",
    "tree",
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
_POOL_SOURCE_LIKE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:candidate-asset|automatic-tree-leaf-selection|"
    r"cross-matrix-cell-selection)-"
    r"[A-Za-z0-9_-]+(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_POOL_SOURCE_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:candidate-asset|automatic-tree-leaf-selection|"
    r"cross-matrix-cell-selection)-",
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
_VOTING_SUBJECT_RE = re.compile(
    r"(?:投票|(?<![A-Za-z0-9_])(?:Voting|n[-_ ]?of[-_ ]?k)"
    r"(?![A-Za-z0-9_]))",
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
    r"[^，,；;。\n]{0,40}(?:拒绝|审批|通过|复核)|"
    r"(?<![A-Za-z0-9_])(?:add\s+to\s+(?:the\s+)?(?:strategy\s+)?pool|"
    r"set\s+action|adopt|deploy|publish|write[- ]?back)(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:and(?:\s+then)?|then|also)"
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
    r"(?:报告|文档|汇报|总结)\s*(?:即可|就行|only)|"
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
) -> StrategyRequestCompilation:
    """Validate an already parsed LLM payload without invoking an LLM."""

    return _validate_payload(
        payload,
        _column_whitelist(allowed_columns),
        target_col=_normalized_target_col(target_col),
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
    return _validate_payload(payload, whitelist, target_col=target_col)


def _validate_payload(
    payload: object,
    whitelist: tuple[str, ...],
    *,
    target_col: str | None,
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
) -> _ValidationOutcome:
    missing = [
        field for field in ("workflow", "workflow_inputs") if field not in payload
    ]
    if missing:
        return _invalid("标准 Workflow 请求缺少字段：" + "、".join(missing) + "。")
    workflow = payload["workflow"]
    if not isinstance(workflow, str) or workflow not in STANDARD_STRATEGY_WORKFLOWS:
        return _invalid(
            "不支持的标准 Workflow；可选值为："
            + "、".join(STANDARD_STRATEGY_WORKFLOWS)
            + "。"
        )
    raw_inputs = payload["workflow_inputs"]
    if not isinstance(raw_inputs, Mapping):
        return _invalid("workflow_inputs 必须是一个对象。")
    if any(not isinstance(key, str) for key in raw_inputs):
        return _invalid("workflow_inputs 的字段名必须是文本。")
    try:
        if workflow == "profit_calc":
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
        elif workflow == "automatic_tree_candidate_build":
            normalized = _validate_automatic_tree_candidate_build_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "automatic_tree_leaf_materialization":
            normalized = _validate_automatic_tree_leaf_materialization_inputs(
                raw_inputs
            )
        elif workflow == "voting_candidate_build":
            normalized = _validate_voting_candidate_build_inputs(raw_inputs)
        elif workflow == "cross_matrix_analysis":
            normalized = _validate_cross_matrix_workflow_inputs(
                raw_inputs,
                whitelist,
                target_col=target_col,
            )
        elif workflow == "cross_matrix_cell_selection":
            normalized = _validate_cross_matrix_cell_selection_inputs(raw_inputs)
        elif workflow in _STRATEGY_POOL_MEASUREMENT_WORKFLOWS:
            normalized = _validate_strategy_pool_impact_inputs(
                raw_inputs,
                whitelist,
            )
        elif workflow in _STRATEGY_POOL_WORKFLOWS:
            normalized = _validate_strategy_pool_workflow_inputs(
                workflow,
                raw_inputs,
            )
        else:  # pragma: no cover - guarded by STANDARD_STRATEGY_WORKFLOWS
            raise _DraftValidationError(f"不支持的标准 Workflow：{workflow}。")
    except _DraftValidationError as exc:
        return _invalid(str(exc))

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
) -> dict[str, Any]:
    allowed = {
        "features",
        "methods",
        "bin_count",
        "min_bin_pct",
        "loan_amount_col",
        "overdue_amount_col",
        "sentinel_values",
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
    )
    return {
        **normalized,
        "x_feature": x_feature,
        "x_method": x_method,
        "y_feature": y_feature,
        "y_method": y_method,
    }


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

    feature = _workflow_column(
        inputs["feature"],
        name=f"{workflow} feature",
        whitelist=whitelist,
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
            and _CROSS_MATRIX_CELL_SELECTION_ID_RE.fullmatch(source_id) is None
        ):
            raise _DraftValidationError(
                f"{workflow} selection_id 必须是 automatic-tree-leaf-selection- "
                "或 cross-matrix-cell-selection- 后接 32 位小写十六进制字符。"
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


def _ground_refinement_request(
    utterance: str,
    result: StrategyRequestCompilation,
    *,
    whitelist: tuple[str, ...],
) -> StrategyRequestCompilation:
    draft = result.draft
    if _utterance_targets_strategy_pool_impact(utterance) and not (
        isinstance(draft, StandardWorkflowRequestDraft)
        and draft.workflow == "strategy_pool_impact"
    ):
        return _clarification(
            "原话明确要求 Strategy Pool 影响测算，只能编译为 strategy_pool_impact；"
            "不能改路由到 Pool 修改、通用策略生命周期、报告或其他 Workflow。",
            code="strategy_pool_impact_workflow_required",
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
            in {"cross_matrix_analysis", "cross_matrix_cell_selection"}
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
        and _utterance_targets_voting_candidate(utterance)
        and not (
            isinstance(draft, StandardWorkflowRequestDraft)
            and draft.workflow == "voting_candidate_build"
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
    if draft.workflow == "automatic_tree_leaf_materialization":
        return _ground_automatic_tree_leaf_materialization(utterance, result)
    if draft.workflow == "voting_candidate_build":
        return _ground_voting_candidate_build(utterance, result)
    if draft.workflow == "cross_matrix_cell_selection":
        return _ground_cross_matrix_cell_selection(utterance, result)
    if draft.workflow == "cross_matrix_analysis":
        return _ground_cross_matrix_analysis(
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


def _utterance_targets_strategy_pool_impact(utterance: str) -> bool:
    if _POOL_IMPACT_TARGET_RE.search(utterance) is None:
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
            _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE,
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
            _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE,
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
            "selection_id 必须是 automatic-tree-leaf-selection- 后接 32 位"
            "小写十六进制字符，不能同时给出两类来源。",
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
    if draft.workflow == "profit_calc":
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
        "对于 automatic_tree_candidate_build，只能抄录用户明确提供的 features、"
        "权重/金额字段、方向和树参数；不得填写平台拥有的数据绑定、目标列、标签策略、"
        "预算、结果、叶子、动作、排名或推荐，也不得串联选叶或 Strategy Pool。"
        "对于 automatic_tree_leaf_materialization，只能逐字抄录用户原话中唯一的"
        "完整 tree_asset_id、唯一的完整 leaf_id，以及用户用“选择理由/理由/原因/说明”"
        "显式标注时的逐字 selection_reason；未显式标注时必须省略。它只创建"
        "pointer，不得复制规则、条件、指标、动作或平台 artifact/hash，也不得串联"
        "Strategy Pool、业务动作、采纳、部署或 leaf ID 写回。selection_reason 中也"
        "不得藏入理由替换、后续动作、生命周期操作或极值/排名选叶语义；它只接受"
        "人工/业务/风险/合规/样本评审依据类短说明。"
        "对于 voting_candidate_build，只能逐字抄录用户明确标注的 strategy_type、"
        "2 到 50 个完整 candidate-rule ID 和整数 n；不得输出 entry_id、Pool revision/hash、"
        "condition、指标、动作、推荐或平台数据绑定。规则集合必须全部来自同一条正向"
        "Voting/n-of-k 构建命令；‘最好规则’‘刚才那些’等启发式引用，或同一句串联入池、"
        "动作、采纳、部署、写回时必须澄清。问句、假设/未来/历史描述、演示文本、句尾"
        "撤销以及多个 strategy_type/n 候选也必须澄清；显式 k 必须与 rule_ids 数量一致。"
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
        "必须澄清。selection_id 只允许 automatic-tree-leaf-selection- 或"
        "cross-matrix-cell-selection- 后接 32 位小写十六进制；完整 Cross Matrix asset"
        "不能直接入池。source ID 必须与唯一正向入池命令位于同一子句，不能从否定子句、"
        "reason、引用或代词上下文借用；未来/条件指令、问句、how-to、演示和测试也"
        "必须澄清。"
        "对于 strategy_pool_impact，只能抄录用户明确的 approval/reject Pool 类型，"
        "可选 absolute/vs_baseline 比较模式、完整 baseline_strategy_id、精确 month_col/"
        "loan_amount_col/overdue_amount_col 和明确的 drop_nan_labels 布尔授权。普通肯定式"
        "请求可默认 absolute；vs_baseline 必须同时有原话中的比较表达和完整基线 ID。"
        "禁止输出 dataset/target、Pool revision/hash、workspace、sample binding、semantic hash、"
        "metrics、conditions 或 strategy_spec。用户未指定月份/金额列时必须省略，平台只会"
        "使用唯一确认的语义角色；没有角色则 unavailable，多个角色则澄清，Agent 不得猜列。"
        "limit/pricing/segmentation、否定/问句/历史/仅报告或同轮修改/采纳/部署必须澄清。"
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
    "validate_strategy_request",
]
