"""tree request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Sequence
import math
import re
import unicodedata

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import AUTOMATIC_TREE_DIRECTIONS
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _SCORECARD_SECOND_OPERATION_RE
    from . import _clarification
    from . import _ratio_token_value
    from . import _utterance_requests_automatic_tree_follow_up
    from . import _utterance_supports_automatic_tree_column_role
    from . import _utterance_supports_automatic_tree_direction
    from . import _utterance_supports_automatic_tree_feature
    from . import _utterance_supports_automatic_tree_number

_AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])automatic-tree-leaf-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-frontier-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"interactive-tree-frontier-group-selection-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_AUTOMATIC_TREE_ASSET_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])candidate-asset-[0-9a-f]{32}(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_SOURCE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:candidate-asset|interactive-tree-revision)-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_NODE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])node-[0-9a-f]{20}(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_SPLIT_SEARCH_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-split-search-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-split-candidate-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
)

_INTERACTIVE_TREE_REVISION_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])interactive-tree-revision-[0-9a-f]{32}"
    r"(?![A-Za-z0-9_-])"
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

_INTERACTIVE_TREE_FEATURE_ACTION_RE = re.compile(
    r"(?:替换|更换|修改|调整|改用|换成|设置)"
    r"[^，,；;。.!?！？\n]{0,180}(?:分裂|切分)?(?:特征|字段|变量)|"
    r"(?:分裂|切分)?(?:特征|字段|变量)"
    r"[^，,；;。.!?！？\n]{0,180}(?:替换|更换|修改|调整|改为|改成|改用)|"
    r"(?<![A-Za-z0-9_])replace_split_feature(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?:replace|change|set)\s+(?:the\s+)?"
    r"(?:split\s+)?feature(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_INTERACTIVE_TREE_FEATURE_VALUE_RE = re.compile(
    r"(?:新\s*)?(?:分裂|切分)?(?:特征|字段|变量)\s*"
    r"(?:替换|更换|修改|调整|设置|改)?\s*"
    r"(?:为|成|到|=|:|：)\s*"
    r"(?P<zh_feature>[A-Za-z0-9_.\-\u4e00-\u9fff]+)|"
    r"(?<![A-Za-z0-9_])(?:replace|change|set)\s+(?:the\s+)?"
    r"(?:new\s+)?(?:split\s+)?feature\s+(?:to|=|:)\s*"
    r"(?P<en_feature>[A-Za-z0-9_.\-]+)(?![A-Za-z0-9_])",
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

_INTERACTIVE_TREE_FEATURE_AMBIGUOUS_RE = re.compile(
    r"(?:最佳特征|最优特征|最合适(?:的)?(?:特征|字段|变量)|"
    r"自动(?:选择|推荐|替换|更换)(?:特征|字段|变量)|"
    r"全部特征|所有特征|每个特征)|"
    r"(?<![A-Za-z0-9_])(?:best|optimal)\s+(?:split\s+)?feature|"
    r"(?<![A-Za-z0-9_])automatically\s+(?:select|recommend|replace)"
    r"\s+(?:the\s+)?(?:split\s+)?feature(?![A-Za-z0-9_])",
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

def _ground_interactive_tree_split_search(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one aggregate-only node search without authorizing a tree edit."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if re.search(
        r"(?:不要|别|无需|仅讨论|以后|将来|曾经|是否|能否|可否|"
        r"\b(?:not|do\s+not|don't|later|future|historical|whether|can\s+you)\b)",
        utterance,
        re.IGNORECASE,
    ):
        return _clarification(
            "原话必须是当前、肯定的一次节点候选搜索命令；否定、问句、"
            "历史、假设或未来描述不会启动搜索。",
            code="interactive_tree_split_search_intent_negated",
            fields=("search_intent",),
        )
    if not (
        re.search(
            r"(?:搜索|分析|遍历|候选|排名|试算|search|rank|candidate|"
            r"evaluate)",
            utterance,
            re.IGNORECASE,
        )
        and re.search(
            r"(?:树|节点|分裂|特征|tree|node|split|feature)",
            utterance,
            re.IGNORECASE,
        )
    ):
        return _clarification(
            "请明确要求对一个交互树节点搜索或分析分裂候选。",
            code="interactive_tree_split_search_explicit_intent_required",
            fields=("search_intent",),
        )
    if re.search(
        r"(?:直接替换|直接修改|自动选(?:最优|最佳|冠军)|采用第一名|"
        r"继续建树|自动续建|加入策略池|入池|应用|采纳|部署|投产|生成报告|"
        r"replace\s+and\s+apply|select\s+(?:the\s+)?winner|"
        r"auto[- ]?build|continue\s+(?:the\s+)?tree|add\s+to\s+pool|"
        r"adopt|deploy|generate\s+(?:a\s+)?report)",
        utterance,
        re.IGNORECASE,
    ):
        return _clarification(
            "本轮只生成节点分裂候选证据；选定候选、修改树、自动续建、"
            "入池、应用、报告、采纳或部署必须拆成后续明确请求。",
            code="interactive_tree_split_search_single_step_required",
            fields=("next_action",),
        )
    if re.search(
        r"(?:artifact|content_hash|evidence_hash|sample_design_ref|"
        r"dataset_id|workspace_revision|registry_metadata_hash|"
        r"制品哈希|样本绑定|数据集绑定)",
        utterance,
        re.IGNORECASE,
    ):
        return _clarification(
            "artifact、hash、数据集、workspace、样本与树父链由平台恢复，"
            "不能由本次自然语言搜索覆盖。",
            code="interactive_tree_split_search_platform_controls_forbidden",
            fields=("platform_bindings",),
        )
    source_matches = tuple(
        _INTERACTIVE_TREE_SOURCE_ID_TOKEN_RE.finditer(utterance)
    )
    node_matches = tuple(_INTERACTIVE_TREE_NODE_ID_TOKEN_RE.finditer(utterance))
    source_ids = frozenset(match.group(0) for match in source_matches)
    node_ids = frozenset(match.group(0) for match in node_matches)
    if (
        len(source_matches) != 1
        or len(source_ids) != 1
        or len(node_matches) != 1
        or len(node_ids) != 1
    ):
        return _clarification(
            "请逐字提供且只提供一个完整 automatic-tree/revision ID 和一个"
            "完整 node ID；平台不会用“那棵树”“最差节点”等代词或排名"
            "替你选择。",
            code="interactive_tree_split_search_explicit_ids_required",
            fields=("source_tree_id", "node_id"),
        )
    ungrounded: list[str] = []
    if source_ids != {inputs["source_tree_id"]}:
        ungrounded.append("source_tree_id")
    if node_ids != {inputs["node_id"]}:
        ungrounded.append("node_id")
    all_feature_intent = re.search(
        r"(?:全部特征|所有特征|全特征|完整特征全集|"
        r"all\s+(?:authenticated\s+)?features?|full\s+feature\s+universe)",
        utterance,
        re.IGNORECASE,
    )
    if inputs["mode"] == "all_features":
        if all_feature_intent is None or "features" in inputs:
            ungrounded.append("mode")
    else:
        features = inputs.get("features")
        if (
            all_feature_intent is not None
            or not isinstance(features, list)
            or any(feature not in utterance for feature in features)
        ):
            ungrounded.append("features")
    for field in (
        "max_thresholds_per_feature",
        "max_row_evaluations",
    ):
        value = inputs[field]
        if re.search(
            rf"(?<![0-9A-Fa-f.]){re.escape(str(value))}(?![0-9A-Fa-f.])",
            utterance,
        ) is None:
            ungrounded.append(field)
    if ungrounded:
        return _clarification(
            "搜索范围和预算必须与用户原话完全一致：明确全特征或逐字给出"
            "特征子集，并给出每特征阈值数与总行评估预算；平台不会补默认值"
            "或猜测。",
            code="interactive_tree_split_search_controls_not_grounded",
            fields=tuple(dict.fromkeys(ungrounded)),
        )
    return result

def _ground_interactive_tree_auto_continuation(
    utterance: str,
    result: StrategyRequestCompilation,
) -> StrategyRequestCompilation:
    """Ground one explicitly seeded and fully bounded subtree continuation."""

    draft = result.draft
    assert isinstance(draft, StandardWorkflowRequestDraft)
    inputs = draft.to_dict()["workflow_inputs"]
    if re.search(
        r"(?:不要|别|无需|仅讨论|以后|将来|是否|能否|可否|"
        r"\b(?:not|do\s+not|don't|later|future|whether|can\s+you)\b)",
        utterance,
        re.IGNORECASE,
    ) or re.search(
        r"(?:自动续建|自动继续|继续建树|续建子树|继续生长|"
        r"auto[- ]?continue|continue\s+(?:the\s+)?tree)",
        utterance,
        re.IGNORECASE,
    ) is None:
        return _clarification(
            "请用当前、肯定的单步命令明确要求自动续建交互树子树。",
            code="interactive_tree_auto_continuation_intent_required",
            fields=("continuation_intent",),
        )
    if re.search(
        r"(?:加入策略池|入池|应用|采纳|部署|投产|生成报告|出报告|"
        r"同时修改|顺便调整|再搜索|add\s+to\s+pool|adopt|deploy|"
        r"generate\s+(?:a\s+)?report|and\s+apply)",
        utterance,
        re.IGNORECASE,
    ):
        return _clarification(
            "本轮只允许从已明确选择的候选续建子树；入池、应用、报告、"
            "采纳、部署或另一项树编辑必须拆成后续请求。",
            code="interactive_tree_auto_continuation_single_step_required",
            fields=("next_action",),
        )
    if re.search(
        r"(?:artifact|content_hash|evidence_hash|sample_design_ref|"
        r"dataset_id|workspace_revision|registry_metadata_hash|"
        r"制品哈希|样本绑定|数据集绑定)",
        utterance,
        re.IGNORECASE,
    ):
        return _clarification(
            "artifact、hash、数据集、workspace、样本与父链由平台恢复，"
            "不能由续建请求覆盖。",
            code="interactive_tree_auto_continuation_platform_controls_forbidden",
            fields=("platform_bindings",),
        )
    search_matches = tuple(
        _INTERACTIVE_TREE_SPLIT_SEARCH_ID_TOKEN_RE.finditer(utterance)
    )
    candidate_matches = tuple(
        _INTERACTIVE_TREE_SPLIT_CANDIDATE_ID_TOKEN_RE.finditer(utterance)
    )
    search_ids = frozenset(match.group(0) for match in search_matches)
    candidate_ids = frozenset(match.group(0) for match in candidate_matches)
    if (
        len(search_matches) != 1
        or len(search_ids) != 1
        or len(candidate_matches) != 1
        or len(candidate_ids) != 1
    ):
        return _clarification(
            "请逐字提供且只提供一个完整 split search ID 和一个完整"
            " eligible candidate ID；平台不会按排名、最佳或“第一个”"
            "替你选择种子候选。",
            code="interactive_tree_auto_continuation_explicit_ids_required",
            fields=("search_id", "candidate_id"),
        )
    ungrounded: list[str] = []
    if search_ids != {inputs["search_id"]}:
        ungrounded.append("search_id")
    if candidate_ids != {inputs["candidate_id"]}:
        ungrounded.append("candidate_id")
    for field in (
        "max_additional_depth",
        "min_gini_gain",
        "max_generated_nodes",
        "max_thresholds_per_feature",
        "max_row_evaluations",
    ):
        value = inputs[field]
        tokens = {str(value)}
        if isinstance(value, float) and value.is_integer():
            tokens.add(str(int(value)))
        if not any(
            re.search(
                rf"(?<![0-9A-Fa-f.]){re.escape(token)}"
                r"(?![0-9A-Fa-f.])",
                utterance,
            )
            for token in tokens
        ):
            ungrounded.append(field)
    for field in ("objective", "tie_break"):
        if inputs[field] not in utterance:
            ungrounded.append(field)
    reason = inputs.get("reason")
    if reason is not None and reason not in utterance:
        ungrounded.append("reason")
    if ungrounded:
        return _clarification(
            "续建必须逐字给出 search/candidate ID、追加深度、最小 Gini "
            "增益、节点上限、每特征阈值上限、总行评估预算，以及固定的 "
            "objective 和 tie_break；平台不会补默认值或代选候选。",
            code="interactive_tree_auto_continuation_controls_not_grounded",
            fields=tuple(dict.fromkeys(ungrounded)),
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
    feature_action = (
        _INTERACTIVE_TREE_FEATURE_ACTION_RE.search(utterance) is not None
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
    if _INTERACTIVE_TREE_FEATURE_AMBIGUOUS_RE.search(utterance) is not None:
        return _clarification(
            "换分裂特征必须点名一个当前可见 split node、一个认证特征和一个"
            "有限阈值；平台不会按“最佳特征”“自动推荐”或“全部特征”直接"
            "替用户修改树。请先单独运行节点候选分析，再精确选择。",
            code="interactive_tree_revision_feature_ambiguous",
            fields=("feature", "threshold"),
        )
    if (
        _INTERACTIVE_TREE_NEGATED_OR_NONCURRENT_RE.search(utterance) is not None
        or not (prune_action or threshold_action or feature_action)
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
        and (threshold_action or feature_action)
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
        "replace_split_feature"
        if feature_action
        else (
            "adjust_split_threshold"
            if threshold_action
            else "prune_subtree"
        )
    )
    if expected_operation in {
        "adjust_split_threshold",
        "replace_split_feature",
    } and len(threshold_values) != 1:
        return _clarification(
            "分裂调整必须在同一条命令中明确且只给出一个有限的新 threshold "
            "数值；平台不会从描述、指标或历史树中推断。",
            code="interactive_tree_revision_explicit_threshold_required",
            fields=("threshold",),
        )
    feature_values = _interactive_tree_feature_values(utterance)
    if (
        expected_operation == "replace_split_feature"
        and len(feature_values) != 1
    ):
        return _clarification(
            "换分裂特征必须在同一条命令中逐字给出且只给出一个新 feature；"
            "平台不会从排名或树结构中推断。",
            code="interactive_tree_revision_explicit_feature_required",
            fields=("feature",),
        )

    ungrounded: list[str] = []
    if source_ids != {inputs["source_tree_id"]}:
        ungrounded.append("source_tree_id")
    if node_ids != {inputs["node_id"]}:
        ungrounded.append("node_id")
    if inputs["operation"] != expected_operation:
        ungrounded.append("operation")
    if expected_operation in {
        "adjust_split_threshold",
        "replace_split_feature",
    }:
        supplied_threshold = inputs.get("threshold")
        if (
            isinstance(supplied_threshold, bool)
            or not isinstance(supplied_threshold, int | float)
            or float(supplied_threshold) != threshold_values[0]
        ):
            ungrounded.append("threshold")
    elif "threshold" in inputs:
        ungrounded.append("threshold")
    if expected_operation == "replace_split_feature":
        if inputs.get("feature") != feature_values[0]:
            ungrounded.append("feature")
    elif "feature" in inputs:
        ungrounded.append("feature")
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

def _interactive_tree_feature_values(utterance: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _INTERACTIVE_TREE_FEATURE_VALUE_RE.finditer(utterance):
        value = match.group("zh_feature") or match.group("en_feature")
        if value:
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

def _automatic_tree_value_is_replaced(utterance: str, *, end: int) -> bool:
    return (
        re.match(
            r"\s*(?:改为|改成|调整为|替换为|而非|不是而是)",
            utterance[end:],
        )
        is not None
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
