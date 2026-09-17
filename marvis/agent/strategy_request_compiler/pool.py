"""pool request-compiler handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any
import unicodedata

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import StandardWorkflowRequestDraft
    from . import StrategyRequestCompilation
    from . import _AUTOMATIC_TREE_ASSET_ID_TOKEN_RE
    from . import _AUTOMATIC_TREE_LEAF_SELECTION_ID_TOKEN_RE
    from . import _CROSS_MATRIX_CELL_SELECTION_ID_TOKEN_RE
    from . import _INTERACTIVE_TREE_FRONTIER_GROUP_SELECTION_ID_TOKEN_RE
    from . import _INTERACTIVE_TREE_FRONTIER_SELECTION_ID_TOKEN_RE
    from . import _SCORECARD_CUTOFF_SELECTION_ID_TOKEN_RE
    from . import _STRATEGY_POOL_WORKFLOWS
    from . import _clarification
    from . import _impact_cube_strategy_type_mentions
    from . import _ungrounded_pool_actions
    from . import _utterance_contains_token
    from . import _voting_strategy_type_mentions
    from . import utterance_targets_candidate_monthly_stability

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

_POOL_MUTATION_WORKFLOWS = _STRATEGY_POOL_WORKFLOWS - {"strategy_pool_compile"}

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
