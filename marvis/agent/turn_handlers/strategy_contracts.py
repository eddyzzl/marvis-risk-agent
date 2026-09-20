"""Strategy contracts for governed Agent turns."""

from __future__ import annotations

from marvis.agent.strategy_setup import StrategySetupError
from marvis.artifacts.transactional import ArtifactTransactionError
from marvis.repositories.task_artifacts import TaskArtifactConflictError
from marvis.repositories.task_artifacts import TaskArtifactDataError
from marvis.repositories.task_artifacts import TaskArtifactNotFoundError
import re
import sqlite3


_STRATEGY_SAMPLE_BOUND_TOOLS = frozenset(
    {
        "analyze_univariate_candidates",
        "backtest_strategy",
        "build_automatic_tree_candidate",
        "compare_strategies",
        "design_cutoff_bands",
        "design_strategy_candidate",
        "evaluate_rule_set",
        "limit_pricing_matrix",
        "measure_pool_impact",
        "mine_rules",
        "tradeoff_view",
    }
)

_STRATEGY_SAMPLE_DESIGN_REQUIRED_FIELDS = (
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
)

_STRATEGY_SAMPLE_DESIGN_V2_MISSING_CONTROLS = (
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
)

_STRATEGY_SAMPLE_BOUND_CANDIDATE_WORKFLOWS = frozenset(
    {
        "univariate_candidate_analysis",
        "univariate_candidate_refinement",
        "automatic_tree_candidate_build",
        "cross_matrix_analysis",
    }
)

_STRATEGY_NAN_LABEL_META_KEY = "strategy_nan_label_confirmation"

_STRATEGY_SAMPLE_V2_POLICY = {
    "minimum_partition_count": 1,
    "minimum_bad_count": 1,
    "minimum_label_coverage": 0.8,
    "minimum_historical_score_coverage": 0.8,
    "maximum_group_coverage_gap": 0.2,
    "diagnostic_severities": {
        "entity_overlap": "fail",
        "temporal_oot": "fail",
        "risk_outside_approval": "fail",
        "maturity": "fail",
        "label_coverage": "fail",
        "historical_score_coverage": "warn",
        "group_coverage_gap": "warn",
        "sufficiency": "fail",
    },
}

_STRATEGY_DROP_NAN_CONFIRM_RE = re.compile(
    r"(?:确认|同意|允许|可以).{0,12}(?:丢弃|排除|剔除|删除).{0,12}"
    r"(?:NaN|nan|空标签|缺失标签|无效标签)|"
    r"(?:确认|同意|允许|可以).{0,12}"
    r"(?:NaN|nan|空标签|缺失标签|无效标签).{0,24}"
    r"(?:风险|坏账).{0,8}分母.{0,8}(?:排除|剔除)|"
    r"(?:confirm|allow).{0,12}(?:drop|exclude).{0,12}(?:nan|missing)\s+labels?",
    re.IGNORECASE,
)

_STRATEGY_DROP_NAN_CANCEL_RE = re.compile(
    r"(?:不丢弃|不排除|不剔除|不删除|取消|停止|"
    r"do\s+not\s+(?:drop|exclude)|don't\s+(?:drop|exclude))",
    re.IGNORECASE,
)

_STRATEGY_REQUEST_META_KEY = "strategy_request"

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

_STRATEGY_REQUEST_ACTION_RE = re.compile(
    r"(?:开发|设计|制定|创建|生成|构建|训练|物化|固化|冻结|探索|整理|梳理|汇总|归集|收集|刷新|更新|复盘|盘点|记录|做|计算|测算|分析|评估|查看|看一下|看下|回测|测试|验证|回放|应用|执行|写回|回写|回填|打标|"
    r"对比|比较|搜索|查找|检索|枚举|采纳|采用|上线|报告|文档|监控|漂移|挖掘|选择|筛选|保留|合并|编辑|"
    r"添加|加入|入池|删除|移除|排序|重排|改为|编译|预览|"
    r"develop|design|create|build|train|materialize|aggregate|collect|compute|calculate|analy[sz]e|evaluate|backtest|validate|replay|run|apply|compare|"
    r"search|find|enumerate|screen|adopt|report|monitor|mine|refine|select|merge|add|remove|delete|reorder|compile|preview)",
    re.IGNORECASE,
)

_STRATEGY_REQUEST_SUBJECT_RE = re.compile(
    r"(?:策略|策略项目上下文|项目上下文|当前项目(?:现状|情况)|历史(?:版本)?策略|策略样本|样本设计|样本边界|策略池|规则池|准入|审批|拒绝|额度|授信|定价|利率|分群|分层|规则|候选|候选箱|单变量|分箱|自动树|决策树|叶子|叶节点|投票|Voting|n[-_ ]?of[-_ ]?k|(?:二维|2\s*[dD])?\s*(?:交叉|cross)\s*(?:矩阵|matrix)|cutoff|利润|收益|"
    r"催收|滚动率|迁徙率|迁徙矩阵|定价矩阵|额度矩阵|网格|ROA|"
    r"roll(?:\s|-|_)*rate|strategy(?:\s|-|_)*pool|pool|strategy|approval|reject|limit|pricing|segment|rule|candidate|automatic(?:\s|-|_)*tree|decision(?:\s|-|_)*tree|leaf|"
    r"candidate\s+bins?|\bbins?\b|univariate|binning|sample(?:\s|-|_)*design|profit|collection)",
    re.IGNORECASE,
)

_STRATEGY_AUTOMATIC_TREE_SHORTHAND_RE = re.compile(
    r"(?:建\s*(?:一棵)?\s*(?:自动)?(?:决策)?树(?!状|莓|屋)|"
    r"训练\s*(?:一棵)?\s*(?:自动)?(?:决策)?树(?:模型)?|"
    r"(?<![A-Za-z0-9_])(?:build|train)\s+(?:an?\s+)?"
    r"(?:(?:automatic|decision)\s+)?tree(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)

_STRATEGY_REQUEST_CANCEL_RE = re.compile(
    r"(?:先别|不要|不用|不执行|先不|暂不|暂停|停止|取消|"
    r"do\s*not|don't|dont|stop|cancel|wait)",
    re.IGNORECASE,
)

_STRATEGY_REQUEST_NON_EXECUTION_RE = re.compile(
    r"(?:不要执行|不要运行|先别执行|先别运行|先不执行|先不运行|"
    r"只预览|仅预览|只讨论|仅讨论|只聊|仅供讨论|"
    r"do\s+not\s+(?:execute|run)|don't\s+(?:execute|run)|"
    r"preview\s+only|discussion\s+only|discuss\s+only)",
    re.IGNORECASE,
)

_STRATEGY_POOL_COMPILE_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:编译|预览|compile|preview))",
    re.IGNORECASE,
)

_STRATEGY_POOL_IMPACT_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:影响|效果|瀑布|逐月|通过率|坏账率|风险率|测算|评估|计算|回测|"
    r"impact|effect|waterfall|monthly|approval\s+rate|bad\s+rate|risk\s+rate|"
    r"measure|assess|evaluat|calculate|backtest))",
    re.IGNORECASE,
)

_STRATEGY_POOL_VALIDATION_REQUEST_RE = re.compile(
    r"(?=.*(?:策略池|规则池|strategy(?:\s|-|_)*pool|\bpool\b))"
    r"(?=.*(?:独立样本|独立回放|回放验证|独立验证|"
    r"independent\s+(?:sample\s+)?replay|independent\s+validation|"
    r"replay\s+validation))"
    r"(?=.*(?:验证集|验证样本|验证分区|"
    r"(?<![A-Za-z0-9_])(?:validation|oot)(?![A-Za-z0-9_])))",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_UNAVAILABLE_ANSWER_RE = re.compile(
    r"(?:暂时没有|暂缺|暂无|没有|未提供|不可用|不知道|未知|待补充|"
    r"unavailable|not\s+available|unknown|missing)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_ALL_PENDING_RE = re.compile(
    r"(?:这些|上述|以上|全部|所有|都|all\s+of\s+them|all)",
    re.IGNORECASE,
)

_PROJECT_CONTEXT_ANSWER_PATTERNS = {
    "current.status_fields.volume": re.compile(
        r"申请量|进件量|放款量|业务量|规模|volume", re.IGNORECASE
    ),
    "current.status_fields.approval": re.compile(
        r"通过率|审批率|准入率|approval", re.IGNORECASE
    ),
    "current.status_fields.risk": re.compile(
        r"坏账率|风险率|逾期率|risk|bad\s+rate", re.IGNORECASE
    ),
    "current.status_fields.economics": re.compile(
        r"收益|利润|成本|经济|economics|profit", re.IGNORECASE
    ),
    "current.maturity_summary": re.compile(
        r"成熟度|表现窗|观察窗|maturity|performance\s+window", re.IGNORECASE
    ),
    "historical_strategy_reviews": re.compile(
        r"历史(?:版本)?策略|历史材料|旧版策略|上一版策略|history|historical",
        re.IGNORECASE,
    ),
}


class _StrategySampleDesignRequiredError(StrategySetupError):
    """The current strategy request has no exact mature sample-design binding."""


class _StrategySampleDesignPolicyMismatchError(StrategySetupError):
    """A valid newest sample exists, but the probed null-label policy differs."""


class _StrategyV2EvidenceSetupError(StrategySetupError):
    """Typed preflight failure for platform-owned V2 evidence discovery."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


_STRATEGY_V2_ARTIFACT_ERRORS = (
    ArtifactTransactionError,
    TaskArtifactConflictError,
    TaskArtifactDataError,
    TaskArtifactNotFoundError,
    sqlite3.Error,
)

_STRATEGY_MODEL_EVIDENCE_V2_REQUEST_RE = re.compile(
    r"(?:Strategy\s+Model\s*Evidence(?:\s+V2)?|"
    r"Model\s*Evidence(?:\s+V2)?|模型证据(?:\s*V2)?|"
    r"单变量(?:候选)?证据(?:包|汇总)?|认证单变量(?:候选)?(?:证据|结果))",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_TYPE_PATTERNS = {
    "approval": re.compile(
        r"(?:审批|准入)|(?<![A-Za-z0-9_])approval(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"拒绝|(?<![A-Za-z0-9_])reject(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"额度|(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"定价|利率|(?<![A-Za-z0-9_])pricing(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "segmentation": re.compile(
        r"分群|分层|客群|"
        r"(?<![A-Za-z0-9_])segmentation(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}

_STRATEGY_REPORT_POOL_COMMAND_RE = re.compile(
    r"(?:生成|创建|制作|编制|形成|出一份|出个|给我|导出|构建)"
    r"[^；;。.!?？\n]{0,80}(?:报告|Report)|"
    r"(?<![A-Za-z0-9_])(?:generate|create|build|produce|prepare|render|export)"
    r"[^;.!?\n]{0,80}\breport(?:\s+bundle)?\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_TITLE_RE = re.compile(
    r"(?:报告标题|标题|report\s+title|title)\s*"
    r"(?:为|是|叫|is|=|:|：)\s*"
    r"(?:[“\"'《][^”\"'》\n]{1,200}[”\"'》]|"
    r"[^，,；;。.!?？\n]{1,200})",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_SELECTOR_RE = re.compile(
    r"(?:选择|选用|使用|采用|针对|指定|按|基于|改用|就用|要用|而是|"
    r"(?:Pool|策略)\s*类型\s*(?:为|是|=|:|：)|"
    r"(?<![A-Za-z0-9_])(?:select|choose|use|using|for|on|but|instead)"
    r"(?![A-Za-z0-9_]))\s*$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_TYPE_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|禁止|排除|剔除|而非|不是|并非|"
    r"不使用|不选|不选择)\s*(?:(?:选择|选用|使用|采用|针对|指定)\s*)?"
    r"[^，,；;。.!?？\n]{0,16}$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without|exclude)"
    r"[^,;.!?\n]{0,20}$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_HISTORY_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史|已归档|已生成)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historical|"
    r"last\s+time|archived|already\s+generated)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT = 64

_STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT = _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT

_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT = _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT

_STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT = _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
