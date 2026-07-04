"""Labeling pack: 从还款/DPD 长表构造 0/1 坏标签 + cohort 成熟度检查 + 定坏口径建议 (C1).

标签构造是信贷风控建模最重要的前置：把"贷款×期"的逾期长表，按业务给定的
观察期/表现期/逾期阈值，落成一个带 0/1 目标的衍生数据集，并携带定坏口径元数据
（进 T3 血缘）。成熟度检查在此之上加一道确认门：表现期未闭合的 cohort 不静默纳入。
"""

from marvis.data.label_construction import (
    BadDefinition,
    BadDefinitionSuggestion,
    CohortMaturity,
    LabelConstruction,
    MaturityReport,
    check_cohort_maturity,
    construct_label,
    suggest_bad_definition,
)

__all__ = [
    "BadDefinition",
    "BadDefinitionSuggestion",
    "CohortMaturity",
    "LabelConstruction",
    "MaturityReport",
    "check_cohort_maturity",
    "construct_label",
    "suggest_bad_definition",
]
