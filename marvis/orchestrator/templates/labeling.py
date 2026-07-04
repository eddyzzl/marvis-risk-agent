"""标签构造 workflow 模板 (C1).

标签构造是建模前置 stage：先按 vintage 判定各 cohort 表现期是否闭合（成熟度检查，
决策点），再按观察期/表现期/逾期阈值构造 0/1 坏标签（define_label，确认门）。
与既有 NaN 标签门、label_semantics 门形成完整标签防线——标签先造好、成熟度先确认，
再进特征/建模。goal_patterns 与其它模板不相交，关键词路由不交叉。
"""

from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
)
from marvis.plugins.manifest import ToolRef


LABEL_CONSTRUCTION = WorkflowTemplate(
    id="label_construction",
    title="标签构造",
    goal_patterns=(
        "标签构造",
        "构造标签",
        "定坏",
        "定义坏样本",
        "label construction",
        "define label",
    ),
    slots=(
        SlotSpec("dataset_id", True, "task_context", "Registered DPD/repayment long-table dataset id"),
        SlotSpec("id_col", True, "task_context", "Loan id column"),
        SlotSpec("mob_col", True, "task_context", "Month-on-book column"),
        SlotSpec("cohort_col", True, "task_context", "Vintage/放款月 cohort column (maturity check)"),
        SlotSpec("observation_window", True, "user", "Observation window end MOB"),
        SlotSpec("performance_window", True, "user", "Performance window length (MOBs)"),
        SlotSpec("at_mob", False, "user", "Bad-definition evaluation MOB (default obs+perf)"),
        SlotSpec("dpd_col", False, "task_context", "Numeric days-past-due column (dpd 口径)"),
        SlotSpec("threshold_dpd", False, "user", "DPD threshold in days (e.g. 30/60/90)"),
        SlotSpec("status_col", False, "task_context", "Overdue-bucket status column (status 口径)"),
        SlotSpec("threshold_status", False, "user", "Overdue-bucket threshold (in states)"),
        SlotSpec("states", False, "user", "Overdue-bucket order good->bad (status 口径)"),
        SlotSpec("target_col", False, "user", "Output target column name (default target)"),
    ),
    steps=(
        StepTemplate(
            title="Cohort 成熟度检查",
            tool_ref=ToolRef("labeling", "check_cohort_maturity"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "id_col": "{slot:id_col}",
                "mob_col": "{slot:mob_col}",
                "cohort_col": "{slot:cohort_col}",
                "observation_window": "{slot:observation_window}",
                "performance_window": "{slot:performance_window}",
                "required_mob": "{slot:at_mob}",
            },
            depends_on_titles=(),
            post_checks=(PostCheck("nonempty", {"field": "cohorts"}),),
            decision_point=True,
        ),
        StepTemplate(
            title="构造标签",
            tool_ref=ToolRef("labeling", "define_label"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "id_col": "{slot:id_col}",
                "mob_col": "{slot:mob_col}",
                "cohort_col": "{slot:cohort_col}",
                "observation_window": "{slot:observation_window}",
                "performance_window": "{slot:performance_window}",
                "at_mob": "{slot:at_mob}",
                "dpd_col": "{slot:dpd_col}",
                "threshold_dpd": "{slot:threshold_dpd}",
                "status_col": "{slot:status_col}",
                "threshold_status": "{slot:threshold_status}",
                "states": "{slot:states}",
                "target_col": "{slot:target_col}",
                # Literal False default (not {slot:...}): SlotSpec has no default-value
                # mechanism and an omitted slot key is dropped by planner._fill_inputs,
                # so the apply_adjust gate-override channel (which only overwrites keys
                # already present in the instantiated inputs) needs this baked in. An
                # unanswered maturity gate leaves False -> define_label raises
                # CohortMaturityNotConfirmedError (mirrors vintage_analysis' baked
                # drop_nan_labels=False / label_semantics=None precedent).
                "confirm_immature_cohorts": False,
            },
            depends_on_titles=("Cohort 成熟度检查",),
            post_checks=(
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("nonempty", {"field": "target_col"}),
            ),
            # Mandatory label gate: constructing the modeling target is a口径 decision
            # that must not be auto-accepted — the driver pauses so the user confirms
            # the bad definition (and any immature-cohort inclusion) before the labeled
            # dataset is written.
            needs_confirmation=True,
        ),
    ),
    default_autonomy=1,
    source="builtin",
)


__all__ = ["LABEL_CONSTRUCTION"]
