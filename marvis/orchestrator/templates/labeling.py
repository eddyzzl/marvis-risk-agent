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
        SlotSpec("dataset_id", True, "task_context", "Active task-owned repayment long-table dataset id"),
        SlotSpec("expected_content_hash", True, "task_context", "SHA-256 of the confirmed active dataset"),
        SlotSpec("workspace_revision", True, "task_context", "Confirmed data-workspace revision"),
        SlotSpec("analysis_generation", True, "task_context", "Confirmed active-dataset analysis generation"),
        SlotSpec("id_col", True, "task_context", "Loan id column"),
        SlotSpec("mob_col", True, "task_context", "Month-on-book column"),
        SlotSpec("cohort_col", True, "task_context", "Vintage/放款月 cohort column (maturity check)"),
        SlotSpec("date_col", True, "user", "Observation/snapshot date column used for the as-of cutoff"),
        SlotSpec("as_of_date", True, "user", "Explicit ISO as-of cutoff date"),
        SlotSpec("observation_window", True, "user", "Observation window end MOB"),
        SlotSpec("performance_window", True, "user", "Performance window length (MOBs)"),
        SlotSpec("at_mob", True, "user", "Explicit bad-definition evaluation MOB"),
        SlotSpec("rule_kind", True, "user", "Explicit label rule kind: dpd or status"),
        SlotSpec("dpd_col", False, "task_context", "Numeric days-past-due column (dpd 口径)"),
        SlotSpec("threshold_dpd", False, "user", "DPD threshold in days (e.g. 30/60/90)"),
        SlotSpec("status_col", False, "task_context", "Overdue-bucket status column (status 口径)"),
        SlotSpec("threshold_status", False, "user", "Overdue-bucket threshold (in states)"),
        SlotSpec("states", False, "user", "Overdue-bucket order good->bad (status 口径)"),
        SlotSpec("target_col", True, "user", "Explicit output target column name"),
        SlotSpec("proposal_hash", True, "task_context", "SHA-256 of the human-confirmed pre-plan proposal"),
        SlotSpec("confirm_immature_cohorts", True, "user", "Explicit decision to include immature cohorts as NaN labels"),
    ),
    steps=(
        StepTemplate(
            title="Cohort 成熟度检查",
            tool_ref=ToolRef("labeling", "check_cohort_maturity"),
            inputs_template={
                "dataset_id": "{slot:dataset_id}",
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "id_col": "{slot:id_col}",
                "mob_col": "{slot:mob_col}",
                "cohort_col": "{slot:cohort_col}",
                "date_col": "{slot:date_col}",
                "as_of_date": "{slot:as_of_date}",
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
                "expected_content_hash": "{slot:expected_content_hash}",
                "workspace_revision": "{slot:workspace_revision}",
                "analysis_generation": "{slot:analysis_generation}",
                "id_col": "{slot:id_col}",
                "mob_col": "{slot:mob_col}",
                "cohort_col": "{slot:cohort_col}",
                "date_col": "{slot:date_col}",
                "as_of_date": "{slot:as_of_date}",
                "observation_window": "{slot:observation_window}",
                "performance_window": "{slot:performance_window}",
                "at_mob": "{slot:at_mob}",
                "rule_kind": "{slot:rule_kind}",
                "dpd_col": "{slot:dpd_col}",
                "threshold_dpd": "{slot:threshold_dpd}",
                "status_col": "{slot:status_col}",
                "threshold_status": "{slot:threshold_status}",
                "states": "{slot:states}",
                "target_col": "{slot:target_col}",
                "proposal_hash": "{slot:proposal_hash}",
                "confirm_immature_cohorts": "{slot:confirm_immature_cohorts}",
            },
            depends_on_titles=("Cohort 成熟度检查",),
            post_checks=(
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("nonempty", {"field": "target_col"}),
                PostCheck("nonempty", {"field": "result_content_hash"}),
                PostCheck("nonempty", {"field": "dataset_artifact_id"}),
                PostCheck("nonempty", {"field": "dataset_download_url"}),
                PostCheck("nonempty", {"field": "evidence_artifact_id"}),
                PostCheck("nonempty", {"field": "evidence_download_url"}),
                PostCheck(
                    "range",
                    {
                        "field": "bad_rate",
                        "min": 0.0,
                        "max": 1.0,
                        "allow_null": True,
                    },
                ),
                PostCheck(
                    "invariant",
                    {"rule": "source_dataset_id==lineage.parent_dataset_id"},
                ),
                PostCheck(
                    "invariant",
                    {"rule": "result_dataset_id==lineage.child_dataset_id"},
                ),
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
