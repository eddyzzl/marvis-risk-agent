from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import SlotSpec, StepTemplate, WorkflowTemplate
from marvis.plugins.manifest import GovernancePolicy, ToolRef


DATASET_ASOF_JOIN = WorkflowTemplate(
    id="dataset_asof_join",
    title="时点一致数据拼接",
    goal_patterns=("时点拼接", "按历史可见时间拼接", "as-of join", "point-in-time join"),
    slots=(
        SlotSpec("decision_contract", True, "user", "Decision dataset id/hash, row/entity identity, decision time column, timezone and recorded source evidence"),
        SlotSpec("feature_contract", True, "user", "Feature dataset id/hash, event/version/availability columns with timezone/source evidence; unavailable historical availability must be explicit null"),
        SlotSpec("spec", True, "user", "Explicit verified/exploration mode, timezone-aware as_of, feature allowlist and optional lookback/partition constraints"),
    ),
    steps=(
        StepTemplate(
            title="确认时间契约并执行时点拼接",
            tool_ref=ToolRef("data_ops", "asof_join"),
            inputs_template={name: "{slot:" + name + "}" for name in ("decision_contract", "feature_contract", "spec")},
            depends_on_titles=(),
            post_checks=(
                PostCheck("one_of", {"field": "schema_version", "values": ["dataset-asof-tool-result.v1"]}),
                PostCheck("one_of", {"field": "assurance", "values": ["verified", "inferred", "unknown"]}),
                PostCheck("rowcount", {"field": "row_count", "min": 1}),
                PostCheck("nonempty", {"field": "result_dataset_id"}),
                PostCheck("nonempty", {"field": "result_content_hash"}),
                PostCheck("nonempty", {"field": "evidence.artifact_id"}),
                PostCheck("nonempty", {"field": "membership_sha256"}),
            ),
            needs_confirmation=True,
            policy=GovernancePolicy(human_decision_gate="required"),
            phase="数据准备",
        ),
    ),
    default_autonomy=1,
    source="builtin",
)
