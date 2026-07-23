from marvis.orchestrator.templates import get_template
from marvis.orchestrator.templates.sample import BUILTIN_TEMPLATES
from marvis.orchestrator.templates.strategy import STRATEGY_DSL_DELIVERY


def test_strategy_dsl_delivery_template_is_builtin_and_non_gated():
    template = get_template("strategy_dsl_delivery")

    assert template == STRATEGY_DSL_DELIVERY
    assert template in BUILTIN_TEMPLATES
    assert [slot.name for slot in template.slots] == [
        "strategy_ref",
        "dataset_ref",
        "workspace_ref",
        "maximum_equivalence_rows",
    ]
    assert all(slot.source == "task_context" for slot in template.slots)
    assert len(template.steps) == 1
    step = template.steps[0]
    assert step.tool_ref.plugin == "strategy"
    assert step.tool_ref.tool == "export_strategy_delivery"
    assert step.needs_confirmation is False
    assert step.decision_point is False
    assert {
        check.spec["field"]
        for check in step.post_checks
        if check.kind == "nonempty"
    } == {
        "delivery_id",
        "equivalence.equivalence_id",
        "artifacts",
    }
