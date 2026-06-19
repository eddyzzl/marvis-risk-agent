import pytest

from marvis.orchestrator.contracts import PostCheck
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
    builtin_template_ids,
    clear_user_templates,
    get_template,
    list_templates,
    load_builtin_templates,
    register_template,
    register_user_template,
)
from marvis.plugins.manifest import ToolRef


def _template(template_id: str, *, source: str = "builtin") -> WorkflowTemplate:
    return WorkflowTemplate(
        id=template_id,
        title=f"Template {template_id}",
        goal_patterns=(template_id,),
        slots=(SlotSpec("task_id", True, "task_context", "Current task"),),
        steps=(
            StepTemplate(
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={"message": "{slot:task_id}"},
                depends_on_titles=(),
                post_checks=(PostCheck("nonempty", {"field": "echoed"}),),
            ),
        ),
        source=source,
    )


def test_register_get_and_list_templates():
    template = _template("test_builtin_template")

    register_template(template)

    assert get_template("test_builtin_template") == template
    assert template in list_templates()
    with pytest.raises(ValueError, match="duplicate"):
        register_template(template)


def test_load_builtin_templates_registers_sample_echo_idempotently():
    load_builtin_templates()
    load_builtin_templates()

    template = get_template("sample_echo")
    assert template.source == "builtin"
    assert template.steps[0].tool_ref == ToolRef("_sample", "echo")
    assert template.slots[0].name == "message"
    assert "sample_echo" in builtin_template_ids()
    model_validation = get_template("model_validation")
    assert model_validation.steps[0].tool_ref == ToolRef("v1_compat", "scan_materials")
    assert model_validation.steps[-1].needs_confirmation is True
    assert "model_validation" in builtin_template_ids()


def test_user_template_registration_cannot_shadow_builtin_and_can_reload():
    load_builtin_templates()
    clear_user_templates()
    user_v1 = _template("user_echo", source="user")
    user_v2 = _template("user_echo", source="user")

    register_user_template(user_v1)
    register_user_template(user_v2)

    assert get_template("user_echo") == user_v2
    with pytest.raises(ValueError, match="builtin"):
        register_user_template(_template("sample_echo", source="user"))

    clear_user_templates()
    with pytest.raises(KeyError):
        get_template("user_echo")
    assert get_template("sample_echo").source == "builtin"
