from marvis.orchestrator.intent import IntentRouter
from marvis.orchestrator.templates import (
    SlotSpec,
    StepTemplate,
    WorkflowTemplate,
    clear_user_templates,
    load_builtin_templates,
    register_template,
    register_user_template,
)
from marvis.plugins.manifest import ToolRef


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _router(llm: FakeLLM) -> IntentRouter:
    return IntentRouter(lambda: llm, tool_registry=None)


def _template(template_id: str, pattern: str, *, source: str = "builtin") -> WorkflowTemplate:
    return WorkflowTemplate(
        id=template_id,
        title=template_id,
        goal_patterns=(pattern,),
        slots=(SlotSpec("task_id", True, "task_context", "Task id"),),
        steps=(
            StepTemplate(
                title="Echo",
                tool_ref=ToolRef("_sample", "echo"),
                inputs_template={"message": "{slot:task_id}"},
                depends_on_titles=(),
                post_checks=(),
            ),
        ),
        source=source,
    )


def test_intent_router_strong_keyword_match_does_not_call_llm():
    load_builtin_templates()
    llm = FakeLLM("novel")

    result = _router(llm).route("please run sample echo", {"message": "hi"})

    assert result.kind == "template"
    assert result.template_id == "sample_echo"
    assert result.confidence >= 0.75
    assert llm.calls == []


def test_intent_router_matches_active_user_skill_templates():
    clear_user_templates()
    register_user_template(_template("user_intent_echo", "custom echo", source="user"))
    llm = FakeLLM("novel")

    result = _router(llm).route("run custom echo now", {"task_id": "task-1"})

    assert result.kind == "template"
    assert result.template_id == "user_intent_echo"
    assert result.slots == {"task_id": "task-1"}


def test_intent_router_uses_llm_classification_when_no_keyword_matches():
    load_builtin_templates()
    llm = FakeLLM('{"choice":"sample_echo"}')

    result = _router(llm).route("do the thing", {"message": "hi"})

    assert result.kind == "template"
    assert result.template_id == "sample_echo"
    assert result.rationale == "llm classified"
    assert llm.calls


def test_intent_router_falls_back_to_novel_for_invalid_llm_choice():
    load_builtin_templates()
    llm = FakeLLM("invent_a_new_flow")

    result = _router(llm).route("do the thing", {})

    assert result.kind == "novel"
    assert result.template_id is None
    assert result.slots == {}


def test_intent_router_extracts_task_context_slots():
    template = _template("context_slot_template", "context slot")
    try:
        register_template(template)
    except ValueError:
        pass
    llm = FakeLLM("novel")

    result = _router(llm).route("context slot", {"task_id": "task-7"})

    assert result.template_id == "context_slot_template"
    assert result.slots == {"task_id": "task-7"}


def test_intent_router_matches_standard_modeling_without_llm_for_common_chinese_goal():
    load_builtin_templates()
    llm = FakeLLM("novel")
    task_context = {
        "dataset_id": "dataset-1",
        "target_col": "bad_flag",
        "feature_cols": ["income", "age"],
        "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "recipe": "lr",
        "seed": 7,
    }

    result = _router(llm).route("请帮我建模，训练一个A卡模型", task_context)

    assert result.kind == "template"
    assert result.template_id == "standard_modeling"
    assert result.slots == task_context
    assert llm.calls == []
