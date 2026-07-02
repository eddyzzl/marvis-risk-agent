import json

import pytest

from marvis.drafts import AuthoringError, DraftTool, LearningNote
from marvis.drafts.authoring import draft_script


class _FakeLLM:
    def __init__(self, response: dict | str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, str):
            return self.response
        return json.dumps(self.response)


def _valid_spec(**overrides):
    spec = {
        "name": "calc_margin",
        "summary": "Calculate margin.",
        "code": "def calc_margin(inputs: dict, ctx) -> dict:\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        "input_schema": {
            "type": "object",
            "properties": {
                "revenue": {"type": "number"},
                "cost": {"type": "number"},
            },
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"margin": {"type": "number"}},
            "required": ["margin"],
            "additionalProperties": False,
        },
        "determinism": "deterministic",
    }
    spec.update(overrides)
    return spec


def _note() -> LearningNote:
    return LearningNote(
        id="note-1",
        query="margin formula",
        sources=("https://example.test/a",),
        distilled="Use revenue - cost.",
        created_at="2026-06-19T00:00:00Z",
    )


def test_draft_script_creates_valid_draft_from_learning_note():
    llm = _FakeLLM(_valid_spec())

    draft = draft_script("task-1", "build margin calculator", learning_note=_note(), llm_factory=lambda: llm)

    assert isinstance(draft, DraftTool)
    assert draft.id.startswith("draft-")
    assert draft.task_id == "task-1"
    assert draft.name == "calc_margin"
    assert draft.source == "web_learning"
    assert draft.learning_note_id == "note-1"
    assert draft.status == "draft"
    assert draft.input_schema["required"] == ["revenue", "cost"]
    assert "Use revenue - cost." in llm.calls[0]["user_prompt"]
    assert llm.calls[0]["response_format"] == {"type": "json_object"}


def test_draft_script_without_learning_note_is_llm_generated():
    llm = _FakeLLM(
        _valid_spec(
            name="calc_roi",
            code="def calc_roi(inputs: dict, ctx) -> dict:\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        )
    )

    draft = draft_script("task-1", "build ROI calculator", learning_note=None, llm_factory=lambda: llm)

    assert draft.name == "calc_roi"
    assert draft.source == "llm_generated"
    assert draft.learning_note_id is None


def test_draft_script_rejects_missing_schema_and_invalid_determinism():
    missing_schema = _valid_spec()
    missing_schema.pop("output_schema")
    with pytest.raises(AuthoringError, match="output_schema"):
        draft_script("task-1", "bad", learning_note=None, llm_factory=lambda: _FakeLLM(missing_schema))

    with pytest.raises(AuthoringError, match="determinism"):
        draft_script(
            "task-1",
            "bad",
            learning_note=None,
            llm_factory=lambda: _FakeLLM(_valid_spec(determinism="unknown")),
        )


def test_draft_script_rejects_dangerous_code_and_invalid_json():
    with pytest.raises(AuthoringError, match="banned"):
        draft_script(
            "task-1",
            "bad",
            learning_note=None,
            llm_factory=lambda: _FakeLLM(_valid_spec(code="def calc(inputs, ctx):\n    os.system('rm -rf /')\n")),
        )
    with pytest.raises(AuthoringError, match="JSON"):
        draft_script("task-1", "bad", learning_note=None, llm_factory=lambda: _FakeLLM("not json"))


@pytest.mark.parametrize(
    "code",
    [
        "def calc_margin(inputs: dict, ctx) -> dict:\n"
        "    urllib.request.urlopen('https://example.test')\n"
        "    return {'margin': 0}\n",
        "def calc_margin(inputs: dict, ctx) -> dict:\n"
        "    Path('/tmp/marvis-draft').write_text('bad')\n"
        "    return {'margin': 0}\n",
        "def calc_margin(inputs: dict, ctx) -> dict:\n"
        "    secret = Path('/tmp/marvis-draft').read_text()\n"
        "    return {'margin': len(secret)}\n",
        "def calc_margin(inputs: dict, ctx) -> dict:\n"
        "    os.remove('/tmp/marvis-draft')\n"
        "    return {'margin': 0}\n",
        "import os\n"
        "def calc_margin(inputs: dict, ctx) -> dict:\n"
        "    return {'margin': len(os.environ)}\n",
    ],
)
def test_draft_script_rejects_network_file_write_and_file_delete_calls(code):
    with pytest.raises(AuthoringError, match="banned"):
        draft_script(
            "task-1",
            "bad",
            learning_note=None,
            llm_factory=lambda: _FakeLLM(_valid_spec(code=code)),
        )


class _SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, dict):
            return json.dumps(value)
        return value


def test_draft_script_retries_after_unparseable_first_reply():
    fenced = "```json\n" + json.dumps(_valid_spec()) + "\n```"
    llm = _SequenceLLM(["garbage not json", fenced])

    draft = draft_script(
        "task-1",
        "build margin calculator",
        learning_note=None,
        llm_factory=lambda: llm,
    )

    assert isinstance(draft, DraftTool)
    assert draft.name == "calc_margin"
    # Two LLM calls: the failed first, then the retry with error feedback.
    assert len(llm.calls) == 2
    assert "未通过校验" in llm.calls[1]["user_prompt"]


def test_draft_script_recovers_fence_wrapped_json_on_first_reply():
    fenced = "```json\n" + json.dumps(_valid_spec()) + "\n```"
    llm = _SequenceLLM([fenced])

    draft = draft_script(
        "task-1",
        "build margin calculator",
        learning_note=None,
        llm_factory=lambda: llm,
    )

    assert draft.name == "calc_margin"
    assert len(llm.calls) == 1


def test_draft_script_raises_after_two_failed_attempts():
    bad = _valid_spec(determinism="unknown")
    llm = _SequenceLLM([bad, bad])

    with pytest.raises(AuthoringError, match="determinism"):
        draft_script(
            "task-1",
            "bad",
            learning_note=None,
            llm_factory=lambda: llm,
        )

    assert len(llm.calls) == 2


def test_draft_script_retry_still_enforces_safety_floor():
    unsafe = _valid_spec(code="def calc_margin(inputs, ctx):\n    os.system('rm -rf /')\n")
    llm = _SequenceLLM([unsafe, unsafe])

    with pytest.raises(AuthoringError, match="banned"):
        draft_script(
            "task-1",
            "bad",
            learning_note=None,
            llm_factory=lambda: llm,
        )

    assert len(llm.calls) == 2
