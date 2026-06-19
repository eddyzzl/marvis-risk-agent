from marvis.drafts import LearningNote
from marvis.drafts.learning import MAX_NOTE_CHARS, distill_learning


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_distill_learning_returns_bounded_note_with_sources():
    llm = _FakeLLM("  step 1\n\nstep 2\x00" + ("x" * (MAX_NOTE_CHARS + 100)))

    note = distill_learning(
        "scorecard monitoring",
        ["A" * 6000, "B" * 6000],
        ["https://example.test/a", "https://example.test/b"],
        llm_factory=lambda: llm,
    )

    assert isinstance(note, LearningNote)
    assert note.id.startswith("note-")
    assert note.query == "scorecard monitoring"
    assert note.sources == ("https://example.test/a", "https://example.test/b")
    assert "\x00" not in note.distilled
    assert len(note.distilled) <= MAX_NOTE_CHARS
    assert note.created_at

    call = llm.calls[0]
    assert call["stream"] is False
    assert "scorecard monitoring" in call["user_prompt"]
    assert "A" * 5000 in call["user_prompt"]
    assert "A" * 5001 not in call["user_prompt"]


def test_distill_learning_truncates_joined_prompt_input():
    llm = _FakeLLM("short note")
    contents = ["A" * 9000, "B" * 9000, "C" * 9000, "D" * 9000, "E" * 9000]

    distill_learning(
        "large docs",
        contents,
        ["https://example.test/a"],
        llm_factory=lambda: llm,
    )

    user_prompt = llm.calls[0]["user_prompt"]
    assert len(user_prompt) < 23_000
    assert "E" * 1000 not in user_prompt
