import json

import pytest

from marvis.drafts import LearningNote
from marvis.drafts.errors import DraftError
from marvis.drafts.learning import distill_learning
from marvis.llm_outbound_learning import _fetched_public_source


class _FakeLLM:
    def __init__(self, response):
        self.response, self.calls = response, []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.response)


def _source(text="Public WOE and PSI methodology."):
    return _fetched_public_source("https://example.test/docs", "https://example.test/docs", text.encode(), text)


def test_distill_learning_returns_note_with_exact_verified_public_sources():
    source = _source()
    llm = _FakeLLM({"note": "Define WOE bins and measure PSI locally."})
    note = distill_learning("scorecard monitoring", [source.text], [source.requested_url],
        public_sources=[source], llm_factory=lambda: llm)
    assert isinstance(note, LearningNote)
    assert note.query == "scorecard monitoring" and note.sources == (source.requested_url,)
    assert note.distilled == "Define WOE bins and measure PSI locally."
    assert note.id.startswith("note-") and note.created_at
    context = json.loads(llm.calls[0]["user_prompt"])
    assert context["public_sources"][0]["public_text"] == source.text
    assert context["public_sources"][0]["text_sha256"] == source.text_sha256


@pytest.mark.parametrize("mode", ["missing_provenance", "mismatched_body", "oversized_body", "private_public_body"])
def test_learning_refuses_unsupported_sources_without_requests_or_silent_truncation(mode):
    llm = _FakeLLM({"note": "should not run"})
    source = _source("A" * 5001 if mode == "oversized_body" else "api_key=secret123" if mode == "private_public_body" else "Public source.")
    with pytest.raises(DraftError):
        distill_learning("large docs", ["different body" if mode == "mismatched_body" else source.text],
            [source.requested_url], public_sources=[] if mode == "missing_provenance" else [source], llm_factory=lambda: llm)
    assert llm.calls == []


def test_learning_does_not_save_invalid_or_oversized_model_response_as_success():
    source = _source()
    llm = _FakeLLM({"note": "x" * 4001})
    with pytest.raises(DraftError, match="未保存学习成功"):
        distill_learning("methodology", [source.text], [source.requested_url], public_sources=[source], llm_factory=lambda: llm)
    assert len(llm.calls) == 1
