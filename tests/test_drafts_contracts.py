from dataclasses import asdict

import pytest

import marvis.drafts as draft_contracts
from marvis.drafts import (
    DraftRun,
    DraftStateError,
    DraftTool,
    LearningNote,
    PromotionCheck,
    assert_draft_status_transition,
)


def test_learning_note_contract_round_trips_sources():
    note = LearningNote(
        id="note-1",
        query="how to compute reject inference",
        sources=("https://example.test/a", "https://example.test/b"),
        distilled="步骤、公式和关键 API 摘要。",
        created_at="2026-06-19T00:00:00Z",
    )

    payload = asdict(note)

    assert LearningNote(**payload) == note
    assert payload["sources"] == ("https://example.test/a", "https://example.test/b")


def test_draft_tool_contract_round_trips_schema_and_status():
    draft = DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code="def calc_margin(inputs, ctx):\n    return {'margin': 1}\n",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {"margin": {"type": "number"}}},
        determinism="deterministic",
        source="llm_generated",
        learning_note_id=None,
        status="draft",
        created_at="2026-06-19T00:00:00Z",
    )

    payload = asdict(draft)

    assert DraftTool(**payload) == draft
    assert payload["input_schema"]["type"] == "object"
    assert payload["learning_note_id"] is None


def test_draft_run_and_promotion_check_contracts_round_trip():
    run = DraftRun(
        id="run-1",
        draft_id="draft-1",
        task_id="task-1",
        inputs_hash="abc123",
        ok=True,
        output={"value": 3},
        error=None,
        at="2026-06-19T00:01:00Z",
    )
    check = PromotionCheck(
        passed=False,
        problems=("at least one test case required",),
        test_result=None,
    )

    assert DraftRun(**asdict(run)) == run
    assert PromotionCheck(**asdict(check)) == check


def test_draft_status_transition_guard_allows_governed_flow():
    assert_draft_status_transition("draft", "tested")
    assert_draft_status_transition("draft", "rejected")
    assert_draft_status_transition("tested", "promoted")
    assert_draft_status_transition("tested", "rejected")
    assert_draft_status_transition("promoted", "promoted")


def test_draft_status_transition_guard_rejects_bypass_to_promoted():
    with pytest.raises(DraftStateError, match="draft -> promoted"):
        assert_draft_status_transition("draft", "promoted")
    with pytest.raises(DraftStateError, match="promoted -> tested"):
        assert_draft_status_transition("promoted", "tested")
    with pytest.raises(DraftStateError, match="unknown"):
        assert_draft_status_transition("unknown", "draft")


def test_drafts_package_exports_contract_surface():
    assert draft_contracts.LearningNote is LearningNote
    assert draft_contracts.DraftTool is DraftTool
    assert draft_contracts.DraftRun is DraftRun
    assert draft_contracts.PromotionCheck is PromotionCheck
    assert draft_contracts.DraftStateError is DraftStateError
    assert "DraftTool" in draft_contracts.__all__
