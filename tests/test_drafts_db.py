import pytest

from marvis.db import DraftRepository, connect, init_db
from marvis.drafts import DraftRun, DraftStateError, DraftTool, LearningNote


def _note() -> LearningNote:
    return LearningNote(
        id="note-1",
        query="scorecard monitoring",
        sources=("https://example.test/a",),
        distilled="bounded implementation notes",
        created_at="2026-06-19T00:00:00Z",
    )


def _draft() -> DraftTool:
    return DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code="def calc_margin(inputs, ctx):\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        input_schema={"type": "object", "properties": {"revenue": {"type": "number"}}},
        output_schema={"type": "object", "properties": {"margin": {"type": "number"}}},
        determinism="deterministic",
        source="web_learning",
        learning_note_id="note-1",
        status="draft",
        created_at="2026-06-19T00:01:00Z",
    )


def _run() -> DraftRun:
    return DraftRun(
        id="run-1",
        draft_id="draft-1",
        task_id="task-1",
        inputs_hash="abc123",
        ok=True,
        output={"margin": 12.5},
        error=None,
        at="2026-06-19T00:02:00Z",
    )


def test_draft_repository_round_trips_note_draft_and_run(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    note = _note()
    draft = _draft()
    run = _run()

    repo.save_learning_note(note)
    repo.save_draft(draft)
    repo.save_draft_run(run)

    assert repo.get_learning_note(note.id) == note
    assert repo.get_draft(draft.id) == draft
    assert repo.list_drafts("task-1") == [draft]
    assert repo.list_drafts("task-1", status="draft") == [draft]
    assert repo.list_runs(draft.id) == [run]


def test_draft_repository_status_transition_guard(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft()
    repo.save_draft(draft)

    with pytest.raises(DraftStateError, match="draft -> promoted"):
        repo.set_status(draft.id, "promoted")

    repo.set_status(draft.id, "tested")
    assert repo.get_draft(draft.id).status == "tested"
    repo.set_status(draft.id, "promoted")
    assert repo.get_draft(draft.id).status == "promoted"


def test_draft_runs_cascade_when_draft_is_deleted(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    note = _note()
    draft = _draft()
    run = _run()
    repo.save_learning_note(note)
    repo.save_draft(draft)
    repo.save_draft_run(run)

    with connect(db_path) as conn:
        conn.execute("DELETE FROM draft_tools WHERE id = ?", (draft.id,))

    assert repo.get_draft(draft.id) is None
    assert repo.list_runs(draft.id) == []
    assert repo.get_learning_note(note.id) == note
