import pytest

import marvis.db as db_module
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


def test_draft_repository_saves_learning_note_with_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    note = _note()

    repo.save_learning_note_with_audit(
        note,
        audit={
            "kind": "draft.learning_note.create",
            "target_ref": note.id,
            "outcome": "succeeded",
            "detail": {"query": note.query, "source_count": len(note.sources)},
        },
    )

    assert repo.get_learning_note(note.id) == note
    audit = db_module.PluginRepository(db_path).list_audit(kind="draft.learning_note.create")[0]
    assert audit["target_ref"] == note.id
    assert audit["detail"]["source_count"] == 1


def test_draft_repository_rolls_back_learning_note_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    note = _note()

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.save_learning_note_with_audit(
            note,
            audit={
                "kind": "draft.learning_note.create",
                "target_ref": note.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_learning_note(note.id) is None


def test_draft_repository_saves_draft_with_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft()

    repo.save_draft_with_audit(
        draft,
        audit={
            "kind": "draft.author",
            "target_ref": draft.id,
            "outcome": "succeeded",
            "detail": {"task_id": draft.task_id, "source": draft.source},
        },
    )

    assert repo.get_draft(draft.id) == draft
    audit = db_module.PluginRepository(db_path).list_audit(kind="draft.author")[0]
    assert audit["target_ref"] == draft.id
    assert audit["detail"]["task_id"] == draft.task_id


def test_draft_repository_rolls_back_draft_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft()

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.save_draft_with_audit(
            draft,
            audit={
                "kind": "draft.author",
                "target_ref": draft.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_draft(draft.id) is None


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


def test_draft_repository_saves_run_status_and_audit_atomically(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft()
    run = _run()
    repo.save_draft(draft)

    repo.save_draft_run_with_status_audit(
        run,
        status="tested",
        audit={
            "kind": "draft.run.record",
            "target_ref": draft.id,
            "outcome": "succeeded",
            "detail": {"run_id": run.id, "task_id": run.task_id},
        },
    )

    assert repo.list_runs(draft.id) == [run]
    assert repo.get_draft(draft.id).status == "tested"
    audit = db_module.PluginRepository(db_path).list_audit(kind="draft.run.record")[0]
    assert audit["target_ref"] == draft.id
    assert audit["detail"]["run_id"] == run.id


def test_draft_repository_rolls_back_run_and_status_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    draft = _draft()
    run = _run()
    repo.save_draft(draft)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(db_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.save_draft_run_with_status_audit(
            run,
            status="tested",
            audit={
                "kind": "draft.run.record",
                "target_ref": draft.id,
                "outcome": "succeeded",
            },
        )

    assert repo.list_runs(draft.id) == []
    assert repo.get_draft(draft.id).status == "draft"


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
