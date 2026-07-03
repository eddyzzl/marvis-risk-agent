import pytest

from marvis.db import DraftRepository, PluginRepository, init_db
from marvis.drafts import DraftNotFound, DraftTool
from marvis.drafts.registry import DraftRegistry
from marvis.plugins.registry import PluginRegistry, ToolRegistry


def _draft() -> DraftTool:
    return DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code="def calc_margin(inputs, ctx):\n    return {'margin': 1}\n",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        determinism="deterministic",
        source="hand_written",
        learning_note_id=None,
        status="draft",
        created_at="2026-06-19T00:00:00Z",
    )


def test_draft_registry_crud_delegates_to_repository(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = DraftRegistry(DraftRepository(db_path))
    draft = _draft()

    registry.add(draft)
    assert registry.get(draft.id) == draft
    assert registry.list_for_task("task-1") == [draft]
    assert registry.list_for_task("task-1", status="draft") == [draft]

    registry.set_status(draft.id, "tested")
    assert registry.get(draft.id).status == "tested"
    assert registry.list_for_task("task-1", status="draft") == []
    assert registry.list_for_task("task-1", status="tested") == [registry.get(draft.id)]


def test_draft_registry_add_with_audit_delegates_to_repository(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = DraftRepository(db_path)
    registry = DraftRegistry(repo)
    draft = _draft()

    registry.add_with_audit(
        draft,
        audit={
            "kind": "draft.author",
            "target_ref": draft.id,
            "outcome": "succeeded",
            "detail": {"task_id": draft.task_id},
        },
    )

    assert registry.get(draft.id) == draft
    audit = PluginRepository(db_path).list_audit(kind="draft.author")[0]
    assert audit["target_ref"] == draft.id


def test_draft_registry_raises_for_missing_draft(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    registry = DraftRegistry(DraftRepository(db_path))

    with pytest.raises(DraftNotFound, match="missing"):
        registry.get("missing")


def test_draft_registry_is_not_visible_to_planner_catalog(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    drafts = DraftRegistry(DraftRepository(db_path))
    drafts.add(_draft())
    plugin_registry = PluginRegistry(PluginRepository(db_path))
    planner_catalog = ToolRegistry(plugin_registry).catalog_for_planner()

    assert planner_catalog == []
    assert all(item.get("tool") != "calc_margin" for item in planner_catalog)
