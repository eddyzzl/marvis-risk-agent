from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.drafts import DraftTool


def _admin_headers(client) -> dict:
    return {"X-MARVIS-Plugin-Admin": client.app.state.plugin_admin_token}


def _draft() -> DraftTool:
    return DraftTool(
        id="draft-1",
        task_id="task-1",
        name="calc_margin",
        summary="Calculate margin.",
        code="def calc_margin(inputs, ctx):\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        input_schema={
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"margin": {"type": "number"}},
            "required": ["margin"],
            "additionalProperties": False,
        },
        determinism="deterministic",
        source="hand_written",
        learning_note_id=None,
        status="draft",
        created_at="2026-06-19T00:00:00Z",
    )


def test_draft_stays_out_of_planner_catalog_until_admin_promotion(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.app.state.draft_registry.add(_draft())

    before = client.app.state.tool_registry.catalog_for_planner()
    run = client.post("/api/drafts/draft-1/run", json={"inputs": {"revenue": 10, "cost": 3}})
    after_run = client.app.state.tool_registry.catalog_for_planner()
    promoted = client.post(
        "/api/drafts/draft-1/promote",
        json={"test_cases": [{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}]},
        headers=_admin_headers(client),
    )
    after_promote = client.app.state.tool_registry.catalog_for_planner()

    assert run.status_code == 200
    assert run.json()["ok"] is True
    assert promoted.status_code == 200
    assert not _catalog_has_tool(before, "calc_margin")
    assert not _catalog_has_tool(after_run, "calc_margin")
    assert _catalog_has_tool(after_promote, "calc_margin")


def _catalog_has_tool(catalog: list[dict], tool_name: str) -> bool:
    return any(item["tool"] == tool_name for item in catalog)
