from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.drafts import DraftTool


ADMIN_HEADERS = {"X-MARVIS-Plugin-Admin": "local-dev"}


def _draft(**overrides) -> DraftTool:
    payload = {
        "id": "draft-1",
        "task_id": "task-1",
        "name": "calc_margin",
        "summary": "Calculate margin.",
        "code": "def calc_margin(inputs, ctx):\n    return {'margin': inputs['revenue'] - inputs['cost']}\n",
        "input_schema": {
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
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
        "source": "hand_written",
        "learning_note_id": None,
        "status": "draft",
        "created_at": "2026-06-19T00:00:00Z",
    }
    payload.update(overrides)
    return DraftTool(**payload)


def _client_with_draft(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.app.state.draft_registry.add(_draft())
    return client


def test_draft_routes_are_served_from_dedicated_router():
    from marvis.routers import drafts

    route_paths = {route.path for route in drafts.router.routes}

    assert "/api/drafts" in route_paths
    assert "/api/drafts/{draft_id}" in route_paths
    assert all(route.endpoint.__module__ == "marvis.routers.drafts" for route in drafts.router.routes)


def test_list_and_detail_draft_endpoints(tmp_path):
    client = _client_with_draft(tmp_path)
    client.app.state.draft_registry.add(
        _draft(
            id="draft-2",
            task_id="task-2",
            status="tested",
            created_at="2026-06-19T00:01:00Z",
        )
    )

    listed = client.get("/api/drafts?task_id=task-1")
    all_listed = client.get("/api/drafts")
    tested = client.get("/api/drafts?status=tested")
    detail = client.get("/api/drafts/draft-1")

    assert listed.status_code == 200
    assert listed.json()["drafts"][0]["id"] == "draft-1"
    assert listed.json()["drafts"][0]["code"] is None
    assert [draft["id"] for draft in all_listed.json()["drafts"]] == ["draft-1", "draft-2"]
    assert [draft["id"] for draft in tested.json()["drafts"]] == ["draft-2"]
    assert detail.status_code == 200
    assert detail.json()["draft"]["code"].startswith("def calc_margin")
    assert detail.json()["runs"] == []


def test_run_draft_endpoint_records_run(tmp_path):
    client = _client_with_draft(tmp_path)

    response = client.post(
        "/api/drafts/draft-1/run",
        json={"inputs": {"revenue": 10, "cost": 3}},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["output"] == {"margin": 7}
    detail = client.get("/api/drafts/draft-1").json()
    assert detail["runs"][0]["ok"] is True
    assert detail["draft"]["status"] == "tested"


def test_draft_web_search_endpoint_returns_offline_guidance(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))

    def fake_web_search(payload, _ctx):
        assert payload == {"query": "learn joins", "max_results": 3}
        return {
            "results": [],
            "offline": True,
            "guidance": "No network. Produce the tool externally, then upload it as a plugin.",
        }

    monkeypatch.setattr("marvis.routers.drafts.tool_web_search", fake_web_search)

    response = client.post(
        "/api/drafts/web-search",
        json={"query": "learn joins", "max_results": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [],
        "offline": True,
        "guidance": "No network. Produce the tool externally, then upload it as a plugin.",
    }


def test_promote_draft_endpoint_requires_admin_and_registers_plugin(tmp_path):
    client = _client_with_draft(tmp_path)
    body = {"test_cases": [{"inputs": {"revenue": 10, "cost": 3}, "expect": {"margin": 7}}]}

    denied = client.post("/api/drafts/draft-1/promote", json=body)
    promoted = client.post("/api/drafts/draft-1/promote", json=body, headers=ADMIN_HEADERS)

    assert denied.status_code == 403
    assert promoted.status_code == 200
    assert promoted.json()["check"]["passed"] is True
    assert promoted.json()["plugin"]["tool_count"] == 1
    catalog = client.app.state.tool_registry.catalog_for_planner()
    assert any(item["tool"] == "calc_margin" for item in catalog)
    assert client.get("/api/drafts/draft-1").json()["draft"]["status"] == "promoted"


def test_promote_draft_endpoint_rejects_malformed_test_cases(tmp_path):
    client = _client_with_draft(tmp_path)

    response = client.post(
        "/api/drafts/draft-1/promote",
        json={"test_cases": [{"expect": {"margin": 7}}]},
        headers=ADMIN_HEADERS,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["check"]["passed"] is False
    assert response.json()["detail"]["check"]["problems"] == [
        "test case 1 inputs must be an object"
    ]


def test_reject_draft_endpoint_requires_admin_and_writes_audit(tmp_path):
    client = _client_with_draft(tmp_path)

    denied = client.post("/api/drafts/draft-1/reject", json={"reason": "not useful"})
    rejected = client.post(
        "/api/drafts/draft-1/reject",
        json={"reason": "not useful"},
        headers=ADMIN_HEADERS,
    )

    assert denied.status_code == 403
    assert rejected.status_code == 200
    assert client.get("/api/drafts/draft-1").json()["draft"]["status"] == "rejected"
    audits = client.app.state.plugin_repo.list_audit(kind="draft.reject")
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "draft-1"
    assert audits[0]["detail"]["reason"] == "not useful"
