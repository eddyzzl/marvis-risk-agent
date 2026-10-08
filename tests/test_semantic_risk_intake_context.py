"""Risk intake replies carry the platform question, never arbitrary prior text."""

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.agent.turn_handlers.semantic import _semantic_intent_route_contract
from marvis.app import create_app
from test_semantic_join_intent import _install_semantic_client, _intent_reply


def test_risk_intake_context_projects_only_known_business_choices():
    state = {
        "phase": "request_materials",
        "analysis_kind": "standard_vintage",
        "label_semantics": "incremental",
        "analysis_scope": "PRIVATE_SCOPE",
        "raw_sample": "PRIVATE_SAMPLE",
    }
    messages = [
        {"role": "assistant", "metadata": {"risk_analysis_intake": state}},
        {
            "role": "user",
            "metadata": {
                "risk_analysis_intake": {
                    "phase": "ready",
                    "analysis_kind": "profitability",
                }
            },
        },
    ]
    repo = SimpleNamespace(list_agent_messages=lambda task_id: messages)
    task = SimpleNamespace(id="risk", task_type="vintage")
    context, allowed, pending = _semantic_intent_route_contract(None, repo, task)
    assert context["risk_setup_phase"] == "request_materials"
    assert context["analysis_kind"] == "standard_vintage"
    assert context["label_semantics"] == "incremental"
    assert "材料" in context["pending_question"]
    assert "PRIVATE" not in json.dumps(context)
    assert "current_workflow" in allowed and pending is None
    state["label_semantics"] = "PRIVATE_UNKNOWN"
    assert "label_semantics" not in _semantic_intent_route_contract(None, repo, task)[0]
    state["analysis_kind"] = "PRIVATE_UNKNOWN"
    assert _semantic_intent_route_contract(None, repo, task)[0] == {
        "risk_setup_phase": "request_materials"
    }


@pytest.mark.parametrize(
    "blocked_flag",
    ["withholds_action", "requests_change", "is_question", "is_conditional"],
)
def test_actual_api_intake_context_never_overrides_independent_review_flags(
    tmp_path, monkeypatch, blocked_flag
):
    class Client:
        def __init__(self):
            self.requests = []

        def complete(self, **kwargs):
            request = json.loads(kwargs["user_prompt"])
            self.requests.append((kwargs["caller"], request))
            context = request["current_context"]
            intent = (
                "risk_standard_vintage"
                if context["risk_setup_phase"] == "ask_goal"
                else "current_workflow"
            )
            overrides = {}
            if (
                context["risk_setup_phase"] != "ask_goal"
                and kwargs["caller"] == "semantic_intent_reviewer"
            ):
                overrides[blocked_flag] = True
            return _intent_reply(request["instruction"], intent=intent, **overrides)

    llm = Client()
    _install_semantic_client(monkeypatch, llm)
    source = tmp_path / "workspace" / "materials"
    source.mkdir(parents=True)
    pd.DataFrame(
        {
            "cohort": ["2025-01", "2025-01", "2025-02", "2025-02"],
            "mob": [0, 1, 0, 1],
            "bad": [0, 1, 0, 0],
        }
    ).to_csv(source / "panel.csv", index=False)
    with TestClient(create_app(tmp_path / "workspace")) as client:
        created = client.post(
            "/api/tasks",
            json={
                "model_name": "intake",
                "model_version": "v1",
                "validator": "qa",
                "source_dir": str(source),
                "task_type": "vintage",
                "run_mode": "agent",
                "target_col": "bad",
                "time_col": "cohort",
            },
        )
        assert created.status_code == 200, created.text
        task = created.json()
        url = f"/api/tasks/{task['id']}/agent"
        assert client.post(url + "/start", json={}).status_code == 202
        first = client.post(
            url + "/messages",
            json={"content": "做标准 Vintage，bad 是 incremental，不是 snapshot。"},
        )
        assert first.status_code == 202, first.text
        before = client.get(f"/api/tasks/{task['id']}/plans").json()["plans"]
        second = client.post(
            url + "/messages",
            json={
                "content": "材料已上传，覆盖全部 cohort 和 MOB；bad 是 incremental 当期新增，不是 snapshot。"
            },
        )
        assert second.status_code == 202, second.text
        last = [m for m in second.json()["messages"] if m["role"] == "assistant"][-1]
        diagnostic = last["metadata"]["semantic_diagnostics"]
        assert diagnostic["failure_code"] == "semantic_intent_unsafe_decision"
        assert diagnostic["passes"][1]["attempts"][0]["decision"][blocked_flag] is True
        contexts = [r["current_context"] for _, r in llm.requests[-2:]]
        assert contexts[0] == contexts[1]
        assert contexts[0]["analysis_kind"] == "standard_vintage"
        assert contexts[0]["label_semantics"] == "incremental"
        assert contexts[0]["pending_question"]
        assert (
            client.get(f"/api/tasks/{task['id']}/plans").json()["plans"] == before == []
        )
