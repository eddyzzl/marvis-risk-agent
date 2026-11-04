"""Questions about finished results get read-only answers (eval S7).

"所以哪个月放款的资产质量最差？参考 MOB 是第几期？" on a finished standard
Vintage task used to hit the top-level router and come back as "我还不能安全
确定这句话要进入哪条流程".  The answer must come from the stored step output,
without a plan, a Tool or an LLM call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from marvis.agent.turn_handlers import semantic as semantic_lane
from marvis.app import create_app


class _CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs):  # pragma: no cover - must never be called
        self.calls += 1
        raise AssertionError("result questions must not call an LLM")


def _panel(source: Path) -> None:
    bads_at_mob3 = {"2025-01": 1, "2025-02": 3, "2025-03": 2}
    rows = []
    for cohort, bad_loans in bads_at_mob3.items():
        for loan in range(10):
            for mob in range(7):
                rows.append(
                    {
                        "cohort": cohort,
                        "loan_id": f"{cohort}-{loan}",
                        "mob": mob,
                        "bad": int(loan < bad_loans and mob == 3),
                    }
                )
    pd.DataFrame(rows).to_csv(source / "vintage_panel.csv", index=False)


def test_finished_vintage_question_is_answered_from_platform_output(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    client = TestClient(create_app(workspace))
    source = workspace / "vintage"
    source.mkdir(parents=True)
    _panel(source)
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "看一下这几批放款的 vintage",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "vintage",
            "run_mode": "manual",
            "target_col": "bad",
            "time_col": "cohort",
        },
    )
    assert created.status_code == 200, created.text
    task_id = created.json()["id"]
    url = f"/api/tasks/{task_id}/agent"
    assert client.post(url + "/start", json={}).status_code == 202
    planned = client.post(
        url + "/messages", json={"content": "标准 Vintage，bad 是当期新增"}
    )
    assert planned.json()["messages"][-1]["metadata"]["kind"] == "plan_overview"
    done = client.post(url + "/messages", json={"content": "确认"})
    assert done.status_code == 202, done.text
    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    assert [plan["status"] for plan in plans] == ["done"]

    llm = _CountingLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    asked = client.post(
        url + "/messages",
        json={"content": "所以哪个月放款的资产质量最差？参考 MOB 是第几期？"},
    )

    assert asked.status_code == 202, asked.text
    answer = asked.json()["messages"][-1]
    assert answer["metadata"]["kind"] == "result_answer"
    assert answer["metadata"]["read_only"] is True
    content = answer["content"]
    assert "参考 MOB 是 MOB6" in content
    assert "资产质量最差：2025-02，MOB6 累计坏账率 30.0%" in content
    assert "最好：2025-01，10.0%" in content
    assert "没有重新计算" in content
    assert llm.calls == 0
    assert client.get(f"/api/tasks/{task_id}/plans").json()["plans"] == plans


def test_vintage_answer_names_immature_cohorts_and_ties() -> None:
    answer = semantic_lane._vintage_result_answer(
        {
            "cohorts": ["2025-01", "2025-02", "2025-03"],
            "mob_axis": [1, 2, 3],
            "summary": {
                "at_ref": {"2025-01": 0.04, "2025-02": 0.04},
                "trend": "stable",
            },
            "label_semantics": "snapshot",
        },
        {"ref_mob": 3},
    )

    assert answer is not None
    assert "最差：2025-01、2025-02，MOB3 坏账率 4.0%" in answer
    assert "还没到 MOB3 的批次没有参与比较：2025-03" in answer
    assert "从 0 开始编号" not in answer
    assert semantic_lane._vintage_result_answer({"summary": {}}, {}) is None


def test_refusal_on_finished_task_does_not_list_workflow_names() -> None:
    class _Repo:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        def add_agent_message(self, task_id, *, role, stage, content, metadata):
            self.messages.append(
                {"role": role, "content": content, "metadata": metadata}
            )

        def list_agent_messages(self, task_id):
            return list(self.messages)

    class _Task:
        id = "task"

    repo = _Repo()
    semantic_lane._semantic_intent_clarification_response(
        repo,
        _Task(),
        user_text="为什么 2025-02 这么差？",
        reason="用户是在追问，属于疑问，未明确授权继续当前节点。",
        about_results=True,
    )

    reply = repo.messages[-1]
    assert "没有创建计划" in reply["content"]
    assert "未明确授权" not in reply["content"]
    assert "Vintage/VTG、策略" not in reply["content"]
    assert reply["metadata"]["reason"].startswith("用户是在追问")
