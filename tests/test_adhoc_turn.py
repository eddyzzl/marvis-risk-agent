"""S6 收尾接线: ad-hoc 问数 intent branch inside dispatch_driver_turn.

The wiring only ever fires when (a) the task has a ready dataset, (b) there is no
active plan / open gate, and (c) detect_question_intent is true; otherwise the
turn falls straight through to the normal type handler (never hijacked). A
validated口径确认门 is stored on its own message metadata (the join-C1 /
portfolio-states pending-state precedent), a confirm runs the single-step
slice_aggregate plan on the REAL tool (hand-calculated numbers + audit row), a
non-confirm drops the pending spec, and a hallucinated column yields a Chinese
clarification with no stored state (INV-1: the LLM only ever emits a spec).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.db_schema import connect


class _FakeLLM:
    """Returns one fixed JSON spec (or clarify) payload; records call count so a
    test can prove INV-1 — the LLM is invoked once for parsing, never for numbers."""

    def __init__(self, payload):
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = 0

    def complete(self, **kwargs) -> str:
        self.calls += 1
        return self._payload


def _install_llm(monkeypatch, llm) -> None:
    # dispatch_driver_turn builds runtime.llm_client via driver_llm_client(); the
    # adhoc branch calls .complete on it. Manual mode normally returns None, so we
    # inject the FakeLLM at the module boundary the service resolves it from.
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


def _frame() -> pd.DataFrame:
    # 4-row hand-calculable dataset. May-only bad_rate by channel:
    #   A -> rows bad=1, bad=0            -> 1/2 = 0.5
    #   B -> rows bad=1, bad=1            -> 2/2 = 1.0
    return pd.DataFrame({
        "channel": ["A", "A", "B", "B"],
        "month": ["2026-05", "2026-05", "2026-05", "2026-05"],
        "bad": [1, 0, 1, 1],
        "amount": [100, 200, 300, 400],
    })


def _register_dataset(client: TestClient, task_id: str, tmp_path: Path) -> str:
    settings = client.app.state.settings
    root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(root), root)
    path = tmp_path / "adhoc.csv"
    _frame().to_csv(path, index=False)
    return registry.register_from_upload(task_id, path, role="sample").id


def _make_task_with_ready_dataset(client: TestClient, tmp_path: Path) -> str:
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "问数任务",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "feature_analysis",
            "run_mode": "manual",
            "target_col": "bad",
        },
    ).json()["id"]
    _register_dataset(client, task_id, tmp_path)
    return task_id


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


def _post(client: TestClient, task_id: str, content: str):
    return client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": content})


# -- (1) intent does NOT trigger: a plain instruction takes the normal path -----
def test_non_question_instruction_does_not_enter_adhoc_branch(client, tmp_path, monkeypatch):
    spy = _FakeLLM({"clarify": "should never be called"})
    _install_llm(monkeypatch, spy)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    # "确认" is not a data question -> detect_question_intent False -> normal flow.
    resp = _post(client, task_id, "确认")
    assert resp.status_code in (202, 409), resp.text
    last = _last_assistant(resp.json()["messages"])
    assert "adhoc_spec" not in (last.get("metadata") or {})
    # The adhoc parser LLM was never consulted for a non-question turn.
    assert spy.calls == 0


# -- (2) intent triggers -> 口径确认门 with the pending spec stored on metadata --
def test_question_intent_opens_confirmation_gate(client, tmp_path, monkeypatch):
    llm = _FakeLLM({
        "group_by": ["channel"],
        "metrics": [{"op": "bad_rate", "col": "bad"}],
        "month_col": "month",
        "months": ["2026-05"],
    })
    _install_llm(monkeypatch, llm)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    resp = _post(client, task_id, "按渠道看 5 月坏率")
    assert resp.status_code == 202, resp.text
    last = _last_assistant(resp.json()["messages"])
    # 口径确认门先行: a plain-Chinese echo, ending in 确认？, spec stashed on metadata.
    assert "将按〔channel〕统计〔bad 的坏率〕" in last["content"]
    assert last["content"].endswith("，确认？")
    spec = last["metadata"]["adhoc_spec"]
    assert spec["group_by"] == ["channel"]
    assert spec["metrics"] == [{"op": "bad_rate", "col": "bad"}]
    assert spec["months"] == ["2026-05"]
    assert llm.calls == 1  # INV-1: one parse call, no number computed by the LLM.


# -- (3) confirm -> single-step slice_aggregate runs on the REAL tool -----------
def test_confirm_runs_slice_aggregate_and_renders_table(client, tmp_path, monkeypatch):
    llm = _FakeLLM({
        "group_by": ["channel"],
        "metrics": [{"op": "bad_rate", "col": "bad"}, {"op": "count"}],
        "month_col": "month",
        "months": ["2026-05"],
    })
    _install_llm(monkeypatch, llm)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    _post(client, task_id, "按渠道看 5 月坏率")
    done = _post(client, task_id, "确认")
    assert done.status_code == 202, done.text
    last = _last_assistant(done.json()["messages"])
    # append_driver_messages stores every driver message with stage="chat"; the
    # terminal done message is identified by its content marker + rendered tables.
    assert "计划已全部完成" in last["content"]
    assert "即席问数结果" in last["content"]
    tables = last["metadata"]["tables"]
    agg = next(t for t in tables if t["title"] == "聚合结果")
    by_channel = {row[0]: row for row in agg["rows"]}
    cols = agg["columns"]
    br = cols.index("bad_rate_bad")
    cnt = cols.index("count")
    # Hand-calculated on the 4-row dataset (all May): A bad_rate 0.5 / count 2,
    # B bad_rate 1.0 / count 2. Numbers come from the tool, not the LLM (INV-1).
    assert by_channel["A"][br] == "0.5000"
    assert by_channel["A"][cnt] == "2"
    assert by_channel["B"][br] == "1.0000"
    assert by_channel["B"][cnt] == "2"


# -- (3b) audit row: the deterministic tool wrote its data.slice_aggregate audit --
def test_confirm_writes_slice_aggregate_audit_row(client, tmp_path, monkeypatch):
    llm = _FakeLLM({"group_by": ["channel"], "metrics": [{"op": "count"}]})
    _install_llm(monkeypatch, llm)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    _post(client, task_id, "统计各渠道数量")
    _post(client, task_id, "确认")

    with connect(client.app.state.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT detail_json FROM audit WHERE kind = 'data.slice_aggregate'"
        ).fetchall()
    assert len(rows) == 1
    assert f'"task_id":"{task_id}"' in rows[0]["detail_json"]


# -- (4) clarification path: a hallucinated column -> Chinese clarify, no state --
def test_hallucinated_column_clarifies_without_storing_state(client, tmp_path, monkeypatch):
    llm = _FakeLLM({"group_by": ["region"], "metrics": [{"op": "count"}]})
    _install_llm(monkeypatch, llm)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    resp = _post(client, task_id, "按地区看数量")
    assert resp.status_code == 202, resp.text
    last = _last_assistant(resp.json()["messages"])
    assert "region" in last["content"]
    # A guess is never stored: no pending口径, so the next turn re-parses.
    assert "adhoc_spec" not in (last.get("metadata") or {})


# -- (5) deny/rephrase drops the pending spec and returns to the normal flow -----
def test_non_confirm_reply_discards_pending_spec(client, tmp_path, monkeypatch):
    llm = _FakeLLM({
        "group_by": ["channel"],
        "metrics": [{"op": "bad_rate", "col": "bad"}],
    })
    _install_llm(monkeypatch, llm)
    task_id = _make_task_with_ready_dataset(client, tmp_path)

    opened = _post(client, task_id, "按渠道看坏率")
    assert "adhoc_spec" in _last_assistant(opened.json()["messages"])["metadata"]

    # "先别执行" is a negated-confirm -> the adhoc branch returns None and the turn
    # falls through to the normal feature handler; no slice_aggregate plan runs.
    denied = _post(client, task_id, "先别执行")
    assert denied.status_code in (202, 409), denied.text
    with connect(client.app.state.settings.db_path) as conn:
        audit = conn.execute(
            "SELECT count(*) AS n FROM audit WHERE kind = 'data.slice_aggregate'"
        ).fetchone()
    assert audit["n"] == 0
