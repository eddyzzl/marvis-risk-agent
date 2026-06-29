"""Agent-mode operations layer: the LLM operates the gates the user clicks in manual
mode (decision #: "agent 模式就是手动模式里需要操作的部分给 llm 操作和判断").

Two invariants are covered:
  1. Agent mode REQUIRES a configured LLM — with none, /agent/start must error
     (HTTP 409) rather than silently running the manual flow.
  2. With an LLM (here a FakeLLM, since the platform may have none configured yet),
     the LLM auto-confirms each gate and the whole deterministic flow runs end-to-end
     in a single request — C1 file-role gate -> C2 join gate -> executed join.

The LLM is injected at ``marvis.api._resolve_driver_agent_client``, so a FakeLLM
drives the real FastAPI endpoints with no network and no LLM configuration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.auto_drive import decide_gate, parse_decision
from marvis.app import create_app


def _join_dir(root: Path, n: int = 50) -> Path:
    src = root / "join_material"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    pd.DataFrame({
        "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
        "balance": list(range(n)),
    }).to_parquet(src / "features.parquet")
    return src


def _strategy_dir(root: Path) -> Path:
    src = root / "strategy_material"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "bad": [1, 0, 0, 0, 1, 0],
        "score": [580, 620, 730, 760, 590, 800],
    }).to_csv(src / "strategy.csv", index=False)
    return src


def _vintage_dir(root: Path) -> Path:
    src = root / "vintage_material"
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "cohort": ["202601", "202601", "202602", "202602"],
        "mob": [0, 1, 0, 1],
        "bad": [0, 1, 0, 0],
    }).to_csv(src / "vintage.csv", index=False)
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


class _FakeLLM:
    """Records prompts; returns a fixed JSON gate decision."""

    def __init__(self, action: str = "confirm", reason: str = "结果正常,继续"):
        self._payload = json.dumps({"action": action, "reason": reason})
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self._payload


class _SequencedLLM:
    """Returns one payload per call, then repeats the last payload."""

    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if len(self.calls) <= len(self.payloads):
            return self.payloads[len(self.calls) - 1]
        return self.payloads[-1]


# -- invariant 1: agent mode requires an LLM ---------------------------------
def test_agent_mode_data_join_without_llm_errors(client: TestClient, tmp_path: Path):
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接agent", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "agent",
    }).json()["id"]
    # No LLM configured in the test workspace → starting an *agent* task must 409,
    # not fall through to the manual flow.
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 409, resp.text


def test_manual_mode_data_join_without_llm_runs(client: TestClient, tmp_path: Path):
    """The same task in MANUAL mode runs with no LLM — manual mode needs no LLM."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接手动", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text


@pytest.mark.parametrize(
    ("task_type", "source_factory", "extra_payload"),
    [
        ("strategy", _strategy_dir, {"target_col": "bad", "score_col": "score"}),
        ("vintage", _vintage_dir, {"target_col": "bad", "time_col": "cohort"}),
    ],
)
def test_agent_mode_strategy_and_vintage_without_llm_error(
    client: TestClient,
    tmp_path: Path,
    task_type: str,
    source_factory,
    extra_payload: dict,
):
    src = source_factory(tmp_path)
    task_payload = {
        "model_name": f"{task_type} agent",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": task_type,
        "run_mode": "agent",
        **extra_payload,
    }
    task_id = client.post("/api/tasks", json=task_payload).json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})

    assert resp.status_code == 409, resp.text


# -- invariant 2: with an LLM, agent mode auto-drives the gates ---------------
def test_agent_mode_autodrives_join_to_completion(client: TestClient, tmp_path: Path, monkeypatch):
    fake = _FakeLLM(action="confirm", reason="命中率正常,继续")
    monkeypatch.setattr("marvis.api._resolve_driver_agent_client", lambda request, task, payload: fake)
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接自动", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "agent",
    }).json()["id"]

    # A single start request in AUTO(自动审查) mode: the LLM confirms the C1 file-role
    # gate AND the C2 join gate, so the executed-join result is the final message.
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={"acceptance_mode": "auto_accept"})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    done = _last_assistant(msgs)
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]
    # The LLM was consulted at each gate, and its rationale is visible in the transcript.
    assert len(fake.calls) >= 2
    assert any(m["metadata"].get("intent") == "agent_decision" for m in msgs if m["role"] == "assistant")


def test_agent_mode_halt_decision_stops_at_gate(client: TestClient, tmp_path: Path, monkeypatch):
    fake = _FakeLLM(action="halt", reason="命中率过低,请人工核对")
    monkeypatch.setattr("marvis.api._resolve_driver_agent_client", lambda request, task, payload: fake)
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接暂停", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "agent",
    }).json()["id"]
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={"acceptance_mode": "auto_accept"})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    # Halt at the FIRST (C1) gate: no join is executed, the LLM's halt reason shows.
    assert not any("拼接执行完成" in m["content"] for m in msgs)
    assert any("请人工核对" in m["content"] for m in msgs if m["role"] == "assistant")


def test_agent_normal_mode_stops_at_first_gate(client: TestClient, tmp_path: Path, monkeypatch):
    """NORMAL(默认权限) honors the human-in-the-loop: even with an LLM that would
    confirm, the agent runs to the FIRST gate and stops — it does NOT auto-drive the
    whole plan. (spec §6: 默认权限每个大步后停, 自动审查全自动.)"""
    fake = _FakeLLM(action="confirm", reason="结果正常,继续")
    monkeypatch.setattr("marvis.api._resolve_driver_agent_client", lambda request, task, payload: fake)
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接默认权限", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "agent",
    }).json()["id"]
    # No acceptance_mode → defaults to NORMAL.
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    # Stops at the C1 file-role gate; the join is NOT executed and the LLM was not
    # consulted to auto-confirm.
    assert any("样本主表" in m["content"] for m in msgs if m["role"] == "assistant")
    assert not any("拼接执行完成" in m["content"] for m in msgs)
    assert fake.calls == []


def test_agent_mode_autodrives_strategy_to_completion(client: TestClient, tmp_path: Path, monkeypatch):
    fake = _FakeLLM(action="confirm", reason="策略规则可回测,继续")
    monkeypatch.setattr("marvis.api._resolve_driver_agent_client", lambda request, task, payload: fake)
    src = _strategy_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "策略自动",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "strategy",
        "run_mode": "agent",
        "target_col": "bad",
        "score_col": "score",
    }).json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={"acceptance_mode": "auto_accept"})

    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    done = _last_assistant(msgs)
    assert "策略权衡视图完成" in done["content"]
    assert len(fake.calls) >= 2
    assert any(m["metadata"].get("intent") == "agent_decision" for m in msgs if m["role"] == "assistant")


def test_agent_mode_autodrives_vintage_to_completion(client: TestClient, tmp_path: Path, monkeypatch):
    fake = _FakeLLM(action="confirm", reason="字段已识别,继续")
    monkeypatch.setattr("marvis.api._resolve_driver_agent_client", lambda request, task, payload: fake)
    src = _vintage_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "Vintage 自动",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "vintage",
        "run_mode": "agent",
        "target_col": "bad",
        "time_col": "cohort",
    }).json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={"acceptance_mode": "auto_accept"})

    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    done = _last_assistant(msgs)
    assert "Vintage 曲线完成" in done["content"]
    assert len(fake.calls) >= 1


# -- unit: gate-decision parsing ---------------------------------------------
def test_parse_decision_confirm():
    assert parse_decision('{"action":"confirm","reason":"ok"}') == {"action": "confirm", "reason": "ok"}


def test_parse_decision_extracts_json_from_markdown():
    out = parse_decision('```json\n{"action":"confirm","reason":"指标稳定"}\n```')
    assert out == {"action": "confirm", "reason": "指标稳定"}


def test_parse_decision_defaults_to_halt_on_junk():
    out = parse_decision("not json at all")
    assert out["action"] == "halt"


def test_parse_decision_unknown_action_is_halt():
    out = parse_decision('{"action":"maybe","reason":"x"}')
    assert out["action"] == "halt"


def test_parse_decision_adjust_when_gate_allows_it():
    out = parse_decision(
        '{"action":"adjust","reason":"降低泄漏阈值重算","params":{"leakage_ks":0.35},'
        '"selection":["x1","x2"],"confidence":0.82}',
        allowed_actions=("confirm", "adjust", "halt"),
    )

    assert out == {
        "action": "adjust",
        "reason": "降低泄漏阈值重算",
        "params": {"leakage_ks": 0.35},
        "selection": ["x1", "x2"],
        "confidence": 0.82,
    }


def test_parse_decision_disallowed_adjust_becomes_safe_halt():
    out = parse_decision(
        '{"action":"adjust","reason":"想调参","params":{"n_trials":10}}',
        allowed_actions=("confirm", "halt"),
    )

    assert out["action"] == "halt"
    assert "不允许" in out["reason"]


def test_decide_gate_uses_gate_envelope_allowed_actions():
    fake = _FakeLLM(action="adjust", reason="泄漏阈值偏保守,重算")
    fake._payload = json.dumps({
        "action": "adjust",
        "reason": "泄漏阈值偏保守,重算",
        "params": {"leakage_ks": 0.35},
    })
    gate = {
        "content": "特征筛选完成",
        "metadata": {
            "gate_envelope": {
                "kind": "screen",
                "target_step_id": "gate-screen",
                "allowed_actions": ["confirm", "adjust", "halt"],
            }
        },
    }

    decision = decide_gate(fake, gate=gate)

    assert decision["action"] == "adjust"
    assert decision["params"] == {"leakage_ks": 0.35}
    assert "confirm, adjust, halt" in fake.calls[0]["system_prompt"]


def test_decide_gate_passes_table_context_to_llm():
    fake = _FakeLLM()
    gate = {"content": "拼接诊断完成", "metadata": {"tables": [
        {"title": "拼接诊断", "columns": ["特征表", "命中率"], "rows": [["features", "1.0000"]]},
    ]}}
    decision = decide_gate(fake, gate=gate)
    assert decision["action"] == "confirm"
    assert "拼接诊断" in fake.calls[0]["user_prompt"]
    assert "命中率" in fake.calls[0]["user_prompt"]


def test_decide_gate_injects_screen_red_flags_into_prompt():
    fake = _FakeLLM()
    gate = {
        "content": "特征筛选完成",
        "metadata": {
            "screen": {
                "leakage": [["leak_col", 0.55, "suspected target leakage"]],
                "suspected": [["score_x", 0.31, "model output"]],
                "unusable": [["all_null", "high_missing"]],
            }
        },
    }

    decide_gate(fake, gate=gate)

    prompt = fake.calls[0]["user_prompt"]
    assert "平台红旗 checklist" in prompt
    assert "疑似硬泄漏特征" in prompt
    assert "可疑模型输出/泄漏特征" in prompt
    assert "不可用特征" in prompt


def test_decide_gate_injects_join_red_flags_from_tables_into_prompt():
    fake = _FakeLLM()
    gate = {
        "content": "拼接诊断完成",
        "metadata": {
            "tables": [
                {
                    "title": "拼接诊断(逐特征表)",
                    "columns": ["特征表", "命中率", "膨胀", "去重(安全/冲突键)"],
                    "rows": [["features.parquet", "0.1000", "⚠️是", "安全0/⚠️冲突3"]],
                }
            ]
        },
    }

    decide_gate(fake, gate=gate)

    prompt = fake.calls[0]["user_prompt"]
    assert "命中率偏低(10.00%)" in prompt
    assert "存在拼接膨胀风险" in prompt
    assert "存在同键冲突去重风险" in prompt


def test_decide_gate_omits_red_flag_section_for_clean_gate():
    fake = _FakeLLM()
    gate = {"content": "拼接诊断完成", "metadata": {"tables": [
        {
            "title": "拼接诊断(逐特征表)",
            "columns": ["特征表", "命中率", "膨胀"],
            "rows": [["features.parquet", "1.0000", "否"]],
        }
    ]}}

    decide_gate(fake, gate=gate)

    assert "平台红旗 checklist" not in fake.calls[0]["user_prompt"]


def test_decide_gate_retries_once_after_unparseable_reply():
    fake = _SequencedLLM(["not json", '{"action":"confirm","reason":"重试后可解析"}'])
    gate = {"content": "调参结果完成", "metadata": {}}

    decision = decide_gate(fake, gate=gate)

    assert decision == {"action": "confirm", "reason": "重试后可解析"}
    assert len(fake.calls) == 2
    assert "上一次返回无法解析" in fake.calls[1]["user_prompt"]
