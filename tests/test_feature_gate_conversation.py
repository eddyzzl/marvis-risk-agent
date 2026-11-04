"""Feature-analysis gate conversation (eval S3).

* "为什么没算稳定性？" got the router's rationale back as the reply.
* "把按 apply_month 的 PSI 加进特征分析，重新跑" failed as a replan with a
  "budget exhausted" message pointing at a non-existent 「调参重算」 entry.
* "对 sig1 和 score 做分箱，然后生成报告" shrank the metrics feature set to
  two features and echoed a Python dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.gate_execution_adapter import (
    FEATURE_STABILITY_COMMAND,
    GateExecutionAdapter,
    describe_adjust_params,
)
from marvis.agent.quick_replies import QUICK_REPLIES_KEY
from marvis.app import create_app

_ROUTER_RATIONALE = "用户是在追问为何未计算稳定性，属于疑问和新增分析诉求，未明确授权继续当前节点。"


class _GateLLM:
    def __init__(self, route: dict, *, authorize: bool = True) -> None:
        self.route = route
        self.authorize = authorize
        self.callers: list[str] = []

    def complete(self, **kwargs) -> str:
        caller = str(kwargs.get("caller") or "")
        self.callers.append(caller)
        if caller == "router":
            return json.dumps(
                {
                    "params": {},
                    "constraint": "",
                    "reason": _ROUTER_RATIONALE,
                    "confidence": "high",
                    "explicit_authorization": self.route.get("action") == "confirm",
                    **self.route,
                },
                ensure_ascii=False,
            )
        if caller == "semantic_authorization_reviewer":
            request = json.loads(kwargs["user_prompt"])
            instruction = str(request.get("instruction") or "")
            return json.dumps(
                {
                    "verdict": "authorize" if self.authorize else "ambiguous",
                    "evidence_quote": instruction if self.authorize else "",
                    "reason": "",
                    "confidence": "high" if self.authorize else "low",
                    "is_question": False,
                    "is_conditional": False,
                    "requests_change": False,
                    "withholds_authorization": False,
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"unexpected LLM caller {caller}")


def _feature_task(client: TestClient, tmp_path: Path) -> str:
    source = tmp_path / "feature_material"
    source.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    n = 400
    s1, s2, s3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.8 * s1 + 0.6 * s2 - 0.5 * s3 - 1.2)))
    pd.DataFrame(
        {
            "sig1": s1,
            "sig2": s2,
            "sig3": s3,
            "apply_month": np.repeat(["2025-01", "2025-02", "2025-03", "2025-04"], n // 4),
            "bad_flag": (rng.uniform(size=n) < p).astype(float),
        }
    ).to_parquet(source / "sample.parquet")
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "特征哪些能用",
            "validator": "qa",
            "source_dir": str(source),
            "task_type": "feature_analysis",
            "run_mode": "manual",
            "target_col": "bad_flag",
            "metrics": ["iv", "ks", "auc", "coverage"],
        },
    ).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    return task_id


def _install(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )


def _messages(client: TestClient, task_id: str) -> list[dict]:
    return client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]


def _steps(client: TestClient, task_id: str) -> dict[str, dict]:
    plan = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    return {step["title"]: step for step in plan["steps"]}


def _send(client: TestClient, task_id: str, content: str) -> list[dict]:
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": content}
    )
    assert response.status_code == 202, response.text
    return _messages(client, task_id)


def _first_reply(messages: list[dict]) -> dict:
    """The Agent's direct reply to the latest user turn (the gate card may follow)."""

    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    return next(m for m in messages[last_user + 1 :] if m["role"] == "assistant")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_binning_request_selects_features_without_shrinking_metrics(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    task_id = _feature_task(client, tmp_path)
    metrics_features = _steps(client, task_id)["特征指标"]["inputs"]["features"]
    assert len(metrics_features) == 3
    _install(
        monkeypatch,
        _GateLLM({"action": "adjust", "params": {"features": ["sig1", "sig3"]}}),
    )

    messages = _send(client, task_id, "那先对 sig1 和 sig3 做分箱，然后生成报告")

    steps = _steps(client, task_id)
    assert steps["特征指标"]["inputs"]["features"] == metrics_features
    assert steps["可选分箱分析"]["inputs"]["features"] == ["sig1", "sig3"]
    assert steps["生成特征分析报告"]["status"] == "done"
    done = [m for m in messages if m["role"] == "assistant"][-1]
    binning = next(t for t in done["metadata"]["tables"] if t["title"] == "分箱分析")
    assert {row[0] for row in binning["rows"]} == {"sig1", "sig3"}
    assert all("{'features'" not in m["content"] for m in messages)


def test_unauthorized_binning_request_changes_nothing_and_points_to_card(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    task_id = _feature_task(client, tmp_path)
    before = _steps(client, task_id)
    _install(
        monkeypatch,
        _GateLLM(
            {"action": "adjust", "params": {"features": ["sig1"]}}, authorize=False
        ),
    )

    reply = _first_reply(_send(client, task_id, "sig1 分箱看看"))

    after = _steps(client, task_id)
    assert after["特征指标"]["inputs"] == before["特征指标"]["inputs"]
    assert after["可选分箱分析"]["status"] == "awaiting_confirm"
    assert "分析所选特征并生成报告" in reply["content"]
    assert "没有执行" in reply["content"]


def test_stability_question_is_explained_with_a_working_quick_reply(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    task_id = _feature_task(client, tmp_path)
    llm = _GateLLM({"action": "clarify"})
    _install(monkeypatch, llm)

    reply = _first_reply(
        _send(client, task_id, "为什么没算稳定性？我要看每个特征按 apply_month 的 PSI")
    )

    assert _ROUTER_RATIONALE not in reply["content"]
    assert "未明确授权" not in reply["content"]
    assert "IV、KS、AUC、覆盖率" in reply["content"]
    assert "没有选择稳定性" in reply["content"]
    assert reply["metadata"][QUICK_REPLIES_KEY] == [
        {"label": "补算按月 PSI", "command": FEATURE_STABILITY_COMMAND}
    ]
    calls_before = len(llm.callers)

    _send(client, task_id, FEATURE_STABILITY_COMMAND)

    steps = _steps(client, task_id)
    assert {"psi_month_first", "psi_month_previous"} <= set(
        steps["特征指标"]["inputs"]["metrics"]
    )
    assert len(steps["特征指标"]["inputs"]["features"]) == 3
    assert steps["可选分箱分析"]["status"] == "awaiting_confirm"
    # The chip is a literal command: no router call was needed.
    assert len(llm.callers) == calls_before


def test_add_psi_instruction_recomputes_metrics_instead_of_failing_replan(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    task_id = _feature_task(client, tmp_path)
    _install(monkeypatch, _GateLLM({"action": "replan"}))

    messages = _send(
        client, task_id, "把按 apply_month 的 PSI 加进特征分析，重新跑一下特征指标"
    )

    contents = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    assert "重规划未成功" not in contents
    assert "调参重算" not in contents
    assert "按月 PSI（时间列 apply_month）" in contents
    inputs = _steps(client, task_id)["特征指标"]["inputs"]
    assert inputs["time_col"] == "apply_month"
    assert {"psi_month_first", "psi_month_previous"} <= set(inputs["metrics"])


@pytest.mark.parametrize(
    ("tier", "replan_count", "expected"),
    [
        ("balanced", 0, "没能把这条要求改写成"),
        ("balanced", 4, "重规划次数（4 次）已经用完"),
    ],
)
def test_replan_failure_copy_names_real_cause_and_entry(
    tier: str, replan_count: int, expected: str
) -> None:
    plan = SimpleNamespace(tier=tier, replan_count=replan_count)
    gate = SimpleNamespace(title="可选分箱分析")

    text = GateExecutionAdapter._replan_failure_text(plan, gate)

    assert expected in text
    assert "当前仍停在「可选分箱分析」" in text
    assert "调参重算" not in text
    assert "预算用尽或指令无法执行" not in text


def test_adjust_summary_is_business_chinese_not_a_dict() -> None:
    assert describe_adjust_params({"features": ["sig1", "score"], "bins": 5}) == (
        "特征改为 sig1、score；分箱数改为 5"
    )
    assert "{" not in describe_adjust_params({"key_overrides": {"cust_id": "cid"}})
    # Keyed by dataset id; only the chosen columns reach the sentence (eval S6).
    assert describe_adjust_params({"key_overrides": {"ds_" + "f" * 32: ["mobile"]}}) == "拼接键改为 mobile"
    assert describe_adjust_params({"key_overrides": {"ds_a": ["mobile"], "ds_b": ["cust_id", "apply_date"]}}) == (
        "拼接键改为 第 1 张特征表 mobile、第 2 张特征表 cust_id + apply_date"
    )
