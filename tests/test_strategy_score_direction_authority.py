"""D-3: who decides a strategy score's direction.

Order: the user's own words in the current request -> the direction frozen in
the sample contract -> the workspace-confirmed field direction -> stop and ask.
Sample data only checks the chosen direction; a disagreement waits at the
plan's start gate showing both, and nothing ever defaults to higher_is_better.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.strategy_request_compiler.development import (
    resolve_development_score_direction,
    score_direction_conflict_message,
    stated_development_score_direction,
    workspace_score_direction,
)
from marvis.app import create_app
from marvis.data.field_directions import confirm_field_directions
from marvis.data.workspace import data_semantic_mapping_hash
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.data_workspace import DataWorkspaceRepository
from tests.strategy_sample_design_support import (
    materialize_mature_strategy_sample_design,
)


# ---------------------------------------------------------------- resolver


def test_request_words_are_the_first_source() -> None:
    assert stated_development_score_direction("score 越高风险越高", "score") == (
        "higher_is_riskier"
    )
    assert stated_development_score_direction("score 越低风险越高", "score") == (
        "higher_is_better"
    )
    assert stated_development_score_direction("开始策略开发", "score") is None
    decision = resolve_development_score_direction(
        score_col="score",
        request_direction="higher_is_riskier",
        sample_contract_direction=None,
        workspace_direction="higher_is_better",
    )
    assert decision == {
        "status": "resolved",
        "direction": "higher_is_riskier",
        "source": "request",
    }


def test_sample_contract_then_workspace_are_next() -> None:
    assert resolve_development_score_direction(
        score_col="score",
        request_direction=None,
        sample_contract_direction="higher_is_better",
        workspace_direction="higher_is_better",
    )["source"] == "sample_contract"
    assert resolve_development_score_direction(
        score_col="score",
        request_direction=None,
        sample_contract_direction="higher_is_better",
        workspace_direction=None,
    )["source"] == "sample_contract"
    workspace =resolve_development_score_direction(
        score_col="score",
        request_direction=None,
        sample_contract_direction=None,
        workspace_direction=workspace_score_direction("lower_is_riskier"),
    )
    assert workspace == {
        "status": "resolved",
        "direction": "higher_is_better",
        "source": "workspace",
    }


def test_contract_and_workspace_disagreeing_stops_instead_of_contract_winning() -> None:
    decision = resolve_development_score_direction(
        score_col="score",
        request_direction=None,
        sample_contract_direction="higher_is_better",
        workspace_direction="higher_is_riskier",
    )

    assert decision["status"] == "contradiction"
    message = str(decision["message"])
    assert "score 越低风险越高" in message and "score 越高风险越高" in message
    assert "higher_is" not in message
    # The user's own words in the request still decide.
    assert resolve_development_score_direction(
        score_col="score",
        request_direction="higher_is_better",
        sample_contract_direction="higher_is_better",
        workspace_direction="higher_is_riskier",
    )["source"] == "request"


@pytest.mark.parametrize("field_direction", [None, "unknown", "non_monotonic"])
def test_no_source_stops_and_asks(field_direction) -> None:
    decision = resolve_development_score_direction(
        score_col="score",
        request_direction=None,
        sample_contract_direction=None,
        workspace_direction=workspace_score_direction(field_direction),
        workspace_non_monotonic=field_direction == "non_monotonic",
    )

    assert decision["status"] == "missing"
    message = str(decision["message"])
    assert "score 越高风险越高" in message and "score 越低风险越高" in message
    assert "higher_is" not in message
    if field_direction == "non_monotonic":
        assert "非单调" in message


def test_request_opposite_to_the_frozen_sample_contract_stops() -> None:
    decision = resolve_development_score_direction(
        score_col="score",
        request_direction="higher_is_riskier",
        sample_contract_direction="higher_is_better",
        workspace_direction=None,
    )

    assert decision["status"] == "contradiction"
    assert "重新固化样本设计" in str(decision["message"])


def test_conflict_message_shows_both_directions_in_business_words() -> None:
    message = score_direction_conflict_message(
        "score", "higher_is_better", "higher_is_riskier", corr=0.62, labeled=40
    )

    assert "score 越低风险越高" in message
    assert "score 越高风险越高" in message
    assert "40 笔" in message and "0.620" in message
    assert "higher_is" not in message


def test_tool_never_defaults_a_missing_direction() -> None:
    binding = SimpleNamespace(score_direction_for=lambda column: None)

    with pytest.raises(StrategyError, match="分数方向还没有确认"):
        strategy_tools._sample_score_direction({}, binding, "score")
    assert strategy_tools._sample_score_direction(
        {"score_direction": "higher_is_riskier"}, binding, "score"
    ) == "higher_is_riskier"
    frozen = SimpleNamespace(score_direction_for=lambda column: "higher_is_better")
    assert strategy_tools._sample_score_direction({}, frozen, "score") == (
        "higher_is_better"
    )


# ---------------------------------------------------------------- end to end


def _risky_high_source(tmp_path: Path) -> Path:
    """40 labeled rows where a higher score clearly means more bad loans."""

    src = tmp_path / "strategy"
    src.mkdir(parents=True, exist_ok=True)
    scores = list(range(500, 900, 10))
    pd.DataFrame(
        {"score": scores, "bad": [1 if score >= 760 else 0 for score in scores]}
    ).to_csv(src / "strategy.csv", index=False)
    return src


def _task(client: TestClient, tmp_path: Path) -> str:
    created = client.post(
        "/api/tasks",
        json={
            "model_name": "额度准入策略",
            "validator": "qa",
            "source_dir": str(_risky_high_source(tmp_path)),
            "task_type": "strategy",
            "run_mode": "manual",
            "target_col": "bad",
            "score_col": "score",
            "strategy_input": {"objective": "max_approval", "max_bad_rate": 0.20},
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _development_plan(client: TestClient, task_id: str):
    plans = [
        plan
        for plan in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        if plan["template_id"] == "strategy_development"
    ]
    return plans[-1] if plans else None


def test_data_conflict_waits_at_the_start_gate_showing_both(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "开始策略开发，score 越低风险越高"},
    )

    assert response.status_code == 202, response.text
    plan = _development_plan(client, task_id)
    assert plan is not None
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].inputs["score_direction"] == "higher_is_better"
    # Not pre-confirmed: only a person's start of this gate confirms it.
    assert stored.steps[0].inputs["confirm_direction_conflict"] is False
    assert all(step.status.value == "pending" for step in stored.steps)
    notice = next(
        message["content"]
        for message in response.json()["messages"]
        if (message.get("metadata") or {}).get("intent") == "full_development"
    )
    assert "score 越低风险越高" in notice
    assert "样本数据显示 score 越高风险越高" in notice
    assert "不会自动开始" in notice

    # AUTO: the overview gate is a mandatory human decision, so an LLM
    # "confirm" is turned into a halt, and the driver refuses an AUTO start.
    from marvis.agent.auto_drive import decide_gate
    from marvis.agent.plan_driver import (
        CONFIRMATION_SOURCE_AUTO,
        DriverError,
        PlanDriver,
    )

    overview = next(
        message
        for message in reversed(response.json()["messages"])
        if (message.get("metadata") or {}).get("kind") == "plan_overview"
    )
    assert overview["metadata"]["human_decision_gate"] == "required"

    class _ConfirmingLLM:
        def complete(self, **kwargs) -> str:
            return '{"action": "confirm", "reason": "继续", "confidence": 0.99}'

    assert decide_gate(_ConfirmingLLM(), gate=overview)["action"] == "halt"
    driver = PlanDriver(
        client.app.state.plan_repo,
        client.app.state.plan_executor,
        planner=client.app.state.planner,
        validator=client.app.state.plan_validator,
        governance_service=getattr(client.app.state, "governance_service", None),
    )
    with pytest.raises(DriverError, match="AUTO 不能代替人工确认"):
        driver.resume(
            plan_id=plan["id"],
            user_text="开始",
            confirmation_source=CONFIRMATION_SOURCE_AUTO,
        )
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].inputs["confirm_direction_conflict"] is False
    assert all(step.status.value == "pending" for step in stored.steps)

    # A person's start of that gate is the confirmation.
    driver.resume(plan_id=plan["id"], user_text="开始")
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert all(
        step.inputs["confirm_direction_conflict"] is True
        for step in stored.steps
        if "confirm_direction_conflict" in step.inputs
    )
    assert stored.steps[0].status.value != "failed"


def test_workspace_confirmed_field_direction_is_read_for_the_exact_column(
    tmp_path: Path,
) -> None:
    from marvis.agent.turn_handlers import strategy_turns
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository
    from tests.strategy_sample_design_support import _ensure_active_data_workspace

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    settings = client.app.state.settings
    _ensure_active_data_workspace(client, task_id)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    runtime = SimpleNamespace(settings=settings)
    task = SimpleNamespace(id=task_id)
    assert strategy_turns._workspace_field_direction(runtime, task, registry, "score") == (
        None,
        False,
    )

    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    confirm_field_directions(
        settings,
        task_id=task_id,
        actor_id="maker-1",
        expected_revision=snapshot.revision,
        expected_semantic_mapping_hash=data_semantic_mapping_hash(
            snapshot.semantic_mapping
        ),
        dataset_id=snapshot.active_dataset_id,
        dataset_content_hash=snapshot.active_dataset_content_hash,
        directions={"score": "lower_is_riskier"},
    )

    assert strategy_turns._workspace_field_direction(runtime, task, registry, "score") == (
        "higher_is_better",
        False,
    )
    assert strategy_turns._workspace_field_direction(runtime, task, registry, "bad") == (
        None,
        False,
    )


def _confirm_workspace_direction(client: TestClient, task_id: str, direction: str) -> None:
    from tests.strategy_sample_design_support import _ensure_active_data_workspace

    settings = client.app.state.settings
    _ensure_active_data_workspace(client, task_id)
    snapshot = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    confirm_field_directions(
        settings,
        task_id=task_id,
        actor_id="maker-1",
        expected_revision=snapshot.revision,
        expected_semantic_mapping_hash=data_semantic_mapping_hash(
            snapshot.semantic_mapping
        ),
        dataset_id=snapshot.active_dataset_id,
        dataset_content_hash=snapshot.active_dataset_content_hash,
        directions={"score": direction},
    )


def test_workspace_direction_survives_sample_writes_and_drives_development(
    tmp_path: Path, monkeypatch
) -> None:
    """D-3 step 3 end to end: confirm, fix the sample, develop without stating it."""

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    settings = client.app.state.settings
    _confirm_workspace_direction(client, task_id, "higher_is_riskier")

    # The governed sample-design writes (workspace re-binding + materialize).
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    after = DataWorkspaceRepository(settings.db_path).get_or_default(task_id)
    assert after.semantic_mapping.risk_directions["score"].direction == (
        "higher_is_riskier"
    )

    from marvis.agent.turn_handlers import strategy_turns

    decisions: list[dict] = []
    resolve = strategy_turns._development_score_direction

    def recording_resolve(*args, **kwargs):
        decision = resolve(*args, **kwargs)
        decisions.append(decision)
        return decision

    monkeypatch.setattr(strategy_turns, "_development_score_direction", recording_resolve)
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "开始策略开发"}
    )

    assert response.status_code == 202, response.text
    assert response.json().get("code") != "strategy_score_direction_required"
    assert [(item["direction"], item["source"]) for item in decisions] == [
        ("higher_is_riskier", "workspace")
    ]
    plan = _development_plan(client, task_id)
    assert plan is not None
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].inputs["score_direction"] == "higher_is_riskier"
    # The data agrees (higher score -> more bad), so no conflict gate.
    assert "confirm_direction_conflict" not in stored.steps[0].inputs
    assert any(
        "来源：数据工作区已确认的字段方向" in message["content"]
        for message in response.json()["messages"]
    )


def test_role_confirmation_keeps_confirmed_field_directions(tmp_path: Path) -> None:
    """The sample-design time-role write keeps the user's direction declarations."""

    from marvis.agent.turn_handlers import strategy_sample

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    settings = client.app.state.settings
    _confirm_workspace_direction(client, task_id, "lower_is_riskier")
    draft = SimpleNamespace(workflow_inputs={"field_bindings": {"time_field": "score"}})
    strategy_sample._confirm_manual_sample_design_time_semantics(
        SimpleNamespace(settings=settings),
        SimpleNamespace(id=task_id),
        draft=draft,
        preview=SimpleNamespace(columns=("score", "bad")),
    )

    mapping = DataWorkspaceRepository(settings.db_path).get_or_default(task_id).semantic_mapping
    assert mapping.field_roles["score"] == "date"
    assert mapping.risk_directions["score"].direction == "lower_is_riskier"


# ---------------------------------------------------------------- quick analysis


def _quick_plan(client: TestClient, task_id: str):
    plans = [
        plan
        for plan in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        if plan["template_id"] == "strategy_analysis"
    ]
    return plans[-1] if plans else None


def test_quick_analysis_without_a_direction_stops_and_offers_quick_replies(
    tmp_path: Path, monkeypatch
) -> None:
    """The data-implied direction is never declared on the quick path either."""

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "快速策略分析"}
    )

    assert response.status_code == 202, response.text
    assert response.json().get("code") == "strategy_score_direction_required"
    assert _quick_plan(client, task_id) is None
    replies = [
        reply.get("text") or reply.get("label") or reply.get("command")
        for reply in response.json()["messages"][-1]["metadata"]["quick_replies"]
    ]
    assert replies == [
        "快速策略分析，score 越高风险越高",
        "快速策略分析，score 越低风险越高",
    ]


def test_quick_analysis_uses_the_workspace_direction_and_cuts_the_rule_with_it(
    tmp_path: Path, monkeypatch
) -> None:
    """Workspace says lower score is riskier; the data says the opposite."""

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    _confirm_workspace_direction(client, task_id, "lower_is_riskier")
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": "快速策略分析"}
    )

    assert response.status_code == 202, response.text
    plan = _quick_plan(client, task_id)
    assert plan is not None
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    build, _backtest, tradeoff = stored.steps
    # The candidate rule rejects the low-score tail (declared direction), not
    # the high-score tail the data would imply.
    assert build.inputs["rules"][0]["condition"].startswith("score < ")
    assert tradeoff.inputs["score_direction"] == "higher_is_better"
    # The data disagrees: the plan waits for a person to confirm at its start.
    assert tradeoff.inputs["confirm_direction_conflict"] is False
    assert all(step.status.value == "pending" for step in stored.steps)
    content = "\n".join(message["content"] for message in response.json()["messages"])
    assert "来源：数据工作区已确认的字段方向" in content
    assert "样本数据显示 score 越高风险越高" in content
    overview = next(
        message
        for message in reversed(response.json()["messages"])
        if (message.get("metadata") or {}).get("kind") == "plan_overview"
    )
    assert overview["metadata"]["human_decision_gate"] == "required"


def test_quick_analysis_stated_direction_agreeing_with_data_has_no_gate(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={"content": "快速策略分析，score 越高风险越高"},
    )

    assert response.status_code == 202, response.text
    plan = _quick_plan(client, task_id)
    assert plan is not None
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    assert stored.steps[0].inputs["rules"][0]["condition"].startswith("score >= ")
    assert stored.steps[2].inputs["score_direction"] == "higher_is_riskier"
    assert "confirm_direction_conflict" not in stored.steps[2].inputs
    assert any(
        "来源：你在本次请求里说明" in message["content"]
        for message in response.json()["messages"]
    )
