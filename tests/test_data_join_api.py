"""End-to-end HTTP test of the data-join entry through the generic PlanDriver.

Drives the real FastAPI endpoints (/agent/start, /agent/messages) for a
task_type='data_join' task whose materials are two joinable tables (an anchor
sample with a label + a feature table keyed by md5(mobile)). No LLM is
configured — this is exactly the no-LLM preview scenario the manual-first build
targets. Asserts the driver pauses at the forced-confirm gate (showing join
diagnostics) and, on confirmation, executes a 1:1-anchored join.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

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


def _join_dir_with_conflicts(root: Path, n: int = 50) -> Path:
    """Like _join_dir but the feature table repeats the first 5 keys with a DIFFERENT
    value — a same-key conflict that makes the join key non-unique, so confirm_join
    leaves the feature awaiting a dedup strategy (the §4 dedup picker scenario)."""
    src = root / "join_conflict"
    src.mkdir(parents=True, exist_ok=True)
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(src / "sample.parquet")
    md5s = [hashlib.md5(p.encode()).hexdigest() for p in phones]
    pd.DataFrame({
        "phone_md5": md5s + md5s[:5],          # 5 duplicate keys
        "balance": list(range(n)) + [999] * 5,  # ...with a conflicting value
    }).to_parquet(src / "features.parquet")
    return src


def _last_assistant(messages: list[dict]) -> dict:
    return [m for m in messages if m["role"] == "assistant"][-1]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_data_join_conversation_end_to_end(client: TestClient, tmp_path: Path):
    src = _join_dir(tmp_path)
    resp = client.post("/api/tasks", json={
        "model_name": "拼接测试",
        "validator": "qa",
        "source_dir": str(src),
        "task_type": "data_join",
        "run_mode": "manual",
    })
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["id"]

    # turn 0 — start: C1 file-role assignment gate (propose anchor/feature + target)
    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert resp.status_code == 202, resp.text
    msgs = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    c1 = _last_assistant(msgs)
    assert "样本主表" in c1["content"]
    assert c1["metadata"]["join_c1"]["anchor_id"]  # proposal carried for the form
    assert any(t["title"].startswith("输入文件") for t in c1["metadata"].get("tables", []))

    # turn 1 — confirm C1 roles: build the join plan, pause at the plan-overview gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]

    # turn 2 — 开始: run the plan, pause at the C2 diagnostics gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert resp.status_code == 202, resp.text
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]
    assert any(t["title"].startswith("拼接诊断") for t in gate["metadata"].get("tables", []))

    # turn 3 — confirm C2: confirm_join + execute_join run, anchor preserved 1:1
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


def test_data_join_dedup_picker_resolves_conflicts(client: TestClient, tmp_path: Path):
    """§4 join dedup picker: a non-unique feature key leaves confirm_join awaiting a
    strategy; the C2 gate surfaces it via metadata.dedup; posting a per-feature
    strategy re-confirms (resolving the conflict) and the join then completes 1:1."""
    src = _join_dir_with_conflicts(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接去重", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    dedup = gate["metadata"].get("dedup")
    assert dedup is not None, gate["metadata"]
    assert dedup["needs_dedup"], dedup
    assert dedup["strategies"] == ["first", "last"]
    feature_id = dedup["needs_dedup"][0]
    # the picker shows the conflict count from the propose-step diagnostics
    assert dedup["features"][0]["conflict_keys"] >= 1

    # pick a strategy -> re-confirm; conflict resolved, re-pause at the now-clear gate
    resp = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": "确认",
            "dedup_strategies": {feature_id: "first"},
            "expected_step_id": gate["metadata"]["step_id"],
        },
    )
    assert resp.status_code == 202, resp.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert gate2["metadata"].get("kind") == "gate"
    assert not gate2["metadata"].get("dedup")  # no strategy still needed

    # confirm execute -> 1:1 anchored join completes
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


def test_data_join_dedup_text_instruction_resolves_conflicts(client: TestClient, tmp_path: Path):
    """Manual-mode TEXT resolution of a same-key conflict when no §4 picker is wired: the C2
    gate surfaces the conflict + the 「去重 first/last」 hint, and replying with that text
    applies the strategy to every needs_dedup feature so the join completes 1:1. Without this
    a pure-text manual user dead-ends at 'all joins must be confirmed before execute'."""
    src = _join_dir_with_conflicts(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接去重文字", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    # the gate surfaces the conflict + the text-resolution hint, naming the feature by its
    # friendly file name (.parquet), not a raw ds_<hash> id
    assert "同键冲突" in gate["content"] and "去重 first" in gate["content"]
    assert ".parquet" in gate["content"] and "`ds_" not in gate["content"]

    # plain-text dedup choice resolves every conflicting feature, re-pause at the clear gate
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "去重 first"})
    assert resp.status_code == 202, resp.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert not gate2["metadata"].get("dedup")  # no strategy still needed

    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"] and "1:1 保持" in done["content"]


def test_data_join_c1_form_assignment_drives_the_join(client: TestClient, tmp_path: Path):
    """The C1 control form posts a structured [C1] assignment; the driver honors it."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接表单", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    state = c1["metadata"]["join_c1"]
    anchor = state["anchor_id"]
    features = state["feature_ids"]
    payload = json.dumps({"anchor_id": anchor, "feature_ids": features, "target_col": state["target_col"]})
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": f"[C1]{payload}"})
    assert resp.status_code == 202, resp.text
    overview = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "确认「开始」后按计划执行" in overview["content"]
    # 开始 → run to the C2 diagnostics gate
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接诊断完成" in gate["content"]


def test_data_join_c1_form_rejects_duplicate_sample_primary_role(client: TestClient, tmp_path: Path):
    """UX-7: the C1 form must reject a payload that marks two datasets as the
    sample anchor table with a typed, explicit error rather than silently
    dropping the second dataset from the join (join_setup.JoinSetupError)."""
    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "拼接双主表", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    state = c1["metadata"]["join_c1"]
    anchor = state["anchor_id"]
    other = state["feature_ids"][0]
    payload = json.dumps({
        "anchor_id": anchor,
        "anchor_ids": [anchor, other],
        "feature_ids": [],
        "target_col": state["target_col"],
    })
    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": f"[C1]{payload}"})
    assert resp.status_code == 202, resp.text
    error = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert error["metadata"].get("error") is True
    assert "样本主表只能有一个" in error["content"]
    assert "特征表" in error["content"] or "忽略" in error["content"]

    # the gate must still be the same C1 form (not silently advanced) so the
    # user can correct the roles and resubmit.
    still_open = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert still_open["metadata"].get("error") is True


def test_data_join_single_file_confirms_then_skips(client: TestClient, tmp_path: Path):
    src = tmp_path / "single_material"
    src.mkdir()
    pd.DataFrame({"mobile": ["a", "b"], "bad_flag": [0, 1]}).to_parquet(src / "only.parquet")
    task_id = client.post("/api/tasks", json={
        "model_name": "单文件", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    # C1 still confirms the sample + target even with one file (spec §1 / §3 single-file degenerate)
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    c1 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert c1["metadata"]["join_c1"]["skip"] is True
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    skip = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "无需拼接" in skip["content"]


def test_data_join_double_confirm_second_request_gets_409_without_second_turn(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-1: a double-sent confirm (dual tab / retried request) while the first
    driver turn is still executing must be rejected with 409 by the
    idx_jobs_active_task guard *before* it reaches dispatch_plan_driver_turn a
    second time — not raced into a second PlanExecutor.run that could mis-flag
    the in-flight step as a restart orphan (REL-1) and clobber it with a 500."""
    from marvis import api as marvis_api

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "并发确认", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    turn_calls: list[str] = []
    real_dispatch = marvis_api.dispatch_plan_driver_turn

    def racing_dispatch(runtime, repo, task, **kwargs):
        turn_calls.append(task.id)
        # Simulate Tab B re-sending the confirmation while Tab A's turn (this
        # call) is still executing: it must be rejected before ever reaching
        # dispatch_plan_driver_turn again.
        second = client.post(
            f"/api/tasks/{task_id}/agent/messages",
            json={"content": "确认"},
        )
        assert second.status_code == 409, second.text
        assert second.json()["detail"] == "该任务正在执行上一步，请等待完成"
        return real_dispatch(runtime, repo, task, **kwargs)

    monkeypatch.setattr(marvis_api, "dispatch_plan_driver_turn", racing_dispatch)

    first = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})

    assert first.status_code == 202, first.text
    # dispatch_plan_driver_turn (the real turn / PlanExecutor.run path) ran
    # exactly once for this confirm — the racing second request never got in.
    assert turn_calls == [task_id]
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]


def test_data_join_active_driver_job_reports_kind_driver_on_task_payload(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-6: GET /api/tasks (and the single-task endpoint) must surface
    active_job_kind == "driver" while a driver turn is executing, so the
    frontend's busy state / 1s polling can pick it up after a refresh or from
    any other entry point (UX-1)."""
    from marvis import api as marvis_api

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "任务忙碌态", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    observed: dict[str, str | None] = {}
    real_dispatch = marvis_api.dispatch_plan_driver_turn

    def observing_dispatch(runtime, repo, task, **kwargs):
        mid_turn = client.get(f"/api/tasks/{task_id}")
        assert mid_turn.status_code == 200, mid_turn.text
        observed["single"] = mid_turn.json()["active_job_kind"]
        listed = client.get("/api/tasks")
        listed_task = next(t for t in listed.json() if t["id"] == task_id)
        observed["listed"] = listed_task["active_job_kind"]
        return real_dispatch(runtime, repo, task, **kwargs)

    monkeypatch.setattr(marvis_api, "dispatch_plan_driver_turn", observing_dispatch)

    resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 202, resp.text

    assert observed["single"] == "driver"
    assert observed["listed"] == "driver"
    # The job is finished (synchronous turn) once the HTTP response returns.
    after = client.get(f"/api/tasks/{task_id}")
    assert after.json()["active_job_kind"] is None


def test_data_join_driver_turn_job_finishes_on_exception(
    client: TestClient, tmp_path: Path, monkeypatch,
):
    """REL-1: the job must be released (finish_job in the finally-equivalent
    except branch) even when the turn function raises an unexpected exception,
    so a single failed turn doesn't permanently lock the task behind the
    idx_jobs_active_task unique index."""
    from marvis import api as marvis_api
    from marvis.app import create_app
    from marvis.db import TaskRepository

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "异常清理", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    def boom(runtime, repo, task, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(marvis_api, "dispatch_plan_driver_turn", boom)

    # A fresh client sharing the same tmp_path-backed app/db, configured not to
    # re-raise the server exception, so the 500 can be asserted on the response
    # instead of propagating through TestClient.
    error_client = TestClient(create_app(tmp_path), raise_server_exceptions=False)
    resp = error_client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resp.status_code == 500

    repo = TaskRepository(tmp_path / "marvis.sqlite")
    assert repo.get_active_job_kind(task_id) is None

    # The task is not stuck: a fresh confirm (now unpatched) can still claim
    # the job and complete the turn.
    monkeypatch.undo()
    resumed = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert resumed.status_code == 202, resumed.text


def test_data_join_plan_payload_carries_started_at_for_running_step(
    client: TestClient, tmp_path: Path,
):
    """UX-1/REL-6: a step's plan payload gets a started_at field once it is
    RUNNING (sourced from plan_step_runs, already recorded per attempt), so the
    plan rail can show elapsed time via the same formatStepElapsed() range the
    validation stepper already uses, instead of a plain spinner."""
    from marvis.db import PlanRepository
    from marvis.orchestrator.contracts import StepStatus

    src = _join_dir(tmp_path)
    task_id = client.post("/api/tasks", json={
        "model_name": "耗时展示", "validator": "qa", "source_dir": str(src),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})  # C1 roles
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})  # run to C2 gate

    plans = client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
    plan = plans[-1]
    execute_step = next(step for step in plan["steps"] if step["tool_ref"]["tool"] == "execute_join")
    assert execute_step.get("started_at") in (None, "")  # not running yet

    repo = PlanRepository(tmp_path / "marvis.sqlite")
    loaded = repo.load_plan(plan["id"])
    running_step = next(step for step in loaded.steps if step.id == execute_step["id"])
    running_step.status = StepStatus.RUNNING
    repo.update_step(running_step)
    repo.start_step_run(
        plan_id=plan["id"],
        step_id=execute_step["id"],
        tool_ref="data_ops.execute_join",
        inputs={},
    )

    refreshed = client.get(f"/api/tasks/{task_id}/plans").json()["plans"][-1]
    refreshed_step = next(step for step in refreshed["steps"] if step["id"] == execute_step["id"])
    assert refreshed_step["status"] == "running"
    assert refreshed_step["started_at"]
