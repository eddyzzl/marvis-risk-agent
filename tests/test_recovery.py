import sqlite3

from marvis.agent.orchestrator import is_metrics_failure
from marvis.db import PlanRepository, TaskRepository, init_db
from marvis.domain import (
    TASK_STATUS_REASON_SERVER_RESTART,
    TASK_TYPE_DATA_JOIN,
    TaskCreate,
    TaskStatus,
)
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.orchestrator.harness_state import HarnessState
from marvis.orchestrator.reviewer import Reviewer
from marvis.pipeline import METRICS_STAGE_FAILURE_PREFIX
from marvis.plugins.manifest import ToolRef
from marvis.recovery import (
    last_completed_step,
    reclaim_running_plans,
    reclaim_stale_running_tasks,
)


def test_last_completed_step_returns_none_when_dir_empty(tmp_path):
    assert last_completed_step(tmp_path) is None


def test_last_completed_step_detects_notebook_step(tmp_path):
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    (execution_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    (execution_dir / "code_model_scores.csv").write_text(
        "row_index,code_model_score\n0,0.1\n",
        encoding="utf-8",
    )
    (execution_dir / "runtime_contract.json").write_text("{}", encoding="utf-8")

    assert last_completed_step(tmp_path) == "notebook"


def test_last_completed_step_detects_artifacts_step(tmp_path):
    execution_dir = tmp_path / "execution"
    execution_dir.mkdir()
    (execution_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    (execution_dir / "code_model_scores.csv").write_text(
        "row_index,code_model_score\n0,0.1\n",
        encoding="utf-8",
    )
    (execution_dir / "runtime_contract.json").write_text("{}", encoding="utf-8")
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "validation.xlsx").write_bytes(b"PK\x03\x04")
    (outputs_dir / "validation_report.docx").write_bytes(b"PK\x03\x04")

    assert last_completed_step(tmp_path) == "artifacts"


def test_reclaim_stale_running_tasks_marks_orphan_running_tasks_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED
    assert loaded.status_message == "reclaimed: server restart while running"
    assert loaded.status_reason_code == TASK_STATUS_REASON_SERVER_RESTART


def test_reclaim_stale_running_tasks_marks_later_active_states_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED


def test_reclaim_stale_running_tasks_preserves_agent_writing_artifacts_without_active_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    assert repo.list_agent_messages(task.id) == []


def test_reclaim_stale_running_tasks_preserves_agent_writing_artifacts_with_active_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    job_id = repo.start_job(task.id, "report")

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row == ("failed", "ServerRestart")
    messages = repo.list_agent_messages(task.id)
    assert messages[-1]["stage"] == "failure"
    assert messages[-1]["metadata"]["interrupted_by_restart"] is True


def test_reclaim_stale_running_tasks_skips_recent_active_agent_job_within_stale_window(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    job_id = repo.start_job(task.id, "report")

    # With a one-hour stale window a just-updated task is not yet stale, so it must
    # NOT be treated as interrupted: no premature "server restart" notice is
    # inserted (regression guard for the cutoff-gated active-job UNION half).
    reclaimed = reclaim_stale_running_tasks(db_path, stale_after_seconds=3600)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    assert repo.list_agent_messages(task.id) == []
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row == ("queued", None)


def test_reclaim_stale_running_tasks_finalizes_agent_draft_message_for_writing_artifacts_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    repo.update_status(
        task.id,
        TaskStatus.WRITING_ARTIFACTS,
        "metrics generated",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    job_id = repo.start_job(task.id, "agent")
    draft_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="word_conclusion_draft",
        content="",
        metadata={"streaming": True, "model_id": "m1"},
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 0
    assert loaded.status == TaskStatus.WRITING_ARTIFACTS
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row == ("failed", "ServerRestart")
    messages = {message["id"]: message for message in repo.list_agent_messages(task.id)}
    assert messages[draft_message["id"]]["metadata"]["streaming"] is False
    assert messages[draft_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[draft_message["id"]]["content"]


def test_reclaim_stale_running_tasks_marks_orphan_jobs_failed(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs(id, task_id, kind, status, created_at)
            VALUES('job-1', 'task-1', 'notebook', 'running', '2026-01-01T00:00:00+00:00')
            """
        )
        conn.commit()

    reclaim_stale_running_tasks(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, error_name FROM jobs WHERE id='job-1'"
        ).fetchone()
    assert row == ("failed", "ServerRestart")


def test_reclaim_stale_running_tasks_finalizes_streaming_agent_messages(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    empty_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="",
        metadata={"streaming": True, "model_id": "m1"},
    )
    partial_message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="metrics",
        content="已经写入的半截分析",
        metadata={"streaming": True, "model_id": "m1"},
    )

    reclaim_stale_running_tasks(db_path)

    messages = {message["id"]: message for message in repo.list_agent_messages(task.id)}
    assert messages[empty_message["id"]]["metadata"]["streaming"] is False
    assert messages[empty_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[empty_message["id"]]["content"]
    assert messages[partial_message["id"]]["metadata"]["streaming"] is False
    assert messages[partial_message["id"]]["metadata"]["interrupted_by_restart"] is True
    assert messages[partial_message["id"]]["content"].startswith("已经写入的半截分析")
    assert "输出在此处中断" in messages[partial_message["id"]]["content"]


def test_reclaim_stale_running_tasks_adds_agent_restart_notice(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            run_mode="agent",
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)

    reclaim_stale_running_tasks(db_path)

    messages = repo.list_agent_messages(task.id)
    assert messages[-1]["stage"] == "failure"
    assert messages[-1]["metadata"]["interrupted_by_restart"] is True
    assert "服务器重启" in messages[-1]["content"]


def test_reclaim_computing_metrics_with_complete_execution_resumes_via_metrics(tmp_path):
    # REL-2: a restart during COMPUTING_METRICS with intact execution/ artifacts
    # (runtime_contract.json + code_model_scores.csv + model_meta.json already
    # written by the notebook stage) must reclaim into a metrics-specific
    # failure so retry goes through the cheap metrics-only path instead of a
    # full notebook re-run.
    db_path = tmp_path / "app.sqlite"
    workspace = tmp_path / "workspace"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    execution_dir = workspace / "tasks" / task.id / "execution"
    execution_dir.mkdir(parents=True)
    (execution_dir / "model_meta.json").write_text("{}", encoding="utf-8")
    (execution_dir / "code_model_scores.csv").write_text(
        "row_index,code_model_score\n0,0.1\n",
        encoding="utf-8",
    )
    (execution_dir / "runtime_contract.json").write_text("{}", encoding="utf-8")

    reclaimed = reclaim_stale_running_tasks(db_path, tasks_dir=workspace / "tasks")

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED
    assert loaded.status_message.startswith(METRICS_STAGE_FAILURE_PREFIX)
    assert loaded.status_reason_code == TASK_STATUS_REASON_SERVER_RESTART
    assert is_metrics_failure(loaded) is True


def test_reclaim_computing_metrics_without_execution_artifacts_keeps_generic_message(tmp_path):
    # A restart during COMPUTING_METRICS whose execution/ artifacts are NOT
    # intact (e.g. crashed mid-notebook before contract files were written)
    # must NOT be misrouted into the metrics-only retry path; it keeps the
    # generic reclaim message so the user reruns from the notebook stage.
    db_path = tmp_path / "app.sqlite"
    workspace = tmp_path / "workspace"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )
    execution_dir = workspace / "tasks" / task.id / "execution"
    execution_dir.mkdir(parents=True)
    # incomplete: missing runtime_contract.json / model_meta.json

    reclaimed = reclaim_stale_running_tasks(db_path, tasks_dir=workspace / "tasks")

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED
    assert loaded.status_message == "reclaimed: server restart while running"
    assert is_metrics_failure(loaded) is False


def test_reclaim_without_tasks_dir_keeps_generic_message_for_computing_metrics(tmp_path):
    # Backward compatibility: callers that do not pass tasks_dir (or pass
    # None) must keep today's behavior exactly.
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
        )
    )
    repo.update_status(task.id, TaskStatus.SCANNED, "scanned", expected=TaskStatus.CREATED)
    repo.update_status(task.id, TaskStatus.RUNNING, "running", expected=TaskStatus.SCANNED)
    repo.update_status(task.id, TaskStatus.EXECUTED, "executed", expected=TaskStatus.RUNNING)
    repo.update_status(
        task.id,
        TaskStatus.COMPUTING_METRICS,
        "computing",
        expected=TaskStatus.EXECUTED,
    )

    reclaimed = reclaim_stale_running_tasks(db_path)

    loaded = repo.get_task(task.id)
    assert reclaimed == 1
    assert loaded.status == TaskStatus.FAILED
    assert loaded.status_message == "reclaimed: server restart while running"


def _driver_task(repo: TaskRepository, tmp_path):
    return repo.create_task(
        TaskCreate(
            model_name="模型",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            task_type=TASK_TYPE_DATA_JOIN,
            run_mode="agent",
        )
    )


def _running_plan_with_step(task_id: str, *, step_status: StepStatus) -> Plan:
    step = PlanStep(
        id="step-1",
        plan_id="plan-1",
        index=0,
        title="join it",
        tool_ref=ToolRef("data", "execute_join"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=step_status,
    )
    return Plan(
        id="plan-1",
        task_id=task_id,
        goal="join two tables",
        source="template",
        template_id="data_join",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.RUNNING,
    )


def _reviewer():
    return Reviewer(lambda: None)


def test_reclaim_running_plans_pauses_plan_and_marks_step_interrupted(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = _driver_task(task_repo, tmp_path)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(_running_plan_with_step(task.id, step_status=StepStatus.RUNNING))

    reclaimed = reclaim_running_plans(
        plan_repo,
        _reviewer(),
        None,
        HarnessState(plan_repo),
        task_repo,
    )

    assert reclaimed == 1
    plan = plan_repo.load_plan("plan-1")
    assert plan.status == PlanStatus.FAILED
    assert plan.steps[0].status == StepStatus.FAILED
    assert "explicit retry required" in plan.steps[0].error

    messages = task_repo.list_agent_messages(task.id)
    assert messages, "expected a restart notice in the task conversation"
    last = messages[-1]
    assert last["metadata"]["plan_interrupted_by_restart"] is True
    assert last["metadata"]["plan_id"] == "plan-1"
    assert last["metadata"]["error"] is True
    assert last["metadata"]["error_diagnostic"] == {
        "schema_version": "workflow_error.v1",
        "workflow": TASK_TYPE_DATA_JOIN,
        "code": "server_restart_interrupted",
        "phase": "execution",
        "title": "数据处理执行中断",
        "summary": "服务重启中断了“join it”步骤，已完成步骤和中间产物均已保留。",
        "cause": "MARVIS 服务在该步骤执行期间重启；源材料没有被修改。",
        "location": "join it",
        "evidence": [
            {"label": "计划", "value": "plan-1"},
            {"label": "失败步骤", "value": "step-1"},
        ],
        "actions": ["回复“重试当前步骤”，从该失败步骤继续。"],
        "agent_prompt": "请回复“重试当前步骤”，Agent 将复用已完成步骤和中间产物继续执行。",
        "recovery_actions": [
            {"label": "重试当前步骤", "command": "重试当前步骤"}
        ],
        "retryable": True,
        "auto_recoverable": True,
        "impact": "失败步骤之后的依赖步骤尚未执行。",
        "exception_type": "ServerRestart",
        "technical_detail": (
            "ServerRestart: interrupted during running before output was persisted; "
            "explicit retry required"
        ),
    }
    assert last["metadata"]["failure_envelope"] == {
        "schema_version": "failure.v1",
        "failed_step_id": "step-1",
        "error_kind": "ServerRestart",
        "message": "服务重启中断了“join it”步骤，已完成步骤和中间产物均已保留。",
        "retryable": True,
        "editable_input_schema": {},
        "suggested_actions": ["retry"],
        "downstream_reset": "dependent_steps",
        "downstream_reset_steps": [],
    }
    assert "服务已重启" in last["content"]
    assert "计划已暂停" in last["content"]
    assert "回复“重试当前步骤”" in last["content"]
    assert "点击" not in last["content"]


def test_reclaim_running_plan_finalizes_tool_progress_message_in_place(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = _driver_task(task_repo, tmp_path)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(
        _running_plan_with_step(task.id, step_status=StepStatus.RUNNING)
    )
    progress = {
        "kind": "model_tuning",
        "algorithm": "xgb",
        "trial": 9,
        "trial_total": 40,
    }
    message = task_repo.add_agent_message(
        task.id,
        role="assistant",
        stage="chat",
        content="模型调参正在执行：xgb，当前轮次 9/40。",
        metadata={
            "kind": "tool_progress",
            "plan_id": "plan-1",
            "step_id": "step-1",
            "run_id": "run-1",
            "status": "running",
            "streaming": True,
            "progress": progress,
        },
    )

    reclaimed = reclaim_running_plans(
        plan_repo,
        _reviewer(),
        None,
        HarnessState(plan_repo),
        task_repo,
    )

    assert reclaimed == 1
    messages = {item["id"]: item for item in task_repo.list_agent_messages(task.id)}
    recovered = messages[message["id"]]
    assert recovered["metadata"]["streaming"] is False
    assert recovered["metadata"]["status"] == "interrupted"
    assert recovered["metadata"]["interrupted_by_restart"] is True
    assert recovered["metadata"]["progress"] == progress
    assert "调参已中断" in recovered["content"]


def test_reclaim_running_plan_preserves_awaiting_confirmation_boundary(tmp_path):
    """A restart can land after the step became awaiting_confirm but before the
    plan-level status write.  Recovery must finish that transition, not create
    the contradictory FAILED-plan / awaiting-step state."""
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = _driver_task(task_repo, tmp_path)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(
        _running_plan_with_step(task.id, step_status=StepStatus.AWAITING_CONFIRM)
    )

    reclaimed = reclaim_running_plans(
        plan_repo,
        _reviewer(),
        None,
        HarnessState(plan_repo),
        task_repo,
    )

    assert reclaimed == 1
    plan = plan_repo.load_plan("plan-1")
    assert plan.status == PlanStatus.AWAITING_CONFIRM
    assert plan.steps[0].status == StepStatus.AWAITING_CONFIRM
    messages = task_repo.list_agent_messages(task.id)
    assert messages[-1]["metadata"]["plan_resumed_at_confirmation"] is True
    assert "等待你的确认" in messages[-1]["content"]
    assert "重试步骤" not in messages[-1]["content"]


def test_reclaim_running_plans_is_idempotent_across_two_startups(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = _driver_task(task_repo, tmp_path)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(_running_plan_with_step(task.id, step_status=StepStatus.RUNNING))

    first = reclaim_running_plans(plan_repo, _reviewer(), None, HarnessState(plan_repo), task_repo)
    second = reclaim_running_plans(plan_repo, _reviewer(), None, HarnessState(plan_repo), task_repo)

    assert first == 1
    # The plan is FAILED after the first pass, so list_plans_by_status(RUNNING)
    # finds nothing the second time around — no duplicate notice, no crash.
    assert second == 0
    messages = task_repo.list_agent_messages(task.id)
    restart_notices = [
        m for m in messages if m["metadata"].get("plan_interrupted_by_restart") is True
    ]
    assert len(restart_notices) == 1


def test_reclaim_running_plans_releases_orphan_driver_job(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    task = _driver_task(task_repo, tmp_path)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(_running_plan_with_step(task.id, step_status=StepStatus.RUNNING))
    job_id = task_repo.start_job(task.id, "driver_turn")
    task_repo.mark_job_running(job_id)
    # The unique partial index only allows one active job per task; assert the
    # guard is actually armed before reclaim, so the "released" assertion below
    # is meaningful rather than vacuously true.
    assert task_repo.task_has_active_job(task.id)

    reclaim_running_plans(plan_repo, _reviewer(), None, HarnessState(plan_repo), task_repo)

    assert not task_repo.task_has_active_job(task.id)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, error_name FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    assert row == ("failed", "ServerRestart")
    # And the task can immediately accept a new job (the whole point of
    # releasing the guard) instead of staying wedged behind a 409.
    new_job_id = task_repo.start_job(task.id, "driver_turn")
    assert new_job_id


def test_reclaim_running_plans_ignores_normally_running_job_for_other_task(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task_repo = TaskRepository(db_path)
    running_task = _driver_task(task_repo, tmp_path)
    # A second, unrelated task with a genuinely healthy in-flight job and no
    # RUNNING plan at all must be left completely untouched by the reclaim pass.
    healthy_task = task_repo.create_task(
        TaskCreate(
            model_name="模型2",
            model_version="v1",
            validator="验证人员",
            source_dir=str(tmp_path),
            task_type=TASK_TYPE_DATA_JOIN,
            run_mode="agent",
        )
    )
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(
        _running_plan_with_step(running_task.id, step_status=StepStatus.RUNNING)
    )
    healthy_job_id = task_repo.start_job(healthy_task.id, "driver_turn")
    task_repo.mark_job_running(healthy_job_id)

    reclaim_running_plans(plan_repo, _reviewer(), None, HarnessState(plan_repo), task_repo)

    assert task_repo.task_has_active_job(healthy_task.id)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM jobs WHERE id = ?", (healthy_job_id,)
        ).fetchone()
    assert row == ("running",)
    assert task_repo.list_agent_messages(healthy_task.id) == []
