import pytest

from marvis.db import TaskRepository, _ensure_column, connect, init_db
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    TaskCreate,
    TaskStatus,
)
from marvis.state_machine import ConflictError, IllegalTransition


def _task_create(model_name: str = "模型", **overrides) -> TaskCreate:
    values = {
        "model_name": model_name,
        "model_version": "v1",
        "validator": "验证人员",
        "source_dir": "/tmp/source",
        "algorithm": "lgb",
        "run_mode": "manual",
        "target_col": "y",
        "score_col": "pred",
        "split_col": "split",
        "time_col": "apply_month",
        "feature_columns": [],
        "notebook_path": None,
        "sample_path": None,
        "pmml_path": None,
        "dictionary_path": None,
        "report_values": {},
    }
    values.update(overrides)
    return TaskCreate(**values)


def test_create_and_get_task_round_trips_v2_fields(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    task = repo.create_task(
        _task_create(
            model_name="贷前评分卡",
            model_version="202604",
            validator="验证人员A",
            algorithm="lr",
            run_mode="agent",
            target_col="target",
            score_col="score",
            split_col="sample_type",
            time_col="month",
            feature_columns=["x1", "x2"],
            sample_weight_col="sample_weight",
            notebook_path="/tmp/source/model.ipynb",
            sample_path="/tmp/source/sample.csv",
            pmml_path="/tmp/source/model.pmml",
            dictionary_path="/tmp/source/dictionary.xlsx",
            report_values={"TEXT:report_title": "自定义标题"},
        )
    )

    loaded = repo.get_task(task.id)
    assert loaded.id == task.id
    assert loaded.model_name == "贷前评分卡"
    assert loaded.model_version == "202604"
    assert loaded.validator == "验证人员A"
    assert loaded.algorithm == "lr"
    assert loaded.run_mode == "agent"
    assert loaded.target_col == "target"
    assert loaded.score_col == "score"
    assert loaded.split_col == "sample_type"
    assert loaded.time_col == "month"
    assert loaded.feature_columns == ["x1", "x2"]
    assert loaded.sample_weight_col == "sample_weight"
    assert loaded.notebook_path == "/tmp/source/model.ipynb"
    assert loaded.sample_path == "/tmp/source/sample.csv"
    assert loaded.pmml_path == "/tmp/source/model.pmml"
    assert loaded.dictionary_path == "/tmp/source/dictionary.xlsx"
    assert loaded.report_values_revision == 0
    assert loaded.status == TaskStatus.CREATED
    assert loaded.status_message == "created"
    assert loaded.status_reason_code == ""

    values, revision = repo.get_report_values(task.id)
    assert values == {"TEXT:report_title": "自定义标题"}
    assert revision == 0


def test_create_task_rejects_unknown_algorithm_and_normalizes_run_mode_defaults(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    with pytest.raises(ValueError, match="unsupported model algorithm"):
        repo.create_task(_task_create(algorithm="unknown", run_mode="auto"))

    task = repo.create_task(_task_create(algorithm="", run_mode="auto"))
    loaded = repo.get_task(task.id)
    assert loaded.algorithm == ""
    assert loaded.run_mode == "manual"


def test_create_task_accepts_supported_algorithm_choices(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    algorithms = ["xgb", "lgb", "lr", "catboost", "scorecard", "dnn"]
    loaded = [
        repo.get_task(repo.create_task(_task_create(algorithm=algorithm)).id).algorithm
        for algorithm in algorithms
    ]

    assert loaded == algorithms


def test_get_task_missing_raises_key_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    with pytest.raises(KeyError, match="Task not found: missing"):
        repo.get_task("missing")


def test_delete_task_removes_task_record(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.delete_task(task.id)

    assert repo.list_tasks() == []
    with pytest.raises(KeyError, match=f"Task not found: {task.id}"):
        repo.get_task(task.id)


def test_delete_task_missing_raises_key_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    with pytest.raises(KeyError, match="Task not found: missing"):
        repo.delete_task("missing")


def test_start_job_allows_only_one_active_job_per_task(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())
    other_task = repo.create_task(_task_create(model_name="另一个模型"))

    first_job_id = repo.start_job(task.id, "notebook")
    other_job_id = repo.start_job(other_task.id, "notebook")

    assert first_job_id
    assert other_job_id
    assert first_job_id != other_job_id
    assert repo.task_has_active_job(task.id) is True
    assert repo.get_active_job_kind(task.id) == "notebook"
    assert repo.get_active_job_kind(other_task.id) == "notebook"
    with pytest.raises(ConflictError, match="already has an active job"):
        repo.start_job(task.id, "metrics")

    repo.finish_job(first_job_id, status="succeeded")
    assert repo.get_active_job_kind(task.id) is None
    retry_job_id = repo.start_job(task.id, "metrics")

    assert retry_job_id != first_job_id
    assert repo.get_active_job_kind(task.id) == "metrics"


def test_list_tasks_returns_created_tasks(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    first = repo.create_task(_task_create(model_name="模型A"))
    second = repo.create_task(_task_create(model_name="模型B"))

    tasks = repo.list_tasks()

    assert {task.id for task in tasks} == {first.id, second.id}
    assert {task.model_name for task in tasks} == {"模型A", "模型B"}


def test_update_report_values_merges_and_increments_revision(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(
        _task_create(
            report_values={
                "TEXT:report_title": "旧标题",
                "TEXT:model_scope": "旧范围",
            }
        )
    )

    new_revision = repo.update_report_values(
        task.id,
        {"TEXT:report_title": "新标题"},
        expected_revision=0,
    )

    assert new_revision == 1
    assert repo.get_report_values(task.id) == (
        {"TEXT:report_title": "新标题", "TEXT:model_scope": "旧范围"},
        1,
    )
    assert repo.get_task(task.id).report_values_revision == 1


def test_update_report_values_rejects_conflict(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.update_report_values(
        task.id,
        {"TEXT:report_title": "新标题"},
        expected_revision=0,
    )

    with pytest.raises(ConflictError):
        repo.update_report_values(
            task.id,
            {"TEXT:model_scope": "范围"},
            expected_revision=0,
        )


@pytest.mark.parametrize(
    "values",
    [
        {"report_title": "missing prefix"},
        {"TEXT:report_title": 123},
    ],
)
def test_update_report_values_rejects_invalid_values(tmp_path, values):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    with pytest.raises(ValueError):
        repo.update_report_values(task.id, values, expected_revision=0)


def test_update_report_values_rejects_platform_computed_values(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    with pytest.raises(ValueError, match="platform-computed"):
        repo.update_report_values(
            task.id,
            {"TEXT:train_test_period": "人工覆盖"},
            expected_revision=0,
        )


def test_update_agent_report_conclusions_allows_only_final_three_keys(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(run_mode="agent"))

    revision = repo.update_agent_report_conclusions(
        task.id,
        {
            "TEXT:pressure_test_summary": "压力测试显示整体稳定。",
            "TEXT:pressure_impact_recommendation": "建议持续监控关键数据源。",
            "TEXT:final_validation_conclusion": "从当前验证结果看，模型可复现性、区分效果和稳定性整体满足验证要求。",
        },
        expected_revision=0,
    )

    assert revision == 1
    values, stored_revision = repo.get_report_values(task.id)
    assert stored_revision == 1
    assert values["TEXT:pressure_test_summary"] == "压力测试显示整体稳定。"
    assert values["TEXT:final_validation_conclusion"].startswith("从当前验证结果看")

    with pytest.raises(ValueError, match="only update agent conclusion keys"):
        repo.update_agent_report_conclusions(
            task.id,
            {"TEXT:report_title": "不允许"},
            expected_revision=1,
        )


def test_agent_messages_round_trip_with_metadata(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(run_mode="agent"))

    repo.add_agent_message(
        task.id,
        role="user",
        stage="chat",
        content="开始验证",
        metadata={"model_id": "m1"},
    )
    repo.add_agent_message(
        task.id,
        role="assistant",
        stage="scan",
        content="材料齐全。",
        metadata={"checks": 4},
    )

    messages = repo.list_agent_messages(task.id)

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["metadata"]["model_id"] == "m1"
    assert messages[1]["metadata"]["checks"] == 4


def test_agent_message_can_be_updated_for_streaming_chunks(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(run_mode="agent"))

    message = repo.add_agent_message(
        task.id,
        role="assistant",
        stage="metrics",
        content="",
        metadata={"streaming": True, "model_id": "m1"},
    )

    repo.update_agent_message(
        message["id"],
        content="第一段",
        metadata={"streaming": True, "model_id": "m1"},
    )
    updated = repo.update_agent_message(
        message["id"],
        content="第一段第二段",
        metadata={"streaming": False, "model_id": "m1", "streamed": True},
    )
    messages = repo.list_agent_messages(task.id)

    assert updated["created_at"] == message["created_at"]
    assert messages == [updated]
    assert messages[0]["content"] == "第一段第二段"
    assert messages[0]["metadata"]["streaming"] is False
    assert messages[0]["metadata"]["streamed"] is True


def test_update_task_status_through_v2_pipeline_states(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.update_status(
        task.id,
        TaskStatus.SCANNED,
        "scanned",
        expected=TaskStatus.CREATED,
    )
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
        "writing",
        expected=TaskStatus.COMPUTING_METRICS,
    )
    repo.update_status(
        task.id,
        TaskStatus.SUCCEEDED,
        "done",
        expected=TaskStatus.WRITING_ARTIFACTS,
    )

    loaded = repo.get_task(task.id)
    assert loaded.status == TaskStatus.SUCCEEDED
    assert loaded.status_message == "done"
    assert loaded.status_reason_code == ""


def test_update_status_and_message_can_persist_structured_reason_code(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.update_status(
        task.id,
        TaskStatus.SCANNED,
        "stopped",
        expected=TaskStatus.CREATED,
        reason_code=TASK_STATUS_REASON_USER_CANCELLED,
    )
    assert repo.get_task(task.id).status_reason_code == TASK_STATUS_REASON_USER_CANCELLED

    repo.update_status_message(task.id, "still stopped")
    assert repo.get_task(task.id).status_reason_code == TASK_STATUS_REASON_USER_CANCELLED

    repo.update_status(
        task.id,
        TaskStatus.RUNNING,
        "running",
        expected=TaskStatus.SCANNED,
    )
    assert repo.get_task(task.id).status_reason_code == ""


def test_update_status_rejects_illegal_transition(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.update_status(task.id, TaskStatus.SCANNED, "ok", expected=TaskStatus.CREATED)

    with pytest.raises(IllegalTransition, match="scanned -> succeeded"):
        repo.update_status(task.id, TaskStatus.SUCCEEDED, "x", expected=TaskStatus.SCANNED)


def test_update_status_rejects_stale_expected_status(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    repo.update_status(task.id, TaskStatus.SCANNED, "ok", expected=TaskStatus.CREATED)

    with pytest.raises(IllegalTransition, match="scanned -> running"):
        repo.update_status(task.id, TaskStatus.RUNNING, "stale", expected=TaskStatus.CREATED)


def test_update_task_status_missing_raises_key_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    with pytest.raises(KeyError, match="Task not found: missing"):
        repo.update_status(
            "missing",
            TaskStatus.RUNNING,
            message="执行中",
            expected=TaskStatus.SCANNED,
        )


def test_ensure_column_rejects_unsafe_identifiers(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        with pytest.raises(ValueError, match="unsupported migration table"):
            _ensure_column(conn, "tasks;DROP", "safe_column", "TEXT")
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            _ensure_column(conn, "tasks", "bad-column", "TEXT")
