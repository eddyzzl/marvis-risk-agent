import pytest

import marvis.db as db_module
import marvis.repositories.tasks as task_repo_module
from marvis.db import (
    DatasetRepository,
    ModelingRepository,
    PlanRepository,
    StrategyRepository,
    TaskRepository,
    _ensure_column,
    _list_audit_rows,
    connect,
    init_db,
)
from marvis.data.contracts import ColumnFingerprint, ColumnProfile, Dataset, JoinPlan
from marvis.domain import (
    TASK_STATUS_REASON_USER_CANCELLED,
    TaskCreate,
    TaskStatus,
)
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep
from marvis.packs.modeling import Experiment, ModelArtifact, ModelMetrics, TrainConfig
from marvis.packs.strategy import build_strategy
from marvis.plugins.manifest import ToolRef
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


def test_task_repository_is_reexported_from_db_for_compatibility():
    assert TaskRepository is task_repo_module.TaskRepository


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


def test_get_latest_job_returns_filtered_status_and_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create())

    first_job_id = repo.start_job(task.id, "join")
    repo.mark_job_running(first_job_id)
    repo.finish_job(
        first_job_id,
        status="failed",
        error_name="FanOutError",
        error_value="join produced 12 > anchor 10 rows",
        traceback="hidden traceback",
    )
    second_job_id = repo.start_job(task.id, "metrics")

    latest_metrics = repo.get_latest_job(task.id, kind="metrics")
    latest_join = repo.get_latest_job(task.id, kind="join")

    assert latest_metrics["id"] == second_job_id
    assert latest_metrics["kind"] == "metrics"
    assert latest_join == {
        "id": first_job_id,
        "task_id": task.id,
        "kind": "join",
        "status": "failed",
        "progress_message": "",
        "error_name": "FanOutError",
        "error_value": "join produced 12 > anchor 10 rows",
        "created_at": latest_join["created_at"],
        "started_at": latest_join["started_at"],
        "finished_at": latest_join["finished_at"],
        "log_path": None,
    }
    assert "traceback" not in latest_join


def test_list_tasks_returns_created_tasks(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    first = repo.create_task(_task_create(model_name="模型A"))
    second = repo.create_task(_task_create(model_name="模型B"))

    tasks = repo.list_tasks()

    assert {task.id for task in tasks} == {first.id, second.id}
    assert {task.model_name for task in tasks} == {"模型A", "模型B"}
    assert repo.list_tasks(limit=1)[0].id == tasks[0].id
    assert [task.id for task in repo.list_tasks(limit=1, offset=1)] == [tasks[1].id]


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


def test_update_report_values_with_audit_records_changed_keys(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(report_values={"TEXT:report_title": "旧标题"}))

    revision = repo.update_report_values_with_audit(
        task.id,
        {"TEXT:report_title": "新标题"},
        expected_revision=0,
        audit={
            "kind": "report.values.update",
            "target_ref": task.id,
            "outcome": "succeeded",
            "detail": {"keys": ["TEXT:report_title"], "expected_revision": 0},
        },
    )

    assert revision == 1
    assert repo.get_report_values(task.id) == ({"TEXT:report_title": "新标题"}, 1)
    audit = db_module.PluginRepository(db_path).list_audit(kind="report.values.update")[0]
    assert audit["target_ref"] == task.id
    assert audit["detail"]["keys"] == ["TEXT:report_title"]


def test_update_report_values_with_audit_rolls_back_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(report_values={"TEXT:report_title": "旧标题"}))

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(task_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.update_report_values_with_audit(
            task.id,
            {"TEXT:report_title": "新标题"},
            expected_revision=0,
            audit={
                "kind": "report.values.update",
                "target_ref": task.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_report_values(task.id) == ({"TEXT:report_title": "旧标题"}, 0)


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


def test_update_agent_report_conclusions_with_audit_rolls_back_when_audit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(run_mode="agent"))
    values = {
        "TEXT:pressure_test_summary": "压力测试显示整体稳定。",
        "TEXT:pressure_impact_recommendation": "建议持续监控关键数据源。",
        "TEXT:final_validation_conclusion": "从当前验证结果看，模型可复现性、区分效果和稳定性整体满足验证要求。",
    }

    def fail_audit(*args, **kwargs):
        raise RuntimeError("audit down")

    monkeypatch.setattr(task_repo_module, "_write_audit_row", fail_audit)

    with pytest.raises(RuntimeError, match="audit down"):
        repo.update_agent_report_conclusions_with_audit(
            task.id,
            values,
            expected_revision=0,
            audit={
                "kind": "report.agent_conclusions.confirm",
                "target_ref": task.id,
                "outcome": "succeeded",
            },
        )

    assert repo.get_report_values(task.id) == ({}, 0)


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


def test_agent_messages_can_list_after_cursor(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)
    task = repo.create_task(_task_create(run_mode="agent"))

    first = repo.add_agent_message(task.id, role="user", stage="chat", content="first")
    second = repo.add_agent_message(task.id, role="assistant", stage="scan", content="second")
    third = repo.add_agent_message(task.id, role="assistant", stage="scan", content="third")

    messages = repo.list_agent_messages(task.id, after_id=first["id"])

    assert [message["id"] for message in messages] == [second["id"], third["id"]]
    limited_messages = repo.list_agent_messages(task.id, after_id=first["id"], limit=1)
    assert [message["id"] for message in limited_messages] == [second["id"]]
    first_page = repo.list_agent_messages(task.id, limit=2)
    assert [message["id"] for message in first_page] == [first["id"], second["id"]]
    assert repo.has_agent_message(task.id, first["id"]) is True
    assert repo.has_agent_message(task.id, "missing-message") is False


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


def test_init_db_migrates_pre_heartbeat_jobs_table(tmp_path):
    # REL-5: simulate a database created before jobs.heartbeat_at existed —
    # init_db must add the column via _ensure_column without touching existing
    # rows, so an upgrade doesn't lose in-flight job state.
    db_path = tmp_path / "app.sqlite"
    with connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_message TEXT NOT NULL DEFAULT '',
                error_name TEXT,
                error_value TEXT,
                traceback TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                log_path TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO jobs(id, task_id, kind, status, created_at)
            VALUES ('job-1', 'task-1', 'join', 'running', '2026-01-01T00:00:00+00:00')
            """
        )

    init_db(db_path)

    with connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "heartbeat_at" in columns
        row = conn.execute(
            "SELECT status, heartbeat_at FROM jobs WHERE id = 'job-1'"
        ).fetchone()
    assert row["status"] == "running"
    assert row["heartbeat_at"] is None


def test_ensure_column_rejects_unsafe_identifiers(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)

    with connect(db_path) as conn:
        with pytest.raises(ValueError, match="unsupported migration table"):
            _ensure_column(conn, "tasks;DROP", "safe_column", "TEXT")
        with pytest.raises(ValueError, match="unsafe SQL identifier"):
            _ensure_column(conn, "tasks", "bad-column", "TEXT")


def _profile(name: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="object",
        semantic_role="id",
        fingerprint=ColumnFingerprint("categorical", None, None, False, None, None, None),
        null_rate=0.0,
        cardinality=2,
        sample_values=("a***", "b***"),
    )


def _dataset(dataset_id: str, *, task_id: str, source_path: str | None = None) -> Dataset:
    return Dataset(
        id=dataset_id,
        task_id=task_id,
        role="sample",
        source_path=source_path or f"{task_id}/{dataset_id}.parquet",
        format="parquet",
        sheet=None,
        row_count=2,
        columns=(_profile("customer_id"),),
        has_target=False,
        target_col=None,
        created_at="2026-06-19T00:00:00Z",
    )


def _plan(plan_id: str, task_id: str) -> Plan:
    step = PlanStep(
        id=f"{plan_id}-step-1",
        plan_id=plan_id,
        index=0,
        title="step",
        tool_ref=ToolRef("_sample", "echo"),
        inputs={},
        depends_on=[],
        post_checks=[],
        needs_confirmation=False,
        decision_point=False,
        sub_agent_scope=None,
        granted_tools=[],
    )
    return Plan(
        id=plan_id,
        task_id=task_id,
        goal="finish",
        source="template",
        template_id="test",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.CONFIRMED,
    )


def _experiment(experiment_id: str, task_id: str) -> Experiment:
    config = TrainConfig(
        dataset_id="dataset-1",
        features=("score",),
        target_col="bad",
        split_col="split",
        split_values={"train": "train", "test": "test", "oot": "oot"},
        params={},
        seed=7,
        early_stopping_rounds=10,
    )
    return Experiment(
        id=experiment_id,
        task_id=task_id,
        recipe_id="lgb",
        config=config,
        metrics=ModelMetrics(
            train_ks=0.4,
            test_ks=0.37,
            oot_ks=0.35,
            train_auc=0.78,
            test_auc=0.74,
            oot_auc=0.72,
            psi_test_vs_train=0.04,
            psi_oot_vs_train=0.08,
            overfit_train_test_gap=0.03,
            overfit_train_oot_gap=0.05,
            overfit_flag=False,
        ),
        artifact_id=None,
        status="pending",
        created_at="2026-06-19T00:00:00Z",
    )


def _seed_full_task_graph(db_path, task_id: str, *, shared_dataset_path: str | None = None):
    task_repo = TaskRepository(db_path)
    task = task_repo.create_task(_task_create(model_name=f"model-{task_id}"))
    dataset_repo = DatasetRepository(db_path)
    dataset = _dataset(f"ds-{task.id}", task_id=task.id, source_path=shared_dataset_path)
    dataset_repo.create_dataset(dataset)
    join_plan = JoinPlan(
        id=f"join-{task.id}",
        task_id=task.id,
        anchor_dataset_id=dataset.id,
        joins=[],
        status="draft",
    )
    dataset_repo.create_join_plan(join_plan)
    plan_repo = PlanRepository(db_path)
    plan_repo.create_plan(_plan(f"plan-{task.id}", task.id))
    modeling_repo = ModelingRepository(db_path)
    experiment = _experiment(f"exp-{task.id}", task.id)
    modeling_repo.create_experiment(experiment)
    artifact = ModelArtifact(
        id=f"artifact-{task.id}",
        experiment_id=experiment.id,
        algorithm="lgb",
        model_path=f"{task.id}/model.pkl",
        pmml_path=None,
        feature_list=("score",),
        params={},
        woe_maps=None,
        created_at="2026-06-19T00:00:00Z",
    )
    modeling_repo.create_model_artifact(artifact)
    strategy_repo = StrategyRepository(db_path)
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="baseline cutoff",
    )
    strategy_repo.create_strategy(task.id, strategy, created_at="2026-06-19T00:00:00Z")
    return task, dataset


def test_purge_task_removes_all_task_scoped_rows_and_writes_audit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, dataset = _seed_full_task_graph(db_path, "task-a")
    repo = TaskRepository(db_path)

    summary = repo.purge_task(task.id, actor="tester")

    assert summary["datasets"] == 1
    assert summary["joins"] == 1
    assert summary["plans"] == 1
    assert summary["plan_steps"] == 1
    assert summary["experiments"] == 1
    assert summary["model_artifacts"] == 1
    assert summary["strategies"] == 1
    assert summary["datasets_shared_with_other_tasks"] == 0
    assert summary["dataset_source_paths"] == [dataset.source_path]

    with pytest.raises(KeyError):
        repo.get_task(task.id)
    with connect(db_path) as conn:
        for table, column in (
            ("datasets", "task_id"),
            ("joins", "task_id"),
            ("plans", "task_id"),
            ("experiments", "task_id"),
            ("strategies", "task_id"),
            ("model_artifacts", "experiment_id"),
            ("plan_steps", "plan_id"),
        ):
            remaining = conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            assert remaining == 0, f"{table} still has rows after purge"

    audit_rows = _list_audit_rows(db_path, kind="task.delete")
    assert len(audit_rows) == 1
    assert audit_rows[0]["target_ref"] == task.id
    assert audit_rows[0]["actor"] == "tester"
    assert audit_rows[0]["detail"]["purge_summary"]["datasets"] == 1


def test_purge_task_missing_raises_key_error(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = TaskRepository(db_path)

    with pytest.raises(KeyError, match="Task not found: missing"):
        repo.purge_task("missing")
    with pytest.raises(KeyError, match="Task not found: missing"):
        repo.purge_preview("missing")


def test_purge_preview_matches_purge_task_summary_without_deleting(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    task, _dataset = _seed_full_task_graph(db_path, "task-b")
    repo = TaskRepository(db_path)

    preview = repo.purge_preview(task.id)

    assert preview["datasets"] == 1
    assert preview["plans"] == 1
    assert "dataset_source_paths" not in preview
    # dry-run must not touch any rows
    assert repo.get_task(task.id).id == task.id
    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 1


def test_purge_task_does_not_report_dataset_file_shared_with_another_task(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    shared_path = "task-owner/shared.parquet"
    owner_task, _owner_dataset = _seed_full_task_graph(
        db_path, "task-owner", shared_dataset_path=shared_path
    )
    reuser_task_repo = TaskRepository(db_path)
    reuser_task = reuser_task_repo.create_task(_task_create(model_name="reuser"))
    DatasetRepository(db_path).create_dataset(
        _dataset(f"ds-{reuser_task.id}", task_id=reuser_task.id, source_path=shared_path)
    )

    repo = TaskRepository(db_path)
    reuser_summary = repo.purge_task(reuser_task.id)

    # the reuser's own dataset row referenced the shared file, but the owner task
    # still has a dataset row pointing at it -- the file must not be marked removable.
    assert reuser_summary["datasets_shared_with_other_tasks"] == 1
    assert reuser_summary["dataset_source_paths"] == []

    with connect(db_path) as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM datasets WHERE source_path = ?", (shared_path,)
        ).fetchone()[0]
    assert remaining == 1  # the owner task's dataset row survives

