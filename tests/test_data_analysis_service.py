from __future__ import annotations

import json

import pandas as pd
import pytest

from marvis.data.analysis_service import (
    DATA_ANALYSIS_SECTIONS,
    DataAnalysisArtifactError,
    DataAnalysisRequest,
    DataAnalysisRetryRequiredError,
    DataAnalysisService,
)
from marvis.data.backend import DataBackend
from marvis.data.descriptive import DescriptiveConfig, DescriptiveInputError
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
)
from marvis.db import DatasetRepository, TaskRepository, connect, init_db
from marvis.domain import TaskCreate
from marvis.repositories.data_analysis import DataAnalysisRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from marvis.state_machine import ConflictError


def _seed_workspace(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="data analysis",
            model_version="v1",
            validator="owner",
            source_dir=str(settings.workspace),
            task_type="strategy",
        )
    )
    dataset_dir = settings.datasets_dir / task.id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    parquet = dataset_dir / "sample.parquet"
    pd.DataFrame(
        {
            "account_no": [10001, 10002, 10003, 10004],
            "phone": ["13800138000", "13900139000", "13800138000", None],
            "customer_name": ["Alice", "Bob", "Alice", None],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "bad": [0, 1, 0, 1],
        }
    ).to_parquet(parquet, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(
        parquet,
        task_id=task.id,
        role="sample",
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "phone": "phone",
            "customer_name": "name",
            "amount": "amount",
            "bad": "target",
        },
        business_names={"amount": "申请金额"},
    )
    workspace_repo = DataWorkspaceRepository(settings.db_path)
    activated = workspace_repo.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    snapshot = workspace_repo.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="statistics",
            selected_field="amount",
            semantic_mapping=mapping,
        ),
        expected_revision=activated.revision,
    )
    return settings, task, dataset, snapshot


def _request(*, retry: bool = False) -> DataAnalysisRequest:
    return DataAnalysisRequest(
        sections=DATA_ANALYSIS_SECTIONS,
        columns=("account_no", "phone", "customer_name", "amount", "bad"),
        config=DescriptiveConfig(
            low_cardinality_threshold=10,
            histogram_bins=4,
        ),
        retry=retry,
    )


def test_service_dispatches_completes_canonical_private_safe_artifact_and_audit(
    tmp_path,
):
    settings, task, dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)

    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )

    assert dispatch.http_status == 202
    assert dispatch.should_execute is True
    assert dispatch.record.status == "queued"
    assert dispatch.record.job_id == dispatch.job_id
    service.run_job(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )

    view = service.get_run(task.id, dispatch.record.id)
    assert view is not None
    assert view.record.status == "succeeded"
    assert view.result_artifact_id
    assert view.download_url == (
        f"/api/tasks/{task.id}/task-artifacts/"
        f"{view.result_artifact_id}/download"
    )
    assert view.result is not None
    assert set(view.result) == {
        "schema_version",
        "identity",
        "request",
        "semantics",
        "analysis",
    }
    assert view.result["identity"]["dataset_content_hash"] == dataset.content_hash
    assert set(view.result["analysis"]) == set(DATA_ANALYSIS_SECTIONS)

    artifact = TaskArtifactRepository(settings.db_path).get_for_task(
        task.id,
        view.result_artifact_id,
    )
    assert artifact is not None
    assert artifact["path"] == (
        f"tasks/{task.id}/data_analysis/{dispatch.record.input_hash}.json"
    )
    artifact_path = settings.workspace / artifact["path"]
    raw = artifact_path.read_text(encoding="utf-8")
    assert raw == json.dumps(
        json.loads(raw),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert "13800138000" not in raw
    assert "13900139000" not in raw
    assert "Alice" not in raw
    assert "Bob" not in raw
    assert "10001" not in raw
    phone = next(
        field
        for field in view.result["analysis"]["distribution"]
        if field["name"] == "phone"
    )
    assert sum(
        item["count"] for item in phone["frequency"]["items"]
    ) + phone["frequency"]["other_count"] == 4
    assert all(
        item["value"]["value"].startswith("token:")
        for item in phone["frequency"]["items"]
        if item["value"]["type"] != "null"
    )
    account_no = next(
        field
        for field in view.result["analysis"]["distribution"]
        if field["name"] == "account_no"
    )
    assert next(
        column.semantic_role
        for column in dataset.columns
        if column.name == "account_no"
    ) == "idcard"
    assert "account_no" not in snapshot.semantic_mapping.field_roles
    assert account_no["numeric"] is None
    assert account_no["histogram"] is None
    assert account_no["sensitive_value_policy"] == (
        "frequency_tokenized_numeric_distribution_suppressed"
    )
    assert "account_no" not in view.result["analysis"]["correlation"]["columns"]

    audits = TaskRepository(settings.db_path).list_audit(
        target_ref=dispatch.record.id
    )
    assert [row["kind"] for row in audits] == [
        "data.analysis.started",
        "data.analysis.completed",
    ]
    audit_json = json.dumps(audits, ensure_ascii=False, sort_keys=True)
    assert "13800138000" not in audit_json
    assert "Alice" not in audit_json
    assert all("input_hash" in row["detail"] for row in audits)
    assert audits[-1]["detail"]["artifact_id"] == view.result_artifact_id
    assert service.task_repo.get_job(dispatch.job_id)["status"] == "succeeded"


def test_atomic_job_finish_failure_rolls_back_artifact_run_and_completion_audit(
    tmp_path,
    monkeypatch,
):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )

    original_finish = service.task_repo.finish_job_on_connection

    def fail_atomic_finish(*args, **kwargs):
        if kwargs.get("status") == "succeeded":
            raise RuntimeError("synthetic atomic job finish failure")
        return original_finish(*args, **kwargs)

    monkeypatch.setattr(
        service.task_repo,
        "finish_job_on_connection",
        fail_atomic_finish,
    )
    service.run_job(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )

    failed = service.get_run(task.id, dispatch.record.id)
    assert failed is not None
    assert failed.record.status == "failed"
    assert service.task_repo.get_job(dispatch.job_id)["status"] == "failed"
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    assert not (
        settings.tasks_dir
        / task.id
        / "data_analysis"
        / f"{dispatch.record.input_hash}.json"
    ).exists()
    audits = TaskRepository(settings.db_path).list_audit(
        target_ref=dispatch.record.id
    )
    assert [row["kind"] for row in audits] == [
        "data.analysis.started",
        "data.analysis.failed",
    ]


def test_succeeded_cache_survives_page_only_revision_without_new_job_or_audit(
    tmp_path,
):
    settings, task, dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    first = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    service.run_job(task_id=task.id, run_id=first.record.id, job_id=first.job_id)
    saved = DataWorkspaceRepository(settings.db_path).save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="fields",
            selected_field="phone",
            semantic_mapping=snapshot.semantic_mapping,
        ),
        expected_revision=snapshot.revision,
    )

    cached = service.request_analysis(
        task.id,
        expected_workspace_revision=saved.revision,
        request=_request(),
    )

    assert cached.http_status == 200
    assert cached.cached is True
    assert cached.should_execute is False
    assert cached.record.id == first.record.id
    assert cached.record.workspace_revision == snapshot.revision
    assert cached.result is not None
    with connect(settings.db_path) as conn:
        job_count = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE task_id = ? AND kind = 'data_analysis'",
            (task.id,),
        ).fetchone()[0]
    assert job_count == 1
    audits = TaskRepository(settings.db_path).list_audit(target_ref=first.record.id)
    assert [row["kind"] for row in audits] == [
        "data.analysis.started",
        "data.analysis.completed",
    ]


def test_execution_fails_closed_when_semantics_change_after_dispatch(tmp_path):
    settings, task, dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    DataWorkspaceRepository(settings.db_path).save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            page="semantics",
            selected_field="amount",
            semantic_mapping=DataSemanticMapping(
                target_col="bad",
                field_roles={"bad": "target", "amount": "loan_amount"},
                business_names={},
            ),
        ),
        expected_revision=snapshot.revision,
    )

    service.run_job(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )

    record = DataAnalysisRepository(settings.db_path).get_for_task(
        task.id,
        dispatch.record.id,
    )
    assert record is not None
    assert record.status == "failed"
    assert record.error_kind == "stale_data_analysis_identity"
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == []
    job = TaskRepository(settings.db_path).get_job(dispatch.job_id)
    assert job["status"] == "failed"
    audits = TaskRepository(settings.db_path).list_audit(target_ref=record.id)
    assert [row["kind"] for row in audits] == [
        "data.analysis.started",
        "data.analysis.failed",
    ]


def test_failed_run_requires_explicit_retry_and_reuses_stable_run_id(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    calls = 0

    def flaky_analyzer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic failure")
        from marvis.data.descriptive import analyze_parquet

        return analyze_parquet(*args, **kwargs)

    service = DataAnalysisService(settings, analyzer=flaky_analyzer)
    first = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    service.run_job(task_id=task.id, run_id=first.record.id, job_id=first.job_id)
    failed_view = service.get_run(task.id, first.record.id)
    assert failed_view is not None
    assert failed_view.record.error_kind == "data_analysis_failed"
    assert failed_view.record.error_message == "data analysis failed"
    with connect(settings.db_path) as conn:
        job_error = conn.execute(
            "SELECT error_value, traceback FROM jobs WHERE id = ?",
            (first.job_id,),
        ).fetchone()
    assert job_error["error_value"] == "data analysis failed"
    assert "synthetic failure" not in (job_error["traceback"] or "")

    with pytest.raises(DataAnalysisRetryRequiredError) as raised:
        service.request_analysis(
            task.id,
            expected_workspace_revision=snapshot.revision,
            request=_request(),
        )
    assert raised.value.record.status == "failed"

    retried = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(retry=True),
    )
    assert retried.record.id == first.record.id
    assert retried.job_id != first.job_id
    service.run_job(
        task_id=task.id,
        run_id=retried.record.id,
        job_id=retried.job_id,
    )
    assert service.get_run(task.id, retried.record.id).record.status == "succeeded"


def test_running_run_with_terminal_job_is_stranded_then_explicitly_retried(
    tmp_path,
):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    first = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    assert service.task_repo.mark_job_running(first.job_id) is True
    service.analysis_repo.mark_running(
        task_id=task.id,
        run_id=first.record.id,
        job_id=first.job_id,
    )
    service.task_repo.finish_job(
        first.job_id,
        status="failed",
        error_name="HeartbeatLost",
        error_value="heartbeat lost",
        traceback="",
    )

    with pytest.raises(DataAnalysisRetryRequiredError) as raised:
        service.request_analysis(
            task.id,
            expected_workspace_revision=snapshot.revision,
            request=_request(),
        )
    assert raised.value.record.status == "failed"
    assert raised.value.record.error_kind == "data_analysis_job_lost"

    retried = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(retry=True),
    )
    assert retried.record.id == first.record.id
    assert retried.job_id != first.job_id
    assert retried.should_execute is True
    audits = TaskRepository(settings.db_path).list_audit(
        target_ref=first.record.id
    )
    assert [row["kind"] for row in audits] == [
        "data.analysis.started",
        "data.analysis.failed",
        "data.analysis.started",
    ]


def test_dispatch_race_replays_winners_attached_active_job(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    winner_tasks = TaskRepository(settings.db_path)

    def winner_attaches_before_conflict(task_id: str, kind: str) -> str:
        job_id = winner_tasks.start_job(task_id, kind)
        run = service.analysis_repo.list_for_task(task_id)[0]
        service.analysis_repo.attach_job(
            task_id=task_id,
            run_id=run.id,
            job_id=job_id,
        )
        raise ConflictError("simulated losing start_job race")

    service.task_repo.start_job = winner_attaches_before_conflict
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )

    assert dispatch.http_status == 202
    assert dispatch.should_execute is False
    assert dispatch.record.status == "queued"
    assert dispatch.job_id is not None
    with connect(settings.db_path) as conn:
        jobs = conn.execute(
            "SELECT id, kind, status FROM jobs WHERE task_id = ?",
            (task.id,),
        ).fetchall()
    assert [dict(job) for job in jobs] == [
        {
            "id": dispatch.job_id,
            "kind": "data_analysis",
            "status": "queued",
        }
    ]


def test_duplicate_running_callback_is_a_noop(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)

    def must_not_execute(*_args, **_kwargs):
        raise AssertionError("duplicate callback must not execute analyzer")

    service = DataAnalysisService(settings, analyzer=must_not_execute)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    assert service.task_repo.mark_job_running(dispatch.job_id) is True
    running = service.analysis_repo.mark_running(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )

    service.run_job(
        task_id=task.id,
        run_id=running.id,
        job_id=dispatch.job_id,
    )

    assert service.get_run(task.id, running.id).record.status == "running"
    assert service.task_repo.get_job(dispatch.job_id)["status"] == "running"


def test_late_callback_from_prior_job_cannot_fail_retried_attempt(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    first = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    service.fail_dispatch(
        task_id=task.id,
        run_id=first.record.id,
        job_id=first.job_id,
        error_kind="background_registration_failed",
        error_message="background registration failed",
    )
    retried = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(retry=True),
    )

    service.run_job(
        task_id=task.id,
        run_id=retried.record.id,
        job_id=first.job_id,
    )

    current = service.get_run(task.id, retried.record.id).record
    assert current.status == "queued"
    assert current.job_id == retried.job_id
    assert service.task_repo.get_job(retried.job_id)["status"] == "queued"


def test_typed_chained_analyzer_failure_does_not_persist_sensitive_traceback(
    tmp_path,
):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    secret = "raw sanitizer value 13800138000 Alice"

    def chained_failure(*_args, **_kwargs):
        try:
            raise RuntimeError(secret)
        except RuntimeError as exc:
            raise DescriptiveInputError(
                "frequency value sanitizer failed for column 'phone'"
            ) from exc

    service = DataAnalysisService(settings, analyzer=chained_failure)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    service.run_job(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )

    view = service.get_run(task.id, dispatch.record.id)
    assert view is not None
    assert view.record.error_kind == "descriptive_input_error"
    assert secret not in (view.record.error_message or "")
    with connect(settings.db_path) as conn:
        job = conn.execute(
            "SELECT error_name, error_value, traceback FROM jobs WHERE id = ?",
            (dispatch.job_id,),
        ).fetchone()
    assert job["error_name"] == "descriptive_input_error"
    assert secret not in json.dumps(dict(job), ensure_ascii=False, sort_keys=True)


def test_get_run_conceals_cross_task_identity(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    other = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="other",
            model_version="v1",
            validator="owner",
            source_dir=str(settings.workspace),
        )
    )
    service = DataAnalysisService(settings)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )

    assert service.get_run(other.id, dispatch.record.id) is None


def test_cached_read_rejects_task_artifact_registry_hash_drift(tmp_path):
    settings, task, _dataset, snapshot = _seed_workspace(tmp_path)
    service = DataAnalysisService(settings)
    dispatch = service.request_analysis(
        task.id,
        expected_workspace_revision=snapshot.revision,
        request=_request(),
    )
    service.run_job(
        task_id=task.id,
        run_id=dispatch.record.id,
        job_id=dispatch.job_id,
    )
    view = service.get_run(task.id, dispatch.record.id)
    assert view is not None
    with connect(settings.db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            ("f" * 64, view.result_artifact_id),
        )

    with pytest.raises(DataAnalysisArtifactError, match="artifact registry drifted"):
        service.get_run(task.id, dispatch.record.id)
