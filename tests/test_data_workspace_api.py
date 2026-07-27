import io
import json

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.routers.data import router as data_router
from marvis.settings import build_settings
import marvis.data.registry as registry_module


def _app(settings) -> FastAPI:
    app = FastAPI()
    app.state.settings = settings
    app.include_router(data_router)
    return app


def _client(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    return TestClient(_app(settings)), settings


def _create_task(settings):
    return TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="strategy-workbench",
            model_version="v1",
            validator="validator",
            source_dir=str(settings.workspace),
            algorithm="",
            run_mode="manual",
            target_col="bad_flag",
            score_col="score",
            split_col="split",
            time_col="apply_month",
            feature_columns=[],
            notebook_path=None,
            sample_path=None,
            pmml_path=None,
            dictionary_path=None,
            report_values={},
        )
    )


def _upload_csv(client: TestClient, task_id: str, *, suffix: str = "") -> dict:
    response = client.post(
        f"/api/tasks/{task_id}/datasets/upload",
        data={"role": "sample"},
        files={
            "file": (
                f"sample{suffix}.csv",
                (
                    "mobile,bad_flag,loan_amount\n"
                    "13800138000,0,1000\n"
                    "13900139000,1,2000\n"
                ).encode(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 201
    return response.json()["datasets"][0]


def _upload_xlsx(client: TestClient, task_id: str) -> dict:
    buffer = io.BytesIO()
    pd.DataFrame(
        {
            "mobile": ["13800138000", "13900139000"],
            "bad_flag": [0, 1],
            "loan_amount": [1000, 2000],
        }
    ).to_excel(buffer, index=False)
    response = client.post(
        f"/api/tasks/{task_id}/datasets/upload",
        data={"role": "sample"},
        files={
            "file": (
                "sample.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert response.status_code == 201
    return response.json()["datasets"][0]


def _workspace_draft(
    dataset: dict | None,
    *,
    page: str = "overview",
    selected_field: str | None = None,
    target_col: str | None = None,
    field_roles: dict[str, str] | None = None,
    business_names: dict[str, str] | None = None,
) -> dict:
    return {
        "active_dataset_id": None if dataset is None else dataset["id"],
        "active_dataset_content_hash": (
            None if dataset is None else dataset["content_hash"]
        ),
        "page": page,
        "selected_field": selected_field,
        "semantic_mapping": {
            "target_col": target_col,
            "field_roles": field_roles or {},
            "business_names": business_names or {},
        },
    }


def _put_workspace(
    client: TestClient,
    task_id: str,
    draft: dict,
    *,
    revision: int | str | None,
):
    headers = {} if revision is None else {"If-Match": str(revision)}
    return client.put(
        f"/api/tasks/{task_id}/data-workspace",
        headers=headers,
        json=draft,
    )


def test_get_data_workspace_returns_stable_default_snapshot(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)

    first = client.get(f"/api/tasks/{task.id}/data-workspace")
    second = client.get(f"/api/tasks/{task.id}/data-workspace")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    payload = first.json()
    assert payload == {
        "schema_version": "data-workspace.v1",
        "task_id": task.id,
        "revision": 0,
        "active_dataset_id": None,
        "active_dataset_content_hash": None,
        "analysis_generation": 0,
        "page": "overview",
        "selected_field": None,
        "semantic_mapping": {
            "target_col": None,
            "field_roles": {},
            "business_names": {},
        },
        "updated_at": payload["updated_at"],
    }
    assert payload["updated_at"]
    assert client.get("/api/tasks/missing/data-workspace").status_code == 404


@pytest.mark.parametrize("source_kind", ["csv", "xlsx"])
def test_data_workspace_activates_import_and_saves_semantics_in_two_steps(
    tmp_path,
    source_kind,
):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    dataset = (
        _upload_csv(client, task.id)
        if source_kind == "csv"
        else _upload_xlsx(client, task.id)
    )

    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset),
        revision=0,
    )

    assert activated.status_code == 200
    assert activated.json()["revision"] == 1
    assert activated.json()["analysis_generation"] == 1
    assert activated.json()["active_dataset_content_hash"] == dataset["content_hash"]

    semantic_draft = _workspace_draft(
        dataset,
        page="semantics",
        selected_field="loan_amount",
        target_col="bad_flag",
        field_roles={"bad_flag": "target", "loan_amount": "loan_amount"},
        business_names={"loan_amount": "贷款金额"},
    )
    saved = _put_workspace(client, task.id, semantic_draft, revision=1)

    assert saved.status_code == 200
    payload = saved.json()
    assert payload["revision"] == 2
    assert payload["analysis_generation"] == 1
    assert payload["page"] == "semantics"
    assert payload["selected_field"] == "loan_amount"
    assert payload["semantic_mapping"] == semantic_draft["semantic_mapping"]


def test_data_workspace_recovers_saved_snapshot_after_app_restart(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset),
        revision=0,
    )
    assert activated.status_code == 200
    saved = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset, page="fields", selected_field="loan_amount"),
        revision=1,
    )
    assert saved.status_code == 200

    restarted_client = TestClient(_app(settings))
    recovered = restarted_client.get(f"/api/tasks/{task.id}/data-workspace")

    assert recovered.status_code == 200
    assert recovered.json() == saved.json()


def test_data_workspace_requires_valid_if_match_and_rejects_stale_revision(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    draft = _workspace_draft(None, page="fields")

    missing = _put_workspace(client, task.id, draft, revision=None)
    malformed = _put_workspace(client, task.id, draft, revision="not-an-int")
    negative = _put_workspace(client, task.id, draft, revision=-1)
    first = _put_workspace(client, task.id, draft, revision=0)
    stale = _put_workspace(client, task.id, draft, revision=0)

    assert missing.status_code == 428
    assert missing.json()["detail"] == "If-Match header is required"
    assert malformed.status_code == 400
    assert malformed.json()["detail"] == "If-Match must be a non-negative integer"
    assert negative.status_code == 400
    assert negative.json()["detail"] == "If-Match must be a non-negative integer"
    assert first.status_code == 200
    assert stale.status_code == 412
    assert "stale data workspace revision" in stale.json()["detail"]


def test_data_workspace_request_is_strict_complete_replacement(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    draft = _workspace_draft(None)

    missing_field = dict(draft)
    missing_field.pop("selected_field")
    unexpected_field = {**draft, "revision": 0}
    invalid_page = {**draft, "page": "unknown"}
    coerced_id = {**draft, "active_dataset_id": 123}
    padded_selection = {**draft, "selected_field": " loan_amount "}

    for payload in (
        missing_field,
        unexpected_field,
        invalid_page,
        coerced_id,
        padded_selection,
    ):
        response = _put_workspace(client, task.id, payload, revision=0)
        assert response.status_code == 422


def test_data_workspace_rejects_cross_task_dataset_and_hash_mismatch(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    other_task = _create_task(settings)
    dataset = _upload_csv(client, task.id)

    cross_task = _put_workspace(
        client,
        other_task.id,
        _workspace_draft(dataset),
        revision=0,
    )
    mismatched_hash_draft = _workspace_draft(dataset)
    mismatched_hash_draft["active_dataset_content_hash"] = "f" * 64
    hash_mismatch = _put_workspace(
        client,
        task.id,
        mismatched_hash_draft,
        revision=0,
    )
    invalid_hash_draft = _workspace_draft(dataset)
    invalid_hash_draft["active_dataset_content_hash"] = "not-a-sha256"
    invalid_hash = _put_workspace(
        client,
        task.id,
        invalid_hash_draft,
        revision=0,
    )

    assert cross_task.status_code == 404
    assert cross_task.json()["detail"] == "dataset not found"
    assert hash_mismatch.status_code == 422
    assert "hash" in hash_mismatch.json()["detail"]
    assert invalid_hash.status_code == 422


def test_data_workspace_requires_explicit_reset_when_active_dataset_changes(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    first_dataset = _upload_csv(client, task.id, suffix="_one")
    second_dataset = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={
            "file": (
                "sample_two.csv",
                b"customer_id,bad_flag,loan_amount\n1,0,3000\n2,1,4000\n",
                "text/csv",
            )
        },
    ).json()["datasets"][0]
    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(first_dataset),
        revision=0,
    )
    assert activated.status_code == 200
    semantic_draft = _workspace_draft(
        first_dataset,
        page="semantics",
        selected_field="loan_amount",
        target_col="bad_flag",
        field_roles={"bad_flag": "target"},
    )
    saved = _put_workspace(client, task.id, semantic_draft, revision=1)
    assert saved.status_code == 200

    unsafe_switch = _workspace_draft(
        second_dataset,
        page="semantics",
        selected_field="loan_amount",
        target_col="bad_flag",
        field_roles={"bad_flag": "target"},
    )
    rejected = _put_workspace(client, task.id, unsafe_switch, revision=2)

    assert rejected.status_code == 422
    assert "reset" in rejected.json()["detail"]

    reset = _put_workspace(
        client,
        task.id,
        _workspace_draft(second_dataset),
        revision=2,
    )
    assert reset.status_code == 200
    assert reset.json()["analysis_generation"] == 2
    assert reset.json()["selected_field"] is None
    assert reset.json()["semantic_mapping"]["field_roles"] == {}


def test_data_workspace_rejects_unknown_semantic_columns(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset),
        revision=0,
    )
    assert activated.status_code == 200
    draft = _workspace_draft(
        dataset,
        page="semantics",
        selected_field="missing_column",
        target_col="missing_column",
        field_roles={"missing_column": "target"},
        business_names={"also_missing": "不存在"},
    )

    response = _put_workspace(client, task.id, draft, revision=1)

    assert response.status_code == 422
    assert "column" in response.json()["detail"]


def test_task_owned_dataset_preview_masks_rows_and_enforces_ownership(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    other_task = _create_task(settings)
    dataset = _upload_csv(client, task.id)

    preview = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/preview?rows=1"
    )
    legacy = client.get(f"/api/datasets/{dataset['id']}/preview?rows=1")
    too_many = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/preview?rows=501"
    )
    cross_task = client.get(
        f"/api/tasks/{other_task.id}/datasets/{dataset['id']}/preview?rows=1"
    )
    download = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/download"
    )
    cross_task_download = client.get(
        f"/api/tasks/{other_task.id}/datasets/{dataset['id']}/download"
    )
    unscoped_download = client.get(f"/api/datasets/{dataset['id']}/download")

    assert preview.status_code == 200
    assert preview.json() == legacy.json()
    assert preview.json()["truncated"] is True
    assert preview.json()["rows"][0]["mobile"] == "138******00"
    assert "13800138000" not in json.dumps(preview.json(), ensure_ascii=False)
    assert too_many.status_code == 422
    assert cross_task.status_code == 404
    assert cross_task.json()["detail"] == "dataset not found"
    assert download.status_code == 200
    assert download.content
    assert cross_task_download.status_code == 404
    assert cross_task_download.json()["detail"] == "dataset not found"
    assert unscoped_download.status_code == 404


def test_workspace_and_preview_fail_closed_after_registered_file_drift(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset),
        revision=0,
    )
    assert activated.status_code == 200

    stored = DatasetRepository(settings.db_path).get_dataset(dataset["id"])
    assert stored is not None
    dataset_path = settings.datasets_dir / stored.source_path
    dataset_path.write_bytes(b"out-of-band mutation")

    workspace = client.get(f"/api/tasks/{task.id}/data-workspace")
    update = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset, page="fields"),
        revision=1,
    )
    task_preview = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/preview?rows=1"
    )
    legacy_preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=1")
    task_download = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/download"
    )

    for response in (workspace, update, task_preview, legacy_preview, task_download):
        assert response.status_code == 409
        assert "failed integrity verification" in response.json()["detail"]


def test_unchanged_dataset_integrity_is_hashed_once_across_workspace_requests(
    tmp_path,
    monkeypatch,
):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    dataset = _upload_csv(client, task.id)
    real_sha256_file = registry_module.sha256_file
    hash_calls = 0

    def counted_sha256_file(path):
        nonlocal hash_calls
        hash_calls += 1
        return real_sha256_file(path)

    monkeypatch.setattr(registry_module, "sha256_file", counted_sha256_file)

    activated = _put_workspace(
        client,
        task.id,
        _workspace_draft(dataset),
        revision=0,
    )
    workspace = client.get(f"/api/tasks/{task.id}/data-workspace")
    first_preview = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/preview?rows=1"
    )
    second_preview = client.get(
        f"/api/tasks/{task.id}/datasets/{dataset['id']}/preview?rows=1"
    )

    assert activated.status_code == 200
    assert workspace.status_code == 200
    assert first_preview.status_code == 200
    assert second_preview.status_code == 200
    assert hash_calls == 1
