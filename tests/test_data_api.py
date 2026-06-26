import hashlib
import json

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.api import router
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.settings import build_settings


class FakeHookDispatcher:
    def __init__(self):
        self.calls = []

    def dispatch(self, event, payload, *, task_id):
        self.calls.append((event, payload, task_id))
        return []


def _client(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    app = FastAPI()
    app.state.settings = settings
    app.include_router(router)
    return TestClient(app), settings


def _create_task(settings):
    return TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="A-card",
            model_version="v1",
            validator="validator",
            source_dir=str(settings.workspace),
            algorithm="lgb",
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


def _registry(settings):
    repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    return DatasetRegistry(repo, backend, settings.datasets_dir)


def _register_csv(settings, tmp_path, task_id: str, name: str, frame: pd.DataFrame, role: str):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return _registry(settings).register_from_upload(task_id, path, role=role)


def test_dataset_upload_list_and_preview_api(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n13900139000,1\n"

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )

    assert upload.status_code == 201
    dataset = upload.json()["datasets"][0]
    assert dataset["role"] == "sample"
    assert dataset["has_target"] is True
    assert dataset["target_col"] == "bad_flag"

    listed = client.get(f"/api/tasks/{task.id}/datasets")
    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=1")
    invalid_preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=9999")

    assert listed.status_code == 200
    assert listed.json()["datasets"][0]["id"] == dataset["id"]
    assert preview.status_code == 200
    assert preview.json()["columns"] == ["mobile", "bad_flag"]
    assert preview.json()["truncated"] is True
    assert invalid_preview.status_code == 422


def test_dataset_preview_returns_column_profiles_and_masked_samples(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = b"mobile,bad_flag\n13800138000,0\n,1\n"

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    dataset = upload.json()["datasets"][0]

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=2")

    assert preview.status_code == 200
    payload = preview.json()
    profiles = {profile["name"]: profile for profile in payload["column_profiles"]}
    assert profiles["mobile"]["semantic_role"] == "phone"
    assert profiles["mobile"]["null_rate"] == 0.5
    assert profiles["mobile"]["sample_values"] == ["138******00"]
    assert payload["rows"][0]["mobile"] == "138******00"
    assert payload["rows"][1]["mobile"] is None
    assert "13800138000" not in json.dumps(payload, ensure_ascii=False)


def test_dataset_preview_masks_names_and_long_card_values(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    csv_bytes = (
        "customer_name,bank_card,bad_flag\n"
        "张三丰,6222020202020202020,0\n"
        "李四,6222020202020202021,1\n"
    ).encode()

    upload = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    dataset = upload.json()["datasets"][0]

    preview = client.get(f"/api/datasets/{dataset['id']}/preview?rows=2")

    assert preview.status_code == 200
    payload = preview.json()
    dumped = json.dumps(payload, ensure_ascii=False)
    profiles = {profile["name"]: profile for profile in payload["column_profiles"]}
    assert profiles["customer_name"]["semantic_role"] == "categorical"
    assert profiles["customer_name"]["sample_values"][0].startswith("value:")
    assert payload["rows"][0]["customer_name"].startswith("value:")
    assert payload["rows"][0]["bank_card"].startswith("6222")
    assert "*" in payload["rows"][0]["bank_card"]
    assert "张三丰" not in dumped
    assert "李四" not in dumped
    assert "6222020202020202020" not in dumped


def test_dataset_upload_dispatches_dataset_registered_hook(tmp_path):
    client, settings = _client(tmp_path)
    dispatcher = FakeHookDispatcher()
    client.app.state.hook_dispatcher = dispatcher
    task = _create_task(settings)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "sample"},
        files={"file": ("sample.csv", b"mobile,bad_flag\n13800138000,0\n", "text/csv")},
    )

    assert response.status_code == 201
    dataset = response.json()["datasets"][0]
    assert dispatcher.calls == [
        (
            "dataset.registered",
            {
                "task_id": task.id,
                "dataset_id": dataset["id"],
                "role": "sample",
            },
            task.id,
        )
    ]


def test_dataset_upload_rejects_invalid_excel_sheet(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    workbook_path = tmp_path / "book.xlsx"
    pd.DataFrame({"id": [1]}).to_excel(workbook_path, sheet_name="Present", index=False)

    response = client.post(
        f"/api/tasks/{task.id}/datasets/upload",
        data={"role": "feature", "sheet": "Missing"},
        files={
            "file": (
                "book.xlsx",
                workbook_path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 422


def test_join_api_propose_confirm_execute_flow(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )

    propose = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    )

    assert propose.status_code == 201
    plan = propose.json()
    assert plan["status"] == "draft"
    assert plan["anchor_dataset_id"] == anchor.id
    assert plan["joins"][0]["key_pairs"][0]["match_method"] == "hash:md5"

    blocked = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    assert blocked.status_code == 409

    confirm = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )
    fetched = client.get(f"/api/joins/{plan['join_plan_id']}")
    execute = client.post(f"/api/joins/{plan['join_plan_id']}/execute")
    repeat = client.post(f"/api/joins/{plan['join_plan_id']}/execute")

    assert confirm.status_code == 200
    assert confirm.json()["anchor_dataset_id"] == anchor.id
    assert confirm.json()["joins"][0]["confirmed"] is True
    assert fetched.status_code == 200
    assert fetched.json()["anchor_dataset_id"] == anchor.id
    assert execute.status_code == 200
    assert execute.json()["anchor_rows"] == 2
    assert execute.json()["joined_rows"] == 2
    assert repeat.status_code == 409
    audits = PluginRepository(settings.db_path).list_audit()
    audit_kinds = [audit["kind"] for audit in audits]
    assert "join.confirmed" in audit_kinds
    assert "join.executed" in audit_kinds
    executed_audit = next(audit for audit in audits if audit["kind"] == "join.executed")
    assert executed_audit["target_ref"] == plan["join_plan_id"]
    assert executed_audit["detail"]["result_dataset_id"] == execute.json()["result_dataset_id"]


def test_join_confirm_dispatches_join_confirmed_hook(tmp_path):
    client, settings = _client(tmp_path)
    dispatcher = FakeHookDispatcher()
    client.app.state.hook_dispatcher = dispatcher
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )

    assert response.status_code == 200
    assert dispatcher.calls == [
        (
            "join.confirmed",
            {
                "task_id": task.id,
                "join_plan_id": plan["join_plan_id"],
                "feature_id": feature.id,
                "confirmed": True,
            },
            task.id,
        )
    ]


def test_join_confirm_accepts_feature_dataset_id_alias(tmp_path):
    client, settings = _client(tmp_path)
    task = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task.id,
        "anchor",
        pd.DataFrame({"mobile": ["13800138000", "13900139000"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task.id,
        "feature",
        pd.DataFrame({
            "phone_md5": [
                hashlib.md5(value.encode()).hexdigest()
                for value in ["13800138000", "13900139000"]
            ],
            "balance": [10, 20],
        }),
        "feature",
    )
    plan = client.post(
        f"/api/tasks/{task.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    ).json()

    response = client.post(
        f"/api/joins/{plan['join_plan_id']}/confirm",
        json={"feature_dataset_id": feature.id, "confirmed": True, "dedup_strategy": None},
    )

    assert response.status_code == 200
    assert response.json()["joins"][0]["confirmed"] is True


def test_join_api_keeps_task_dataset_boundaries(tmp_path):
    client, settings = _client(tmp_path)
    task_1 = _create_task(settings)
    task_2 = _create_task(settings)
    anchor = _register_csv(
        settings,
        tmp_path,
        task_1.id,
        "anchor",
        pd.DataFrame({"acct_num": ["A1", "B2"]}),
        "sample",
    )
    feature = _register_csv(
        settings,
        tmp_path,
        task_2.id,
        "feature",
        pd.DataFrame({"acct_no": ["A1", "B2"]}),
        "feature",
    )

    response = client.post(
        f"/api/tasks/{task_1.id}/joins/propose",
        json={
            "anchor_dataset_id": anchor.id,
            "feature_dataset_ids": [feature.id],
        },
    )

    assert response.status_code == 404
