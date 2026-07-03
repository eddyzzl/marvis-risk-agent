"""Tests for the built-in sample-data generator (UX-9) and its wiring into the
one-click "用示例数据试跑" first-run entry.

Covers: deterministic generation (marvis.sample_data), the /api/sample-data
endpoint's material-upload-shaped response, and an end-to-end flow that creates
a manual modeling task from the generated materials and confirms the data
dictionary (GAP-4) is picked up by the screen gate — the two review items are
meant to be exercised together by this entry, so one test dogfoods both.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.sample_data import (
    DEMO_TASK_NAME_PREFIX,
    DICTIONARY_TABLE_NAME,
    SAMPLE_TABLE_NAME,
    generate_dictionary_frame,
    generate_sample_frame,
    write_sample_materials,
)


def test_generate_sample_frame_is_deterministic():
    first = generate_sample_frame()
    second = generate_sample_frame()
    pd.testing.assert_frame_equal(first, second)


def test_generate_sample_frame_shape_and_columns():
    frame = generate_sample_frame()
    assert len(frame) == 1500
    assert "apply_month" in frame.columns
    assert "y" in frame.columns
    assert set(frame["y"].unique()) <= {0, 1}
    feature_cols = [c for c in frame.columns if c not in {"apply_month", "y"}]
    assert len(feature_cols) == 6
    # Realistic-ish bad rate: a demo sample that's ~50/50 doesn't read as credible
    # to a credit-risk reviewer, and it also isn't exercising a leakage/imbalance
    # screen the way a skewed real portfolio would.
    bad_rate = frame["y"].mean()
    assert 0.05 < bad_rate < 0.35


def test_generate_dictionary_frame_covers_every_sample_column():
    sample = generate_sample_frame()
    dictionary = generate_dictionary_frame()
    assert list(dictionary.columns) == ["特征名", "含义"]
    named = set(dictionary["特征名"])
    assert named == set(sample.columns)
    # every meaning is non-empty text
    assert all(str(value).strip() for value in dictionary["含义"])


def test_write_sample_materials_writes_expected_files(tmp_path: Path):
    target = tmp_path / "materials"
    result = write_sample_materials(target)
    assert result == target
    assert (target / SAMPLE_TABLE_NAME).exists()
    assert (target / DICTIONARY_TABLE_NAME).exists()
    # The dictionary filename must trip marvis.files.classify_file's dictionary
    # detection (keyword "字典") so it gets registered as a data dictionary, not a
    # second sample table.
    assert "字典" in DICTIONARY_TABLE_NAME


def test_write_sample_materials_is_repeatable(tmp_path: Path):
    target = tmp_path / "materials"
    write_sample_materials(target)
    first = (target / SAMPLE_TABLE_NAME).read_bytes()
    write_sample_materials(target)
    second = (target / SAMPLE_TABLE_NAME).read_bytes()
    assert first == second


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def test_sample_data_endpoint_returns_material_upload_shape(client: TestClient):
    resp = client.post("/api/sample-data")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "source_dir" in body
    assert Path(body["source_dir"]).exists()
    names = {item["relative_path"] for item in body["files"]}
    assert names == {SAMPLE_TABLE_NAME, DICTIONARY_TABLE_NAME}
    for item in body["files"]:
        assert item["size_bytes"] > 0


def test_sample_data_task_creation_and_dictionary_reach_screen_gate(client: TestClient):
    """End-to-end UX-9 + GAP-4 dogfood: generate sample data, create a manual
    modeling task from it exactly like the welcome-page entry does, and confirm
    the screen gate shows the data-dictionary business-name map."""
    upload = client.post("/api/sample-data").json()
    create_resp = client.post(
        "/api/tasks",
        json={
            "task_type": "modeling",
            "model_name": f"{DEMO_TASK_NAME_PREFIX}演示建模",
            "model_version": "",
            "validator": "演示",
            "source_dir": upload["source_dir"],
            "run_mode": "manual",
            "report_values": {},
        },
    )
    assert create_resp.status_code == 200, create_resp.text
    task = create_resp.json()
    assert task["model_name"].startswith(DEMO_TASK_NAME_PREFIX)
    task_id = task["id"]

    start_resp = client.post(f"/api/tasks/{task_id}/agent/start", json={})
    assert start_resp.status_code == 202, start_resp.text

    begin_resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})
    assert begin_resp.status_code == 202, begin_resp.text
    confirm_resp = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    assert confirm_resp.status_code == 202, confirm_resp.text

    messages = client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"]
    gate = messages[-1]
    screen = gate.get("metadata", {}).get("screen")
    assert screen is not None, "expected the screen gate to have been reached"
    dictionary = screen.get("dictionary") or {}
    # Every generated feature/label/time column has a dictionary entry (GAP-4).
    assert dictionary.get("y")
    assert dictionary.get("apply_month")
    assert any(dictionary.get(col) for col in screen.get("selected", []))

    datasets = client.get(f"/api/tasks/{task_id}/datasets").json()["datasets"]
    roles = {d["role"] for d in datasets}
    assert "feature_dictionary" in roles
