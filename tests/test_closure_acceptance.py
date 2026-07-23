from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pytest
from openpyxl import Workbook

import scripts.closure_acceptance as closure_acceptance
from scripts.closure_acceptance import inspect, render_markdown
from scripts.closure_test_evidence import REQUIRED_TESTS, build_command


_TEST_COMMIT_SHA = "a" * 40
_TOOL_NAMES = {
    "切分样本": "make_split",
    "特征筛选": "screen_features",
    "精选特征": "select_features",
    "训练模型": "train_models",
    "选择实验": "select_experiment",
    "生成模型开发报告": "generate_model_reports",
    "模型交付动作": "post_training_action",
}


class _DictBackedModelFixture(dict):
    """Exercise estimator payloads that persist a custom mapping subclass."""


@pytest.fixture(autouse=True)
def _stable_repo_state(monkeypatch):
    monkeypatch.setattr(
        closure_acceptance,
        "_current_repo_state",
        lambda: (_TEST_COMMIT_SHA, True),
    )


def _inspect(workspace: Path, **kwargs):
    return inspect(workspace, "task-1", **kwargs)


def _check(result: dict, check_id: str) -> dict:
    return next(item for item in result["checks"] if item["check_id"] == check_id)


def _mutate_step_evidence(workspace: Path, title: str, mutate) -> None:
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        row = connection.execute(
            """
            SELECT o.step_id, o.evidence_json
              FROM plan_step_outputs o
              JOIN plan_steps s ON s.id = o.step_id
             WHERE s.title = ?
            """,
            (title,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[1])
        mutate(payload)
        connection.execute(
            "UPDATE plan_step_outputs SET evidence_json = ? WHERE step_id = ?",
            (json.dumps(payload), row[0]),
        )


def _mutate_test_evidence(workspace: Path, mutate) -> None:
    path = workspace / "closure_test_evidence.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mutate_step_output(workspace: Path, title: str, mutate) -> None:
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        row = connection.execute(
            """
            SELECT o.step_id, o.output_json
              FROM plan_step_outputs o
              JOIN plan_steps s ON s.id = o.step_id
             WHERE s.title = ?
            """,
            (title,),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[1])
        mutate(payload)
        connection.execute(
            "UPDATE plan_step_outputs SET output_json = ? WHERE step_id = ?",
            (json.dumps(payload), row[0]),
        )


def _seed_workspace(
    root: Path,
    *,
    sentinel: bool,
    traceable: bool,
    selected: list[str] | None = None,
    sentinel_columns: dict | None = None,
    preprocessing_steps: list[dict] | None = None,
    special_value_governance: dict | None = None,
    recipe: str = "lr",
    include_test_evidence: bool = True,
) -> Path:
    workspace = root / "workspace"
    workspace.mkdir()
    connection = sqlite3.connect(workspace / "marvis.sqlite")
    connection.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, task_type TEXT, model_name TEXT, model_version TEXT,
            target_col TEXT, time_col TEXT
        );
        CREATE TABLE plans (
            id TEXT PRIMARY KEY, task_id TEXT, template_id TEXT, status TEXT,
            updated_at TEXT
        );
        CREATE TABLE plan_steps (
            id TEXT PRIMARY KEY, plan_id TEXT, idx INTEGER, title TEXT,
            tool_plugin TEXT, tool_name TEXT, status TEXT, output_ref TEXT
        );
        CREATE TABLE plan_step_outputs (
            step_id TEXT PRIMARY KEY, output_json TEXT, evidence_json TEXT
        );
        CREATE TABLE experiments (
            id TEXT PRIMARY KEY, task_id TEXT, recipe_id TEXT
        );
        CREATE TABLE model_artifacts (
            id TEXT PRIMARY KEY, params_json TEXT, feature_list_json TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO tasks VALUES (?,?,?,?,?,?)",
        ("task-1", "modeling", "closure", "v1", "y", "apply_month"),
    )
    connection.execute(
        "INSERT INTO plans VALUES (?,?,?,?,?)",
        ("plan-1", "task-1", "modeling", "done", "2026-07-24T00:00:00Z"),
    )

    outputs_dir = workspace / "tasks" / "task-1" / "outputs"
    outputs_dir.mkdir(parents=True)
    report = outputs_dir / "report.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "closure report"
    workbook.save(report)
    model_dir = workspace / "tasks" / "task-1" / "modeling_artifacts"
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(pickle.dumps({"model": "closure fixture"}))
    (model_dir / "model.pmml").write_text(
        """
        <PMML version="4.4" xmlns="http://www.dmg.org/PMML-4_4">
          <DataDictionary numberOfFields="1">
            <DataField name="x1" optype="continuous" dataType="double"/>
          </DataDictionary>
          <RegressionModel modelName="fixture" functionName="classification">
            <MiningSchema>
              <MiningField name="x1" usageType="active"/>
            </MiningSchema>
          </RegressionModel>
        </PMML>
        """,
        encoding="utf-8",
    )
    card = model_dir / "card.json"
    card.write_text(
        json.dumps({"artifact_id": "artifact-1", "recipe": recipe}),
        encoding="utf-8",
    )

    base_evidence = {
        "step_run_id": "run",
        "manifest_hash": "sha256:" + "1" * 64,
        "input_hash": "sha256:" + "2" * 64,
        "source_dataset_refs": ["dataset:ds-1"],
        "parent_output_refs": ["ref:parent"],
        "input_summary": {},
        "random_seed": 23,
    }
    outputs = {
        "切分样本": {
            "split_col": "split",
            "holdout_values": ["oot"],
            "sample_analysis": {"split_counts": {"train": 60, "test": 20, "oot": 20}},
        },
        "特征筛选": {
            "sentinel_columns": (
                sentinel_columns
                if sentinel_columns is not None
                else ({"x1": [[-999, 0.1]]} if sentinel else {})
            ),
            "nan_labels_dropped": 0,
        },
        "精选特征": {
            "fit_split": "train",
            "fit_rows": 60,
            "selected": selected if selected is not None else ["x1"],
        },
        "训练模型": {"selection_metric": "test_ks(overfit-penalized)"},
        "选择实验": {
            "artifact_id": "artifact-1",
            "selection_reason": "selected by policy",
            "metrics": {"train_ks": 0.5, "test_ks": 0.4, "oot_ks": 0.3},
            "refit": {"enabled": False},
        },
        "生成模型开发报告": {"report_path": str(report)},
        "模型交付动作": {
            "artifact_id": "artifact-1",
            "native_model_path": "model.pkl",
            "pmml_path": str(model_dir / "model.pmml"),
            "model_card_path": str(card),
            "model_card": {
                "recipe": recipe,
                "key_metrics": [
                    {"metric": "train_ks", "value": 0.5},
                    {"metric": "test_ks", "value": 0.4},
                    {"metric": "oot_ks", "value": 0.3},
                ],
                "limitations": [],
            },
            "monitoring_policy": {"status": "pass", "recommendation": "ready"},
        },
    }
    for idx, (title, output) in enumerate(outputs.items()):
        step_id = f"step-{idx}"
        tool_name = _TOOL_NAMES[title]
        evidence = {
            **base_evidence,
            "step_run_id": f"run-{idx}",
            "input_summary": (
                {"seed": 23}
                if tool_name
                in {
                    "make_split",
                    "select_features",
                    "train_models",
                }
                else {}
            ),
        }
        connection.execute(
            "INSERT INTO plan_steps VALUES (?,?,?,?,?,?,?,?)",
            (
                step_id,
                "plan-1",
                idx,
                title,
                "modeling",
                tool_name,
                "done",
                f"ref:{idx}",
            ),
        )
        connection.execute(
            "INSERT INTO plan_step_outputs VALUES (?,?,?)",
            (
                step_id,
                json.dumps(output),
                json.dumps(evidence),
            ),
        )
    connection.execute(
        "INSERT INTO model_artifacts VALUES (?,?,?)",
        (
            "artifact-1",
            json.dumps(
                {
                    "preprocessing_chain_traceable": traceable,
                    "preprocessing_steps": preprocessing_steps or [],
                    "special_value_governance": special_value_governance or {},
                }
            ),
            json.dumps(selected if selected is not None else ["x1"]),
        ),
    )
    connection.commit()
    connection.close()

    if include_test_evidence:
        log_path = workspace / "closure-regression.log"
        log_path.write_text("66 passed in 2.00s\n", encoding="utf-8")
        log_hash = "sha256:" + hashlib.sha256(log_path.read_bytes()).hexdigest()
        (workspace / "closure_test_evidence.json").write_text(
            json.dumps(
                {
                    "schema_version": "closure-test-evidence.v1",
                    "command": (
                        "python -m pytest -q "
                        "tests/test_dirty_shape_regression.py "
                        "tests/test_reconcile_reference_numbers.py"
                    ),
                    "exit_code": 0,
                    "commit_sha": _TEST_COMMIT_SHA,
                    "commit_sha_after": _TEST_COMMIT_SHA,
                    "timestamp": "2026-07-24T00:00:00Z",
                    "log_path": log_path.name,
                    "log_hash": log_hash,
                    "worktree_clean": True,
                }
            ),
            encoding="utf-8",
        )
    return workspace


def _seed_multi_reports(workspace: Path) -> list[dict]:
    outputs_dir = workspace / "tasks" / "task-1" / "outputs"
    reports: list[dict] = []
    for experiment_id, recipe in (
        ("experiment-first", "lr"),
        ("experiment-second", "xgb"),
    ):
        digest = hashlib.sha256(experiment_id.encode("utf-8")).hexdigest()[:12]
        report_path = outputs_dir / f"model_report_{recipe}_{digest}.xlsx"
        workbook = Workbook()
        workbook.active["A1"] = experiment_id
        workbook.save(report_path)
        reports.append(
            {
                "experiment_id": experiment_id,
                "recipe": recipe,
                "report_path": str(report_path),
            }
        )
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        connection.executemany(
            "INSERT INTO experiments VALUES (?,?,?)",
            [
                ("experiment-first", "task-1", "lr"),
                ("experiment-second", "task-1", "xgb"),
            ],
        )
    _mutate_step_output(
        workspace,
        "生成模型开发报告",
        lambda payload: payload.update(
            {"report_path": reports[0]["report_path"], "reports": reports}
        ),
    )
    return reports


def test_machine_precheck_separates_passes_from_manual_external_reconciliation(
    tmp_path,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    result = _inspect(workspace)
    statuses = {item["check_id"]: item["status"] for item in result["checks"]}
    assert result["machine_verdict"] == "PASS"
    assert result["closure_verdict"] == "BLOCKED_MANUAL"
    assert statuses["D1"] == "PASS"
    assert statuses["B1"] == statuses["B2"] == statuses["B3"] == "MANUAL"
    assert statuses["B4-INTERNAL"] == "PASS"
    assert statuses["B4-EXTERNAL"] == "MANUAL"
    assert "cannot and will not fabricate" in result["manual_blocker"]
    assert "真实材料机器预检报告" in render_markdown(result)


def test_machine_precheck_blocks_untraceable_sentinel_preprocessing(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=True, traceable=False)
    result = _inspect(workspace)
    assert result["machine_verdict"] == "FAIL"
    assert result["closure_verdict"] == "BLOCKED_MACHINE"
    assert "A1" in result["machine_failures"]


def test_machine_precheck_rejects_traceable_flag_without_real_sentinel_step(
    tmp_path,
):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=True,
        special_value_governance={
            "x1": {
                "action": "mask",
                "detected_values": [-999],
                "fingerprint": "sha256:decision",
            }
        },
    )
    result = _inspect(workspace)
    assert result["machine_verdict"] == "FAIL"
    assert "A1" in result["machine_failures"]


def test_machine_precheck_accepts_exact_mask_step_for_selected_column(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=True,
        preprocessing_steps=[
            {
                "kind": "sentinel",
                "columns": ["x1"],
                "params": {"x1": [-999]},
            }
        ],
        special_value_governance={
            "x1": {
                "action": "mask",
                "detected_values": [-999],
                "fingerprint": "sha256:decision",
            }
        },
    )
    result = _inspect(workspace)
    statuses = {item["check_id"]: item["status"] for item in result["checks"]}
    assert statuses["A1"] == "PASS"
    assert result["machine_verdict"] == "PASS"


def test_machine_precheck_rejects_mask_with_different_step_values(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=True,
        preprocessing_steps=[
            {
                "kind": "sentinel",
                "columns": ["x1"],
                "params": {"x1": [-998]},
            }
        ],
        special_value_governance={
            "x1": {
                "action": "mask",
                "detected_values": [-999],
                "fingerprint": "sha256:decision",
            }
        },
    )
    result = _inspect(workspace)
    assert "A1" in result["machine_failures"]


def test_machine_precheck_accepts_explicit_retain_with_fingerprint(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=False,
        special_value_governance={
            "x1": {
                "action": "retain",
                "detected_values": [-999],
                "confirmed": True,
                "reason": "业务约定的有效状态码",
                "source_dataset_id": "dataset-1",
                "source_dataset_content_hash": "sha256:dataset",
                "fingerprint": "sha256:decision",
            }
        },
    )
    result = _inspect(workspace)
    statuses = {item["check_id"]: item["status"] for item in result["checks"]}
    assert statuses["A1"] == "PASS"


def test_machine_precheck_rejects_unconfirmed_retain(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=False,
        special_value_governance={
            "x1": {
                "action": "retain",
                "detected_values": [-999],
                "confirmed": False,
                "reason": "业务约定的有效状态码",
                "source_dataset_id": "dataset-1",
                "source_dataset_content_hash": "sha256:dataset",
                "fingerprint": "sha256:decision",
            }
        },
    )
    result = _inspect(workspace)
    assert "A1" in result["machine_failures"]


@pytest.mark.parametrize(
    ("governance_patch", "expected_reason"),
    [
        ({"reason": ""}, "retain missing reason"),
        (
            {"source_dataset_content_hash": ""},
            "retain missing source dataset fingerprint",
        ),
    ],
)
def test_machine_precheck_rejects_incomplete_retain_evidence(
    tmp_path,
    governance_patch,
    expected_reason,
):
    evidence = {
        "action": "retain",
        "detected_values": [-999],
        "confirmed": True,
        "reason": "业务约定的有效状态码",
        "source_dataset_id": "dataset-1",
        "source_dataset_content_hash": "sha256:dataset",
        "fingerprint": "sha256:decision",
        **governance_patch,
    }
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=False,
        special_value_governance={"x1": evidence},
    )
    result = _inspect(workspace)
    a1 = next(item for item in result["checks"] if item["check_id"] == "A1")
    assert a1["status"] == "FAIL"
    assert expected_reason in a1["evidence"]


def test_machine_precheck_ignores_detected_column_dropped_before_training(
    tmp_path,
):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=False,
        selected=["x2"],
    )
    result = _inspect(workspace)
    statuses = {item["check_id"]: item["status"] for item in result["checks"]}
    assert statuses["A1"] == "PASS"


def test_machine_precheck_checks_every_selected_detected_column(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=True,
        traceable=True,
        selected=["x1", "x2"],
        sentinel_columns={
            "x1": [[-999, 0.1]],
            "x2": [[9999, 0.1]],
        },
        preprocessing_steps=[
            {
                "kind": "sentinel",
                "columns": ["x1"],
                "params": {"x1": [-999]},
            }
        ],
        special_value_governance={
            "x1": {
                "action": "mask",
                "detected_values": [-999],
                "fingerprint": "sha256:x1",
            }
        },
    )
    result = _inspect(workspace)
    assert "A1" in result["machine_failures"]


def test_machine_precheck_requires_completed_plan(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        connection.execute("UPDATE plans SET status='failed'")
    with pytest.raises(LookupError, match="not 'done'"):
        inspect(workspace, "task-1")


def test_c2_fails_closed_when_no_executed_tool_steps_exist(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        connection.execute("DELETE FROM plan_step_outputs")
        connection.execute("DELETE FROM plan_steps")

    result = _inspect(workspace)
    c2 = _check(result, "C2")
    assert c2["status"] == "FAIL"
    assert "required_executed_tool_steps=0" in c2["evidence"]


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("step_run_id", "step_run_id"),
        ("manifest_hash", "manifest_hash"),
        ("input_hash", "input_hash"),
        ("source_refs", "source_refs"),
        ("random_seed", "random_seed"),
    ],
)
def test_c2_checks_each_executed_tool_step_individually(
    tmp_path,
    field,
    expected,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)

    def remove_field(payload):
        if field == "source_refs":
            payload["source_dataset_refs"] = []
            payload["parent_output_refs"] = []
        else:
            payload.pop(field, None)

    _mutate_step_evidence(workspace, "切分样本", remove_field)
    result = _inspect(workspace)
    c2 = _check(result, "C2")
    assert c2["status"] == "FAIL"
    assert "step-0/切分样本" in c2["evidence"]
    assert expected in c2["evidence"]


def test_c2_allows_seedless_deterministic_step(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)

    def remove_seed(payload):
        payload.pop("random_seed", None)
        payload["input_summary"] = {}

    _mutate_step_evidence(workspace, "模型交付动作", remove_seed)
    result = _inspect(workspace)
    assert _check(result, "C2")["status"] == "PASS"


def test_c2_rejects_malformed_hashes_not_just_missing_hashes(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    _mutate_step_evidence(
        workspace,
        "特征筛选",
        lambda payload: payload.update(
            {"manifest_hash": "manifest", "input_hash": "sha256:short"}
        ),
    )
    result = _inspect(workspace)
    c2 = _check(result, "C2")
    assert c2["status"] == "FAIL"
    assert "manifest_hash,input_hash" in c2["evidence"]


def test_c2_does_not_hide_duplicate_title_with_missing_evidence(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        connection.execute(
            "INSERT INTO plan_steps VALUES (?,?,?,?,?,?,?,?)",
            (
                "step-duplicate",
                "plan-1",
                99,
                "训练模型",
                "modeling",
                "train_models",
                "done",
                "ref:duplicate",
            ),
        )
        connection.execute(
            "INSERT INTO plan_step_outputs VALUES (?,?,?)",
            ("step-duplicate", "{}", "{}"),
        )

    result = _inspect(workspace)
    c2 = _check(result, "C2")
    assert c2["status"] == "FAIL"
    assert "step-duplicate/训练模型" in c2["evidence"]


def test_c3_fails_when_machine_test_evidence_is_missing(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=False,
        traceable=True,
        include_test_evidence=False,
    )
    result = _inspect(workspace)
    c3 = _check(result, "C3")
    assert c3["status"] == "FAIL"
    assert "present=false" in c3["evidence"]


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"exit_code": 1}, "exit_code=1"),
        ({"commit_sha": "b" * 40}, "commit mismatch"),
        ({"timestamp": "not-a-time"}, "invalid timezone-aware timestamp"),
        (
            {"command": "python -m pytest -q tests/test_dirty_shape_regression.py"},
            "command must run exactly",
        ),
        (
            {
                "command": (
                    "true -m pytest -q tests/test_dirty_shape_regression.py "
                    "tests/test_reconcile_reference_numbers.py"
                )
            },
            "command executable is not Python",
        ),
        ({"log_hash": "sha256:not-a-hash"}, "invalid log_hash"),
    ],
)
def test_c3_rejects_invalid_or_stale_machine_test_evidence(
    tmp_path,
    patch,
    expected,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    _mutate_test_evidence(workspace, lambda payload: payload.update(patch))
    result = _inspect(workspace)
    c3 = _check(result, "C3")
    assert c3["status"] == "FAIL"
    assert expected in c3["evidence"]


def test_c3_rejects_tampered_regression_log(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    (workspace / "closure-regression.log").write_text(
        "tampered\n",
        encoding="utf-8",
    )
    result = _inspect(workspace)
    c3 = _check(result, "C3")
    assert c3["status"] == "FAIL"
    assert "log hash mismatch" in c3["evidence"]


def test_c3_rejects_evidence_from_dirty_worktree(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    _mutate_test_evidence(
        workspace,
        lambda payload: payload.update({"worktree_clean": False}),
    )
    result = _inspect(workspace)
    c3 = _check(result, "C3")
    assert c3["status"] == "FAIL"
    assert "not run from a clean worktree" in c3["evidence"]


def test_c3_rechecks_current_worktree_instead_of_trusting_recorded_flag(
    tmp_path,
    monkeypatch,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    monkeypatch.setattr(
        closure_acceptance,
        "_current_repo_state",
        lambda: (_TEST_COMMIT_SHA, False),
    )
    result = _inspect(workspace)
    c3 = _check(result, "C3")
    assert c3["status"] == "FAIL"
    assert "current worktree is not clean" in c3["evidence"]


def test_c3_rejects_commit_change_during_test_run(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    _mutate_test_evidence(
        workspace,
        lambda payload: payload.update({"commit_sha_after": "b" * 40}),
    )
    c3 = _check(_inspect(workspace), "C3")
    assert c3["status"] == "FAIL"
    assert "commit changed during tests" in c3["evidence"]


def test_c3_rejects_log_outside_evidence_directory(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    outside = tmp_path / "outside.log"
    outside.write_text("66 passed in 2.00s\n", encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(outside.read_bytes()).hexdigest()
    _mutate_test_evidence(
        workspace,
        lambda payload: payload.update({"log_path": str(outside), "log_hash": digest}),
    )
    c3 = _check(_inspect(workspace), "C3")
    assert c3["status"] == "FAIL"
    assert "log_path escapes evidence directory" in c3["evidence"]


def test_test_evidence_helper_runs_exact_required_regression_net():
    command = build_command("/python")
    assert command[:4] == ["/python", "-m", "pytest", "-q"]
    assert tuple(command[4:]) == REQUIRED_TESTS


def test_b4_internal_ks_mismatch_is_a_machine_failure(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        row = connection.execute(
            """
            SELECT o.step_id, o.output_json
              FROM plan_step_outputs o
              JOIN plan_steps s ON s.id = o.step_id
             WHERE s.title = '模型交付动作'
            """
        ).fetchone()
        output = json.loads(row[1])
        output["model_card"]["key_metrics"][0]["value"] = 0.1
        connection.execute(
            "UPDATE plan_step_outputs SET output_json = ? WHERE step_id = ?",
            (json.dumps(output), row[0]),
        )

    result = _inspect(workspace)
    assert _check(result, "B4-INTERNAL")["status"] == "FAIL"
    assert _check(result, "B4-EXTERNAL")["status"] == "MANUAL"
    assert "B4-INTERNAL" in result["machine_failures"]


def test_b4_internal_nonfinite_ks_is_a_machine_failure(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    with sqlite3.connect(workspace / "marvis.sqlite") as connection:
        row = connection.execute(
            """
            SELECT o.step_id, o.output_json
              FROM plan_step_outputs o
              JOIN plan_steps s ON s.id = o.step_id
             WHERE s.title = '选择实验'
            """
        ).fetchone()
        output = json.loads(row[1])
        output["metrics"]["test_ks"] = "nan"
        connection.execute(
            "UPDATE plan_step_outputs SET output_json = ? WHERE step_id = ?",
            (json.dumps(output), row[0]),
        )

    result = _inspect(workspace)
    check = _check(result, "B4-INTERNAL")
    assert check["status"] == "FAIL"
    assert "test_ks: nonfinite" in check["evidence"]


@pytest.mark.parametrize(
    ("path", "check_id"),
    [
        ("tasks/task-1/outputs/report.xlsx", "E1"),
        ("tasks/task-1/modeling_artifacts/model.pkl", "E2"),
        ("tasks/task-1/modeling_artifacts/card.json", "E3"),
    ],
)
def test_mandatory_delivery_artifacts_fail_machine_closure(
    tmp_path,
    path,
    check_id,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    (workspace / path).unlink()
    result = _inspect(workspace)
    assert _check(result, check_id)["status"] == "FAIL"
    assert check_id in result["machine_failures"]


def test_missing_pmml_fails_when_recipe_supports_pmml(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=False,
        traceable=True,
        recipe="lr",
    )
    (workspace / "tasks/task-1/modeling_artifacts/model.pmml").unlink()
    result = _inspect(workspace)
    assert _check(result, "E4")["status"] == "FAIL"
    assert "E4" in result["machine_failures"]


def test_missing_pmml_is_na_for_explicitly_unsupported_recipe(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=False,
        traceable=True,
        recipe="catboost",
    )
    (workspace / "tasks/task-1/modeling_artifacts/model.pmml").unlink()
    result = _inspect(workspace)
    assert _check(result, "E4")["status"] == "N/A"
    assert "E4" not in result["machine_failures"]


def test_unknown_recipe_cannot_silently_mark_pmml_not_applicable(tmp_path):
    workspace = _seed_workspace(
        tmp_path,
        sentinel=False,
        traceable=True,
        recipe="mystery",
    )
    result = _inspect(workspace)
    assert _check(result, "E4")["status"] == "FAIL"
    assert "E4" in result["machine_failures"]


@pytest.mark.parametrize(
    ("title", "field", "source_relative", "check_id"),
    [
        ("生成模型开发报告", "report_path", "tasks/task-1/outputs/report.xlsx", "E1"),
        (
            "模型交付动作",
            "native_model_path",
            "tasks/task-1/modeling_artifacts/model.pkl",
            "E2",
        ),
        (
            "模型交付动作",
            "model_card_path",
            "tasks/task-1/modeling_artifacts/card.json",
            "E3",
        ),
        (
            "模型交付动作",
            "pmml_path",
            "tasks/task-1/modeling_artifacts/model.pmml",
            "E4",
        ),
    ],
)
def test_delivery_artifacts_cannot_escape_current_task_governed_directories(
    tmp_path,
    title,
    field,
    source_relative,
    check_id,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    source = workspace / source_relative
    outside = tmp_path / "outside" / source.name
    outside.parent.mkdir()
    outside.write_bytes(source.read_bytes())
    _mutate_step_output(
        workspace,
        title,
        lambda payload: payload.update({field: str(outside)}),
    )
    check = _check(_inspect(workspace), check_id)
    assert check["status"] == "FAIL"
    assert "outside current task governed directory" in check["evidence"]


def test_delivery_artifact_from_different_task_cannot_satisfy_e1(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    other_report = workspace / "tasks" / "task-2" / "outputs" / "report.xlsx"
    other_report.parent.mkdir(parents=True)
    other_report.write_bytes(
        (workspace / "tasks/task-1/outputs/report.xlsx").read_bytes()
    )
    _mutate_step_output(
        workspace,
        "生成模型开发报告",
        lambda payload: payload.update({"report_path": str(other_report)}),
    )
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "FAIL"
    assert "outside current task governed directory" in check["evidence"]


def test_delivery_symlink_cannot_escape_current_task_directory(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    outside = tmp_path / "outside.xlsx"
    outside.write_bytes((workspace / "tasks/task-1/outputs/report.xlsx").read_bytes())
    link = workspace / "tasks/task-1/outputs/linked-report.xlsx"
    link.symlink_to(outside)
    _mutate_step_output(
        workspace,
        "生成模型开发报告",
        lambda payload: payload.update({"report_path": str(link)}),
    )
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "FAIL"
    assert "outside current task governed directory" in check["evidence"]


@pytest.mark.parametrize(
    ("relative_path", "check_id"),
    [
        ("tasks/task-1/outputs/report.xlsx", "E1"),
        ("tasks/task-1/modeling_artifacts/model.pkl", "E2"),
        ("tasks/task-1/modeling_artifacts/card.json", "E3"),
        ("tasks/task-1/modeling_artifacts/model.pmml", "E4"),
    ],
)
def test_delivery_artifacts_require_nonempty_real_file_type(
    tmp_path,
    relative_path,
    check_id,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    (workspace / relative_path).write_bytes(b"not-the-declared-artifact-type")
    check = _check(_inspect(workspace), check_id)
    assert check["status"] == "FAIL"
    assert "file type or suffix invalid" in check["evidence"]


def test_e1_validates_every_experiment_bound_report(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    reports = _seed_multi_reports(workspace)
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "PASS"
    assert f"validated {len(reports)} experiment-bound reports" in check["evidence"]


@pytest.mark.parametrize("damage", ["delete", "garbage-xlsx"])
def test_e1_fails_when_any_nonfirst_report_is_missing_or_broken(
    tmp_path,
    damage,
):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    reports = _seed_multi_reports(workspace)
    second = Path(reports[1]["report_path"])
    if damage == "delete":
        second.unlink()
    else:
        with zipfile.ZipFile(second, "w") as archive:
            archive.writestr("[Content_Types].xml", b"garbage")
            archive.writestr("xl/workbook.xml", b"garbage")
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "FAIL"
    assert "reports[1]" in check["evidence"]


def test_e1_fails_when_report_experiment_binding_is_forged(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    _seed_multi_reports(workspace)

    def forge(payload):
        payload["reports"][1]["experiment_id"] = "experiment-other-task"

    _mutate_step_output(workspace, "生成模型开发报告", forge)
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "FAIL"
    assert "experiment binding mismatch" in check["evidence"]


def test_broken_pickle_protocol_prefix_does_not_satisfy_e2(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    model = workspace / "tasks/task-1/modeling_artifacts/model.pkl"
    model.write_bytes(b"\x80garbage")
    check = _check(_inspect(workspace), "E2")
    assert check["status"] == "FAIL"
    assert "file type or suffix invalid" in check["evidence"]


def test_real_joblib_numpy_payload_satisfies_e2_without_loading_model(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    model = workspace / "tasks/task-1/modeling_artifacts/model.pkl"
    joblib.dump({"coef": np.array([1.0, 2.0])}, model)
    assert _check(_inspect(workspace), "E2")["status"] == "PASS"


def test_dict_backed_estimator_payload_satisfies_e2_without_importing_class(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    model = workspace / "tasks/task-1/modeling_artifacts/model.pkl"
    joblib.dump(
        _DictBackedModelFixture(
            {
                "params": {"max_depth": 3},
                "coef": np.array([1.0, 2.0]),
            }
        ),
        model,
    )
    assert _check(_inspect(workspace), "E2")["status"] == "PASS"


def test_two_garbage_xlsx_entries_do_not_satisfy_e1(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    report = workspace / "tasks/task-1/outputs/report.xlsx"
    with zipfile.ZipFile(report, "w") as archive:
        archive.writestr("[Content_Types].xml", b"garbage")
        archive.writestr("xl/workbook.xml", b"garbage")
    check = _check(_inspect(workspace), "E1")
    assert check["status"] == "FAIL"
    assert "file type or suffix invalid" in check["evidence"]


def test_empty_pmml_shell_does_not_satisfy_e4(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    pmml = workspace / "tasks/task-1/modeling_artifacts/model.pmml"
    pmml.write_text("<PMML/>", encoding="utf-8")
    check = _check(_inspect(workspace), "E4")
    assert check["status"] == "FAIL"
    assert "file type or suffix invalid" in check["evidence"]


def test_model_card_must_be_bound_to_selected_artifact(tmp_path):
    workspace = _seed_workspace(tmp_path, sentinel=False, traceable=True)
    card = workspace / "tasks/task-1/modeling_artifacts/card.json"
    card.write_text(
        json.dumps({"artifact_id": "artifact-other", "recipe": "lr"}),
        encoding="utf-8",
    )
    check = _check(_inspect(workspace), "E3")
    assert check["status"] == "FAIL"
    assert "file type or suffix invalid" in check["evidence"]
