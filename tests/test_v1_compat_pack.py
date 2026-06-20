import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

from docx import Document
from fastapi.testclient import TestClient
import nbformat
import pytest

from marvis.app import create_app
from marvis.db import PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate, TaskStatus
from marvis.orchestrator.planner import Planner
from marvis.orchestrator.templates import get_template, load_builtin_templates
from marvis.orchestrator.validator import PlanValidator
from marvis.packs.v1_compat.adapters import (
    artifact_ref,
    report_artifacts,
    validation_metric_summary,
)
from marvis.packs.v1_compat.tools import (
    tool_compute_validation_metrics,
    tool_render_reports,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_v1_compat_manifest_registers_builtin_tools(tmp_path):
    registry, _repo = _registry(tmp_path)

    manifest = registry.get("v1_compat")

    assert manifest.builtin is True
    assert {tool.name for tool in manifest.tools} == {
        "scan_materials",
        "run_notebook",
        "compute_validation_metrics",
        "render_reports",
    }
    metrics_tool = next(tool for tool in manifest.tools if tool.name == "compute_validation_metrics")
    assert {"ks", "auc", "psi"} <= set(metrics_tool.output_schema["properties"])


def test_scan_materials_tool_runner_scans_without_file_contents(tmp_path):
    runner, _plugin_repo = _runner(tmp_path)
    repo = TaskRepository(tmp_path / "workspace" / "marvis.sqlite")
    source_dir = tmp_path / "materials"
    _write_materials(source_dir)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
        )
    )

    result = runner.invoke(
        ToolRef("v1_compat", "scan_materials"),
        {"task_id": task.id},
        task_id=task.id,
    )

    assert result.ok is True
    assert result.output["status"] == "scanned"
    assert {item["role"] for item in result.output["materials"]} == {
        "notebook",
        "sample",
        "model_pmml",
        "data_dictionary",
    }
    assert all(
        "secret notebook source" not in json.dumps(item)
        for item in result.output["materials"]
    )
    assert repo.get_task(task.id).status == TaskStatus.SCANNED


def test_scan_materials_marks_task_failed_when_required_material_is_missing(tmp_path):
    runner, _plugin_repo = _runner(tmp_path)
    repo = TaskRepository(tmp_path / "workspace" / "marvis.sqlite")
    source_dir = tmp_path / "materials"
    source_dir.mkdir(parents=True)
    (source_dir / "model.ipynb").write_text("secret notebook source", encoding="utf-8")
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
        )
    )

    result = runner.invoke(
        ToolRef("v1_compat", "scan_materials"),
        {"task_id": task.id},
        task_id=task.id,
    )

    assert result.ok is True
    assert result.output["status"] == "failed"
    assert any(check["status"] == "missing" for check in result.output["checks"])
    assert repo.get_task(task.id).status == TaskStatus.FAILED


def test_v1_compat_tool_runner_converts_missing_task_to_execution_error(tmp_path):
    runner, _plugin_repo = _runner(tmp_path)

    result = runner.invoke(
        ToolRef("v1_compat", "scan_materials"),
        {"task_id": "missing"},
        task_id="missing",
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "Task not found" in result.error


def test_run_notebook_tool_runner_converts_missing_material_to_structured_error(tmp_path):
    runner, plugin_repo = _runner(tmp_path)
    repo = TaskRepository(tmp_path / "workspace" / "marvis.sqlite")
    source_dir = tmp_path / "materials"
    source_dir.mkdir(parents=True)
    (source_dir / "model.ipynb").write_text("secret notebook source", encoding="utf-8")
    (source_dir / "model.pmml").write_text("<PMML>secret</PMML>", encoding="utf-8")
    (source_dir / "dictionary.csv").write_text("特征名,类别\nx,base\n", encoding="utf-8")
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
        )
    )

    result = runner.invoke(
        ToolRef("v1_compat", "run_notebook"),
        {"task_id": task.id},
        task_id=task.id,
    )

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "missing required input: sample" in result.error
    combined_output = f"{result.error}\n{result.stdout_tail}\n{result.stderr_tail}"
    assert "secret notebook source" not in combined_output
    assert "<PMML>secret</PMML>" not in combined_output
    assert repo.get_task(task.id).status == TaskStatus.FAILED
    audit = plugin_repo.list_audit(kind="tool.invoke")[-1]
    assert audit["target_ref"] == "v1_compat.run_notebook"
    assert audit["outcome"] == "failed"


def test_tool_runner_runs_metrics_and_reports_after_notebook_in_separate_workers(tmp_path):
    runner, _plugin_repo = _runner(tmp_path)
    repo = TaskRepository(tmp_path / "workspace" / "marvis.sqlite")
    _write_report_template(tmp_path / "workspace" / "report_templates" / "default.docx")
    source_dir = _write_validation_project(tmp_path / "materials")
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
            algorithm="lgb",
            target_col="y",
            score_col="pred",
            split_col="split",
            time_col="apply_month",
        )
    )

    notebook = runner.invoke(
        ToolRef("v1_compat", "run_notebook"),
        {"task_id": task.id},
        task_id=task.id,
    )
    metrics = runner.invoke(
        ToolRef("v1_compat", "compute_validation_metrics"),
        {"task_id": task.id},
        task_id=task.id,
    )
    report = runner.invoke(
        ToolRef("v1_compat", "render_reports"),
        {"task_id": task.id},
        task_id=task.id,
    )

    assert notebook.ok is True, notebook.error
    assert notebook.output["status"] == "executed"
    assert metrics.ok is True, metrics.error
    assert metrics.output["status"] == "writing_artifacts"
    assert 0.0 <= metrics.output["ks"] <= 1.0
    assert 0.0 <= metrics.output["auc"] <= 1.0
    assert metrics.output["score_consistency_passed"] is True
    assert report.ok is True, report.error
    assert report.output["status"] == "succeeded"
    assert {artifact["kind"] for artifact in report.output["artifacts"]} == {"excel", "word"}
    assert {
        artifact["path"] for artifact in report.output["artifacts"]
    } == {
        f"tasks/{task.id}/outputs/validation.xlsx",
        f"tasks/{task.id}/outputs/validation_report.docx",
    }


def test_compute_metrics_summary_keeps_only_top_level_metrics(tmp_path, monkeypatch):
    context = _task_context(tmp_path)
    outputs_dir = context.task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "validation_results.json").write_text(
        json.dumps({
            "reproducibility": {"summary": {"status": "pass"}},
            "effectiveness": {
                "overall": [
                    {
                        "split": "oot",
                        "ks": 0.321,
                        "auc": 0.745,
                        "psi_vs_train": None,
                    }
                ]
            },
            "raw_sample_preview": [{"customer_id": "should-not-leak"}],
        }),
        encoding="utf-8",
    )
    _advance_status(context.repo, context.task_id, TaskStatus.EXECUTED)

    def fake_run_metrics_stage(*, task_id, settings):
        context.repo.update_status(
            task_id,
            TaskStatus.COMPUTING_METRICS,
            message="computing",
            expected=TaskStatus.EXECUTED,
        )
        context.repo.update_status(
            task_id,
            TaskStatus.WRITING_ARTIFACTS,
            message="metrics generated",
            expected=TaskStatus.COMPUTING_METRICS,
        )

    monkeypatch.setattr(
        "marvis.packs.v1_compat.tools.run_metrics_stage",
        fake_run_metrics_stage,
    )
    monkeypatch.setattr(
        "marvis.packs.v1_compat.tools.get_live_notebook_session",
        lambda _task_id: object(),
    )

    output = tool_compute_validation_metrics({"task_id": context.task_id}, context.ctx)

    assert output == {
        "task_id": context.task_id,
        "status": "writing_artifacts",
        "ks": 0.321,
        "auc": 0.745,
        "psi": None,
        "score_consistency_passed": True,
        "validation_results_ref": f"artifact:tasks/{context.task_id}/outputs/validation_results.json",
    }
    assert "customer_id" not in json.dumps(output)
    assert validation_metric_summary(context.v1)["psi"] is None


def test_validation_metric_summary_rejects_missing_required_metrics(tmp_path):
    context = _task_context(tmp_path)
    outputs_dir = context.task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "validation_results.json").write_text(
        json.dumps({
            "reproducibility": {"summary": {"status": "pass"}},
            "effectiveness": {
                "overall": [
                    {
                        "split": "oot",
                        "auc": 0.745,
                    }
                ]
            },
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="metrics output missing ks"):
        validation_metric_summary(context.v1)


def test_v1_compat_artifact_refs_are_workspace_relative_and_safe(tmp_path):
    context = _task_context(tmp_path)

    ref = artifact_ref(context.v1, context.task_dir / "outputs" / "validation_results.json")

    assert ref == f"artifact:tasks/{context.task_id}/outputs/validation_results.json"
    assert str(context.task_dir) not in ref


def test_render_reports_wraps_existing_download_routes(tmp_path, monkeypatch):
    context = _task_context(tmp_path)
    outputs_dir = context.task_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "validation.xlsx").write_bytes(b"xlsx")
    (outputs_dir / "validation_report.docx").write_bytes(b"docx")
    _advance_status(context.repo, context.task_id, TaskStatus.WRITING_ARTIFACTS)

    def fake_run_report_stage(*, task_id, settings):
        context.repo.update_status(
            task_id,
            TaskStatus.SUCCEEDED,
            message="done",
            expected=TaskStatus.WRITING_ARTIFACTS,
        )

    monkeypatch.setattr(
        "marvis.packs.v1_compat.tools.run_report_stage",
        fake_run_report_stage,
    )

    output = tool_render_reports({"task_id": context.task_id}, context.ctx)

    assert output["status"] == "succeeded"
    assert output["artifacts"] == report_artifacts(context.v1)
    assert {artifact["download_url"] for artifact in output["artifacts"]} == {
        f"/api/tasks/{context.task_id}/analysis/download",
        f"/api/tasks/{context.task_id}/report/download",
    }


def test_model_validation_template_resolves_v1_compat_tools(tmp_path):
    registry, _repo = _registry(tmp_path)
    tool_registry = ToolRegistry(registry)
    load_builtin_templates()
    template = get_template("model_validation")
    planner = Planner(tool_registry, lambda: _NoLLM(), PlanValidator(tool_registry))

    plan = planner.from_template(template, {"task_id": "task-1"}, "task-1")
    problems = PlanValidator(tool_registry).validate(plan)

    assert [step.tool_ref.plugin for step in plan.steps] == ["v1_compat"] * 4
    assert plan.steps[-1].needs_confirmation is True
    assert problems == []


def test_model_validation_plan_api_uses_v1_compat_template(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    source_dir = tmp_path / "materials"
    _write_materials(source_dir)
    task = TaskRepository(tmp_path / "marvis.sqlite").create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
        )
    )

    response = client.post(
        f"/api/tasks/{task.id}/plans",
        json={"goal": "模型验证"},
    )

    assert response.status_code == 201, response.text
    plan = response.json()["plan"]
    assert plan["template_id"] == "model_validation"
    assert [step["tool_ref"]["plugin"] for step in plan["steps"]] == ["v1_compat"] * 4
    assert [step["tool_ref"]["tool"] for step in plan["steps"]] == [
        "scan_materials",
        "run_notebook",
        "compute_validation_metrics",
        "render_reports",
    ]


class _NoLLM:
    def complete(self, **_kwargs):
        raise AssertionError("template path should not call LLM")


def _runner(tmp_path):
    workspace = tmp_path / "workspace"
    init_db(workspace / "marvis.sqlite")
    registry, plugin_repo = _registry(tmp_path)
    return (
        ToolRunner(
            ToolRegistry(registry),
            plugin_repo,
            python_executable=sys.executable,
            datasets_root=workspace / "datasets",
            workspace=workspace,
        ),
        plugin_repo,
    )


def _registry(tmp_path):
    db_path = tmp_path / "plugins.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    load_builtin_packs(registry, Path(__file__).parents[1] / "marvis" / "packs")
    return registry, repo


def _write_materials(source_dir: Path) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "model.ipynb").write_text("secret notebook source", encoding="utf-8")
    (source_dir / "sample.csv").write_text("y,pred\n1,0.9\n", encoding="utf-8")
    (source_dir / "model.pmml").write_text("<PMML>secret</PMML>", encoding="utf-8")
    (source_dir / "dictionary.csv").write_text("特征名,类别\nx,base\n", encoding="utf-8")


def _write_validation_project(source_dir: Path) -> Path:
    source_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("train", "test", "oot"):
        for index in range(40):
            x1 = (index + 1) / 41
            rows.append(
                {
                    "x1": x1,
                    "x2": 0.0,
                    "pred": 1 / (1 + math.exp(-x1)),
                    "y": int(index >= 20),
                    "split": split,
                    "apply_month": "202503" if split == "train" else "202505",
                }
            )
    csv_lines = ["x1,x2,pred,y,split,apply_month"]
    csv_lines.extend(
        f"{row['x1']},{row['x2']},{row['pred']},{row['y']},{row['split']},{row['apply_month']}"
        for row in rows
    )
    (source_dir / "sample.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    shutil.copy(FIXTURES / "min_lr.pmml", source_dir / "fr_final.pmml")
    (source_dir / "data_dictionary.csv").write_text(
        "特征名,类别\nx1,征信\nx2,基础信息\n",
        encoding="utf-8",
    )
    _write_contract_notebook(source_dir / "dev.ipynb")
    return source_dir


def _write_contract_notebook(path: Path) -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "\n".join(
                    [
                        "import pandas as pd",
                        "RMC_SAMPLE_DF = pd.read_csv('sample.csv')",
                        "RMC_TARGET_COL = 'y'",
                        "RMC_ALGORITHM = 'lgb'",
                        "RMC_SPLIT_COL = 'split'",
                        "RMC_TIME_COL = 'apply_month'",
                        "def RMC_SCORE_FN(df):",
                        "    return df['pred']",
                    ]
                )
            )
        ]
    )
    nbformat.write(notebook, path)


def _write_report_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    document.add_paragraph("MARVIS validation report")
    document.save(path)


def _task_context(tmp_path):
    from marvis.packs.v1_compat.adapters import load_v1_task_context

    workspace = tmp_path / "workspace"
    init_db(workspace / "marvis.sqlite")
    repo = TaskRepository(workspace / "marvis.sqlite")
    source_dir = tmp_path / "materials"
    _write_materials(source_dir)
    task = repo.create_task(
        TaskCreate(
            model_name="A卡",
            model_version="v1",
            validator="qa",
            source_dir=str(source_dir),
        )
    )
    ctx = SimpleNamespace(workspace=workspace, task_id=task.id)
    return SimpleNamespace(
        ctx=ctx,
        repo=repo,
        task_id=task.id,
        task_dir=workspace / "tasks" / task.id,
        v1=load_v1_task_context(ctx, task.id),
    )


def _advance_status(repo: TaskRepository, task_id: str, target: TaskStatus) -> None:
    transitions = [
        (TaskStatus.SCANNED, TaskStatus.CREATED),
        (TaskStatus.RUNNING, TaskStatus.SCANNED),
        (TaskStatus.EXECUTED, TaskStatus.RUNNING),
        (TaskStatus.COMPUTING_METRICS, TaskStatus.EXECUTED),
        (TaskStatus.WRITING_ARTIFACTS, TaskStatus.COMPUTING_METRICS),
    ]
    for status, expected in transitions:
        repo.update_status(task_id, status, message=status.value, expected=expected)
        if status is target:
            return
