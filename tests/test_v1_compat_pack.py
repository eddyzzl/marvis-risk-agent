import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

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
