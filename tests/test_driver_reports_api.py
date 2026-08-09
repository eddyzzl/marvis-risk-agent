from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from marvis.api_report_helpers import driver_report_id
from marvis.app import create_app
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef
from marvis.repositories.task_artifacts import TaskArtifactRepository


def test_each_plural_model_report_has_a_task_scoped_download(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "materials"
    materials.mkdir()
    response = client.post("/api/tasks", json={
        "model_name": "多模型报告",
        "validator": "qa",
        "source_dir": str(materials),
        "task_type": "modeling",
        "run_mode": "manual",
    })
    assert response.status_code == 200, response.text
    task_id = response.json()["id"]

    outputs_dir = client.app.state.settings.tasks_dir / task_id / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    lgb_path = outputs_dir / "model_report_lgb.xlsx"
    xgb_path = outputs_dir / "model_report_xgb.xlsx"
    outside_path = tmp_path / "outside.xlsx"
    lgb_path.write_bytes(b"PK-lgb")
    xgb_path.write_bytes(b"PK-xgb")
    outside_path.write_bytes(b"PK-outside")

    plan_id = "plan-multi-report"
    step_id = "step-multi-report"
    step = PlanStep(
        id=step_id,
        plan_id=plan_id,
        index=0,
        title="生成模型开发报告",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    plan = Plan(
        id=plan_id,
        task_id=task_id,
        goal="modeling",
        source="template",
        template_id="modeling",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.DONE,
    )
    repo = client.app.state.plan_repo
    repo.create_plan(plan)
    step.output_ref = repo.store_step_output(step_id, {
        "report_path": str(lgb_path),
        "reports": [
            {
                "experiment_id": "exp-lgb",
                "recipe": "lgb",
                "report_path": str(lgb_path),
            },
            {
                "experiment_id": "exp-xgb",
                "recipe": "xgb",
                "report_path": str(xgb_path),
            },
            {
                "experiment_id": "exp-outside",
                "recipe": "outside",
                "report_path": str(outside_path),
            },
        ],
    })
    repo.update_step(step)

    first = client.get(
        f"/api/tasks/{task_id}/driver-reports/"
        f"{driver_report_id(plan_id, step_id, 0)}/download"
    )
    second = client.get(
        f"/api/tasks/{task_id}/driver-reports/"
        f"{driver_report_id(plan_id, step_id, 1)}/download"
    )
    assert first.status_code == 200
    assert first.content == b"PK-lgb"
    assert "model_report_lgb.xlsx" in first.headers["content-disposition"]
    assert second.status_code == 200
    assert second.content == b"PK-xgb"
    assert "model_report_xgb.xlsx" in second.headers["content-disposition"]

    # The client never supplies a path.  Even the opaque id for a persisted
    # out-of-task candidate is rejected by the server containment check.
    escaped = client.get(
        f"/api/tasks/{task_id}/driver-reports/"
        f"{driver_report_id(plan_id, step_id, 2)}/download"
    )
    assert escaped.status_code == 404
    assert client.get(
        f"/api/tasks/{task_id}/driver-reports/not-a-real-report/download"
    ).status_code == 404

    # Backward-compatible primary download still resolves the first safe report.
    legacy = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert legacy.status_code == 200
    assert legacy.content == b"PK-lgb"


def test_driver_report_downloads_reject_symlinked_outputs_directory(tmp_path: Path):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "materials"
    materials.mkdir()
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "模型报告路径校验",
            "validator": "qa",
            "source_dir": str(materials),
            "task_type": "modeling",
            "run_mode": "manual",
        },
    ).json()["id"]
    task_dir = client.app.state.settings.tasks_dir / task_id
    task_dir.mkdir(parents=True)
    outside_outputs = tmp_path / "outside-outputs"
    outside_outputs.mkdir()
    report_path = outside_outputs / "model_report.xlsx"
    report_path.write_bytes(b"OUTSIDE_REPORT_SECRET")
    (task_dir / "outputs").symlink_to(
        outside_outputs,
        target_is_directory=True,
    )

    plan_id = "plan-symlink-report"
    step_id = "step-symlink-report"
    step = PlanStep(
        id=step_id,
        plan_id=plan_id,
        index=0,
        title="生成模型开发报告",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    plan = Plan(
        id=plan_id,
        task_id=task_id,
        goal="modeling",
        source="template",
        template_id="modeling",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.DONE,
    )
    repo = client.app.state.plan_repo
    repo.create_plan(plan)
    step.output_ref = repo.store_step_output(
        step_id,
        {"report_path": str(task_dir / "outputs" / report_path.name)},
    )
    repo.update_step(step)

    opaque = client.get(
        f"/api/tasks/{task_id}/driver-reports/"
        f"{driver_report_id(plan_id, step_id, 0)}/download"
    )
    legacy = client.get(f"/api/tasks/{task_id}/driver-report/download")

    assert opaque.status_code == 404
    assert legacy.status_code == 404
    assert opaque.content != b"OUTSIDE_REPORT_SECRET"
    assert legacy.content != b"OUTSIDE_REPORT_SECRET"


def test_registered_portfolio_report_outside_outputs_downloads_only_while_hash_matches(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "portfolio-materials"
    materials.mkdir()
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "组合报告下载",
            "validator": "qa",
            "source_dir": str(materials),
            "task_type": "portfolio",
            "run_mode": "agent",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["id"]

    report_dir = client.app.state.settings.tasks_dir / task_id / "portfolio"
    report_dir.mkdir(parents=True)
    report_path = report_dir / "portfolio_report.xlsx"
    report_bytes = b"PK-portfolio-report"
    report_path.write_bytes(report_bytes)
    content_hash = hashlib.sha256(report_bytes).hexdigest()
    artifact = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).register(
        task_id=task_id,
        kind="portfolio_report_xlsx",
        path=str(report_path),
        content_hash=content_hash,
        origin_tool="analysis.portfolio_report",
        provenance={"schema_version": "portfolio-report-artifact.v1"},
    )

    plan_id = "plan-portfolio-report"
    step_id = "step-portfolio-report"
    step = PlanStep(
        id=step_id,
        plan_id=plan_id,
        index=0,
        title="生成组合报告",
        tool_ref=ToolRef("analysis", "portfolio_report"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    plan = Plan(
        id=plan_id,
        task_id=task_id,
        goal="portfolio",
        source="template",
        template_id="portfolio_analysis_no_trend",
        steps=[step],
        autonomy_level=1,
        status=PlanStatus.DONE,
    )
    repo = client.app.state.plan_repo
    repo.create_plan(plan)
    step.output_ref = repo.store_step_output(
        step_id,
        {
            "report_path": str(report_path),
            "artifact_id": artifact["id"],
            "artifact_content_hash": content_hash,
        },
    )
    repo.update_step(step)

    downloaded = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == report_bytes

    step.output_ref = repo.store_step_output(
        step_id,
        {
            "report_path": str(report_path),
            "artifact_id": "not-the-registered-artifact",
            "artifact_content_hash": content_hash,
        },
    )
    repo.update_step(step)
    assert (
        client.get(f"/api/tasks/{task_id}/driver-report/download").status_code
        == 404
    )

    step.output_ref = repo.store_step_output(
        step_id,
        {
            "report_path": str(report_path),
            "artifact_id": artifact["id"],
            "artifact_content_hash": content_hash,
        },
    )
    repo.update_step(step)
    report_path.write_bytes(b"PK-tampered")
    assert (
        client.get(f"/api/tasks/{task_id}/driver-report/download").status_code
        == 404
    )


def test_latest_invalid_registered_report_does_not_fall_back_to_older_plan(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "portfolio-materials"
    materials.mkdir()
    response = client.post(
        "/api/tasks",
        json={
            "model_name": "组合报告禁止旧版回退",
            "validator": "qa",
            "source_dir": str(materials),
            "task_type": "portfolio",
            "run_mode": "agent",
        },
    )
    assert response.status_code == 200, response.text
    task_id = response.json()["id"]
    repo = client.app.state.plan_repo

    outputs_dir = client.app.state.settings.tasks_dir / task_id / "outputs"
    outputs_dir.mkdir(parents=True)
    old_report_path = outputs_dir / "old_report.xlsx"
    old_report_path.write_bytes(b"PK-old-report")
    old_step = PlanStep(
        id="step-old-report",
        plan_id="plan-old-report",
        index=0,
        title="旧报告",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    old_plan = Plan(
        id="plan-old-report",
        task_id=task_id,
        goal="old report",
        source="template",
        template_id="modeling",
        steps=[old_step],
        autonomy_level=1,
        status=PlanStatus.DONE,
        created_at="2026-08-01T00:00:00+00:00",
    )
    repo.create_plan(old_plan)
    old_step.output_ref = repo.store_step_output(
        old_step.id,
        {"report_path": str(old_report_path)},
    )
    repo.update_step(old_step)

    portfolio_dir = client.app.state.settings.tasks_dir / task_id / "portfolio"
    portfolio_dir.mkdir()
    newest_report_path = portfolio_dir / "newest_report.xlsx"
    newest_bytes = b"PK-newest-report"
    newest_report_path.write_bytes(newest_bytes)
    actual_hash = hashlib.sha256(newest_bytes).hexdigest()
    artifact = TaskArtifactRepository(
        client.app.state.settings.db_path
    ).register(
        task_id=task_id,
        kind="portfolio_report_xlsx",
        path=str(newest_report_path),
        content_hash=actual_hash,
        origin_tool="analysis.portfolio_report",
        provenance={"schema_version": "portfolio-report-artifact.v1"},
    )
    newest_step = PlanStep(
        id="step-newest-report",
        plan_id="plan-newest-report",
        index=0,
        title="最新报告",
        tool_ref=ToolRef("analysis", "portfolio_report"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    newest_plan = Plan(
        id="plan-newest-report",
        task_id=task_id,
        goal="newest report",
        source="template",
        template_id="portfolio_analysis_no_trend",
        steps=[newest_step],
        autonomy_level=1,
        status=PlanStatus.DONE,
        created_at="2026-08-01T01:00:00+00:00",
    )
    repo.create_plan(newest_plan)
    newest_step.output_ref = repo.store_step_output(
        newest_step.id,
        {
            "report_path": str(newest_report_path),
            "artifact_id": artifact["id"],
            "artifact_content_hash": "0" * 64,
        },
    )
    repo.update_step(newest_step)

    response = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert response.status_code == 404
    assert response.json()["detail"] == "report not generated"


@pytest.mark.parametrize("new_output", [None, {}, {"status": "done"}])
def test_latest_completed_report_step_with_missing_output_fails_closed(
    tmp_path: Path,
    new_output: dict | None,
):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "materials"
    materials.mkdir()
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "缺失报告输出禁止旧版回退",
            "validator": "qa",
            "source_dir": str(materials),
            "task_type": "modeling",
            "run_mode": "manual",
        },
    ).json()["id"]
    repo = client.app.state.plan_repo
    outputs_dir = client.app.state.settings.tasks_dir / task_id / "outputs"
    outputs_dir.mkdir(parents=True)
    old_path = outputs_dir / "old_report.xlsx"
    old_path.write_bytes(b"PK-old")

    old_step = PlanStep(
        id="step-old-complete-report",
        plan_id="plan-old-complete-report",
        index=0,
        title="旧报告",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    old_plan = Plan(
        id="plan-old-complete-report",
        task_id=task_id,
        goal="old report",
        source="template",
        template_id="modeling",
        steps=[old_step],
        autonomy_level=1,
        status=PlanStatus.DONE,
        created_at="2026-08-01T00:00:00+00:00",
    )
    repo.create_plan(old_plan)
    old_step.output_ref = repo.store_step_output(
        old_step.id,
        {"report_path": str(old_path)},
    )
    repo.update_step(old_step)

    newest_step = PlanStep(
        id="step-new-missing-report",
        plan_id="plan-new-missing-report",
        index=0,
        title="最新报告",
        tool_ref=ToolRef("analysis", "portfolio_report"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    newest_plan = Plan(
        id="plan-new-missing-report",
        task_id=task_id,
        goal="new report",
        source="template",
        template_id="portfolio_analysis_no_trend",
        steps=[newest_step],
        autonomy_level=1,
        status=PlanStatus.DONE,
        created_at="2026-08-01T01:00:00+00:00",
    )
    repo.create_plan(newest_plan)
    if new_output is not None:
        newest_step.output_ref = repo.store_step_output(newest_step.id, new_output)
        repo.update_step(newest_step)

    response = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert response.status_code == 404
    assert response.json()["detail"] == "report not generated"


@pytest.mark.parametrize(
    ("step_status", "plan_status"),
    [
        (StepStatus.FAILED, PlanStatus.FAILED),
        (StepStatus.PENDING, PlanStatus.RUNNING),
        (StepStatus.RUNNING, PlanStatus.RUNNING),
        (StepStatus.SKIPPED, PlanStatus.CANCELLED),
    ],
)
def test_latest_incomplete_report_attempt_blocks_older_successful_report(
    tmp_path: Path,
    step_status: StepStatus,
    plan_status: PlanStatus,
):
    client = TestClient(create_app(tmp_path / "workspace"))
    materials = client.app.state.settings.workspace / "materials"
    materials.mkdir()
    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "最新报告尝试失败时禁止旧版回退",
            "validator": "qa",
            "source_dir": str(materials),
            "task_type": "modeling",
            "run_mode": "manual",
        },
    ).json()["id"]
    repo = client.app.state.plan_repo
    outputs_dir = client.app.state.settings.tasks_dir / task_id / "outputs"
    outputs_dir.mkdir(parents=True)
    old_path = outputs_dir / "old_report.xlsx"
    old_path.write_bytes(b"PK-OLD-REPORT")

    old_step = PlanStep(
        id="step-old-successful-report",
        plan_id="plan-old-successful-report",
        index=0,
        title="旧成功报告",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
    )
    old_plan = Plan(
        id="plan-old-successful-report",
        task_id=task_id,
        goal="old report",
        source="template",
        template_id="modeling",
        steps=[old_step],
        autonomy_level=1,
        status=PlanStatus.DONE,
        created_at="2026-08-01T00:00:00+00:00",
    )
    repo.create_plan(old_plan)
    old_step.output_ref = repo.store_step_output(
        old_step.id,
        {"report_path": str(old_path)},
    )
    repo.update_step(old_step)

    newest_step = PlanStep(
        id=f"step-new-{step_status.value}-report",
        plan_id=f"plan-new-{step_status.value}-report",
        index=0,
        title="最新报告尝试",
        tool_ref=ToolRef("modeling", "generate_model_reports"),
        inputs={},
        depends_on=[],
        post_checks=[],
        status=step_status,
        error="report generation failed" if step_status is StepStatus.FAILED else None,
    )
    newest_plan = Plan(
        id=newest_step.plan_id,
        task_id=task_id,
        goal="new report",
        source="template",
        template_id="modeling",
        steps=[newest_step],
        autonomy_level=1,
        status=plan_status,
        created_at="2026-08-01T01:00:00+00:00",
    )
    repo.create_plan(newest_plan)

    response = client.get(f"/api/tasks/{task_id}/driver-report/download")
    assert response.status_code == 404
    assert response.json()["detail"] == "report not generated"
    assert response.content != b"PK-OLD-REPORT"
