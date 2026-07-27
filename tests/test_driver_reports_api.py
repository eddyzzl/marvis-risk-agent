from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from marvis.api_report_helpers import driver_report_id
from marvis.app import create_app
from marvis.orchestrator.contracts import Plan, PlanStatus, PlanStep, StepStatus
from marvis.plugins.manifest import ToolRef


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
