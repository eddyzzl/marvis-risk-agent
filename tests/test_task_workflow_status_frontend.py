import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).parents[1] / "marvis" / "static"


def _run_node(source: str) -> dict:
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_workflow_status_snapshot_maps_backend_fallback_for_task_pills():
    module_url = (STATIC_DIR / "js" / "v2" / "plan_rail_controller.js").as_uri()
    payload = _run_node(
        "\n".join(
            [
                (
                    "const { workflowStatusSnapshot } = "
                    f"await import({json.dumps(module_url)});"
                ),
                "console.log(JSON.stringify({",
                "  failed: workflowStatusSnapshot('failed'),",
                "  waiting: workflowStatusSnapshot('awaiting_confirm'),",
                "  running: workflowStatusSnapshot('running'),",
                "  done: workflowStatusSnapshot('done'),",
                "}));",
            ]
        )
    )

    assert payload["failed"]["label"] == "失败"
    # Batch 4 (2026-10-14 redesign): snapshots use the shared tone set (ok|fail|run|review|neutral).
    assert payload["failed"]["tone"] == "fail"
    assert payload["waiting"]["label"] == "待确认"
    assert payload["running"]["label"] == "执行中"
    assert payload["running"]["tone"] == "run"
    assert payload["done"]["label"] == "已完成"
    # Batch 4 (2026-10-14 redesign): success -> ok in the shared tone set.
    assert payload["done"]["tone"] == "ok"


def test_task_status_rendering_prefers_plan_then_workflow_then_task_status():
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    helper_start = app_js.index("function taskPlanWorkflowStatusSnapshot")
    label_start = app_js.index("function taskStatusLabel")
    tone_start = app_js.index("function taskStatusTone")
    helper = app_js[helper_start:label_start]
    label = app_js[label_start:tone_start]
    tone = app_js[tone_start:app_js.index("function notebookReproducibilityComplete")]

    assert helper.index("planRailController.statusSnapshot") < helper.index(
        "workflowStatusSnapshot(task?.workflow_status)"
    )
    # Batch 4 (2026-10-14 redesign): the row label/tone come from the shared resolver;
    # the plan/workflow snapshot is its planStatus input, ahead of task.status.
    context = app_js[app_js.index("function taskStatusContext"):label_start]
    assert "planStatus: taskPlanWorkflowStatusSnapshot(task)" in context
    assert "resolveTaskStatus(task, taskStatusContext(task))" in context
    assert "taskResolvedStatus(task).label" in label
    assert "taskResolvedStatus(task).tone" in tone
    status_js = (STATIC_DIR / "js" / "task-status.js").read_text(encoding="utf-8")
    resolver = status_js[status_js.index("export function resolveTaskStatus"):]
    assert resolver.index("if (planKey) return statusPresentation(planKey);") < resolver.index(
        "V1_TASK_KEYS[status]"
    )
