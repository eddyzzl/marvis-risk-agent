"""Driver report-download affordance (drives the /driver-report/download endpoint).

The backend endpoint + 404/200 behavior is covered in test_feature_analysis_api;
this guards the frontend wiring (button + handler). The interactive 下载报告 button
now lives in the middle-workspace driver-actions panel (planDriverActionsHtml), NOT
inline on the narrow rail step row. The rail contains steps and status icons only.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _slice_function(source: str, signature: str) -> str:
    start = source.index(signature)
    depth = 0
    body_started = False
    for index in range(start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
            body_started = True
        elif char == "}":
            depth -= 1
            if body_started and depth == 0:
                return source[start : index + 1]
    return source[start:]


def test_driver_report_download_button_lives_in_middle_panel_with_handler():
    app_js = _read("app.js")
    plan_rail_js = _read("js/v2/plan_rail_controller.js")

    # The 下载报告 button HTML is produced by the middle driver-actions panel
    # builder (planDriverActionsHtml), driven by a completed report step.
    actions_body = _slice_function(plan_rail_js, "function planDriverActionsHtml")
    assert 'data-driver-report-download="1"' in actions_body
    assert "plan-step-download" in actions_body
    assert "下载报告" in actions_body
    assert "特征分析报告" in actions_body
    assert "模型开发报告" in actions_body
    # The done-report predicate (which report tools count) lives in doneReportStep.
    report_step_body = _slice_function(plan_rail_js, "function doneReportStep")
    assert "generate_model_report" in report_step_body
    assert "generate_model_reports" in report_step_body
    assert "generate_feature_report" in report_step_body
    assert "generate_risk_analysis_report" in report_step_body

    # The rail substep row carries status/progress only. Report readiness and
    # every interactive report action live in the middle workspace.
    substep_body = _slice_function(plan_rail_js, "function planSubstepHtml")
    assert 'data-driver-report-download="1"' not in substep_body
    assert "plan-step-ready" not in substep_body
    assert "data-plan-report-locate" not in substep_body
    assert "build_report_bundle_v2" not in substep_body
    assert 'openDriverActionCard("strategy-artifacts")' not in substep_body
    assert "renderStepChecker(checkerStatus)" in substep_body
    assert "tuningProgress" in substep_body

    # the handler navigates to the driver-report download endpoint
    assert "function handleDriverReportDownloadClick" in app_js
    assert "`api/tasks/${encodeURIComponent(taskSession.taskId)}/driver-report/download`" in app_js
    assert "`/api/tasks/${encodeURIComponent(taskSession.taskId)}/driver-report/download`" not in app_js
    # Completion messages also carry their own center-stream download card, so
    # users do not depend on a rail poll or have to hunt elsewhere on the page.
    manual_js = _read("js/v2/driver_manual_analysis.js")
    assert "function agentMessageReportDownloadHtml" in app_js
    assert "metadata?.report_downloads" in app_js
    assert "reports.map((report)" in app_js
    assert 'data-agent-report-download' in app_js
    assert "renderReportDownload: agentMessageReportDownloadHtml" in app_js
    assert "renderReportDownload(message)" in manual_js
    assert (
        "${renderModelDelivery(message)}${renderTables(message)}"
        "${renderReportDownload(message)}"
    ) in manual_js


def test_agent_message_report_download_renders_every_report_link():
    app_js = _read("app.js")
    function_source = _slice_function(app_js, "function agentMessageReportDownloadHtml")
    script = f"""
      import assert from "node:assert/strict";
      import {{ safeSameOriginApiHref }} from "./marvis/static/js/url-safety.js";
      const escapeHtml = (value) => String(value);
      const taskSession = {{ task: {{ task_type: "modeling" }} }};
      {function_source}
      const html = agentMessageReportDownloadHtml({{
        metadata: {{
          report_download: {{
            label: "legacy primary",
            download_url: "/api/tasks/task-1/driver-report/download",
          }},
          report_downloads: [
            {{
              label: "下载 lgb 模型报告",
              download_url: "/api/tasks/task-1/driver-reports/id-lgb/download",
              recipe: "lgb",
              experiment_id: "exp-lgb",
              detail_label: "实验 1 · LightGBM",
            }},
            {{
              label: "下载 xgb 模型报告",
              download_url: "/api/tasks/task-1/driver-reports/id-xgb/download",
              recipe: "xgb",
              experiment_id: "exp-xgb",
            }},
          ],
        }},
      }});
      assert.equal((html.match(/data-agent-report-download/g) || []).length, 2);
      assert.equal(html.includes("/driver-reports/id-lgb/download"), true);
      assert.equal(html.includes("/driver-reports/id-xgb/download"), true);
      assert.equal(html.includes("legacy primary"), false);
      assert.equal(html.includes("已生成 2 份分析报告"), true);
      // Readable detail; the experiment id is only a tooltip.
      assert.equal(html.includes(">实验 1 · LightGBM<"), true);
      assert.equal(html.includes("lgb · exp-lgb"), false);
      assert.equal(html.includes('title="exp-lgb"'), true);

      taskSession.task = {{ task_type: "portfolio" }};
      const historical = agentMessageReportDownloadHtml({{
        metadata: {{
          report_download: {{
            label: "下载模型开发报告",
            download_url: "/api/tasks/task-1/driver-report/download",
          }},
        }},
      }});
      assert.equal(historical.includes("下载组合分析报告"), true);
      assert.equal(historical.includes("下载模型开发报告"), false);
    """
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_STATIC.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def test_delivery_created_tasks_are_named_without_artifact_ids():
    app_js = _read("app.js")
    script = f"""
      import assert from "node:assert/strict";
      const isValidationBatchTask = () => false;
      {_slice_function(app_js, "function taskDisplayName")}
      {_slice_function(app_js, "function displayModelVersion")}
      const artifact = "artifact_5995" + "a".repeat(28);
      assert.equal(taskDisplayName({{ model_name: "贷前A卡 建模", model_version: artifact }}), "贷前A卡 建模 · 交付验证");
      assert.equal(
        taskDisplayName({{ model_name: "贷前A卡 建模", model_version: `${{artifact}}-challenger-backtest` }}),
        "贷前A卡 建模 · 挑战者回测",
      );
      assert.equal(taskDisplayName({{ model_name: "A卡", model_version: "v2" }}), "A卡 · v2");
    """
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_STATIC.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )


def test_multi_report_download_labels_name_experiments_readably():
    from marvis.api_report_helpers import driver_report_download_metadata

    items = driver_report_download_metadata(
        plan_id="plan-1",
        task_id="task-1",
        step_id="reports",
        output={"reports": [
            {"experiment_id": "experiment_" + "a" * 32, "recipe": "lr", "report_path": "/tmp/a.xlsx"},
            {"experiment_id": "experiment_" + "b" * 32, "recipe": "lr", "report_path": "/tmp/b.xlsx"},
        ]},
        default_label="下载模型开发报告",
    )
    assert [item["label"] for item in items] == [
        "下载实验 1 · 逻辑回归模型报告",
        "下载实验 2 · 逻辑回归模型报告",
    ]
    assert [item["detail_label"] for item in items] == ["实验 1 · 逻辑回归", "实验 2 · 逻辑回归"]
    assert all(item["experiment_id"].startswith("experiment_") for item in items)
