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

    # The rail substep row no longer renders the download BUTTON — only a
    # "报告已就绪" status badge + a lightweight locate entry.
    substep_body = _slice_function(plan_rail_js, "function planSubstepHtml")
    assert 'data-driver-report-download="1"' not in substep_body
    assert "plan-step-ready" in substep_body
    assert 'data-plan-report-locate="1"' in substep_body
    assert "build_report_bundle_v2" in substep_body
    assert 'data-plan-report-locate="strategy"' in substep_body
    assert 'openDriverActionCard("strategy-artifacts")' not in substep_body

    # the handler navigates to the driver-report download endpoint
    assert "function handleDriverReportDownloadClick" in app_js
    assert "`/api/tasks/${encodeURIComponent(selectedTaskId)}/driver-report/download`" in app_js
    assert "`api/tasks/${encodeURIComponent(selectedTaskId)}/driver-report/download`" not in app_js
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
      const escapeHtml = (value) => String(value);
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
    """
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_STATIC.parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
