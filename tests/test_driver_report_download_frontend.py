"""Plan-rail report-download affordance (drives the /driver-report/download endpoint).

The backend endpoint + 404/200 behavior is covered in test_feature_analysis_api;
this guards the frontend wiring (button + handler) since the rail is the only way a
user reaches a driver task's generated Excel report.
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def test_plan_rail_has_report_download_button_and_handler():
    app_js = _read("app.js")

    # a 下载报告 button sits inline on the producing report step row once it completes
    assert 'data-driver-report-download="1"' in app_js
    assert "plan-step-download" in app_js  # inline on the step row, not a floating rail button
    assert "下载报告" in app_js
    assert "generate_model_report" in app_js
    assert "generate_feature_report" in app_js

    # the handler navigates to the driver-report download endpoint
    assert "function handleDriverReportDownloadClick" in app_js
    assert "/driver-report/download" in app_js
