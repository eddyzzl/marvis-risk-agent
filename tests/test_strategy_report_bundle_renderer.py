"""Fail-closed Agent rendering for StrategyReportBundle V2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from marvis.agent.renderers import render_tool_output
from test_strategy_report_bundle_tools import _run, _setup


def test_report_renderer_shows_governed_identity_warnings_and_three_downloads(
    tmp_path: Path,
) -> None:
    output = _run(_setup(tmp_path))

    text, tables = render_tool_output("build_report_bundle_v2", output)

    assert output["report_id"] in text
    assert f"revision **{output['report_revision']}**" in text
    assert f"状态 **{output['status']}**" in text
    assert "未创建策略、未采纳、未部署或上线" in text
    for warning in output["warnings"]:
        assert warning in text
    assert text.count("](/api/tasks/") == 3
    assert "[JSON]" in text
    assert "[Markdown]" in text
    assert "[XLSX]" in text
    for artifact in output["artifacts"]:
        assert (
            f"expected_content_hash={artifact['content_hash']}"
            in text
        )
    assert tables == []


def test_report_renderer_fails_closed_on_forged_cached_output(
    tmp_path: Path,
) -> None:
    output = _run(_setup(tmp_path))
    forged = deepcopy(output)
    forged["report_revision"] += 1

    text, tables = render_tool_output("build_report_bundle_v2", forged)

    assert "结果完整性校验失败" in text
    assert output["report_id"] not in text
    assert "/task-artifacts/" not in text
    assert tables == []


def test_report_renderer_fails_closed_on_unknown_envelope_fields(
    tmp_path: Path,
) -> None:
    output = _run(_setup(tmp_path))
    output["caller_metric"] = 0.99

    text, tables = render_tool_output("build_report_bundle_v2", output)

    assert "结果完整性校验失败" in text
    assert "0.99" not in text
    assert tables == []


def test_report_renderer_fails_closed_on_unpinned_download_link(
    tmp_path: Path,
) -> None:
    output = _run(_setup(tmp_path))
    output["artifacts"][0]["download_url"] = output["artifacts"][0][
        "download_url"
    ].split("?", 1)[0]

    text, tables = render_tool_output("build_report_bundle_v2", output)

    assert "结果完整性校验失败" in text
    assert "/task-artifacts/" not in text
    assert tables == []
