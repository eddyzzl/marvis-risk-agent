from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from marvis.agent.renderers import render_tool_output
from test_strategy_apply_tool import _runtime_fixture
from test_strategy_dsl_delivery_tools import _inputs, _run


def _trusted_artifacts(runtime, output: dict) -> dict[str, dict]:
    names = ("python", "sql", "strategy_json", "equivalence_json")
    return {
        name: runtime.task_artifacts.get_for_task(
            output["task_id"],
            artifact["artifact_id"],
        )
        for name, artifact in zip(names, output["artifacts"], strict=True)
    }


@pytest.mark.parametrize(
    ("maximum_rows", "expected_sample_count", "expected_status"),
    [
        (2, 2, "bounded"),
        (4096, 3, "full"),
    ],
)
def test_delivery_renderer_shows_exact_scope_and_four_pinned_downloads(
    tmp_path: Path,
    maximum_rows: int,
    expected_sample_count: int,
    expected_status: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    request["maximum_equivalence_rows"] = maximum_rows
    output, runtime = _run(fixture, request)

    text, tables = render_tool_output(
        "export_strategy_delivery",
        output,
        trusted_task_id=output["task_id"],
        trusted_inputs=request,
        trusted_artifacts=_trusted_artifacts(runtime, output),
    )

    assert output["delivery_id"] in text
    assert f"sample_count **{expected_sample_count}**" in text
    assert "source_row_count **3**" in text
    assert f"（**{expected_status}**）" in text
    assert "**offline-only**" in text
    assert (
        "**not_applied=true / not_adopted=true / not_deployed=true**"
        in text
    )
    assert "未应用、未采纳、未部署" in text
    for label in ("Python", "DuckDB SQL", "Strategy JSON", "Equivalence JSON"):
        assert f"[{label}]" in text
    assert text.count("](/api/tasks/") == 4
    for artifact in output["artifacts"]:
        assert artifact["download_url"] in text
        assert (
            f"expected_content_hash={artifact['content_hash']}"
            in artifact["download_url"]
        )
    assert tables == []


def test_delivery_renderer_fails_closed_on_unpinned_download(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    output, runtime = _run(fixture)
    forged = deepcopy(output)
    forged["artifacts"][0]["download_url"] = forged["artifacts"][0][
        "download_url"
    ].split("?", 1)[0]

    text, tables = render_tool_output(
        "export_strategy_delivery",
        forged,
        trusted_task_id=output["task_id"],
        trusted_inputs=_inputs(fixture),
        trusted_artifacts=_trusted_artifacts(runtime, output),
    )

    assert "策略交付结果完整性校验失败" in text
    assert "/task-artifacts/" not in text
    assert output["delivery_id"] not in text
    assert tables == []


def test_delivery_renderer_rejects_coherent_task_and_url_drift(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, runtime = _run(fixture, request)
    forged = deepcopy(output)
    forged["task_id"] = "another-task"
    for artifact in forged["artifacts"]:
        artifact["download_url"] = artifact["download_url"].replace(
            f"/tasks/{output['task_id']}/",
            "/tasks/another-task/",
        )

    text, tables = render_tool_output(
        "export_strategy_delivery",
        forged,
        trusted_task_id=output["task_id"],
        trusted_inputs=request,
        trusted_artifacts=_trusted_artifacts(runtime, output),
    )

    assert "策略交付结果完整性校验失败" in text
    assert "/tasks/another-task/" not in text
    assert output["delivery_id"] not in text
    assert tables == []


def test_delivery_renderer_rejects_wrong_kind_or_reused_registry_rows(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, runtime = _run(fixture, request)
    trusted = _trusted_artifacts(runtime, output)
    trusted["sql"] = trusted["python"]

    text, tables = render_tool_output(
        "export_strategy_delivery",
        output,
        trusted_task_id=output["task_id"],
        trusted_inputs=request,
        trusted_artifacts=trusted,
    )

    assert "策略交付结果完整性校验失败" in text
    assert output["delivery_id"] not in text
    assert "/task-artifacts/" not in text
    assert tables == []
