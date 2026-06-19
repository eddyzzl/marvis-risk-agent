import sys
from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner


def _runner(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(registry, packs_root)
    return ToolRunner(
        ToolRegistry(registry),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )


def test_tool_runner_invokes_sample_echo_in_subprocess(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is True
    assert result.output == {"echoed": "hi"}
    assert result.error is None
    assert result.duration_ms >= 0


def test_tool_runner_returns_schema_error_before_worker(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "schema"
    assert "message" in result.error


def test_tool_runner_converts_tool_exception_to_execution_error(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "fail"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "execution"
    assert "sample failure" in result.error
    assert "RuntimeError" in result.stderr_tail


def test_tool_runner_rejects_output_schema_mismatch(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "bad_output"), {}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "schema"
    assert "echoed" in result.error


def test_tool_runner_kills_timed_out_worker(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "sleep"), {"seconds": 2}, task_id="task-1")

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert "timed out" in result.error
