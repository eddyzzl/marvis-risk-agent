import sys
from pathlib import Path

from marvis.db import PluginRepository, init_db
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner


def _runner(tmp_path):
    runner, _repo = _runtime(tmp_path)
    return runner


def _runtime(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = PluginRepository(db_path)
    registry = PluginRegistry(repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(registry),
        repo,
        python_executable=sys.executable,
        datasets_root=tmp_path / "datasets",
        workspace=tmp_path / "workspace",
    )
    return runner, repo


def test_tool_runner_invokes_sample_echo_in_subprocess(tmp_path):
    runner = _runner(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    assert result.ok is True
    assert result.output == {"echoed": "hi"}
    assert result.error is None
    assert result.duration_ms >= 0


def test_tool_runner_records_invocation_audit(tmp_path):
    runner, repo = _runtime(tmp_path)

    result = runner.invoke(ToolRef("_sample", "echo"), {"message": "hi"}, task_id="task-1")

    audits = repo.list_audit(kind="tool.invoke")
    assert result.ok is True
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "_sample.echo"
    assert audits[0]["outcome"] == "succeeded"
    assert audits[0]["inputs_hash"]


def test_tool_runner_derives_seed_for_stochastic_tools(tmp_path):
    runner = _runner(tmp_path)

    first = runner.invoke(ToolRef("_sample", "random"), {"key": "a"}, task_id="task-1")
    second = runner.invoke(ToolRef("_sample", "random"), {"key": "a"}, task_id="task-1")

    assert first.ok is True
    assert second.ok is True
    assert first.output == second.output
    assert isinstance(first.output["seed"], int)


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


def test_tool_runner_invokes_adhoc_module_in_subprocess_and_audits_draft(tmp_path):
    runner, repo = _runtime(tmp_path)
    module_path = tmp_path / "draft_calc.py"
    module_path.write_text(
        "def calc_margin(inputs, ctx):\n"
        "    return {'margin': inputs['revenue'] - inputs['cost'], 'task_id': ctx.task_id}\n",
        encoding="utf-8",
    )

    result = runner.invoke_adhoc(
        module=module_path,
        entrypoint="calc_margin",
        inputs={"revenue": 10, "cost": 3},
        input_schema={
            "type": "object",
            "properties": {"revenue": {"type": "number"}, "cost": {"type": "number"}},
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"margin": {"type": "number"}, "task_id": {"type": "string"}},
            "required": ["margin", "task_id"],
            "additionalProperties": False,
        },
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert result.ok is True, result.error
    assert result.output == {"margin": 7, "task_id": "task-1"}
    audits = repo.list_audit(kind="draft.invoke")
    assert len(audits) == 1
    assert audits[0]["target_ref"] == "draft.calc_margin"
    assert audits[0]["outcome"] == "succeeded"


def test_tool_runner_adhoc_validates_input_and_output_schema(tmp_path):
    runner = _runner(tmp_path)
    module_path = tmp_path / "draft_bad.py"
    module_path.write_text(
        "def bad_output(inputs, ctx):\n"
        "    return {'wrong': True}\n",
        encoding="utf-8",
    )
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    }
    output_schema = {
        "type": "object",
        "properties": {"echoed": {"type": "string"}},
        "required": ["echoed"],
        "additionalProperties": False,
    }

    missing = runner.invoke_adhoc(
        module=module_path,
        entrypoint="bad_output",
        inputs={},
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )
    mismatch = runner.invoke_adhoc(
        module=module_path,
        entrypoint="bad_output",
        inputs={"message": "hi"},
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=10,
        task_id="task-1",
        mode="draft",
    )

    assert missing.ok is False
    assert missing.error_kind == "schema"
    assert mismatch.ok is False
    assert mismatch.error_kind == "schema"
    assert "echoed" in mismatch.error
