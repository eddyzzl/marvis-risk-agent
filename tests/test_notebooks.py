import json
from pathlib import Path
import subprocess
import threading
import time

import nbformat
import psutil
import pytest

from marvis.notebook_cancellation import NotebookCancellationToken
from marvis.notebooks import (
    _build_step_events,
    _notebook_worker_env,
    _parse_notebook_worker_result,
    _record_cell_complete,
    _record_cell_start,
    _worker_protocol_error_message,
    AppendedCellExecutionPolicy,
    NOTEBOOK_RESULT_SENTINEL,
    NotebookExecutionSession,
    prepare_execution_notebook_v3,
    run_notebook,
)
from marvis.notebook_steps import notebook_step_plan


def test_parse_worker_result_finds_sentinel_amid_kernel_chatter():
    # Windows kernels print TCP/zmq chatter to the worker's stdout; the sentinel
    # line must still be recovered even when it is NOT the last line.
    payload = {"ok": True, "result": {"succeeded": True}}
    stdout = "\n".join([
        "[IPKernelApp] WARNING | Kernel is running over TCP without encryption.",
        NOTEBOOK_RESULT_SENTINEL + json.dumps(payload),
        "[IPKernelApp] WARNING | Parent appears to have exited, shutting down.",
    ])
    assert _parse_notebook_worker_result(stdout) == payload


def test_parse_worker_result_legacy_bare_last_line_still_parsed():
    payload = {"ok": False, "error": "boom"}
    assert _parse_notebook_worker_result(json.dumps(payload)) == payload


def test_parse_worker_result_returns_none_on_pure_chatter():
    assert _parse_notebook_worker_result("[IPKernelApp] WARNING | over TCP\nshutting down") is None
    assert _parse_notebook_worker_result("") is None


def test_worker_emit_output_round_trips_through_parser():
    # The worker prefixes with the sentinel; the host parser must read it back.
    from marvis.notebook_worker import _emit as worker_emit  # noqa: PLC0415
    import io

    captured = io.BytesIO()

    class _Stdout:
        buffer = captured

    original = __import__("sys").stdout
    try:
        __import__("sys").stdout = _Stdout()
        worker_emit({"ok": True, "result": {"succeeded": True}})
    finally:
        __import__("sys").stdout = original

    text = captured.getvalue().decode("utf-8")
    assert text.startswith(NOTEBOOK_RESULT_SENTINEL)
    assert _parse_notebook_worker_result(text) == {"ok": True, "result": {"succeeded": True}}


def test_worker_protocol_error_message_surfaces_stderr_tail():
    stderr = "Traceback (most recent call last):\nModuleNotFoundError: No module named 'sklearn_pandas'"
    message = _worker_protocol_error_message(1, stderr)
    assert "exit code 1" in message
    assert "sklearn_pandas" in message
    # No stderr -> just the base message, no dangling separator.
    assert _worker_protocol_error_message(1, "") == "worker returned invalid protocol with exit code 1"


def test_run_notebook_executes_relative_to_notebook_directory(tmp_path: Path):
    notebook_dir = tmp_path / "notebooks" / "submitted"
    notebook_dir.mkdir(parents=True)
    notebook_path = notebook_dir / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    output_name = "relative-output-task-5.txt"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                f"from pathlib import Path\nPath({output_name!r}).write_text('ok')"
            )
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    result = run_notebook(notebook_path, executed_path, log_path, timeout=60)

    assert result.succeeded is True
    assert (notebook_dir / output_name).read_text(encoding="utf-8") == "ok"
    assert not (tmp_path / output_name).exists()


def test_run_notebook_isolated_executes_in_worker_process(tmp_path: Path):
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    notebook_path = notebook_dir / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    output_name = "isolated-output.txt"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                f"Path({output_name!r}).write_text('ok', encoding='utf-8')"
            )
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    result = run_notebook(notebook_path, executed_path, log_path, timeout=60, isolated=True)

    assert result.succeeded is True
    assert result.resource_usage is not None
    assert result.resource_usage["subprocess_isolated"] is True
    assert (notebook_dir / output_name).read_text(encoding="utf-8") == "ok"
    assert executed_path.exists()
    assert log_path.read_text(encoding="utf-8") == "succeeded\n"


def test_notebook_worker_env_strips_host_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("PYTHONPATH", "/tmp/marvis")

    env = _notebook_worker_env()

    assert env["PATH"] == "/usr/bin"
    assert env["PYTHONPATH"] == "/tmp/marvis"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_run_notebook_passes_env_allowlist_to_kernel_client(tmp_path: Path, monkeypatch):
    # TST-7: the live interactive-modeling kernel previously inherited the
    # full host os.environ with no filtering, unlike the isolated batch
    # worker and the plugin/draft subprocess worker which both already use
    # an env allowlist. execute_notebook must now pass the same
    # _notebook_worker_env() allowlist through to NotebookClient.execute so
    # jupyter_client launches the kernel subprocess with a filtered env
    # (sensitive host secrets such as API keys must not be visible to
    # user-authored notebook code).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-should-not-reach-kernel")
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(nbformat.v4.new_notebook(), notebook_path)
    captured = {}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name):
            pass

        def execute(self, *, cwd, env=None):
            captured["env"] = env

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    result = run_notebook(notebook_path, executed_path, log_path, timeout=60)

    assert result.succeeded is True
    assert captured["env"] is not None
    assert "OPENAI_API_KEY" not in captured["env"]
    assert captured["env"] == _notebook_worker_env()


def test_run_notebook_uses_configured_kernel_name(tmp_path: Path, monkeypatch):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(nbformat.v4.new_notebook(), notebook_path)
    captured = {}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name):
            captured["timeout"] = timeout
            captured["kernel_name"] = kernel_name

        def execute(self, *, cwd, env=None):
            captured["cwd"] = cwd

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        kernel_name="marvis-kernel",
    )

    assert result.succeeded is True
    assert captured["timeout"] == 60
    assert captured["kernel_name"] == "marvis-kernel"
    assert captured["cwd"] == str(tmp_path)


def test_execution_session_close_falls_back_to_kernel_manager_shutdown():
    class FakeKernelManager:
        def __init__(self):
            self.calls = []

        def shutdown_kernel(self, *, now):
            self.calls.append(now)

    class FakeClient:
        def __init__(self):
            self.km = FakeKernelManager()

    session = object.__new__(NotebookExecutionSession)
    session.closed = False
    session.client = FakeClient()

    session.close()

    assert session.closed is True
    assert session.client.km.calls == [True]


def test_execution_session_close_falls_back_when_private_cleanup_fails():
    class FakeKernelManager:
        def __init__(self):
            self.calls = []

        def shutdown_kernel(self, *, now):
            self.calls.append(now)

    class FakeClient:
        def __init__(self):
            self.km = FakeKernelManager()

        def _cleanup_kernel(self):
            raise RuntimeError("cleanup unavailable")

    session = object.__new__(NotebookExecutionSession)
    session.closed = False
    session.client = FakeClient()

    session.close()

    assert session.client.km.calls == [True]


def test_run_notebook_uses_explicit_execution_cwd(tmp_path: Path, monkeypatch):
    notebook_path = tmp_path / "execution" / "prepared.ipynb"
    notebook_path.parent.mkdir()
    executed_path = tmp_path / "execution" / "executed.ipynb"
    log_path = tmp_path / "execution" / "run.log"
    source_dir = tmp_path / "submitted" / "input"
    source_dir.mkdir(parents=True)
    nbformat.write(nbformat.v4.new_notebook(), notebook_path)
    captured = {}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            pass

        def execute(self, *, cwd, env=None):
            captured["cwd"] = cwd

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        execution_cwd=source_dir,
    )

    assert result.succeeded is True
    assert captured["cwd"] == str(source_dir)


def test_run_notebook_streams_step_progress_file(tmp_path: Path, monkeypatch):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    progress_path = tmp_path / "notebook_steps.json"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 数据准备"),
            nbformat.v4.new_code_cell("load_data()"),
            nbformat.v4.new_markdown_cell("# 模型训练"),
            nbformat.v4.new_code_cell("fit_model()"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    progress_snapshots = []

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.notebook = notebook
            self.callbacks = callbacks

        def execute(self, *, cwd, env=None):
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            self.callbacks["on_cell_start"](cell=self.notebook.cells[1], cell_index=1)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            self.callbacks["on_cell_executed"](cell=self.notebook.cells[1], cell_index=1)
            self.callbacks["on_cell_complete"](cell=self.notebook.cells[1], cell_index=1)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            self.callbacks["on_cell_start"](cell=self.notebook.cells[3], cell_index=3)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        progress_path=progress_path,
    )

    assert result.succeeded is True
    assert [step["status"] for step in progress_snapshots[0]["steps"]] == [
        "pending",
        "pending",
    ]
    assert [step["status"] for step in progress_snapshots[1]["steps"]] == [
        "running",
        "pending",
    ]
    assert [step["status"] for step in progress_snapshots[2]["steps"]] == [
        "running",
        "pending",
    ]
    assert [step["status"] for step in progress_snapshots[3]["steps"]] == [
        "succeeded",
        "running",
    ]
    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert final_progress["steps"][0]["title"] == "数据准备"
    assert final_progress["steps"][1]["title"] == "模型训练"


def test_run_notebook_defers_step_elapsed_until_next_cell_or_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    progress_path = tmp_path / "notebook_steps.json"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 数据准备"),
            nbformat.v4.new_code_cell("load_data()"),
            nbformat.v4.new_markdown_cell("# 模型训练"),
            nbformat.v4.new_code_cell("fit_model()"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    progress_snapshots = []
    clock = {"value": "2026-05-25T00:00:00+00:00"}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.notebook = notebook
            self.callbacks = callbacks

        def execute(self, *, cwd, env=None):
            self.callbacks["on_cell_start"](cell=self.notebook.cells[1], cell_index=1)
            self.callbacks["on_cell_executed"](cell=self.notebook.cells[1], cell_index=1)
            self.callbacks["on_cell_complete"](cell=self.notebook.cells[1], cell_index=1)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            clock["value"] = "2026-05-25T00:00:10+00:00"
            self.callbacks["on_cell_start"](cell=self.notebook.cells[3], cell_index=3)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            self.callbacks["on_cell_executed"](cell=self.notebook.cells[3], cell_index=3)
            self.callbacks["on_cell_complete"](cell=self.notebook.cells[3], cell_index=3)
            clock["value"] = "2026-05-25T00:00:15+00:00"

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)
    monkeypatch.setattr("marvis.notebooks._utc_now", lambda: clock["value"])

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        progress_path=progress_path,
    )

    assert result.succeeded is True
    assert progress_snapshots[0]["steps"][0]["status"] == "running"
    assert progress_snapshots[0]["steps"][0]["ended_at"] is None
    assert progress_snapshots[1]["steps"][0]["status"] == "succeeded"
    assert progress_snapshots[1]["steps"][0]["ended_at"] == "2026-05-25T00:00:10+00:00"
    assert progress_snapshots[1]["steps"][0]["elapsed_seconds"] == 10.0
    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert final_progress["steps"][1]["status"] == "succeeded"
    assert final_progress["steps"][1]["ended_at"] == "2026-05-25T00:00:15+00:00"
    assert final_progress["steps"][1]["elapsed_seconds"] == 5.0


def test_appended_system_cells_are_visible_before_execution(tmp_path: Path, monkeypatch):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    progress_path = tmp_path / "notebook_steps.json"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 建模"),
            nbformat.v4.new_code_cell("fit_model()"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.notebook = notebook
            self.callbacks = callbacks
            self.code_cells_executed = 0

        def execute_cell(self, cell, cell_index, execution_count):
            self.code_cells_executed = execution_count
            self.callbacks["on_cell_start"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_executed"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_complete"](cell=cell, cell_index=cell_index)

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
        progress_path=progress_path,
        allow_appended_execution=True,
        appended_execution_policy=AppendedCellExecutionPolicy(
            scope="test-repro-progress",
            reason="exercise injected reproducibility progress cells",
            allowed_marvis_kinds=("repro-pmml", "repro-compare"),
        ),
    )
    try:
        pmml_index = session.append_code_cell(
            "score_pmml()",
            metadata={"marvis": "repro-pmml"},
            record_progress=True,
        )
        compare_index = session.append_code_cell(
            "compare_scores()",
            metadata={"marvis": "repro-compare"},
            record_progress=True,
        )
        planned = json.loads(progress_path.read_text(encoding="utf-8"))

        assert [step["title"] for step in planned["steps"]] == [
            "建模",
            "PMML 打分",
            "分数一致性对比",
        ]
        assert [step["status"] for step in planned["steps"]] == [
            "pending",
            "pending",
            "pending",
        ]

        first = session.execute_existing_code_cell(
            pmml_index,
            log_path=tmp_path / "pmml.log",
            record_progress=True,
        )
        after_pmml = json.loads(progress_path.read_text(encoding="utf-8"))
        second = session.execute_existing_code_cell(
            compare_index,
            log_path=tmp_path / "compare.log",
            record_progress=True,
        )
    finally:
        session.close()

    assert first.succeeded is True
    assert second.succeeded is True
    assert [step["status"] for step in after_pmml["steps"]] == [
        "pending",
        "succeeded",
        "pending",
    ]
    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert [step["status"] for step in final_progress["steps"]] == [
        "pending",
        "succeeded",
        "succeeded",
    ]


def test_execute_existing_code_cell_uses_return_time_for_completed_elapsed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    progress_path = tmp_path / "notebook_steps.json"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# PMML 打分"),
            nbformat.v4.new_code_cell("score_pmml()"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    clock = {"value": "2026-05-25T00:00:00+00:00"}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.notebook = notebook
            self.callbacks = callbacks
            self.code_cells_executed = 0

        def execute_cell(self, cell, cell_index, execution_count):
            self.code_cells_executed = execution_count
            self.callbacks["on_cell_start"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_executed"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_complete"](cell=cell, cell_index=cell_index)
            clock["value"] = "2026-05-25T00:00:05+00:00"

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)
    monkeypatch.setattr("marvis.notebooks._utc_now", lambda: clock["value"])

    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
        progress_path=progress_path,
    )
    try:
        result = session.execute_existing_code_cell(
            1,
            log_path=tmp_path / "pmml.log",
            record_progress=True,
        )
    finally:
        session.close()

    assert result.succeeded is True
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["steps"][0]["ended_at"] == "2026-05-25T00:00:05+00:00"
    assert progress["steps"][0]["elapsed_seconds"] == 5.0


def test_execute_existing_code_cell_stays_running_until_execute_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    progress_path = tmp_path / "notebook_steps.json"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# PMML 打分"),
            nbformat.v4.new_code_cell("score_pmml()"),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    progress_snapshots = []
    clock = {"value": "2026-05-25T00:00:00+00:00"}

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.notebook = notebook
            self.callbacks = callbacks
            self.code_cells_executed = 0

        def execute_cell(self, cell, cell_index, execution_count):
            self.code_cells_executed = execution_count
            self.callbacks["on_cell_start"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_executed"](cell=cell, cell_index=cell_index)
            self.callbacks["on_cell_complete"](cell=cell, cell_index=cell_index)
            progress_snapshots.append(json.loads(progress_path.read_text(encoding="utf-8")))
            clock["value"] = "2026-05-25T00:00:05+00:00"

    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)
    monkeypatch.setattr("marvis.notebooks._utc_now", lambda: clock["value"])

    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
        progress_path=progress_path,
    )
    try:
        result = session.execute_existing_code_cell(
            1,
            log_path=tmp_path / "pmml.log",
            record_progress=True,
        )
    finally:
        session.close()

    assert result.succeeded is True
    assert progress_snapshots[0]["steps"][0]["status"] == "running"
    assert progress_snapshots[0]["steps"][0]["ended_at"] is None
    final_progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert final_progress["steps"][0]["status"] == "succeeded"
    assert final_progress["steps"][0]["ended_at"] == "2026-05-25T00:00:05+00:00"
    assert final_progress["steps"][0]["elapsed_seconds"] == 5.0


def test_cell_complete_marks_running_step_succeeded_when_executed_callback_is_missing():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 输出"),
            nbformat.v4.new_code_cell("print('done')"),
        ]
    )
    plan = notebook_step_plan(notebook)
    cell_events = {}

    _record_cell_start(cell_events, cell=notebook.cells[1], cell_index=1)
    _record_cell_complete(cell_events, cell=notebook.cells[1], cell_index=1)

    progress = _build_step_events(plan, cell_events)

    assert progress["steps"][0]["status"] == "succeeded"
    assert progress["cells"][0]["status"] == "succeeded"


def test_step_events_include_elapsed_time_for_running_and_completed_steps():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell("# 数据准备"),
            nbformat.v4.new_code_cell("load_data()"),
            nbformat.v4.new_markdown_cell("# 模型训练"),
            nbformat.v4.new_code_cell("fit_model()"),
        ]
    )
    plan = notebook_step_plan(notebook)
    cell_events = {
        1: {
            "cell_index": 1,
            "cell_type": "code",
            "status": "succeeded",
            "started_at": "2026-05-25T00:00:00+00:00",
            "ended_at": "2026-05-25T00:00:03+00:00",
        },
        3: {
            "cell_index": 3,
            "cell_type": "code",
            "status": "running",
            "started_at": "2026-05-25T00:00:10+00:00",
            "ended_at": None,
        },
    }

    progress = _build_step_events(plan, cell_events)

    assert progress["steps"][0]["started_at"] == "2026-05-25T00:00:00+00:00"
    assert progress["steps"][0]["ended_at"] == "2026-05-25T00:00:03+00:00"
    assert progress["steps"][0]["elapsed_seconds"] == 3.0
    assert progress["steps"][1]["started_at"] == "2026-05-25T00:00:10+00:00"
    assert progress["steps"][1]["ended_at"] is None
    assert isinstance(progress["steps"][1]["elapsed_seconds"], float)
    assert progress["steps"][1]["elapsed_seconds"] >= 0.0


def test_retried_system_step_progress_uses_latest_attempt_status():
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("prepare_old()"),
            nbformat.v4.new_code_cell("prepare_new()"),
        ]
    )
    notebook.cells[0].metadata["marvis"] = "metrics-prepare"
    notebook.cells[1].metadata["marvis"] = "metrics-prepare"
    plan = notebook_step_plan(notebook)
    cell_events = {
        0: {
            "cell_index": 0,
            "cell_type": "code",
            "status": "failed",
            "started_at": "2026-05-28T06:30:14+00:00",
            "ended_at": "2026-05-28T06:30:15+00:00",
            "exception_name": "TypeError",
        },
        1: {
            "cell_index": 1,
            "cell_type": "code",
            "status": "succeeded",
            "started_at": "2026-05-28T06:37:21+00:00",
            "ended_at": "2026-05-28T06:37:22+00:00",
            "exception_name": None,
        },
    }

    progress = _build_step_events(plan, cell_events)

    assert progress["steps"][0]["status"] == "succeeded"
    assert progress["steps"][0]["cell_indexes"] == [1]
    assert progress["steps"][0]["started_at"] == "2026-05-28T06:37:21+00:00"
    assert progress["steps"][0]["ended_at"] == "2026-05-28T06:37:22+00:00"


def test_run_notebook_preserves_artifacts_on_timeout(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("import time\ntime.sleep(2)")],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    result = run_notebook(notebook_path, executed_path, log_path, timeout=1)

    assert result.succeeded is False
    assert executed_path.exists()
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "failed" in log_text
    assert result.error_name in log_text
    assert "timeout" in log_text.lower() or "timed out" in log_text.lower()


def test_run_notebook_isolated_cell_timeout_preserves_artifacts(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("import time\ntime.sleep(10)")],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    result = run_notebook(notebook_path, executed_path, log_path, timeout=1, isolated=True)

    assert result.succeeded is False
    assert result.error_name == "CellTimeoutError"
    assert result.resource_usage is not None
    assert result.resource_usage["subprocess_isolated"] is True
    assert executed_path.exists()
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "CellTimeoutError" in log_text
    assert "timed out" in log_text


def test_run_notebook_isolated_parent_timeout_kills_worker_and_preserves_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("while True: pass")]),
        notebook_path,
    )
    killed = {"value": False}

    class FakeProcess:
        pid = 12345
        returncode = None

        def communicate(self, input=None, timeout=None):
            if input is not None and timeout is not None:
                raise subprocess.TimeoutExpired(["python"], timeout)
            return "partial stdout", "partial stderr"

        def poll(self):
            return None

    monkeypatch.setattr("marvis.notebooks.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr("marvis.notebooks._kill_process_tree", lambda process: killed.update(value=True))

    result = run_notebook(notebook_path, executed_path, log_path, timeout=1, isolated=True)

    assert result.succeeded is False
    assert result.error_name == "NotebookSubprocessTimeout"
    assert killed["value"] is True
    assert executed_path.exists()
    assert "NotebookSubprocessTimeout" in log_path.read_text(encoding="utf-8")


def test_prepare_execution_notebook_appends_generated_system_cells(tmp_path: Path):
    source_notebook = tmp_path / "source.ipynb"
    output_notebook = tmp_path / "prepared.ipynb"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell(
                    "\n".join(
                        [
                            "import pandas as pd",
                            "RMC_SAMPLE_DF = pd.DataFrame({'x': [1.0, 2.0], 'y': [0, 1]})",
                            "RMC_TARGET_COL = 'y'",
                            "RMC_ALGORITHM = 'lgb'",
                            "def RMC_SCORE_FN(df):",
                            "    return [0.1 for _ in range(len(df))]",
                        ]
                    )
                )
            ],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        ),
        source_notebook,
    )

    prepare_execution_notebook_v3(
        source_notebook=source_notebook,
        output_notebook=output_notebook,
        sample_path=tmp_path / "sample.csv",
        contract_meta_path=tmp_path / "runtime_contract.json",
        code_scores_path=tmp_path / "code_model_scores.csv",
        feature_importance_path=tmp_path / "feature_importance.csv",
        model_params_path=tmp_path / "model_params.json",
        extra_code_cells=[
            ("repro-pmml", "score_pmml()"),
            ("metrics-output", "write_metrics()"),
        ],
    )

    prepared = nbformat.read(output_notebook, as_version=4)

    assert [cell.metadata.get("marvis") for cell in prepared.cells] == [
        "head",
        None,
        "tail",
        "repro-pmml",
        "metrics-output",
    ]
    assert prepared.cells[-2].source == "score_pmml()"
    assert prepared.cells[-1].source == "write_metrics()"


def test_run_notebook_isolated_worker_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(nbformat.v4.new_notebook(cells=[]), notebook_path)

    class FakeProcess:
        pid = 12345
        returncode = 1

        def communicate(self, input=None, timeout=None):
            return json.dumps({"ok": False, "error": "worker boom"}) + "\n", ""

        def poll(self):
            return 1

    monkeypatch.setattr("marvis.notebooks.subprocess.Popen", lambda *args, **kwargs: FakeProcess())

    result = run_notebook(notebook_path, executed_path, log_path, timeout=1, isolated=True)

    assert result.succeeded is False
    assert result.error_name == "NotebookWorkerError"
    assert result.error_value == "worker boom"
    assert "worker boom" in log_path.read_text(encoding="utf-8")


def test_run_notebook_returns_cancelled_when_token_is_cancelled(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("x = 1")],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    cancellation_token = NotebookCancellationToken(task_id="task-1")
    cancellation_token.cancel()

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        cancellation_token=cancellation_token,
    )

    assert result.succeeded is False
    assert result.cancelled is True
    assert result.error_name == "NotebookCancelled"
    assert executed_path.exists()
    assert log_path.read_text(encoding="utf-8") == "cancelled\n"


def test_run_notebook_isolated_returns_cancelled_when_token_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(
        nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")]),
        notebook_path,
    )
    killed = {"value": False}

    class FakeProcess:
        pid = 12345
        returncode = None
        args = ["python", "-m", "marvis.notebook_worker"]

        def communicate(self, input=None, timeout=None):
            return "", ""

        def poll(self):
            return None

    monkeypatch.setattr("marvis.notebooks.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        "marvis.notebooks._kill_process_tree",
        lambda process: killed.update(value=True),
    )
    cancellation_token = NotebookCancellationToken(task_id="task-1")
    cancellation_token.cancel()

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        cancellation_token=cancellation_token,
        isolated=True,
    )

    assert result.succeeded is False
    assert result.cancelled is True
    assert result.error_name == "NotebookCancelled"
    assert killed["value"] is True
    assert executed_path.exists()
    assert log_path.read_text(encoding="utf-8") == "cancelled\n"


def test_run_notebook_stops_kernel_when_rss_exceeds_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("x = 'large allocation'")],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    calls = {
        "process_pid": None,
        "interrupts": 0,
        "shutdowns": [],
        "terminates": 0,
    }

    class FakeMemoryInfo:
        rss = 2 * 1024 * 1024

    class FakeProcess:
        def __init__(self, pid):
            calls["process_pid"] = pid

        def memory_info(self):
            return FakeMemoryInfo()

        def children(self, recursive):
            return []

        def terminate(self):
            calls["terminates"] += 1

        def kill(self):
            raise AssertionError("process should terminate without kill fallback")

    class FakePsutil:
        @staticmethod
        def Process(pid):
            return FakeProcess(pid)

        @staticmethod
        def wait_procs(targets, timeout):
            return list(targets), []

    class FakeKernelManager:
        def __init__(self):
            self.provisioner = type("FakeProvisioner", (), {"pid": 12345})()

        def interrupt_kernel(self):
            calls["interrupts"] += 1

        def shutdown_kernel(self, *, now):
            calls["shutdowns"].append(now)

    class FakeNotebookClient:
        def __init__(self, notebook, timeout, kernel_name, **callbacks):
            self.km = FakeKernelManager()

        def execute(self, *, cwd, env=None):
            deadline = time.monotonic() + 2.0
            while not calls["shutdowns"] and time.monotonic() < deadline:
                time.sleep(0.01)
            raise RuntimeError("kernel stopped")

    monkeypatch.setattr("marvis.notebooks.psutil", FakePsutil)
    monkeypatch.setattr("marvis.notebooks.NotebookClient", FakeNotebookClient)

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        memory_limit_mb=1,
        resource_poll_interval_seconds=0.01,
    )

    assert result.succeeded is False
    assert result.error_name == "NotebookResourceLimitExceeded"
    assert result.resource_usage is not None
    assert result.resource_usage["memory_limit_exceeded"] is True
    assert result.resource_usage["memory_limit_mb"] == 1
    assert result.resource_usage["peak_rss_mb"] == 2.0
    assert result.resource_usage["kernel_pid"] == 12345
    assert calls["process_pid"] == 12345
    assert calls["interrupts"] == 1
    assert calls["shutdowns"] == [True, True]
    assert calls["terminates"] == 1
    log_text = log_path.read_text(encoding="utf-8")
    assert "NotebookResourceLimitExceeded" in log_text
    assert "memory_limit_mb=1" in log_text
    assert executed_path.exists()


@pytest.mark.slow
def test_run_notebook_isolated_cancel_kills_real_kernel_process_tree(tmp_path: Path):
    """TST-4d: real kill, notebook isolation path. marvis/notebooks.py's
    isolated=True subprocess mechanism (_run_notebook_in_subprocess /
    _kill_process_tree) is a distinct code path from ToolRunner's worker kill
    (different PID source: a Jupyter kernel launched by nbclient inside the
    notebook worker process, not a direct Popen child). The existing coverage
    for this branch
    (test_run_notebook_isolated_returns_cancelled_when_token_is_cancelled)
    mocks subprocess.Popen and _kill_process_tree entirely, so it never
    proves an actual kernel process goes away.

    The notebook cell busy-loops forever with a long nbclient timeout (so
    nbclient's own cell-timeout never fires); a background thread cancels the
    NotebookCancellationToken shortly after the kernel starts, which drives
    _run_notebook_in_subprocess's NotebookCancelled branch and its real
    os.killpg-based _kill_process_tree call. Assert both the isolated worker
    process and the real Jupyter kernel process (the process tree's
    grandchild from this test's point of view) are gone afterwards -- proving
    the kill reaps the whole group, not just the immediate worker.
    """
    notebook_dir = tmp_path / "notebooks"
    notebook_dir.mkdir()
    notebook_path = notebook_dir / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    pid_file = notebook_dir / "kernel.pid"
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "import os, pathlib\n"
                f"pathlib.Path({pid_file.name!r}).write_text(str(os.getpid()))\n"
                "while True:\n"
                "    pass\n"
            )
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)

    cancellation_token = NotebookCancellationToken(task_id="task-1")

    def _cancel_once_kernel_started() -> None:
        deadline = time.monotonic() + 20.0
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        cancellation_token.cancel()

    canceller = threading.Thread(target=_cancel_once_kernel_started, daemon=True)
    canceller.start()

    result = run_notebook(
        notebook_path,
        executed_path,
        log_path,
        timeout=60,
        cancellation_token=cancellation_token,
        isolated=True,
    )
    canceller.join(timeout=5.0)

    assert result.succeeded is False
    assert result.cancelled is True
    assert result.error_name == "NotebookCancelled"
    assert result.resource_usage is not None
    worker_pid = result.resource_usage["worker_pid"]
    assert worker_pid is not None
    assert pid_file.exists(), "kernel never started; test setup is broken"
    kernel_pid = int(pid_file.read_text().strip())

    # INV-6: poll briefly for SIGKILL delivery, then require both the worker
    # and the real kernel grandchild to be gone -- a process-tree-wide kill,
    # not a single-process kill.
    deadline = time.monotonic() + 5.0
    worker_alive = True
    kernel_alive = True
    while time.monotonic() < deadline:
        worker_alive = psutil.pid_exists(worker_pid)
        kernel_alive = psutil.pid_exists(kernel_pid)
        if not worker_alive and not kernel_alive:
            break
        time.sleep(0.1)
    assert not worker_alive, f"isolated notebook worker pid {worker_pid} survived cancel kill"
    assert not kernel_alive, f"kernel pid {kernel_pid} survived process-tree kill"


def test_live_notebook_session_reuses_kernel_for_appended_cells(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    appended_log_path = tmp_path / "appended.log"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("live_value = 41")],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )
    nbformat.write(notebook, notebook_path)
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
        allow_appended_execution=True,
        appended_execution_policy=AppendedCellExecutionPolicy(
            scope="test-live-reuse",
            reason="exercise live kernel reuse in a unit test",
        ),
    )

    try:
        first = session.execute_notebook()
        second = session.execute_code_cell(
            "assert live_value == 41\nprint(f'live={live_value + 1}')",
            log_path=appended_log_path,
        )
    finally:
        session.close()

    assert first.succeeded is True
    assert second.succeeded is True
    executed = nbformat.read(executed_path, as_version=4)
    assert len(executed.cells) == 2
    assert executed.cells[-1].outputs[0]["text"] == "live=42\n"
    assert appended_log_path.read_text(encoding="utf-8") == "succeeded\n"


def test_live_notebook_appended_execution_requires_policy(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("x = 1")],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        ),
        notebook_path,
    )

    with pytest.raises(ValueError, match="requires an explicit policy"):
        NotebookExecutionSession(
            notebook_path=notebook_path,
            executed_path=executed_path,
            log_path=log_path,
            timeout=60,
            kernel_name="python3",
            allow_appended_execution=True,
        )


def test_live_notebook_appended_policy_rejects_unlisted_marvis_kind(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("x = 1")],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        ),
        notebook_path,
    )
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
        appended_execution_policy=AppendedCellExecutionPolicy(
            scope="test-restricted",
            reason="exercise policy rejection",
            allowed_marvis_kinds=("repro-pmml",),
        ),
    )

    try:
        with pytest.raises(RuntimeError, match="only allows marvis cell kinds"):
            session.append_code_cell(
                "x = 2",
                metadata={"marvis": "metrics-output"},
            )
    finally:
        session.close()


def test_live_notebook_session_rejects_appended_cells_by_default(tmp_path: Path):
    notebook_path = tmp_path / "source.ipynb"
    executed_path = tmp_path / "executed.ipynb"
    log_path = tmp_path / "run.log"
    nbformat.write(
        nbformat.v4.new_notebook(
            cells=[nbformat.v4.new_code_cell("x = 1")],
            metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        ),
        notebook_path,
    )
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=executed_path,
        log_path=log_path,
        timeout=60,
        kernel_name="python3",
    )

    try:
        with pytest.raises(RuntimeError, match="appended-cell execution is disabled"):
            session.execute_code_cell("x = 2")
    finally:
        session.close()
