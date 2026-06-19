from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

from marvis.plugins.runner import ToolContext


def worker_main() -> None:
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        job = json.loads(raw)
    except Exception as exc:
        _emit({"ok": False, "error_kind": "protocol", "error": f"bad job json: {exc}"})
        _hard_exit(1)

    _apply_resource_limits(job.get("memory_limit_mb"))
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            output = _run_tool(job)
    except MemoryError as exc:
        _emit({
            "ok": False,
            "error_kind": "resource",
            "error": str(exc),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "traceback": traceback.format_exc(),
        })
        _hard_exit(0)
    except Exception as exc:
        _emit({
            "ok": False,
            "error_kind": "execution",
            "error": str(exc),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "traceback": traceback.format_exc(),
        })
        _hard_exit(0)

    _emit({
        "ok": True,
        "output": output,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
    })
    _hard_exit(0)


def _run_tool(job: dict) -> dict:
    for path in job.get("plugin_paths") or []:
        path_text = str(path)
        if path_text and path_text not in sys.path:
            sys.path.insert(0, path_text)
    module = importlib.import_module(job["module"])
    func = getattr(module, job["entrypoint"])
    ctx = ToolContext(
        task_id=str(job["task_id"]),
        seed=job.get("seed"),
        datasets_root=Path(job["datasets_root"]),
        workspace=Path(job["workspace"]),
    )
    if ctx.seed is not None:
        random.seed(ctx.seed)
        try:
            import numpy as np

            np.random.seed(ctx.seed)
        except Exception:
            pass
    result = func(job["inputs"], ctx)
    if not isinstance(result, dict):
        raise TypeError(f"tool must return dict, got {type(result).__name__}")
    return result


def _apply_resource_limits(memory_mb: int | None) -> None:
    if memory_mb is None:
        return
    try:
        import resource

        limit = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, OSError, ValueError):
        return


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _hard_exit(code: int) -> None:
    # Tool workers are isolated one-shot subprocesses. Some native data
    # libraries can leave interpreter shutdown waiting after the JSON protocol
    # line has been flushed; exit immediately so the parent can consume it.
    os._exit(code)


if __name__ == "__main__":
    worker_main()
