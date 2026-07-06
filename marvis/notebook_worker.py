"""Isolated notebook execution worker."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

from marvis.notebooks import (
    NOTEBOOK_RESULT_SENTINEL,
    notebook_run_result_to_dict,
    run_notebook,
)


def _configure_windows_event_loop() -> None:
    # On Windows, Python 3.8+ defaults to the Proactor event loop, but
    # jupyter_client / pyzmq need the selector loop to start a kernel cleanly
    # (otherwise a noisy "Proactor event loop does not implement add_reader"
    # warning + a slower selector-thread fallback). Isolated to this subprocess.
    if not sys.platform.startswith("win"):
        return
    try:
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


def worker_main() -> None:
    _configure_windows_event_loop()
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        job = json.loads(raw)
    except Exception as exc:
        _emit({"ok": False, "error": f"bad job json: {exc}", "traceback": traceback.format_exc()})
        raise SystemExit(1)
    try:
        result = run_notebook(
            Path(job["notebook_path"]),
            Path(job["executed_path"]),
            Path(job["log_path"]),
            timeout=int(job.get("timeout") or 3600),
            kernel_name=str(job.get("kernel_name") or "python3"),
            progress_path=_optional_path(job.get("progress_path")),
            execution_cwd=_optional_path(job.get("execution_cwd")),
            memory_limit_mb=job.get("memory_limit_mb"),
            resource_poll_interval_seconds=float(job.get("resource_poll_interval_seconds") or 0.5),
            isolated=False,
        )
    except Exception as exc:
        _emit({
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise SystemExit(1)
    _emit({"ok": True, "result": notebook_run_result_to_dict(result)})
    raise SystemExit(0)


def _optional_path(value) -> Path | None:
    return None if value in (None, "") else Path(str(value))


def _emit(payload: dict) -> None:
    # Tag the result with the sentinel so the host can find it even when the
    # kernel prints TCP/zmq chatter onto this stdout (Windows), and write raw
    # UTF-8 bytes so we never depend on the platform's stdout text encoding
    # (cp936 on Chinese Windows) -- the host reads this pipe as UTF-8.
    line = NOTEBOOK_RESULT_SENTINEL + json.dumps(payload, ensure_ascii=False) + "\n"
    data = line.encode("utf-8")
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except (AttributeError, ValueError):
        sys.stdout.write(line)
        sys.stdout.flush()


if __name__ == "__main__":
    worker_main()
