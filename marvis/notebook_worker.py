"""Isolated notebook execution worker."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

from marvis.notebooks import notebook_run_result_to_dict, run_notebook


def worker_main() -> None:
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
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    worker_main()
