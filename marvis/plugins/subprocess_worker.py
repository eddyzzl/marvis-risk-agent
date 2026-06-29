from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import errno
from io import StringIO
import importlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import random
import socket
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

    resource_limits = _apply_resource_limits(
        job.get("memory_limit_mb"),
        cpu_seconds=job.get("cpu_limit_seconds"),
        file_size_mb=job.get("file_size_limit_mb"),
    )
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
            "resource_limits": resource_limits,
        })
        _hard_exit(1)
    except Exception as exc:
        payload = {
            "ok": False,
            "error_kind": "execution",
            "error": str(exc),
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "traceback": traceback.format_exc(),
            "resource_limits": resource_limits,
        }
        detail = _structured_error_detail(exc)
        if detail is not None:
            payload["error_kind"] = str(detail.get("kind") or "execution")
            payload["error_detail"] = detail
        _emit(payload)
        _hard_exit(1)

    _emit({
        "ok": True,
        "output": output,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "resource_limits": resource_limits,
    })
    _hard_exit(0)


def _run_tool(job: dict) -> dict:
    _install_network_guard(job.get("side_effects") or [])
    for path in job.get("plugin_paths") or []:
        path_text = str(path)
        if path_text and path_text not in sys.path:
            sys.path.insert(0, path_text)
    module = _load_module(job)
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


def _install_network_guard(side_effects: list[str]) -> None:
    allowed = set(str(item) for item in side_effects)
    if "network:optional" in allowed or "llm" in allowed:
        return

    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class _GuardedSocket(original_socket):
        def connect(self, address):
            _assert_local_address(address)
            return super().connect(address)

        def connect_ex(self, address):
            try:
                _assert_local_address(address)
            except PermissionError:
                return errno.EACCES
            return super().connect_ex(address)

        def sendto(self, data, *args):
            if args:
                address = args[-1]
                _assert_local_address(address)
            return super().sendto(data, *args)

    def _guarded_create_connection(address, *args, **kwargs):
        _assert_local_address(address)
        return original_create_connection(address, *args, **kwargs)

    socket.socket = _GuardedSocket
    socket.create_connection = _guarded_create_connection


def _assert_local_address(address) -> None:
    if not isinstance(address, tuple) or not address:
        return
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode("utf-8", errors="ignore")
    host_text = str(host or "").strip().lower()
    if host_text in {"localhost", "ip6-localhost"}:
        return
    try:
        ip = ipaddress.ip_address(host_text)
    except ValueError as exc:
        raise PermissionError("network access requires network:optional or llm side_effect") from exc
    if not ip.is_loopback:
        raise PermissionError("network access requires network:optional or llm side_effect")


def _load_module(job: dict):
    module_path = job.get("module_path")
    if module_path:
        path = Path(str(module_path))
        module_name = f"_marvis_adhoc_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(job["module"])


def _apply_resource_limits(
    memory_mb: int | None,
    *,
    cpu_seconds: int | None = None,
    file_size_mb: int | None = None,
) -> dict:
    meta = {
        "memory_limit_mb": None if memory_mb is None else int(memory_mb),
        "memory_limit_applied": False,
        "cpu_limit_seconds": None if cpu_seconds is None else int(cpu_seconds),
        "cpu_limit_applied": False,
        "file_size_limit_mb": None if file_size_mb is None else int(file_size_mb),
        "file_size_limit_applied": False,
        "degraded": False,
        "error": None,
        "errors": [],
    }
    if memory_mb is None and cpu_seconds is None and file_size_mb is None:
        return meta
    try:
        import resource
    except ImportError as exc:
        meta["degraded"] = True
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return meta

    if memory_mb is not None:
        memory_limit = int(memory_mb) * 1024 * 1024
        if _apply_one_limit(resource, "RLIMIT_DATA", memory_limit, meta, "memory_data"):
            meta["memory_limit_applied"] = True
        if _apply_one_limit(resource, "RLIMIT_AS", memory_limit, meta, "memory_as"):
            meta["memory_limit_applied"] = True
    if cpu_seconds is not None:
        if _apply_one_limit(resource, "RLIMIT_CPU", int(cpu_seconds), meta, "cpu"):
            meta["cpu_limit_applied"] = True
    if file_size_mb is not None:
        file_limit = int(file_size_mb) * 1024 * 1024
        if _apply_one_limit(resource, "RLIMIT_FSIZE", file_limit, meta, "file_size"):
            meta["file_size_limit_applied"] = True

    if meta["errors"]:
        meta["degraded"] = True
        meta["error"] = "; ".join(meta["errors"])
    return meta


def _apply_one_limit(resource_module, limit_name: str, limit: int, meta: dict, label: str) -> bool:
    kind = getattr(resource_module, limit_name, None)
    if kind is None:
        meta["errors"].append(f"{label}: unsupported")
        return False
    try:
        current_soft, current_hard = resource_module.getrlimit(kind)
        hard_infinity = current_hard == getattr(resource_module, "RLIM_INFINITY", -1)
        new_hard = limit if hard_infinity else current_hard
        new_soft = min(limit, new_hard)
        resource_module.setrlimit(kind, (new_soft, new_hard))
    except (OSError, ValueError) as exc:
        meta["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
        return False
    return True


def _structured_error_detail(exc: BaseException) -> dict | None:
    """Return a JSON-serializable structured payload for errors that expose ``to_detail()``.

    Lets typed errors (e.g. NanLabelNotConfirmedError) carry diagnostics across the
    subprocess boundary as structured data instead of free text.
    """
    to_detail = getattr(exc, "to_detail", None)
    if not callable(to_detail):
        return None
    try:
        detail = to_detail()
    except Exception:
        return None
    if not isinstance(detail, dict):
        return None
    try:
        json.dumps(detail)
    except (TypeError, ValueError):
        return None
    return detail


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
