from __future__ import annotations

import builtins
from contextlib import redirect_stderr, redirect_stdout
import errno
import io
from io import StringIO
import importlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import traceback

from marvis.plugins.contracts import PROTOCOL_VERSION, ToolContext


def worker_main() -> None:
    raw = sys.stdin.buffer.read().decode("utf-8")
    try:
        job = json.loads(raw)
    except Exception as exc:
        _emit({"ok": False, "error_kind": "protocol", "error": f"bad job json: {exc}"})
        _hard_exit(1)

    _check_protocol_version(job)

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
        "worker_protocol_version": PROTOCOL_VERSION,
    })
    _hard_exit(0)


def _check_protocol_version(job: dict) -> None:
    """ARCH-5: verify the host's protocol_version matches this worker's before
    doing any real work. A version mismatch means the job dict schema, result
    protocol shape, or guard semantics may have drifted between host and
    worker (e.g. execution_environment.python_executable points at a separate
    interpreter running a different marvis checkout) -- continuing would fail
    in confusing, hard-to-diagnose ways deeper in the run. Fail fast with a
    typed, Chinese-readable error instead."""
    host_version = job.get("protocol_version")
    if host_version == PROTOCOL_VERSION:
        return
    _emit({
        "ok": False,
        "error_kind": "protocol_version_mismatch",
        "error": (
            f"插件 worker 协议版本不匹配：宿主={host_version!r}，"
            f"worker={PROTOCOL_VERSION!r}；请确认 execution_environment 配置的 "
            f"python_executable 与宿主使用同一份 marvis 代码"
        ),
        "error_detail": {
            "kind": "protocol_version_mismatch",
            "host_protocol_version": host_version,
            "worker_protocol_version": PROTOCOL_VERSION,
        },
        "worker_protocol_version": PROTOCOL_VERSION,
    })
    _hard_exit(1)


def _run_tool(job: dict) -> dict:
    side_effects = [str(item) for item in (job.get("side_effects") or [])]
    plugin_paths = [Path(str(path)) for path in (job.get("plugin_paths") or []) if str(path)]
    _install_network_guard(side_effects)
    if not bool(job.get("builtin")):
        _install_process_guard(side_effects)
    if _should_install_file_guard(job):
        _install_file_guard(
            workspace=Path(job["workspace"]),
            datasets_root=Path(job["datasets_root"]),
            side_effects=side_effects,
            module_path=Path(str(job["module_path"])) if job.get("module_path") else None,
            plugin_paths=plugin_paths,
        )
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


def _should_install_file_guard(job: dict) -> bool:
    return bool(job.get("module_path")) or not bool(job.get("builtin"))


def _install_file_guard(
    *,
    workspace: Path,
    datasets_root: Path,
    side_effects: list[str],
    module_path: Path | None,
    plugin_paths: list[Path],
) -> None:
    sys.dont_write_bytecode = True
    read_roots = _guard_roots([
        *plugin_paths,
        *(_python_runtime_roots()),
        _project_root(),
        module_path.parent if module_path is not None else None,
        workspace if _has_read_effect(side_effects) else None,
        datasets_root if _has_read_effect(side_effects) else None,
    ])
    write_roots = _guard_roots([
        workspace if _has_write_effect(side_effects) else None,
        datasets_root if _has_write_effect(side_effects) else None,
    ])
    original_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open
    original_remove = os.remove
    original_unlink = os.unlink
    original_rmdir = os.rmdir
    original_mkdir = os.mkdir
    original_makedirs = os.makedirs
    original_rename = os.rename
    original_replace = os.replace
    original_link = os.link if hasattr(os, "link") else None
    original_symlink = os.symlink if hasattr(os, "symlink") else None
    original_path_open = Path.open
    original_path_read_text = Path.read_text
    original_path_read_bytes = Path.read_bytes
    original_path_write_text = Path.write_text
    original_path_write_bytes = Path.write_bytes
    original_path_touch = Path.touch
    original_path_unlink = Path.unlink
    original_path_mkdir = Path.mkdir
    original_path_rmdir = Path.rmdir
    original_path_rename = Path.rename
    original_path_replace = Path.replace
    original_path_glob = Path.glob
    original_path_rglob = Path.rglob
    original_shutil_copy = shutil.copy
    original_shutil_copy2 = shutil.copy2
    original_shutil_copyfile = shutil.copyfile
    original_shutil_move = shutil.move
    original_shutil_rmtree = shutil.rmtree

    def guarded_open(file, mode="r", *args, **kwargs):
        if _is_path_like(file):
            _assert_file_access(file, _access_from_mode(str(mode)), read_roots, write_roots)
        return original_open(file, mode, *args, **kwargs)

    def guarded_io_open(file, mode="r", *args, **kwargs):
        if _is_path_like(file):
            _assert_file_access(file, _access_from_mode(str(mode)), read_roots, write_roots)
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        _assert_file_access(path, _access_from_flags(int(flags)), read_roots, write_roots)
        return original_os_open(path, flags, *args, **kwargs)

    def guarded_remove(path, *args, **kwargs):
        _assert_file_access(path, "write", read_roots, write_roots)
        return original_remove(path, *args, **kwargs)

    def guarded_unlink(path, *args, **kwargs):
        _assert_file_access(path, "write", read_roots, write_roots)
        return original_unlink(path, *args, **kwargs)

    def guarded_rmdir(path, *args, **kwargs):
        _assert_file_access(path, "write", read_roots, write_roots)
        return original_rmdir(path, *args, **kwargs)

    def guarded_mkdir(path, *args, **kwargs):
        _assert_file_access(path, "write", read_roots, write_roots)
        return original_mkdir(path, *args, **kwargs)

    def guarded_makedirs(name, *args, **kwargs):
        _assert_file_access(name, "write", read_roots, write_roots)
        return original_makedirs(name, *args, **kwargs)

    def guarded_rename(src, dst, *args, **kwargs):
        _assert_file_access(src, "write", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_rename(src, dst, *args, **kwargs)

    def guarded_replace(src, dst, *args, **kwargs):
        _assert_file_access(src, "write", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_replace(src, dst, *args, **kwargs)

    def guarded_link(src, dst, *args, **kwargs):
        _assert_file_access(src, "read", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_link(src, dst, *args, **kwargs)

    def guarded_symlink(src, dst, *args, **kwargs):
        _assert_file_access(src, "read", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_symlink(src, dst, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        _assert_file_access(self, _access_from_mode(str(mode)), read_roots, write_roots)
        return original_path_open(self, mode, *args, **kwargs)

    def guarded_path_read_text(self, *args, **kwargs):
        _assert_file_access(self, "read", read_roots, write_roots)
        return original_path_read_text(self, *args, **kwargs)

    def guarded_path_read_bytes(self, *args, **kwargs):
        _assert_file_access(self, "read", read_roots, write_roots)
        return original_path_read_bytes(self, *args, **kwargs)

    def guarded_path_write_text(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_write_text(self, *args, **kwargs)

    def guarded_path_write_bytes(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_write_bytes(self, *args, **kwargs)

    def guarded_path_touch(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_touch(self, *args, **kwargs)

    def guarded_path_unlink(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_unlink(self, *args, **kwargs)

    def guarded_path_mkdir(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_mkdir(self, *args, **kwargs)

    def guarded_path_rmdir(self, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        return original_path_rmdir(self, *args, **kwargs)

    def guarded_path_rename(self, target, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        _assert_file_access(target, "write", read_roots, write_roots)
        return original_path_rename(self, target, *args, **kwargs)

    def guarded_path_replace(self, target, *args, **kwargs):
        _assert_file_access(self, "write", read_roots, write_roots)
        _assert_file_access(target, "write", read_roots, write_roots)
        return original_path_replace(self, target, *args, **kwargs)

    def guarded_path_glob(self, *args, **kwargs):
        _assert_file_access(self, "read", read_roots, write_roots)
        return original_path_glob(self, *args, **kwargs)

    def guarded_path_rglob(self, *args, **kwargs):
        _assert_file_access(self, "read", read_roots, write_roots)
        return original_path_rglob(self, *args, **kwargs)

    def guarded_shutil_copy(src, dst, *args, **kwargs):
        _assert_file_access(src, "read", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_shutil_copy(src, dst, *args, **kwargs)

    def guarded_shutil_copy2(src, dst, *args, **kwargs):
        _assert_file_access(src, "read", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_shutil_copy2(src, dst, *args, **kwargs)

    def guarded_shutil_copyfile(src, dst, *args, **kwargs):
        _assert_file_access(src, "read", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_shutil_copyfile(src, dst, *args, **kwargs)

    def guarded_shutil_move(src, dst, *args, **kwargs):
        _assert_file_access(src, "write", read_roots, write_roots)
        _assert_file_access(dst, "write", read_roots, write_roots)
        return original_shutil_move(src, dst, *args, **kwargs)

    def guarded_shutil_rmtree(path, *args, **kwargs):
        _assert_file_access(path, "write", read_roots, write_roots)
        return original_shutil_rmtree(path, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    os.open = guarded_os_open
    os.remove = guarded_remove
    os.unlink = guarded_unlink
    os.rmdir = guarded_rmdir
    os.mkdir = guarded_mkdir
    os.makedirs = guarded_makedirs
    os.rename = guarded_rename
    os.replace = guarded_replace
    if original_link is not None:
        os.link = guarded_link
    if original_symlink is not None:
        os.symlink = guarded_symlink
    Path.open = guarded_path_open
    Path.read_text = guarded_path_read_text
    Path.read_bytes = guarded_path_read_bytes
    Path.write_text = guarded_path_write_text
    Path.write_bytes = guarded_path_write_bytes
    Path.touch = guarded_path_touch
    Path.unlink = guarded_path_unlink
    Path.mkdir = guarded_path_mkdir
    Path.rmdir = guarded_path_rmdir
    Path.rename = guarded_path_rename
    Path.replace = guarded_path_replace
    Path.glob = guarded_path_glob
    Path.rglob = guarded_path_rglob
    shutil.copy = guarded_shutil_copy
    shutil.copy2 = guarded_shutil_copy2
    shutil.copyfile = guarded_shutil_copyfile
    shutil.move = guarded_shutil_move
    shutil.rmtree = guarded_shutil_rmtree


def _has_read_effect(side_effects: list[str]) -> bool:
    return any(item.startswith("read:") for item in side_effects)


def _has_write_effect(side_effects: list[str]) -> bool:
    return any(item.startswith("write:") for item in side_effects)


def _guard_roots(paths) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        try:
            resolved = Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        key = str(resolved)
        if key and key not in seen:
            roots.append(resolved)
            seen.add(key)
    return tuple(roots)


def _python_runtime_roots() -> list[Path]:
    roots = [
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(getattr(sys, "exec_prefix", sys.prefix)),
        Path(sys.executable).resolve().parent,
    ]
    return roots


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_path_like(value) -> bool:
    return isinstance(value, (str, bytes, os.PathLike))


def _access_from_mode(mode: str) -> str:
    return "write" if any(flag in mode for flag in ("w", "a", "x", "+")) else "read"


def _access_from_flags(flags: int) -> str:
    write_flags = [
        getattr(os, "O_WRONLY", 0),
        getattr(os, "O_RDWR", 0),
        getattr(os, "O_CREAT", 0),
        getattr(os, "O_TRUNC", 0),
        getattr(os, "O_APPEND", 0),
    ]
    return "write" if any(flags & flag for flag in write_flags if flag) else "read"


def _assert_file_access(path, access: str, read_roots: tuple[Path, ...], write_roots: tuple[Path, ...]) -> None:
    if not _is_path_like(path):
        return
    allowed_roots = write_roots if access == "write" else read_roots
    resolved = _coerce_path(path).expanduser().resolve(strict=False)
    if _is_under_any(resolved, allowed_roots):
        return
    if access == "write":
        raise PermissionError(
            f"file write access denied for {resolved}; declare write:* side_effect and use workspace/datasets paths"
        )
    raise PermissionError(
        f"file read access denied for {resolved}; declare read:* side_effect and use workspace/datasets paths"
    )


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        if path == root or root in path.parents:
            return True
    return False


def _coerce_path(path) -> Path:
    if isinstance(path, bytes):
        return Path(os.fsdecode(path))
    return Path(path)


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


def _install_process_guard(side_effects: list[str]) -> None:
    if "process:spawn" in set(str(item) for item in side_effects):
        return

    def _blocked_process_spawn(*_args, **_kwargs):
        raise PermissionError("process spawn access requires process:spawn side_effect")

    subprocess.Popen = _blocked_process_spawn
    for name in ("call", "check_call", "check_output", "getoutput", "getstatusoutput", "run"):
        if hasattr(subprocess, name):
            setattr(subprocess, name, _blocked_process_spawn)
    for name in (
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "forkpty",
        "popen",
        "posix_spawn",
        "posix_spawnp",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "startfile",
        "system",
    ):
        if hasattr(os, name):
            setattr(os, name, _blocked_process_spawn)
    try:
        import pty
    except ImportError:
        return
    pty.spawn = _blocked_process_spawn


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
