from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from marvis.db import PluginRepository
from marvis.plugins.contracts import PROTOCOL_VERSION, WORKER_RESULT_SENTINEL
from marvis.plugins.contracts import ToolContext as ToolContext  # noqa: F401 (re-exported for compatibility)
from marvis.plugins.manifest import PluginManifest, ToolRef
from marvis.plugins.registry import ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema
from marvis.plugins.errors import SchemaValidationError
from marvis.redaction import redact_text
from marvis.safe_paths import assert_within
from marvis.resource_monitor import ProcessTreeResourceMonitor

logger = logging.getLogger(__name__)


_WORKER_ENV_ALLOWLIST = frozenset({
    "CONDA_PREFIX",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "MARVIS_PROBE_URL",
    "MARVIS_SEARCH_ENDPOINT",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "RMC_MATERIAL_ROOTS",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    # Windows-essential vars (absent on POSIX, so a no-op there). Dropping
    # SYSTEMROOT breaks Winsock init in the spawned worker -- any socket then
    # fails with OSError [WinError 10106/10104].
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "USERNAME",
    "LOCALAPPDATA",
    "APPDATA",
    "PROGRAMDATA",
    # Home-directory resolution (Path.home()/expanduser("~") on Windows).
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "HOMESHARE",
})


# Default soft RSS ceiling (MB) for tool worker process trees when the host
# does not configure execution_environment.rss_memory_limit_mb explicitly.
# This psutil-based RSS monitor is the sole memory enforcement layer:
# rlimit-based caps (RLIMIT_AS/RLIMIT_DATA) were removed because they bound
# virtual address space, which the JVM and OpenBLAS legitimately reserve in
# multi-GB quantities — on Linux that broke PMML export and scipy imports
# while macOS ignored the limits entirely. RSS measures resident memory,
# which is what the ceiling is meant to bound; the kill path is real-process
# verified (TST-4). CPU/file-size rlimits remain in the worker.
DEFAULT_RSS_MEMORY_LIMIT_MB = 4096


class WorkerResourceLimitExceeded(Exception):
    """Raised when a tool worker process tree's RSS exceeds the soft limit."""

    def __init__(self, resource_usage: dict[str, Any]) -> None:
        self.resource_usage = resource_usage
        super().__init__(_resource_limit_message(resource_usage))


@dataclass
class ToolResult:
    ok: bool
    output: dict | None
    error: str | None
    error_kind: str | None
    duration_ms: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_detail: dict | None = None
    resource_limits: dict | None = None


class ToolRunner:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        repo: PluginRepository,
        *,
        python_executable: str,
        datasets_root: Path,
        workspace: Path,
        plugin_paths: list[Path] | None = None,
        rss_memory_limit_mb: int | None = DEFAULT_RSS_MEMORY_LIMIT_MB,
    ):
        self._tools = tool_registry
        self._repo = repo
        self._python_executable = python_executable
        self._datasets_root = datasets_root
        self._workspace = workspace
        self._plugin_paths = tuple(Path(path) for path in (plugin_paths or ()))
        self._rss_memory_limit_mb = rss_memory_limit_mb

    def invoke(
        self,
        ref: ToolRef,
        inputs: dict,
        *,
        task_id: str,
        seed: int | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        target_ref = ref.label()
        logger.debug("tool invoke starting target_ref=%s task_id=%s", target_ref, task_id)
        try:
            manifest, tool = self._tools.resolve_with_manifest(ref)
            _require_tool_permissions(manifest, tool.side_effects)
            validate_against_schema(inputs, tool.input_schema, label="inputs")
        except SchemaValidationError as exc:
            result = _failed_result(started, "schema", str(exc))
            return self._finalize_audited_result(started, target_ref, inputs, result)
        except PermissionError as exc:
            result = _failed_result(started, "permission", str(exc))
            return self._finalize_audited_result(started, target_ref, inputs, result)
        effective_seed = seed
        if effective_seed is None and tool.determinism == "stochastic":
            effective_seed = _input_seed(inputs)
        if effective_seed is None and tool.determinism == "stochastic":
            effective_seed = _derive_seed(target_ref, task_id, inputs)

        checkpoint_error = self._write_started_audit(
            started,
            target_ref,
            inputs,
            seed=effective_seed,
            side_effects=tool.side_effects,
            timeout_seconds=tool.timeout_seconds,
        )
        if checkpoint_error is not None:
            return checkpoint_error

        job = {
            "protocol_version": PROTOCOL_VERSION,
            "module": manifest.module,
            "entrypoint": tool.entrypoint,
            "inputs": inputs,
            "task_id": task_id,
            "seed": effective_seed,
            "datasets_root": str(self._datasets_root),
            "workspace": str(self._workspace),
            "memory_limit_mb": tool.memory_limit_mb,
            "cpu_limit_seconds": int(tool.timeout_seconds) + 2,
            "file_size_limit_mb": 2048,
            "plugin_paths": [str(path) for path in self._plugin_paths],
            "side_effects": list(tool.side_effects),
            "builtin": bool(manifest.builtin),
        }
        try:
            completed = _run_worker(
                self._python_executable,
                job,
                timeout=tool.timeout_seconds,
                rss_limit_mb=self._rss_memory_limit_mb,
            )
        except subprocess.TimeoutExpired as exc:
            result = _failed_result(
                started,
                "timeout",
                f"tool {target_ref} timed out after {tool.timeout_seconds}s",
                stdout_tail=_tail(exc.stdout),
                stderr_tail=_tail(exc.stderr),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )
        except WorkerResourceLimitExceeded as exc:
            result = _failed_result(
                started,
                "resource_limit",
                str(exc),
                resource_limits=exc.resource_usage,
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )

        protocol = _parse_worker_result(completed.stdout)
        if protocol is None:
            result = _failed_result(
                started,
                "protocol",
                f"worker returned invalid protocol with exit code {completed.returncode}",
                stdout_tail=_tail(completed.stdout),
                stderr_tail=_tail(completed.stderr),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )

        version_error = _check_worker_protocol_version(protocol)
        if version_error is not None:
            result = _failed_result(
                started,
                "protocol_version_mismatch",
                version_error,
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                error_detail=_protocol_version_error_detail(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )

        if not protocol.get("ok"):
            error_detail = protocol.get("error_detail")
            result = _failed_result(
                started,
                str(protocol.get("error_kind") or "execution"),
                str(protocol.get("error") or "tool execution failed"),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("traceback") or protocol.get("stderr") or completed.stderr),
                error_detail=error_detail if isinstance(error_detail, dict) else None,
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )

        output = protocol.get("output")
        try:
            validate_against_schema(output, tool.output_schema, label=f"output:{target_ref}")
            _validate_output_paths(
                output,
                workspace=self._workspace,
                datasets_root=self._datasets_root,
            )
        except SchemaValidationError as exc:
            result = _failed_result(
                started,
                "schema",
                str(exc),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )
        except PermissionError as exc:
            result = _failed_result(
                started,
                "permission",
                str(exc),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=effective_seed,
            )

        result = ToolResult(
            ok=True,
            output=output,
            error=None,
            error_kind=None,
            duration_ms=_duration_ms(started),
            stdout_tail=_tail(protocol.get("stdout") or ""),
            stderr_tail=_tail(protocol.get("stderr") or ""),
            resource_limits=_protocol_resource_limits(protocol),
        )
        return self._finalize_audited_result(
            started,
            target_ref,
            inputs,
            result,
            seed=effective_seed,
        )

    def invoke_adhoc(
        self,
        *,
        module: Path,
        entrypoint: str,
        inputs: dict,
        input_schema: dict,
        output_schema: dict,
        timeout_seconds: int,
        task_id: str,
        mode: str = "adhoc",
        seed: int | None = None,
        memory_limit_mb: int = 2048,
    ) -> ToolResult:
        started = time.monotonic()
        target_ref = f"{mode}.{entrypoint}"
        audit_kind = f"{mode}.invoke"
        try:
            validate_against_schema(inputs, input_schema, label="inputs")
        except SchemaValidationError as exc:
            result = _failed_result(started, "schema", str(exc))
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        checkpoint_error = self._write_started_audit(
            started,
            target_ref,
            inputs,
            seed=seed,
            kind=f"{audit_kind}.started",
            mode=mode,
            side_effects=(),
            timeout_seconds=timeout_seconds,
        )
        if checkpoint_error is not None:
            return checkpoint_error

        job = {
            "protocol_version": PROTOCOL_VERSION,
            "module_path": str(Path(module)),
            "entrypoint": entrypoint,
            "inputs": inputs,
            "task_id": task_id,
            "seed": seed,
            "datasets_root": str(self._datasets_root),
            "workspace": str(self._workspace),
            "memory_limit_mb": int(memory_limit_mb),
            "cpu_limit_seconds": int(timeout_seconds) + 2,
            "file_size_limit_mb": 2048,
            "plugin_paths": [str(path) for path in self._plugin_paths],
            "side_effects": [],
            "builtin": False,
        }
        try:
            completed = _run_worker(
                self._python_executable,
                job,
                timeout=int(timeout_seconds),
                rss_limit_mb=self._rss_memory_limit_mb,
            )
        except subprocess.TimeoutExpired as exc:
            result = _failed_result(
                started,
                "timeout",
                f"tool {target_ref} timed out after {timeout_seconds}s",
                stdout_tail=_tail(exc.stdout),
                stderr_tail=_tail(exc.stderr),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )
        except WorkerResourceLimitExceeded as exc:
            result = _failed_result(
                started,
                "resource_limit",
                str(exc),
                resource_limits=exc.resource_usage,
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        protocol = _parse_worker_result(completed.stdout)
        if protocol is None:
            result = _failed_result(
                started,
                "protocol",
                f"worker returned invalid protocol with exit code {completed.returncode}",
                stdout_tail=_tail(completed.stdout),
                stderr_tail=_tail(completed.stderr),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        version_error = _check_worker_protocol_version(protocol)
        if version_error is not None:
            result = _failed_result(
                started,
                "protocol_version_mismatch",
                version_error,
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                error_detail=_protocol_version_error_detail(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        if not protocol.get("ok"):
            error_detail = protocol.get("error_detail")
            result = _failed_result(
                started,
                str(protocol.get("error_kind") or "execution"),
                str(protocol.get("error") or "tool execution failed"),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("traceback") or protocol.get("stderr") or completed.stderr),
                error_detail=error_detail if isinstance(error_detail, dict) else None,
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        output = protocol.get("output")
        try:
            validate_against_schema(output, output_schema, label=f"output:{target_ref}")
            _validate_output_paths(
                output,
                workspace=self._workspace,
                datasets_root=self._datasets_root,
            )
        except SchemaValidationError as exc:
            result = _failed_result(
                started,
                "schema",
                str(exc),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )
        except PermissionError as exc:
            result = _failed_result(
                started,
                "permission",
                str(exc),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
                resource_limits=_protocol_resource_limits(protocol),
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
                kind=audit_kind,
                mode=mode,
            )

        result = ToolResult(
            ok=True,
            output=output,
            error=None,
            error_kind=None,
            duration_ms=_duration_ms(started),
            stdout_tail=_tail(protocol.get("stdout") or ""),
            stderr_tail=_tail(protocol.get("stderr") or ""),
            resource_limits=_protocol_resource_limits(protocol),
        )
        return self._finalize_audited_result(
            started,
            target_ref,
            inputs,
            result,
            seed=seed,
            kind=audit_kind,
            mode=mode,
        )

    def _write_started_audit(
        self,
        started: float,
        target_ref: str,
        inputs: dict,
        *,
        seed: int | None,
        kind: str = "tool.invoke.started",
        mode: str | None = None,
        side_effects: tuple[str, ...],
        timeout_seconds: int,
    ) -> ToolResult | None:
        detail: dict[str, Any] = {
            "seed": seed,
            "side_effects": list(side_effects),
            "timeout_seconds": int(timeout_seconds),
        }
        if mode:
            detail["mode"] = mode
        try:
            self._repo.write_audit(
                kind=kind,
                target_ref=target_ref,
                inputs_hash=_hash_inputs(inputs),
                outcome="started",
                detail=detail,
            )
        except Exception as exc:
            return _audit_failure_result(started, "start", exc)
        return None

    def _finalize_audited_result(
        self,
        started: float,
        target_ref: str,
        inputs: dict,
        result: ToolResult,
        *,
        seed: int | None = None,
        kind: str = "tool.invoke",
        mode: str | None = None,
    ) -> ToolResult:
        # Single choke point every invoke()/invoke_adhoc() branch (success or
        # any of the failure kinds) funnels through -- logging here covers the
        # whole call without needing a log line in each individual branch.
        if result.ok:
            logger.info(
                "tool invoke ok target_ref=%s duration_ms=%d",
                target_ref, int(result.duration_ms),
            )
        else:
            logger.warning(
                "tool invoke failed target_ref=%s error_kind=%s error=%s",
                target_ref, result.error_kind, redact_text(result.error or ""),
            )
        try:
            self._write_audit(target_ref, inputs, result, seed=seed, kind=kind, mode=mode)
        except Exception as exc:
            return _audit_failure_result(started, "finish", exc, result=result)
        return result

    def _write_audit(
        self,
        target_ref: str,
        inputs: dict,
        result: ToolResult,
        *,
        seed: int | None = None,
        kind: str = "tool.invoke",
        mode: str | None = None,
    ) -> None:
        detail = {
            "error_kind": result.error_kind,
            "duration_ms": result.duration_ms,
            "seed": seed,
        }
        if result.error_detail:
            detail["error_detail"] = result.error_detail
        if result.resource_limits:
            detail["resource_limits"] = result.resource_limits
        if mode:
            detail["mode"] = mode
        self._repo.write_audit(
            kind=kind,
            target_ref=target_ref,
            inputs_hash=_hash_inputs(inputs),
            outcome="succeeded" if result.ok else "failed",
            detail=detail,
        )


def _require_tool_permissions(manifest: PluginManifest, side_effects: tuple[str, ...]) -> None:
    allowed = set(manifest.permissions)
    missing = [effect for effect in side_effects if effect not in allowed]
    if missing:
        raise PermissionError(
            f"tool side_effects not allowed by plugin permissions: {', '.join(missing)}"
        )


def _validate_output_paths(output: Any, *, workspace: Path, datasets_root: Path) -> None:
    allowed_roots = (Path(workspace), Path(datasets_root))
    for location, value in _iter_output_path_values(output):
        _validate_output_path_value(str(value), location=location, allowed_roots=allowed_roots)


def _iter_output_path_values(value: Any, *, prefix: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{prefix}.{key}"
            if isinstance(item, str):
                if item and (
                    (isinstance(key, str) and _is_path_output_key(key))
                    or _is_artifact_ref(item)
                ):
                    yield location, item
                continue
            yield from _iter_output_path_values(item, prefix=location)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_output_path_values(item, prefix=f"{prefix}[{index}]")
    elif isinstance(value, str) and _is_artifact_ref(value):
        yield prefix, value


def _is_path_output_key(key: str) -> bool:
    normalized = key.lower()
    return normalized == "path" or normalized.endswith("_path")


def _is_artifact_ref(value: str) -> bool:
    return value.strip().startswith("artifact:")


def _validate_output_path_value(value: str, *, location: str, allowed_roots: tuple[Path, ...]) -> None:
    text = value.strip()
    if not text:
        return
    if text.startswith("artifact:"):
        _validate_relative_output_path(text.split(":", 1)[1], location=location)
        return
    path = Path(text)
    if path.is_absolute():
        for root in allowed_roots:
            try:
                assert_within(root, path)
                return
            except PermissionError:
                continue
        raise PermissionError(f"output path {location} escapes allowed roots: {text}")
    _validate_relative_output_path(text, location=location)


def _validate_relative_output_path(value: str, *, location: str) -> None:
    path = Path(value)
    if path.is_absolute() or path.drive:
        raise PermissionError(f"output path {location} must be relative: {value}")
    if value.startswith("~") or any(part == ".." for part in path.parts):
        raise PermissionError(f"output path {location} contains unsafe relative path: {value}")


def _failed_result(
    started: float,
    error_kind: str,
    error: str,
    *,
    stdout_tail: str = "",
    stderr_tail: str = "",
    error_detail: dict | None = None,
    resource_limits: dict | None = None,
) -> ToolResult:
    return ToolResult(
        ok=False,
        output=None,
        error=error,
        error_kind=error_kind,
        duration_ms=_duration_ms(started),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error_detail=error_detail,
        resource_limits=resource_limits,
    )


def _audit_failure_result(
    started: float,
    phase: str,
    exc: Exception,
    *,
    result: ToolResult | None = None,
) -> ToolResult:
    detail: dict[str, Any] = {
        "audit_phase": phase,
        "audit_error": str(exc),
    }
    if result is not None:
        detail["result_ok"] = result.ok
        detail["result_error_kind"] = result.error_kind
    return _failed_result(
        started,
        "audit",
        f"audit {phase} failed: {exc}",
        error_detail=detail,
        resource_limits=result.resource_limits if result is not None else None,
    )


def _run_worker(
    python_executable: str,
    job: dict,
    *,
    timeout: int,
    rss_limit_mb: int | None = None,
) -> subprocess.CompletedProcess:
    args = [python_executable, "-m", "marvis.plugins.subprocess_worker"]
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_worker_env(),
        start_new_session=(os.name != "nt"),
    )
    monitor = ProcessTreeResourceMonitor(
        pid_getter=lambda: process.pid,
        memory_limit_mb=rss_limit_mb,
        on_limit=lambda: _kill_worker_tree(process),
    )
    try:
        with monitor:
            stdout, stderr = process.communicate(json.dumps(job, ensure_ascii=False), timeout=int(timeout))
    except subprocess.TimeoutExpired as exc:
        _kill_worker_tree(process)
        stdout, stderr = process.communicate()
        if monitor.memory_limit_exceeded:
            raise WorkerResourceLimitExceeded(monitor.snapshot()) from exc
        raise subprocess.TimeoutExpired(args, int(timeout), output=stdout, stderr=stderr) from exc
    if monitor.memory_limit_exceeded:
        raise WorkerResourceLimitExceeded(monitor.snapshot())
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def _kill_worker_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _worker_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _WORKER_ENV_ALLOWLIST and value
    }
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _parse_worker_result(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        index = line.rfind(WORKER_RESULT_SENTINEL)
        if index != -1:
            result = _load_worker_result(
                line[index + len(WORKER_RESULT_SENTINEL):]
            )
            if result is not None:
                return result
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    return _load_worker_result(line)


def _load_worker_result(line: str) -> dict[str, Any] | None:
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _protocol_resource_limits(protocol: dict) -> dict | None:
    value = protocol.get("resource_limits")
    return value if isinstance(value, dict) else None


def _check_worker_protocol_version(protocol: dict) -> str | None:
    """ARCH-5: host-side half of the version handshake. The worker validates
    protocol_version itself (subprocess_worker._check_protocol_version) and
    reports back worker_protocol_version on every response, but the host must
    not simply trust that self-check -- an old, pre-handshake worker binary
    silently ignores the unrecognized protocol_version job field and returns
    ok=true with no worker_protocol_version at all, which would otherwise slip
    through as a false success. Returns a Chinese-readable error message when
    the worker's reported version is missing or does not match the host's,
    else None.
    """
    worker_version = protocol.get("worker_protocol_version")
    if worker_version == PROTOCOL_VERSION:
        return None
    if worker_version is None and protocol.get("ok"):
        return (
            f"插件 worker 协议版本不匹配：宿主={PROTOCOL_VERSION!r}，worker 未上报版本号"
            f"（可能是握手协议之前的旧 worker）；请确认 execution_environment 配置的 "
            f"python_executable 与宿主使用同一份 marvis 代码"
        )
    if worker_version is None:
        # Worker already failed for an unrelated reason (execution/timeout/etc)
        # before it could report its version -- let that original error surface
        # unchanged rather than masking it with a version complaint.
        return None
    return (
        f"插件 worker 协议版本不匹配：宿主={PROTOCOL_VERSION!r}，worker={worker_version!r}；"
        f"请确认 execution_environment 配置的 python_executable 与宿主使用同一份 marvis 代码"
    )


def _protocol_version_error_detail(protocol: dict) -> dict:
    detail = protocol.get("error_detail")
    if isinstance(detail, dict) and detail.get("kind") == "protocol_version_mismatch":
        return detail
    return {
        "kind": "protocol_version_mismatch",
        "host_protocol_version": PROTOCOL_VERSION,
        "worker_protocol_version": protocol.get("worker_protocol_version"),
    }


def _resource_limit_message(resource_usage: dict[str, Any]) -> str:
    limit = resource_usage.get("memory_limit_mb")
    peak = resource_usage.get("peak_rss_mb")
    if peak is None:
        return f"tool worker RSS exceeded memory limit {limit} MB"
    return f"tool worker RSS {peak} MB exceeded memory limit {limit} MB"


def _tail(value: str | bytes | None, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return redact_text(text[-limit:])


def _hash_inputs(inputs: dict) -> str:
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _input_seed(inputs: dict) -> int | None:
    value = inputs.get("seed")
    if value is None:
        return None
    return int(value)


def _derive_seed(target_ref: str, task_id: str, inputs: dict) -> int:
    raw = json.dumps(
        {"target_ref": target_ref, "task_id": task_id, "inputs": inputs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)
