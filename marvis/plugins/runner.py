from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Callable
import uuid

from marvis.db import PluginRepository
from marvis.governance.errors import AuthorizationError
from marvis.job_cancellation import JobCancelled
from marvis.plugins.contracts import MAX_PROGRESS_BYTES, PROTOCOL_VERSION, WORKER_RESULT_SENTINEL
from marvis.plugins.contracts import ToolContext as ToolContext  # noqa: F401 (re-exported for compatibility)
from marvis.plugins.manifest import PluginManifest, ToolRef
from marvis.plugins.registry import ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema
from marvis.plugins.errors import SchemaValidationError
from marvis.redaction import redact_text
from marvis.safe_paths import assert_within
from marvis.resource_monitor import (
    ProcessTreeResourceMonitor,
    terminate_process_tree_by_pid,
)

logger = logging.getLogger(__name__)


_WORKER_ENV_ALLOWLIST = frozenset({
    "CONDA_PREFIX",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "MARVIS_MAX_CSV_UPLOAD_BYTES",
    "MARVIS_MAX_EXCEL_ROWS",
    "MARVIS_MAX_EXCEL_UPLOAD_BYTES",
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
WORKER_REAP_TIMEOUT_SECONDS = 2.0
PROGRESS_STALE_AFTER_SECONDS = 24 * 60 * 60
PROGRESS_SWEEP_MAX_ENTRIES = 256


class WorkerTimeoutExpired(subprocess.TimeoutExpired):
    """Timeout with a JSON-safe detail contract for nested worker boundaries."""

    def to_detail(self) -> dict:
        return {
            "kind": "timeout",
            "error_detail": {
                "kind": "worker_timeout",
                "timeout_seconds": self.timeout,
            },
        }


class WorkerResourceLimitExceeded(Exception):
    """Raised when a tool worker process tree's RSS exceeds the soft limit."""

    def __init__(
        self,
        resource_usage: dict[str, Any],
        *,
        message: str | None = None,
        error_detail: dict[str, Any] | None = None,
    ) -> None:
        self.resource_usage = resource_usage
        self.error_detail = error_detail or {
            "kind": "worker_rss_limit",
            **resource_usage,
        }
        super().__init__(message or _resource_limit_message(resource_usage))

    def to_detail(self) -> dict:
        return {
            "kind": "resource_limit",
            "resource_limits": dict(self.resource_usage),
            "error_detail": dict(self.error_detail),
        }


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
        governance=None,
        binding_resolver=None,
    ):
        self._tools = tool_registry
        self._repo = repo
        self._python_executable = python_executable
        self._datasets_root = datasets_root
        self._workspace = workspace
        self._plugin_paths = tuple(Path(path) for path in (plugin_paths or ()))
        self._rss_memory_limit_mb = rss_memory_limit_mb
        self._governance = governance
        self._binding_resolver = binding_resolver
        _sweep_stale_progress_files(self._workspace)

    def invoke(
        self,
        ref: ToolRef,
        inputs: dict,
        *,
        task_id: str,
        seed: int | None = None,
        execution_context=None,
        progress_callback: Callable[[dict], None] | None = None,
        cancellation_check: Callable[[], None] | None = None,
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
        manifest_human_required = tool.policy.human_decision_gate == "required"
        manifest_effect_required = tool.policy.effect_authorization == "required"
        context_human_required = bool(
            execution_context is not None
            and getattr(execution_context, "human_decision_required", False) is True
        )
        context_effect_required = bool(
            execution_context is not None
            and getattr(execution_context, "effect_authorization_required", False) is True
        )
        effect_authorization_required = (
            manifest_effect_required or context_effect_required
        )
        human_decision_required = (
            manifest_human_required
            or context_human_required
            or effect_authorization_required
        )
        protected_execution = human_decision_required or effect_authorization_required
        governance_method = (
            getattr(self._governance, "reserve_effect", None)
            if effect_authorization_required
            else getattr(self._governance, "verify_decision", None)
        )
        if protected_execution and (
            not callable(governance_method)
            or not _has_binding_resolver(self._binding_resolver)
            or execution_context is None
        ):
            result = _failed_result(
                started,
                "authorization",
                f"tool {target_ref} requires a live, bound governance decision",
            )
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
            )
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

        if cancellation_check is not None:
            try:
                cancellation_check()
            except JobCancelled as exc:
                result = _failed_result(
                    started,
                    "cancelled",
                    f"tool {target_ref} cancelled by user",
                    error_detail={"kind": "user_cancelled", "message": str(exc)},
                )
                return self._finalize_audited_result(
                    started,
                    target_ref,
                    inputs,
                    result,
                    seed=effective_seed,
                )

        effect_execution = None
        if protected_execution:
            authorization_phase = "binding"
            try:
                live_binding = self._resolve_governance_binding(
                    task_id=task_id,
                    ref=ref,
                    inputs=inputs,
                    execution_context=execution_context,
                    manifest=manifest,
                    tool=tool,
                )
                if effect_authorization_required:
                    authorization_phase = "reserve"
                    effect_execution = self._governance.reserve_effect(
                        execution_context,
                        live_binding,
                    )
                else:
                    authorization_phase = "verify"
                    self._governance.verify_decision(
                        execution_context,
                        live_binding,
                    )
            except AuthorizationError as exc:
                result = _failed_result(
                    started,
                    "authorization",
                    f"governance authorization rejected for {target_ref}: {exc}",
                    error_detail={"authorization_phase": authorization_phase},
                )
                return self._finalize_audited_result(
                    started,
                    target_ref,
                    inputs,
                    result,
                    seed=effective_seed,
                )

        progress_path = None
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
            "cpu_limit_seconds": _aggregate_cpu_limit_seconds(tool.timeout_seconds),
            "file_size_limit_mb": 2048,
            "plugin_paths": [str(path) for path in self._plugin_paths],
            "side_effects": list(tool.side_effects),
            "builtin": bool(manifest.builtin),
        }
        if effect_execution is not None:
            # Execution authorization is platform metadata, not a business
            # input.  Keeping it out-of-band means a model/client cannot forge
            # it through the tool's JSON input contract.
            try:
                job["effect_execution_id"] = _effect_execution_id(effect_execution)
                job["runtime_generation"] = _runtime_generation(
                    effect_execution,
                    execution_context,
                )
                self._governance.mark_effect_dispatched(
                    _effect_execution_id(effect_execution),
                    reservation_id=_effect_reservation_id(effect_execution),
                )
            except Exception as exc:
                self._release_prepared_effect(
                    effect_execution,
                    reason=f"dispatch checkpoint failed before worker start: {exc}",
                )
                if not isinstance(exc, AuthorizationError):
                    raise
                result = _failed_result(
                    started,
                    "authorization",
                    f"effect dispatch checkpoint failed for {target_ref}: {exc}",
                    error_detail={"authorization_phase": "dispatch"},
                )
                return self._finalize_audited_result(
                    started,
                    target_ref,
                    inputs,
                    result,
                    seed=effective_seed,
                )
        if bool(manifest.builtin) and callable(progress_callback):
            progress_path = _try_new_progress_path(self._workspace)
            if progress_path is not None:
                job["progress_path"] = str(progress_path)
        try:
            completed = _run_worker(
                self._python_executable,
                job,
                timeout=tool.timeout_seconds,
                rss_limit_mb=self._rss_memory_limit_mb,
                progress_path=progress_path,
                progress_callback=progress_callback,
                cancellation_check=cancellation_check,
            )
        except JobCancelled as exc:
            result = _failed_result(
                started,
                "cancelled",
                f"tool {target_ref} cancelled by user",
                error_detail={"kind": "user_cancelled", "message": str(exc)},
            )
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
                seed=effective_seed,
            )
        except subprocess.TimeoutExpired as exc:
            result = _failed_result(
                started,
                "timeout",
                f"tool {target_ref} timed out after {tool.timeout_seconds}s",
                stdout_tail=_tail(exc.stdout),
                stderr_tail=_tail(exc.stderr),
            )
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
                seed=effective_seed,
            )
        except WorkerResourceLimitExceeded as exc:
            result = _failed_result(
                started,
                "resource_limit",
                str(exc),
                error_detail=exc.error_detail,
                resource_limits=exc.resource_usage,
            )
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
                seed=effective_seed,
            )
        except Exception as exc:
            if effect_execution is None:
                raise
            result = _failed_result(
                started,
                "execution",
                f"tool {target_ref} worker launch failed: {exc}",
            )
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
                seed=effective_seed,
            )
        finally:
            _cleanup_progress_path(progress_path)

        protocol = _parse_worker_result(completed.stdout)
        if protocol is None:
            result = _missing_worker_protocol_result(
                started,
                completed,
                cpu_limit_seconds=job["cpu_limit_seconds"],
            )
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
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
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
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
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
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
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
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
            return self._finalize_effect_result(
                started,
                target_ref,
                inputs,
                result,
                effect_execution=effect_execution,
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
        return self._finalize_effect_result(
            started,
            target_ref,
            inputs,
            result,
            effect_execution=effect_execution,
            seed=effective_seed,
        )

    def _resolve_governance_binding(
        self,
        *,
        task_id: str,
        ref: ToolRef,
        inputs: dict,
        execution_context,
        manifest: PluginManifest,
        tool,
    ):
        resolver = self._binding_resolver
        resolve = getattr(resolver, "resolve_binding", None)
        if resolve is None:
            resolve = resolver
        if not callable(resolve):
            raise TypeError("binding_resolver must be callable or expose resolve_binding")
        return resolve(
            task_id=task_id,
            ref=ref,
            inputs=inputs,
            execution_context=execution_context,
            manifest=manifest,
            tool=tool,
        )

    def _release_prepared_effect(self, effect_execution, *, reason: str) -> None:
        try:
            self._governance.release_prepared_effect(
                _effect_execution_id(effect_execution),
                reservation_id=_effect_reservation_id(effect_execution),
                reason=reason,
            )
        except Exception:
            # The worker was never dispatched.  Startup reconciliation can
            # safely release a stranded PREPARED record; never start the worker
            # merely because this cleanup checkpoint failed.
            logger.exception("failed to release prepared effect execution")

    def _finalize_effect_result(
        self,
        started: float,
        target_ref: str,
        inputs: dict,
        result: ToolResult,
        *,
        effect_execution,
        seed: int | None,
    ) -> ToolResult:
        if effect_execution is None:
            return self._finalize_audited_result(
                started,
                target_ref,
                inputs,
                result,
                seed=seed,
            )
        effect_id = _effect_execution_id(effect_execution)
        reservation_id = _effect_reservation_id(effect_execution)
        if result.ok:
            try:
                result_hash = "sha256:" + _hash_inputs(result.output or {})
                if self._effect_is_committed(effect_id):
                    # Some domain tools (strategy adoption) atomically commit
                    # the domain transition and effect receipt in one DB
                    # transaction. Re-validating that terminal state must not
                    # replace its domain receipt hash with a host output hash.
                    result_hash = None
                self._governance.mark_effect_committed(
                    effect_id,
                    reservation_id=reservation_id,
                    result_hash=result_hash,
                )
            except Exception as exc:
                self._mark_effect_uncertain(
                    effect_execution,
                    reason=f"effect result could not be committed: {exc}",
                )
                if not isinstance(exc, AuthorizationError):
                    raise
                result = _failed_result(
                    started,
                    "authorization",
                    f"effect commit checkpoint failed for {target_ref}: {exc}",
                    stdout_tail=result.stdout_tail,
                    stderr_tail=result.stderr_tail,
                    error_detail={
                        "authorization_phase": "commit",
                        "effect_execution_id": effect_id,
                    },
                    resource_limits=result.resource_limits,
                )
        else:
            self._mark_effect_uncertain(
                effect_execution,
                reason=f"post-dispatch {result.error_kind or 'execution'} failure: {result.error or ''}",
            )
        return self._finalize_audited_result(
            started,
            target_ref,
            inputs,
            result,
            seed=seed,
        )

    def _mark_effect_uncertain(self, effect_execution, *, reason: str) -> None:
        if self._effect_is_committed(_effect_execution_id(effect_execution)):
            # A domain transaction may have committed before a later artifact
            # or worker-protocol failure. Never downgrade/revoke that terminal
            # effect and never make its approval replayable.
            return
        try:
            self._governance.mark_effect_uncertain(
                _effect_execution_id(effect_execution),
                reservation_id=_effect_reservation_id(effect_execution),
                reason=reason,
            )
        except Exception:
            # A DISPATCHED effect must never be reissued on a cleanup failure.
            # Startup reconciliation keeps the approval terminal/uncertain.
            logger.exception("failed to mark dispatched effect execution uncertain")

    def _effect_is_committed(self, effect_id: str) -> bool:
        load = getattr(self._governance, "get_effect_execution", None)
        if not callable(load):
            return False
        current = load(effect_id)
        state = getattr(current, "state", None)
        return str(getattr(state, "value", state) or "") == "committed"

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
        registered_manifest = _registered_manifest_for_module(
            self._tools,
            Path(module),
            plugin_paths=self._plugin_paths,
        )
        if registered_manifest is not None:
            # Ad-hoc execution has no manifest policy or bound ExecutionContext.
            # Letting it point at an installed module would therefore create a
            # second, policy-blind entry to every registered Tool (including
            # strategy adoption). Registered code must always use invoke().
            result = _failed_result(
                started,
                "authorization",
                f"registered plugin module {registered_manifest} must be invoked through its manifest-aware Tool path",
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
            "cpu_limit_seconds": _aggregate_cpu_limit_seconds(timeout_seconds),
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
                error_detail=exc.error_detail,
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
            result = _missing_worker_protocol_result(
                started,
                completed,
                cpu_limit_seconds=job["cpu_limit_seconds"],
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


def _has_binding_resolver(resolver) -> bool:
    return callable(getattr(resolver, "resolve_binding", None)) or callable(resolver)


def _registered_manifest_for_module(
    tools,
    module_path: Path,
    *,
    plugin_paths: tuple[Path, ...],
) -> str | None:
    """Identify an installed module even when it is addressed by filesystem path."""

    list_manifests = getattr(tools, "manifests", None)
    if not callable(list_manifests):
        return None
    requested = module_path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    roots = (project_root, *plugin_paths)
    for manifest in list_manifests():
        relative = Path(*str(manifest.module).split("."))
        candidates: set[Path] = set()
        for root in roots:
            base = Path(root).resolve() / relative
            candidates.add(base.with_suffix(".py"))
            candidates.add(base / "__init__.py")
        if any(_same_module_file(requested, candidate) for candidate in candidates):
            return f"{manifest.name}@{manifest.version}"
    return None


def _same_module_file(requested: Path, candidate: Path) -> bool:
    """Compare file identity, including aliases on case-insensitive filesystems."""

    try:
        return requested.samefile(candidate)
    except OSError:
        # Preserve a deterministic lexical fallback for missing paths. Existing
        # files use samefile(), which also closes symlink and hard-link aliases.
        return requested == candidate.resolve()


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


def _available_cpu_count() -> int:
    """Return the logical CPU capacity available to the worker process.

    ``RLIMIT_CPU`` measures aggregate process CPU time, including time consumed
    concurrently by native-library threads.  A wall-clock tool timeout must
    therefore be multiplied by the CPU capacity available to the process; using
    the wall timeout verbatim kills a healthy multi-threaded learner many times
    earlier than the host's actual wall-clock deadline.
    """

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            count = len(affinity(0))
            if count > 0:
                return count
        except (OSError, TypeError, ValueError):
            pass
    return max(1, int(os.cpu_count() or 1))


def _aggregate_cpu_limit_seconds(timeout_seconds: int) -> int:
    """Translate a wall timeout into an aggregate CPU-time safety ceiling."""

    return (max(1, int(timeout_seconds)) + 2) * _available_cpu_count()


def _missing_worker_protocol_result(
    started: float,
    completed: subprocess.CompletedProcess,
    *,
    cpu_limit_seconds: int,
) -> ToolResult:
    """Classify signal termination before falling back to protocol failure."""

    sigxcpu = getattr(signal, "SIGXCPU", None)
    if sigxcpu is not None and completed.returncode == -int(sigxcpu):
        resource_limits = {
            "cpu_limit_seconds": int(cpu_limit_seconds),
            "cpu_limit_exceeded": True,
            "termination_signal": "SIGXCPU",
            "worker_returncode": int(completed.returncode),
        }
        return _failed_result(
            started,
            "resource_limit",
            (
                "tool worker 累计 CPU 时间达到平台上限 "
                f"{cpu_limit_seconds} 秒（SIGXCPU）；这是平台资源约束，"
                "不是数据错误或 worker 协议错误。"
            ),
            stdout_tail=_tail(completed.stdout),
            stderr_tail=_tail(completed.stderr),
            error_detail={"kind": "worker_cpu_limit", **resource_limits},
            resource_limits=resource_limits,
        )
    return _failed_result(
        started,
        "protocol",
        f"worker returned invalid protocol with exit code {completed.returncode}",
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def _nested_worker_signal_resource_error(
    completed: subprocess.CompletedProcess,
    *,
    cpu_limit_seconds: int,
) -> WorkerResourceLimitExceeded | None:
    """Map nested signal exits before they can be mislabeled as protocol errors."""

    sigxcpu = getattr(signal, "SIGXCPU", None)
    if sigxcpu is None or completed.returncode != -int(sigxcpu):
        return None
    resource_limits = {
        "cpu_limit_seconds": int(cpu_limit_seconds),
        "cpu_limit_exceeded": True,
        "termination_signal": "SIGXCPU",
        "worker_returncode": int(completed.returncode),
    }
    return WorkerResourceLimitExceeded(
        resource_limits,
        message=(
            "tool worker aggregate CPU time exceeded "
            f"{int(cpu_limit_seconds)} seconds (SIGXCPU)"
        ),
        error_detail={"kind": "worker_cpu_limit", **resource_limits},
    )


def _run_worker(
    python_executable: str,
    job: dict,
    *,
    timeout: int,
    rss_limit_mb: int | None = None,
    progress_path: Path | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancellation_check: Callable[[], None] | None = None,
    start_new_session: bool = True,
) -> subprocess.CompletedProcess:
    if cancellation_check is not None:
        cancellation_check()
    args = [python_executable, "-m", "marvis.plugins.subprocess_worker"]
    process = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=_worker_env(),
        # Top-level tool workers own a process group so the host can terminate
        # their whole tree.  A nested, short-lived worker must remain in that
        # same group: if the aggregate worker is killed by its host, the nested
        # recipe worker then dies with it instead of becoming an orphan.  The
        # default preserves the public/top-level behaviour unchanged.
        start_new_session=(os.name != "nt" and start_new_session),
    )
    # A top-level POSIX worker is its own process-group leader.  Preserve the
    # group id while the leader is alive: after a native crash ``getpgid(pid)``
    # can no longer resolve it, even though grandchildren in that same group
    # may still be alive and holding our stdout/stderr pipes open.
    if os.name != "nt" and start_new_session:
        process._marvis_process_group_id = int(process.pid)  # type: ignore[attr-defined]

    def kill_worker() -> None:
        _terminate_worker_process(
            process,
            owns_process_group=start_new_session,
        )
    monitor = ProcessTreeResourceMonitor(
        pid_getter=lambda: process.pid,
        memory_limit_mb=rss_limit_mb,
        # The monitor already uses psutil to terminate a precise descendant
        # tree. Top-level workers additionally own a process group, so retain
        # the group kill there; nested workers must never kill that shared group.
        on_limit=kill_worker if start_new_session else None,
    )
    watcher = (
        _ProgressFileWatcher(progress_path, progress_callback)
        if progress_path is not None and callable(progress_callback)
        else None
    )
    cancellation_watcher = (
        _WorkerCancellationWatcher(cancellation_check, kill_worker)
        if callable(cancellation_check)
        else None
    )
    if watcher is not None:
        try:
            watcher.start()
        except Exception:
            logger.warning("failed to start tool progress watcher", exc_info=True)
            watcher = None
    if cancellation_watcher is not None:
        cancellation_watcher.start()
    try:
        try:
            with monitor:
                stdout, stderr = process.communicate(
                    json.dumps(job, ensure_ascii=False),
                    timeout=int(timeout),
                )
            # Once a worker has emitted a complete protocol result, its governed
            # side effect and evidence boundary is complete.  A stop arriving in
            # the tiny gap between result emission and ``communicate`` returning
            # belongs to the next plan boundary and must not relabel or discard
            # this completed result.
            if (
                cancellation_watcher is not None
                and _parse_worker_result(stdout) is None
            ):
                cancellation_watcher.raise_if_cancelled()
        except subprocess.TimeoutExpired as exc:
            kill_worker()
            stdout, stderr = _bounded_reap_worker(process)
            if monitor.memory_limit_exceeded:
                raise WorkerResourceLimitExceeded(monitor.snapshot()) from exc
            raise WorkerTimeoutExpired(
                args,
                int(timeout),
                output=stdout or exc.output,
                stderr=stderr or exc.stderr,
            ) from exc
        except BaseException as exc:
            # Cancellation and interpreter shutdown must not strand a worker that
            # was deliberately started in its own session. Cleanup failures are
            # secondary: retain the original cancellation/interrupt for the caller.
            try:
                kill_worker()
            except BaseException:
                logger.warning("failed to kill interrupted tool worker tree", exc_info=True)
            try:
                _bounded_reap_worker(process)
            except BaseException:
                logger.warning("failed to reap interrupted tool worker", exc_info=True)
            if cancellation_watcher is not None:
                try:
                    cancellation_watcher.raise_if_cancelled()
                except BaseException as cancellation_exc:
                    raise cancellation_exc from exc
            raise
        if monitor.memory_limit_exceeded:
            raise WorkerResourceLimitExceeded(monitor.snapshot())
        completed = subprocess.CompletedProcess(
            args,
            process.returncode,
            stdout,
            stderr,
        )
        if (
            os.name != "nt"
            and start_new_session
            and int(completed.returncode or 0) < 0
        ):
            # A native-crashed aggregate can close the host-facing pipes while
            # nested recipe workers (which own independent PIPEs) keep running.
            # That path never reaches TimeoutExpired, so synchronously reap the
            # cached worker group before returning the signal result.
            _terminate_worker_process(process, owns_process_group=True)
        if not start_new_session:
            resource_error = _nested_worker_signal_resource_error(
                completed,
                cpu_limit_seconds=int(job.get("cpu_limit_seconds") or 0),
            )
            if resource_error is not None:
                raise resource_error
        return completed
    finally:
        if cancellation_watcher is not None:
            cancellation_watcher.close()
        if watcher is not None:
            watcher.close()
        _cleanup_progress_path(progress_path)


class _WorkerCancellationWatcher:
    """Bridge a cooperative callback to a process-tree interrupt.

    ``Popen.communicate`` cannot observe an in-memory cancellation token while
    it is blocked waiting for a long tool.  This tiny watcher polls the token,
    terminates the worker's owned process tree on cancellation, and relays the
    original exception back to the invoking thread after ``communicate`` wakes.
    """

    def __init__(
        self,
        cancellation_check: Callable[[], None],
        terminate: Callable[[], None],
    ) -> None:
        self._cancellation_check = cancellation_check
        self._terminate = terminate
        self._stop = threading.Event()
        self._cancelled: list[BaseException] = []
        self._thread = threading.Thread(
            target=self._run,
            name="marvis-worker-cancellation",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise self._cancelled[0]

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self._cancellation_check()
            except BaseException as exc:
                self._cancelled.append(exc)
                try:
                    self._terminate()
                except BaseException:
                    logger.warning(
                        "failed to terminate cancelled tool worker tree",
                        exc_info=True,
                    )
                return


class _ProgressFileWatcher:
    """Poll a worker's atomic progress file without affecting its result."""

    def __init__(self, path: Path, callback: Callable[[dict], None]) -> None:
        self._path = path
        self._callback = callback
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_raw: bytes | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"marvis-progress-{path.stem[:12]}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        # Read before and after stopping so a worker's final atomic replace is
        # not lost in the gap between process exit and the poll interval.
        self._deliver_latest()
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._deliver_latest()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self._deliver_latest()

    def _deliver_latest(self) -> None:
        try:
            raw = self._path.read_bytes()
            if not raw or len(raw) > MAX_PROGRESS_BYTES:
                return
            with self._lock:
                if raw == self._last_raw:
                    return
                payload = _sanitize_progress_payload(json.loads(raw.decode("utf-8")))
                if payload is None:
                    return
                self._last_raw = raw
            try:
                self._callback(payload)
            except Exception:
                logger.warning("tool progress callback failed", exc_info=True)
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return


_PROGRESS_SCALAR_FIELDS = frozenset({
    "kind",
    "algorithm",
    "algorithm_index",
    "algorithm_total",
    "trial",
    "trial_total",
    "stage",
    "completed_trials",
    "total_trials",
    "percent",
    "selection_score",
    "test_ks",
    "best_selection_score",
    "best_test_ks",
    "cache_hit",
    "checkpoint_saved",
})
_PROGRESS_BEST_FIELDS = frozenset({"selection_score", "test_ks"})
_PROGRESS_BOOLEAN_FIELDS = frozenset({"cache_hit", "checkpoint_saved"})


def _sanitize_progress_payload(value) -> dict | None:
    if not isinstance(value, dict) or value.get("kind") != "model_tuning":
        return None
    result: dict[str, Any] = {}
    for key in _PROGRESS_SCALAR_FIELDS:
        item = value.get(key)
        if isinstance(item, str):
            result[key] = item[:120]
        elif isinstance(item, bool):
            if key in _PROGRESS_BOOLEAN_FIELDS:
                result[key] = item
        elif isinstance(item, int):
            result[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            result[key] = item
        elif item is None and key in {
            "selection_score",
            "test_ks",
            "best_selection_score",
            "best_test_ks",
        }:
            result[key] = None
    best = value.get("best_by_algorithm")
    if isinstance(best, dict):
        clean_best: dict[str, dict] = {}
        for algorithm, metrics in list(best.items())[:20]:
            if not isinstance(metrics, dict):
                continue
            clean_metrics = {}
            for key in _PROGRESS_BEST_FIELDS:
                metric = metrics.get(key)
                if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                    metric = float(metric)
                    if math.isfinite(metric):
                        clean_metrics[key] = metric
                elif metric is None:
                    clean_metrics[key] = None
            clean_best[str(algorithm)[:80]] = clean_metrics
        result["best_by_algorithm"] = clean_best
    return result


def _new_progress_path(workspace: Path) -> Path:
    root = assert_within(workspace, workspace / ".runtime" / "progress")
    root.mkdir(parents=True, exist_ok=True)
    return assert_within(workspace, root / f"{uuid.uuid4().hex}.json")


def _try_new_progress_path(workspace: Path) -> Path | None:
    try:
        return _new_progress_path(workspace)
    except Exception:
        logger.warning("failed to prepare tool progress path", exc_info=True)
        return None


def _cleanup_progress_path(path: Path | None) -> None:
    if path is None:
        return
    for candidate in (path, path.with_name(f"{path.name}.tmp")):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            logger.warning("failed to clean worker progress file %s", candidate, exc_info=True)


def _sweep_stale_progress_files(workspace: Path) -> None:
    """Bounded cleanup for host-owned telemetry left by a hard process crash."""

    try:
        root = assert_within(workspace, Path(workspace) / ".runtime" / "progress")
        if not root.is_dir():
            return
        deadline = time.time() - PROGRESS_STALE_AFTER_SECONDS
        inspected = 0
        for candidate in root.iterdir():
            if inspected >= PROGRESS_SWEEP_MAX_ENTRIES:
                break
            inspected += 1
            if not _is_host_progress_file(candidate.name):
                continue
            try:
                if candidate.stat().st_mtime < deadline:
                    candidate.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "failed to sweep stale worker progress file %s",
                    candidate,
                    exc_info=True,
                )
    except (OSError, PermissionError):
        logger.warning("failed to sweep stale worker progress files", exc_info=True)


def _is_host_progress_file(name: str) -> bool:
    suffixes = (".json", ".json.tmp", ".jsonl", ".jsonl.tmp")
    suffix = next((item for item in suffixes if name.endswith(item)), None)
    if suffix is None:
        return False
    token = name[: -len(suffix)]
    return len(token) == 32 and all(char in "0123456789abcdef" for char in token)


def _terminate_worker_process(
    process: subprocess.Popen,
    *,
    owns_process_group: bool,
) -> None:
    """Terminate a worker without ever widening a nested kill to its parent group."""

    if owns_process_group:
        # Do this even when the direct worker already exited.  A native-crashed
        # group leader can leave learner subprocesses behind; their inherited
        # pipes make ``communicate`` time out while ``poll()`` is non-None.
        _kill_worker_tree(process)
        return
    if process.poll() is not None:
        return
    if terminate_process_tree_by_pid(
        int(process.pid),
        timeout_seconds=WORKER_REAP_TIMEOUT_SECONDS,
    ):
        return
    if os.name == "nt":
        _taskkill_worker_tree(int(process.pid))
        return
    # psutil is a required runtime dependency in normal deployments.  Keep a
    # narrow last-resort fallback for stripped test/repair environments; the
    # bounded reap below closes inherited pipes so cancellation cannot hang.
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass


def _bounded_reap_worker(
    process: subprocess.Popen,
    *,
    timeout_seconds: float = WORKER_REAP_TIMEOUT_SECONDS,
) -> tuple[str | bytes | None, str | bytes | None]:
    """Collect worker pipes within a hard deadline, even if a descendant owns them."""

    try:
        return process.communicate(timeout=max(0.05, float(timeout_seconds)))
    except subprocess.TimeoutExpired as exc:
        _close_worker_pipes(process)
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except (OSError, ProcessLookupError):
                pass
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=max(0.05, float(timeout_seconds)))
            except (subprocess.TimeoutExpired, OSError):
                pass
        return exc.output, exc.stderr


def _close_worker_pipes(process: subprocess.Popen) -> None:
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, name, None)
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except OSError:
                pass


def _taskkill_worker_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(int(pid))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _kill_worker_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        _taskkill_worker_tree(int(process.pid))
        return
    try:
        process_group_id = getattr(process, "_marvis_process_group_id", None)
        if process_group_id is None:
            if process.poll() is not None:
                return
            process_group_id = os.getpgid(process.pid)
        # Never widen a cleanup bug into killing MARVIS itself.  Dead leaders
        # use only the PGID cached at successful start_new_session launch — we
        # deliberately do not call getpgid(dead_pid), which could observe a
        # subsequently reused PID.
        if int(process_group_id) == int(os.getpgrp()):
            logger.error(
                "refusing to terminate host process group %s for worker %s",
                process_group_id,
                process.pid,
            )
            return
        os.killpg(int(process_group_id), signal.SIGKILL)
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
    return redact_text(text)[-limit:]


def _hash_inputs(inputs: dict) -> str:
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _effect_execution_id(effect_execution) -> str:
    value = getattr(effect_execution, "effect_execution_id", None)
    if value is None:
        value = getattr(effect_execution, "id", None)
    if not str(value or "").strip():
        raise ValueError("governance reservation did not return an effect execution id")
    return str(value)


def _effect_reservation_id(effect_execution) -> str:
    value = getattr(effect_execution, "reservation_id", None)
    if not str(value or "").strip():
        raise ValueError("governance reservation did not return a reservation id")
    return str(value)


def _runtime_generation(effect_execution, execution_context) -> str:
    value = getattr(effect_execution, "runtime_generation", None)
    if not str(value or "").strip():
        value = getattr(execution_context, "runtime_generation", None)
    if not str(value or "").strip():
        raise ValueError("execution context did not include a runtime generation")
    return str(value)


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
