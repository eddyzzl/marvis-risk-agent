from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from marvis.db import PluginRepository
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import ToolRegistry
from marvis.plugins.schema_validation import validate_against_schema
from marvis.plugins.errors import SchemaValidationError


@dataclass(frozen=True)
class ToolContext:
    task_id: str
    seed: int | None
    datasets_root: Path
    workspace: Path

    def load_dataset_path(self, dataset_id: str) -> Path:
        return self.datasets_root / dataset_id


@dataclass
class ToolResult:
    ok: bool
    output: dict | None
    error: str | None
    error_kind: str | None
    duration_ms: int
    stdout_tail: str = ""
    stderr_tail: str = ""


class ToolRunner:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        repo: PluginRepository,
        *,
        python_executable: str,
        datasets_root: Path,
        workspace: Path,
    ):
        self._tools = tool_registry
        self._repo = repo
        self._python_executable = python_executable
        self._datasets_root = datasets_root
        self._workspace = workspace

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
        try:
            manifest, tool = self._tools.resolve_with_manifest(ref)
            validate_against_schema(inputs, tool.input_schema, label="inputs")
        except SchemaValidationError as exc:
            result = _failed_result(started, "schema", str(exc))
            self._write_audit(target_ref, inputs, result)
            return result

        job = {
            "module": manifest.module,
            "entrypoint": tool.entrypoint,
            "inputs": inputs,
            "task_id": task_id,
            "seed": seed,
            "datasets_root": str(self._datasets_root),
            "workspace": str(self._workspace),
            "memory_limit_mb": tool.memory_limit_mb,
        }
        try:
            completed = subprocess.run(
                [self._python_executable, "-m", "marvis.plugins.subprocess_worker"],
                input=json.dumps(job, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=tool.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            result = _failed_result(
                started,
                "timeout",
                f"tool {target_ref} timed out after {tool.timeout_seconds}s",
                stdout_tail=_tail(exc.stdout),
                stderr_tail=_tail(exc.stderr),
            )
            self._write_audit(target_ref, inputs, result)
            return result

        protocol = _parse_worker_result(completed.stdout)
        if protocol is None:
            result = _failed_result(
                started,
                "protocol",
                f"worker returned invalid protocol with exit code {completed.returncode}",
                stdout_tail=_tail(completed.stdout),
                stderr_tail=_tail(completed.stderr),
            )
            self._write_audit(target_ref, inputs, result)
            return result

        if not protocol.get("ok"):
            result = _failed_result(
                started,
                str(protocol.get("error_kind") or "execution"),
                str(protocol.get("error") or "tool execution failed"),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("traceback") or protocol.get("stderr") or completed.stderr),
            )
            self._write_audit(target_ref, inputs, result)
            return result

        output = protocol.get("output")
        try:
            validate_against_schema(output, tool.output_schema, label=f"output:{target_ref}")
        except SchemaValidationError as exc:
            result = _failed_result(
                started,
                "schema",
                str(exc),
                stdout_tail=_tail(protocol.get("stdout") or completed.stdout),
                stderr_tail=_tail(protocol.get("stderr") or completed.stderr),
            )
            self._write_audit(target_ref, inputs, result)
            return result

        result = ToolResult(
            ok=True,
            output=output,
            error=None,
            error_kind=None,
            duration_ms=_duration_ms(started),
            stdout_tail=_tail(protocol.get("stdout") or ""),
            stderr_tail=_tail(protocol.get("stderr") or ""),
        )
        self._write_audit(target_ref, inputs, result)
        return result

    def _write_audit(self, target_ref: str, inputs: dict, result: ToolResult) -> None:
        self._repo.write_audit(
            kind="tool.invoke",
            target_ref=target_ref,
            inputs_hash=_hash_inputs(inputs),
            outcome="succeeded" if result.ok else "failed",
            detail={
                "error_kind": result.error_kind,
                "duration_ms": result.duration_ms,
            },
        )


def _failed_result(
    started: float,
    error_kind: str,
    error: str,
    *,
    stdout_tail: str = "",
    stderr_tail: str = "",
) -> ToolResult:
    return ToolResult(
        ok=False,
        output=None,
        error=error,
        error_kind=error_kind,
        duration_ms=_duration_ms(started),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _parse_worker_result(stdout: str) -> dict[str, Any] | None:
    line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
    if not line:
        return None
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _tail(value: str | bytes | None, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    return text[-limit:]


def _hash_inputs(inputs: dict) -> str:
    raw = json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
