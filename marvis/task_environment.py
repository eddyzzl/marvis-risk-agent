"""Task-scoped tool interpreter override (B-line extension, 2026-10-23 spec §4.4).

The global execution environment (``execution_environment.json``) stays the
default for every task. A repair can pin one task's modeling and feature tools
to another interpreter the platform itself detected and probed; undoing the
repair deletes the file and the task falls back to the global interpreter.

The file lives under ``tasks/<task_id>/repairs/`` and only ever names an
interpreter returned by :func:`probe_python_candidates`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import re
import subprocess
import sys

from marvis.files import write_json_atomic


logger = logging.getLogger(__name__)

TASK_ENVIRONMENT_SCHEMA = "task_execution_environment.v1"
APPLIES_TO = ("modeling", "feature")
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_PROBE_TIMEOUT_SECONDS = 10
_MAX_PROBES = 4


def task_environment_path(tasks_dir: Path, task_id: str) -> Path:
    if not _SAFE_TASK_ID.fullmatch(str(task_id or "")):
        raise ValueError("task id is unsafe for a task environment path")
    return Path(tasks_dir) / str(task_id) / "repairs" / "execution_environment.json"


def load_task_environment(tasks_dir: Path, task_id: str) -> dict | None:
    try:
        payload = json.loads(task_environment_path(tasks_dir, task_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != TASK_ENVIRONMENT_SCHEMA:
        return None
    if not str(payload.get("python_executable") or "").strip():
        return None
    return payload


def save_task_environment(
    tasks_dir: Path,
    task_id: str,
    *,
    python_executable: str,
    label: str,
    python_version: str,
    repair_id: str,
) -> dict:
    payload = {
        "schema_version": TASK_ENVIRONMENT_SCHEMA,
        "python_executable": str(python_executable),
        "label": str(label),
        "python_version": str(python_version),
        "applies_to": list(APPLIES_TO),
        "repair_id": str(repair_id),
        "set_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = task_environment_path(tasks_dir, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload, ensure_ascii=False, indent=2)
    return payload


def clear_task_environment(tasks_dir: Path, task_id: str) -> None:
    task_environment_path(tasks_dir, task_id).unlink(missing_ok=True)


def task_python_resolver(tasks_dir: Path) -> Callable[[str, str], str | None]:
    """``(task_id, plugin_name) -> interpreter`` for :class:`ToolRunner`."""

    def resolve(task_id: str, plugin_name: str) -> str | None:
        if str(plugin_name) not in APPLIES_TO:
            return None
        try:
            payload = load_task_environment(tasks_dir, task_id)
        except ValueError:
            return None
        if payload is None or str(plugin_name) not in (payload.get("applies_to") or ()):
            return None
        executable = Path(str(payload["python_executable"]))
        if not executable.is_file():
            logger.warning("task environment interpreter is missing task_id=%s", task_id)
            return None
        return str(executable)

    return resolve


def _probe_interpreter(executable: str, modules: Sequence[str]) -> str | None:
    """Python version text when ``executable`` imports marvis and ``modules``."""

    imports = "".join(f"importlib.import_module({name!r});" for name in modules)
    code = (
        "import importlib, sys;import marvis;"
        + imports
        + "print('.'.join(str(part) for part in sys.version_info[:3]))"
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or "").strip().splitlines()
    return lines[-1].strip() if lines else None


def _detected_interpreters() -> list[tuple[str, str]]:
    """(label, executable) pairs from the platform's own environment detection."""

    from marvis.execution_environment import detect_execution_environment_options

    found: list[tuple[str, str]] = []
    try:
        options = detect_execution_environment_options()
    except Exception:  # noqa: BLE001 - detection is best effort
        options = []
    for option in options:
        executable = str(getattr(option, "python_executable", "") or "").strip()
        if executable and option.available:
            found.append((str(option.label or executable), executable))
    found.append(("MARVIS 当前运行环境", sys.executable))
    unique: dict[str, tuple[str, str]] = {}
    for label, executable in found:
        try:
            key = str(Path(executable).resolve())
        except OSError:
            continue
        unique.setdefault(key, (label, executable))
    return list(unique.values())


def probe_python_candidates(
    *,
    python_requires: str,
    exclude: Sequence[str] = (),
    required_modules: Sequence[str] = (),
) -> list[dict]:
    """Detected interpreters that satisfy ``python_requires`` and import marvis.

    Probes are bounded (at most four, ten seconds each) and read-only.
    """

    from marvis.plugins.manifest import python_requires_satisfied

    excluded = set()
    for item in exclude:
        try:
            excluded.add(str(Path(item).resolve()))
        except OSError:
            continue
    candidates: list[dict] = []
    probes = 0
    for label, executable in _detected_interpreters():
        try:
            if str(Path(executable).resolve()) in excluded:
                continue
        except OSError:
            continue
        if probes >= _MAX_PROBES:
            break
        probes += 1
        version = _probe_interpreter(executable, required_modules)
        if version is None:
            continue
        try:
            parts = tuple(int(part) for part in version.split(".")[:3])
        except ValueError:
            continue
        if len(parts) != 3 or (python_requires and not python_requires_satisfied(python_requires, parts)):
            continue
        candidates.append({"label": label, "python_executable": executable, "python_version": version})
    return candidates


__all__ = [
    "APPLIES_TO",
    "TASK_ENVIRONMENT_SCHEMA",
    "clear_task_environment",
    "load_task_environment",
    "probe_python_candidates",
    "save_task_environment",
    "task_environment_path",
    "task_python_resolver",
]
