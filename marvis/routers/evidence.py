from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading

from fastapi import APIRouter, Request

from marvis.api_task_helpers import get_task_or_404
from marvis.db import TaskRepository
from marvis.execution_environment import load_execution_environment


router = APIRouter(prefix="/api", tags=["evidence"])

# PERF-6: the 1s polling loop re-reads and json.loads these files on every tick.
# Cache the parsed result per absolute path, keyed by (mtime_ns, size) so a cache
# hit skips both the read_text() and json.loads() calls; any change to the file
# (including truncation/rewrite with the same mtime granularity but a different
# size) invalidates the entry. Process-local and unbounded by task count is fine
# here -- entries are small parsed JSON dicts, one per (task, file) pair actually
# polled, not per task in existence.
_JSON_CACHE_LOCK = threading.Lock()
_JSON_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}


def _repo(request: Request) -> TaskRepository:
    return TaskRepository(request.app.state.settings.db_path)


@router.get("/tasks/{task_id}/evidence")
def get_task_evidence(task_id: str, request: Request) -> dict:
    repo = _repo(request)
    get_task_or_404(repo, task_id)
    settings = request.app.state.settings
    task_dir = settings.tasks_dir / task_id
    notebook_steps = _read_json(task_dir / "execution" / "notebook_steps.json")
    scan_result = _read_json(task_dir / "execution" / "scan_result.json")
    contract = _read_json(task_dir / "execution" / "runtime_contract.json")
    notebook_reproducibility = _read_json(
        task_dir / "outputs" / "reproducibility_result.json"
    )
    results = _read_json(task_dir / "outputs" / "validation_results.json")
    environment = load_execution_environment(settings.workspace)
    if scan_result and notebook_steps:
        scan_result = {
            **scan_result,
            "notebook_steps": notebook_steps.get("steps", []),
        }
    return {
        "scan": scan_result,
        "notebook_steps": (notebook_steps or {}).get("steps", []),
        "notebook_cells": (notebook_steps or {}).get("cells", []),
        "contract": contract or {},
        "reproducibility": (results or {}).get("reproducibility", {})
        or notebook_reproducibility
        or {},
        "execution_environment": asdict(environment),
    }


def _read_json(path: Path) -> dict:
    cache_key = str(path)
    try:
        stat_result = path.stat()
    except OSError:
        with _JSON_CACHE_LOCK:
            _JSON_CACHE.pop(cache_key, None)
        return {}
    fingerprint = (stat_result.st_mtime_ns, stat_result.st_size)
    with _JSON_CACHE_LOCK:
        cached = _JSON_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    parsed = payload if isinstance(payload, dict) else {}
    with _JSON_CACHE_LOCK:
        _JSON_CACHE[cache_key] = (fingerprint, parsed)
    return parsed
