from __future__ import annotations

import json
from pathlib import Path

from marvis.validation.overfitting import overfitting_check_from_validation_results


def agent_evidence_from_settings(settings, task_id: str) -> dict:
    task_dir = settings.tasks_dir / task_id
    validation_results = _read_json(task_dir / "outputs" / "validation_results.json")
    return {
        "scan": _read_json(task_dir / "execution" / "scan_result.json"),
        "notebook_steps": _read_json(task_dir / "execution" / "notebook_steps.json"),
        "contract": _read_json(task_dir / "execution" / "runtime_contract.json"),
        "reproducibility": _read_json(task_dir / "outputs" / "reproducibility_result.json"),
        "validation_results": agent_validation_results_with_overfitting_check(validation_results),
    }


def agent_validation_results_with_overfitting_check(validation_results):
    if not isinstance(validation_results, dict):
        return validation_results
    return {
        **validation_results,
        "overfitting_check": overfitting_check_from_validation_results(validation_results),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = [
    "agent_evidence_from_settings",
    "agent_validation_results_with_overfitting_check",
]
