from __future__ import annotations

import json
from pathlib import Path

from marvis.validation.overfitting import overfitting_check_from_validation_results
from marvis.validation.results import (
    normalize_validation_results_lift_order,
    pmml_scoring_result_from_dict,
    pmml_scoring_result_to_dict,
)


def agent_evidence_from_settings(settings, task_id: str) -> dict:
    task_dir = settings.tasks_dir / task_id
    validation_results = normalize_validation_results_lift_order(
        _read_json(task_dir / "outputs" / "validation_results.json")
    )
    pmml_scoring = _read_pmml_scoring_result(
        task_dir / "outputs" / "pmml_scoring_result.json"
    )
    return {
        "scan": _read_json(task_dir / "execution" / "scan_result.json"),
        "notebook_steps": _read_json(task_dir / "execution" / "notebook_steps.json"),
        "contract": _read_json(task_dir / "execution" / "runtime_contract.json"),
        "reproducibility": _read_json(task_dir / "outputs" / "reproducibility_result.json"),
        "pmml_scoring": pmml_scoring,
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


def _read_pmml_scoring_result(path: Path) -> dict:
    payload = _read_json(path)
    if not payload:
        return {}
    try:
        return pmml_scoring_result_to_dict(pmml_scoring_result_from_dict(payload))
    except ValueError:
        return {}


__all__ = [
    "agent_evidence_from_settings",
    "agent_validation_results_with_overfitting_check",
]
