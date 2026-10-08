"""Real monitoring intake actions and source-bound post-run inspection."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import time

from pydantic import BaseModel, ConfigDict, Field


class MonitoringBusinessInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    material_path: str = Field(min_length=1)
    target_col: str | None = Field(default=None, min_length=1)


def monitoring_entry(case):
    if any(a.kind == "submit_model_monitoring_request" for a in case.actions):
        return "standard_model_monitoring_agent"
    if any(a.kind == "start_model_monitoring_workflow" for a in case.actions):
        return "manual_monitoring_workflow"
    return None


def monitoring_materials(case):
    return {a.monitoring_request.material_path for a in case.actions if a.monitoring_request is not None}


def start_monitoring(journey, action, dataset_root):
    from marvis.api_schemas import DataWorkspaceSnapshotResponse, DataWorkspaceUpdateRequest
    from marvis.agent.monitoring_setup import ModelMonitoringSetupRequest
    from .runtime_runner import RuntimeJourneyError, RuntimeBudgetExceeded

    material = next(m for m in journey.case.materials if m.path == action.monitoring_request.material_path)
    path = (dataset_root / material.path).resolve()
    if not path.is_relative_to(dataset_root.resolve()):
        raise RuntimeJourneyError("monitoring_material_identity_mismatch")
    with path.open("rb") as source, tempfile.TemporaryFile() as stream:
        sha = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            if time.monotonic() >= journey.deadline:
                raise RuntimeBudgetExceeded("monitoring material snapshot wall budget")
            sha.update(chunk)
            stream.write(chunk)
        if sha.hexdigest() != material.sha256:
            raise RuntimeJourneyError("monitoring_material_identity_mismatch")
        stream.seek(0)
        journey.record_human_action(action, phase="material_upload")
        datasets = journey.json_request(
            "POST", f"/api/tasks/{journey.task_id}/datasets/upload", label="upload_monitoring_material",
            files={"file": (path.name, stream)}, data={"role": material.role},
        )["datasets"]
    if len(datasets) != 1 or datasets[0]["task_id"] != journey.task_id:
        raise RuntimeJourneyError("monitoring_dataset_selection_not_unique")
    sample = datasets[0]
    experiments = journey.json_request(
        "GET", f"/api/tasks/{journey.task_id}/experiments", label="read_monitoring_experiments",
    )["experiments"]
    selected = [e for e in experiments if e["status"] in {"selected", "handed_off", "validated"} and e["artifact_id"]]
    if len(selected) != 1 or selected[0]["task_id"] != journey.task_id:
        raise RuntimeJourneyError("monitoring_selected_experiment_not_unique")
    route = f"/api/tasks/{journey.task_id}/data-workspace"
    workspace = DataWorkspaceSnapshotResponse.model_validate(
        journey.json_request("GET", route, label="read_monitoring_workspace"))
    reset = DataWorkspaceUpdateRequest(
        active_dataset_id=sample["id"], active_dataset_content_hash=sample["content_hash"],
        page="overview", selected_field=None,
        semantic_mapping={"target_col": None, "field_roles": {}, "business_names": {}},
    )
    workspace = DataWorkspaceSnapshotResponse.model_validate(journey.json_request(
        "PUT", route, label="human_monitoring_dataset_selection",
        headers={"If-Match": str(workspace.revision)}, json=reset.model_dump(),
    ))
    request = ModelMonitoringSetupRequest(
        experiment_id=selected[0]["id"], dataset_id=sample["id"],
        expected_content_hash=sample["content_hash"], workspace_revision=workspace.revision,
        analysis_generation=workspace.analysis_generation, target_col=action.monitoring_request.target_col,
    )
    journey.record_human_action(action, phase="monitoring_proposal")
    if action.kind == "submit_model_monitoring_request":
        journey.json_request(
            "POST", f"/api/tasks/{journey.task_id}/agent/messages", label="human_monitoring_proposal",
            json={"content": action.content, "model_monitoring_request": request.model_dump(),
                  "acceptance_mode": "manual_review"},
        )
    else:
        journey.json_request(
            "POST", f"/api/tasks/{journey.task_id}/plans", label="create_manual_monitoring_workflow",
            json={"goal": "模型监控", "slots": {"experiment_id": selected[0]["id"],
                  "dataset_id": sample["id"], "target_col": request.target_col}},
        )
    plans = journey.wait_idle()
    monitors = [p for p in plans if p["template_id"] == "monitoring_run"]
    if len(monitors) != 1 or monitors[0]["status"] != "validated" or any(
        s["status"] != "pending" or s.get("output_ref") for s in monitors[0]["steps"]
    ):
        raise RuntimeJourneyError("monitoring_proposal_executed_before_confirmation")
    journey.monitoring_submission = {"request": request.model_dump(), "plan_id": monitors[0]["id"],
                                     "runtime_entry": monitoring_entry(journey.case)}


def confirm_monitoring(journey, action):
    from .runtime_runner import RuntimeJourneyError

    submission = getattr(journey, "monitoring_submission", None)
    if submission is None:
        raise RuntimeJourneyError("monitoring_confirmation_has_no_proposal")
    plan = journey.json_request("GET", f"/api/plans/{submission['plan_id']}", label="read_monitoring_plan")["plan"]
    if plan["status"] != "validated" or plan["task_id"] != journey.task_id:
        raise RuntimeJourneyError("monitoring_proposal_is_stale")
    journey.record_human_action(action)
    # Agent tasks reject the generic /plan/confirm bypass even if a human
    # created the workflow through /plans. Both entries use its existing gate.
    journey.json_request(
        "POST", f"/api/tasks/{journey.task_id}/agent/messages", label="human_monitoring_plan_start",
        json={"content": action.content, "ui_action": "start_plan", "expected_plan_id": plan["id"],
              **plan["confirmation_snapshot"], "acceptance_mode": "manual_review"},
    )
    journey.wait_idle()


def monitoring_receipt(workspace: Path, task_id, evidence, outputs, submission):
    """Verify actual prediction rows and PSI, rather than accepting output counters."""
    import numpy as np
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.repositories.datasets import DatasetRepository
    from marvis.repositories.modeling import ModelingRepository
    from marvis.repositories.plans import PlanRepository
    from marvis.packs.modeling.scoring import _ModelArtifactScorer
    from marvis.packs.modeling.monitor_binding import monitoring_binding_hash
    from marvis.packs.modeling._runtime import _artifact_base_dir
    from marvis.settings import Settings
    from .runtime_contracts import digest

    if not submission:
        raise ValueError("monitoring entry is unobserved")
    settings = Settings(workspace.resolve())
    plans = PlanRepository(settings.db_path)
    plan = plans.load_plan(submission["plan_id"])
    request = submission["request"]
    if plan.task_id != task_id or plan.template_id != "monitoring_run":
        raise ValueError("monitoring plan owner differs")
    steps = [s for s in evidence["steps"] if s["tool"] in {"modeling.score_dataset", "modeling.monitor_run"}]
    if len(steps) != 2 or not all(s.get("binding_verified") for s in steps):
        raise ValueError("monitoring tool evidence unavailable")
    scored = outputs[next(s["id"] for s in steps if s["tool"] == "modeling.score_dataset")]
    result = outputs[next(s["id"] for s in steps if s["tool"] == "modeling.monitor_run")]
    model_repo = ModelingRepository(settings.db_path)
    experiment = model_repo.get_experiment(request["experiment_id"])
    artifact = model_repo.get_model_artifact(experiment.artifact_id)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    raw = registry.get(request["dataset_id"])
    child = registry.get(scored["result_dataset_id"])
    if (raw.task_id != task_id or child.task_id != task_id
            or raw.content_hash != request["expected_content_hash"] or raw.id == child.id
            or result["dataset_id"] != child.id or result["artifact_id"] != artifact.id
            or result["experiment_id"] != experiment.id or scored["artifact_id"] != artifact.id):
        raise ValueError("monitoring data/model identity differs")
    source = registry.read_authenticated_parquet_snapshot(raw.id)
    predictions = registry.read_authenticated_parquet_snapshot(child.id)
    model = _ModelArtifactScorer(artifact, base_dir=_artifact_base_dir(settings, task_id), replay_preprocessing=True)
    actual_scores = predictions[scored["score_col"]].to_numpy(dtype=float)
    if not np.allclose(actual_scores, model.score(source), rtol=1e-12, atol=1e-12):
        raise ValueError("monitoring scored data differs from selected model")
    if not predictions[source.columns].equals(source) or len(source) != result["row_count"]:
        raise ValueError("monitoring source rows changed")
    baseline = artifact.baseline_distributions
    edges = np.array(baseline["score_edges"], dtype=float)
    indices = np.searchsorted(edges[1:-1], actual_scores, side="right")
    observed = np.bincount(indices, minlength=len(edges) - 1) / len(actual_scores)
    expected = np.array(baseline["score_distribution"]["train"]["bin_proportions"], dtype=float)
    observed, expected = np.where(observed <= 0, 1e-6, observed), np.where(expected <= 0, 1e-6, expected)
    observed, expected = observed / observed.sum(), expected / expected.sum()
    psi = float(np.sum((observed - expected) * np.log(observed / expected)))
    check = next(c for c in result["checks"] if c["id"] == "score_psi")
    if not np.isclose(check["value"], psi, rtol=1e-10, atol=1e-12):
        raise ValueError("monitoring PSI differs from baseline and actual scored rows")
    if request["target_col"] is None:
        label_checks = [c for c in result["checks"] if c["id"] in {"ks_drop", "auc_drop"}]
        if len(label_checks) != 2 or any(c["value"] is not None or c["status"] != "n/a" for c in label_checks):
            raise ValueError("unlabeled monitoring invented label performance")
    frozen = plan.steps[0].inputs.get("monitoring_binding")
    if submission["runtime_entry"] == "standard_model_monitoring_agent":
        from marvis.packs.modeling.monitor_binding import validate_monitoring_binding
        if not frozen:
            raise ValueError("Agent monitoring has no frozen binding")
        validate_monitoring_binding(settings, task_id, frozen, scored_dataset_id=child.id)
    return {"verified": True, "runtime_entry": submission["runtime_entry"],
            "source_content_hash": raw.content_hash, "scored_content_hash": child.content_hash,
            "experiment_id": experiment.id, "artifact_id": artifact.id,
            "baseline_sha256": monitoring_binding_hash(baseline), "binding_sha256": digest(frozen) if frozen else None,
            "actual_rows": len(source), "score_psi": psi,
            "label_maturity_assurance": "unknown", "business_acceptance": "not_established",
            "monitoring_semantic_text_inference": "not_exercised",
            "scope": "actual_source_model_scoring_and_baseline_psi; not_external_outcome_truth"}
