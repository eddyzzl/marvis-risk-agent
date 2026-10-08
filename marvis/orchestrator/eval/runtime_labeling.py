"""Public label-workflow actions and post-run evidence, never expected answers."""

from __future__ import annotations

import json
import csv
import io
from pathlib import Path
from types import SimpleNamespace

from pydantic import ConfigDict, create_model

from marvis.api_schemas import LabelingSetupRequest

_BINDING_FIELDS = {
    "dataset_id",
    "expected_content_hash",
    "workspace_revision",
    "analysis_generation",
}
# Reuse the product's strict business-field types; dynamic identities may only
# come from this run's upload and current workspace, never from the case file.
LabelingBusinessInput = create_model(
    "LabelingBusinessInput",
    __config__=ConfigDict(extra="forbid", strict=True, allow_inf_nan=False),
    **{
        name: (field.annotation, field)
        for name, field in LabelingSetupRequest.model_fields.items()
        if name not in _BINDING_FIELDS
    },
)


def submit_labeling_request(journey, action):
    from marvis.api_schemas import (
        DataWorkspaceSnapshotResponse,
        DataWorkspaceUpdateRequest,
    )
    from .runtime_runner import RuntimeJourneyError

    if journey.case.task.task_type != "data_join" or len(journey.uploaded_samples) != 1:
        raise RuntimeJourneyError("labeling_sample_binding_not_unique")
    sample = journey.uploaded_samples[0]
    if sample.get("task_id") != journey.task_id or sample.get("role") != "sample":
        raise RuntimeJourneyError("labeling_sample_binding_wrong_owner")
    route = f"/api/tasks/{journey.task_id}/data-workspace"
    snapshot = DataWorkspaceSnapshotResponse.model_validate(
        journey.json_request("GET", route, label="read_labeling_workspace")
    )
    if snapshot.task_id != journey.task_id:
        raise RuntimeJourneyError("labeling_sample_binding_wrong_owner")
    journey.record_human_action(action, phase="dataset_selection")
    if (snapshot.active_dataset_id, snapshot.active_dataset_content_hash) != (
        sample["id"],
        sample["content_hash"],
    ):
        reset = DataWorkspaceUpdateRequest(
            active_dataset_id=sample["id"],
            active_dataset_content_hash=sample["content_hash"],
            page="overview",
            selected_field=None,
            semantic_mapping={
                "target_col": None,
                "field_roles": {},
                "business_names": {},
            },
        )
        snapshot = DataWorkspaceSnapshotResponse.model_validate(
            journey.json_request(
                "PUT",
                route,
                label="human_labeling_dataset_selection",
                headers={"If-Match": str(snapshot.revision)},
                json=reset.model_dump(),
            )
        )
    if snapshot.task_id != journey.task_id or (
        snapshot.active_dataset_id,
        snapshot.active_dataset_content_hash,
    ) != (sample["id"], sample["content_hash"]):
        raise RuntimeJourneyError("labeling_sample_binding_wrong_owner")
    request = LabelingSetupRequest(
        **action.labeling_request.model_dump(exclude_none=True),
        dataset_id=sample["id"],
        expected_content_hash=sample["content_hash"],
        workspace_revision=snapshot.revision,
        analysis_generation=snapshot.analysis_generation,
    )
    journey.record_human_action(action, phase="label_declaration")
    journey.json_request(
        "POST",
        f"/api/tasks/{journey.task_id}/agent/messages",
        label="human_labeling_proposal",
        json={
            "content": action.content,
            "acceptance_mode": journey.case.acceptance_mode,
            "labeling_request": request.model_dump(exclude_none=True),
        },
    )
    if journey.wait_idle():
        raise RuntimeJourneyError("labeling_proposal_created_plan_before_confirmation")
    messages = journey.json_request(
        "GET",
        f"/api/tasks/{journey.task_id}/agent/messages",
        label="read_labeling_proposal",
    )["messages"]
    assistant = [m for m in messages if m.get("role") == "assistant"]
    metadata = assistant[-1].get("metadata", {}) if assistant else {}
    proposal = metadata.get("labeling_proposal", {})
    from marvis.packs.labeling.contracts import LabelingRequest

    contract = LabelingRequest(**request.model_dump())
    if (
        metadata.get("kind") != "labeling_preplan_confirmation"
        or proposal.get("request") != contract.to_dict()
        or proposal.get("proposal_hash") != contract.contract_hash
        or proposal.get("requires_human_confirmation") is not True
    ):
        raise RuntimeJourneyError("labeling_proposal_contract_mismatch")


def download_labeling_results(journey, action):
    from .runtime_contracts import digest
    from marvis.packs.labeling.tools import _task_artifact_download_url
    from .runtime_runner import RuntimeJourneyError, _tool_name

    matches = [
        s
        for p in journey.plans()
        for s in p["steps"]
        if _tool_name(s["tool_ref"]) == "labeling.define_label"
        and s["status"] == "done"
    ]
    if len(matches) != 1:
        raise RuntimeJourneyError("labeling_result_not_unique")
    output = journey.json_request(
        "GET", f"/api/step-outputs/{matches[0]['id']}", label="read_labeling_output"
    )
    journey.record_human_action(action)
    for kind in ("dataset", "evidence"):
        route = _task_artifact_download_url(
            journey.task_id,
            output[f"{kind}_artifact_id"],
            output[f"{kind}_content_hash"],
        )
        if route != output[f"{kind}_download_url"]:
            raise RuntimeJourneyError("labeling_download_binding_mismatch")
        response = journey.request("GET", route, label=f"download_labeling_{kind}")
        if (
            response.status_code != 200
            or digest(response.content) != output[f"{kind}_content_hash"]
        ):
            raise RuntimeJourneyError("labeling_download_integrity_failed")
        journey.events[-1].update(
            sha256=digest(response.content), size_bytes=len(response.content)
        )


def labeling_receipt(workspace: Path, task_id: str, output: dict) -> dict:
    """Authenticate actual registered label artifacts and replay their source.

    Only digests/counts leave the private scoring process. No model narration or
    output counter can substitute for the actual result Parquet and CSV bytes.
    """
    from .runtime_contracts import digest
    from marvis.data.backend import DataBackend
    from marvis.data.dataset_export import _csv_cell, _safe_string, _SafetyCounts
    from marvis.data.registry import DatasetRegistry
    from marvis.packs.labeling.evidence import _replay_labeling_evidence
    from marvis.repositories.datasets import DatasetRepository
    from marvis.repositories.task_artifacts import TaskArtifactRepository
    from marvis.settings import Settings

    settings = Settings(workspace.resolve())
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    runtime = SimpleNamespace(settings=settings, registry=registry)
    request, _, labels, evidence = _replay_labeling_evidence(
        runtime,
        task_id,
        {
            "artifact_id": output["evidence_artifact_id"],
            "content_hash": output["evidence_content_hash"],
        },
    )
    if any(
        output[key] != evidence["quality"][key]
        for key in ("n_loans", "n_bad", "n_good", "n_unmatured", "bad_rate")
    ) or (
        output["quality"] != evidence["quality"]
        or output["maturity"] != evidence["maturity"]
        or output["source_dataset_id"] != request.dataset_id
        or output["source_content_hash"] != request.expected_content_hash
        or output["result_dataset_id"] != evidence["result"]["dataset_id"]
        or output["result_content_hash"] != evidence["result"]["content_hash"]
        or output["proposal_hash"] != request.contract_hash
        or output["target_col"] != request.target_col
    ):
        raise ValueError("label output differs from authenticated evidence")
    exported_artifact = TaskArtifactRepository(settings.db_path).get_for_task(
        task_id, output["dataset_artifact_id"]
    )
    path = Path(exported_artifact["path"])
    if (
        exported_artifact["kind"] != "labeling_dataset_csv"
        or exported_artifact["origin_tool"] != "labeling.define_label"
        or exported_artifact["content_hash"] != output["dataset_content_hash"]
        or any(item.is_symlink() for item in (path, *path.parents))
        or not path.resolve().is_relative_to(
            (settings.tasks_dir / task_id / "labeling").resolve()
        )
        or exported_artifact["provenance"]["result_content_hash"]
        != output["result_content_hash"]
    ):
        raise ValueError("label CSV identity differs from native receipt")
    csv_bytes = path.read_bytes()
    if digest(csv_bytes) != exported_artifact["content_hash"]:
        raise ValueError("label CSV bytes differ from native receipt")
    # Honor the existing Excel-safe text-marker contract. Never strip arbitrary
    # apostrophes from source IDs or mistake a protected ID for a changed label.
    safety = _SafetyCounts()
    expected_csv = [[_safe_string(c, safety=safety) for c in labels.columns]] + [
        [
            _csv_cell(
                row[c],
                force_text=c in {request.id_col, request.cohort_col},
                safety=safety,
            )
            for c in labels.columns
        ]
        for row in json.loads(labels.to_json(orient="records"))
    ]
    if (
        list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        != expected_csv
    ):
        raise ValueError("label CSV differs from registered label values")
    columns = [request.id_col, request.cohort_col, request.target_col]

    def records(frame):
        values = frame[columns].sort_values(request.id_col).to_json(orient="values")
        return json.loads(values)

    return {
        "verified": True,
        "labels_sha256": digest(records(labels)),
        "dataset_download_sha256": exported_artifact["content_hash"],
        "evidence_download_sha256": output["evidence_content_hash"],
        "result_rows": len(labels),
        "rows_excluded_after_as_of": evidence["source"]["rows_excluded_after_as_of"],
        "scope": "registered_source_replay_and_actual_labels; not_external_outcome_truth",
    }
