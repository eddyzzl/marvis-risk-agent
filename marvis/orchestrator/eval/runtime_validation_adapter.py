"""Benchmark adapter for the native V2 validation Agent's job/stage carrier.

This observes existing product APIs and files. It does not create plans, tool
receipts or a second runtime, and never feeds scoring expectations to the app.
"""
from __future__ import annotations

import json
from pathlib import Path

from .runtime_contracts import digest


def validation_entry(case):
    if case.task.task_type != "validation":
        return None
    return ("manual_compatibility_workflow" if case.actions[0].kind == "start_validation_workflow"
            else "standard_validation_agent_v2")


def start_agent(journey, action):
    from .runtime_runner import RuntimeJourneyError

    if validation_entry(journey.case) != "standard_validation_agent_v2" or not action.content.strip():
        raise RuntimeJourneyError("validation_agent_start_not_declared")
    if journey.plans():
        raise RuntimeJourneyError("validation_agent_must_not_use_plan")
    journey.interventions += 1
    response = journey.json_request(
        "POST", f"/api/tasks/{journey.task_id}/agent/start", label="agent_initial_turn",
        json={"acceptance_mode": journey.case.acceptance_mode},
    )
    if response.get("task_id") != journey.task_id or response.get("status") != "accepted":
        raise RuntimeJourneyError("validation_agent_start_not_accepted")
    journey.wait_idle()


def confirm_current_report(journey, action):
    from marvis.agent.validation_app_service import latest_pending_agent_report_draft
    from .runtime_runner import RuntimeJourneyError

    if validation_entry(journey.case) != "standard_validation_agent_v2" or not action.content.strip():
        raise RuntimeJourneyError("validation_report_confirmation_not_declared")
    response = journey.json_request(
        "GET", f"/api/tasks/{journey.task_id}/agent/messages", label="read_validation_report_draft",
    )
    messages = response.get("messages", [])
    draft = latest_pending_agent_report_draft(messages)
    if not draft:
        raise RuntimeJourneyError("validation_report_draft_not_confirmable")
    matches = [m for m in messages if m.get("id") == draft["message_id"]]
    metadata = matches[0].get("metadata", {}) if len(matches) == 1 else {}
    if (len(matches) != 1 or matches[0].get("task_id") != journey.task_id
            or metadata.get("fallback") is not False or metadata.get("confirmable") is False
            or metadata.get("streaming") is True
            or type(draft["draft_edit_revision"]) is not int or draft["draft_edit_revision"] < 0
            or type(draft["report_revision"]) is not int or draft["report_revision"] < 0):
        raise RuntimeJourneyError("validation_report_draft_not_confirmable")
    journey.interventions += 1
    journey.validation_confirmation = {
        "draft_message_id": draft["message_id"], "draft_edit_revision": draft["draft_edit_revision"],
        "revision": draft["report_revision"], "values_sha256": digest(draft["values"]),
    }
    journey.json_request(
        "POST", f"/api/tasks/{journey.task_id}/agent/report-draft/confirm",
        label="human_validation_report_confirmation",
        json={"draft_message_id": draft["message_id"], "draft_edit_revision": draft["draft_edit_revision"],
              "revision": draft["report_revision"], "text_values": draft["values"]},
    )
    journey.wait_idle()
    journey.validation_downloads = []
    for kind, route in (("word", "report"), ("excel", "analysis")):
        downloaded = journey.request(
            "GET", f"/api/tasks/{journey.task_id}/{route}/download",
            label="download_validation_" + kind,
        )
        if downloaded.status_code != 200 or not (0 < len(downloaded.content) <= 32_000_000):
            raise RuntimeJourneyError("validation_report_download_failed")
        journey.validation_downloads.append({"kind": kind, "sha256": digest(downloaded.content)})


def _bounded_file(root, relative, *, maximum=32_000_000):
    path = root / relative
    if path.resolve() != path or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("validation artifact boundary")
    raw = path.read_bytes()
    if not raw or len(raw) > maximum:
        raise ValueError("validation artifact size")
    return path, raw


def pipeline_receipt(workspace, task_id, case, messages, *, confirmation=None, downloads=()):
    """Read the settled isolated app's own evidence; no signed tool proof claim."""
    from marvis.db_schema import connect
    from marvis.repositories.tasks import TaskRepository, AGENT_REPORT_CONCLUSION_KEYS
    from marvis.repositories.validation_contracts import ValidationContractRepository
    from marvis.settings import Settings
    from marvis.validation.input_contracts import input_contract_to_dict
    from marvis.validation.pmml_score_artifacts import (
        build_pmml_scoring_identity, validate_pmml_score_artifact,
    )
    from marvis.validation.results import (
        pmml_scoring_result_from_dict, validation_results_from_dict,
        VALIDATION_RESULTS_SCHEMA_V2,
    )
    from marvis.validation.sample_chunks import iter_sample_chunks
    from .runtime_runner import _validation_report_files

    workspace = Path(workspace).resolve()
    repo = TaskRepository(Settings(workspace).db_path)
    task = repo.get_task(task_id)
    result = {
        "carrier": "native_validation_job_stage_files",
        "evidence_scope": "isolated_runtime_material_hashes_native_db_and_result_files; not_signed_tool_receipt",
        "task_status": task.status.value, "workflow_version": task.validation_workflow_version,
        "material_binding_verified": False, "scoring_verified": False,
        "metrics_verified": False, "report_confirmation_verified": False,
        "execution_complete": False, "business_acceptance": "not_established",
        "notebook_consistency": "not_in_v2_entry", "report_files": [],
    }
    # Only enums, timestamps, IDs and hashes leave the isolated workspace.
    with connect(repo.db_path) as conn:
        result["jobs"] = [dict(r) for r in conn.execute(
            "SELECT id, kind, status, started_at, finished_at FROM jobs WHERE task_id=? ORDER BY created_at,id",
            (task_id,),
        ).fetchall()]
    result["stages"] = [
        {"id": m["id"], "stage": m["stage"], "streaming": m.get("metadata", {}).get("streaming") is True,
         "fallback": m.get("metadata", {}).get("fallback") is True}
        for m in messages if m.get("role") == "assistant" and m.get("stage") in {
            "scan", "reproducibility", "metrics", "word_conclusion_draft", "word_conclusion_confirmed"}
    ]
    check = "task_identity"
    try:
        if (task.task_type != "validation" or task.validation_workflow_version != 2
                or task.run_mode != "agent" or not task.notebook_path):
            raise ValueError("wrong validation entry")
        check = "material_binding"
        record = ValidationContractRepository(repo.db_path).get(task_id)
        if record is None or record.status != "ready":
            raise ValueError("input contract not ready")
        source = Path(task.source_dir).resolve()
        if not source.is_relative_to(workspace / "material_uploads"):
            raise ValueError("source not uploaded")
        for material in case.materials:
            if getattr(task, material.role + "_path") != material.path:
                raise ValueError("selection mismatch")
            _, raw = _bounded_file(source, material.path)
            if digest(raw) != material.sha256 or record.contract.material_hashes.get(material.role) != material.sha256:
                raise ValueError("source binding mismatch")
        result.update(material_binding_verified=True, input_contract_revision=record.revision,
                      input_contract_sha256=digest(input_contract_to_dict(record.contract)))
        check = "pmml_scoring"
        outputs = workspace / "tasks" / task_id / "outputs"
        _, scoring_raw = _bounded_file(outputs, "pmml_scoring_result.json")
        scoring = pmml_scoring_result_from_dict(json.loads(scoring_raw))
        hashes = {m.role: m.sha256 for m in case.materials}
        identity = build_pmml_scoring_identity(
            contract=record.contract, pmml_sha256=hashes["pmml"], sample_sha256=hashes["sample"],
            chunk_size=scoring.chunk_size,
        )
        # CSV inspection deliberately leaves total rows unknown. Count the
        # authenticated sample through the existing projected streaming reader;
        # never turn preview size or missing metadata into the true population.
        schema = record.contract.require_sample_schema()
        source_rows = 0
        for chunk in iter_sample_chunks(source / task.sample_path, columns=(), chunk_size=100_000, schema=schema):
            source_rows += len(chunk.row_ids)
            if source_rows > 1_000_000:
                raise ValueError("runtime validation row verification limit")
        if (scoring.pmml_sha256 != hashes["pmml"] or scoring.sample_sha256 != hashes["sample"]
                or scoring.output_field != record.contract.require_output_field()
                or scoring.score_artifact_path != "pmml_scores.parquet"
                or scoring.input_row_count != source_rows
                or (schema.row_count is not None and schema.row_count != source_rows)):
            raise ValueError("score identity mismatch")
        score_path, _ = _bounded_file(outputs, "pmml_scores.parquet")
        validate_pmml_score_artifact(scoring, score_path, expected_cache_key=identity.cache_key)
        result.update(scoring_verified=True, scored_rows=scoring.input_row_count,
                      pmml_scoring_sha256=digest(scoring_raw), scores_sha256=scoring.score_artifact_sha256)
        check = "metrics"
        _, metrics_raw = _bounded_file(outputs, "validation_results.json")
        metrics = validation_results_from_dict(json.loads(metrics_raw))
        if (metrics.schema_version != VALIDATION_RESULTS_SCHEMA_V2 or metrics.pmml_scoring != scoring
                or metrics.reproducibility is not None or metrics.model_name != task.model_name
                or metrics.algorithm != task.algorithm):
            raise ValueError("metrics scoring mismatch")
        result.update(metrics_verified=True, metrics_sha256=digest(metrics_raw))
        check = "report_confirmation"
        if not confirmation:
            raise ValueError("no observed human confirmation")
        draft = next((m for m in messages if m["id"] == confirmation["draft_message_id"]), None)
        metadata = draft.get("metadata", {}) if draft else {}
        values, revision = repo.get_report_values(task_id)
        draft_values = metadata.get("draft_values", {})
        if (not draft or draft.get("stage") != "word_conclusion_draft" or metadata.get("fallback") is not False
                or metadata.get("report_revision") != confirmation["revision"]
                or metadata.get("draft_edit_revision", 0) != confirmation["draft_edit_revision"]
                or digest(draft_values) != confirmation["values_sha256"]
                or revision != confirmation["revision"] + 1
                or any(values.get(k) != v for k, v in draft_values.items())
                or not AGENT_REPORT_CONCLUSION_KEYS <= draft_values.keys()):
            raise ValueError("draft confirmation mismatch")
        audits = repo.list_audit(kind="report.agent_conclusions.confirm", target_ref=task_id)
        if not any(a.get("outcome") == "succeeded" and a.get("detail", {}).get("expected_revision") == confirmation["revision"] for a in audits):
            raise ValueError("confirmation audit missing")
        result.update(report_confirmation_verified=True, report_revision=revision,
                      confirmed_draft_sha256=confirmation["values_sha256"])
        check = "report_files"
        artifacts = {"artifacts": [
            {"kind": kind, "path": f"tasks/{task_id}/outputs/{name}"}
            for kind, name in (("word", "validation_report.docx"), ("excel", "validation.xlsx"))
        ]}
        files = _validation_report_files(workspace, task_id, artifacts)
        if {f["kind"] for f in files} != {"word", "excel"} or any(
                {"kind": f["kind"], "sha256": f["sha256"]} not in downloads for f in files):
            raise ValueError("download/file mismatch")
        result["report_files"] = files
        jobs = result["jobs"]
        result["execution_complete"] = (
            task.status.value in {"succeeded", "review_required"}
            and {"agent", "report"} <= {j["kind"] for j in jobs}
            and all(j["status"] == "succeeded" and j["started_at"] and j["finished_at"] for j in jobs)
            and {"scan", "reproducibility", "metrics", "word_conclusion_draft", "word_conclusion_confirmed"}
                <= {s["stage"] for s in result["stages"]}
        )
    except (ValueError, KeyError, TypeError, OSError):
        result["failed_check"] = check
    return result
