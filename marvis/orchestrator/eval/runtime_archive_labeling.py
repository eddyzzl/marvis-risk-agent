"""Read-only independent label checks over externally bound retained originals.

No producer label algorithm, archived code, model, plugin, restored signing key,
or original workspace is executed/opened. The archive is copied and verified
before parsing. Native records establish consistency; external trust still owns
actor authentication, hidden-data provenance and real outcome truth.
"""
from __future__ import annotations

from contextlib import ExitStack, closing
import csv
import io
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_archive_reader import (
    _Unsupported, _file, _json, _authenticated_snapshot, _public_source_identity, _original_path,
)
from .runtime_contracts import RuntimeCase, digest
from .runtime_custody import verify_case_archive
from .runtime_labeling_reference import reference_labels
from .runtime_archive_datasets import _dataset, _source_input

_HASH = r"^[0-9a-f]{64}$"
_ID = r"^[a-zA-Z0-9_-]{1,160}$"
_TOOLS = {"labels": "labeling.define_label", "maturity": "labeling.check_cohort_maturity"}
_CHECKS = (
    "archive_integrity", "recomputation_snapshot", "external_bindings", "native_steps",
    "label_contract", "source_material", "registered_datasets", "label_recomputation",
    "quality_and_maturity", "artifact_records", "csv_values", "download_binding", "workspace_binding",
)


class FrozenLabelingBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case: RuntimeCase
    run_id: str = Field(min_length=1, max_length=160)
    task_id: str = Field(pattern=_ID)
    plan_id: str = Field(pattern=_ID)
    step_ids: dict[str, str]
    output_sha256: dict[str, str]
    download_sha256: dict[str, str]
    cases_sha256: str = Field(pattern=_HASH)
    expected_sha256: str = Field(pattern=_HASH)
    source: dict
    model_connection_sha256: str = Field(pattern=_HASH)

    @model_validator(mode="after")
    def required_bindings(self):
        if (self.case.task.task_type != "data_join" or not self.case.actions
                or self.case.actions[0].kind != "submit_labeling_request"):
            raise ValueError("a standard labeling proposal entry is required")
        if (set(self.step_ids) != set(_TOOLS) or len(set(self.step_ids.values())) != 2
                or any(not re.fullmatch(_ID, value) for value in self.step_ids.values())
                or set(self.output_sha256) != set(_TOOLS)
                or set(self.download_sha256) != {"dataset", "evidence"}
                or any(not re.fullmatch(_HASH, value) for value in [*self.output_sha256.values(), *self.download_sha256.values()])):
            raise ValueError("complete distinct native step and download bindings are required")
        if not isinstance(self.source.get("source_sha256"), str) or not re.fullmatch(_HASH, self.source["source_sha256"]):
            raise ValueError("external source identity is required")
        return self


def _equal(actual, expected):
    # Canonical JSON keeps booleans distinct from numeric counts. Arithmetic
    # below follows the declared contract, so rounded candidate metrics cannot
    # replace the exact numerator/denominator values either.
    if digest(actual) != digest(expected):
        raise ValueError("label evidence mismatch")


def _native_outputs(conn, binding):
    from marvis.orchestrator.evidence import payload_hash

    task = conn.execute("SELECT * FROM tasks WHERE id=?", (binding.task_id,)).fetchone()
    plan = conn.execute("SELECT * FROM plans WHERE id=?", (binding.plan_id,)).fetchone()
    if (task is None or plan is None or task["task_type"] != "data_join" or task["run_mode"] != "agent"
            or plan["task_id"] != binding.task_id or plan["status"] != "done" or plan["template_id"] != "label_construction"):
        raise ValueError("label task or plan binding mismatch")
    outputs, inputs, native, refs = {}, {}, {}, {}
    for key, tool in _TOOLS.items():
        step_id = binding.step_ids[key]
        step = conn.execute("SELECT * FROM plan_steps WHERE id=?", (step_id,)).fetchone()
        if (step is None or step["plan_id"] != binding.plan_id or step["status"] != "done"
                or f"{step['tool_plugin']}.{step['tool_name']}" != tool):
            raise ValueError("label step binding mismatch")
        match = re.fullmatch(r"metrics:" + re.escape(step_id) + r":v([1-9][0-9]*)", step["output_ref"] or "")
        if match is None:
            raise ValueError("label output is not version bound")
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? AND version=?",
                             (step_id, int(match[1]))).fetchone()
        if saved is None:
            raise ValueError("label output missing")
        output, evidence = json.loads(saved["output_json"]), json.loads(saved["evidence_json"])
        run = conn.execute("SELECT * FROM plan_step_runs WHERE id=?", (evidence.get("step_run_id"),)).fetchone()
        if run is None:
            raise ValueError("label producing run missing")
        run_input = json.loads(run["input_json"])
        _equal(digest(output), binding.output_sha256[key])
        if (run["plan_id"] != binding.plan_id or run["step_id"] != step_id or run["tool_ref"] != tool
                or run["status"] != "succeeded" or run["output_ref"] != step["output_ref"]
                or run["output_hash"] != payload_hash(output) or evidence.get("output_hash") != payload_hash(output)
                or evidence.get("input_hash") != payload_hash(run_input)
                or evidence.get("task_id") != binding.task_id or evidence.get("plan_id") != binding.plan_id
                or evidence.get("step_id") != step_id or evidence.get("tool_name") != tool
                or evidence.get("producer_invocation_id") != run["id"] or run["invocation_id"] != run["id"]
                or evidence.get("raw_output_hash") != run["raw_output_hash"]
                or evidence.get("manifest_hash") != run["manifest_hash"]
                or evidence.get("tool_version") != run["tool_version"]):
            raise ValueError("label native run identity mismatch")
        if evidence.get("schema_version") != "evidence.v1" or evidence.get("output_ref") != step["output_ref"]:
            raise ValueError("label native evidence version mismatch")
        _equal(evidence.get("source_dataset_refs"), [f"dataset:{run_input['dataset_id']}"])
        _equal(evidence.get("result_dataset_bindings"), [] if key == "maturity" else [
            {"dataset_id": output["result_dataset_id"], "content_hash": output["result_content_hash"]}])
        native[key], refs[key] = evidence, step["output_ref"]
        outputs[key], inputs[key] = output, run_input
    _equal(native["maturity"].get("parent_output_refs"), [])
    _equal(native["maturity"].get("parent_output_bindings"), [])
    _equal(native["labels"].get("parent_output_refs"), [refs["maturity"]])
    _equal(native["labels"].get("parent_output_bindings"), [{"step_id": binding.step_ids["maturity"],
        "output_ref": refs["maturity"], "output_hash": payload_hash(outputs["maturity"])}])
    return dict(task), outputs, inputs


def _artifact(conn, archive, manifest, binding, output, kind):
    from marvis.repositories.task_artifacts import stable_task_artifact_id

    row = conn.execute("SELECT * FROM task_artifacts WHERE id=?", (output[f"{kind}_artifact_id"],)).fetchone()
    artifact_kind = "labeling_dataset_csv" if kind == "dataset" else "labeling_quality_evidence_json"
    if (row is None or row["task_id"] != binding.task_id or row["kind"] != artifact_kind
            or row["origin_tool"] != "labeling.define_label" or row["content_hash"] != output[f"{kind}_content_hash"]
            or row["id"] != stable_task_artifact_id(task_id=binding.task_id, kind=artifact_kind, path=row["path"])):
        raise ValueError("label artifact identity mismatch")
    root = Path(manifest["original_workspace"]) / "tasks" / binding.task_id / "labeling"
    if Path(row["path"]).parent != root:
        raise ValueError("label artifact outside native task directory")
    path = _original_path(archive, manifest["original_workspace"], row["path"])
    if digest(path.read_bytes()) != row["content_hash"]:
        raise ValueError("label artifact bytes mismatch")
    provenance = {"schema_version": "labeling-dataset-export.v1" if kind == "dataset" else "labeling-quality-evidence.v1",
                  "producer_version": "labeling.define_label.v1", "proposal_hash": output["proposal_hash"],
                  "source_dataset_id": output["source_dataset_id"], "source_content_hash": output["source_content_hash"],
                  "result_dataset_id": output["result_dataset_id"], "result_content_hash": output["result_content_hash"]}
    if kind == "dataset":
        provenance["target_col"] = output["target_col"]
    _equal(json.loads(row["provenance_json"]), provenance)
    return path


def revalidate_labeling_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenLabelingBinding):
    from marvis.packs.labeling.contracts import LabelingRequest
    from marvis.orchestrator.eval.runtime_runner import _source_identity
    from marvis.data.dataset_export import _csv_cell, _safe_string, _SafetyCounts

    if not isinstance(frozen_binding, FrozenLabelingBinding):
        raise ValueError("an explicit FrozenLabelingBinding is required")
    binding = frozen_binding.model_copy(deep=True)
    archive = original_archive = Path(archive).absolute()
    temporary = ExitStack()
    verifier_source = _source_identity()
    result = {"schema": "marvis.runtime-labeling-revalidation.v1", "case_id": binding.case.id,
              "algorithm": "independent_observed_repayment_label_reference.v1",
              "verifier_source": verifier_source, "original_source_binding": _public_source_identity(binding.source),
              "original_source_binding_sha256": digest(binding.source),
              "verifier_source_matches_original": verifier_source["source_sha256"] == binding.source["source_sha256"],
              "source_authentication": "not_established", "acceptance_claim": "not_established",
              "recomputation_isolation": "private_authenticated_copy; excludes_adversarial_same_uid_or_root_access",
              "checks": {key: {"status": "unverified"} for key in _CHECKS},
              "unsupported": {"native_signature_authentication": "original_key_not_available",
                              "business_outcome_truth": "requires_independent_source_verification",
                              "actor_authorization_and_duplicate_side_effect_safety": "not_established_by_label_arithmetic",
                              "original_runtime_cost_and_timing": "not_recomputed"}}
    checks, current = result["checks"], "archive_integrity"
    try:
        verify_case_archive(archive, expected_manifest_sha256=expected_manifest_sha256)
        checks[current] = {"status": "verified", "scope": "bytes_match_caller_held_manifest_digest"}
        current = "recomputation_snapshot"
        snapshot = Path(temporary.enter_context(tempfile.TemporaryDirectory(prefix="marvis-label-archive-"))).resolve()
        _authenticated_snapshot(archive, snapshot, expected_manifest_sha256)
        archive = snapshot
        scratch = Path(temporary.enter_context(tempfile.TemporaryDirectory(prefix="marvis-label-input-"))).resolve()
        checks[current] = {"status": "verified", "scope": "all_parser_inputs_in_private_authenticated_copy"}
        current = "external_bindings"
        manifest, bindings = _json(archive, "manifest.json"), _json(archive, "bindings.json")
        for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
            _equal(bindings.get(key), getattr(binding, key))
        _equal(_json(archive, "case.json"), binding.case.model_dump())
        observed = bindings["execution_before_scoring"]
        if observed["execution"]["task_id"] != binding.task_id or manifest["case_id"] != binding.case.id:
            raise ValueError("label run task binding mismatch")
        checks[current] = {"status": "verified", "scope": "external_binding_consistency_not_origin_authentication"}
        database = _file(archive, "workspace/marvis.sqlite")
        conn = temporary.enter_context(closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)))
        conn.row_factory = sqlite3.Row
        current = "native_steps"
        task, outputs, inputs = _native_outputs(conn, binding)
        output = outputs["labels"]
        checks[current] = {"status": "verified", "steps": 2, "scope": "native_task_plan_steps_runs_and_versioned_output_hashes"}
        current = "label_contract"
        evidence_path = _artifact(conn, archive, manifest, binding, output, "evidence")
        evidence = json.loads(evidence_path.read_bytes())
        request = LabelingRequest(**evidence["contract"])
        business = binding.case.actions[0].labeling_request.model_dump(exclude_none=True)
        declared = LabelingRequest(**business, **{key: getattr(request, key) for key in
                                  ("dataset_id", "expected_content_hash", "workspace_revision", "analysis_generation")})
        _equal(request.to_dict(), declared.to_dict())
        _equal(request.to_dict(), {key: inputs["labels"].get(key) for key in request.to_dict()})
        if (evidence["schema_version"] != "labeling-quality-evidence.v1" or evidence["producer_version"] != "labeling.define_label.v1"
                or request.contract_hash != output["proposal_hash"] or evidence["proposal_hash"] != request.contract_hash
                or request.dataset_id != output["source_dataset_id"] or request.expected_content_hash != output["source_content_hash"]
                or output["target_col"] != request.target_col):
            raise ValueError("label contract differs from frozen business rule")
        maturity_fields = {key: getattr(request, key) for key in ("dataset_id", "expected_content_hash", "workspace_revision",
                           "analysis_generation", "id_col", "mob_col", "cohort_col", "date_col", "as_of_date")}
        maturity_fields["required_mob"] = request.at_mob
        _equal({key: inputs["maturity"].get(key) for key in maturity_fields}, maturity_fields)
        checks[current] = {"status": "verified", "scope": "frozen_business_rule_and_both_native_tool_inputs"}
        current = "registered_datasets"
        source, source_frame = _dataset(conn, archive, binding.task_id, request.dataset_id, request.expected_content_hash)
        derived, derived_frame = _dataset(conn, archive, binding.task_id, output["result_dataset_id"], output["result_content_hash"])
        if source["role"] != "sample" or derived["role"] != "derived" or derived["target_col"] != request.target_col:
            raise ValueError("label dataset roles or target differ")
        checks[current] = {"status": "verified", "source_rows": len(source_frame), "result_rows": len(derived_frame)}
        current = "source_material"
        _source_input(archive, manifest, binding, source, source_frame, scratch)
        checks[current] = {"status": "verified", "scope": "frozen_input_upload_identity_and_registered_source_values"}
        current = "label_recomputation"
        reference = reference_labels(source_frame, request)
        labels = reference["frame"]
        pd.testing.assert_frame_equal(derived_frame, labels, check_dtype=False, check_exact=True)
        _equal(evidence["source"], {"dataset_id": request.dataset_id, "content_hash": request.expected_content_hash,
                "row_count": len(source_frame), "rows_at_as_of": reference["rows_at_as_of"],
                "rows_excluded_after_as_of": reference["rows_excluded_after_as_of"], "date_col": request.date_col,
                "as_of_date": request.as_of_date})
        _equal(evidence["result"], {"dataset_id": derived["id"], "content_hash": derived["content_hash"],
                                   "target_col": request.target_col, "row_count": len(labels)})
        checks[current] = {"status": "verified", "rows_recomputed": len(labels),
                           "rows_excluded_after_as_of": reference["rows_excluded_after_as_of"]}
        current = "quality_and_maturity"
        for key in ("quality", "maturity", "bad_definition"):
            _equal(output[key], reference[key])
            _equal(evidence[key], reference[key])
        for key in ("n_loans", "n_bad", "n_good", "n_unmatured", "bad_rate"):
            _equal(output[key], reference["quality"][key])
        _equal({key: outputs["maturity"][key] for key in reference["maturity"]}, reference["maturity"])
        checks[current] = {"status": "verified", "scope": "independent_labels_counts_rate_cohort_maturity_and_definition"}
        current = "artifact_records"
        csv_path = _artifact(conn, archive, manifest, binding, output, "dataset")
        _equal(evidence["lineage"], {"parent_dataset_id": source["id"], "child_dataset_id": derived["id"],
                                    "relation_kind": "label_construction", "edge_order": 0})
        _equal(output["lineage"], evidence["lineage"])
        checks[current] = {"status": "verified", "scope": "native_owner_kind_provenance_and_actual_file_bytes"}
        current = "csv_values"
        safety = _SafetyCounts()
        expected_csv = [[_safe_string(c, safety=safety) for c in labels.columns]] + [
            [_csv_cell(row[c], force_text=c in {request.id_col, request.cohort_col}, safety=safety) for c in labels.columns]
            for row in json.loads(labels.to_json(orient="records"))]
        _equal(list(csv.reader(io.StringIO(csv_path.read_bytes().decode("utf-8-sig")))), expected_csv)
        checks[current] = {"status": "verified", "rows_compared": len(labels), "scope": "reference_labels_with_native_Excel_safe_text_encoding"}
        current = "download_binding"
        for kind, path in (("dataset", csv_path), ("evidence", evidence_path)):
            _equal(digest(path.read_bytes()), binding.download_sha256[kind])
            if not any(event.get("stage") == f"download_labeling_{kind}" and event.get("status_code") == 200
                       and event.get("sha256") == binding.download_sha256[kind] for event in observed["http_events"]):
                raise ValueError("actual label download missing")
        checks[current] = {"status": "verified", "scope": "separately_bound_download_bytes_and_native_HTTP_records"}
        current = "workspace_binding"
        workspace = conn.execute("SELECT * FROM data_workspaces WHERE task_id=?", (binding.task_id,)).fetchone()
        expected_workspace = {"revision": request.workspace_revision, "analysis_generation": request.analysis_generation,
                              "active_dataset_id": source["id"], "active_dataset_content_hash": source["content_hash"],
                              "active_dataset_changed": False}
        _equal(output["workspace"], expected_workspace)
        _equal(evidence["workspace"], expected_workspace)
        if workspace is None:
            raise ValueError("label workspace absent")
        _equal({"revision": workspace["revision"], **{key: workspace[key] for key in
               ("analysis_generation", "active_dataset_id", "active_dataset_content_hash")}, "active_dataset_changed": False}, expected_workspace)
        checks[current] = {"status": "verified", "scope": "original_source_still_active_not_silently_replaced_by_labels"}
        ordered = json.loads(labels.sort_values(request.id_col).to_json(orient="values"))
        result["values"] = {"labels_sha256": digest(ordered), "result_rows": len(labels),
                            "quality": reference["quality"], "all_matured": reference["maturity"]["all_matured"],
                            "active_dataset_changed": False}
    except _Unsupported as exc:
        checks[current] = {"status": "unsupported", "reason": str(exc)}
    except Exception as exc:
        checks[current] = {"status": "mismatch", "error_type": type(exc).__name__}
    finally:
        if checks["archive_integrity"]["status"] == "verified":
            try:
                verify_case_archive(original_archive, expected_manifest_sha256=expected_manifest_sha256)
                verify_case_archive(archive, expected_manifest_sha256=expected_manifest_sha256)
                result["archive_unchanged_during_revalidation"] = True
            except Exception:
                checks["archive_integrity"] = {"status": "mismatch", "reason": "archive_changed_during_revalidation"}
                result["archive_unchanged_during_revalidation"] = False
        temporary.close()
    result["supported_checks_verified"] = all(item["status"] == "verified" for item in checks.values())
    return result
