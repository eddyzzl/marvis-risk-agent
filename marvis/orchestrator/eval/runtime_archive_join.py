"""Independent source, native execution and row-level ordinary-JOIN revalidation."""
from contextlib import ExitStack, closing
import json
from pathlib import Path
import re
import sqlite3
import tempfile

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_archive_datasets import _dataset, _source_input, _parquet
from .runtime_archive_reader import _Unsupported, _file, _json, _authenticated_snapshot, _public_source_identity
from .runtime_contracts import RuntimeCase, digest
from .runtime_custody import verify_case_archive
from .runtime_join_reference import join_reference

_HASH = r"^[0-9a-f]{64}$"
_ID = r"^[a-zA-Z0-9_-]{1,160}$"
_TOOLS = {"propose": "data_ops.propose_join", "confirm": "data_ops.confirm_join", "execute": "data_ops.execute_join"}
_PARENTS = {"propose": (), "confirm": ("propose",), "execute": ("propose", "confirm")}
_CHECKS = ("archive_integrity", "recomputation_snapshot", "external_bindings", "native_steps",
           "join_contract", "registered_datasets", "source_materials", "receipt_artifacts",
           "independent_join_values", "physical_membership", "result_counts")
_KEY_FIELDS = ("anchor_col", "feature_col", "match_method", "transform_side")


class FrozenJoinBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case: RuntimeCase
    run_id: str = Field(min_length=1, max_length=160)
    task_id: str = Field(pattern=_ID)
    plan_id: str = Field(pattern=_ID)
    join_plan_id: str = Field(pattern=_ID)
    source_dataset_ids: list[str] = Field(min_length=2, max_length=32)
    source_sha256: list[str] = Field(min_length=2, max_length=32)
    result_dataset_id: str = Field(pattern=_ID)
    result_sha256: str = Field(pattern=_HASH)
    join_specs: list[dict] = Field(min_length=1, max_length=31)
    step_ids: dict[str, str]
    output_sha256: dict[str, str]
    cases_sha256: str = Field(pattern=_HASH)
    expected_sha256: str = Field(pattern=_HASH)
    source: dict
    model_connection_sha256: str = Field(pattern=_HASH)

    @model_validator(mode="after")
    def required_bindings(self):
        size = len(self.source_dataset_ids)
        if (self.case.task.task_type != "data_join" or len(self.case.materials) != size
                or len(self.source_sha256) != size or len(self.join_specs) != size - 1
                or [m.role for m in self.case.materials] != ["sample"] + ["feature"] * (size - 1)
                or len(set([*self.source_dataset_ids, self.result_dataset_id])) != size + 1
                or any(not re.fullmatch(_ID, v) for v in self.source_dataset_ids)
                or any(not re.fullmatch(_HASH, v) for v in self.source_sha256)
                or not any(a.kind == "approve_step" and a.tool == _TOOLS["execute"] for a in self.case.actions)):
            raise ValueError("complete source order and explicit JOIN confirmation required")
        if (set(self.step_ids) != set(_TOOLS) or len(set(self.step_ids.values())) != 3
                or any(not re.fullmatch(_ID, v) for v in self.step_ids.values())
                or set(self.output_sha256) != set(_TOOLS)
                or any(not re.fullmatch(_HASH, v) for v in self.output_sha256.values())
                or not re.fullmatch(_HASH, self.source.get("source_sha256", ""))):
            raise ValueError("complete external native execution bindings required")
        for spec in self.join_specs:
            if (set(spec) != {"keys", "dedup_strategy"} or not isinstance(spec["keys"], list) or not spec["keys"]
                    or spec["dedup_strategy"] not in {None, "abort", "first", "last", "agg_mean", "agg_max"}
                    or any(set(rule) != set(_KEY_FIELDS) or any(not isinstance(v, str) or not v for v in rule.values())
                           or rule["transform_side"] not in {"anchor", "feature", "both"} for rule in spec["keys"])):
                raise ValueError("external key and dedup rules required")
        return self


def _equal(actual, expected):
    if digest(actual) != digest(expected):
        raise ValueError("JOIN identity mismatch")


def _native(conn, binding):
    from marvis.orchestrator.evidence import payload_hash

    task = conn.execute("SELECT * FROM tasks WHERE id=?", (binding.task_id,)).fetchone()
    plan = conn.execute("SELECT * FROM plans WHERE id=?", (binding.plan_id,)).fetchone()
    if (task is None or plan is None or task["task_type"] != "data_join" or task["run_mode"] != "agent"
            or plan["task_id"] != binding.task_id or plan["status"] != "done"):
        raise ValueError("JOIN task or plan mismatch")
    if plan["template_id"] != "data_join":
        raise _Unsupported("JOIN template requires additional independent domain checks")
    _equal(sorted(s[0] for s in conn.execute("SELECT id FROM plan_steps WHERE plan_id=?", (binding.plan_id,))),
           sorted(binding.step_ids.values()))
    outputs, inputs, evidence, refs = {}, {}, {}, {}
    for key, tool in _TOOLS.items():
        step_id = binding.step_ids[key]
        step = conn.execute("SELECT * FROM plan_steps WHERE id=?", (step_id,)).fetchone()
        if (step is None or step["plan_id"] != binding.plan_id or step["status"] != "done"
                or f"{step['tool_plugin']}.{step['tool_name']}" != tool):
            raise ValueError("JOIN step identity mismatch")
        _equal(json.loads(step["depends_on_json"]), [binding.step_ids[parent] for parent in _PARENTS[key]])
        if key == "execute" and (step["confirmed"] != 1 or step["needs_confirmation"] != 1
                              or json.loads(step["policy_json"]).get("human_decision_gate") != "required"):
            raise ValueError("JOIN gate native confirmation missing")
        match = re.fullmatch(r"metrics:" + re.escape(step_id) + r":v([1-9][0-9]*)", step["output_ref"] or "")
        if match is None:
            raise ValueError("JOIN output is not version bound")
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? AND version=?",
                             (step_id, int(match[1]))).fetchone()
        if saved is None:
            raise ValueError("JOIN output missing")
        output, native = json.loads(saved["output_json"]), json.loads(saved["evidence_json"])
        run = conn.execute("SELECT * FROM plan_step_runs WHERE id=?", (native.get("step_run_id"),)).fetchone()
        if run is None:
            raise ValueError("JOIN producing run missing")
        run_input = json.loads(run["input_json"])
        _equal(digest(output), binding.output_sha256[key])
        if (run["plan_id"] != binding.plan_id or run["step_id"] != step_id or run["tool_ref"] != tool
                or run["status"] != "succeeded" or run["output_ref"] != step["output_ref"]
                or run["output_hash"] != payload_hash(output) or native.get("output_hash") != payload_hash(output)
                or native.get("input_hash") != payload_hash(run_input)
                or native.get("task_id") != binding.task_id or native.get("plan_id") != binding.plan_id
                or native.get("step_id") != step_id or native.get("tool_name") != tool
                or native.get("producer_invocation_id") != run["id"] or run["invocation_id"] != run["id"]
                or native.get("raw_output_hash") != run["raw_output_hash"]
                or native.get("manifest_hash") != run["manifest_hash"]
                or native.get("tool_version") != run["tool_version"]
                or native.get("schema_version") != "evidence.v1" or native.get("output_ref") != step["output_ref"]):
            raise ValueError("JOIN native run binding mismatch")
        _equal(native.get("source_dataset_refs"), [])
        _equal(native.get("result_dataset_bindings"), [{"dataset_id": binding.result_dataset_id,
            "content_hash": binding.result_sha256}] if key == "execute" else [])
        outputs[key], inputs[key], evidence[key], refs[key] = output, run_input, native, step["output_ref"]
    for key, parents in _PARENTS.items():
        _equal(evidence[key].get("parent_output_refs"), [refs[p] for p in parents])
        _equal(evidence[key].get("parent_output_bindings"), [
            {"step_id": binding.step_ids[p], "output_ref": refs[p], "output_hash": payload_hash(outputs[p])}
            for p in parents])
    return outputs, inputs


def _contract(conn, binding, outputs, inputs):
    row = conn.execute("SELECT * FROM joins WHERE id=?", (binding.join_plan_id,)).fetchone()
    if (row is None or row["task_id"] != binding.task_id or row["status"] != "executed"
            or row["anchor_dataset_id"] != binding.source_dataset_ids[0]
            or row["result_dataset_id"] != binding.result_dataset_id):
        raise ValueError("native JOIN contract mismatch")
    specs = json.loads(row["joins_json"])
    _equal(len(specs), len(binding.join_specs))
    for index, (native, expected) in enumerate(zip(specs, binding.join_specs, strict=True)):
        _equal(native["feature_dataset_id"], binding.source_dataset_ids[index + 1])
        _equal({"keys": [{k: rule[k] for k in _KEY_FIELDS} for rule in native["key_pairs"]],
                "dedup_strategy": native["dedup_strategy"]}, expected)
        _equal(native["confirmed"], True)
    _equal(inputs["propose"], {"anchor_id": binding.source_dataset_ids[0],
                             "feature_ids": binding.source_dataset_ids[1:], "key_overrides": {}})
    confirm_input = {"join_plan_id": binding.join_plan_id}
    strategies = {binding.source_dataset_ids[i + 1]: spec["dedup_strategy"]
                  for i, spec in enumerate(binding.join_specs) if spec["dedup_strategy"] is not None}
    if strategies:
        confirm_input["dedup_strategies"] = strategies
    _equal(inputs["confirm"], confirm_input)
    _equal(inputs["execute"], {"join_plan_id": binding.join_plan_id})
    _equal(outputs["propose"]["join_plan_id"], binding.join_plan_id)
    _equal(outputs["propose"]["anchor_dataset_id"], binding.source_dataset_ids[0])
    _equal(outputs["confirm"]["join_plan_id"], binding.join_plan_id)
    _equal(outputs["confirm"]["confirmed"], binding.source_dataset_ids[1:])
    _equal(outputs["confirm"]["status"], "confirmed")
    _equal(outputs["confirm"]["needs_dedup"], [])
    _equal(outputs["confirm"]["needs_dtype_ack"], [])
    _equal(outputs["execute"]["result_dataset_id"], binding.result_dataset_id)
    for proposed, saved in zip(outputs["propose"]["joins"], specs, strict=True):
        _equal(proposed["feature_id"], saved["feature_dataset_id"])
        _equal(proposed["key_pairs"], saved["key_pairs"])
        _equal(proposed["diagnostics"], saved["diagnostics"])
    return {"join_plan_id": binding.join_plan_id, "task_id": binding.task_id,
            "anchor_dataset_id": binding.source_dataset_ids[0], "joins": [
                {"feature_dataset_id": spec["feature_dataset_id"], "keys": spec["key_pairs"],
                 "dedup_strategy": spec["dedup_strategy"], "confirmed": True} for spec in specs]}


def _artifact(conn, archive, binding, *, kind, provenance, path=None):
    from marvis.repositories.task_artifacts import stable_task_artifact_id

    records = [dict(row) for row in conn.execute("SELECT * FROM task_artifacts WHERE task_id=? AND kind=?",
               (binding.task_id, kind)) if json.loads(row["provenance_json"]) == provenance]
    if path is not None:
        records = [row for row in records if row["path"] == path]
    if len(records) != 1:
        raise ValueError("JOIN artifact is missing or ambiguous")
    row = records[0]
    relative = Path(row["path"])
    if (row["origin_tool"] != _TOOLS["execute"] or relative.is_absolute()
            or not relative.is_relative_to(Path("datasets") / binding.task_id / ".join")
            or row["id"] != stable_task_artifact_id(task_id=binding.task_id, kind=kind, path=row["path"])):
        raise ValueError("JOIN artifact owner, identity or path mismatch")
    physical = _file(archive, Path("workspace") / relative)
    _equal(digest(physical.read_bytes()), row["content_hash"])
    return row, physical


def _receipt(conn, archive, binding, contract, datasets, frames):
    record, path = _artifact(conn, archive, binding, kind="dataset_join_evidence_v1",
        provenance={"output_dataset_id": binding.result_dataset_id, "output_hash": binding.result_sha256})
    proof = _json(archive, path.relative_to(archive))
    _equal(proof["schema_version"], "ordinary-join-evidence.v1")
    _equal(proof["assurance"], "unknown")
    _equal(proof["reason"], "ordinary_join_does_not_establish_point_in_time_availability")
    _equal(proof["contract"], contract)
    _equal(proof["output_hash"], binding.result_sha256)
    _equal(proof["output_columns"], list(frames[-1].columns))
    _equal(proof["sources"], [{"dataset_id": row["id"], "content_hash": row["content_hash"],
        "row_count": len(frame), "columns": list(frame.columns), "has_target": bool(row["has_target"]),
        "target_col": row["target_col"]} for row, frame in zip(datasets[:-1], frames[:-1], strict=True)])
    _equal(len(proof["steps"]), len(binding.join_specs))
    members = []
    for index, step in enumerate(proof["steps"]):
        _equal(step["feature_dataset_id"], binding.source_dataset_ids[index + 1])
        member, physical = _artifact(conn, archive, binding, kind="dataset_join_membership_v1",
            provenance={"output_dataset_id": binding.result_dataset_id, "evidence_hash": record["content_hash"]},
            path=step["member_path"])
        _equal(member["content_hash"], step["member_hash"])
        members.append(_parquet(physical))
    _equal(proof["steps"][-1]["output_hash"], binding.result_sha256)
    return proof, members


def _membership(frame):
    if list(frame.columns) != ["output_row", "left_row", "right_rows"]:
        raise ValueError("JOIN membership columns mismatch")
    rows = []
    for row in frame.to_dict("records"):
        right = row["right_rows"]
        if hasattr(right, "tolist"):
            right = right.tolist()
        if right is not None and not isinstance(right, list):
            raise ValueError("JOIN membership must be physical row ordinal arrays")
        if any(type(row[key]) is not int for key in ("output_row", "left_row")) or (
                right is not None and any(type(v) is not int for v in right)):
            raise ValueError("JOIN membership ordinals must be integers")
        rows.append({**row, "right_rows": right})
    return rows


def revalidate_join_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenJoinBinding):
    from .runtime_runner import _source_identity

    if not isinstance(frozen_binding, FrozenJoinBinding):
        raise ValueError("an explicit FrozenJoinBinding is required")
    binding = FrozenJoinBinding.model_validate(frozen_binding.model_dump())
    archive = original_archive = Path(archive).absolute()
    temporary = ExitStack()
    verifier = _source_identity()
    result = {"schema": "marvis.runtime-join-revalidation.v1", "case_id": binding.case.id,
              "algorithm": "independent_python_key_join.v1", "verifier_source": verifier,
              "original_source_binding": _public_source_identity(binding.source),
              "original_source_binding_sha256": digest(binding.source),
              "verifier_source_matches_original": verifier["source_sha256"] == binding.source["source_sha256"],
              "source_authentication": "not_established", "acceptance_claim": "not_established",
              "recomputation_isolation": "private_authenticated_copy; excludes_adversarial_same_uid_or_root_access",
              "checks": {key: {"status": "unverified"} for key in _CHECKS},
              "unsupported": {"native_signature_authentication": "original_key_not_available",
                  "point_in_time_availability": "ordinary_join_does_not_establish_availability",
                  "business_outcome_truth": "requires_independent_source_verification",
                  "actor_authorization_and_duplicate_side_effect_safety": "not_established_by_join_arithmetic",
                  "diagnostic_fingerprints_and_conflict_narratives": "not_recomputed",
                  "intermediate_parquet_encoding_hashes": "only_final_registered_output_bytes_verified",
                  "original_runtime_cost_and_timing": "not_recomputed"}}
    checks, current = result["checks"], "archive_integrity"
    try:
        verify_case_archive(archive, expected_manifest_sha256=expected_manifest_sha256)
        checks[current] = {"status": "verified", "scope": "bytes_match_caller_held_manifest_digest"}
        current = "recomputation_snapshot"
        snapshot = Path(temporary.enter_context(tempfile.TemporaryDirectory(prefix="marvis-join-archive-"))).resolve()
        _authenticated_snapshot(archive, snapshot, expected_manifest_sha256)
        archive = snapshot
        scratch = Path(temporary.enter_context(tempfile.TemporaryDirectory(prefix="marvis-join-input-"))).resolve()
        checks[current] = {"status": "verified", "scope": "all_parser_inputs_in_private_authenticated_copy"}
        current = "external_bindings"
        manifest, bindings = _json(archive, "manifest.json"), _json(archive, "bindings.json")
        for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
            _equal(bindings.get(key), getattr(binding, key))
        _equal(_json(archive, "case.json"), binding.case.model_dump())
        observed = bindings["execution_before_scoring"]
        if observed["execution"]["task_id"] != binding.task_id or manifest["case_id"] != binding.case.id:
            raise ValueError("JOIN external task binding mismatch")
        checks[current] = {"status": "verified", "scope": "external_binding_consistency_not_origin_authentication"}
        conn = temporary.enter_context(closing(sqlite3.connect(
            _file(archive, "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)))
        conn.row_factory = sqlite3.Row
        current = "native_steps"
        outputs, inputs = _native(conn, binding)
        checks[current] = {"status": "verified", "steps": 3, "scope": "native_runs_outputs_parents_and_result_dataset_binding"}
        current = "join_contract"
        contract = _contract(conn, binding, outputs, inputs)
        checks[current] = {"status": "verified", "scope": "externally_frozen_keys_dedup_rules_and_native_confirmation"}
        current = "registered_datasets"
        datasets, frames = [], []
        for dataset_id, sha in zip([*binding.source_dataset_ids, binding.result_dataset_id],
                                  [*binding.source_sha256, binding.result_sha256], strict=True):
            row, frame = _dataset(conn, archive, binding.task_id, dataset_id, sha)
            _equal([c["name"] for c in json.loads(row["columns_json"])], list(frame.columns))
            datasets.append(row)
            frames.append(frame)
        _equal([row["role"] for row in datasets], ["sample"] + ["feature"] * len(binding.join_specs) + ["derived"])
        _equal((datasets[-1]["has_target"], datasets[-1]["target_col"]),
               (datasets[0]["has_target"], datasets[0]["target_col"]))
        checks[current] = {"status": "verified", "datasets": len(datasets)}
        current = "source_materials"
        for index, (row, frame) in enumerate(zip(datasets[:-1], frames[:-1], strict=True)):
            _source_input(archive, manifest, binding, row, frame, scratch / str(index), material_index=index)
        checks[current] = {"status": "verified", "scope": "every_frozen_upload_and_registered_source_value"}
        current = "receipt_artifacts"
        proof, memberships = _receipt(conn, archive, binding, contract, datasets, frames)
        checks[current] = {"status": "verified", "scope": "registered_owner_provenance_contract_columns_and_actual_bytes"}
        current = "independent_join_values"
        expected, stages = join_reference(frames[0], frames[1:-1], binding.join_specs)
        _equal(list(frames[-1].columns), list(expected.columns))
        approximate = {alias for feature, spec, stage in zip(frames[1:-1], binding.join_specs, stages, strict=True)
                       if spec["dedup_strategy"] == "agg_mean" for name, alias in stage["mapping"].items()
                       if pd.api.types.is_numeric_dtype(feature[name]) and not pd.api.types.is_bool_dtype(feature[name])}
        for column in expected:
            pd.testing.assert_series_equal(frames[-1][column], expected[column], check_dtype=False,
                check_exact=column not in approximate, rtol=1e-12, atol=1e-12)
        checks[current] = {"status": "verified", "rows": len(expected), "columns": len(expected.columns),
                           "scope": "independent_key_matching_dedup_column_mapping_and_all_result_values"}
        current = "physical_membership"
        for step, frame, reference in zip(proof["steps"], memberships, stages, strict=True):
            _equal(step["feature_columns"], reference["mapping"])
            _equal(_membership(frame), reference["members"])
        checks[current] = {"status": "verified", "scope": "every_physical_output_left_and_right_row_for_each_join"}
        current = "result_counts"
        executed = outputs["execute"]
        _equal(executed["anchor_rows"], len(frames[0]))
        _equal(executed["joined_rows"], len(expected))
        _equal(executed["fan_out"], False)
        if len(frames[0]) > 5000:
            raise _Unsupported("sampled JOIN diagnostic match rates require their independently bound sample")
        for index, (table, reference) in enumerate(zip(executed["per_table"], stages, strict=True)):
            _equal(table["feature_id"], binding.source_dataset_ids[index + 1])
            _equal(table["new_columns"], len(reference["mapping"]))
            _equal(table["match_rate"], round(reference["match_rate"], 4))
        checks[current] = {"status": "verified", "scope": "anchor_output_and_per_feature_match_and_column_counts"}
        result["values"] = {"source_rows": [len(frame) for frame in frames[:-1]], "output_rows": len(expected),
                            "output_columns": len(expected.columns), "join_stages": len(stages)}
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
