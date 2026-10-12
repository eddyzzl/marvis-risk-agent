"""Private archive identity plumbing for the three independent domain readers.

No original workspace path is reopened. This is evidence consistency, not a
signature authority or a replacement for separate safety adjudication.
"""
from contextlib import ExitStack, closing
import json
import hashlib
import re
import sqlite3
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime_archive_reader import _Unsupported, _authenticated_snapshot, _file, _json, _public_source_identity
from .runtime_archive_reader import _original_path
from .runtime_contracts import RuntimeCase, digest
from .runtime_custody import verify_case_archive

HASH = r"^[0-9a-f]{64}$"
ID = r"^[a-zA-Z0-9_-]{1,160}$"


class FrozenDomainBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    case: RuntimeCase
    run_id: str = Field(min_length=1, max_length=160)
    task_id: str = Field(pattern=ID)
    plan_id: str = Field(pattern=ID)
    step_ids: dict[str, str]
    output_sha256: dict[str, str]
    cases_sha256: str = Field(pattern=HASH)
    expected_sha256: str = Field(pattern=HASH)
    source: dict
    model_connection_sha256: str = Field(pattern=HASH)

    @model_validator(mode="after")
    def native_inventory(self):
        if (not self.step_ids or len(self.step_ids) > 100
                or set(self.step_ids) != set(self.output_sha256)
                or len(set(self.step_ids.values())) != len(self.step_ids)
                or any(not re.fullmatch(ID, v) for v in self.step_ids.values())
                or any(not re.fullmatch(HASH, v) for v in self.output_sha256.values())
                or not re.fullmatch(HASH, self.source.get("source_sha256", ""))):
            raise ValueError("complete external native step bindings required")
        return self


def equal(actual, expected):
    if digest(actual) != digest(expected):
        raise ValueError("domain identity mismatch")


def numbers(actual, expected, *, rel_tol=1e-9, abs_tol=1e-11):
    import math
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError("domain object mismatch")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError("domain metric missing")
            numbers(actual[key], value, rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("domain list mismatch")
        for a, b in zip(actual, expected, strict=True):
            numbers(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise ValueError("independent domain value mismatch")
    elif actual != expected or (expected is None and actual is not None):
        raise ValueError("domain literal mismatch")


def _declared_refs(value, *, mode):
    refs = []
    bindings = []
    def visit(item, key=""):
        if isinstance(item, dict):
            if mode == "artifact" and isinstance(item.get("artifact_id"), str) and re.fullmatch(HASH, str(item.get("content_hash", ""))):
                binding = {"artifact_id": item["artifact_id"].strip(), "content_hash": item["content_hash"]}
                if isinstance(item.get("kind"), str) and item["kind"].strip():
                    binding["kind"] = item["kind"].strip()
                if binding not in bindings:
                    bindings.append(binding)
            for child_key, child in item.items():
                visit(child, child_key)
        elif isinstance(item, list):
            for child in item:
                visit(child, key)
        elif isinstance(item, str) and item.strip():
            text, key = item.strip(), key.lower()
            if mode == "dataset":
                ref = text if text.startswith("dataset:") else f"dataset:{text}" if key.endswith(("dataset_id", "dataset_ids")) else None
            elif mode == "result":
                ref = text if key == "result_dataset_id" else None
            else:
                ref = text if text.startswith("artifact:") else f"artifact:{text}" if key == "path" or key.endswith(("_path", "artifact_id", "artifact_ref")) else None
            if ref is not None and ref not in refs:
                refs.append(ref)
    visit(value)
    return refs, bindings


def native_steps(conn, binding):
    from marvis.orchestrator.evidence import payload_hash
    task = conn.execute("SELECT * FROM tasks WHERE id=?", (binding.task_id,)).fetchone()
    plan = conn.execute("SELECT * FROM plans WHERE id=?", (binding.plan_id,)).fetchone()
    if (task is None or plan is None or task["task_type"] != binding.case.task.task_type
            or task["run_mode"] != "agent" or plan["task_id"] != binding.task_id or plan["status"] != "done"):
        raise ValueError("native domain task/plan mismatch")
    rows = conn.execute("SELECT * FROM plan_steps WHERE plan_id=?", (binding.plan_id,)).fetchall()
    equal(sorted(r["id"] for r in rows), sorted(binding.step_ids.values()))
    result = {}
    for key, step_id in binding.step_ids.items():
        row = next(r for r in rows if r["id"] == step_id)
        match = re.fullmatch(r"metrics:" + re.escape(step_id) + r":v([1-9][0-9]*)", row["output_ref"] or "")
        if row["status"] != "done" or match is None:
            raise _Unsupported("complete successful native plan required for normal domain revalidation")
        saved = conn.execute("SELECT * FROM plan_step_output_versions WHERE step_id=? AND version=?", (step_id, int(match[1]))).fetchone()
        if saved is None:
            raise ValueError("native versioned output missing")
        output, evidence = json.loads(saved["output_json"]), json.loads(saved["evidence_json"])
        run = conn.execute("SELECT * FROM plan_step_runs WHERE id=?", (evidence.get("step_run_id"),)).fetchone()
        if run is None:
            raise ValueError("native producer invocation missing")
        inputs = json.loads(run["input_json"])
        tool = f"{row['tool_plugin']}.{row['tool_name']}"
        # Runtime execution captures the platform's native payload-hash
        # contract (including ±inf bucket endpoints), not strict public JSON.
        try:
            observed_hash = digest(output)
        except ValueError:
            observed_hash = payload_hash(output).removeprefix("sha256:")
        equal(observed_hash, binding.output_sha256[key])
        if (run["plan_id"] != binding.plan_id or run["step_id"] != step_id
                or run["tool_ref"] != tool or run["status"] != "succeeded" or run["output_ref"] != row["output_ref"]
                or run["output_hash"] != payload_hash(output) or evidence.get("output_hash") != payload_hash(output)
                or evidence.get("input_hash") != payload_hash(inputs) or evidence.get("task_id") != binding.task_id
                or evidence.get("plan_id") != binding.plan_id or evidence.get("step_id") != step_id
                or evidence.get("tool_name") != tool or evidence.get("producer_invocation_id") != run["id"]
                or run["invocation_id"] != run["id"] or evidence.get("raw_output_hash") != run["raw_output_hash"]
                or evidence.get("manifest_hash") != run["manifest_hash"] or evidence.get("tool_version") != run["tool_version"]
                or evidence.get("schema_version") != "evidence.v1" or evidence.get("output_ref") != row["output_ref"]):
            raise ValueError("native producer input/output evidence differs")
        result[step_id] = {"tool": tool, "input": inputs, "output": output, "evidence": evidence, "step": dict(row)}
        equal(evidence.get("source_dataset_refs"), _declared_refs(inputs, mode="dataset")[0])
        equal(evidence.get("artifact_refs"), _declared_refs(output, mode="artifact")[0])
        equal(evidence.get("artifact_bindings"), _declared_refs(output, mode="artifact")[1])
        declared_results = []
        for identity in _declared_refs(output, mode="result")[0]:
            dataset = conn.execute("SELECT * FROM datasets WHERE id=? AND task_id=?", (identity, binding.task_id)).fetchone()
            if dataset is None or not re.fullmatch(HASH, dataset["content_hash"]):
                raise ValueError("native result dataset identity unregistered")
            declared_results.append({"dataset_id": identity, "content_hash": dataset["content_hash"]})
        equal(evidence.get("result_dataset_bindings"), declared_results)
    for item in result.values():
        parents = json.loads(item["step"]["depends_on_json"])
        if any(p not in result for p in parents):
            raise ValueError("native dependency outside frozen plan")
        equal(item["evidence"].get("parent_output_refs"), [result[p]["step"]["output_ref"] for p in parents])
        equal(item["evidence"].get("parent_output_bindings"), [
            {"step_id": p, "output_ref": result[p]["step"]["output_ref"], "output_hash": payload_hash(result[p]["output"])} for p in parents])
        if item["step"]["needs_confirmation"] and not item["step"]["confirmed"]:
            # The installed conditional sentinel gate is a deterministic no-op
            # only when its authenticated screen parent found no selected
            # sentinel-bearing field. Any actual decision still needs assent.
            screens = [result[p]["output"] for p in parents if result[p]["tool"] == "modeling.screen_features"]
            no_op = (item["tool"] == "modeling.resolve_special_values" and len(screens) == 1
                and isinstance(screens[0].get("selected"), list) and isinstance(screens[0].get("sentinel_columns"), dict)
                and not any(name in screens[0]["selected"] and rows for name, rows in screens[0]["sentinel_columns"].items())
                and item["output"].get("governance") == {} and item["output"].get("masked") == []
                and item["output"].get("retained") == [] and item["output"].get("dropped") == []
                and item["output"].get("result_dataset_id") == item["input"].get("dataset_id"))
            if not no_op:
                raise ValueError("native required confirmation absent")
    return result


def one_step(steps, tool):
    matching = [s for s in steps.values() if s["tool"] == tool]
    if len(matching) != 1:
        raise _Unsupported("one exact native domain action required")
    return matching[0]


def authenticate_step_sources(session):
    dataset_ids, artifact_bindings = set(), {}
    for step in session.steps.values():
        for reference in step["evidence"]["source_dataset_refs"]:
            dataset_ids.add(reference.removeprefix("dataset:"))
        for binding in step["evidence"]["result_dataset_bindings"]:
            dataset_ids.add(binding["dataset_id"])
        for binding in step["evidence"]["artifact_bindings"]:
            if binding["artifact_id"] in artifact_bindings and artifact_bindings[binding["artifact_id"]] != binding:
                raise ValueError("conflicting native artifact bindings")
            artifact_bindings[binding["artifact_id"]] = binding
    if len(dataset_ids) + len(artifact_bindings) > 1000:
        raise _Unsupported("native source binding count limit")
    for identity in dataset_ids:
        row = session.conn.execute("SELECT * FROM datasets WHERE id=? AND task_id=?", (identity, session.binding.task_id)).fetchone()
        if row is None or Path(row["source_path"]).is_absolute():
            raise ValueError("native dataset source is missing or unbound")
        path = _file(session.archive, Path("workspace/datasets") / row["source_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["content_hash"]:
            raise ValueError("native step source bytes changed")
    for identity, expected in artifact_bindings.items():
        row = session.conn.execute("SELECT * FROM task_artifacts WHERE id=? AND task_id=?", (identity, session.binding.task_id)).fetchone()
        if row is None:
            # Installed strategy adoption retains its immutable deliverables
            # in the strategy-owned registry, whose task owner is the policy.
            row = session.conn.execute("SELECT a.* FROM strategy_artifacts a JOIN strategies s ON s.id=a.strategy_id WHERE a.id=? AND s.task_id=?",
                                       (identity, session.binding.task_id)).fetchone()
        if row is None or row["content_hash"] != expected["content_hash"] or ("kind" in expected and row["kind"] != expected["kind"]):
            raise ValueError("native step artifact registration differs")
        stored = Path(row["path"])
        path = _original_path(session.archive, session.manifest["original_workspace"], stored) if stored.is_absolute() else _file(session.archive, Path("workspace") / stored)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["content_hash"]:
            raise ValueError("native step artifact bytes changed")


class DomainArchive:
    def __init__(self, archive, sha, binding, *, family, checks):
        self.original = Path(archive).absolute()
        self.archive = self.original
        self.sha, self.binding = sha, binding
        self.stack = ExitStack()
        self.current = "archive_integrity"
        self.result = {"schema": f"marvis.runtime-{family}-revalidation.v1", "case_id": binding.case.id,
            "algorithm": f"independent_{family}_reference.v1", "source_authentication": "not_established",
            "acceptance_claim": "not_established", "original_source_binding": _public_source_identity(binding.source),
            "original_source_binding_sha256": digest(binding.source),
            "recomputation_isolation": "private_authenticated_copy; excludes_adversarial_same_uid_or_root_access",
            "checks": {k: {"status": "unverified"} for k in ("archive_integrity", "recomputation_snapshot", "external_bindings", "native_steps", *checks)},
            "unsupported": {"native_signature_authentication": "requires_independently_held_authority",
                "point_in_time_maturity_and_business_outcome_truth": "requires_independent_source_verification",
                "actor_authorization_and_duplicate_side_effect_safety": "requires_separate_process_adjudication",
                "original_runtime_cost_and_timing": "not_recomputed"}}

    def mark(self, name, **details):
        self.current = name
        self.result["checks"][name] = {"status": "verified", **details}

    def __enter__(self):
        return self

    def open(self):
        verify_case_archive(self.original, expected_manifest_sha256=self.sha)
        self.mark("archive_integrity", scope="caller_pinned_manifest_bytes")
        self.current = "recomputation_snapshot"
        self.archive = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="marvis-domain-originals-"))).resolve()
        _authenticated_snapshot(self.original, self.archive, self.sha)
        self.mark("recomputation_snapshot", scope="bounded_authenticated_parser_inputs")
        self.current = "external_bindings"
        self.manifest = _json(self.archive, "manifest.json")
        self.bindings = _json(self.archive, "bindings.json")
        for key in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256"):
            equal(self.bindings.get(key), getattr(self.binding, key))
        # The pinned archive authenticates the original encoded bytes. Compare
        # installed typed semantics here so adding optional contract defaults
        # cannot make an unchanged historical case's encoding unverifiable.
        encoded_case = _json(self.archive, "case.json")
        typed_case = RuntimeCase.model_validate(encoded_case).model_dump()
        equal(typed_case, self.binding.case.model_dump())
        if (self.manifest["case_id"] != self.binding.case.id
                or self.bindings["execution_before_scoring"]["execution"]["task_id"] != self.binding.task_id):
            raise ValueError("externally frozen case/task differs")
        self.mark("external_bindings", scope="consistency_not_authority_authentication",
                  original_case_sha256=digest(encoded_case), typed_case_sha256=digest(typed_case),
                  normalization_applied=digest(encoded_case) != digest(typed_case))
        self.conn = self.stack.enter_context(closing(sqlite3.connect(_file(self.archive, "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)))
        self.conn.row_factory = sqlite3.Row
        self.current = "native_steps"
        self.steps = native_steps(self.conn, self.binding)
        authenticate_step_sources(self)
        self.mark("native_steps", count=len(self.steps))
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.result["checks"][self.current] = {"status": "unsupported" if isinstance(exc, _Unsupported) else "mismatch", "error_type": type(exc).__name__}
        if self.result["checks"]["archive_integrity"]["status"] == "verified":
            try:
                verify_case_archive(self.original, expected_manifest_sha256=self.sha)
                verify_case_archive(self.archive, expected_manifest_sha256=self.sha)
                self.result["archive_unchanged_during_revalidation"] = True
            except Exception:
                self.result["checks"]["archive_integrity"] = {"status": "mismatch", "reason": "archive_changed"}
                self.result["archive_unchanged_during_revalidation"] = False
        self.stack.close()
        self.result["supported_checks_verified"] = all(v["status"] == "verified" for v in self.result["checks"].values())
        return exc is not None
