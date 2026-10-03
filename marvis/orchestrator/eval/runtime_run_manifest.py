"""Bind final public runtime records without certifying their business truth.

A final manifest is emitted only by a newly finishing run. Its digest must be
held outside the run directory. No API repairs or backfills historical runs;
missing final evidence is explicitly unavailable. Parsed records returned by the
verifier come from the same bytes whose hashes it checked, not later path reads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

from .runtime_contracts import digest
from .runtime_custody import CustodyError, _private_dir


SCHEMA = "marvis.runtime-final-evidence.v1"
FINAL_MANIFEST_NAME = "final-manifest.json"
MAX_CASES = 1_000
MAX_FILES = 3 * MAX_CASES + 2
MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_MANIFEST_BYTES = 4 * 1024**2
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"[a-zA-Z0-9_-]{1,80}\Z")
_RUNTIME_STATUSES = {"completed", "interrupted", "budget_exceeded", "error", "receipt_error", "not_run_after_interrupt"}
_FROZEN_FIELDS = (
    "cases_sha256", "expected_sha256", "source", "model_connection_sha256",
    "model_source", "model_id", "model_name",
)


class RunManifestError(ValueError):
    """Bounded diagnostic; callers must not treat invalid evidence as success."""


class RunManifestUnavailable(RunManifestError):
    """Historical runs without a final manifest cannot gain one retroactively."""


def _transport_runtime_failure(attempts):
    """Shared runner precedence; flags describe failure, never successful truth."""
    if any(
        item["event"] == "budget_blocked"
        or item.get("output_limit_exceeded") is True
        or item.get("aggregate_budget_violation") is True
        or (item["event"] == "finished" and item.get("error_type") == "RuntimeBudgetExceeded")
        or (item["event"] == "measurement_closed" and item.get("reason") == "wall_deadline")
        for item in attempts
    ):
        return {"runtime_status": "budget_exceeded", "error_code": "llm_attempt_or_wall_budget"}
    if any(item["event"] == "transport_rejected" for item in attempts):
        return {"runtime_status": "error", "error_code": "model_gateway_rejected_request"}
    return None


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RunManifestError("duplicate_json_key")
        result[key] = value
    return result


def _json(raw):
    def invalid_constant(_):
        raise RunManifestError("nonfinite_json_value")

    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RunManifestError("invalid_json_record") from exc
    if not isinstance(value, dict):
        raise RunManifestError("record_must_be_object")
    return value


def _equal(actual, expected):
    if digest(actual) != digest(expected):
        raise RunManifestError("inconsistent_final_record")


def _read(path, *, maximum=MAX_FILE_BYTES):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600 or path.resolve() != path
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())):
        raise RunManifestError("unsafe_run_file")
    if info.st_size > maximum:
        raise RunManifestError("run_evidence_size_limit")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or len(raw) != info.st_size:
        raise RunManifestError("run_file_changed_or_oversized")
    return raw


def _case_ids(value):
    if (not isinstance(value, list) or not 1 <= len(value) <= MAX_CASES
            or any(not isinstance(v, str) or not _CASE_ID.fullmatch(v) for v in value)
            or len(set(value)) != len(value)):
        raise RunManifestError("invalid_complete_case_denominator")
    return value


def _allowed_files(case_ids):
    required = {"manifest.json", "report.json"}
    optional = set()
    for case_id in case_ids:
        required.update({f"{case_id}/execution.json", f"{case_id}/score.json"})
        optional.add(f"{case_id}/llm-attempts.jsonl")
    return required, optional


def _directory_files(root, case_ids, *, finalized):
    required, optional = _allowed_files(case_ids)
    expected_directories = set(case_ids)
    files, directories = set(), set()
    total = 0
    for ordinal, path in enumerate(root.rglob("*"), 1):
        if ordinal > MAX_FILES + MAX_CASES + 1:
            raise RunManifestError("run_evidence_count_limit")
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            _private_dir(path)
            directories.add(relative)
            if relative not in expected_directories:
                raise RunManifestError("unexpected_run_directory")
            continue
        # Check aliases, modes and size before reading any file, including the
        # final manifest. Only this finite public carrier set is permitted.
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.resolve() != path
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise RunManifestError("unsafe_run_file")
        if relative == FINAL_MANIFEST_NAME and finalized:
            continue
        if relative not in required | optional:
            raise RunManifestError("unexpected_run_file")
        if info.st_size > MAX_FILE_BYTES:
            raise RunManifestError("run_evidence_size_limit")
        total += info.st_size
        if total > MAX_TOTAL_BYTES:
            raise RunManifestError("run_evidence_size_limit")
        files.add(relative)
    if not required <= files or directories != expected_directories:
        raise RunManifestError("missing_final_case_evidence")
    return files


def _inventory_files(root, case_ids, inventory, *, finalized):
    if not isinstance(inventory, dict) or len(inventory) > MAX_FILES:
        raise RunManifestError("invalid_run_inventory")
    required, optional = _allowed_files(case_ids)
    declared_total = 0
    for relative, identity in inventory.items():
        if (relative not in required | optional or not isinstance(identity, dict)
                or set(identity) != {"sha256", "size_bytes"} or not _hash(identity["sha256"])
                or type(identity["size_bytes"]) is not int or not 0 <= identity["size_bytes"] <= MAX_FILE_BYTES):
            raise RunManifestError("invalid_run_inventory")
        declared_total += identity["size_bytes"]
        if declared_total > MAX_TOTAL_BYTES:
            raise RunManifestError("run_evidence_size_limit")
    if not required <= inventory.keys() or set(inventory) != _directory_files(root, case_ids, finalized=finalized):
        raise RunManifestError("run_inventory_does_not_match_files")
    originals = {}
    for relative, identity in inventory.items():
        raw = _read(root / relative)
        if len(raw) != identity["size_bytes"] or digest(raw) != identity["sha256"]:
            raise RunManifestError("run_original_bytes_changed")
        originals[relative] = raw
    return originals


def _attempts(raw):
    if raw is None:
        return []
    # Match the runner's preserved malformed-line marker. An incomplete
    # transport receipt remains visible and is not made into a successful call.
    result = []
    for line in raw.decode("utf-8").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            result.append({"event": "incomplete_receipt"})
    return result


def _case_entry(case_id, execution, inventory):
    custody = execution.get("evidence_custody")
    status, custody_digest = "not_recorded", None
    if custody is not None:
        if not isinstance(custody, dict) or custody.get("status") not in {"retained", "failed"}:
            raise RunManifestError("invalid_custody_binding")
        status = custody["status"]
        custody_digest = custody.get("manifest_sha256")
        if ((status == "retained" and not _hash(custody_digest))
                or (status != "retained" and custody_digest is not None)):
            raise RunManifestError("invalid_custody_binding")
    attempts = inventory.get(f"{case_id}/llm-attempts.jsonl")
    return {"case_id": case_id, "runtime_status": execution["runtime_status"],
            "execution_sha256": inventory[f"{case_id}/execution.json"]["sha256"],
            "score_sha256": inventory[f"{case_id}/score.json"]["sha256"],
            "attempts_sha256": attempts["sha256"] if attempts else None,
            "custody_status": status, "custody_manifest_sha256": custody_digest}


def _validate_usage(score, events, frozen):
    from .runtime_scoring import usage_summary

    usage = score.get("usage")
    if usage is None and isinstance(score.get("scorer_error"), str) and score["passed"] is False:
        return
    if not isinstance(usage, dict):
        raise RunManifestError("missing_usage_or_scorer_failure")
    _equal(usage.get("price_book_sha256"), frozen["price_book_sha256"])
    recomputed = usage_summary(events, model_name="", price_bytes=None)
    # Prices are bound by digest, not reconstructed from a copied claim. Call,
    # token, retry and aggregate-budget summaries reuse the original algorithm.
    for key, value in recomputed.items():
        if key not in {"cost", "cost_status"}:
            _equal(usage.get(key), value)


def _validate_records(manifest, originals):
    initial = _json(originals["manifest.json"])
    report = _json(originals["report.json"])
    case_ids = _case_ids(manifest["case_ids"])
    if type(manifest["denominator"]) is not int or manifest["denominator"] != len(case_ids):
        raise RunManifestError("invalid_complete_case_denominator")
    if (not isinstance(manifest["run_id"], str) or not 1 <= len(manifest["run_id"]) <= 160
            or manifest["run_id"] != initial.get("run_id")):
        raise RunManifestError("run_identity_mismatch")
    _equal(initial.get("case_ids"), case_ids)
    _equal(initial.get("denominator"), len(case_ids))
    for key, value in initial.items():
        _equal(report.get(key), value)
    frozen = manifest["frozen"]
    if not isinstance(frozen, dict) or set(frozen) != set(_FROZEN_FIELDS) | {"price_book_sha256", "baseline_sha256"}:
        raise RunManifestError("invalid_frozen_bindings")
    for field in _FROZEN_FIELDS:
        _equal(frozen[field], initial.get(field))
    for field in ("cases_sha256", "expected_sha256", "model_connection_sha256"):
        if not _hash(frozen[field]):
            raise RunManifestError("invalid_frozen_bindings")
    if not isinstance(frozen["source"], dict) or not _hash(frozen["source"].get("source_sha256")):
        raise RunManifestError("invalid_frozen_bindings")
    for field in ("price_book_sha256", "baseline_sha256"):
        if frozen[field] is not None and not _hash(frozen[field]):
            raise RunManifestError("invalid_frozen_bindings")
    _equal(report.get("baseline_sha256"), frozen["baseline_sha256"])
    cases = report.get("cases")
    if not isinstance(cases, list) or [c.get("case_id") for c in cases] != case_ids:
        raise RunManifestError("invalid_complete_case_denominator")
    executions, scores, attempts, entries = {}, {}, {}, []
    for case_id, case in zip(case_ids, cases, strict=True):
        execution = _json(originals[f"{case_id}/execution.json"])
        score = _json(originals[f"{case_id}/score.json"])
        if execution.get("case_id") != case_id or execution.get("runtime_status") not in _RUNTIME_STATUSES:
            raise RunManifestError("case_identity_mismatch")
        if type(score.get("passed")) is not bool:
            raise RunManifestError("invalid_score_record")
        if "a_evidence_eligible" in score and type(score["a_evidence_eligible"]) is not bool:
            raise RunManifestError("invalid_score_record")
        _equal({k: v for k, v in case.items() if k not in {"score", "execution_sha256"}}, execution)
        _equal(case.get("score"), score)
        _equal(case.get("execution_sha256"), manifest["files"][f"{case_id}/execution.json"]["sha256"])
        observed = _attempts(originals.get(f"{case_id}/llm-attempts.jsonl"))
        _equal(execution.get("llm_events"), observed)
        _validate_usage(score, observed, frozen)
        failure = _transport_runtime_failure(observed)
        if failure is not None and any(execution.get(key) != value for key, value in failure.items()):
            raise RunManifestError("transport_failure_contradicts_final_status")
        if score["passed"] and execution["runtime_status"] != "completed":
            raise RunManifestError("passing_score_contradicts_runtime_status")
        if score["passed"] and "scorer_error" in score:
            raise RunManifestError("passing_score_contradicts_scorer_failure")
        if score["passed"]:
            assertions = score.get("assertions")
            job = execution.get("execution", {}).get("latest_job") or {}
            if (score.get("terminal_ok") is not True or not isinstance(assertions, list) or not assertions
                    or not all(isinstance(item, dict) and item.get("passed") is True for item in assertions)
                    or job.get("status") in {"queued", "running"}):
                raise RunManifestError("passing_score_contradicts_reported_checks")
        if score.get("a_evidence_eligible") is True:
            usage = score.get("usage") or {}
            if (not score["passed"] or frozen["model_source"] != "real_model"
                    or type(usage.get("transport_attempts")) is not int or usage["transport_attempts"] <= 0
                    or usage.get("trace_complete") is not True):
                raise RunManifestError("eligibility_flag_contradicts_final_records")
        executions[case_id], scores[case_id], attempts[case_id] = execution, score, observed
        entries.append(_case_entry(case_id, execution, manifest["files"]))
    _equal(manifest["cases"], entries)
    # This verifies arithmetic consistency of reported flags, not their truth.
    passed = sum(score["passed"] for score in scores.values())
    summary = report.get("summary", {})
    for key, value in (("denominator", len(case_ids)), ("passed", passed), ("pass_rate", passed / len(case_ids))):
        _equal(summary.get(key), value)
    _equal(report.get("all_passed"), passed == len(case_ids))
    _equal(report.get("a_evidence_case_count"), sum(score.get("a_evidence_eligible", False) for score in scores.values()))
    _equal(report.get("regression_gate_passed"), report["all_passed"] and report.get("regression_ok", True))
    for field in ("family", "scenario", "case_set"):
        groups = report.get("groups", {}).get(field, {})
        if set(groups) != {case[field] for case in cases}:
            raise RunManifestError("inconsistent_final_groups")
        for key, group in groups.items():
            members = [case for case in cases if case[field] == key]
            successes = sum(case["score"]["passed"] for case in members)
            for name, value in (("denominator", len(members)), ("passed", successes), ("pass_rate", successes / len(members))):
                _equal(group.get(name), value)
    return {"report": report, "executions": executions, "scores": scores, "attempts": attempts}


def _verify(root, expected_manifest_sha256):
    if not _hash(expected_manifest_sha256):
        raise RunManifestError("independently_held_final_digest_required")
    _private_dir(root)
    path = root / FINAL_MANIFEST_NAME
    if not path.exists() and not path.is_symlink():
        raise RunManifestUnavailable("final_runtime_evidence_unavailable_for_legacy_run")
    raw = _read(path, maximum=MAX_MANIFEST_BYTES)
    if digest(raw) != expected_manifest_sha256:
        raise RunManifestError("final_manifest_digest_mismatch")
    manifest = _json(raw)
    if (manifest.get("schema") != SCHEMA or set(manifest) != {
            "schema", "run_id", "case_ids", "denominator", "frozen", "files", "cases"}):
        raise RunManifestError("unsupported_final_manifest_schema")
    ids = _case_ids(manifest["case_ids"])
    originals = _inventory_files(root, ids, manifest["files"], finalized=True)
    parsed = _validate_records(manifest, originals)
    # The caller consumes authenticated parsed values even if paths are changed
    # after this call. Same-UID/root attackers remain outside filesystem privacy.
    return {"manifest": manifest, **parsed, "integrity": "files_match_caller_held_manifest",
            "record_consistency": "verified_not_truth", "source_authentication": "not_established",
            "usage_verification_scope": "record_arithmetic_calls_tokens_retries_and_budget_summaries",
            "usage_cost_recomputed": False,
            "custody_archives_reverified": False, "acceptance_claim": "not_established"}


def verify_run_manifest(run_dir: Path, *, expected_manifest_sha256: str) -> dict:
    """Authenticate final carrier bytes against an independently held digest.

    This does not authenticate the operator/model/source, validate private
    custody bytes, or establish that archived passed/eligible flags are true.
    """
    try:
        return _verify(Path(run_dir).absolute(), expected_manifest_sha256)
    except RunManifestError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, CustodyError) as exc:
        raise RunManifestError("invalid_final_runtime_evidence") from exc


def _finalize_run_manifest(
    run_dir: Path, *, initial_manifest_sha256: str,
    price_book_sha256: str | None, baseline_sha256: str | None,
) -> dict:
    """Runner-only finalization of a current run; never repair old artifacts."""
    root = _private_dir(Path(run_dir).absolute())
    path = root / FINAL_MANIFEST_NAME
    if path.exists() or path.is_symlink():
        raise RunManifestError("final_manifest_already_exists")
    initial_bytes = _read(root / "manifest.json")
    if not _hash(initial_manifest_sha256) or digest(initial_bytes) != initial_manifest_sha256:
        raise RunManifestError("initial_manifest_changed_since_run_started")
    initial = _json(initial_bytes)
    ids = _case_ids(initial.get("case_ids"))
    originals, total = {}, 0
    for relative in sorted(_directory_files(root, ids, finalized=False)):
        raw = _read(root / relative)
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            raise RunManifestError("run_evidence_size_limit")
        originals[relative] = raw
    inventory = {relative: {"sha256": digest(raw), "size_bytes": len(raw)} for relative, raw in originals.items()}
    if inventory["manifest.json"]["sha256"] != initial_manifest_sha256:
        raise RunManifestError("initial_manifest_changed_since_run_started")
    manifest = {"schema": SCHEMA, "run_id": initial["run_id"], "case_ids": ids, "denominator": len(ids),
                "frozen": {**{key: initial[key] for key in _FROZEN_FIELDS},
                           "price_book_sha256": price_book_sha256, "baseline_sha256": baseline_sha256},
                "files": inventory,
                "cases": [_case_entry(case_id, _json(originals[f"{case_id}/execution.json"]), inventory) for case_id in ids]}
    _validate_records(manifest, originals)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_MANIFEST_BYTES:
        raise RunManifestError("run_evidence_size_limit")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    final_digest = digest(raw)
    verify_run_manifest(root, expected_manifest_sha256=final_digest)
    return {"final_manifest_path": str(path), "final_manifest_sha256": final_digest}
