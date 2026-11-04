"""Execute frozen hidden slots and retain unsigned candidate originals.

This entry never manufactures independent executor/reviewer signatures, and no
local score is promoted to formal acceptance. All frozen attempts remain visible.
"""
from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from .runtime_contracts import digest
from .runtime_runner import _write_new, run_runtime_suite


def _load_binding(prepared, mode):
    load = prepared.environment["load"]
    return {"load_mode": mode, "warmup": prepared.load_policy.warmup.model_dump() if mode == "steady" else None,
            "load_spec": load, "load_sha256": prepared.benchmark.environment.load_sha256}


def _execute_attempt(prepared, attempt, inputs, runs, custody, secret_env=None):
    """One actual app process per slot, using only the frozen runtime definition."""
    from .process_timing_collector import build_formal_observation
    cases = {c.id: c for c in prepared.suite.cases}
    case = cases[attempt.case_id]
    public = inputs / (attempt.id + ".cases.json")
    expected = inputs / (attempt.id + ".expected.json")
    row = {"attempt_id": attempt.id, "case_id": case.id, "load_mode": attempt.load_mode,
           "status": "unavailable", "error_code": None, "independent_acceptance": "not_established"}
    try:
        _write_new(public, {"schema_version": 1, "cases": [case.model_dump()]})
        _write_new(expected, {"schema_version": 1, "cases": {case.id: prepared.expected.cases[case.id].model_dump()}})
        report = run_runtime_suite(cases_path=public, expected_path=expected, dataset_root=prepared.dataset_root,
            output_dir=runs, model=prepared.model, model_source="real_model", price_book_path=prepared.price_book_path,
            evidence_custody_dir=custody, load_binding=_load_binding(prepared, attempt.load_mode),
            formal_provider_limits=prepared.provider_limits, secret_env=secret_env,
            expected_source_identity=prepared.source_identity)
        record = report["cases"][0]
        row.update(status=record["runtime_status"], error_code=record.get("error_code"),
                   run_id=report["run_id"], run_dir=str(Path(report["report_path"]).parent),
                   final_manifest_sha256=report["final_manifest_sha256"],
                   cases_file=str(public), expected_file=str(expected), price_book=str(prepared.price_book_path),
                   local_predicates_passed=record["score"].get("passed", False),
                   evidence_custody=record.get("evidence_custody"),
                   load_originals_sha256=digest(record.get("load_originals")))
        load = record.get("load_originals") or {}
        events = {e["event"]: e for e in load.get("events", [])}
        if "journey_started" in events and "journey_finished" in events:
            row["journey_interval"] = {"clock": "host_monotonic_ns", "pid": load["child_identity"]["pid"],
                "workspace_sha256": load["child_identity"]["workspace_sha256"],
                "start_ns": load["origin_ns"] + events["journey_started"]["elapsed_ns"],
                "end_ns": load["origin_ns"] + events["journey_finished"]["elapsed_ns"]}
        try:
            observation, coverage = build_formal_observation(record, attempt_id=attempt.id, case_id=case.id,
                run_id=report["run_id"], load_mode=attempt.load_mode, model=prepared.benchmark.model.model_dump(),
                environment=prepared.benchmark.environment.model_dump())
            path = inputs.parent / "timing" / (attempt.id + ".json")
            _write_new(path, observation)
            _write_new(path.with_suffix(".coverage.json"), coverage)
            row["telemetry"] = {"path": str(path), "sha256": digest(path.read_bytes())}
        except (ValueError, KeyError, TypeError):
            row["telemetry_unavailable"] = "fixed_collector_originals_incomplete"
    except KeyboardInterrupt:
        row.update(status="interrupted", error_code="operator_interrupt")
    except Exception as exc:
        # Exception text can contain credentials, prompts or input paths.
        row.update(status="unavailable", error_code=type(exc).__name__)
    return row


def _bind_concurrency(rows):
    concurrent = [row for row in rows if row["load_mode"] == "concurrent"]
    for row in concurrent:
        current = row.get("journey_interval")
        peers = []
        if current and current["end_ns"] > current["start_ns"]:
            for other in concurrent:
                interval = other.get("journey_interval")
                if (other is not row and interval and interval["pid"] != current["pid"]
                        and interval["workspace_sha256"] != current["workspace_sha256"]
                        and max(current["start_ns"], interval["start_ns"]) < min(current["end_ns"], interval["end_ns"])):
                    peers.append(other["attempt_id"])
        row["concurrency"] = {"observed_overlap": bool(peers), "peer_attempt_ids": peers,
                              "scope": "actual_application_journeys"}
        if not peers:
            row["status"] = "load_unavailable"
            row["error_code"] = "concurrent_journey_overlap_not_observed"


def _concurrency_carriers(prepared, batch, rows):
    from .runtime_concurrency import build_concurrency_evidence, verify_concurrency_evidence
    from .runtime_run_manifest import verify_run_manifest
    hashes = {}
    for row in rows:
        if row.get("run_dir"):
            relative = Path(row["run_dir"]).relative_to(batch).as_posix()
            hashes[relative + "/final-manifest.json"] = row["final_manifest_sha256"]
    planned = {a.id: a for a in prepared.benchmark.attempts}
    for row in rows:
        if row["load_mode"] != "concurrent":
            continue
        current = row.get("journey_interval")
        try:
            if not current or not row.get("telemetry"):
                raise ValueError("concurrent_originals_unavailable")
            peers = []
            for other in rows:
                interval = other.get("journey_interval")
                if (other["load_mode"] == "concurrent" and interval and other.get("run_dir")
                        and max(current["start_ns"], interval["start_ns"]) < min(current["end_ns"], interval["end_ns"])):
                    peers.append({key: other[key] for key in ("attempt_id", "case_id", "run_id", "final_manifest_sha256")} |
                                 {"run_dir": Path(other["run_dir"]).relative_to(batch).as_posix()})
            evidence = build_concurrency_evidence(batch, attempt_id=row["attempt_id"], peers=peers)
            run = verify_run_manifest(Path(row["run_dir"]), expected_manifest_sha256=row["final_manifest_sha256"])
            observed = verify_concurrency_evidence(batch, evidence, artifact_hashes=hashes,
                benchmark=prepared.benchmark, attempt=planned[row["attempt_id"]], run=run)
            path = batch / "concurrency" / (row["attempt_id"] + ".json")
            _write_new(path, evidence)
            reference = {"path": path.relative_to(batch).as_posix(), "sha256": digest(path.read_bytes())}
            hashes[reference["path"]] = reference["sha256"]
            row["telemetry"]["concurrency"] = reference
            row["concurrency"] = {"observed_overlap": True, **observed}
        except (ValueError, KeyError, TypeError):
            row["concurrency"] = {"observed_overlap": False, "scope": "actual_application_journeys"}
            if row["status"] not in {"not_run_after_interrupt", "interrupted", "not_run"}:
                row.update(status="load_unavailable", error_code="concurrent_frozen_load_not_verified")
    for row in rows:
        for key in ("run_dir", "cases_file", "expected_file", "price_book"):
            if row.get(key):
                row[key] = Path(row[key]).relative_to(batch).as_posix()
        if row.get("telemetry"):
            reference = row["telemetry"]
            reference["path"] = Path(reference["path"]).relative_to(batch).as_posix()
            hashes[reference["path"]] = reference["sha256"]
    return hashes


def execute_prepared(prepared, *, binding_path, freeze_dir, output_dir, evidence_custody_dir, secret_env=None):
    """Only called after every source, material, approval and total cap is checked."""
    if len(prepared.benchmark.attempts) != 193 or len(prepared.benchmark.cases) != 133:
        raise ValueError("hidden_fixed_denominator_required")
    output_dir, evidence_custody_dir = Path(output_dir).resolve(), Path(evidence_custody_dir).resolve()
    protected = (prepared.dataset_root, Path(binding_path).parent, Path(freeze_dir), Path(__file__).resolve().parents[3])
    if (output_dir == evidence_custody_dir or output_dir.is_relative_to(evidence_custody_dir)
            or evidence_custody_dir.is_relative_to(output_dir)):
        raise ValueError("hidden_output_and_custody_must_be_separate")
    if any(output_dir.is_relative_to(path.resolve()) or evidence_custody_dir.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("hidden_execution_outputs_overlap_inputs")
    # Re-authenticate every input once and only retain/use these validated bytes.
    # In particular, never parse a mutable approval again to discover local paths.
    from .hidden_benchmark import load_frozen_case_set
    from .hidden_formal_contract import FileBinding, _read
    from .runtime_formal_budget import ProviderApproval
    root = Path(binding_path).parent
    originals = {}
    originals["execution-binding.json"] = _read(root, FileBinding(path=Path(binding_path).name, sha256=prepared.binding_sha256))[1]
    for name, reference in prepared.binding.files.model_dump().items():
        originals[name + ".json"] = _read(root, FileBinding.model_validate(reference))[1]
    approval_raw = _read(root, FileBinding.model_validate(prepared.provider_limits.approval.model_dump()))[1]
    approval = ProviderApproval.model_validate_json(approval_raw)
    originals["provider-approval.json"] = approval_raw
    originals["provider-accounting-document"] = _read(root, FileBinding.model_validate(approval.source_document.model_dump()))[1]
    proposal = load_frozen_case_set(Path(freeze_dir))
    manifest_raw = (Path(freeze_dir) / "freeze-manifest.json").read_bytes()
    if digest(manifest_raw) != prepared.binding.proposal_manifest_sha256:
        raise ValueError("hidden_proposal_changed_after_preflight")
    originals["proposal-freeze-manifest.json"] = manifest_raw
    for name, expected in (("cases", proposal["manifest"]["case_set_sha256"]), ("attempts", proposal["manifest"]["attempts_sha256"])):
        raw = (Path(freeze_dir) / (name + ".json")).read_bytes()
        if digest(json.loads(raw)) != expected:
            raise ValueError("hidden_proposal_changed_after_preflight")
        originals["proposal-" + name + ".json"] = raw
    for name in ("ledger", "profile"):
        source = getattr(prepared, name + "_path")
        raw = source.read_bytes()
        if digest(raw) != getattr(prepared, name + "_sha256"):
            raise ValueError("hidden_authority_changed_after_preflight")
        originals[name + ".json"] = raw
    batch = output_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12])
    batch.mkdir(parents=True, mode=0o700)
    for name in ("inputs", "runs", "timing", "frozen", "slots", "concurrency"):
        (batch / name).mkdir(mode=0o700)
    frozen = batch / "frozen"
    inventory = {}
    for name, raw in originals.items():
        with os.fdopen(os.open(frozen / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as stream:
            stream.write(raw)
        inventory[name] = digest(raw)
    # Every attempt reads the retained, hash-checked price instead of a mutable
    # authority path; the runtime reads and verifies that price exactly once.
    prepared = replace(prepared, price_book_path=frozen / "price_book.json")
    rows = {a.id: {"attempt_id": a.id, "case_id": a.case_id, "load_mode": a.load_mode,
                   "status": "not_run", "error_code": None, "independent_acceptance": "not_established"}
            for a in prepared.benchmark.attempts}
    _write_new(batch / "planned-slots.json", list(rows.values()))
    _write_new(batch / "frozen-inputs.json", inventory)
    interrupted = False
    for attempt in prepared.benchmark.attempts:
        if attempt.load_mode == "concurrent" or interrupted:
            continue
        rows[attempt.id] = _execute_attempt(prepared, attempt, batch / "inputs", batch / "runs", evidence_custody_dir, secret_env)
        _write_new(batch / "slots" / (attempt.id + ".json"), rows[attempt.id])
        interrupted = rows[attempt.id]["status"] == "interrupted"
    if not interrupted:
        with ThreadPoolExecutor(max_workers=prepared.load_policy.concurrent_workers) as pool:
            futures = {pool.submit(_execute_attempt, prepared, attempt, batch / "inputs", batch / "runs", evidence_custody_dir, secret_env): attempt
                       for attempt in prepared.benchmark.attempts if attempt.load_mode == "concurrent"}
            try:
                for future in as_completed(futures):
                    attempt = futures[future]
                    try:
                        rows[attempt.id] = future.result()
                    except Exception as exc:
                        rows[attempt.id].update(status="unavailable", error_code=type(exc).__name__)
                    _write_new(batch / "slots" / (attempt.id + ".json"), rows[attempt.id])
            except KeyboardInterrupt:
                interrupted = True
                for future in futures:
                    future.cancel()
            finally:
                for future, attempt in futures.items():
                    if future.done() and not future.cancelled():
                        try:
                            rows[attempt.id] = future.result()
                        except Exception as exc:
                            rows[attempt.id].update(status="unavailable", error_code=type(exc).__name__)
        # The executor context waits for in-flight cases after cancellation.
        # Harvest those originals too; do not label an executed call as unrun.
        for future, attempt in futures.items():
            if future.done() and not future.cancelled():
                try:
                    rows[attempt.id] = future.result()
                except Exception as exc:
                    rows[attempt.id].update(status="unavailable", error_code=type(exc).__name__)
                slot_path = batch / "slots" / (attempt.id + ".json")
                if not slot_path.exists():
                    _write_new(slot_path, rows[attempt.id])
    ordered = list(rows.values())
    if interrupted:
        for row in ordered:
            if row["status"] == "not_run":
                row.update(status="not_run_after_interrupt", error_code="operator_interrupt")
    artifact_hashes = _concurrency_carriers(prepared, batch, ordered)
    _write_new(batch / "candidate-artifact-hashes.json", artifact_hashes)
    report = {"schema": "marvis.hidden-execution-candidate.v1", "benchmark_id": prepared.benchmark.benchmark_id,
        "execution_binding_sha256": prepared.binding_sha256, "frozen_inputs_sha256": digest(inventory),
        "denominator": 193, "case_count": 133, "planned_cost_upper_bound": prepared.total_cost_bound,
        "budget_scope": "193 full per-attempt ceilings; provider accounting approval separately bound",
        "provider_billing_guarantee": False, "acceptance_claim": "none",
        "execution_scope": "local_fixture/protocol_only" if approval.source_scope == "local_test_fixture" else "unsigned_formal_candidate",
        "signature_status": "unsigned_candidate_originals", "attempts": ordered,
        "interrupted": interrupted, "report_path": str(batch / "candidate.json")}
    _write_new(batch / "candidate.json", report)
    return report
