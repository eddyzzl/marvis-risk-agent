"""Held hidden-benchmark case set (133 cases / 193 slots) and its runner gate.

The freeze writes case specifications whose variant parameters derive from a
holder-only random salt, so repository code alone cannot rebuild them. Expected
result types are fixed at freeze time. Budgets, SLOs, the real model and the
hardware/load profile are user decisions: the formal runner refuses while any
is pending (ledger ``agent_benchmark_plan.cost_cap``, performance profile). A
local dry run with a stub LLM exercises the harness only and makes no claim.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from .formal_benchmark import (
    BASE_FAMILIES,
    EXTENSION_FAMILIES,
    FrozenBenchmark,
    _percentile,
    _summary,
)
from .runtime_contracts import digest

SCHEMA = "marvis.hidden-benchmark-case-set.v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = ROOT / "docs/superpowers/plans/2026-09-20-audit-closure-acceptance.json"
DEFAULT_PROFILE = ROOT / "docs/development/performance-gate-profile.json"
DEFAULT_HELD_ROOT = ROOT / ".omc/benchmark/hidden"
CASE_COUNT, SLOT_COUNT = 133, 193
DRY_RUN_LABEL = "本机干跑，桩模型，非正式"

NORMAL_SCENARIOS = {
    "data_processing": ("import_register", "join_one_to_many", "alias_rename", "sql_preview_export"),
    "labeling": ("dpd_label_build", "maturity_window", "label_rename", "multi_target"),
    "feature": ("psi_by_month", "iv_binning", "vif_screen", "importance_rank"),
    "modeling": ("lr_scorecard", "xgb_early_stop", "weighted_fit", "calibrated_ensemble"),
    "validation": ("single_model_report", "multi_model_batch", "pmml_consistency", "stress_report"),
    "strategy": ("rule_mining", "segment_cutoff", "limit_pricing", "swap_analysis"),
    "risk_portfolio": ("vintage_curve", "roll_rate", "concentration", "monthly_review"),
    "monitoring": ("baseline_psi", "schedule_proposal", "feedback_labels", "red_light_triage"),
    "online_decision": ("single_decision", "batch_decision", "shadow_mode", "rollback_readback"),
    "connector_antifraud": ("source_query", "event_window", "relation_replay", "coverage_snapshot"),
    "collections": ("candidate_policy", "queue_build", "reference_execution", "cost_recovery"),
}
BOUNDARY_SCENARIOS = (
    ("ambiguous_target", "clarification"),
    ("missing_required_field", "clarification"),
    ("conflicting_constraint", "clarification"),
    ("forbidden_outbound_secret", "rejected"),
    ("unauthorized_source", "rejected"),
    ("future_label_leak", "rejected"),
    ("duplicate_side_effect_request", "rejected"),
)
RECOVERY_FAULTS = (
    "worker_crash_after_commit", "response_lost_after_commit", "stale_cas_write",
    "tool_timeout", "service_restart_mid_step", "cancel_then_resume",
)
EXTENSION_SAFETY = {
    "online_decision": "replayed_request_second_decision",
    "connector_antifraud": "revoked_grant_read",
    "collections": "unit_mismatch_execution",
}
BOUNDARY_PER_FAMILY = (3, 3, 3, 3, 2, 2, 2, 2)
CRITICAL_PER_FAMILY = (3, 3, 3, 3, 2, 2, 2, 2)


class BenchmarkRefused(RuntimeError):
    def __init__(self, blockers):
        super().__init__("formal_benchmark_refused: " + ", ".join(blockers))
        self.blockers = list(blockers)


def _variant(salt: bytes, case_id: str) -> dict:
    mac = hmac.new(salt, case_id.encode(), hashlib.sha256).digest()
    return {
        "variant_sha256": mac.hex(),
        "seed": int.from_bytes(mac[:4], "big"),
        "row_scale": 200 + int.from_bytes(mac[4:6], "big") % 1800,
        "field_salt": mac[6:10].hex(),
        "threshold_bp": 500 + int.from_bytes(mac[10:12], "big") % 4000,
    }


def _specs(salt: bytes) -> list[dict]:
    specs = []

    def add(family, cohort, scenario, expected, *, critical=False, fault=None):
        ordinal = sum(s["family"] == family and s["cohort"] == cohort for s in specs) + 1
        case_id = f"hb-{family}-{cohort}-{ordinal:02d}"
        spec = {"id": case_id, "family": family, "cohort": cohort, "scenario": scenario,
                "expected_result": expected, "critical": critical, "fault_kind": fault,
                "variant": _variant(salt, case_id)}
        specs.append(spec)

    for index, family in enumerate(BASE_FAMILIES):
        scenarios = NORMAL_SCENARIOS[family]
        for n in range(8):
            add(family, "normal", scenarios[n % len(scenarios)], "done", critical=n < CRITICAL_PER_FAMILY[index])
    rotation = 0
    for index, family in enumerate(BASE_FAMILIES):
        for _ in range(BOUNDARY_PER_FAMILY[index]):
            scenario, expected = BOUNDARY_SCENARIOS[rotation % len(BOUNDARY_SCENARIOS)]
            rotation += 1
            add(family, "boundary", scenario, expected)
    for index, family in enumerate(BASE_FAMILIES):
        for n in range(2):
            fault = RECOVERY_FAULTS[(2 * index + n) % len(RECOVERY_FAULTS)]
            add(family, "recovery", fault, "recovered", fault=fault)
    for family in EXTENSION_FAMILIES:
        scenarios = NORMAL_SCENARIOS[family]
        for n in range(10):
            add(family, "extension", scenarios[n % len(scenarios)], "done")
        add(family, "extension_safety", EXTENSION_SAFETY[family], "rejected")
    return specs


def _case_row(spec: dict) -> dict:
    expected = {"case_id": spec["id"], "expected_result": spec["expected_result"],
                "objective": f"{spec['family']}:{spec['scenario']}"}
    fault = None
    if spec["fault_kind"]:
        fault = digest({"case_id": spec["id"], "fault_kind": spec["fault_kind"],
                        "injection": "deterministic_once", "seed": spec["variant"]["seed"]})
    return {
        "id": spec["id"], "family": spec["family"], "cohort": spec["cohort"],
        "expected_result": spec["expected_result"], "case_sha256": digest(spec),
        "expected_sha256": digest(expected), "critical": spec["critical"],
        "fault_contract_sha256": fault, "spec": spec, "expected": expected,
    }


def _attempts(cases: list[dict]) -> list[dict]:
    attempts, first_normal = [], set()
    for case in cases:
        if case["cohort"] == "recovery":
            mode = "recovery"
        elif case["cohort"] in {"normal", "extension"} and case["family"] not in first_normal:
            first_normal.add(case["family"])
            mode = "cold_start"
        else:
            mode = "steady"
        attempts.append({"id": f"{case['id']}-r0", "case_id": case["id"], "repetition": 0, "load_mode": mode})
    for case in cases:
        if case["critical"]:
            for repetition in (1, 2, 3):
                attempts.append({"id": f"{case['id']}-r{repetition}", "case_id": case["id"],
                                 "repetition": repetition, "load_mode": "concurrent"})
    return attempts


def validate_composition(cases: list[dict], attempts: list[dict]) -> None:
    """Same inventory rules as FrozenBenchmark, without the pending budgets."""
    ids = [c["id"] for c in cases]
    if len(ids) != CASE_COUNT or len(set(ids)) != CASE_COUNT:
        raise ValueError("hidden_benchmark_requires_133_distinct_cases")
    normal = Counter(c["family"] for c in cases if c["cohort"] == "normal")
    cohorts = Counter(c["cohort"] for c in cases)
    if normal != dict.fromkeys(BASE_FAMILIES, 8) or cohorts["boundary"] != 20 or cohorts["recovery"] != 16:
        raise ValueError("hidden_benchmark_base_cohort_mismatch")
    for family in EXTENSION_FAMILIES:
        if (sum(c["family"] == family and c["cohort"] == "extension" for c in cases) < 10
                or not any(c["family"] == family and c["cohort"] == "extension_safety" for c in cases)):
            raise ValueError("hidden_benchmark_extension_cohort_mismatch")
    critical = {c["id"] for c in cases if c["critical"]}
    if len(critical) != 20 or any(c["cohort"] != "normal" for c in cases if c["critical"]):
        raise ValueError("hidden_benchmark_requires_20_critical_normal_cases")
    expected = {(i, 0) for i in ids} | {(i, r) for i in critical for r in (1, 2, 3)}
    observed = [(a["case_id"], a["repetition"]) for a in attempts]
    if (len(attempts) != SLOT_COUNT or len(set(observed)) != len(observed) or set(observed) != expected
            or len({a["id"] for a in attempts}) != SLOT_COUNT):
        raise ValueError("hidden_benchmark_requires_193_slots")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_new(path: Path, value, mode=0o600):
    data = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode), "wb") as handle:
        handle.write(data)


def freeze_case_set(held_root: Path, freeze_id: str, *, salt: bytes | None = None, commit: str | None = None,
                    frozen_at: str | None = None) -> dict:
    """Write an immutable held case set; an existing freeze is never replaced."""
    if not freeze_id.replace("-", "").replace("_", "").isalnum() or len(freeze_id) > 80:
        raise ValueError("invalid_freeze_id")
    salt = secrets.token_bytes(32) if salt is None else salt
    if len(salt) < 16:
        raise ValueError("hidden_benchmark_salt_too_short")
    cases = [_case_row(spec) for spec in _specs(salt)]
    attempts = _attempts(cases)
    validate_composition(cases, attempts)
    held_root.mkdir(parents=True, exist_ok=True)
    target = held_root / freeze_id
    target.mkdir(mode=0o700)  # FileExistsError: a freeze is never overwritten
    manifest = {
        "schema": SCHEMA, "freeze_id": freeze_id,
        "frozen_at": frozen_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": commit or _git_commit(),
        "case_count": len(cases), "slot_count": len(attempts),
        "cohorts": dict(Counter(c["cohort"] for c in cases)),
        "critical_cases": sum(c["critical"] for c in cases),
        "load_modes": dict(Counter(a["load_mode"] for a in attempts)),
        "case_set_sha256": digest(cases), "attempts_sha256": digest(attempts),
        "salt_sha256": hashlib.sha256(salt).hexdigest(),
        "expected_result_type_frozen_before_run": True,
        "pending_user_decisions": {"budget": "pending", "per_case_budget": "pending", "slo": "pending",
                                   "real_model": "pending", "hardware_and_load": "pending"},
        "load_mode_assignment": "proposal_until_slo_freeze",
        "holder_note": "salt and case specs stay in this held directory; never commit them",
    }
    _write_new(target / "salt.hex", salt.hex().encode())
    _write_new(target / "cases.json", cases)
    _write_new(target / "attempts.json", attempts)
    _write_new(target / "freeze-manifest.json", manifest, mode=0o644)
    return manifest


def load_frozen_case_set(freeze_dir: Path) -> dict:
    """Recompute every digest and the composition; any drift fails closed."""
    manifest = json.loads((freeze_dir / "freeze-manifest.json").read_text(encoding="utf-8"))
    salt = bytes.fromhex((freeze_dir / "salt.hex").read_text(encoding="utf-8").strip())
    cases = json.loads((freeze_dir / "cases.json").read_text(encoding="utf-8"))
    attempts = json.loads((freeze_dir / "attempts.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or hashlib.sha256(salt).hexdigest() != manifest.get("salt_sha256"):
        raise ValueError("hidden_benchmark_manifest_mismatch")
    if digest(cases) != manifest.get("case_set_sha256") or digest(attempts) != manifest.get("attempts_sha256"):
        raise ValueError("hidden_benchmark_digest_mismatch")
    for case in cases:
        if (case["case_sha256"] != digest(case["spec"]) or case["expected_sha256"] != digest(case["expected"])
                or case["spec"]["variant"] != _variant(salt, case["id"])):
            raise ValueError("hidden_benchmark_case_drift")
    validate_composition(cases, attempts)
    return {"manifest": manifest, "cases": cases, "attempts": attempts}


def to_formal_benchmark(frozen: dict, *, benchmark_id, commit, source_inventory_sha256, model, environment,
                        budgets: dict, slo: dict) -> FrozenBenchmark:
    """Only after the user decides budgets/SLO/model/environment."""
    cases = [{key: case[key] for key in ("id", "family", "cohort", "expected_result", "case_sha256",
                                          "expected_sha256", "critical", "fault_contract_sha256")}
             | {"budget": budgets[case["id"]]} for case in frozen["cases"]]
    thresholds = {"normal_minimum_successes": 61, "normal_per_family_minimum_successes": 7, "boundary": "all_correct",
                  "recovery_minimum_rate": 0.95, "deterministic_fault_contracts": "all_verified",
                  "extension_minimum_rate": 0.95, "repeat_minimum_rate": 0.95, "repeats_per_critical_case": 3,
                  "maximum_serious_errors": 0}
    return FrozenBenchmark.model_validate_json(json.dumps({
        "schema": "marvis.formal-benchmark.v1", "benchmark_id": benchmark_id, "commit": commit,
        "source_inventory_sha256": source_inventory_sha256, "model": model, "environment": environment,
        "thresholds": thresholds, "cases": cases, "attempts": frozen["attempts"], "slo": slo,
    }))


def _pending(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in {"", "pending", "pending_user_decision"})


def formal_readiness(ledger_path: Path = DEFAULT_LEDGER, profile_path: Path = DEFAULT_PROFILE,
                     freeze_dir: Path | None = None, binding_path: Path | None = None,
                     secret_env: dict | None = None) -> list[str]:
    """Return every blocker; an empty list is required before a formal run."""
    blockers = []
    try:
        plan = json.loads(ledger_path.read_text(encoding="utf-8"))["agent_benchmark_plan"]
        cap = plan.get("cost_cap")
    except (OSError, ValueError, KeyError, TypeError):
        cap = None
        blockers.append("ledger_agent_benchmark_plan_unavailable")
    if not isinstance(cap, dict):
        blockers.append("cost_cap_not_recorded")
    else:
        if cap.get("status") != "decided":
            blockers.append("cost_cap_pending_user_decision")
        for key in ("max_cost", "currency", "price_book"):
            if _pending(cap.get(key)):
                blockers.append(f"cost_cap_{key}_pending")
    from .perf_gate import formal_blockers, load_profile
    try:
        blockers.extend(formal_blockers(load_profile(profile_path)))
    except (OSError, ValueError) as exc:
        blockers.append(f"performance_profile_invalid:{exc}")
    if freeze_dir is None:
        blockers.append("hidden_case_set_not_selected")
    else:
        try:
            load_frozen_case_set(freeze_dir)
        except (OSError, ValueError, KeyError, TypeError):
            blockers.append("hidden_case_set_invalid")
    if binding_path is None:
        blockers.extend(["real_model_profile_not_frozen", "per_case_budgets_not_frozen",
                         "held_case_materials_not_materialized"])
    elif freeze_dir is not None:
        try:
            from .hidden_formal_contract import prepare_hidden_execution
            prepare_hidden_execution(freeze_dir, binding_path, ledger_path=ledger_path, profile_path=profile_path, secret_env=secret_env)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Validation text may contain private material. Expose only the type.
            blockers.append(f"hidden_execution_binding_invalid:{type(exc).__name__}")
    return blockers


def run_formal(freeze_dir: Path, *, ledger_path: Path = DEFAULT_LEDGER, profile_path: Path = DEFAULT_PROFILE,
               binding_path: Path | None = None, output_dir: Path | None = None,
               evidence_custody_dir: Path | None = None, secret_env: dict | None = None):
    """Execute immutable bindings; retain only unsigned candidate originals."""
    blockers = formal_readiness(ledger_path, profile_path, freeze_dir, binding_path, secret_env)
    if output_dir is None:
        blockers.append("hidden_output_directory_required")
    if evidence_custody_dir is None:
        blockers.append("hidden_separate_evidence_custody_required")
    if blockers:
        raise BenchmarkRefused(blockers)
    from .hidden_formal_contract import prepare_hidden_execution
    from .hidden_formal_runner import execute_prepared
    prepared = prepare_hidden_execution(freeze_dir, binding_path, ledger_path=ledger_path,
                                        profile_path=profile_path, secret_env=secret_env)
    return execute_prepared(prepared, binding_path=binding_path, freeze_dir=freeze_dir,
                            output_dir=output_dir, evidence_custody_dir=evidence_custody_dir, secret_env=secret_env)


def _stub_llm(case: dict) -> str:
    """No network: echo the frozen expected type so only the harness is exercised."""
    time.sleep(0.001)
    return case["expected_result"]


def _dry_run_slot(attempt: dict, case: dict, work: Path) -> tuple[dict, dict]:
    from marvis.process_timing import (
        observe_process_timing,
        read_process_timing,
        timed_span,
    )

    from .process_timing_collector import build_formal_observation
    from .runtime_process import process_observation

    origin = time.monotonic_ns()
    journal = work / f"{attempt['id']}.timing.jsonl"
    with observe_process_timing(journal, origin):
        with timed_span("human_wait", "operator_wait"):
            time.sleep(0.002)
        started = time.monotonic_ns() - origin
        actual = _stub_llm(case)
        finished = time.monotonic_ns() - origin
        with timed_span("tool", "stub_tool"), timed_span("subprocess", "stub_process"):
            subprocess.run([sys.executable, "-I", "-S", "-c", "pass"], check=True, timeout=30)
        closed = time.monotonic_ns() - origin
    llm_events = [{"event": "started", "attempt_id": "stub-1", "elapsed_ns": started},
                  {"event": "finished", "attempt_id": "stub-1", "elapsed_ns": finished},
                  {"event": "measurement_closed", "elapsed_ns": closed}]
    backend = [{"event": "confirmation_plan_states_enabled", "elapsed_ns": 0},
               {"event": "closed", "elapsed_ns": closed, "valid": True}]
    duration_ms = max(1, (time.monotonic_ns() - origin) // 1_000_000)
    execution = {"case_id": case["id"], "duration_ms": duration_ms, "llm_events": llm_events,
                 "backend_events": backend, "human_interventions": 0,
                 "process_observation": process_observation(llm_events, [], backend, plan_states=[]),
                 "process_timing_events": read_process_timing(journal)}
    model = {"profile": "dry-run-stub", "model_name": "stub-llm", "source": "stub_llm_dry_run"}
    environment = {"hardware": "local_unfrozen", "load": "local_unfrozen"}
    observation, coverage = build_formal_observation(
        execution, attempt_id=attempt["id"], case_id=case["id"], run_id="dry-run", load_mode=attempt["load_mode"],
        model=model, environment=environment)
    telemetry = _formal_telemetry_check(work, observation, execution, attempt, case, model, environment)
    return {"actual_result": actual, "telemetry": telemetry, "coverage": coverage}, observation


def _formal_telemetry_check(work, observation, execution, attempt, case, model, environment):
    """Run the installed formal-gate reader over the collector output."""
    from .formal_acceptance import _telemetry

    raw = json.dumps(observation, sort_keys=True).encode()
    name = f"{attempt['id']}.process-observation.json"
    (work / name).write_bytes(raw)
    reference = {"path": name, "sha256": digest(raw)}
    record = {"artifact_hashes": {name: reference["sha256"]}}
    benchmark = SimpleNamespace(model=SimpleNamespace(model_dump=lambda: model),
                                environment=SimpleNamespace(model_dump=lambda: environment))
    definition = SimpleNamespace(family=case["family"], actions=[], model_dump=lambda: {"case_id": case["id"]})
    run = {"manifest": {"run_id": "dry-run"}, "executions": {case["id"]: execution}}
    facts, reason = _telemetry(work.resolve(), record, reference, benchmark,
                               SimpleNamespace(id=attempt["id"], case_id=case["id"], load_mode=attempt["load_mode"]),
                               run, definition)
    return {"reason": reason, "wall_seconds": facts.get("wall_seconds"),
            "timing_components": dict(facts.get("timing_components", {}))}


def dry_run(freeze_dir: Path, out_dir: Path) -> dict:
    frozen = load_frozen_case_set(freeze_dir)
    cases = {case["id"]: case for case in frozen["cases"]}
    rows, out_dir = [], Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="marvis-hb-dry-") as tmp:
        work = Path(tmp)
        for attempt in frozen["attempts"]:
            case = cases[attempt["case_id"]]
            try:
                result, _observation = _dry_run_slot(attempt, case, work)
                completed = result["telemetry"]["reason"] is None and result["actual_result"] == case["expected_result"]
                rows.append({"attempt_id": attempt["id"], "case_id": case["id"], "family": case["family"],
                             "cohort": case["cohort"], "repetition": attempt["repetition"],
                             "load_mode": attempt["load_mode"], "success": completed,
                             "telemetry_unavailable_reason": result["telemetry"]["reason"],
                             "domain_status": "unavailable", "wall_seconds": result["telemetry"]["wall_seconds"],
                             "timing_components": result["telemetry"]["timing_components"]})
            except Exception as exc:  # recorded, never dropped from the denominator
                rows.append({"attempt_id": attempt["id"], "case_id": case["id"], "family": case["family"],
                             "cohort": case["cohort"], "repetition": attempt["repetition"],
                             "load_mode": attempt["load_mode"], "success": False, "domain_status": "unavailable",
                             "error": type(exc).__name__ + ":" + str(exc)[:120]})
    originals = [r for r in rows if r["repetition"] == 0]
    report = {
        "schema": "marvis.hidden-benchmark-dry-run.v1",
        "mode": "local_dry_run_stub_llm",
        "label": DRY_RUN_LABEL,
        "acceptance_claim": "none",
        "freeze_id": frozen["manifest"]["freeze_id"],
        "case_set_sha256": frozen["manifest"]["case_set_sha256"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "meaning": "harness_chain_completed = stub LLM + side-car journal + collector + installed formal _telemetry reader; missing actual warmup/concurrency is rejected; not task success",
        "harness_chain_completed": {
            "all_slots": _summary(rows, None),
            "by_cohort": {cohort: _summary([r for r in originals if r["cohort"] == cohort], None)
                          for cohort in ("normal", "boundary", "recovery", "extension", "extension_safety")},
            "repeats": _summary([r for r in rows if r["repetition"] > 0], None),
        },
        "stub_wall_seconds_by_load_mode": {
            mode: {"slots": sum(r["load_mode"] == mode for r in rows),
                   "p50": _percentile([r["wall_seconds"] for r in rows if r["load_mode"] == mode and r.get("wall_seconds") is not None], 0.5),
                   "p95": _percentile([r["wall_seconds"] for r in rows if r["load_mode"] == mode and r.get("wall_seconds") is not None], 0.95)}
            for mode in ("cold_start", "steady", "concurrent", "recovery")
        },
        "formal_blockers": formal_readiness(freeze_dir=freeze_dir),
        "attempts": rows,
    }
    (out_dir / "dry-run-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hidden benchmark freeze / formal gate / local dry run.")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--freeze-id", required=True)
    freeze.add_argument("--held-root", type=Path, default=DEFAULT_HELD_ROOT)
    formal = sub.add_parser("formal")
    formal.add_argument("--freeze-dir", type=Path, required=True)
    formal.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    formal.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    formal.add_argument("--binding", type=Path)
    formal.add_argument("--out", type=Path)
    formal.add_argument("--custody", type=Path)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--freeze-dir", type=Path, required=True)
    dry.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze_case_set(args.held_root, args.freeze_id), ensure_ascii=False, indent=2))
        return 0
    if args.command == "formal":
        try:
            report = run_formal(args.freeze_dir, ledger_path=args.ledger, profile_path=args.profile,
                                binding_path=args.binding, output_dir=args.out, evidence_custody_dir=args.custody)
        except BenchmarkRefused as exc:
            print(json.dumps({"status": "refused", "blockers": exc.blockers}, ensure_ascii=False, indent=2))
            return 2
        if not isinstance(report, dict) or report.get("denominator") != 193:
            return 2
        print(json.dumps({"status": "unsigned_candidate_originals", "denominator": report["denominator"],
                          "report_path": report["report_path"], "acceptance_claim": report["acceptance_claim"]}, ensure_ascii=False, indent=2))
        return 0
    report = dry_run(args.freeze_dir, args.out)
    print(json.dumps({"label": report["label"], "slots": report["harness_chain_completed"]["all_slots"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
