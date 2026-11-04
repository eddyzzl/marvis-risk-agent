"""Frozen local task-ingestion/read workload; never a software-S attestation.

The caller supplies every load/SLO decision and the exact dataset/package bytes.
This runner observes the actual local HTTP service. It cannot attest who approved
a supplied profile, and does not benchmark learner training or production traffic.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time

from . import perf_gate as base

SCHEMA = "marvis.performance-workload-binding.v1"
WORKLOAD = "task-ingest-read-v1"
_KEYS = {"schema", "workload", "scope", "dataset", "source_inventory_sha256", "execution"}
_EXECUTION = {"cold_start_runs": (1, 20), "recovery_runs": (1, 10), "requests_per_worker": (1, 1000),
              "request_timeout_seconds": (1, 120), "startup_timeout_seconds": (5, 600)}


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def source_inventory():
    """Pin package code/assets, excluding interpreter caches and runtime output."""
    package = base.ROOT / "marvis"
    return {p.relative_to(base.ROOT).as_posix(): _hash(p.read_bytes())
            for p in sorted(package.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}}


def hardware_identity():
    return {k: v for k, v in base._machine().items() if k != "scope"}


def hardware_fingerprint():
    return _hash(_canonical(hardware_identity()))


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _prepare(profile, binding_path):
    base.validate_profile(profile)
    if base.formal_blockers(profile):
        raise ValueError("performance_profile_not_frozen")
    binding_bytes = binding_path.read_bytes()
    binding = json.loads(binding_bytes)
    if (type(binding) is not dict or set(binding) != _KEYS or binding["schema"] != SCHEMA
            or binding["workload"] != WORKLOAD or binding["scope"] not in {"local_synthetic_fixture", "supplied_acceptance_environment"}):
        raise ValueError("unsupported_frozen_performance_workload")
    dataset, execution = binding["dataset"], binding["execution"]
    if (type(dataset) is not dict or set(dataset) != {"path", "sha256", "rows"}
            or type(dataset["path"]) is not str or not _digest(dataset["sha256"])
            or type(dataset["rows"]) is not int or not 1 <= dataset["rows"] <= 10_000_000
            or dataset["rows"] != profile["load"]["data_rows"]):
        raise ValueError("frozen_dataset_binding_invalid")
    if type(execution) is not dict or set(execution) != set(_EXECUTION):
        raise ValueError("frozen_execution_keys_invalid")
    for key, (minimum, maximum) in _EXECUTION.items():
        if type(execution[key]) is not int or not minimum <= execution[key] <= maximum:
            raise ValueError(f"frozen_execution_outside_bound:{key}")
    if not 1 <= profile["load"]["concurrent_users"] <= 64 or not 1 <= profile["load"]["sustained_seconds"] <= 600:
        raise ValueError("frozen_load_outside_supported_profile")
    counts = {"cold_start": execution["cold_start_runs"], "recovery": execution["recovery_runs"],
              "concurrent": execution["requests_per_worker"] * profile["load"]["concurrent_users"]}
    if any(count < profile["slo"][mode]["minimum_observations"] for mode, count in counts.items()):
        raise ValueError("frozen_load_cannot_meet_observation_minimum")
    path = Path(dataset["path"])
    if not path.is_absolute() or path.is_symlink() or path.suffix.lower() != ".csv" or not path.is_file():
        raise ValueError("frozen_dataset_requires_regular_absolute_csv")
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("frozen_dataset_exceeds_supported_size")
    raw = path.read_bytes()
    if _hash(raw) != dataset["sha256"]:
        raise ValueError("frozen_dataset_hash_mismatch")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    header = next(reader, [])
    if not header or len(set(header)) != len(header) or any(not name.strip() for name in header):
        raise ValueError("frozen_dataset_header_invalid")
    rows = 0
    for row in reader:
        if len(row) != len(header):
            raise ValueError("frozen_dataset_row_shape_invalid")
        rows += 1
    if rows != dataset["rows"]:
        raise ValueError("frozen_dataset_row_count_mismatch")
    inventory = source_inventory()
    if not _digest(binding["source_inventory_sha256"]) or _hash(_canonical(inventory)) != binding["source_inventory_sha256"]:
        raise ValueError("frozen_source_inventory_mismatch")
    if hardware_fingerprint() != profile["hardware"]["fingerprint_sha256"]:
        raise ValueError("frozen_hardware_fingerprint_mismatch")
    return binding, raw, inventory, binding_bytes


def _task_ids(payload):
    rows = json.loads(payload)
    if type(rows) is not list or any(type(row) is not dict or type(row.get("id")) is not str for row in rows):
        raise ValueError("task_inventory_shape_invalid")
    ids = sorted(row["id"] for row in rows)
    if len(set(ids)) != len(ids):
        raise ValueError("task_inventory_duplicate_identity")
    return ids


def _registered_dataset_rows(workspace):
    """Read the real registry to ensure the workload actually ingested its rows."""
    with sqlite3.connect(f"file:{workspace / 'marvis.sqlite'}?mode=ro", uri=True) as connection:
        return connection.execute("SELECT id, task_id, row_count, content_hash FROM datasets ORDER BY id").fetchall()


def _ingest_task(server, source, index, timeout):
    started = time.monotonic()
    status, _, payload = base._request(server.base + "/api/tasks", method="POST", body=base._task_body(source, index), timeout=timeout)
    if status == 200:
        task_id = json.loads(payload)["id"]
        status, _, _ = base._request(server.base + f"/api/tasks/{task_id}/datasets/register-path", method="POST",
                                    body={"path": str(source / "sample.csv"), "role": "sample"}, timeout=timeout)
    return status, time.monotonic() - started


def _evaluate(profile, samples, integrity_errors):
    results, failures = {}, list(integrity_errors)
    for mode in base.LOAD_MODES:
        group = [s for s in samples if s["mode"] == mode]
        # Failed requests remain in the timing denominator. Timeout durations
        # cannot disappear just because no successful response was measured.
        latencies = [s["seconds"] for s in group if s["seconds"] is not None]
        result = base._stats(latencies, sum(not s["ok"] for s in group), len(group))
        slo = profile["slo"][mode]
        reasons = []
        if result["observations"] < slo["minimum_observations"]:
            reasons.append("insufficient_observations")
        if len(latencies) != len(group) or result["p95_seconds"] is None or result["p95_seconds"] > slo["p95_seconds"]:
            reasons.append("p95_exceeded_or_unmeasured")
        if result["error_rate"] is None or result["error_rate"] > slo["max_error_rate"]:
            reasons.append("error_rate_exceeded_or_unmeasured")
        results[mode] = {**result, "passed": not reasons, "failure_reasons": reasons, "slo": slo}
        failures.extend(f"{mode}:{reason}" for reason in reasons)
    return results, failures


def run_frozen_workload(profile, binding_path, out_dir):
    """Validate before touching output; then retain every measured failure."""
    binding, dataset_bytes, inventory, binding_bytes = _prepare(profile, binding_path)
    binding_hash = _hash(binding_bytes)
    if out_dir.exists():
        raise ValueError("performance_output_must_not_already_exist")
    out_dir.mkdir(parents=True)
    cfg, load = binding["execution"], profile["load"]
    timeout, startup = cfg["request_timeout_seconds"], cfg["startup_timeout_seconds"]
    samples, integrity, recoveries = [], [], []
    journal = out_dir / "observations.jsonl"

    def record(mode, operation, seconds, ok, *, status=None):
        if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
            raise ValueError("invalid_performance_duration")
        row = {"id": len(samples), "mode": mode, "operation": operation, "seconds": seconds,
               "ok": bool(ok), "http_status": status, "retries": 0}
        samples.append(row)
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    (out_dir / "frozen-profile.json").write_bytes(_canonical(profile))
    (out_dir / "frozen-binding.json").write_bytes(binding_bytes)
    (out_dir / "source-inventory.json").write_bytes(_canonical(inventory))
    (out_dir / "sample.csv").write_bytes(dataset_bytes)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="marvis-perf-frozen-") as temporary:
        temp = Path(temporary)
        for run in range(cfg["cold_start_runs"]):
            server = base._Server(temp / f"cold-{run}", base._free_port(), out_dir / "server.log")
            try:
                beginning = time.monotonic()
                seconds = server.start(startup)
                record("cold_start", "fresh_workspace_start", time.monotonic() - beginning, seconds is not None)
            finally:
                server.stop()
        workspace = temp / "load"
        source = workspace / "source"
        source.mkdir(parents=True)
        (source / "sample.csv").write_bytes(dataset_bytes)
        server = base._Server(workspace, base._free_port(), out_dir / "server.log")
        try:
            if server.start(startup) is None:
                integrity.append("workload_server_failed_to_start")
            else:
                status, _ = _ingest_task(server, source, 0, timeout)
                if status != 201:
                    integrity.append("seed_task_ingestion_failed")
                deadline, index = time.monotonic() + load["sustained_seconds"], 0
                while time.monotonic() < deadline:
                    path = ("/api/tasks", "/", "/api/branding")[index % 3]
                    status, seconds, _ = base._request(server.base + path, timeout=timeout)
                    record("steady", "loaded_workspace_read", seconds, status == 200, status=status)
                    index += 1

                def worker(worker_id):
                    rows = []
                    for number in range(cfg["requests_per_worker"]):
                        create = number % 4 == 0
                        if create:
                            status, seconds = _ingest_task(server, source, 1 + worker_id * cfg["requests_per_worker"] + number, timeout)
                        else:
                            status, seconds, _ = base._request(server.base + "/api/tasks", timeout=timeout)
                        rows.append(("material_ingest" if create else "task_list", seconds, status, status == (201 if create else 200)))
                    return rows

                with ThreadPoolExecutor(load["concurrent_users"]) as pool:
                    for rows in pool.map(worker, range(load["concurrent_users"])):
                        for operation, seconds, status, ok in rows:
                            record("concurrent", operation, seconds, ok, status=status)
                registered = _registered_dataset_rows(workspace)
                expected_ingestions = 1 + load["concurrent_users"] * math.ceil(cfg["requests_per_worker"] / 4)
                if len(registered) != expected_ingestions or any(row[2] != load["data_rows"] for row in registered):
                    integrity.append("registered_dataset_population_mismatch")
                for _ in range(cfg["recovery_runs"]):
                    status, _, payload = base._request(server.base + "/api/tasks", timeout=timeout)
                    before = _task_ids(payload) if status == 200 else None
                    before_datasets = _registered_dataset_rows(workspace)
                    server.stop(kill=True)
                    beginning = time.monotonic()
                    seconds = server.start(startup)
                    elapsed = time.monotonic() - beginning
                    status, _, payload = base._request(server.base + "/api/tasks", timeout=timeout) if seconds is not None else (None, 0, b"")
                    after = _task_ids(payload) if status == 200 else None
                    after_datasets = _registered_dataset_rows(workspace)
                    preserved = before is not None and bool(before) and before == after and before_datasets == after_datasets
                    recoveries.append({"before_sha256": _hash(_canonical([before, before_datasets])),
                                       "after_sha256": _hash(_canonical([after, after_datasets])), "preserved": preserved})
                    record("recovery", "sigkill_restart_same_workspace", elapsed, seconds is not None and preserved)
                    if not preserved:
                        integrity.append("recovery_task_identity_changed")
        except Exception as exc:  # keep measured failures; never convert a partial run to success
            integrity.append(f"workload_error:{type(exc).__name__}")
        finally:
            server.stop()
    if source_inventory() != inventory:
        integrity.append("source_inventory_changed_during_run")
    if hardware_fingerprint() != profile["hardware"]["fingerprint_sha256"]:
        integrity.append("hardware_identity_changed_during_run")
    try:
        if _hash(binding_path.read_bytes()) != binding_hash or _hash(Path(binding["dataset"]["path"]).read_bytes()) != binding["dataset"]["sha256"]:
            integrity.append("input_identity_changed_during_run")
    except OSError:
        integrity.append("input_identity_unavailable_after_run")
    results, failures = _evaluate(profile, samples, integrity)
    report = {"schema": "marvis.performance-frozen-result.v1", "workload": WORKLOAD, "scope": binding["scope"],
              "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "acceptance_claim": "none",
              "limitations": ["caller_supplied_approval_not_independently_authenticated", "task_ingestion_and_control_plane_only",
                              "no_model_training_or_production_traffic_claim", "not_software_S_acceptance"],
              "gate_verdict": "failed" if failures else "passed", "failure_reasons": failures,
              "profile_sha256": _hash(_canonical(profile)), "binding_sha256": binding_hash,
              "source_inventory_sha256": binding["source_inventory_sha256"], "dataset_sha256": binding["dataset"]["sha256"],
              "hardware": hardware_identity(), "elapsed_seconds": time.monotonic() - started,
              "results": results, "recovery_identities": recoveries,
              "workloads": {"cold_start": "fresh empty workspace start until GET /api/tasks is healthy",
                            "steady": "sequential GET /api/tasks, /, /api/branding on populated workspace for frozen sustained_seconds",
                            "concurrent": "frozen worker count; every fourth operation creates a task and registers exact CSV bytes, other operations GET /api/tasks",
                            "recovery": "SIGKILL server and restart same workspace; task and dataset identities must be unchanged"},
              "observations_sha256": _hash(journal.read_bytes()) if journal.exists() else None}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (out_dir / "report.md").write_text("# 冻结配置性能检查\n\n"
        f"范围：`{WORKLOAD}`；结果：`{report['gate_verdict']}`；正式验收声明：无。\n\n"
        "仅测量材料摄取与任务界面请求，不表示模型训练、机构生产或软件 S 验收完成。\n\n"
        "| 模式 | 观测 | 错误 | p95 秒 | 结果 |\n| --- | --- | --- | --- | --- |\n" + "\n".join(
            f"| {mode} | {row['observations']} | {row['errors']} | {row['p95_seconds']} | {row['passed']} |"
            for mode, row in results.items()) + "\n")
    return report
