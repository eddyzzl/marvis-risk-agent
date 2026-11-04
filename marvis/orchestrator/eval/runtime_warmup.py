"""Fixed read-only warmup and original load-boundary verification.

This module does not certify a run. Its originals are still bound by the run
manifest, source inventory and separately held executor/reviewer signatures.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from .runtime_contracts import digest

SCHEMA = "marvis.runtime-load-originals.v1"
READ_ONLY_ENDPOINTS = frozenset({"/", "/api/branding", "/api/tasks"})
LOAD_MODES = frozenset({"cold_start", "steady", "concurrent", "recovery"})


class LoadBindingError(ValueError):
    pass


def _need(value, reason):
    if not value:
        raise LoadBindingError(reason)


def validate_load_binding(value):
    _need(isinstance(value, dict) and set(value) == {"load_mode", "warmup", "load_sha256", "load_spec"}, "invalid_runtime_load_binding")
    _need(value["load_mode"] in LOAD_MODES, "invalid_runtime_load_mode")
    commitment = value["load_sha256"]
    _need(isinstance(commitment, str) and len(commitment) == 64
          and set(commitment) <= set("0123456789abcdef"), "invalid_runtime_load_hash")
    specification = value["load_spec"]
    _need(isinstance(specification, dict) and set(specification) in ({"profile_load", "load_policy"}, {"profile_load", "load_policy", "profile_slo"})
          and digest(specification) == commitment, "runtime_load_spec_hash_mismatch")
    policy = value["warmup"]
    if value["load_mode"] != "steady":
        _need(policy is None, "warmup_only_for_steady")
        return value
    _need(policy == specification["load_policy"].get("warmup"), "warmup_frozen_policy_mismatch")
    _need(isinstance(policy, dict) and set(policy) == {"endpoints", "requests", "minimum_seconds"}, "invalid_warmup_policy")
    endpoints = policy["endpoints"]
    _need(isinstance(endpoints, list) and 0 < len(endpoints) <= 3
          and all(isinstance(item, str) and item in READ_ONLY_ENDPOINTS for item in endpoints)
          and len(set(endpoints)) == len(endpoints), "warmup_requires_fixed_readonly_endpoints")
    _need(type(policy["requests"]) is int and len(endpoints) <= policy["requests"] <= 1000,
          "invalid_warmup_request_count")
    _need(type(policy["minimum_seconds"]) in {int, float}
          and math.isfinite(policy["minimum_seconds"])
          and 0 <= policy["minimum_seconds"] <= 600, "invalid_warmup_duration")
    return value


def child_process_identity(config):
    """Written once by the spawned process before its application is created."""
    identity = {"schema": "marvis.runtime-child-identity.v1", "pid": os.getpid(),
                "workspace_sha256": digest(str(Path(config["workspace"]).resolve())),
                "nonce": config["load_nonce"]}
    suffix = "" if config.get("epoch", 0) == 0 else f"-epoch-{config['epoch']}"
    path = Path(config["workspace"]).parent / f"runtime-child-identity{suffix}.json"
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as stream:
        stream.write(raw)


def append_event(evidence, name, origin_ns, *, pid, **fields):
    evidence["events"].append({"event": name, "elapsed_ns": time.monotonic_ns() - origin_ns,
                               "pid": pid, **fields})


def perform_warmup(client, policy, *, process, evidence, origin_ns, deadline):
    """Execute the frozen count, then hold to its duration; never call a model."""
    start = time.monotonic()
    append_event(evidence, "warmup_started", origin_ns, pid=process.pid)
    for ordinal in range(policy["requests"]):
        _need(process.poll() is None, "warmup_process_exited")
        remaining = deadline - time.monotonic()
        _need(remaining > 0, "warmup_wall_budget_exhausted")
        endpoint = policy["endpoints"][ordinal % len(policy["endpoints"]) ]
        response = client.get(endpoint, timeout=min(remaining, 10), follow_redirects=False)
        _need(response.status_code == 200, "warmup_read_failed")
        append_event(evidence, "warmup_response", origin_ns, pid=process.pid,
                     ordinal=ordinal + 1, endpoint=endpoint, status_code=response.status_code)
    while time.monotonic() - start < policy["minimum_seconds"]:
        _need(process.poll() is None and time.monotonic() < deadline, "warmup_wall_budget_exhausted")
        time.sleep(min(0.05, policy["minimum_seconds"] - (time.monotonic() - start)))
    _need(process.poll() is None, "warmup_process_exited")
    append_event(evidence, "warmup_completed", origin_ns, pid=process.pid)


def measurement_window(execution, *, load_mode, frozen_load_sha256):
    """Derive the only allowed window from fixed runner originals.

    The caller cannot select boundaries. Steady starts exactly at the recorded
    successful warmup completion and runs through the original case end; cold,
    concurrent and recovery retain the whole process duration.
    """
    _need(not execution.get("process_epochs"), "restarted_process_requires_epoch_timing")
    duration = execution.get("duration_ms")
    _need(type(duration) is int and duration > 0, "original_duration_unavailable")
    evidence = execution.get("load_originals")
    _need(isinstance(evidence, dict) and set(evidence) == {
        "schema", "binding", "child_identity", "child_identity_sha256", "events", "origin_ns"
    } and evidence["schema"] == SCHEMA, "load_originals_unavailable")
    _need(type(evidence["origin_ns"]) is int and evidence["origin_ns"] > 0, "load_origin_unavailable")
    binding = validate_load_binding(evidence["binding"])
    _need(binding["load_mode"] == load_mode and binding["load_sha256"] == frozen_load_sha256,
          "load_originals_frozen_binding_mismatch")
    child = evidence["child_identity"]
    _need(isinstance(child, dict) and set(child) == {"schema", "pid", "workspace_sha256", "nonce"}
          and child["schema"] == "marvis.runtime-child-identity.v1"
          and type(child["pid"]) is int and child["pid"] > 0
          and isinstance(child["nonce"], str) and len(child["nonce"]) == 32
          and isinstance(child["workspace_sha256"], str) and len(child["workspace_sha256"]) == 64
          and digest(child) == evidence["child_identity_sha256"], "load_child_identity_mismatch")
    events = evidence["events"]
    _need(isinstance(events, list) and 4 <= len(events) <= 1010, "load_events_unavailable")
    previous = -1
    for row in events:
        _need(isinstance(row, dict) and row.get("pid") == child["pid"]
              and type(row.get("elapsed_ns")) is int and previous <= row["elapsed_ns"] <= (duration + 1) * 1_000_000,
              "load_event_process_or_order_mismatch")
        previous = row["elapsed_ns"]
    names = [row["event"] for row in events]
    expected = ["process_ready"]
    if load_mode == "steady":
        expected += ["warmup_started", *(["warmup_response"] * binding["warmup"]["requests"]), "warmup_completed"]
    expected += ["journey_started", "journey_finished", "process_stopped"]
    _need(names == expected, "load_event_sequence_mismatch")
    ready = events[0]
    _need(ready.get("workspace_sha256") == child["workspace_sha256"] and ready.get("nonce") == child["nonce"],
          "load_ready_identity_mismatch")
    start_ns = 0
    if load_mode == "steady":
        policy = binding["warmup"]
        start, end = events[1], events[-4]
        _need(end["elapsed_ns"] - start["elapsed_ns"] >= policy["minimum_seconds"] * 1_000_000_000,
              "warmup_duration_incomplete")
        for ordinal, row in enumerate(events[2:-4]):
            _need(set(row) == {"event", "elapsed_ns", "pid", "ordinal", "endpoint", "status_code"}
                  and row["ordinal"] == ordinal + 1
                  and row["endpoint"] == policy["endpoints"][ordinal % len(policy["endpoints"])]
                  and row["status_code"] == 200, "warmup_response_incomplete")
        start_ns = end["elapsed_ns"]
    _need(start_ns < duration * 1_000_000, "empty_measurement_window")
    return {"schema": "marvis.runtime-measurement-window.v1", "start_ns": start_ns,
            "end_ns": duration * 1_000_000, "load_originals_sha256": digest(evidence),
            "load_sha256": frozen_load_sha256, "scope": "post_warmup_to_original_case_end" if load_mode == "steady" else "whole_original_case"}
