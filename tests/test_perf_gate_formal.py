"""Frozen performance execution uses actual local HTTP with explicit test-only SLOs."""
import copy
import hashlib
import json

import pytest

from marvis.orchestrator.eval import perf_gate as base
from marvis.orchestrator.eval import perf_gate_formal as formal


def _fixture(tmp_path):
    source = tmp_path / "fixture.csv"
    source.write_text("x,y,split\n1,0,train\n2,1,test\n3,0,oot\n")
    profile = copy.deepcopy(base.load_profile())
    profile.update(status="frozen", approved_by="local-test-fixture-only", approved_on="2026-11-04")
    profile["hardware"] = {"description": "current local fixture host", "fingerprint_sha256": formal.hardware_fingerprint()}
    profile["load"] = {"data_rows": 3, "concurrent_users": 2, "sustained_seconds": 1}
    profile["slo"] = {mode: {"p95_seconds": 120., "minimum_observations": 1, "max_error_rate": 0.}
                      for mode in base.LOAD_MODES}
    binding = {"schema": formal.SCHEMA, "workload": formal.WORKLOAD, "scope": "local_synthetic_fixture",
               "dataset": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": 3},
               "source_inventory_sha256": formal._hash(formal._canonical(formal.source_inventory())),
               "execution": {"cold_start_runs": 1, "recovery_runs": 1, "requests_per_worker": 2,
                             "request_timeout_seconds": 10, "startup_timeout_seconds": 120}}
    profile_path, binding_path = tmp_path / "profile.json", tmp_path / "binding.json"
    profile_path.write_text(json.dumps(profile))
    binding_path.write_text(json.dumps(binding))
    return profile, binding, profile_path, binding_path


@pytest.mark.e2e
def test_frozen_actual_http_ingestion_cold_steady_concurrent_and_sigkill_recovery(tmp_path):
    _, binding, profile_path, binding_path = _fixture(tmp_path)
    output = tmp_path / "result"
    report = base.run_formal_gate(profile_path, binding_path=binding_path, out_dir=output)
    assert report["gate_verdict"] == "passed", report
    assert report["acceptance_claim"] == "none" and report["scope"] == "local_synthetic_fixture"
    assert report["dataset_sha256"] == binding["dataset"]["sha256"]
    assert report["results"]["concurrent"]["observations"] == 4
    assert report["recovery_identities"][0]["preserved"] is True
    assert report["observations_sha256"] == hashlib.sha256((output / "observations.jsonl").read_bytes()).hexdigest()
    rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert set(r["mode"] for r in rows) == set(base.LOAD_MODES)
    assert len(rows) == sum(r["observations"] for r in report["results"].values())
    assert all(r["retries"] == 0 for r in rows)
    assert (output / "sample.csv").read_bytes() == (tmp_path / "fixture.csv").read_bytes()
    assert report["binding_sha256"] == hashlib.sha256((output / "frozen-binding.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="must_not_already_exist"):
        base.run_formal_gate(profile_path, binding_path=binding_path, out_dir=output)


@pytest.mark.parametrize("mutation,reason", [
    ("dataset_hash", "dataset_hash"), ("dataset_rows", "dataset_row_count"),
    ("source", "source_inventory"), ("hardware", "hardware_fingerprint"),
    ("unknown_key", "unsupported_frozen"), ("load_minimum", "observation_minimum"),
    ("execution_unknown", "execution_keys"), ("pending", "not_frozen"),
])
def test_preflight_rejects_missing_or_changed_frozen_identity_before_server(tmp_path, monkeypatch, mutation, reason):
    profile, binding, _, binding_path = _fixture(tmp_path)
    if mutation == "dataset_hash":
        (tmp_path / "fixture.csv").write_text("x,y\n99,1\n")
    elif mutation == "dataset_rows":
        profile["load"]["data_rows"] = binding["dataset"]["rows"] = 4
    elif mutation == "source":
        binding["source_inventory_sha256"] = "a" * 64
    elif mutation == "hardware":
        profile["hardware"]["fingerprint_sha256"] = "a" * 64
    elif mutation == "unknown_key":
        binding["command"] = "must-not-run"
    elif mutation == "execution_unknown":
        binding["execution"]["url"] = "http://example.com"
    elif mutation == "pending":
        profile["status"] = "pending_user_freeze"
    else:
        profile["slo"]["concurrent"]["minimum_observations"] = 5
    binding_path.write_text(json.dumps(binding))
    def forbidden(*args, **kwargs):
        pytest.fail("invalid binding started application")
    monkeypatch.setattr(base._Server, "start", forbidden)
    output = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match=reason):
        formal.run_frozen_workload(profile, binding_path, output)
    assert not output.exists()


def test_failed_requests_and_missing_observations_cannot_disappear_from_slo_denominator(tmp_path):
    profile, _, _, _ = _fixture(tmp_path)
    profile["slo"]["steady"]["p95_seconds"] = 2.
    samples = [{"mode": "steady", "seconds": 1., "ok": True}, {"mode": "steady", "seconds": 10., "ok": False}]
    results, failures = formal._evaluate(profile, samples, ["recovery_task_identity_changed"])
    assert results["steady"]["observations"] == 2
    assert results["steady"]["p95_seconds"] == 10.
    assert results["steady"]["error_rate"] == .5
    assert "recovery_task_identity_changed" in failures
    assert "cold_start:insufficient_observations" in failures
    assert "steady:p95_exceeded_or_unmeasured" in failures
    assert "steady:error_rate_exceeded_or_unmeasured" in failures


def test_recovery_compares_exact_task_ids_and_rejects_duplicate_inventory():
    before = formal._task_ids(b'[{"id":"a"},{"id":"b"}]')
    after = formal._task_ids(b'[{"id":"a"},{"id":"c"}]')
    assert len(before) == len(after) and before != after
    with pytest.raises(ValueError, match="duplicate_identity"):
        formal._task_ids(b'[{"id":"a"},{"id":"a"}]')


@pytest.mark.parametrize("section,key,value", [("load", "data_rows", 2.5), ("load", "concurrent_users", True),
    ("slo", "minimum_observations", 1.5), ("slo", "max_error_rate", -.1), ("slo", "max_error_rate", 1.1)])
def test_frozen_profile_requires_integral_load_counts_and_valid_error_rate(tmp_path, section, key, value):
    profile, _, _, _ = _fixture(tmp_path)
    (profile["load"] if section == "load" else profile["slo"]["steady"])[key] = value
    with pytest.raises(ValueError):
        base.validate_profile(profile)


def test_formal_cli_never_falls_through_to_baseline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(base, "run_formal_gate", lambda *a, **kw: {"gate_verdict": "failed", "acceptance_claim": "none"})
    monkeypatch.setattr(base, "run_local_baseline", lambda *a, **kw: pytest.fail("formal fell through to baseline"))
    assert base.main(["formal", "--binding", str(tmp_path / "binding.json"), "--out", str(tmp_path / "result")]) == 1
    assert '"gate_verdict": "failed"' in capsys.readouterr().out


@pytest.mark.parametrize("key,value", [("approved_by", "   "), ("approved_on", "2026-99-99"), ("approved_on", "2026-1-2")])
def test_frozen_approval_metadata_requires_explicit_name_and_real_iso_date(tmp_path, key, value):
    profile, _, _, _ = _fixture(tmp_path)
    profile[key] = value
    with pytest.raises(ValueError):
        base.validate_profile(profile)
