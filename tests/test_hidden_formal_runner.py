"""Local protocol fixtures: never historical, paid or independent acceptance."""
import json
from pathlib import Path
from copy import deepcopy

import pytest
from test_hidden_formal_contract import (
    _materialized, _prepare,
    hidden_source_checkout as hidden_source_checkout,
    isolated_hidden_source as isolated_hidden_source,
)

from marvis.orchestrator.eval import hidden_benchmark
from marvis.orchestrator.eval import hidden_formal_runner as runner


def test_pending_default_still_refuses_before_creating_outputs(tmp_path):
    fixture = _materialized(tmp_path)
    with pytest.raises(hidden_benchmark.BenchmarkRefused):
        hidden_benchmark.run_formal(fixture["freeze_dir"], output_dir=tmp_path / "runs", evidence_custody_dir=tmp_path / "custody")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_193_slot_coordinator_preserves_failed_and_unrun_denominator(tmp_path, monkeypatch, interrupt):
    fixture = _materialized(tmp_path)
    prepared = _prepare(fixture)
    observed = []
    def execute(_prepared, attempt, *_args):
        observed.append(attempt.id)
        return {"attempt_id": attempt.id, "case_id": attempt.case_id, "load_mode": attempt.load_mode,
                "status": "interrupted" if interrupt else "error", "error_code": "local_protocol_fixture",
                "independent_acceptance": "not_established"}
    monkeypatch.setattr(runner, "_execute_attempt", execute)
    report = runner.execute_prepared(prepared, binding_path=fixture["binding_path"], freeze_dir=fixture["freeze_dir"],
                                     output_dir=tmp_path / "runs", evidence_custody_dir=tmp_path / "custody")
    assert len(report["attempts"]) == report["denominator"] == 193
    assert [r["attempt_id"] for r in report["attempts"]] == [a.id for a in prepared.benchmark.attempts]
    assert len(observed) == (1 if interrupt else 193)
    assert report["acceptance_claim"] == "none" and report["execution_scope"] == "local_fixture/protocol_only"
    assert report["signature_status"] == "unsigned_candidate_originals"
    stored = json.loads(Path(report["report_path"]).read_bytes())
    assert stored == report and not report["provider_billing_guarantee"]
    assert all(not row.get("concurrency", {}).get("observed_overlap") for row in report["attempts"])


def test_concurrency_requires_distinct_owned_processes_and_actual_overlapping_journeys():
    row = {"attempt_id": "a", "load_mode": "concurrent", "status": "completed",
           "journey_interval": {"pid": 10, "workspace_sha256": "a", "start_ns": 100, "end_ns": 200}}
    other = deepcopy(row)
    other.update(attempt_id="b")
    other["journey_interval"].update(pid=20, workspace_sha256="b", start_ns=199, end_ns=250)
    runner._bind_concurrency([row, other])
    assert row["concurrency"]["peer_attempt_ids"] == ["b"]
    for changed in ({"pid": 10}, {"workspace_sha256": "a"}, {"start_ns": 200}):
        bad = deepcopy(other)
        bad["journey_interval"].update(changed)
        first = deepcopy(row)
        runner._bind_concurrency([first, bad])
        assert first["status"] == "load_unavailable"
        assert not first["concurrency"]["observed_overlap"]


def test_public_formal_entry_delegates_only_after_valid_materialization(tmp_path, monkeypatch):
    fixture = _materialized(tmp_path)
    seen = []
    def execute(prepared, **kwargs):
        seen.append((prepared, kwargs))
        return {"denominator": 193, "acceptance_claim": "none"}
    monkeypatch.setattr(runner, "execute_prepared", execute)
    result = hidden_benchmark.run_formal(fixture["freeze_dir"], binding_path=fixture["binding_path"],
        ledger_path=fixture["ledger_path"], profile_path=fixture["profile_path"],
        output_dir=tmp_path / "runs", evidence_custody_dir=tmp_path / "custody",
        secret_env={"MARVIS_HIDDEN_TEST_KEY": "local-fixture-placeholder"})
    assert result["denominator"] == 193 and len(seen) == 1
    assert seen[0][0].total_cost_bound == 19.3


@pytest.mark.parametrize("changed", ["approval", "prices", "ledger", "profile", "binding"])
def test_coordinator_rechecks_authority_bytes_before_retaining_or_execution(tmp_path, monkeypatch, changed):
    fixture = _materialized(tmp_path)
    prepared = _prepare(fixture)
    called = []
    monkeypatch.setattr(runner, "_execute_attempt", lambda *_args: called.append(True))
    name = {"approval": "provider-approval.json", "prices": "prices.json", "ledger": "ledger.json",
            "profile": "profile.json", "binding": "binding.json"}[changed]
    path = fixture["root"] / name
    if changed == "approval":
        raw = json.loads(path.read_bytes())
        raw["source_document"]["path"] = str(tmp_path / "private-sentinel.txt")
        (tmp_path / "private-sentinel.txt").write_text("PRIVATE_NOT_TO_BE_RETAINED")
        path.write_text(json.dumps(raw))
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash_mismatch|authority_changed"):
        runner.execute_prepared(prepared, binding_path=fixture["binding_path"], freeze_dir=fixture["freeze_dir"],
                                output_dir=tmp_path / "runs", evidence_custody_dir=tmp_path / "custody")
    assert not called and not (tmp_path / "runs").exists()
