from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvis.orchestrator.eval import hidden_benchmark as hb
from marvis.orchestrator.eval.perf_gate import DEFAULT_PROFILE

SALT = b"s" * 32


@pytest.fixture()
def frozen(tmp_path):
    hb.freeze_case_set(tmp_path, "f1", salt=SALT, commit="0" * 40, frozen_at="2026-11-02T00:00:00+00:00")
    return tmp_path / "f1"


def test_freeze_has_133_cases_and_193_slots(frozen):
    data = hb.load_frozen_case_set(frozen)
    assert len(data["cases"]) == 133 and len(data["attempts"]) == 193
    manifest = data["manifest"]
    assert manifest["cohorts"] == {"normal": 64, "boundary": 20, "recovery": 16, "extension": 30, "extension_safety": 3}
    assert manifest["critical_cases"] == 20
    assert set(manifest["pending_user_decisions"].values()) == {"pending"}
    assert {c["expected_result"] for c in data["cases"] if c["cohort"] == "boundary"} == {"clarification", "rejected"}
    assert (frozen / "salt.hex").stat().st_mode & 0o077 == 0


def test_frozen_set_matches_formal_benchmark_inventory(frozen):
    data = hb.load_frozen_case_set(frozen)
    # Test-only budgets/SLO prove composition compatibility; they are not decisions.
    budget = {"wall_seconds": 60, "max_llm_attempts": 1, "max_http_requests": 1,
              "max_output_tokens_per_attempt": 1, "max_total_tokens": 1, "max_cost": 1.0, "currency": "TEST"}
    benchmark = hb.to_formal_benchmark(
        data, benchmark_id="t", commit="0" * 40, source_inventory_sha256="1" * 64,
        model={"profile": "p", "model_name": "m", "connection_sha256": "2" * 64, "source": "real_model",
               "evaluation_mode": "blind"},
        environment={k: "3" * 64 for k in ("hardware_sha256", "load_sha256", "runtime_identity_sha256",
                                            "dependency_inventory_sha256")},
        budgets={c["id"]: budget for c in data["cases"]},
        slo={m: {"minimum_observations": 1, "p95_wall_seconds": 1.0}
             for m in ("cold_start", "steady", "concurrent", "recovery")},
    )
    assert len(benchmark.cases) == 133 and len(benchmark.attempts) == 193


def test_freeze_is_never_overwritten(frozen):
    with pytest.raises(FileExistsError):
        hb.freeze_case_set(frozen.parent, "f1", salt=SALT)


def test_variants_depend_on_held_salt(tmp_path):
    hb.freeze_case_set(tmp_path, "a", salt=SALT)
    hb.freeze_case_set(tmp_path, "b", salt=b"t" * 32)
    a = json.loads((tmp_path / "a/cases.json").read_text())
    b = json.loads((tmp_path / "b/cases.json").read_text())
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert all(x["case_sha256"] != y["case_sha256"] for x, y in zip(a, b))


def test_tampered_case_set_is_rejected(frozen):
    path = frozen / "cases.json"
    cases = json.loads(path.read_text())
    cases[0]["expected_result"] = "clarification"
    path.chmod(0o600)
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="digest_mismatch"):
        hb.load_frozen_case_set(frozen)


def _ledger(tmp_path: Path, cost_cap):
    plan = {"agent_benchmark_plan": {} if cost_cap is None else {"cost_cap": cost_cap}}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.mark.parametrize("cap, blocker", [
    (None, "cost_cap_not_recorded"),
    ({"status": "pending_user_decision", "max_cost": "pending", "currency": "pending", "price_book": "pending"},
     "cost_cap_pending_user_decision"),
    ({"status": "decided", "max_cost": 10, "currency": "pending", "price_book": "x"}, "cost_cap_currency_pending"),
])
def test_formal_run_refuses_while_budget_pending(tmp_path, frozen, cap, blocker):
    with pytest.raises(hb.BenchmarkRefused) as refused:
        hb.run_formal(frozen, ledger_path=_ledger(tmp_path, cap), profile_path=DEFAULT_PROFILE)
    assert blocker in refused.value.blockers
    assert "performance_profile_pending_user_freeze" in refused.value.blockers


def test_repository_ledger_and_profile_block_formal_run(frozen):
    blockers = hb.formal_readiness(freeze_dir=frozen)
    assert any(b.startswith("cost_cap_") for b in blockers)
    assert "hidden_case_set_invalid" not in blockers


def test_dry_run_exercises_every_slot_through_formal_reader(tmp_path, frozen):
    report = hb.dry_run(frozen, tmp_path / "out")
    assert report["mode"] == "local_dry_run_stub_llm" and report["acceptance_claim"] == "none"
    assert report["label"] == hb.DRY_RUN_LABEL
    slots = report["harness_chain_completed"]["all_slots"]
    expected = sum(row["load_mode"] not in {"steady", "concurrent"} for row in report["attempts"])
    assert slots["denominator"] == 193 and slots["successes"] == expected
    assert all(row["telemetry_unavailable_reason"] in {"steady_warmup_originals_unavailable", "concurrent_originals_unavailable"}
               for row in report["attempts"] if row["load_mode"] in {"steady", "concurrent"})
    assert report["formal_blockers"]
    assert (tmp_path / "out/dry-run-report.json").exists()


def test_formal_cli_never_falls_through_to_dry_run(frozen, monkeypatch):
    def no_dry_run(*_args, **_kwargs):
        raise AssertionError("formal request reached dry_run")

    monkeypatch.setattr(hb, "dry_run", no_dry_run)
    assert hb.main(["formal", "--freeze-dir", str(frozen)]) == 2
    # Even if run_formal returned without raising, the formal branch exits 2.
    monkeypatch.setattr(hb, "run_formal", lambda *_a, **_k: None)
    assert hb.main(["formal", "--freeze-dir", str(frozen)]) == 2
