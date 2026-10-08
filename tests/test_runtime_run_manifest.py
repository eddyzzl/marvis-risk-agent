"""Final carrier binding is distinct from scoring truth and business acceptance."""
import json
from pathlib import Path
import shutil

import pytest

from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import ModelConnection, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_run_manifest import (
    FINAL_MANIFEST_NAME, RunManifestError, RunManifestUnavailable,
    _finalize_run_manifest, verify_run_manifest,
)
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_validation_agent import _v2_protocol


@pytest.fixture(scope="module")
def actual_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-v2")
    paths = write_synthetic_suite(root / "suite", normal_validation_agent_only=True)
    case = RuntimeCase.model_validate(json.loads(paths["cases"].read_text())["cases"][0])
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "test-only", "currency": "TEST",
        "source": "public synthetic rate table", "effective_date": "2026-10-03",
        "prompt_per_million": 2.0, "completion_per_million": 5.0}))
    baseline = root / "baseline.json"
    baseline.write_text(json.dumps({"schema_version": 1, "execution_mode": "real_http_validation_native_entries",
        "model_source": "fixture_model", "cases_sha256": digest(paths["cases"].read_bytes()),
        "expected_sha256": digest(paths["expected"].read_bytes()), "case_ids": [case.id], "denominator": 1,
        "cases": [{"case_id": case.id, "case_sha256": digest(case.model_dump()), "score": {"passed": False}}],
        "private_fixture_marker": "BASELINE_ORIGINAL_NEVER_COPIED_8293"}))
    with fixture_model(answer_factory=_v2_protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            price_book_path=price, baseline_path=baseline,
            model=model, model_source="fixture_model", secret_env={"UNUSED_SECRET": "PRIVATE_SECRET_NEVER_COPIED_3241"})
    assert calls and report["all_passed"] and report["regression_gate_passed"]
    assert report["cases"][0]["isolated_workspace_removed"]
    return Path(report["report_path"]).parent, report, {"cases": paths["cases"], "expected": paths["expected"],
                                                       "price_book": price, "baseline": baseline}


def _copy(actual_run, tmp_path):
    original, report, _ = actual_run
    target = tmp_path / "run"
    shutil.copytree(original, target)
    return target, report["final_manifest_sha256"], report["case_ids"][0]


def _rewrite_inventory(root, *, update_case_hashes=True):
    path = root / FINAL_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    for relative in list(manifest["files"]):
        raw = (root / relative).read_bytes()
        manifest["files"][relative] = {"sha256": digest(raw), "size_bytes": len(raw)}
    if update_case_hashes:
        for case in manifest["cases"]:
            for key, filename in (("execution_sha256", "execution.json"), ("score_sha256", "score.json"),
                                  ("attempts_sha256", "llm-attempts.jsonl")):
                entry = manifest["files"].get(f"{case['case_id']}/{filename}")
                case[key] = entry["sha256"] if entry else None
    path.write_text(json.dumps(manifest))
    return digest(path.read_bytes())


def test_actual_final_records_are_bound_after_custody_and_scoring(actual_run):
    root, report, frozen_paths = actual_run
    final_path = root / FINAL_MANIFEST_NAME
    expected_digest = report["final_manifest_sha256"]
    before = {p.relative_to(root): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
    verified = verify_run_manifest(root, expected_manifest_sha256=expected_digest)
    manifest, retained = verified["manifest"], verified["report"]
    assert retained == {k: v for k, v in report.items() if k not in {"final_manifest_path", "final_manifest_sha256"}}
    assert "final_manifest_sha256" not in retained and "final_manifest_path" not in retained
    assert FINAL_MANIFEST_NAME not in manifest["files"]
    assert manifest["files"]["report.json"]["sha256"] == digest((root / "report.json").read_bytes())
    assert manifest["files"]["manifest.json"]["sha256"] == digest((root / "manifest.json").read_bytes())
    assert manifest["case_ids"] == retained["case_ids"]
    assert manifest["denominator"] == len(retained["cases"]) == 1
    for key, path in frozen_paths.items():
        assert manifest["frozen"][key + "_sha256"] == digest(path.read_bytes())
    case = retained["cases"][0]
    entry = manifest["cases"][0]
    assert entry["custody_status"] == "retained"
    assert entry["custody_manifest_sha256"] == case["evidence_custody"]["manifest_sha256"]
    execution = verified["executions"][case["case_id"]]
    assert all(key in execution for key in ("duration_ms", "runtime_entry", "budget", "llm_events", "evidence_custody"))
    assert verified["scores"][case["case_id"]] == case["score"]
    assert verified["attempts"][case["case_id"]] == execution["llm_events"]
    assert verified["record_consistency"] == "verified_not_truth"
    assert verified["source_authentication"] == verified["acceptance_claim"] == "not_established"
    assert verified["custody_archives_reverified"] is False
    assert "BASELINE_ORIGINAL_NEVER_COPIED_8293" not in final_path.read_text()
    assert "PRIVATE_SECRET_NEVER_COPIED_3241" not in final_path.read_text()
    assert before == {p.relative_to(root): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()}


def test_rehashed_summary_cannot_change_original_measured_time(actual_run, tmp_path):
    root, _, case_id = _copy(actual_run, tmp_path)
    execution_path = root / case_id / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    assert execution["process_observation"]["llm_transport"]["complete"]
    execution["process_observation"]["llm_transport"]["known_busy_duration_ns"] += 1
    execution_path.write_text(json.dumps(execution))
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    report["cases"][0].update(execution)
    report["cases"][0]["execution_sha256"] = digest(execution_path.read_bytes())
    report_path.write_text(json.dumps(report))
    new_digest = _rewrite_inventory(root)
    with pytest.raises(RunManifestError, match="invalid_original_process_observation"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


@pytest.mark.parametrize("change", ["summary", "snapshot"])
def test_rehashed_revision_claim_cannot_change_native_observation(actual_run, tmp_path, change):
    root, _, case_id = _copy(actual_run, tmp_path)
    execution_path = root / case_id / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    assert execution["process_observation"]["revisions"]["complete"]
    if change == "summary":
        execution["process_observation"]["revisions"]["known_counts"]["structural_replan"] += 1
    else:
        execution["execution"]["plans"].append({"id": "not-observed", "replan_count": 0})
    execution_path.write_text(json.dumps(execution))
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    report["cases"][0].update(execution)
    report["cases"][0]["execution_sha256"] = digest(execution_path.read_bytes())
    report_path.write_text(json.dumps(report))
    new_digest = _rewrite_inventory(root)
    with pytest.raises(RunManifestError, match="invalid_original_process_observation"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


@pytest.mark.parametrize("change", ["summary", "binding", "missing_end"])
def test_rehashed_confirmation_wait_claim_cannot_change_native_observation(actual_run, tmp_path, change):
    root, _, case_id = _copy(actual_run, tmp_path)
    execution_path = root / case_id / "execution.json"
    execution = json.loads(execution_path.read_bytes())
    waiting = execution["process_observation"]["backend"]["scopes"]["report_confirmation_wait"]
    assert waiting["complete"] and waiting["measured_intervals"] == 1
    if change == "summary":
        waiting["known_busy_duration_ns"] += 1
    else:
        endpoint = next(row for row in execution["backend_events"]
                        if row.get("scope") == "report_confirmation_wait" and row["event"] == "resumed")
        if change == "binding":
            endpoint["binding_sha256"] = "a" * 64
        else:
            execution["backend_events"].remove(endpoint)
    execution_path.write_text(json.dumps(execution))
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    report["cases"][0].update(execution)
    report["cases"][0]["execution_sha256"] = digest(execution_path.read_bytes())
    report_path.write_text(json.dumps(report))
    new_digest = _rewrite_inventory(root)
    with pytest.raises(RunManifestError, match="invalid_original_process_observation"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


@pytest.mark.parametrize("file", ["report.json", "manifest.json", "execution.json", "score.json", "llm-attempts.jsonl"])
def test_original_bytes_cannot_change_under_external_digest(actual_run, tmp_path, file):
    root, expected_digest, case_id = _copy(actual_run, tmp_path)
    path = root / file if file in {"report.json", "manifest.json"} else root / case_id / file
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RunManifestError):
        verify_run_manifest(root, expected_manifest_sha256=expected_digest)
    _rewrite_inventory(root)
    with pytest.raises(RunManifestError, match="final_manifest_digest_mismatch"):
        verify_run_manifest(root, expected_manifest_sha256=expected_digest)


@pytest.mark.parametrize("file", ["execution.json", "score.json", "llm-attempts.jsonl"])
def test_redone_inventory_cannot_hide_inconsistent_final_records(actual_run, tmp_path, file):
    root, _, case_id = _copy(actual_run, tmp_path)
    path = root / case_id / file
    if file == "llm-attempts.jsonl":
        path.write_bytes(path.read_bytes() + b'{"event":"transport_rejected"}\n')
    else:
        value = json.loads(path.read_text())
        value["duration_ms" if file == "execution.json" else "passed"] = 9999 if file == "execution.json" else False
        path.write_text(json.dumps(value))
    new_digest = _rewrite_inventory(root)
    with pytest.raises(RunManifestError, match="inconsistent_final_record"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


@pytest.mark.parametrize("damage", ["deleted_case", "removed_report_case", "extra_file", "extra_directory", "symlink", "hardlink"])
def test_complete_denominator_and_exact_safe_file_set_are_required(actual_run, tmp_path, damage):
    root, expected_digest, case_id = _copy(actual_run, tmp_path)
    if damage == "deleted_case":
        shutil.rmtree(root / case_id)
    elif damage == "removed_report_case":
        path = root / "report.json"
        report = json.loads(path.read_text())
        report["cases"] = []
        report["case_ids"] = []
        report["denominator"] = 0
        path.write_text(json.dumps(report))
        expected_digest = _rewrite_inventory(root)
    elif damage == "extra_file":
        path = root / case_id / "unbound-extra.json"
        path.touch(mode=0o600)
    elif damage == "extra_directory":
        (root / "unbound-extra").mkdir(mode=0o700)
    else:
        path = root / case_id / "execution.json"
        original = tmp_path / "outside.json"
        original.write_bytes(path.read_bytes())
        original.chmod(0o600)
        path.unlink()
        path.symlink_to(original) if damage == "symlink" else path.hardlink_to(original)
    with pytest.raises(RunManifestError):
        verify_run_manifest(root, expected_manifest_sha256=expected_digest)


@pytest.mark.parametrize("field", ["source", "cases_sha256", "expected_sha256", "price_book_sha256", "baseline_sha256"])
def test_manifest_frozen_bindings_cannot_contradict_original_records(actual_run, tmp_path, field):
    root, _, _ = _copy(actual_run, tmp_path)
    path = root / FINAL_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["frozen"][field] = {"source_sha256": "a" * 64} if field == "source" else "a" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(RunManifestError):
        verify_run_manifest(root, expected_manifest_sha256=digest(path.read_bytes()))


def test_missing_legacy_manifest_is_explicitly_unavailable_and_not_created(actual_run, tmp_path):
    root, expected_digest, _ = _copy(actual_run, tmp_path)
    (root / FINAL_MANIFEST_NAME).unlink()
    with pytest.raises(RunManifestUnavailable, match="unavailable_for_legacy_run"):
        verify_run_manifest(root, expected_manifest_sha256=expected_digest)
    assert not (root / FINAL_MANIFEST_NAME).exists()


def test_finalize_rejects_collision_and_changed_initial_manifest(actual_run, tmp_path):
    root, report, _ = actual_run
    kwargs = {"initial_manifest_sha256": digest((root / "manifest.json").read_bytes()),
              "price_book_sha256": None, "baseline_sha256": None}
    with pytest.raises(RunManifestError, match="already_exists"):
        _finalize_run_manifest(root, **kwargs)
    copy, _, _ = _copy(actual_run, tmp_path)
    (copy / FINAL_MANIFEST_NAME).unlink()
    (copy / "manifest.json").write_text('{}')
    with pytest.raises(RunManifestError, match="initial_manifest_changed"):
        _finalize_run_manifest(copy, **kwargs)
    assert not (copy / FINAL_MANIFEST_NAME).exists()


def test_verifier_returns_authenticated_parsed_values_instead_of_paths(actual_run, tmp_path):
    root, expected_digest, case_id = _copy(actual_run, tmp_path)
    checked = verify_run_manifest(root, expected_manifest_sha256=expected_digest)
    (root / case_id / "execution.json").write_text('{}')
    assert checked["executions"][case_id]["llm_events"]
    assert checked["report"]["cases"][0]["case_id"] == case_id
    with pytest.raises(RunManifestError):
        verify_run_manifest(root, expected_manifest_sha256=expected_digest)


@pytest.mark.parametrize("interruption", ["journey", "scorer"])
def test_interrupt_keeps_current_and_unstarted_cases_in_final_denominator(tmp_path, monkeypatch, interruption):
    from marvis.orchestrator.eval import runtime_runner, runtime_scoring

    paths = write_synthetic_suite(tmp_path / "suite")
    cases = json.loads(paths["cases"].read_text())["cases"]
    if interruption == "journey":
        def interrupt(*_):
            raise KeyboardInterrupt
        monkeypatch.setattr(runtime_runner.Journey, "run", interrupt)
    else:
        def finished(case, *args, **kwargs):
            return {"case_id": case.id, "family": case.family, "task_type": case.task.task_type,
                    "case_set": case.case_set, "scenario": case.scenario, "runtime_status": "completed",
                    "execution": {"plans": [], "steps": []}, "llm_events": [], "http_events": []}, {}
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt
        monkeypatch.setattr(runtime_runner, "_run_case", finished)
        monkeypatch.setattr(runtime_scoring, "score_case", interrupt)
    with fixture_model() as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs", model=model, model_source="fixture_model")
    checked = verify_run_manifest(Path(report["report_path"]).parent, expected_manifest_sha256=report["final_manifest_sha256"])
    assert checked["manifest"]["case_ids"] == [c["id"] for c in cases]
    assert checked["manifest"]["denominator"] == len(cases)
    assert all(c["runtime_status"] == "not_run_after_interrupt" for c in report["cases"][1:])
    assert not report["all_passed"]
    assert report["summary"]["passed"] == 0
    if interruption == "scorer":
        assert report["cases"][0]["score"]["scorer_error"] == "KeyboardInterrupt"


def test_infrastructure_failures_have_final_bindings_without_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("MANIFEST_TEST_NONEXISTENT_CREDENTIAL", raising=False)
    paths = write_synthetic_suite(tmp_path / "suite")
    # No network provider can be contacted: explicit credential resolution fails
    # before application startup. Both failures still receive final records.
    report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
        dataset_root=paths["dataset_root"], output_dir=tmp_path / "runs", model_source="real_model",
        model=ModelConnection(model_id="missing", model_name="missing", api_base_url="http://127.0.0.1:1/v1",
                              api_key_env="MANIFEST_TEST_NONEXISTENT_CREDENTIAL"))
    checked = verify_run_manifest(Path(report["report_path"]).parent, expected_manifest_sha256=report["final_manifest_sha256"])
    assert checked["manifest"]["denominator"] == report["denominator"]
    assert not report["all_passed"]
    assert all(c["runtime_status"] == "error" for c in report["cases"])
    assert all(c["custody_status"] == "not_recorded" for c in checked["manifest"]["cases"])
    assert all(not any(e.get("event") == "started" for e in events) for events in checked["attempts"].values())
    assert all(c["attempts_sha256"] is not None for c in checked["manifest"]["cases"])


def _rewrite_consistent_case(root, case_id, *, mutate):
    from marvis.orchestrator.eval.runtime_scoring import summarize, usage_summary

    execution_path = root / case_id / "execution.json"
    score_path = root / case_id / "score.json"
    attempts_path = root / case_id / "llm-attempts.jsonl"
    execution = json.loads(execution_path.read_text())
    score = json.loads(score_path.read_text())
    events = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    mutate(execution, score, events)
    execution["llm_events"] = events
    # Keep every duplicated count/hash honest so negative tests exercise the
    # contradiction under test, not an earlier accidental inventory mismatch.
    if score.pop("_refresh_usage_from_events", False):
        score["usage"].update({k: v for k, v in usage_summary(events, model_name="", price_bytes=None).items()
                               if k not in {"cost", "cost_status"}})
    attempts_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    execution_path.write_text(json.dumps(execution))
    score_path.write_text(json.dumps(score))
    report_path = root / "report.json"
    report = json.loads(report_path.read_text())
    report["cases"] = [{**execution, "score": score, "execution_sha256": digest(execution_path.read_bytes())}
                       if item["case_id"] == case_id else item for item in report["cases"]]
    report.update(summarize(report["cases"]))
    report["regression_gate_passed"] = report["all_passed"] and report.get("regression_ok", True)
    report_path.write_text(json.dumps(report))
    manifest_path = root / FINAL_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["cases"]:
        if item["case_id"] == case_id:
            item["runtime_status"] = execution["runtime_status"]
    manifest_path.write_text(json.dumps(manifest))
    return _rewrite_inventory(root)


@pytest.mark.parametrize("contradiction", ["unknown_status", "budget_pass", "interrupted_pass", "eligible_without_pass", "fixture_eligible"])
def test_repeated_consistent_flags_do_not_override_necessary_failure_conditions(actual_run, tmp_path, contradiction):
    root, _, case_id = _copy(actual_run, tmp_path)

    def mutate(execution, score, events):
        if contradiction in {"unknown_status", "budget_pass", "interrupted_pass"}:
            execution["runtime_status"] = {"unknown_status": "success-by-assertion", "budget_pass": "budget_exceeded", "interrupted_pass": "interrupted"}[contradiction]
        else:
            score["a_evidence_eligible"] = True
            if contradiction == "eligible_without_pass":
                score["passed"] = False

    with pytest.raises(RunManifestError, match="case_identity_mismatch|passing_score_contradicts|eligibility_flag_contradicts"):
        verify_run_manifest(root, expected_manifest_sha256=_rewrite_consistent_case(root, case_id, mutate=mutate))


@pytest.mark.parametrize("event", [
    {"event": "budget_blocked"},
    {"event": "finished", "attempt_id": "extra", "output_limit_exceeded": True},
    {"event": "aggregate_budget_snapshot", "aggregate_budget_violation": True},
    {"event": "finished", "attempt_id": "extra", "error_type": "RuntimeBudgetExceeded"},
    {"event": "measurement_closed", "reason": "wall_deadline"},
    {"event": "transport_rejected"},
])
def test_transport_failure_cannot_be_recorded_as_completed_even_with_self_consistent_hashes(actual_run, tmp_path, event):
    root, _, case_id = _copy(actual_run, tmp_path)

    def mutate(execution, score, events):
        events.append(event)
        score["_refresh_usage_from_events"] = True

    new_digest = _rewrite_consistent_case(root, case_id, mutate=mutate)
    with pytest.raises(RunManifestError, match="transport_failure_contradicts_final_status"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


@pytest.mark.parametrize("field", ["transport_attempts", "logical_calls", "retry_attempts", "prompt_tokens", "completion_tokens", "known_prompt_tokens_subtotal", "incomplete_attempts", "token_budget_status", "cost_budget_status"])
def test_usage_summary_is_recomputed_from_attempt_bytes(actual_run, tmp_path, field):
    root, _, case_id = _copy(actual_run, tmp_path)

    def mutate(execution, score, events):
        score["usage"][field] = "incorrect-summary" if field.endswith("status") else 999999

    new_digest = _rewrite_consistent_case(root, case_id, mutate=mutate)
    with pytest.raises(RunManifestError, match="inconsistent_final_record"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)


def test_final_digest_and_inventory_limits_are_required_before_original_reads(actual_run, tmp_path, monkeypatch):
    import marvis.orchestrator.eval.runtime_run_manifest as module

    root, _, _ = _copy(actual_run, tmp_path)
    with pytest.raises(TypeError):
        verify_run_manifest(root)
    with pytest.raises(RunManifestError, match="independently_held"):
        verify_run_manifest(root, expected_manifest_sha256="")
    manifest_path = root / FINAL_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["report.json"]["size_bytes"] = module.MAX_TOTAL_BYTES + 1
    manifest_path.write_text(json.dumps(manifest))
    reads = []
    original_read = module._read

    def capture(path, **kwargs):
        reads.append(path.name)
        return original_read(path, **kwargs)

    monkeypatch.setattr(module, "_read", capture)
    with pytest.raises(RunManifestError, match="invalid_run_inventory"):
        verify_run_manifest(root, expected_manifest_sha256=digest(manifest_path.read_bytes()))
    assert reads == [FINAL_MANIFEST_NAME]


@pytest.mark.parametrize("contradiction", ["terminal_check", "assertion", "empty_assertions", "active_job"])
def test_passing_flag_cannot_contradict_its_own_reported_checks(actual_run, tmp_path, contradiction):
    root, _, case_id = _copy(actual_run, tmp_path)

    def mutate(execution, score, events):
        if contradiction == "terminal_check":
            score["terminal_ok"] = False
        elif contradiction == "assertion":
            score["assertions"][0]["passed"] = False
        elif contradiction == "empty_assertions":
            score["assertions"] = []
        else:
            execution["execution"]["latest_job"]["status"] = "running"

    new_digest = _rewrite_consistent_case(root, case_id, mutate=mutate)
    with pytest.raises(RunManifestError, match="passing_score_contradicts_reported_checks"):
        verify_run_manifest(root, expected_manifest_sha256=new_digest)
