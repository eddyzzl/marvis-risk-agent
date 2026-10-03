"""Exercise the independent-review entry point through its actual CLI carrier."""

import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

from marvis.orchestrator.eval.runtime_contracts import digest
from test_runtime_archive_validation import actual_archive as retained_archive_fixture


actual_archive = retained_archive_fixture


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/revalidate_runtime_archive.py"


def _binding_file(tmp_path, binding):
    path = tmp_path / "independently-held-binding.json"
    path.write_text(binding.model_dump_json(), encoding="utf-8")
    return path, digest(path.read_bytes())


def _invoke(archive, manifest_digest, binding_path, binding_digest, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(archive), "--manifest-sha256", manifest_digest,
         "--binding", str(binding_path), "--binding-sha256", binding_digest, *map(str, extra)],
        capture_output=True, text=True, timeout=90, check=False,
    )


def test_real_cli_recomputes_retained_archive_without_claiming_final_runtime_or_acceptance(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    output = tmp_path / "review.json"
    result = _invoke(archive, manifest_digest, path, binding_digest, "--output", output)
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert json.loads(output.read_text()) == payload
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert payload["supported_checks_verified"]
    assert payload["checks"]["pmml_rescoring"]["rows_recomputed"] == 180
    assert payload["requested_scope"] == "domain_only"
    assert payload["final_runtime_evidence"]["status"] == "unavailable"
    assert payload["source_authentication"] == payload["acceptance_claim"] == "not_established"


@pytest.mark.parametrize("wrong", ["manifest", "binding"])
def test_wrong_external_digest_never_uses_archived_self_attestation(actual_archive, tmp_path, wrong):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    result = _invoke(archive, "f" * 64 if wrong == "manifest" else manifest_digest,
                     path, "f" * 64 if wrong == "binding" else binding_digest)
    assert result.returncode == (1 if wrong == "manifest" else 2)
    payload = json.loads(result.stdout)
    assert payload["acceptance_claim"] == "not_established"
    if wrong == "manifest":
        assert payload["checks"]["archive_integrity"]["status"] == "mismatch"
        assert payload["checks"]["pmml_rescoring"]["status"] == "unverified"


def test_absent_confirmation_cannot_be_promoted_by_domain_recomputation(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    binding = binding.model_copy(update={"confirmed_draft_sha256": None, "report_revision": None,
                                         "report_sha256": {"word": None, "excel": None}})
    path, binding_digest = _binding_file(tmp_path, binding)
    result = _invoke(archive, manifest_digest, path, binding_digest)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["checks"]["metrics_recomputation"]["status"] == "verified"
    assert payload["checks"]["confirmation_record_binding"]["status"] == "unsupported"


@pytest.mark.parametrize("target_kind", ["existing", "archive", "binding", "symlink"])
def test_output_never_overwrites_or_mutates_input_evidence(actual_archive, tmp_path, target_kind):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    existing = tmp_path / "keep.json"
    existing.write_text("keep", encoding="utf-8")
    output = {"existing": existing, "archive": archive / "injected.json", "binding": path,
              "symlink": tmp_path / "link.json"}[target_kind]
    if target_kind == "symlink":
        output.symlink_to(existing)
    before_binding = path.read_bytes()
    result = _invoke(archive, manifest_digest, path, binding_digest, "--output", output)
    assert result.returncode == 2
    assert existing.read_text() == "keep"
    assert path.read_bytes() == before_binding
    assert not (archive / "injected.json").exists()


def test_malformed_binding_does_not_echo_private_contents(tmp_path):
    path = tmp_path / "binding.json"
    path.write_text('{"private_field":"sensitive-business-material"}', encoding="utf-8")
    result = _invoke(tmp_path / "no-archive", "a" * 64, path, digest(path.read_bytes()))
    assert result.returncode == 2
    assert "sensitive-business-material" not in result.stdout + result.stderr


def test_missing_explicit_binding_is_a_usage_error(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "--manifest-sha256", "a" * 64],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 2
    assert "--binding" in result.stderr


def _run_records(archive, binding):
    run_dir = archive.parents[2] / "public" / binding.run_id
    return run_dir, digest((run_dir / "final-manifest.json").read_bytes())


def test_real_cli_binds_domain_review_to_original_final_run_records(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    run_dir, run_digest = _run_records(archive, binding)
    result = _invoke(archive, manifest_digest, path, binding_digest,
                     "--run-dir", run_dir, "--run-manifest-sha256", run_digest)
    assert result.returncode == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["requested_scope"] == "selected_case_domain_and_final_run_records"
    assert payload["final_runtime_evidence"]["status"] == "bound"
    assert payload["final_runtime_evidence"]["recorded_case_passed"]
    assert payload["final_runtime_evidence"]["independent_runtime_authentication"] == "not_established"
    assert payload["acceptance_claim"] == "not_established"


@pytest.mark.parametrize("change", ["case", "source", "input_contract", "confirmation", "report", "custody"])
def test_final_run_cannot_bind_to_another_external_case_or_domain_result(actual_archive, tmp_path, change):
    archive, manifest_digest, binding = actual_archive
    run_dir, run_digest = _run_records(archive, binding)
    updates = {
        "case": {"case": binding.case.model_copy(update={"revision": "different-frozen-revision"})},
        "source": {"source": {**binding.source, "source_sha256": "f" * 64}},
        "input_contract": {"input_contract_sha256": "f" * 64},
        "confirmation": {"confirmed_draft_sha256": "f" * 64},
        "report": {"report_sha256": {**binding.report_sha256, "word": "f" * 64}},
        "custody": {},
    }[change]
    path, binding_digest = _binding_file(tmp_path, binding.model_copy(update=updates))
    result = _invoke(archive, "f" * 64 if change == "custody" else manifest_digest,
                     path, binding_digest, "--run-dir", run_dir, "--run-manifest-sha256", run_digest)
    assert result.returncode == 2, result.stdout
    assert json.loads(result.stdout)["acceptance_claim"] == "not_established"


def test_old_run_without_final_manifest_stays_explicitly_unavailable(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    run_dir, run_digest = _run_records(archive, binding)
    old = tmp_path / "old-run"
    shutil.copytree(run_dir, old)
    (old / "final-manifest.json").unlink()
    result = _invoke(archive, manifest_digest, path, binding_digest,
                     "--run-dir", old, "--run-manifest-sha256", run_digest)
    assert result.returncode == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["supported_checks_verified"]
    assert not payload["requested_checks_verified"]
    assert payload["final_runtime_evidence"] == {"status": "unavailable", "reason": "original_final_run_manifest_missing"}
    assert not (old / "final-manifest.json").exists()


def test_partial_final_run_arguments_cannot_silently_select_domain_only(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    path, binding_digest = _binding_file(tmp_path, binding)
    result = _invoke(archive, manifest_digest, path, binding_digest, "--run-dir", tmp_path)
    assert result.returncode == 2
    assert "must be supplied together" in result.stderr


def test_valid_binding_extensions_are_not_echoed_on_a_normal_mismatch_result(actual_archive, tmp_path):
    archive, manifest_digest, binding = actual_archive
    source = {**binding.source, "api_key": "private-accidental-credential",
              "institution": {"customer": "private-customer-information"},
              "commit": "private-malformed-commit-value"}
    path, binding_digest = _binding_file(tmp_path, binding.model_copy(update={"source": source}))
    output = tmp_path / "sanitized-result.json"
    result = _invoke(archive, manifest_digest, path, binding_digest, "--output", output)
    assert result.returncode == 1
    serialized = result.stdout + result.stderr + output.read_text()
    assert not any(value in serialized for value in
                   ("private-accidental-credential", "private-customer-information", "private-malformed-commit-value"))
    payload = json.loads(result.stdout)
    assert payload["checks"]["external_bindings"]["status"] == "mismatch"
    assert payload["original_source_binding_sha256"] == digest(source)
    assert "commit" not in payload["original_source_binding"]


def test_completed_domain_cannot_promote_an_original_scoring_interrupt(tmp_path, monkeypatch):
    from marvis.orchestrator.eval import runtime_scoring
    from marvis.orchestrator.eval.runtime_archive_validation import FrozenValidationBinding
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_contracts import RuntimeCase
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_validation_agent import _v2_protocol

    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    case = RuntimeCase.model_validate(json.loads(paths["cases"].read_text())["cases"][0])

    def interrupt_scoring(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_scoring, "score_case", interrupt_scoring)
    with fixture_model(answer_factory=_v2_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "public",
            evidence_custody_dir=tmp_path / "custody", model=model, model_source="fixture_model")
    observed = report["cases"][0]
    assert not report["all_passed"]
    assert observed["runtime_status"] == "completed"
    assert observed["score"]["scorer_error"] == "KeyboardInterrupt"
    pipeline = observed["execution"]["validation_pipeline"]
    binding = FrozenValidationBinding(case=case, run_id=report["run_id"], task_id=observed["execution"]["task_id"],
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], input_contract_sha256=pipeline["input_contract_sha256"],
        confirmed_draft_sha256=pipeline["confirmed_draft_sha256"], report_revision=pipeline["report_revision"],
        report_sha256={item["kind"]: item["sha256"] for item in pipeline["report_files"]}, bin_count=10)
    archive = tmp_path / "custody" / report["run_id"] / case.id
    path, binding_digest = _binding_file(tmp_path, binding)
    result = _invoke(archive, observed["evidence_custody"]["manifest_sha256"], path, binding_digest,
        "--run-dir", tmp_path / "public" / report["run_id"],
        "--run-manifest-sha256", report["final_manifest_sha256"])
    assert result.returncode == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["supported_checks_verified"]
    assert not payload["requested_checks_verified"]
    assert payload["final_runtime_evidence"]["recorded_case_failure"]
    assert not payload["final_runtime_evidence"]["recorded_run_passed"]
    assert payload["acceptance_claim"] == "not_established"
