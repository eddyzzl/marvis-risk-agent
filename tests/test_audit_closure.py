from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from marvis.orchestrator.eval.acceptance import EVIDENCE_SCHEMA, check_ledger, load_acceptance_spec, validate_evidence


COMMIT = "a" * 40


@pytest.fixture
def bundle(tmp_path):
    def write(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    finding = {
        "id": "F01", "work_packages": ["WP01"], "criterion_version": "v1",
        "source_paths": ["source.py"], "required_evidence_tiers": ["C"],
        "status": "ready_for_review", "evidence": [],
    }
    record = {
        "schema_version": EVIDENCE_SCHEMA, "run_id": "run-1", "finding_ids": ["F01"],
        "work_package": "WP01", "criterion_version": "v1", "commit": COMMIT,
        "evidence_tier": "C", "environment_kind": "test_process",
        "source_kind": "synthetic_fixture", "target_id": "local-test-suite",
        "source_hashes": {"source.py": write("source.py", "source under test")},
        "input_hashes": {"input.json": write("input.json", "{}")},
        "dataset_hashes": {}, "real_executor": None, "tool_receipts": [],
        "artifact_hashes": {
            "test.log": write("test.log", "raw test result"),
            "review.json": write("review.json", '{"reviewer_id":"independent-reviewer"}'),
        },
        "evidence_paths": ["test.log"],
        "independent_review": {
            "reviewer_id": "independent-reviewer", "artifact_path": "review.json",
        },
    }
    return record, finding, tmp_path


def inspect(bundle, **kwargs):
    record, finding, root = bundle
    return validate_evidence(
        record, finding, source_root=root, evidence_root=root, expected_commit=COMMIT, **kwargs,
    )


def acceptance_spec(*findings):
    return {"schema_version": "marvis.audit_requirements.v1", "findings": deepcopy(list(findings))}


def test_intact_self_report_cannot_authenticate_execution_or_review(bundle):
    record, _, _ = bundle
    record["passed"] = True
    record["independent_review"]["passed"] = True
    result = inspect(bundle)
    assert not result.integrity_errors
    assert not result.verified
    assert len(result.verification_errors) == 2


@pytest.mark.parametrize("mutation,expected", [
    (lambda r: r.update(commit="b" * 40), "commit binding"),
    (lambda r: r.update(criterion_version="v2"), "criterion version"),
    (lambda r: r.update(finding_ids=["F99"]), "finding binding"),
    (lambda r: r.update(source_hashes={}), "source snapshot"),
    (lambda r: r.update(input_hashes={}), "empty input_hashes"),
    (lambda r: r.update(evidence_paths=["missing.log"]), "unhashed"),
    (lambda r: r.update(independent_review=None), "independent review required"),
    (lambda r: r.update(evidence_tier=[]), "invalid evidence tier"),
    (lambda r: r["independent_review"].update(artifact_path=[]), "not hash-bound"),
])
def test_bad_bindings_never_reach_trusted_verifiers(bundle, mutation, expected):
    mutation(bundle[0])
    calls = []
    result = inspect(bundle, receipt_verifiers={"C": lambda *_: calls.append(True)})
    assert expected in " ".join(result.integrity_errors)
    assert not result.verified
    assert calls == []


def test_tampered_evidence_and_symlink_escape_are_rejected(bundle, tmp_path):
    record, _, root = bundle
    (root / "test.log").write_text("replacement", encoding="utf-8")
    assert "hash mismatch" in " ".join(inspect(bundle).integrity_errors)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "escape.log"
    link.symlink_to(outside)
    record["artifact_hashes"] = {"escape.log": hashlib.sha256(b"outside").hexdigest()}
    assert "escapes" in " ".join(inspect(bundle).integrity_errors)


@pytest.mark.parametrize("tier,environment,source,expected", [
    ("R", "test_process", "synthetic_fixture", "test process"),
    ("H", "local_reference", "synthetic_fixture", "historical acceptance"),
    ("P", "local_reference", "reference_runtime", "institution production"),
    ("P", "institution_test", "institution_runtime", "institution production"),
    ("A", "local_reference", "reference_runtime", "real model profile"),
])
def test_evidence_tiers_cannot_be_upgraded_by_labels(bundle, tier, environment, source, expected):
    record, _, _ = bundle
    record.update(
        evidence_tier=tier, environment_kind=environment, source_kind=source,
        real_executor={"id": "claimed-executor"}, tool_receipts=["test.log"],
    )
    result = inspect(bundle)
    assert expected in " ".join(result.integrity_errors)
    assert not result.verified


def test_trusted_adapter_errors_or_truthy_strings_cannot_pass(bundle):
    def broken(*_):
        raise RuntimeError("sensitive provider details must not be included")

    result = inspect(bundle, receipt_verifiers={"C": broken}, review_verifier=lambda *_: "true")
    assert not result.verified
    assert result.verification_errors == ("receipt verification failed", "independent review not verified")


def test_ledger_retains_open_findings_and_rejects_false_closure(bundle):
    record, finding, root = bundle
    raw = json.dumps(record).encode()
    (root / "record.json").write_bytes(raw)
    finding["evidence"] = [{"path": "record.json", "sha256": hashlib.sha256(raw).hexdigest()}]
    ledger = {"schema_version": 1, "overall_status": "in_progress", "findings": [finding]}
    original = deepcopy(ledger)
    kwargs = dict(source_root=root, evidence_root=root, expected_commit=COMMIT)
    report = check_ledger(ledger, **kwargs)
    assert report["valid"] and not report["complete"]
    assert report["findings"][0]["missing_verified_tiers"] == ["C"]
    assert ledger == original
    finding["status"] = "closed"
    assert not check_ledger(ledger, **kwargs)["valid"]
    # Only environment-owned adapters can attest a record. These test doubles
    # exercise the integration seam, and are never installed by the real CLI.
    verified = check_ledger(
        ledger, **kwargs, receipt_verifiers={"C": lambda *_: True}, review_verifier=lambda *_: True,
        trusted_spec=acceptance_spec(finding),
    )
    assert verified["valid"] and verified["complete"]
    finding["required_evidence_tiers"] = ["C", "R"]
    report = check_ledger(
        ledger, **kwargs, receipt_verifiers={"C": lambda *_: True}, review_verifier=lambda *_: True,
        trusted_spec=acceptance_spec(finding),
    )
    assert not report["valid"] and not report["complete"]
    assert report["findings"][0]["missing_verified_tiers"] == ["R"]


def test_cli_distinguishes_valid_open_ledger_from_completion(tmp_path, capsys):
    from scripts.check_audit_closure import main

    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({
        "schema_version": 1, "overall_status": "planned", "findings": [{
            "id": "F01", "required_evidence_tiers": ["C"], "status": "planned", "evidence": [],
            "criterion_version": "v1", "work_packages": ["WP01"], "source_paths": ["source.py"],
        }],
    }))
    args = [str(path), "--evidence-root", str(tmp_path)]
    assert main(args) == 0
    assert not json.loads(capsys.readouterr().out)["complete"]
    assert main([*args, "--require-complete"]) == 1
    assert not json.loads(capsys.readouterr().out)["complete"]


@pytest.mark.parametrize("change", [
    lambda f: f.update(required_evidence_tiers=["C"]),
    lambda f: f.update(criterion_version="relaxed-v2"),
    lambda f: f.update(source_paths=["different.py"]),
    lambda f: f.update(work_packages=["WP02"]),
])
def test_candidate_ledger_cannot_reduce_frozen_requirements(bundle, change):
    _, finding, root = bundle
    finding["required_evidence_tiers"] = ["C", "P"]
    frozen = acceptance_spec(finding)
    change(finding)
    report = check_ledger(
        {"schema_version": 1, "overall_status": "in_progress", "findings": [finding]},
        source_root=root, evidence_root=root, expected_commit=COMMIT, trusted_spec=frozen,
    )
    assert not report["valid"] and not report["complete"]
    assert any("frozen acceptance requirements mismatch" in message for message in report["errors"])


def test_missing_finding_cannot_shrink_the_frozen_scope(bundle):
    _, finding, root = bundle
    other = {**finding, "id": "F02"}
    report = check_ledger(
        {"schema_version": 1, "overall_status": "in_progress", "findings": [finding]},
        source_root=root, evidence_root=root, expected_commit=COMMIT,
        trusted_spec=acceptance_spec(finding, other),
    )
    assert "finding set differs from frozen acceptance scope" in report["errors"]


@pytest.mark.parametrize("key,value", [
    ("work_packages", None), ("source_paths", None), ("source_paths", [{}]),
    ("status", []), ("required_evidence_tiers", [None]),
])
def test_malformed_finding_returns_structured_rejection(bundle, key, value):
    _, finding, root = bundle
    finding[key] = value
    report = check_ledger(
        {"schema_version": 1, "overall_status": "in_progress", "findings": [finding]},
        source_root=root, evidence_root=root, expected_commit=COMMIT,
    )
    assert not report["valid"] and not report["complete"]
    assert report["errors"]


def test_malformed_overall_status_and_missing_trust_anchor_cannot_close(bundle):
    _, finding, root = bundle
    report = check_ledger(
        {"schema_version": 1, "overall_status": ["closed"], "findings": [finding]},
        source_root=root, evidence_root=root, expected_commit=COMMIT,
    )
    assert "invalid overall status" in report["errors"]
    finding["status"] = "closed"
    report = check_ledger(
        {"schema_version": 1, "overall_status": "closed", "findings": [finding]},
        source_root=root, evidence_root=root, expected_commit=COMMIT,
    )
    assert any("trusted acceptance spec unavailable" in message for message in report["errors"])


def test_acceptance_spec_is_bound_to_independently_supplied_digest(bundle):
    _, finding, root = bundle
    path = root / "requirements.json"
    original = json.dumps(acceptance_spec(finding)).encode()
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    assert load_acceptance_spec(path, expected_sha256=digest)["findings"][0]["id"] == "F01"
    path.write_text('{"findings": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_acceptance_spec(path, expected_sha256=digest)


def test_spec_swap_after_hashing_does_not_replace_verified_bytes(bundle, monkeypatch):
    from marvis.orchestrator.eval import acceptance

    _, finding, root = bundle
    finding["required_evidence_tiers"] = ["C", "P"]
    path = root / "requirements.json"
    raw = json.dumps(acceptance_spec(finding)).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    real_sha256 = hashlib.sha256

    def swap_after_hash(data):
        result = real_sha256(data)
        finding["required_evidence_tiers"] = ["C"]
        path.write_text(json.dumps(acceptance_spec(finding)), encoding="utf-8")
        return result

    monkeypatch.setattr(acceptance.hashlib, "sha256", swap_after_hash)
    loaded = load_acceptance_spec(path, expected_sha256=digest)
    assert loaded["findings"][0]["required_evidence_tiers"] == ["C", "P"]
