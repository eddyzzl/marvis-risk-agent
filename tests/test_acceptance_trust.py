"""Only externally pinned, independently signed observations can close a record."""

import base64
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance import check_ledger
from marvis.orchestrator.eval.acceptance_trust import (
    ATTESTATION_SCHEMA,
    SCHEMA,
    SOURCE_SCOPE,
    TrustInputError,
    load_trusted_environment,
    source_inventory,
    verify_junit,
)
from marvis.orchestrator.eval.runtime_contracts import digest
from test_runtime_archive_validation import actual_archive as actual_archive


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical(value)
    path.write_bytes(raw)
    return {"path": name, "sha256": digest(raw)}


def junit(*outcomes):
    return (
        '<testsuites><testsuite name="checks" tests="%d" failures="%d" errors="%d" skipped="%d">'
        % (
            len(outcomes),
            outcomes.count("failure"),
            outcomes.count("error"),
            outcomes.count("skipped"),
        )
        + "".join(
            f'<testcase classname="tests.test_scope" name="case_{i}" passed="true">'
            + (f"<{outcome}/>" if outcome else "")
            + "</testcase>"
            for i, outcome in enumerate(outcomes)
        )
        + "</testsuite></testsuites>"
    ).encode()


def invocation():
    # Test-only remote observations. No environment values or credentials.
    return {
        "entrypoint": "python_module:pytest",
        "cwd": "source_root",
        "argv_sha256": digest(["python", "-m", "pytest", "tests/test_scope.py"]),
        "runtime_identity_sha256": digest({"fixture_runtime": "cpython"}),
        "dependency_inventory_sha256": digest({"fixture_dependencies": ["pytest"]}),
        "environment_sha256": digest({"fixture_environment": "isolated"}),
        "isolation": "fresh_process_isolated_empty_caches",
    }


def sign_entries(bundle):
    """Temporary fixture keys only. Production code exposes no signer."""
    config, record = bundle.config, bundle.record
    inventory = digest(
        [
            {k: row[k] for k in ("path", "sha256", "adapter", "inputs")}
            for row in config["records"]
        ]
    )
    for entry in config["records"]:
        bound_record = json.loads((bundle.evidence / entry["path"]).read_bytes())
        for kind, key in bundle.keys.items():
            body = {
                "schema": ATTESTATION_SCHEMA,
                "kind": kind,
                "authority_id": kind,
                "record_sha256": entry["sha256"],
                "spec_sha256": config["spec_sha256"],
                "commit": config["commit"],
                "source_inventory_sha256": digest(config["source_inventory"]),
                "record_inventory_sha256": inventory,
                "adapter": entry["adapter"],
                "inputs_sha256": digest(entry["inputs"]),
                "run_id": bound_record["run_id"],
                "target_id": bound_record["target_id"],
                **{
                    name: bound_record[name]
                    for name in ("evidence_tier", "environment_kind", "source_kind")
                },
                "facts": deepcopy(
                    getattr(bundle, "per_record_facts", {})
                    .get(entry["path"], {})
                    .get(
                        kind, bundle.execution if kind == "execution" else bundle.review
                    )
                ),
            }
            signature = key.sign(ATTESTATION_SCHEMA.encode() + b"\0" + canonical(body))
            entry[kind] = put(
                bundle.trust,
                f"{entry['path']}.{kind}.json",
                {"body": body, "signature": base64.b64encode(signature).decode()},
            )
    bundle.config_ref = put(bundle.trust, "trust.json", config)
    bundle.record = record


@pytest.fixture
def bundle(tmp_path):
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    source, evidence, trust = [
        tmp_path / part for part in ("source", "evidence", "external")
    ]
    for root in (source, evidence, trust):
        root.mkdir()
    put(source, "marvis/app.py", b"value = 1\n")
    put(source, "scripts/accept.py", b"value = 2\n")
    put(source, "tests/test_app.py", b"assert True\n")
    put(source, "pyproject.toml", b"[project]\nname='fixture'\n")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    finding = {
        "id": "F01",
        "criterion_version": "v1",
        "work_packages": ["WP01"],
        "source_paths": ["marvis/app.py"],
        "required_evidence_tiers": ["C"],
        "status": "closed",
        "evidence": [],
    }
    spec = {
        "schema_version": "marvis.audit_requirements.v1",
        "findings": [deepcopy(finding)],
    }
    spec_ref = put(trust, "spec.json", spec)
    xml = put(evidence, "results.xml", junit("", ""))
    review_doc = put(
        evidence,
        "review.json",
        {"reviewed_findings": ["F01"], "notes": "Temporary fixture review"},
    )
    input_ref = put(evidence, "input.json", {"fixture": True})
    record = {
        "schema_version": "marvis.audit_evidence.v1",
        "run_id": "run-1",
        "target_id": "scoped-tests",
        "finding_ids": ["F01"],
        "work_package": "WP01",
        "criterion_version": "v1",
        "commit": commit,
        "evidence_tier": "C",
        "environment_kind": "test_process",
        "source_kind": "synthetic_fixture",
        "source_hashes": {
            "marvis/app.py": digest((source / "marvis/app.py").read_bytes())
        },
        "input_hashes": {input_ref["path"]: input_ref["sha256"]},
        "dataset_hashes": {},
        "artifact_hashes": {r["path"]: r["sha256"] for r in (xml, review_doc)},
        "evidence_paths": [xml["path"]],
        "tool_receipts": [],
        "real_executor": None,
        "independent_review": {
            "reviewer_id": "reviewer-person",
            "artifact_path": review_doc["path"],
        },
    }
    ref = put(evidence, "record.json", record)
    finding["evidence"] = [ref]
    keys = {role: ed.Ed25519PrivateKey.generate() for role in ("execution", "review")}
    authorities = {
        role: {
            "principal_id": "executor-person"
            if role == "execution"
            else "reviewer-person",
            "organization_id": "fixture-only",
            "role": "executor" if role == "execution" else "reviewer",
            "public_key": base64.b64encode(
                key.public_key().public_bytes_raw()
            ).decode(),
            "tiers": ["C"],
            "environments": ["test_process"],
            "sources": ["synthetic_fixture"],
        }
        for role, key in keys.items()
    }
    entry = {
        **ref,
        "adapter": "code_checks.v1",
        "inputs": {
            "checks": [
                {
                    "id": "pytest",
                    "junit": xml,
                    "test_ids": [
                        "tests.test_scope::case_0",
                        "tests.test_scope::case_1",
                    ],
                    "invocation": invocation(),
                }
            ]
        },
        "execution": {},
        "review": {},
    }
    config = {
        "schema": SCHEMA,
        "commit": commit,
        "spec_sha256": spec_ref["sha256"],
        "source_scope": SOURCE_SCOPE,
        "source_inventory": source_inventory(source),
        "authorities": authorities,
        "records": [entry],
    }
    value = SimpleNamespace(
        source=source,
        evidence=evidence,
        trust=trust,
        record=record,
        finding=finding,
        spec=spec,
        spec_ref=spec_ref,
        config=config,
        keys=keys,
        ledger={"schema_version": 1, "overall_status": "closed", "findings": [finding]},
        execution={
            "checks": {
                "pytest": {
                    "exit_code": 0,
                    "junit_sha256": xml["sha256"],
                    "invocation": invocation(),
                }
            }
        },
        review={
            "decision": "accepted",
            "review_artifact_sha256": review_doc["sha256"],
            "reviewed_findings": ["F01"],
            "coverage": ["frozen_criteria", "complete_run_denominator"],
            "unresolved_findings": [],
        },
    )
    sign_entries(value)
    return value


def environment(bundle, *, expected_digest=None):
    return load_trusted_environment(
        bundle.trust / "trust.json",
        expected_sha256=expected_digest or bundle.config_ref["sha256"],
        source_root=bundle.source,
        evidence_root=bundle.evidence,
        expected_commit=bundle.config["commit"],
        spec_sha256=bundle.spec_ref["sha256"],
    )


def check(bundle, env=None):
    return check_ledger(
        bundle.ledger,
        source_root=bundle.source,
        evidence_root=bundle.evidence,
        expected_commit=bundle.config["commit"],
        trusted_spec=bundle.spec,
        trusted_environment=env or environment(bundle),
    )


def update_record(bundle):
    ref = put(bundle.evidence, "record.json", bundle.record)
    bundle.finding["evidence"] = [ref]
    bundle.config["records"][0].update(ref)


def test_real_signature_and_complete_junit_close_only_the_fixture_c_tier(bundle):
    before = {p: p.read_bytes() for p in bundle.trust.rglob("*") if p.is_file()}
    result = check(bundle)
    assert result["valid"] and result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == facts["passed"] == 2
    assert before == {p: p.read_bytes() for p in bundle.trust.rglob("*") if p.is_file()}
    assert result["findings"][0]["missing_verified_tiers"] == []


def test_cli_wires_only_explicit_external_configuration(bundle, monkeypatch, capsys):
    from scripts import check_audit_closure as cli

    monkeypatch.setattr(cli, "REPO_ROOT", bundle.source)
    ledger = bundle.evidence / "ledger.json"
    ledger.write_bytes(canonical(bundle.ledger))
    args = [
        str(ledger),
        "--evidence-root",
        str(bundle.evidence),
        "--spec",
        str(bundle.trust / "spec.json"),
        "--spec-sha256",
        bundle.spec_ref["sha256"],
        "--require-complete",
    ]
    assert cli.main(args) == 1
    assert not json.loads(capsys.readouterr().out)["complete"]
    assert (
        cli.main(
            [
                *args,
                "--trust-config",
                str(bundle.trust / "trust.json"),
                "--trust-config-sha256",
                bundle.config_ref["sha256"],
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["complete"]


@pytest.mark.parametrize("outcome", ["failure", "error", "skipped"])
def test_signed_failure_cannot_be_erased_by_candidate_passed_flags(bundle, outcome):
    xml = put(bundle.evidence, "results.xml", junit("", outcome))
    bundle.record.update(passed=True, verified=True)
    bundle.record["artifact_hashes"][xml["path"]] = xml["sha256"]
    bundle.config["records"][0]["inputs"]["checks"][0]["junit"] = xml
    bundle.execution["checks"]["pytest"] = {
        "exit_code": 0 if outcome == "skipped" else 1,
        "junit_sha256": xml["sha256"],
        "invocation": invocation(),
    }
    update_record(bundle)
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == 2 and facts["passed"] == 1


@pytest.mark.parametrize(
    "attack", ["record", "result", "review", "signature", "config", "spec", "commit"]
)
def test_modified_bound_bytes_or_context_never_close(bundle, attack):
    if attack in {"record", "result", "review"}:
        name = {
            "record": "record.json",
            "result": "results.xml",
            "review": "review.json",
        }[attack]
        (bundle.evidence / name).write_bytes(
            (bundle.evidence / name).read_bytes() + b" "
        )
    elif attack == "signature":
        ref = bundle.config["records"][0]["execution"]
        envelope = json.loads((bundle.trust / ref["path"]).read_bytes())
        envelope["signature"] = base64.b64encode(bytes(64)).decode()
        ref.update(put(bundle.trust, ref["path"], envelope))
        bundle.config_ref = put(bundle.trust, "trust.json", bundle.config)
    elif attack == "config":
        (bundle.trust / "trust.json").write_bytes(canonical(bundle.config) + b" ")
    elif attack == "spec":
        bundle.spec_ref["sha256"] = "f" * 64
    else:
        bundle.config["commit"] = "f" * 40
    try:
        result = check(bundle)
    except TrustInputError:
        return
    assert not result["complete"]


@pytest.mark.parametrize("same", ["principal", "key"])
def test_renaming_the_executor_does_not_create_an_independent_reviewer(bundle, same):
    if same == "principal":
        bundle.config["authorities"]["review"]["principal_id"] = "executor-person"
        bundle.record["independent_review"]["reviewer_id"] = "executor-person"
        update_record(bundle)
    else:
        bundle.keys["review"] = bundle.keys["execution"]
        bundle.config["authorities"]["review"]["public_key"] = bundle.config[
            "authorities"
        ]["execution"]["public_key"]
    sign_entries(bundle)
    assert not check(bundle)["complete"]


@pytest.mark.parametrize(
    "target",
    [
        "scripts/accept.py",
        "tests/test_app.py",
        "pyproject.toml",
        "marvis/ignored/new.py",
        "conftest.py",
        "pytest.ini",
        "sitecustomize.py",
        "uv.lock",
        "extra_plugin/customize.py",
        ".hidden/pytest.ini",
        "orphan.pyc",
    ],
)
def test_all_current_code_is_bound_beyond_the_finding_source_subset(bundle, target):
    env = environment(bundle)
    put(bundle.source, target, b"modified = True\n")
    assert not check(bundle, env)["complete"]
    with pytest.raises(TrustInputError, match="current_source_inventory_mismatch"):
        environment(bundle)


@pytest.mark.parametrize(
    "failure", ["junit_failure", "exit_status", "missing_review", "wrong_invocation"]
)
def test_all_frozen_records_must_pass_for_c_tier(bundle, failure):
    second = deepcopy(bundle.record)
    second["run_id"] = "second-required-check-run"
    xml = put(
        bundle.evidence,
        "second.xml",
        junit("", "failure" if failure == "junit_failure" else ""),
    )
    second["artifact_hashes"][xml["path"]] = xml["sha256"]
    ref = put(bundle.evidence, "second.json", second)
    bundle.finding["evidence"].append(ref)
    entry = {**deepcopy(bundle.config["records"][0]), **ref}
    entry["inputs"]["checks"][0]["junit"] = xml
    bundle.config["records"].append(entry)
    observed = {
        "checks": {
            "pytest": {
                "exit_code": 1 if failure in {"junit_failure", "exit_status"} else 0,
                "junit_sha256": xml["sha256"],
                "invocation": invocation(),
            }
        }
    }
    review = deepcopy(bundle.review)
    if failure == "missing_review":
        review["coverage"] = []
    if failure == "wrong_invocation":
        observed["checks"]["pytest"]["invocation"]["argv_sha256"] = "f" * 64
    bundle.per_record_facts = {ref["path"]: {"execution": observed, "review": review}}
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    assert result["findings"][0]["missing_verified_tiers"] == ["C"]
    records = result["trusted_environment"]["records"]
    assert len(records) == 2 and records[0]["status"] == "verified"
    assert records[1]["status"] != "verified"
    if failure == "junit_failure":
        assert sum(row["observed"]["denominator"] for row in records) == 4
        assert sum(row["observed"]["passed"] for row in records) == 3


@pytest.mark.parametrize(
    "field",
    [
        "argv_sha256",
        "cwd",
        "runtime_identity_sha256",
        "dependency_inventory_sha256",
        "environment_sha256",
        "isolation",
    ],
)
def test_signed_execution_must_match_frozen_invocation(bundle, field):
    bundle.execution["checks"]["pytest"]["invocation"][field] = "different"
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    assert (
        result["trusted_environment"]["records"][0]["reason"]
        == "observed_check_invocation_mismatch"
    )


def test_old_check_without_execution_environment_binding_is_unavailable(bundle):
    del bundle.execution["checks"]["pytest"]["invocation"]
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    assert result["trusted_environment"]["records"][0]["status"] == "unavailable"


def test_untracked_conftest_actually_changes_pytest_but_invalidates_source_pin(bundle):
    put(
        bundle.source,
        "tests/test_app.py",
        b"def test_pass():\n    assert True\ndef test_fail():\n    assert False\n",
    )
    bundle.config["source_inventory"] = source_inventory(bundle.source)
    sign_entries(bundle)
    env = environment(bundle)
    process_env = dict(
        os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1"
    )
    process_env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_app.py",
    ]
    before = subprocess.run(
        command, cwd=bundle.source, env=process_env, capture_output=True
    )
    put(
        bundle.source,
        "conftest.py",
        b"def pytest_collection_modifyitems(items):\n    items[:] = [item for item in items if item.name != 'test_fail']\n",
    )
    after = subprocess.run(
        command, cwd=bundle.source, env=process_env, capture_output=True
    )
    assert before.returncode == 1 and after.returncode == 0
    assert source_inventory(bundle.source) != bundle.config["source_inventory"]
    result = check(bundle, env)
    assert not result["complete"]
    assert any(
        "current source inventory mismatch" in error for error in result["errors"]
    )


def test_commit_change_with_identical_source_bytes_invalidates_binding(bundle):
    env = environment(bundle)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "different commit",
        ],
        cwd=bundle.source,
        check=True,
    )
    assert source_inventory(bundle.source) == bundle.config["source_inventory"]
    assert not check(bundle, env)["complete"]
    with pytest.raises(TrustInputError, match="current_source_commit_mismatch"):
        environment(bundle)


def test_one_record_cannot_omit_any_bound_finding_from_ledger_coverage(bundle):
    second_finding = deepcopy(bundle.finding)
    second_finding["id"] = "F02"
    second_finding["evidence"] = []
    bundle.ledger["findings"].append(second_finding)
    bundle.spec["findings"].append(deepcopy(second_finding))
    bundle.spec_ref = put(bundle.trust, "spec.json", bundle.spec)
    bundle.config["spec_sha256"] = bundle.spec_ref["sha256"]
    bundle.record["finding_ids"] = ["F01", "F02"]
    bundle.review["reviewed_findings"] = ["F01", "F02"]
    update_record(bundle)
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    assert any("complete record" in error for error in result["errors"])
    second_finding["evidence"] = deepcopy(bundle.finding["evidence"])
    assert check(bundle)["complete"]


def test_omitting_a_record_from_the_frozen_inventory_is_rejected(bundle):
    extra = deepcopy(bundle.record)
    extra["run_id"] = "original-failed-run"
    reference = put(bundle.evidence, "failed-record.json", extra)
    entry = {**deepcopy(bundle.config["records"][0]), **reference}
    bundle.config["records"].append(entry)
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    assert any("complete record" in error for error in result["errors"])


@pytest.mark.parametrize(
    "change", ["adapter", "scope", "review_facts", "review_rejected"]
)
def test_unknown_code_or_missing_external_scope_fails_closed(bundle, change):
    if change == "adapter":
        bundle.config["records"][0]["adapter"] = "candidate.verifier:always_true"
    elif change == "scope":
        bundle.config["authorities"]["execution"]["tiers"] = ["R"]
    elif change == "review_facts":
        bundle.review["coverage"] = ["frozen_criteria"]
    else:
        bundle.review["decision"] = "rejected"
    sign_entries(bundle)
    if change == "adapter":
        with pytest.raises(TrustInputError, match="unsupported_acceptance_adapter"):
            environment(bundle)
    else:
        assert not check(bundle)["complete"]


def test_absent_optional_crypto_is_explicitly_unavailable(bundle, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("fixture missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    result = check(bundle)
    assert not result["complete"]
    assert (
        result["trusted_environment"]["records"][0]["reason"]
        == "signature_dependency_unavailable"
    )


def test_candidate_configuration_and_path_aliases_are_rejected(bundle):
    candidate = bundle.evidence / "trust.json"
    candidate.write_bytes((bundle.trust / "trust.json").read_bytes())
    with pytest.raises(TrustInputError, match="must_be_external"):
        load_trusted_environment(
            candidate,
            expected_sha256=bundle.config_ref["sha256"],
            source_root=bundle.source,
            evidence_root=bundle.evidence,
            expected_commit=bundle.config["commit"],
            spec_sha256=bundle.spec_ref["sha256"],
        )
    alias = bundle.trust / "alias.json"
    alias.symlink_to(bundle.trust / "trust.json")
    with pytest.raises(TrustInputError, match="unsafe_trust_path"):
        load_trusted_environment(
            alias,
            expected_sha256=bundle.config_ref["sha256"],
            source_root=bundle.source,
            evidence_root=bundle.evidence,
            expected_commit=bundle.config["commit"],
            spec_sha256=bundle.spec_ref["sha256"],
        )


@pytest.mark.parametrize(
    "raw,ids",
    [
        (
            b'<testsuite><testcase classname="a" name="x"/><testcase classname="a" name="x"/></testsuite>',
            ["a::x"],
        ),
        (
            b'<testsuite tests="0"><testcase classname="a" name="x"/></testsuite>',
            ["a::x"],
        ),
        (b'<testsuite><testcase classname="a" name="x"/></testsuite>', ["a::y"]),
        (b'<testsuite><testcase name="x"/></testsuite>', ["a::x"]),
        (
            b'<testsuite><testcase classname="a" name="x"><error/><failure/></testcase></testsuite>',
            ["a::x"],
        ),
        (
            b'<testsuite><testcase classname="a" name="x"/><testcase classname="a" name="y"/></testsuite>',
            ["a::x"],
        ),
        (
            b'<testsuite><testcase classname="a" name="x"/><system-out><testcase classname="b" name="hidden"><failure/></testcase></system-out></testsuite>',
            ["a::x"],
        ),
        ((b"<testsuite>" * 100) + (b"</testsuite>" * 100), ["a::x"]),
        (
            '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE testsuite [<!ENTITY x "unsafe">]><testsuite>&x;</testsuite>'.encode(
                "utf-16"
            ),
            ["a::x"],
        ),
    ],
)
def test_junit_cannot_hide_missing_duplicate_or_failed_cases(raw, ids):
    with pytest.raises(TrustInputError):
        verify_junit(raw, expected_test_ids=ids)


def test_nested_junit_counts_are_computed_from_case_outcomes():
    raw = b'<testsuites tests="2" failures="1"><testsuite tests="1"><testcase classname="a" name="x"/></testsuite><testsuite><testsuite><testcase classname="b" name="y"><failure>actual failure</failure></testcase></testsuite></testsuite></testsuites>'
    result = verify_junit(raw, expected_test_ids=["a::x", "b::y"])
    assert result == {
        "tests": 2,
        "failures": 1,
        "errors": 0,
        "skipped": 0,
        "passed": 1,
        "case_ids": ["a::x", "b::y"],
    }


def configure_runtime_bundle(bundle, actual_archive):
    archive, custody_digest, binding = actual_archive
    original_run = archive.parents[2] / "public" / binding.run_id
    copied_run = bundle.evidence / "run"
    shutil.copytree(original_run, copied_run)
    copied_archive = bundle.evidence / "archive"
    shutil.copytree(archive, copied_archive)
    binding_ref = put(bundle.evidence, "binding.json", binding.model_dump())
    final_ref = {
        "path": "run/final-manifest.json",
        "sha256": digest((copied_run / "final-manifest.json").read_bytes()),
    }
    bundle.source = Path(__file__).resolve().parents[1]
    bundle.finding.update(
        source_paths=["marvis/orchestrator/eval/runtime_runner.py"],
        required_evidence_tiers=["R"],
    )
    bundle.spec = {
        "schema_version": "marvis.audit_requirements.v1",
        "findings": [deepcopy(bundle.finding)],
    }
    bundle.spec_ref = put(bundle.trust, "spec.json", bundle.spec)
    bundle.record.update(
        run_id=binding.run_id,
        commit=binding.source["commit"],
        evidence_tier="R",
        environment_kind="local_reference",
        real_executor={"id": "executor-person"},
        source_hashes={
            bundle.finding["source_paths"][0]: digest(
                (bundle.source / bundle.finding["source_paths"][0]).read_bytes()
            )
        },
        tool_receipts=[final_ref["path"]],
        evidence_paths=[final_ref["path"]],
    )
    bundle.record["artifact_hashes"][final_ref["path"]] = final_ref["sha256"]
    bundle.record["input_hashes"][binding_ref["path"]] = binding_ref["sha256"]
    update_record(bundle)
    bundle.config.update(
        commit=binding.source["commit"],
        spec_sha256=bundle.spec_ref["sha256"],
        source_inventory=source_inventory(bundle.source),
    )
    for authority in bundle.config["authorities"].values():
        authority.update(tiers=["R"], environments=["local_reference"])
    bundle.config["records"][0].update(
        adapter="runtime_validation_v2.v1",
        inputs={
            "run_dir": "run",
            "final_manifest_sha256": final_ref["sha256"],
            "cases": [
                {
                    "case_id": binding.case.id,
                    "archive": "archive",
                    "manifest_sha256": custody_digest,
                    "binding": binding_ref,
                }
            ],
        },
    )
    bundle.execution = {
        "case_ids": [binding.case.id],
        "environment_id": "fixture-http-only",
        "model": None,
        "historical_data": None,
    }
    bundle.review["coverage"].append("narrative_semantics")
    sign_entries(bundle)
    return bundle


@pytest.fixture
def runtime_bundle(bundle, actual_archive):
    return configure_runtime_bundle(bundle, actual_archive)


def test_fixed_runtime_adapter_recomputes_actual_http_originals(runtime_bundle):
    result = check(runtime_bundle)
    assert result["complete"], result
    observed = result["trusted_environment"]["records"][0]["observed"]
    assert observed["denominator"] == observed["passed"] == 1
    assert (
        observed["cases"][0]["domain_checks"]["pmml_rescoring"]["status"] == "verified"
    )
    # This external fixture attestation does not restore native signing keys.
    assert (
        observed["cases"][0]["unsupported"]["native_signature_authentication"]
        == "original_key_not_available"
    )


@pytest.mark.parametrize(
    "change", ["missing_archive", "omit_case", "wrong_binding", "changed_source"]
)
def test_runtime_binding_failure_retains_authenticated_full_denominator(
    runtime_bundle, change
):
    entry = runtime_bundle.config["records"][0]
    if change == "missing_archive":
        entry["inputs"]["cases"][0]["archive"] = "missing-archive"
    elif change == "omit_case":
        entry["inputs"]["cases"] = []
    elif change == "wrong_binding":
        path = runtime_bundle.evidence / "binding.json"
        value = json.loads(path.read_bytes())
        value["task_id"] = "wrong-task"
        reference = put(runtime_bundle.evidence, "binding.json", value)
        entry["inputs"]["cases"][0]["binding"] = reference
        runtime_bundle.record["input_hashes"]["binding.json"] = reference["sha256"]
        update_record(runtime_bundle)
    else:
        source = json.loads((runtime_bundle.evidence / "binding.json").read_bytes())
        source["source"]["commit"] = "f" * 40
        reference = put(runtime_bundle.evidence, "binding.json", source)
        entry["inputs"]["cases"][0]["binding"] = reference
        runtime_bundle.record["input_hashes"]["binding.json"] = reference["sha256"]
        update_record(runtime_bundle)
    sign_entries(runtime_bundle)
    result = check(runtime_bundle)
    assert not result["complete"]
    observed = result["trusted_environment"]["records"][0]["observed"]
    assert observed["denominator"] == 1 and observed["passed"] == 0
    assert len(observed["cases"]) == 1
    assert observed["cases"][0]["runtime_status"] == "completed"


def test_reused_environment_recomputes_changed_private_originals(runtime_bundle):
    env = environment(runtime_bundle)
    assert check(runtime_bundle, env)["complete"]
    material = next((runtime_bundle.evidence / "archive/inputs").glob("*.csv"))
    material.write_bytes(material.read_bytes() + b"changed")
    result = check(runtime_bundle, env)
    assert not result["complete"]
    assert result["trusted_environment"]["records"][0]["observed"]["denominator"] == 1


@pytest.mark.parametrize("tier", ["A", "H", "P"])
def test_fixture_runtime_cannot_gain_higher_tiers_from_external_labels(
    runtime_bundle, tier
):
    bundle = runtime_bundle
    bundle.record.update(
        evidence_tier=tier,
        source_kind="deidentified_historical",
        model_profile="claimed-model",
        model_source="real_model",
        evaluation_mode="blind",
    )
    bundle.record["dataset_hashes"] = {
        "results.xml": digest((bundle.evidence / "results.xml").read_bytes())
    }
    bundle.finding["required_evidence_tiers"] = [tier]
    bundle.spec["findings"][0]["required_evidence_tiers"] = [tier]
    bundle.spec_ref = put(bundle.trust, "spec.json", bundle.spec)
    bundle.config["spec_sha256"] = bundle.spec_ref["sha256"]
    if tier == "P":
        bundle.record.update(
            environment_kind="institution_production", source_kind="institution_runtime"
        )
    for authority in bundle.config["authorities"].values():
        authority.update(
            tiers=[tier],
            sources=[bundle.record["source_kind"]],
            environments=[bundle.record["environment_kind"]],
        )
    bundle.execution["model"] = {
        "profile": "claimed-model",
        "source": "real_model",
        "connection_sha256": "f" * 64,
        "evaluation_mode": "blind",
    }
    label = put(bundle.evidence, "labels.json", {"source": "self-reported"})
    bundle.record["input_hashes"][label["path"]] = label["sha256"]
    bundle.execution["historical_data"] = {
        "source_kind": "deidentified_historical",
        "dataset_hashes": bundle.record["dataset_hashes"],
        "label_contract": label,
    }
    bundle.review["coverage"].extend(
        [
            "historical_data_origin",
            "business_label_validity",
            "blind_real_model_execution",
        ]
    )
    update_record(bundle)
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    if tier == "A":
        observed = result["trusted_environment"]["records"][0]
        assert observed["status"] == "unavailable"
        assert observed["reason"] == "formal_acceptance_aggregate_unavailable"
        assert observed["observed"]["denominator"] == 1


def test_original_scoring_interrupt_remains_failed_after_domain_recomputation(
    bundle, tmp_path, monkeypatch
):
    from marvis.orchestrator.eval import runtime_scoring
    from marvis.orchestrator.eval.runtime_archive_validation import (
        FrozenValidationBinding,
    )
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_contracts import RuntimeCase
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_validation_agent import _v2_protocol

    paths = write_synthetic_suite(tmp_path / "suite", normal_validation_agent_only=True)
    case = RuntimeCase.model_validate(
        json.loads(paths["cases"].read_bytes())["cases"][0]
    )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime_scoring, "score_case", interrupt)
    with fixture_model(answer_factory=_v2_protocol) as (model, _):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "public",
            evidence_custody_dir=tmp_path / "custody",
            model=model,
            model_source="fixture_model",
        )
    observed = report["cases"][0]
    assert (
        not report["all_passed"]
        and observed["score"]["scorer_error"] == "KeyboardInterrupt"
    )
    pipeline = observed["execution"]["validation_pipeline"]
    binding = FrozenValidationBinding(
        case=case,
        run_id=report["run_id"],
        task_id=observed["execution"]["task_id"],
        cases_sha256=report["cases_sha256"],
        expected_sha256=report["expected_sha256"],
        source=report["source"],
        model_connection_sha256=report["model_connection_sha256"],
        input_contract_sha256=pipeline["input_contract_sha256"],
        confirmed_draft_sha256=pipeline["confirmed_draft_sha256"],
        report_revision=pipeline["report_revision"],
        report_sha256={
            item["kind"]: item["sha256"] for item in pipeline["report_files"]
        },
        bin_count=10,
    )
    archive = tmp_path / "custody" / report["run_id"] / case.id
    configured = configure_runtime_bundle(
        bundle, (archive, observed["evidence_custody"]["manifest_sha256"], binding)
    )
    result = check(configured)
    assert not result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == 1 and facts["passed"] == 0
    assert facts["cases"][0]["recorded_passed"] is False
    assert all(
        item["status"] == "verified"
        for item in facts["cases"][0]["domain_checks"].values()
    )
