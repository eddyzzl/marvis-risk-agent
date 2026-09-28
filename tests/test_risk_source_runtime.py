from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import pandas as pd
import pytest
import httpx

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.plugins.manifest import ToolRef
from marvis.risk_context.source_contracts import (
    SourceEnvelope,
    SourceError,
    SourceProfile,
    assess,
    iso,
)
from marvis.risk_context.source_service import SourceService

from test_modeling_pack import _runtime
from test_operations_api import _claim_role

SUBJECT = "a" * 64


def source_record(**changes):
    now = time.time()
    return {
        "subject_namespace": "reference.people",
        "subject_token": SUBJECT,
        "record_version": 1,
        "observed_at": iso(now - 60),
        "available_at": iso(now - 30),
        "expires_at": iso(now + 3600),
        "kyc": {"identity_match": "match", "document_status": "valid"},
        "bureau": {"coverage": "partial", "accounts": []},
        **changes,
    }


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def provider(root, record=None, *, mode="normal", port=None):
    root.mkdir(parents=True, exist_ok=True)
    fixture = root / "records.json"
    fixture.write_text(json.dumps([record or source_record()]))
    port = port or free_port()
    log = (root / "server.log").open("w")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "marvis",
            "reference-source",
            "--state-dir",
            str(root),
            "--records",
            str(fixture),
            "--port",
            str(port),
            "--fault-mode",
            mode,
        ],
        cwd=Path(__file__).parents[1],
        stdout=log,
        stderr=log,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            if process.poll() is not None:
                raise AssertionError((root / "server.log").read_text())
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise AssertionError("reference service startup timeout")
                time.sleep(0.05)
        yield f"http://127.0.0.1:{port}", (root / "access_token").read_text(), root
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


@pytest.fixture
def runtime(tmp_path):
    runner, _, registry, _, settings, task = _runtime(tmp_path)
    app = create_app(settings)
    maker, admin, other = TestClient(app), TestClient(app), TestClient(app)
    actor = _claim_role(app, maker, "maker")
    _claim_role(app, admin, "admin")
    _claim_role(app, other, "maker")
    basis = admin.post(
        f"/api/tasks/{task.id}/risk-sources/authorization-bases",
        json={
            "basis_id": "basis-one",
            "reference": "synthetic-test-consent",
            "declaration": "synthetic_reference_test",
        },
    )
    assert basis.status_code == 201, basis.text
    return app, runner, registry, task, maker, admin, other, actor, basis.json()["id"]


def authorize(runtime, endpoint=None, token=None, **overrides):
    app, runner, registry, task, maker, admin, other, actor, basis = runtime
    profile = {
        "profile_id": overrides.pop("profile_id", "reference"),
        "mode": "reference_http" if endpoint else "historical_file",
        **({"endpoint": endpoint} if endpoint else {}),
        **overrides,
    }
    response = admin.post(
        "/api/risk-sources/profiles",
        json={"profile": profile, **({"token": token} if token else {})},
    )
    assert response.status_code == 201, response.text
    grant_id = "grant-" + profile["profile_id"]
    response = admin.post(
        "/api/risk-sources/grants",
        json={
            "grant_id": grant_id,
            "profile_id": profile["profile_id"],
            "task_id": task.id,
            "grantee_id": actor["id"],
            "subject_namespace": "reference.people",
            "subject_token": SUBJECT,
            "purpose": "credit_review",
            "basis_artifact_id": basis,
            "starts_at": iso(time.time() - 10),
            "expires_at": iso(time.time() + 3600),
        },
    )
    assert response.status_code == 201, response.text
    return grant_id


def prepare(runtime, grant_id, request_id="query-one", **extra):
    response = runtime[4].post(
        f"/api/tasks/{runtime[3].id}/risk-sources/requests",
        json={"request_id": request_id, "grant_id": grant_id, **extra},
    )
    assert response.status_code == 201, response.text
    return response.json()


def execute(runtime, request_id="query-one"):
    response = runtime[4].post(
        f"/api/tasks/{runtime[3].id}/risk-sources/requests/{request_id}/execute"
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_cli_http_real_worker_receipt_restart_and_safe_summary(runtime, tmp_path):
    with provider(tmp_path / "provider") as (endpoint, token, state):
        grant = authorize(runtime, endpoint, token)
        frozen = prepare(runtime, grant)
        result = execute(runtime)
        assert result["state"] == "completed"
        assert result["status"] == "available"
        assert result["artifact_id"]
        assert result["automated_clearance"] is False
        assert SUBJECT not in json.dumps(result) and token not in json.dumps(result)
        assert "bureau:partial" in result["missing_reasons"]
        service = SourceService(runtime[0].state.settings)
        restored = service.execute(runtime[3].id, "query-one")
        assert restored == result
        assert prepare(runtime, grant)["contract_hash"] == frozen["contract_hash"]
        details = runtime[4].get(result["evidence_url"]).json()
        assert details["assessment"]["account_count"] is None
        assert details["assessment"]["totals_by_currency"] is None
        with sqlite3.connect(state / "reference.sqlite") as conn:
            assert conn.execute("SELECT count(*) FROM queries").fetchone()[0] == 1
        with connect(service.repo.db_path) as conn:
            assert (
                conn.execute("SELECT count(*) FROM source_attempts").fetchone()[0] == 1
            )


def test_timeout_after_provider_commit_recovers_by_get_only(runtime, tmp_path):
    with provider(tmp_path / "provider", mode="timeout_after_commit") as (
        endpoint,
        token,
        state,
    ):
        grant = authorize(runtime, endpoint, token, timeout_seconds=0.1)
        prepare(runtime, grant)
        result = execute(runtime)
        assert result["state"] == "unknown_effect"
        assert result["next_action"] == "readback_same_request"
        result = SourceService(runtime[0].state.settings).execute(
            runtime[3].id, "query-one"
        )
        assert result["state"] == "completed" and result["artifact_id"]
        with connect(runtime[0].state.settings.db_path) as conn:
            assert [
                r[0]
                for r in conn.execute(
                    "SELECT method FROM source_attempts ORDER BY started_at"
                )
            ] == ["POST", "GET"]
        with sqlite3.connect(state / "reference.sqlite") as conn:
            assert conn.execute("SELECT count(*) FROM queries").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("mode", "status", "state"),
    [
        ("schema_drift", "schema_drift", "completed"),
        ("unauthorized", "unauthorized", "completed"),
        ("unavailable", "unavailable", "unknown_effect"),
    ],
)
def test_http_conformance_errors_are_not_clearance(
    runtime, tmp_path, mode, status, state
):
    with provider(tmp_path / "provider", mode=mode) as (endpoint, token, _):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant)
        result = execute(runtime)
        assert (result["status"], result["state"]) == (status, state)
        assert result["artifact_id"] is None and not result["automated_clearance"]


def test_roles_scope_revocation_and_no_client_authorized_flag(runtime, tmp_path):
    app, _, _, task, maker, admin, other, _, _ = runtime
    denied = maker.post(
        "/api/risk-sources/profiles",
        json={"profile": {"profile_id": "x", "mode": "historical_file"}},
    )
    assert denied.status_code == 403
    grant = authorize(runtime)
    prepare(
        runtime,
        grant,
        historical={
            "dataset_id": "pending",
            "expected_content_hash": "1" * 64,
            "row_id_column": "id",
            "row_id": "one",
            "response_column": "response",
        },
    )
    path = f"/api/tasks/{task.id}/risk-sources/requests/query-one"
    assert other.get(path).status_code == 403
    assert other.post(path + "/execute").status_code == 403
    assert other.get(f"/api/tasks/{task.id}/risk-sources/grants").json()["grants"] == []
    assert (
        maker.post(
            f"/api/tasks/{task.id}/risk-sources/requests",
            json={"request_id": "fake", "grant_id": grant, "authorized": True},
        ).status_code
        == 422
    )
    assert admin.post(f"/api/risk-sources/grants/{grant}/revoke").status_code == 200
    assert maker.post(path + "/execute").status_code == 403
    with pytest.raises(SourceError, match="grant_revoked"):
        app.state.risk_sources.execute(task.id, "query-one")


def test_history_import_authenticates_dataset_and_does_not_fill_zero(runtime, tmp_path):
    app, runner, registry, task, maker, *_ = runtime
    grant = authorize(runtime)
    path = tmp_path / "history.parquet"
    pd.DataFrame(
        [{"id": "record-one", "response": json.dumps(source_record())}]
    ).to_parquet(path, index=False)
    dataset = registry.register_existing(path, task_id=task.id, role="source_history")
    history = {
        "dataset_id": dataset.id,
        "expected_content_hash": dataset.content_hash,
        "row_id_column": "id",
        "row_id": "record-one",
        "response_column": "response",
    }
    prepare(runtime, grant, historical=history)
    result = runner.invoke(
        ToolRef("risk_context", "query_source"),
        {"request_id": "query-one"},
        task_id=task.id,
    )
    assert result.ok, result.error
    assert result.output["origin"] == "historical_import_unverified"
    assert result.output["status"] == "available"
    details = maker.get(result.output["evidence_url"]).json()
    assert details["assessment"]["account_count"] is None
    assert (
        details["receipt"]["dataset_binding"]["expected_content_hash"]
        == dataset.content_hash
    )
    prepare(
        runtime,
        grant,
        "wronghash",
        historical={**history, "expected_content_hash": "f" * 64},
    )
    assert execute(runtime, "wronghash")["status"] == "source_material_invalid"


def test_rate_limit_is_shared_durable_and_applies_to_recovery(runtime, tmp_path):
    with provider(tmp_path / "provider", mode="timeout_after_commit") as (
        endpoint,
        token,
        _,
    ):
        grant = authorize(
            runtime, endpoint, token, timeout_seconds=0.1, requests_per_minute=1
        )
        prepare(runtime, grant)
        assert execute(runtime)["state"] == "unknown_effect"
        service = SourceService(runtime[0].state.settings)
        with pytest.raises(SourceError, match="source_rate_limited"):
            service.execute(runtime[3].id, "query-one")
        with connect(service.repo.db_path) as conn:
            assert (
                conn.execute("SELECT count(*) FROM source_attempts").fetchone()[0] == 1
            )


def test_circuit_breaker_persists_and_does_not_retry_post(runtime, tmp_path):
    with provider(tmp_path / "provider", mode="unavailable") as (endpoint, token, _):
        grant = authorize(runtime, endpoint, token, breaker_threshold=1)
        prepare(runtime, grant)
        assert execute(runtime)["status"] == "unavailable"
        service = SourceService(runtime[0].state.settings)
        with pytest.raises(SourceError, match="source_circuit_open"):
            service.execute(runtime[3].id, "query-one")


def test_concurrent_claims_issue_one_post_and_conflicting_request_id_rejected(
    runtime, tmp_path
):
    with provider(tmp_path / "provider", mode="timeout_after_commit") as (
        endpoint,
        token,
        state,
    ):
        grant = authorize(runtime, endpoint, token, timeout_seconds=0.1)
        prepare(runtime, grant)
        service = runtime[0].state.risk_sources

        def run(_):
            try:
                return service.execute(runtime[3].id, "query-one")["state"]
            except SourceError as exc:
                return exc.code

        with ThreadPoolExecutor(max_workers=4) as pool:
            states = list(pool.map(run, range(4)))
        assert "unknown_effect" in states
        with connect(service.repo.db_path) as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM source_attempts WHERE method='POST'"
                ).fetchone()[0]
                == 1
            )
        response = runtime[4].post(
            f"/api/tasks/{runtime[3].id}/risk-sources/requests",
            json={
                "request_id": "query-one",
                "grant_id": grant,
                "expires_in_seconds": 1,
            },
        )
        assert response.status_code == 409


def test_expired_record_is_explicit_and_evidence_tamper_blocks(runtime, tmp_path):
    now = time.time()
    expired = source_record(
        observed_at=iso(now - 90), available_at=iso(now - 60), expires_at=iso(now - 1)
    )
    with provider(tmp_path / "provider", expired) as (endpoint, token, _):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant)
        result = execute(runtime)
        assert result["status"] == "expired"
        service = runtime[0].state.risk_sources
        record = service.repo.artifacts.get_for_task(
            runtime[3].id, result["artifact_id"]
        )
        Path(record["path"]).write_text("{}")
        response = runtime[4].get(result["evidence_url"])
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "source_receipt_integrity_failed"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:1234",
        "http://127.0.0.1:1234/redirect",
        "https://127.0.0.1:1234",
        "http://user:secret@127.0.0.1:1234",
        "http://169.254.169.254:80",
    ],
)
def test_profile_cannot_choose_arbitrary_endpoint(endpoint):
    with pytest.raises(ValueError):
        SourceProfile(profile_id="bad", mode="reference_http", endpoint=endpoint)


def test_contract_missing_defaults_are_unknown_and_credit_nulls_are_not_zero():
    record = SourceEnvelope.model_validate(
        source_record(
            kyc={},
            bureau={
                "coverage": "complete",
                "accounts": [{"account_token": "b" * 64, "currency": "CNY"}],
            },
        )
    )
    result = assess(record, time.time())
    assert result["kyc_claims"]["identity_match"] == "unknown"
    assert result["totals_by_currency"]["CNY"]["balance_minor"] is None
    assert result["automated_clearance"] is False


def test_expired_actor_blocks_queued_tool_and_request_scope_is_signed(runtime):
    app, _, _, task, maker, _, _, actor, _ = runtime
    grant = authorize(runtime)
    # Freeze a valid history intent without reading the dataset until execution.
    history = {
        "dataset_id": "pending",
        "expected_content_hash": "1" * 64,
        "row_id_column": "id",
        "row_id": "one",
        "response_column": "response",
    }
    prepare(runtime, grant, historical=history)
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE local_principals SET expires_at=? WHERE id=?",
            (iso(time.time() - 1), actor["id"]),
        )
    with pytest.raises(SourceError, match="source_role_forbidden"):
        app.state.risk_sources.execute(task.id, "query-one")
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM source_attempts").fetchone()[0] == 0
        conn.execute(
            "UPDATE local_principals SET expires_at=? WHERE id=?",
            (iso(time.time() + 3600), actor["id"]),
        )
        conn.execute(
            "UPDATE source_requests SET actor_id='changed' WHERE task_id=?", (task.id,)
        )
    with pytest.raises(SourceError, match="source_request_binding_failed"):
        app.state.risk_sources.execute(task.id, "query-one")


def test_normal_cli_main_http_to_worker_and_restarted_provider(runtime, tmp_path):
    app, _, _, task, maker, *_ = runtime
    fixture = source_record()
    state = tmp_path / "reference-cli"
    port = free_port()
    with provider(state, fixture, port=port) as (endpoint, token, _):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant)
    # The provider reloads its stored corpus, request table and credential after restart.
    with provider(state, fixture, port=port):
        app_port = free_port()
        with (tmp_path / "app-cli.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "marvis",
                    "serve",
                    "--workspace",
                    str(app.state.settings.workspace),
                    "--port",
                    str(app_port),
                ],
                cwd=Path(__file__).parents[1],
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{app_port}",
                    cookies=dict(maker.cookies),
                    timeout=30,
                ) as client:
                    deadline = time.monotonic() + 25
                    while True:
                        if process.poll() is not None:
                            raise AssertionError((tmp_path / "app-cli.log").read_text())
                        try:
                            response = client.get("/api/risk-sources/capabilities")
                            break
                        except httpx.TransportError:
                            if time.monotonic() > deadline:
                                raise AssertionError("main CLI startup timeout")
                            time.sleep(0.05)
                    assert response.status_code == 200, response.text
                    result = client.post(
                        f"/api/tasks/{task.id}/risk-sources/requests/query-one/execute"
                    )
                    assert result.status_code == 200, result.text
                    receipt = result.json()
                    assert receipt["status"] == "available" and receipt["artifact_id"]
                    assert client.get(receipt["evidence_url"]).status_code == 200
                    (tmp_path / "source-cli-proof.json").write_text(
                        json.dumps(
                            {
                                "scope": "synthetic_reference_only",
                                "cli_provider_restarted": True,
                                "cli_main_http_worker": True,
                                "receipt": receipt,
                                "external_supplier_acceptance": "not_evaluated",
                            },
                            indent=2,
                        )
                    )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def test_no_record_is_not_zero_credit_or_kyc_pass(runtime, tmp_path):
    with provider(tmp_path / "provider", source_record(subject_token="b" * 64)) as (
        endpoint,
        token,
        _,
    ):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant)
        result = execute(runtime)
        assert result["status"] == "no_record"
        assert result["missing_reasons"] == ["provider:no_record"]
        evidence = runtime[4].get(result["evidence_url"]).json()
        assert evidence["normalized_source"] is None and evidence["assessment"] is None


def test_private_response_tamper_blocks_idempotent_summary(runtime, tmp_path):
    with provider(tmp_path / "provider") as (endpoint, token, _):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant)
        result = execute(runtime)
        service = runtime[0].state.risk_sources
        details = service.evidence(runtime[3].id, "query-one", runtime[7]["id"])
        path = (
            service.repo.private
            / f"wire-{details['receipt']['wire_response_hash']}.bin"
        )
        path.write_bytes(b"changed wire response")
        with pytest.raises(SourceError, match="source_response_integrity_failed"):
            SourceService(runtime[0].state.settings).execute(runtime[3].id, "query-one")
        assert (
            runtime[4].get(result["evidence_url"].removesuffix("/evidence")).status_code
            == 409
        )


def test_expired_prepared_intent_never_sends_and_crashed_running_only_gets(
    runtime, tmp_path
):
    with provider(tmp_path / "provider") as (endpoint, token, state):
        grant = authorize(runtime, endpoint, token)
        prepare(runtime, grant, expires_in_seconds=1)
        # Advance only past the recorded deadline; the grant remains live.
        time.sleep(1.05)
        with pytest.raises(SourceError, match="source_request_expired"):
            runtime[0].state.risk_sources.execute(runtime[3].id, "query-one")
        prepare(runtime, grant, "crashed")
        repo = runtime[0].state.risk_sources.repo
        owner, _, _, _ = repo.claim(runtime[3].id, "crashed")
        assert owner
        # Simulate a process crash after durable intent, before any response was stored.
        with connect(repo.db_path) as conn:
            conn.execute("UPDATE source_requests SET lease_until=0 WHERE id='crashed'")
        result = SourceService(runtime[0].state.settings).execute(
            runtime[3].id, "crashed"
        )
        assert result["state"] == "unknown_effect"
        with connect(repo.db_path) as conn:
            assert [
                r[0]
                for r in conn.execute(
                    "SELECT method FROM source_attempts ORDER BY started_at"
                )
            ] == ["POST", "GET"]
        with sqlite3.connect(state / "reference.sqlite") as conn:
            assert conn.execute("SELECT count(*) FROM queries").fetchone()[0] == 0


def test_known_adverse_kyc_claims_have_deterministic_reason_codes():
    record = SourceEnvelope.model_validate(
        source_record(kyc={"identity_match": "mismatch", "watchlist": "hit"})
    )
    result = assess(record, time.time())
    assert result["finding_codes"] == [
        "kyc.identity_match:mismatch",
        "kyc.watchlist:hit",
    ]
    assert "kyc.document_status:unknown" in result["missing_reasons"]
