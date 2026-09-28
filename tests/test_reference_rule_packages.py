"""Model-free packages use the same HTTP, worker, deployment and replay kernel."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import joblib
import pytest
from pydantic import ValidationError

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.decision_twin.batch import _economics
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.evaluator import evaluate_strategy_row
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.reference_decision.contracts import DecisionError, RulePackageRequest
from marvis.reference_decision.evaluation import evaluate

from test_modeling_pack import _runtime
from test_operations_api import _claim_role
from test_reference_deployment import _promotion, _install_activate
from test_risk_source_runtime import free_port


def register(app, task_id, spec, identity):
    strategy = replace(
        build_strategy_from_spec(spec, score_col="x1", description="rules"), id=identity
    )
    StrategyRepository(app.state.settings.db_path).create_strategy(task_id, strategy)
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status='validated', status='draft' WHERE id=?",
            (identity,),
        )


@pytest.fixture(scope="module")
def rule_runtime(tmp_path_factory):
    root = tmp_path_factory.mktemp("rule-runtime")
    *_, settings, task = _runtime(root)
    app = create_app(settings)
    clients = [TestClient(app) for _ in range(3)]
    for client, role in zip(clients, ("maker", "checker", "admin"), strict=True):
        _claim_role(app, client, role)
    spec = StrategySpec(
        strategy_type="approval",
        rules=(
            StrategyRuleSpec(
                rule_id="low",
                priority=1,
                condition={
                    "op": "compare",
                    "field": "x1",
                    "operator": "<",
                    "value": 0.5,
                },
                action=StrategyAction(type="approval", reason_code="LOW_RULE"),
            ),
        ),
        default_action=StrategyAction(type="review", reason_code="RULE_REVIEW"),
    )
    register(app, task.id, spec, "online-strategy")
    request = RulePackageRequest(
        package_kind="rule_only",
        strategy_id="online-strategy",
        strategy_version=1,
        decision_node="underwriting",
        raw_schema=[{"name": "x1", "type": "number"}, {"name": "x2", "type": "number"}],
    )
    return app, *clients, task, request, root


def test_rule_package_http_is_explicit_and_never_loads_a_model(
    rule_runtime, monkeypatch
):
    app, maker, checker, _, _, request, _ = rule_runtime
    monkeypatch.setattr(
        joblib, "load", lambda *a, **k: pytest.fail("rule-only loaded a model")
    )
    assert (
        checker.post(
            "/api/reference-decision/packages", json=request.model_dump()
        ).status_code
        == 403
    )
    response = maker.post("/api/reference-decision/packages", json=request.model_dump())
    assert response.status_code == 201, response.text
    body = response.json()
    manifest = body["manifest"]
    assert manifest["schema_version"] == "reference-decision-package.v2"
    assert manifest["model"] is None and manifest["model_producer_receipt"] is None
    assert (
        manifest["files"] == [] and manifest["configuration"]["score_product"] is None
    )
    with app.state.reference_decision.packages.snapshot(body["package_hash"]) as (
        frozen,
        directory,
    ):
        for x in (0.1, 0.9):
            result = evaluate(frozen, directory, {"x1": x, "x2": 2})
            assert result["score"] is None and result["score_product"] is None
            assert result["timing_ms"]["scoring"] is None
            assert (
                result["action"]
                == evaluate_strategy_row(
                    {"x1": x}, manifest["strategy_spec"]
                ).to_dict()["action"]
            )
    row = maker.get("/api/reference-decision/packages").json()["packages"][0]
    assert row["model_artifact_id"] is None and row["package_kind"] == "rule_only"


@pytest.mark.parametrize(
    "extra",
    [{"model_artifact_id": "fake"}, {"score_field": "pd"}, {"score_product": "raw_pd"}],
)
def test_rule_model_metadata_is_rejected(rule_runtime, extra):
    request = rule_runtime[-2]
    with pytest.raises(ValidationError):
        RulePackageRequest.model_validate({**request.model_dump(), **extra})


def test_unbound_or_reserved_features_cannot_be_external_rule_inputs(rule_runtime):
    app, _, _, _, _, request, _ = rule_runtime
    with pytest.raises(DecisionError, match="strategy_inputs_unbound"):
        app.state.reference_decision.packages.build(
            request.model_copy(update={"raw_schema": []}), actor_id="maker"
        )
    with pytest.raises(ValidationError):
        RulePackageRequest.model_validate(
            {
                **request.model_dump(),
                "raw_schema": [{"name": "__marvis_model_pd_fake", "type": "number"}],
            }
        )
    hash_, _ = app.state.reference_decision.packages.build(request, actor_id="maker")
    with app.state.reference_decision.packages.snapshot(hash_) as (manifest, directory):
        with pytest.raises(DecisionError, match="raw_schema_mismatch"):
            evaluate(manifest, directory, {"x1": 0, "x2": 0, "pd": 0})


def test_default_only_limit_pricing_and_structured_collection(rule_runtime):
    app, _, _, _, task, request, _ = rule_runtime
    from test_collection_planning import policy, spec as collection_spec

    collection = collection_spec(policy()).to_dict()
    collection["default_action"]["reason_code"] = "COLLECTION_POLICY"
    specs = [
        StrategySpec(
            strategy_type="limit",
            default_action=StrategyAction(
                type="limit", value=10000, reason_code="LIMIT"
            ),
        ),
        StrategySpec(
            strategy_type="pricing",
            default_action=StrategyAction(
                type="pricing", value=0.12, reason_code="PRICE"
            ),
        ),
        StrategySpec.from_dict(collection),
    ]
    for index, spec in enumerate(specs):
        identity = f"default-only-{index}"
        register(app, task.id, spec, identity)
        req = request.model_copy(update={"strategy_id": identity, "raw_schema": []})
        hash_, _ = app.state.reference_decision.packages.build(req, actor_id="maker")
        result = app.state.reference_decision.evaluate(hash_, {}, 20)
        assert result["action"] == spec.default_action.to_dict()
        assert result["score"] is None
    assert _economics([], [], object(), None)["status"] == "unknown"


def test_rule_http_governed_install_restart_rollback_and_real_cli(
    rule_runtime, monkeypatch
):
    app, maker, checker, admin, _, request, root = rule_runtime
    governed = app, maker, checker, admin
    store = app.state.reference_decision.packages
    hash_, _ = store.build(request, actor_id="maker")
    first_promotion = _promotion(governed, hash_)
    receipt, _ = _install_activate(governed, first_promotion)
    assert receipt["readback"]["model_artifact_id"] is None
    assert receipt["probe"]["score"] is None
    payload = {
        "request_id": "rule-application",
        "decision_node": "underwriting",
        "expected_package_hash": hash_,
        "features": {"x1": 0.1, "x2": 0.2},
    }
    response = admin.post("/api/reference-decision/decisions", json=payload)
    assert response.status_code == 200, response.text
    original = response.json()
    assert (
        original["action"]["reason_code"] == "LOW_RULE"
        and original["status"] == "decided"
    )
    assert (
        original["versions"]["model_artifact_id"] is None and original["score"] is None
    )
    newer, _ = store.build(
        request.model_copy(update={"failure_action": "reject"}), actor_id="maker"
    )
    _, active = _install_activate(governed, _promotion(governed, newer))
    assert (
        admin.post("/api/reference-decision/decisions", json=payload).json() == original
    )
    rollback = admin.post(
        f"/api/production-governance/environments/local-reference/deployments/{active['id']}/rollback",
        json={"expected_active_deployment_id": active["id"], "reason": "rule rollback"},
    )
    assert rollback.status_code == 200, rollback.text
    fresh = create_app(app.state.settings)
    assert fresh.state.reference_decision.head()["package_hash"] == hash_
    port = free_port()
    with (root / "rule-cli.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "marvis",
                "serve",
                "--workspace",
                str(app.state.settings.workspace),
                "--port",
                str(port),
            ],
            cwd=Path(__file__).parents[1],
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                cookies=dict(admin.cookies),
                timeout=30,
            ) as client:
                deadline = time.monotonic() + 25
                while True:
                    assert process.poll() is None, (root / "rule-cli.log").read_text()
                    try:
                        ready = client.get("/api/reference-decision/status")
                        break
                    except httpx.TransportError:
                        assert time.monotonic() < deadline
                        time.sleep(0.05)
                assert ready.json()["package_hash"] == hash_
                assert (
                    client.post(
                        "/api/reference-decision/decisions", json=payload
                    ).json()
                    == original
                )
                result = client.post(
                    "/api/reference-decision/decisions",
                    json={**payload, "request_id": "rule-cli-new"},
                )
                assert result.status_code == 200, result.text
                assert (
                    result.json()["status"] == "decided"
                    and result.json()["score"] is None
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def unavailable(*args):
        raise DecisionError("scoring_timeout", 503)

    monkeypatch.setattr(app.state.reference_decision, "evaluate", unavailable)
    fallback = admin.post(
        "/api/reference-decision/decisions",
        json={**payload, "request_id": "rule-timeout"},
    ).json()
    assert fallback["status"] == "fallback" and fallback["action"]["type"] == "review"
    assert fallback["score"] is None and fallback["score_product"] is None
