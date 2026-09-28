import json

from fastapi.testclient import TestClient
import joblib
import pytest

from marvis.app import create_app
from marvis.db_schema import connect
from marvis.packs.modeling.producer_receipts import KIND
from marvis.plugins.manifest import ToolRef
from marvis.reference_decision.readiness import package_readiness, required_raw_fields
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings

from test_feature_preprocessing_provenance import scenario
from test_operations_api import _claim_role
from test_reference_decision import packaged as packaged


def test_readiness_authenticates_native_closure_without_deserializing_or_guessing_schema(
    packaged, monkeypatch
):
    app, store, request, _, _, artifact, *_ = packaged
    monkeypatch.setattr(
        joblib, "load", lambda *a, **k: pytest.fail("readiness unpickled source")
    )
    result = package_readiness(
        store, artifact.id, strategy_id=request.strategy_id, strategy_version=1
    )
    assert result["state"] == "authenticated" and result["build_ready"] is False
    assert result["producer"]["state"] == "authenticated"
    assert result["raw_requirements"] == [
        {"name": "x1", "type": None, "nullable": None, "declaration_required": True},
        {"name": "x2", "type": None, "nullable": None, "declaration_required": True},
    ]
    assert result["score_field_candidates"] == ["pd"]
    assert result["score_products"] == [
        {"value": "raw_pd", "available": True, "reason_code": None},
        {
            "value": "calibrated_pd",
            "available": False,
            "reason_code": "calibration_required",
        },
        {
            "value": "scorecard_points",
            "available": False,
            "reason_code": "scorecard_required",
        },
    ]
    client = TestClient(app)
    assert (
        client.get(
            "/api/reference-decision/readiness",
            params={"model_artifact_id": artifact.id},
        ).status_code
        == 403
    )
    _claim_role(app, client, "maker")
    response = client.get(
        "/api/reference-decision/readiness",
        params={
            "model_artifact_id": artifact.id,
            "strategy_id": request.strategy_id,
            "strategy_version": 1,
        },
    )
    assert response.status_code == 200 and response.json() == result


def test_missing_native_receipt_stays_unknown_without_backfill(packaged, monkeypatch):
    _, store, _, _, _, artifact, *_ = packaged
    original = TaskArtifactRepository.list_for_task
    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_for_task",
        lambda self, task_id: [r for r in original(self, task_id) if r["kind"] != KIND],
    )
    result = package_readiness(store, artifact.id)
    assert result["state"] == "blocked"
    assert result["reason_codes"] == [
        "native_model_producer_evidence_unknown_retrain_required"
    ]
    assert result["raw_requirements"] == []


def test_altered_native_bytes_or_calibration_metadata_fail_closed(packaged):
    _, store, _, _, _, artifact, *_ = packaged
    from marvis.repositories.modeling import ModelingRepository

    experiment = ModelingRepository(store.settings.db_path).get_experiment(
        artifact.experiment_id
    )
    path = (
        store.settings.tasks_dir
        / experiment.task_id
        / "modeling_artifacts"
        / artifact.model_path
    )
    original = path.read_bytes()
    try:
        path.write_bytes(b"not the produced model")
        assert package_readiness(store, artifact.id)["reason_codes"] == [
            "native_model_source_integrity_failed"
        ]
    finally:
        path.write_bytes(original)
    with connect(store.settings.db_path) as conn:
        row = conn.execute(
            "SELECT params_json FROM model_artifacts WHERE id=?", (artifact.id,)
        ).fetchone()
        original_params = row["params_json"]
        params = json.loads(original_params)
        params["calibration"] = {"path": "arbitrary.pkl"}
        conn.execute(
            "UPDATE model_artifacts SET params_json=? WHERE id=?",
            (json.dumps(params), artifact.id),
        )
    try:
        result = package_readiness(store, artifact.id)
        assert result["state"] == "blocked"
        assert result["score_products"] == []
    finally:
        with connect(store.settings.db_path) as conn:
            conn.execute(
                "UPDATE model_artifacts SET params_json=? WHERE id=?",
                (original_params, artifact.id),
            )


def test_real_preprocessing_receipt_recovers_raw_inputs_and_blocks_tamper(tmp_path):
    runner, registry, _, source, transformed = scenario(tmp_path)
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": transformed.id,
            "recipe": "lr",
            "features": ["x"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 7,
        },
        task_id="task-feature",
    )
    assert trained.ok, trained.error
    store = create_app(
        build_settings(registry.datasets_root.parent)
    ).state.reference_decision.packages
    result = package_readiness(store, trained.output["artifact_id"])
    assert result["state"] == "authenticated", result
    assert result["preprocessing"]["state"] == "training_only"
    assert result["preprocessing"]["receipt_ids"]
    assert result["preprocessing"]["source_binding"]["dataset_id"] == source.id
    assert [r["name"] for r in result["raw_requirements"]] == ["x"]
    raw_path = registry.resolve_path(source.id)
    raw_path.chmod(0o600)
    raw_path.write_bytes(b"changed source")
    assert package_readiness(store, trained.output["artifact_id"])["state"] == "blocked"


def test_dependency_walk_does_not_request_derived_or_unconsumed_population_columns():
    steps = [
        {"kind": "normalize", "columns": ["income"], "params": {}},
        {
            "kind": "onehot",
            "columns": ["channel"],
            "params": {"channel": ["web", "app"]},
        },
        {"kind": "fitted_rank", "columns": ["income"], "params": {"column": "income"}},
    ]
    names, outputs, derived = required_raw_fields(
        ["income__rank", "channel_web"],
        steps,
        ["income", "channel", "y", "split", "unused_age"],
    )
    assert names == ["channel", "income"]
    assert "unused_age" not in outputs and "income__rank" in derived


def test_mismatched_strategy_version_never_advertises_authenticated_package(packaged):
    _, store, request, _, _, artifact, *_ = packaged
    result = package_readiness(
        store, artifact.id, strategy_id=request.strategy_id, strategy_version=999
    )
    assert result["state"] == "blocked"
    assert result["reason_codes"] == ["strategy_not_ready"]


def test_rule_readiness_requires_no_model_or_score_and_discovers_fields(packaged, monkeypatch):
    from marvis.repositories.modeling import ModelingRepository

    app, _, request, *_ = packaged
    client = TestClient(app)
    _claim_role(app, client, "maker")
    monkeypatch.setattr(ModelingRepository, "get_model_artifact", lambda *a: pytest.fail("rule readiness queried model"))
    params = {"package_kind": "rule_only", "strategy_id": request.strategy_id,
              "strategy_version": request.strategy_version}
    response = client.get("/api/reference-decision/readiness", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "authenticated" and body["model"] is None
    assert body["raw_requirements"] == [{"name": "pd", "type": None, "nullable": None, "declaration_required": True}]
    assert body["score_field_candidates"] == body["score_products"] == []
    assert "score_field" not in body["build_requires_explicit_declaration"]
    assert client.get("/api/reference-decision/readiness", params={**params, "model_artifact_id": "fake"}).status_code == 422
    assert client.get("/api/reference-decision/readiness", params={**params, "strategy_version": 999}).json()["state"] == "blocked"
