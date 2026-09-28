from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time

import joblib

from fastapi.testclient import TestClient
import pytest

from marvis.app import create_app
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.packs.strategy.dsl import StrategyAction, StrategyRuleSpec, StrategySpec
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.reference_decision.contracts import DecisionError, PackageRequest
from marvis.reference_decision.evaluation import evaluate
from marvis.reference_decision.features import feature_outputs
from marvis.reference_decision.ledger import DecisionLedger
from marvis.repositories.modeling import ModelingRepository

from test_modeling_pack import _runtime
from test_modeling_monitor import _train_lr_experiment
from test_operations_api import _claim_role


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference-package")
    runner, _, registry, _, settings, task = _runtime(root)
    trained, frame = _train_lr_experiment(runner, registry, root, task)
    app = create_app(settings)
    repo = ModelingRepository(settings.db_path)
    artifact = repo.get_model_artifact(
        repo.get_experiment(trained.output["experiment_id"]).artifact_id
    )
    spec = StrategySpec(
        strategy_type="approval",
        rules=(
            StrategyRuleSpec(
                rule_id="low_pd",
                priority=1,
                condition={
                    "op": "compare",
                    "field": "pd",
                    "operator": "<",
                    "value": 0.5,
                },
                action=StrategyAction(type="approval", reason_code="LOW_PD"),
            ),
        ),
        default_action=StrategyAction(type="reject", reason_code="HIGH_PD"),
    )
    strategy = replace(
        build_strategy_from_spec(spec, score_col="pd", description="test"),
        id="online-strategy",
    )
    StrategyRepository(settings.db_path).create_strategy(task.id, strategy)
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status='validated', status='draft' WHERE id='online-strategy'"
        )
    request = PackageRequest(
        strategy_id="online-strategy",
        strategy_version=1,
        model_artifact_id=artifact.id,
        decision_node="underwriting",
        score_field="pd",
        raw_schema=[{"name": "x1", "type": "number"}, {"name": "x2", "type": "number"}],
    )
    store = app.state.reference_decision.packages
    package_hash, manifest = store.build(request, actor_id="test-maker")
    return app, store, request, package_hash, manifest, artifact, frame, spec


def test_authenticated_package_freezes_source_and_matches_offline(packaged):
    app, store, request, package_hash, manifest, artifact, frame, spec = packaged
    assert store.get(package_hash) == manifest
    assert store.build(request, actor_id="test-maker")[0] == package_hash
    scorer = _ModelArtifactScorer(artifact, base_dir=store.root / package_hash)
    rows = frame.iloc[[0, 17, 79]][["x1", "x2"]]
    offline = rows.copy()
    offline["pd"] = scorer.raw_score(rows)
    decisions = evaluate_strategy_frame(offline, spec).to_frame()
    for index, (_, row) in enumerate(rows.iterrows()):
        result = evaluate(manifest, store.root / package_hash, row.to_dict())
        assert result["score"] == pytest.approx(offline.iloc[index]["pd"])
        assert result["action"]["type"] == decisions.iloc[index]["action_type"]
        assert result["action"]["reason_code"] == decisions.iloc[index]["reason_code"]
    source = (
        app.state.settings.tasks_dir
        / ModelingRepository(app.state.settings.db_path)
        .get_experiment(artifact.experiment_id)
        .task_id
        / "modeling_artifacts"
        / artifact.model_path
    )
    original = source.read_bytes()
    try:
        source.write_bytes(b"changed after publication")
        assert store.get(package_hash) == manifest
        assert (
            evaluate(manifest, store.root / package_hash, {"x1": 0.1, "x2": 0.2})[
                "score"
            ]
            >= 0
        )
    finally:
        source.write_bytes(original)


def test_package_corruption_rejected_before_deserialization(packaged, monkeypatch):
    _, store, _, package_hash, manifest, _, _, _ = packaged
    target = store.root / package_hash / manifest["files"][0]["path"]
    original = target.read_bytes()
    try:
        target.write_bytes(b"tampered pickle")
        with pytest.raises(DecisionError, match="package_file_integrity_failed"):
            store.get(package_hash)
    finally:
        target.write_bytes(original)


@pytest.mark.parametrize(
    "features",
    [
        {"x1": 1},
        {"x1": 1, "x2": 2, "pd": 0},
        {"x1": True, "x2": 2},
        {"x1": float("nan"), "x2": 2},
        {"x1": None, "x2": 2},
        {"x1": 10**400, "x2": 2},
    ],
)
def test_invalid_or_forged_features_never_reach_scoring(packaged, features):
    _, store, _, package_hash, manifest, _, _, _ = packaged
    with pytest.raises(DecisionError):
        evaluate(manifest, store.root / package_hash, features)


def test_fitted_steps_require_raw_inputs_and_block_derived_overrides():
    steps = [
        {
            "kind": "missing_indicator",
            "columns": ["income"],
            "params": {"income": "income__missing"},
        },
        {"kind": "impute", "columns": ["income"], "params": {"income": 5}},
        {
            "kind": "normalize",
            "columns": ["income"],
            "params": {"income": {"mean": 2, "std": 1}},
        },
    ]
    assert feature_outputs(["income"], steps)[0] == {"income", "income__missing"}
    with pytest.raises(DecisionError, match="derived_features"):
        feature_outputs(["income", "income__missing"], steps)
    with pytest.raises(DecisionError, match="unsupported_preprocessing"):
        feature_outputs(["income"], [{"kind": "mystery", "columns": ["income"]}])


def test_real_worker_uses_frozen_package_without_injection(packaged):
    app, _, _, package_hash, _, _, _, _ = packaged
    result = app.state.reference_decision.evaluate(
        package_hash, {"x1": 0.1, "x2": 0.2}, 20
    )
    assert result["action"]["reason_code"] == "LOW_PD"
    assert result["score"] < 0.5


def test_ledger_is_durable_fenced_idempotent_and_does_not_store_features(packaged):
    app, _, _, package_hash, _, _, _, _ = packaged
    ledger = DecisionLedger(app.state.settings.db_path)
    scope = "ledger-test"
    owner, _ = ledger.claim(scope, "same", "input", package_hash, 1)
    with pytest.raises(DecisionError, match="decision_in_progress"):
        ledger.claim(scope, "same", "input", package_hash, 1)
    with pytest.raises(DecisionError, match="idempotency_payload_conflict"):
        ledger.claim(scope, "same", "changed", package_hash, 1)
    with connect(app.state.settings.db_path) as conn:
        conn.execute(
            "UPDATE reference_decisions SET lease_until=? WHERE environment=?",
            (time.time() - 1, scope),
        )
    owner2, _ = ledger.claim(scope, "same", "input", package_hash, 1)
    with pytest.raises(DecisionError, match="decision_lease_lost"):
        ledger.finish(scope, "same", owner, {"timing_ms": {}})
    response = ledger.finish(
        scope, "same", owner2, {"decision": "reject", "timing_ms": {}}
    )
    fresh = DecisionLedger(app.state.settings.db_path)
    assert fresh.existing(scope, "same", "input") == response
    assert fresh.claim(scope, "same", "input", package_hash, 1) == (None, response)


def test_package_api_uses_existing_role_authentication(packaged):
    app, _, request, package_hash, _, _, _, _ = packaged
    client = TestClient(app)
    assert client.get("/api/reference-decision/capabilities").status_code == 403
    _claim_role(app, client, "checker")
    assert (
        client.post(
            "/api/reference-decision/packages", json=request.model_dump()
        ).status_code
        == 403
    )
    assert (
        client.get(f"/api/reference-decision/packages/{package_hash}").status_code
        == 200
    )
    assert client.get("/api/reference-decision/status").json()["state"] == "unavailable"


def test_parameter_and_manifest_mutation_fail_closed(packaged):
    app, store, request, _, _, artifact, _, _ = packaged
    repo = ModelingRepository(app.state.settings.db_path)
    original = artifact.params
    try:
        repo.set_model_artifact_params(
            artifact.id,
            {
                **original,
                "preprocessing_steps": [{"kind": "unknown", "columns": ["x1"]}],
            },
        )
        with pytest.raises(DecisionError, match="unsupported_preprocessing"):
            store.build(request, actor_id="test-maker")
    finally:
        repo.set_model_artifact_params(artifact.id, original)


def test_same_request_concurrent_workers_have_one_owner(packaged):
    app, _, _, package_hash, _, _, _, _ = packaged
    ledger = DecisionLedger(app.state.settings.db_path)

    def claim(_):
        try:
            return ledger.claim("parallel", "once", "same", package_hash, 30)[0]
        except DecisionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, range(2)))
    assert claims.count("decision_in_progress") == 1


def test_replacing_trained_model_with_arbitrary_ensemble_is_rejected(packaged):
    from sklearn.isotonic import IsotonicRegression

    app, store, request, _, _, artifact, _, _ = packaged
    repo = ModelingRepository(app.state.settings.db_path)
    experiment = repo.get_experiment(artifact.experiment_id)
    source = app.state.settings.tasks_dir / experiment.task_id / "modeling_artifacts"
    original = (source / artifact.model_path).read_bytes()
    names = ["ensemble-a.joblib", "ensemble-b.joblib"]
    for name in names:
        (source / name).write_bytes(original)
    members = [
        {"artifact_id": f"member-{i}", "algorithm": "lr", "model_path": name}
        for i, name in enumerate(names)
    ]
    joblib.dump(
        {"members": members, "weights": [0.5, 0.5]}, source / "ensemble-root.joblib"
    )
    joblib.dump(
        {
            "method": "isotonic",
            "calibrator": IsotonicRegression(out_of_bounds="clip").fit(
                [0, 0.5, 1], [0, 0.4, 1]
            ),
        },
        source / "calibration.joblib",
    )
    params = {
        **artifact.params,
        "ensemble_member_artifact_ids": [m["artifact_id"] for m in members],
        "calibration": {"path": "calibration.joblib"},
    }
    try:
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE model_artifacts SET algorithm='ensemble', model_path='ensemble-root.joblib', params_json=? WHERE id=?",
                (json.dumps(params), artifact.id),
            )
        with pytest.raises(
            DecisionError, match="native_model_metadata_authentication_failed"
        ):
            store.build(
                request.model_copy(update={"score_product": "calibrated_pd"}),
                actor_id="test-maker",
            )
    finally:
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE model_artifacts SET algorithm=?, model_path=?, params_json=? WHERE id=?",
                (
                    artifact.algorithm,
                    artifact.model_path,
                    json.dumps(artifact.params),
                    artifact.id,
                ),
            )


def test_new_derived_recipe_inputs_exclude_intermediate_and_population_values():
    steps = [
        {
            "kind": "derive",
            "columns": ["a", "b"],
            "params": {"recipe": [{"kind": "ratio", "num": "a", "den": "b"}]},
        },
        {
            "kind": "fitted_rank",
            "columns": ["a_ratio_b"],
            "params": {"column": "a_ratio_b", "values": [0, 1], "ranks": [0.5, 1]},
        },
        {
            "kind": "group_aggregate",
            "columns": ["g"],
            "params": {"group": "g", "value": "training_only_value", "aggs": ["mean"]},
        },
        {
            "kind": "derive",
            "columns": ["date"],
            "params": {"recipe": [{"kind": "month", "col": "date"}]},
        },
    ]
    output, _ = feature_outputs(["a", "b", "g", "date"], steps)
    assert {
        "a_ratio_b",
        "a_ratio_b__rank",
        "training_only_value_by_g_mean",
        "date__month",
    } <= output
    with pytest.raises(DecisionError):
        feature_outputs(["a", "b", "g", "date", "a_ratio_b"], steps)


def test_selected_exact_final_artifact_can_be_packaged(packaged):
    app, store, request, package_hash, _, artifact, _, _ = packaged
    repo = ModelingRepository(app.state.settings.db_path)
    try:
        repo.set_experiment_status(artifact.experiment_id, "selected")
        assert store.build(request, actor_id="maker")[0] == package_hash
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET artifact_id=NULL WHERE id=?",
                (artifact.experiment_id,),
            )
        with pytest.raises(DecisionError, match="completed_registered_model_required"):
            store.build(request, actor_id="maker")
    finally:
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE experiments SET status='trained', artifact_id=? WHERE id=?",
                (artifact.id, artifact.experiment_id),
            )


def test_source_tampered_before_build_fails_without_deserialization(
    packaged, monkeypatch
):
    app, store, request, _, _, artifact, _, _ = packaged
    experiment = ModelingRepository(app.state.settings.db_path).get_experiment(
        artifact.experiment_id
    )
    source = (
        app.state.settings.tasks_dir
        / experiment.task_id
        / "modeling_artifacts"
        / artifact.model_path
    )
    original = source.read_bytes()
    try:
        source.write_bytes(b"malicious before the package existed")
        monkeypatch.setattr(
            joblib,
            "load",
            lambda *a, **k: pytest.fail("unverified pickle reached deserializer"),
        )
        with pytest.raises(DecisionError, match="native_model_source_integrity_failed"):
            store.build(request, actor_id="maker")
    finally:
        source.write_bytes(original)


def test_missing_original_producer_evidence_cannot_be_backfilled(packaged):
    app, store, request, _, _, artifact, _, _ = packaged
    with connect(app.state.settings.db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM task_artifacts WHERE kind='modeling_native_model_closure'"
        ).fetchall()
        conn.execute(
            "DELETE FROM task_artifacts WHERE kind='modeling_native_model_closure'"
        )
    try:
        with pytest.raises(DecisionError, match="evidence_unknown_retrain_required"):
            store.build(request, actor_id="maker")
    finally:
        with connect(app.state.settings.db_path) as conn:
            conn.executemany(
                "INSERT INTO task_artifacts VALUES (?,?,?,?,?,?,?,?)",
                [tuple(row) for row in rows],
            )


def test_real_ensemble_and_calibrator_producer_authenticate_all_source_bytes(
    tmp_path, packaged, monkeypatch
):
    from marvis.plugins.manifest import ToolRef

    runner, _, registry, _, settings, task = _runtime(tmp_path)
    initial, _ = _train_lr_experiment(runner, registry, tmp_path, task)
    repo = ModelingRepository(settings.db_path)
    dataset_id = repo.get_experiment(initial.output["experiment_id"]).config.dataset_id
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset_id,
            "recipe": "ensemble",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"base_recipe": "lr", "n_members": 2},
            "seed": 23,
        },
        task_id=task.id,
    )
    assert trained.ok, trained.error
    artifact_id = trained.output["artifact_id"]
    calibrated = runner.invoke(
        ToolRef("modeling", "calibrate_model"),
        {
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "method": "sigmoid",
            "split": "test",
            "min_samples": 20,
            "n_bins": 5,
        },
        task_id=task.id,
    )
    assert calibrated.ok, calibrated.error
    strategy = replace(
        build_strategy_from_spec(
            packaged[-1], score_col="pd", description="native closure"
        ),
        id="online-strategy",
    )
    StrategyRepository(settings.db_path).create_strategy(task.id, strategy)
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status='validated', status='draft' WHERE id='online-strategy'"
        )
    store = create_app(settings).state.reference_decision.packages
    request = packaged[2].model_copy(
        update={"model_artifact_id": artifact_id, "score_product": "calibrated_pd"}
    )
    # Package construction has no pickle deserialization at all.
    with monkeypatch.context() as patch:
        patch.setattr(
            joblib,
            "load",
            lambda *a, **k: pytest.fail("package builder deserialized pickle"),
        )
        package_hash, manifest = store.build(request, actor_id="maker")
    assert len(manifest["files"]) == 4
    with store.snapshot(package_hash) as (frozen, directory):
        assert 0 <= evaluate(frozen, directory, {"x1": 0.1, "x2": 0.2})["score"] <= 1
    source = settings.tasks_dir / task.id / "modeling_artifacts"
    for entry in manifest["files"]:
        path = source / entry["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(b"changed before build")
            with pytest.raises(
                DecisionError, match="native_model_source_integrity_failed"
            ):
                store.build(request, actor_id="maker")
        finally:
            path.write_bytes(original)
    # A second calibration creates a new immutable receipt without destroying
    # the first package's calibrated artifact or original producer records.
    repeated = runner.invoke(
        ToolRef("modeling", "calibrate_model"),
        {
            "artifact_id": artifact_id,
            "dataset_id": dataset_id,
            "method": "sigmoid",
            "split": "test",
            "min_samples": 20,
            "n_bins": 5,
        },
        task_id=task.id,
    )
    assert repeated.ok, repeated.error
    assert repeated.output["calibration_path"] != calibrated.output["calibration_path"]
    assert store.build(request, actor_id="maker")[0] != package_hash
    assert store.get(package_hash) == manifest


def test_calibration_cannot_certify_or_overwrite_concurrently_changed_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from marvis.packs.modeling import calibrate_tools
    from marvis.packs.modeling.errors import ModelingError
    from marvis.repositories.task_artifacts import TaskArtifactRepository

    runner, _, registry, _, settings, task = _runtime(tmp_path)
    trained, _ = _train_lr_experiment(runner, registry, tmp_path, task)
    repo = ModelingRepository(settings.db_path)
    artifact = repo.get_model_artifact(trained.output["artifact_id"])
    experiment = repo.get_experiment(artifact.experiment_id)
    original_fit = calibrate_tools._fit_calibrator
    changed = {**artifact.params, "concurrent_note": "new declaration"}
    before = TaskArtifactRepository(settings.db_path).list_for_task(task.id)
    def change_while_fitting(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        repo.set_model_artifact_params(artifact.id, changed)
        return result
    monkeypatch.setattr(calibrate_tools, "_fit_calibrator", change_while_fitting)
    with pytest.raises(ModelingError, match="changed_during_calibration"):
        calibrate_tools.tool_calibrate_model({"artifact_id": artifact.id, "dataset_id": experiment.config.dataset_id,
            "method": "sigmoid", "split": "test", "min_samples": 20, "n_bins": 5},
            SimpleNamespace(task_id=task.id, workspace=settings.workspace, datasets_root=settings.datasets_dir, seed=None))
    assert repo.get_model_artifact(artifact.id).params == changed
    assert TaskArtifactRepository(settings.db_path).list_for_task(task.id) == before
    assert not list((settings.tasks_dir / task.id / "modeling_artifacts").glob("*.calibration.*.joblib"))
