"""Authenticated model consumers retain the limits of native timing evidence."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.app import create_app
from marvis.data.asof_join import AsOfJoinEngine
from marvis.data.time_contracts import DatasetTimeContract
from marvis.db import StrategyRepository
from marvis.db_schema import connect
from marvis.packs.strategy.dsl import StrategyAction, StrategySpec
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.plugins.manifest import ToolRef
from marvis.reference_decision.contracts import DecisionError, PackageRequest
from marvis.reference_decision.evaluation import evaluate
from marvis.reference_decision.readiness import package_readiness
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings

from test_data_time_contracts import spec, tc
from test_decision_twin_batch import batch as batch, _replay
from test_feature_preprocessing_provenance import scenario as fitted_scenario
from test_feature_time_evidence import scenario as temporal_scenario
from test_join_time_evidence import register, run_join
from test_reference_decision import packaged as packaged


def train(runner, registry, dataset_id, features):
    result = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset_id,
            "recipe": "lr",
            "features": features,
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "seed": 7,
        },
        task_id="task-feature",
    )
    assert result.ok, result.error
    return ModelingRepository(registry._repo.db_path).get_model_artifact(
        result.output["artifact_id"]
    )


def test_no_recorded_steps_does_not_mean_preprocessing_was_unnecessary(packaged):
    _, store, _, _, manifest, artifact, *_ = packaged
    assert artifact.params["preprocessing_assurance"] == "unknown"
    assert artifact.params["preprocessing_chain_traceable"] is False
    result = package_readiness(store, artifact.id)
    assert result["state"] == "authenticated"
    assert result["preprocessing"]["state"] == "unknown"
    assert result["feature_time_evidence"] == artifact.params["feature_time_evidence"]
    assert result["feature_time_evidence"]["assurance"] == "unknown"
    assert result["scope"] == "local_reference_package_preparation"
    assert (
        result["preprocessing"]["scope"] == "recorded_transforms_and_fitting_membership"
    )
    assert result["build_ready"] is False
    assert (
        manifest["model"]["params"]["feature_time_evidence"]
        == result["feature_time_evidence"]
    )
    assert manifest["assurance"] == "local_reference_only"


def test_readiness_does_not_repin_a_previously_frozen_training_source(tmp_path):
    runner, registry, _, source, transformed = fitted_scenario(tmp_path)
    artifact = train(runner, registry, transformed.id, ["x"])
    before = registry.get(source.id)
    store = create_app(
        build_settings(registry.datasets_root.parent)
    ).state.reference_decision.packages
    result = package_readiness(store, artifact.id)
    assert result["state"] == "authenticated", result
    assert registry.get(source.id) == before
    assert result["preprocessing"]["state"] == "training_only"
    assert result["feature_time_evidence"] == artifact.params["feature_time_evidence"]


@pytest.mark.parametrize(
    ("kind", "expected", "preprocessing"),
    [
        ("native", "verified", "unknown"),
        ("inferred", "inferred", "unknown"),
        ("normalize", "unknown", "training_only"),
        ("right_join", "unknown", "unknown"),
        ("row_local", "verified", "row_local"),
    ],
)
def test_native_timing_scope_survives_delivery_package_and_readiness(
    tmp_path,
    kind,
    expected,
    preprocessing,
):
    runner, registry, backend, native = temporal_scenario.__wrapped__(tmp_path)
    dataset_id, feature = native.dataset.id, "asof__amount"
    if kind == "inferred":
        original = json.loads(native.evidence_path.read_text())
        fc = DatasetTimeContract.model_validate_json(
            json.dumps(original["feature_contract"])
        )
        fc = fc.model_copy(update={"available_at": tc("available", basis="inferred")})
        engine = AsOfJoinEngine(
            registry,
            TaskArtifactRepository(registry._repo.db_path),
            workspace_root=registry.datasets_root.parent,
        )
        exploratory = engine.execute(
            task_id="task-feature",
            decision_contract=DatasetTimeContract.model_validate_json(
                json.dumps(original["decision_contract"])
            ),
            feature_contract=fc,
            spec=spec(mode="exploration"),
        )
        dataset_id = exploratory.dataset.id
    elif kind in {"normalize", "row_local"}:
        tool = "normalize" if kind == "normalize" else "cross_features"
        inputs = {"dataset_id": dataset_id}
        inputs.update(
            {"columns": [feature], "method": "zscore", "split_col": "split"}
            if kind == "normalize"
            else {"recipe": [{"kind": "transform", "col": feature, "ops": ["log1p"]}]}
        )
        result = runner.invoke(ToolRef("feature", tool), inputs, task_id="task-feature")
        assert result.ok, result.error
        dataset_id = result.output["result_dataset_id"]
        if kind == "row_local":
            feature += "__log1p"
    elif kind == "right_join":
        right = register(
            registry,
            pd.DataFrame(
                {
                    "subject": [f"s{i}" for i in range(40)],
                    "right_amount": [float(i % 5) for i in range(40)],
                }
            ),
            "untimed",
        )
        joined, _, _ = run_join(registry, backend, native.dataset, [right])
        dataset_id, feature = joined.id, "right_amount"
    artifact = train(runner, registry, dataset_id, [feature])
    evidence = artifact.params["feature_time_evidence"]
    assert evidence["assurance"] == expected
    if kind == "normalize":
        assert "fitted_parameter_availability_not_recorded" in evidence["reasons"]
    if kind == "right_join":
        assert "ordinary_join_right_field_availability_unknown" in evidence["reasons"]
    assert evidence["scope"] == "selected_field_availability_at_recorded_decisions"
    assert "model_evaluation" in evidence["excluded_assurances"]

    app = create_app(build_settings(registry.datasets_root.parent))
    store = app.state.reference_decision.packages
    ready = package_readiness(store, artifact.id)
    assert ready["state"] == "authenticated", ready
    assert ready["preprocessing"]["state"] == preprocessing
    assert ready["feature_time_evidence"] == evidence
    assert ready["build_ready"] is False
    strategy = replace(
        build_strategy_from_spec(
            StrategySpec(
                strategy_type="approval",
                rules=(),
                default_action=StrategyAction(
                    type="approval", reason_code="REFERENCE_ONLY"
                ),
            ),
            score_col="pd",
            description="consumer evidence fixture",
        ),
        id="evidence-strategy",
    )
    StrategyRepository(store.settings.db_path).create_strategy("task-feature", strategy)
    with connect(store.settings.db_path) as conn:
        conn.execute(
            "UPDATE strategies SET asset_status='validated' WHERE id=?", (strategy.id,)
        )
    request = PackageRequest(
        strategy_id=strategy.id,
        strategy_version=1,
        model_artifact_id=artifact.id,
        decision_node="underwriting",
        score_field="pd",
        raw_schema=[
            {"name": item["name"], "type": "number"}
            for item in ready["raw_requirements"]
        ],
    )
    package_hash, _ = store.build(request, actor_id="test-maker")
    with store.snapshot(package_hash) as (manifest, directory):
        assert manifest["model"]["params"]["feature_time_evidence"] == evidence
        assert manifest["assurance"] == "local_reference_only"
        # Unknown or inferred historical availability remains usable for honest
        # local scoring. No evidence is changed to force a package to be usable.
        result = evaluate(
            manifest,
            directory,
            {item["name"]: 1.0 for item in ready["raw_requirements"]},
        )
        assert 0 <= result["score"] <= 1
        assert result["action"]["reason_code"] == "REFERENCE_ONLY"
    delivered = runner.invoke(
        ToolRef("modeling", "post_training_action"),
        {
            "experiment_id": artifact.experiment_id,
            "sample_dataset_id": dataset_id,
            "actions": ["export_pmml"],
        },
        task_id="task-feature",
    )
    assert delivered.ok, delivered.error
    card = json.loads(Path(delivered.output["model_card_path"]).read_text())
    assert card["training"]["feature_time_evidence"] == evidence
    assert any("特征缺少可认证的历史可得时间" in item for item in card["limitations"]) is (
        expected != "verified"
    )
    assert card["training"]["parameter_time_evidence"]["assurance"] == "unknown"
    assert any("模型及拟合参数的历史可得时间未验证" in item for item in card["limitations"])


def test_edited_unknown_timing_cannot_be_displayed_as_authenticated(packaged):
    _, store, request, package_hash, _, artifact, *_ = packaged
    repo = ModelingRepository(store.settings.db_path)
    original = artifact.params
    forged = {**original, "feature_time_evidence": {"assurance": "verified"}}
    try:
        repo.set_model_artifact_params(artifact.id, forged)
        ready = package_readiness(store, artifact.id)
        assert ready["state"] == "blocked"
        assert ready["feature_time_evidence"]["assurance"] == "unknown"
        with pytest.raises(
            DecisionError, match="native_model_metadata_authentication_failed"
        ):
            store.build(request, actor_id="test-maker")
        with store.snapshot(package_hash) as (manifest, directory):
            assert (
                manifest["model"]["params"]["feature_time_evidence"]
                == original["feature_time_evidence"]
            )
            assert evaluate(manifest, directory, {"x1": 0.1, "x2": 0.2})["score"] >= 0
    finally:
        repo.set_model_artifact_params(artifact.id, original)


def test_historical_replay_of_untimed_model_remains_retrospective(batch):
    _, material, contract, _, _ = batch

    # Replay uses an immutable native model whose training feature timing is
    # unknown; historical per-record timestamps cannot establish model timing.
    manifest = material.packages.get(contract.scenarios[0].package_hash)
    assert (
        manifest["model"]["params"]["feature_time_evidence"]["assurance"] == "unknown"
    )
    result = _replay(material, contract)["payload"]
    assert result["package_time_scope"] == "retrospective_policy_simulation"
    assert result["historical_package_availability"] == "not_established"
    assert result["causal_assessment"]["causal_gain_verified"] is False
    assert result["automatic_action_permitted"] is False
