"""Real learners consume selected fold frames without using outer holdouts for selection."""

import base64

import numpy as np
import pandas as pd
import pytest

from marvis.packs.modeling.fold_policy import build_fold_selection_plan, selection_session
from marvis.packs.modeling.fold_tuning import tune_fold_selection
from marvis.plugins.manifest import ToolRef
from tests.test_feature_pack import _runtime


def _source(tmp_path, target_type="binary", mutate=None, return_runtime=False):
    runner, registry, _, _ = _runtime(tmp_path)
    rng = np.random.default_rng(413)
    n = 360
    target = rng.normal(size=n) if target_type == "continuous" else np.arange(n) % (3 if target_type == "multiclass" else 2)
    frame = pd.DataFrame({"x": target * .3 + rng.normal(size=n), "z": rng.normal(size=n),
        "y": target, "split": ["train"] * 240 + ["test"] * 60 + ["oot"] * 60,
        "borrower": 2**53 + np.arange(n) // 3})
    if mutate:
        mutate(frame)
    path = tmp_path / "source.parquet"
    frame.to_parquet(path, index=False)
    source = registry.register_from_upload("task-feature", path, role="sample")
    references, selected = [], ["x", "z"]
    for tool, parameters in (("screen_features", {"top_k": 1, "leakage_ks": 1.0}),
                             ("select_features", {"iv_min": 0.0, "top_k": 1, "vif_max": 1e9})):
        result = runner.invoke(ToolRef("modeling", tool), {
            "dataset_id": source.id, "features": ["x", "z"], "target_col": "y",
            "split_col": "split", "target_type": target_type, **parameters,
        }, task_id="task-feature")
        assert result.ok, result.error
        references.append(result.output["selection_evidence_ref"])
        selected = result.output["selected"]
    inputs = {"dataset_id": source.id, "target_col": "y", "features": selected,
        "split_col": "split", "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "selection_evidence_refs": references,
        "fold_selection": {"source_dataset_id": source.id, "candidates": ["x", "z"]}}
    plan = build_fold_selection_plan(registry, "task-feature", inputs)
    return (runner, registry, plan, frame) if return_runtime else (registry, plan, frame)


def _tune(registry, plan, target_type, recipe, **kwargs):
    with selection_session(registry, plan, target_type=target_type, group_columns=["borrower"]) as session:
        return tune_fold_selection(session, plan, recipe=recipe, seed=19, cv_folds=3, n_trials=1,
            base_params={"valid_group_cols": ["borrower"]}, **kwargs)


def _rows(entry, count):
    return np.flatnonzero(np.unpackbits(np.frombuffer(base64.b64decode(entry["membership"]), dtype=np.uint8),
                                      bitorder="little", count=count))


@pytest.mark.parametrize("target_type,recipe", [("binary", "lr"), ("continuous", "lr_regressor"),
                                               ("multiclass", "lr_multiclass")])
def test_real_cv_selects_full_candidates_and_keeps_groups_and_excludes_outer_labels(tmp_path, monkeypatch, target_type, recipe):
    from marvis.packs.modeling import fold_selection

    registry, plan, frame = _source(tmp_path, target_type)
    reads = []
    original = fold_selection._ProjectedBackend.read_frame

    def read(self, path, *, columns=None):
        reads.append((tuple(self.columns if columns is None else columns), self.positions.copy()))
        return original(self, path, columns=columns)

    monkeypatch.setattr(fold_selection._ProjectedBackend, "read_frame", read)
    result = _tune(registry, plan, target_type, recipe)
    assert len(result.selected_features) == 1
    assert result.n_trials == 1
    assert result.fold_selection_evidence["oot_evaluation"] == "not_used_for_selection_or_tuning"
    for columns, positions in reads:
        if "y" in columns:
            assert (positions < 240).all()
    for fold in result.fold_selection_evidence["folds"]:
        proof = fold["selection"]
        assert proof["candidates"] == ["x", "z"]
        fit = _rows(proof["memberships"]["fit"], len(frame))
        test = _rows(proof["memberships"]["test"], len(frame))
        assert not set(frame.loc[fit, "borrower"]) & set(frame.loc[test, "borrower"])
    final = result.fold_selection_evidence["final_selection"]
    assert np.array_equal(_rows(final["memberships"]["fit"], len(frame)), np.arange(240))


def test_heldout_label_changes_do_not_change_that_fold_selection_or_fit_params(tmp_path):
    from marvis.packs.modeling.tune import _cv_folds

    registry, plan, frame = _source(tmp_path / "before")
    first = _tune(registry, plan, "binary", "lr")
    heldout = _cv_folds(frame.iloc[:240], cv_folds=3, seed=19, group_cols=["borrower"])[0]

    def mutate(frame):
        frame.loc[heldout, "y"] = 1 - frame.loc[heldout, "y"]

    registry, plan, _ = _source(tmp_path / "after", mutate=mutate)
    second = _tune(registry, plan, "binary", "lr")
    assert first.fold_selection_evidence["folds"][0]["selection"]["selected"] == second.fold_selection_evidence["folds"][0]["selection"]["selected"]
    assert first.trials[0]["cv_fold_metrics"][0]["params"] == second.trials[0]["cv_fold_metrics"][0]["params"]


@pytest.mark.parametrize("target_type,recipe", [("binary", "lr"), ("continuous", "lr_regressor"),
    ("multiclass", "lr_multiclass"), ("binary", "lgb"), ("binary", "catboost"),
    ("continuous", "lgb_regressor"), ("continuous", "xgb_regressor")])
def test_actual_tool_child_cache_and_final_training_bind_the_same_features(tmp_path, target_type, recipe):
    from marvis.packs.modeling.fold_tuning_evidence import load_fold_tuning
    from marvis.repositories.modeling import ModelingRepository
    from marvis.repositories.task_artifacts import TaskArtifactRepository
    from marvis.repositories.plugins import PluginRepository

    runner, registry, plan, _ = _source(tmp_path, target_type, return_runtime=True)
    audit = PluginRepository(registry._repo.db_path)
    tree = recipe in {"lgb", "catboost", "lgb_regressor", "xgb_regressor"}
    parameters = {"valid_group_cols": ["borrower"]}
    if recipe == "lgb":
        parameters["scale_pos_weight"] = "auto"
    if recipe.startswith(("lgb", "xgb")):
        parameters["monotone_constraints"] = {"x": 1, "z": 0}
    inputs = {"dataset_id": plan["training_dataset_id"], "features": ["x"], "target_col": "y",
        "split_col": "split", "split_values": plan["split_values"], "recipe": recipe, "seed": 19,
        "n_trials": 3 if tree else 1, "cv_folds": 3, "params": parameters,
        "early_stopping_rounds": 2, "max_boost_round": 10,
        "selection_evidence_refs": plan["references"],
        "fold_selection": {"source_dataset_id": plan["source_dataset_id"], "candidates": ["x", "z"]}}
    result = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), inputs, task_id="task-feature")
    assert result.ok, result.error
    repeat = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), inputs, task_id="task-feature")
    assert repeat.ok, repeat.error
    assert repeat.output == result.output
    assert len(audit.list_audit(kind="modeling.tuning_checkpoint.saved")) == 1
    assert len(audit.list_audit(kind="modeling.tuning_checkpoint.hit")) == 1
    ref = result.output["fold_tuning_evidence_refs"][recipe]
    if recipe == "lgb_regressor":
        denied = runner.invoke(ToolRef("modeling", "train_model"), {
            "dataset_id": inputs["dataset_id"], "recipe": recipe,
            "features": result.output["features_by_recipe"][recipe], "target_col": "y",
            "split_col": "split", "split_values": inputs["split_values"], "seed": 19,
            "params": result.output["best_params"], "scenario": "income",
            "fold_tuning_evidence_ref": ref,
            "selection_evidence_refs": plan["references"],
        }, task_id="task-feature")
        assert not denied.ok and "scenario changes frozen fold parameters" in str(denied.error)
    proof = load_fold_tuning(registry, "task-feature", ref)
    assert proof["features"] == result.output["features_by_recipe"][recipe]
    assert proof["evidence"]["oot_evaluation"] == "not_used_for_selection_or_tuning"
    trained = runner.invoke(ToolRef("modeling", "train_models"), {
        "dataset_id": inputs["dataset_id"], "features": inputs["features"], "target_col": "y",
        "split_col": "split", "split_values": inputs["split_values"], "recipes": [recipe], "seed": 19,
        "params": result.output["best_params"], "features_by_recipe": result.output["features_by_recipe"],
        "fold_tuning_evidence_refs": result.output["fold_tuning_evidence_refs"],
        "selection_evidence_refs": plan["references"],
    }, task_id="task-feature")
    assert trained.ok, trained.error
    from marvis.packs.modeling.experiment import ExperimentStore
    experiment = ExperimentStore(registry._repo.db_path).get(trained.output["best_experiment_id"])
    artifact = ModelingRepository(registry._repo.db_path).get_model_artifact(experiment.artifact_id)
    assert list(artifact.feature_list) == proof["features"]
    assert artifact.params["fold_tuning_evidence_ref"] == ref
    assert artifact.params["fold_tuning_evidence"]["historical_availability"] == "unknown"
    assert experiment.config.early_stopping_rounds is None
    from marvis.packs.modeling.fold_tuning_evidence import validate_fold_experiment
    assert validate_fold_experiment(registry, "task-feature", experiment, artifact) == proof["evidence"]
    if recipe == "lr":
        refit = runner.invoke(ToolRef("modeling", "select_experiment"), {
            "experiment_ids": [experiment.id], "refit_on_train_plus_test": True,
        }, task_id="task-feature")
        assert refit.ok, refit.error
        assert not refit.output["refit"]["applied"]
        assert refit.output["artifact_id"] == artifact.id
        delivered = runner.invoke(ToolRef("modeling", "post_training_action"), {
            "experiment_id": experiment.id, "sample_dataset_id": inputs["dataset_id"],
            "actions": [],
        }, task_id="task-feature")
        assert delivered.ok, delivered.error
    if tree:
        assert [row["search_stage"] for row in result.output["trials"]] == ["coarse", "coarse", "fine"]
        rounds_key = "iterations" if recipe == "catboost" else "num_boost_round"
        assert artifact.params[rounds_key] == proof["best_params"][rounds_key]
        for trial in result.output["trials"]:
            folds = trial["cv_fold_metrics"]
            assert trial["best_iteration"] == int(np.ceil(np.median([fold["best_iteration"] for fold in folds])))
        if recipe == "lgb":
            for i, fold in enumerate(result.output["trials"][0]["cv_fold_metrics"]):
                members = _rows(proof["evidence"]["folds"][i]["selection"]["memberships"]["fit"], plan["row_count"])
                # Labels are deterministic in this fixture; membership is native.
                labels = members % 2
                assert fold["params"]["scale_pos_weight"] == pytest.approx(float((labels == 0).sum() / (labels == 1).sum()))
            assert proof["best_params"]["scale_pos_weight"] == 1.0
        if recipe == "catboost":
            import joblib
            artifact_path, = tmp_path.rglob(artifact.model_path)
            model = joblib.load(artifact_path)
            assert model.tree_count_ == proof["best_params"]["iterations"]
    record = TaskArtifactRepository(registry._repo.db_path).get_for_task("task-feature", ref["artifact_id"])
    (registry.datasets_root.parent / record["path"]).write_bytes(b"tampered fold receipt")
    rejected = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), inputs, task_id="task-feature")
    assert not rejected.ok
    assert "fold tuning evidence content changed" in str(rejected.error)
    assert len(audit.list_audit(kind="modeling.tuning_checkpoint.hit")) == 1
    if recipe == "lr":
        delivered = runner.invoke(ToolRef("modeling", "post_training_action"), {
            "experiment_id": experiment.id, "sample_dataset_id": inputs["dataset_id"],
            "actions": [],
        }, task_id="task-feature")
        assert not delivered.ok and "fold tuning evidence content changed" in str(delivered.error)


def test_native_receipt_rejects_metric_tampering_and_cross_fold_cache_rebinding(tmp_path):
    import copy
    import json
    from marvis.packs.modeling.tune_checkpoint import _payload_checksum

    runner, registry, plan, _ = _source(tmp_path, return_runtime=True)
    inputs = {"dataset_id": plan["training_dataset_id"], "features": ["x"], "target_col": "y",
        "split_col": "split", "split_values": plan["split_values"], "recipe": "lr", "seed": 19,
        "n_trials": 1, "cv_folds": 3, "params": {"valid_group_cols": ["borrower"]},
        "selection_evidence_refs": plan["references"],
        "fold_selection": {"source_dataset_id": plan["source_dataset_id"], "candidates": ["x", "z"]}}
    first = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), inputs, task_id="task-feature")
    assert first.ok, first.error
    cache, = tmp_path.rglob("tuning_checkpoints/lr.json")
    original = json.loads(cache.read_text())
    altered = copy.deepcopy(original)
    altered["result"]["best_metrics"]["test_ks"] = .999999
    altered["result"]["trials"][0]["test_ks"] = .999999
    altered["checksum"] = _payload_checksum(altered["identity"], altered["result"])
    cache.write_text(json.dumps(altered))
    rejected = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), inputs, task_id="task-feature")
    assert not rejected.ok and "native fold receipt" in str(rejected.error)
    fourth = {**inputs, "cv_folds": 4}
    fresh = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), fourth, task_id="task-feature")
    assert fresh.ok, fresh.error
    envelope = json.loads(cache.read_text())
    envelope["result"] = original["result"]
    envelope["checksum"] = _payload_checksum(envelope["identity"], envelope["result"])
    cache.write_text(json.dumps(envelope))
    rejected = runner.invoke(ToolRef("modeling", "tune_hyperparameters"), fourth, task_id="task-feature")
    assert not rejected.ok and "native fold receipt" in str(rejected.error)


@pytest.mark.parametrize("key", ["fold_tuning_evidence", "fold_tuning_evidence_ref", "fold_selection_plan"])
def test_governed_training_rejects_caller_fold_proofs(key):
    from marvis.packs.modeling.evidence_tools import _reject_caller_owned_platform_params
    from marvis.packs.modeling.errors import ModelingError
    with pytest.raises(ModelingError, match="platform-owned"):
        _reject_caller_owned_platform_params({"nested": [{key: {"fabricated": True}}]})
