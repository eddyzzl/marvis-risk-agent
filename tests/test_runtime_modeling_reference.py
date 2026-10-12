"""Real native ToolRunner artifacts; independent math and installed learners only."""
from contextlib import closing
from pathlib import Path
import pickle
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_modeling import _model_values
from marvis.orchestrator.eval.runtime_archive_reader import _Unsupported
from marvis.orchestrator.eval.runtime_modeling_native_reference import read_linear_reference
from marvis.orchestrator.eval.runtime_modeling_reference import binary_reference, metrics_reference
from marvis.plugins.manifest import ToolRef
from test_feature_pack import _runtime


def _native(tmp_path, recipe, *, early_stop=None, fold=False):
    runner, registry, _, _ = _runtime(tmp_path)
    rng = np.random.RandomState(921)
    n = 300
    frame = pd.DataFrame({"x": rng.randn(n), "z": rng.randn(n),
        "split": ["train"] * 180 + ["test"] * 60 + ["oot"] * 60})
    kind = "regression" if recipe.endswith("regressor") else "multiclass" if recipe.endswith("multiclass") else "binary"
    frame["y"] = frame.x * .8 + frame.z * .1 + rng.randn(n) * .2 if kind == "regression" else np.arange(n) % (3 if kind == "multiclass" else 2)
    if kind == "multiclass":
        frame["y"] = frame.y.map({0: "A", 1: "B", 2: "C"})
    frame.loc[frame.split == "oot", "y"] = np.nan if kind == "regression" else None
    source = tmp_path / "synthetic-source.parquet"
    frame.to_parquet(source, index=False)
    dataset = registry.register_from_upload("task-feature", source, role="sample")
    params = {"num_boost_round": 6, "verbosity": -1} if recipe.startswith(("xgb", "lgb")) else {}
    if recipe.startswith("xgb"):
        params["verbosity"] = 0
    if early_stop:
        params["early_stopping_rounds"] = early_stop
    trained = runner.invoke(ToolRef("modeling", "train_models"), {
        "dataset_id": dataset.id, "features": ["x", "z"], "target_col": "y",
        "target_type": "continuous" if kind == "regression" else kind, "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "recipes": [recipe], "seed": 21, "params": params,
    }, task_id="task-feature")
    assert trained.ok, trained.error
    workspace = registry._repo.db_path.parent
    archive = tmp_path / "retained"
    shutil.copytree(workspace, archive / "workspace")
    conn = sqlite3.connect(archive / "workspace/marvis.sqlite")
    conn.row_factory = sqlite3.Row
    experiment = conn.execute("SELECT * FROM experiments WHERE id=?", (trained.output["best_experiment_id"],)).fetchone()
    session = SimpleNamespace(conn=conn, archive=archive,
        binding=SimpleNamespace(task_id="task-feature", dataset_id=dataset.id,
            fit_representations={experiment["id"]: {"representation": "compact_float32", "noncompact_features": []}}))
    return session, dict(experiment), frame, runner, registry


@pytest.mark.parametrize("recipe", ["lr", "lr_regressor", "lr_multiclass", "xgb", "lgb",
    "xgb_regressor", "lgb_regressor", "xgb_multiclass", "lgb_multiclass"])
def test_actual_native_profiles_recompute_without_producer_metrics_or_scoring(tmp_path, monkeypatch, recipe):
    session, experiment, frame, _, _ = _native(tmp_path, recipe)
    import marvis.packs.modeling.recipes.common as common
    import marvis.packs.modeling.scoring as scoring
    def forbidden(*args, **kwargs):
        raise AssertionError("producer is not an independent mathematical reference")
    for name in ("compute_model_metrics", "compute_regression_metrics", "compute_multiclass_model_metrics"):
        monkeypatch.setattr(common, name, forbidden)
    monkeypatch.setattr(scoring, "_ModelArtifactScorer", forbidden)
    with closing(session.conn):
        values, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["fit_rows"] == 180
    assert scope["test_evaluation"] == "independent_heldout"
    assert scope["historical_availability"] == "unknown"
    if recipe.endswith("regressor"):
        assert values["test_rmse"] > 0 and values["oot_rmse"] is None
    elif recipe.endswith("multiclass"):
        assert values["test_logloss"] > 0 and values["oot_macro_auc"] is None
    else:
        assert 0 <= values["test_auc"] <= 1 and values["oot_auc"] is None


@pytest.mark.parametrize("recipe", ["xgb", "lgb", "xgb_regressor", "lgb_regressor", "xgb_multiclass", "lgb_multiclass"])
def test_native_early_stopping_replays_only_frozen_outer_train(tmp_path, recipe):
    session, experiment, frame, _, _ = _native(tmp_path, recipe, early_stop=2)
    with closing(session.conn):
        values, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["fit_rows"] == 180
    assert values["oot_auc"] is None


def test_native_coefficients_cannot_be_replaced_by_rehashed_claimed_metrics(tmp_path):
    session, experiment, frame, _, _ = _native(tmp_path, "lr")
    changed = frame.copy()
    changed.loc[changed.split == "train", "y"] = 1 - changed.loc[changed.split == "train", "y"]
    with closing(session.conn), pytest.raises(ValueError, match="independent approved fit population"):
        _model_values(session, experiment["id"], experiment["artifact_id"], changed)


def test_archive_pickle_cannot_invoke_arbitrary_globals(tmp_path):
    target = tmp_path / "would-execute"
    class Evil:
        def __reduce__(self):
            return Path.write_text, (target, "executed")
    file = tmp_path / "model.pkl"
    file.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(_Unsupported):
        read_linear_reference(file, "lr", ["x"])
    assert not target.exists()


@pytest.mark.parametrize("structured", [False, True])
def test_archive_joblib_nested_object_array_cannot_execute_pickle(tmp_path, structured):
    import joblib
    target = tmp_path / "nested-array-executed"
    class Evil:
        def __reduce__(self):
            return Path.write_text, (target, "executed")
    file = tmp_path / "object-array.joblib"
    values = (np.array([(Evil(),)], dtype=[("payload", object)]) if structured
        else np.array([Evil()], dtype=object))
    joblib.dump(values, file)
    assert not target.exists(), "serialization must not execute the callback"
    with pytest.raises(_Unsupported):
        read_linear_reference(file, "lr", ["x"])
    assert not target.exists()


def test_archive_joblib_object_metadata_keeps_restricted_reader(tmp_path):
    import joblib
    from marvis.orchestrator.eval.runtime_modeling_native_reference import _LinearUnpickler

    values = np.array(["feature-name", None], dtype=object)
    file = tmp_path / "object-metadata.joblib"
    joblib.dump(values, file)
    with file.open("rb") as stream:
        restored = _LinearUnpickler(str(file), stream, ensure_native_byte_order=True).load()
    np.testing.assert_array_equal(restored, values)


def test_archive_joblib_declared_shape_is_bounded_before_allocation(tmp_path):
    from joblib.numpy_pickle import NumpyArrayWrapper
    wrapper = NumpyArrayWrapper(np.ndarray, (1_000_000_000,), "C", np.dtype("float64"), False)
    file = tmp_path / "oversize-array.joblib"
    file.write_bytes(pickle.dumps(wrapper))
    with pytest.raises(_Unsupported):
        read_linear_reference(file, "lr", ["x"])


def test_missing_oot_is_unknown_and_development_union_consumes_heldout():
    frame = pd.DataFrame({"split": ["train", "train", "test", "test", "oot"], "y": [0., 1., 0., 1., np.nan], "x": [.1, .8, .2, .7, .9]})
    config = {"split_col": "split", "split_values": {r: r for r in ("train", "test", "oot")}, "target_col": "y", "target_type": "binary", "seed": 42}
    values, masks = metrics_reference(frame, config, lambda part: part.x.to_numpy(), development=True)
    assert values["test_auc"] is values["oot_auc"] is values["overfit_flag"] is None
    assert masks["development"] == [True, True, True, True, False]
    assert binary_reference([.5, .5], [0, 1]) == {"ks": 0., "auc": .5}
