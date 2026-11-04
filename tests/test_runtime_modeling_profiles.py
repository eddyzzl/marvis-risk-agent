"""Actual producer files are independently read, refitted and checked for tampering."""
import pickle
from contextlib import closing
from pathlib import Path

import pytest
from test_runtime_modeling_reference import _native

from marvis.orchestrator.eval.runtime_archive_modeling import _model_values
from marvis.orchestrator.eval.runtime_archive_reader import _Unsupported
from marvis.orchestrator.eval.runtime_modeling_profile_reference import (
    read_numerical_payload,
)


@pytest.mark.slow
@pytest.mark.parametrize("recipe,params", [
    ("mlp", {"max_iter": 15, "hidden_layer_sizes": [4]}),
    ("mlp_regressor", {"max_iter": 15, "hidden_layer_sizes": [4]}),
    ("mlp_multiclass", {"max_iter": 15, "hidden_layer_sizes": [4]}),
    ("catboost", {"iterations": 8, "depth": 2}),
    ("scorecard", {"scorecard_max_bins": 4}),
    ("ensemble", {"base_recipe": "lr", "n_members": 2}),
])
def test_extended_native_model_artifacts_have_independent_fit_closure(tmp_path, monkeypatch, recipe, params):
    session, experiment, frame, _, _ = _native(tmp_path, recipe, extra_params=params)
    from marvis.packs.modeling import scoring
    from marvis.packs.modeling.recipes import common
    def forbidden(*args, **kwargs):
        pytest.fail("production scoring or metrics are not independent")
    monkeypatch.setattr(scoring, "_ModelArtifactScorer", forbidden)
    monkeypatch.setattr(common, "compute_model_metrics", forbidden)
    with closing(session.conn):
        values, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["profile"] == recipe and scope["fit_rows"] == 180
    assert scope["test_evaluation"] == "independent_heldout"
    assert any(v is not None for v in values.values())


@pytest.mark.slow
@pytest.mark.parametrize("recipe,params", [
    ("xgb", {}), ("lgb", {}), ("catboost", {"iterations": 8, "depth": 2}),
    ("mlp", {"max_iter": 15, "hidden_layer_sizes": [4]}),
    ("scorecard", {"scorecard_max_bins": 4}),
])
def test_weighted_native_profiles_recompute_weights_and_metrics(tmp_path, recipe, params):
    session, experiment, frame, _, _ = _native(tmp_path, recipe, extra_params=params, weighted=True)
    with closing(session.conn):
        metrics, _ = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert 0 <= metrics["weighted_test_auc"] <= 1
    assert metrics["weighted_oot_auc"] is None


@pytest.mark.slow
@pytest.mark.parametrize("recipe,params", [
    ("catboost", {"iterations": 15, "depth": 2}),
    ("mlp", {"max_iter": 40, "hidden_layer_sizes": [4], "early_stopping": True, "n_iter_no_change": 3}),
    ("ensemble", {"base_recipe": "mlp", "n_members": 2, "max_iter": 20, "hidden_layer_sizes": [4], "early_stopping": True}),
])
def test_non_tree_and_ensemble_early_stop_replay(tmp_path, recipe, params):
    # train_models owns the tree early-stop window; MLP owns its explicit
    # early_stopping/n_iter_no_change parameters inside the train population.
    session, experiment, frame, _, _ = _native(tmp_path, recipe, extra_params=params)
    with closing(session.conn):
        _, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["test_evaluation"] == "independent_heldout"


@pytest.mark.slow
@pytest.mark.parametrize("best_model", [True, False])
def test_catboost_native_test_selected_trees_are_not_called_independent_holdout(tmp_path, best_model):
    session, experiment, frame, _, _ = _native(
        tmp_path, "catboost", single=True, extra_params={"iterations": 10, "depth": 2, "use_best_model": best_model})
    with closing(session.conn):
        _, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["test_evaluation"] == ("used_for_model_selection" if best_model else "independent_heldout")


@pytest.mark.slow
@pytest.mark.parametrize("base", ["lgb", "xgb", "catboost"])
def test_weighted_tree_ensemble_authenticates_and_refits_each_member(tmp_path, base):
    params = {"base_recipe": base, "n_members": 2, "num_boost_round": 6}
    if base == "catboost":
        params["depth"] = 2
    session, experiment, frame, _, _ = _native(tmp_path, "ensemble", extra_params=params, weighted=True)
    with closing(session.conn):
        values, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert values["weighted_test_auc"] is not None
    assert scope["test_evaluation"] == ("used_for_model_selection" if base == "catboost" else "independent_heldout")


@pytest.mark.slow
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_native_receipts_authenticate_model_and_calibrator(tmp_path, method):
    session, experiment, frame, _, _ = _native(tmp_path, "lr", calibration=method)
    with closing(session.conn):
        _, scope = _model_values(session, experiment["id"], experiment["artifact_id"], frame)
    assert scope["calibration"]["method"] == method
    assert scope["calibration"]["fit_rows"] == 27
    assert scope["calibration"]["evaluated_on"] == "test"


@pytest.mark.slow
@pytest.mark.parametrize("recipe,params", [
    ("mlp", {"max_iter": 15, "hidden_layer_sizes": [4]}),
    ("catboost", {"iterations": 8, "depth": 2}),
    ("scorecard", {"scorecard_max_bins": 4}),
    ("ensemble", {"base_recipe": "lr", "n_members": 2}),
])
def test_extended_fit_refuses_reassigned_training_labels(tmp_path, recipe, params):
    session, experiment, frame, _, _ = _native(tmp_path, recipe, extra_params=params)
    changed = frame.copy()
    chosen = changed.split == "train"
    changed.loc[chosen, "y"] = 1 - changed.loc[chosen, "y"]
    with closing(session.conn), pytest.raises(ValueError):
        _model_values(session, experiment["id"], experiment["artifact_id"], changed)


def test_extended_numerical_loader_still_rejects_arbitrary_globals(tmp_path):
    target = tmp_path / "must-not-execute"
    class Evil:
        def __reduce__(self):
            return Path.write_text, (target, "executed")
    path = tmp_path / "native.pkl"
    path.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(_Unsupported):
        read_numerical_payload(path)
    assert not target.exists()
