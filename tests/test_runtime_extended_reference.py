"""Independent feature/portfolio/modeling references agree with installed producers.

The tests import production code; the reference modules must not.
"""
import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marvis.orchestrator.eval import runtime_feature_extended_reference as feat
from marvis.orchestrator.eval import runtime_modeling_extended_reference as model_ref
from marvis.orchestrator.eval import runtime_portfolio_trend_reference as port


@pytest.mark.parametrize("module", [feat, model_ref, port])
def test_reference_modules_import_no_production_calculator(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [n for n in names if n.startswith("marvis")]


def _close(actual, expected, path=""):
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (path, set(actual) ^ set(expected))
        for key in expected:
            _close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _close(a, b, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(actual, (str, type(None))):
        assert actual == expected or math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def _frame(seed=5, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(size=n)
    x3 = rng.integers(0, 4, size=n).astype(float)
    x1[rng.random(n) < .05] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp(-(x2 - .5)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


# --- features ---------------------------------------------------------------

@pytest.mark.parametrize("bins", [5, 10])
def test_psi_equals_feature_tool_kernel(bins):
    from marvis.feature.binning import equal_frequency_edges
    from marvis.feature.metrics import feature_psi

    frame = _frame()
    base, compare = frame["x2"].to_numpy()[:200], frame["x2"].to_numpy()[200:] + .3
    expected = feature_psi(base, compare, equal_frequency_edges(base, bins))
    assert feat.psi_reference(base, compare, bins=bins)["psi"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("view", ["psi_split", "psi_month_first", "psi_month_last", "psi_mom"])
def test_psi_views_follow_the_installed_pairing_rule(view):
    from marvis.packs.feature.tools import _feature_psi_series, _psi_comparison_pairs, _split_role_labels

    frame = _frame(n=360)
    raw = ["Train", "valid", "OOT"] * 120 if view == "psi_split" else ["2026-01", "2026-02", "2026-04"] * 120
    labels = _split_role_labels(pd.Series(raw)) if view == "psi_split" else np.asarray(raw, dtype=object)
    series, _ = _feature_psi_series(frame["x2"].to_numpy(dtype=float), labels, _psi_comparison_pairs(labels, view), bins=10)
    ours = feat.psi_series_reference(frame["x2"].tolist(), raw, view)
    _close(ours["series"], series)
    assert ours["psi"] == pytest.approx(max(r["psi"] for r in series), rel=1e-12)


def test_vif_equals_installed_correlation_kernel_including_cap_and_unavailable():
    from marvis.feature.correlation import vif

    frame = _frame()
    frame["x4"] = 2 * frame["x2"] + 1  # perfectly collinear
    features = ["x1", "x2", "x3", "x4"]
    expected, ours = vif(frame, features), feat.vif_reference(frame.to_dict("records"), features)
    for name in ("x1", "x3"):
        assert ours[name] == pytest.approx(expected[name], rel=1e-9)
    assert ours["x2"] == ours["x4"] == feat.VIF_CAP and expected["x2"] >= 1e6
    small = frame.head(20)
    assert feat.vif_reference(small.to_dict("records"), features) == vif(small, features)


def test_importance_retrain_sums_dumped_tree_gains_like_installed_tool():
    from marvis.feature.importance import feature_importance

    frame = _frame()
    expected = feature_importance(frame, ["x1", "x2", "x3"], "y")
    ours = feat.importance_reference(frame, ["x1", "x2", "x3"], "y")
    for name in expected:
        assert ours[name] == pytest.approx(expected[name], rel=1e-9, abs=1e-12)
    assert feat.importance_reference(frame.assign(y=0), ["x1"], "y") == {"x1": None}


@pytest.mark.parametrize("column,bins", [("x1", 10), ("x2", 5), ("x3", 10)])
def test_bin_detail_equals_installed_bin_analysis(column, bins):
    from marvis.feature.bin_analysis import feature_bin_analysis

    frame = _frame()
    expected = json.loads(json.dumps(feature_bin_analysis(frame[column].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float),
                                                          feature=column, requested_bins=bins)))
    ours = feat.bin_analysis_reference(frame[column].tolist(), frame["y"].tolist(), feature=column, requested_bins=bins)
    _close(ours, expected)


# --- portfolio --------------------------------------------------------------

def _baseline(rng):
    from marvis.feature.binning import equal_frequency_edges
    from marvis.validation.binning import bin_distribution

    train = rng.normal(size=500)
    edges = equal_frequency_edges(train, 10)
    return {"score_edges": edges.tolist(), "score_distribution": {"train": {"bin_proportions": bin_distribution(train, edges).tolist()}},
            "feature_distributions": {"a": {"quantile_edges": edges.tolist()}, "b": {"quantile_edges": (edges * 2).tolist()}}}


def test_score_and_csi_trend_equal_installed_trend_kernels():
    from marvis.packs.analysis.trend import feature_csi_trend, score_stability_trend
    from dataclasses import asdict

    rng = np.random.default_rng(9)
    baseline = _baseline(rng)
    months = [(m, pd.DataFrame({"score": rng.normal(shift, 1, 200), "a": rng.normal(shift, 1, 200), "b": rng.normal(0, 3, 200)}))
              for m, shift in (("2026-01", 0), ("2026-02", .4), ("2026-05", 1.2))]
    score = asdict(score_stability_trend(baseline, months, score_col="score"))
    _close(port.score_trend_reference(baseline["score_edges"], baseline["score_distribution"]["train"]["bin_proportions"],
                                      [(m, f["score"].tolist()) for m, f in months]), score)
    csi = asdict(feature_csi_trend(baseline, months))
    _close(port.csi_trend_reference(baseline["feature_distributions"], [(m, {c: f[c].tolist() for c in f}) for m, f in months]), csi)
    assert score["red_flags"][0]["kind"] == "month_gap"


def test_trend_thresholds_mirror_monitor_thresholds():
    from marvis.packs.modeling.monitor_tools import MONITOR_RUN_THRESHOLDS

    assert (port.PSI_WARN, port.PSI_FAIL) == (MONITOR_RUN_THRESHOLDS["score_psi"]["warn"], MONITOR_RUN_THRESHOLDS["score_psi"]["fail"])


@pytest.mark.parametrize("semantics", ["incremental", "snapshot"])
def test_vintage_equals_installed_curve_and_summary(semantics):
    from marvis.packs.strategy.vintage import vintage_curve, vintage_summary

    rng = np.random.default_rng(4)
    rows = [{"cohort": c, "mob": m, "bad": int(rng.random() < .1 * (m + 1) / 4)}
            for c in ("202601", "2026-02", "202603") for m in range(1, 8 if c != "202603" else 5) for _ in range(30)]
    curve = vintage_curve(pd.DataFrame(rows), cohort_col="cohort", mob_col="mob", bad_col="bad", mob_max=6, label_semantics=semantics)
    ours = port.vintage_reference(rows, cohort_col="cohort", mob_col="mob", bad_col="bad", mob_max=6, label_semantics=semantics)
    assert ours["cohorts"] == list(curve.cohorts) and ours["mob_axis"] == list(curve.mob_axis)
    assert ours["counts"] == curve.counts
    _close(ours["curves"], {k: list(v) for k, v in curve.curves.items()})
    _close(port.vintage_summary_reference(ours, ref_mob=4), vintage_summary(curve, ref_mob=4))


# --- modeling ---------------------------------------------------------------

def test_calibration_metrics_rows_and_appliers_equal_installed_scoring():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from marvis.packs.modeling.scoring import _apply_calibrator, _calibration_bin_rows, _calibration_metrics

    rng = np.random.default_rng(2)
    scores = np.round(rng.random(300), 1)  # values on bin edges
    labels = (rng.random(300) < scores).astype(int)
    _close(model_ref.calibration_metrics_reference(labels.tolist(), scores.tolist(), n_bins=10),
           _calibration_metrics(labels, scores, n_bins=10))
    _close(model_ref.calibration_rows_reference(labels.tolist(), scores.tolist(), n_bins=10, score_type="raw"),
           _calibration_bin_rows(labels, scores, n_bins=10, score_type="raw"))
    sigmoid = LogisticRegression(solver="lbfgs").fit(scores.reshape(-1, 1), labels)
    _close(model_ref.apply_sigmoid_reference(float(sigmoid.coef_[0][0]), float(sigmoid.intercept_[0]), scores.tolist()),
           _apply_calibrator("sigmoid", sigmoid, scores).tolist())
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
    probe = np.linspace(-.1, 1.1, 50)
    _close(model_ref.apply_isotonic_reference(isotonic.X_thresholds_, isotonic.y_thresholds_, probe.tolist()),
           _apply_calibrator("isotonic", isotonic, probe).tolist())


def test_calibration_fold_equals_installed_carve():
    from marvis.packs.modeling.calibrate_tools import DEFAULT_CALIBRATION_FIT_FRACTION, _calibration_fold_seed
    from marvis.packs.modeling.recipes.common import carve_early_stop_fold

    train = pd.DataFrame({"x": range(173)})
    _, fold = carve_early_stop_fold(train, seed=_calibration_fold_seed(11), valid_fraction=DEFAULT_CALIBRATION_FIT_FRACTION)
    assert model_ref.calibration_fold_membership(173, 11) == sorted(fold.index.tolist())


def test_early_stopping_membership_equals_installed_carve():
    from marvis.orchestrator.eval.runtime_modeling_tree_reference import early_stop_membership
    from marvis.packs.modeling.recipes.common import carve_early_stop_fold

    train = pd.DataFrame({"x": range(120), "g": [i // 3 for i in range(120)]})
    for params in ({}, {"valid_group_cols": ["g"]}):
        fit, valid = early_stop_membership(train, {"seed": 7, "params": params})
        _, expected = carve_early_stop_fold(train, seed=7, group_cols=params.get("valid_group_cols"))
        held = valid.index.tolist() if hasattr(valid, "index") else list(valid)
        assert sorted(held) == sorted(expected.index.tolist())


def test_ensemble_weighted_average_equals_numpy_average():
    rng = np.random.default_rng(1)
    members = rng.random((3, 20))
    weights = [.2, .3, .5]
    assert model_ref.ensemble_reference(members.tolist(), weights) == pytest.approx(np.average(members, axis=0, weights=weights).tolist(), rel=1e-12)
    assert model_ref.ensemble_reference(members.tolist()) == pytest.approx(members.mean(axis=0).tolist(), rel=1e-12)


def test_catboost_json_evaluation_equals_library_probabilities(tmp_path):
    catboost = pytest.importorskip("catboost")
    frame = _frame().fillna(0.0)
    features = ["x1", "x2", "x3"]
    model = catboost.CatBoostClassifier(iterations=30, depth=4, verbose=0, random_seed=3, allow_writing_files=False)
    model.fit(frame[features], frame["y"])
    path = tmp_path / "model.json"
    model.save_model(str(path), format="json")
    ours = model_ref.catboost_json_reference(json.loads(path.read_text()), frame[features].to_numpy().tolist())
    assert ours == pytest.approx(model.predict_proba(frame[features])[:, 1].tolist(), rel=1e-6, abs=1e-9)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer")
def test_mlp_forward_pass_from_fitted_numbers_equals_installed_recipe_pipeline():
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = _frame()
    features = ["x1", "x2", "x3"]
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300, random_state=0))])
    pipeline.fit(frame[features], frame["y"])
    ours = model_ref.mlp_reference(model_ref.sklearn_mlp_params(pipeline), frame[features].to_numpy().tolist())
    assert ours == pytest.approx(pipeline.predict_proba(frame[features])[:, 1].tolist(), rel=1e-9, abs=1e-12)


def test_scorecard_points_follow_the_installed_pdo_convention():
    factor = 50 / math.log(2)
    result = model_ref.scorecard_points_reference(-1.2, {"x": 0.8}, [("x", 0, 0.25), ("x", -1, -0.4)])
    assert result["offset"] == pytest.approx(600 - factor * math.log(50))
    assert result["base_points"] == pytest.approx(result["offset"] + factor * 1.2)
    assert [r["points"] for r in result["points"]] == pytest.approx([-factor * .8 * .25, factor * .8 * .4])


def test_weighted_ks_auc_equal_installed_weighted_metrics():
    from marvis.feature.metrics import weighted_feature_auc, weighted_feature_ks

    rng = np.random.default_rng(8)
    scores = np.round(rng.random(400), 2)
    labels = (rng.random(400) < scores).astype(int)
    weights = rng.random(400) * 3
    ours = model_ref.weighted_binary_reference(scores.tolist(), labels.tolist(), weights.tolist())
    assert ours["ks"] == pytest.approx(weighted_feature_ks(scores, labels, weights), rel=1e-9)
    assert ours["auc"] == pytest.approx(weighted_feature_auc(scores, labels, weights), rel=1e-9)
