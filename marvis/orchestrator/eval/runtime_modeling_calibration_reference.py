"""Revalidate authenticated calibrator fits and reliability evidence independently."""
import math

import numpy as np

from .runtime_archive_domain import equal, numbers
from .runtime_archive_reader import _Unsupported
from .runtime_modeling_extended_reference import (
    apply_isotonic_reference,
    apply_sigmoid_reference,
    calibration_fold_membership,
    calibration_metrics_reference,
    calibration_rows_reference,
)
from .runtime_modeling_profile_reference import read_numerical_payload


def _sample(frame, score, target):
    predictions = np.asarray(score(frame), dtype=float)
    labels = np.asarray(frame[target], dtype=float)
    selected = np.isfinite(predictions) & np.isfinite(labels)
    predictions, labels = predictions[selected], labels[selected]
    if not set(labels).issubset({0., 1.}):
        raise ValueError("calibration labels outside binary contract")
    return predictions, labels.astype(int)


def revalidate_calibration(path, metadata, config, frame, raw_predict):
    """The original model metrics stay raw; calibration gets its own fit check.

    Calibration is retained in a separately hashed numerical payload. The public
    metadata records which fit population was used, including explicit in-sample
    calibration. Such a population is never relabelled independent holdout.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    calibration = metadata["params"]["calibration"]
    if (calibration["dataset_id"] != config["dataset_id"] or calibration["target_col"] != config["target_col"]
            or calibration["split_col"] != config["split_col"]):
        raise ValueError("calibration source/target contract differs from model source")
    payload = read_numerical_payload(path)
    if not isinstance(payload, dict) or payload.get("method") != calibration["method"]:
        raise ValueError("calibration numerical payload method mismatch")
    method, learner = payload["method"], payload.get("calibrator")
    expected_type = {"sigmoid": LogisticRegression, "isotonic": IsotonicRegression}.get(method)
    if expected_type is None or type(learner) is not expected_type:
        raise ValueError("calibration native class mismatch")
    declared = (LogisticRegression(solver="lbfgs") if method == "sigmoid"
                else IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0))
    if learner.get_params(deep=False) != declared.get_params(deep=False):
        raise _Unsupported("calibrator configuration outside declared production profile")
    if method == "sigmoid":
        if (not np.array_equal(learner.classes_, [0, 1]) or learner.n_features_in_ != 1
                or np.asarray(learner.coef_).shape != (1, 1)
                or np.asarray(learner.intercept_).shape != (1,)):
            raise ValueError("calibrator binary class/coefficient shape mismatch")
    elif (np.asarray(learner.X_thresholds_).ndim != 1 or np.asarray(learner.y_thresholds_).ndim != 1
          or learner.increasing_ is not True):
        raise ValueError("calibrator isotonic shape/direction mismatch")
    split = calibration["fit_split"]
    if calibration["split"] != split:
        raise ValueError("calibration fit split disagreement")
    if split == "train_calibration_fold":
        if calibration["split_value"] != config["split_values"]["train"]:
            raise ValueError("calibration fold source differs from train")
        train = frame[frame[config["split_col"]] == config["split_values"]["train"]]
        fit = train.iloc[calibration_fold_membership(len(train), config["seed"])]
    else:
        fit = frame[frame[config["split_col"]] == calibration["split_value"]]
        if calibration["evaluated_on"] != "fit_sample(in-sample)" or calibration["eval_split"] != split:
            raise ValueError("explicit calibration split must remain in-sample")
    scores, labels = _sample(fit, raw_predict, config["target_col"])
    if len(labels) < 2 or len(set(labels)) != 2:
        raise ValueError("calibration fit requires both binary classes")
    equal({"fit_sample_count": calibration["fit_sample_count"], "positive_count": calibration["positive_count"],
           "negative_count": calibration["negative_count"]},
          {"fit_sample_count": len(labels), "positive_count": int(sum(labels)), "negative_count": int(len(labels) - sum(labels))})
    if method == "sigmoid":
        rebuilt = LogisticRegression(solver="lbfgs").fit(scores.reshape(-1, 1), labels)
        if not (np.allclose(learner.coef_, rebuilt.coef_, rtol=1e-8, atol=1e-10)
                and np.allclose(learner.intercept_, rebuilt.intercept_, rtol=1e-8, atol=1e-10)):
            raise ValueError("calibrator differs from independent declared-population fit")
        def apply(values):
            return apply_sigmoid_reference(float(learner.coef_[0, 0]), float(learner.intercept_[0]), values)
    else:
        rebuilt = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
        if learner.X_min_ != rebuilt.X_min_ or learner.X_max_ != rebuilt.X_max_:
            raise ValueError("calibrator extrapolation bounds differ from independent fit")
        for actual, expected in ((learner.X_thresholds_, rebuilt.X_thresholds_), (learner.y_thresholds_, rebuilt.y_thresholds_)):
            if np.asarray(actual).shape != np.asarray(expected).shape or not np.allclose(actual, expected, rtol=1e-8, atol=1e-10):
                raise ValueError("calibrator differs from independent declared-population fit")
        def apply(values):
            return apply_isotonic_reference(learner.X_thresholds_, learner.y_thresholds_, values)
    n_bins = calibration["n_bins"]
    if type(n_bins) is not int or not 2 <= n_bins <= 10000:
        raise ValueError("calibration bin contract outside bound")
    calibrated = apply(scores)
    reliability = [*calibration_rows_reference(labels, scores, n_bins=n_bins, score_type="raw"),
                   *calibration_rows_reference(labels, calibrated, n_bins=n_bins, score_type="calibrated")]
    numbers(calibration["reliability_curve"], reliability)
    per_split = {}
    for role, observed in calibration["per_split_metrics"].items():
        if split != "train_calibration_fold" or role not in {"test", "oot"}:
            raise ValueError("calibration evaluation split outside declared contract")
        subset = frame[frame[config["split_col"]] == config["split_values"][role]]
        s, y = _sample(subset, raw_predict, config["target_col"])
        if not len(y) or len(set(y)) != 2:
            raise ValueError("calibration evaluation requires retained binary labels")
        before = calibration_metrics_reference(y, s, n_bins=n_bins)
        after = calibration_metrics_reference(y, apply(s), n_bins=n_bins)
        expected = {"sample_count": len(y), "brier_raw": before["brier"], "brier_calibrated": after["brier"],
                    "ece_raw": before["ece"], "ece_calibrated": after["ece"]}
        numbers(observed, expected)
        per_split[role] = expected
    if per_split:
        headline = "test" if "test" in per_split else "oot"
        if calibration["eval_split"] != headline or calibration["evaluated_on"] != headline:
            raise ValueError("calibration headline evaluation binding mismatch")
        metrics = per_split[headline]
    else:
        if calibration["evaluated_on"] != "fit_sample(in-sample)":
            raise ValueError("calibration without independent evidence must remain in-sample")
        before = calibration_metrics_reference(labels, scores, n_bins=n_bins)
        after = calibration_metrics_reference(labels, calibrated, n_bins=n_bins)
        metrics = {"sample_count": len(labels), "brier_raw": before["brier"], "brier_calibrated": after["brier"],
                   "ece_raw": before["ece"], "ece_calibrated": after["ece"]}
    numbers({k: calibration[k] for k in metrics}, metrics)
    if any(not math.isfinite(float(value)) for key, value in metrics.items() if key != "sample_count"):
        raise ValueError("calibration metric is not finite")
    return {"method": method, "fit_split": split, "fit_rows": len(labels), "evaluated_on": calibration["evaluated_on"],
            "metrics": metrics, "independent_fit_verified": True}
