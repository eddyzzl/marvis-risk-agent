from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from marvis.feature.encode import woe_encode
from marvis.feature.preprocessing import apply_preprocessing_steps
from marvis.packs.modeling._common import CALIBRATION_PARAMS_KEY, _optional_str, _resolve_artifact_path
from marvis.packs.modeling.artifact import load_model
from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import normalize_multiclass_probabilities


def scorecard_points_from_raw_pd(
    raw_pd: Sequence[float] | np.ndarray,
    *,
    factor: float,
    offset: float,
) -> np.ndarray:
    """Convert bad-event probabilities to higher-is-better scorecard points."""

    scale_factor, scale_offset = _scorecard_scale(factor=factor, offset=offset)
    probability = _scorecard_numeric_vector(raw_pd, name="raw PD")
    if not np.all(np.isfinite(probability)):
        raise ModelingError("scorecard raw PD values must all be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ModelingError("scorecard raw PD values must all be in [0, 1]")
    logits = np.empty(probability.shape, dtype=np.float64)
    at_zero = probability == 0.0
    at_one = probability == 1.0
    interior = ~(at_zero | at_one)
    logits[at_zero] = -np.inf
    logits[at_one] = np.inf
    logits[interior] = (
        np.log(probability[interior])
        - np.log1p(-probability[interior])
    )
    with np.errstate(over="ignore"):
        points = np.asarray(
            scale_offset - scale_factor * logits,
            dtype=np.float64,
        )
    points.setflags(write=False)
    return points


def raw_pd_from_scorecard_points(
    points: Sequence[float] | np.ndarray,
    *,
    factor: float,
    offset: float,
) -> np.ndarray:
    """Convert finite higher-is-better scorecard points to bad-event probabilities."""

    scale_factor, scale_offset = _scorecard_scale(factor=factor, offset=offset)
    score = _scorecard_numeric_vector(points, name="points")
    if not np.all(np.isfinite(score)):
        raise ModelingError("scorecard points values must all be finite")
    with np.errstate(over="ignore"):
        logits = (scale_offset - score) / scale_factor
    probability = np.empty(score.shape, dtype=np.float64)
    nonnegative = logits >= 0.0
    probability[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    probability[~nonnegative] = exp_logits / (1.0 + exp_logits)
    probability.setflags(write=False)
    return probability


def _scorecard_numeric_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ModelingError(
            f"scorecard {name} must be a one-dimensional numeric vector"
        ) from exc
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ModelingError(
            f"scorecard {name} must be a one-dimensional numeric vector"
        )
    return np.ascontiguousarray(raw, dtype=np.float64)


def _scorecard_scale(*, factor: float, offset: float) -> tuple[float, float]:
    scale_factor = _scorecard_finite_scalar(factor, name="factor")
    if scale_factor <= 0.0:
        raise ModelingError("scorecard factor must be finite and greater than zero")
    scale_offset = _scorecard_finite_scalar(offset, name="offset")
    return scale_factor, scale_offset


def _scorecard_finite_scalar(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ModelingError(f"scorecard {name} must be a finite number")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"scorecard {name} must be a finite number") from exc
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ModelingError(f"scorecard {name} must be a finite number")
    normalized = float(raw)
    if not np.isfinite(normalized):
        raise ModelingError(f"scorecard {name} must be a finite number")
    return normalized


def _fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray):
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(x, y)
        return calibrator
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(np.asarray(scores, dtype=float), y)
        return calibrator
    raise ModelingError(f"unsupported calibration method: {method}")


def _apply_calibrator(method: str, calibrator, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if method == "sigmoid":
        calibrated = calibrator.predict_proba(values.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrated = calibrator.predict(values)
    else:
        raise ModelingError(f"unsupported calibration method: {method}")
    return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


def _calibration_metrics(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> dict:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "ece": _expected_calibration_error(y, p, n_bins=n_bins),
    }


def _expected_calibration_error(labels: np.ndarray, scores: np.ndarray, *, n_bins: int) -> float:
    rows = _calibration_bin_rows(labels, scores, n_bins=n_bins, score_type="")
    total = sum(int(row["sample_count"]) for row in rows)
    if total == 0:
        return 0.0
    return float(
        sum(
            (int(row["sample_count"]) / total) * abs(float(row["calibration_gap"]))
            for row in rows
        )
    )


def _calibration_curve_rows(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    *,
    n_bins: int,
) -> list[dict]:
    return [
        *_calibration_bin_rows(labels, raw_scores, n_bins=n_bins, score_type="raw"),
        *_calibration_bin_rows(labels, calibrated_scores, n_bins=n_bins, score_type="calibrated"),
    ]


def _calibration_bin_rows(labels: np.ndarray, scores: np.ndarray, *, n_bins: int, score_type: str) -> list[dict]:
    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict] = []
    for index in range(int(n_bins)):
        lower = edges[index]
        upper = edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(p[mask]))
        bad_rate = float(np.mean(y[mask]))
        rows.append({
            "score_type": score_type,
            "bin": index + 1,
            "prob_lower": float(lower),
            "prob_upper": float(upper),
            "sample_count": int(np.sum(mask)),
            "positive_count": int(np.sum(y[mask] == 1.0)),
            "avg_predicted_pd": avg_pred,
            "observed_bad_rate": bad_rate,
            "calibration_gap": avg_pred - bad_rate,
            "abs_gap": abs(avg_pred - bad_rate),
        })
    return rows


def _artifact_calibration_rows(artifact: ModelArtifact | None) -> list[dict]:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return []
    # DOM-4: fit_split/eval_split/evaluated_on annotate whether brier/ece below were
    # computed on an independent out-of-sample split or (fallback) on the
    # calibrator's own fitting sample -- absent on calibration payloads persisted
    # before DOM-4, so these read as None on old artifacts rather than erroring.
    fit_split = calibration.get("fit_split", calibration.get("split"))
    eval_split = calibration.get("eval_split", calibration.get("split"))
    evaluated_on = calibration.get("evaluated_on")
    rows = []
    rows.append({
        "score_type": "summary",
        "method": calibration.get("method"),
        "split": calibration.get("split"),
        "split_value": calibration.get("split_value"),
        "fit_split": fit_split,
        "eval_split": eval_split,
        "evaluated_on": evaluated_on,
        "sample_count": calibration.get("sample_count"),
        "positive_count": calibration.get("positive_count"),
        "brier_raw": calibration.get("brier_raw"),
        "brier_calibrated": calibration.get("brier_calibrated"),
        "ece_raw": calibration.get("ece_raw"),
        "ece_calibrated": calibration.get("ece_calibrated"),
        "pmml_includes_calibration": calibration.get("pmml_includes_calibration", False),
        "bin": None,
        "prob_lower": None,
        "prob_upper": None,
        "avg_predicted_pd": None,
        "observed_bad_rate": None,
        "calibration_gap": None,
        "abs_gap": None,
    })
    for row in calibration.get("reliability_curve") or []:
        if not isinstance(row, dict):
            continue
        rows.append({
            "score_type": row.get("score_type"),
            "method": calibration.get("method"),
            "split": calibration.get("split"),
            "split_value": calibration.get("split_value"),
            "fit_split": fit_split,
            "eval_split": eval_split,
            "evaluated_on": evaluated_on,
            "sample_count": row.get("sample_count"),
            "positive_count": row.get("positive_count"),
            "brier_raw": None,
            "brier_calibrated": None,
            "ece_raw": None,
            "ece_calibrated": None,
            "pmml_includes_calibration": calibration.get("pmml_includes_calibration", False),
            "bin": row.get("bin"),
            "prob_lower": row.get("prob_lower"),
            "prob_upper": row.get("prob_upper"),
            "avg_predicted_pd": row.get("avg_predicted_pd"),
            "observed_bad_rate": row.get("observed_bad_rate"),
            "calibration_gap": row.get("calibration_gap"),
            "abs_gap": row.get("abs_gap"),
        })
    return rows


def _artifact_calibration_metadata(artifact: ModelArtifact | None) -> dict | None:
    params = getattr(artifact, "params", None)
    if not isinstance(params, dict):
        return None
    calibration = params.get(CALIBRATION_PARAMS_KEY)
    return calibration if isinstance(calibration, dict) else None


def _load_calibration_payload(artifact: ModelArtifact, *, base_dir: Path) -> dict | None:
    calibration = _artifact_calibration_metadata(artifact)
    if not calibration:
        return None
    calibration_path = _optional_str(calibration.get("path"))
    if not calibration_path:
        return None
    path = _resolve_artifact_path(calibration_path, base_dir=base_dir)
    if not path.exists():
        raise ModelingError(f"calibration file does not exist: {calibration_path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "method" not in payload or "calibrator" not in payload:
        raise ModelingError(f"invalid calibration payload: {calibration_path}")
    return payload


class _ModelArtifactScorer:
    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        base_dir: Path,
        load_calibration: bool = True,
        replay_preprocessing: bool = False,
    ):
        self.artifact = artifact
        self.base_dir = Path(base_dir)
        self.model = load_model(artifact, base_dir=base_dir)
        self.calibration = (
            _load_calibration_payload(artifact, base_dir=self.base_dir)
            if load_calibration
            else None
        )
        # PREP-2: replay is opt-in. Existing report/stress-test/calibration call sites
        # score the SAME already-transformed modeling frame the model was trained on
        # (impute/cap/normalize already applied in place), so replaying again would
        # double-apply a non-idempotent transform like zscore/minmax normalize and
        # silently corrupt those scores. Only a caller scoring genuinely new raw data
        # (e.g. a future score_dataset tool) should pass replay_preprocessing=True.
        self.replay_preprocessing = bool(replay_preprocessing)

    def score(self, dataframe: pd.DataFrame, *, use_calibration: bool = True) -> list[float]:
        if "multiclass" in self.artifact.algorithm:
            raise ModelingError(
                "multiclass artifacts do not expose a scalar score; "
                "use predict_proba() and the persisted class mapping"
            )
        scores = np.asarray(self.raw_score(dataframe), dtype=float)
        if use_calibration and self.calibration is not None:
            scores = _apply_calibrator(str(self.calibration["method"]), self.calibration["calibrator"], scores)
        return [float(value) for value in scores]

    def raw_score(self, dataframe: pd.DataFrame) -> list[float]:
        dataframe = self._replay_preprocessing(dataframe)
        features = list(self.artifact.feature_list)
        if "multiclass" in self.artifact.algorithm:
            raise ModelingError(
                "multiclass artifacts do not expose a scalar raw score; "
                "use predict_proba()"
            )
        if self.artifact.algorithm == "ensemble" and isinstance(self.model, dict):
            return self._ensemble_raw_score(dataframe, features)
        if self.artifact.algorithm == "xgb" and not hasattr(self.model, "predict_proba"):
            import xgboost as xgb

            matrix = xgb.DMatrix(dataframe[features], feature_names=features)
            return [float(value) for value in self.model.predict(matrix)]
        if self.artifact.algorithm == "scorecard" and isinstance(self.model, dict):
            encoded = pd.DataFrame(index=dataframe.index)
            woe_maps = self.model["woe_maps"]
            for feature in features:
                encoded[feature] = woe_encode(dataframe, feature, woe_maps[feature]).to_numpy(dtype=float)
            return [float(value) for value in self.model["model"].predict_proba(encoded)[:, 1]]
        if hasattr(self.model, "predict_proba"):
            return [float(value) for value in self.model.predict_proba(dataframe[features])[:, 1]]
        return [float(value) for value in self.model.predict(dataframe[features])]

    def predict_proba(self, dataframe: pd.DataFrame) -> np.ndarray:
        """Return the complete N x K probability matrix for a multiclass artifact.

        A multiclass model has no canonical positive class. Returning column 1 as
        a fake PD silently discards K-1 outcomes, so callers must consume all
        persisted class columns explicitly.
        """

        if "multiclass" not in self.artifact.algorithm:
            raise ModelingError(
                f"predict_proba is only available for multiclass artifacts: "
                f"{self.artifact.algorithm}"
            )
        dataframe = self._replay_preprocessing(dataframe)
        features = list(self.artifact.feature_list)
        if self.artifact.algorithm == "lgb_multiclass":
            probabilities = np.asarray(
                self.model.predict(dataframe[features]),
                dtype=float,
            )
        elif hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(
                self.model.predict_proba(dataframe[features]),
                dtype=float,
            )
        else:
            raise ModelingError(
                f"multiclass artifact cannot produce probabilities: "
                f"{self.artifact.algorithm}"
            )
        classes = list((self.artifact.params or {}).get("classes") or [])
        return normalize_multiclass_probabilities(
            probabilities,
            expected_rows=len(dataframe),
            expected_classes=len(classes),
        )

    def _ensemble_raw_score(self, dataframe: pd.DataFrame, features: list[str]) -> list[float]:
        """SEL-6: score every member (each loaded fresh via its own algorithm's
        load_model dispatch, relative to this artifact's base_dir) and return
        the weight-averaged probability. Mirrors
        recipes/ensemble.py._member_score_fn's per-algorithm dispatch, kept
        separate since that module cannot import tools.py (circular import)."""
        members = self.model.get("members") or []
        weights = self.model.get("weights") or [1.0 / len(members)] * len(members)
        if not members:
            raise ModelingError(f"ensemble artifact has no members: {self.artifact.id}")
        predictions = []
        for member in members:
            member_artifact = ModelArtifact(
                id=str(member.get("artifact_id") or ""),
                experiment_id="",
                algorithm=str(member.get("algorithm") or ""),
                model_path=str(member.get("model_path") or ""),
                pmml_path=None,
                feature_list=tuple(features),
                params={},
                woe_maps=None,
                created_at="",
            )
            member_model = load_model(member_artifact, base_dir=self.base_dir)
            if member_artifact.algorithm == "xgb" and not hasattr(member_model, "predict_proba"):
                import xgboost as xgb

                matrix = xgb.DMatrix(dataframe[features], feature_names=features)
                predictions.append(np.asarray(member_model.predict(matrix), dtype=float))
            else:
                predictions.append(
                    np.asarray(member_model.predict_proba(dataframe[features])[:, 1], dtype=float)
                )
        averaged = np.average(np.array(predictions, dtype=float), axis=0, weights=weights)
        return [float(value) for value in averaged]

    def scorecard_points(self, dataframe: pd.DataFrame) -> list[float] | None:
        if self.artifact.algorithm != "scorecard" or not isinstance(self.model, dict):
            return None
        dataframe = self._replay_preprocessing(dataframe)
        params = dict(self.model.get("params") or {})
        if "factor" not in params or "offset" not in params:
            return None
        features = list(self.artifact.feature_list)
        encoded = pd.DataFrame(index=dataframe.index)
        woe_maps = self.model["woe_maps"]
        for feature in features:
            encoded[feature] = woe_encode(dataframe, feature, woe_maps[feature]).to_numpy(dtype=float)
        logits = (
            float(self.model["model"].intercept_[0])
            + encoded.to_numpy(dtype=float) @ self.model["model"].coef_[0]
        )
        scores = float(params["offset"]) - float(params["factor"]) * logits
        return [float(value) for value in scores]

    def _replay_preprocessing(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Replay this artifact's persisted preprocessing chain (PREP-2) before scoring,
        so predict-time input matches the exact impute/cap/normalize/onehot transforms
        the model was trained on — the fix for silent scoring drift on new raw data.
        No-op unless the caller opted in via replay_preprocessing=True (see __init__),
        or when the artifact carries no chain (e.g. a pre-PREP-2 artifact, or one
        trained straight off a dataset with no traceable lineage)."""
        if not self.replay_preprocessing:
            return dataframe
        steps = self.artifact.params.get("preprocessing_steps") if self.artifact.params else None
        if not steps:
            return dataframe
        return apply_preprocessing_steps(dataframe, list(steps))
