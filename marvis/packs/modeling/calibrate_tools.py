from __future__ import annotations

import hashlib
import joblib
import numpy as np
import pandas as pd
from dataclasses import replace
from datetime import UTC, datetime
from marvis.artifacts import ArtifactUnitOfWork
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling.artifact import persist_model_meta
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.recipes.common import carve_early_stop_fold

from marvis.packs.modeling._common import CALIBRATION_PARAMS_KEY, _jsonable, _unique_columns
from marvis.packs.modeling._runtime import _artifact, _artifact_model_base_dir, _runtime
from marvis.packs.modeling.scoring import _ModelArtifactScorer, _apply_calibrator, _calibration_curve_rows, _calibration_metrics, _fit_calibrator


def _calibration_fold_seed(seed: int) -> int:
    """Deterministic seed derivation for the calibration-fitting fold, distinct from
    the early-stopping valid-fold seed and the base training seed (same derivation
    pattern as ``recipes.common._valid_fold_seed`` / ``_bootstrap_ci_seed``) so the
    calibration fold draw doesn't collide with either RNG stream."""
    digest = hashlib.sha256(f"{int(seed)}:calibration_fold".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


#: DOM-4: default fraction of ``train`` carved out to fit the calibrator when the
#: caller does not explicitly pick a fit split -- mirrors the early-stopping fold's
#: default fraction (recipes.common.DEFAULT_EARLY_STOP_VALID_FRACTION).
DEFAULT_CALIBRATION_FIT_FRACTION = 0.15


def _calibration_valid_labeled_scores(
    scorer: "_ModelArtifactScorer",
    sample: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score ``sample`` and drop rows without a finite raw score or a clean binary
    label -- shared filtering logic for every split calibrate_model touches."""
    raw_scores = np.asarray(scorer.score(sample, use_calibration=False), dtype=float)
    labels = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw_scores) & np.isfinite(labels) & np.isin(labels, [0.0, 1.0])
    return raw_scores[valid], labels[valid].astype(int)


def tool_calibrate_model(inputs: dict, ctx) -> dict:
    """Fit a post-training binary probability calibrator and report Brier/ECE.

    DOM-4: when the caller does not explicitly pick a ``split``/``fit_split`` to fit
    on, the calibrator is fit on a held-out fold carved from ``train`` (same
    deterministic carving paradigm as the early-stopping fold, TUNE-3/SEL-4, but its
    own seed namespace so the draw never collides) instead of on ``test`` -- fitting
    and immediately evaluating on the same set gives a mathematically-guaranteed-
    optimistic ECE/Brier reading, especially for isotonic regression. Calibration
    quality (Brier/ECE/reliability curve) is then reported on ``test`` (and ``oot``
    when it carries labels), i.e. evaluated out of sample.

    Back-compat: when the caller explicitly passes ``split`` (or ``fit_split``), the
    calibrator fits AND is evaluated on that exact split, exactly like before --
    existing callers/tests that pin ``split="test"`` see byte-identical output,
    just with the metrics now additionally labelled ``evaluated_on:
    "fit_sample(in-sample)"`` so the in-sample caveat is explicit instead of silent.
    """
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    experiment = runtime.experiments.get(artifact.experiment_id)
    config = experiment.config
    if getattr(config, "target_type", "binary") != "binary":
        raise ModelingError("probability calibration is only supported for binary models")

    method = str(inputs.get("method") or "sigmoid").strip().lower()
    if method not in {"sigmoid", "isotonic"}:
        raise ModelingError(f"unsupported calibration method: {method}")
    n_bins = int(inputs.get("n_bins") or 10)
    min_samples = int(inputs.get("min_samples") or 30)
    if n_bins < 2:
        raise ModelingError("n_bins must be at least 2")
    if min_samples < 1:
        raise ModelingError("min_samples must be at least 1")

    dataset_id = str(inputs.get("dataset_id") or config.dataset_id)
    dataset = runtime.registry.get(dataset_id)
    target_col = str(inputs.get("target_col") or config.target_col)
    split_col = str(inputs.get("split_col") or config.split_col)
    explicit_split = inputs.get("split") or inputs.get("fit_split")
    frame = runtime.backend.read_frame(
        runtime.registry.resolve_path(dataset.id),
        columns=_unique_columns([*artifact.feature_list, target_col, split_col]),
    )
    scorer = _ModelArtifactScorer(
        artifact,
        base_dir=_artifact_model_base_dir(runtime, artifact),
        load_calibration=False,
    )

    if explicit_split:
        # Back-compat path: fit and evaluate on the exact split the caller named --
        # output shape/values are unchanged from before DOM-4, only the new
        # `evaluated_on`/`fit_split`/`eval_split` fields are added.
        split_name = str(explicit_split)
        split_value = inputs.get("split_value", config.split_values.get(split_name, split_name))
        sample = frame[frame[split_col] == split_value].copy()
        if sample.empty:
            raise ModelingError(f"calibration split has no rows: {split_col}={split_value}")
        raw_scores, labels = _calibration_valid_labeled_scores(scorer, sample, target_col)
        if labels.size < min_samples:
            raise ModelingError(
                f"calibration sample has {labels.size} valid labeled rows; require at least {min_samples}"
            )
        if np.unique(labels).size < 2:
            raise ModelingError("calibration sample must contain both positive and negative labels")

        calibrator = _fit_calibrator(method, raw_scores, labels)
        calibrated_scores = _apply_calibrator(method, calibrator, raw_scores)
        raw_metrics = _calibration_metrics(labels, raw_scores, n_bins=n_bins)
        calibrated_metrics = _calibration_metrics(labels, calibrated_scores, n_bins=n_bins)
        reliability_curve = _calibration_curve_rows(labels, raw_scores, calibrated_scores, n_bins=n_bins)
        eval_split_name = split_name
        eval_sample_count = int(labels.size)
        evaluated_on = "fit_sample(in-sample)"
        per_split_metrics: dict[str, dict] = {}
    else:
        # DOM-4 default path: fit on a held-out fold carved from train, evaluate on
        # test (and OOT when it carries labels) -- an independent labeled set the
        # calibrator never saw during fitting.
        train_value = config.split_values.get("train", "train")
        train_frame = frame[frame[split_col] == train_value].copy()
        if train_frame.empty:
            raise ModelingError(f"calibration fit split has no rows: {split_col}={train_value}")
        _fit_train, fit_fold = carve_early_stop_fold(
            train_frame,
            seed=_calibration_fold_seed(int(getattr(config, "seed", 0) or 0)),
            valid_fraction=DEFAULT_CALIBRATION_FIT_FRACTION,
        )
        raw_scores, labels = _calibration_valid_labeled_scores(scorer, fit_fold, target_col)
        if labels.size < min_samples:
            raise ModelingError(
                f"calibration sample has {labels.size} valid labeled rows; require at least {min_samples}"
            )
        if np.unique(labels).size < 2:
            raise ModelingError("calibration sample must contain both positive and negative labels")

        calibrator = _fit_calibrator(method, raw_scores, labels)
        calibrated_scores = _apply_calibrator(method, calibrator, raw_scores)
        # In-sample readings on the fitting fold itself -- reported but explicitly
        # labelled, never the headline metric (DOM-4 fix item #2).
        raw_metrics = _calibration_metrics(labels, raw_scores, n_bins=n_bins)
        calibrated_metrics = _calibration_metrics(labels, calibrated_scores, n_bins=n_bins)
        reliability_curve = _calibration_curve_rows(labels, raw_scores, calibrated_scores, n_bins=n_bins)
        split_name = "train_calibration_fold"
        split_value = train_value

        per_split_metrics = {}
        eval_split_name = None
        eval_sample_count = 0
        for candidate_split in ("test", "oot"):
            candidate_value = config.split_values.get(candidate_split)
            if candidate_value is None:
                continue
            candidate_frame = frame[frame[split_col] == candidate_value].copy()
            if candidate_frame.empty:
                continue
            eval_scores, eval_labels = _calibration_valid_labeled_scores(scorer, candidate_frame, target_col)
            if eval_labels.size < min_samples or np.unique(eval_labels).size < 2:
                continue
            eval_calibrated = _apply_calibrator(method, calibrator, eval_scores)
            eval_raw_metrics = _calibration_metrics(eval_labels, eval_scores, n_bins=n_bins)
            eval_calibrated_metrics = _calibration_metrics(eval_labels, eval_calibrated, n_bins=n_bins)
            per_split_metrics[candidate_split] = {
                "sample_count": int(eval_labels.size),
                "brier_raw": eval_raw_metrics["brier"],
                "brier_calibrated": eval_calibrated_metrics["brier"],
                "ece_raw": eval_raw_metrics["ece"],
                "ece_calibrated": eval_calibrated_metrics["ece"],
            }
            if eval_split_name is None:
                # test is evaluated first in the loop order, so it is preferred as
                # the headline out-of-sample reading; oot only fills in when test
                # itself didn't have enough labeled rows.
                eval_split_name = candidate_split
                eval_sample_count = int(eval_labels.size)
                raw_metrics = eval_raw_metrics
                calibrated_metrics = eval_calibrated_metrics

        if eval_split_name is not None:
            evaluated_on = eval_split_name
        else:
            # No independent labeled split available (e.g. OOT unlabeled and no
            # test split) -- fall back to the in-sample fitting-fold metrics
            # already computed above, but say so explicitly.
            evaluated_on = "fit_sample(in-sample)"
            eval_split_name = split_name
            eval_sample_count = int(labels.size)

    base_dir = _artifact_model_base_dir(runtime, artifact)
    calibration_path = f"{artifact.id}.calibration.{method}.joblib"
    calibration_payload = {
        "method": method,
        "calibrator": calibrator,
        "created_at": datetime.now(UTC).isoformat(),
    }
    uow = ArtifactUnitOfWork()
    calibration_artifact = uow.stage_file(base_dir, calibration_path)
    try:
        joblib.dump(calibration_payload, calibration_artifact.path)
    except Exception:
        uow.rollback()
        raise
    calibration = {
        "method": method,
        "path": calibration_path,
        "dataset_id": dataset.id,
        "target_col": target_col,
        "split_col": split_col,
        "split": split_name,
        "split_value": split_value,
        "fit_split": split_name,
        "eval_split": eval_split_name,
        "evaluated_on": evaluated_on,
        "sample_count": eval_sample_count,
        "fit_sample_count": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "brier_raw": raw_metrics["brier"],
        "brier_calibrated": calibrated_metrics["brier"],
        "ece_raw": raw_metrics["ece"],
        "ece_calibrated": calibrated_metrics["ece"],
        "per_split_metrics": per_split_metrics,
        "n_bins": n_bins,
        "pmml_includes_calibration": False,
        "reliability_curve": reliability_curve,
    }
    params = {**dict(artifact.params or {}), CALIBRATION_PARAMS_KEY: calibration}
    updated_artifact = replace(artifact, params=params)
    try:
        persist_model_meta(base_dir, updated_artifact, config=config, uow=uow)
    except Exception:
        uow.rollback()
        raise
    audit = {
        "kind": "modeling.artifact.calibrate",
        "target_ref": artifact.id,
        "outcome": "succeeded",
        "detail": {
            "method": method,
            "dataset_id": dataset.id,
            "sample_count": eval_sample_count,
            "calibration_path": calibration_path,
        },
    }
    set_params_on_connection = getattr(
        runtime.modeling_repo,
        "set_model_artifact_params_with_audit_on_connection",
        None,
    )
    transaction = getattr(runtime.modeling_repo, "transaction", None)
    if callable(set_params_on_connection) and callable(transaction):
        uow.finalize_with_connection(
            transaction,
            lambda conn: set_params_on_connection(conn, artifact.id, params, audit=audit),
        )
    else:
        uow.finalize(
            lambda: runtime.modeling_repo.set_model_artifact_params_with_audit(
                artifact.id,
                params,
                audit=audit,
            )
        )
    return {
        "artifact_id": artifact.id,
        "method": method,
        "calibration_path": str(base_dir / calibration_path),
        "split": split_name,
        "split_value": split_value,
        "fit_split": split_name,
        "eval_split": eval_split_name,
        "evaluated_on": evaluated_on,
        "sample_count": eval_sample_count,
        "fit_sample_count": int(labels.size),
        "brier_raw": raw_metrics["brier"],
        "brier_calibrated": calibrated_metrics["brier"],
        "ece_raw": raw_metrics["ece"],
        "ece_calibrated": calibrated_metrics["ece"],
        "per_split_metrics": per_split_metrics,
        "pmml_includes_calibration": False,
        "reliability_curve": reliability_curve,
    }


#: SEL-8: a segment (after merging groups below the floor) must retain at
#: least this many rows to get its own KS/AUC row -- otherwise the statistic is
#: too noisy to act on and the group is folded into __default__.
SEGMENT_MIN_GROUP_ROWS = 500


#: SEL-8: label used for every row whose original segment fell below
#: SEGMENT_MIN_GROUP_ROWS and was merged together.
SEGMENT_DEFAULT_GROUP_LABEL = "__default__"


def tool_segment_value_evaluation(inputs: dict, ctx) -> dict:
    """SEL-8 (scoped-down): diagnose whether a candidate segment column looks
    worth building separate per-segment models for, WITHOUT training any new
    model. Scores an already-trained artifact once (its normal single champion
    scorer) against the dataset, then reports the pooled KS/AUC alongside a
    per-segment breakdown of the SAME model's KS/AUC -- large spread between
    segments (a segment where the shared model performs far better/worse than
    pooled) is the signal that a dedicated per-segment model could plausibly
    help; a flat spread means segmentation is unlikely to pay for the added
    complexity of maintaining N models.

    Full per-segment model training/routing (the un-scoped SEL-8 spec) was
    judged to exceed the ~600-line budget once artifact routing, scoring
    replay, and report/monitoring plumbing for a genuinely new multi-model
    artifact shape were accounted for (see recipes/ensemble.py's SEL-6 blast
    radius as a lower-complexity precedent that alone ran ~600 lines) --
    this diagnostic-only tool is the explicitly-sanctioned degraded scope.
    Segments below SEGMENT_MIN_GROUP_ROWS rows are merged into
    SEGMENT_DEFAULT_GROUP_LABEL before scoring (mirrors the floor the full
    spec's per-segment training would have needed for the same reason: a
    KS/AUC computed on a handful of rows is not a usable statistic).
    """
    runtime = _runtime(ctx)
    artifact = _artifact(runtime, str(inputs["artifact_id"]))
    experiment = runtime.experiments.get(artifact.experiment_id)
    config = experiment.config
    if getattr(config, "target_type", "binary") != "binary":
        raise ModelingError("segment value evaluation is only supported for binary models")

    segment_col = str(inputs.get("segment_col") or "").strip()
    if not segment_col:
        raise ModelingError("segment_col is required")
    dataset_id = str(inputs.get("dataset_id") or config.dataset_id)
    dataset = runtime.registry.get(dataset_id)
    target_col = str(inputs.get("target_col") or config.target_col)
    split_col = str(inputs.get("split_col") or config.split_col)
    split_name = str(inputs.get("split") or "test")
    split_value = inputs.get("split_value", config.split_values.get(split_name, split_name))
    min_group_rows = int(inputs.get("min_group_rows") or SEGMENT_MIN_GROUP_ROWS)
    if min_group_rows < 1:
        raise ModelingError("min_group_rows must be at least 1")

    dataset_path = runtime.registry.resolve_path(dataset.id)
    columns = _unique_columns([*artifact.feature_list, target_col, split_col, segment_col])
    if segment_col not in runtime.backend.column_names(dataset_path):
        raise ModelingError(f"segment_col not found in dataset: {segment_col}")
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    sample = frame[frame[split_col] == split_value].copy()
    if sample.empty:
        raise ModelingError(f"segment evaluation split has no rows: {split_col}={split_value}")

    scorer = _ModelArtifactScorer(
        artifact,
        base_dir=_artifact_model_base_dir(runtime, artifact),
        load_calibration=False,
    )
    scores = np.asarray(scorer.score(sample, use_calibration=False), dtype=float)
    labels = pd.to_numeric(sample[target_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores) & np.isfinite(labels) & np.isin(labels, [0.0, 1.0])
    segment_values = sample[segment_col].astype("object").where(sample[segment_col].notna(), None)
    scores = scores[valid]
    labels = labels[valid].astype(int)
    segment_labels = [
        "(missing)" if value is None else str(value)
        for value in segment_values.to_numpy()[valid]
    ]
    if labels.size < 2 or len(set(labels.tolist())) < 2:
        raise ModelingError("segment evaluation sample must contain both positive and negative labels")

    grouped_labels = _merge_small_segments(segment_labels, min_group_rows=min_group_rows)
    pooled = {
        "sample_count": int(labels.size),
        "bad_rate": float(np.mean(labels)),
        "ks": feature_ks(scores, labels.astype(float)),
        "auc": feature_auc(scores, labels.astype(float)),
    }
    segment_rows = _segment_breakdown_rows(scores, labels, grouped_labels)
    segment_kss = [row["ks"] for row in segment_rows if row["segment"] != SEGMENT_DEFAULT_GROUP_LABEL]
    ks_spread = (max(segment_kss) - min(segment_kss)) if len(segment_kss) >= 2 else 0.0
    return {
        "artifact_id": artifact.id,
        "segment_col": segment_col,
        "split": split_name,
        "split_value": split_value,
        "min_group_rows": min_group_rows,
        "pooled": _jsonable(pooled),
        "segments": _jsonable(segment_rows),
        "segment_ks_spread": _jsonable(ks_spread),
        "note": (
            "诊断口径:同一模型在各分群上的 KS/AUC 分布,不训练分群模型;"
            "spread 越大越可能通过分群建模获益,仅供投入前评估参考。"
        ),
    }


def _merge_small_segments(segment_labels: list[str], *, min_group_rows: int) -> list[str]:
    """SEL-8: any segment label with fewer than min_group_rows rows is folded
    into SEGMENT_DEFAULT_GROUP_LABEL -- deterministic (pure function of the
    input labels + threshold, no randomness)."""
    counts: dict[str, int] = {}
    for label in segment_labels:
        counts[label] = counts.get(label, 0) + 1
    return [
        label if counts[label] >= min_group_rows else SEGMENT_DEFAULT_GROUP_LABEL
        for label in segment_labels
    ]


def _segment_breakdown_rows(
    scores: np.ndarray, labels: np.ndarray, grouped_labels: list[str],
) -> list[dict]:
    grouped = np.asarray(grouped_labels)
    rows = []
    for segment in sorted(set(grouped_labels)):
        mask = grouped == segment
        segment_scores = scores[mask]
        segment_labels_arr = labels[mask].astype(float)
        has_both_classes = len(set(labels[mask].tolist())) >= 2
        rows.append({
            "segment": segment,
            "sample_count": int(mask.sum()),
            "bad_rate": float(np.mean(labels[mask])) if mask.sum() else None,
            "ks": feature_ks(segment_scores, segment_labels_arr) if has_both_classes else None,
            "auc": feature_auc(segment_scores, segment_labels_arr) if has_both_classes else None,
        })
    return rows
