from __future__ import annotations

import numpy as np
import pandas as pd
import uuid
from marvis.artifacts import ArtifactUnitOfWork
from marvis.feature.metrics import feature_auc, feature_ks
from marvis.packs.modeling.errors import ModelingError
from marvis.validation.binning import bin_distribution, compute_psi

from marvis.packs.modeling._common import _effective_seed, _optional_float, _optional_str, _unique_columns
from marvis.packs.modeling._runtime import _artifact, _artifact_base_dir, _runtime
from marvis.packs.modeling.delivery_tools import _monitoring_check_status
from marvis.packs.modeling.scoring import _ModelArtifactScorer


def tool_score_dataset(inputs: dict, ctx) -> dict:
    """S1b/DOM-3: apply a trained artifact to a new (unscored) dataset, closing the
    "no tool applies a trained model to new data" gap. Reuses load_model +
    _ModelArtifactScorer with replay_preprocessing=True (the PREP-2 escape hatch
    reserved for exactly this: scoring genuinely new raw data, as opposed to the
    already-transformed modeling frame the model trained on) so impute/cap/
    normalize/onehot/woe steps are replayed deterministically before scoring.
    Missing preprocessing input columns / unseen WOE categories fall back to each
    step's own existing tolerance (marvis.feature.preprocessing); a feature that is
    still absent after replay surfaces as a normal KeyError from the scorer, not a
    silent zero-fill.

    Writes a PD column (``model_score``) and, for scorecard artifacts, a points
    column (``scorecard_points``), then registers the scored frame as a derived
    dataset with direction metadata copied verbatim from the artifact (never
    re-inferred) and a ``modeling.dataset.scored`` audit entry.
    """
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)

    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    frame = runtime.backend.read_frame(dataset_path)
    row_count = int(len(frame))

    score_col = str(inputs.get("output_col") or "model_score").strip() or "model_score"
    scorer = _ModelArtifactScorer(artifact, base_dir=base_dir, replay_preprocessing=True)
    scores = scorer.score(frame)
    frame[score_col] = scores
    score_missing_rate = float(np.mean(~np.isfinite(np.asarray(scores, dtype=float)))) if row_count else 0.0

    points_col = None
    points_missing_rate = None
    scorecard_points = scorer.scorecard_points(frame)
    if scorecard_points is not None:
        points_col = str(inputs.get("points_col") or "scorecard_points").strip() or "scorecard_points"
        frame[points_col] = scorecard_points
        points_missing_rate = (
            float(np.mean(~np.isfinite(np.asarray(scorecard_points, dtype=float)))) if row_count else 0.0
        )

    out_dir = runtime.datasets_root / str(ctx.task_id) / "modeling"
    uow = ArtifactUnitOfWork()
    staged = uow.stage_file(out_dir, f"scored_{uuid.uuid4().hex}.parquet")
    try:
        frame.to_parquet(staged.path, index=False)

        def audit_factory(registered_dataset):
            return {
                "kind": "modeling.dataset.scored",
                "target_ref": registered_dataset.id,
                "outcome": "succeeded",
                "detail": {
                    "source_dataset_id": dataset.id,
                    "experiment_id": experiment.id,
                    "artifact_id": artifact.id,
                    "score_col": score_col,
                    "points_col": points_col,
                    "score_direction": artifact.score_direction,
                    "points_direction": artifact.points_direction,
                    "row_count": row_count,
                    "score_missing_rate": score_missing_rate,
                },
            }

        registered = uow.finalize_with_connection(
            runtime.repo.transaction,
            lambda conn: runtime.registry.register_existing_with_audit_on_connection(
                conn,
                staged.final_path,
                audit_factory=audit_factory,
                task_id=str(ctx.task_id),
                role="modeling.scored",
                anchor_target=dataset.id,
                seed=_effective_seed(inputs, ctx),
            ),
        )
    except Exception:
        uow.rollback()
        raise
    return {
        "result_dataset_id": registered.id,
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "score_col": score_col,
        "points_col": points_col,
        "score_direction": artifact.score_direction,
        "points_direction": artifact.points_direction,
        "row_count": row_count,
        "score_missing_rate": score_missing_rate,
        "points_missing_rate": points_missing_rate,
    }


#: S1b/DOM-3: monitor_run's own threshold defaults (distinct from
#: DEFAULT_MONITORING_THRESHOLDS above, which compares train/test/oot splits at
#: *training* time -- these compare a *new* dataset against the training-time
#: baseline snapshot). score_psi/feature_csi follow the industry-standard PSI
#: bands (<0.10 stable, 0.10-0.25 moderate drift/warn, >=0.25 material drift/
#: fail); oot_ks_drop/oot_auc_drop flag when live discrimination has fallen well
#: below the model's own development-period reading.
MONITOR_RUN_THRESHOLDS = {
    "score_psi": {
        "label": "Score PSI vs baseline",
        "metric": "score_psi",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "feature_csi_max": {
        "label": "Max feature CSI vs baseline",
        "metric": "feature_csi_max",
        "direction": "max",
        "warn": 0.10,
        "fail": 0.25,
    },
    "ks_drop": {
        "label": "KS drop vs development",
        "metric": "ks_drop",
        "direction": "max",
        "warn": 0.05,
        "fail": 0.10,
    },
    "auc_drop": {
        "label": "AUC drop vs development",
        "metric": "auc_drop",
        "direction": "max",
        "warn": 0.03,
        "fail": 0.05,
    },
}


#: Maps the shared _monitoring_check_status vocabulary (pass/warn/fail/missing/
#: needs_policy) onto monitor_run's green/amber/red judgement (DOM-3 spec
#: wording). "missing" is intentionally amber here (a PSI/CSI check that
#: couldn't be computed is a real data-quality problem on a check that should
#: always have a value) -- the no-label KS/AUC case is handled separately as an
#: explicit "n/a" check, never silently downgraded to missing/amber.
_MONITOR_STATUS_TO_LEVEL = {
    "pass": "green",
    "warn": "amber",
    "missing": "amber",
    "needs_policy": "amber",
    "fail": "red",
}


def tool_monitor_run(inputs: dict, ctx) -> dict:
    """S1b/DOM-3: execute one monitoring run against the training-time baseline
    snapshot -- score PSI, per-feature CSI (top drifted features), and (only when
    the sample carries valid binary labels) KS/AUC compared against the model's
    own development-period reading. Judged against monitor_run's own threshold
    policy (MONITOR_RUN_THRESHOLDS, overridable via ``monitoring_policy``) into a
    green/amber/red verdict per check plus an overall verdict, written as a
    ``modeling.monitor.run`` audit entry.

    Accepts either ``dataset_id`` (raw new data -- scored internally via
    _ModelArtifactScorer with replay_preprocessing=True, exactly like
    score_dataset) or ``scored_dataset_id`` (already scored, e.g. by a prior
    score_dataset call; ``score_col`` names the column, default "model_score").

    Old artifacts trained before S1b carry no baseline_distributions snapshot;
    this raises ModelingError explicitly rather than fabricating a reference
    distribution -- the caller must retrain or supply an explicit baseline.
    """
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    if experiment.artifact_id is None:
        raise ModelingError(f"experiment has no artifact: {experiment.id}")
    artifact = _artifact(runtime, experiment.artifact_id)
    baseline = artifact.baseline_distributions
    if not baseline:
        raise ModelingError(
            f"artifact {artifact.id!r} has no training-time baseline distribution snapshot "
            "(trained before S1b, or a non-binary target that baseline snapshots don't cover); "
            "monitor_run has no reference to compare against -- retrain this experiment to "
            "capture a baseline, or run monitoring against a different experiment that has one."
        )
    base_dir = _artifact_base_dir(runtime.settings, experiment.task_id)

    scored_dataset_id = _optional_str(inputs.get("scored_dataset_id"))
    dataset_id = _optional_str(inputs.get("dataset_id"))
    if not scored_dataset_id and not dataset_id:
        raise ModelingError("monitor_run requires either scored_dataset_id or dataset_id")
    score_col = str(inputs.get("score_col") or "model_score").strip() or "model_score"
    target_col = _optional_str(inputs.get("target_col"))

    if scored_dataset_id:
        dataset = runtime.registry.get(scored_dataset_id)
        dataset_path = runtime.registry.resolve_path(dataset.id)
        # LT-6: monitor_run only ever reads score_col/feature_list/target_col off this
        # frame (never writes it back), so project instead of pulling the full modeling
        # frame -- filtered against the dataset's actual columns first (not requested
        # blind) so a feature_list entry or target_col absent from THIS dataset degrades
        # the same way it always has (_monitor_run_feature_csi_checks' `feature not in
        # frame.columns` skip / _monitor_run_label_checks' n/a rows) instead of read_frame
        # raising on an unknown column.
        available = set(runtime.backend.column_names(dataset_path))
        columns = _unique_columns([score_col, target_col, *artifact.feature_list])
        columns = [column for column in columns if column in available]
        frame = runtime.backend.read_frame(dataset_path, columns=columns)
        if score_col not in frame.columns:
            raise ModelingError(
                f"scored dataset {dataset.id!r} has no column {score_col!r}; "
                "pass score_col to name the column score_dataset actually wrote."
            )
        scores = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=float)
    else:
        dataset = runtime.registry.get(str(dataset_id))
        dataset_path = runtime.registry.resolve_path(dataset.id)
        # LT-6: NOT column-projected, unlike the scored_dataset_id branch above --
        # replay_preprocessing=True means the preprocessing chain may reference raw
        # input columns (e.g. a pre-onehot/pre-woe source column) that are not
        # themselves in artifact.feature_list, and apply_preprocessing_steps silently
        # SKIPS a step whose input column is absent from the frame rather than raising
        # (see marvis.feature.preprocessing.apply_preprocessing_steps). Projecting here
        # could silently drop a preprocessing step instead of erroring, corrupting the
        # replayed scores/CSI -- exactly the drift the task's hard constraint forbids.
        frame = runtime.backend.read_frame(dataset_path)
        scorer = _ModelArtifactScorer(artifact, base_dir=base_dir, replay_preprocessing=True)
        scores = np.asarray(scorer.score(frame), dtype=float)

    monitoring_policy_source = inputs.get("monitoring_policy") if isinstance(inputs.get("monitoring_policy"), dict) else {}
    thresholds = _monitor_run_thresholds(monitoring_policy_source.get("thresholds"))

    score_check = _monitor_run_score_psi_check(scores, baseline, thresholds["score_psi"])
    feature_checks, drifted_features = _monitor_run_feature_csi_checks(
        frame, artifact.feature_list, baseline, thresholds["feature_csi_max"]
    )
    label_checks = _monitor_run_label_checks(
        frame, scores, target_col, experiment.metrics, thresholds
    )

    checks = [score_check, *feature_checks, *label_checks]
    overall_level = _monitor_run_overall_level(check["level"] for check in checks)
    recommendation = _monitor_run_recommendation(overall_level)

    row_count = int(len(frame))
    runtime.repo.write_audit(
        kind="modeling.monitor.run",
        target_ref=experiment.id,
        outcome="succeeded",
        detail={
            "artifact_id": artifact.id,
            "dataset_id": dataset.id,
            "row_count": row_count,
            "overall_level": overall_level,
            "score_psi": score_check.get("value"),
            "feature_csi_max": (feature_checks[0].get("value") if feature_checks else None),
        },
    )
    return {
        "experiment_id": experiment.id,
        "artifact_id": artifact.id,
        "dataset_id": dataset.id,
        "row_count": row_count,
        "score_direction": artifact.score_direction,
        "overall_level": overall_level,
        "recommendation": recommendation,
        "checks": checks,
        "top_drifted_features": drifted_features,
    }


def _monitor_run_thresholds(source) -> dict:
    thresholds = {key: dict(value) for key, value in MONITOR_RUN_THRESHOLDS.items()}
    if not isinstance(source, dict):
        return thresholds
    for key, override in source.items():
        if not isinstance(override, dict) or key not in thresholds:
            continue
        merged = dict(thresholds[key])
        for field in ("label", "metric", "direction", "warn", "fail"):
            if field in override:
                merged[field] = override[field]
        thresholds[key] = merged
    return thresholds


def _monitor_run_level(value, spec: dict) -> tuple[str, str, str]:
    """Returns (level, status, message) -- level is the green/amber/red verdict,
    status is the underlying pass/warn/fail/missing/needs_policy reason code."""
    status, message = _monitoring_check_status(
        value, direction=str(spec.get("direction") or "max"), warn=_optional_float(spec.get("warn")), fail=_optional_float(spec.get("fail"))
    )
    return _MONITOR_STATUS_TO_LEVEL.get(status, "amber"), status, message


def _monitor_run_score_psi_check(scores: np.ndarray, baseline: dict, spec: dict) -> dict:
    edges = np.asarray(baseline.get("score_edges") or [], dtype=float)
    train_dist = ((baseline.get("score_distribution") or {}).get("train") or {}).get("bin_proportions")
    finite_scores = scores[np.isfinite(scores)]
    if edges.size < 2 or not train_dist or finite_scores.size == 0:
        level, status, message = _monitor_run_level(None, spec)
        return {
            "id": "score_psi", "label": spec.get("label"), "metric": "score_psi",
            "value": None, "level": level, "status": status, "message": message,
        }
    actual_dist = bin_distribution(finite_scores, edges)
    psi = compute_psi(np.asarray(train_dist, dtype=float), actual_dist)
    level, status, message = _monitor_run_level(psi, spec)
    return {
        "id": "score_psi", "label": spec.get("label"), "metric": "score_psi",
        "value": float(psi), "level": level, "status": status, "message": message,
        "sample_count": int(finite_scores.size),
    }


def _monitor_run_feature_csi_checks(
    frame: pd.DataFrame, feature_list: tuple[str, ...], baseline: dict, spec: dict
) -> tuple[list[dict], list[dict]]:
    feature_baselines = baseline.get("feature_distributions") or {}
    rows: list[dict] = []
    for feature in feature_list:
        feature_baseline = feature_baselines.get(feature)
        if not isinstance(feature_baseline, dict) or feature not in frame.columns:
            continue
        edges = np.asarray(feature_baseline.get("quantile_edges") or [], dtype=float)
        if edges.size < 2:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        actual_dist = bin_distribution(finite_values, edges)
        bin_count = edges.size - 1
        expected_dist = np.full(bin_count, 1.0 / bin_count, dtype=float)
        csi = compute_psi(expected_dist, actual_dist)
        rows.append({"feature": str(feature), "csi": float(csi), "sample_count": int(finite_values.size)})
    rows.sort(key=lambda item: (-item["csi"], item["feature"]))
    worst_value = rows[0]["csi"] if rows else None
    level, status, message = _monitor_run_level(worst_value, spec)
    summary_check = {
        "id": "feature_csi_max", "label": spec.get("label"), "metric": "feature_csi_max",
        "value": worst_value, "level": level, "status": status, "message": message,
        "top_feature": rows[0]["feature"] if rows else None,
    }
    return [summary_check], rows[:10]


def _monitor_run_label_checks(
    frame: pd.DataFrame,
    scores: np.ndarray,
    target_col: str | None,
    dev_metrics,
    thresholds: dict,
) -> list[dict]:
    """When the sample has no target_col or no valid binary labels, both
    ks_drop/auc_drop checks are explicit n/a rows (level "n/a", not fabricated as
    missing/amber) -- an unlabeled monitoring run is a completely normal scenario
    (labels mature months after scoring), not a data quality problem."""
    labels = None
    if target_col and target_col in frame.columns:
        raw_labels = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(scores) & np.isfinite(raw_labels) & np.isin(raw_labels, [0.0, 1.0])
        if int(finite_mask.sum()) > 0:
            labels = raw_labels[finite_mask].astype(int)
            live_scores = scores[finite_mask]

    if labels is None or np.unique(labels).size < 2:
        return [
            {
                "id": check_id, "label": thresholds[check_id].get("label"), "metric": check_id,
                "value": None, "level": "n/a", "status": "n/a",
                "message": "本次样本无有效标签(或标签单一类别),无法计算 KS/AUC 与开发期对比;这是正常的评分期监控场景,不代表数据质量问题。",
            }
            for check_id in ("ks_drop", "auc_drop")
        ]

    live_ks = feature_ks(live_scores, labels)
    live_auc = feature_auc(live_scores, labels, direction_agnostic=True)
    dev_ks = getattr(dev_metrics, "train_ks", None) if dev_metrics is not None else None
    dev_auc = getattr(dev_metrics, "train_auc", None) if dev_metrics is not None else None

    checks: list[dict] = []
    for check_id, live_value, dev_value in (("ks_drop", live_ks, dev_ks), ("auc_drop", live_auc, dev_auc)):
        spec = thresholds[check_id]
        drop = (float(dev_value) - float(live_value)) if isinstance(dev_value, (int, float)) else None
        level, status, message = _monitor_run_level(drop, spec)
        checks.append({
            "id": check_id, "label": spec.get("label"), "metric": check_id,
            "value": drop, "level": level, "status": status, "message": message,
            "live_value": float(live_value), "dev_value": (float(dev_value) if isinstance(dev_value, (int, float)) else None),
            "sample_count": int(labels.size),
        })
    return checks


def _monitor_run_overall_level(levels) -> str:
    values = {str(level) for level in levels}
    if "red" in values:
        return "red"
    if "amber" in values:
        return "amber"
    return "green"


def _monitor_run_recommendation(level: str) -> str:
    if level == "green":
        return "分布稳定,可继续按监控计划执行。"
    if level == "red":
        return "存在显著漂移或性能下降,建议触发模型风险复核或重训评估。"
    return "存在轻中度漂移或指标缺口,建议加强观察频率并复核相关特征。"
