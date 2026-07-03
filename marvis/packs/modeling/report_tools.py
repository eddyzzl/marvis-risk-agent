from __future__ import annotations

import json
import numpy as np
import pandas as pd
import re
from marvis.artifacts import ArtifactUnitOfWork, TransactionalArtifactStore
from marvis.feature.binning import equal_frequency_edges
from marvis.feature.metrics import DEFAULT_IV_BINS, feature_metrics, feature_psi
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.llm_prompts import REPORT_NARRATIVE_SYS as _REPORT_NARRATIVE_SYS_SPEC
from marvis.llm_settings import LLMSettingsError, resolve_llm_model
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.contracts import ModelArtifact, TrainConfig
from marvis.packs.modeling.errors import ModelingError, ReportScoreMissingError
from marvis.packs.modeling.report_compute import BusinessColumns, build_feature_dictionary, compute_amount_bin_table, compute_sample_analysis, compute_vintage_report, resolve_report_sections, stress_low_pricing
from marvis.validation.config import ValidationConfig
from marvis.validation.stress_test import run_stress_test
from pathlib import Path

from marvis.packs.modeling._common import MODEL_REPORT_SCORE_COL, SCORECARD_POINTS_COL, _NUMBER_TOKEN_RE, _allowed_number_tokens, _business_columns, _jsonable, _number_token_allowed, _optional_str, _ratio, _section_available, _unique_columns
from marvis.packs.modeling._runtime import _Runtime, _artifact, _artifact_model_base_dir, _cached_dataset_runtime, _runtime
from marvis.packs.modeling.scoring import _ModelArtifactScorer, _artifact_calibration_rows


def tool_generate_model_report(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    experiment = runtime.experiments.get(str(inputs["experiment_id"]))
    report_path = Path(runtime.settings.tasks_dir) / ctx.task_id / "outputs" / "model_report.xlsx"
    return _generate_model_report_for(runtime, ctx, experiment, inputs, report_path)


def tool_generate_model_reports(inputs: dict, ctx) -> dict:
    """MODELING §5 multi-version fan-out: render one report per requested experiment
    using version-specific output paths. Each report reuses the single-report pipeline.
    report_path mirrors the first report so the existing download endpoint stays
    compatible."""
    runtime = _runtime(ctx)
    experiment_ids = [str(item) for item in inputs.get("experiment_ids") or []]
    if not experiment_ids:
        raise ModelingError("experiment_ids must not be empty")
    outputs_dir = Path(runtime.settings.tasks_dir) / ctx.task_id / "outputs"
    reports: list[dict] = []
    for experiment_id in experiment_ids:
        experiment = runtime.experiments.get(experiment_id)
        recipe = str(experiment.recipe_id)
        report_path = outputs_dir / _report_filename(recipe, experiment_id)
        generated = _generate_model_report_for(runtime, ctx, experiment, inputs, report_path)
        reports.append({
            "experiment_id": experiment_id,
            "recipe": recipe,
            "report_path": generated["report_path"],
        })
    return {
        "reports": reports,
        "report_path": reports[0]["report_path"] if reports else "",
    }


_REPORT_FILENAME_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_-]+")


def _report_filename(recipe: str, experiment_id: str) -> str:
    safe_recipe = _REPORT_FILENAME_UNSAFE_RE.sub("_", recipe).strip("_") or "model"
    safe_id = _REPORT_FILENAME_UNSAFE_RE.sub("_", experiment_id)[:8]
    return f"model_report_{safe_recipe}_{safe_id}.xlsx"


def _generate_model_report_for(runtime: _Runtime, ctx, experiment, inputs: dict, report_path: Path) -> dict:
    # The full report is binary-credit-specific (bad-rate / Vintage / OOT bins / stress).
    # For a non-binary target (regression / multiclass) write a compact metrics report so
    # the flow finishes with a downloadable artifact instead of crashing on binary-only math.
    if getattr(experiment.config, "target_type", "binary") != "binary":
        from marvis.output.model_report_minimal import render_minimal_model_report

        statuses = [
            {"section": "汇总", "status": "ok"},
            {"section": "模型指标", "status": "ok"},
        ]
        uow = ArtifactUnitOfWork()
        staged_report = uow.stage_file(report_path.parent, report_path.name)
        try:
            render_minimal_model_report(experiment, staged_report.path)
            _finalize_model_report_write(
                runtime,
                uow,
                experiment=experiment,
                report_path=staged_report.final_path,
                section_status=statuses,
            )
        except Exception:
            uow.rollback()
            raise
        return {
            "report_path": str(report_path),
            "section_status": statuses,
            "scorecard_table": [],
            "score_bands": [],
        }
    artifact = _artifact(runtime, experiment.artifact_id) if experiment.artifact_id else None
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    dataset_path = runtime.registry.resolve_path(dataset.id)
    business = _business_columns(inputs.get("business_columns") or {})
    statuses = resolve_report_sections(
        business,
        _optional_str(inputs.get("feature_dictionary_id")),
    )
    sample = None
    if _section_available(statuses, "sample_analysis") and business.loan_month_col:
        sample = compute_sample_analysis(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            target_col=experiment.config.target_col,
            business=business,
            mob_cols=business.mob_observe_cols,
        )
    vintage = None
    if _section_available(statuses, "vintage") and business.loan_month_col:
        vintage = compute_vintage_report(
            runtime.backend,
            dataset_path,
            loan_month_col=business.loan_month_col,
            mob_observe_cols=business.mob_observe_cols,
            amount_col=business.loan_amount_col,
        )

    report_dataset_path, score_col, report_frame = _report_scored_dataset(
        runtime,
        dataset_path,
        artifact,
        experiment.config,
        task_id=ctx.task_id,
        experiment_id=experiment.id,
        dataset_id=dataset.id,
    )
    report_runtime = _cached_dataset_runtime(runtime, report_dataset_path, frame=report_frame)
    low_pricing = None
    if _section_available(statuses, "low_pricing") and business.interest_rate_col:
        low_pricing = stress_low_pricing(
            report_runtime.backend,
            report_dataset_path,
            score_col=score_col,
            target_col=experiment.config.target_col,
            interest_rate_col=business.interest_rate_col,
            low_pricing_threshold=None,
        )
    oot_bin = _report_bin_table(
        report_runtime,
        report_dataset_path,
        score_col=score_col,
        target_col=experiment.config.target_col,
        config=experiment.config,
        business=business,
    )
    feature_dictionary_id = _optional_str(inputs.get("feature_dictionary_id"))
    feature_dictionary = (
        build_feature_dictionary(runtime.backend, feature_dictionary_id, runtime.registry)
        if feature_dictionary_id
        else {}
    )
    feature_importance = _feature_importance_rows(artifact, feature_dictionary=feature_dictionary)
    scorecard_table = _scorecard_table_rows(artifact)
    score_band_col = (
        SCORECARD_POINTS_COL
        if artifact is not None and artifact.algorithm == "scorecard"
        else score_col
    )
    score_bands = _score_band_rows(
        report_runtime,
        report_dataset_path,
        score_col=score_band_col,
        target_col=experiment.config.target_col,
        config=experiment.config,
    )
    stress_product_removal = _stress_product_removal(
        report_runtime,
        report_dataset_path,
        artifact,
        experiment.config,
        feature_dictionary,
    )
    split_profile = _dataset_split_profile(
        report_runtime,
        report_dataset_path,
        experiment.config,
        window_col=business.loan_month_col,
    )
    calibration = _artifact_calibration_rows(artifact)
    structured_summary = _report_structured_summary(
        project_meta=dict(inputs.get("project_meta") or {}),
        dataset_split=_dataset_split_rows(experiment.metrics, split_profile=split_profile),
        stability=_stability_rows(experiment.metrics),
        sample_analysis=sample,
        vintage=vintage,
        feature_importance=feature_importance,
        scorecard_table=scorecard_table,
        score_bands=score_bands,
        calibration=calibration,
        univariate=_univariate_rows(report_runtime, report_dataset_path, artifact, experiment.config),
        oot_bin_table=oot_bin,
        stress_product_removal=stress_product_removal,
        stress_low_pricing=low_pricing,
        section_status=statuses,
    )
    narratives = _guard_no_invented_numbers(
        _draft_report_narratives(
            structured_summary,
            llm_factory=_report_llm_factory(runtime.settings.workspace, _optional_str(inputs.get("model_id"))),
        ),
        structured_summary,
    )
    scored_dataset_path = report_dataset_path if report_dataset_path != dataset_path else None
    uow = ArtifactUnitOfWork()
    staged_report = uow.stage_file(report_path.parent, report_path.name)
    try:
        render_model_report(
            ModelReportPayload(
                project_meta=structured_summary["project_meta"],
                dataset_split=structured_summary["dataset_split"],
                stability=structured_summary["stability"],
                sample_analysis=sample,
                vintage=vintage,
                feature_importance=structured_summary["feature_importance"],
                scorecard_table=structured_summary["scorecard_table"],
                score_bands=structured_summary["score_bands"],
                calibration=structured_summary["calibration"],
                univariate=structured_summary["univariate"],
                oot_bin_table=oot_bin,
                stress_product_removal=stress_product_removal,
                stress_low_pricing=low_pricing,
                narratives=narratives,
                section_status=statuses,
            ),
            staged_report.path,
        )
        _finalize_model_report_write(
            runtime,
            uow,
            experiment=experiment,
            report_path=staged_report.final_path,
            section_status=statuses,
            scored_dataset_path=scored_dataset_path,
        )
    except Exception:
        uow.rollback()
        # The scored dataset (when freshly written by _report_scored_dataset above)
        # already promoted/committed under its OWN transactional store before this
        # try block started -- deleting it here is a deliberate policy choice (a
        # scored parquet with no matching report is not useful on its own), not a
        # transaction-boundary bug, so it stays a best-effort unlink rather than
        # joining the report's UoW.
        if scored_dataset_path is not None and scored_dataset_path.name == "model_report_scored.parquet":
            scored_dataset_path.unlink(missing_ok=True)
        raise
    return {
        "report_path": str(report_path),
        "section_status": [_jsonable(status) for status in statuses],
        "scorecard_table": structured_summary["scorecard_table"],
        "score_bands": structured_summary["score_bands"],
        "calibration": structured_summary["calibration"],
    }


def _model_report_audit_kwargs(
    *,
    experiment,
    report_path: Path,
    section_status: list[dict],
    scored_dataset_path: Path | None = None,
) -> dict:
    artifact_id = experiment.artifact_id or ""
    return {
        "kind": "modeling.report.generated",
        "target_ref": experiment.id,
        "outcome": "succeeded",
        "detail": {
            "artifact_id": artifact_id,
            "report_path": str(report_path),
            "scored_dataset_path": str(scored_dataset_path) if scored_dataset_path else "",
            "section_status": [_jsonable(status) for status in section_status],
        },
    }


def _finalize_model_report_write(
    runtime: _Runtime,
    uow: ArtifactUnitOfWork,
    *,
    experiment,
    report_path: Path,
    section_status: list[dict],
    scored_dataset_path: Path | None = None,
) -> None:
    """LT-5: promote the staged report file and write its audit row as one unit --
    a process kill between "report bytes written" and "audit row committed" must
    not leave a torn/partial report at the real path with no record of it. Prefers
    a connection-scoped audit write sharing the promote/commit boundary when the
    repo exposes one; falls back to the older single-call ``write_audit`` (still a
    strict improvement over the previous non-staged direct write)."""
    audit_kwargs = _model_report_audit_kwargs(
        experiment=experiment,
        report_path=report_path,
        section_status=section_status,
        scored_dataset_path=scored_dataset_path,
    )
    write_audit_on_connection = getattr(runtime.repo, "write_audit_on_connection", None)
    transaction = getattr(runtime.repo, "transaction", None)
    if callable(write_audit_on_connection) and callable(transaction):
        uow.finalize_with_connection(
            transaction,
            lambda conn: write_audit_on_connection(conn, **audit_kwargs),
        )
    else:
        uow.finalize(lambda: runtime.repo.write_audit(**audit_kwargs))


def _dataset_split_rows(metrics, *, split_profile: dict[str, dict] | None = None) -> list[dict]:
    if metrics is None:
        return []
    split_profile = split_profile or {}
    if metrics.train_rmse is not None:
        return [
            {
                "split": "train",
                **split_profile.get("train", {}),
                "rmse": metrics.train_rmse,
                "mae": metrics.train_mae,
                "r2": metrics.train_r2,
            },
            {
                "split": "test",
                **split_profile.get("test", {}),
                "rmse": metrics.test_rmse,
                "mae": metrics.test_mae,
                "r2": metrics.test_r2,
            },
            {
                "split": "oot",
                **split_profile.get("oot", {}),
                "rmse": metrics.oot_rmse,
                "mae": metrics.oot_mae,
                "r2": metrics.oot_r2,
            },
        ]
    return [
        {"split": "train", **split_profile.get("train", {}), "ks": metrics.train_ks, "auc": metrics.train_auc},
        {"split": "test", **split_profile.get("test", {}), "ks": metrics.test_ks, "auc": metrics.test_auc},
        {"split": "oot", **split_profile.get("oot", {}), "ks": metrics.oot_ks, "auc": metrics.oot_auc},
    ]


def _dataset_split_profile(
    runtime: _Runtime,
    dataset_path: Path,
    config: TrainConfig,
    *,
    window_col: str | None = None,
) -> dict[str, dict]:
    frame = runtime.backend.read_frame(dataset_path, columns=_unique_columns([config.split_col, config.target_col, window_col]))
    target = pd.to_numeric(frame[config.target_col], errors="coerce")
    binary_target = target.dropna().isin([0, 1]).all()
    profile = {}
    for split in ("train", "test", "oot"):
        split_value = config.split_values.get(split, split)
        split_mask = frame[config.split_col] == split_value
        group_target = target[split_mask]
        row = {"sample_count": int(len(group_target))}
        if binary_target:
            row["bad_rate"] = _ratio(float((group_target == 1).sum()), float(len(group_target)))
        if window_col and window_col in frame.columns:
            window_values = sorted(str(value) for value in frame.loc[split_mask, window_col].dropna().unique())
            if window_values:
                row["window_start"] = window_values[0]
                row["window_end"] = window_values[-1]
        profile[split] = row
    return profile


def _stability_rows(metrics) -> list[dict]:
    if metrics is None:
        return []
    if metrics.train_rmse is not None:
        return [
            {"metric": "rmse_test_minus_train", "value": metrics.overfit_train_test_gap},
            {"metric": "rmse_oot_minus_train", "value": metrics.overfit_train_oot_gap},
            {"metric": "overfit_flag", "value": metrics.overfit_flag},
        ]
    return [
        {"metric": "psi_test_vs_train", "value": metrics.psi_test_vs_train},
        {"metric": "psi_oot_vs_train", "value": metrics.psi_oot_vs_train},
        {"metric": "overfit_flag", "value": metrics.overfit_flag},
    ]


def _feature_importance_rows(artifact: ModelArtifact | None, *, feature_dictionary: dict | None = None) -> list[dict]:
    if artifact is None:
        return []
    dictionary = feature_dictionary or {}
    metadata_keys = ("含义", "产品名称", "厂商名称")
    importance_pairs = artifact.feature_importance or tuple((feature, 0.0) for feature in artifact.feature_list)
    total_importance = sum(float(importance) for _, importance in importance_pairs)
    cumulative_importance = 0.0
    rows = []
    for feature, importance in importance_pairs:
        importance_value = float(importance)
        cumulative_importance += importance_value
        row = {
            "feature": feature,
            "importance": importance_value,
            "importance_pct": _ratio(importance_value, total_importance),
            "cumulative_importance_pct": _ratio(cumulative_importance, total_importance),
        }
        if dictionary:
            metadata = dictionary.get(str(feature))
            row.update({
                key: metadata.get(key) if isinstance(metadata, dict) and metadata.get(key) not in ("",) else None
                for key in metadata_keys
            })
        rows.append(row)
    return rows


def _scorecard_table_rows(artifact: ModelArtifact | None) -> list[dict]:
    if artifact is None or artifact.algorithm != "scorecard":
        return []
    if not artifact.scorecard_table:
        return [{
            "feature": "__missing__",
            "bin_label": "旧 artifact 未包含评分卡表,需重训或回填后查看 points 明细",
            "points": None,
        }]
    return [dict(row) for row in artifact.scorecard_table]


def _score_band_direction(score_col: str) -> str:
    """DOM-5: cumulation direction for the score-band sheet. Scorecard points are
    higher-is-better (higher points = lower risk); the native model score is a PD
    (higher = higher risk). No `score_direction` parameter exists platform-wide yet
    (DOM-2, separate item) so direction is inferred from which column is being
    banded -- the only two score columns this function is ever called with."""
    return "higher_is_better" if score_col == SCORECARD_POINTS_COL else "higher_is_riskier"


def _score_band_rows(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    config: TrainConfig,
    bin_count: int = 10,
) -> list[dict]:
    """DOM-5: score bands on a single set of bin edges shared by every split.

    Edges are computed once on the ``train`` split (equal-frequency) so that bin N
    means the same score range in train/test/oot -- required to read cutoff impact
    (e.g. "if cutoff = edge of bin 6, OOT approval/bad rate is X") off the table and
    to compare distribution migration across splits. test/oot reuse train's edges
    verbatim; when train is missing or has no finite scores, edges fall back to the
    first available split (so the sheet degrades instead of going empty) and the
    fallback source is recorded in every row via ``bin_edges_source``.

    Cumulation direction (cum_count_pct / cum_bad_rate / cum_bad_capture) follows
    ``_score_band_direction``: for a higher-is-riskier PD score, cumulation runs from
    the highest bin down (an "approve everyone at or below this score" cutoff reading);
    for higher-is-better scorecard points, it runs from the lowest bin up. Either way
    ``cum_count_pct`` reads as "share of population that would be approved if the
    cutoff were placed at this band's boundary".
    """
    from marvis.validation.binning import assign_bins, equal_frequency_bin_edges

    columns = _unique_columns([score_col, target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    direction = _score_band_direction(score_col)

    train_value = config.split_values.get("train")
    edges: np.ndarray | None = None
    edges_source = "train"
    if train_value is not None:
        train_scores = pd.to_numeric(
            frame.loc[frame[config.split_col] == train_value, score_col], errors="coerce"
        ).to_numpy(dtype=float)
        finite_train = train_scores[np.isfinite(train_scores)]
        if finite_train.size > 0:
            edges = equal_frequency_bin_edges(finite_train, int(bin_count))
    if edges is None:
        # No usable train split (missing/empty/all-NaN score) -- fall back to the first
        # split with finite scores so the sheet still renders, and say so per row.
        for split_name, split_value in config.split_values.items():
            candidate = pd.to_numeric(
                frame.loc[frame[config.split_col] == split_value, score_col], errors="coerce"
            ).to_numpy(dtype=float)
            finite_candidate = candidate[np.isfinite(candidate)]
            if finite_candidate.size > 0:
                edges = equal_frequency_bin_edges(finite_candidate, int(bin_count))
                edges_source = split_name
                break
    if edges is None:
        return []

    overall_bad_rate_by_split: dict[str, float | None] = {}
    rows: list[dict] = []
    for split_name, split_value in config.split_values.items():
        split_frame = frame[frame[config.split_col] == split_value]
        if split_frame.empty:
            continue
        scores = pd.to_numeric(split_frame[score_col], errors="coerce").to_numpy(dtype=float)
        assigned = assign_bins(scores, edges)
        labels = pd.to_numeric(split_frame[target_col], errors="coerce").to_numpy(dtype=float)
        labeled_mask = np.isfinite(labels)
        total_count = int(np.sum(assigned > 0))
        total_labeled = int(np.sum(labeled_mask & (assigned > 0)))
        total_bad = int(np.sum(labels[labeled_mask & (assigned > 0)] == 1)) if total_labeled else 0
        overall_bad_rate = (total_bad / total_labeled) if total_labeled else None
        overall_bad_rate_by_split[split_name] = overall_bad_rate

        # Cumulation walks bin indices in "approval order": worst-score-first for a
        # higher-is-riskier score (approving low scores first) or best-score-first for
        # higher-is-better points -- either way starting from bin index len(edges)-1
        # down to 1 for higher_is_riskier, or 1 up to len(edges)-1 for higher_is_better.
        bin_indices = list(range(1, len(edges)))
        cum_order = list(reversed(bin_indices)) if direction == "higher_is_riskier" else bin_indices
        band_rows: dict[int, dict] = {}
        cum_count = 0
        cum_labeled = 0
        cum_bad = 0
        for bin_index in cum_order:
            mask = assigned == bin_index
            count = int(np.sum(mask))
            label_mask = mask & labeled_mask
            labeled_count = int(np.sum(label_mask))
            bad_count = int(np.sum(labels[label_mask] == 1)) if labeled_count else 0
            bad_rate = (bad_count / labeled_count) if labeled_count else None
            cum_count += count
            cum_labeled += labeled_count
            cum_bad += bad_count
            cum_count_pct = (cum_count / total_count) if total_count else None
            cum_bad_rate = (cum_bad / cum_labeled) if cum_labeled else None
            cum_bad_capture = (cum_bad / total_bad) if total_bad else None
            lift = (bad_rate / overall_bad_rate) if bad_rate is not None and overall_bad_rate else None
            band_rows[bin_index] = {
                "split": split_name,
                "bin": int(bin_index),
                "score_lower": float(edges[bin_index - 1]) if np.isfinite(edges[bin_index - 1]) else None,
                "score_upper": float(edges[bin_index]) if np.isfinite(edges[bin_index]) else None,
                "sample_count": count,
                "labeled_count": labeled_count if count else None,
                "bad_count": bad_count if count and labeled_count else None,
                "bad_rate": bad_rate if count else None,
                "avg_score": float(np.mean(scores[mask])) if count else None,
                "cum_count_pct": cum_count_pct if count else None,
                "cum_bad_rate": cum_bad_rate if count else None,
                "cum_bad_capture": cum_bad_capture if count else None,
                "cum_pass_rate": cum_count_pct if count else None,
                "lift": lift if count else None,
                "bin_edges_source": edges_source,
                "cum_direction": direction,
            }
        for bin_index in bin_indices:
            row = band_rows.get(bin_index)
            if row is None or row["sample_count"] == 0:
                continue
            rows.append(row)

    if rows:
        # FS-KS-band-contribution: per-band KS contribution = |cum_bad_pct - cum_good_pct|
        # at that band, computed within each split from the already-cumulated counts above.
        _annotate_score_band_ks(rows, overall_bad_rate_by_split)
    return rows


def _annotate_score_band_ks(rows: list[dict], overall_bad_rate_by_split: dict[str, float | None]) -> None:
    """Adds a per-row ``ks_contribution`` = |cum_bad_pct - cum_good_pct| within each
    split, walking rows in the same cumulation order they were produced in (already
    grouped by split, in cum_order). Purely derived from fields already on each row --
    no extra data pass, deterministic (INV-1)."""
    by_split: dict[str, list[dict]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)
    for split_name, split_rows in by_split.items():
        overall_bad_rate = overall_bad_rate_by_split.get(split_name)
        total_labeled = sum(row["labeled_count"] or 0 for row in split_rows)
        total_bad = sum(row["bad_count"] or 0 for row in split_rows)
        total_good = total_labeled - total_bad
        if total_bad <= 0 or total_good <= 0 or overall_bad_rate is None:
            for row in split_rows:
                row["ks_contribution"] = None
            continue
        cum_bad = 0
        cum_good = 0
        for row in split_rows:
            bad_count = row["bad_count"] or 0
            labeled_count = row["labeled_count"] or 0
            good_count = labeled_count - bad_count
            cum_bad += bad_count
            cum_good += good_count
            cum_bad_pct = cum_bad / total_bad
            cum_good_pct = cum_good / total_good
            row["ks_contribution"] = float(abs(cum_bad_pct - cum_good_pct))


def _univariate_rows(runtime: _Runtime, dataset_path: Path, artifact, config: TrainConfig) -> list[dict]:
    """DOM-7a: per-feature/per-split univariate metrics, plus a train-vs-split PSI
    column. PSI needs no labels (feature_psi compares two value distributions), so it
    is computed for every non-train split against train regardless of whether that
    split carries labels — train itself reports psi=None (nothing to compare against).
    """
    if artifact is None:
        return []
    frame = runtime.backend.read_frame(dataset_path, columns=[*artifact.feature_list, config.target_col, config.split_col])
    train_value = config.split_values.get("train")
    train_frame = frame[frame[config.split_col] == train_value] if train_value is not None else frame.iloc[0:0]
    rows = []
    for feature in artifact.feature_list:
        train_values = pd.to_numeric(train_frame[feature], errors="coerce").to_numpy(dtype=float) if not train_frame.empty else np.array([])
        psi_edges = equal_frequency_edges(train_values, DEFAULT_IV_BINS) if train_values.size else None
        for split_name, split_value in config.split_values.items():
            split_frame = frame[frame[config.split_col] == split_value]
            if split_frame.empty:
                continue
            split_values_arr = pd.to_numeric(split_frame[feature], errors="coerce").to_numpy(dtype=float)
            psi = None
            if psi_edges is not None and split_name != "train":
                try:
                    psi = feature_psi(train_values, split_values_arr, psi_edges)
                except Exception:
                    psi = None
            target_series = pd.to_numeric(split_frame[config.target_col], errors="coerce")
            if target_series.notna().sum() == 0:
                # Scoring-only split (no labels): skip label-dependent metrics, but PSI
                # (no labels required) still gets reported for this split.
                rows.append({
                    "feature": feature,
                    "split": split_name,
                    "iv": None,
                    "ks": None,
                    "auc": None,
                    "sample_count": int(len(split_frame)),
                    "coverage": None,
                    "missing_rate": None,
                    "unique_count": None,
                    "psi_vs_train": psi,
                })
                continue
            metrics = feature_metrics(
                split_values_arr,
                target_series.to_numpy(dtype=float),
                feature=feature,
            )
            rows.append({
                "feature": feature,
                "split": split_name,
                "iv": metrics.iv,
                "ks": metrics.ks,
                "auc": metrics.auc,
                "sample_count": int(len(split_frame)),
                "coverage": 1.0 - metrics.missing_rate,
                "missing_rate": metrics.missing_rate,
                "unique_count": metrics.unique_count,
                "psi_vs_train": psi,
            })
    return rows


def _stress_product_removal(
    runtime: _Runtime,
    dataset_path: Path,
    artifact: ModelArtifact | None,
    config: TrainConfig,
    feature_dictionary: dict,
) -> dict:
    if artifact is None or not feature_dictionary:
        return {}
    categories = _stress_feature_categories(feature_dictionary, artifact.feature_list)
    if not categories:
        return {}
    columns = _unique_columns([*artifact.feature_list, config.target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    oot_value = config.split_values.get("oot", "oot")
    oot_sample = frame[frame[config.split_col] == oot_value]
    if oot_sample.empty:
        return {"baseline": {"status": "skipped", "reason": "OOT sample is required for stress test"}}
    result = run_stress_test(
        oot_sample=oot_sample,
        config=ValidationConfig(
            target_col=config.target_col,
            score_col=MODEL_REPORT_SCORE_COL,
            split_col=config.split_col,
            time_col=str(config.params.get("time_col") or "apply_month"),
            feature_columns=list(artifact.feature_list),
            bin_count=10,
            random_seed=config.seed,
            split_values={key: str(value) for key, value in config.split_values.items()},
        ),
        feature_categories=categories,
        input_scorer=_ModelArtifactScorer(artifact, base_dir=_artifact_model_base_dir(runtime, artifact)),
    )
    return _stress_product_rows(result)


def _report_scored_dataset(
    runtime: _Runtime,
    dataset_path: Path,
    artifact: ModelArtifact | None,
    config: TrainConfig,
    *,
    task_id: str,
    experiment_id: str,
    dataset_id: str,
) -> tuple[Path, str, pd.DataFrame | None]:
    columns = runtime.backend.column_names(dataset_path)
    if "score" in columns:
        return dataset_path, "score", None
    if artifact is None:
        # No trained artifact and no explicit `score` column: there is no real model
        # score to report on. Previously this silently substituted the first feature
        # column as a fake "score", producing a plausible-looking but semantically wrong
        # formal report (DOM-10) — fail loudly instead.
        raise ReportScoreMissingError(experiment_id=experiment_id, dataset_id=dataset_id)

    frame = runtime.backend.read_frame(dataset_path)
    scorer = _ModelArtifactScorer(artifact, base_dir=_artifact_model_base_dir(runtime, artifact))
    frame[MODEL_REPORT_SCORE_COL] = scorer.score(frame)
    scorecard_points = scorer.scorecard_points(frame)
    if scorecard_points is not None:
        frame[SCORECARD_POINTS_COL] = scorecard_points
    out_path = Path(runtime.settings.tasks_dir) / task_id / "outputs" / "model_report_scored.parquet"
    artifact = TransactionalArtifactStore(out_path.parent).stage(out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
        final_path = artifact.promote()
        artifact.commit()
        return final_path, MODEL_REPORT_SCORE_COL, frame
    except Exception:
        artifact.rollback()
        raise


def _stress_feature_categories(feature_dictionary: dict, feature_list: tuple[str, ...]) -> dict[str, list[str]]:
    allowed = set(feature_list)
    categories: dict[str, list[str]] = {}
    for feature, metadata in feature_dictionary.items():
        if feature not in allowed or not isinstance(metadata, dict):
            continue
        product = _optional_str(metadata.get("产品名称"))
        if not product:
            continue
        categories.setdefault(product, []).append(str(feature))
    return categories


def _stress_product_rows(result) -> dict:
    rows = {
        "baseline": {
            "status": result.status,
            "sample_count": result.baseline.sample_count,
            "ks": result.baseline.ks,
            "dropped_features": "",
            "dropped_feature_count": "",
            "ks_after": "",
            "ks_delta": "",
            "psi_vs_baseline": "",
            "error": "",
        }
    }
    for row in result.per_category:
        rows[row.category] = {
            "status": row.status,
            "sample_count": result.baseline.sample_count,
            "ks": result.baseline.ks,
            "dropped_features": ", ".join(row.dropped_features),
            "dropped_feature_count": len(row.dropped_features),
            "ks_after": row.ks_after,
            "ks_delta": row.ks_delta,
            "psi_vs_baseline": row.psi_vs_baseline,
            "error": row.error or "",
        }
    return rows


def _report_bin_table(
    runtime: _Runtime,
    dataset_path: Path,
    *,
    score_col: str,
    target_col: str,
    config: TrainConfig,
    business: BusinessColumns,
) -> list[dict]:
    columns = _unique_columns([score_col, target_col, config.split_col])
    frame = runtime.backend.read_frame(dataset_path, columns=columns)
    oot_value = config.split_values.get("oot", "oot")
    oot_frame = frame[frame[config.split_col] == oot_value]
    if oot_frame.empty:
        return []
    from marvis.validation.binning import equal_frequency_bin_edges

    edges = equal_frequency_bin_edges(oot_frame[score_col].to_numpy(dtype=float), 10)
    return compute_amount_bin_table(
        runtime.backend,
        dataset_path,
        score_col=score_col,
        target_col=target_col,
        edges=edges,
        business=business,
        filters={config.split_col: oot_value},
    )


def _report_structured_summary(**payload) -> dict:
    return _jsonable(payload)


# LLM-10: text/version now live in marvis.llm_prompts; kept as a module-level
# constant so existing imports of REPORT_NARRATIVE_SYS from here keep working
# unchanged.
REPORT_NARRATIVE_SYS = _REPORT_NARRATIVE_SYS_SPEC.text


REPORT_NARRATIVE_KEYS = ("sample", "vintage", "model", "stress")


REPORT_NUMERIC_EVIDENCE_KEYS = (
    "dataset_split",
    "stability",
    "sample_analysis",
    "vintage",
    "feature_importance",
    "scorecard_table",
    "score_bands",
    "calibration",
    "univariate",
    "oot_bin_table",
    "stress_product_removal",
    "stress_low_pricing",
)


def _draft_report_narratives(structured_summary: dict, *, llm_factory=None) -> dict:
    fallback = _fallback_report_narratives()
    if llm_factory is None:
        return fallback
    try:
        raw = llm_factory().complete(
            system_prompt=REPORT_NARRATIVE_SYS,
            user_prompt=_report_narrative_prompt(structured_summary),
            response_format={"type": "json_object"},
            stream=False,
        )
        payload = json.loads(str(raw))
    except (LLMClientError, LLMSettingsError, json.JSONDecodeError, TypeError, ValueError):
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return {
        key: str(payload.get(key) or fallback[key])
        for key in REPORT_NARRATIVE_KEYS
    }


def _fallback_report_narratives() -> dict:
    return {
        "sample": "样本分析基于平台聚合结果生成。",
        "vintage": "Vintage 结论基于平台计算曲线生成。",
        "model": "模型结论基于平台指标与特征重要性生成。",
        "stress": "压力测试结论基于平台压测结果生成。",
    }


def _report_narrative_prompt(structured_summary: dict) -> str:
    return (
        "请基于以下结构化摘要，输出 JSON："
        "{sample, vintage, model, stress}。\n"
        "要求：只写文字解释；所有数字必须来自摘要原文；缺少数据时说明缺业务数据。\n\n"
        f"结构化摘要：\n{json.dumps(structured_summary, ensure_ascii=False, sort_keys=True)}"
    )


def _report_llm_factory(workspace: Path, model_id: str | None):
    def factory():
        return OpenAICompatibleLLMClient(resolve_llm_model(workspace, model_id))

    return factory


def _guard_no_invented_numbers(narratives: dict, structured_summary: dict) -> dict:
    allowed = _allowed_number_tokens(_report_numeric_evidence(structured_summary))
    guarded: dict[str, str] = {}
    for key, value in narratives.items():
        text = str(value)
        guarded[str(key)] = _NUMBER_TOKEN_RE.sub(
            lambda match: match.group(0) if _number_token_allowed(match.group(0), allowed) else "[平台未提供该数字]",
            text,
        )
    return guarded


def _report_numeric_evidence(structured_summary: dict) -> dict:
    return {
        key: structured_summary.get(key)
        for key in REPORT_NUMERIC_EVIDENCE_KEYS
        if key in structured_summary
    }
