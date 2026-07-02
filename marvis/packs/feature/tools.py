from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.labels import require_labels_confirmed
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.feature.candidates import (
    candidate_numeric_features,
    excluded_categorical_columns,
    suspected_categorical_columns,
)
from marvis.feature.binning import (
    chimerge_edges,
    equal_frequency_edges,
    equal_width_edges,
    manual_edges,
    monotonic_direction,
    monotonic_edges,
    tree_edges,
)
from marvis.feature.correlation import correlation_report
from marvis.feature.derive import derive_batch, derive_date_features
from marvis.feature.encode import apply_categorical_woe, categorical_woe_encode, onehot_encode, woe_encode
from marvis.feature.errors import FeatureError, FitRequiresSplitError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.feature.metrics import feature_metrics, feature_psi, head_tail_lift
from marvis.feature.preprocessing import (
    read_preprocessing_chain,
    sidecar_path,
    write_preprocessing_chain,
)
from marvis.feature.transform import (
    apply_scaler,
    cap_outliers,
    impute_missing,
    mask_sentinel_values,
    minmax_normalize,
    zscore_standardize,
)
from marvis.settings import build_settings


def tool_compute_feature_metrics(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    features = _resolve_feature_cols(
        runtime,
        str(inputs["dataset_id"]),
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
    )
    dataset, frame = _read_frame(
        runtime,
        str(inputs["dataset_id"]),
        _unique([*features, inputs["target_col"]]),
    )
    nan_labels_dropped = require_labels_confirmed(
        frame,
        str(inputs["target_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    compare_frame = None
    if inputs.get("compare_dataset_id"):
        _compare_dataset, compare_frame = _read_frame(
            runtime,
            str(inputs["compare_dataset_id"]),
            features,
        )
    metrics = []
    for feature in features:
        compare_values = None if compare_frame is None else compare_frame[str(feature)].to_numpy(dtype=float)
        item = feature_metrics(
            frame[str(feature)].to_numpy(dtype=float),
            _target_values(frame, str(inputs["target_col"])),
            feature=str(feature),
            bins=int(inputs.get("bins") or 10),
            compare_values=compare_values,
        )
        metrics.append(_jsonable(item))
    result = {"dataset_id": dataset.id, "metrics": metrics, "nan_labels_dropped": nan_labels_dropped}
    # Optional metrics are computed only when selected (spec §2: 选了才算). VIF /
    # collinear is the first wired one; missing selection → not computed.
    selected = {str(metric).strip().lower() for metric in (inputs.get("metrics") or [])}
    if selected & {"vif", "collinear", "共线"}:
        report = correlation_report(
            frame,
            features,
            method="pearson",
            threshold=float(inputs.get("corr_threshold", 0.8)),
        )
        result["collinear"] = _jsonable(report)
    if selected & {"head_tail_lift", "headtail_lift", "头尾lift"}:
        # Merge the risk-direction-aware head/tail lift into each per-feature row so it
        # rides the existing metrics echo (no new output key / $ref needed).
        target_values = _target_values(frame, str(inputs["target_col"]))
        for index, feature in enumerate(features):
            metrics[index].update(
                head_tail_lift(frame[str(feature)].to_numpy(dtype=float), target_values)
            )
    if selected & {"importance", "feature_importance", "重要性"}:
        # Multivariate gain importance: train ONE capped, seed-pinned model over all
        # features and merge each feature's share into its row (lazy lightgbm import).
        from marvis.feature.importance import feature_importance

        feature_names = list(features)
        importance = feature_importance(frame, feature_names, str(inputs["target_col"]))
        for index, feature in enumerate(feature_names):
            metrics[index]["importance"] = importance.get(feature)
    return result


def tool_screen_features(inputs: dict, ctx) -> dict:
    """Leakage-aware feature screening (spec form B §4 backend; shared screen with
    MODELING via marvis.feature.screen). Flags hard leakage (KS>=leakage_ks), model-output
    names, and unusable (constant/sparse) columns, and ranks the rest — yielding a selected
    feature set for the downstream model.

    For a non-binary target (``target_type != "binary"``, e.g. a regression task) the
    leakage KS screen is skipped: ``feature_ks`` is a binary-only statistic and would
    miscompute or crash on a continuous target. In that case every candidate is kept as
    ``selected`` (ks=None) and only missing_rate / unique_count are reported."""
    target_type = str(inputs.get("target_type", "binary"))
    if target_type != "binary":
        return _screen_features_non_binary(inputs, ctx)

    from marvis.feature.screen import screen_features, sentinel_screen_notice

    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    split_col = inputs.get("split_col")
    requested_features = inputs.get("features") or []
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    excluded_categorical = _excluded_categorical_for_screen(
        runtime,
        dataset.id,
        requested_features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    suspected_categorical = _suspected_categorical_for_screen(
        runtime,
        dataset.id,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
    )
    holdout = inputs.get("holdout_values")
    top_k = inputs.get("top_k")
    result = screen_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        split_col=str(split_col) if split_col else None,
        holdout_values=tuple(str(value) for value in holdout) if holdout else ("oot",),
        leakage_ks=float(inputs.get("leakage_ks", 0.40)),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=int(top_k) if top_k is not None else None,
        batch_size=int(inputs.get("batch_size", 500)),
    )
    payload = {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "excluded_categorical": excluded_categorical,
    }
    if suspected_categorical:
        payload["suspected_categorical"] = suspected_categorical
    if result.split_shift:
        payload["split_shift"] = [[feature, delta, reason] for feature, delta, reason in result.split_shift]
    if result.leakage_watch:
        payload["leakage_watch"] = [[feature, ks, reason] for feature, ks, reason in result.leakage_watch]
    if result.sentinel_columns:
        payload["sentinel_columns"] = _jsonable(result.sentinel_columns)
        payload["sentinel_notice"] = sentinel_screen_notice(result.sentinel_columns)
    return payload


def _excluded_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    requested_features: list,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """String/object columns silently dropped by candidate inference (PREP-3/FS-3).

    Only meaningful when ``features`` was NOT explicitly provided — an explicit
    feature list is the caller's own choice, not an inference the platform made
    on their behalf, so there is nothing to surface."""
    if [str(item) for item in requested_features if str(item).strip()]:
        return []
    dataset = runtime.registry.get(str(dataset_id))
    excluded = excluded_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in excluded]


def _suspected_categorical_for_screen(
    runtime: "_Runtime",
    dataset_id: str,
    *,
    target_col: str,
    split_col: str | None,
) -> list[dict]:
    """Numeric columns that look like nominal codes rather than continuous measures
    (PREP-5), e.g. a zip/industry code — surfaced as a screen-gate hint, always (even
    with an explicit feature list) since these columns keep being modeled as continuous
    numeric today; nothing about candidate inference or the selected set changes."""
    dataset = runtime.registry.get(str(dataset_id))
    suspected = suspected_categorical_columns(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=target_col,
        split_col=split_col,
    )
    return [{"column": item.column, "cardinality": item.cardinality} for item in suspected]


def _screen_features_non_binary(inputs: dict, ctx) -> dict:
    """Screen path for a non-binary (continuous/multiclass) target: the binary-only leakage
    KS screen is skipped, but unusable columns are still dropped into ``unusable`` — mirroring
    the binary screen — namely constant (unique_count<=1) or mostly-missing
    (missing_rate>=max_missing_rate) columns; the rest are kept as selected (ks=None)."""
    from marvis.feature.screen import screen_features_non_binary

    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    holdout = inputs.get("holdout_values")
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]) if inputs.get("split_col") else None,
    )
    result = screen_features_non_binary(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        features=features,
        target_col=str(inputs["target_col"]),
        split_col=str(inputs["split_col"]) if inputs.get("split_col") else None,
        holdout_values=tuple(str(value) for value in holdout) if holdout else ("oot",),
        max_missing_rate=float(inputs.get("max_missing_rate", 0.95)),
        top_k=int(inputs["top_k"]) if inputs.get("top_k") is not None else None,
    )
    return {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [],
        "suspected": [],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
        "note": "非二分类目标：跳过泄漏KS筛选，已剔除常量/高缺失列",
    }


def tool_generate_feature_report(inputs: dict, ctx) -> dict:
    """Write the per-feature metrics into a downloadable Excel report (FEATURE form A)."""
    from marvis.output.feature_report import render_feature_report

    metrics = [item for item in (inputs.get("metrics") or []) if isinstance(item, dict)]
    collinear = inputs.get("collinear") if isinstance(inputs.get("collinear"), dict) else None
    settings = build_settings(ctx.workspace)
    out_path = Path(settings.tasks_dir) / ctx.task_id / "outputs" / "feature_report.xlsx"
    render_feature_report(metrics, out_path, collinear=collinear)
    # Echo metrics (+ optional collinear) so the driver renders the wide table, the VIF
    # section, and the report link together.
    out = {"report_path": str(out_path), "feature_count": len(metrics), "metrics": metrics}
    if collinear is not None:
        out["collinear"] = collinear
    return out


def tool_bin_feature(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    _dataset, frame = _read_frame(
        runtime,
        str(inputs["dataset_id"]),
        [str(inputs["feature"]), str(inputs["target_col"])],
    )
    nan_labels_dropped = require_labels_confirmed(
        frame,
        str(inputs["target_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    feature = str(inputs["feature"])
    target_col = str(inputs["target_col"])
    target = _target_values(frame, target_col)
    sentinel_values = inputs.get("sentinel_values")
    if sentinel_values:
        frame = frame.copy()
        frame[feature] = mask_sentinel_values(frame[feature], [float(v) for v in sentinel_values])
    values = frame[feature].to_numpy(dtype=float)
    edges = _edges_for(frame, inputs, ctx)
    before = None
    resolved_direction = None
    if bool(inputs.get("enforce_monotonic")):
        before = compute_woe_iv(values, target, edges, feature=feature)
        resolved_direction = monotonic_direction(
            values,
            target,
            edges,
            direction=str(inputs.get("monotonic_direction") or "auto"),
        )
        edges = monotonic_edges(values, target, edges, direction=resolved_direction)
    result = compute_woe_iv(
        values,
        target,
        edges,
        feature=feature,
    )
    payload = _jsonable(result)
    payload["bins"] = [_jsonable(bin_row) for bin_row in result.bins]
    payload["na_bin"] = _jsonable(result.na_bin) if result.na_bin else None
    payload["nan_labels_dropped"] = nan_labels_dropped
    if before is not None:
        payload["monotonic_enforced"] = True
        payload["monotonic_direction"] = resolved_direction
        payload["monotonic_before"] = bool(before.monotonic)
        payload["total_iv_before_monotonic"] = before.total_iv
        payload["edges_before_monotonic"] = [float(value) for value in before.edges]
    return payload


def tool_compute_psi(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    feature = str(inputs["feature"])
    columns = _unique([feature, *_filter_columns(inputs.get("base_filter")), *_filter_columns(inputs.get("compare_filter"))])
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]), columns)
    base = _apply_filter(frame, inputs.get("base_filter"))[feature].to_numpy(dtype=float)
    compare = _apply_filter(frame, inputs.get("compare_filter"))[feature].to_numpy(dtype=float)
    edges = equal_frequency_edges(base, int(inputs.get("bins") or 10))
    psi = feature_psi(base, compare, edges)
    return {
        "dataset_id": dataset.id,
        "feature": feature,
        "psi": float(psi),
        "edges": _jsonable(edges),
        "bin_distributions": {
            "base": _jsonable(_bin_distribution(base, edges)),
            "compare": _jsonable(_bin_distribution(compare, edges)),
        },
    }


def tool_correlation_analysis(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]), [str(item) for item in inputs["features"]])
    report = correlation_report(
        frame,
        [str(item) for item in inputs["features"]],
        method=str(inputs.get("method") or "pearson"),
        threshold=float(inputs.get("threshold", 0.8)),
    )
    payload = _jsonable(report)
    payload["dataset_id"] = dataset.id
    return payload


def tool_woe_encode(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    features = [str(item) for item in inputs["features"]]
    target_col = str(inputs["target_col"])
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    required_columns = [*features, target_col]
    if inputs.get("split_col"):
        required_columns.append(str(inputs["split_col"]))
    _assert_columns(frame, required_columns)
    out = frame.copy()
    sentinel_values = _sentinel_values_for(inputs, features)
    for feature, column_sentinels in sentinel_values.items():
        out[feature] = mask_sentinel_values(out[feature], column_sentinels)
    fit_frame, fit_split = _woe_fit_frame(out, inputs, dataset.id)
    nan_labels_dropped = require_labels_confirmed(
        fit_frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        scope="woe fit",
    )
    woe_maps = {}
    new_columns = []
    for feature in features:
        edges = _edges_for(fit_frame, {**inputs, "feature": feature}, ctx)
        binning = compute_woe_iv(
            fit_frame[feature].to_numpy(dtype=float),
            _target_values(fit_frame, target_col),
            edges,
            feature=feature,
        )
        woe = woe_result_from_binning(binning)
        encoded = woe_encode(out, feature, woe)
        out[encoded.name] = encoded
        new_columns.append(encoded.name)
        woe_maps[feature] = _jsonable(woe)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "woe",
        preprocessing_step={"kind": "woe", "columns": features, "params": _jsonable(woe_maps)},
    )
    return {
        "result_dataset_id": result.id,
        "new_columns": new_columns,
        "woe_maps": woe_maps,
        "nan_labels_dropped": nan_labels_dropped,
        "fit_rows": int(len(fit_frame)),
        "fit_split": fit_split,
    }


def _woe_fit_frame(
    frame: pd.DataFrame, inputs: dict, dataset_id: str, *, tool: str = "woe_encode"
) -> tuple[pd.DataFrame, str]:
    """Rows used to fit the WOE mapping — excludes holdout (default test+OOT) so the
    mapping never peeks at evaluation labels (PREP-1). No ``split_col`` means the caller
    cannot express train-only fitting; that's a typed-error stop unless the caller
    explicitly confirms a full-pool fit via ``allow_full_fit``."""
    split_col = inputs.get("split_col")
    if not split_col:
        if bool(inputs.get("allow_full_fit")):
            return frame, "full"
        raise FitRequiresSplitError(tool=tool, dataset_id=dataset_id)
    holdout_values = tuple(str(value) for value in (inputs.get("holdout_values") or ("test", "oot")))
    mask = ~frame[str(split_col)].astype(str).isin(holdout_values)
    fit_frame = frame.loc[mask]
    if fit_frame.empty:
        raise FeatureError("WOE fit frame is empty after excluding holdout rows")
    return fit_frame, "train"


def tool_woe_encode_categorical(inputs: dict, ctx) -> dict:
    """Category -> WOE encode string/object columns (PREP-3/FS-3) — the categorical
    analogue of ``woe_encode``. Same train-only fitting contract: fits on the non-
    holdout rows (default excludes test+OOT) and raises ``FitRequiresSplitError``
    unless ``split_col`` is given or the caller passes ``allow_full_fit=true``."""
    runtime = _runtime(ctx)
    features = [str(item) for item in inputs["features"]]
    target_col = str(inputs["target_col"])
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    required_columns = [*features, target_col]
    if inputs.get("split_col"):
        required_columns.append(str(inputs["split_col"]))
    _assert_columns(frame, required_columns)
    out = frame.copy()
    fit_frame, fit_split = _woe_fit_frame(out, inputs, dataset.id, tool="woe_encode_categorical")
    nan_labels_dropped = require_labels_confirmed(
        fit_frame,
        target_col,
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        scope="categorical woe fit",
    )
    min_count = inputs.get("min_count")
    smoothing = float(inputs.get("smoothing", 0.5))
    woe_maps = {}
    new_columns = []
    for feature in features:
        woe = categorical_woe_encode(
            fit_frame[feature],
            _target_values(fit_frame, target_col),
            feature=feature,
            min_count=int(min_count) if min_count is not None else None,
            smoothing=smoothing,
        )
        encoded = apply_categorical_woe(out, feature, woe)
        out[encoded.name] = encoded
        new_columns.append(encoded.name)
        woe_maps[feature] = _jsonable(woe)
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "catwoe",
        preprocessing_step={"kind": "categorical_woe", "columns": features, "params": _jsonable(woe_maps)},
    )
    return {
        "result_dataset_id": result.id,
        "new_columns": new_columns,
        "woe_maps": woe_maps,
        "nan_labels_dropped": nan_labels_dropped,
        "fit_rows": int(len(fit_frame)),
        "fit_split": fit_split,
    }


def tool_onehot_encode(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    columns = [str(item) for item in inputs["columns"]]
    _assert_columns(frame, columns)
    encoded, mapping = onehot_encode(
        frame,
        columns,
        max_categories=int(inputs.get("max_categories") or 50),
    )
    result = _register_frame(
        runtime,
        encoded,
        dataset,
        ctx,
        "onehot",
        preprocessing_step={"kind": "onehot", "columns": columns, "params": _jsonable(mapping)},
    )
    return {"result_dataset_id": result.id, "mapping": _jsonable(mapping)}


def tool_normalize(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    method = str(inputs["method"])
    out = frame.copy()
    params = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "normalize", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    for col in columns:
        column_sentinels = sentinel_values.get(col)
        fit_series = mask_sentinel_values(out.loc[fit_mask, col], column_sentinels)
        fit_values = fit_series.to_numpy(dtype=float)
        full_series = mask_sentinel_values(out[col], column_sentinels)
        full_values = full_series.to_numpy(dtype=float)
        if method == "minmax":
            _fit_values, column_params = minmax_normalize(
                fit_values,
                feature_range=tuple(inputs.get("feature_range") or (0, 1)),
            )
            values = apply_scaler(full_values, column_params, kind="minmax")
        elif method == "zscore":
            _fit_values, column_params = zscore_standardize(fit_values)
            values = apply_scaler(full_values, column_params, kind="zscore")
        else:
            raise FeatureError("method must be minmax or zscore")
        out[col] = values
        params[col] = column_params
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "normalize",
        preprocessing_step={"kind": "normalize", "columns": columns, "params": _jsonable(params)},
    )
    return {
        "result_dataset_id": result.id,
        "scaler_params": _jsonable(params),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


def tool_impute_missing(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    out = frame.copy()
    fill_values = {}
    indicators = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "impute_missing", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    add_indicators = bool(inputs.get("add_indicators"))
    indicator_columns: list[str] = []
    for column in columns:
        column_sentinels = sentinel_values.get(column)
        _filled_fit, value = impute_missing(
            out.loc[fit_mask, column],
            strategy=str(inputs["strategy"]),
            fill_value=inputs.get("fill_value"),
            sentinel_values=column_sentinels,
        )
        masked = mask_sentinel_values(out[column], column_sentinels)
        if add_indicators and masked.isna().any():
            # PREP-8: preserve the "missing" signal that plain imputation erases —
            # a col__was_missing 0/1 column, guarded against colliding with an
            # existing column name.
            indicator_name = _unique_column_name(f"{column}__was_missing", out.columns)
            out[indicator_name] = masked.isna().astype(int)
            indicators[column] = indicator_name
            indicator_columns.append(indicator_name)
        out[column] = masked.fillna(value)
        fill_values[column] = value
    preprocessing_steps = []
    if indicators:
        # Ordered before "impute" so replay computes the pre-fill NaN mask first.
        preprocessing_steps.append(
            {"kind": "missing_indicator", "columns": list(indicators), "params": _jsonable(indicators)}
        )
    preprocessing_steps.append({"kind": "impute", "columns": columns, "params": _jsonable(fill_values)})
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "impute",
        preprocessing_steps=preprocessing_steps,
    )
    return {
        "result_dataset_id": result.id,
        "fill_values": _jsonable(fill_values),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
        "indicator_columns": indicator_columns,
    }


def tool_cap_outliers(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    out = frame.copy()
    bounds = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "cap_outliers", dataset.id)
    sentinel_values = _sentinel_values_for(inputs, columns)
    for column in columns:
        column_sentinels = sentinel_values.get(column)
        fit_values = out.loc[fit_mask, column].to_numpy(dtype=float)
        _capped_fit, params = cap_outliers(
            fit_values,
            method=str(inputs.get("method") or "iqr"),
            lower_q=float(inputs.get("lower_q", 0.01)),
            upper_q=float(inputs.get("upper_q", 0.99)),
            sentinel_values=column_sentinels,
        )
        all_values = mask_sentinel_values(out[column], column_sentinels).to_numpy(dtype=float)
        mask = np.isfinite(all_values)
        clipped = all_values.copy()
        lower = params["lower"]
        upper = params["upper"]
        if np.isfinite(lower) and np.isfinite(upper):
            clipped[mask] = np.clip(clipped[mask], lower, upper)
        out[column] = clipped
        bounds[column] = params
    result = _register_frame(
        runtime,
        out,
        dataset,
        ctx,
        "cap",
        preprocessing_step={"kind": "cap", "columns": columns, "params": _jsonable(bounds)},
    )
    return {
        "result_dataset_id": result.id,
        "bounds": _jsonable(bounds),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


def _sentinel_values_for(inputs: dict, columns: list[str]) -> dict[str, list[float]]:
    """Resolve the ``sentinel_values`` input (PREP-4) into a per-column map.

    Accepts either a flat list (applied to every column in ``columns``) or a
    ``{column: [values, ...]}`` mapping (applied per-column only). Columns with
    no sentinel values configured are omitted from the result."""
    raw = inputs.get("sentinel_values")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {
            str(column): [float(v) for v in values]
            for column, values in raw.items()
            if str(column) in columns and values
        }
    flat = [float(v) for v in raw]
    return {column: flat for column in columns} if flat else {}


def _unique_column_name(candidate: str, existing) -> str:
    """A column name guaranteed not to collide with ``existing`` (PREP-8): appends
    an incrementing numeric suffix (``_2``, ``_3``, ...) until it is unique."""
    existing_set = set(str(column) for column in existing)
    if candidate not in existing_set:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_set:
        suffix += 1
    return f"{candidate}_{suffix}"


def _stat_fit_mask(frame: pd.DataFrame, inputs: dict, tool: str, dataset_id: str) -> tuple[np.ndarray, str]:
    """Rows used to fit statistical transforms (impute/normalize/cap) — excludes holdout
    (default test+OOT) so fill values / scaler params / capping bounds never absorb
    evaluation-set distribution (PREP-1). No ``split_col`` means the caller cannot
    express train-only fitting; that's a typed-error stop unless the caller explicitly
    confirms a full-pool fit via ``allow_full_fit``."""
    split_col = inputs.get("split_col")
    if not split_col:
        if bool(inputs.get("allow_full_fit")):
            return np.ones(len(frame), dtype=bool), "full"
        raise FitRequiresSplitError(tool=tool, dataset_id=dataset_id)
    _assert_columns(frame, [str(split_col)])
    holdout_values = tuple(str(value) for value in (inputs.get("holdout_values") or ("test", "oot")))
    mask = (~frame[str(split_col)].astype(str).isin(holdout_values)).to_numpy()
    if not mask.any():
        raise FeatureError(f"{tool} fit frame is empty after excluding holdout rows")
    return mask, "train"


def tool_cross_features(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    derived, new_columns = derive_batch(frame, list(inputs["recipe"]), dataset_id=dataset.id)
    result = _register_frame(runtime, derived, dataset, ctx, "cross")
    return {"result_dataset_id": result.id, "new_columns": new_columns}


def tool_derive_date_features(inputs: dict, ctx) -> dict:
    """Derive datediff/month/tenure-months numeric columns from date-role columns
    (PREP-7). Opt-in: never runs as part of any default template, so a caller must
    explicitly invoke it (with a date column identified e.g. via profiling/schema
    inference) to pull date information into the modeling frame."""
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    derived, new_columns = derive_date_features(frame, list(inputs["recipe"]))
    result = _register_frame(runtime, derived, dataset, ctx, "datefeat")
    return {"result_dataset_id": result.id, "new_columns": new_columns}


class _Runtime:
    def __init__(self, ctx):
        settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.backend = DataBackend(self.datasets_root)
        self.repo = DatasetRepository(settings.db_path)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _resolve_feature_cols(
    runtime: _Runtime,
    dataset_id: str,
    features,
    *,
    target_col: str,
    split_col: str | None = None,
) -> list[str]:
    provided = [str(item) for item in (features or []) if str(item).strip()]
    if provided:
        return provided
    dataset = runtime.registry.get(str(dataset_id))
    inferred = candidate_numeric_features(
        runtime.backend,
        runtime.registry.resolve_path(dataset.id),
        target_col=str(target_col),
        split_col=split_col,
    )
    if not inferred:
        raise FeatureError("未找到可用候选特征列;请检查拼接结果或指定特征列。")
    return inferred


def _read_frame(
    runtime: _Runtime,
    dataset_id: str,
    columns: list[str] | None = None,
):
    dataset = runtime.registry.get(dataset_id)
    frame = runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)
    return dataset, frame


def _register_frame(
    runtime: _Runtime,
    frame: pd.DataFrame,
    source_dataset,
    ctx,
    suffix: str,
    *,
    preprocessing_step: dict[str, Any] | None = None,
    preprocessing_steps: list[dict[str, Any]] | None = None,
):
    out_path = runtime.datasets_root / ctx.task_id / "feature" / f"{source_dataset.id}_{suffix}.parquet"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_path.parent, out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
        steps_to_append = list(preprocessing_steps or [])
        if preprocessing_step is not None:
            steps_to_append.append(preprocessing_step)
        if steps_to_append:
            # PREP-2/PREP-8: persist the accumulated preprocessing chain (source
            # dataset's chain + these new steps, in order) as a sidecar next to the
            # derived parquet, so a model trained downstream can replay every fit
            # param at scoring time instead of only seeing it in this tool's JSON
            # response. Staged via the same unit of work as the parquet so both
            # promote/commit atomically.
            source_path = None
            try:
                source_path = runtime.registry.resolve_path(source_dataset.id)
            except KeyError:
                source_path = None
            chain = read_preprocessing_chain(source_path) if source_path else []
            for step in steps_to_append:
                chain = [
                    *chain,
                    {
                        "kind": str(step["kind"]),
                        "columns": [str(c) for c in step["columns"]],
                        "params": step["params"],
                    },
                ]
            sidecar_name = sidecar_path(Path(out_path.name)).name
            sidecar_artifact = uow.stage_file(out_path.parent, sidecar_name)
            write_preprocessing_chain(sidecar_artifact.path, chain)
        register_kwargs = {
            "task_id": ctx.task_id,
            "role": "derived",
            "anchor_target": source_dataset.id,
            "seed": int(ctx.seed or 0),
        }
        register_on_connection = getattr(runtime.registry, "register_existing_on_connection", None)
        transaction = getattr(runtime.registry, "transaction", None)
        if callable(register_on_connection) and callable(transaction):
            return uow.finalize_with_connection(
                transaction,
                lambda conn: register_on_connection(conn, artifact.final_path, **register_kwargs),
            )
        return uow.finalize(
            lambda: runtime.registry.register_existing(artifact.final_path, **register_kwargs)
        )
    except Exception:
        uow.rollback()
        raise


def _edges_for(frame: pd.DataFrame, inputs: dict, ctx) -> np.ndarray:
    feature = str(inputs["feature"])
    target_col = str(inputs.get("target_col") or "")
    values = frame[feature].to_numpy(dtype=float)
    method = str(inputs.get("method") or "equal_frequency")
    max_bins = int(inputs.get("max_bins") or inputs.get("bins") or 10)
    # PREP-9: minimum bin share (default 5%) merges small bins so WOE stays stable
    # across time periods; only meaningful for the frequency-driven methods below
    # (equal_width/manual/tree already control bin size some other way).
    min_bin_pct = float(inputs.get("min_bin_pct", 0.05))
    if method in {"equal_frequency", "quantile"}:
        return equal_frequency_edges(values, max_bins, min_bin_pct=min_bin_pct)
    if method == "equal_width":
        return equal_width_edges(values, max_bins)
    if method == "manual":
        return manual_edges([float(item) for item in inputs.get("breakpoints") or []])
    if method == "chimerge":
        return chimerge_edges(
            values, _target_values(frame, target_col), max_bins=max_bins, min_bin_pct=min_bin_pct
        )
    if method == "tree":
        return tree_edges(values, _target_values(frame, target_col), max_bins=max_bins, seed=int(ctx.seed or 0))
    raise FeatureError("method must be equal_frequency, equal_width, manual, chimerge, or tree")


def _target_values(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    if not target_col or target_col not in frame.columns:
        raise FeatureError(f"missing target column: {target_col}")
    return pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)


def _apply_filter(frame: pd.DataFrame, spec: Any) -> pd.DataFrame:
    if not spec:
        return frame
    if not isinstance(spec, dict):
        raise FeatureError("filter must be an object")
    if "column" not in spec:
        if len(spec) == 1:
            column, value = next(iter(spec.items()))
            return frame[frame[str(column)] == value]
        raise FeatureError("filter requires column, op, and value")
    column = str(spec["column"])
    op = str(spec.get("op") or "eq")
    value = spec.get("value")
    _assert_columns(frame, [column])
    series = frame[column]
    if op == "eq":
        mask = series == value
    elif op == "ne":
        mask = series != value
    elif op == "lt":
        mask = series < value
    elif op == "lte":
        mask = series <= value
    elif op == "gt":
        mask = series > value
    elif op == "gte":
        mask = series >= value
    elif op == "in":
        mask = series.isin(value or [])
    else:
        raise FeatureError("filter op must be eq, ne, lt, lte, gt, gte, or in")
    return frame[mask]


def _filter_columns(spec: Any) -> list[str]:
    if not isinstance(spec, dict) or not spec:
        return []
    if "column" in spec:
        return [str(spec["column"])]
    return [str(key) for key in spec]


def _bin_distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    from marvis.feature.binning import assign_bins

    assigned = assign_bins(values, edges)
    valid = assigned >= 0
    if not np.any(valid):
        return np.zeros(len(edges) - 1, dtype=float)
    counts = np.bincount(assigned[valid], minlength=len(edges) - 1).astype(float)
    return counts / counts.sum()


def _assert_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureError(f"missing columns: {', '.join(missing)}")


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
