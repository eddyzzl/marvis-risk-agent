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
from marvis.feature.candidates import candidate_numeric_features
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
from marvis.feature.derive import derive_batch
from marvis.feature.encode import onehot_encode, woe_encode
from marvis.feature.errors import FeatureError, FitRequiresSplitError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.feature.metrics import feature_metrics, feature_psi, head_tail_lift
from marvis.feature.transform import (
    apply_scaler,
    cap_outliers,
    impute_missing,
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

    from marvis.feature.screen import screen_features

    runtime = _runtime(ctx)
    dataset = runtime.registry.get(str(inputs["dataset_id"]))
    split_col = inputs.get("split_col")
    features = _resolve_feature_cols(
        runtime,
        dataset.id,
        inputs.get("features") or [],
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
    return {
        "selected": list(result.selected),
        "ranked": [[feature, ks] for feature, ks in result.ranked],
        "leakage": [[feature, ks, reason] for feature, ks, reason in result.leakage],
        "suspected": [[feature, ks, reason] for feature, ks, reason in result.suspected],
        "unusable": [[feature, reason] for feature, reason in result.unusable],
        "scores": _jsonable(result.scores),
        "n_screened": result.n_screened,
    }


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
    result = _register_frame(runtime, out, dataset, ctx, "woe")
    return {
        "result_dataset_id": result.id,
        "new_columns": new_columns,
        "woe_maps": woe_maps,
        "nan_labels_dropped": nan_labels_dropped,
        "fit_rows": int(len(fit_frame)),
        "fit_split": fit_split,
    }


def _woe_fit_frame(frame: pd.DataFrame, inputs: dict, dataset_id: str) -> tuple[pd.DataFrame, str]:
    """Rows used to fit the WOE mapping — excludes holdout (default test+OOT) so the
    mapping never peeks at evaluation labels (PREP-1). No ``split_col`` means the caller
    cannot express train-only fitting; that's a typed-error stop unless the caller
    explicitly confirms a full-pool fit via ``allow_full_fit``."""
    split_col = inputs.get("split_col")
    if not split_col:
        if bool(inputs.get("allow_full_fit")):
            return frame, "full"
        raise FitRequiresSplitError(tool="woe_encode", dataset_id=dataset_id)
    holdout_values = tuple(str(value) for value in (inputs.get("holdout_values") or ("test", "oot")))
    mask = ~frame[str(split_col)].astype(str).isin(holdout_values)
    fit_frame = frame.loc[mask]
    if fit_frame.empty:
        raise FeatureError("WOE fit frame is empty after excluding holdout rows")
    return fit_frame, "train"


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
    result = _register_frame(runtime, encoded, dataset, ctx, "onehot")
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
    for col in columns:
        fit_values = out.loc[fit_mask, col].to_numpy(dtype=float)
        if method == "minmax":
            _fit_values, column_params = minmax_normalize(
                fit_values,
                feature_range=tuple(inputs.get("feature_range") or (0, 1)),
            )
            values = apply_scaler(out[col].to_numpy(dtype=float), column_params, kind="minmax")
        elif method == "zscore":
            _fit_values, column_params = zscore_standardize(fit_values)
            values = apply_scaler(out[col].to_numpy(dtype=float), column_params, kind="zscore")
        else:
            raise FeatureError("method must be minmax or zscore")
        out[col] = values
        params[col] = column_params
    result = _register_frame(runtime, out, dataset, ctx, "normalize")
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
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "impute_missing", dataset.id)
    for column in columns:
        _filled_fit, value = impute_missing(
            out.loc[fit_mask, column],
            strategy=str(inputs["strategy"]),
            fill_value=inputs.get("fill_value"),
        )
        out[column] = out[column].fillna(value)
        fill_values[column] = value
    result = _register_frame(runtime, out, dataset, ctx, "impute")
    return {
        "result_dataset_id": result.id,
        "fill_values": _jsonable(fill_values),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


def tool_cap_outliers(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    out = frame.copy()
    bounds = {}
    columns = [str(column) for column in inputs["columns"]]
    _assert_columns(out, columns)
    fit_mask, fit_split = _stat_fit_mask(out, inputs, "cap_outliers", dataset.id)
    for column in columns:
        fit_values = out.loc[fit_mask, column].to_numpy(dtype=float)
        _capped_fit, params = cap_outliers(
            fit_values,
            method=str(inputs.get("method") or "iqr"),
            lower_q=float(inputs.get("lower_q", 0.01)),
            upper_q=float(inputs.get("upper_q", 0.99)),
        )
        all_values = out[column].to_numpy(dtype=float)
        mask = np.isfinite(all_values)
        clipped = all_values.copy()
        lower = params["lower"]
        upper = params["upper"]
        if np.isfinite(lower) and np.isfinite(upper):
            clipped[mask] = np.clip(clipped[mask], lower, upper)
        out[column] = clipped
        bounds[column] = params
    result = _register_frame(runtime, out, dataset, ctx, "cap")
    return {
        "result_dataset_id": result.id,
        "bounds": _jsonable(bounds),
        "fit_rows": int(fit_mask.sum()),
        "fit_split": fit_split,
    }


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
    derived, new_columns = derive_batch(frame, list(inputs["recipe"]))
    result = _register_frame(runtime, derived, dataset, ctx, "cross")
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


def _register_frame(runtime: _Runtime, frame: pd.DataFrame, source_dataset, ctx, suffix: str):
    out_path = runtime.datasets_root / ctx.task_id / "feature" / f"{source_dataset.id}_{suffix}.parquet"
    uow = ArtifactUnitOfWork()
    artifact = uow.stage_file(out_path.parent, out_path.name)
    try:
        frame.to_parquet(artifact.path, index=False)
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
    if method in {"equal_frequency", "quantile"}:
        return equal_frequency_edges(values, max_bins)
    if method == "equal_width":
        return equal_width_edges(values, max_bins)
    if method == "manual":
        return manual_edges([float(item) for item in inputs.get("breakpoints") or []])
    if method == "chimerge":
        return chimerge_edges(values, _target_values(frame, target_col), max_bins=max_bins)
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
