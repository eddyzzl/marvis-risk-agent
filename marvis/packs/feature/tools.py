from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository
from marvis.feature.binning import (
    chimerge_edges,
    equal_frequency_edges,
    equal_width_edges,
    manual_edges,
    tree_edges,
)
from marvis.feature.correlation import correlation_report
from marvis.feature.derive import derive_batch
from marvis.feature.encode import onehot_encode, woe_encode
from marvis.feature.errors import FeatureError
from marvis.feature.iv import compute_woe_iv, woe_result_from_binning
from marvis.feature.metrics import feature_metrics, feature_psi
from marvis.feature.transform import (
    cap_outliers,
    impute_missing,
    minmax_normalize,
    zscore_standardize,
)
from marvis.settings import build_settings


def tool_compute_feature_metrics(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(
        runtime,
        str(inputs["dataset_id"]),
        _unique([*inputs["features"], inputs["target_col"]]),
    )
    compare_frame = None
    if inputs.get("compare_dataset_id"):
        _compare_dataset, compare_frame = _read_frame(
            runtime,
            str(inputs["compare_dataset_id"]),
            [str(feature) for feature in inputs["features"]],
        )
    metrics = []
    for feature in inputs["features"]:
        compare_values = None if compare_frame is None else compare_frame[str(feature)].to_numpy(dtype=float)
        item = feature_metrics(
            frame[str(feature)].to_numpy(dtype=float),
            frame[str(inputs["target_col"])].to_numpy(dtype=int),
            feature=str(feature),
            bins=int(inputs.get("bins") or 10),
            compare_values=compare_values,
        )
        metrics.append(_jsonable(item))
    return {"dataset_id": dataset.id, "metrics": metrics}


def tool_bin_feature(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    _dataset, frame = _read_frame(
        runtime,
        str(inputs["dataset_id"]),
        [str(inputs["feature"]), str(inputs["target_col"])],
    )
    edges = _edges_for(frame, inputs, ctx)
    result = compute_woe_iv(
        frame[str(inputs["feature"])].to_numpy(dtype=float),
        frame[str(inputs["target_col"])].to_numpy(dtype=int),
        edges,
        feature=str(inputs["feature"]),
    )
    payload = _jsonable(result)
    payload["bins"] = [_jsonable(bin_row) for bin_row in result.bins]
    payload["na_bin"] = _jsonable(result.na_bin) if result.na_bin else None
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
    _assert_columns(frame, [*features, target_col])
    out = frame.copy()
    woe_maps = {}
    new_columns = []
    for feature in features:
        edges = _edges_for(out, {**inputs, "feature": feature}, ctx)
        binning = compute_woe_iv(
            out[feature].to_numpy(dtype=float),
            out[target_col].to_numpy(dtype=int),
            edges,
            feature=feature,
        )
        woe = woe_result_from_binning(binning)
        encoded = woe_encode(out, feature, woe)
        out[encoded.name] = encoded
        new_columns.append(encoded.name)
        woe_maps[feature] = _jsonable(woe)
    result = _register_frame(runtime, out, dataset, ctx, "woe")
    return {"result_dataset_id": result.id, "new_columns": new_columns, "woe_maps": woe_maps}


def tool_onehot_encode(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    encoded, mapping = onehot_encode(
        frame,
        [str(item) for item in inputs["columns"]],
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
    for column in inputs["columns"]:
        col = str(column)
        if method == "minmax":
            values, column_params = minmax_normalize(
                out[col].to_numpy(dtype=float),
                feature_range=tuple(inputs.get("feature_range") or (0, 1)),
            )
        elif method == "zscore":
            values, column_params = zscore_standardize(out[col].to_numpy(dtype=float))
        else:
            raise FeatureError("method must be minmax or zscore")
        out[col] = values
        params[col] = column_params
    result = _register_frame(runtime, out, dataset, ctx, "normalize")
    return {"result_dataset_id": result.id, "scaler_params": _jsonable(params)}


def tool_impute_missing(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    out = frame.copy()
    fill_values = {}
    for column in inputs["columns"]:
        filled, value = impute_missing(
            out[str(column)],
            strategy=str(inputs["strategy"]),
            fill_value=inputs.get("fill_value"),
        )
        out[str(column)] = filled
        fill_values[str(column)] = value
    result = _register_frame(runtime, out, dataset, ctx, "impute")
    return {"result_dataset_id": result.id, "fill_values": _jsonable(fill_values)}


def tool_cap_outliers(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset, frame = _read_frame(runtime, str(inputs["dataset_id"]))
    out = frame.copy()
    bounds = {}
    for column in inputs["columns"]:
        values, params = cap_outliers(
            out[str(column)].to_numpy(dtype=float),
            method=str(inputs.get("method") or "iqr"),
            lower_q=float(inputs.get("lower_q", 0.01)),
            upper_q=float(inputs.get("upper_q", 0.99)),
        )
        out[str(column)] = values
        bounds[str(column)] = params
    result = _register_frame(runtime, out, dataset, ctx, "cap")
    return {"result_dataset_id": result.id, "bounds": _jsonable(bounds)}


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    return runtime.registry.register_existing(
        out_path,
        task_id=ctx.task_id,
        role="derived",
        anchor_target=source_dataset.id,
        seed=int(ctx.seed or 0),
    )


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
        return chimerge_edges(values, frame[target_col].to_numpy(dtype=int), max_bins=max_bins)
    if method == "tree":
        return tree_edges(values, frame[target_col].to_numpy(dtype=int), max_bins=max_bins, seed=int(ctx.seed or 0))
    raise FeatureError("method must be equal_frequency, equal_width, manual, chimerge, or tree")


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
