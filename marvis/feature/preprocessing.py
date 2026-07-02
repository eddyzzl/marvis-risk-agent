"""Preprocessing chain persistence and scoring-time replay (PREP-2).

Feature-pack transforms (impute/cap/normalize/onehot) fit their parameters on
train-only rows and apply them in place, but historically only echoed the fitted
params in the tool's JSON response -- nothing was written to disk alongside the
derived dataset, so a model trained downstream had no way to replay those exact
transforms on new raw data at scoring time.

This module defines the on-disk lineage format (a JSON sidecar next to each
derived dataset's parquet file) and the replay primitives that both the
in-process ``_ModelArtifactScorer`` and the generated handoff notebook use to
turn ``preprocessing_steps`` back into concrete column transforms.

A ``PreprocessingStep`` is a plain, JSON-safe dict:

    {"kind": "impute" | "cap" | "normalize" | "onehot",
     "columns": [...],
     "params": {...}}

``params`` mirrors what each tool already returns today (fill_values /
bounds / scaler_params / mapping), keyed by column name so ``kind`` +
``columns`` + ``params`` fully describes the transform. WOE is intentionally
excluded -- scorecard/woe_encode already replay their own WOE maps and are not
duplicated here (see module docstring in tools.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marvis.feature.errors import FeatureError
from marvis.feature.transform import apply_scaler


_SIDECAR_SUFFIX = ".preprocessing.json"


def sidecar_path(dataset_path: Path) -> Path:
    """The lineage sidecar path for a dataset parquet file."""
    path = Path(dataset_path)
    return path.with_name(path.name + _SIDECAR_SUFFIX) if path.suffix != ".json" else path


def read_preprocessing_chain(dataset_path: Path) -> list[dict[str, Any]]:
    """Read the accumulated preprocessing chain for a dataset, or ``[]`` when the
    dataset has no lineage sidecar (e.g. a historical / non-derived dataset)."""
    path = sidecar_path(dataset_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    steps = payload.get("preprocessing_steps") if isinstance(payload, dict) else None
    return [dict(step) for step in steps] if isinstance(steps, list) else []


def write_preprocessing_chain(dataset_path: Path, steps: list[dict[str, Any]]) -> Path:
    """Write the accumulated preprocessing chain sidecar next to ``dataset_path``."""
    path = sidecar_path(dataset_path)
    payload = {"preprocessing_steps": steps}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def append_step(
    source_dataset_path: Path | None,
    *,
    kind: str,
    columns: list[str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the accumulated chain for a newly derived dataset: the source
    dataset's chain (if any) plus this one new step."""
    chain = read_preprocessing_chain(source_dataset_path) if source_dataset_path else []
    chain = [*chain, {"kind": str(kind), "columns": [str(c) for c in columns], "params": params}]
    return chain


def apply_preprocessing_steps(frame: pd.DataFrame, steps: list[dict[str, Any]]) -> pd.DataFrame:
    """Replay a preprocessing chain on new raw data, in order.

    Mirrors each tool's own apply semantics (in-place overwrite for
    impute/cap/normalize; drop+dummy for onehot) so scoring-time output matches
    what the platform computed at training/derivation time. Missing input
    columns are skipped per-column (a step targeting a column that does not
    exist in ``frame`` is a no-op for that column) rather than raising, mirroring
    each tool's existing missing-value tolerance.
    """
    out = frame.copy()
    for step in steps:
        kind = str(step.get("kind") or "")
        columns = [str(c) for c in step.get("columns") or []]
        params = step.get("params") or {}
        if kind == "impute":
            out = _apply_impute(out, columns, params)
        elif kind == "cap":
            out = _apply_cap(out, columns, params)
        elif kind == "normalize":
            out = _apply_normalize(out, columns, params)
        elif kind == "onehot":
            out = _apply_onehot(out, columns, params)
        else:
            raise FeatureError(f"unsupported preprocessing step kind: {kind!r}")
    return out


def _apply_impute(frame: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        value = params.get(column)
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        out[column] = out[column].fillna(value)
    return out


def _apply_cap(frame: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        bounds = params.get(column) or {}
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if lower is not None and upper is not None and np.isfinite(lower) and np.isfinite(upper):
            values[mask] = np.clip(values[mask], float(lower), float(upper))
        out[column] = values
    return out


def _apply_normalize(frame: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            continue
        column_params = params.get(column) or {}
        kind = "minmax" if "min" in column_params else "zscore"
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        out[column] = apply_scaler(values, column_params, kind=kind)
    return out


def _apply_onehot(frame: pd.DataFrame, columns: list[str], params: dict) -> pd.DataFrame:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return frame.copy()
    out = frame.copy()
    dummy_frames = []
    for column in present:
        categories = params.get(column) or []
        data = {
            f"{column}_{category}": (out[column] == category).astype(int)
            for category in categories
        }
        dummy_frames.append(pd.DataFrame(data, index=out.index))
    out = out.drop(columns=present)
    if dummy_frames:
        out = pd.concat([out, *dummy_frames], axis=1)
    return out


__all__ = [
    "append_step",
    "apply_preprocessing_steps",
    "read_preprocessing_chain",
    "sidecar_path",
    "write_preprocessing_chain",
]
