from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.errors import DataBackendError


def sample_dataset(
    backend: DataBackend,
    path: Path,
    n: int,
    *,
    strategy: str = "random",
    seed: int = 0,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    warnings.warn(
        "sample_dataset() is deprecated and retained for one compatibility "
        "release cycle; use DataBackend.read_frame(..., nrows=n) for head "
        "sampling or DataBackend.sample_rows(..., seed=seed) for random "
        "sampling, and DataBackend.sample_rows_stratified(...) for stratified "
        "sampling.",
        DeprecationWarning,
        stacklevel=2,
    )
    if n <= 0:
        raise DataBackendError("sample size must be positive")
    if strategy == "head":
        return backend.read_frame(path, nrows=n)
    if strategy == "stratified":
        if not stratify_col:
            raise DataBackendError("stratified sampling requires stratify_col")
        return backend.sample_rows_stratified(
            path,
            n,
            stratify_col=stratify_col,
            seed=seed,
        )
    if strategy == "random":
        return backend.sample_rows(path, n, seed=seed)
    raise DataBackendError(f"unsupported sampling strategy: {strategy}")
__all__ = ["sample_dataset"]
