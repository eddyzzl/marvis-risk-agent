from __future__ import annotations

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
    if n <= 0:
        raise DataBackendError("sample size must be positive")
    if strategy == "head":
        return backend.read_frame(path, nrows=n)
    if strategy == "stratified":
        if not stratify_col:
            raise DataBackendError("stratified sampling requires stratify_col")
        frame = backend.read_frame(path)
        return _stratified_sample(frame, n=n, stratify_col=stratify_col, seed=seed)
    if strategy == "random":
        return backend.sample_rows(path, n, seed=seed)
    raise DataBackendError(f"unsupported sampling strategy: {strategy}")


def _stratified_sample(
    frame: pd.DataFrame,
    *,
    n: int,
    stratify_col: str,
    seed: int,
) -> pd.DataFrame:
    if stratify_col not in frame.columns:
        raise DataBackendError(f"unknown stratify_col: {stratify_col}")
    if frame.empty or len(frame) <= n:
        return frame.reset_index(drop=True)

    groups = [
        (key, group)
        for key, group in frame.groupby(stratify_col, group_keys=False, dropna=False)
    ]
    allocations = _allocate_strata(n, [len(group) for _, group in groups])
    samples = [
        group.sample(n=count, random_state=int(seed))
        for (_, group), count in zip(groups, allocations)
        if count > 0
    ]
    if not samples:
        return frame.head(0).reset_index(drop=True)
    return pd.concat(samples).sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def _allocate_strata(n: int, sizes: list[int]) -> list[int]:
    total = sum(sizes)
    if total == 0:
        return [0 for _ in sizes]
    allocations = [min(size, max(1, round(n * size / total))) for size in sizes]
    while sum(allocations) > n:
        candidates = [
            index
            for index, allocation in enumerate(allocations)
            if allocation > 1
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: allocations[item])
        allocations[index] -= 1
    while sum(allocations) < n:
        candidates = [
            index
            for index, size in enumerate(sizes)
            if allocations[index] < size
        ]
        if not candidates:
            break
        index = max(candidates, key=lambda item: sizes[item] - allocations[item])
        allocations[index] += 1
    return allocations


__all__ = ["sample_dataset"]
