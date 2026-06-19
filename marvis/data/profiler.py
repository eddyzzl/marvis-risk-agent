from __future__ import annotations

from pathlib import Path

from marvis.data.backend import DataBackend
from marvis.data.contracts import SMALL_SAMPLE_N, ColumnProfile
from marvis.data.schema_infer import infer_dataset_schema


PROFILE_SAMPLE_N = SMALL_SAMPLE_N


def profile_dataset(
    backend: DataBackend,
    path: Path,
    *,
    seed: int = 0,
) -> list[ColumnProfile]:
    sample = backend.sample_rows(path, PROFILE_SAMPLE_N, seed=seed)
    return infer_dataset_schema(sample, seed=seed)


__all__ = ["PROFILE_SAMPLE_N", "profile_dataset"]
