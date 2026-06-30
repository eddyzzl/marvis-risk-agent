from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class TrainingDataset:
    """Cached modeling frame shared by multi-recipe training.

    Recipes still accept the historical ``backend, dataset_path`` arguments. The
    adapter below keeps that API stable while ensuring the train_models loop does
    not re-read the same full dataset for every algorithm.
    """

    path: Path
    frame: pd.DataFrame

    @classmethod
    def load(cls, backend, path: Path) -> "TrainingDataset":
        resolved = Path(path)
        return cls(path=resolved, frame=backend.read_frame(resolved))

    def backend_adapter(self, fallback_backend):
        return _TrainingDatasetBackend(self, fallback_backend)


class _TrainingDatasetBackend:
    def __init__(self, dataset: TrainingDataset, fallback_backend):
        self._dataset = dataset
        self._fallback = fallback_backend

    def read_frame(
        self,
        path: Path,
        *,
        columns: Sequence[str] | None = None,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        if Path(path) != self._dataset.path:
            return self._fallback.read_frame(path, columns=columns, nrows=nrows)
        frame = self._dataset.frame
        if columns is not None:
            frame = frame[list(columns)]
        if nrows is not None:
            frame = frame.head(int(nrows))
        return frame.copy()

    def column_names(self, path: Path) -> list[str]:
        if Path(path) != self._dataset.path:
            return self._fallback.column_names(path)
        return [str(column) for column in self._dataset.frame.columns]
