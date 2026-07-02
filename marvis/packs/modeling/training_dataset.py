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
    """Serves every recipe's ``read_frame`` from the single cached modeling frame.

    PERF-10: the cached frame used to be deep-copied on *every* ``read_frame``
    call, so an N-recipe ``train_models`` run duplicated the whole (potentially
    very wide) modeling frame N times and the resident-memory peak scaled
    linearly with the recipe count. Every recipe only ever *reads* from the
    frame (it slices splits via boolean masks / ``.loc`` and never assigns back
    into it), so that eager duplication was pure overhead. Each read now returns
    a shallow Copy-on-Write view (``copy(deep=False)`` for the whole frame, a
    column projection when ``columns`` is given): it is a distinct object that
    shares the cached blocks with zero up-front duplication, yet -- because
    pandas Copy-on-Write is always on here (pandas >= 3) -- a consumer that did
    write into the returned frame would transparently copy only the touched
    columns at that moment, leaving the cache intact. The defensive isolation
    the old deep copy provided is thus preserved, but the per-run memory
    footprint is now flat in the recipe count instead of linear.
    """

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
            # A column projection is already a distinct CoW view; it also shrinks
            # the working set to just the requested columns.
            frame = frame[list(columns)]
        else:
            # Shallow CoW view of the whole frame: no eager copy, but a distinct
            # object so any downstream write copies-on-write instead of touching
            # the shared cache.
            frame = frame.copy(deep=False)
        if nrows is not None:
            frame = frame.head(int(nrows))
        return frame

    def column_names(self, path: Path) -> list[str]:
        if Path(path) != self._dataset.path:
            return self._fallback.column_names(path)
        return [str(column) for column in self._dataset.frame.columns]
