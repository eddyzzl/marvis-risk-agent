from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from marvis.validation.input_contracts import SampleSchema
from marvis.validation.sample_schema import (
    inspect_sample_schema,
    iter_sample_projection,
)


@dataclass(frozen=True)
class SampleChunk:
    """A projected sample batch and its stable, zero-based source row IDs."""

    row_ids: np.ndarray
    frame: pd.DataFrame


def iter_sample_chunks(
    path: Path,
    *,
    columns: tuple[str, ...],
    chunk_size: int,
    schema: SampleSchema | None = None,
) -> Iterator[SampleChunk]:
    """Yield projected sample rows with contiguous ``int64`` row IDs.

    Format, encoding, and Excel-sheet behavior is deliberately delegated to
    :func:`iter_sample_projection`.  An empty projection is supported because
    a PMML model may have no raw input fields: one schema column is then read
    as a row-count carrier and removed before the chunk is returned.
    """

    _validate_chunk_size(chunk_size)

    selected_schema = schema
    projection_columns = columns
    carrier_column: str | None = None
    if not columns:
        if selected_schema is None:
            selected_schema = inspect_sample_schema(path)
        elif not isinstance(selected_schema, SampleSchema):
            # Keep the public error deterministic instead of trying to access
            # ``columns`` on an arbitrary object.
            raise TypeError("schema must be a SampleSchema")
        if not selected_schema.columns:
            # SampleSchema inspection rejects this state.  Keep a bounded
            # guard for manually constructed contracts.
            raise ValueError("sample schema contains no carrier column")
        carrier_column = selected_schema.columns[0]
        projection_columns = (carrier_column,)

    offset = 0
    for projected in iter_sample_projection(
        path,
        columns=projection_columns,
        chunk_size=chunk_size,
        schema=selected_schema,
    ):
        if projected.empty:
            continue
        frame = projected.reset_index(drop=True)
        if carrier_column is not None:
            # Selecting no columns from a DataFrame retains its row index, so
            # this preserves row cardinality without exposing the carrier.
            frame = frame.iloc[:, 0:0].copy()
        row_ids = np.arange(offset, offset + len(frame), dtype=np.int64)
        yield SampleChunk(row_ids=row_ids, frame=frame)
        offset += len(frame)


def read_selected_columns(
    path: Path,
    *,
    columns: tuple[str, ...],
    chunk_size: int = 100_000,
    schema: SampleSchema | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> pd.DataFrame:
    """Materialize selected columns, checking cancellation once per chunk."""

    _validate_chunk_size(chunk_size)
    frames: list[pd.DataFrame] = []
    for chunk in iter_sample_chunks(
        path,
        columns=columns,
        chunk_size=chunk_size,
        schema=schema,
    ):
        if cancellation_check is not None:
            cancellation_check()
        frames.append(chunk.frame)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, axis=0, ignore_index=True).loc[:, list(columns)]


def _validate_chunk_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sample chunk size must be a positive integer")
