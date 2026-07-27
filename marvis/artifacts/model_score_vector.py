"""Deterministic immutable Parquet contract for governed model scores.

The artifact deliberately carries only a zero-based row ordinal and one
``float64`` probability.  It is neutral infrastructure: model, task, and
Strategy lineage belong to the registering Tool's canonical JSON/provenance,
not to Parquet metadata that callers could accidentally reinterpret.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import hmac
import os
from pathlib import Path
import stat

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MODEL_SCORE_VECTOR_SCHEMA_VERSION = "marvis.model-score-vector.v1"
MODEL_SCORE_VECTOR_WRITER_VERSION = "marvis.model-score-vector-parquet/1"
MAX_MODEL_SCORE_VECTOR_ROWS = 2_000_000
MAX_MODEL_SCORE_VECTOR_BYTES = 128 * 1024 * 1024
MODEL_SCORE_VECTOR_ROW_GROUP_SIZE = 65_536
MODEL_SCORE_VECTOR_SCHEMA = pa.schema(
    [
        pa.field("row_ordinal", pa.int64(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ],
    metadata={
        b"marvis.schema_version": MODEL_SCORE_VECTOR_SCHEMA_VERSION.encode("ascii"),
        b"marvis.writer_version": MODEL_SCORE_VECTOR_WRITER_VERSION.encode("ascii"),
    },
)


class ModelScoreVectorError(ValueError):
    """A score vector does not satisfy the immutable Parquet contract."""


@dataclass(frozen=True, eq=False)
class ModelScoreVector:
    """Fully validated score-vector bytes and values."""

    path: Path
    content_hash: str
    row_count: int
    row_ordinals: np.ndarray
    scores: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelScoreVector):
            return NotImplemented
        return (
            self.path == other.path
            and self.content_hash == other.content_hash
            and self.row_count == other.row_count
            and np.array_equal(self.row_ordinals, other.row_ordinals)
            and np.array_equal(self.scores, other.scores)
        )

    @property
    def score_min(self) -> float:
        return float(np.min(self.scores))

    @property
    def score_max(self) -> float:
        return float(np.max(self.scores))


def write_model_score_vector(
    path: Path | str,
    scores: Sequence[float] | np.ndarray,
) -> ModelScoreVector:
    """Write one deterministic Parquet vector, then read it through the validator."""

    destination = Path(path)
    normalized = _scores(scores)
    if destination.is_symlink():
        raise ModelScoreVectorError("model score vector path must not be a symlink")
    if destination.exists() and not destination.is_file():
        raise ModelScoreVectorError("model score vector path must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(normalized.size, dtype=np.int64), type=pa.int64()),
            pa.array(normalized, type=pa.float64()),
        ],
        schema=MODEL_SCORE_VECTOR_SCHEMA,
    )
    try:
        pq.write_table(
            table,
            destination,
            row_group_size=MODEL_SCORE_VECTOR_ROW_GROUP_SIZE,
            compression="NONE",
            use_dictionary=False,
            write_statistics=False,
            version="2.6",
            data_page_version="2.0",
            store_schema=True,
        )
    except (OSError, pa.ArrowException) as exc:
        raise ModelScoreVectorError("model score vector could not be written") from exc
    return validate_model_score_vector(
        destination,
        expected_row_count=int(normalized.size),
    )


def validate_model_score_vector(
    path: Path | str,
    *,
    expected_content_hash: str | None = None,
    expected_row_count: int | None = None,
) -> ModelScoreVector:
    """Authenticate schema, ordinals, probabilities, row count, and file bytes."""

    source = Path(path)
    before = _regular_file_stat(source)
    try:
        parquet = pq.ParquetFile(source)
    except (OSError, pa.ArrowException) as exc:
        raise ModelScoreVectorError(
            "model score vector is not readable Parquet"
        ) from exc
    if parquet.schema_arrow != MODEL_SCORE_VECTOR_SCHEMA:
        raise ModelScoreVectorError("model score vector schema is not canonical")
    row_count = int(parquet.metadata.num_rows)
    _row_count(row_count)
    if expected_row_count is not None and row_count != _expected_row_count(
        expected_row_count
    ):
        raise ModelScoreVectorError("model score vector row count changed")

    ordinal_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    offset = 0
    try:
        for batch in parquet.iter_batches(
            batch_size=MODEL_SCORE_VECTOR_ROW_GROUP_SIZE,
            columns=["row_ordinal", "score"],
            use_threads=False,
        ):
            if batch.schema != MODEL_SCORE_VECTOR_SCHEMA:
                raise ModelScoreVectorError(
                    "model score vector batch schema is not canonical"
                )
            if batch.column(0).null_count or batch.column(1).null_count:
                raise ModelScoreVectorError(
                    "model score vector columns must not contain nulls"
                )
            ordinals = np.asarray(
                batch.column(0).to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            scores = np.asarray(
                batch.column(1).to_numpy(zero_copy_only=False),
                dtype=np.float64,
            )
            expected = np.arange(offset, offset + len(ordinals), dtype=np.int64)
            if not np.array_equal(ordinals, expected):
                raise ModelScoreVectorError(
                    "model score vector row_ordinal is not zero-based consecutive"
                )
            _probabilities(scores)
            ordinal_parts.append(ordinals)
            score_parts.append(scores)
            offset += len(ordinals)
    except ModelScoreVectorError:
        raise
    except (OSError, pa.ArrowException, ValueError) as exc:
        raise ModelScoreVectorError(
            "model score vector could not be validated"
        ) from exc
    if offset != row_count:
        raise ModelScoreVectorError(
            "model score vector row count does not match Parquet metadata"
        )

    after = _regular_file_stat(source)
    if before != after:
        raise ModelScoreVectorError("model score vector changed while reading")
    observed_hash = _sha256_file(source, expected_stat=after)
    final = _regular_file_stat(source)
    if after != final:
        raise ModelScoreVectorError("model score vector changed while hashing")
    if expected_content_hash is not None:
        expected_hash = _content_hash(expected_content_hash)
        if not hmac.compare_digest(observed_hash, expected_hash):
            raise ModelScoreVectorError("model score vector content hash changed")
    row_ordinals = np.concatenate(ordinal_parts).astype(np.int64, copy=False)
    scores = np.concatenate(score_parts).astype(np.float64, copy=False)
    row_ordinals.setflags(write=False)
    scores.setflags(write=False)
    return ModelScoreVector(
        path=source,
        content_hash=observed_hash,
        row_count=row_count,
        row_ordinals=row_ordinals,
        scores=scores,
    )


def _scores(value: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ModelScoreVectorError("model scores must be a numeric vector") from exc
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ModelScoreVectorError(
            "model scores must be a one-dimensional numeric vector"
        )
    normalized = np.ascontiguousarray(raw, dtype=np.float64)
    _row_count(int(normalized.size))
    _probabilities(normalized)
    return normalized


def _probabilities(scores: np.ndarray) -> None:
    if not np.all(np.isfinite(scores)):
        raise ModelScoreVectorError("model scores must all be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ModelScoreVectorError("model scores must all be in [0, 1]")


def _row_count(value: int) -> int:
    if value <= 0:
        raise ModelScoreVectorError("model score vector must not be empty")
    if value > MAX_MODEL_SCORE_VECTOR_ROWS:
        raise ModelScoreVectorError("model score vector exceeds row budget")
    return value


def _expected_row_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelScoreVectorError("expected row count must be an integer")
    return _row_count(value)


def _regular_file_stat(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise ModelScoreVectorError("model score vector must not be a symlink")
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ModelScoreVectorError("model score vector is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise ModelScoreVectorError("model score vector must be a regular file")
    if observed.st_size <= 0 or observed.st_size > MAX_MODEL_SCORE_VECTOR_BYTES:
        raise ModelScoreVectorError("model score vector file size is invalid")
    return (
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_mode),
    )


def _content_hash(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ModelScoreVectorError("expected content hash must be lowercase SHA-256")
    return value


def _sha256_file(
    path: Path,
    *,
    expected_stat: tuple[int, int, int, int],
) -> str:
    digest = hashlib.sha256()
    descriptor = -1
    try:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        before = _descriptor_stat(descriptor)
        if before != expected_stat:
            raise ModelScoreVectorError("model score vector changed before hashing")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        if _descriptor_stat(descriptor) != before:
            raise ModelScoreVectorError("model score vector changed while hashing")
    except ModelScoreVectorError:
        raise
    except OSError as exc:
        raise ModelScoreVectorError("model score vector could not be hashed") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _descriptor_stat(descriptor: int) -> tuple[int, int, int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        raise ModelScoreVectorError("model score vector must be a regular file")
    if observed.st_size <= 0 or observed.st_size > MAX_MODEL_SCORE_VECTOR_BYTES:
        raise ModelScoreVectorError("model score vector file size is invalid")
    return (
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_mode),
    )


__all__ = [
    "MAX_MODEL_SCORE_VECTOR_BYTES",
    "MAX_MODEL_SCORE_VECTOR_ROWS",
    "MODEL_SCORE_VECTOR_ROW_GROUP_SIZE",
    "MODEL_SCORE_VECTOR_SCHEMA",
    "MODEL_SCORE_VECTOR_SCHEMA_VERSION",
    "MODEL_SCORE_VECTOR_WRITER_VERSION",
    "ModelScoreVector",
    "ModelScoreVectorError",
    "validate_model_score_vector",
    "write_model_score_vector",
]
