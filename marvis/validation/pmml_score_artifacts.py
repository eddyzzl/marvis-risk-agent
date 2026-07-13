from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
from time import monotonic
from typing import TypeVar
from uuid import uuid4

from filelock import FileLock, Timeout
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pypmml

from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
    topologically_sorted_transformations,
)
from marvis.validation.input_contracts import (
    TransformationSpec,
    ValidationInputContract,
)
from marvis.validation.pmml_scoring import PmmlScorer, load_pmml_scorer
from marvis.validation.results import (
    PMML_SCORING_RESULT_SCHEMA,
    PmmlScoringResult,
    validate_pmml_scoring_result_fields,
)
from marvis.validation.sample_chunks import iter_sample_chunks


SCORING_SCHEMA = PMML_SCORING_RESULT_SCHEMA
SCORING_ENGINE = "pypmml-pmml4s-batch"
SCORE_ARTIFACT_SCHEMA = pa.schema(
    [("row_id", pa.int64()), ("pmml_score", pa.float64())]
)
DEFAULT_IO_BLOCK_SIZE = 8 * 1024 * 1024
DEFAULT_VERIFY_BATCH_SIZE = 100_000
MAX_SCORING_ERROR_CHARS = 1_024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


@dataclass(frozen=True)
class _FileFingerprint:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _file_fingerprint(path: Path, *, label: str) -> _FileFingerprint:
    try:
        current = path.stat()
    except OSError as exc:
        raise ValueError(f"unable to inspect {label} file") from exc
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} path must be a regular file")
    return _FileFingerprint(
        device=current.st_dev,
        inode=current.st_ino,
        size=current.st_size,
        modified_ns=current.st_mtime_ns,
        changed_ns=current.st_ctime_ns,
    )


def _require_unchanged_file(
    path: Path, expected: _FileFingerprint, *, label: str
) -> None:
    if _file_fingerprint(path, label=label) != expected:
        raise ValueError(f"{label} file changed during PMML scoring")


def raise_if_cancelled(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _bounded_error(value: object, *, prefix: str = "") -> str:
    message = prefix + str(value)
    if len(message) <= MAX_SCORING_ERROR_CHARS:
        return message
    suffix = "... [truncated]"
    return message[: MAX_SCORING_ERROR_CHARS - len(suffix)] + suffix


def sha256_file_cancellable(
    path: Path,
    cancellation_check: Callable[[], None] | None = None,
    *,
    block_size: int = DEFAULT_IO_BLOCK_SIZE,
) -> str:
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("hash block size must be a positive integer")
    digest = sha256()
    try:
        with Path(path).open("rb") as handle:
            while True:
                raise_if_cancelled(cancellation_check)
                block = handle.read(block_size)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ValueError("unable to hash file") from exc
    raise_if_cancelled(cancellation_check)
    return digest.hexdigest()


def copy_file_cancellable(
    source: Path,
    destination: Path,
    *,
    cancellation_check: Callable[[], None] | None = None,
    block_size: int = DEFAULT_IO_BLOCK_SIZE,
) -> None:
    """Copy to a caller-owned staging path, removing partial output on failure."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("copy block size must be a positive integer")
    source = Path(source)
    destination = Path(destination)
    source_fingerprint = _file_fingerprint(source, label="source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as reader, destination.open("wb") as writer:
            while True:
                raise_if_cancelled(cancellation_check)
                block = reader.read(block_size)
                if not block:
                    break
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        _require_unchanged_file(source, source_fingerprint, label="source")
        shutil.copystat(source, destination)
        raise_if_cancelled(cancellation_check)
    except BaseException:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@contextmanager
def cancellable_file_lock(
    path: Path,
    cancellation_check: Callable[[], None] | None = None,
    *,
    poll_seconds: float = 0.25,
) -> Iterator[None]:
    """Acquire an explicit, externally located lock file without deleting it."""

    if (
        isinstance(poll_seconds, bool)
        or not isinstance(poll_seconds, (int, float))
        or not math.isfinite(float(poll_seconds))
        or poll_seconds <= 0
    ):
        raise ValueError("lock poll interval must be positive and finite")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    while True:
        raise_if_cancelled(cancellation_check)
        try:
            lock.acquire(timeout=float(poll_seconds))
            break
        except Timeout:
            continue
    try:
        raise_if_cancelled(cancellation_check)
        yield
    finally:
        lock.release()


class AtomicScoreWriter:
    """Write one score sidecar while preserving any prior complete artifact."""

    def __init__(self, final_path: Path):
        self.final_path = Path(final_path)
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = self.final_path.with_name(
            f".{self.final_path.name}.{uuid4().hex}.staging"
        )
        self.schema = SCORE_ARTIFACT_SCHEMA
        self.writer: pq.ParquetWriter | None = None
        self.closed = False
        self.committed = False
        try:
            self.writer = pq.ParquetWriter(self.staging_path, self.schema)
        except BaseException:
            try:
                self.staging_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def write(self, row_ids: np.ndarray, scores: np.ndarray) -> None:
        if self.closed or self.committed or self.writer is None:
            raise RuntimeError("PMML score writer is closed")
        row_values = np.asarray(row_ids)
        score_values = np.asarray(scores)
        if row_values.ndim != 1 or score_values.ndim != 1:
            raise ValueError("row_id and PMML score arrays must be one-dimensional")
        if len(row_values) != len(score_values):
            raise ValueError("row_id and PMML score lengths differ")
        if row_values.dtype.kind not in {"i", "u"}:
            raise ValueError("row_id values must be integers")
        try:
            table = pa.table(
                {
                    "row_id": pa.array(row_values, type=pa.int64()),
                    "pmml_score": pa.array(score_values, type=pa.float64()),
                },
                schema=self.schema,
            )
        except (pa.ArrowException, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid PMML score artifact values") from exc
        if table.column("row_id").null_count or table.column("pmml_score").null_count:
            raise ValueError("PMML score artifact values must not be null")
        self.writer.write_table(table)

    def _close(self) -> None:
        if self.closed:
            return
        if self.writer is None:
            self.closed = True
            return
        self.writer.close()
        self.closed = True

    def commit(
        self,
        *,
        cancellation_check: Callable[[], None] | None = None,
        prepare: Callable[[str], _T] | None = None,
    ) -> tuple[Path, str] | _T:
        """Close, hash and prepare first; replacement is the final fallible step."""

        if self.committed:
            raise RuntimeError("PMML score writer is already committed")
        raise_if_cancelled(cancellation_check)
        self._close()
        digest = sha256_file_cancellable(
            self.staging_path, cancellation_check=cancellation_check
        )
        prepared = prepare(digest) if prepare is not None else None
        # Nothing after this replace performs I/O, cancellation, hashing or
        # validation. A failed replace leaves the old final artifact untouched.
        os.replace(self.staging_path, self.final_path)
        self.committed = True
        if prepare is not None:
            return prepared  # type: ignore[return-value]
        return self.final_path, digest

    def rollback(self) -> None:
        if self.committed:
            return
        try:
            self._close()
        finally:
            self.staging_path.unlink(missing_ok=True)


def atomic_score_writer(final_path: Path) -> AtomicScoreWriter:
    return AtomicScoreWriter(final_path)


@dataclass
class _ScoreCounts:
    input_row_count: int = 0
    null_count: int = 0
    non_finite_count: int = 0

    def observe(self, scores: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(scores, errors="coerce")
        values = numeric.to_numpy(dtype=np.float64)
        null_mask = pd.isna(numeric).to_numpy(dtype=bool)
        non_finite_mask = (~null_mask) & ~np.isfinite(values)
        null_count = int(null_mask.sum())
        non_finite_count = int(non_finite_mask.sum())
        self.input_row_count += len(values)
        self.null_count += null_count
        self.non_finite_count += non_finite_count
        if null_count or non_finite_count:
            raise ValueError(
                "PMML scoring produced invalid values in a chunk: "
                f"null={null_count}, non_finite={non_finite_count}"
            )
        return values


def _pmml4s_version() -> str:
    package_dir = Path(pypmml.__file__).resolve().parent
    candidates = sorted((package_dir / "jars").glob("pmml4s_*-*.jar"))
    if len(candidates) != 1:
        return "unknown"
    name = candidates[0].stem
    match = re.search(r"-(?P<version>[^-]+)$", name)
    return match.group("version") if match else "unknown"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _runtime_identity() -> dict[str, str]:
    return {
        "engine": SCORING_ENGINE,
        "pypmml": _distribution_version("pypmml"),
        "pmml4s": _pmml4s_version(),
        "pandas": _distribution_version("pandas"),
    }


def pypmml_engine_version() -> str:
    identity = _runtime_identity()
    return ";".join(
        f"{name}={identity[name]}" for name in ("pypmml", "pmml4s", "pandas")
    )


def _transformation_sha256(specs: Sequence[TransformationSpec]) -> str:
    canonical = json.dumps(
        [asdict(item) for item in specs],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def pmml_scoring_cache_key(
    *,
    pmml_sha256: str,
    sample_sha256: str,
    output_field: str,
    engine_version: str,
    transformation_sha256: str,
    chunk_size: int | None = None,
) -> str:
    # Chunk size remains part of the identity until every supported source
    # format, notably quoted multi-line CSV, has a proven cross-chunk parity gate.
    canonical = json.dumps(
        {
            "schema": SCORING_SCHEMA,
            "sidecar_schema": str(SCORE_ARTIFACT_SCHEMA),
            "runtime": _runtime_identity(),
            "engine_version": engine_version,
            "pmml": pmml_sha256,
            "sample": sample_sha256,
            "output": output_field,
            "transformations": transformation_sha256,
            "chunk_size": chunk_size,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _transformation_closure(
    output_fields: Sequence[str], specs: Sequence[TransformationSpec]
) -> tuple[TransformationSpec, ...]:
    ordered = topologically_sorted_transformations(specs)
    by_output = {item.output_field: item for item in ordered}
    needed: set[str] = set()
    stack = list(output_fields)
    while stack:
        field = stack.pop()
        spec = by_output.get(field)
        if spec is None or field in needed:
            continue
        needed.add(field)
        stack.extend(spec.input_fields)
    return tuple(item for item in ordered if item.output_field in needed)


def _material_digest(
    *,
    supplied: str | None,
    contract: ValidationInputContract,
    role: str,
    path: Path,
    cancellation_check: Callable[[], None] | None,
) -> str:
    expected = contract.material_hashes.get(role)
    if expected is None or _SHA256_PATTERN.fullmatch(expected) is None:
        raise ValueError(f"invalid confirmed {role} SHA-256")
    if supplied is not None:
        if _SHA256_PATTERN.fullmatch(supplied) is None:
            raise ValueError(f"invalid supplied {role} SHA-256")
        if supplied != expected:
            raise ValueError(f"supplied {role} SHA-256 does not match contract")
    # A supplied digest is only corroborating stage evidence.  It must never
    # turn into a trust shortcut: the selected file may have been replaced
    # after confirmation (or after a prior stage calculated that digest).
    actual = sha256_file_cancellable(path, cancellation_check)
    if actual != expected:
        raise ValueError(f"current {role} file does not match confirmed SHA-256")
    return actual


def _build_scoring_result(
    *,
    cache_key: str,
    pmml_sha256: str,
    sample_sha256: str,
    engine_version: str,
    output_field: str,
    input_row_count: int,
    elapsed_seconds: float,
    chunk_size: int,
    required_input_count: int,
    score_path: Path,
    score_digest: str,
) -> PmmlScoringResult:
    elapsed = max(float(elapsed_seconds), 0.0)
    result = PmmlScoringResult(
        schema_version=SCORING_SCHEMA,
        cache_key=cache_key,
        pmml_sha256=pmml_sha256,
        sample_sha256=sample_sha256,
        engine=SCORING_ENGINE,
        engine_version=engine_version,
        output_field=output_field,
        input_row_count=input_row_count,
        success_count=input_row_count,
        failure_count=0,
        null_count=0,
        non_finite_count=0,
        elapsed_seconds=elapsed,
        rows_per_second=input_row_count / elapsed if elapsed > 0 else 0.0,
        chunk_size=chunk_size,
        required_input_count=required_input_count,
        missing_inputs=[],
        score_artifact_path=str(score_path),
        score_artifact_sha256=score_digest,
        status="pass",
        bounded_errors=[],
    )
    return validate_pmml_scoring_result_fields(result)


def run_pmml_scoring(
    *,
    contract: ValidationInputContract,
    sample_path: Path,
    pmml_path: Path,
    score_path: Path,
    chunk_size: int,
    scorer: PmmlScorer | None = None,
    pmml_sha256: str | None = None,
    sample_sha256: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
) -> PmmlScoringResult:
    if contract.status != "ready":
        raise ValueError("validation input contract is not ready for PMML scoring")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("PMML scoring chunk size must be a positive integer")
    sample_path = Path(sample_path)
    pmml_path = Path(pmml_path)
    score_path = Path(score_path)
    try:
        source_paths = {
            sample_path.expanduser().resolve(),
            pmml_path.expanduser().resolve(),
        }
        resolved_score_path = score_path.expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("invalid PMML scoring artifact path") from exc
    if resolved_score_path in source_paths:
        raise ValueError("PMML score artifact path must differ from input materials")
    sample_fingerprint = _file_fingerprint(sample_path, label="sample")
    pmml_fingerprint = _file_fingerprint(pmml_path, label="PMML")
    raise_if_cancelled(cancellation_check)

    resolved_pmml_sha256 = _material_digest(
        supplied=pmml_sha256,
        contract=contract,
        role="pmml",
        path=pmml_path,
        cancellation_check=cancellation_check,
    )
    resolved_sample_sha256 = _material_digest(
        supplied=sample_sha256,
        contract=contract,
        role="sample",
        path=sample_path,
        cancellation_check=cancellation_check,
    )
    _require_unchanged_file(sample_path, sample_fingerprint, label="sample")
    _require_unchanged_file(pmml_path, pmml_fingerprint, label="PMML")

    manifest = contract.require_pmml_manifest()
    output_field = contract.require_output_field()
    transformations = _transformation_closure(
        manifest.raw_required_fields, contract.transformations
    )
    projected_columns = required_transformation_inputs(
        manifest.raw_required_fields, transformations
    )
    transformation_digest = _transformation_sha256(transformations)
    engine_version = pypmml_engine_version()
    cache_key = pmml_scoring_cache_key(
        pmml_sha256=resolved_pmml_sha256,
        sample_sha256=resolved_sample_sha256,
        output_field=output_field,
        engine_version=engine_version,
        transformation_sha256=transformation_digest,
        chunk_size=chunk_size,
    )
    active_scorer = scorer or load_pmml_scorer(pmml_path, output_field)
    writer: AtomicScoreWriter | None = None
    counts = _ScoreCounts()
    started = monotonic()
    try:
        writer = atomic_score_writer(score_path)
        for chunk in iter_sample_chunks(
            sample_path,
            columns=projected_columns,
            chunk_size=chunk_size,
            schema=contract.require_sample_schema(),
        ):
            raise_if_cancelled(cancellation_check)
            scoring_frame = apply_confirmed_transformations(
                chunk.frame, transformations
            )
            scoring_frame = scoring_frame.loc[:, list(manifest.raw_required_fields)]
            scores = active_scorer.score_chunk(scoring_frame)
            if not isinstance(scores, pd.Series):
                raise ValueError("PMML scorer must return a one-dimensional Series")
            if len(scores) != len(chunk.row_ids):
                raise ValueError(
                    "PMML scorer row count mismatch: "
                    f"expected={len(chunk.row_ids)}, actual={len(scores)}"
                )
            values = counts.observe(scores)
            writer.write(chunk.row_ids, values)
        if counts.input_row_count == 0:
            raise ValueError("validation sample contains no rows")
        expected_rows = contract.require_sample_schema().row_count
        if expected_rows is not None and counts.input_row_count != expected_rows:
            raise ValueError(
                "validation sample row count changed after inspection: "
                f"expected={expected_rows}, actual={counts.input_row_count}"
            )
        raise_if_cancelled(cancellation_check)
        _require_unchanged_file(sample_path, sample_fingerprint, label="sample")
        _require_unchanged_file(pmml_path, pmml_fingerprint, label="PMML")
        elapsed = monotonic() - started

        def prepare_result(score_digest: str) -> PmmlScoringResult:
            # The staging hash may take long enough for a selected source to
            # be replaced after the pre-commit check.  Recheck at the last
            # fallible point before publishing the sidecar.
            _require_unchanged_file(sample_path, sample_fingerprint, label="sample")
            _require_unchanged_file(pmml_path, pmml_fingerprint, label="PMML")
            return _build_scoring_result(
                cache_key=cache_key,
                pmml_sha256=resolved_pmml_sha256,
                sample_sha256=resolved_sample_sha256,
                engine_version=engine_version,
                output_field=output_field,
                input_row_count=counts.input_row_count,
                elapsed_seconds=elapsed,
                chunk_size=chunk_size,
                required_input_count=len(manifest.raw_required_fields),
                score_path=score_path,
                score_digest=score_digest,
            )

        prepared = writer.commit(
            cancellation_check=cancellation_check,
            prepare=prepare_result,
        )
        # AtomicScoreWriter guarantees that prepare returned this exact type and
        # that no fallible operation occurs after the final replacement.
        return prepared  # type: ignore[return-value]
    except BaseException as exc:
        if writer is not None:
            try:
                writer.rollback()
            except BaseException:
                pass
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if isinstance(exc, Exception) and exc.__class__.__module__.startswith(
            "marvis."
        ):
            raise
        if isinstance(exc, ValueError) and len(str(exc)) <= MAX_SCORING_ERROR_CHARS:
            raise
        raise ValueError(_bounded_error(exc, prefix="PMML scoring failed: ")) from exc


def validate_pmml_score_artifact(
    result: PmmlScoringResult,
    score_path: Path,
    *,
    expected_cache_key: str | None = None,
    cancellation_check: Callable[[], None] | None = None,
    batch_size: int = DEFAULT_VERIFY_BATCH_SIZE,
) -> PmmlScoringResult:
    validate_pmml_scoring_result_fields(result)
    if result.schema_version != SCORING_SCHEMA or result.status != "pass":
        raise ValueError("PMML scoring result is not a passing v1 artifact")
    if expected_cache_key is not None and result.cache_key != expected_cache_key:
        raise ValueError("PMML scoring result/cache key mismatch")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("PMML score verification batch size must be positive")
    score_path = Path(score_path)
    fingerprint = _file_fingerprint(score_path, label="PMML score sidecar")
    raise_if_cancelled(cancellation_check)
    digest = sha256_file_cancellable(score_path, cancellation_check)
    _require_unchanged_file(
        score_path, fingerprint, label="PMML score sidecar"
    )
    if result.score_artifact_sha256 != digest:
        raise ValueError("PMML score sidecar hash mismatch")
    try:
        parquet = pq.ParquetFile(score_path)
        if parquet.schema_arrow != SCORE_ARTIFACT_SCHEMA:
            raise ValueError("PMML score sidecar schema mismatch")
        if parquet.metadata.num_rows != result.input_row_count:
            raise ValueError("PMML score sidecar row count mismatch")
        offset = 0
        for batch in parquet.iter_batches(
            columns=["row_id", "pmml_score"], batch_size=batch_size
        ):
            raise_if_cancelled(cancellation_check)
            row_column = batch.column(0)
            score_column = batch.column(1)
            if row_column.null_count or score_column.null_count:
                raise ValueError("PMML score sidecar contains null values")
            row_ids = row_column.to_numpy(zero_copy_only=False)
            scores = score_column.to_numpy(zero_copy_only=False)
            expected = np.arange(offset, offset + batch.num_rows, dtype=np.int64)
            if not np.array_equal(row_ids, expected):
                raise ValueError("PMML score sidecar row_id is not contiguous")
            if not np.isfinite(scores).all():
                raise ValueError("PMML score sidecar contains a non-finite score")
            offset += batch.num_rows
        if offset != result.input_row_count:
            raise ValueError("PMML score sidecar row count mismatch")
    except ValueError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("invalid PMML score sidecar") from exc
    raise_if_cancelled(cancellation_check)
    _require_unchanged_file(
        score_path, fingerprint, label="PMML score sidecar"
    )
    return replace(result, score_artifact_path=str(score_path))
