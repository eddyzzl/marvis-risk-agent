from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import stat
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marvis.validation.binning import (
    bin_distribution,
    bin_table,
    compute_ks,
    compute_psi,
    equal_frequency_bin_edges,
)
from marvis.validation.config import ValidationConfig
from marvis.validation.field_transformations import (
    apply_confirmed_transformations,
    required_transformation_inputs,
    topologically_sorted_transformations,
)
from marvis.validation.input_contracts import (
    TransformationSpec,
    ValidationInputContract,
)
from marvis.validation.pmml_score_artifacts import (
    SCORE_ARTIFACT_SCHEMA,
    atomic_score_writer,
    cancellable_file_lock,
    copy_file_cancellable,
    raise_if_cancelled,
    sha256_file_cancellable,
)
from marvis.validation.pmml_scoring import PmmlScorer
from marvis.validation.platform_metrics import load_pmml_analysis_frame
from marvis.validation.results import (
    PmmlScoringResult,
    StressBaseline,
    StressCategoryResult,
    StressTestResult,
)
from marvis.validation.sample_chunks import iter_sample_chunks
from marvis.validation.stress_test import (
    STRESS_MISSING_VALUE,
    require_complete_stress_result,
)


STRESS_CACHE_SCHEMA = "marvis.pmml_stress_cache.v1"
_STRESS_CACHE_METADATA = "metadata.json"
_STRESS_CACHE_SCORES = "scores.parquet"
_STRESS_CACHE_METADATA_FIELDS = {
    "schema_version",
    "cache_key",
    "category",
    "row_count",
    "score_sha256",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_CACHE_METADATA_BYTES = 64 * 1024


@dataclass(frozen=True)
class OotStressContext:
    row_ids: np.ndarray
    labels: np.ndarray
    baseline_scores: np.ndarray


@dataclass(frozen=True)
class ScenarioScoreArtifact:
    category: str
    path: Path
    row_count: int
    sha256: str


@dataclass(frozen=True)
class OotInputArtifact:
    path: Path
    row_count: int
    sha256: str


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


class _CorruptStressCache(ValueError):
    """A cache-local validation failure that is safe to heal by recomputing."""


def stress_cache_key(
    *,
    baseline_cache_key: str,
    category: str,
    raw_fields: tuple[str, ...],
    chunk_size: int,
    sentinel: int | float = STRESS_MISSING_VALUE,
) -> str:
    """Return the immutable identity of one PMML stress scenario."""

    _require_sha256(baseline_cache_key, label="baseline cache key")
    _require_positive_chunk_size(chunk_size)
    if not isinstance(category, str) or not category.strip():
        raise ValueError("model stress category must be non-empty")
    if (
        not isinstance(raw_fields, tuple)
        or not raw_fields
        or any(not isinstance(field, str) or not field for field in raw_fields)
        or len(set(raw_fields)) != len(raw_fields)
    ):
        raise ValueError("model stress cache raw fields must be unique strings")
    if (
        isinstance(sentinel, bool)
        or not isinstance(sentinel, (int, float))
        or not np.isfinite(float(sentinel))
    ):
        raise ValueError("model stress sentinel must be finite")
    canonical = json.dumps(
        {
            "schema": STRESS_CACHE_SCHEMA,
            "score_schema": str(SCORE_ARTIFACT_SCHEMA),
            "baseline_cache_key": baseline_cache_key,
            "category": category,
            "raw_fields": list(raw_fields),
            "sentinel": sentinel,
            "chunk_size": chunk_size,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def materialize_cached_stress_scenario(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    output_path: Path,
    cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact:
    """Validate and atomically materialize one complete cached scenario."""

    _require_sha256(cache_key, label="stress cache key")
    expected_ids = _validated_row_ids(expected_row_ids, label="stress row IDs")
    if len(expected_ids) == 0:
        raise ValueError("stress cache materialization requires rows")
    with cancellable_file_lock(
        _stress_cache_lock_path(Path(cache_dir), cache_key), cancellation_check
    ):
        try:
            cached = _load_valid_cache_entry_locked(
                cache_dir=Path(cache_dir),
                cache_key=cache_key,
                category=category,
                expected_row_ids=expected_ids,
                cancellation_check=cancellation_check,
            )
        except _CorruptStressCache:
            _discard_cache_entry_locked(Path(cache_dir), cache_key)
            raise ValueError("model stress cache entry is corrupt") from None
        if cached is None:
            raise ValueError("model stress cache entry does not exist")
        return _materialize_cache_artifact_locked(
            cached,
            expected_row_ids=expected_ids,
            output_path=Path(output_path),
            cancellation_check=cancellation_check,
        )


def load_or_run_stress_scenario(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    output_path: Path,
    runner: Callable[[Path], ScenarioScoreArtifact],
    cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact:
    """Materialize one verified cache hit or produce it exactly once per key."""

    _require_sha256(cache_key, label="stress cache key")
    expected_ids = _validated_row_ids(expected_row_ids, label="stress row IDs")
    if len(expected_ids) == 0:
        raise ValueError("stress scenario cache requires rows")
    return _produce_or_materialize_cached_scenario(
        cache_dir=Path(cache_dir),
        cache_key=cache_key,
        category=category,
        expected_row_ids=expected_ids,
        output_path=Path(output_path),
        producer=runner,
        cancellation_check=cancellation_check,
    )


def load_oot_stress_context(
    *,
    sample_path: Path,
    baseline_score_path: Path,
    scoring_result: PmmlScoringResult,
    contract: ValidationInputContract,
    config: ValidationConfig,
    cancellation_check: Callable[[], None] | None = None,
) -> OotStressContext:
    frame = load_pmml_analysis_frame(
        sample_path=sample_path,
        score_path=baseline_score_path,
        contract=contract,
        scoring_result=scoring_result,
        cancellation_check=cancellation_check,
    )
    if config.split_col not in frame or config.target_col not in frame:
        raise ValueError("model stress config does not match PMML analysis frame")
    if config.score_col not in frame:
        raise ValueError("model stress config has no PMML score column")
    mask = frame[config.split_col].eq(config.split_values["oot"]).to_numpy(dtype=bool)
    row_ids = np.flatnonzero(mask).astype(np.int64)
    if len(row_ids) == 0:
        raise ValueError("OOT sample is required for model stress test")
    labels = frame.loc[mask, config.target_col].to_numpy(dtype=np.int8)
    baseline_scores = frame.loc[mask, config.score_col].to_numpy(dtype=np.float64)
    if len(labels) != len(row_ids) or not np.isfinite(baseline_scores).all():
        raise ValueError("invalid OOT baseline evidence for model stress test")
    return OotStressContext(row_ids, labels, baseline_scores)


def materialize_oot_pmml_inputs(
    *,
    contract: ValidationInputContract,
    sample_path: Path,
    oot_row_ids: np.ndarray,
    output_path: Path,
    chunk_size: int,
    cancellation_check: Callable[[], None] | None = None,
) -> OotInputArtifact:
    _require_positive_chunk_size(chunk_size)
    expected_ids = _validated_row_ids(oot_row_ids, label="OOT row IDs")
    if len(expected_ids) == 0:
        raise ValueError("OOT input materialization requires rows")
    manifest = contract.require_pmml_manifest()
    internal_row_id = _oot_row_id_field(manifest.raw_required_fields)
    transformations = _transformation_closure(
        manifest.raw_required_fields, contract.transformations
    )
    projection = required_transformation_inputs(
        manifest.raw_required_fields, transformations
    )
    output_path = Path(output_path)
    sample_path = Path(sample_path)
    if output_path.expanduser().resolve() == sample_path.expanduser().resolve():
        raise ValueError("OOT PMML input path must differ from validation sample")
    sample_identity = _file_identity(sample_path, label="validation sample")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.staging")
    writer: pq.ParquetWriter | None = None
    written = 0
    cursor = 0
    try:
        for chunk in iter_sample_chunks(
            sample_path,
            columns=projection,
            chunk_size=chunk_size,
            schema=contract.require_sample_schema(),
        ):
            raise_if_cancelled(cancellation_check)
            chunk_start = int(chunk.row_ids[0])
            chunk_end = int(chunk.row_ids[-1]) + 1
            start = cursor
            while cursor < len(expected_ids) and expected_ids[cursor] < chunk_end:
                cursor += 1
            selected_ids = expected_ids[start:cursor]
            if len(selected_ids) == 0:
                continue
            if int(selected_ids[0]) < chunk_start:
                raise ValueError("OOT row IDs are not aligned with validation sample")
            positions = selected_ids - chunk_start
            if np.any(positions < 0) or np.any(positions >= len(chunk.frame)):
                raise ValueError("OOT row IDs are not aligned with validation sample")
            selected = chunk.frame.iloc[positions].reset_index(drop=True)
            transformed = apply_confirmed_transformations(selected, transformations)
            model_inputs = transformed.loc[:, list(manifest.raw_required_fields)].copy()
            model_inputs.insert(0, internal_row_id, selected_ids)
            table = pa.Table.from_pandas(model_inputs, preserve_index=False)
            if table.schema.field(internal_row_id).type != pa.int64():
                table = table.set_column(
                    0, internal_row_id, pa.array(selected_ids, type=pa.int64())
                )
            if writer is None:
                writer = pq.ParquetWriter(staging, table.schema)
            elif writer.schema != table.schema:
                raise ValueError("OOT PMML input schema changed between chunks")
            writer.write_table(table)
            written += len(selected_ids)
        if writer is None:
            raise ValueError("OOT PMML input materialization produced no rows")
        writer.close()
        writer = None
        if cursor != len(expected_ids) or written != len(expected_ids):
            raise ValueError(
                "OOT PMML input row count mismatch: "
                f"{written} != {len(expected_ids)}"
            )
        _verify_oot_input_artifact(
            staging,
            expected_row_ids=expected_ids,
            raw_fields=manifest.raw_required_fields,
            internal_row_id=internal_row_id,
            cancellation_check=cancellation_check,
        )
        digest = sha256_file_cancellable(staging, cancellation_check)
        _require_file_identity(
            sample_path, sample_identity, label="validation sample"
        )
        prepared = OotInputArtifact(output_path, written, digest)
        # Publishing is the final fallible operation. Nothing after replacement
        # hashes, validates, checks cancellation or reads the file.
        os.replace(staging, output_path)
        return prepared
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def score_oot_category(
    *,
    category: str,
    raw_fields: tuple[str, ...],
    scorer: PmmlScorer,
    contract: ValidationInputContract,
    oot_input_path: Path,
    expected_row_ids: np.ndarray,
    output_path: Path,
    chunk_size: int,
    cancellation_check: Callable[[], None] | None = None,
) -> ScenarioScoreArtifact:
    _require_positive_chunk_size(chunk_size)
    if not isinstance(category, str) or not category.strip():
        raise ValueError("model stress category must be non-empty")
    expected_ids = _validated_row_ids(expected_row_ids, label="stress row IDs")
    manifest = contract.require_pmml_manifest()
    internal_row_id = _oot_row_id_field(manifest.raw_required_fields)
    if Path(output_path).expanduser().resolve() == Path(
        oot_input_path
    ).expanduser().resolve():
        raise ValueError("stress score path must differ from OOT PMML inputs")
    unknown = [field for field in raw_fields if field not in manifest.raw_required_fields]
    if not raw_fields or unknown:
        raise ValueError(
            f"stress category {category} has invalid raw input fields"
        )
    source_identity = _file_identity(Path(oot_input_path), label="OOT PMML inputs")
    writer = atomic_score_writer(Path(output_path))
    offset = 0
    try:
        parquet = pq.ParquetFile(oot_input_path)
        for batch in parquet.iter_batches(
            batch_size=chunk_size,
            columns=[internal_row_id, *manifest.raw_required_fields],
        ):
            raise_if_cancelled(cancellation_check)
            selected = batch.to_pandas()
            selected_ids = selected.pop(internal_row_id).to_numpy(dtype=np.int64)
            expected_batch = expected_ids[offset : offset + len(selected_ids)]
            if not np.array_equal(selected_ids, expected_batch):
                raise ValueError(f"stress row alignment failed for category {category}")
            for field in raw_fields:
                selected[field] = STRESS_MISSING_VALUE
            scores = scorer.score_chunk(
                selected.loc[:, list(manifest.raw_required_fields)]
            )
            if not isinstance(scores, pd.Series):
                raise ValueError("PMML stress scorer must return a Series")
            numeric = pd.to_numeric(scores, errors="coerce").to_numpy(
                dtype=np.float64
            )
            if len(numeric) != len(selected_ids) or not np.isfinite(numeric).all():
                raise ValueError(f"invalid PMML stress scores for category {category}")
            writer.write(selected_ids, numeric)
            offset += len(selected_ids)
        if offset != len(expected_ids):
            raise ValueError(
                f"stress row count mismatch for {category}: "
                f"{offset} != {len(expected_ids)}"
            )

        def prepare(digest: str) -> ScenarioScoreArtifact:
            _require_file_identity(
                Path(oot_input_path), source_identity, label="OOT PMML inputs"
            )
            return ScenarioScoreArtifact(category, Path(output_path), offset, digest)

        return writer.commit(
            cancellation_check=cancellation_check,
            prepare=prepare,
        )
    except BaseException:
        try:
            writer.rollback()
        except BaseException:
            pass
        raise


def load_aligned_scenario_scores(
    artifact: ScenarioScoreArtifact,
    expected_row_ids: np.ndarray,
    *,
    cancellation_check: Callable[[], None] | None = None,
    batch_size: int = 100_000,
) -> np.ndarray:
    _require_positive_chunk_size(batch_size)
    expected_ids = _validated_row_ids(expected_row_ids, label="stress row IDs")
    identity = _file_identity(artifact.path, label="PMML stress sidecar")
    if sha256_file_cancellable(artifact.path, cancellation_check) != artifact.sha256:
        raise ValueError(f"stress artifact hash mismatch for {artifact.category}")
    if artifact.row_count != len(expected_ids):
        raise ValueError(f"stress row count mismatch for category {artifact.category}")
    values = np.empty(len(expected_ids), dtype=np.float64)
    offset = 0
    try:
        parquet = pq.ParquetFile(artifact.path)
        if parquet.schema_arrow != SCORE_ARTIFACT_SCHEMA:
            raise ValueError(f"stress schema mismatch for category {artifact.category}")
        for batch in parquet.iter_batches(
            columns=["row_id", "pmml_score"], batch_size=batch_size
        ):
            raise_if_cancelled(cancellation_check)
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            expected_batch = expected_ids[offset : offset + batch.num_rows]
            if not np.array_equal(ids, expected_batch):
                raise ValueError(
                    f"stress row alignment failed for category {artifact.category}"
                )
            scores = batch.column(1).to_numpy(zero_copy_only=False)
            if not np.isfinite(scores).all():
                raise ValueError(
                    f"non-finite stress score for category {artifact.category}"
                )
            values[offset : offset + batch.num_rows] = scores
            offset += batch.num_rows
    except ValueError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise ValueError(f"invalid stress artifact for {artifact.category}") from exc
    if offset != len(expected_ids):
        raise ValueError(f"stress row count mismatch for category {artifact.category}")
    _require_file_identity(artifact.path, identity, label="PMML stress sidecar")
    raise_if_cancelled(cancellation_check)
    return values


def _stress_cache_entry_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / "stress-v1" / "entries" / cache_key


def _stress_cache_lock_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / "stress-v1" / "locks" / f"{cache_key}.lock"


def _discard_cache_entry_locked(cache_dir: Path, cache_key: str) -> None:
    entry = _stress_cache_entry_path(cache_dir, cache_key)
    try:
        if entry.is_symlink() or (entry.exists() and not entry.is_dir()):
            entry.unlink(missing_ok=True)
        elif entry.exists():
            shutil.rmtree(entry)
    except OSError as exc:
        raise ValueError("unable to remove corrupt model stress cache entry") from exc


def _cache_sha256(
    path: Path,
    cancellation_check: Callable[[], None] | None,
) -> str:
    cancellation: list[BaseException] = []

    def relay_cancellation() -> None:
        try:
            raise_if_cancelled(cancellation_check)
        except BaseException as exc:
            cancellation.append(exc)
            raise

    try:
        return sha256_file_cancellable(path, relay_cancellation)
    except BaseException as exc:
        if cancellation and exc is cancellation[-1]:
            raise
        if isinstance(exc, ValueError):
            raise _CorruptStressCache("unable to hash model stress cache score") from exc
        raise


def _read_cache_metadata(
    path: Path,
    *,
    cache_key: str,
    category: str,
) -> tuple[int, str]:
    try:
        identity = _file_identity(path, label="model stress cache metadata")
        if identity.size > _MAX_CACHE_METADATA_BYTES:
            raise _CorruptStressCache("model stress cache metadata is too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _require_file_identity(path, identity, label="model stress cache metadata")
    except _CorruptStressCache:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _CorruptStressCache("invalid model stress cache metadata") from exc
    if not isinstance(payload, dict) or set(payload) != _STRESS_CACHE_METADATA_FIELDS:
        raise _CorruptStressCache("model stress cache metadata fields mismatch")
    if payload["schema_version"] != STRESS_CACHE_SCHEMA:
        raise _CorruptStressCache("model stress cache schema mismatch")
    if payload["cache_key"] != cache_key or payload["category"] != category:
        raise _CorruptStressCache("model stress cache identity mismatch")
    row_count = payload["row_count"]
    if type(row_count) is not int or row_count <= 0:
        raise _CorruptStressCache("model stress cache row count is invalid")
    digest = payload["score_sha256"]
    if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
        raise _CorruptStressCache("model stress cache score digest is invalid")
    return row_count, digest


def _validate_scenario_score_file(
    path: Path,
    *,
    expected_row_ids: np.ndarray,
    expected_digest: str,
    cancellation_check: Callable[[], None] | None,
    batch_size: int = 100_000,
) -> str:
    try:
        identity = _file_identity(path, label="model stress cache score")
    except ValueError as exc:
        raise _CorruptStressCache("invalid model stress cache score") from exc
    digest = _cache_sha256(path, cancellation_check)
    if digest != expected_digest:
        raise _CorruptStressCache("model stress cache score hash mismatch")
    try:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != SCORE_ARTIFACT_SCHEMA:
            raise _CorruptStressCache("model stress cache score schema mismatch")
        if parquet.metadata.num_rows != len(expected_row_ids):
            raise _CorruptStressCache("model stress cache score row count mismatch")
        offset = 0
        for batch in parquet.iter_batches(
            columns=["row_id", "pmml_score"], batch_size=batch_size
        ):
            raise_if_cancelled(cancellation_check)
            row_column = batch.column(0)
            score_column = batch.column(1)
            if row_column.null_count or score_column.null_count:
                raise _CorruptStressCache("model stress cache score contains nulls")
            row_ids = row_column.to_numpy(zero_copy_only=False)
            scores = score_column.to_numpy(zero_copy_only=False)
            expected = expected_row_ids[offset : offset + batch.num_rows]
            if not np.array_equal(row_ids, expected):
                raise _CorruptStressCache("model stress cache row alignment mismatch")
            if not np.isfinite(scores).all():
                raise _CorruptStressCache("model stress cache score is non-finite")
            offset += batch.num_rows
        if offset != len(expected_row_ids):
            raise _CorruptStressCache("model stress cache score row count mismatch")
    except _CorruptStressCache:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise _CorruptStressCache("invalid model stress cache score") from exc
    raise_if_cancelled(cancellation_check)
    try:
        _require_file_identity(path, identity, label="model stress cache score")
    except ValueError as exc:
        raise _CorruptStressCache("model stress cache score changed") from exc
    return digest


def _load_valid_cache_entry_path(
    entry: Path,
    *,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact:
    if entry.is_symlink() or not entry.is_dir():
        raise _CorruptStressCache("model stress cache entry is not a directory")
    try:
        names = {item.name for item in entry.iterdir()}
    except OSError as exc:
        raise _CorruptStressCache("unable to inspect model stress cache entry") from exc
    if names != {_STRESS_CACHE_METADATA, _STRESS_CACHE_SCORES}:
        raise _CorruptStressCache("model stress cache entry files mismatch")
    row_count, digest = _read_cache_metadata(
        entry / _STRESS_CACHE_METADATA,
        cache_key=cache_key,
        category=category,
    )
    if row_count != len(expected_row_ids):
        raise _CorruptStressCache("model stress cache metadata row count mismatch")
    score_path = entry / _STRESS_CACHE_SCORES
    _validate_scenario_score_file(
        score_path,
        expected_row_ids=expected_row_ids,
        expected_digest=digest,
        cancellation_check=cancellation_check,
    )
    return ScenarioScoreArtifact(category, score_path, row_count, digest)


def _load_valid_cache_entry_locked(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact | None:
    entry = _stress_cache_entry_path(cache_dir, cache_key)
    if not entry.exists() and not entry.is_symlink():
        return None
    return _load_valid_cache_entry_path(
        entry,
        cache_key=cache_key,
        category=category,
        expected_row_ids=expected_row_ids,
        cancellation_check=cancellation_check,
    )


def _write_cache_metadata(
    path: Path,
    *,
    cache_key: str,
    category: str,
    row_count: int,
    score_sha256: str,
) -> None:
    payload = {
        "schema_version": STRESS_CACHE_SCHEMA,
        "cache_key": cache_key,
        "category": category,
        "row_count": row_count,
        "score_sha256": score_sha256,
    }
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ValueError("unable to write model stress cache metadata") from exc


def _publish_cache_entry_locked(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    producer: ScenarioScoreArtifact,
    expected_row_ids: np.ndarray,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact:
    if (
        producer.category != category
        or type(producer.row_count) is not int
        or producer.row_count != len(expected_row_ids)
        or _SHA256_PATTERN.fullmatch(producer.sha256) is None
    ):
        raise ValueError("model stress cache producer returned invalid evidence")
    try:
        _validate_scenario_score_file(
            producer.path,
            expected_row_ids=expected_row_ids,
            expected_digest=producer.sha256,
            cancellation_check=cancellation_check,
        )
    except _CorruptStressCache as exc:
        raise ValueError("model stress cache producer artifact is invalid") from exc

    entry = _stress_cache_entry_path(cache_dir, cache_key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    staging = entry.parent / f".{cache_key}.{uuid4().hex}.staging"
    try:
        staging.mkdir()
        copied = staging / _STRESS_CACHE_SCORES
        copy_file_cancellable(
            producer.path,
            copied,
            cancellation_check=cancellation_check,
        )
        copied_digest = sha256_file_cancellable(copied, cancellation_check)
        if copied_digest != producer.sha256:
            raise ValueError("model stress cache copy hash mismatch")
        _write_cache_metadata(
            staging / _STRESS_CACHE_METADATA,
            cache_key=cache_key,
            category=category,
            row_count=producer.row_count,
            score_sha256=copied_digest,
        )
        prepared = _load_valid_cache_entry_path(
            staging,
            cache_key=cache_key,
            category=category,
            expected_row_ids=expected_row_ids,
            cancellation_check=cancellation_check,
        )
        if entry.exists() or entry.is_symlink():
            raise ValueError("model stress cache entry appeared during production")
        published = ScenarioScoreArtifact(
            prepared.category,
            entry / _STRESS_CACHE_SCORES,
            prepared.row_count,
            prepared.sha256,
        )
        # The complete cache unit has been validated. Directory replacement is
        # its publication point; readers use the same per-key lock. Nothing
        # fallible is performed after this replacement.
        os.replace(staging, entry)
        return published
    except BaseException:
        try:
            if staging.exists():
                shutil.rmtree(staging)
        except OSError:
            pass
        raise


def _materialize_cache_artifact_locked(
    cached: ScenarioScoreArtifact,
    *,
    expected_row_ids: np.ndarray,
    output_path: Path,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact:
    output_path = Path(output_path)
    if output_path.expanduser().resolve() == cached.path.expanduser().resolve():
        raise ValueError("task stress artifact must differ from cache artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_identity = _file_identity(cached.path, label="model stress cache score")
    staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.staging")
    try:
        raise_if_cancelled(cancellation_check)
        try:
            os.link(cached.path, staging)
            linked_identity = _file_identity(
                cached.path, label="model stress cache score"
            )
            # Adding a hard link legitimately updates ctime/link count. Every
            # content-bearing identity field must still match the pre-link file.
            if (
                linked_identity.device != source_identity.device
                or linked_identity.inode != source_identity.inode
                or linked_identity.size != source_identity.size
                or linked_identity.modified_ns != source_identity.modified_ns
            ):
                raise ValueError("model stress cache score changed during hardlink")
            source_identity = linked_identity
        except OSError:
            copy_file_cancellable(
                cached.path,
                staging,
                cancellation_check=cancellation_check,
            )
        _validate_scenario_score_file(
            staging,
            expected_row_ids=expected_row_ids,
            expected_digest=cached.sha256,
            cancellation_check=cancellation_check,
        )
        _require_file_identity(
            cached.path, source_identity, label="model stress cache score"
        )
        prepared = ScenarioScoreArtifact(
            cached.category,
            output_path,
            cached.row_count,
            cached.sha256,
        )
        # No validation, hashing, cancellation or I/O occurs after publication.
        os.replace(staging, output_path)
        return prepared
    except BaseException:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _try_materialize_cached_scenario(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    output_path: Path,
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact | None:
    with cancellable_file_lock(
        _stress_cache_lock_path(cache_dir, cache_key), cancellation_check
    ):
        try:
            cached = _load_valid_cache_entry_locked(
                cache_dir=cache_dir,
                cache_key=cache_key,
                category=category,
                expected_row_ids=expected_row_ids,
                cancellation_check=cancellation_check,
            )
        except _CorruptStressCache:
            _discard_cache_entry_locked(cache_dir, cache_key)
            return None
        if cached is None:
            return None
        return _materialize_cache_artifact_locked(
            cached,
            expected_row_ids=expected_row_ids,
            output_path=output_path,
            cancellation_check=cancellation_check,
        )


def _produce_or_materialize_cached_scenario(
    *,
    cache_dir: Path,
    cache_key: str,
    category: str,
    expected_row_ids: np.ndarray,
    output_path: Path,
    producer: Callable[[Path], ScenarioScoreArtifact],
    cancellation_check: Callable[[], None] | None,
) -> ScenarioScoreArtifact:
    with cancellable_file_lock(
        _stress_cache_lock_path(cache_dir, cache_key), cancellation_check
    ):
        try:
            cached = _load_valid_cache_entry_locked(
                cache_dir=cache_dir,
                cache_key=cache_key,
                category=category,
                expected_row_ids=expected_row_ids,
                cancellation_check=cancellation_check,
            )
        except _CorruptStressCache:
            _discard_cache_entry_locked(cache_dir, cache_key)
            cached = None
        if cached is None:
            producer_path = output_path.with_name(
                f".{output_path.name}.{uuid4().hex}.producer"
            )
            produced: ScenarioScoreArtifact | None = None
            try:
                produced = producer(producer_path)
                cached = _publish_cache_entry_locked(
                    cache_dir=cache_dir,
                    cache_key=cache_key,
                    category=category,
                    producer=produced,
                    expected_row_ids=expected_row_ids,
                    cancellation_check=cancellation_check,
                )
            finally:
                try:
                    producer_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return _materialize_cache_artifact_locked(
            cached,
            expected_row_ids=expected_row_ids,
            output_path=output_path,
            cancellation_check=cancellation_check,
        )


def run_pmml_stress(
    *,
    contract: ValidationInputContract,
    config: ValidationConfig,
    sample_path: Path,
    baseline_score_path: Path,
    scoring_result: PmmlScoringResult,
    scenario_dir: Path,
    feature_categories: Mapping[str, tuple[str, ...]],
    scorer: PmmlScorer | None = None,
    scorer_factory: Callable[[], PmmlScorer] | None = None,
    chunk_size: int,
    cancellation_check: Callable[[], None] | None = None,
    category_source_counts: Mapping[str, int] | None = None,
    baseline_cache_key: str | None = None,
    cache_dir: Path | None = None,
) -> StressTestResult:
    if contract.status != "ready":
        raise ValueError("validation input contract is not ready for model stress test")
    if scorer is not None and scorer_factory is not None:
        raise ValueError("provide either a PMML scorer or scorer_factory, not both")
    _require_positive_chunk_size(chunk_size)
    categories = _validated_feature_categories(
        feature_categories, contract.require_pmml_manifest().raw_required_fields
    )
    cache_enabled = baseline_cache_key is not None or cache_dir is not None
    if cache_enabled and (baseline_cache_key is None or cache_dir is None):
        raise ValueError(
            "baseline_cache_key and cache_dir must be provided together"
        )
    if baseline_cache_key is not None:
        _require_sha256(baseline_cache_key, label="baseline cache key")
        if baseline_cache_key != scoring_result.cache_key:
            raise ValueError("baseline cache key does not match PMML scoring result")
    sample_identity = _file_identity(Path(sample_path), label="validation sample")
    context = load_oot_stress_context(
        sample_path=sample_path,
        baseline_score_path=baseline_score_path,
        scoring_result=scoring_result,
        contract=contract,
        config=config,
        cancellation_check=cancellation_check,
    )
    raise_if_cancelled(cancellation_check)
    edges = equal_frequency_bin_edges(context.baseline_scores, config.bin_count)
    baseline_frame = pd.DataFrame(
        {"__target__": context.labels, "__score__": context.baseline_scores}
    )
    baseline_ks = float(compute_ks(context.baseline_scores, context.labels))
    baseline = StressBaseline(
        ks=baseline_ks,
        sample_count=len(context.row_ids),
        bin_table=bin_table(
            baseline_frame,
            edges,
            score_col="__score__",
            target_col="__target__",
        ),
    )
    scenario_dir = Path(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, ScenarioScoreArtifact] = {}
    missing: list[tuple[int, str, tuple[str, ...], str | None]] = []
    for index, (category, raw_fields) in enumerate(categories, start=1):
        output_path = scenario_dir / f"category_{index:03d}.parquet"
        scenario_key: str | None = None
        if baseline_cache_key is not None and cache_dir is not None:
            scenario_key = stress_cache_key(
                baseline_cache_key=baseline_cache_key,
                category=category,
                raw_fields=raw_fields,
                sentinel=STRESS_MISSING_VALUE,
                chunk_size=chunk_size,
            )
            cached = _try_materialize_cached_scenario(
                cache_dir=Path(cache_dir),
                cache_key=scenario_key,
                category=category,
                expected_row_ids=context.row_ids,
                output_path=output_path,
                cancellation_check=cancellation_check,
            )
            if cached is not None:
                artifacts[category] = cached
                continue
        missing.append((index, category, raw_fields, scenario_key))

    oot_inputs: OotInputArtifact | None = None
    if missing:
        active_scorer = scorer
        if active_scorer is None:
            if scorer_factory is None:
                raise ValueError("PMML scorer is required for uncached stress scenarios")
            active_scorer = scorer_factory()
        oot_inputs = materialize_oot_pmml_inputs(
            contract=contract,
            sample_path=sample_path,
            oot_row_ids=context.row_ids,
            output_path=scenario_dir / "oot_pmml_inputs.parquet",
            chunk_size=chunk_size,
            cancellation_check=cancellation_check,
        )
        for index, category, raw_fields, scenario_key in missing:
            raise_if_cancelled(cancellation_check)
            output_path = scenario_dir / f"category_{index:03d}.parquet"

            def produce(producer_path: Path) -> ScenarioScoreArtifact:
                assert oot_inputs is not None
                return score_oot_category(
                    category=category,
                    raw_fields=raw_fields,
                    scorer=active_scorer,
                    contract=contract,
                    oot_input_path=oot_inputs.path,
                    expected_row_ids=context.row_ids,
                    output_path=producer_path,
                    chunk_size=chunk_size,
                    cancellation_check=cancellation_check,
                )

            if scenario_key is not None and cache_dir is not None:
                artifact = load_or_run_stress_scenario(
                    cache_dir=Path(cache_dir),
                    cache_key=scenario_key,
                    category=category,
                    expected_row_ids=context.row_ids,
                    output_path=output_path,
                    runner=produce,
                    cancellation_check=cancellation_check,
                )
            else:
                artifact = produce(output_path)
            artifacts[category] = artifact

    _require_file_identity(
        Path(sample_path), sample_identity, label="validation sample"
    )
    baseline_distribution = bin_distribution(context.baseline_scores, edges)
    per_category: list[StressCategoryResult] = []
    for category, raw_fields in categories:
        raise_if_cancelled(cancellation_check)
        artifact = artifacts[category]
        scenario_scores = load_aligned_scenario_scores(
            artifact,
            context.row_ids,
            cancellation_check=cancellation_check,
        )
        ks_after = float(compute_ks(scenario_scores, context.labels))
        scenario_frame = pd.DataFrame(
            {"__target__": context.labels, "__score__": scenario_scores}
        )
        per_category.append(
            StressCategoryResult(
                category=category,
                dropped_features=list(raw_fields),
                ks_after=ks_after,
                ks_delta=ks_after - baseline_ks,
                psi_vs_baseline=float(
                    compute_psi(
                        baseline_distribution,
                        bin_distribution(scenario_scores, edges),
                    )
                ),
                bin_table=bin_table(
                    scenario_frame,
                    edges,
                    score_col="__score__",
                    target_col="__target__",
                ),
                error=None,
                status="completed",
            )
        )
        del scenario_scores
        raise_if_cancelled(cancellation_check)
    result = StressTestResult(
        baseline=baseline,
        per_category=per_category,
        status="completed",
        unclassified_features=[],
        category_source_counts=dict(category_source_counts or {}),
    )
    return require_complete_stress_result(result)


def _validated_feature_categories(
    feature_categories: Mapping[str, tuple[str, ...]],
    raw_required_fields: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(feature_categories, Mapping) or not feature_categories:
        raise ValueError("model stress test requires at least one feature category")
    allowed = set(raw_required_fields)
    normalized: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for category, fields in feature_categories.items():
        if not isinstance(category, str) or not category.strip() or category in seen:
            raise ValueError("model stress categories must be unique non-empty strings")
        if not isinstance(fields, tuple) or not fields:
            raise ValueError(f"stress category {category} has no raw input fields")
        if len(set(fields)) != len(fields) or any(
            not isinstance(field, str) or field not in allowed for field in fields
        ):
            raise ValueError(f"stress category {category} has invalid raw input fields")
        seen.add(category)
        normalized.append((category, fields))
    return tuple(normalized)


def _transformation_closure(
    output_fields: Sequence[str],
    specs: Sequence[TransformationSpec],
) -> tuple[TransformationSpec, ...]:
    ordered = topologically_sorted_transformations(specs)
    by_output = {spec.output_field: spec for spec in ordered}
    needed: set[str] = set()
    stack = list(reversed(tuple(output_fields)))
    while stack:
        field = stack.pop()
        spec = by_output.get(field)
        if spec is None or field in needed:
            continue
        needed.add(field)
        stack.extend(reversed(spec.input_fields))
    return tuple(spec for spec in ordered if spec.output_field in needed)


def _verify_oot_input_artifact(
    path: Path,
    *,
    expected_row_ids: np.ndarray,
    raw_fields: Sequence[str],
    internal_row_id: str,
    cancellation_check: Callable[[], None] | None,
) -> None:
    try:
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow.names != [internal_row_id, *raw_fields]:
            raise ValueError("OOT PMML input schema mismatch")
        if parquet.schema_arrow.field(internal_row_id).type != pa.int64():
            raise ValueError("OOT PMML input row_id type mismatch")
        offset = 0
        for batch in parquet.iter_batches(
            columns=[internal_row_id], batch_size=100_000
        ):
            raise_if_cancelled(cancellation_check)
            ids = batch.column(0).to_numpy(zero_copy_only=False)
            expected = expected_row_ids[offset : offset + batch.num_rows]
            if not np.array_equal(ids, expected):
                raise ValueError("OOT PMML input row alignment mismatch")
            offset += batch.num_rows
        if offset != len(expected_row_ids):
            raise ValueError("OOT PMML input row count mismatch")
    except ValueError:
        raise
    except (OSError, pa.ArrowException) as exc:
        raise ValueError("invalid OOT PMML input artifact") from exc


def _validated_row_ids(values: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must be a one-dimensional integer array")
    result = result.astype(np.int64, copy=False)
    if len(result) and (
        int(result[0]) < 0 or np.any(result[1:] <= result[:-1])
    ):
        raise ValueError(f"{label} must be strictly increasing and non-negative")
    return result


def _oot_row_id_field(raw_fields: Sequence[str]) -> str:
    candidate = "__marvis_source_row_id__"
    used = set(raw_fields)
    while candidate in used:
        candidate += "_"
    return candidate


def _file_identity(path: Path, *, label: str) -> _FileIdentity:
    try:
        current = path.stat()
    except OSError as exc:
        raise ValueError(f"unable to inspect {label}") from exc
    if not stat.S_ISREG(current.st_mode):
        raise ValueError(f"{label} must be a regular file")
    return _FileIdentity(
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )


def _require_file_identity(path: Path, expected: _FileIdentity, *, label: str) -> None:
    if _file_identity(path, label=label) != expected:
        raise ValueError(f"{label} changed during model stress test")


def _require_positive_chunk_size(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model stress chunk size must be a positive integer")


def _require_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
