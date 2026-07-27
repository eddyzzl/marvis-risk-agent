"""Pure, bounded full-tree writeback for committed automatic rule trees.

This module is deliberately below the governed Tool boundary.  It owns strict
tree/asset validation, complete feature-domain validation, canonical weighted
tree replay, and deterministic Parquet projection.  It does not read or write a
registry, database, task artifact, Strategy Pool, action, or adoption decision.

The caller supplies a source Parquet file and a fresh staging path.  Every
source column and row is copied in source order, then two non-null UTF-8 columns
are appended: the canonical leaf id and its canonical rule id.  Routing always
delegates to :func:`apply_weighted_rule_tree`; the leaf-to-rule relationship is
derived only from the strictly validated tree rules.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, BinaryIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from marvis.feature.weighted_rule_tree import (
    WEIGHTED_RULE_TREE_SCHEMA_VERSION,
    WeightedRuleTreeError,
    apply_weighted_rule_tree,
    validate_weighted_rule_tree,
)
from marvis.files import sha256_file
from marvis.packs.strategy.automatic_tree_asset import (
    AUTOMATIC_TREE_ASSET_SCHEMA_VERSION,
    AutomaticTreeAssetError,
    validate_automatic_tree_asset,
)
from marvis.packs.strategy.codegen import (
    AutomaticTreeCodegenError,
    validate_automatic_tree_duckdb_input_frame,
)
from marvis.packs.strategy.errors import StrategyError


AUTOMATIC_TREE_APPLY_SCHEMA_VERSION = "strategy.automatic-tree-apply-result.v1"
AUTOMATIC_TREE_APPLY_PRODUCER_VERSION = "strategy.automatic-tree-apply/1"
AUTOMATIC_TREE_APPLY_WRITER_CONTRACT = "strategy.automatic-tree-apply-parquet-writer/1"

_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_BATCH_ROWS = 8_192
_MAX_DECODED_BATCH_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_ROWS = 1_000_000
_MAX_FEATURES = 50
_MAX_FEATURE_CELLS = 50_000_000
_MAX_TREE_NODES = 511
_MAX_TREE_LEAVES = 256
_MAX_SOURCE_COLUMNS = 500
_PARQUET_WRITE_BATCH_ROWS = 1_024
_PARQUET_DICTIONARY_PAGE_BYTES = 1_048_576
_PARQUET_COMPRESSION_LEVEL = 3


class AutomaticTreeApplyError(StrategyError):
    """A full automatic-tree writeback failed closed."""


class AutomaticTreeApplyBudgetError(AutomaticTreeApplyError):
    """A non-overridable writeback resource limit was exceeded."""

    def __init__(self, *, dimension: str, actual: int, limit: int) -> None:
        self.dimension = str(dimension)
        self.actual = int(actual)
        self.limit = int(limit)
        super().__init__(
            f"automatic-tree apply {self.dimension} budget exceeded: "
            f"actual={self.actual}, limit={self.limit}"
        )

    def to_detail(self) -> dict[str, object]:
        return {
            "kind": "automatic_tree_apply_budget_exceeded",
            "dimension": self.dimension,
            "actual": self.actual,
            "limit": self.limit,
        }


@dataclass
class _SourceSnapshot:
    """Private immutable copy bound to one retained regular-file descriptor."""

    source_path: Path
    content_hash: str
    source_fd: int
    source_stat: os.stat_result
    snapshot_file: BinaryIO
    snapshot_stat: os.stat_result

    @classmethod
    def create(
        cls,
        source_path: Path,
        *,
        directory: Path,
    ) -> _SourceSnapshot:
        source_fd, source_stat = _open_verified_regular_file(source_path)
        snapshot_file: BinaryIO | None = None
        try:
            snapshot_file = tempfile.TemporaryFile(
                mode="w+b",
                dir=directory,
            )
            snapshot_fd = snapshot_file.fileno()
            content_hash, copied_bytes = _copy_and_hash_fd(source_fd, snapshot_fd)
            os.fsync(snapshot_fd)
            source_after_copy = os.fstat(source_fd)
            if _stable_stat(source_after_copy) != _stable_stat(source_stat):
                raise AutomaticTreeApplyError(
                    "source Parquet changed while its private snapshot was created"
                )
            snapshot_stat = os.fstat(snapshot_fd)
            if int(snapshot_stat.st_size) != copied_bytes:
                raise AutomaticTreeApplyError(
                    "private source snapshot size does not match copied bytes"
                )
            return cls(
                source_path=source_path,
                content_hash=content_hash,
                source_fd=source_fd,
                source_stat=source_stat,
                snapshot_file=snapshot_file,
                snapshot_stat=snapshot_stat,
            )
        except Exception:
            if snapshot_file is not None:
                snapshot_file.close()
            os.close(source_fd)
            raise

    @property
    def snapshot_fd(self) -> int:
        return self.snapshot_file.fileno()

    def verify_unchanged(self) -> None:
        """Reject source/snapshot content, identity, symlink, or ABA changes."""

        _verify_retained_fd(
            self.snapshot_fd,
            expected_stat=self.snapshot_stat,
            expected_hash=self.content_hash,
            label="private source snapshot",
        )
        _verify_retained_fd(
            self.source_fd,
            expected_stat=self.source_stat,
            expected_hash=self.content_hash,
            label="source Parquet",
        )
        path_fd = -1
        try:
            path_fd, path_stat = _open_verified_regular_file(
                self.source_path,
                expected_stat=self.source_stat,
            )
            _verify_retained_fd(
                path_fd,
                expected_stat=path_stat,
                expected_hash=self.content_hash,
                label="source Parquet path",
            )
            final_path_stat = os.lstat(self.source_path)
            if _stable_stat(final_path_stat) != _stable_stat(self.source_stat):
                raise AutomaticTreeApplyError(
                    "source Parquet path changed or was replaced during writeback"
                )
        except OSError as exc:
            raise AutomaticTreeApplyError(
                "source Parquet path changed or was replaced during writeback"
            ) from exc
        finally:
            if path_fd >= 0:
                os.close(path_fd)
        if _stable_stat(os.fstat(self.source_fd)) != _stable_stat(self.source_stat):
            raise AutomaticTreeApplyError(
                "source Parquet changed or was replaced during writeback"
            )

    def open_reader(self) -> BinaryIO:
        """Open a reader duplicated from the retained immutable snapshot fd."""

        reader_fd = os.dup(self.snapshot_fd)
        try:
            os.lseek(reader_fd, 0, os.SEEK_SET)
            return os.fdopen(reader_fd, "rb", closefd=True)
        except Exception:
            os.close(reader_fd)
            raise

    def close(self) -> None:
        try:
            self.snapshot_file.close()
        except OSError:
            pass
        try:
            os.close(self.source_fd)
        except OSError:
            pass


@dataclass(frozen=True)
class AutomaticTreeApplyResult:
    """Path-free evidence for one deterministic full-tree Parquet result."""

    schema_version: str
    producer_version: str
    result_id: str
    source_content_hash: str
    source_row_count: int
    output_schema: dict[str, Any]
    output_columns: dict[str, str]
    leaf_distribution: tuple[dict[str, Any], ...]
    tree_result_hash: str
    asset_id: str | None
    asset_hash: str | None
    writer_contract: dict[str, Any]
    output_content_hash: str
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe evidence projection with no local paths."""

        body = {
            "schema_version": self.schema_version,
            "producer_version": self.producer_version,
            "source": {
                "content_hash": self.source_content_hash,
                "row_count": self.source_row_count,
            },
            "tree": {
                "result_hash": self.tree_result_hash,
                "asset_id": self.asset_id,
                "asset_hash": self.asset_hash,
            },
            "output": {
                "content_hash": self.output_content_hash,
                "row_count": self.source_row_count,
                "schema": _detach_json(self.output_schema),
                "columns": dict(self.output_columns),
                "leaf_distribution": [dict(item) for item in self.leaf_distribution],
            },
            "writer": dict(self.writer_contract),
        }
        expected_hash = _canonical_hash(body)
        expected_id = f"automatic-tree-apply-{expected_hash[:32]}"
        if self.result_hash != expected_hash or self.result_id != expected_id:
            raise AutomaticTreeApplyError(
                "automatic-tree apply result evidence was mutated after creation"
            )
        return {
            **body,
            "result_id": self.result_id,
            "result_hash": self.result_hash,
        }


def apply_automatic_tree_to_parquet(
    tree_or_asset: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    *,
    leaf_id_column: str,
    rule_id_column: str,
) -> AutomaticTreeApplyResult:
    """Apply every strict tree leaf and write a deterministic derived Parquet.

    ``output_path`` is a caller-owned staging destination.  It must differ from
    the source and must not exist.  The function never chooses a temporary path,
    registers the file, or records governance state; a Tool can do those things
    after this result and the staged bytes have both been verified.
    """

    source, output = _validated_paths(source_path, output_path)
    _validated_tree_and_asset_identity(tree_or_asset)
    _validated_output_columns(
        leaf_id_column=leaf_id_column,
        rule_id_column=rule_id_column,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _SourceSnapshot.create(
        source,
        directory=output.parent,
    )
    result: AutomaticTreeApplyResult | None = None
    try:
        with snapshot.open_reader() as snapshot_reader:
            result = _apply_automatic_tree_snapshot_to_parquet(
                tree_or_asset,
                snapshot_reader,
                output,
                source_content_hash=snapshot.content_hash,
                leaf_id_column=leaf_id_column,
                rule_id_column=rule_id_column,
            )
        snapshot.verify_unchanged()
        return result
    except Exception:
        if result is not None:
            output.unlink(missing_ok=True)
        raise
    finally:
        snapshot.close()


def _apply_automatic_tree_snapshot_to_parquet(
    tree_or_asset: Mapping[str, Any],
    source_stream: BinaryIO,
    output_path: Path,
    *,
    source_content_hash: str,
    leaf_id_column: str,
    rule_id_column: str,
) -> AutomaticTreeApplyResult:
    """Apply a tree only to one private, descriptor-bound source snapshot."""

    tree, asset_id, asset_hash = _validated_tree_and_asset_identity(tree_or_asset)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise AutomaticTreeApplyError("output_path must not already exist")
    output_columns = _validated_output_columns(
        leaf_id_column=leaf_id_column,
        rule_id_column=rule_id_column,
    )
    parquet_file = _open_source_parquet(source_stream)
    source_schema = parquet_file.schema_arrow
    source_row_count = int(parquet_file.metadata.num_rows)
    features = tuple(tree["training"]["feature_order"])
    _enforce_static_budgets(
        source_rows=source_row_count,
        source_columns=len(source_schema),
        feature_count=len(features),
        node_count=int(tree["tree"]["node_count"]),
        leaf_count=int(tree["tree"]["leaf_count"]),
    )
    feature_indexes = _validate_source_schema(
        source_schema,
        features=features,
        output_columns=output_columns,
    )
    # Fail schema/type/casefold errors before the output file is opened.  Value
    # validation is repeated on every decoded batch below, covering every cell.
    empty_feature_frame = _empty_feature_frame(
        source_schema,
        feature_indexes=feature_indexes,
    )
    _preflight_feature_frame(empty_feature_frame, tree, features=features)

    result_schema = _result_arrow_schema(
        source_schema,
        leaf_id_column=leaf_id_column,
        rule_id_column=rule_id_column,
    )
    rule_by_leaf = {
        str(rule["leaf_id"]): str(rule["rule_id"]) for rule in tree["rules"]
    }
    leaf_counts = Counter({leaf_id: 0 for leaf_id in rule_by_leaf})
    rows_written = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output.open("xb") as output_stream:
            with pq.ParquetWriter(
                output_stream,
                result_schema,
                version="2.6",
                use_dictionary=True,
                compression="zstd",
                write_statistics=True,
                use_deprecated_int96_timestamps=False,
                compression_level=_PARQUET_COMPRESSION_LEVEL,
                use_byte_stream_split=False,
                data_page_version="1.0",
                use_compliant_nested_type=True,
                write_batch_size=_PARQUET_WRITE_BATCH_ROWS,
                dictionary_pagesize_limit=_PARQUET_DICTIONARY_PAGE_BYTES,
                store_schema=True,
                write_page_index=False,
                write_page_checksum=False,
            ) as writer:
                for batch in parquet_file.iter_batches(
                    batch_size=_BATCH_ROWS,
                    use_threads=False,
                ):
                    _enforce_budget(
                        "decoded_batch_bytes",
                        int(batch.nbytes),
                        _MAX_DECODED_BATCH_BYTES,
                    )
                    feature_frame = batch.select(feature_indexes).to_pandas(
                        types_mapper=pd.ArrowDtype,
                    )
                    _preflight_feature_frame(feature_frame, tree, features=features)
                    leaf_ids = _canonical_leaf_ids(feature_frame, tree)
                    try:
                        rule_ids = [rule_by_leaf[leaf_id] for leaf_id in leaf_ids]
                    except KeyError as exc:  # defensive: strict tree replay owns ids
                        raise AutomaticTreeApplyError(
                            "canonical weighted-tree replay returned an unknown leaf id"
                        ) from exc
                    leaf_counts.update(leaf_ids)
                    output_batch = pa.RecordBatch.from_arrays(
                        [
                            *batch.columns,
                            pa.array(leaf_ids, type=pa.string()),
                            pa.array(rule_ids, type=pa.string()),
                        ],
                        schema=result_schema,
                    )
                    writer.write_batch(output_batch, row_group_size=_BATCH_ROWS)
                    rows_written += int(batch.num_rows)
    except FileExistsError as exc:
        raise AutomaticTreeApplyError("output_path must not already exist") from exc
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        parquet_file.close()

    try:
        if rows_written != source_row_count:
            raise AutomaticTreeApplyError(
                "source Parquet row metadata changed during full-tree writeback"
            )
        output_content_hash = sha256_file(output)
        emitted_schema = _verify_emitted_parquet(
            output,
            expected_schema=result_schema,
            expected_rows=source_row_count,
        )
        if sha256_file(output) != output_content_hash:
            raise AutomaticTreeApplyError(
                "automatic-tree output changed during result verification"
            )
        writer_contract = _writer_contract()
        leaf_distribution = tuple(
            {
                "leaf_id": str(rule["leaf_id"]),
                "rule_id": str(rule["rule_id"]),
                "row_count": int(leaf_counts[str(rule["leaf_id"])]),
            }
            for rule in tree["rules"]
        )
        if sum(item["row_count"] for item in leaf_distribution) != source_row_count:
            raise AutomaticTreeApplyError(
                "leaf distribution does not conserve the source row count"
            )
        output_schema = _schema_projection(emitted_schema)
        evidence_body = {
            "schema_version": AUTOMATIC_TREE_APPLY_SCHEMA_VERSION,
            "producer_version": AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
            "source": {
                "content_hash": source_content_hash,
                "row_count": source_row_count,
            },
            "tree": {
                "result_hash": tree["result_hash"],
                "asset_id": asset_id,
                "asset_hash": asset_hash,
            },
            "output": {
                "content_hash": output_content_hash,
                "row_count": source_row_count,
                "schema": output_schema,
                "columns": output_columns,
                "leaf_distribution": list(leaf_distribution),
            },
            "writer": writer_contract,
        }
        result_hash = _canonical_hash(evidence_body)
        return AutomaticTreeApplyResult(
            schema_version=AUTOMATIC_TREE_APPLY_SCHEMA_VERSION,
            producer_version=AUTOMATIC_TREE_APPLY_PRODUCER_VERSION,
            result_id=f"automatic-tree-apply-{result_hash[:32]}",
            source_content_hash=source_content_hash,
            source_row_count=source_row_count,
            output_schema=output_schema,
            output_columns=output_columns,
            leaf_distribution=leaf_distribution,
            tree_result_hash=str(tree["result_hash"]),
            asset_id=asset_id,
            asset_hash=asset_hash,
            writer_contract=writer_contract,
            output_content_hash=output_content_hash,
            result_hash=result_hash,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _validated_tree_and_asset_identity(
    tree_or_asset: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None, str | None]:
    if not isinstance(tree_or_asset, Mapping):
        raise AutomaticTreeApplyError("tree_or_asset must be an object")
    schema_version = tree_or_asset.get("schema_version")
    try:
        if schema_version == WEIGHTED_RULE_TREE_SCHEMA_VERSION:
            return validate_weighted_rule_tree(tree_or_asset), None, None
        if schema_version == AUTOMATIC_TREE_ASSET_SCHEMA_VERSION:
            asset = validate_automatic_tree_asset(tree_or_asset)
            return (
                asset["tree_result"],
                str(asset["asset_id"]),
                str(asset["asset_hash"]),
            )
    except (
        WeightedRuleTreeError,
        AutomaticTreeAssetError,
        TypeError,
        ValueError,
    ) as exc:
        raise AutomaticTreeApplyError(
            f"automatic tree asset/tree failed strict validation: {exc}"
        ) from exc
    raise AutomaticTreeApplyError(
        "tree_or_asset schema_version must identify a strict weighted rule tree "
        "or automatic tree asset"
    )


def _validated_paths(source_path: Path, output_path: Path) -> tuple[Path, Path]:
    try:
        source = Path(os.path.abspath(os.fspath(source_path)))
        output = Path(os.path.abspath(os.fspath(output_path)))
    except (TypeError, ValueError) as exc:
        raise AutomaticTreeApplyError(
            "source_path and output_path must be paths"
        ) from exc
    try:
        source_stat = os.lstat(source)
    except OSError as exc:
        raise AutomaticTreeApplyError("source_path is not a readable file") from exc
    if stat.S_ISLNK(source_stat.st_mode):
        raise AutomaticTreeApplyError("source_path must not be a symlink")
    if not stat.S_ISREG(source_stat.st_mode):
        raise AutomaticTreeApplyError("source_path must be a regular Parquet file")
    if os.path.normcase(os.fspath(source)) == os.path.normcase(os.fspath(output)):
        raise AutomaticTreeApplyError(
            "source_path and output_path must be different paths"
        )
    if source.suffix.lower() != ".parquet":
        raise AutomaticTreeApplyError("source_path must have a .parquet suffix")
    if output.suffix.lower() != ".parquet":
        raise AutomaticTreeApplyError("output_path must have a .parquet suffix")
    if output.exists() or output.is_symlink():
        raise AutomaticTreeApplyError("output_path must not already exist")
    return source, output


def _open_verified_regular_file(
    path: Path,
    *,
    expected_stat: os.stat_result | None = None,
) -> tuple[int, os.stat_result]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise AutomaticTreeApplyError(
            "source Parquet path changed or is not readable"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise AutomaticTreeApplyError("source Parquet path must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise AutomaticTreeApplyError("source Parquet path must be a regular file")
    if expected_stat is not None and _stable_stat(before) != _stable_stat(
        expected_stat
    ):
        raise AutomaticTreeApplyError(
            "source Parquet path changed or was replaced during writeback"
        )

    flags = os.O_RDONLY
    for optional_flag in ("O_BINARY", "O_CLOEXEC", "O_NOFOLLOW"):
        flags |= int(getattr(os, optional_flag, 0))
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AutomaticTreeApplyError(
            "source Parquet path changed or is not readable"
        ) from exc
    try:
        opened = os.fstat(fd)
        after = os.lstat(path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(after.st_mode)
            or not _same_file_identity(before, opened)
            or not _same_file_identity(opened, after)
            or _stable_stat(before) != _stable_stat(opened)
            or _stable_stat(opened) != _stable_stat(after)
        ):
            raise AutomaticTreeApplyError(
                "source Parquet path changed or was replaced while it was opened"
            )
        if expected_stat is not None and _stable_stat(opened) != _stable_stat(
            expected_stat
        ):
            raise AutomaticTreeApplyError(
                "source Parquet path changed or was replaced during writeback"
            )
        return fd, opened
    except Exception:
        os.close(fd)
        raise


def _copy_and_hash_fd(source_fd: int, target_fd: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    copied_bytes = 0
    os.lseek(source_fd, 0, os.SEEK_SET)
    os.lseek(target_fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        copied_bytes += len(chunk)
        remaining = memoryview(chunk)
        while remaining:
            written = os.write(target_fd, remaining)
            if written <= 0:
                raise AutomaticTreeApplyError(
                    "private source snapshot could not be written completely"
                )
            remaining = remaining[written:]
    return digest.hexdigest(), copied_bytes


def _verify_retained_fd(
    fd: int,
    *,
    expected_stat: os.stat_result,
    expected_hash: str,
    label: str,
) -> None:
    before = os.fstat(fd)
    if _stable_stat(before) != _stable_stat(expected_stat):
        raise AutomaticTreeApplyError(
            f"{label} changed or was replaced during writeback"
        )
    content_hash = _hash_fd(fd)
    after = os.fstat(fd)
    if (
        _stable_stat(after) != _stable_stat(expected_stat)
        or content_hash != expected_hash
    ):
        raise AutomaticTreeApplyError(
            f"{label} changed or was replaced during writeback"
        )


def _hash_fd(fd: int) -> str:
    position = os.lseek(fd, 0, os.SEEK_CUR)
    digest = hashlib.sha256()
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    finally:
        os.lseek(fd, position, os.SEEK_SET)
    return digest.hexdigest()


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
        stat.S_IFMT(left.st_mode),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        stat.S_IFMT(right.st_mode),
    )


def _stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(stat.S_IFMT(value.st_mode)),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _validated_output_columns(
    *,
    leaf_id_column: str,
    rule_id_column: str,
) -> dict[str, str]:
    columns = {"leaf_id": leaf_id_column, "rule_id": rule_id_column}
    for role, column in columns.items():
        if not isinstance(column, str) or not column:
            raise AutomaticTreeApplyError(
                f"{role}_column must be a non-empty safe identifier"
            )
        if len(column) > 64:
            raise AutomaticTreeApplyError(
                f"{role}_column must be at most 64 characters"
            )
        if not column.isascii() or _SAFE_OUTPUT_NAME.fullmatch(column) is None:
            if column[:1].isascii() and column[:1].isdigit():
                raise AutomaticTreeApplyError(
                    f"{role}_column cannot start with a digit"
                )
            raise AutomaticTreeApplyError(
                f"{role}_column must contain only ASCII letters, digits, and "
                "underscores and cannot start with a digit"
            )
    if leaf_id_column.casefold() == rule_id_column.casefold():
        raise AutomaticTreeApplyError(
            "automatic-tree output column names must be case-insensitively unique"
        )
    return columns


def _open_source_parquet(source: Path | BinaryIO) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(source)
    except (OSError, pa.ArrowException) as exc:
        raise AutomaticTreeApplyError(
            "source_path must be a readable normalized Parquet file"
        ) from exc


def _enforce_static_budgets(
    *,
    source_rows: int,
    source_columns: int,
    feature_count: int,
    node_count: int,
    leaf_count: int,
) -> None:
    _enforce_budget("source_rows", source_rows, _MAX_SOURCE_ROWS)
    _enforce_budget("source_columns", source_columns, _MAX_SOURCE_COLUMNS)
    _enforce_budget("features", feature_count, _MAX_FEATURES)
    _enforce_budget("feature_cells", source_rows * feature_count, _MAX_FEATURE_CELLS)
    _enforce_budget("tree_nodes", node_count, _MAX_TREE_NODES)
    _enforce_budget("tree_leaves", leaf_count, _MAX_TREE_LEAVES)


def _enforce_budget(dimension: str, actual: int, limit: int) -> None:
    if actual > limit:
        raise AutomaticTreeApplyBudgetError(
            dimension=dimension,
            actual=actual,
            limit=limit,
        )


def _validate_source_schema(
    schema: pa.Schema,
    *,
    features: tuple[str, ...],
    output_columns: Mapping[str, str],
) -> tuple[int, ...]:
    names = schema.names
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise AutomaticTreeApplyError(
            "source Parquet has duplicate column names: " + ", ".join(duplicate_names)
        )
    name_set = set(names)
    missing = [feature for feature in features if feature not in name_set]
    if missing:
        raise AutomaticTreeApplyError(
            "source Parquet is missing exact selected training features: "
            + ", ".join(missing)
        )
    source_casefold = {name.casefold() for name in names}
    collisions = sorted(
        column
        for column in output_columns.values()
        if column.casefold() in source_casefold
    )
    if collisions:
        raise AutomaticTreeApplyError(
            "automatic-tree output columns already exist (case-insensitive): "
            + ", ".join(collisions)
        )
    feature_indexes = tuple(schema.get_field_index(feature) for feature in features)
    for feature, index in zip(features, feature_indexes, strict=True):
        data_type = schema.field(index).type
        if not (pa.types.is_integer(data_type) or pa.types.is_floating(data_type)):
            raise AutomaticTreeApplyError(
                "weighted-tree feature requires an integer/float physical type: "
                + feature
            )
    return feature_indexes


def _empty_feature_frame(
    schema: pa.Schema,
    *,
    feature_indexes: tuple[int, ...],
) -> pd.DataFrame:
    feature_schema = pa.schema([schema.field(index) for index in feature_indexes])
    return pa.Table.from_batches([], schema=feature_schema).to_pandas(
        types_mapper=pd.ArrowDtype,
    )


def _preflight_feature_frame(
    frame: pd.DataFrame,
    tree: Mapping[str, Any],
    *,
    features: tuple[str, ...],
) -> None:
    try:
        validate_automatic_tree_duckdb_input_frame(
            frame,
            tree,
            additional_feature_fields=features,
        )
    except AutomaticTreeCodegenError as exc:
        raise AutomaticTreeApplyError(
            f"automatic-tree feature domain preflight failed: {exc}"
        ) from exc


def _canonical_leaf_ids(
    frame: pd.DataFrame,
    tree: Mapping[str, Any],
) -> list[str]:
    try:
        return [str(value) for value in apply_weighted_rule_tree(frame, tree).tolist()]
    except WeightedRuleTreeError as exc:
        raise AutomaticTreeApplyError(
            f"canonical weighted-tree replay failed: {exc}"
        ) from exc


def _result_arrow_schema(
    source_schema: pa.Schema,
    *,
    leaf_id_column: str,
    rule_id_column: str,
) -> pa.Schema:
    return pa.schema(
        [
            *list(source_schema),
            pa.field(leaf_id_column, pa.string(), nullable=False),
            pa.field(rule_id_column, pa.string(), nullable=False),
        ],
        metadata=source_schema.metadata,
    )


def _verify_emitted_parquet(
    output: Path,
    *,
    expected_schema: pa.Schema,
    expected_rows: int,
) -> pa.Schema:
    try:
        with pq.ParquetFile(output) as parquet_file:
            emitted_schema = parquet_file.schema_arrow
            emitted_rows = int(parquet_file.metadata.num_rows)
    except (OSError, pa.ArrowException) as exc:
        raise AutomaticTreeApplyError(
            "automatic-tree output is not a readable Parquet file"
        ) from exc
    if emitted_rows != expected_rows:
        raise AutomaticTreeApplyError(
            "automatic-tree output row count does not conserve the source"
        )
    if not emitted_schema.equals(expected_schema, check_metadata=True):
        raise AutomaticTreeApplyError(
            "automatic-tree output schema does not match the canonical projection"
        )
    return emitted_schema


def _schema_projection(schema: pa.Schema) -> dict[str, Any]:
    return {
        "fields": [
            {
                "name": field.name,
                "physical_type": str(field.type),
                "nullable": bool(field.nullable),
                "metadata_hash": _metadata_hash(field.metadata),
            }
            for field in schema
        ],
        "metadata_hash": _metadata_hash(schema.metadata),
    }


def _metadata_hash(metadata: Mapping[bytes, bytes] | None) -> str | None:
    if metadata is None:
        return None
    projection = [
        {"key_hex": bytes(key).hex(), "value_hex": bytes(value).hex()}
        for key, value in sorted(metadata.items())
    ]
    return _canonical_hash(projection)


def _writer_contract() -> dict[str, Any]:
    return {
        "contract": AUTOMATIC_TREE_APPLY_WRITER_CONTRACT,
        "engine": "pyarrow.parquet",
        "engine_version": pa.__version__,
        "threads": 1,
        "preserve_insertion_order": True,
        "batch_rows": _BATCH_ROWS,
        "max_decoded_batch_bytes": _MAX_DECODED_BATCH_BYTES,
        "row_group_rows": _BATCH_ROWS,
        "write_batch_rows": _PARQUET_WRITE_BATCH_ROWS,
        "parquet_version": "2.6",
        "data_page_version": "1.0",
        "compression": "zstd",
        "compression_level": _PARQUET_COMPRESSION_LEVEL,
        "dictionary_encoding": True,
        "dictionary_page_bytes": _PARQUET_DICTIONARY_PAGE_BYTES,
        "write_statistics": True,
        "byte_stream_split": False,
        "use_deprecated_int96_timestamps": False,
        "use_compliant_nested_type": True,
        "store_arrow_schema": True,
        "write_page_index": False,
        "write_page_checksum": False,
        "source_schema_metadata": "preserved",
        "appended_id_type": "utf8_non_null",
    }


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _detach_json(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


__all__ = [
    "AUTOMATIC_TREE_APPLY_PRODUCER_VERSION",
    "AUTOMATIC_TREE_APPLY_SCHEMA_VERSION",
    "AUTOMATIC_TREE_APPLY_WRITER_CONTRACT",
    "AutomaticTreeApplyBudgetError",
    "AutomaticTreeApplyError",
    "AutomaticTreeApplyResult",
    "apply_automatic_tree_to_parquet",
]
