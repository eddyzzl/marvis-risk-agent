from __future__ import annotations

from collections import OrderedDict
import hmac
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.csv_ingest import CsvIngestReport, read_csv_with_fallback_encoding
from marvis.data.errors import DataBackendError, DatasetContentDriftError
from marvis.data.excel_ingest import (
    detect_excel_container_format,
    ingest_sheet,
    list_sheets,
    require_excel_format,
)
from marvis.data.profiler import profile_dataset
from marvis.data.schema_infer import detect_target_column
from marvis.files import sha256_file

import logging

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_FILE_CACHE_MAX = 512
_VERIFIED_FILE_CACHE: OrderedDict[
    tuple[str, str],
    tuple[int, int, int, int, int],
] = OrderedDict()
_VERIFIED_FILE_CACHE_LOCK = Lock()


class DatasetRegistry:
    def __init__(self, repo, backend: DataBackend, datasets_root: Path):
        self._repo = repo
        self._backend = backend
        self._root = Path(datasets_root)
        self._root.mkdir(parents=True, exist_ok=True)
        # GAP-1: side-channel for the most recent CSV encoding/dtype-defense
        # decision, populated during register_from_upload/_write_upload_as_parquet.
        # A router handling a single upload request calls register_from_upload
        # once and reads this immediately after -- mirrors the existing
        # single-request-scoped usage pattern of this registry instance.
        self.last_csv_ingest_report: CsvIngestReport | None = None
        self._pending_ingest_notice: dict | None = None
        self._ingest_notices_by_task: dict[str, list[dict]] = {}

    def consume_ingest_notices(self, task_id: str) -> list[dict]:
        """Return and clear user-facing material recovery notices for a task."""

        notices = self._ingest_notices_by_task.pop(str(task_id), [])
        unique: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for notice in notices:
            key = (str(notice.get("code") or ""), str(notice.get("file") or ""))
            if key in seen:
                continue
            seen.add(key)
            unique.append(dict(notice))
        return unique

    def register_from_upload(
        self,
        task_id: str,
        source_path: Path,
        *,
        role: str = "unknown",
        seed: int = 0,
        max_excel_rows: int | None = None,
        audit_factory: Callable[[Dataset], dict] | None = None,
    ) -> Dataset:
        if audit_factory is not None:
            self._require_atomic_upload_audit_support()
        source_path = Path(source_path)
        dataset_dir = self._dataset_dir(task_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        uow = ArtifactUnitOfWork()
        final_name = f"{source_path.stem}_{uuid.uuid4().hex[:8]}.parquet"
        artifact = uow.stage_file(dataset_dir, final_name)
        pending_notice: dict | None = None
        try:
            self.last_csv_ingest_report = None
            self._pending_ingest_notice = None
            sheet = self._write_upload_as_parquet(
                source_path, artifact.path, max_excel_rows=max_excel_rows
            )
            if self._pending_ingest_notice is not None:
                pending_notice = dict(self._pending_ingest_notice)
            content_hash = sha256_file(artifact.path)
            find_by_hash = getattr(self._repo, "find_dataset_by_content_hash", None)
            existing = find_by_hash(content_hash) if callable(find_by_hash) else None
            if existing is not None:
                profiles = existing.columns
                target = existing.target_col if existing.has_target else None
                row_count = existing.row_count
            else:
                profiles = profile_dataset(self._backend, artifact.path, seed=seed)
                sample = self._backend.sample_rows(artifact.path, 1000, seed=seed)
                target = detect_target_column(profiles, sample)
                row_count = self._backend.row_count(artifact.path)
            dataset = Dataset(
                id=_new_dataset_id(),
                task_id=task_id,
                role=role,
                source_path=self._relative_path(artifact.final_path),
                format="parquet",
                sheet=sheet,
                row_count=row_count,
                columns=tuple(profiles),
                has_target=target is not None,
                target_col=target,
                created_at=_now_iso(),
                content_hash=content_hash,
            )
            atomic_result = self._register_upload_atomically(
                uow,
                dataset,
                audit_factory=audit_factory,
            )
            if atomic_result is not None:
                result = atomic_result
            else:
                create_on_connection = getattr(
                    self._repo,
                    "create_dataset_on_connection",
                    None,
                )
                transaction = getattr(self._repo, "transaction", None)
                if callable(create_on_connection) and callable(transaction):
                    result = uow.finalize_with_connection(
                        transaction,
                        lambda conn: _create_dataset_on_connection(
                            create_on_connection,
                            conn,
                            dataset,
                        ),
                    )
                else:
                    result = uow.finalize(
                        lambda: _create_dataset(self._repo.create_dataset, dataset)
                    )
            if pending_notice is not None:
                self._ingest_notices_by_task.setdefault(str(task_id), []).append(
                    pending_notice
                )
            return result
        except Exception:
            uow.rollback()
            self._pending_ingest_notice = None
            raise

    def _register_upload_atomically(
        self,
        uow: ArtifactUnitOfWork,
        dataset: Dataset,
        *,
        audit_factory: Callable[[Dataset], dict] | None = None,
    ) -> Dataset | None:
        transaction = getattr(self._repo, "transaction", None)
        find_on_connection = getattr(
            self._repo,
            "find_dataset_by_content_hash_on_connection",
            None,
        )
        create_on_connection = getattr(self._repo, "create_dataset_on_connection", None)
        create_with_audit_on_connection = getattr(
            self._repo,
            "create_dataset_with_audit_on_connection",
            None,
        )
        write_audit_on_connection = getattr(
            self._repo,
            "write_audit_on_connection",
            None,
        )
        required_methods = (
            transaction,
            find_on_connection,
            create_on_connection,
            create_with_audit_on_connection,
        )
        if audit_factory is not None:
            required_methods += (write_audit_on_connection,)
        if not all(callable(method) for method in required_methods):
            if audit_factory is not None:
                self._require_atomic_upload_audit_support()
            return None

        promoted = False
        try:
            with transaction() as conn:
                # Serialize the content-hash lookup and reference insertion with
                # task purge. A WAL read followed by a later write can otherwise
                # retain a source_path that purge has already removed.
                conn.execute("BEGIN IMMEDIATE")
                existing = find_on_connection(conn, dataset.content_hash)
                if existing is None:
                    uow.promote_all()
                    promoted = True
                    create_on_connection(conn, dataset)
                    result = dataset
                else:
                    result = _dedup_reference_dataset(dataset, existing)
                    create_with_audit_on_connection(
                        conn,
                        result,
                        audit=_dedup_reference_audit(result, existing),
                    )
                if audit_factory is not None:
                    write_audit_on_connection(conn, **audit_factory(result))
        except Exception:
            uow.rollback()
            raise
        if promoted:
            uow.commit()
        else:
            uow.rollback()
        return result

    def _require_atomic_upload_audit_support(self) -> None:
        required_methods = (
            "transaction",
            "find_dataset_by_content_hash_on_connection",
            "create_dataset_on_connection",
            "create_dataset_with_audit_on_connection",
            "write_audit_on_connection",
        )
        missing = [
            name
            for name in required_methods
            if not callable(getattr(self._repo, name, None))
        ]
        if missing:
            raise DataBackendError(
                "dataset repository does not support atomic audited upload registration; "
                f"missing connection-scoped methods: {', '.join(missing)}"
            )

    def register_existing(
        self,
        parquet_path: Path,
        *,
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        self._repo.create_dataset(dataset)
        return dataset

    def register_existing_on_connection(
        self,
        conn,
        parquet_path: Path,
        *,
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        create_on_connection = getattr(self._repo, "create_dataset_on_connection", None)
        if not callable(create_on_connection):
            raise DataBackendError("dataset repository does not support connection-scoped dataset writes")
        create_on_connection(conn, dataset)
        return dataset

    def register_existing_with_audit_on_connection(
        self,
        conn,
        parquet_path: Path,
        *,
        audit_factory: Callable[[Dataset], dict],
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        audit = audit_factory(dataset)
        create_with_audit_on_connection = getattr(
            self._repo,
            "create_dataset_with_audit_on_connection",
            None,
        )
        if not callable(create_with_audit_on_connection):
            raise DataBackendError("dataset repository does not support connection-scoped audited dataset writes")
        create_with_audit_on_connection(conn, dataset, audit=audit)
        return dataset

    def register_existing_with_audit(
        self,
        parquet_path: Path,
        *,
        audit_factory: Callable[[Dataset], dict],
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        audit = audit_factory(dataset)
        try:
            self._repo.create_dataset_with_audit(dataset, audit=audit)
        except Exception:
            parquet_path.unlink(missing_ok=True)
            raise
        return dataset

    def register_join_result_with_audit(
        self,
        parquet_path: Path,
        *,
        join_plan_id: str,
        audit_factory: Callable[[Dataset], dict],
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        audit = audit_factory(dataset)
        try:
            self._repo.record_join_result_with_audit(
                join_plan_id,
                dataset,
                audit=audit,
            )
        except Exception:
            parquet_path.unlink(missing_ok=True)
            raise
        return dataset

    def register_join_result_with_audit_on_connection(
        self,
        conn,
        parquet_path: Path,
        *,
        join_plan_id: str,
        audit_factory: Callable[[Dataset], dict],
        task_id: str,
        role: str,
        anchor_target: str | None = None,
        seed: int = 0,
    ) -> Dataset:
        parquet_path = self._ensure_under_root(Path(parquet_path), task_id)
        dataset = self._dataset_from_existing(
            parquet_path,
            task_id=task_id,
            role=role,
            anchor_target=anchor_target,
            seed=seed,
        )
        audit = audit_factory(dataset)
        record_on_connection = getattr(
            self._repo,
            "record_join_result_with_audit_on_connection",
            None,
        )
        if not callable(record_on_connection):
            raise DataBackendError("dataset repository does not support connection-scoped join result writes")
        record_on_connection(
            conn,
            join_plan_id,
            dataset,
            audit=audit,
        )
        return dataset

    def transaction(self):
        return self._repo.transaction()

    def get(self, dataset_id: str) -> Dataset:
        dataset = self._repo.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        return dataset

    def list_for_task(self, task_id: str) -> list[Dataset]:
        return self._repo.list_datasets(task_id)

    def resolve_path(self, dataset_id: str) -> Path:
        return self._root / self.get(dataset_id).source_path

    def resolve_verified_path(self, dataset_id: str) -> Path:
        """Resolve an immutable dataset and fail closed on out-of-band drift.

        Dataset hashes identify the canonical parquet bytes, not merely the
        database row.  Workspace reads use this boundary before returning any
        preview or accepting a semantic snapshot so an in-place file change
        cannot silently reuse the previous analysis generation.
        """

        dataset = self.get(dataset_id)
        expected_hash = dataset.content_hash
        if (
            not isinstance(expected_hash, str)
            or _SHA256_RE.fullmatch(expected_hash) is None
        ):
            raise DatasetContentDriftError(
                dataset_id,
                reason="registered content hash is missing or invalid",
            )
        candidate = self._root / dataset.source_path
        try:
            resolved_root = self._root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
            if candidate.is_symlink() or not resolved.is_file():
                raise OSError("dataset path is not a regular workspace file")
            _verify_stable_file_hash(
                resolved,
                expected_hash=expected_hash,
                dataset_id=dataset_id,
            )
        except (OSError, ValueError) as exc:
            raise DatasetContentDriftError(
                dataset_id,
                reason="registered file is missing or outside the dataset workspace",
            ) from exc
        return resolved

    def set_role(self, dataset_id: str, role: str) -> None:
        self._repo.set_dataset_role(dataset_id, role)

    def _normalize_to_parquet(self, source_path: Path, dataset_dir: Path) -> tuple[Path, str | None]:
        out_path = dataset_dir / f"{source_path.stem}_{uuid.uuid4().hex[:8]}.parquet"
        sheet = self._write_upload_as_parquet(source_path, out_path)
        return out_path, sheet

    def _write_upload_as_parquet(
        self,
        source_path: Path,
        out_path: Path,
        *,
        max_excel_rows: int | None = None,
    ) -> str | None:
        suffix = source_path.suffix.lower()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".parquet":
            if source_path.resolve() == out_path.resolve():
                return None
            shutil.copy2(source_path, out_path)
            return None
        if suffix == ".csv":
            if detect_excel_container_format(source_path) is not None:
                detected_excel_format = require_excel_format(source_path)
                sheet = self._write_excel_upload_as_parquet(
                    source_path,
                    out_path,
                    max_excel_rows=max_excel_rows,
                    copy_to_xlsx=True,
                )
                self._pending_ingest_notice = {
                    "code": "extension_content_mismatch",
                    "severity": "warning",
                    "file": source_path.name,
                    "declared_format": "csv",
                    "detected_format": detected_excel_format,
                    "message": (
                        f"`{source_path.name}` 扩展名是 `.csv`，但内容是 "
                        f"{detected_excel_format.upper()} Excel 工作簿；"
                        "已按 Excel 工作簿读取，原文件未修改。"
                    ),
                }
                return sheet
            frame, report = read_csv_with_fallback_encoding(source_path)
            self.last_csv_ingest_report = report
            if report.encoding_used != "utf-8-sig":
                logger.info(
                    "CSV %s decoded with fallback encoding %s",
                    source_path.name,
                    report.encoding_used,
                )
            if report.long_id_columns:
                logger.info(
                    "CSV %s: %d long numeric id column(s) read as string to avoid "
                    "float64 precision truncation: %s",
                    source_path.name,
                    len(report.long_id_columns),
                    ", ".join(report.long_id_columns),
                )
            frame.to_parquet(out_path, index=False)
            return None
        if suffix == ".feather":
            frame = pd.read_feather(source_path)
            frame.to_parquet(out_path, index=False)
            return None
        if suffix in {".xls", ".xlsx", ".xlsm"}:
            return self._write_excel_upload_as_parquet(
                source_path,
                out_path,
                max_excel_rows=max_excel_rows,
                copy_to_xlsx=False,
            )
        raise DataBackendError(f"unsupported dataset upload format: {suffix}")

    def _write_excel_upload_as_parquet(
        self,
        source_path: Path,
        out_path: Path,
        *,
        max_excel_rows: int | None,
        copy_to_xlsx: bool,
    ) -> str:
        with tempfile.TemporaryDirectory(
            prefix=".xlsx_ingest_", dir=out_path.parent
        ) as temp_name:
            temp_dir = Path(temp_name)
            workbook_path = source_path
            if copy_to_xlsx:
                workbook_path = temp_dir / f"{source_path.stem}.xlsx"
                shutil.copy2(source_path, workbook_path)
            sheets = list_sheets(workbook_path)
            if not sheets:
                raise DataBackendError(f"workbook has no sheets: {source_path}")
            parquet_path, report = ingest_sheet(
                workbook_path,
                sheets[0],
                temp_dir / "normalized",
                max_rows=max_excel_rows,
            )
            shutil.move(parquet_path, out_path)
        return report.sheet

    def _ensure_under_root(self, parquet_path: Path, task_id: str) -> Path:
        if parquet_path.suffix.lower() != ".parquet":
            raise DataBackendError("register_existing requires a parquet file")
        try:
            parquet_path.resolve().relative_to(self._root.resolve())
            return parquet_path
        except ValueError:
            dataset_dir = self._dataset_dir(task_id)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            out_path = dataset_dir / f"{parquet_path.stem}_{uuid.uuid4().hex[:8]}.parquet"
            shutil.copy2(parquet_path, out_path)
            return out_path

    def _dataset_dir(self, task_id: str) -> Path:
        return self._root / task_id

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self._root.resolve()).as_posix()

    def _dataset_from_existing(
        self,
        parquet_path: Path,
        *,
        task_id: str,
        role: str,
        anchor_target: str | None,
        seed: int,
    ) -> Dataset:
        profiles = profile_dataset(self._backend, parquet_path, seed=seed)
        target = None
        if anchor_target:
            anchor = self.get(anchor_target)
            target = anchor.target_col if anchor.has_target else None
        if target is None:
            sample = self._backend.sample_rows(parquet_path, 1000, seed=seed)
            target = detect_target_column(profiles, sample)
        return Dataset(
            id=_new_dataset_id(),
            task_id=task_id,
            role=role,
            source_path=self._relative_path(parquet_path),
            format="parquet",
            sheet=None,
            row_count=self._backend.row_count(parquet_path),
            columns=tuple(profiles),
            has_target=target is not None,
            target_col=target,
            created_at=_now_iso(),
            content_hash=sha256_file(parquet_path),
        )


def _new_dataset_id() -> str:
    return f"ds_{uuid.uuid4().hex}"


def _verify_stable_file_hash(
    path: Path,
    *,
    expected_hash: str,
    dataset_id: str,
) -> None:
    """Hash once per stable file identity and re-hash after any stat change."""

    before = path.stat()
    signature = _file_signature(before)
    cache_key = (str(path), expected_hash)
    with _VERIFIED_FILE_CACHE_LOCK:
        if _VERIFIED_FILE_CACHE.get(cache_key) == signature:
            _VERIFIED_FILE_CACHE.move_to_end(cache_key)
            return

    actual_hash = sha256_file(path)
    after_signature = _file_signature(path.stat())
    if after_signature != signature:
        _forget_verified_path(path)
        raise DatasetContentDriftError(
            dataset_id,
            reason="registered file changed during integrity verification",
        )
    if not hmac.compare_digest(actual_hash, expected_hash):
        _forget_verified_path(path)
        raise DatasetContentDriftError(dataset_id)

    with _VERIFIED_FILE_CACHE_LOCK:
        for key in tuple(_VERIFIED_FILE_CACHE):
            if key[0] == str(path) and key != cache_key:
                del _VERIFIED_FILE_CACHE[key]
        _VERIFIED_FILE_CACHE[cache_key] = signature
        _VERIFIED_FILE_CACHE.move_to_end(cache_key)
        while len(_VERIFIED_FILE_CACHE) > _VERIFIED_FILE_CACHE_MAX:
            _VERIFIED_FILE_CACHE.popitem(last=False)


def _file_signature(stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _forget_verified_path(path: Path) -> None:
    path_text = str(path)
    with _VERIFIED_FILE_CACHE_LOCK:
        for key in tuple(_VERIFIED_FILE_CACHE):
            if key[0] == path_text:
                del _VERIFIED_FILE_CACHE[key]


def _create_dataset(create_dataset, dataset: Dataset) -> Dataset:
    create_dataset(dataset)
    return dataset


def _create_dataset_on_connection(create_dataset_on_connection, conn, dataset: Dataset) -> Dataset:
    create_dataset_on_connection(conn, dataset)
    return dataset


def _dedup_reference_dataset(dataset: Dataset, existing: Dataset) -> Dataset:
    return Dataset(
        id=dataset.id,
        task_id=dataset.task_id,
        role=dataset.role,
        source_path=existing.source_path,
        format=existing.format,
        sheet=existing.sheet,
        row_count=existing.row_count,
        columns=existing.columns,
        has_target=existing.has_target,
        target_col=existing.target_col,
        created_at=dataset.created_at,
        content_hash=dataset.content_hash,
    )


def _dedup_reference_audit(dataset: Dataset, existing: Dataset) -> dict:
    return {
        "kind": "dataset.dedup_reference",
        "target_ref": dataset.id,
        "outcome": "succeeded",
        "detail": {
            "task_id": dataset.task_id,
            "content_hash": dataset.content_hash,
            "reused_dataset_id": existing.id,
            "reused_task_id": existing.task_id,
            "source_path": dataset.source_path,
        },
    }


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["DatasetRegistry"]
