from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from marvis.artifacts import ArtifactUnitOfWork
from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.csv_ingest import CsvIngestReport, read_csv_with_fallback_encoding
from marvis.data.errors import DataBackendError
from marvis.data.excel_ingest import ingest_sheet, list_sheets
from marvis.data.profiler import profile_dataset
from marvis.data.schema_infer import detect_target_column
from marvis.files import sha256_file

import logging

logger = logging.getLogger(__name__)


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

    def register_from_upload(
        self,
        task_id: str,
        source_path: Path,
        *,
        role: str = "unknown",
        seed: int = 0,
        max_excel_rows: int | None = None,
    ) -> Dataset:
        source_path = Path(source_path)
        dataset_dir = self._dataset_dir(task_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        uow = ArtifactUnitOfWork()
        final_name = f"{source_path.stem}_{uuid.uuid4().hex[:8]}.parquet"
        artifact = uow.stage_file(dataset_dir, final_name)
        try:
            self.last_csv_ingest_report = None
            sheet = self._write_upload_as_parquet(
                source_path, artifact.path, max_excel_rows=max_excel_rows
            )
            content_hash = sha256_file(artifact.path)
            find_by_hash = getattr(self._repo, "find_dataset_by_content_hash", None)
            existing = find_by_hash(content_hash) if callable(find_by_hash) else None
            if existing is not None:
                # GAP-7: identical file content already registered (possibly by a
                # different task) -- reuse its parquet + profiling instead of
                # writing a duplicate file and re-profiling. The staged upload
                # parquet is never promoted; only a new dataset row is created,
                # pointing at the existing dataset's source_path.
                uow.rollback()
                dataset = Dataset(
                    id=_new_dataset_id(),
                    task_id=task_id,
                    role=role,
                    source_path=existing.source_path,
                    format=existing.format,
                    sheet=existing.sheet,
                    row_count=existing.row_count,
                    columns=existing.columns,
                    has_target=existing.has_target,
                    target_col=existing.target_col,
                    created_at=_now_iso(),
                    content_hash=content_hash,
                )
                return self._create_dedup_reference(dataset, existing)
            profiles = profile_dataset(self._backend, artifact.path, seed=seed)
            sample = self._backend.sample_rows(artifact.path, 1000, seed=seed)
            target = detect_target_column(profiles, sample)
            dataset = Dataset(
                id=_new_dataset_id(),
                task_id=task_id,
                role=role,
                source_path=self._relative_path(artifact.final_path),
                format="parquet",
                sheet=sheet,
                row_count=self._backend.row_count(artifact.path),
                columns=tuple(profiles),
                has_target=target is not None,
                target_col=target,
                created_at=_now_iso(),
                content_hash=content_hash,
            )
            create_on_connection = getattr(self._repo, "create_dataset_on_connection", None)
            transaction = getattr(self._repo, "transaction", None)
            if callable(create_on_connection) and callable(transaction):
                return uow.finalize_with_connection(
                    transaction,
                    lambda conn: _create_dataset_on_connection(create_on_connection, conn, dataset),
                )
            return uow.finalize(lambda: _create_dataset(self._repo.create_dataset, dataset))
        except Exception:
            uow.rollback()
            raise

    def _create_dedup_reference(self, dataset: Dataset, existing: Dataset) -> Dataset:
        audit = {
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
        create_with_audit = getattr(self._repo, "create_dataset_with_audit", None)
        if callable(create_with_audit):
            create_with_audit(dataset, audit=audit)
            return dataset
        self._repo.create_dataset(dataset)
        return dataset

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
        if suffix in {".xlsx", ".xlsm"}:
            sheets = list_sheets(source_path)
            if not sheets:
                raise DataBackendError(f"workbook has no sheets: {source_path}")
            with tempfile.TemporaryDirectory(prefix=".xlsx_ingest_", dir=out_path.parent) as temp_name:
                parquet_path, report = ingest_sheet(
                    source_path, sheets[0], Path(temp_name), max_rows=max_excel_rows
                )
                shutil.move(parquet_path, out_path)
            return report.sheet
        raise DataBackendError(f"unsupported dataset upload format: {suffix}")

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
        )


def _new_dataset_id() -> str:
    return f"ds_{uuid.uuid4().hex}"


def _create_dataset(create_dataset, dataset: Dataset) -> Dataset:
    create_dataset(dataset)
    return dataset


def _create_dataset_on_connection(create_dataset_on_connection, conn, dataset: Dataset) -> Dataset:
    create_dataset_on_connection(conn, dataset)
    return dataset


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["DatasetRegistry"]
