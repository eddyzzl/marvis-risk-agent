from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.contracts import Dataset
from marvis.data.errors import DataBackendError
from marvis.data.excel_ingest import ingest_sheet, list_sheets
from marvis.data.profiler import profile_dataset
from marvis.data.schema_infer import detect_target_column


class DatasetRegistry:
    def __init__(self, repo, backend: DataBackend, datasets_root: Path):
        self._repo = repo
        self._backend = backend
        self._root = Path(datasets_root)
        self._root.mkdir(parents=True, exist_ok=True)

    def register_from_upload(
        self,
        task_id: str,
        source_path: Path,
        *,
        role: str = "unknown",
        seed: int = 0,
    ) -> Dataset:
        source_path = Path(source_path)
        dataset_dir = self._dataset_dir(task_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        parquet_path, sheet = self._normalize_to_parquet(source_path, dataset_dir)
        profiles = profile_dataset(self._backend, parquet_path, seed=seed)
        sample = self._backend.sample_rows(parquet_path, 1000, seed=seed)
        target = detect_target_column(profiles, sample)
        dataset = Dataset(
            id=_new_dataset_id(),
            task_id=task_id,
            role=role,
            source_path=self._relative_path(parquet_path),
            format="parquet",
            sheet=sheet,
            row_count=self._backend.row_count(parquet_path),
            columns=tuple(profiles),
            has_target=target is not None,
            target_col=target,
            created_at=_now_iso(),
        )
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
        suffix = source_path.suffix.lower()
        out_path = dataset_dir / f"{source_path.stem}_{uuid.uuid4().hex[:8]}.parquet"
        if suffix == ".parquet":
            if source_path.resolve() == out_path.resolve():
                return out_path, None
            shutil.copy2(source_path, out_path)
            return out_path, None
        if suffix == ".csv":
            frame = pd.read_csv(source_path, encoding="utf-8-sig")
            frame.to_parquet(out_path, index=False)
            return out_path, None
        if suffix == ".feather":
            frame = pd.read_feather(source_path)
            frame.to_parquet(out_path, index=False)
            return out_path, None
        if suffix in {".xlsx", ".xlsm"}:
            sheets = list_sheets(source_path)
            if not sheets:
                raise DataBackendError(f"workbook has no sheets: {source_path}")
            parquet_path, report = ingest_sheet(source_path, sheets[0], dataset_dir)
            return parquet_path, report.sheet
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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["DatasetRegistry"]
