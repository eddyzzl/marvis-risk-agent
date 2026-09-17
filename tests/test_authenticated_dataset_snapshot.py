from __future__ import annotations

import hashlib
import stat
from pathlib import Path

import pandas as pd
import pytest

import marvis.data.authenticated_snapshot as snapshot_module
from marvis.data.authenticated_snapshot import (
    AuthenticatedSnapshotError,
    SnapshotFailureReason,
    materialize_authenticated_file_snapshot,
    read_authenticated_parquet_snapshot,
)


def _write_parquet(path: Path) -> str:
    pd.DataFrame(
        {
            "customer_id": ["A", "B"],
            "score": [610, 720],
            "unused": [1, 2],
        }
    ).to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_authenticated_snapshot_reads_only_requested_columns(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    digest = _write_parquet(dataset)

    frame = read_authenticated_parquet_snapshot(
        dataset,
        root=tmp_path,
        expected_sha256=digest,
        columns=["customer_id", "score"],
    )

    assert frame.to_dict(orient="records") == [
        {"customer_id": "A", "score": 610},
        {"customer_id": "B", "score": 720},
    ]


def test_authenticated_snapshot_rejects_hash_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_parquet(dataset)

    with pytest.raises(AuthenticatedSnapshotError) as captured:
        read_authenticated_parquet_snapshot(
            dataset,
            root=tmp_path,
            expected_sha256="0" * 64,
        )

    assert captured.value.reason is SnapshotFailureReason.SOURCE_BYTES_CHANGED


def test_authenticated_snapshot_rejects_path_outside_root(tmp_path: Path) -> None:
    governed_root = tmp_path / "governed"
    governed_root.mkdir()
    dataset = tmp_path / "outside.parquet"
    digest = _write_parquet(dataset)

    with pytest.raises(AuthenticatedSnapshotError) as captured:
        read_authenticated_parquet_snapshot(
            dataset,
            root=governed_root,
            expected_sha256=digest,
        )

    assert captured.value.reason is SnapshotFailureReason.PATH_OUTSIDE_ROOT


def test_materialized_snapshot_is_read_only_content_addressed_and_source_independent(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.parquet"
    digest = _write_parquet(dataset)
    destination = tmp_path / "_cas" / digest / f"{digest}.parquet"

    pinned = materialize_authenticated_file_snapshot(
        dataset,
        root=tmp_path,
        expected_sha256=digest,
        destination=destination,
    )
    pinned_bytes = pinned.read_bytes()
    pd.DataFrame({"customer_id": ["MUTATED"]}).to_parquet(dataset, index=False)

    assert pinned == destination.resolve()
    assert hashlib.sha256(pinned_bytes).hexdigest() == digest
    assert pinned.read_bytes() == pinned_bytes
    assert stat.S_IMODE(pinned.stat().st_mode) == 0o444
    assert stat.S_IMODE(pinned.parent.stat().st_mode) == 0o555
    with pytest.raises(OSError):
        pd.DataFrame({"customer_id": ["ATTACK"]}).to_parquet(pinned, index=False)


def test_materialized_snapshot_reuses_only_matching_existing_object(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.parquet"
    digest = _write_parquet(dataset)
    destination = tmp_path / "_cas" / digest / f"{digest}.parquet"
    materialize_authenticated_file_snapshot(
        dataset,
        root=tmp_path,
        expected_sha256=digest,
        destination=destination,
    )

    reused = materialize_authenticated_file_snapshot(
        dataset,
        root=tmp_path,
        expected_sha256=digest,
        destination=destination,
    )

    assert hashlib.sha256(reused.read_bytes()).hexdigest() == digest


def test_materialized_snapshot_publishes_before_windows_read_only_hardening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows rejects renaming a directory after chmod marks it read-only."""

    dataset = tmp_path / "dataset.parquet"
    digest = _write_parquet(dataset)
    destination = tmp_path / "_cas" / digest / f"{digest}.parquet"
    read_only_paths: set[Path] = set()
    original_chmod = snapshot_module.os.chmod
    original_rename = snapshot_module.os.rename

    def windows_chmod(path, mode):
        normalized = Path(path)
        original_chmod(path, mode)
        if mode & stat.S_IWRITE:
            read_only_paths.discard(normalized)
        else:
            read_only_paths.add(normalized)

    def windows_rename(source, target):
        if Path(source) in read_only_paths:
            raise PermissionError("Windows cannot rename a read-only directory")
        return original_rename(source, target)

    monkeypatch.setattr(snapshot_module.os, "chmod", windows_chmod)
    monkeypatch.setattr(snapshot_module.os, "rename", windows_rename)

    pinned = materialize_authenticated_file_snapshot(
        dataset,
        root=tmp_path,
        expected_sha256=digest,
        destination=destination,
    )

    assert pinned == destination.resolve()
    assert hashlib.sha256(pinned.read_bytes()).hexdigest() == digest


def test_materialized_snapshot_rejects_drifted_source_when_cas_already_exists(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.parquet"
    digest = _write_parquet(first)
    second = tmp_path / "second.parquet"
    second.write_bytes(first.read_bytes())
    destination = tmp_path / "_cas" / digest / f"{digest}.parquet"
    materialize_authenticated_file_snapshot(
        first,
        root=tmp_path,
        expected_sha256=digest,
        destination=destination,
    )
    pd.DataFrame({"customer_id": ["DRIFT"]}).to_parquet(second, index=False)

    with pytest.raises(AuthenticatedSnapshotError) as captured:
        materialize_authenticated_file_snapshot(
            second,
            root=tmp_path,
            expected_sha256=digest,
            destination=destination,
        )

    assert captured.value.reason is SnapshotFailureReason.SOURCE_BYTES_CHANGED
