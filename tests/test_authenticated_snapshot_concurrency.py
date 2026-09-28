from concurrent.futures import ThreadPoolExecutor
from threading import Event

import marvis.data.authenticated_snapshot as snapshots

from test_authenticated_dataset_snapshot import _write_parquet


def _pinned(tmp_path):
    source = tmp_path / "source.parquet"
    digest = _write_parquet(source)
    destination = tmp_path / "_cas" / digest / f"{digest}.parquet"
    path = snapshots.materialize_authenticated_file_snapshot(
        source, root=tmp_path, expected_sha256=digest, destination=destination
    )
    return path, digest


def test_repin_and_verify_do_not_mutate_hardened_file_or_directory(
    tmp_path, monkeypatch
):
    path, digest = _pinned(tmp_path)
    before = (path.stat().st_ctime_ns, path.parent.stat().st_ctime_ns)
    changed = []
    original = snapshots.os.chmod

    def record_chmod(target, mode):
        changed.append((target, mode))
        return original(target, mode)

    monkeypatch.setattr(snapshots.os, "chmod", record_chmod)
    snapshots.materialize_authenticated_file_snapshot(
        path, root=tmp_path, expected_sha256=digest, destination=path
    )
    snapshots.verify_content_addressed_file_snapshot(
        path, root=tmp_path, expected_sha256=digest
    )
    assert changed == []  # Also deterministic on filesystems with coarse ctime.
    assert before == (path.stat().st_ctime_ns, path.parent.stat().st_ctime_ns)


def test_repin_during_retained_descriptor_read_keeps_identity_stable(
    tmp_path, monkeypatch
):
    path, digest = _pinned(tmp_path)
    decoded = Event()
    repinned = Event()
    original = snapshots.pd.read_parquet

    def coordinated_read(*args, **kwargs):
        frame = original(*args, **kwargs)
        decoded.set()
        assert repinned.wait(5), "concurrent authenticated pin did not complete"
        return frame

    monkeypatch.setattr(snapshots.pd, "read_parquet", coordinated_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(
            snapshots.read_authenticated_parquet_snapshot,
            path,
            root=tmp_path,
            expected_sha256=digest,
            columns=["customer_id", "score"],
        )
        try:
            assert decoded.wait(5), (
                "reader did not reach its authenticated private snapshot"
            )
            pin = pool.submit(
                snapshots.materialize_authenticated_file_snapshot,
                path,
                root=tmp_path,
                expected_sha256=digest,
                destination=path,
            )
            assert pin.result(timeout=5) == path
        finally:
            repinned.set()
        assert read.result(timeout=5).to_dict(orient="records") == [
            {"customer_id": "A", "score": 610},
            {"customer_id": "B", "score": 720},
        ]
