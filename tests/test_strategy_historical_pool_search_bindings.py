from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

import marvis.packs.strategy.voting_candidate_search_tools as search_tools
from marvis.data.workspace import DataWorkspaceDraft
from marvis.db import connect
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_tools import (
    POOL_ARTIFACT_KIND,
    load_current_strategy_candidate_pool_artifact,
    load_strategy_candidate_pool_revision_artifact,
    require_strategy_candidate_pool_revision_artifact_binding_on_connection,
    run_set_pool_entry_action,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.packs.strategy.voting_candidate_search_tools import (
    load_historical_voting_candidate_search_artifact,
    load_voting_candidate_search_artifact,
    require_historical_voting_candidate_search_artifact_binding_on_connection,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND,
    load_historical_strategy_sample_design_v2_artifacts,
    load_strategy_sample_design_v2_artifacts,
    require_historical_strategy_sample_design_v2_artifact_binding_on_connection,
    run_materialize_sample_design_v2,
)
from tests.test_strategy_voting_search_selection_tools import _searched_fixture


def _advance_pool(fixture: dict) -> dict:
    pool = fixture["pool"]
    return run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
            "rule_id": pool["entries"][0]["rule_id"],
            "action": {
                "type": "review",
                "value": "review",
                "reason_code": "HISTORICAL_AUTH_TEST",
                "stop": True,
            },
        },
        fixture["ctx"],
        fixture["runtime"],
    )


def _advance_workspace_ui(fixture: dict) -> None:
    repository = DataWorkspaceRepository(fixture["settings"].db_path)
    current = repository.get_or_default(fixture["task"].id)
    changed = repository.save(
        fixture["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=current.active_dataset_id,
            active_dataset_content_hash=current.active_dataset_content_hash,
            page=("history" if current.page != "history" else "statistics"),
            selected_field=current.selected_field,
            semantic_mapping=current.semantic_mapping,
        ),
        expected_revision=current.revision,
    )
    assert changed.revision == current.revision + 1
    assert changed.analysis_generation == current.analysis_generation


def _search_record(fixture: dict) -> dict:
    descriptor = fixture["search"]["artifacts"][0]
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        descriptor["artifact_id"],
    )
    assert record is not None
    return record


def _load_historical_search(fixture: dict):
    descriptor = fixture["search"]["artifacts"][0]
    return load_historical_voting_candidate_search_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
    )


def test_historical_pool_revision_authenticates_after_head_advances(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    old_pool = fixture["pool"]
    search_artifact = fixture["search"]["artifacts"][0]
    search_record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        search_artifact["artifact_id"],
    )
    assert search_record is not None
    pool_ref = search_record["provenance"]["pool_ref"]
    changed = _advance_pool(fixture)
    assert changed["revision"] == old_pool["revision"] + 1

    binding = load_strategy_candidate_pool_revision_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        strategy_type="approval",
        revision_id=old_pool["revision_id"],
        artifact_id=pool_ref["artifact_id"],
        expected_artifact_content_hash=pool_ref["artifact_content_hash"],
    )

    assert binding.pool == old_pool
    assert binding.artifact_id == pool_ref["artifact_id"]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_candidate_pool_revision_artifact_binding_on_connection(
            conn,
            binding,
        )
        assert conn.in_transaction
        conn.rollback()


def test_current_pool_loader_still_rejects_historical_revision(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    old_pool = fixture["pool"]
    _advance_pool(fixture)

    with pytest.raises(StrategyError, match="stale|current"):
        load_current_strategy_candidate_pool_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            strategy_type="approval",
            expected_pool_revision=old_pool["revision"],
            expected_pool_snapshot_hash=old_pool["snapshot_hash"],
        )


def test_exact_pool_artifact_lookup_survives_more_than_64_pool_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _searched_fixture(tmp_path)
    task_id = fixture["task"].id
    old_pool = fixture["pool"]
    old_pool_ref = _search_record(fixture)["provenance"]["pool_ref"]
    changed = _advance_pool(fixture)
    current_pool = changed["pool"]
    current_descriptor = changed["artifacts"][0]
    artifact_repository = fixture["runtime"].task_artifacts
    distractor_dir = (
        Path(fixture["settings"].tasks_dir)
        / task_id
        / "strategy_candidate_pools"
        / "distractors"
    )
    distractor_dir.mkdir(parents=True)
    for index in range(65):
        path = distractor_dir / f"pool-{index:03d}.json"
        raw = f'{{"distractor":{index}}}'.encode()
        path.write_bytes(raw)
        artifact_repository.register(
            task_id=task_id,
            kind=POOL_ARTIFACT_KIND,
            path=str(path),
            content_hash=hashlib.sha256(raw).hexdigest(),
            origin_tool="strategy.test_pool_artifact",
            provenance={"index": index},
            created_at=f"2099-01-01T00:00:00.{index:06d}+00:00",
        )

    original_list_for_task = artifact_repository.list_for_task

    def bounded_recent_list(owner_task_id: str) -> list[dict]:
        return original_list_for_task(owner_task_id)[-64:]

    monkeypatch.setattr(
        artifact_repository,
        "list_for_task",
        bounded_recent_list,
    )

    historical = load_strategy_candidate_pool_revision_artifact(
        fixture["runtime"],
        task_id=task_id,
        strategy_type="approval",
        revision_id=old_pool["revision_id"],
        artifact_id=old_pool_ref["artifact_id"],
        expected_artifact_content_hash=old_pool_ref["artifact_content_hash"],
    )
    current = load_current_strategy_candidate_pool_artifact(
        fixture["runtime"],
        task_id=task_id,
        strategy_type="approval",
        expected_pool_revision=current_pool["revision"],
        expected_pool_snapshot_hash=current_pool["snapshot_hash"],
        expected_artifact_id=current_descriptor["artifact_id"],
        expected_artifact_content_hash=current_descriptor["content_hash"],
    )

    assert historical.pool == old_pool
    assert current.pool == current_pool


def test_historical_search_authenticates_after_pool_head_advances(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    search = fixture["search"]
    descriptor = search["artifacts"][0]
    _advance_pool(fixture)

    historical = load_historical_voting_candidate_search_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
    )

    assert historical.result == search["search_result"]
    assert historical.pool_development.pool.pool == fixture["pool"]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_historical_voting_candidate_search_artifact_binding_on_connection(
            conn,
            historical,
        )
        assert conn.in_transaction
        conn.rollback()

    with pytest.raises(StrategyError, match="current|stale"):
        load_voting_candidate_search_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=descriptor["artifact_id"],
            expected_artifact_content_hash=descriptor["content_hash"],
            expected_search_id=search["search_id"],
            expected_search_content_hash=search["content_hash"],
        )


def test_historical_search_ignores_later_workspace_ui_revision(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    descriptor = fixture["search"]["artifacts"][0]
    _advance_workspace_ui(fixture)
    _advance_pool(fixture)

    binding = load_historical_voting_candidate_search_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
    )

    assert binding.result == fixture["search"]["search_result"]


def test_historical_search_rejects_artifact_file_tamper(tmp_path: Path) -> None:
    fixture = _searched_fixture(tmp_path)
    record = _search_record(fixture)
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(StrategyError, match="content hash|artifact"):
        _load_historical_search(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("search_id", "voting-search-" + "f" * 32),
        ("requirement_bindings", {}),
    ],
)
def test_historical_search_rejects_provenance_tamper(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _searched_fixture(tmp_path)
    record = _search_record(fixture)
    provenance = dict(record["provenance"])
    provenance[field] = value
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
            (
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                record["id"],
            ),
        )

    with pytest.raises(StrategyError, match="provenance|requirement"):
        _load_historical_search(fixture)


def test_historical_search_rejects_pool_revision_link_tamper(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "DROP TRIGGER trg_strategy_candidate_pool_revisions_immutable_update"
        )
        conn.execute(
            """
            UPDATE strategy_candidate_pool_revisions
               SET artifact_content_hash = ?
             WHERE id = ?
            """,
            ("0" * 64, fixture["pool"]["revision_id"]),
        )

    with pytest.raises(StrategyError, match="artifact|revision"):
        _load_historical_search(fixture)


def test_historical_search_rejects_candidate_source_registry_tamper(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    source_id = fixture["pool"]["entries"][0]["source"]["artifact_id"]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            ("0" * 64, source_id),
        )

    with pytest.raises(StrategyError, match="source|artifact|content hash"):
        _load_historical_search(fixture)


def test_historical_search_rejects_sample_artifact_registry_tamper(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    sample_id = _search_record(fixture)["provenance"]["sample_design_ref"][
        "artifact_id"
    ]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            ("0" * 64, sample_id),
        )

    with pytest.raises(StrategyError, match="sample-design|artifact"):
        _load_historical_search(fixture)


def test_historical_search_rejects_dataset_registry_tamper(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    dataset_id = _search_record(fixture)["provenance"]["dataset_binding"][
        "dataset_id"
    ]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE datasets SET row_count = row_count + 1 WHERE id = ?",
            (dataset_id,),
        )

    with pytest.raises(StrategyError, match="dataset|metadata"):
        _load_historical_search(fixture)


def test_historical_search_rejects_dataset_byte_tamper(tmp_path: Path) -> None:
    fixture = _searched_fixture(tmp_path)
    dataset_id = _search_record(fixture)["provenance"]["dataset_binding"][
        "dataset_id"
    ]
    path = Path(fixture["runtime"].registry.resolve_verified_path(dataset_id))
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(StrategyError, match="dataset|content hash"):
        _load_historical_search(fixture)


def test_voting_search_reader_uses_bounded_descriptor_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"bounded":true}'
    path = (tmp_path / "search.json").absolute()
    path.write_bytes(raw)
    read_sizes: list[int] = []
    original_read = os.read

    def tracked_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(descriptor, size)

    def reject_unbounded_read(_path: Path) -> bytes:
        raise AssertionError("Path.read_bytes must not be used")

    monkeypatch.setattr(search_tools.os, "read", tracked_read)
    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_read)

    loaded = search_tools._read_exact_file(
        path,
        root=tmp_path.absolute(),
        expected_content_hash=hashlib.sha256(raw).hexdigest(),
    )

    assert loaded == raw
    assert read_sizes
    assert max(read_sizes) <= 1024 * 1024


def test_voting_search_reader_uses_windows_backend_without_posix_open_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"windows":true}'
    root = tmp_path.absolute()
    path = (root / "search.json").absolute()
    path.write_bytes(raw)
    opened: list[tuple[Path, Path]] = []
    original_open = os.open

    def open_windows(candidate: Path, *, root: Path) -> int:
        opened.append((candidate, root))
        return original_open(candidate, os.O_RDONLY)

    monkeypatch.setattr(
        search_tools,
        "_open_file_beneath_root_windows",
        open_windows,
        raising=False,
    )
    monkeypatch.setattr(search_tools.os, "name", "nt")
    monkeypatch.delattr(search_tools.os, "O_DIRECTORY", raising=False)
    monkeypatch.delattr(search_tools.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(search_tools.os, "supports_dir_fd", ())

    loaded = search_tools._read_exact_file(
        path,
        root=root,
        expected_content_hash=hashlib.sha256(raw).hexdigest(),
    )

    assert loaded == raw
    assert opened == [(path, root), (path, root)]


@pytest.mark.parametrize("location", ["ancestor", "final"])
@pytest.mark.parametrize("unsafe_kind", ["symlink", "junction", "reparse"])
def test_voting_search_windows_backend_rejects_every_reparse_path_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    unsafe_kind: str,
) -> None:
    root = (tmp_path / "tasks").absolute()
    task_dir = root / "task-owned"
    output_dir = task_dir / "strategy_voting_candidate_searches"
    path = output_dir / "search.json"
    unsafe_path = task_dir if location == "ancestor" else path
    identities = {
        root: 1,
        task_dir: 2,
        output_dir: 3,
        path: 4,
    }

    def fake_lstat(candidate: Path) -> SimpleNamespace:
        candidate = Path(candidate)
        is_file = candidate == path
        mode = (stat.S_IFREG if is_file else stat.S_IFDIR) | 0o700
        attributes = 0
        if candidate == unsafe_path:
            if unsafe_kind == "symlink":
                mode = stat.S_IFLNK | 0o700
            elif unsafe_kind == "reparse":
                attributes = 0x400
        return SimpleNamespace(
            st_dev=1,
            st_ino=identities[candidate],
            st_mode=mode,
            st_uid=0,
            st_gid=0,
            st_size=0,
            st_mtime_ns=0,
            st_ctime_ns=0,
            st_file_attributes=attributes,
        )

    monkeypatch.setattr(search_tools.os, "lstat", fake_lstat)
    monkeypatch.setattr(
        search_tools.os.path,
        "isjunction",
        lambda candidate: (
            unsafe_kind == "junction" and Path(candidate) == unsafe_path
        ),
        raising=False,
    )

    with pytest.raises(StrategyError, match="symlink|junction|reparse"):
        search_tools._open_file_beneath_root_windows(path, root=root)


def test_voting_search_windows_backend_reads_regular_artifact_with_handle_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"windows-handle":true}'
    root = (tmp_path / "tasks").absolute()
    output_dir = (
        root
        / "task-owned"
        / "strategy_voting_candidate_searches"
    )
    output_dir.mkdir(parents=True)
    path = output_dir / "search.json"
    path.write_bytes(raw)
    opened: list[tuple[Path, bool, int]] = []
    closed: list[int] = []
    handle_paths: dict[int, tuple[Path, bool]] = {}
    next_handle = 100

    def open_handle(candidate: Path, *, directory: bool) -> int:
        nonlocal next_handle
        handle = next_handle
        next_handle += 1
        opened.append((candidate, directory, handle))
        handle_paths[handle] = (candidate, directory)
        return handle

    def handle_attributes(handle: int) -> int:
        return 0x10 if handle_paths[handle][1] else 0

    def transfer_handle(handle: int) -> int:
        candidate, directory = handle_paths[handle]
        assert directory is False
        return os.open(candidate, os.O_RDONLY)

    monkeypatch.setattr(
        search_tools,
        "_open_windows_path_handle",
        open_handle,
        raising=False,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_file_attributes",
        handle_attributes,
        raising=False,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_to_descriptor",
        transfer_handle,
        raising=False,
    )
    monkeypatch.setattr(
        search_tools,
        "_close_windows_handle",
        lambda handle: closed.append(handle),
        raising=False,
    )
    monkeypatch.setattr(search_tools.os, "name", "nt")

    loaded = search_tools._read_exact_file(
        path,
        root=root,
        expected_content_hash=hashlib.sha256(raw).hexdigest(),
    )

    assert loaded == raw
    assert [item[:2] for item in opened] == [
        (root, True),
        (root / "task-owned", True),
        (output_dir, True),
        (path, False),
    ] * 2
    assert len(closed) == 6
    assert all(handle_paths[handle][1] for handle in closed)


@pytest.mark.parametrize("location", ["ancestor", "final"])
def test_voting_search_windows_backend_rejects_handle_reparse_and_closes_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    root = (tmp_path / "tasks").absolute()
    task_dir = root / "task-owned"
    output_dir = task_dir / "strategy_voting_candidate_searches"
    output_dir.mkdir(parents=True)
    path = output_dir / "search.json"
    path.write_bytes(b"{}")
    unsafe_path = task_dir if location == "ancestor" else path
    handle_paths: dict[int, tuple[Path, bool]] = {}
    opened: list[int] = []
    closed: list[int] = []

    def open_handle(candidate: Path, *, directory: bool) -> int:
        handle = 100 + len(opened)
        opened.append(handle)
        handle_paths[handle] = (candidate, directory)
        return handle

    def handle_attributes(handle: int) -> int:
        candidate, directory = handle_paths[handle]
        attributes = 0x10 if directory else 0
        if candidate == unsafe_path:
            attributes |= 0x400
        return attributes

    monkeypatch.setattr(
        search_tools,
        "_open_windows_path_handle",
        open_handle,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_file_attributes",
        handle_attributes,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_to_descriptor",
        lambda _handle: pytest.fail("reparse handle must not become a descriptor"),
    )
    monkeypatch.setattr(
        search_tools,
        "_close_windows_handle",
        lambda handle: closed.append(handle),
    )

    with pytest.raises(StrategyError, match="symlink|junction|reparse"):
        search_tools._open_file_beneath_root_windows(path, root=root)

    assert sorted(closed) == sorted(opened)


def test_voting_search_windows_backend_cleanup_failure_closes_every_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "tasks").absolute()
    output_dir = root / "task-owned" / "searches"
    output_dir.mkdir(parents=True)
    path = output_dir / "search.json"
    path.write_bytes(b"{}")
    handle_paths: dict[int, tuple[Path, bool]] = {}
    close_attempts: list[int] = []
    transferred: list[int] = []

    def open_handle(candidate: Path, *, directory: bool) -> int:
        handle = 100 + len(handle_paths)
        handle_paths[handle] = (candidate, directory)
        return handle

    def handle_attributes(handle: int) -> int:
        return 0x10 if handle_paths[handle][1] else 0

    def transfer_handle(handle: int) -> int:
        descriptor = os.open(handle_paths[handle][0], os.O_RDONLY)
        transferred.append(descriptor)
        return descriptor

    def fail_first_close(handle: int) -> None:
        close_attempts.append(handle)
        if len(close_attempts) == 1:
            raise OSError("forced Windows cleanup failure")

    monkeypatch.setattr(
        search_tools,
        "_open_windows_path_handle",
        open_handle,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_file_attributes",
        handle_attributes,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_to_descriptor",
        transfer_handle,
    )
    monkeypatch.setattr(
        search_tools,
        "_close_windows_handle",
        fail_first_close,
    )

    try:
        with pytest.raises(OSError, match="forced Windows cleanup failure"):
            search_tools._open_file_beneath_root_windows(path, root=root)

        assert len(close_attempts) == 3
        assert len(transferred) == 1
        with pytest.raises(OSError):
            os.fstat(transferred[0])
    finally:
        for descriptor in transferred:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_voting_search_windows_backend_cleanup_does_not_mask_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (tmp_path / "tasks").absolute()
    output_dir = root / "task-owned" / "searches"
    output_dir.mkdir(parents=True)
    path = output_dir / "search.json"
    path.write_bytes(b"{}")
    handle_paths: dict[int, tuple[Path, bool]] = {}
    close_attempts: list[int] = []
    transferred: list[int] = []

    def open_handle(candidate: Path, *, directory: bool) -> int:
        handle = 100 + len(handle_paths)
        handle_paths[handle] = (candidate, directory)
        return handle

    def transfer_handle(handle: int) -> int:
        descriptor = os.open(handle_paths[handle][0], os.O_RDONLY)
        transferred.append(descriptor)
        return descriptor

    def fail_close(handle: int) -> None:
        close_attempts.append(handle)
        raise OSError(f"forced cleanup failure for {handle}")

    def fail_chain_check(*_args, **_kwargs) -> None:
        raise StrategyError("primary Windows identity failure")

    monkeypatch.setattr(
        search_tools,
        "_open_windows_path_handle",
        open_handle,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_file_attributes",
        lambda handle: 0x10 if handle_paths[handle][1] else 0,
    )
    monkeypatch.setattr(
        search_tools,
        "_windows_handle_to_descriptor",
        transfer_handle,
    )
    monkeypatch.setattr(
        search_tools,
        "_require_windows_path_chain_unchanged",
        fail_chain_check,
    )
    monkeypatch.setattr(
        search_tools,
        "_close_windows_handle",
        fail_close,
    )

    try:
        with pytest.raises(
            StrategyError,
            match="primary Windows identity failure",
        ):
            search_tools._open_file_beneath_root_windows(path, root=root)

        assert len(close_attempts) == 3
        assert len(transferred) == 1
        with pytest.raises(OSError):
            os.fstat(transferred[0])
    finally:
        for descriptor in transferred:
            try:
                os.close(descriptor)
            except OSError:
                pass


def test_voting_search_reader_rejects_path_swap_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b'{"stable":true}'
    path = (tmp_path / "search.json").absolute()
    replacement = (tmp_path / "replacement.json").absolute()
    path.write_bytes(raw)
    replacement.write_bytes(raw)
    original_read = os.read
    swapped = False

    def swap_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, size)
        if chunk and not swapped:
            os.replace(replacement, path)
            swapped = True
        return chunk

    monkeypatch.setattr(search_tools.os, "read", swap_after_first_read)

    with pytest.raises(StrategyError, match="file is invalid"):
        search_tools._read_exact_file(
            path,
            root=tmp_path.absolute(),
            expected_content_hash=hashlib.sha256(raw).hexdigest(),
        )
    assert swapped is True


def test_voting_search_reader_rejects_ancestor_directory_symlink(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "root").absolute()
    outside = (tmp_path / "outside").absolute()
    root.mkdir()
    outside.mkdir()
    raw = b'{"outside":true}'
    outside_path = outside / "search.json"
    outside_path.write_bytes(raw)
    (root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(StrategyError, match="invalid|unavailable|canonical"):
        search_tools._read_exact_file(
            root / "linked" / "search.json",
            root=root,
            expected_content_hash=hashlib.sha256(raw).hexdigest(),
        )


def test_historical_sample_design_v2_survives_workspace_ui_revision(
    tmp_path: Path,
) -> None:
    from tests.test_strategy_sample_design_v2_tool import _setup as _v2_setup

    fixture = _v2_setup(tmp_path)
    output = run_materialize_sample_design_v2(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    records = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_for_task(fixture["task"].id)
    membership = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_MEMBERSHIP_ARTIFACT_KIND
    )
    bundle = next(
        item
        for item in records
        if item["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    )
    _advance_workspace_ui(fixture)

    historical = load_historical_strategy_sample_design_v2_artifacts(
        fixture["runtime"],
        task_id=fixture["task"].id,
        membership_artifact_id=membership["id"],
        expected_membership_artifact_content_hash=membership["content_hash"],
        bundle_artifact_id=bundle["id"],
        expected_bundle_artifact_content_hash=bundle["content_hash"],
        expected_bundle_id=output["bundle_id"],
        expected_sample_design_id=output["sample_design_id"],
        expected_sample_design_content_hash=output[
            "sample_design_content_hash"
        ],
    )
    assert historical.bundle == output["bundle"]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_historical_strategy_sample_design_v2_artifact_binding_on_connection(
            conn,
            historical,
        )
        conn.rollback()

    with pytest.raises(StrategyError, match="DataWorkspace|workspace"):
        load_strategy_sample_design_v2_artifacts(
            fixture["runtime"],
            task_id=fixture["task"].id,
            membership_artifact_id=membership["id"],
            expected_membership_artifact_content_hash=membership[
                "content_hash"
            ],
            bundle_artifact_id=bundle["id"],
            expected_bundle_artifact_content_hash=bundle["content_hash"],
            expected_bundle_id=output["bundle_id"],
            expected_sample_design_id=output["sample_design_id"],
            expected_sample_design_content_hash=output[
                "sample_design_content_hash"
            ],
        )


@pytest.mark.slow
def test_historical_scorecard_search_survives_workspace_ui_revision(
    tmp_path: Path,
) -> None:
    from tests.test_strategy_voting_scorecard import (
        _two_scorecard_pool_entries,
    )

    fixture = _two_scorecard_pool_entries(tmp_path)
    task_id = fixture["fx"]["task"].id
    search_inputs = search_tools.resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=task_id,
        user_controls={
            "strategy_type": "approval",
            "member_count": 2,
            "n": 1,
            "objective": {
                "metric": "bad_capture_rate",
                "direction": "maximize",
            },
            "constraints": [
                {"metric": "hit_share", "operator": "gte", "value": 0.01}
            ],
            "include_rule_ids": [],
            "exclude_rule_ids": [],
            "max_combinations": 10,
        },
    )
    search = search_tools.run_search_voting_candidates(
        search_inputs,
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    descriptor = search["artifacts"][0]
    pool = fixture["pool"]
    changed = run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
            "rule_id": pool["entries"][0]["rule_id"],
            "action": {
                "type": "review",
                "value": "review",
                "reason_code": "HISTORICAL_SCORE_REQUIREMENT",
                "stop": True,
            },
        },
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    assert changed["revision"] == pool["revision"] + 1
    _advance_workspace_ui(fixture["fx"])

    historical = load_historical_voting_candidate_search_artifact(
        fixture["runtime"],
        task_id=task_id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
    )
    assert historical.resolved_requirements is not None
    assert historical.resolved_requirements.evidence_bindings
    with connect(fixture["fx"]["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_historical_voting_candidate_search_artifact_binding_on_connection(
            conn,
            historical,
        )
        conn.rollback()

    with pytest.raises(StrategyError, match="current|stale|DataWorkspace"):
        load_voting_candidate_search_artifact(
            fixture["runtime"],
            task_id=task_id,
            artifact_id=descriptor["artifact_id"],
            expected_artifact_content_hash=descriptor["content_hash"],
            expected_search_id=search["search_id"],
            expected_search_content_hash=search["content_hash"],
        )

    requirement_provenance = json.loads(
        json.dumps(historical.artifact_provenance)
    )
    requirement_provenance["requirement_bindings"]["requirements_hash"] = (
        "0" * 64
    )
    tampered_requirement_binding = replace(
        historical,
        artifact_provenance=requirement_provenance,
        artifact_provenance_json=json.dumps(
            requirement_provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    with connect(fixture["fx"]["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="provenance|requirement"):
            require_historical_voting_candidate_search_artifact_binding_on_connection(
                conn,
                tampered_requirement_binding,
            )
        conn.rollback()

    score_binding = historical.resolved_requirements.evidence_bindings[0]
    with connect(fixture["fx"]["settings"].db_path) as conn:
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE task_artifacts SET content_hash = ? WHERE id = ?",
            ("0" * 64, score_binding.vector_record["id"]),
        )
        with pytest.raises(
            StrategyError,
            match="score|TaskArtifact|content hash|artifact",
        ):
            require_historical_voting_candidate_search_artifact_binding_on_connection(
                conn,
                historical,
            )
        conn.rollback()
