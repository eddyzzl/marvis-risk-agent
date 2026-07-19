from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from marvis.db import TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.packs.strategy.automatic_tree_asset import (
    build_automatic_tree_asset,
    canonical_automatic_tree_asset_json,
)
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION,
    AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
    AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
    canonical_automatic_tree_leaf_fragment_json,
)
from marvis.packs.strategy import automatic_tree_leaf_tools as leaf_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _asset(task_id: str) -> dict:
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "bad": [0, 0, 1, 0, 1, 1],
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=["x", "z"],
        target_col="bad",
        max_depth=2,
        min_leaf_count=1,
    )
    return build_automatic_tree_asset(
        tree,
        task_id=task_id,
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=[f"workspace:{task_id}:3", "dataset:dataset-labelled"],
    )


def _task(repository: TaskRepository, name: str):
    return repository.create_task(
        TaskCreate(
            model_name=name,
            model_version="dev",
            validator="qa",
            source_dir=f"/tmp/{name}",
            task_type="strategy",
            target_col="bad",
        )
    )


def _fixture(tmp_path: Path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = _task(tasks, "leaf-selection")
    foreign = _task(tasks, "foreign-leaf-selection")
    repository = TaskArtifactRepository(settings.db_path)
    runtime = SimpleNamespace(settings=settings, task_artifacts=repository)
    ctx = SimpleNamespace(task_id=task.id)
    asset = _asset(task.id)
    content = canonical_automatic_tree_asset_json(asset).encode("utf-8")
    content_hash = hashlib.sha256(content).hexdigest()
    source_path = leaf_tools.canonical_automatic_tree_source_path(
        settings.tasks_dir,
        task_id=task.id,
        asset_id=asset["asset_id"],
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(content)
    provenance = leaf_tools.automatic_tree_source_provenance_from_asset(asset)
    record = repository.register(
        task_id=task.id,
        kind=AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        path=str(source_path),
        content_hash=content_hash,
        origin_tool=AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        provenance=provenance,
    )
    inputs = {
        "source_artifact_id": record["id"],
        "expected_artifact_content_hash": content_hash,
        "expected_asset_id": asset["asset_id"],
        "expected_asset_hash": asset["asset_hash"],
        "expected_tree_result_hash": asset["tree_result"]["result_hash"],
        "leaf_id": asset["fragments"][0]["leaf_id"],
    }
    return SimpleNamespace(
        settings=settings,
        task=task,
        foreign=foreign,
        repository=repository,
        runtime=runtime,
        ctx=ctx,
        asset=asset,
        content=content,
        source_path=source_path,
        source_record=record,
        inputs=inputs,
    )


def _update_artifact(fx, **changes: object) -> None:
    assignments = ", ".join(f"{field} = ?" for field in changes)
    with fx.repository.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        # Deliberately bypass the production immutability trigger to exercise
        # the Tool's independent fail-closed verification against a corrupted
        # registry snapshot.
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            f"UPDATE task_artifacts SET {assignments} WHERE id = ?",  # noqa: S608
            (*changes.values(), fx.source_record["id"]),
        )


def _selection_records(fx) -> list[dict]:
    return [
        record
        for record in fx.repository.list_for_task(fx.task.id)
        if record["kind"] == AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND
    ]


def test_materialize_leaf_is_canonical_exact_and_idempotent(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)

    first = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        fx.inputs, fx.ctx, fx.runtime
    )
    repeated = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        fx.inputs, fx.ctx, fx.runtime
    )

    assert repeated == first
    assert set(first) == {
        "schema_version",
        "selection_id",
        "selection_hash",
        "selection_reason",
        "tree_asset_id",
        "tree_asset_hash",
        "tree_result_hash",
        "leaf_id",
        "fragment_id",
        "fragment_hash",
        "rule_id",
        "effect_id",
        "artifacts",
    }
    assert first["schema_version"] == leaf_tools.TOOL_SCHEMA_VERSION
    assert first["selection_reason"] is None
    assert len(first["artifacts"]) == 1
    descriptor = first["artifacts"][0]
    assert set(descriptor) == {
        "artifact_id",
        "kind",
        "format",
        "filename",
        "content_hash",
        "download_url",
    }
    assert "path" not in descriptor
    assert descriptor["kind"] == AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_KIND
    assert descriptor["format"] == "json"

    [record] = _selection_records(fx)
    expected_path = leaf_tools.canonical_automatic_tree_leaf_selection_path(
        fx.settings.tasks_dir,
        task_id=fx.task.id,
        selection_id=first["selection_id"],
    )
    assert Path(record["path"]) == expected_path
    persisted = expected_path.read_bytes()
    selection = json.loads(persisted.decode("utf-8"))
    assert persisted == canonical_automatic_tree_leaf_fragment_json(selection).encode(
        "utf-8"
    )
    assert hashlib.sha256(persisted).hexdigest() == record["content_hash"]
    assert record["origin_tool"] == AUTOMATIC_TREE_LEAF_FRAGMENT_ORIGIN_TOOL
    assert set(record["provenance"]) == leaf_tools.SELECTION_PROVENANCE_FIELDS
    assert record["provenance"] == (
        leaf_tools.automatic_tree_leaf_selection_provenance(selection)
    )
    assert record["provenance"]["schema_version"] == (
        AUTOMATIC_TREE_LEAF_FRAGMENT_ARTIFACT_SCHEMA_VERSION
    )
    forbidden = {"action", "pool", "metrics", "condition", "requirements"}
    assert forbidden.isdisjoint(first)


def test_different_leaf_or_reason_is_a_distinct_audit_artifact(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    first = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        fx.inputs, fx.ctx, fx.runtime
    )
    other_leaf = fx.asset["fragments"][1]["leaf_id"]
    second = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        {**fx.inputs, "leaf_id": other_leaf}, fx.ctx, fx.runtime
    )
    reasoned = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        {**fx.inputs, "selection_reason": "  analyst\tselected  "},
        fx.ctx,
        fx.runtime,
    )

    assert (
        len({first["selection_id"], second["selection_id"], reasoned["selection_id"]})
        == 3
    )
    assert reasoned["selection_reason"] == "analyst selected"
    assert len(_selection_records(fx)) == 3


@pytest.mark.parametrize(
    "field",
    [
        "condition",
        "requirements",
        "metrics",
        "action",
        "fragment",
        "rule",
        "effect",
        "selection_id",
        "selection_hash",
    ],
)
def test_input_whitelist_rejects_caller_derived_fields(
    tmp_path: Path, field: str
) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match="unsupported"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**fx.inputs, field: "forged"}, fx.ctx, fx.runtime
        )


def test_selection_reason_is_bounded(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match="500"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**fx.inputs, "selection_reason": "x" * 501}, fx.ctx, fx.runtime
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_artifact_content_hash", "e" * 64),
        ("expected_asset_id", "candidate-asset-" + "e" * 32),
        ("expected_asset_hash", "e" * 64),
        ("expected_tree_result_hash", "e" * 64),
        ("leaf_id", "leaf-unknown"),
    ],
)
def test_expected_bindings_and_leaf_are_fail_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**fx.inputs, field: value}, fx.ctx, fx.runtime
        )


def test_unknown_and_foreign_source_artifacts_are_rejected(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    with pytest.raises(StrategyError, match="not found"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**fx.inputs, "source_artifact_id": "unknown"}, fx.ctx, fx.runtime
        )

    foreign_asset = _asset(fx.foreign.id)
    content = canonical_automatic_tree_asset_json(foreign_asset).encode("utf-8")
    path = leaf_tools.canonical_automatic_tree_source_path(
        fx.settings.tasks_dir,
        task_id=fx.foreign.id,
        asset_id=foreign_asset["asset_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    foreign_record = fx.repository.register(
        task_id=fx.foreign.id,
        kind=AUTOMATIC_TREE_SOURCE_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(content).hexdigest(),
        origin_tool=AUTOMATIC_TREE_SOURCE_ARTIFACT_ORIGIN_TOOL,
        provenance=leaf_tools.automatic_tree_source_provenance_from_asset(
            foreign_asset
        ),
    )
    with pytest.raises(StrategyError, match="not found"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {**fx.inputs, "source_artifact_id": foreign_record["id"]},
            fx.ctx,
            fx.runtime,
        )


@pytest.mark.parametrize("column", ["kind", "origin_tool"])
def test_wrong_source_kind_or_origin_is_rejected(tmp_path: Path, column: str) -> None:
    fx = _fixture(tmp_path)
    _update_artifact(fx, **{column: "forged"})
    with pytest.raises(StrategyError):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "unexpected": True},
        lambda value: {**value, "schema_version": "forged"},
        lambda value: {**value, "asset_hash": "e" * 64},
        lambda value: {**value, "task_id": "foreign"},
    ],
)
def test_source_provenance_must_be_exact_and_asset_derived(
    tmp_path: Path, mutation
) -> None:
    fx = _fixture(tmp_path)
    provenance = mutation(deepcopy(fx.source_record["provenance"]))
    _update_artifact(
        fx,
        provenance_json=json.dumps(
            provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    with pytest.raises(StrategyError):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )


@pytest.mark.parametrize("mode", ["noncanonical", "duplicate", "invalid_utf8"])
def test_source_json_must_be_strict_utf8_duplicate_safe_and_canonical(
    tmp_path: Path, mode: str
) -> None:
    fx = _fixture(tmp_path)
    if mode == "noncanonical":
        changed = fx.content + b"\n"
    elif mode == "duplicate":
        changed = (
            b'{"asset_hash":"'
            + fx.asset["asset_hash"].encode("ascii")
            + b'",'
            + fx.content[1:]
        )
    else:
        changed = fx.content + b"\xff"
    fx.source_path.write_bytes(changed)
    changed_hash = hashlib.sha256(changed).hexdigest()
    _update_artifact(fx, content_hash=changed_hash)
    with pytest.raises(StrategyError):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            {
                **fx.inputs,
                "expected_artifact_content_hash": changed_hash,
            },
            fx.ctx,
            fx.runtime,
        )


def test_source_path_must_be_exact_absolute_and_regular(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    _update_artifact(fx, path="relative/tree.json")
    with pytest.raises(StrategyError, match="canonical"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )

    fx = _fixture(tmp_path / "second")
    fx.source_path.unlink()
    fx.source_path.mkdir()
    with pytest.raises(StrategyError, match="regular"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )


def test_source_path_rejects_symlink_ancestor(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    source_dir = fx.source_path.parent
    real_dir = source_dir.with_name("real-automatic-trees")
    source_dir.rename(real_dir)
    source_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(StrategyError, match="symlink"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )


def test_under_lock_source_drift_rolls_back_selection(
    monkeypatch, tmp_path: Path
) -> None:
    fx = _fixture(tmp_path)
    original = leaf_tools.ArtifactUnitOfWork.stage_file

    def stage_and_drift(self, root, final_name):
        staged = original(self, root, final_name)
        fx.source_path.write_bytes(fx.content + b"\n")
        return staged

    monkeypatch.setattr(leaf_tools.ArtifactUnitOfWork, "stage_file", stage_and_drift)
    with pytest.raises(StrategyError, match="changed|canonical|hash"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )
    assert _selection_records(fx) == []
    assert not list(
        (fx.settings.tasks_dir / fx.task.id).glob(
            "strategy_automatic_tree_leaf_fragments/*.json"
        )
    )


def test_registration_failure_rolls_back_promoted_file(
    monkeypatch, tmp_path: Path
) -> None:
    fx = _fixture(tmp_path)

    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(
        fx.repository,
        "register_on_connection",
        fail_registration,
    )
    with pytest.raises(RuntimeError, match="registration failed"):
        leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )
    assert _selection_records(fx) == []
    assert not list(
        (fx.settings.tasks_dir / fx.task.id).glob(
            "strategy_automatic_tree_leaf_fragments/*.json"
        )
    )


def test_writer_lock_is_acquired_before_promote(monkeypatch, tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    original = leaf_tools.ArtifactUnitOfWork.promote_all
    observed = {"locked": False}

    def assert_locked(self):
        contender = sqlite3.connect(fx.settings.db_path, timeout=0)
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")
            observed["locked"] = True
        finally:
            contender.close()
        return original(self)

    monkeypatch.setattr(leaf_tools.ArtifactUnitOfWork, "promote_all", assert_locked)
    leaf_tools.run_materialize_automatic_tree_leaf_fragment(
        fx.inputs, fx.ctx, fx.runtime
    )
    assert observed["locked"] is True


def test_identical_concurrent_materialization_is_idempotent(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)

    def run():
        return leaf_tools.run_materialize_automatic_tree_leaf_fragment(
            fx.inputs, fx.ctx, fx.runtime
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run(), range(2)))
    assert results[0] == results[1]
    assert len(_selection_records(fx)) == 1


def test_failed_writer_rolls_back_before_identical_peer_promotes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    failed_writer_exited_db = threading.Event()
    release_failed_writer = threading.Event()
    failure_injected = threading.Event()
    original_register = TaskArtifactRepository.register_on_connection
    original_transaction = TaskArtifactRepository.transaction

    def fail_first_writer(self, conn, **kwargs):
        if threading.current_thread().name == "failing-writer":
            failure_injected.set()
            raise RuntimeError("injected post-promotion registration failure")
        return original_register(self, conn, **kwargs)

    @contextmanager
    def pause_failed_writer_after_db_exit(self):
        try:
            with original_transaction(self) as conn:
                yield conn
        finally:
            if (
                threading.current_thread().name == "failing-writer"
                and failure_injected.is_set()
            ):
                failed_writer_exited_db.set()
                if not release_failed_writer.wait(timeout=15):
                    raise RuntimeError("timed out waiting to release failed writer")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_first_writer,
    )
    monkeypatch.setattr(
        TaskArtifactRepository,
        "transaction",
        pause_failed_writer_after_db_exit,
    )
    outputs: dict[str, dict] = {}
    failures: dict[str, BaseException] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = leaf_tools.run_materialize_automatic_tree_leaf_fragment(
                fx.inputs,
                fx.ctx,
                fx.runtime,
            )
        except BaseException as exc:  # captured for main-thread assertions
            failures[name] = exc

    failing = threading.Thread(
        target=invoke,
        args=("failing",),
        name="failing-writer",
    )
    peer = threading.Thread(target=invoke, args=("peer",), name="peer-writer")
    failing.start()
    assert failed_writer_exited_db.wait(timeout=20)
    peer.start()
    peer.join(timeout=30)
    assert not peer.is_alive()
    assert "peer" not in failures
    release_failed_writer.set()
    failing.join(timeout=30)

    assert not failing.is_alive()
    assert isinstance(failures.get("failing"), RuntimeError)
    assert outputs["peer"]["selection_id"].startswith("automatic-tree-leaf-selection-")
    [record] = _selection_records(fx)
    persisted = Path(record["path"])
    assert persisted.is_file()
    assert hashlib.sha256(persisted.read_bytes()).hexdigest() == record["content_hash"]
