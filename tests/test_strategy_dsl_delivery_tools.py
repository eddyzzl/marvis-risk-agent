from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from marvis.data.workspace import (
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.files import sha256_file
from marvis.packs.strategy.dsl_delivery import (
    validate_strategy_delivery_equivalence,
)
from marvis.packs.strategy.dsl_delivery_tools import (
    DELIVERY_ARTIFACT_KINDS,
    DELIVERY_AUDIT_KIND,
    StrategyDeliveryToolError,
    run_export_strategy_delivery,
    validate_export_strategy_delivery_tool_output,
    validate_strategy_delivery_artifact_records,
)
import marvis.packs.strategy.dsl_delivery_tools as delivery_tools
from marvis.packs.strategy.tools import _runtime
from marvis.repositories.data_workspace import DataWorkspaceRepository
from test_strategy_apply_tool import _runtime_fixture


def _inputs(fixture: tuple) -> dict:
    settings, _task, _registry, dataset, strategy, _ctx = fixture
    strategies = _runtime(fixture[-1]).strategies
    meta = strategies.get_strategy_meta(strategy.id)
    assert meta is not None
    spec_hash = strategies.get_strategy_spec_hash(strategy.id)
    assert spec_hash is not None
    workspace = DataWorkspaceRepository(settings.db_path).get_or_default(
        fixture[1].id
    )
    return {
        "strategy_ref": {
            "strategy_id": strategy.id,
            "expected_strategy_type": strategy.strategy_type,
            "expected_version": meta["version"],
            "expected_spec_hash": spec_hash,
        },
        "dataset_ref": {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
        },
        "workspace_ref": {
            "revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "active_dataset_id": workspace.active_dataset_id,
            "active_dataset_content_hash": (
                workspace.active_dataset_content_hash
            ),
        },
        "maximum_equivalence_rows": 4096,
    }


def _run(fixture: tuple, inputs: dict | None = None) -> tuple[dict, object]:
    runtime = _runtime(fixture[-1])
    request = _inputs(fixture) if inputs is None else inputs
    return (
        run_export_strategy_delivery(request, fixture[-1], runtime),
        runtime,
    )


def _artifact_rows(runtime, task_id: str) -> list[dict]:
    return [
        row
        for row in runtime.task_artifacts.list_for_task(task_id)
        if row["kind"] in set(DELIVERY_ARTIFACT_KINDS.values())
    ]


def _artifact_projections(output: dict) -> dict[str, dict[str, str]]:
    return {
        name: {
            "artifact_id": output["artifacts"][index]["artifact_id"],
            "content_hash": output["artifacts"][index]["content_hash"],
        }
        for index, name in enumerate(
            ("python", "sql", "strategy_json", "equivalence_json")
        )
    }


def _artifact_record_mapping(runtime, task_id: str) -> dict[str, dict]:
    rows = _artifact_rows(runtime, task_id)
    return {
        name: next(
            row for row in rows if row["kind"] == DELIVERY_ARTIFACT_KINDS[name]
        )
        for name in (
            "python",
            "sql",
            "strategy_json",
            "equivalence_json",
        )
    }


def _audit_count(settings, *, delivery_id: str) -> int:
    with sqlite3.connect(settings.db_path) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = ? AND target_ref = ?",
                (DELIVERY_AUDIT_KIND, delivery_id),
            ).fetchone()[0]
        )


@pytest.mark.parametrize(
    "strategy_type",
    ["approval", "reject", "limit", "pricing", "segmentation"],
)
def test_export_strategy_delivery_publishes_authenticated_code_and_equivalence(
    tmp_path: Path,
    strategy_type: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, strategy_type)
    request = _inputs(fixture)

    output, runtime = _run(fixture, request)

    assert validate_export_strategy_delivery_tool_output(
        output,
        expected_task_id=fixture[1].id,
        expected_strategy_ref=request["strategy_ref"],
        expected_dataset_ref=request["dataset_ref"],
        expected_workspace_ref=request["workspace_ref"],
        expected_artifacts=_artifact_projections(output),
    ) == output
    assert output["strategy_type"] == strategy_type
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    assert {item["kind"] for item in output["artifacts"]} == set(
        DELIVERY_ARTIFACT_KINDS.values()
    )
    rows = _artifact_rows(runtime, fixture[1].id)
    assert len(rows) == 4
    for artifact in output["artifacts"]:
        row = next(item for item in rows if item["id"] == artifact["artifact_id"])
        path = Path(row["path"])
        assert path.is_file()
        assert sha256_file(path) == artifact["content_hash"]
        assert row["provenance"]["delivery_id"] == output["delivery_id"]
        assert row["provenance"]["strategy_ref"] == request["strategy_ref"]
        assert row["provenance"]["dataset_ref"] == request["dataset_ref"]
        assert row["provenance"]["workspace_ref"] == request["workspace_ref"]
    python_path = Path(
        next(
            row["path"]
            for row in rows
            if row["kind"] == DELIVERY_ARTIFACT_KINDS["python"]
        )
    )
    python_source = python_path.read_text(encoding="utf-8").lower()
    assert "from marvis" not in python_source
    assert "import marvis" not in python_source
    equivalence = output["equivalence"]
    assert validate_strategy_delivery_equivalence(
        equivalence,
        expected_strategy_spec_hash=request["strategy_ref"][
            "expected_spec_hash"
        ],
        expected_sample_hash=equivalence["sample_hash"],
        expected_content_hash=equivalence["content_hash"],
    ) == equivalence
    assert _audit_count(
        fixture[0],
        delivery_id=output["delivery_id"],
    ) == 1


def test_export_strategy_delivery_revalidates_workspace_in_publish_transaction(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    workspace_repo = DataWorkspaceRepository(fixture[0].db_path)
    workspace_repo.save(
        fixture[1].id,
        DataWorkspaceDraft(
            active_dataset_id=fixture[3].id,
            active_dataset_content_hash=fixture[3].content_hash,
        ),
        expected_revision=request["workspace_ref"]["revision"],
    )
    runtime = _runtime(fixture[-1])

    with pytest.raises(
        StrategyDeliveryToolError,
        match="DataWorkspace no longer matches",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0


def test_export_strategy_delivery_rejects_legacy_row_before_publication(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET dsl_json = NULL,
                   dsl_schema_version = NULL,
                   dsl_content_hash = NULL
             WHERE id = ?
            """,
            (fixture[4].id,),
        )
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])

    with pytest.raises(
        StrategyDeliveryToolError,
        match="canonical Strategy DSL snapshot",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    assert not delivery_root.exists()
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0


def test_export_strategy_delivery_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)

    first, runtime = _run(fixture, request)
    second, _runtime_again = _run(fixture, request)

    assert second == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


def test_export_strategy_delivery_concurrent_exact_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(
            pool.map(
                lambda _index: _run(fixture, request)[0],
                range(2),
            )
        )

    runtime = _runtime(fixture[-1])
    assert outputs[0] == outputs[1]
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=outputs[0]["delivery_id"],
    ) == 1


def test_export_strategy_delivery_rejects_tampered_registered_file(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    row = next(
        item
        for item in _artifact_rows(runtime, fixture[1].id)
        if item["kind"] == DELIVERY_ARTIFACT_KINDS["python"]
    )
    path = Path(row["path"])
    path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        StrategyDeliveryToolError,
        match="existing.*bytes|artifact.*changed",
    ):
        _run(fixture, request)

    assert path.read_text(encoding="utf-8") == "tampered\n"
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


def test_export_strategy_delivery_recovers_exact_promoted_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())


@pytest.mark.parametrize(
    "force_path_backend",
    [False, True],
    ids=["dirfd", "path-fallback"],
)
def test_export_strategy_delivery_recovers_exact_files_without_rows_or_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_path_backend: bool,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    if force_path_backend:
        monkeypatch.setattr(
            delivery_tools,
            "_supports_secure_delivery_dir_fds",
            lambda: False,
        )
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
        conn.execute(
            "DELETE FROM audit WHERE kind = ? AND target_ref = ?",
            (DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1
    assert all(
        Path(path).read_bytes() == raw for path, raw in original.items()
    )


def test_export_strategy_delivery_recovers_exact_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    retained_path = Path(rows[0]["path"])
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert retained_path.read_bytes() == original[str(retained_path)]
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(Path(path).read_bytes() == raw for path, raw in original.items())


def test_export_strategy_delivery_uses_path_safe_fallback_without_dir_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    monkeypatch.setattr(
        delivery_tools,
        "_supports_secure_delivery_dir_fds",
        lambda: False,
    )

    output, runtime = _run(fixture, request)

    assert output["strategy_ref"] == request["strategy_ref"]
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(
        Path(row["path"]).is_file()
        for row in _artifact_rows(runtime, fixture[1].id)
    )


def test_path_safe_fallback_preserves_partial_orphan_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    monkeypatch.setattr(
        delivery_tools,
        "_supports_secure_delivery_dir_fds",
        lambda: False,
    )
    first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    original = {
        str(row["path"]): Path(row["path"]).read_bytes()
        for row in rows
    }
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()

    replay, _runtime_again = _run(fixture, request)

    assert replay == first
    assert len(_artifact_rows(runtime, fixture[1].id)) == 4
    assert all(
        Path(path).read_bytes() == raw for path, raw in original.items()
    )


def test_path_safe_fallback_rejects_windows_reparse_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    original = delivery_tools._is_delivery_reparse_point

    def force_reparse(path, metadata):
        if Path(path).name == "strategy_delivery":
            return True
        return original(path, metadata)

    monkeypatch.setattr(
        delivery_tools,
        "_supports_secure_delivery_dir_fds",
        lambda: False,
    )
    monkeypatch.setattr(
        delivery_tools,
        "_is_delivery_reparse_point",
        force_reparse,
    )
    runtime = _runtime(fixture[-1])

    with pytest.raises(
        StrategyDeliveryToolError,
        match="junction|reparse point",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_rejects_drifted_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    _first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    retained_path = Path(rows[0]["path"])
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()
    retained_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(StrategyDeliveryToolError, match="artifact bytes changed"):
        _run(fixture, request)

    assert retained_path.read_text(encoding="utf-8") == "tampered\n"
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_rejects_symlinked_partial_orphan_set(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    _first, runtime = _run(fixture, request)
    rows = _artifact_rows(runtime, fixture[1].id)
    retained_path = Path(rows[0]["path"])
    outside = tmp_path / "outside-delivery.py"
    outside.write_bytes(retained_path.read_bytes())
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.executemany(
            "DELETE FROM task_artifacts WHERE id = ?",
            [(row["id"],) for row in rows],
        )
    for row in rows[1:]:
        Path(row["path"]).unlink()
    retained_path.unlink()
    retained_path.symlink_to(outside)

    with pytest.raises(
        StrategyDeliveryToolError,
        match="regular file|unavailable|must stay under",
    ):
        _run(fixture, request)

    assert retained_path.is_symlink()
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_rejects_symlinked_parent_directory(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    task_dir = fixture[0].tasks_dir / fixture[1].id
    redirected = fixture[0].tasks_dir / "other-task" / "redirect"
    task_dir.mkdir(parents=True, exist_ok=True)
    redirected.mkdir(parents=True, exist_ok=True)
    (task_dir / "strategy_delivery").symlink_to(
        redirected,
        target_is_directory=True,
    )
    runtime = _runtime(fixture[-1])

    with pytest.raises(StrategyDeliveryToolError, match="symlink"):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert list(redirected.iterdir()) == []
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_export_strategy_delivery_parent_swap_cannot_redirect_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    redirected = tmp_path / "redirected-delivery"
    detached = tmp_path / "detached-task"
    redirected.mkdir()
    original = delivery_tools._open_delivery_directory_chain
    swapped = False

    def open_then_swap(tasks_root, *, task_id, delivery_id):
        nonlocal swapped
        uow = original(
            tasks_root,
            task_id=task_id,
            delivery_id=delivery_id,
        )
        task_dir = Path(tasks_root) / task_id
        task_dir.rename(detached)
        task_dir.symlink_to(redirected, target_is_directory=True)
        swapped = True
        return uow

    monkeypatch.setattr(
        delivery_tools,
        "_open_delivery_directory_chain",
        open_then_swap,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="directory.*changed|became a symlink",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert swapped is True
    assert list(redirected.iterdir()) == []
    assert not list(detached.rglob("*.py"))
    assert not list(detached.rglob("*.sql"))
    assert not list(detached.rglob("*.json"))
    assert _artifact_rows(runtime, fixture[1].id) == []
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0


def test_export_strategy_delivery_rejects_drifted_audit_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET detail_json = '{}'
             WHERE kind = ? AND target_ref = ?
            """,
            (DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("actor", "operator"), ("outcome", "failed")),
)
def test_export_strategy_delivery_rejects_drifted_audit_outcome_on_retry(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            f"UPDATE audit SET {field} = ? "
            "WHERE kind = ? AND target_ref = ?",
            (replacement, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)


def test_export_strategy_delivery_rejects_drifted_audit_inputs_hash_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET inputs_hash = ?
             WHERE kind = ? AND target_ref = ?
            """,
            ("0" * 64, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 1


@pytest.mark.parametrize("field", ("kind", "target_ref"))
def test_export_strategy_delivery_rejects_drifted_audit_identity_on_retry(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    replacement = (
        f"{DELIVERY_AUDIT_KIND}.tampered"
        if field == "kind"
        else f"{first['delivery_id']}-tampered"
    )
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            f"UPDATE audit SET {field} = ? "
            "WHERE kind = ? AND target_ref = ?",
            (replacement, DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_export_strategy_delivery_rejects_joint_audit_identity_drift_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            UPDATE audit
               SET kind = ?, target_ref = ?
             WHERE kind = ? AND target_ref = ?
            """,
            (
                f"{DELIVERY_AUDIT_KIND}.tampered",
                f"{first['delivery_id']}-tampered",
                DELIVERY_AUDIT_KIND,
                first["delivery_id"],
            ),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit").fetchone()[0] == 1


def test_export_strategy_delivery_rejects_duplicate_audit_on_retry(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    first, _runtime_value = _run(fixture, request)
    with sqlite3.connect(fixture[0].db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit(
                id, kind, actor, target_ref, inputs_hash, outcome,
                detail_json, at
            )
            SELECT lower(hex(randomblob(16))), kind, actor, target_ref,
                   inputs_hash, outcome, detail_json, at
              FROM audit
             WHERE kind = ? AND target_ref = ?
            """,
            (DELIVERY_AUDIT_KIND, first["delivery_id"]),
        )

    with pytest.raises(StrategyDeliveryToolError, match="audit.*changed"):
        _run(fixture, request)

    assert _audit_count(
        fixture[0],
        delivery_id=first["delivery_id"],
    ) == 2


def test_export_strategy_delivery_rejects_strategy_or_dataset_drift(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    wrong_strategy = deepcopy(request)
    wrong_strategy["strategy_ref"]["expected_spec_hash"] = "f" * 64
    with pytest.raises(StrategyDeliveryToolError, match="strategy.*exact"):
        _run(fixture, wrong_strategy)

    wrong_dataset = deepcopy(request)
    wrong_dataset["dataset_ref"]["expected_content_hash"] = "e" * 64
    with pytest.raises(StrategyDeliveryToolError, match="dataset.*exact"):
        _run(fixture, wrong_dataset)


def test_export_strategy_delivery_rolls_back_all_files_and_rows_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    original = runtime.task_artifacts.register_on_connection
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected registry failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        runtime.task_artifacts,
        "register_on_connection",
        fail_second,
    )

    with pytest.raises(RuntimeError, match="injected registry failure"):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    assert not list(delivery_root.rglob("*.py"))
    assert not list(delivery_root.rglob("*.sql"))
    assert not list(delivery_root.rglob("*.json"))
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = ?",
                (DELIVERY_AUDIT_KIND,),
            ).fetchone()[0]
            == 0
        )


def test_output_contract_failure_rolls_back_files_rows_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])

    def reject_output(*_args, **_kwargs):
        raise StrategyDeliveryToolError(
            "injected public output contract failure"
        )

    monkeypatch.setattr(
        delivery_tools,
        "validate_export_strategy_delivery_tool_output",
        reject_output,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="public output contract failure",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    assert not [
        path
        for path in delivery_root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    "force_path_backend",
    [False, True],
    ids=["dirfd", "path-fallback"],
)
@pytest.mark.parametrize(
    "with_existing_final",
    [False, True],
    ids=["without-old-final", "with-old-final"],
)
def test_post_publication_validation_failure_removes_only_new_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_path_backend: bool,
    with_existing_final: bool,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    if force_path_backend:
        monkeypatch.setattr(
            delivery_tools,
            "_supports_secure_delivery_dir_fds",
            lambda: False,
        )
    runtime = _runtime(fixture[-1])
    retained_path: Path | None = None
    retained_bytes: bytes | None = None
    if with_existing_final:
        first = run_export_strategy_delivery(
            request,
            fixture[-1],
            runtime,
        )
        rows = _artifact_rows(runtime, fixture[1].id)
        retained = next(
            row
            for row in rows
            if row["kind"] == DELIVERY_ARTIFACT_KINDS["python"]
        )
        retained_path = Path(retained["path"])
        retained_bytes = retained_path.read_bytes()
        with sqlite3.connect(fixture[0].db_path) as conn:
            conn.executemany(
                "DELETE FROM task_artifacts WHERE id = ?",
                [(row["id"],) for row in rows],
            )
            conn.execute(
                "DELETE FROM audit WHERE kind = ? AND target_ref = ?",
                (DELIVERY_AUDIT_KIND, first["delivery_id"]),
            )
        for row in rows:
            path = Path(row["path"])
            if path != retained_path:
                path.unlink()

    def fail_promoted_entry(*_args, **_kwargs):
        raise OSError("injected post-link stat failure")

    monkeypatch.setattr(
        delivery_tools,
        "_require_promoted_delivery_entry",
        fail_promoted_entry,
    )
    monkeypatch.setattr(
        delivery_tools,
        "_require_promoted_delivery_path",
        fail_promoted_entry,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="promotion failed safely",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert _artifact_rows(runtime, fixture[1].id) == []
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    published = [
        path
        for path in delivery_root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    if retained_path is None:
        assert published == []
    else:
        assert retained_bytes is not None
        assert retained_path.read_bytes() == retained_bytes
        assert published == [retained_path]


@pytest.mark.parametrize(
    "force_path_backend",
    [False, True],
    ids=["dirfd", "path-fallback"],
)
def test_no_clobber_race_preserves_external_final_and_rolls_back_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    force_path_backend: bool,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    dirfd_supported = delivery_tools._supports_secure_delivery_dir_fds()
    if not force_path_backend and not dirfd_supported:
        pytest.skip("secure dirfd publication is unavailable on this platform")
    monkeypatch.setattr(
        delivery_tools,
        "_supports_secure_delivery_dir_fds",
        lambda: not force_path_backend,
    )
    original_link = delivery_tools.os.link
    external_bytes = b"external-race-winner\n"
    injected = False

    def inject_external_final_before_link(source, target, *args, **kwargs):
        nonlocal injected
        if not injected:
            destination_fd = kwargs.get("dst_dir_fd")
            if destination_fd is None:
                Path(target).write_bytes(external_bytes)
            else:
                descriptor = os.open(
                    target,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=destination_fd,
                )
                try:
                    assert os.write(descriptor, external_bytes) == len(
                        external_bytes
                    )
                finally:
                    os.close(descriptor)
            injected = True
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(
        delivery_tools.os,
        "link",
        inject_external_final_before_link,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="promotion failed safely",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert injected is True
    delivery_root = (
        Path(fixture[0].tasks_dir)
        / fixture[1].id
        / "strategy_delivery"
    )
    published = [
        path
        for path in delivery_root.rglob("*")
        if path.is_file() or path.is_symlink()
    ]
    assert len(published) == 1
    assert published[0].name == "strategy.py"
    assert published[0].read_bytes() == external_bytes
    assert _artifact_rows(runtime, fixture[1].id) == []
    with sqlite3.connect(fixture[0].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (DELIVERY_AUDIT_KIND,),
        ).fetchone()[0] == 0


def test_export_strategy_delivery_rejects_symlink_swap_after_promotion_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    outside = tmp_path / "outside-strategy.py"
    original = delivery_tools._require_exact_delivery_entry
    swapped = False

    def swap_after_first_check(
        directory_fd,
        name,
        *,
        expected,
        expected_hash,
        expected_identity=None,
    ):
        nonlocal swapped
        original(
            directory_fd,
            name,
            expected=expected,
            expected_hash=expected_hash,
            expected_identity=expected_identity,
        )
        if not swapped and name == "strategy.py":
            outside.write_bytes(expected)
            os.unlink(name, dir_fd=directory_fd)
            os.symlink(outside, name, dir_fd=directory_fd)
            swapped = True

    monkeypatch.setattr(
        delivery_tools,
        "_require_exact_delivery_entry",
        swap_after_first_check,
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="regular file|unavailable|bytes changed",
    ):
        run_export_strategy_delivery(request, fixture[-1], runtime)

    assert swapped is True
    assert _artifact_rows(runtime, fixture[1].id) == []


def test_delivery_output_validator_requires_external_exact_refs(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged["strategy_ref"]["expected_spec_hash"] = "0" * 64
    body = {
        key: value
        for key, value in forged["equivalence"].items()
        if key not in {"equivalence_id", "content_hash"}
    }
    body["strategy_spec_hash"] = "0" * 64
    digest = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    forged["equivalence"] = {
        **body,
        "equivalence_id": "strategy-dsl-equivalence-" + digest[:24],
    }
    forged["equivalence"]["content_hash"] = hashlib.sha256(
        json.dumps(
            forged["equivalence"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(StrategyDeliveryToolError, match="strategy_ref"):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_artifacts=_artifact_projections(output),
        )


def test_artifact_record_validator_rejects_forged_stable_id(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, runtime = _run(fixture, request)
    records = _artifact_record_mapping(runtime, fixture[1].id)
    forged = deepcopy(records)
    forged["python"]["id"] = "0" * 64
    assert forged["python"]["id"] != records["python"]["id"]

    with pytest.raises(
        StrategyDeliveryToolError,
        match="stable identity drifted",
    ):
        validate_strategy_delivery_artifact_records(
            forged,
            expected_task_id=fixture[1].id,
            expected_delivery_id=output["delivery_id"],
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_maximum_equivalence_rows=request[
                "maximum_equivalence_rows"
            ],
            expected_equivalence=output["equivalence"],
        )


def test_delivery_output_validator_rejects_sample_count_above_declared_budget(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged["maximum_equivalence_rows"] = 1
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        workspace_ref=forged["workspace_ref"],
        maximum_equivalence_rows=1,
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(StrategyDeliveryToolError, match="sample_count.*budget"):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_artifacts=_artifact_projections(output),
        )


@pytest.mark.parametrize(
    ("artifact_index", "artifact_name"),
    tuple(enumerate(("python", "sql", "strategy_json", "equivalence_json"))),
)
def test_delivery_output_validator_binds_all_published_artifact_content(
    tmp_path: Path,
    artifact_index: int,
    artifact_name: str,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    forged_hash = "0" * 64
    assert forged_hash != forged["artifacts"][artifact_index]["content_hash"]
    forged["artifacts"][artifact_index]["content_hash"] = forged_hash
    forged["artifacts"][artifact_index]["download_url"] = (
        forged["artifacts"][artifact_index]["download_url"].rsplit("=", 1)[0]
        + f"={forged_hash}"
    )
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        workspace_ref=forged["workspace_ref"],
        maximum_equivalence_rows=forged["maximum_equivalence_rows"],
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match=f"artifacts.*{artifact_name}|authenticated publication",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_artifacts=_artifact_projections(output),
        )


def test_delivery_output_validator_binds_equivalence_artifact_to_document_bytes(
    tmp_path: Path,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    output, _runtime_value = _run(fixture, request)
    forged = deepcopy(output)
    equivalence_artifact = forged["artifacts"][3]
    forged_hash = "0" * 64
    assert forged_hash != equivalence_artifact["content_hash"]
    equivalence_artifact["content_hash"] = forged_hash
    equivalence_artifact["download_url"] = (
        equivalence_artifact["download_url"].rsplit("=", 1)[0]
        + f"={forged_hash}"
    )
    forged["delivery_id"] = delivery_tools._delivery_id(
        strategy_ref=forged["strategy_ref"],
        dataset_ref=forged["dataset_ref"],
        workspace_ref=forged["workspace_ref"],
        maximum_equivalence_rows=forged["maximum_equivalence_rows"],
        equivalence=forged["equivalence"],
        content_hashes={
            name: forged["artifacts"][index]["content_hash"]
            for index, name in enumerate(
                ("python", "sql", "strategy_json", "equivalence_json")
            )
        },
    )

    with pytest.raises(
        StrategyDeliveryToolError,
        match="equivalence.*artifact.*content",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_artifacts=_artifact_projections(forged),
        )


def test_export_strategy_delivery_authenticates_snapshot_and_artifact_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    private_reads: list[object] = []
    original_read_parquet = delivery_tools.pd.read_parquet

    def reject_live_path_read(*args, **kwargs):
        raise AssertionError("delivery must not reopen the live dataset path")

    def record_private_read(source, *args, **kwargs):
        assert not isinstance(source, (str, Path))
        private_reads.append(source)
        return original_read_parquet(source, *args, **kwargs)

    monkeypatch.setattr(runtime.backend, "read_frame", reject_live_path_read)
    monkeypatch.setattr(
        delivery_tools.pd,
        "read_parquet",
        record_private_read,
    )

    output = run_export_strategy_delivery(
        request,
        fixture[-1],
        runtime,
    )
    assert len(private_reads) == 1
    for artifact in output["artifacts"]:
        assert artifact["download_url"].endswith(
            f"?expected_content_hash={artifact['content_hash']}"
        )

    forged = deepcopy(output)
    forged["artifacts"][0]["artifact_id"] = "0" * 64
    forged["artifacts"][0]["download_url"] = forged["artifacts"][0][
        "download_url"
    ].replace(
        output["artifacts"][0]["artifact_id"],
        "0" * 64,
    )
    with pytest.raises(
        StrategyDeliveryToolError,
        match="authenticated publication",
    ):
        validate_export_strategy_delivery_tool_output(
            forged,
            expected_task_id=fixture[1].id,
            expected_strategy_ref=request["strategy_ref"],
            expected_dataset_ref=request["dataset_ref"],
            expected_workspace_ref=request["workspace_ref"],
            expected_artifacts=_artifact_projections(output),
        )


def test_export_strategy_delivery_revalidates_source_without_path_hash_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _runtime_fixture(tmp_path, "approval")
    request = _inputs(fixture)
    runtime = _runtime(fixture[-1])
    dataset_path = Path(runtime.registry.resolve_path(fixture[3].id))
    original_authenticate = delivery_tools._require_authenticated_file_hash
    authenticated: list[Path] = []

    def record_authenticated(path, *, root, expected_hash):
        authenticated.append(Path(path))
        return original_authenticate(
            path,
            root=root,
            expected_hash=expected_hash,
        )

    monkeypatch.setattr(
        delivery_tools,
        "_require_authenticated_file_hash",
        record_authenticated,
    )

    output = run_export_strategy_delivery(request, fixture[-1], runtime)

    assert output["dataset_ref"] == request["dataset_ref"]
    assert authenticated == [dataset_path]
