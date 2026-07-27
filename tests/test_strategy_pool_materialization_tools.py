from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import sqlite3
from threading import Barrier

import pytest

import marvis.db_schema as db_schema_module
import marvis.packs.strategy.pool_materialization_tools as materialization_tools
from marvis.db import StrategyRepository, TaskRepository
from marvis.db_schema import connect, init_db
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.dsl import canonical_strategy_json
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH, compile_strategy_pool
from marvis.packs.strategy.pool_materialization_tools import (
    MATERIALIZATION_AUDIT_KIND,
    MATERIALIZATION_TOOL_SCHEMA_VERSION,
    run_materialize_strategy_from_pool,
)
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.pool_requirement_resolver import (
    model_score_virtual_field,
)
from test_strategy_pool_scorecard import (
    _add_inputs as _scorecard_add_inputs,
    _real_scorecard,
    _selection as _scorecard_selection,
)
from test_strategy_pool_tools import _add_inputs, _setup


def _materialization_input(added: dict) -> dict:
    artifact = added["artifacts"][0]
    design = compile_strategy_pool(added["pool"])
    return {
        "strategy_type": added["pool"]["strategy_type"],
        "expected_pool_revision": added["revision"],
        "expected_pool_snapshot_hash": added["snapshot_hash"],
        "expected_pool_artifact_id": artifact["artifact_id"],
        "expected_pool_artifact_content_hash": artifact["content_hash"],
        "expected_design_hash": design["design_hash"],
    }


def test_materializes_exact_compiled_spec_and_retries_idempotently(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    expected_design = compile_strategy_pool(added["pool"])

    first = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    retry = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert retry == first
    assert set(first) == {
        "schema_version",
        "materialization_id",
        "strategy_ref",
        "pool_ref",
        "design_hash",
        "requirements",
        "lifecycle",
    }
    assert first["schema_version"] == MATERIALIZATION_TOOL_SCHEMA_VERSION
    assert first["design_hash"] == expected_design["design_hash"]
    assert first["pool_ref"] == {
        "pool_id": added["pool"]["pool_id"],
        "revision_id": added["pool"]["revision_id"],
        "revision": added["revision"],
        "snapshot_hash": added["snapshot_hash"],
        "artifact_id": added["artifacts"][0]["artifact_id"],
        "artifact_content_hash": added["artifacts"][0]["content_hash"],
    }
    assert first["requirements"] == {
        "requirements_hash": hashlib.sha256(b"[]").hexdigest(),
        "requirement_count": 0,
        "virtual_fields": [],
        "runtime_requirements_supported": True,
        "blocker_code": None,
    }
    assert first["lifecycle"] == {
        "created_status": "draft",
        "created_asset_status": "draft",
        "current_status": "draft",
        "current_asset_status": "draft",
        "adopted_by_this_tool": False,
        "deployed_by_this_tool": False,
    }

    strategy_id = first["strategy_ref"]["strategy_id"]
    strategies = StrategyRepository(fixture["settings"].db_path)
    persisted = strategies.get_strategy(strategy_id)
    assert persisted is not None
    assert persisted.spec is not None
    assert persisted.spec.to_dict() == expected_design["strategy_spec"]
    assert first["strategy_ref"] == {
        "strategy_id": strategy_id,
        "strategy_type": "approval",
        "version": 1,
        "strategy_spec_hash": strategies.get_strategy_spec_hash(strategy_id),
        "strategy_dsl_content_hash": hashlib.sha256(
            canonical_strategy_json(expected_design["strategy_spec"]).encode("utf-8")
        ).hexdigest(),
    }
    ledger = strategies.get_pool_materialization(first["materialization_id"])
    assert ledger is not None
    assert ledger["strategy_id"] == strategy_id
    assert ledger["pool_revision_id"] == added["pool"]["revision_id"]
    assert TaskRepository(fixture["settings"].db_path).count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
        target_ref=strategy_id,
    ) == 1


def test_retry_rejects_a_second_creation_audit_even_with_wrong_inputs_hash(
    tmp_path,
) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    first = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    strategy_id = first["strategy_ref"]["strategy_id"]
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit(
                id, kind, actor, target_ref, inputs_hash, outcome,
                detail_json, at
            )
            VALUES (
                'forged-pool-materialization-audit',
                ?, 'system', ?, ?, 'succeeded', '{}',
                '2026-07-26T00:00:00+00:00'
            )
            """,
            (MATERIALIZATION_AUDIT_KIND, strategy_id, "f" * 64),
        )

    with pytest.raises(
        StrategyError,
        match="exactly one creation audit",
    ):
        run_materialize_strategy_from_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )


def test_retry_rejects_materialized_strategy_creation_timestamp_drift(
    tmp_path,
) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    first = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE strategies SET created_at = ? WHERE id = ?",
            (
                "2026-07-26T01:00:00+00:00",
                first["strategy_ref"]["strategy_id"],
            ),
        )

    with pytest.raises(
        StrategyError,
        match="Strategy binding changed",
    ):
        run_materialize_strategy_from_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )


def test_retry_returns_same_strategy_with_current_adopted_lifecycle(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    first = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    strategy_id = first["strategy_ref"]["strategy_id"]
    strategies = StrategyRepository(fixture["settings"].db_path)
    strategies.adopt_strategy_with_audit(
        strategy_id,
        reason="approved by committee",
        audit={
            "kind": "strategy.adopt",
            "target_ref": strategy_id,
            "outcome": "succeeded",
            "detail": {"strategy_id": strategy_id},
        },
        adopted_at="2026-07-26T02:00:00+00:00",
    )

    retry = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert retry["materialization_id"] == first["materialization_id"]
    assert retry["strategy_ref"] == first["strategy_ref"]
    assert retry["lifecycle"] == {
        "created_status": "draft",
        "created_asset_status": "draft",
        "current_status": "adopted",
        "current_asset_status": "adopted_local",
        "adopted_by_this_tool": False,
        "deployed_by_this_tool": False,
    }
    assert TaskRepository(fixture["settings"].db_path).count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
        target_ref=strategy_id,
    ) == 1


def test_retry_returns_same_strategy_with_current_retired_lifecycle(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    first = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    strategy_id = first["strategy_ref"]["strategy_id"]
    strategies = StrategyRepository(fixture["settings"].db_path)
    strategies.adopt_strategy_with_audit(
        strategy_id,
        reason="initial committee approval",
        audit={
            "kind": "strategy.adopt",
            "target_ref": strategy_id,
            "outcome": "succeeded",
        },
        adopted_at="2026-07-26T02:00:00+00:00",
    )
    materialized = strategies.get_strategy(strategy_id)
    assert materialized is not None
    replacement = replace(materialized, id="replacement-strategy")
    strategies.create_strategy(fixture["task"].id, replacement)
    strategies.adopt_strategy_with_audit(
        replacement.id,
        reason="replacement committee approval",
        audit={
            "kind": "strategy.adopt",
            "target_ref": replacement.id,
            "outcome": "succeeded",
        },
        adopted_at="2026-07-26T03:00:00+00:00",
    )

    retry = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert retry["materialization_id"] == first["materialization_id"]
    assert retry["strategy_ref"] == first["strategy_ref"]
    assert retry["lifecycle"] == {
        "created_status": "draft",
        "created_asset_status": "draft",
        "current_status": "retired",
        "current_asset_status": "retired",
        "adopted_by_this_tool": False,
        "deployed_by_this_tool": False,
    }


def test_concurrent_identical_materializations_converge_on_one_identity(
    tmp_path,
    monkeypatch,
) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    rendezvous = Barrier(2)
    original_loader = (
        materialization_tools.load_current_strategy_candidate_pool_artifact
    )

    def synchronized_loader(*args, **kwargs):
        binding = original_loader(*args, **kwargs)
        rendezvous.wait(timeout=10)
        return binding

    monkeypatch.setattr(
        materialization_tools,
        "load_current_strategy_candidate_pool_artifact",
        synchronized_loader,
    )

    def materialize() -> dict:
        return run_materialize_strategy_from_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(lambda _index: materialize(), range(2)))

    assert outputs[0] == outputs[1]
    strategy_id = outputs[0]["strategy_ref"]["strategy_id"]
    with connect(fixture["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategies WHERE id = ?",
            (strategy_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_pool_materializations"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ? AND target_ref = ?",
            (MATERIALIZATION_AUDIT_KIND, strategy_id),
        ).fetchone()[0] == 1


def test_repository_read_reauthenticates_the_exact_creation_audit(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    result = run_materialize_strategy_from_pool(
        _materialization_input(added),
        fixture["ctx"],
        fixture["runtime"],
    )
    strategies = StrategyRepository(fixture["settings"].db_path)
    materialization = strategies.get_pool_materialization(
        result["materialization_id"]
    )
    assert materialization is not None
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "UPDATE audit SET detail_json = '{}' WHERE id = ?",
            (materialization["audit_id"],),
        )

    with pytest.raises(StrategyError, match="audit changed"):
        strategies.get_pool_materialization(result["materialization_id"])


def test_repository_read_reauthenticates_the_exact_pool_revision(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    result = run_materialize_strategy_from_pool(
        _materialization_input(added),
        fixture["ctx"],
        fixture["runtime"],
    )
    strategies = StrategyRepository(fixture["settings"].db_path)
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            "DROP TRIGGER trg_strategy_candidate_pool_revisions_immutable_update"
        )
        conn.execute(
            """
            UPDATE strategy_candidate_pool_revisions
               SET snapshot_hash = ?
             WHERE id = ?
            """,
            ("f" * 64, added["pool"]["revision_id"]),
        )

    with pytest.raises(StrategyError, match="Pool revision binding changed"):
        strategies.get_pool_materialization(result["materialization_id"])


@pytest.mark.parametrize(
    ("field", "expected_message"),
    [
        ("expected_pool_artifact_id", "artifact id changed"),
        ("expected_design_hash", "design hash changed"),
    ],
)
def test_stale_exact_input_creates_no_strategy_ledger_or_audit(
    tmp_path,
    field: str,
    expected_message: str,
) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(added)
    request[field] = "f" * 64

    with pytest.raises(StrategyError, match=expected_message):
        run_materialize_strategy_from_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )

    strategies = StrategyRepository(fixture["settings"].db_path)
    assert strategies.list_for_task(fixture["task"].id) == []
    assert TaskRepository(fixture["settings"].db_path).count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
    ) == 0
    with connect(fixture["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_pool_materializations"
        ).fetchone()[0] == 0


def test_ledger_insert_failure_rolls_back_strategy_and_creation_audit(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    with connect(fixture["settings"].db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER force_pool_materialization_ledger_failure
            BEFORE INSERT ON strategy_pool_materializations
            BEGIN
                SELECT RAISE(ABORT, 'forced Pool materialization ledger failure');
            END
            """
        )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="forced Pool materialization ledger failure",
    ):
        run_materialize_strategy_from_pool(
            _materialization_input(added),
            fixture["ctx"],
            fixture["runtime"],
        )

    assert StrategyRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    ) == []
    assert TaskRepository(fixture["settings"].db_path).count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
    ) == 0
    with connect(fixture["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_pool_materializations"
        ).fetchone()[0] == 0


@pytest.mark.slow
def test_model_score_requirements_are_materialized_and_runtime_ready(
    tmp_path,
) -> None:
    real = _real_scorecard(tmp_path)
    selection = _scorecard_selection(real)
    added = run_add_candidate_to_pool(
        _scorecard_add_inputs(
            selection,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    compiled = compile_strategy_pool(added["pool"])
    [requirement] = compiled["requirements"]
    virtual_field = model_score_virtual_field(
        requirement["requirement"]["score_vector_artifact_id"]
    )

    output = run_materialize_strategy_from_pool(
        _materialization_input(added),
        real["fx"]["ctx"],
        real["runtime"],
    )

    assert output["requirements"] == {
        "requirements_hash": hashlib.sha256(
            _canonical_json(compiled["requirements"]).encode("utf-8")
        ).hexdigest(),
        "requirement_count": 1,
        "virtual_fields": [virtual_field],
        "runtime_requirements_supported": True,
        "blocker_code": None,
    }
    strategy = StrategyRepository(real["fx"]["settings"].db_path).get_strategy(
        output["strategy_ref"]["strategy_id"]
    )
    assert strategy is not None
    assert strategy.spec is not None
    assert strategy.spec.to_dict() == compiled["strategy_spec"]


def test_materialization_ledger_updates_are_immutable(tmp_path) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    output = run_materialize_strategy_from_pool(
        _materialization_input(added),
        fixture["ctx"],
        fixture["runtime"],
    )

    with connect(fixture["settings"].db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                """
                UPDATE strategy_pool_materializations
                   SET selected_design_hash = ?
                 WHERE id = ?
                """,
                ("f" * 64, output["materialization_id"]),
            )


def test_materialized_task_purge_removes_owned_graph_and_retains_audits(
    tmp_path,
) -> None:
    fixture = _setup(tmp_path)
    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    output = run_materialize_strategy_from_pool(
        _materialization_input(added),
        fixture["ctx"],
        fixture["runtime"],
    )
    task_id = fixture["task"].id
    strategy_id = output["strategy_ref"]["strategy_id"]
    tasks = TaskRepository(fixture["settings"].db_path)

    summary = tasks.purge_task(task_id, actor="tester")

    assert summary["strategies"] == 1
    with pytest.raises(KeyError):
        tasks.get_task(task_id)
    with connect(fixture["settings"].db_path) as conn:
        for table, column in (
            ("strategies", "task_id"),
            ("strategy_pool_materializations", "task_id"),
            ("strategy_candidate_pools", "task_id"),
            ("strategy_candidate_pool_revisions", "task_id"),
            ("task_artifacts", "task_id"),
        ):
            remaining = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {column} = ?",
                (task_id,),
            ).fetchone()[0]
            assert remaining == 0, f"{table} still has task-owned rows after purge"
    assert tasks.count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
        target_ref=strategy_id,
    ) == 1
    assert tasks.count_audit(kind="task.delete", target_ref=task_id) == 1


def test_retry_rejects_a_pool_revision_that_is_no_longer_current(tmp_path) -> None:
    fixture = _setup(tmp_path)
    first_pool = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    request = _materialization_input(first_pool)
    materialized = run_materialize_strategy_from_pool(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    run_add_candidate_to_pool(
        _add_inputs(
            fixture["refine"](1),
            expected_revision=first_pool["revision"],
            expected_hash=first_pool["snapshot_hash"],
        ),
        fixture["ctx"],
        fixture["runtime"],
    )

    with pytest.raises(
        StrategyError,
        match="stale strategy candidate pool revision",
    ):
        run_materialize_strategy_from_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )

    strategies = StrategyRepository(fixture["settings"].db_path)
    assert [
        item.id for item in strategies.list_for_task(fixture["task"].id)
    ] == [materialized["strategy_ref"]["strategy_id"]]
    assert TaskRepository(fixture["settings"].db_path).count_audit(
        kind=MATERIALIZATION_AUDIT_KIND,
    ) == 1


def test_migration_022_is_registered_and_recreates_the_guarded_ledger(
    tmp_path,
) -> None:
    db_path = tmp_path / "migration.sqlite"
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DROP TABLE strategy_pool_materializations")
        conn.execute("PRAGMA user_version = 21")

    init_db(db_path)

    with connect(db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 22
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(strategy_pool_materializations)"
            ).fetchall()
        }
        triggers = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                  FROM sqlite_master
                 WHERE type = 'trigger'
                   AND name LIKE 'trg_strategy_pool_materializations_%'
                """
            ).fetchall()
        }
    assert db_schema_module.SCHEMA_VERSION == 22
    assert {
        "strategy_version",
        "requirements_json",
        "audit_id",
        "strategy_spec_hash",
        "strategy_dsl_content_hash",
    } <= columns
    assert {
        "trg_strategy_pool_materializations_current_pool",
        "trg_strategy_pool_materializations_immutable_update",
    } <= triggers
    assert "trg_strategy_pool_materializations_immutable_delete" not in triggers


def _canonical_json(value: object) -> str:
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
