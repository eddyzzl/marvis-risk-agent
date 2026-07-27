from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import marvis.db_schema as db_schema
from marvis.db_schema import connect, init_db
from marvis.packs.strategy.monitoring_plan import (
    MonitoringPlan,
    canonical_economics_bindings_hash,
)
from marvis.repositories.strategy_monitoring import (
    StrategyMonitoringConflictError,
    StrategyMonitoringDataError,
    StrategyMonitoringDuplicateError,
    StrategyMonitoringRepository,
    validate_monitoring_run_result,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_domain(db_path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tasks(
                id, model_name, model_version, validator, source_dir,
                status, status_message, created_at, updated_at
            ) VALUES (
                'task-1', 'strategy task', 'v1', 'tester', '/tmp/source',
                'draft', '', '2026-07-18T00:00:00+00:00',
                '2026-07-18T00:00:00+00:00'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO strategies(
                id, task_id, strategy_type, rules_json, score_col,
                default_decision_json, description, created_at, version, status
            ) VALUES (
                'strategy-1', 'task-1', 'limit', '[]', 'score',
                'null', 'test strategy', '2026-07-18T00:00:00+00:00', 3, 'adopted'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO datasets(
                id, task_id, role, source_path, format, row_count, columns_json,
                has_target, target_col, created_at, content_hash
            ) VALUES (
                'dataset-1', 'task-1', 'monitoring', '/tmp/monitor.csv', 'csv',
                10, '["score", "pd"]', 0, NULL,
                '2026-07-18T00:00:00+00:00', ?
            )
            """,
            (_sha("dataset-1"),),
        )


def _plan(*, revision: int = 1, supersedes: str | None = None) -> MonitoringPlan:
    return MonitoringPlan(
        strategy_id="strategy-1",
        version=3,
        revision=revision,
        supersedes_plan_id=supersedes,
        thresholds={"expected_loss": {"metric": "expected_loss", "direction": "max"}},
        expectation_baseline={"strategy_effect_hash": _sha("strategy-effect")},
        economics_bindings={
            "lgd": {"kind": "scalar", "value": 0.45},
            "pd": {"kind": "column", "column": "pd"},
        },
    )


def test_migration_008_upgrades_and_is_idempotent(tmp_path):
    db_path = tmp_path / "v7.sqlite"
    with connect(db_path) as conn:
        conn.execute("CREATE TABLE strategies(id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE datasets(id TEXT PRIMARY KEY)")
        conn.execute("PRAGMA user_version = 7")

    init_db(db_path)
    init_db(db_path)

    with connect(db_path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        plan_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(strategy_monitoring_plans)")
        }
        run_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(strategy_monitoring_runs)")
        }

    assert version == db_schema.SCHEMA_VERSION
    assert db_schema.SCHEMA_VERSION >= 15
    assert {"strategy_monitoring_plans", "strategy_monitoring_runs"} <= tables
    assert {
        "id",
        "strategy_id",
        "strategy_version",
        "revision",
        "schema_version",
        "payload_json",
        "payload_hash",
        "supersedes_plan_id",
        "created_at",
    } <= plan_columns
    assert {
        "id",
        "strategy_id",
        "monitoring_plan_id",
        "dataset_id",
        "dataset_content_hash",
        "strategy_effect_hash",
        "economics_binding_hash",
        "result_json",
        "result_hash",
        "overall_level",
        "created_at",
    } <= run_columns


def test_repository_creates_immutable_plan_revisions_with_cas(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)

    first = repo.create_plan(_plan(), expected_revision=0)
    second = repo.create_plan(
        _plan(revision=2, supersedes=first.id),
        expected_revision=1,
        expected_payload_hash=first.payload_hash,
    )

    assert first.revision == 1
    assert first.plan.monitoring_plan_id == first.id
    assert second.revision == 2
    assert second.supersedes_plan_id == first.id
    assert repo.get_plan(first.id) == first
    assert repo.latest_plan("strategy-1") == second
    assert repo.list_plans("strategy-1") == [first, second]
    with connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute(
                "UPDATE strategy_monitoring_plans SET revision = 99 WHERE id = ?",
                (first.id,),
            )


def test_repository_uses_plan_embedded_id_when_no_override_is_supplied(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    plan = MonitoringPlan(
        strategy_id="strategy-1",
        version=3,
        monitoring_plan_id="embedded-plan-id",
    )

    created = repo.create_plan(plan, expected_revision=0)

    assert created.id == "embedded-plan-id"
    assert created.plan.monitoring_plan_id == "embedded-plan-id"


def test_repository_rejects_stale_and_duplicate_plan_writes(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    first = repo.create_plan(_plan(), expected_revision=0, plan_id="plan-1")

    with pytest.raises(StrategyMonitoringConflictError, match="stale"):
        repo.create_plan(
            _plan(revision=2, supersedes=first.id),
            expected_revision=0,
        )
    with pytest.raises(StrategyMonitoringConflictError, match="hash"):
        repo.create_plan(
            _plan(revision=2, supersedes=first.id),
            expected_revision=1,
            expected_payload_hash=_sha("wrong"),
        )
    with pytest.raises(StrategyMonitoringDuplicateError, match="duplicate"):
        repo.create_plan(
            _plan(revision=2, supersedes=first.id),
            expected_revision=1,
            expected_payload_hash=first.payload_hash,
            plan_id="plan-1",
        )


def test_repository_records_bound_monitoring_run_evidence(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    plan = repo.create_plan(_plan(), expected_revision=0)
    result = {
        "overall_level": "amber",
        "checks": [
            {
                "id": "expected_loss",
                "level": "amber",
                "metric": "expected_loss",
                "value": 12.5,
            }
        ],
    }

    run = repo.create_run(
        strategy_id="strategy-1",
        monitoring_plan_id=plan.id,
        expected_plan_revision=plan.revision,
        expected_plan_payload_hash=plan.payload_hash,
        dataset_id="dataset-1",
        dataset_content_hash=_sha("dataset-1"),
        strategy_effect_hash=_sha("strategy-effect"),
        economics_binding_hash=canonical_economics_bindings_hash(
            plan.plan.economics_bindings
        ),
        result=result,
        overall_level="amber",
    )

    assert run.result == result
    assert run.result_hash == _sha(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    assert repo.get_run(run.id) == run
    assert repo.list_runs("strategy-1") == [run]


def test_repository_rejects_stale_mismatched_and_duplicate_runs(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    first = repo.create_plan(_plan(), expected_revision=0)
    second = repo.create_plan(
        _plan(revision=2, supersedes=first.id),
        expected_revision=1,
        expected_payload_hash=first.payload_hash,
    )
    kwargs = {
        "strategy_id": "strategy-1",
        "monitoring_plan_id": second.id,
        "expected_plan_revision": second.revision,
        "expected_plan_payload_hash": second.payload_hash,
        "dataset_id": "dataset-1",
        "dataset_content_hash": _sha("dataset-1"),
        "strategy_effect_hash": _sha("strategy-effect"),
        "economics_binding_hash": canonical_economics_bindings_hash(
            second.plan.economics_bindings
        ),
        "result": {
            "overall_level": "green",
            "checks": [{"id": "expected_loss", "level": "green"}],
        },
        "overall_level": "green",
    }

    with pytest.raises(StrategyMonitoringConflictError, match="latest"):
        repo.create_run(**{**kwargs, "monitoring_plan_id": first.id})
    with pytest.raises(StrategyMonitoringDataError, match="dataset content hash"):
        repo.create_run(**{**kwargs, "dataset_content_hash": _sha("changed")})
    with pytest.raises(StrategyMonitoringDataError, match="economics binding hash"):
        repo.create_run(**{**kwargs, "economics_binding_hash": _sha("wrong")})

    repo.create_run(**kwargs)
    with pytest.raises(StrategyMonitoringDuplicateError, match="duplicate"):
        repo.create_run(**kwargs)


def test_repository_rejects_plan_strategy_version_mismatch_and_mutable_timestamp(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)

    with pytest.raises(StrategyMonitoringDataError, match="strategy version"):
        repo.create_plan(
            MonitoringPlan(strategy_id="strategy-1", version=2),
            expected_revision=0,
        )
    with pytest.raises(StrategyMonitoringDataError, match="last_run_at"):
        repo.create_plan(
            MonitoringPlan(
                strategy_id="strategy-1",
                version=3,
                last_run_at="2026-07-18T00:00:00+00:00",
            ),
            expected_revision=0,
        )


def test_on_connection_api_participates_in_callers_atomic_transaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)

    with pytest.raises(RuntimeError, match="abort disposition"):
        with connect(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            repo.create_plan_on_connection(conn, _plan(), expected_revision=0)
            raise RuntimeError("abort disposition")

    assert repo.latest_plan("strategy-1") is None


def test_monitoring_run_result_contract_aggregates_material_check_levels():
    validate_monitoring_run_result(
        {
            "overall_level": "green",
            "checks": [
                {"id": "not_matured", "level": "n/a"},
                {"id": "approval_rate", "level": "green"},
            ],
        },
        overall_level="green",
    )
    validate_monitoring_run_result(
        {
            "overall_level": "red",
            "checks": [
                {"id": "approval_rate", "level": "green"},
                {"id": "expected_loss", "level": "amber"},
                {"id": "score_psi", "level": "red"},
                {"id": "not_matured", "level": "n/a"},
            ],
        },
        overall_level="red",
    )
    validate_monitoring_run_result(
        {"overall_level": "n/a", "checks": []},
        overall_level="n/a",
    )


@pytest.mark.parametrize(
    ("result", "overall_level", "message"),
    [
        ([], "n/a", "must be an object"),
        ({"overall_level": "n/a"}, "n/a", "checks must be a list"),
        (
            {"overall_level": "n/a", "checks": {}},
            "n/a",
            "checks must be a list",
        ),
        (
            {"overall_level": "n/a", "checks": ["invalid"]},
            "n/a",
            r"checks\[0\] must be an object",
        ),
        (
            {"overall_level": "green", "checks": [{"level": "green"}]},
            "green",
            r"checks\[0\]\.id must be non-empty",
        ),
        (
            {
                "overall_level": "green",
                "checks": [{"id": "approval_rate", "level": "blue"}],
            },
            "green",
            r"checks\[0\]\.level must be one of",
        ),
        (
            {"checks": []},
            "n/a",
            "overall_level must be present",
        ),
        (
            {"overall_level": "green", "checks": []},
            "amber",
            "does not match run overall_level",
        ),
        (
            {"overall_level": "green", "checks": []},
            "green",
            "does not match check levels",
        ),
        (
            {
                "overall_level": "green",
                "checks": [{"id": "expected_loss", "level": "red"}],
            },
            "green",
            "does not match check levels",
        ),
        (
            {
                "overall_level": "green",
                "checks": [
                    {"id": "approval_rate", "level": "green"},
                    {"id": "approval_rate", "level": "green"},
                ],
            },
            "green",
            "duplicate check id",
        ),
    ],
)
def test_monitoring_run_result_contract_rejects_invalid_semantics(
    result, overall_level, message
):
    with pytest.raises(StrategyMonitoringDataError, match=message):
        validate_monitoring_run_result(result, overall_level=overall_level)


def test_repository_rejects_semantically_inconsistent_run_before_insert(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    plan = repo.create_plan(_plan(), expected_revision=0)

    with pytest.raises(StrategyMonitoringDataError, match="check levels"):
        repo.create_run(
            strategy_id="strategy-1",
            monitoring_plan_id=plan.id,
            expected_plan_revision=plan.revision,
            expected_plan_payload_hash=plan.payload_hash,
            dataset_id="dataset-1",
            dataset_content_hash=_sha("dataset-1"),
            strategy_effect_hash=_sha("strategy-effect"),
            economics_binding_hash=canonical_economics_bindings_hash(
                plan.plan.economics_bindings
            ),
            result={
                "overall_level": "red",
                "checks": [{"id": "expected_loss", "level": "green"}],
            },
            overall_level="red",
        )

    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM strategy_monitoring_runs").fetchone()[
            0
        ] == 0


def test_repository_rejects_hash_valid_but_semantically_tampered_run(tmp_path):
    db_path = tmp_path / "app.sqlite"
    _seed_domain(db_path)
    repo = StrategyMonitoringRepository(db_path)
    plan = repo.create_plan(_plan(), expected_revision=0)
    run = repo.create_run(
        strategy_id="strategy-1",
        monitoring_plan_id=plan.id,
        expected_plan_revision=plan.revision,
        expected_plan_payload_hash=plan.payload_hash,
        dataset_id="dataset-1",
        dataset_content_hash=_sha("dataset-1"),
        strategy_effect_hash=_sha("strategy-effect"),
        economics_binding_hash=canonical_economics_bindings_hash(
            plan.plan.economics_bindings
        ),
        result={
            "overall_level": "green",
            "checks": [{"id": "expected_loss", "level": "green"}],
        },
        overall_level="green",
    )
    tampered = {
        "overall_level": "red",
        "checks": [{"id": "expected_loss", "level": "green"}],
    }
    tampered_json = json.dumps(
        tampered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with connect(db_path) as conn:
        conn.execute("DROP TRIGGER trg_strategy_monitoring_runs_immutable_update")
        conn.execute(
            """
            UPDATE strategy_monitoring_runs
               SET result_json = ?, result_hash = ?, overall_level = 'red'
             WHERE id = ?
            """,
            (tampered_json, _sha(tampered_json), run.id),
        )

    with pytest.raises(StrategyMonitoringDataError, match="check levels"):
        repo.get_run(run.id)
