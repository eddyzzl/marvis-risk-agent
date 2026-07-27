from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from marvis.data.contracts import ColumnFingerprint, ColumnProfile, Dataset
from marvis.db_schema import connect, init_db
from marvis.domain import (
    TASK_TYPE_STRATEGY,
    StrategyProfitInput,
    StrategyTaskInput,
    TaskCreate,
)
from marvis.packs.strategy.dsl import canonical_strategy_json
from marvis.packs.strategy.strategy import build_strategy_from_spec
from marvis.repositories.datasets import DatasetRepository
from marvis.repositories.strategy import StrategyRepository
from marvis.repositories.strategy_handoff import StrategyHandoffRepository
from marvis.repositories.tasks import TaskRepository
from marvis.state_machine import ConflictError


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile(name: str) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        dtype="float64",
        semantic_role="numeric",
        fingerprint=ColumnFingerprint(
            value_kind="numeric",
            length_mode=None,
            regex_pattern=None,
            is_hashed=False,
            hash_type=None,
            hex_case=None,
            date_format=None,
        ),
        null_rate=0.0,
        cardinality=3,
        sample_values=(1.0, 2.0, 3.0),
    )


def _task_payload(
    *,
    name: str = "approval champion",
    metrics: tuple[str, ...] | None = ("psi",),
) -> TaskCreate:
    return TaskCreate(
        task_type=TASK_TYPE_STRATEGY,
        model_name=name,
        model_version="2026-07",
        validator="strategy-owner",
        source_dir="/governed/source",
        algorithm="lgb",
        run_mode="agent",
        target_col="bad_flag",
        score_col="risk_score",
        split_col="split",
        time_col="apply_month",
        feature_columns=["risk_score", "income"],
        strategy_input=StrategyTaskInput(
            strategy_type="approval",
            objective="max_profit",
            max_bad_rate=0.08,
            min_approval_rate=0.55,
            profit=StrategyProfitInput(
                ead_col="ead",
                pd_col="pd",
                annual_rate=0.18,
                funding_rate=0.05,
                lgd=0.6,
                operating_cost_per_loan=20.0,
                term_months=12,
            ),
        ),
        metrics=None if metrics is None else list(metrics),
        capability_tier="balanced",
        notebook_path="/must/not/copy.ipynb",
        sample_path="/must/not/copy.parquet",
        report_values={"TEXT:private": "must not copy"},
    )


def _parent_strategy():
    return build_strategy_from_spec(
        {
            "strategy_type": "approval",
            "default_action": {"type": "approval"},
            "rules": [
                {
                    "rule_id": "reject-risky",
                    "priority": 10,
                    "condition": {
                        "op": "and",
                        "args": [
                            {
                                "op": "compare",
                                "field": "risk_score",
                                "operator": ">=",
                                "value": 700,
                            },
                            {
                                "op": "compare",
                                "field": "income",
                                "operator": "<",
                                "value": 3000,
                            },
                        ],
                    },
                    "action": {"type": "reject", "reason_code": "RISKY"},
                }
            ],
            "metadata": {"owner": "risk-team", "note": "keep canonical DSL"},
        },
        score_col="risk_score",
        description="adopted approval champion",
    )


def _seed_monitoring_evidence(
    db_path: Path,
    *,
    strategy_id: str,
    dataset_id: str,
    dataset_hash: str,
    plan_id: str = "plan-1",
    plan_revision: int = 1,
    run_id: str = "run-red",
    level: str = "red",
    check_level: str | None = None,
    created_at: str = "2026-07-18T01:00:00+00:00",
) -> None:
    plan_json = json.dumps(
        {"strategy_id": strategy_id, "revision": plan_revision},
        sort_keys=True,
        separators=(",", ":"),
    )
    result_json = json.dumps(
        {
            "overall_level": level,
            "checks": [{"id": "bad_rate", "level": check_level or level}],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO strategy_monitoring_plans(
                id, strategy_id, strategy_version, revision, schema_version,
                payload_json, payload_hash, supersedes_plan_id, created_at
            ) VALUES (?, ?, 1, ?, 'strategy.monitoring_plan.v2', ?, ?, NULL, ?)
            """,
            (
                plan_id,
                strategy_id,
                plan_revision,
                plan_json,
                _sha_text(plan_json),
                "2026-07-18T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO strategy_monitoring_runs(
                id, strategy_id, monitoring_plan_id, dataset_id,
                dataset_content_hash, strategy_effect_hash,
                economics_binding_hash, result_json, result_hash,
                overall_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                strategy_id,
                plan_id,
                dataset_id,
                dataset_hash,
                _sha_text("strategy-effect"),
                _sha_text("economics"),
                result_json,
                _sha_text(result_json),
                level,
                created_at,
            ),
        )


def _fixture(
    tmp_path: Path,
    *,
    run_check_level: str | None = None,
    metrics: tuple[str, ...] | None = ("psi",),
) -> dict:
    db_path = tmp_path / "marvis.sqlite"
    datasets_root = tmp_path / "datasets"
    source_path = datasets_root / "content" / "monitoring.parquet"
    source_path.parent.mkdir(parents=True)
    source_bytes = b"immutable-monitoring-dataset"
    source_path.write_bytes(source_bytes)
    dataset_hash = _sha_bytes(source_bytes)
    init_db(db_path)

    task_repo = TaskRepository(db_path)
    source_task = task_repo.create_task(_task_payload(metrics=metrics))
    parent = _parent_strategy()
    strategy_repo = StrategyRepository(db_path)
    strategy_repo.create_strategy(source_task.id, parent)
    with connect(db_path) as conn:
        conn.execute(
            """
            UPDATE strategies
               SET status = 'adopted', asset_status = 'adopted_local', adopted_at = ?
             WHERE id = ?
            """,
            ("2026-07-17T00:00:00+00:00", parent.id),
        )

    source_dataset = Dataset(
        id="dataset-monitoring",
        task_id=source_task.id,
        role="strategy.monitoring",
        source_path="content/monitoring.parquet",
        format="parquet",
        sheet=None,
        row_count=3,
        columns=(_profile("risk_score"), _profile("bad_flag")),
        has_target=True,
        target_col="bad_flag",
        created_at="2026-07-18T00:30:00+00:00",
        content_hash=dataset_hash,
    )
    DatasetRepository(db_path).create_dataset(source_dataset)
    _seed_monitoring_evidence(
        db_path,
        strategy_id=parent.id,
        dataset_id=source_dataset.id,
        dataset_hash=dataset_hash,
        check_level=run_check_level,
    )
    return {
        "db_path": db_path,
        "datasets_root": datasets_root,
        "task_repo": task_repo,
        "strategy_repo": strategy_repo,
        "source_task": source_task,
        "parent": parent,
        "source_dataset": source_dataset,
        "dataset_hash": dataset_hash,
        "source_path": source_path,
    }


def test_new_version_from_on_connection_targets_new_task_and_preserves_dsl(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    target_task = fx["task_repo"].create_task(_task_payload(name="target"))
    repo: StrategyRepository = fx["strategy_repo"]

    with repo.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        child = repo.new_version_from_on_connection(
            conn,
            fx["parent"].id,
            target_task_id=target_task.id,
            new_strategy_id="strategy-child",
            created_at="2026-07-18T02:00:00+00:00",
        )

    meta = repo.get_strategy_meta(child.id)
    assert meta == {
        "id": "strategy-child",
        "task_id": target_task.id,
        "strategy_type": "approval",
        "version": 2,
        "status": "draft",
        "asset_status": "draft",
        "adopted_at": None,
        "adoption_reason": None,
        "parent_strategy_id": fx["parent"].id,
        "created_at": "2026-07-18T02:00:00+00:00",
        "description": "adopted approval champion",
    }
    assert child.spec is not None
    assert child.spec.rules == fx["parent"].spec.rules
    assert child.spec.default_action == fx["parent"].spec.default_action
    assert child.spec.metadata["owner"] == "risk-team"
    assert child.spec.metadata["lineage"]["parent_strategy_id"] == fx["parent"].id
    stored = json.loads(canonical_strategy_json(child.spec))
    assert stored["rules"][0]["condition"]["op"] == "and"


def test_existing_new_version_from_keeps_max_version_compatibility(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    repo: StrategyRepository = fx["strategy_repo"]
    first = repo.new_version_from(
        fx["parent"].id,
        new_strategy_id="branch-a",
    )
    second = repo.new_version_from(
        fx["parent"].id,
        new_strategy_id="branch-b",
    )

    assert repo.get_strategy_meta(first.id)["version"] == 2
    assert repo.get_strategy_meta(second.id)["version"] == 3


def test_red_monitoring_handoff_creates_governed_child_task_dataset_and_strategy(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    repo = StrategyHandoffRepository(fx["db_path"], fx["datasets_root"])

    result = repo.create_new_version_from_red_run(
        source_task_id=fx["source_task"].id,
        parent_strategy_id=fx["parent"].id,
        monitoring_run_id="run-red",
        new_task_id="task-child",
        new_strategy_id="strategy-child",
        new_dataset_id="dataset-child",
        actor="risk-owner",
        created_at="2026-07-18T02:00:00+00:00",
    )

    assert result == {
        "source_task_id": fx["source_task"].id,
        "new_task_id": "task-child",
        "parent_strategy_id": fx["parent"].id,
        "new_strategy_id": "strategy-child",
        "monitoring_run_id": "run-red",
        "monitoring_plan_id": "plan-1",
        "source_dataset_id": "dataset-monitoring",
        "new_dataset_id": "dataset-child",
        "dataset_content_hash": fx["dataset_hash"],
    }
    child_task = fx["task_repo"].get_task("task-child")
    assert child_task.task_type == TASK_TYPE_STRATEGY
    assert child_task.target_col == "bad_flag"
    assert child_task.score_col == "risk_score"
    assert child_task.feature_columns == ["risk_score", "income"]
    assert child_task.strategy_input is not None
    assert child_task.strategy_input.baseline_strategy_id == fx["parent"].id
    assert child_task.strategy_input.max_bad_rate == 0.08
    assert child_task.strategy_input.profit is not None
    assert child_task.strategy_input.profit.ead_col == "ead"
    assert child_task.notebook_path is None
    assert child_task.sample_path is None
    assert child_task.report_values_revision == 0

    child_meta = fx["strategy_repo"].get_strategy_meta("strategy-child")
    assert child_meta["task_id"] == "task-child"
    assert child_meta["version"] == 2
    assert child_meta["status"] == "draft"
    assert child_meta["parent_strategy_id"] == fx["parent"].id

    child_dataset = DatasetRepository(fx["db_path"]).get_dataset("dataset-child")
    assert child_dataset is not None
    assert child_dataset.task_id == "task-child"
    assert child_dataset.source_path == fx["source_dataset"].source_path
    assert child_dataset.content_hash == fx["dataset_hash"]
    assert child_dataset.role == "strategy.new_version_source"

    with connect(fx["db_path"]) as conn:
        audit = conn.execute(
            "SELECT actor, target_ref, outcome, detail_json FROM audit WHERE kind = ?",
            ("strategy.monitoring.new_version_handoff",),
        ).fetchone()
    assert audit is not None
    assert audit["actor"] == "risk-owner"
    assert audit["target_ref"] == "run-red"
    assert audit["outcome"] == "succeeded"
    detail = json.loads(str(audit["detail_json"]))
    assert detail["new_task_id"] == "task-child"
    assert detail["new_strategy_id"] == "strategy-child"
    assert detail["dataset_content_hash"] == fx["dataset_hash"]
    assert "rows" not in detail


def test_red_monitoring_handoff_preserves_unconfigured_metrics(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, metrics=None)
    repo = StrategyHandoffRepository(fx["db_path"], fx["datasets_root"])

    repo.create_new_version_from_red_run(
        source_task_id=fx["source_task"].id,
        parent_strategy_id=fx["parent"].id,
        monitoring_run_id="run-red",
        new_task_id="task-child",
        new_strategy_id="strategy-child",
        new_dataset_id="dataset-child",
        actor="risk-owner",
        created_at="2026-07-18T02:00:00+00:00",
    )

    child_task = fx["task_repo"].get_task("task-child")
    assert child_task.metrics is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("foreign_parent_task", "does not belong to source task"),
        ("non_red", "must be red"),
        ("stale_run", "latest monitoring run"),
        ("stale_plan", "latest monitoring plan"),
        ("foreign_dataset", "does not belong to source task"),
        ("live_hash_drift", "live content hash"),
        ("semantic_mismatch", "semantic contract"),
    ],
)
def test_handoff_fails_closed_for_stale_or_cross_task_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    fx = _fixture(
        tmp_path,
        run_check_level="green" if mutation == "semantic_mismatch" else None,
    )
    source_task_id = fx["source_task"].id
    if mutation == "foreign_parent_task":
        source_task_id = fx["task_repo"].create_task(_task_payload(name="foreign")).id
    elif mutation == "non_red":
        with connect(fx["db_path"]) as conn:
            conn.execute("DELETE FROM strategy_monitoring_runs WHERE id = 'run-red'")
        _seed_monitoring_evidence(
            fx["db_path"],
            strategy_id=fx["parent"].id,
            dataset_id=fx["source_dataset"].id,
            dataset_hash=fx["dataset_hash"],
            plan_id="plan-1",
            plan_revision=1,
            run_id="run-red",
            level="amber",
        )
    elif mutation == "stale_run":
        _seed_monitoring_evidence(
            fx["db_path"],
            strategy_id=fx["parent"].id,
            dataset_id=fx["source_dataset"].id,
            dataset_hash=fx["dataset_hash"],
            plan_id="plan-2",
            plan_revision=2,
            run_id="run-newer",
            level="green",
            created_at="2026-07-18T03:00:00+00:00",
        )
    elif mutation == "stale_plan":
        with connect(fx["db_path"]) as conn:
            payload = '{"revision":2}'
            conn.execute(
                """
                INSERT INTO strategy_monitoring_plans(
                    id, strategy_id, strategy_version, revision, schema_version,
                    payload_json, payload_hash, supersedes_plan_id, created_at
                ) VALUES ('plan-2', ?, 1, 2, 'strategy.monitoring_plan.v2', ?, ?, 'plan-1', ?)
                """,
                (
                    fx["parent"].id,
                    payload,
                    _sha_text(payload),
                    "2026-07-18T03:00:00+00:00",
                ),
            )
    elif mutation == "foreign_dataset":
        foreign_task = fx["task_repo"].create_task(_task_payload(name="foreign")).id
        with connect(fx["db_path"]) as conn:
            conn.execute(
                "UPDATE datasets SET task_id = ? WHERE id = ?",
                (foreign_task, fx["source_dataset"].id),
            )
    elif mutation == "live_hash_drift":
        fx["source_path"].write_bytes(b"tampered")

    repo = StrategyHandoffRepository(fx["db_path"], fx["datasets_root"])
    with connect(fx["db_path"]) as conn:
        before_counts = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "datasets", "strategies", "audit")
        )
    with pytest.raises(ConflictError, match=message):
        repo.create_new_version_from_red_run(
            source_task_id=source_task_id,
            parent_strategy_id=fx["parent"].id,
            monitoring_run_id="run-red",
        )

    with connect(fx["db_path"]) as conn:
        after_counts = tuple(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "datasets", "strategies", "audit")
        )
    assert after_counts == before_counts


def test_handoff_rejects_replay_of_same_monitoring_run(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    repo = StrategyHandoffRepository(fx["db_path"], fx["datasets_root"])
    first = repo.create_new_version_from_red_run(
        source_task_id=fx["source_task"].id,
        parent_strategy_id=fx["parent"].id,
        monitoring_run_id="run-red",
    )

    with pytest.raises(ConflictError, match="already created a new strategy version"):
        repo.create_new_version_from_red_run(
            source_task_id=fx["source_task"].id,
            parent_strategy_id=fx["parent"].id,
            monitoring_run_id="run-red",
        )

    with connect(fx["db_path"]) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE kind = ? AND target_ref = ?",
                ("strategy.monitoring.new_version_handoff", "run-red"),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE parent_strategy_id = ?",
                (fx["parent"].id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE id = ?", (first["new_task_id"],)
            ).fetchone()[0]
            == 1
        )


def test_handoff_rolls_back_task_dataset_strategy_and_audit_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    import marvis.repositories.strategy_handoff as handoff_module

    def fail_audit(*_args, **_kwargs):
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(handoff_module, "_write_audit_row", fail_audit)
    repo = StrategyHandoffRepository(fx["db_path"], fx["datasets_root"])
    with pytest.raises(RuntimeError, match="injected audit failure"):
        repo.create_new_version_from_red_run(
            source_task_id=fx["source_task"].id,
            parent_strategy_id=fx["parent"].id,
            monitoring_run_id="run-red",
            new_task_id="task-rolled-back",
            new_strategy_id="strategy-rolled-back",
            new_dataset_id="dataset-rolled-back",
        )

    with connect(fx["db_path"]) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE id = 'task-rolled-back'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM datasets WHERE id = 'dataset-rolled-back'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE id = 'strategy-rolled-back'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM audit WHERE target_ref = 'run-red'"
            ).fetchone()[0]
            == 0
        )
