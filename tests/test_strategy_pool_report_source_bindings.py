from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from marvis.packs.strategy import pool_impact_tools as pool_impact_tools_module
from marvis.packs.strategy import pool_tools as pool_tools_module
from marvis.data.workspace import DataSemanticMapping, DataWorkspaceDraft
from marvis.db import TaskRepository
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool_impact_tools import (
    StrategyPoolImpactArtifactBinding,
    load_strategy_pool_impact_artifact,
    require_strategy_pool_impact_artifact_binding_on_connection,
    run_measure_pool_impact,
)
from marvis.packs.strategy.pool_tools import (
    StrategyCandidatePoolArtifactBinding,
    load_current_strategy_candidate_pool_artifact,
    require_strategy_candidate_pool_artifact_binding_on_connection,
    run_set_pool_entry_action,
)
from marvis.packs.strategy.strategy import build_strategy
from marvis.repositories.data_workspace import DataWorkspaceRepository

from test_strategy_pool_impact_tools import _action, _setup


def _bindings(
    tmp_path: Path,
) -> tuple[
    dict,
    StrategyCandidatePoolArtifactBinding,
    StrategyPoolImpactArtifactBinding,
]:
    fixture = _setup(tmp_path)
    pool = load_current_strategy_candidate_pool_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        strategy_type="approval",
        expected_pool_revision=fixture["pool"]["revision"],
        expected_pool_snapshot_hash=fixture["pool"]["snapshot_hash"],
    )
    output = run_measure_pool_impact(
        fixture["request"],
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    impact = load_strategy_pool_impact_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_assessment_id=output["assessment_id"],
        expected_assessment_content_hash=output["content_hash"],
    )
    return fixture, pool, impact


def test_pool_and_impact_bindings_are_authenticated_and_caller_owned(
    tmp_path: Path,
) -> None:
    fixture, pool, impact = _bindings(tmp_path)

    assert pool.compiled_design["design_hash"] == impact.assessment["identity"][
        "design_hash"
    ]
    assert impact.stage == "development_backtest"
    assert impact.validation_status == "unvalidated"
    assert impact.assessment["lifecycle"] == {
        "candidate_stage": "development",
        "observation_stage": "backtested",
        "validation_status": "unvalidated",
        "creates_strategy": False,
        "adopted": False,
        "deployed": False,
    }
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
        require_strategy_pool_impact_artifact_binding_on_connection(conn, impact)
        assert conn.in_transaction
        conn.rollback()
    with connect(fixture["settings"].db_path) as conn:
        with pytest.raises(StrategyError, match="caller-owned transaction"):
            require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
        with pytest.raises(StrategyError, match="caller-owned transaction"):
            require_strategy_pool_impact_artifact_binding_on_connection(conn, impact)


def test_pool_and_impact_bindings_reject_stale_pool_head(tmp_path: Path) -> None:
    fixture, pool, impact = _bindings(tmp_path)
    entry = fixture["pool"]["entries"][0]
    changed = run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "rule_id": entry["rule_id"],
            "action": _action("approval"),
            "expected_pool_revision": fixture["pool"]["revision"],
            "expected_pool_snapshot_hash": fixture["pool"]["snapshot_hash"],
            "reason": "exercise downstream stale-head rejection",
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert changed["revision"] == fixture["pool"]["revision"] + 1

    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="no longer current"):
            require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)
        with pytest.raises(StrategyError, match="no longer current"):
            require_strategy_pool_impact_artifact_binding_on_connection(conn, impact)


def test_pool_and_impact_bindings_reject_cross_task_or_artifact_drift(
    tmp_path: Path,
) -> None:
    fixture, pool, impact = _bindings(tmp_path)
    foreign = TaskRepository(fixture["settings"].db_path).create_task(
        TaskCreate(
            model_name="foreign",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
        )
    )
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(
            StrategyError,
            match="ownership|identity|another task|no longer current",
        ):
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                replace(pool, task_id=foreign.id),
            )
        with pytest.raises(
            StrategyError,
            match="another task|identity|no longer current",
        ):
            require_strategy_pool_impact_artifact_binding_on_connection(
                conn,
                replace(impact, task_id=foreign.id),
            )
        with pytest.raises(
            StrategyError,
            match="registered|binding|disappeared|artifact link",
        ):
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                replace(pool, artifact_id="f" * 64),
            )
        with pytest.raises(StrategyError, match="registered|binding|disappeared"):
            require_strategy_pool_impact_artifact_binding_on_connection(
                conn,
                replace(impact, artifact_id="e" * 64),
            )


@pytest.mark.parametrize("source", ["pool_artifact", "candidate_source"])
def test_pool_binding_rejects_file_or_authenticated_lineage_toctou(
    tmp_path: Path,
    source: str,
) -> None:
    fixture, pool, _impact = _bindings(tmp_path)
    if source == "pool_artifact":
        path = pool.artifact_path
    else:
        path = pool.lineages[0].asset_record.path
    path.write_bytes(path.read_bytes() + b"tampered")

    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="hash|bytes|content"):
            require_strategy_candidate_pool_artifact_binding_on_connection(conn, pool)


@pytest.mark.parametrize("source", ["impact_artifact", "dataset", "sample_design"])
def test_impact_binding_rejects_file_or_upstream_toctou(
    tmp_path: Path,
    source: str,
) -> None:
    fixture, _pool, impact = _bindings(tmp_path)
    paths = {
        "impact_artifact": impact.artifact_path,
        "dataset": impact.dataset.path,
        "sample_design": impact.sample_design.artifact.path,
    }
    path = paths[source]
    path.write_bytes(path.read_bytes() + b"tampered")

    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="hash|bytes|binding"):
            require_strategy_pool_impact_artifact_binding_on_connection(conn, impact)


def test_impact_binding_rejects_lifecycle_or_compiled_design_drift(
    tmp_path: Path,
) -> None:
    fixture, pool, impact = _bindings(tmp_path)
    changed_design = {
        **pool.compiled_design,
        "requirements": [{"forged": True}],
    }
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="stage"):
            require_strategy_pool_impact_artifact_binding_on_connection(
                conn,
                replace(impact, stage="independent_oot"),
            )
        with pytest.raises(StrategyError, match="compiled.*changed"):
            require_strategy_candidate_pool_artifact_binding_on_connection(
                conn,
                replace(pool, compiled_design=changed_design),
            )


def test_impact_binding_authenticates_baseline_and_workspace_inputs(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    baseline = build_strategy(
        "approval",
        [
            {
                "condition": "loan_amount < 150",
                "decision": "reject",
                "value": None,
            }
        ],
        score_col="score",
        default_decision="approve",
        description="report-source baseline",
    )
    fixture["runtime"].strategies.create_strategy(fixture["task"].id, baseline)
    output = run_measure_pool_impact(
        {
            **fixture["request"],
            "comparison_mode": "vs_baseline",
            "baseline_strategy_id": baseline.id,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = output["artifacts"][0]
    impact = load_strategy_pool_impact_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
    )
    assert impact.baseline is not None
    assert impact.baseline.strategy_id == baseline.id

    advanced = DataWorkspaceRepository(fixture["settings"].db_path).save(
        fixture["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=fixture["dataset"].id,
            active_dataset_content_hash=fixture["dataset"].content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col=fixture["mapping"].target_col,
                field_roles=dict(fixture["mapping"].field_roles),
                business_names={"score": "评分"},
            ),
        ),
        expected_revision=fixture["workspace"].revision,
    )
    assert advanced.revision == fixture["workspace"].revision + 1
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="DataWorkspace|workspace"):
            require_strategy_pool_impact_artifact_binding_on_connection(conn, impact)


@pytest.mark.parametrize("source", ["pool", "impact"])
def test_report_source_binding_rejects_same_content_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    fixture, pool, impact = _bindings(tmp_path)
    if source == "pool":
        module = pool_tools_module
        path = pool.artifact_path

        def revalidate(conn):
            return require_strategy_candidate_pool_artifact_binding_on_connection(
                conn, pool
            )
    else:
        module = pool_impact_tools_module
        path = impact.artifact_path

        def revalidate(conn):
            return require_strategy_pool_impact_artifact_binding_on_connection(
                conn, impact
            )

    original_read = module.os.read
    original_fstat = module.os.fstat
    target_stat = path.stat()
    raw = path.read_bytes()
    backup = path.with_name(path.name + ".before-race")
    replaced = False

    def racing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, count)
        observed = original_fstat(descriptor)
        if (
            not replaced
            and chunk
            and (observed.st_dev, observed.st_ino)
            == (target_stat.st_dev, target_stat.st_ino)
        ):
            path.rename(backup)
            path.write_bytes(raw)
            replaced = True
        return chunk

    monkeypatch.setattr(module.os, "read", racing_read)
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(StrategyError, match="path changed|content hash drifted"):
            revalidate(conn)
    assert replaced is True
