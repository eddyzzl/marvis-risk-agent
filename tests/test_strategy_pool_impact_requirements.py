from __future__ import annotations

from pathlib import Path

import pytest

from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.packs.strategy import pool_impact_tools as impact_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_impact_tools import (
    load_historical_strategy_pool_impact_artifact,
    load_strategy_pool_impact_artifact,
    require_historical_strategy_pool_impact_artifact_binding_on_connection,
    require_strategy_pool_impact_artifact_binding_on_connection,
    run_measure_pool_impact,
)
from marvis.packs.strategy.pool_requirement_resolver import (
    project_pool_entry_requirements,
)
from marvis.packs.strategy.pool_tools import (
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    run_add_candidate_to_pool,
)
from marvis.packs.strategy.voting_candidate_tools import (
    run_build_voting_candidate,
)
from marvis.packs.strategy.strategy import build_strategy
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_model_evidence_tool import _native_fixture
from tests.test_strategy_pool_tools import _add_inputs as _pool_add_inputs
from tests.test_strategy_voting_scorecard import (
    _two_scorecard_pool_entries,
    _voting_inputs,
)


def _impact_request(real: dict, pool: dict) -> dict:
    runtime = real["runtime"]
    task_id = real["fx"]["task"].id
    pool_binding = load_current_strategy_candidate_pool_artifact(
        runtime,
        task_id=task_id,
        strategy_type="approval",
        expected_pool_revision=pool["revision"],
        expected_pool_snapshot_hash=pool["snapshot_hash"],
    )
    development = bind_strategy_pool_development_execution(
        runtime,
        pool_binding,
    )
    sample = development.sample_design
    return {
        "strategy_type": "approval",
        "expected_pool_revision": pool["revision"],
        "expected_pool_snapshot_hash": pool["snapshot_hash"],
        "dataset_id": development.dataset.dataset_id,
        "expected_dataset_content_hash": development.dataset.content_hash,
        "workspace_revision": sample.workspace_revision,
        "workspace_generation": sample.workspace_generation,
        "semantic_mapping_hash": sample.semantic_mapping_hash,
        "target_col": sample.target_col,
        "sample_design_ref": sample.to_ref_dict(),
        "comparison_mode": "absolute",
        "drop_nan_labels": sample.drop_nan_labels,
        "month_col": sample.month_col,
        "loan_amount_col": sample.loan_amount_col,
        "overdue_amount_col": sample.overdue_amount_col,
    }


@pytest.mark.slow
def test_pool_impact_executes_real_scorecard_and_voting_requirements_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _two_scorecard_pool_entries(tmp_path)
    pool = real["pool"]
    registration_checks: list[bool] = []
    original_requirement_check = (
        impact_tools.require_resolved_pool_requirements_on_connection
    )

    def observe_requirement_check(conn, resolved) -> None:
        registration_checks.append(conn.in_transaction)
        original_requirement_check(conn, resolved)

    monkeypatch.setattr(
        impact_tools,
        "require_resolved_pool_requirements_on_connection",
        observe_requirement_check,
    )
    output = run_measure_pool_impact(
        _impact_request(real, pool),
        real["fx"]["ctx"],
        real["runtime"],
    )

    requirements = list(project_pool_entry_requirements(pool["entries"]))
    assert requirements
    assert output["population_count"] == 6
    record = TaskArtifactRepository(
        real["fx"]["settings"].db_path
    ).get_for_task(
        real["fx"]["task"].id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    provenance = record["provenance"]
    assert provenance["schema_version"] == (
        "strategy.pool-impact-artifact.requirements.v1"
    )
    assert provenance["requirement_bindings"]["requirements"] == requirements
    assert provenance["requirement_bindings"]["virtual_fields"]
    assert registration_checks == [True]
    assert provenance["semantic_mapping_hash"] == data_semantic_mapping_hash(
        real["fx"]["workspace"].semantic_mapping
    )

    source = real["runtime"].backend.read_frame(
        real["runtime"].registry.resolve_verified_path(
            real["fx"]["dataset"].id
        )
    )
    assert set(
        provenance["requirement_bindings"]["virtual_fields"]
    ).isdisjoint(source.columns)

    descriptor = output["artifacts"][0]
    current = load_strategy_pool_impact_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
        expected_assessment_id=output["assessment_id"],
        expected_assessment_content_hash=output["content_hash"],
    )
    assert current.resolved_requirements is not None
    assert current.resolved_requirements.requirements == tuple(requirements)
    assert list(current.resolved_requirements.virtual_fields) == provenance[
        "requirement_bindings"
    ]["virtual_fields"]

    vector_id = requirements[0]["requirement"][
        "score_vector_artifact_id"
    ]
    with real["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM task_artifacts WHERE id = ?",
            (vector_id,),
        )
        with pytest.raises(StrategyError, match="disappeared before commit"):
            require_strategy_pool_impact_artifact_binding_on_connection(
                conn,
                current,
            )
        conn.rollback()

    voting = run_build_voting_candidate(
        _voting_inputs(real),
        real["fx"]["ctx"],
        real["runtime"],
    )
    voting_descriptor = voting["artifacts"][0]
    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": voting_descriptor["artifact_id"],
            "expected_artifact_content_hash": voting_descriptor[
                "content_hash"
            ],
            "expected_asset_id": voting["asset_id"],
            "expected_asset_hash": voting["asset_hash"],
            "strategy_type": "approval",
            "default_action": {
                "type": "approval",
                "value": "approve",
                "reason_code": None,
                "stop": True,
            },
            "action": {
                "type": "review",
                "value": "review",
                "reason_code": "SCORECARD_VOTING",
                "stop": True,
            },
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
            "placement_mode": "replace_selected_members",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    voting_pool = admitted["pool"]
    voting_output = run_measure_pool_impact(
        _impact_request(real, voting_pool),
        real["fx"]["ctx"],
        real["runtime"],
    )
    voting_requirements = list(
        project_pool_entry_requirements(voting_pool["entries"])
    )
    assert voting_requirements
    voting_artifact = voting_output["artifacts"][0]
    voting_record = TaskArtifactRepository(
        real["fx"]["settings"].db_path
    ).get_for_task(
        real["fx"]["task"].id,
        voting_artifact["artifact_id"],
    )
    assert voting_record is not None
    assert voting_record["provenance"]["requirement_bindings"][
        "requirements"
    ] == voting_requirements

    voting_current = load_strategy_pool_impact_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        artifact_id=voting_artifact["artifact_id"],
        expected_artifact_content_hash=voting_artifact["content_hash"],
        expected_assessment_id=voting_output["assessment_id"],
        expected_assessment_content_hash=voting_output["content_hash"],
    )
    assert voting_current.resolved_requirements is not None

    mapping = real["fx"]["workspace"].semantic_mapping
    advanced = DataWorkspaceRepository(
        real["fx"]["settings"].db_path
    ).save(
        real["fx"]["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=real["fx"]["dataset"].id,
            active_dataset_content_hash=real["fx"]["dataset"].content_hash,
            semantic_mapping=DataSemanticMapping(
                target_col=mapping.target_col,
                field_roles=mapping.field_roles,
                business_names={"legacy_score": "advanced workspace head"},
            ),
        ),
        expected_revision=real["fx"]["workspace"].revision,
    )
    assert advanced.revision > real["fx"]["workspace"].revision
    with pytest.raises(StrategyError, match="[Ww]orkspace|current"):
        load_strategy_pool_impact_artifact(
            real["runtime"],
            task_id=real["fx"]["task"].id,
            artifact_id=voting_artifact["artifact_id"],
            expected_artifact_content_hash=voting_artifact["content_hash"],
        )

    historical = load_historical_strategy_pool_impact_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        artifact_id=voting_artifact["artifact_id"],
        expected_artifact_content_hash=voting_artifact["content_hash"],
        expected_assessment_id=voting_output["assessment_id"],
        expected_assessment_content_hash=voting_output["content_hash"],
    )
    assert historical.resolved_requirements is not None
    assert historical.resolved_requirements.requirements == tuple(
        voting_requirements
    )
    with real["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_historical_strategy_pool_impact_artifact_binding_on_connection(
            conn,
            historical,
        )
        conn.rollback()

    vector_record = real["runtime"].task_artifacts.get_for_task(
        real["fx"]["task"].id,
        vector_id,
    )
    assert vector_record is not None
    vector_path = Path(vector_record["path"])
    original_vector = vector_path.read_bytes()
    vector_path.write_bytes(original_vector + b"corruption")
    try:
        with pytest.raises(StrategyError, match="score|vector|artifact|content"):
            load_historical_strategy_pool_impact_artifact(
                real["runtime"],
                task_id=real["fx"]["task"].id,
                artifact_id=voting_artifact["artifact_id"],
                expected_artifact_content_hash=voting_artifact[
                    "content_hash"
                ],
            )
    finally:
        vector_path.write_bytes(original_vector)


def test_native_bad_zero_pool_impact_rejects_target_leaking_baseline(
    tmp_path: Path,
) -> None:
    fixture = _native_fixture(tmp_path, target_bad_value=0)
    analysis = fixture["candidate"]
    report = next(
        artifact
        for artifact in analysis["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    feature_analysis = next(
        item
        for item in analysis["candidate_evidence"]["analysis"]["features"]
        if item["feature"] == "legacy_score"
    )
    method = feature_analysis["methods"][0]
    candidate = strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": report["artifact_id"],
            "expected_artifact_content_hash": report["content_hash"],
            "expected_candidate_id": analysis["candidate_id"],
            "expected_evidence_hash": analysis["evidence_hash"],
            "feature": "legacy_score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {
                "source_bin_ids": [method["bins"][0]["id"]],
            },
        },
        fixture["ctx"],
    )
    added = run_add_candidate_to_pool(
        _pool_add_inputs(
            candidate,
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    baseline = build_strategy(
        "approval",
        [
            {
                "condition": "bad == 0",
                "decision": "reject",
                "value": None,
            }
        ],
        score_col="legacy_score",
        default_decision="approve",
        description="invalid target-leaking baseline",
    )
    fixture["runtime"].strategies.create_strategy(
        fixture["task"].id,
        baseline,
    )
    request = _impact_request(
        {"fx": fixture, "runtime": fixture["runtime"]},
        added["pool"],
    )
    absolute = run_measure_pool_impact(
        request,
        fixture["ctx"],
        fixture["runtime"],
    )
    absolute_record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        absolute["artifacts"][0]["artifact_id"],
    )
    assert absolute_record is not None
    assert absolute["population_count"] == 2
    assert absolute_record["provenance"]["schema_version"] == (
        "strategy.pool-impact-artifact.v2"
    )
    assert "requirement_bindings" not in absolute_record["provenance"]
    assert absolute_record["provenance"]["source_target_bad_value"] == 0
    assert absolute_record["provenance"]["normalized_target_bad_value"] == 1
    before = [
        record["id"]
        for record in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if record["kind"] == "strategy_pool_impact_json"
    ]

    with pytest.raises(
        StrategyError,
        match="baseline strategy columns.*governed columns.*bad",
    ):
        run_measure_pool_impact(
            {
                **request,
                "comparison_mode": "vs_baseline",
                "baseline_strategy_id": baseline.id,
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    assert [
        record["id"]
        for record in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if record["kind"] == "strategy_pool_impact_json"
    ] == before
