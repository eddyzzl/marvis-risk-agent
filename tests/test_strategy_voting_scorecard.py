from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from marvis.packs.strategy import voting_candidate_tools
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.pool_tools import (
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
)
from marvis.packs.strategy.pool_validation_tools import (
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.scorecard_candidate_tools import (
    run_build_scorecard_band_asset,
)
from marvis.packs.strategy.voting_candidate_tools import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    load_verified_voting_candidate_artifact,
    run_build_voting_candidate,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_pool_scorecard import (
    _add_inputs,
    _real_scorecard,
    _selection,
)


def _two_scorecard_pool_entries(tmp_path: Path) -> dict:
    real = _real_scorecard(tmp_path)
    first_selection = _selection(real, ordinal=0)
    first = run_add_candidate_to_pool(
        _add_inputs(
            first_selection,
            expected_revision=0,
            expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    first_asset = real["band"]["scorecard_band_asset"]
    first_cutoff = first_asset["cutoffs"][0]["execution_pd"]
    lower_average = first_asset["bands"][0]["average_pd"]
    assert lower_average is not None
    second_edge = (float(lower_average) + float(first_cutoff)) / 2.0
    score_ref = first_asset["source_refs"]["score_evidence"]
    vector_ref = first_asset["source_refs"]["score_vector"]
    second_band = run_build_scorecard_band_asset(
        {
            "score_evidence_ref": {
                "evidence_artifact_id": score_ref["artifact_id"],
                "expected_evidence_artifact_content_hash": score_ref[
                    "artifact_content_hash"
                ],
                "score_vector_artifact_id": vector_ref["artifact_id"],
                "expected_score_vector_artifact_content_hash": vector_ref[
                    "artifact_content_hash"
                ],
            },
            "sample_design_ref": first_asset["sample_design_ref"],
            "raw_pd_band_edges": [0.0, second_edge, 1.0],
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    second_selection = _selection(
        {**real, "band": second_band},
        ordinal=0,
    )
    second = run_add_candidate_to_pool(
        _add_inputs(
            second_selection,
            expected_revision=first["revision"],
            expected_snapshot_hash=first["snapshot_hash"],
        ),
        real["fx"]["ctx"],
        real["runtime"],
    )
    return {**real, "pool": second["pool"]}


def _voting_inputs(real: dict) -> dict:
    pool = real["pool"]
    return {
        "strategy_type": "approval",
        "expected_pool_revision": pool["revision"],
        "expected_pool_snapshot_hash": pool["snapshot_hash"],
        "selected_entry_ids": [
            entry["entry_id"] for entry in pool["entries"]
        ],
        "n": 2,
    }


def _voting_records(real: dict) -> list[dict]:
    return [
        record
        for record in TaskArtifactRepository(
            real["fx"]["settings"].db_path
        ).list_for_task(real["fx"]["task"].id)
        if record["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]


@pytest.mark.slow
def test_scorecard_cutoffs_build_load_and_replay_as_voting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = _two_scorecard_pool_entries(tmp_path)
    inputs = _voting_inputs(real)

    built = run_build_voting_candidate(
        inputs,
        real["fx"]["ctx"],
        real["runtime"],
    )

    descriptor = built["artifacts"][0]
    loaded = load_verified_voting_candidate_artifact(
        real["runtime"],
        task_id=real["fx"]["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_content_hash=descriptor["content_hash"],
        expected_asset_id=built["asset_id"],
        expected_asset_hash=built["asset_hash"],
    )
    asset = loaded.asset
    assert [
        entry["requirements"] for entry in asset["selected_entries"]
    ] == [
        entry["execution"]["requirements"] for entry in real["pool"]["entries"]
    ]
    assert all(
        len(entry["requirements"]) == 1
        for entry in asset["selected_entries"]
    )
    assert len(asset["fragment"]["requirements"]) == 2
    requirements = [
        item["requirement"]
        for item in asset["fragment"]["requirements"]
    ]
    assert {item["type"] for item in requirements} == {
        "model_score_vector.v1"
    }
    assert len(
        {item["score_vector_artifact_id"] for item in requirements}
    ) == 1
    assert len(_voting_records(real)) == 1

    # The public resolver boundary must fail closed before a second write.
    original_resolve = voting_candidate_tools.resolve_pool_requirements

    def reject_virtual_collision(*args, **kwargs):
        raise voting_candidate_tools.StrategyError(
            "virtual score field conflicts with physical dataset column"
        )

    monkeypatch.setattr(
        voting_candidate_tools,
        "resolve_pool_requirements",
        reject_virtual_collision,
    )
    with pytest.raises(
        voting_candidate_tools.StrategyError,
        match="virtual score field conflicts",
    ):
        run_build_voting_candidate(
            inputs,
            real["fx"]["ctx"],
            real["runtime"],
        )
    assert len(_voting_records(real)) == 1
    monkeypatch.setattr(
        voting_candidate_tools,
        "resolve_pool_requirements",
        original_resolve,
    )

    def reject_sample_mismatch(*args, **kwargs):
        raise voting_candidate_tools.StrategyError(
            "model score evidence does not bind selected SampleDesign V2"
        )

    monkeypatch.setattr(
        voting_candidate_tools,
        "resolve_pool_requirements",
        reject_sample_mismatch,
    )
    with pytest.raises(
        voting_candidate_tools.StrategyError,
        match="does not bind selected SampleDesign V2",
    ):
        run_build_voting_candidate(
            inputs,
            real["fx"]["ctx"],
            real["runtime"],
        )
    assert len(_voting_records(real)) == 1
    monkeypatch.setattr(
        voting_candidate_tools,
        "resolve_pool_requirements",
        original_resolve,
    )

    score_ref = asset["selected_entries"][0]["requirements"][0]
    evidence_record = real["runtime"].task_artifacts.get_for_task(
        real["fx"]["task"].id,
        score_ref["score_evidence_artifact_id"],
    )
    assert evidence_record is not None
    evidence_path = Path(evidence_record["path"])
    evidence_bytes = evidence_path.read_bytes()
    original_transaction = real["runtime"].task_artifacts.transaction

    @contextmanager
    def drift_score_evidence_during_registration():
        evidence_path.write_bytes(evidence_bytes + b" ")
        try:
            with original_transaction() as conn:
                yield conn
        finally:
            evidence_path.write_bytes(evidence_bytes)

    monkeypatch.setattr(
        real["runtime"].task_artifacts,
        "transaction",
        drift_score_evidence_during_registration,
    )
    with pytest.raises(
        voting_candidate_tools.StrategyError,
        match="content|evidence|artifact",
    ):
        run_build_voting_candidate(
            inputs,
            real["fx"]["ctx"],
            real["runtime"],
        )
    assert evidence_path.read_bytes() == evidence_bytes
    assert len(_voting_records(real)) == 1
    monkeypatch.setattr(
        real["runtime"].task_artifacts,
        "transaction",
        original_transaction,
    )

    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": descriptor["artifact_id"],
            "expected_artifact_content_hash": descriptor["content_hash"],
            "expected_asset_id": built["asset_id"],
            "expected_asset_hash": built["asset_hash"],
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
            "expected_pool_revision": real["pool"]["revision"],
            "expected_pool_snapshot_hash": real["pool"]["snapshot_hash"],
            "placement_mode": "replace_selected_members",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    compiled = run_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": admitted["revision"],
            "expected_pool_snapshot_hash": admitted["snapshot_hash"],
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    voting_rule = next(
        rule
        for rule in compiled["strategy_spec"]["rules"]
        if rule["rule_id"] == built["rule_id"]
    )
    assert voting_rule["condition"]["op"] == "n_of_k"
    assert voting_rule["condition"]["n"] == 2

    pool_artifact = admitted["artifacts"][0]
    pool_ref = {
        "artifact_id": pool_artifact["artifact_id"],
        "expected_artifact_content_hash": pool_artifact["content_hash"],
        "expected_pool_id": admitted["pool_id"],
        "expected_revision": admitted["revision"],
        "expected_revision_id": admitted["pool"]["revision_id"],
        "expected_snapshot_hash": admitted["snapshot_hash"],
    }
    validation = run_measure_strategy_pool_validation(
        {
            "strategy_type": "approval",
            "pool_ref": pool_ref,
            "sample_design_ref": real["fx"]["sample_ref"],
            "partition": "validation",
            "population": "risk",
            "comparison_mode": "absolute",
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert validation["population_count"] > 0
    assert validation["evidence"]["identity"]["pool_id"] == admitted["pool_id"]

    impact = run_measure_strategy_impact_cube(
        {
            "strategy_type": "approval",
            "pool_ref": pool_ref,
            "sample_design_ref": real["fx"]["sample_ref"],
            "partitions": ["development", "validation"],
            "population": "risk",
            "dimension_bindings": {
                "month_col": "apply_month",
                "group_col": "channel",
                "segment_col": "sample_partition",
            },
            "current_strategy_ref": None,
            "economics_inputs": None,
        },
        real["fx"]["ctx"],
        real["runtime"],
    )
    assert impact["pool_id"] == admitted["pool_id"]
    assert impact["partitions"] == ["development", "validation"]
