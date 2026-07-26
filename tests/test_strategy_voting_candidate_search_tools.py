from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import pytest

import marvis.packs.strategy.voting_candidate_search_tools as search_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import (
    bind_strategy_pool_development_execution,
    load_current_strategy_candidate_pool_artifact,
    run_add_candidate_to_pool,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
    load_voting_candidate_search_artifact,
    resolve_voting_candidate_search_inputs,
    run_search_voting_candidates,
)
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_candidate_stability_tools import (
    _pool_add_inputs,
    _setup,
)


def _search_fixture(tmp_path: Path) -> dict:
    fixture = _setup(tmp_path)
    pool = None
    for candidate in (
        fixture["first"],
        fixture["refine"](1),
        fixture["refine"](2),
    ):
        added = run_add_candidate_to_pool(
            _pool_add_inputs(
                candidate,
                expected_revision=0 if pool is None else pool["revision"],
                expected_hash=(
                    ABSENT_POOL_SNAPSHOT_HASH if pool is None else pool["snapshot_hash"]
                ),
            ),
            fixture["ctx"],
            fixture["runtime"],
        )
        pool = added["pool"]
    assert pool is not None
    controls = {
        "strategy_type": "approval",
        "member_count": 2,
        "n": 1,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [{"metric": "hit_share", "operator": "gte", "value": 0.05}],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 100,
    }
    return {**fixture, "pool": pool, "controls": controls}


def test_search_recovers_current_pool_and_persists_only_aggregate_evidence(
    tmp_path: Path,
) -> None:
    fixture = _search_fixture(tmp_path)
    task_id = fixture["task"].id
    before_pool = StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(task_id, "approval")

    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=task_id,
        user_controls=fixture["controls"],
    )
    first = run_search_voting_candidates(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    replay = run_search_voting_candidates(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert replay == first
    assert first["search_space"] == 3
    assert first["evaluated"] == 3
    assert first["truncated"] is False
    assert first["excluded_unsupported_rule_ids"] == []
    assert first["not_mutated_pool"] is True
    assert first["not_selected"] is True
    assert first["not_admitted"] is True
    assert first["not_applied"] is True
    assert first["not_adopted"] is True
    assert first["not_deployed"] is True
    assert (
        StrategyCandidatePoolRepository(fixture["settings"].db_path).get_current(
            task_id, "approval"
        )
        == before_pool
    )

    [descriptor] = first["artifacts"]
    assert descriptor["kind"] == VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        task_id, descriptor["artifact_id"]
    )
    assert record is not None
    persisted = json.loads(Path(record["path"]).read_text("utf-8"))
    assert persisted == first["search_result"]
    assert {
        "hit_matrix",
        "target",
        "weights",
        "amounts",
        "winner",
        "champion",
        "selected",
    }.isdisjoint(persisted)
    assert record["provenance"]["pool_ref"] == {
        "artifact_id": inputs["pool_ref"]["artifact_id"],
        "artifact_content_hash": inputs["pool_ref"]["expected_artifact_content_hash"],
        "pool_id": fixture["pool"]["pool_id"],
        "strategy_type": "approval",
        "revision": fixture["pool"]["revision"],
        "revision_id": fixture["pool"]["revision_id"],
        "snapshot_hash": fixture["pool"]["snapshot_hash"],
    }
    assert (
        record["provenance"]["dataset_binding"]["dataset_id"] == fixture["dataset"].id
    )
    assert record["provenance"]["target_binding"]["column"] == "bad"
    assert record["provenance"]["requirement_bindings"] is None

    loaded = load_voting_candidate_search_artifact(
        fixture["runtime"],
        task_id=task_id,
        artifact_id=descriptor["artifact_id"],
        expected_artifact_content_hash=descriptor["content_hash"],
        expected_search_id=first["search_id"],
        expected_search_content_hash=first["content_hash"],
    )
    assert loaded.result == first["search_result"]


def test_search_rejects_user_injection_of_platform_or_row_bindings() -> None:
    controls = {
        "strategy_type": "approval",
        "member_count": 2,
        "n": 1,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 100,
    }
    for field, value in (
        ("pool_ref", {}),
        ("dataset_id", "forged"),
        ("target", [0, 1]),
        ("hit_matrix", [[True, False]]),
        ("weights", [1.0, 1.0]),
        ("amounts", [100.0, 200.0]),
    ):
        with pytest.raises(StrategyError, match="fields are invalid"):
            resolve_voting_candidate_search_inputs(
                None,
                task_id="task-owned",
                user_controls={**controls, field: value},
            )


def test_strategy_pack_registers_the_governed_search_tool() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    [tool] = [
        item for item in manifest["tools"] if item["name"] == "search_voting_candidates"
    ]

    assert tool["entrypoint"] == "tool_search_voting_candidates"
    assert tool["determinism"] == "deterministic"
    assert tool["input_schema"]["properties"]["member_count"]["maximum"] == 50
    assert (
        tool["output_schema"]["properties"]["excluded_unsupported_rule_ids"]["type"]
        == "array"
    )


@pytest.mark.parametrize("objective_metric", ["bad_rate", "lift"])
def test_minimize_rate_or_lift_requires_positive_minimum_hit_constraint(
    objective_metric: str,
) -> None:
    controls = {
        "strategy_type": "approval",
        "member_count": 2,
        "n": 1,
        "objective": {
            "metric": objective_metric,
            "direction": "minimize",
        },
        "constraints": [],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 100,
    }
    with pytest.raises(StrategyError, match="positive minimum hit"):
        resolve_voting_candidate_search_inputs(
            None,
            task_id="task-owned",
            user_controls=controls,
        )
    with pytest.raises(StrategyError, match="positive minimum hit"):
        resolve_voting_candidate_search_inputs(
            None,
            task_id="task-owned",
            user_controls={
                **controls,
                "constraints": [
                    {
                        "metric": "hit_share",
                        "operator": "gte",
                        "value": 0,
                    }
                ],
            },
        )


def test_search_registration_cas_rejects_dataset_byte_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _search_fixture(tmp_path)
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls=fixture["controls"],
    )
    pool_binding = load_current_strategy_candidate_pool_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        strategy_type="approval",
        expected_pool_revision=fixture["pool"]["revision"],
        expected_pool_snapshot_hash=fixture["pool"]["snapshot_hash"],
    )
    dataset_path = bind_strategy_pool_development_execution(
        fixture["runtime"],
        pool_binding,
    ).dataset.path
    original_bytes = dataset_path.read_bytes()
    original_transaction = fixture["runtime"].task_artifacts.transaction
    output_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_voting_candidate_searches"
    )

    @contextmanager
    def drift_only_during_registration():
        drifted = output_dir.exists()
        if drifted:
            dataset_path.write_bytes(original_bytes + b" ")
        try:
            with original_transaction() as conn:
                yield conn
        finally:
            if drifted:
                dataset_path.write_bytes(original_bytes)

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "transaction",
        drift_only_during_registration,
    )

    with pytest.raises(StrategyError, match="dataset|content"):
        run_search_voting_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )

    assert dataset_path.read_bytes() == original_bytes
    assert not [
        record
        for record in TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
            fixture["task"].id
        )
        if record["kind"] == VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND
    ]
    assert not list(output_dir.glob("*.json"))


def test_search_budget_rejects_before_backend_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _search_fixture(tmp_path)
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls=fixture["controls"],
    )
    backend_read = False

    def reject_backend_read(*args, **kwargs):
        nonlocal backend_read
        backend_read = True
        raise AssertionError("backend must not be read before budget admission")

    monkeypatch.setattr(
        fixture["runtime"].backend,
        "read_frame",
        reject_backend_read,
    )
    monkeypatch.setattr(search_tools, "MAX_MATRIX_CELLS", 1)

    with pytest.raises(StrategyError, match="hit matrix exceeds"):
        run_search_voting_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )

    assert backend_read is False


def test_search_artifact_write_rolls_back_when_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _search_fixture(tmp_path)
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls=fixture["controls"],
    )

    def reject_registration(*args, **kwargs):
        raise StrategyError("forced Voting search registration failure")

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        reject_registration,
    )

    with pytest.raises(StrategyError, match="forced Voting search"):
        run_search_voting_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )

    output_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_voting_candidate_searches"
    )
    assert not list(output_dir.glob("*.json"))
    assert not [
        record
        for record in TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
            fixture["task"].id
        )
        if record["kind"] == VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND
    ]


def test_search_uses_governed_optional_amount_semantics(tmp_path: Path) -> None:
    from tests.test_strategy_voting_candidate_tool import (
        _setup as voting_setup,
    )

    fixture = voting_setup(tmp_path)
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls={
            "strategy_type": "approval",
            "member_count": 2,
            "n": 1,
            "objective": {
                "metric": "bad_amount_capture_rate",
                "direction": "maximize",
            },
            "constraints": [],
            "include_rule_ids": [],
            "exclude_rule_ids": [],
            "max_combinations": 10,
        },
    )

    output = run_search_voting_candidates(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["search_result"]["population"]["amount"]["available"] is True
    assert output["search_result"]["population"]["amount"]["total"] == 2520.0
    assert output["search_result"]["population"]["weight"]["available"] is False
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["provenance"]["observation_bindings"] == {
        "weight_col": None,
        "amount_col": "loan_amount",
    }


@pytest.mark.slow
def test_scorecard_search_hydrates_full_universe_and_excludes_existing_voting(
    tmp_path: Path,
) -> None:
    from tests.test_strategy_voting_scorecard import (
        _two_scorecard_pool_entries,
        _voting_inputs,
    )
    from marvis.packs.strategy.voting_candidate_tools import (
        run_build_voting_candidate,
    )

    fixture = _two_scorecard_pool_entries(tmp_path)
    pool = fixture["pool"]
    controls = {
        "strategy_type": "approval",
        "member_count": 2,
        "n": 1,
        "objective": {
            "metric": "bad_capture_rate",
            "direction": "maximize",
        },
        "constraints": [{"metric": "hit_share", "operator": "gte", "value": 0.01}],
        "include_rule_ids": [],
        "exclude_rule_ids": [],
        "max_combinations": 10,
    }
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["fx"]["task"].id,
        user_controls=controls,
    )

    output = run_search_voting_candidates(
        inputs,
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )

    assert output["search_space"] == 1
    assert output["evaluated"] == 1
    assert output["truncated"] is False
    record = TaskArtifactRepository(fixture["fx"]["settings"].db_path).get_for_task(
        fixture["fx"]["task"].id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    requirements = record["provenance"]["requirement_bindings"]
    assert requirements is not None
    assert len(requirements["requirements"]) == 2
    assert len(requirements["virtual_fields"]) == 1
    assert output["search_result"]["configuration"]["candidate_ids"] == sorted(
        entry["rule_id"] for entry in pool["entries"]
    )

    voting = run_build_voting_candidate(
        _voting_inputs(fixture),
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    voting_artifact = voting["artifacts"][0]
    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": voting_artifact["artifact_id"],
            "expected_artifact_content_hash": voting_artifact["content_hash"],
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
            "placement_mode": "before_selected_members",
        },
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    nested_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["fx"]["task"].id,
        user_controls=controls,
    )
    nested = run_search_voting_candidates(
        nested_inputs,
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    nested_record = TaskArtifactRepository(
        fixture["fx"]["settings"].db_path
    ).get_for_task(
        fixture["fx"]["task"].id,
        nested["artifacts"][0]["artifact_id"],
    )
    assert nested_record is not None
    nested_requirements = nested_record["provenance"]["requirement_bindings"]
    assert nested_requirements is not None
    assert len(nested_requirements["requirements"]) == 2
    assert len(nested_requirements["virtual_fields"]) == 1
    assert nested["excluded_unsupported_rule_ids"] == [voting["rule_id"]]
    assert nested_record["provenance"]["excluded_unsupported_rule_ids"] == [
        voting["rule_id"]
    ]
    assert (
        nested["artifacts"][0]["artifact_id"] != output["artifacts"][0]["artifact_id"]
    )
    assert nested["search_result"]["configuration"]["candidate_ids"] == sorted(
        entry["rule_id"]
        for entry in admitted["pool"]["entries"]
        if entry["source"]["asset_type"] != "voting_n_of_k"
    )
