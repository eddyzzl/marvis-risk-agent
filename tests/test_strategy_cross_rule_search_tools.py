from __future__ import annotations

import json
from pathlib import Path

import pytest

from marvis.packs.strategy.cross_rule_search_tools import (
    CROSS_RULE_CANDIDATE_ARTIFACT_KIND,
    CROSS_RULE_SEARCH_ARTIFACT_KIND,
    load_cross_rule_candidate_artifact,
    load_cross_rule_search_artifact,
    run_build_cross_rule_candidate_from_search,
    run_search_cross_threshold_rules,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import (
    ABSENT_POOL_SNAPSHOT_HASH,
    compile_strategy_pool,
)
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.plugins.loader import load_manifest
from marvis.plugins.manifest import ToolRef
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_cross_matrix_candidate_tool import (
    _replace_source_with_native_parallel_evidence,
    _setup,
)


def _inputs(fixture: dict, *, dimension: int = 2, max_trials: int = 20) -> dict:
    source = fixture["source"]
    artifact = next(
        item
        for item in source["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_candidate_id": source["candidate_id"],
        "expected_evidence_hash": source["evidence_hash"],
        "features": ["age", "score"],
        "dimension": dimension,
        "constraints": {
            "min_lift": 0.0,
            "min_bad_count": 0,
            "max_hit_share": 1.0,
            "min_amount_lift": 0.0,
        },
        "max_trials": max_trials,
    }


def test_search_reads_exact_development_sample_and_persists_only_aggregate_rules(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)

    output = run_search_cross_threshold_rules(
        _inputs(fixture, max_trials=4),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["schema_version"] == (
        "strategy.search-cross-threshold-rules-tool.v1"
    )
    assert output["search_result"]["configuration"]["dimension"] == 2
    assert output["search_result"]["population"]["row_count"] == 8
    assert output["evaluated"] == 4
    assert output["truncated"] is True
    assert output["not_selected"] is True
    assert output["not_admitted"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    record = repository.get_for_task(
        fixture["task"].id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["kind"] == CROSS_RULE_SEARCH_ARTIFACT_KIND
    raw = Path(record["path"]).read_text("utf-8")
    assert json.loads(raw) == output["search_result"]
    assert all(
        forbidden not in raw
        for forbidden in (
            '"rows"',
            '"row_ids"',
            '"target"',
            '"assignments"',
            '"winner"',
            '"champion"',
            '"recommended"',
        )
    )
    binding = load_cross_rule_search_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=record["id"],
        expected_artifact_content_hash=record["content_hash"],
        expected_search_id=output["search_id"],
        expected_search_content_hash=output["content_hash"],
    )
    assert binding.result == output["search_result"]


def test_search_missing_inclusion_is_evaluated_as_explicit_rule_branch(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, age_special="missing")

    output = run_search_cross_threshold_rules(
        _inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )

    age = next(
        item
        for item in output["search_result"]["configuration"]["features"]
        if item["feature"] == "age"
    )
    assert age["missing_count"] == 1
    assert any(
        condition["feature"] == "age" and condition["include_missing"]
        for rule in output["search_result"]["rules"]
        for condition in rule["conditions"]
    )


def test_search_supports_native_v2_risk_membership_and_rejects_missing_amount_evidence(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    native_ref = _replace_source_with_native_parallel_evidence(fixture)

    output = run_search_cross_threshold_rules(
        _inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert output["search_result"]["population"]["row_count"] == 6
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        output["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["provenance"]["sample_design_ref"] == native_ref

    no_amount = _setup(
        tmp_path / "no-amount",
        include_amount_columns=False,
    )
    with pytest.raises(StrategyError, match="amount"):
        run_search_cross_threshold_rules(
            _inputs(no_amount),
            no_amount["ctx"],
            no_amount["runtime"],
        )


def test_search_rejects_unavailable_features_dimension_or_row_budget(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    unknown = _inputs(fixture)
    unknown["features"] = ["age", "unknown"]
    with pytest.raises(StrategyError, match="ranking"):
        run_search_cross_threshold_rules(
            unknown,
            fixture["ctx"],
            fixture["runtime"],
        )

    bad_dimension = _inputs(fixture)
    bad_dimension["dimension"] = 3
    with pytest.raises(StrategyError, match="at least 3"):
        run_search_cross_threshold_rules(
            bad_dimension,
            fixture["ctx"],
            fixture["runtime"],
        )

    oversized = _inputs(fixture)
    oversized["max_trials"] = 5_001
    with pytest.raises(StrategyError, match="between 1 and 5000"):
        run_search_cross_threshold_rules(
            oversized,
            fixture["ctx"],
            fixture["runtime"],
        )


def test_exact_rule_pointer_materializes_replayed_candidate_without_pool_mutation(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    searched = run_search_cross_threshold_rules(
        _inputs(fixture, max_trials=4),
        fixture["ctx"],
        fixture["runtime"],
    )
    rule = searched["search_result"]["rules"][0]

    built = run_build_cross_rule_candidate_from_search(
        {
            "search_id": searched["search_id"],
            "rule_id": rule["rule_id"],
            "selection_reason": "业务明确点名该规则，先形成候选。",
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    assert built["source_search_selection"]["search_id"] == searched["search_id"]
    assert built["source_search_selection"]["rule_id"] == rule["rule_id"]
    assert built["candidate"]["metrics"] == rule["metrics"]
    assert built["candidate"]["lifecycle"] == {
        "admitted": False,
        "applied": False,
        "adopted": False,
        "deployed": False,
    }
    assert built["not_admitted"] is True
    assert built["not_applied"] is True
    assert built["not_adopted"] is True
    assert built["not_deployed"] is True

    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        built["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["kind"] == CROSS_RULE_CANDIDATE_ARTIFACT_KIND
    binding = load_cross_rule_candidate_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=record["id"],
        expected_artifact_content_hash=record["content_hash"],
        expected_asset_id=built["candidate"]["asset_id"],
        expected_asset_hash=built["candidate"]["asset_hash"],
    )
    assert binding.candidate == built["candidate"]


def test_rule_materialization_rejects_unknown_pointer_and_search_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    searched = run_search_cross_threshold_rules(
        _inputs(fixture, max_trials=4),
        fixture["ctx"],
        fixture["runtime"],
    )
    with pytest.raises(StrategyError, match="rule_id"):
        run_build_cross_rule_candidate_from_search(
            {
                "search_id": searched["search_id"],
                "rule_id": "cross-rule-" + "f" * 32,
                "selection_reason": None,
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    search_record = repository.get_for_task(
        fixture["task"].id,
        searched["artifacts"][0]["artifact_id"],
    )
    assert search_record is not None
    Path(search_record["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(StrategyError, match="search artifact"):
        run_build_cross_rule_candidate_from_search(
            {
                "search_id": searched["search_id"],
                "rule_id": searched["search_result"]["rules"][0]["rule_id"],
                "selection_reason": None,
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    assert all(
        item["kind"] != CROSS_RULE_CANDIDATE_ARTIFACT_KIND
        for item in repository.list_for_task(fixture["task"].id)
    )


def test_rule_search_and_materialization_tools_are_registered_with_hard_budgets(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tools = {item.name: item for item in manifest.tools}
    search = tools["search_cross_threshold_rules"]
    assert search.input_schema["properties"]["features"]["maxItems"] == 12
    assert search.input_schema["properties"]["dimension"]["enum"] == [2, 3]
    assert search.input_schema["properties"]["max_trials"]["maximum"] == 5_000
    assert search.side_effects == (
        "read:task",
        "read:dataset",
        "write:artifact",
    )
    build = tools["build_cross_rule_candidate_from_search"]
    assert build.input_schema["required"] == [
        "search_id",
        "rule_id",
        "selection_reason",
    ]

    fixture = _setup(tmp_path)
    invocation = fixture["runner"].invoke(
        ToolRef("strategy", "search_cross_threshold_rules"),
        _inputs(fixture, max_trials=4),
        task_id=fixture["task"].id,
    )
    assert invocation.ok, invocation.error
    rule = invocation.output["search_result"]["rules"][0]
    built = fixture["runner"].invoke(
        ToolRef("strategy", "build_cross_rule_candidate_from_search"),
        {
            "search_id": invocation.output["search_id"],
            "rule_id": rule["rule_id"],
            "selection_reason": None,
        },
        task_id=fixture["task"].id,
    )
    assert built.ok, built.error


def test_materialized_cross_rule_replays_into_current_pool_and_compiles(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    searched = run_search_cross_threshold_rules(
        _inputs(fixture, max_trials=4),
        fixture["ctx"],
        fixture["runtime"],
    )
    rule = searched["search_result"]["rules"][0]
    built = run_build_cross_rule_candidate_from_search(
        {
            "search_id": searched["search_id"],
            "rule_id": rule["rule_id"],
            "selection_reason": "明确加入审批策略候选池。",
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = built["artifacts"][0]
    candidate = built["candidate"]

    added = run_add_candidate_to_pool(
        {
            "source_artifact_id": artifact["artifact_id"],
            "expected_artifact_content_hash": artifact["content_hash"],
            "expected_asset_id": candidate["asset_id"],
            "expected_asset_hash": candidate["asset_hash"],
            "strategy_type": "approval",
            "default_action": {
                "type": "approval",
                "value": "approve",
                "reason_code": None,
                "stop": True,
            },
            "action": {
                "type": "reject",
                "value": "reject",
                "reason_code": "CROSS_RULE",
                "stop": True,
            },
            "expected_pool_revision": 0,
            "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    assert added["entry_count"] == 1
    entry = added["entries"][0]
    assert entry["source"]["asset_type"] == "cross_threshold_rule"
    assert entry["rule_id"] == rule["rule_id"]
    compiled = compile_strategy_pool(added["pool"])
    assert compiled["strategy_spec"]["rules"][0]["condition"] == candidate[
        "condition"
    ]
