from __future__ import annotations

import json
from pathlib import Path

import pytest

import marvis.packs.strategy.cross_candidate_search_tools as search_tools
from marvis.packs.strategy.cross_candidate_search_tools import (
    CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
    load_cross_candidate_search_artifact,
    run_build_cross_matrix_candidate_from_search,
    run_search_cross_matrix_candidates,
)
from marvis.packs.strategy.cross_matrix_candidate_tools import (
    ASSET_ARTIFACT_KIND,
    run_build_cross_matrix_candidate,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.tools import (
    tool_build_cross_matrix_candidate_from_search,
    tool_search_cross_matrix_candidates,
)
from marvis.packs.strategy import tools as strategy_tools
from marvis.plugins.loader import load_manifest
from marvis.plugins.manifest import ToolRef
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_cross_matrix_candidate_tool import (
    _replace_source_with_native_parallel_evidence,
    _setup,
)


def _search_inputs(fixture: dict) -> dict:
    source = fixture["source"]
    source_artifact = next(
        item
        for item in source["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    return {
        "source_artifact_id": source_artifact["artifact_id"],
        "expected_artifact_content_hash": source_artifact["content_hash"],
        "expected_candidate_id": source["candidate_id"],
        "expected_evidence_hash": source["evidence_hash"],
        "features": ["score", "age"],
        "max_pairs": 10,
    }


def test_search_persists_only_aggregate_risk_development_and_from_search_replays_exact_asset(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    search_invocation = fixture["runner"].invoke(
        ToolRef("strategy", "search_cross_matrix_candidates"),
        _search_inputs(fixture),
        task_id=fixture["task"].id,
    )
    assert search_invocation.ok, search_invocation.error
    searched = search_invocation.output

    assert searched["search_space"] == 1
    assert searched["evaluated"] == 1
    assert searched["truncated"] is False
    assert searched["not_selected"] is True
    assert searched["not_admitted"] is True
    assert searched["not_applied"] is True
    assert searched["not_adopted"] is True
    assert searched["not_deployed"] is True
    result = searched["search_result"]
    assert result["population"]["row_count"] == 8
    [pair] = result["pairs"]
    assert (pair["x_feature"], pair["y_feature"]) == ("age", "score")
    assert pair["x_method"] == "equal_width"
    assert pair["y_method"] == "equal_width"

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    record = repository.get_for_task(
        fixture["task"].id,
        searched["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["kind"] == CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND
    persisted = json.loads(Path(record["path"]).read_text("utf-8"))
    assert persisted == result
    raw = Path(record["path"]).read_text("utf-8")
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
    assert record["provenance"]["sample_partition"] == "risk/development"

    build_invocation = fixture["runner"].invoke(
        ToolRef("strategy", "build_cross_matrix_candidate_from_search"),
        {
            "search_id": searched["search_id"],
            "pair_id": pair["pair_id"],
        },
        task_id=fixture["task"].id,
    )
    assert build_invocation.ok, build_invocation.error
    built = build_invocation.output
    direct = run_build_cross_matrix_candidate(
        fixture["inputs"],
        fixture["ctx"],
        fixture["runtime"],
    )

    assert built["source_search_selection"]["search_id"] == searched["search_id"]
    assert built["source_search_selection"]["pair_id"] == pair["pair_id"]
    assert built["cross_matrix_candidate"] == direct
    assert built["cross_matrix_candidate"]["cross_matrix_candidate"][
        "asset_hash"
    ] == pair["asset_fingerprint"]["asset_hash"]


def test_search_loader_optionally_freezes_exact_domain_search_identity(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )
    artifact = searched["artifacts"][0]

    binding = load_cross_candidate_search_artifact(
        fixture["runtime"],
        task_id=fixture["task"].id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_search_id=searched["search_id"],
        expected_search_content_hash=searched["content_hash"],
    )

    assert binding.result["search_id"] == searched["search_id"]
    with pytest.raises(StrategyError, match="search identity"):
        load_cross_candidate_search_artifact(
            fixture["runtime"],
            task_id=fixture["task"].id,
            artifact_id=artifact["artifact_id"],
            expected_artifact_content_hash=artifact["content_hash"],
            expected_search_id="cross-search-" + ("f" * 32),
            expected_search_content_hash=searched["content_hash"],
        )


def test_from_search_writer_lock_rejects_deleted_search_and_rolls_everything_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )
    [pair] = searched["search_result"]["pairs"]
    original_require = (
        search_tools.require_cross_candidate_search_artifact_binding_on_connection
    )

    def delete_then_require(conn, binding):
        conn.execute(
            "DELETE FROM task_artifacts WHERE task_id = ? AND id = ?",
            (binding.task_id, binding.artifact_id),
        )
        return original_require(conn, binding)

    monkeypatch.setattr(
        search_tools,
        "require_cross_candidate_search_artifact_binding_on_connection",
        delete_then_require,
    )

    with pytest.raises(StrategyError, match="search artifact"):
        run_build_cross_matrix_candidate_from_search(
            {
                "search_id": searched["search_id"],
                "pair_id": pair["pair_id"],
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    repository = TaskArtifactRepository(fixture["settings"].db_path)
    assert (
        repository.get_for_task(
            fixture["task"].id,
            searched["artifacts"][0]["artifact_id"],
        )
        is not None
    )
    assert all(
        item["kind"] != ASSET_ARTIFACT_KIND
        for item in repository.list_for_task(fixture["task"].id)
    )


def test_search_uses_only_native_v2_risk_development_membership(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, with_split=True)
    native_ref = _replace_source_with_native_parallel_evidence(fixture)

    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert searched["search_result"]["population"]["row_count"] == 6
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        searched["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    assert record["provenance"]["sample_design_ref"] == native_ref
    assert record["provenance"]["sample_partition"] == "risk/development"


def test_search_rejects_nonexplicit_duplicate_or_unknown_features(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    inputs = _search_inputs(fixture)
    inputs["features"] = ["age", "age"]
    with pytest.raises(StrategyError, match="unique"):
        run_search_cross_matrix_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )

    inputs["features"] = ["age", "missing_feature"]
    with pytest.raises(StrategyError, match="ranking"):
        run_search_cross_matrix_candidates(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )


def test_search_uses_each_explicit_features_highest_parent_ranked_available_method(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    source = strategy_tools.tool_analyze_univariate_candidates(
        {
            **fixture["source_inputs"],
            "methods": ["equal_width", "equal_frequency"],
        },
        fixture["ctx"],
    )
    fixture["source"] = source
    expected = {}
    for row in source["rankings"]:
        expected.setdefault(row["feature"], row["method"])

    searched = run_search_cross_matrix_candidates(
        _search_inputs(fixture),
        fixture["ctx"],
        fixture["runtime"],
    )

    assert {
        item["feature"]: item["method"]
        for item in searched["search_result"]["configuration"]["features"]
    } == {
        "age": expected["age"],
        "score": expected["score"],
    }


def test_cross_search_tools_are_registered_with_bounded_public_schemas() -> None:
    manifest = load_manifest(
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy",
        builtin=True,
    )
    tools = {item.name: item for item in manifest.tools}

    search = tools["search_cross_matrix_candidates"]
    assert search.entrypoint == "tool_search_cross_matrix_candidates"
    assert search.input_schema["properties"]["features"]["minItems"] == 2
    assert search.input_schema["properties"]["features"]["maxItems"] == 20
    assert search.input_schema["properties"]["max_pairs"]["maximum"] == 190
    assert search.side_effects == ("read:task", "read:dataset", "write:artifact")

    selection = tools["build_cross_matrix_candidate_from_search"]
    assert selection.entrypoint == "tool_build_cross_matrix_candidate_from_search"
    assert selection.input_schema["required"] == ["search_id", "pair_id"]
    assert selection.side_effects == (
        "read:task",
        "read:dataset",
        "write:artifact",
    )
    assert callable(tool_search_cross_matrix_candidates)
    assert callable(tool_build_cross_matrix_candidate_from_search)
