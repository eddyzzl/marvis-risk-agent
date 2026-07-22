from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.data.workspace import (
    DataSemanticMapping,
    DataWorkspaceDraft,
    data_semantic_mapping_hash,
)
from marvis.db import DatasetRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy import voting_candidate as voting_candidate_domain
from marvis.packs.strategy.automatic_tree_leaf_fragment import (
    AUTOMATIC_TREE_ASSET_ARTIFACT_KIND,
)
from marvis.packs.strategy.automatic_tree_sample_design import (
    sample_design_ref_from_automatic_tree_source_refs,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.dsl import parse_strategy_spec
from marvis.packs.strategy.evaluator import evaluate_strategy_frame
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import (
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
    run_reorder_strategy_pool,
)
from marvis.packs.strategy.voting_candidate import (
    VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1,
    VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
    build_voting_candidate_asset,
)
from marvis.packs.strategy.voting_candidate_tools import (
    TOOL_SCHEMA_VERSION,
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
    VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1,
    VOTING_CANDIDATE_ORIGIN_TOOL,
    build_voting_candidate_artifact_document,
    canonical_voting_candidate_artifact_json,
    canonical_voting_candidate_path,
    load_verified_voting_candidate_artifact,
    run_build_voting_candidate,
    voting_candidate_artifact_provenance,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.settings import build_settings


def _action(action_type: str, reason: str | None = None) -> dict:
    return {
        "type": action_type,
        "value": {
            "approval": "approve",
            "reject": "reject",
            "review": "review",
        }[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _setup(tmp_path: Path) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="voting-candidate",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430],
            "loan_amount": [
                100,
                120,
                140,
                160,
                180,
                200,
                220,
                240,
                260,
                280,
                300,
                320,
            ],
            "overdue_amount": [0, 0, 0, 5, 0, 10, 0, 15, 20, 25, 30, 40],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    source = tmp_path / "voting.parquet"
    frame.to_parquet(source, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    active = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
        ),
        expected_revision=0,
    )
    mapping = DataSemanticMapping(
        target_col="bad",
        field_roles={
            "score": "score",
            "loan_amount": "loan_amount",
            "overdue_amount": "overdue_amount",
            "bad": "target",
        },
    )
    workspace = workspaces.save(
        task.id,
        DataWorkspaceDraft(
            active_dataset_id=dataset.id,
            active_dataset_content_hash=dataset.content_hash,
            semantic_mapping=mapping,
        ),
        expected_revision=active.revision,
    )
    ctx = ToolContext(
        task_id=task.id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    runtime = strategy_tools._runtime(ctx)
    sample = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 90,
            "observation_window_status": "provided",
            "observation_window_start": "2025-01-01",
            "observation_window_end": "2025-12-31",
            "maturity_status": "confirmed_matured",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": False,
        },
        ctx,
    )
    sample_design_ref = {
        "artifact_id": sample["artifact"]["artifact_id"],
        "artifact_content_hash": sample["artifact"]["content_hash"],
        "sample_design_id": sample["sample_design_id"],
        "sample_design_content_hash": sample["content_hash"],
        "partition": "development",
    }
    analyzed = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        ctx,
    )
    report = next(
        artifact
        for artifact in analyzed["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    method = analyzed["candidate_evidence"]["analysis"]["features"][0]["methods"][0]

    def refine(index: int) -> dict:
        return strategy_tools.tool_refine_univariate_candidate(
            {
                "source_artifact_id": report["artifact_id"],
                "expected_artifact_content_hash": report["content_hash"],
                "expected_candidate_id": analyzed["candidate_id"],
                "expected_evidence_hash": analyzed["evidence_hash"],
                "feature": "score",
                "method": "equal_width",
                "merge_groups": [],
                "selection": {"source_bin_ids": [method["bins"][index]["id"]]},
            },
            ctx,
        )

    pool = None
    for index in (0, 2):
        candidate = refine(index)
        artifact = candidate["artifacts"][0]
        expected_revision = 0 if pool is None else pool["revision"]
        expected_hash = (
            ABSENT_POOL_SNAPSHOT_HASH if pool is None else pool["snapshot_hash"]
        )
        added = run_add_candidate_to_pool(
            {
                "source_artifact_id": artifact["artifact_id"],
                "expected_artifact_content_hash": artifact["content_hash"],
                "expected_asset_id": candidate["asset_id"],
                "expected_asset_hash": candidate["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject", f"RISK_{index}"),
                "expected_pool_revision": expected_revision,
                "expected_pool_snapshot_hash": expected_hash,
            },
            ctx,
            runtime,
        )
        pool = added["pool"]
    assert pool is not None
    inputs = {
        "strategy_type": "approval",
        "expected_pool_revision": pool["revision"],
        "expected_pool_snapshot_hash": pool["snapshot_hash"],
        "selected_entry_ids": [
            pool["entries"][1]["entry_id"],
            pool["entries"][0]["entry_id"],
        ],
        "n": 1,
    }
    return {
        "settings": settings,
        "task": task,
        "ctx": ctx,
        "runtime": runtime,
        "dataset": dataset,
        "workspace": workspace,
        "mapping": mapping,
        "pool": pool,
        "inputs": inputs,
        "frame": frame,
        "sample_design_ref": sample_design_ref,
    }


def test_build_voting_candidate_replays_pool_measures_and_persists_exactly(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)

    first = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    repeated = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])

    assert repeated == first
    assert first["schema_version"] == TOOL_SCHEMA_VERSION
    assert first["n"] == 1
    assert first["k"] == 2
    assert [item["pool_position"] for item in first["selected_entries"]] == [0, 1]
    assert len(first["hit_distribution"]) == 3
    assert [row["hit_count"] for row in first["hit_distribution"]] == [0, 1, 2]
    assert sum(row["count"] for row in first["hit_distribution"]) == 12
    assert first["effect"]["population_count"] == 12
    assert first["effect"]["labeled_count"] == 12
    assert first["not_admitted"] is True
    assert first["not_applied"] is True
    assert first["not_adopted"] is True
    assert first["not_deployed"] is True

    [descriptor] = first["artifacts"]
    assert descriptor["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    record = fx["runtime"].task_artifacts.get_for_task(
        fx["task"].id, descriptor["artifact_id"]
    )
    assert record is not None
    assert record["origin_tool"] == VOTING_CANDIDATE_ORIGIN_TOOL
    assert record["provenance"]["schema_version"] == (
        VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION
    )
    assert record["provenance"]["sample_design_ref"] == fx["sample_design_ref"]
    expected_path = canonical_voting_candidate_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
        asset_id=first["asset_id"],
    )
    assert Path(record["path"]) == expected_path
    document = json.loads(expected_path.read_text("utf-8"))
    assert document["asset"]["sample_design_ref"] == fx["sample_design_ref"]
    assert document["asset"]["candidate_evidence"]["sample_design_ref"] == fx[
        "sample_design_ref"
    ]
    assert document["measurement"]["sample_design_ref"] == fx[
        "sample_design_ref"
    ]
    assert expected_path.read_text("utf-8") == canonical_voting_candidate_artifact_json(
        document
    )
    loaded = load_verified_voting_candidate_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_content_hash=descriptor["content_hash"],
        expected_asset_id=first["asset_id"],
        expected_asset_hash=first["asset_hash"],
    )
    assert loaded.asset["asset_id"] == first["asset_id"]
    assert loaded.artifact_binding() == {
        "artifact_id": descriptor["artifact_id"],
        "task_id": fx["task"].id,
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "content_hash": descriptor["content_hash"],
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
        "artifact_schema_version": VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "asset_id": first["asset_id"],
        "asset_hash": first["asset_hash"],
    }
    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": descriptor["artifact_id"],
            "expected_artifact_content_hash": descriptor["content_hash"],
            "expected_asset_id": first["asset_id"],
            "expected_asset_hash": first["asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("review", "VOTING_REVIEW"),
            "expected_pool_revision": fx["pool"]["revision"],
            "expected_pool_snapshot_hash": fx["pool"]["snapshot_hash"],
            "placement_mode": "before_selected_members",
        },
        fx["ctx"],
        fx["runtime"],
    )
    assert admitted["revision"] == fx["pool"]["revision"] + 1
    assert admitted["operation"] == "insert_candidate_before_entries"
    assert admitted["entries"][0]["source"]["asset_type"] == "voting_n_of_k"
    voting_rule_id = admitted["entries"][0]["rule_id"]
    compiled = run_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": admitted["revision"],
            "expected_pool_snapshot_hash": admitted["snapshot_hash"],
        },
        fx["ctx"],
        fx["runtime"],
    )
    voting_rule = next(
        rule
        for rule in compiled["strategy_spec"]["rules"]
        if rule["rule_id"] == voting_rule_id
    )
    assert voting_rule["condition"]["op"] == "n_of_k"
    assert voting_rule["condition"]["n"] == 1
    evaluated = evaluate_strategy_frame(
        fx["frame"],
        parse_strategy_spec(compiled["strategy_spec"]),
    )
    assert int((evaluated.matched_rule_id == voting_rule_id).sum()) == first[
        "effect"
    ]["matched_count"]

    invalid_order = [
        admitted["entries"][1]["rule_id"],
        voting_rule_id,
        admitted["entries"][2]["rule_id"],
    ]
    with pytest.raises(StrategyError, match="Voting candidate is unreachable"):
        run_reorder_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": admitted["revision"],
                "expected_pool_snapshot_hash": admitted["snapshot_hash"],
                "ordered_rule_ids": invalid_order,
            },
            fx["ctx"],
            fx["runtime"],
        )


def test_voting_admission_requires_placement_and_can_replace_members_atomically(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    built = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    [descriptor] = built["artifacts"]
    request = {
        "source_artifact_id": descriptor["artifact_id"],
        "expected_artifact_content_hash": descriptor["content_hash"],
        "expected_asset_id": built["asset_id"],
        "expected_asset_hash": built["asset_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": _action("review", "VOTING_REVIEW"),
        "expected_pool_revision": fx["pool"]["revision"],
        "expected_pool_snapshot_hash": fx["pool"]["snapshot_hash"],
    }

    with pytest.raises(StrategyError, match="requires explicit"):
        run_add_candidate_to_pool(request, fx["ctx"], fx["runtime"])
    assert StrategyCandidatePoolRepository(
        fx["settings"].db_path
    ).get_current(fx["task"].id, "approval")["revision"] == fx["pool"]["revision"]

    replaced = run_add_candidate_to_pool(
        {**request, "placement_mode": "replace_selected_members"},
        fx["ctx"],
        fx["runtime"],
    )
    assert replaced["operation"] == "replace_entries_with_candidate"
    assert replaced["revision"] == fx["pool"]["revision"] + 1
    assert len(replaced["entries"]) == 1
    assert replaced["entries"][0]["source"]["asset_type"] == "voting_n_of_k"


def test_zero_hit_voting_candidate_cannot_enter_pool(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    built = run_build_voting_candidate(
        {**fx["inputs"], "n": 2},
        fx["ctx"],
        fx["runtime"],
    )
    assert built["effect"]["matched_count"] == 0
    [descriptor] = built["artifacts"]

    with pytest.raises(StrategyError, match="no standalone sample hits"):
        run_add_candidate_to_pool(
            {
                "source_artifact_id": descriptor["artifact_id"],
                "expected_artifact_content_hash": descriptor["content_hash"],
                "expected_asset_id": built["asset_id"],
                "expected_asset_hash": built["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("review", "ZERO_HIT_VOTING"),
                "expected_pool_revision": fx["pool"]["revision"],
                "expected_pool_snapshot_hash": fx["pool"]["snapshot_hash"],
                "placement_mode": "before_selected_members",
            },
            fx["ctx"],
            fx["runtime"],
        )
    assert StrategyCandidatePoolRepository(
        fx["settings"].db_path
    ).get_current(fx["task"].id, "approval") == fx["pool"]


def test_rebuilt_equivalent_voting_condition_cannot_enter_pool_twice(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    first = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    [first_artifact] = first["artifacts"]
    admitted = run_add_candidate_to_pool(
        {
            "source_artifact_id": first_artifact["artifact_id"],
            "expected_artifact_content_hash": first_artifact["content_hash"],
            "expected_asset_id": first["asset_id"],
            "expected_asset_hash": first["asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("review", "FIRST_VOTING"),
            "expected_pool_revision": fx["pool"]["revision"],
            "expected_pool_snapshot_hash": fx["pool"]["snapshot_hash"],
            "placement_mode": "before_selected_members",
        },
        fx["ctx"],
        fx["runtime"],
    )
    member_entry_ids = [
        entry["entry_id"]
        for entry in admitted["entries"]
        if entry["source"]["asset_type"] != "voting_n_of_k"
    ]
    rebuilt = run_build_voting_candidate(
        {
            "strategy_type": "approval",
            "expected_pool_revision": admitted["revision"],
            "expected_pool_snapshot_hash": admitted["snapshot_hash"],
            "selected_entry_ids": member_entry_ids,
            "n": 1,
        },
        fx["ctx"],
        fx["runtime"],
    )
    [rebuilt_artifact] = rebuilt["artifacts"]
    with pytest.raises(StrategyError, match="duplicate executable condition"):
        run_add_candidate_to_pool(
            {
                "source_artifact_id": rebuilt_artifact["artifact_id"],
                "expected_artifact_content_hash": rebuilt_artifact["content_hash"],
                "expected_asset_id": rebuilt["asset_id"],
                "expected_asset_hash": rebuilt["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject", "SECOND_VOTING"),
                "expected_pool_revision": admitted["revision"],
                "expected_pool_snapshot_hash": admitted["snapshot_hash"],
                "placement_mode": "before_selected_members",
            },
            fx["ctx"],
            fx["runtime"],
        )

    stricter = run_build_voting_candidate(
        {
            "strategy_type": "approval",
            "expected_pool_revision": admitted["revision"],
            "expected_pool_snapshot_hash": admitted["snapshot_hash"],
            "selected_entry_ids": member_entry_ids,
            "n": 2,
        },
        fx["ctx"],
        fx["runtime"],
    )
    [stricter_artifact] = stricter["artifacts"]
    with pytest.raises(StrategyError, match="logically dominates"):
        run_add_candidate_to_pool(
            {
                "source_artifact_id": stricter_artifact["artifact_id"],
                "expected_artifact_content_hash": stricter_artifact["content_hash"],
                "expected_asset_id": stricter["asset_id"],
                "expected_asset_hash": stricter["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject", "STRICTER_VOTING"),
                "expected_pool_revision": admitted["revision"],
                "expected_pool_snapshot_hash": admitted["snapshot_hash"],
                "placement_mode": "before_selected_members",
            },
            fx["ctx"],
            fx["runtime"],
        )


def test_voting_build_reuses_one_dataset_binding_for_shared_member_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original = pool_tools._load_dataset_binding
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pool_tools, "_load_dataset_binding", counted)

    run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])

    assert calls == 1


def test_voting_artifact_cross_checks_distribution_metrics_and_drop_contract(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    built = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    [descriptor] = built["artifacts"]
    loaded = load_verified_voting_candidate_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=descriptor["artifact_id"],
        expected_content_hash=descriptor["content_hash"],
        expected_asset_id=built["asset_id"],
        expected_asset_hash=built["asset_hash"],
    )
    measurement = loaded.document["measurement"]

    forged_distribution = deepcopy(measurement["hit_distribution"])
    assert forged_distribution[0]["count"] > forged_distribution[0]["bad_count"]
    forged_distribution[0]["count"] -= 1
    forged_distribution[1]["count"] += 1
    total_bad = sum(row["bad_count"] for row in forged_distribution)
    base_bad_rate = total_bad / measurement["labeled_count"]
    for row in forged_distribution:
        row["share"] = row["count"] / measurement["labeled_count"]
        row["bad_rate"] = (
            None if row["count"] == 0 else row["bad_count"] / row["count"]
        )
        row["lift"] = (
            None if row["bad_rate"] is None or base_bad_rate == 0 else row["bad_rate"] / base_bad_rate
        )
    with pytest.raises(StrategyError, match="threshold aggregation"):
        build_voting_candidate_artifact_document(
            loaded.asset,
            target_col=measurement["target_col"],
            drop_nan_labels=measurement["drop_nan_labels"],
            nan_labels_dropped=measurement["nan_labels_dropped"],
            population_count=measurement["population_count"],
            labeled_count=measurement["labeled_count"],
            hit_distribution=forged_distribution,
            metric_observations=measurement["metric_observations"],
        )

    forged_observations = deepcopy(measurement["metric_observations"])
    hit_share = next(
        item
        for item in forged_observations
        if item["metric_name"] == "voting.hit_share"
        and item["dimension"] == "count"
    )
    hit_share["value"] = 0.123456
    with pytest.raises(StrategyError, match="deterministic measurement"):
        build_voting_candidate_artifact_document(
            loaded.asset,
            target_col=measurement["target_col"],
            drop_nan_labels=measurement["drop_nan_labels"],
            nan_labels_dropped=measurement["nan_labels_dropped"],
            population_count=measurement["population_count"],
            labeled_count=measurement["labeled_count"],
            hit_distribution=measurement["hit_distribution"],
            metric_observations=forged_observations,
        )

    missing_observation = measurement["metric_observations"][:-1]
    with pytest.raises(StrategyError, match="identities are incomplete"):
        build_voting_candidate_artifact_document(
            loaded.asset,
            target_col=measurement["target_col"],
            drop_nan_labels=measurement["drop_nan_labels"],
            nan_labels_dropped=measurement["nan_labels_dropped"],
            population_count=measurement["population_count"],
            labeled_count=measurement["labeled_count"],
            hit_distribution=measurement["hit_distribution"],
            metric_observations=missing_observation,
        )

    effect_body = {
        field: loaded.asset["effect"][field]
        for field in (
            "population_count",
            "labeled_count",
            "matched_count",
            "matched_rate",
            "matched_bad_count",
            "matched_bad_rate",
            "unmatched_count",
            "unmatched_bad_count",
            "unmatched_bad_rate",
            "bad_capture_rate",
            "lift",
        )
    }
    effect_body["population_count"] += 1
    population_asset = build_voting_candidate_asset(
        fx["pool"],
        selected_entry_ids=fx["inputs"]["selected_entry_ids"],
        n=fx["inputs"]["n"],
        target_col="bad",
        sample_design_ref=fx["sample_design_ref"],
        effect=effect_body,
    )
    with pytest.raises(StrategyError, match="cannot drop labels"):
        build_voting_candidate_artifact_document(
            population_asset,
            target_col="bad",
            drop_nan_labels=False,
            nan_labels_dropped=1,
            population_count=measurement["population_count"] + 1,
            labeled_count=measurement["labeled_count"],
            hit_distribution=measurement["hit_distribution"],
            metric_observations=measurement["metric_observations"],
        )


def test_voting_stage_write_failure_cleans_partial_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _setup(tmp_path)
    original_write_bytes = Path.write_bytes

    def fail_voting_stage(path: Path, data: bytes) -> int:
        if ".staging" in path.parts and "strategy_voting_candidates" in path.parts:
            original_write_bytes(path, b"partial")
            raise OSError("disk full")
        return original_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_voting_stage)

    with pytest.raises(StrategyError, match="could not be staged"):
        run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])

    output_dir = (
        fx["settings"].tasks_dir / fx["task"].id / "strategy_voting_candidates"
    )
    assert not list(output_dir.rglob("*.json"))
    assert not list((output_dir / ".staging").glob("*"))
    assert not any(
        artifact["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
        for artifact in fx["runtime"].task_artifacts.list_for_task(fx["task"].id)
    )


def test_build_voting_candidate_rejects_injected_results_stale_cas_and_tamper(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    with pytest.raises(StrategyError, match="unsupported: action"):
        run_build_voting_candidate(
            {**fx["inputs"], "action": _action("reject")},
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(StrategyError, match="between 2 and 50"):
        run_build_voting_candidate(
            {**fx["inputs"], "selected_entry_ids": fx["inputs"]["selected_entry_ids"][:1]},
            fx["ctx"],
            fx["runtime"],
        )
    with pytest.raises(StrategyError, match="stale"):
        run_build_voting_candidate(
            {**fx["inputs"], "expected_pool_snapshot_hash": "f" * 64},
            fx["ctx"],
            fx["runtime"],
        )

    built = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    [artifact] = built["artifacts"]
    record = fx["runtime"].task_artifacts.get_for_task(
        fx["task"].id, artifact["artifact_id"]
    )
    assert record is not None
    canonical_bytes = Path(record["path"]).read_bytes()
    Path(record["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(StrategyError, match="content hash changed"):
        run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    with pytest.raises(StrategyError, match="content hash changed"):
        load_verified_voting_candidate_artifact(
            fx["runtime"],
            task_id=fx["task"].id,
            artifact_id=artifact["artifact_id"],
            expected_content_hash=artifact["content_hash"],
            expected_asset_id=built["asset_id"],
            expected_asset_hash=built["asset_hash"],
        )

    Path(record["path"]).write_bytes(canonical_bytes)
    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM task_artifacts WHERE id = ?", (artifact["artifact_id"],))
    with pytest.raises(StrategyError, match="exists without a registry row"):
        run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])

    current = StrategyCandidatePoolRepository(fx["settings"].db_path).get_current(
        fx["task"].id, "approval"
    )
    assert current == fx["pool"]


def test_mixed_univariate_and_automatic_tree_voting_propagates_sample_design_to_impact(
    tmp_path: Path,
) -> None:
    fx = _setup(tmp_path)
    tree = strategy_tools.tool_build_automatic_tree_candidate(
        {
            "dataset_id": fx["dataset"].id,
            "expected_content_hash": fx["dataset"].content_hash,
            "workspace_revision": fx["workspace"].revision,
            "analysis_generation": fx["workspace"].analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(fx["mapping"]),
            "target_col": "bad",
            "sample_design_ref": fx["sample_design_ref"],
            "features": ["score"],
            "directions": {"score": "increasing"},
            "max_depth": 2,
            "min_leaf_count": 2,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "budgets": {
                "max_rows": 100,
                "max_features": 5,
                "max_cells": 500,
                "max_nodes": 31,
                "max_cutpoint_evaluations": 1000,
            },
        },
        fx["ctx"],
    )
    source = next(
        item
        for item in tree["artifacts"]
        if item["kind"] == AUTOMATIC_TREE_ASSET_ARTIFACT_KIND
    )
    selected_leaf = tree["leaf_index"][0]
    leaf = strategy_tools.tool_materialize_automatic_tree_leaf_fragment(
        {
            "source_artifact_id": source["artifact_id"],
            "expected_artifact_content_hash": source["content_hash"],
            "expected_asset_id": tree["summary"]["asset_id"],
            "expected_asset_hash": tree["summary"]["asset_hash"],
            "expected_tree_result_hash": tree["summary"]["tree_result_hash"],
            "leaf_id": selected_leaf["leaf_id"],
        },
        fx["ctx"],
    )
    leaf_artifact = leaf["artifacts"][0]
    extended = run_add_candidate_to_pool(
        {
            "source_artifact_id": leaf_artifact["artifact_id"],
            "expected_artifact_content_hash": leaf_artifact["content_hash"],
            "expected_asset_id": leaf["tree_asset_id"],
            "expected_asset_hash": leaf["tree_asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("reject", "TREE_RISK"),
            "expected_pool_revision": fx["pool"]["revision"],
            "expected_pool_snapshot_hash": fx["pool"]["snapshot_hash"],
        },
        fx["ctx"],
        fx["runtime"],
    )
    selected_ids = [
        extended["entries"][0]["entry_id"],
        extended["entries"][-1]["entry_id"],
    ]
    voting = run_build_voting_candidate(
        {
            "strategy_type": "approval",
            "expected_pool_revision": extended["revision"],
            "expected_pool_snapshot_hash": extended["snapshot_hash"],
            "selected_entry_ids": selected_ids,
            "n": 1,
        },
        fx["ctx"],
        fx["runtime"],
    )

    assert voting["sample_design_ref"] == fx["sample_design_ref"]
    [descriptor] = voting["artifacts"]
    replaced = run_add_candidate_to_pool(
        {
            "source_artifact_id": descriptor["artifact_id"],
            "expected_artifact_content_hash": descriptor["content_hash"],
            "expected_asset_id": voting["asset_id"],
            "expected_asset_hash": voting["asset_hash"],
            "strategy_type": "approval",
            "default_action": _action("approval"),
            "action": _action("review", "MIXED_VOTING"),
            "expected_pool_revision": extended["revision"],
            "expected_pool_snapshot_hash": extended["snapshot_hash"],
            "placement_mode": "replace_selected_members",
        },
        fx["ctx"],
        fx["runtime"],
    )
    identity = extended["entries"][0]["source"]["evidence_identity"]
    impact = strategy_tools.tool_measure_pool_impact(
        {
            "strategy_type": "approval",
            "expected_pool_revision": replaced["revision"],
            "expected_pool_snapshot_hash": replaced["snapshot_hash"],
            "dataset_id": identity["dataset_id"],
            "expected_dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "target_col": "bad",
            "sample_design_ref": fx["sample_design_ref"],
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "comparison_mode": "absolute",
            "drop_nan_labels": False,
        },
        fx["ctx"],
    )
    assert impact["assessment"]["bindings"]["sample_design_ref"] == fx[
        "sample_design_ref"
    ]


@pytest.mark.parametrize(
    "source_refs",
    [
        [],
        ["strategy-sample-design:not-json"],
        [
            "strategy-sample-design:"
            + json.dumps(
                {
                    "kind": "strategy_sample_design",
                    "artifact_id": "1" * 64,
                    "artifact_content_hash": "2" * 64,
                    "sample_design_id": "sample-design-duplicate",
                    "sample_design_content_hash": "3" * 64,
                    "partition": "development",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ]
        * 2,
    ],
)
def test_voting_rejects_missing_malformed_or_duplicate_tree_sample_refs(
    source_refs: list[str],
) -> None:
    with pytest.raises(StrategyError, match="sample-design"):
        sample_design_ref_from_automatic_tree_source_refs(source_refs)


def test_legacy_v1_voting_artifact_remains_strictly_loadable(tmp_path: Path) -> None:
    fx = _setup(tmp_path)
    current = run_build_voting_candidate(fx["inputs"], fx["ctx"], fx["runtime"])
    effect_body = {
        key: current["effect"][key]
        for key in (
            "population_count",
            "labeled_count",
            "matched_count",
            "matched_rate",
            "matched_bad_count",
            "matched_bad_rate",
            "unmatched_count",
            "unmatched_bad_count",
            "unmatched_bad_rate",
            "bad_capture_rate",
            "lift",
        )
    }
    legacy_asset = voting_candidate_domain._build_voting_candidate_asset(
        fx["pool"],
        selected_entry_ids=fx["inputs"]["selected_entry_ids"],
        n=fx["inputs"]["n"],
        target_col="bad",
        sample_design_ref=None,
        effect=effect_body,
        producer_version=VOTING_CANDIDATE_ASSET_PRODUCER_VERSION_V1,
        schema_version=VOTING_CANDIDATE_ASSET_SCHEMA_VERSION_V1,
    )
    document = build_voting_candidate_artifact_document(
        legacy_asset,
        target_col="bad",
        drop_nan_labels=False,
        nan_labels_dropped=0,
        population_count=current["population_count"],
        labeled_count=current["labeled_count"],
        hit_distribution=current["hit_distribution"],
        metric_observations=current["metric_observations"],
    )
    assert document["schema_version"] == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1
    assert "sample_design_ref" not in document["asset"]
    assert "sample_design_ref" not in document["measurement"]

    canonical = canonical_voting_candidate_artifact_json(document).encode("utf-8")
    path = canonical_voting_candidate_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
        asset_id=legacy_asset["asset_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical)
    pool_artifact = pool_tools._load_pool_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        snapshot=fx["pool"],
    )
    provenance = voting_candidate_artifact_provenance(
        document,
        task_id=fx["task"].id,
        pool_artifact=pool_artifact,
    )
    assert provenance["schema_version"] == VOTING_CANDIDATE_ARTIFACT_SCHEMA_VERSION_V1
    assert "sample_design_ref" not in provenance
    record = fx["runtime"].task_artifacts.register(
        task_id=fx["task"].id,
        kind=VOTING_CANDIDATE_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(canonical).hexdigest(),
        origin_tool=VOTING_CANDIDATE_ORIGIN_TOOL,
        provenance=provenance,
    )
    loaded = load_verified_voting_candidate_artifact(
        fx["runtime"],
        task_id=fx["task"].id,
        artifact_id=record["id"],
        expected_content_hash=record["content_hash"],
        expected_asset_id=legacy_asset["asset_id"],
        expected_asset_hash=legacy_asset["asset_hash"],
    )
    assert loaded.asset == legacy_asset
