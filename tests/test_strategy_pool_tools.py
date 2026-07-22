from __future__ import annotations

import json
import hashlib
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
from marvis.db_schema import connect
from marvis.domain import TaskCreate
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
import marvis.repositories.strategy_pool as strategy_pool_repository
from marvis.packs.strategy.errors import (
    StrategyError,
    StrategyPoolLegacyDraftNeedsRebuildError,
)
from marvis.packs.strategy.candidate_fragment import (
    build_verified_candidate_fragment,
    sample_context_hash_from_candidate_evidence,
)
from marvis.packs.strategy.pool import (
    add_verified_candidate_fragment,
    canonical_strategy_pool_json,
)
import marvis.packs.strategy.pool_tools as pool_tools_module
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import (
    POOL_ARTIFACT_KIND,
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
    run_remove_pool_entry,
    run_reorder_strategy_pool,
    run_set_pool_entry_action,
)
from marvis.packs.strategy.voting_candidate_tools import run_build_voting_candidate
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.strategy_pool import (
    StrategyCandidatePoolRepository,
    strategy_pool_id,
    strategy_pool_artifact_content_hash,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {
        "approval": "approve",
        "reject": "reject",
        "review": "review",
    }
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _context(settings, task_id: str) -> ToolContext:
    return ToolContext(
        task_id=task_id,
        seed=0,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )


def _setup(tmp_path: Path) -> dict:
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    tasks = TaskRepository(settings.db_path)
    task = tasks.create_task(
        TaskCreate(
            model_name="candidate-pool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            task_type="strategy",
            target_col="bad",
        )
    )
    other_task = tasks.create_task(
        TaskCreate(
            model_name="foreign-pool",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "foreign"),
            task_type="strategy",
            target_col="bad",
        )
    )
    frame = pd.DataFrame(
        {
            "score": [100, 130, 160, 190, 220, 250, 280, 310, 340, 370, 400, 430],
            "loan_amount": [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320],
            "overdue_amount": [0, 0, 0, 5, 0, 10, 0, 15, 20, 25, 30, 40],
            "bad": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    source_path = tmp_path / "candidate.parquet"
    frame.to_parquet(source_path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(source_path, task_id=task.id, role="derived")
    workspaces = DataWorkspaceRepository(settings.db_path)
    activated = workspaces.save(
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
        expected_revision=activated.revision,
    )
    ctx = _context(settings, task.id)
    runtime = strategy_tools._runtime(ctx)
    sample_design = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": dataset.id,
            "expected_dataset_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-01-31",
            "maturity_status": "confirmed_matured",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": False,
        },
        ctx,
    )
    sample_design_ref = {
        "artifact_id": sample_design["artifact"]["artifact_id"],
        "artifact_content_hash": sample_design["artifact"]["content_hash"],
        "sample_design_id": sample_design["sample_design_id"],
        "sample_design_content_hash": sample_design["content_hash"],
        "partition": "development",
    }
    source_output = strategy_tools.tool_analyze_univariate_candidates(
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
    candidate_report = next(
        item
        for item in source_output["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    method = source_output["candidate_evidence"]["analysis"]["features"][0][
        "methods"
    ][0]

    def refine(bin_index: int) -> dict:
        return strategy_tools.tool_refine_univariate_candidate(
            {
                "source_artifact_id": candidate_report["artifact_id"],
                "expected_artifact_content_hash": candidate_report["content_hash"],
                "expected_candidate_id": source_output["candidate_id"],
                "expected_evidence_hash": source_output["evidence_hash"],
                "feature": "score",
                "method": "equal_width",
                "merge_groups": [],
                "selection": {
                    "source_bin_ids": [method["bins"][bin_index]["id"]]
                },
            },
            ctx,
        )

    return {
        "settings": settings,
        "task": task,
        "other_task": other_task,
        "ctx": ctx,
        "runtime": runtime,
        "dataset": dataset,
        "source_output": source_output,
        "first": refine(0),
        "refine": refine,
    }


def _add_inputs(
    candidate: dict,
    *,
    expected_revision: int,
    expected_hash: str,
    action: dict | None = None,
) -> dict:
    artifact = candidate["artifacts"][0]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": candidate["asset_id"],
        "expected_asset_hash": candidate["asset_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": action or _action("reject", reason="RISK"),
        "expected_pool_revision": expected_revision,
        "expected_pool_snapshot_hash": expected_hash,
    }


def _refine_bins(fixture: dict, indices: list[int]) -> dict:
    source_output = fixture["source_output"]
    candidate_report = next(
        item
        for item in source_output["artifacts"]
        if item["kind"] == "strategy_candidate_json"
    )
    method = source_output["candidate_evidence"]["analysis"]["features"][0][
        "methods"
    ][0]
    return strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": candidate_report["artifact_id"],
            "expected_artifact_content_hash": candidate_report["content_hash"],
            "expected_candidate_id": source_output["candidate_id"],
            "expected_evidence_hash": source_output["evidence_hash"],
            "feature": "score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {
                "source_bin_ids": [method["bins"][index]["id"] for index in indices]
            },
        },
        fixture["ctx"],
    )


def _insert_archived_legacy_draft(fixture: dict) -> dict:
    task_id = fixture["task"].id
    pool_id = strategy_pool_id(task_id, "approval")
    revision_id = "legacy-pool-revision-1"
    snapshot_hash = "d" * 64
    operation_hash = "e" * 64
    old_absent_hash = (
        "9024538661b531de814a43e87e932bf39b4b87522525f7a7afea1bf5bf8968ee"
    )
    legacy_bytes = b'{"schema_version":"strategy.candidate-pool.v1"}'
    path = (
        Path(fixture["settings"].tasks_dir)
        / task_id
        / "strategy_candidate_pools"
        / "legacy-v1.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(legacy_bytes)
    artifact = TaskArtifactRepository(fixture["settings"].db_path).register(
        task_id=task_id,
        kind=POOL_ARTIFACT_KIND,
        path=str(path),
        content_hash=hashlib.sha256(legacy_bytes).hexdigest(),
        origin_tool="strategy.add_candidate_to_pool",
        provenance={"schema_version": "strategy.candidate-pool-artifact.v1"},
    )
    timestamp = "2026-07-19T00:00:00+00:00"
    with connect(fixture["settings"].db_path) as conn:
        conn.execute("PRAGMA defer_foreign_keys = ON")
        conn.execute(
            """
            INSERT INTO strategy_candidate_pools_v1_archive(
                id, schema_version, task_id, strategy_type, current_revision,
                current_revision_id, current_snapshot_hash, created_at, updated_at
            ) VALUES (?, 'strategy.candidate-pool-head.v1', ?, 'approval',
                      1, ?, ?, ?, ?)
            """,
            (pool_id, task_id, revision_id, snapshot_hash, timestamp, timestamp),
        )
        conn.execute(
            """
            INSERT INTO strategy_candidate_pool_revisions_v1_archive(
                id, schema_version, pool_id, task_id, strategy_type, revision,
                parent_revision_id, parent_snapshot_hash, operation_kind,
                operation_hash, operation_reason, default_action_json, status,
                validation_status, snapshot_json, snapshot_hash, artifact_id,
                artifact_content_hash, created_at
            ) VALUES (?, 'strategy.candidate-pool.v1', ?, ?, 'approval', 1,
                      NULL, ?, 'add_candidate', ?, 'legacy draft', '{}', 'draft',
                      'unvalidated', '{}', ?, ?, ?, ?)
            """,
            (
                revision_id,
                pool_id,
                task_id,
                old_absent_hash,
                operation_hash,
                snapshot_hash,
                artifact["id"],
                artifact["content_hash"],
                timestamp,
            ),
        )
    archive = StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_archived_legacy_draft(task_id, "approval")
    assert archive is not None
    return archive


def _refine_for_workspace(fixture: dict, workspace, bin_index: int) -> dict:
    sample_design = strategy_tools.tool_materialize_sample_design(
        {
            "dataset_id": fixture["dataset"].id,
            "expected_dataset_content_hash": fixture["dataset"].content_hash,
            "workspace_revision": workspace.revision,
            "workspace_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "target_col": "bad",
            "target_bad_value": 1,
            "performance_window_status": "provided",
            "performance_window_days": 30,
            "observation_window_status": "provided",
            "observation_window_start": "2026-01-01",
            "observation_window_end": "2026-01-31",
            "maturity_status": "confirmed_matured",
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "drop_nan_labels": False,
        },
        fixture["ctx"],
    )
    sample_design_ref = {
        "artifact_id": sample_design["artifact"]["artifact_id"],
        "artifact_content_hash": sample_design["artifact"]["content_hash"],
        "sample_design_id": sample_design["sample_design_id"],
        "sample_design_content_hash": sample_design["content_hash"],
        "partition": "development",
    }
    source_output = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": fixture["dataset"].id,
            "expected_content_hash": fixture["dataset"].content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(
                workspace.semantic_mapping
            ),
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "features": ["score"],
            "methods": ["equal_width"],
            "bin_count": 3,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
        },
        fixture["ctx"],
    )
    report_artifact = next(
        artifact
        for artifact in source_output["artifacts"]
        if artifact["kind"] == "strategy_candidate_json"
    )
    method = source_output["candidate_evidence"]["analysis"]["features"][0][
        "methods"
    ][0]
    return strategy_tools.tool_refine_univariate_candidate(
        {
            "source_artifact_id": report_artifact["artifact_id"],
            "expected_artifact_content_hash": report_artifact["content_hash"],
            "expected_candidate_id": source_output["candidate_id"],
            "expected_evidence_hash": source_output["evidence_hash"],
            "feature": "score",
            "method": "equal_width",
            "merge_groups": [],
            "selection": {
                "source_bin_ids": [method["bins"][bin_index]["id"]]
            },
        },
        fixture["ctx"],
    )


def test_add_and_compile_persist_governed_pool_without_building_strategy(
    tmp_path: Path,
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

    assert added["revision"] == 1
    assert added["status"] == "draft"
    assert added["validation_status"] == "unvalidated"
    assert added["entry_count"] == 1
    assert added["entries"][0]["rule_id"] == fixture["first"]["rule"]["rule_id"]
    assert added["entries"][0]["source"]["evidence_identity"][
        "sample_context_hash"
    ] == sample_context_hash_from_candidate_evidence(
        fixture["source_output"]["candidate_evidence"]
    )
    assert len(added["artifacts"]) == 1
    assert added["artifacts"][0]["kind"] == POOL_ARTIFACT_KIND

    repository = StrategyCandidatePoolRepository(fixture["settings"].db_path)
    assert repository.get_current(fixture["task"].id, "approval") == added["pool"]
    artifact_record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        added["artifacts"][0]["artifact_id"],
    )
    assert artifact_record is not None
    assert sha256_file(Path(artifact_record["path"])) == artifact_record["content_hash"]
    assert json.loads(Path(artifact_record["path"]).read_text("utf-8")) == added["pool"]

    before = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    compiled = run_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    after = TaskArtifactRepository(fixture["settings"].db_path).list_for_task(
        fixture["task"].id
    )
    assert before == after
    assert compiled["requirements"] == []
    assert compiled["strategy_spec"]["rules"][0]["rule_id"] == added["entries"][0][
        "rule_id"
    ]
    assert compiled["selected_strategy_design"]["design_hash"] == compiled[
        "design_hash"
    ]
    assert fixture["runtime"].strategies.list_for_task(fixture["task"].id) == []


def test_voting_admission_rejects_earlier_logically_dominating_rule(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    pool = None
    for reason, candidate in (
        ("UNION", _refine_bins(fixture, [0, 1, 2])),
        ("LEFT", _refine_bins(fixture, [0, 1])),
        ("RIGHT", _refine_bins(fixture, [1, 2])),
    ):
        added = run_add_candidate_to_pool(
            _add_inputs(
                candidate,
                expected_revision=0 if pool is None else pool["revision"],
                expected_hash=(
                    ABSENT_POOL_SNAPSHOT_HASH
                    if pool is None
                    else pool["snapshot_hash"]
                ),
                action=_action("reject", reason=reason),
            ),
            fixture["ctx"],
            fixture["runtime"],
        )
        pool = added["pool"]
    assert pool is not None

    built = run_build_voting_candidate(
        {
            "strategy_type": "approval",
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
            "selected_entry_ids": [
                entry["entry_id"] for entry in pool["entries"][1:]
            ],
            "n": 2,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert built["effect"]["matched_count"] > 0
    request = {
        **_add_inputs(
            built,
            expected_revision=pool["revision"],
            expected_hash=pool["snapshot_hash"],
            action=_action("review", reason="VOTE"),
        ),
        "placement_mode": "before_selected_members",
    }

    with pytest.raises(StrategyError, match="unreachable"):
        run_add_candidate_to_pool(
            request,
            fixture["ctx"],
            fixture["runtime"],
        )
    assert StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval") == pool


def test_reorder_rejects_new_prefix_that_fully_shadows_voting(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    pool = None
    for reason, candidate in (
        ("LEFT", _refine_bins(fixture, [0, 1])),
        ("RIGHT", _refine_bins(fixture, [1, 2])),
        ("UNION", _refine_bins(fixture, [0, 1, 2])),
    ):
        added = run_add_candidate_to_pool(
            _add_inputs(
                candidate,
                expected_revision=0 if pool is None else pool["revision"],
                expected_hash=(
                    ABSENT_POOL_SNAPSHOT_HASH
                    if pool is None
                    else pool["snapshot_hash"]
                ),
                action=_action("reject", reason=reason),
            ),
            fixture["ctx"],
            fixture["runtime"],
        )
        pool = added["pool"]
    assert pool is not None

    built = run_build_voting_candidate(
        {
            "strategy_type": "approval",
            "expected_pool_revision": pool["revision"],
            "expected_pool_snapshot_hash": pool["snapshot_hash"],
            "selected_entry_ids": [
                entry["entry_id"] for entry in pool["entries"][:2]
            ],
            "n": 2,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    admitted = run_add_candidate_to_pool(
        {
            **_add_inputs(
                built,
                expected_revision=pool["revision"],
                expected_hash=pool["snapshot_hash"],
                action=_action("review", reason="VOTE"),
            ),
            "placement_mode": "before_selected_members",
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    rule_ids = [entry["rule_id"] for entry in admitted["entries"]]
    reordered = [rule_ids[3], rule_ids[0], rule_ids[1], rule_ids[2]]

    with pytest.raises(StrategyError, match="earlier first_match Pool rules shadow"):
        run_reorder_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": admitted["revision"],
                "expected_pool_snapshot_hash": admitted["snapshot_hash"],
                "ordered_rule_ids": reordered,
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    assert StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval") == admitted["pool"]


def test_archived_v1_draft_requires_rebuild_and_is_disclosed_by_v2_mutation(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    archive = _insert_archived_legacy_draft(fixture)

    with pytest.raises(StrategyPoolLegacyDraftNeedsRebuildError) as exc:
        run_compile_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": archive["current_revision"],
                "expected_pool_snapshot_hash": archive["current_snapshot_hash"],
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    detail = exc.value.to_detail()
    assert detail["kind"] == "legacy_pool_draft_needs_rebuild"
    assert detail["archive"] == archive

    added = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    assert added["revision"] == 1
    assert added["archived_legacy_draft"] == archive
    assert len(added["warnings"]) == 1
    assert "separate rebuild" in added["warnings"][0]

    compiled = run_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert compiled["archived_legacy_draft"] == archive
    assert compiled["warnings"] == added["warnings"]
    assert (
        compiled["selected_strategy_design"]["schema_version"]
        == "strategy.selected-strategy-design.v2"
    )

    with connect(fixture["settings"].db_path) as conn:
        detail_json = conn.execute(
            "SELECT detail_json FROM audit "
            "WHERE kind = 'strategy.pool.add_candidate' "
            "ORDER BY at DESC LIMIT 1"
        ).fetchone()[0]
    audit_detail = json.loads(detail_json)
    assert audit_detail["archived_legacy_draft"] == archive
    assert audit_detail["warnings"] == added["warnings"]


def test_mutation_tools_resolve_rule_ids_and_require_complete_reorder(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    first = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    second_candidate = fixture["refine"](2)
    second = run_add_candidate_to_pool(
        _add_inputs(
            second_candidate,
            expected_revision=first["revision"],
            expected_hash=first["snapshot_hash"],
            action=_action("review", reason="MANUAL"),
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    first_rule, second_rule = [row["rule_id"] for row in second["entries"]]

    with pytest.raises(StrategyError, match="complete"):
        run_reorder_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": second["revision"],
                "expected_pool_snapshot_hash": second["snapshot_hash"],
                "ordered_rule_ids": [second_rule],
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    reordered = run_reorder_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": second["revision"],
            "expected_pool_snapshot_hash": second["snapshot_hash"],
            "ordered_rule_ids": [second_rule, first_rule],
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert [row["rule_id"] for row in reordered["entries"]] == [
        second_rule,
        first_rule,
    ]

    changed = run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "expected_pool_revision": reordered["revision"],
            "expected_pool_snapshot_hash": reordered["snapshot_hash"],
            "rule_id": first_rule,
            "action": _action("review", reason="SECOND_REVIEW"),
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    removed = run_remove_pool_entry(
        {
            "strategy_type": "approval",
            "expected_pool_revision": changed["revision"],
            "expected_pool_snapshot_hash": changed["snapshot_hash"],
            "rule_id": second_rule,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert [row["rule_id"] for row in removed["entries"]] == [first_rule]


def test_pool_tools_fail_closed_on_stale_cas_foreign_or_drifted_asset(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    inputs = _add_inputs(
        fixture["first"],
        expected_revision=0,
        expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    added = strategy_tools.tool_add_candidate_to_pool(inputs, fixture["ctx"])

    with pytest.raises(StrategyError, match="snapshot|revision|stale|CAS"):
        run_set_pool_entry_action(
            {
                "strategy_type": "approval",
                "expected_pool_revision": 0,
                "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
                "rule_id": added["entries"][0]["rule_id"],
                "action": _action("review"),
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    foreign_ctx = _context(fixture["settings"], fixture["other_task"].id)
    with pytest.raises(StrategyError, match="artifact not found"):
        run_add_candidate_to_pool(
            inputs,
            foreign_ctx,
            strategy_tools._runtime(foreign_ctx),
        )

    other_candidate = fixture["refine"](1)
    asset_record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        other_candidate["artifacts"][0]["artifact_id"],
    )
    assert asset_record is not None
    Path(asset_record["path"]).write_bytes(Path(asset_record["path"]).read_bytes() + b"\n")
    with pytest.raises(StrategyError, match="content hash drifted"):
        run_add_candidate_to_pool(
            _add_inputs(
                other_candidate,
                expected_revision=added["revision"],
                expected_hash=added["snapshot_hash"],
            ),
            fixture["ctx"],
            fixture["runtime"],
        )


def test_unknown_and_duplicate_rule_ids_fail_without_new_revision(tmp_path: Path) -> None:
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
    with pytest.raises(StrategyError, match="unknown rule_id"):
        run_remove_pool_entry(
            {
                "strategy_type": "approval",
                "expected_pool_revision": added["revision"],
                "expected_pool_snapshot_hash": added["snapshot_hash"],
                "rule_id": "candidate-rule-" + "0" * 32,
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    with pytest.raises(StrategyError, match="duplicate"):
        run_reorder_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": added["revision"],
                "expected_pool_snapshot_hash": added["snapshot_hash"],
                "ordered_rule_ids": [
                    added["entries"][0]["rule_id"],
                    added["entries"][0]["rule_id"],
                ],
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    repository = StrategyCandidatePoolRepository(fixture["settings"].db_path)
    assert repository.get_current(fixture["task"].id, "approval") == added["pool"]


def test_identical_add_retry_is_one_revision_artifact_and_audit(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    inputs = _add_inputs(
        fixture["first"],
        expected_revision=0,
        expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    first = run_add_candidate_to_pool(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    replay = run_add_candidate_to_pool(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )

    assert replay == first
    pool_artifacts = [
        record
        for record in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if record["kind"] == POOL_ARTIFACT_KIND
    ]
    assert len(pool_artifacts) == 1
    with connect(fixture["settings"].db_path) as conn:
        revision_count = conn.execute(
            "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
        ).fetchone()[0]
        audit_count = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = 'strategy.pool.add_candidate'"
        ).fetchone()[0]
    assert revision_count == 1
    assert audit_count == 1


def test_compile_fails_closed_when_pool_artifact_bytes_drift(tmp_path: Path) -> None:
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
    record = TaskArtifactRepository(fixture["settings"].db_path).get_for_task(
        fixture["task"].id,
        added["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_bytes(Path(record["path"]).read_bytes() + b"\n")

    with pytest.raises(StrategyError, match="content hash drifted"):
        run_compile_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": added["revision"],
                "expected_pool_snapshot_hash": added["snapshot_hash"],
            },
            fixture["ctx"],
            fixture["runtime"],
        )


def test_mutation_fails_closed_when_parent_pool_artifact_bytes_drift(
    tmp_path: Path,
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
    artifact_repository = TaskArtifactRepository(fixture["settings"].db_path)
    record = artifact_repository.get_for_task(
        fixture["task"].id,
        added["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    before_artifacts = artifact_repository.list_for_task(fixture["task"].id)
    path = Path(record["path"])
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(StrategyError, match="content hash drifted"):
        run_set_pool_entry_action(
            {
                "strategy_type": "approval",
                "expected_pool_revision": added["revision"],
                "expected_pool_snapshot_hash": added["snapshot_hash"],
                "rule_id": added["entries"][0]["rule_id"],
                "action": _action("reject", reason="DRIFT_MUST_BLOCK"),
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    assert StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval") == added["pool"]
    assert artifact_repository.list_for_task(fixture["task"].id) == before_artifacts
    with connect(fixture["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
        ).fetchone()[0] == 1


@pytest.mark.parametrize("drift_target", ["parent_evidence", "dataset"])
def test_add_fails_closed_when_parent_or_dataset_bytes_drift(
    tmp_path: Path,
    drift_target: str,
) -> None:
    fixture = _setup(tmp_path)
    if drift_target == "parent_evidence":
        asset_artifact = TaskArtifactRepository(
            fixture["settings"].db_path
        ).get_for_task(
            fixture["task"].id,
            fixture["first"]["artifacts"][0]["artifact_id"],
        )
        assert asset_artifact is not None
        parent_artifact = TaskArtifactRepository(
            fixture["settings"].db_path
        ).get_for_task(
            fixture["task"].id,
            asset_artifact["provenance"]["source_artifact_id"],
        )
        assert parent_artifact is not None
        drift_path = Path(parent_artifact["path"])
    else:
        drift_path = Path(
            fixture["runtime"].registry.resolve_path(fixture["dataset"].id)
        )
    drift_path.write_bytes(drift_path.read_bytes() + b"drift")

    with pytest.raises(
        StrategyError,
        match="content hash drifted|failed hash verification",
    ):
        run_add_candidate_to_pool(
            _add_inputs(
                fixture["first"],
                expected_revision=0,
                expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fixture["ctx"],
            fixture["runtime"],
        )


def test_pool_rejects_candidate_from_different_evidence_identity(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    first = run_add_candidate_to_pool(
        _add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    workspaces = DataWorkspaceRepository(fixture["settings"].db_path)
    current = workspaces.get_or_default(fixture["task"].id)
    changed = workspaces.save(
        fixture["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=current.active_dataset_id,
            active_dataset_content_hash=current.active_dataset_content_hash,
            page=current.page,
            selected_field="score",
            semantic_mapping=current.semantic_mapping,
        ),
        expected_revision=current.revision,
    )
    other_candidate = _refine_for_workspace(fixture, changed, 2)

    with pytest.raises(StrategyError, match="evidence identity"):
        run_add_candidate_to_pool(
            _add_inputs(
                other_candidate,
                expected_revision=first["revision"],
                expected_hash=first["snapshot_hash"],
            ),
            fixture["ctx"],
            fixture["runtime"],
        )
    repository = StrategyCandidatePoolRepository(fixture["settings"].db_path)
    assert repository.get_current(fixture["task"].id, "approval") == first["pool"]
    with connect(fixture["settings"].db_path) as conn:
        revision_count = conn.execute(
            "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
        ).fetchone()[0]
    assert revision_count == 1


def test_pool_artifact_and_revision_roll_back_when_audit_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    before_artifacts = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_for_task(fixture["task"].id)

    def fail_audit(*args, **kwargs):
        raise RuntimeError("injected pool audit failure")

    monkeypatch.setattr(
        strategy_pool_repository,
        "_write_audit_row",
        fail_audit,
    )
    with pytest.raises(RuntimeError, match="injected pool audit failure"):
        run_add_candidate_to_pool(
            _add_inputs(
                fixture["first"],
                expected_revision=0,
                expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fixture["ctx"],
            fixture["runtime"],
        )

    after_artifacts = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_for_task(fixture["task"].id)
    assert after_artifacts == before_artifacts
    repository = StrategyCandidatePoolRepository(fixture["settings"].db_path)
    assert repository.get_current(fixture["task"].id, "approval") is None
    pool_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_candidate_pools"
    )
    assert not list(pool_dir.glob("*.json"))
    with connect(fixture["settings"].db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind LIKE 'strategy.pool.%'"
        ).fetchone()[0] == 0


def test_compile_rejects_repository_valid_but_unknown_candidate_adapter(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    task_id = fixture["task"].id
    source_path = (
        Path(fixture["settings"].tasks_dir)
        / task_id
        / "strategy_tree_candidates"
        / "tree-candidate.json"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_bytes = b"{}"
    source_path.write_bytes(source_bytes)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).register(
        task_id=task_id,
        kind="strategy_tree_candidate_json",
        path=str(source_path),
        content_hash=source_hash,
        origin_tool="strategy.build_tree_candidate",
        provenance={"schema_version": "strategy.tree-candidate-artifact.v1"},
    )
    candidate_evidence = fixture["source_output"]["candidate_evidence"]
    identity = candidate_evidence["identity"]
    fragment = build_verified_candidate_fragment(
        artifact={
            "artifact_id": source_record["id"],
            "artifact_kind": source_record["kind"],
            "artifact_schema_version": "strategy.tree-candidate-artifact.v1",
            "artifact_content_hash": source_record["content_hash"],
            "origin_tool": source_record["origin_tool"],
        },
        asset={
            "schema_version": "strategy.tree-candidate.v1",
            "asset_id": "tree-asset-a",
            "asset_hash": "a" * 64,
            "asset_type": "decision_tree",
        },
        fragment_type="strategy_rule",
        rule_id="tree-rule-a",
        condition={
            "op": "compare",
            "field": "score",
            "operator": "<",
            "value": 200,
            "missing": "no_match",
        },
        requirements=[],
        effect_id="tree-effect-a",
        evidence_id="tree-evidence-a",
        evidence_hash="b" * 64,
        evidence_identity={
            "dataset_id": identity["dataset_id"],
            "dataset_content_hash": identity["dataset_content_hash"],
            "workspace_revision": identity["workspace_revision"],
            "workspace_generation": identity["workspace_generation"],
            "semantic_mapping_hash": identity["semantic_mapping_hash"],
            "sample_context_hash": sample_context_hash_from_candidate_evidence(
                candidate_evidence
            ),
        },
    )
    snapshot = add_verified_candidate_fragment(
        None,
        task_id=task_id,
        strategy_type="approval",
        default_action=_action("approval"),
        verified_candidate_fragment=fragment,
        action=_action("reject", reason="TREE"),
    )
    pool_path = (
        Path(fixture["settings"].tasks_dir)
        / task_id
        / "strategy_candidate_pools"
        / pool_tools_module._pool_filename(snapshot)
    )
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(canonical_strategy_pool_json(snapshot), "utf-8")
    pool_record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).register(
        task_id=task_id,
        kind=POOL_ARTIFACT_KIND,
        path=str(pool_path),
        content_hash=strategy_pool_artifact_content_hash(snapshot),
        origin_tool="strategy.add_candidate_to_pool",
        provenance=pool_tools_module._pool_provenance(snapshot),
    )
    StrategyCandidatePoolRepository(fixture["settings"].db_path).apply_snapshot(
        snapshot=snapshot,
        expected_revision=0,
        expected_snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        artifact_id=pool_record["id"],
        artifact_content_hash=pool_record["content_hash"],
        audit={
            "kind": "strategy.pool.add_candidate",
            "target_ref": snapshot["revision_id"],
            "inputs_hash": snapshot["operation"]["operation_hash"],
            "outcome": "succeeded",
            "detail": {"entry_count": 1},
        },
    )

    with pytest.raises(StrategyError, match="unsupported candidate fragment adapter"):
        run_compile_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": 1,
                "expected_pool_snapshot_hash": snapshot["snapshot_hash"],
            },
            fixture["ctx"],
            fixture["runtime"],
        )
