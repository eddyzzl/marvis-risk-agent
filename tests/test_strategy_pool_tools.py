from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import marvis.repositories.strategy_pool as strategy_pool_repository
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
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import (
    POOL_ARTIFACT_KIND,
    run_add_candidate_to_pool,
    run_compile_strategy_pool,
    run_remove_pool_entry,
    run_reorder_strategy_pool,
    run_set_pool_entry_action,
)
from marvis.plugins.contracts import ToolContext
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
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
    source_output = strategy_tools.tool_analyze_univariate_candidates(
        {
            "dataset_id": dataset.id,
            "expected_content_hash": dataset.content_hash,
            "workspace_revision": workspace.revision,
            "analysis_generation": workspace.analysis_generation,
            "semantic_mapping_hash": data_semantic_mapping_hash(mapping),
            "target_col": "bad",
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


def _refine_for_workspace(fixture: dict, workspace, bin_index: int) -> dict:
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
