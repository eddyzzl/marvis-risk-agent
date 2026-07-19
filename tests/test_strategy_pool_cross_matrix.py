from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from marvis.db_schema import connect
from marvis.packs.strategy import pool_tools
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.cross_matrix_cell_selection import (
    CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.repositories.strategy_pool import (
    POOL_ARTIFACT_KIND,
    StrategyCandidatePoolRepository,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_cross_matrix_cell_selection_tool import _fixture


def _action(action_type: str, *, reason: str | None = None) -> dict:
    values = {"approval": "approve", "reject": "reject", "review": "review"}
    return {
        "type": action_type,
        "value": values[action_type],
        "reason_code": reason,
        "stop": True,
    }


def _materialize(
    fx: SimpleNamespace,
    cell_ids: list[str],
    *,
    reason: str | None = None,
) -> dict:
    inputs = {**fx.inputs, "cell_ids": cell_ids}
    if reason is not None:
        inputs["selection_reason"] = reason
    return strategy_tools.tool_materialize_cross_matrix_cell_selection(
        inputs,
        fx.ctx,
    )


def _add_inputs(
    candidate: dict,
    *,
    revision: int,
    snapshot_hash: str,
    action: dict | None = None,
) -> dict:
    artifact = candidate["artifacts"][0]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "expected_artifact_content_hash": artifact["content_hash"],
        "expected_asset_id": candidate["source_asset_id"],
        "expected_asset_hash": candidate["source_asset_hash"],
        "strategy_type": "approval",
        "default_action": _action("approval"),
        "action": action or _action("reject", reason="CROSS_RISK"),
        "expected_pool_revision": revision,
        "expected_pool_snapshot_hash": snapshot_hash,
    }


def _pool_counts(fx: SimpleNamespace) -> tuple[int, int, int]:
    records = TaskArtifactRepository(fx.settings.db_path).list_for_task(fx.task.id)
    artifact_count = sum(record["kind"] == POOL_ARTIFACT_KIND for record in records)
    with connect(fx.settings.db_path) as conn:
        revision_count = conn.execute(
            "SELECT COUNT(*) FROM strategy_candidate_pool_revisions"
        ).fetchone()[0]
        audit_count = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind LIKE 'strategy.pool.%'"
        ).fetchone()[0]
    return int(artifact_count), int(revision_count), int(audit_count)


def test_cross_matrix_cell_group_materialize_add_compile_full_chain(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    selected = _materialize(
        fx,
        [fx.populated[2]["cell_id"], fx.populated[0]["cell_id"]],
        reason="joint risk segment",
    )
    action = _action("review", reason="CROSS_REVIEW")

    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            action=action,
        ),
        fx.ctx,
    )
    compiled = strategy_tools.tool_compile_strategy_pool(
        {
            "strategy_type": "approval",
            "expected_pool_revision": added["revision"],
            "expected_pool_snapshot_hash": added["snapshot_hash"],
        },
        fx.ctx,
    )

    [entry] = added["entries"]
    [rule] = compiled["strategy_spec"]["rules"]
    assert entry["source"]["artifact_id"] == selected["artifacts"][0]["artifact_id"]
    assert entry["source"]["artifact_kind"] == (
        CROSS_MATRIX_CELL_SELECTION_ARTIFACT_KIND
    )
    assert entry["source"]["asset_id"] == fx.matrix["asset_id"]
    assert entry["source"]["fragment_id"] == selected["group_id"]
    assert entry["source"]["effect_id"] == selected["effect_id"]
    assert entry["action"] == action
    assert "metrics" not in entry
    assert rule["condition"] == entry["execution"]["condition"]
    assert rule["action"] == action


def test_complete_cross_matrix_cannot_enter_pool_directly_and_is_zero_mutation(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    matrix = fx.matrix_result

    with pytest.raises(
        StrategyError,
        match="complete Cross Matrix assets cannot be admitted directly",
    ):
        strategy_tools.tool_add_candidate_to_pool(
            {
                "source_artifact_id": fx.matrix_artifact["artifact_id"],
                "expected_artifact_content_hash": fx.matrix_artifact["content_hash"],
                "expected_asset_id": matrix["asset_id"],
                "expected_asset_hash": matrix["asset_hash"],
                "strategy_type": "approval",
                "default_action": _action("approval"),
                "action": _action("reject"),
                "expected_pool_revision": 0,
                "expected_pool_snapshot_hash": ABSENT_POOL_SNAPSHOT_HASH,
            },
            fx.ctx,
        )

    assert _pool_counts(fx) == (0, 0, 0)
    assert (
        StrategyCandidatePoolRepository(fx.settings.db_path).get_current(
            fx.task.id, "approval"
        )
        is None
    )


def test_same_matrix_disjoint_cell_groups_are_admitted(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    first_selection = _materialize(fx, [fx.populated[0]["cell_id"]])
    second_selection = _materialize(fx, [fx.populated[1]["cell_id"]])

    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            first_selection,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx.ctx,
    )
    second = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            second_selection,
            revision=first["revision"],
            snapshot_hash=first["snapshot_hash"],
            action=_action("review", reason="SECOND_GROUP"),
        ),
        fx.ctx,
    )

    assert len(second["entries"]) == 2
    assert len({entry["source"]["asset_id"] for entry in second["entries"]}) == 1
    assert len({entry["source"]["fragment_id"] for entry in second["entries"]}) == 2


def test_same_group_with_different_reason_is_duplicate_without_mutation(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    cell_ids = [fx.populated[0]["cell_id"]]
    first_selection = _materialize(fx, cell_ids, reason="first review")
    second_selection = _materialize(fx, cell_ids, reason="second review")
    assert first_selection["selection_id"] != second_selection["selection_id"]
    assert first_selection["group_id"] == second_selection["group_id"]
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            first_selection,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx.ctx,
    )
    before = _pool_counts(fx)

    with pytest.raises(StrategyError, match="duplicate asset fragment"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                second_selection,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
            ),
            fx.ctx,
        )

    assert _pool_counts(fx) == before
    assert (
        StrategyCandidatePoolRepository(fx.settings.db_path).get_current(
            fx.task.id, "approval"
        )
        == first["pool"]
    )


def test_same_matrix_overlapping_groups_are_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    shared = fx.populated[1]["cell_id"]
    first_selection = _materialize(
        fx,
        [fx.populated[0]["cell_id"], shared],
    )
    second_selection = _materialize(
        fx,
        [shared, fx.populated[2]["cell_id"]],
    )
    first = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            first_selection,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx.ctx,
    )
    before = _pool_counts(fx)

    with pytest.raises(StrategyError, match=shared):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                second_selection,
                revision=first["revision"],
                snapshot_hash=first["snapshot_hash"],
            ),
            fx.ctx,
        )

    assert _pool_counts(fx) == before
    assert (
        StrategyCandidatePoolRepository(fx.settings.db_path).get_current(
            fx.task.id, "approval"
        )
        == first["pool"]
    )


@pytest.mark.parametrize(
    "drift",
    ["selection", "matrix", "parent_evidence", "dataset", "dataset_path"],
)
def test_cross_matrix_under_lock_lineage_drift_is_zero_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    fx = _fixture(tmp_path)
    selected = _materialize(fx, [fx.populated[0]["cell_id"]])
    repository = TaskArtifactRepository(fx.settings.db_path)
    selection_record = repository.get_for_task(
        fx.task.id, selected["artifacts"][0]["artifact_id"]
    )
    parent_record = next(
        record
        for record in repository.list_for_task(fx.task.id)
        if record["kind"] == "strategy_candidate_json"
    )
    assert selection_record is not None
    paths = {
        "selection": Path(selection_record["path"]),
        "matrix": Path(fx.source_record["path"]),
        "parent_evidence": Path(parent_record["path"]),
        "dataset": Path(fx.runtime.registry.resolve_verified_path(fx.dataset.id)),
    }
    original = pool_tools._require_lineage_on_connection
    changed = False

    def drift_then_verify(conn, lineage, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            if drift == "dataset_path":
                conn.execute(
                    "UPDATE datasets SET source_path = ? WHERE id = ?",
                    ("forged/path.parquet", fx.dataset.id),
                )
            else:
                path = paths[drift]
                path.write_bytes(path.read_bytes() + b"\n")
        return original(conn, lineage, **kwargs)

    monkeypatch.setattr(pool_tools, "_require_lineage_on_connection", drift_then_verify)
    with pytest.raises(StrategyError, match="drift|changed|canonical|hash"):
        strategy_tools.tool_add_candidate_to_pool(
            _add_inputs(
                selected,
                revision=0,
                snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
            ),
            fx.ctx,
        )

    assert _pool_counts(fx) == (0, 0, 0)
    assert (
        StrategyCandidatePoolRepository(fx.settings.db_path).get_current(
            fx.task.id, "approval"
        )
        is None
    )


@pytest.mark.parametrize(
    "drift",
    ["selection", "matrix", "parent_evidence", "dataset"],
)
def test_existing_pool_compile_fails_closed_after_cross_lineage_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    fx = _fixture(tmp_path)
    selected = _materialize(fx, [fx.populated[0]["cell_id"]])
    added = strategy_tools.tool_add_candidate_to_pool(
        _add_inputs(
            selected,
            revision=0,
            snapshot_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fx.ctx,
    )
    repository = TaskArtifactRepository(fx.settings.db_path)
    selection_record = repository.get_for_task(
        fx.task.id, selected["artifacts"][0]["artifact_id"]
    )
    parent_record = next(
        record
        for record in repository.list_for_task(fx.task.id)
        if record["kind"] == "strategy_candidate_json"
    )
    assert selection_record is not None
    paths = {
        "selection": Path(selection_record["path"]),
        "matrix": Path(fx.source_record["path"]),
        "parent_evidence": Path(parent_record["path"]),
        "dataset": Path(fx.runtime.registry.resolve_verified_path(fx.dataset.id)),
    }
    paths[drift].write_bytes(paths[drift].read_bytes() + b"\n")
    before = _pool_counts(fx)

    with pytest.raises(StrategyError, match="drift|changed|canonical|hash"):
        strategy_tools.tool_compile_strategy_pool(
            {
                "strategy_type": "approval",
                "expected_pool_revision": added["revision"],
                "expected_pool_snapshot_hash": added["snapshot_hash"],
            },
            fx.ctx,
        )

    assert _pool_counts(fx) == before
    assert (
        StrategyCandidatePoolRepository(fx.settings.db_path).get_current(
            fx.task.id, "approval"
        )
        == added["pool"]
    )
