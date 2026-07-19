"""Plan-time binding for explicit Voting members from the current Pool."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.strategy_setup import StrategySetupError
from marvis.agent.turn_handlers import (
    _candidate_asset_artifact_slots,
    _strategy_voting_candidate_plan_slots,
)
from marvis.packs.strategy.voting_candidate_fragment import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    VOTING_CANDIDATE_ORIGIN_TOOL,
)


RULE_A = "candidate-rule-" + "a" * 32
RULE_B = "candidate-rule-" + "b" * 32
ENTRY_A = "pool-entry-" + "1" * 32
ENTRY_B = "pool-entry-" + "2" * 32
ASSET_ID = "candidate-asset-" + "c" * 32
ASSET_HASH = "d" * 64
CONTENT_HASH = "e" * 64


class _PoolRepository:
    def __init__(self, current) -> None:
        self.current = current

    def get_current(self, task_id: str, strategy_type: str):
        assert task_id == "task-1"
        assert strategy_type == "approval"
        return self.current


def _runtime(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(
            db_path=tmp_path / "db.sqlite",
            tasks_dir=tmp_path / "tasks",
        )
    )


def _draft(rule_ids: list[str] | None = None, *, n: int = 2):
    return StandardWorkflowRequestDraft(
        workflow="voting_candidate_build",
        workflow_inputs={
            "strategy_type": "approval",
            "rule_ids": rule_ids or [RULE_B, RULE_A],
            "n": n,
        },
    )


def _pool(*, nested: bool = False):
    return {
        "revision": 7,
        "entries": [
            {
                "entry_id": ENTRY_A,
                "rule_id": RULE_A,
                "position": 0,
                "enabled": True,
                "source": {"asset_type": "univariate_refinement"},
            },
            {
                "entry_id": ENTRY_B,
                "rule_id": RULE_B,
                "position": 1,
                "enabled": True,
                "source": {
                    "asset_type": "voting_n_of_k" if nested else "automatic_rule_tree"
                },
            },
        ],
    }


def test_voting_turn_binds_current_cas_and_canonical_entry_order(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda _db_path: _PoolRepository(_pool()),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda _snapshot: "f" * 64,
    )

    slots = _strategy_voting_candidate_plan_slots(
        _runtime(tmp_path),
        SimpleNamespace(id="task-1"),
        _draft(),
    )

    assert slots == {
        "strategy_type": "approval",
        "expected_pool_revision": 7,
        "expected_pool_snapshot_hash": "f" * 64,
        "selected_entry_ids": [ENTRY_A, ENTRY_B],
        "n": 2,
    }


def test_voting_turn_requires_existing_exact_current_rules(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda _db_path: _PoolRepository(None),
    )
    with pytest.raises(StrategySetupError, match="没有 approval Strategy Pool"):
        _strategy_voting_candidate_plan_slots(
            _runtime(tmp_path), SimpleNamespace(id="task-1"), _draft()
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda _db_path: _PoolRepository(_pool()),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda _snapshot: "f" * 64,
    )
    with pytest.raises(StrategySetupError, match="没有唯一匹配"):
        _strategy_voting_candidate_plan_slots(
            _runtime(tmp_path),
            SimpleNamespace(id="task-1"),
            _draft([RULE_A, "candidate-rule-" + "c" * 32]),
        )


def test_voting_turn_rejects_nested_voting_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        lambda _db_path: _PoolRepository(_pool(nested=True)),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda _snapshot: "f" * 64,
    )

    with pytest.raises(StrategySetupError, match="拒绝嵌套 Voting"):
        _strategy_voting_candidate_plan_slots(
            _runtime(tmp_path), SimpleNamespace(id="task-1"), _draft()
        )


def _voting_artifact(tmp_path):
    return {
        "id": "task-artifact-voting",
        "task_id": "task-1",
        "kind": VOTING_CANDIDATE_ARTIFACT_KIND,
        "path": str(tmp_path / f"{ASSET_ID}.json"),
        "content_hash": CONTENT_HASH,
        "origin_tool": VOTING_CANDIDATE_ORIGIN_TOOL,
        "provenance": {"asset_id": ASSET_ID, "asset_hash": ASSET_HASH},
    }


class _ArtifactRepository:
    def __init__(self, artifact) -> None:
        self.artifact = artifact

    def list_for_task(self, task_id: str):
        assert task_id == "task-1"
        return [self.artifact]

    def transaction(self):
        return nullcontext(object())


def test_pool_add_turn_binds_verified_voting_artifact(tmp_path, monkeypatch) -> None:
    artifact = _voting_artifact(tmp_path)
    repository = _ArtifactRepository(artifact)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.TaskArtifactRepository",
        lambda _db_path: repository,
    )
    calls = []

    def load(conn, **kwargs):
        calls.append((conn, kwargs))
        return SimpleNamespace(
            artifact_id=artifact["id"],
            content_hash=CONTENT_HASH,
            asset={"asset_id": ASSET_ID, "asset_hash": ASSET_HASH},
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_verified_voting_candidate_artifact_on_connection",
        load,
    )

    slots = _candidate_asset_artifact_slots(
        _runtime(tmp_path), task_id="task-1", asset_id=ASSET_ID
    )

    assert slots == {
        "source_artifact_id": artifact["id"],
        "expected_artifact_content_hash": CONTENT_HASH,
        "expected_asset_id": ASSET_ID,
        "expected_asset_hash": ASSET_HASH,
        "_candidate_asset_type": "voting_n_of_k",
    }
    assert calls[0][1] == {
        "tasks_dir": tmp_path / "tasks",
        "task_id": "task-1",
        "artifact_id": artifact["id"],
        "expected_content_hash": CONTENT_HASH,
        "expected_asset_id": ASSET_ID,
        "expected_asset_hash": ASSET_HASH,
    }


def test_pool_add_turn_rejects_unverified_voting_artifact(
    tmp_path, monkeypatch
) -> None:
    artifact = _voting_artifact(tmp_path)
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.TaskArtifactRepository",
        lambda _db_path: _ArtifactRepository(artifact),
    )

    def reject(*_args, **_kwargs):
        raise ValueError("tampered")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_verified_voting_candidate_artifact_on_connection",
        reject,
    )

    with pytest.raises(StrategySetupError, match="Voting 候选资产.*完整性校验"):
        _candidate_asset_artifact_slots(
            _runtime(tmp_path), task_id="task-1", asset_id=ASSET_ID
        )
