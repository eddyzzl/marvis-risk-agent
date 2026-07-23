from __future__ import annotations

import json
import sqlite3

import pytest

import marvis.repositories.strategy as strategy_repository_module
from marvis.db import StrategyRepository, connect, init_db
from marvis.packs.strategy import build_strategy


_STAMP = "2026-07-18T08:00:00+00:00"
_EXPIRES = "2026-07-18T09:00:00+00:00"


def _strategy(description: str, cutoff: int):
    return build_strategy(
        "approval",
        [{"condition": f"score < {cutoff}", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description=description,
    )


def _adoption_audit(strategy_id: str) -> dict:
    return {
        "kind": "strategy.adopt",
        "target_ref": strategy_id,
        "outcome": "succeeded",
        "detail": {"task_id": "task-1"},
    }


def _artifact_audit(artifact_id: str, *, kind: str = "strategy.artifact") -> dict:
    return {
        "kind": kind,
        "target_ref": artifact_id,
        "outcome": "succeeded",
        "detail": {"task_id": "task-1"},
    }


def _effect_target(
    repo: StrategyRepository,
    strategy_id: str,
    champion_id: str,
) -> dict:
    meta = repo.get_strategy_meta(strategy_id)
    assert meta is not None
    return {
        "kind": "strategy",
        "id": strategy_id,
        "expected_status": "draft",
        "result_status": "adopted",
        "version": meta["version"],
        "task_id": meta["task_id"],
        "strategy_type": meta["strategy_type"],
        "strategy_spec_hash": repo.get_strategy_spec_hash(strategy_id),
        "strategy_description": meta["description"],
        "current_champion_ids": [champion_id],
    }


def _insert_dispatched_effect(db_path, *, target: dict) -> str:
    effect_id = "effect-adopt-atomic"
    approval_id = "approval-adopt-atomic"
    reservation_id = "reservation-adopt-atomic"
    target_json = json.dumps(target, ensure_ascii=False, sort_keys=True)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO local_principals(
                id, kind, display_name, session_token_hash, status,
                created_at, last_seen_at, expires_at
            ) VALUES (?, 'local_session', ?, ?, 'active', ?, ?, ?)
            """,
            (
                "principal-adopt-atomic",
                "本地策略人员",
                "token-hash-atomic",
                _STAMP,
                _STAMP,
                _EXPIRES,
            ),
        )
        conn.execute(
            """
            INSERT INTO decision_records(
                id, task_id, plan_id, plan_revision, step_id, tool_ref,
                principal_id, decision, reason, manifest_hash, policy_hash,
                input_hash, evidence_hash, effect_target_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "decision-adopt-atomic",
                target["task_id"],
                "plan-adopt-atomic",
                1,
                "step-adopt",
                "strategy.adopt_strategy@1.0.0",
                "principal-adopt-atomic",
                "approve",
                "委员会批准",
                "manifest-hash",
                "policy-hash",
                "input-hash",
                "evidence-hash",
                target_json,
                _STAMP,
            ),
        )
        conn.execute(
            """
            INSERT INTO approval_records(
                id, decision_id, task_id, plan_id, plan_revision, step_id,
                tool_ref, principal_id, reason, manifest_hash, policy_hash,
                input_hash, evidence_hash, effect_target_json, nonce, status,
                issued_at, expires_at, reserved_at, reservation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval_id,
                "decision-adopt-atomic",
                target["task_id"],
                "plan-adopt-atomic",
                1,
                "step-adopt",
                "strategy.adopt_strategy@1.0.0",
                "principal-adopt-atomic",
                "委员会批准",
                "manifest-hash",
                "policy-hash",
                "input-hash",
                "evidence-hash",
                target_json,
                "nonce-adopt-atomic",
                "reserved",
                _STAMP,
                _EXPIRES,
                _STAMP,
                reservation_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO effect_executions(
                id, approval_id, reservation_id, runtime_generation, status,
                prepared_at, dispatched_at, detail_json
            ) VALUES (?, ?, ?, ?, 'dispatched', ?, ?, '{}')
            """,
            (
                effect_id,
                approval_id,
                reservation_id,
                "runtime-atomic",
                _STAMP,
                _STAMP,
            ),
        )
    return effect_id


def _setup_governed_adoption(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    champion = _strategy("champion", 600)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", champion)
    repo.create_strategy("task-1", challenger)
    repo.adopt_strategy_with_audit(
        champion.id,
        reason="initial champion",
        audit=_adoption_audit(champion.id),
    )
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id, champion.id),
    )
    return db_path, repo, champion, challenger, effect_id


def _ledger_states(db_path) -> tuple[str, str]:
    with connect(db_path) as conn:
        effect = conn.execute(
            "SELECT status FROM effect_executions WHERE id = 'effect-adopt-atomic'"
        ).fetchone()
        approval = conn.execute(
            "SELECT status FROM approval_records WHERE id = 'approval-adopt-atomic'"
        ).fetchone()
    return str(effect["status"]), str(approval["status"])


def _audit_rows(db_path) -> list[tuple[str, str]]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT kind, target_ref FROM audit ORDER BY rowid"
        ).fetchall()
    return [(str(row["kind"]), str(row["target_ref"])) for row in rows]


def _adopt_on_connection(
    repo: StrategyRepository,
    conn: sqlite3.Connection,
    challenger_id: str,
    effect_id: str,
) -> dict:
    return repo.adopt_strategy_with_audit_on_connection(
        conn,
        challenger_id,
        reason="committee promotes challenger",
        audit=_adoption_audit(challenger_id),
        effect_execution_id=effect_id,
        runtime_generation="runtime-atomic",
    )


def _assert_adoption_rolled_back(
    db_path,
    repo: StrategyRepository,
    champion_id: str,
    challenger_id: str,
) -> None:
    champion_meta = repo.get_strategy_meta(champion_id)
    challenger_meta = repo.get_strategy_meta(challenger_id)
    assert champion_meta is not None
    assert challenger_meta is not None
    assert champion_meta["status"] == "adopted"
    assert challenger_meta["status"] == "draft"
    assert challenger_meta["adopted_at"] is None
    assert challenger_meta["adoption_reason"] is None
    assert _ledger_states(db_path) == ("dispatched", "reserved")
    assert repo.list_strategy_artifacts(challenger_id) == []
    assert _audit_rows(db_path) == [("strategy.adopt", champion_id)]


def test_caller_transaction_rolls_back_when_later_artifact_insert_fails(tmp_path):
    db_path, repo, champion, challenger, effect_id = _setup_governed_adoption(
        tmp_path
    )

    with pytest.raises(sqlite3.IntegrityError):
        with repo.transaction() as conn:
            _adopt_on_connection(repo, conn, challenger.id, effect_id)
            repo.save_strategy_artifact_with_audit_on_connection(
                conn,
                challenger.id,
                kind="decision_table_csv",
                path="decision.csv",
                artifact_id="artifact-duplicate",
                audit=_artifact_audit("artifact-duplicate"),
            )
            repo.save_strategy_artifact_with_audit_on_connection(
                conn,
                challenger.id,
                kind="strategy_json",
                path="strategy.json",
                artifact_id="artifact-duplicate",
                audit=_artifact_audit("artifact-duplicate-second"),
            )

    _assert_adoption_rolled_back(
        db_path,
        repo,
        champion.id,
        challenger.id,
    )


def test_caller_transaction_rolls_back_when_later_artifact_audit_fails(
    tmp_path,
    monkeypatch,
):
    db_path, repo, champion, challenger, effect_id = _setup_governed_adoption(
        tmp_path
    )
    real_write_audit_row = strategy_repository_module._write_audit_row

    def failing_artifact_audit(conn, **audit):
        if audit["kind"] == "strategy.artifact.fail":
            raise RuntimeError("artifact audit unavailable")
        return real_write_audit_row(conn, **audit)

    monkeypatch.setattr(
        strategy_repository_module,
        "_write_audit_row",
        failing_artifact_audit,
    )

    with pytest.raises(RuntimeError, match="artifact audit unavailable"):
        with repo.transaction() as conn:
            _adopt_on_connection(repo, conn, challenger.id, effect_id)
            repo.save_strategy_artifact_with_audit_on_connection(
                conn,
                challenger.id,
                kind="decision_table_csv",
                path="decision.csv",
                artifact_id="artifact-audit-fails",
                audit=_artifact_audit(
                    "artifact-audit-fails",
                    kind="strategy.artifact.fail",
                ),
            )

    _assert_adoption_rolled_back(
        db_path,
        repo,
        champion.id,
        challenger.id,
    )


def test_caller_transaction_commits_adoption_artifacts_and_audits(tmp_path):
    db_path, repo, champion, challenger, effect_id = _setup_governed_adoption(
        tmp_path
    )

    with repo.transaction() as conn:
        pre_adoption_artifact_id = (
            repo.save_strategy_artifact_with_audit_on_connection(
                conn,
                challenger.id,
                kind="review_snapshot_json",
                path="review.json",
                created_at=_STAMP,
                artifact_id="artifact-before-adoption",
                audit=_artifact_audit("artifact-before-adoption"),
            )
        )
        assert conn.in_transaction
        result = _adopt_on_connection(repo, conn, challenger.id, effect_id)
        artifact_id = repo.save_strategy_artifact_with_audit_on_connection(
            conn,
            challenger.id,
            kind="decision_table_csv",
            path="decision.csv",
            created_at=_STAMP,
            artifact_id="artifact-success",
            audit=_artifact_audit("artifact-success"),
        )

    assert result == {"version": 1, "retired_strategy_ids": [champion.id]}
    assert pre_adoption_artifact_id == "artifact-before-adoption"
    assert artifact_id == "artifact-success"
    assert repo.get_strategy_meta(champion.id)["status"] == "retired"
    assert repo.get_strategy_meta(challenger.id)["status"] == "adopted"
    assert _ledger_states(db_path) == ("committed", "consumed")
    assert repo.list_strategy_artifacts(challenger.id) == [
        {
            "id": "artifact-before-adoption",
            "strategy_id": challenger.id,
            "kind": "review_snapshot_json",
            "path": "review.json",
            "created_at": _STAMP,
        },
        {
            "id": "artifact-success",
            "strategy_id": challenger.id,
            "kind": "decision_table_csv",
            "path": "decision.csv",
            "created_at": _STAMP,
        }
    ]
    assert _audit_rows(db_path) == [
        ("strategy.adopt", champion.id),
        ("strategy.artifact", "artifact-before-adoption"),
        ("strategy.retire", champion.id),
        ("strategy.adopt", challenger.id),
        ("strategy.artifact", "artifact-success"),
    ]
