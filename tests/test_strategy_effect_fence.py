from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

import marvis.repositories.strategy as strategy_repository_module
from marvis.db import PluginRepository, StrategyRepository, connect, init_db
from marvis.packs.strategy import BacktestResult, build_strategy, run_typed_backtest
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy import tools as strategy_tools
from marvis.settings import build_settings
from marvis.state_machine import ConflictError


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


def _backtest(strategy_id: str) -> BacktestResult:
    return BacktestResult(
        strategy_id=strategy_id,
        approval_rate=0.7,
        approved_count=70,
        approved_bad_rate=0.04,
        rejected_bad_rate=0.22,
        expected_profit=2300.0,
        swap_in_count=5,
        swap_out_count=8,
        swap_in_bad_rate=0.12,
        swap_out_bad_rate=0.01,
        by_segment=(),
    )


def _audit(strategy_id: str) -> dict:
    return {
        "kind": "strategy.adopt",
        "target_ref": strategy_id,
        "outcome": "succeeded",
        "detail": {"task_id": "task-1"},
    }


def _effect_target(
    repo: StrategyRepository,
    strategy_id: str,
    *,
    champion_ids: list[str] | None = None,
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
        "current_champion_ids": sorted(champion_ids or []),
    }


def _insert_dispatched_effect(
    db_path,
    *,
    target: dict,
    runtime_generation: str = "runtime-2",
    effect_status: str = "dispatched",
    approval_status: str = "reserved",
) -> str:
    effect_id = "effect-adopt-1"
    approval_id = "approval-adopt-1"
    reservation_id = "reservation-adopt-1"
    target_json = json.dumps(target, ensure_ascii=False, sort_keys=True)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO local_principals(
                id, kind, display_name, session_token_hash, status,
                created_at, last_seen_at, expires_at
            ) VALUES (?, 'local_session', ?, ?, 'active', ?, ?, ?)
            """,
            ("principal-1", "本地策略人员", "token-hash-1", _STAMP, _STAMP, _EXPIRES),
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
                "decision-1",
                target["task_id"],
                "plan-1",
                3,
                "step-adopt",
                "strategy.adopt_strategy@1.0.0",
                "principal-1",
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
                "decision-1",
                target["task_id"],
                "plan-1",
                3,
                "step-adopt",
                "strategy.adopt_strategy@1.0.0",
                "principal-1",
                "委员会批准",
                "manifest-hash",
                "policy-hash",
                "input-hash",
                "evidence-hash",
                target_json,
                "nonce-1",
                approval_status,
                _STAMP,
                _EXPIRES,
                _STAMP if approval_status == "reserved" else None,
                reservation_id if approval_status == "reserved" else None,
            ),
        )
        conn.execute(
            """
            INSERT INTO effect_executions(
                id, approval_id, reservation_id, runtime_generation, status,
                prepared_at, dispatched_at, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                effect_id,
                approval_id,
                reservation_id,
                runtime_generation,
                effect_status,
                _STAMP,
                _STAMP if effect_status == "dispatched" else None,
            ),
        )
    return effect_id


def _ledger_states(db_path) -> tuple[str, str]:
    with connect(db_path) as conn:
        effect = conn.execute(
            "SELECT status FROM effect_executions WHERE id = 'effect-adopt-1'"
        ).fetchone()
        approval = conn.execute(
            "SELECT status FROM approval_records WHERE id = 'approval-adopt-1'"
        ).fetchone()
    return str(effect["status"]), str(approval["status"])


def test_governed_adoption_commits_lifecycle_and_effect_receipt_atomically(tmp_path):
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
        audit=_audit(champion.id),
    )
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id, champion_ids=[champion.id]),
    )

    result = repo.adopt_strategy_with_audit(
        challenger.id,
        reason="committee promotes challenger",
        audit=_audit(challenger.id),
        effect_execution_id=effect_id,
        runtime_generation="runtime-2",
    )

    assert result == {"version": 1, "retired_strategy_ids": [champion.id]}
    assert repo.get_strategy_meta(champion.id)["status"] == "retired"
    assert repo.get_strategy_meta(challenger.id)["status"] == "adopted"
    assert _ledger_states(db_path) == ("committed", "consumed")


@pytest.mark.parametrize(
    "target_change",
    [
        {"id": "other-strategy"},
        {"expected_status": "adopted"},
        {"result_status": "retired"},
        {"version": 99},
        {"task_id": "other-task"},
        {"strategy_type": "collection"},
        {"strategy_spec_hash": "stale-reviewed-strategy-spec"},
        {"current_champion_ids": []},
    ],
)
def test_governed_adoption_rejects_stale_or_mismatched_target_without_mutation(
    tmp_path,
    target_change,
):
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
        audit=_audit(champion.id),
    )
    target = _effect_target(repo, challenger.id, champion_ids=[champion.id])
    target.update(target_change)
    effect_id = _insert_dispatched_effect(db_path, target=target)

    with pytest.raises(ConflictError, match="授权|策略效果"):
        repo.adopt_strategy_with_audit(
            challenger.id,
            reason="committee promotes challenger",
            audit=_audit(challenger.id),
            effect_execution_id=effect_id,
            runtime_generation="runtime-2",
        )

    assert repo.get_strategy_meta(champion.id)["status"] == "adopted"
    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"
    assert _ledger_states(db_path) == ("dispatched", "reserved")


def test_governed_adoption_rejects_canonical_dsl_changed_after_approval(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id),
    )

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT dsl_json FROM strategies WHERE id = ?",
            (challenger.id,),
        ).fetchone()
        payload = json.loads(row["dsl_json"])
        payload["rules"][0]["condition"]["value"] = 700
        conn.execute(
            "UPDATE strategies SET dsl_json = ? WHERE id = ?",
            (
                json.dumps(payload, separators=(",", ":")),
                challenger.id,
            ),
        )

    with pytest.raises(ConflictError, match="spec|策略效果|授权"):
        repo.adopt_strategy_with_audit(
            challenger.id,
            reason="committee promotes challenger",
            audit=_audit(challenger.id),
            effect_execution_id=effect_id,
            runtime_generation="runtime-2",
        )

    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"
    assert _ledger_states(db_path) == ("dispatched", "reserved")


@pytest.mark.parametrize(
    ("effect_status", "approval_status", "runtime_generation"),
    [
        ("prepared", "reserved", "runtime-2"),
        ("dispatched", "issued", "runtime-2"),
        ("dispatched", "reserved", "fenced-runtime-1"),
    ],
)
def test_governed_adoption_requires_live_dispatched_reservation_and_generation(
    tmp_path,
    effect_status,
    approval_status,
    runtime_generation,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id),
        effect_status=effect_status,
        approval_status=approval_status,
    )

    with pytest.raises(ConflictError, match="授权|策略效果"):
        repo.adopt_strategy_with_audit(
            challenger.id,
            reason="committee promotes challenger",
            audit=_audit(challenger.id),
            effect_execution_id=effect_id,
            runtime_generation=runtime_generation,
        )

    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"
    assert _ledger_states(db_path) == (effect_status, approval_status)


def test_governed_adoption_replay_is_fenced_after_first_commit(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id),
    )
    kwargs = {
        "reason": "committee promotes challenger",
        "audit": _audit(challenger.id),
        "effect_execution_id": effect_id,
        "runtime_generation": "runtime-2",
    }

    repo.adopt_strategy_with_audit(challenger.id, **kwargs)
    with pytest.raises(ConflictError, match="策略效果|授权"):
        repo.adopt_strategy_with_audit(challenger.id, **kwargs)

    assert repo.get_strategy_meta(challenger.id)["status"] == "adopted"
    assert _ledger_states(db_path) == ("committed", "consumed")
    audits = PluginRepository(db_path).list_audit(kind="strategy.adopt")
    assert [row["target_ref"] for row in audits].count(challenger.id) == 1


def test_effect_receipt_commit_failure_rolls_back_strategy_and_audit(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    repo = StrategyRepository(db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    effect_id = _insert_dispatched_effect(
        db_path,
        target=_effect_target(repo, challenger.id),
    )
    monkeypatch.setattr(
        strategy_repository_module,
        "_commit_strategy_effect_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ledger unavailable")),
    )

    with pytest.raises(RuntimeError, match="ledger unavailable"):
        repo.adopt_strategy_with_audit(
            challenger.id,
            reason="committee promotes challenger",
            audit=_audit(challenger.id),
            effect_execution_id=effect_id,
            runtime_generation="runtime-2",
        )

    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"
    assert _ledger_states(db_path) == ("dispatched", "reserved")
    audits = PluginRepository(db_path).list_audit(kind="strategy.adopt")
    assert [row["target_ref"] for row in audits].count(challenger.id) == 0


def test_tool_lifecycle_commit_survives_artifact_failure_and_cannot_replay(
    tmp_path,
    monkeypatch,
):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = StrategyRepository(settings.db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    repo.save_backtest("backtest-1", challenger.id, "dataset-1", _backtest(challenger.id))
    effect_id = _insert_dispatched_effect(
        settings.db_path,
        target=_effect_target(repo, challenger.id),
    )
    ctx = SimpleNamespace(
        task_id="task-1",
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        seed=None,
        effect_execution_id=effect_id,
        runtime_generation="runtime-2",
    )
    inputs = {
        "strategy_id": challenger.id,
        "backtest_id": "backtest-1",
        "adoption_reason": "committee promotes challenger",
    }
    monkeypatch.setattr(
        strategy_tools,
        "decision_table_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        strategy_tools.tool_adopt_strategy(inputs, ctx)

    assert repo.get_strategy_meta(challenger.id)["status"] == "adopted"
    assert _ledger_states(settings.db_path) == ("committed", "consumed")
    with pytest.raises(ConflictError, match="策略效果|授权"):
        strategy_tools.tool_adopt_strategy(inputs, ctx)


def test_tool_rejects_partial_protected_execution_metadata_before_adoption(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = StrategyRepository(settings.db_path)
    challenger = _strategy("challenger", 625)
    repo.create_strategy("task-1", challenger)
    repo.save_backtest("backtest-1", challenger.id, "dataset-1", _backtest(challenger.id))
    ctx = SimpleNamespace(
        task_id="task-1",
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        seed=None,
        effect_execution_id="effect-only",
        runtime_generation=None,
    )

    with pytest.raises(StrategyError, match="治理执行元数据"):
        strategy_tools.tool_adopt_strategy(
            {
                "strategy_id": challenger.id,
                "backtest_id": "backtest-1",
                "adoption_reason": "committee promotes challenger",
            },
            ctx,
        )

    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"


def test_tool_rejects_typed_adoption_when_approved_bad_rate_is_undefined(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    repo = StrategyRepository(settings.db_path)
    challenger = _strategy("empty approval evidence", 625)
    repo.create_strategy("task-1", challenger)
    result = run_typed_backtest(
        pd.DataFrame({"score": [500, 600], "bad": [1, 0]}),
        challenger.spec,
        target_col="bad",
        strategy_id=challenger.id,
    )
    assert result.metrics["approve_bad_rate"] is None
    repo.save_backtest(
        "typed-empty-approval",
        challenger.id,
        "dataset-1",
        result,
    )
    ctx = SimpleNamespace(
        task_id="task-1",
        workspace=settings.workspace,
        datasets_root=settings.datasets_dir,
        seed=None,
        effect_execution_id=None,
        runtime_generation=None,
    )

    with pytest.raises(StrategyError, match="approved bad rate is undefined"):
        strategy_tools.tool_adopt_strategy(
            {
                "strategy_id": challenger.id,
                "backtest_id": "typed-empty-approval",
                "adoption_reason": "committee requires measurable approved evidence",
            },
            ctx,
        )

    assert repo.get_strategy_meta(challenger.id)["status"] == "draft"
