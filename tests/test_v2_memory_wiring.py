"""MEM-1 / MEM-4: bidirectional wiring between the V2 plan-conversation driver
(JOIN/FEATURE/MODELING) and the agent memory subsystem.

Covers the regression matrix requested in the review fix:
  a) experiment/join completion -> a model_experience/join_experience memory
     entry lands with metrics and source_task_id;
  b) with prior history, a gate payload carries a read-only anchor section and
     a use audit event is recorded;
  c) with no history, the gate/decide_gate prompt is unaffected (no anchor
     section at all — not even an empty one);
  d) with the memory policy OFF, nothing is written and nothing is read;
  e) a field_convention hit reorders sample_setup's target/split candidates
     and annotates the proposal notes ("历史任务曾用"/来自记忆).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.memory_bridge import (
    build_memory_anchor,
    capture_agent_memory_for_driver_done,
    fetch_field_convention_hints,
)
from marvis.agent.sample_setup import detect_setup
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.db import init_db
from marvis.domain import (
    TASK_TYPE_DATA_JOIN,
    TASK_TYPE_FEATURE_ANALYSIS,
    TASK_TYPE_MODELING,
    TASK_TYPE_STRATEGY,
    TASK_TYPE_VINTAGE,
    TaskRecord,
    TaskStatus,
)
from marvis.memory_policy import MemoryPolicySettings, save_memory_policy
from marvis.settings import Settings, build_settings


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(tmp_path))


def _task_record(**overrides) -> TaskRecord:
    base = dict(
        id="task-1",
        model_name="A卡",
        model_version="V1",
        validator="qa",
        source_dir="",
        algorithm="",
        run_mode="agent",
        target_col="",
        score_col="",
        split_col="",
        time_col="",
        feature_columns=[],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.SCANNED,
        status_message="",
        created_at="",
        updated_at="",
        task_type=TASK_TYPE_MODELING,
        target_type="binary",
    )
    base.update(overrides)
    return TaskRecord(**base)


# -- (a) write side: modeling done -> model_experience -----------------------

def test_capture_agent_memory_writes_model_experience_on_modeling_done(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id="task-model-1", task_type=TASK_TYPE_MODELING, target_type="binary")
    done_metadata = {
        "model_delivery": {
            "source_tool": "post_training_action",
            "recipe": "lgb",
            "artifact_id": "art-lgb-1",
            "metrics": {
                "oot_ks": 0.31,
                "test_ks": 0.29,
                "oot_auc": 0.71,
                "psi_oot_vs_train": 0.06,
                "feature_count": 18,
            },
        }
    }

    capture_agent_memory_for_driver_done(
        settings, task, done_message_metadata=done_metadata
    )

    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="model_experience")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.payload["model_name"] == "lgb"
    assert entry.payload["ks"] == 0.31
    assert entry.payload["auc"] == 0.71
    assert entry.payload["source_task_id"] == "task-model-1"
    assert entry.source_task_id == "task-model-1"


def test_capture_agent_memory_writes_join_experience_on_join_done(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id="task-join-1", task_type=TASK_TYPE_DATA_JOIN, model_name="拼接任务")
    content = "**拼接执行完成**:结果数据集 `ds-1`,锚行 100 → 拼接后 100 行(1:1 保持 ✓)"
    metadata = {
        "tables": [
            {
                "title": "各特征表贡献",
                "columns": ["特征表", "命中率", "新增列", "新列缺失率", "去重策略"],
                "rows": [
                    ["feature-a", "0.9600", "3", "0.0100", "无"],
                    ["feature-b", "0.8800", "2", "0.0200", "无"],
                ],
            }
        ]
    }

    capture_agent_memory_for_driver_done(
        settings, task, done_message_content=content, done_message_metadata=metadata
    )

    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="join_experience")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.payload["anchor_rows"] == 100
    assert entry.payload["joined_rows"] == 100
    assert entry.payload["feature_table_count"] == 2
    assert entry.payload["match_rate"] == pytest.approx((0.96 + 0.88) / 2, rel=1e-6)
    assert entry.source_task_id == "task-join-1"


# -- (a2) write side: strategy adoption -> strategy_experience ---------------

def test_capture_agent_memory_writes_strategy_experience_on_adoption(tmp_path: Path):
    """S2: strategy_experience is assembled straight from the persisted adopted
    strategy + its latest backtest (StrategyRepository), not from
    done_message_metadata -- the STRATEGY_DEVELOPMENT template's terminal step is
    render_strategy_doc, which carries no bad_rate/approval_rate/profit."""
    from marvis.packs.strategy import BacktestResult, build_strategy
    from marvis.repositories.strategy import StrategyRepository

    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id="task-strategy-1", task_type=TASK_TYPE_STRATEGY, model_name="策略开发")

    strategies = StrategyRepository(settings.db_path)
    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
        description="S2 memory capture fixture",
    )
    strategies.create_strategy(task.id, strategy)
    strategies.save_backtest(
        "backtest-1",
        strategy.id,
        "dataset-1",
        BacktestResult(
            strategy_id=strategy.id,
            approval_rate=0.8,
            approved_count=80,
            approved_bad_rate=0.03,
            rejected_bad_rate=0.25,
            expected_profit=1500.0,
            swap_in_count=0,
            swap_out_count=0,
            swap_in_bad_rate=0.0,
            swap_out_bad_rate=0.0,
            by_segment=(),
        ),
    )
    strategies.adopt_strategy_with_audit(
        strategy.id, reason="test adoption", audit={"kind": "strategy.adopt", "target_ref": strategy.id}
    )

    capture_agent_memory_for_driver_done(
        settings, task,
        done_message_content="策略文档已生成",
        done_message_metadata={},
    )

    store = AgentMemoryStore(settings.db_path)
    entries = store.list_entries(memory_type="strategy_experience")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.payload["strategy_type"] == "approval"
    assert entry.payload["approval_rate"] == 0.8
    assert entry.payload["approved_bad_rate"] == 0.03
    assert entry.payload["expected_profit"] == 1500.0
    assert entry.payload["source_task_id"] == "task-strategy-1"
    assert "score < 600" in entry.payload["cutoff_summary"]


def test_capture_agent_memory_strategy_is_noop_when_nothing_adopted(tmp_path: Path):
    """A strategy task whose plan hasn't reached adoption (or used the
    lightweight strategy_analysis entry, which never calls adopt_strategy)
    writes nothing -- no exception, no entry."""
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id="task-strategy-noop", task_type=TASK_TYPE_STRATEGY)

    capture_agent_memory_for_driver_done(
        settings, task, done_message_content="", done_message_metadata={}
    )

    store = AgentMemoryStore(settings.db_path)
    assert store.list_entries(memory_type="strategy_experience") == []


def test_capture_agent_memory_strategy_gated_by_auto_distill(tmp_path: Path):
    """Same auto_distill gate as model/join capture (ARCH-4 kwargs fix: strategy
    now passes settings/task into append_driver_messages, so it must honor the
    same single V2-driver gate memory_bridge already enforces)."""
    from marvis.packs.strategy import BacktestResult, build_strategy
    from marvis.repositories.strategy import StrategyRepository

    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    save_memory_policy(
        settings.workspace,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )
    task = _task_record(id="task-strategy-off", task_type=TASK_TYPE_STRATEGY)
    strategies = StrategyRepository(settings.db_path)
    strategy = build_strategy(
        "approval", [{"condition": "score < 600", "decision": "reject"}],
        score_col="score", default_decision="approve", description="gated",
    )
    strategies.create_strategy(task.id, strategy)
    strategies.save_backtest(
        "backtest-off", strategy.id, "dataset-1",
        BacktestResult(
            strategy_id=strategy.id, approval_rate=0.8, approved_count=80,
            approved_bad_rate=0.03, rejected_bad_rate=0.25, expected_profit=1500.0,
            swap_in_count=0, swap_out_count=0, swap_in_bad_rate=0.0, swap_out_bad_rate=0.0,
            by_segment=(),
        ),
    )
    strategies.adopt_strategy_with_audit(
        strategy.id, reason="gated", audit={"kind": "strategy.adopt", "target_ref": strategy.id}
    )

    capture_agent_memory_for_driver_done(settings, task, done_message_metadata={})

    store = AgentMemoryStore(settings.db_path)
    assert store.list_entries(memory_type="strategy_experience") == []


# -- ARCH-4 kwargs fix regression: feature/vintage now pass settings/task too,
# but neither has an extractor wired -- must be a silent, exception-free no-op.

@pytest.mark.parametrize("task_type", [TASK_TYPE_FEATURE_ANALYSIS, TASK_TYPE_VINTAGE])
def test_capture_agent_memory_feature_and_vintage_are_noop_not_errors(tmp_path: Path, task_type):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id=f"task-{task_type}-1", task_type=task_type)

    # Must not raise even though there is no extractor for these two types yet.
    capture_agent_memory_for_driver_done(
        settings, task, done_message_content="done", done_message_metadata={"tables": []}
    )

    store = AgentMemoryStore(settings.db_path)
    assert store.list_entries(memory_type="model_experience") == []
    assert store.list_entries(memory_type="join_experience") == []
    assert store.list_entries(memory_type="strategy_experience") == []


def test_agent_mode_autodrive_join_completion_writes_join_experience_via_real_api(
    tmp_path: Path, monkeypatch
):
    """End-to-end: drive a real data_join task through AUTO mode and confirm the
    memory bridge actually fires from the real turn_handlers.append_driver_messages
    call site (not just the unit-level helper above)."""

    class _FakeLLM:
        def complete(self, **kwargs):
            return json.dumps({"action": "confirm", "reason": "命中率正常,继续"})

    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda request, task, payload: _FakeLLM(),
    )
    client = _client(tmp_path)
    src = tmp_path / "join_material"
    src.mkdir(parents=True, exist_ok=True)
    n = 50
    phones = [f"138{i:08d}" for i in range(n)]
    pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(n)]}).to_parquet(
        src / "sample.parquet"
    )
    pd.DataFrame(
        {
            "phone_md5": [hashlib.md5(p.encode()).hexdigest() for p in phones],
            "balance": list(range(n)),
        }
    ).to_parquet(src / "features.parquet")

    task_id = client.post(
        "/api/tasks",
        json={
            "model_name": "拼接记忆验证",
            "validator": "qa",
            "source_dir": str(src),
            "task_type": "data_join",
            "run_mode": "agent",
        },
    ).json()["id"]

    resp = client.post(f"/api/tasks/{task_id}/agent/start", json={"acceptance_mode": "auto_accept"})
    assert resp.status_code == 202, resp.text

    store = AgentMemoryStore(client.app.state.settings.db_path)
    entries = store.list_entries(memory_type="join_experience", source_task_id=task_id)
    assert len(entries) == 1
    assert entries[0].payload["feature_table_count"] == 1


# -- (d) write side gated by auto_distill -------------------------------------

def test_capture_agent_memory_is_noop_when_auto_distill_disabled(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    save_memory_policy(
        settings.workspace,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )
    task = _task_record(id="task-model-off", task_type=TASK_TYPE_MODELING)
    done_metadata = {
        "model_delivery": {
            "source_tool": "select_experiment",
            "recipe": "lgb",
            "metrics": {"oot_ks": 0.31, "oot_auc": 0.71},
        }
    }

    capture_agent_memory_for_driver_done(settings, task, done_message_metadata=done_metadata)

    store = AgentMemoryStore(settings.db_path)
    assert store.list_entries(memory_type="model_experience") == []


# -- (b)/(c) read side: gate anchor construction ------------------------------

def _seed_history_entry(settings: Settings, *, ks: float, auc: float, scope: str) -> None:
    store = AgentMemoryStore(settings.db_path)
    store.create(
        MemoryCandidate(
            memory_type="model_experience",
            summary=f"lgb 历史实验 KS={ks}",
            payload={
                "ks": ks,
                "auc": auc,
                "psi": 0.05,
                "month": "未标注",
                "channel": "未标注",
                "model_name": "lgb",
                "model_version": "v-hist",
                "scope": scope,
                "source_task_id": "task-history-1",
                "important_feature_sources": ["18"],
            },
            source_task_id="task-history-1",
            confidence="high",
        )
    )


def test_build_memory_anchor_returns_lines_when_history_exists(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    scope = "binary:binary:A卡"
    _seed_history_entry(settings, ks=0.30, auc=0.70, scope=scope)
    task = _task_record(id="task-current", task_type=TASK_TYPE_MODELING, model_name="A卡")
    gate_metadata = {
        "model_delivery": {"source_tool": "select_experiment", "recipe": "lgb", "metrics": {}}
    }

    anchor = build_memory_anchor(settings, task, gate_metadata=gate_metadata)

    assert anchor is not None
    assert len(anchor["lines"]) == 1
    assert "KS=0.3" in anchor["lines"][0]
    assert all(len(line) <= 120 for line in anchor["lines"])
    assert anchor["references"][0]["use_reason"] == "gate_memory_anchor"


def test_build_memory_anchor_none_when_no_history(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    task = _task_record(id="task-current", task_type=TASK_TYPE_MODELING, model_name="A卡")
    gate_metadata = {
        "model_delivery": {"source_tool": "select_experiment", "recipe": "lgb", "metrics": {}}
    }

    assert build_memory_anchor(settings, task, gate_metadata=gate_metadata) is None


def test_build_memory_anchor_none_for_non_delivery_gate(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    _seed_history_entry(settings, ks=0.30, auc=0.70, scope="binary:binary:A卡")
    task = _task_record(id="task-current", task_type=TASK_TYPE_MODELING, model_name="A卡")

    # A gate with no model_delivery / modeling_setup payload at all (e.g. a screen
    # gate) must get no anchor — regression (c): non-modeling-delivery gates are
    # completely unaffected by memory, even when comparable history exists.
    assert build_memory_anchor(settings, task, gate_metadata={"screen": {}}) is None


def test_build_memory_anchor_none_when_reference_cross_task_disabled(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    save_memory_policy(
        settings.workspace,
        MemoryPolicySettings(reference_cross_task=False, auto_distill=True),
    )
    _seed_history_entry(settings, ks=0.30, auc=0.70, scope="binary:binary:A卡")
    task = _task_record(id="task-current", task_type=TASK_TYPE_MODELING, model_name="A卡")
    gate_metadata = {
        "model_delivery": {"source_tool": "select_experiment", "recipe": "lgb", "metrics": {}}
    }

    assert build_memory_anchor(settings, task, gate_metadata=gate_metadata) is None


def test_auto_drive_format_gate_renders_memory_anchor_section():
    from marvis.agent.auto_drive import _format_gate

    gate_without_anchor = {"content": "确认结果", "metadata": {}}
    gate_with_anchor = {
        "content": "确认结果",
        "metadata": {"memory_anchor": ["lgb：KS=0.30、AUC=0.70（来自历史任务 task-history-1，confidence=high）"]},
    }

    rendered_without = _format_gate(gate_without_anchor)
    rendered_with = _format_gate(gate_with_anchor)

    assert "历史同类实验" not in rendered_without
    assert "历史同类实验" in rendered_with
    assert "仅供参考" in rendered_with
    assert "task-history-1" in rendered_with


def test_agent_autodrive_decision_message_carries_memory_references_and_audits_use(
    tmp_path: Path, monkeypatch
):
    """End-to-end (b): a real AUTO-mode modeling gate with comparable history in
    the store produces a decision message whose metadata carries
    memory_references, and the store records a 'use' audit event for it."""
    from marvis.agent.turn_handlers import DRIVER_TURN_FUNCS, agent_autodrive_turn

    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    scope = "binary:binary:A卡"
    _seed_history_entry(settings, ks=0.30, auc=0.70, scope=scope)

    class _TokenRepo:
        def __init__(self):
            self.messages = [
                {
                    "role": "assistant",
                    "metadata": {
                        "kind": "gate",
                        "step_id": "gate-1",
                        "model_delivery": {
                            "source_tool": "select_experiment",
                            "recipe": "lgb",
                            "metrics": {},
                        },
                    },
                }
            ]

        def list_agent_messages(self, task_id):
            return self.messages

        def add_agent_message(self, task_id, *, role, stage, content, metadata):
            message = {
                "id": f"msg-{len(self.messages)}",
                "role": role,
                "stage": stage,
                "content": content,
                "metadata": metadata,
            }
            self.messages.append(message)
            return message

    class _FakeLLM:
        def complete(self, **kwargs):
            return json.dumps({"action": "confirm", "reason": "结果正常,继续"})

    calls = []

    def fake_turn(runtime, repo, task, **kwargs):
        calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setitem(DRIVER_TURN_FUNCS, TASK_TYPE_MODELING, fake_turn)
    repo = _TokenRepo()
    task = _task_record(id="task-current", task_type=TASK_TYPE_MODELING, model_name="A卡")
    runtime = SimpleNamespace(settings=settings)

    agent_autodrive_turn(runtime, repo, task, client=_FakeLLM())

    decision_message = next(
        m for m in repo.messages if m.get("metadata", {}).get("intent") == "agent_decision"
    )
    references = decision_message["metadata"].get("memory_references")
    assert references
    memory_id = references[0]["id"]

    store = AgentMemoryStore(settings.db_path)
    events = store.list_events(memory_id)
    assert any(event["event_type"] == "use" for event in events)


# -- (e) MEM-4: field_convention hints reorder setup candidates --------------

def test_fetch_field_convention_hints_reads_matching_field_convention_memory(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    store = AgentMemoryStore(settings.db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag，样本分组字段=split",
            payload={"target_col": "bad_flag", "split_col": "split"},
            source_task_id="task-prior",
            confidence="high",
        )
    )

    hints = fetch_field_convention_hints(settings, keywords=("sample.parquet",))

    assert hints == {"target_col": "bad_flag", "split_col": "split"}


def test_fetch_field_convention_hints_none_when_reference_cross_task_disabled(tmp_path: Path):
    settings = build_settings(tmp_path)
    init_db(settings.db_path)
    save_memory_policy(
        settings.workspace,
        MemoryPolicySettings(reference_cross_task=False, auto_distill=True),
    )
    store = AgentMemoryStore(settings.db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
            source_task_id="task-prior",
            confidence="high",
        )
    )

    assert fetch_field_convention_hints(settings, keywords=("sample.parquet",)) is None


def _write_sample_parquet(path: Path) -> None:
    n = 200
    frame = pd.DataFrame(
        {
            "bad_flag": [i % 2 for i in range(n)],
            "target": [1 - (i % 2) for i in range(n)],  # a second binary column, worse priority token
            "split": ["train"] * 140 + ["test"] * 40 + ["oot"] * 20,
            "amount": list(range(n)),
        }
    )
    frame.to_parquet(path)


def test_detect_setup_field_hints_reorder_target_and_annotate_notes(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    _write_sample_parquet(path)
    backend = DataBackend(tmp_path)

    baseline = detect_setup(backend, path)
    hinted = detect_setup(backend, path, field_hints={"target_col": "bad_flag", "split_col": "split"})

    # The deterministic heuristics alone already pick "target" over "bad_flag"
    # (higher _TARGET_PRIORITY rank), so this dataset is a genuine case where a
    # memory hint changes the winning candidate versus the no-hint baseline.
    assert baseline.target_col == "target"
    assert hinted.target_col == "bad_flag"
    assert hinted.split_col == "split"
    assert set(hinted.memory_matched_fields) == {"target_col", "split_col"}
    assert any("历史任务口径一致" in note for note in hinted.notes)


def test_detect_setup_field_hints_noop_when_absent(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    _write_sample_parquet(path)
    backend = DataBackend(tmp_path)

    without_hints = detect_setup(backend, path)
    with_none_hints = detect_setup(backend, path, field_hints=None)

    assert without_hints.target_col == with_none_hints.target_col
    assert without_hints.split_col == with_none_hints.split_col
    assert without_hints.notes == with_none_hints.notes
    assert with_none_hints.memory_matched_fields == []


def test_detect_setup_field_hints_ignored_when_configured_target_set(tmp_path: Path):
    path = tmp_path / "sample.parquet"
    _write_sample_parquet(path)
    backend = DataBackend(tmp_path)

    # An explicit configured_target always wins over a memory hint (INV-3: explicit
    # user/task configuration is never overridden by a memory suggestion).
    result = detect_setup(
        backend,
        path,
        configured_target="target",
        field_hints={"target_col": "bad_flag"},
    )

    assert result.target_col == "target"
    assert result.memory_matched_fields == []
