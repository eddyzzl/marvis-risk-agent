from fastapi.testclient import TestClient

from marvis.agent_memory.consolidation import (
    CONSOLIDATION_TRIGGERS,
    ConsolidationScheduler,
)
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.app import create_app
from marvis.db import init_db
from marvis.memory_policy import MemoryPolicySettings, load_memory_policy, save_memory_policy


def test_consolidation_event_distills_and_throttles(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
        )
    )
    scheduler = ConsolidationScheduler(
        DistillationEngine(store),
        EvolutionManager(store),
        store,
        async_mode=False,
        throttle_seconds=3600,
    )

    scheduler.on_event("memory.after_save", {"memory_type": "field_convention"})
    first = store.get_active_distillation("field_convention:target_col")
    scheduler.on_event("memory.after_save", {"memory_type": "field_convention"})
    second = store.get_active_distillation("field_convention:target_col")

    assert first is not None
    assert second is not None
    assert second.id == first.id
    assert store.last_consolidated_at("field_convention") is not None


def test_consolidation_event_respects_auto_distill_policy(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    save_memory_policy(
        tmp_path,
        MemoryPolicySettings(reference_cross_task=True, auto_distill=False),
    )
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
        )
    )
    scheduler = ConsolidationScheduler(
        DistillationEngine(store),
        EvolutionManager(store),
        store,
        async_mode=False,
        auto_enabled=lambda: load_memory_policy(tmp_path).auto_distill,
    )

    scheduler.on_event("memory.after_save", {"memory_type": "field_convention"})

    assert store.get_active_distillation("field_convention:target_col") is None
    assert store.last_consolidated_at("field_convention") is None
    assert scheduler.consolidate_all(["field_convention"]) == {
        "field_convention": {"count": 1, "errors": 0}
    }
    assert store.get_active_distillation("field_convention:target_col") is not None


def test_consolidate_all_returns_counts_and_failures_do_not_escape(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    scheduler = ConsolidationScheduler(
        _BrokenDistiller(),
        EvolutionManager(store),
        store,
        async_mode=False,
    )

    scheduler.on_event("memory.after_save", {"memory_type": "field_convention"})
    assert scheduler.consolidate_all(["field_convention"]) == {
        "field_convention": {"count": 0, "errors": 1}
    }


def test_app_registers_memory_consolidation_hooks(tmp_path):
    client = TestClient(create_app(tmp_path))

    scheduler = client.app.state.memory_consolidation_scheduler
    dispatcher = client.app.state.hook_dispatcher

    assert isinstance(scheduler, ConsolidationScheduler)
    assert {
        event: dispatcher.listener_count(event)
        for event in CONSOLIDATION_TRIGGERS
    } == {event: 1 for event in CONSOLIDATION_TRIGGERS}


class _BrokenDistiller:
    def distill_category(self, _category):
        raise RuntimeError("boom")
