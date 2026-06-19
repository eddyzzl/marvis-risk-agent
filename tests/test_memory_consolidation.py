from marvis.agent_memory.consolidation import ConsolidationScheduler
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db


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
    assert scheduler.consolidate_all(["field_convention"]) == {"field_convention": 0}


class _BrokenDistiller:
    def distill_category(self, _category):
        raise RuntimeError("boom")
