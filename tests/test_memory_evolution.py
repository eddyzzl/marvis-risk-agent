from marvis.agent_memory.distillation import new_distillation
from marvis.agent_memory.evolution import EvolutionManager
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db


def test_evolution_creates_and_updates_non_meaningful_support(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    manager = EvolutionManager(store)
    first = new_distillation(
        category="field_convention",
        scope_key="field_convention:target_col",
        distilled_summary="目标字段常叫 bad_flag。",
        structured={"fields": {"target_col": ["bad_flag"]}},
        source_memory_ids=("mem-1", "mem-2"),
        support_count=2,
    )
    same = new_distillation(
        category="field_convention",
        scope_key="field_convention:target_col",
        distilled_summary="目标字段常叫 bad_flag。",
        structured={"fields": {"target_col": ["bad_flag"]}},
        source_memory_ids=("mem-1", "mem-2", "mem-3"),
        support_count=3,
    )

    created = manager.upsert_with_evolution(first)
    updated = manager.upsert_with_evolution(same)

    assert created.id == updated.id
    assert updated.support_count == 3
    assert updated.confidence == "medium"
    assert store.get_active_distillation("field_convention:target_col").id == created.id


def test_evolution_supersedes_meaningful_update_and_rollback_restores_predecessor(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    manager = EvolutionManager(store)
    old = manager.upsert_with_evolution(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:score_col",
            distilled_summary="分数字段常叫 pred。",
            structured={"fields": {"score_col": ["pred"]}},
            source_memory_ids=("mem-1", "mem-2"),
            support_count=2,
        )
    )
    new = manager.upsert_with_evolution(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:score_col",
            distilled_summary="分数字段常叫 pred 或 score。",
            structured={"fields": {"score_col": ["pred", "score"]}},
            source_memory_ids=("mem-1", "mem-2", "mem-3", "mem-4"),
            support_count=4,
        )
    )

    assert new.id != old.id
    assert store.get_distillation(old.id).superseded_by == new.id
    assert store.get_active_distillation("field_convention:score_col").id == new.id

    manager.rollback(new.id)

    assert store.get_distillation(new.id).superseded_by is None
    assert store.get_active_distillation("field_convention:score_col").id == old.id
