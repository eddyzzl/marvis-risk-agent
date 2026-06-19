from marvis.agent_memory.distillation import (
    MAX_DISTILLED_SUMMARY_CHARS,
    confidence_from_support,
    new_distillation,
)
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db


def test_memory_distillation_contract_bounds_summary_and_confidence():
    distillation = new_distillation(
        category="FIELD_CONVENTION",
        scope_key="field_convention:idcard:机构A",
        distilled_summary="x" * (MAX_DISTILLED_SUMMARY_CHARS + 50),
        structured={"field_role": "idcard", "aliases": ["id_md5"]},
        source_memory_ids=("mem-1", "mem-2"),
        support_count=2,
    )

    assert distillation.category == "field_convention"
    assert len(distillation.distilled_summary) == MAX_DISTILLED_SUMMARY_CHARS
    assert distillation.confidence == "medium"
    assert confidence_from_support(1) == "low"
    assert confidence_from_support(2) == "medium"
    assert confidence_from_support(4) == "high"


def test_store_distillation_crud_and_search_order(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    low = new_distillation(
        category="field_convention",
        scope_key="field_convention:idcard:机构A",
        distilled_summary="机构A 身份证字段常叫 id_md5。",
        structured={"field_role": "idcard", "aliases": ["id_md5"]},
        source_memory_ids=("mem-1",),
        support_count=1,
    )
    high = new_distillation(
        category="validation_pitfall",
        scope_key="validation_pitfall:notebook",
        distilled_summary="Notebook 失败常见原因是缺少依赖包。",
        structured={"pitfall_type": "dependency", "fixes": ["检查依赖"]},
        source_memory_ids=("mem-2", "mem-3", "mem-4", "mem-5"),
        support_count=4,
    )

    created_low = store.create_distillation(low)
    created_high = store.create_distillation(high)
    fetched = store.get_distillation(created_low.id)
    searched = store.search_distillations({"keywords": ["依赖"]}, limit=3)
    all_ranked = store.search_distillations({}, limit=3)

    assert fetched.scope_key == "field_convention:idcard:机构A"
    assert fetched.structured["aliases"] == ["id_md5"]
    assert fetched.source_memory_ids == ("mem-1",)
    assert searched == [created_high]
    assert all_ranked[0].id == created_high.id
    assert store.list_distillations(category="field_convention") == [created_low]


def test_store_supersede_support_status_and_consolidation_state(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    old = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:score",
            distilled_summary="分数字段常叫 pred。",
            structured={"field_role": "score", "aliases": ["pred"]},
            source_memory_ids=("mem-1",),
            support_count=1,
        )
    )
    new = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:score",
            distilled_summary="分数字段常叫 pred 或 score。",
            structured={"field_role": "score", "aliases": ["pred", "score"]},
            source_memory_ids=("mem-1", "mem-2"),
            support_count=2,
        )
    )

    store.set_superseded(old.id, by=new.id)
    updated = store.update_distillation_support(new.id, 4)
    predecessor = store.find_superseded_by(new.id)
    active = store.get_active_distillation("field_convention:score")
    store.mark_consolidated("field_convention", at="2026-06-19T10:00:00+00:00")
    rolled_back = store.set_status_distillation(new.id, "rolled_back")

    assert updated.confidence == "high"
    assert predecessor is not None
    assert predecessor.id == old.id
    assert active is not None
    assert active.id == new.id
    assert store.last_consolidated_at("field_convention") == "2026-06-19T10:00:00+00:00"
    assert rolled_back.id == new.id
    assert store.get_active_distillation("field_convention:score") is None

    store.clear_superseded(old.id)
    assert store.get_distillation(old.id).superseded_by is None
