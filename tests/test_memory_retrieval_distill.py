from marvis.agent_memory.distillation import new_distillation
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.prompting import memory_references, normalize_memory_context
from marvis.agent_memory.retrieval import retrieve_with_distillations
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db


def test_retrieve_with_distillations_prefers_high_confidence_and_backfills_raw(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="A卡验证里坏样本字段常用 bad_flag。",
            payload={"target_col": "bad_flag"},
            source_task_id="task-raw",
            confidence="high",
        )
    )
    high = store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:target_col",
            distilled_summary="A卡坏样本字段常见取值包括 bad_flag。",
            structured={"fields": {"target_col": ["bad_flag"]}},
            source_memory_ids=("mem-1", "mem-2", "mem-3", "mem-4"),
            support_count=4,
        )
    )
    store.create_distillation(
        new_distillation(
            category="field_convention",
            scope_key="field_convention:score_col",
            distilled_summary="分数字段可能叫 score。",
            structured={"fields": {"score_col": ["score"]}},
            source_memory_ids=("mem-low",),
            support_count=1,
        )
    )

    packets = retrieve_with_distillations(
        store,
        {"keywords": ["bad_flag"], "scope": "A卡"},
        limit=2,
    )

    assert packets[0]["kind"] == "distillation"
    assert packets[0]["id"] == high.id
    assert packets[0]["support_count"] == 4
    assert packets[0]["source_memory_ids"] == ["mem-1", "mem-2", "mem-3", "mem-4"]
    assert packets[1]["kind"] == "raw"
    assert packets[1]["source_task_id"] == "task-raw"


def test_distillation_prompt_packet_preserves_audit_fields():
    context = {
        "memories": [
            {
                "kind": "distillation",
                "id": "dist-1",
                "memory_type": "field_convention",
                "summary": "目标字段常见取值包括 bad_flag。",
                "payload": {"fields": {"target_col": ["bad_flag"]}},
                "confidence": "high",
                "support_count": 4,
                "source_memory_ids": ["mem-1", "mem-2"],
            }
        ]
    }
    normalized = normalize_memory_context(context)

    packet = normalized["memories"][0]
    assert packet["kind"] == "distillation"
    assert packet["support_count"] == 4
    assert packet["source_memory_ids"] == ["mem-1", "mem-2"]
    assert packet["payload"] == {"fields": {"target_col": ["bad_flag"]}}
    references = memory_references(context, use_reason="chat")
    assert references == [
        {
            "kind": "distillation",
            "id": "dist-1",
            "memory_type": "field_convention",
            "source_task_id": None,
            "confidence": "high",
            "use_reason": "chat",
            "support_count": 4,
            "source_memory_ids": ["mem-1", "mem-2"],
        }
    ]
