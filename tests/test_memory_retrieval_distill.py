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


def test_retrieve_with_distillations_raw_quota_reserves_slots_for_raw_entries(tmp_path):
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
    # Six independent high-confidence distillations -- without a raw quota,
    # these alone would fill the whole limit=6 budget and squeeze the precise
    # single-task raw entry above out entirely.
    for i in range(6):
        store.create_distillation(
            new_distillation(
                category="field_convention",
                scope_key=f"field_convention:col_{i}",
                distilled_summary=f"字段口径经验 {i}：bad_flag 相关取值。",
                structured={"fields": {f"col_{i}": ["v"]}},
                source_memory_ids=(f"mem-{i}-1", f"mem-{i}-2", f"mem-{i}-3", f"mem-{i}-4"),
                support_count=4,
            )
        )

    packets = retrieve_with_distillations(
        store,
        {"keywords": ["bad_flag"], "scope": "A卡"},
        limit=6,
        raw_quota=3,
    )

    # limit=6, raw_quota=3 reserves 3 slots for raw entries and caps
    # distillations at limit-raw_quota=3, even though 6 distillations exist and
    # would otherwise fill the whole budget on their own.
    assert len(packets) == 4
    raw_packets = [packet for packet in packets if packet["kind"] == "raw"]
    assert len(raw_packets) == 1
    assert raw_packets[0]["source_task_id"] == "task-raw"
    assert sum(1 for packet in packets if packet["kind"] == "distillation") == 3


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
    # The prompt-facing packet bounds source_memory_ids to a count + sample
    # (MEM-11 budget) instead of an unbounded id list.
    assert packet["source_memory_ids_count"] == 2
    assert packet["source_memory_ids_sample"] == ["mem-1", "mem-2"]
    assert packet["payload"] == {"fields": {"target_col": ["bad_flag"]}}
    # The audit trail (memory_references) is built independently and keeps
    # the full source_memory_ids list for traceability.
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



def test_raw_recall_targets_by_kind_so_old_model_experience_is_not_crowded_out(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)

    # An old, high-value model_experience entry created first.
    store.create(
        MemoryCandidate(
            memory_type="model_experience",
            summary="A卡模型历史KS表现记录",
            payload={
                "ks": 30.0,
                "auc": 0.72,
                "psi": 0.08,
                "month": "202501",
                "channel": "自营",
                "model_name": "分润通用A卡模型",
                "model_version": "V2025",
                "scope": "mob3贷前A卡",
                "source_task_id": "task-old",
                "important_feature_sources": ["征信"],
            },
            source_task_id="task-old",
            confidence="high",
        )
    )

    # Flood the store with 250 newer field_convention entries -- under the old
    # "most recent 200 across all types" recall window, these alone would push
    # the model_experience entry above out of list_entries(limit=200).
    for i in range(250):
        store.create(
            MemoryCandidate(
                memory_type="field_convention",
                summary=f"字段口径经验 {i}",
                payload={"target_col": f"col_{i}"},
                source_task_id=f"task-conv-{i}",
            )
        )

    packets = retrieve_with_distillations(
        store,
        {
            "model_name": "分润通用A卡模型",
            "scope": "mob3贷前A卡",
            "channel": "自营",
            "month": "202501",
        },
        limit=6,
    )

    assert any(
        packet["kind"] == "raw" and packet["source_task_id"] == "task-old"
        for packet in packets
    )



def test_memory_packet_bounds_unbounded_source_task_ids_to_count_and_recent_sample():
    task_ids = [f"task-{i:03d}" for i in range(40)]
    context = {
        "memories": [
            {
                "kind": "distillation",
                "id": "dist-big",
                "memory_type": "model_experience",
                "summary": "A卡模型历史经验汇总。",
                "payload": {
                    "model_name": "A卡",
                    "scopes": ["train"],
                    "channels": ["自营"],
                    "metric_ranges": {"ks": {"min": 0.28, "max": 0.34}},
                    "support": len(task_ids),
                    "source_task_ids": task_ids,
                },
                "confidence": "high",
                "support_count": len(task_ids),
            }
        ]
    }

    normalized = normalize_memory_context(context)

    payload = normalized["memories"][0]["payload"]
    assert "source_task_ids" not in payload
    assert payload["source_task_ids_count"] == 40
    assert payload["source_task_ids_sample"] == task_ids[-3:]
    # Other structured fields pass through untouched.
    assert payload["metric_ranges"] == {"ks": {"min": 0.28, "max": 0.34}}
    assert payload["support"] == 40


def test_cross_task_memory_char_budget_drops_low_confidence_packets_first():
    # important_feature_sources is an allowlisted model_experience field, so
    # it survives _bounded_payload trimming and can be used to legitimately
    # inflate a packet past the char budget.
    big_payload = {"important_feature_sources": [f"feature_{i}" for i in range(400)]}
    context = {
        "memories": [
            {
                "kind": "raw",
                "id": "mem-high",
                "memory_type": "model_experience",
                "summary": "高置信度经验",
                "payload": dict(big_payload),
                "confidence": "high",
            },
            {
                "kind": "raw",
                "id": "mem-low",
                "memory_type": "model_experience",
                "summary": "低置信度经验",
                "payload": dict(big_payload),
                "confidence": "low",
            },
        ]
    }

    normalized = normalize_memory_context(context)

    ids = [memory["id"] for memory in normalized["memories"]]
    assert "mem-high" in ids
    assert "mem-low" not in ids
