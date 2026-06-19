from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db


def test_distillation_engine_groups_field_conventions_and_merges_deterministically(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
            source_task_id="task-1",
        )
    )
    store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=y",
            payload={"target_col": "y"},
            source_task_id="task-2",
        )
    )

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category("field_convention")

    assert len(distillations) == 1
    distilled = distillations[0]
    assert distilled.scope_key == "field_convention:target_col"
    assert distilled.support_count == 2
    assert distilled.confidence == "medium"
    assert distilled.structured == {"fields": {"target_col": ["bad_flag", "y"]}, "support": 2}
    assert "字段口径经验" in distilled.distilled_summary
    assert len(distilled.source_memory_ids) == 2


def test_distillation_engine_uses_llm_only_for_summary_wording(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="validation_pitfall",
            summary="notebook validation pitfall: missing RMC_SAMPLE_DF",
            payload={"failure_kind": "notebook", "message": "missing RMC_SAMPLE_DF"},
            source_task_id="task-1",
        )
    )
    llm = _FakeLLM("Notebook 契约缺字段是重复出现的验证坑点。")

    distillations = DistillationEngine(store, llm_factory=lambda: llm).distill_category("validation_pitfall")

    assert distillations[0].distilled_summary == "Notebook 契约缺字段是重复出现的验证坑点。"
    assert "source_summaries" in llm.calls[0]["user_prompt"]
    assert "不要输出任务 ID" in llm.calls[0]["system_prompt"]


def test_distillation_engine_drops_sensitive_distilled_output(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    engine = DistillationEngine(store)
    member = {
        "id": "mem-raw",
        "memory_type": "user_preference",
        "summary": "请联系手机号 13800138000。",
        "payload": {"preference": "手机号 13800138000"},
    }

    assert engine._distill_group("user_preference", "user_preference:general", [member]) is None


class _BrokenLLM:
    def complete(self, **_kwargs):
        raise RuntimeError("unavailable")


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response
