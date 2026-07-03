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


def test_distillation_engine_skips_non_numeric_model_metrics_instead_of_dropping_group(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    for task_id, ks in [("task-1", "N/A"), ("task-2", 0.31)]:
        store.create(
            MemoryCandidate(
                memory_type="model_experience",
                summary=f"模型经验 {task_id}",
                payload={
                    "model_name": "A卡",
                    "model_version": "V1",
                    "scope": "train",
                    "channel": "app",
                    "month": "202601",
                    "ks": ks,
                    "auc": 0.72,
                    "psi": 0.08,
                    "source_task_id": task_id,
                    "important_feature_sources": ["征信"],
                },
                source_task_id=task_id,
            )
        )

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category("model_experience")

    assert len(distillations) == 1
    metric_ranges = distillations[0].structured["metric_ranges"]
    assert metric_ranges["ks"] == {"min": 0.31, "max": 0.31}
    assert metric_ranges["auc"] == {"min": 0.72, "max": 0.72}


def test_distillation_support_counts_distinct_tasks_not_entries(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    payload = {
        "model_name": "A卡",
        "model_version": "V1",
        "scope": "train",
        "channel": "app",
        "month": "202601",
        "ks": 0.31,
        "auc": 0.72,
        "psi": 0.08,
        "source_task_id": "task-1",
        "important_feature_sources": ["征信"],
    }
    for _ in range(4):
        store.create(
            MemoryCandidate(
                memory_type="model_experience",
                summary="模型经验 task-1",
                payload=dict(payload),
                source_task_id="task-1",
            )
        )

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category(
        "model_experience"
    )

    assert len(distillations) == 1
    distilled = distillations[0]
    # Rerunning the same task 4 times deduplicates at capture time (MEM-3), so
    # only one entry exists to distill; support/confidence must reflect the
    # single independent data point, not 4 duplicate rows.
    assert distilled.support_count == 1
    assert distilled.confidence == "low"


def test_distillation_model_experience_carries_months_covered_range(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    for task_id, month in [("task-1", "202503"), ("task-2", "202601")]:
        store.create(
            MemoryCandidate(
                memory_type="model_experience",
                summary=f"模型经验 {task_id}",
                payload={
                    "model_name": "A卡",
                    "model_version": "V1",
                    "scope": "train",
                    "channel": "app",
                    "month": month,
                    "ks": 0.31,
                    "auc": 0.72,
                    "psi": 0.08,
                    "source_task_id": task_id,
                    "important_feature_sources": ["征信"],
                },
                source_task_id=task_id,
            )
        )

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category(
        "model_experience"
    )

    assert len(distillations) == 1
    assert distillations[0].structured["months_covered"] == {"min": "202503", "max": "202601"}


def test_distill_category_logs_and_counts_group_failures_instead_of_swallowing(tmp_path, caplog):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    store.create(
        MemoryCandidate(
            memory_type="user_preference",
            summary="正常偏好经验。",
            payload={"preference": "正常偏好经验。"},
            source_task_id="task-ok",
        )
    )
    engine = DistillationEngine(store)
    original_distill_group = engine._distill_group

    def _boom(category, scope_key, members):
        if scope_key == "user_preference:general":
            raise RuntimeError("boom")
        return original_distill_group(category, scope_key, members)

    engine._distill_group = _boom

    with caplog.at_level("WARNING"):
        results = engine.distill_category("user_preference")

    assert results == []
    assert engine.last_error_count == 1
    assert any("distill group failed" in record.message for record in caplog.records)


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


def test_distillation_support_nets_out_negative_feedback(tmp_path):
    # MEM-7c: distillation support becomes a net value (distinct-task positive
    # support minus distinct-task negative feedback) instead of a raw member
    # count, so a group whose entries were mostly discredited by task
    # failures / user "not useful" reports does not present the same
    # confidence as an equally-sized clean group.
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    payload = {
        "model_name": "A卡",
        "model_version": "V1",
        "scope": "train",
        "channel": "app",
        "month": "202601",
        "ks": 0.31,
        "auc": 0.72,
        "psi": 0.08,
        "important_feature_sources": ["征信"],
    }
    entries = []
    for task_id in ("task-1", "task-2", "task-3", "task-4"):
        entries.append(
            store.create(
                MemoryCandidate(
                    memory_type="model_experience",
                    summary=f"模型经验 {task_id}",
                    payload={**payload, "source_task_id": task_id},
                    source_task_id=task_id,
                ),
                task_id=task_id,
            )
        )
    # Flag two of the four distinct-task entries as negative.
    store.record_negative_feedback(entries[0].id, downgrade=False)
    store.record_negative_feedback(entries[1].id, downgrade=False)

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category(
        "model_experience"
    )

    assert len(distillations) == 1
    distilled = distillations[0]
    # 4 distinct-task positive support - 2 distinct-task negative = 2 net.
    assert distilled.support_count == 2
    assert distilled.confidence == "medium"
    # The raw per-field "support" annotation inside the structured payload is
    # left untouched (still counts merged entries, purely descriptive) --
    # only support_count, which drives confidence_from_support, uses the net
    # value.
    assert distilled.structured["support"] == 4


def test_distillation_support_never_goes_below_zero(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    entry = store.create(
        MemoryCandidate(
            memory_type="field_convention",
            summary="字段口径：目标字段=bad_flag",
            payload={"target_col": "bad_flag"},
            source_task_id="task-1",
        )
    )
    store.record_negative_feedback(entry.id, downgrade=False)

    distillations = DistillationEngine(store, llm_factory=lambda: _BrokenLLM()).distill_category(
        "field_convention"
    )

    assert len(distillations) == 1
    assert distillations[0].support_count == 0
    assert distillations[0].confidence == "low"
