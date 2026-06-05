import pytest

from riskmodel_checker.agent_memory.store import AgentMemoryStore
from riskmodel_checker.db import init_db
from riskmodel_checker.agent_memory.models import (
    MEMORY_STATUSES,
    MEMORY_TYPES,
    MODEL_EXPERIENCE_REQUIRED_FIELDS,
    MemoryCandidate,
    normalize_memory_status,
    normalize_memory_type,
    validate_model_experience_payload,
)
from riskmodel_checker.agent_memory.policy import classify_memory_candidate


def _model_experience_payload(**overrides):
    payload = {
        "ks": 30,
        "auc": 0.72,
        "psi": 0.08,
        "month": "202601",
        "channel": "自营",
        "model_name": "分润通用A卡模型",
        "model_version": "V2026",
        "scope": "mob3贷前A卡",
        "source_task_id": "task-202601",
        "important_feature_sources": ["征信", "交易"],
    }
    payload.update(overrides)
    return payload


def test_memory_types_cover_v1_1_foundation_categories():
    assert set(MEMORY_TYPES) == {
        "user_preference",
        "field_convention",
        "validation_pitfall",
        "task_experience",
        "model_experience",
        "skill_experience_reserved",
    }


def test_memory_statuses_cover_active_disabled_deleted_and_rejected():
    assert set(MEMORY_STATUSES) == {"active", "disabled", "deleted", "rejected"}
    assert normalize_memory_status(" active ") == "active"
    assert normalize_memory_status("DISABLED") == "disabled"

    with pytest.raises(ValueError, match="unsupported memory status"):
        normalize_memory_status("archived")


def test_memory_type_normalization_rejects_unknown_categories():
    assert normalize_memory_type(" model_experience ") == "model_experience"
    assert normalize_memory_type("USER_PREFERENCE") == "user_preference"

    with pytest.raises(ValueError, match="unsupported memory type"):
        normalize_memory_type("metric_snapshot")


def test_model_experience_requires_all_v1_1_fields():
    assert set(MODEL_EXPERIENCE_REQUIRED_FIELDS) == {
        "ks",
        "auc",
        "psi",
        "month",
        "channel",
        "model_name",
        "model_version",
        "scope",
        "source_task_id",
        "important_feature_sources",
    }
    payload = _model_experience_payload()

    assert validate_model_experience_payload(payload) == payload

    incomplete = dict(payload)
    incomplete.pop("important_feature_sources")
    with pytest.raises(ValueError, match="missing required model_experience fields"):
        validate_model_experience_payload(incomplete)


def test_memory_candidate_normalizes_type_and_validates_model_payload():
    candidate = MemoryCandidate(
        memory_type=" MODEL_EXPERIENCE ",
        summary="分润通用A卡模型V2026在202601自营渠道KS为30。",
        payload=_model_experience_payload(),
        source_task_id="task-202601",
        confidence="high",
    )

    assert candidate.memory_type == "model_experience"
    assert candidate.payload["source_task_id"] == "task-202601"


@pytest.mark.parametrize(
    ("summary", "payload", "expected_reason"),
    [
        (
            "客户号 622200000001，手机号 13800138000，y=1，score=0.734。",
            {},
            "customer detail",
        ),
        (
            "```python\nimport pandas as pd\ndf = pd.read_csv('/tmp/raw.csv')\n```",
            {},
            "notebook source",
        ),
        (
            "<PMML><Header/><MiningModel><Segmentation/></MiningModel></PMML>",
            {},
            "pmml or model content",
        ),
        (
            "OPENAI_API_KEY=sk-test-secret-value",
            {},
            "secret",
        ),
        (
            "postgresql://user:pass@example.internal:5432/risk",
            {},
            "database connection",
        ),
        (
            "模型验证报告全文：" + "本报告包含机构敏感信息。" * 60,
            {},
            "long report text",
        ),
        (
            "raw row: age=36 score=0.734 y=1 apply_month=202601 channel=自营",
            {},
            "raw sample row",
        ),
    ],
)
def test_policy_rejects_forbidden_memory_content(summary, payload, expected_reason):
    candidate = MemoryCandidate(
        memory_type="task_experience",
        summary=summary,
        payload=payload,
        source_task_id="task-1",
        confidence="medium",
    )

    decision = classify_memory_candidate(candidate)

    assert decision.allowed is False
    assert expected_reason in decision.reasons


def test_policy_allows_compact_structured_model_experience():
    candidate = MemoryCandidate(
        memory_type="model_experience",
        summary="分润通用A卡模型V2026在202601自营渠道KS为30，AUC为0.72，PSI为0.08。",
        payload=_model_experience_payload(),
        source_task_id="task-202601",
        confidence="high",
    )

    decision = classify_memory_candidate(candidate)

    assert decision.allowed is True
    assert decision.reasons == []


def test_store_creates_active_memory_and_audits_create_and_retrieve(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    candidate = MemoryCandidate(
        memory_type="model_experience",
        summary="分润通用A卡模型V2026在202601自营渠道KS为30。",
        payload=_model_experience_payload(),
        source_task_id="task-202601",
        source_message_id="msg-source",
        confidence="high",
        reason="validation complete",
    )

    entry = store.create(candidate, task_id="task-202601")
    fetched = store.get_entry(entry.id, task_id="task-next")

    assert fetched.id == entry.id
    assert fetched.memory_type == "model_experience"
    assert fetched.status == "active"
    assert fetched.summary == "分润通用A卡模型V2026在202601自营渠道KS为30。"
    assert fetched.payload["model_version"] == "V2026"
    assert fetched.source_task_id == "task-202601"
    assert fetched.source_message_id == "msg-source"
    assert fetched.confidence == "high"
    assert fetched.reason == "validation complete"

    events = store.list_events(entry.id)
    assert [event["event_type"] for event in events] == ["create", "retrieve"]
    assert events[0]["details"]["memory_type"] == "model_experience"
    assert events[1]["task_id"] == "task-next"


def test_store_audits_use_disable_enable_and_delete_with_redaction(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    entry = store.create(
        MemoryCandidate(
            memory_type="task_experience",
            summary="Notebook 环境缺少 xgboost 时需要先检查依赖清单。",
            payload={"failure_type": "dependency", "package": "xgboost"},
            source_task_id="task-old",
        )
    )

    store.record_use(
        entry.id,
        task_id="task-new",
        message_id="msg-new",
        use_reason="提醒检查依赖",
    )
    disabled = store.set_status(entry.id, "disabled", task_id="admin-task")
    assert disabled.status == "disabled"
    assert store.list_entries() == []
    assert [item.id for item in store.list_entries(status="disabled")] == [entry.id]

    enabled = store.set_status(entry.id, "active", task_id="admin-task")
    assert enabled.status == "active"
    assert [item.id for item in store.list_entries()] == [entry.id]

    deleted = store.delete(entry.id, task_id="admin-task")
    assert deleted.status == "deleted"
    assert deleted.summary == ""
    assert deleted.payload == {}
    assert deleted.deleted_at is not None
    assert store.list_entries() == []
    assert store.list_entries(status="active") == []

    tombstone = store.get_entry(entry.id, include_deleted=True, audit=False)
    assert tombstone.status == "deleted"
    assert tombstone.summary == ""
    assert tombstone.payload == {}

    events = store.list_events(entry.id)
    assert [event["event_type"] for event in events] == [
        "create",
        "use",
        "disable",
        "enable",
        "delete",
    ]
    assert events[1]["message_id"] == "msg-new"
    assert events[1]["details"]["use_reason"] == "提醒检查依赖"


def test_store_rejects_candidate_with_audited_redacted_tombstone(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    candidate = MemoryCandidate(
        memory_type="task_experience",
        summary="OPENAI_API_KEY=sk-test-secret-value",
        payload={"raw": "OPENAI_API_KEY=sk-test-secret-value"},
        source_task_id="task-secret",
    )
    decision = classify_memory_candidate(candidate)

    rejected = store.reject(candidate, decision, task_id="task-secret")

    assert rejected.status == "rejected"
    assert rejected.summary == ""
    assert rejected.payload == {}
    assert rejected.source_task_id == "task-secret"
    assert store.list_entries() == []
    assert [item.id for item in store.list_entries(status="rejected")] == [rejected.id]

    events = store.list_events(rejected.id)
    assert [event["event_type"] for event in events] == ["reject"]
    assert events[0]["details"]["reasons"] == ["secret"]

    with pytest.raises(ValueError, match="rejected memory entries are terminal"):
        store.set_status(rejected.id, "active")


def test_store_create_enforces_policy_and_never_saves_unsafe_active_memory(tmp_path):
    db_path = tmp_path / "app.sqlite"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    candidate = MemoryCandidate(
        memory_type="task_experience",
        summary="OPENAI_API_KEY=sk-test-secret-value",
        payload={"raw": "OPENAI_API_KEY=sk-test-secret-value"},
        source_task_id="task-secret",
    )

    rejected = store.create(candidate, task_id="task-secret")

    assert rejected.status == "rejected"
    assert rejected.summary == ""
    assert rejected.payload == {}
    assert store.list_entries() == []
    events = store.list_events(rejected.id)
    assert [event["event_type"] for event in events] == ["reject"]
    assert events[0]["details"]["reasons"] == ["secret"]
