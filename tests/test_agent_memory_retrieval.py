from datetime import UTC, datetime, timedelta

from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.retrieval import (
    MemoryQuery,
    compare_model_experience,
    normalize_model_family,
    retrieve_relevant_memories,
)


def _iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _model_payload(**overrides):
    payload = {
        "ks": 30.0,
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


def _candidate(**overrides):
    payload = _model_payload(**overrides.pop("payload_overrides", {}))
    return MemoryCandidate(
        memory_type=overrides.pop("memory_type", "model_experience"),
        summary=overrides.pop("summary", "A卡模型在自营渠道202601表现稳定。"),
        payload=payload,
        source_task_id=overrides.pop("source_task_id", payload["source_task_id"]),
        confidence=overrides.pop("confidence", "high"),
        reason=overrides.pop("reason", "历史模型经验"),
    )


def test_exact_model_scope_channel_month_match_is_high_confidence():
    entries = [
        _candidate(),
        _candidate(
            summary="B卡历史经验",
            payload_overrides={
                "model_name": "B卡申请模型",
                "scope": "贷中B卡",
                "month": "202512",
                "channel": "联合贷",
                "source_task_id": "task-b",
            },
            source_task_id="task-b",
        ),
    ]
    query = MemoryQuery(
        model_name="分润通用A卡模型",
        scope="mob3贷前A卡",
        channel="自营",
        month="202601",
    )

    results = retrieve_relevant_memories(entries, query)

    assert len(results) == 1
    assert results[0].confidence == "high"
    assert "exact model" in results[0].match_reason
    assert "exact scope" in results[0].match_reason
    assert results[0].context_packet == {
        "id": None,
        "memory_type": "model_experience",
        "summary": "A卡模型在自营渠道202601表现稳定。",
        "payload": {
            "ks": 30.0,
            "auc": 0.72,
            "psi": 0.08,
            "month": "202601",
            "channel": "自营",
            "model_name": "分润通用A卡模型",
            "model_version": "V2026",
            "scope": "mob3贷前A卡",
            "important_feature_sources": ["征信", "交易"],
        },
        "source_task_id": "task-202601",
        "confidence": "high",
        "match_reason": results[0].match_reason,
        "observed_at": None,
        "age_days": None,
    }


def test_fuzzy_model_keyword_and_scope_matches_are_medium_confidence():
    entries = [
        {
            "id": "mem-a",
            "memory_type": "model_experience",
            "summary": "A卡申请模型历史上在mob3客群PSI容易抬升。",
            "payload": _model_payload(
                model_name="A卡申请模型V5",
                scope="贷前申请mob3",
                month="202512",
                source_task_id="task-202512",
            ),
            "source_task_id": "task-202512",
            "confidence": "medium",
            "status": "active",
        }
    ]
    query = MemoryQuery(model_name="A card new model", scope="mob3申请客群")

    results = retrieve_relevant_memories(entries, query)

    assert len(results) == 1
    assert results[0].confidence == "medium"
    assert "model family" in results[0].match_reason
    assert "scope keyword" in results[0].match_reason


def test_retrieval_supports_field_convention_and_task_experience_categories():
    entries = [
        {
            "id": "field-1",
            "memory_type": "field_convention",
            "summary": "A卡验证里坏样本字段常用 bad_flag。",
            "payload": {"target_col": "bad_flag", "time_col": "apply_month"},
            "source_task_id": "task-field",
            "confidence": "high",
            "status": "active",
        },
        {
            "id": "task-1",
            "memory_type": "task_experience",
            "summary": "Notebook 缺少 RMC_SAMPLE_DF 时需要先补运行契约。",
            "payload": {"status": "failed"},
            "source_task_id": "task-pitfall",
            "confidence": "medium",
            "status": "active",
        },
    ]
    query = MemoryQuery(model_name="A卡模型", keywords=("bad_flag", "RMC_SAMPLE_DF"))

    results = retrieve_relevant_memories(entries, query)

    assert [result.context_packet["id"] for result in results] == ["field-1", "task-1"]
    assert results[0].context_packet["payload"] == {
        "target_col": "bad_flag",
        "time_col": "apply_month",
    }


def test_low_confidence_candidates_are_excluded_from_comparison_context():
    entries = [
        _candidate(confidence="low"),
        _candidate(
            confidence="medium",
            payload_overrides={"month": "202512", "source_task_id": "task-202512"},
            source_task_id="task-202512",
        ),
    ]
    current = _model_payload(month="202601", source_task_id="current-task")

    comparison = compare_model_experience(current, entries)

    assert [packet["source_task_id"] for packet in comparison["context_packets"]] == [
        "task-202512"
    ]
    assert all(
        packet["confidence"] != "low" for packet in comparison["context_packets"]
    )


def test_compare_model_experience_excludes_unrelated_model_memories():
    entries = [
        _candidate(
            payload_overrides={
                "model_name": "A卡申请模型",
                "scope": "mob3贷前A卡",
                "month": "202512",
                "channel": "自营",
                "source_task_id": "task-a",
            },
            source_task_id="task-a",
        ),
        _candidate(
            payload_overrides={
                "model_name": "利率模型",
                "scope": "定价利率",
                "month": "202601",
                "channel": "自营",
                "source_task_id": "task-rate",
            },
            source_task_id="task-rate",
        ),
    ]
    current = _model_payload(
        model_name="A卡申请模型",
        scope="mob3贷前A卡",
        month="202601",
        channel="自营",
        source_task_id="current-task",
    )

    comparison = compare_model_experience(current, entries)

    assert [packet["source_task_id"] for packet in comparison["context_packets"]] == [
        "task-a"
    ]
    assert comparison["dimensions"]["models"] == ["A卡申请模型"]


def test_compare_model_experience_supports_multiple_dimensions_and_metrics():
    entries = [
        _candidate(
            payload_overrides={
                "model_name": "A卡申请模型",
                "month": "202512",
                "channel": "自营",
                "ks": 28.5,
                "auc": 0.70,
                "psi": 0.12,
                "source_task_id": "task-a-202512",
            },
            source_task_id="task-a-202512",
        ),
        _candidate(
            payload_overrides={
                "model_name": "A卡申请模型",
                "month": "202601",
                "channel": "联合贷",
                "ks": 31.0,
                "auc": 0.73,
                "psi": 0.09,
                "source_task_id": "task-a-union",
            },
            source_task_id="task-a-union",
        ),
    ]
    current = _model_payload(
        model_name="A卡申请模型",
        month="202601",
        channel="自营",
        ks=30.0,
        auc=0.72,
        psi=0.08,
        source_task_id="current-task",
    )

    comparison = compare_model_experience(current, entries, limit=5)

    assert comparison["current"] == {
        "model_name": "A卡申请模型",
        "model_family": "a_card",
        "model_version": "V2026",
        "scope": "mob3贷前A卡",
        "month": "202601",
        "channel": "自营",
        "metrics": {"ks": 30.0, "auc": 0.72, "psi": 0.08},
    }
    assert comparison["dimensions"] == {
        "models": ["A卡申请模型"],
        "months": ["202512", "202601"],
        "channels": ["联合贷", "自营"],
        "metrics": ["ks", "auc", "psi"],
    }
    assert [packet["source_task_id"] for packet in comparison["context_packets"]] == [
        "task-a-202512",
        "task-a-union",
    ]


def test_normalize_model_family_recognizes_english_and_chinese_examples():
    assert normalize_model_family("A card model") == "a_card"
    assert normalize_model_family("A卡申请模型") == "a_card"
    assert normalize_model_family("B card") == "b_card"
    assert normalize_model_family("B卡贷中模型") == "b_card"
    assert normalize_model_family("C card") == "c_card"
    assert normalize_model_family("C卡模型") == "c_card"
    assert normalize_model_family("amount model") == "amount"
    assert normalize_model_family("额度模型") == "amount"
    assert normalize_model_family("rate model") == "rate"
    assert normalize_model_family("利率模型") == "rate"
    assert normalize_model_family("pre-screening model") == "pre_screening"
    assert normalize_model_family("前筛模型") == "pre_screening"


def test_bounded_payload_preserves_non_model_memory_fields():
    from marvis.agent_memory.prompting import (
        _bounded_payload,
        normalize_memory_context,
    )

    # A field_convention memory keeps its allowlisted structured fields; the old
    # hard-coded model_experience set silently dropped them to {}.
    convention = _bounded_payload(
        {"field": "apply_month", "meaning": "申请月份", "secret": "leak"},
        "field_convention",
    )
    assert convention == {"field": "apply_month", "meaning": "申请月份"}

    # model_experience metrics still pass; non-allowlisted keys are filtered.
    model = _bounded_payload({"ks": 30.0, "auc": 0.72, "bogus": 1}, "model_experience")
    assert model == {"ks": 30.0, "auc": 0.72}

    # Unknown memory_type denies all payload fields by default.
    assert _bounded_payload({"field": "x"}, "mystery") == {}

    normalized = normalize_memory_context(
        {
            "memories": [
                {
                    "id": "m1",
                    "memory_type": "field_convention",
                    "summary": "s",
                    "payload": {"field": "apply_month", "meaning": "申请月份"},
                }
            ]
        }
    )
    assert normalized["memories"][0]["payload"] == {
        "field": "apply_month",
        "meaning": "申请月份",
    }



def test_recent_entry_scores_higher_than_stale_entry_with_same_match_strength():
    recent_entry = {
        "id": "mem-recent",
        "memory_type": "model_experience",
        "summary": "A卡模型近期经验",
        "payload": _model_payload(source_task_id="task-recent"),
        "source_task_id": "task-recent",
        "confidence": "medium",
        "status": "active",
        "created_at": _iso_days_ago(10),
    }
    stale_entry = {
        "id": "mem-stale",
        "memory_type": "model_experience",
        "summary": "A卡模型陈旧经验",
        "payload": _model_payload(source_task_id="task-stale"),
        "source_task_id": "task-stale",
        "confidence": "medium",
        "status": "active",
        "created_at": _iso_days_ago(400),
    }
    query = MemoryQuery(
        model_name="分润通用A卡模型",
        scope="mob3贷前A卡",
        channel="自营",
        month="202601",
    )

    results = retrieve_relevant_memories([recent_entry, stale_entry], query, limit=2)

    scores = {result.entry["id"]: result.score for result in results}
    assert scores["mem-recent"] - scores["mem-stale"] == 20
    recent_result = next(r for r in results if r.entry["id"] == "mem-recent")
    stale_result = next(r for r in results if r.entry["id"] == "mem-stale")
    assert recent_result.context_packet["age_days"] == 10
    assert stale_result.context_packet["age_days"] == 400
    assert "recent" in recent_result.match_reason
    assert "stale" in stale_result.match_reason


def test_missing_created_at_gets_neutral_recency_and_null_age():
    entry = {
        "id": "mem-unknown-age",
        "memory_type": "model_experience",
        "summary": "A卡模型经验",
        "payload": _model_payload(source_task_id="task-unknown"),
        "source_task_id": "task-unknown",
        "confidence": "medium",
        "status": "active",
    }
    query = MemoryQuery(model_name="分润通用A卡模型", scope="mob3贷前A卡")

    results = retrieve_relevant_memories([entry], query, limit=1)

    assert results[0].context_packet["age_days"] is None
    assert results[0].context_packet["observed_at"] is None
    assert "recent" not in results[0].match_reason
    assert "stale" not in results[0].match_reason


def test_memory_packet_annotates_summary_with_age_in_days():
    from marvis.agent_memory.prompting import normalize_memory_context

    normalized = normalize_memory_context(
        {
            "memories": [
                {
                    "id": "mem-1",
                    "memory_type": "model_experience",
                    "summary": "A卡模型历史表现稳定",
                    "confidence": "high",
                    "age_days": 5,
                    "observed_at": "2026-06-27T00:00:00+00:00",
                }
            ]
        }
    )

    packet = normalized["memories"][0]
    assert packet["summary"] == "A卡模型历史表现稳定（5 天前）"
    assert packet["age_days"] == 5
    assert packet["observed_at"] == "2026-06-27T00:00:00+00:00"
