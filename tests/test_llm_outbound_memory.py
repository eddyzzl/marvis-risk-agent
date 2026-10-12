"""Historical aggregate privacy and citations at the actual request boundary."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from marvis.agent.service import generate_word_conclusions, summarize_stage
from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_memory import ModelHistory, model_history_context
from marvis.llm_outbound_validation import validation_metrics_summary_prompt
from tests.test_agent_service import _task
from tests.test_llm_outbound import Wire
from tests.test_llm_outbound_validation import _complete_metrics_summary, _native_result, _report_replies, _v1_report_reply

PRIVATE = "PRIVATE_HISTORY_CONTENT_748"


def _memory(**changes):
    record = {
        "id": PRIVATE + "-memory", "kind": "raw", "memory_type": "model_experience",
        "summary": PRIVATE, "source_task_id": PRIVATE + "-task", "confidence": "high", "age_days": 8,
        "payload": {"model_name": "A卡 " + PRIVATE, "model_version": PRIVATE,
                    "scope": PRIVATE, "channel": PRIVATE, "month": "202601",
                    "source_task_id": PRIVATE + "-task", "important_feature_sources": [PRIVATE],
                    "ks": .22, "auc": .66, "psi": .31, "head_lift_5pct": .97, "tail_lift_5pct": 1.04,
                    "overfitting_status": "fail", "sample_rows": [PRIVATE], "arbitrary_file": PRIVATE},
    }
    record.update(changes)
    return record


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("stage", ["metrics", "report"])
def test_history_metrics_reach_only_relevant_requests_and_exact_local_citations(monkeypatch, version, stage):
    memory = _memory()
    context = {"memories": [memory, _memory(id="unused", memory_type="user_preference")]}
    before = deepcopy(context)
    task = replace(_task(), validation_workflow_version=version)
    wire = Wire(monkeypatch, text_reply=version == 1 and stage == "metrics")
    if stage == "metrics":
        wire.reply = _complete_metrics_summary() if version == 1 else {"summary": "历史记录 1 的 PSI 为 0.31；评估分层未知，谨慎对照。"}
    elif version == 1:
        wire.reply = _v1_report_reply()
    else:
        wire.replies = _report_replies()
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    kwargs = dict(task=task, evidence={"validation_results": _native_result()}, memory_context=context, model_profile={})
    if stage == "metrics":
        _, metadata = summarize_stage(**kwargs, stage="metrics", fallback="本地回退")
    else:
        _, metadata = generate_word_conclusions(**kwargs)
    assert metadata["fallback"] is False and metadata["history_comparison_available"] is True
    assert metadata["memory_references"] == [{
        "kind": "raw", "id": memory["id"], "memory_type": "model_experience",
        "source_task_id": memory["source_task_id"], "confidence": "high", "history_index": 1,
        "use_reason": "metrics" if stage == "metrics" else "word_conclusion_draft",
    }]
    contexts = [json.loads(request["messages"][1]["content"]) for request in wire.requests]
    with_history = [payload for payload in contexts if payload["history_comparison_available"]]
    assert len(with_history) == 1
    if version == 2:
        assert with_history[0]["narrative_topic"] == "overall"
        assert all("historical_models" not in payload for payload in contexts if payload["narrative_topic"] != "overall")
    historical = with_history[0]["historical_models"]["records"][0]
    assert historical["metrics"] == {key: memory["payload"][key] for key in
                                     ("ks", "auc", "psi", "head_lift_5pct", "tail_lift_5pct", "overfitting_status")}
    assert historical["comparison_basis"]["same_model_family"] is True
    assert historical["comparison_basis"]["same_channel"] is False
    assert historical["comparison_basis"]["channel_known_for_current"] is False
    assert historical["metric_units_known"] is False and historical["evaluation_split_known"] is False
    assert historical["month"] == "202601" and historical["age_days"] == 8
    assert PRIVATE not in json.dumps(wire.requests)
    assert context == before


@pytest.mark.parametrize("override", [
    {"kind": "distillation"}, {"memory_type": "user_preference"}, {"confidence": "low"},
    {"confidence": ["high"]}, {"status": "disabled"}, {"source_task_id": "task-1"}, {"source_task_id": "conflicting-source"},
])
def test_unusable_history_does_not_create_outbound_records_or_citations(override):
    history = model_history_context(task=_task(), evidence={}, memory_context={"memories": [_memory(**override)]})
    assert json.loads(history.payload_json) == {"records": []}
    assert history.metadata(use_reason="metrics") == {"history_comparison_available": False}


def test_unrelated_model_unknown_placeholders_and_invalid_metrics_are_not_comparability():
    record = _memory()
    record["payload"].update(model_name="B卡", scope="未标注", channel="未标注", model_version="未标注")
    history = model_history_context(task=replace(_task(), model_name="A卡", model_version="未标注"),
                                    evidence={"validation_results": {"scope": "未标注", "channel": "未标注"}},
                                    memory_context={"memories": [record]})
    assert not history.metadata(use_reason="metrics")["history_comparison_available"]
    record["payload"]["model_name"] = "A卡"
    record["payload"].update(ks=float("nan"), auc=float("inf"), psi=True)
    history = model_history_context(task=_task(), evidence={}, memory_context={"memories": [record]})
    assert json.loads(history.payload_json)["records"] == []


def test_legacy_ks_units_are_preserved_without_automatic_percentage_conversion(monkeypatch):
    record = _memory()
    record["payload"]["ks"] = 20
    history = model_history_context(task=_task(), evidence={}, memory_context={"memories": [record]})
    wire = Wire(monkeypatch)
    wire.client.complete(system_prompt="", user_prompt=validation_metrics_summary_prompt({}, history=history), stream=False)
    payload = wire.last_context()
    assert payload["historical_models"]["records"][0]["metrics"]["ks"] == 20
    assert "不能把历史 KS=20 直接与本次 KS=0.3" in payload["instructions"]
    assert PRIVATE not in json.dumps(wire.requests)


def test_declared_units_cannot_turn_out_of_range_history_into_normalized_metrics():
    record = _memory()
    record["payload"].update(ks=20, metric_units={"ks": "fraction", "auc": "fraction", "psi": "index"},
                             evaluation_split="oot")
    history = model_history_context(task=_task(), evidence={}, memory_context={"memories": [record]})
    sent = json.loads(history.payload_json)["records"][0]
    assert sent["metrics"]["ks"] == 20
    assert sent["metric_units_known"] is False
    assert "ks" not in sent["metric_units"]
    assert sent["evaluation_split"] == "oot"


@pytest.mark.parametrize("version", [1, 2])
def test_history_is_identical_in_repair_without_rejected_response_echo(monkeypatch, version):
    wire = Wire(monkeypatch, text_reply=version == 1)
    wire.replies = [PRIVATE, _complete_metrics_summary() if version == 1 else {"summary": "历史记录 1 可作有限参考。"}]
    wire.reply = {"summary": "其他主题只分析本次测量。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, metadata = summarize_stage(task=replace(_task(), validation_workflow_version=version), stage="metrics",
                                  evidence={"validation_results": _native_result()}, memory_context={"memories": [_memory()]},
                                  model_profile={}, fallback="本地回退")
    assert metadata["fallback"] is False
    contexts = [json.loads(request["messages"][1]["content"]) for request in wire.requests[:2]]
    contexts[1].pop("repair_request" if version == 1 else "response_repair")
    assert contexts[0] == contexts[1]
    assert PRIVATE not in json.dumps(wire.requests)


def test_raw_memory_mapping_cannot_be_used_as_approved_history():
    with pytest.raises(OutboundContextError):
        validation_metrics_summary_prompt({}, history={"records": [_memory()]})
    with pytest.raises(OutboundContextError):
        ModelHistory(json.dumps({"records": [_memory()]}), "[]")


def test_history_projection_keeps_late_adverse_metrics_and_rejects_over_capacity(monkeypatch):
    records = [_memory(id=f"history-{index}") for index in range(215)]
    records[-1]["payload"]["psi"] = .81
    history = model_history_context(task=_task(), evidence={}, memory_context={"memories": records})
    projected = json.loads(history.payload_json)["records"]
    assert len(projected) == 215 and projected[-1]["metrics"]["psi"] == .81
    assert len(history.metadata(use_reason="metrics")["memory_references"]) == 215
    wire = Wire(monkeypatch, text_reply=True)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    _, metadata = summarize_stage(task=_task(), stage="metrics", evidence={},
                                  memory_context={"memories": [_memory()] * 10001},
                                  model_profile={}, fallback="本地回退")
    assert metadata["fallback"] is True and wire.requests == []
    assert "memory_references" not in metadata


def test_native_memory_capture_retrieval_and_persisted_citation_are_connected(tmp_path, monkeypatch):
    from marvis.agent.validation_messages import add_and_stream_agent_message
    from marvis.agent_memory.api_support import agent_memory_context_from_store
    from marvis.agent_memory.extractors import extract_model_experience
    from marvis.agent_memory.store import AgentMemoryStore
    from marvis.db import TaskRepository, init_db
    from marvis.domain import TaskCreate
    from marvis.pipeline_memory import _memory_model_experience_payload

    path = tmp_path / "marvis.sqlite"
    init_db(path)
    repo, store = TaskRepository(path), AgentMemoryStore(path)
    historical = repo.create_task(TaskCreate(model_name="A卡 " + PRIVATE, model_version="old", validator="test", source_dir=str(tmp_path)))
    current = repo.create_task(TaskCreate(model_name="A卡 " + PRIVATE, model_version="new", validator="test", source_dir=str(tmp_path)))
    current = replace(current, validation_workflow_version=1)
    results = _native_result()
    results.update(model_name=historical.model_name, model_version=historical.model_version)
    candidate = extract_model_experience(_memory_model_experience_payload(task=historical, results=results))
    assert candidate is not None
    entry = store.create(candidate)
    evidence = {"validation_results": _native_result()}
    recalled = agent_memory_context_from_store(store, current, stage="metrics", evidence=evidence)
    assert any(item["id"] == entry.id for item in recalled["memories"])
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = _complete_metrics_summary()
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    message = add_and_stream_agent_message(
        repo, current.id, stage="metrics", model_profile={}, raise_if_cancelled=lambda _: None,
        producer=lambda callback: summarize_stage(task=current, stage="metrics", evidence=evidence, memory_context=recalled,
                                                  model_profile={}, fallback="本地回退", on_delta=callback),
    )
    stored = repo.list_agent_messages(current.id)
    assert len(stored) == 1 and stored[0]["id"] == message["id"]
    assert stored[0]["metadata"]["history_comparison_available"] is True
    reference = stored[0]["metadata"]["memory_references"][0]
    assert reference["id"] == entry.id and reference["source_task_id"] == historical.id
    assert reference["memory_type"] == "model_experience" and reference["use_reason"] == "metrics"
    sent = wire.last_context()["historical_models"]["records"][0]
    assert sent["metrics"]["ks"] == results["effectiveness"]["overall"][-1]["ks"]
    assert sent["history_index"] == reference["history_index"]
    assert sent["metric_units_known"] is True
    assert sent["metric_units"] == {"ks": "fraction", "auc": "fraction", "psi": "index",
                                    "head_lift_5pct": "ratio", "tail_lift_5pct": "ratio"}
    assert sent["evaluation_split_known"] is True and sent["evaluation_split"] == "oot"
    assert PRIVATE not in json.dumps(wire.requests)
    assert entry.id not in json.dumps(wire.requests) and historical.id not in json.dumps(wire.requests)
