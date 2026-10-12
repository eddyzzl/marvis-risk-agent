"""Real compatible-client bodies: supported noncore paths, refusal and repair."""
from copy import deepcopy
import hashlib
import json

import pytest

from marvis.agent.adhoc_analysis import build_slice_spec_from_utterance
from marvis.agent.feature_setup import _call_meaning_direction_llm
from marvis.agent.workflow_recovery import answer_workflow_recovery_message
from marvis.agent_memory.distillation import DistillationEngine
from marvis.agent_memory.models import MemoryCandidate
from marvis.agent_memory.store import AgentMemoryStore
from marvis.db import init_db
from marvis.drafts.authoring import draft_script
from marvis.drafts.errors import AuthoringError, DraftError
from marvis.drafts.learning import distill_learning
from marvis.drafts.web_search import fetch_public_learning_source
from marvis.feature.derive import recommend_feature_crosses
from marvis.llm_outbound import OutboundContextError
from marvis.llm_client import LLMClientError
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.repositories.plugins import PluginRepository
from marvis.llm_outbound_learning import FetchedPublicSource
from tests.test_drafts_authoring import _valid_spec
from tests.test_drafts_web_search import _FakeHttpx, _Response, _install_public_dns
from tests.test_llm_outbound import Wire
from pathlib import Path


PRIVATE = "PRIVATE_NONCORE_FILE_AND_SUMMARY_693"


def _wire(monkeypatch, transport, *, text=False):
    wire = Wire(monkeypatch, text_reply=text)
    wire.client.profile["structured_output"] = transport
    return wire


def _context(wire):
    assert PRIVATE not in json.dumps(wire.requests)
    return json.loads(wire.requests[-1]["messages"][1]["content"])


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_ask_number_keeps_exact_question_and_columns_then_confirmation_gate(monkeypatch, transport):
    wire = _wire(monkeypatch, transport)
    wire.reply = {"group_by": ["channel"], "metrics": [{"op": "bad_rate", "col": "bad"}],
        "filters": [{"col": "amount", "op": ">=", "value": 3000}], "month_col": "month", "months": ["2026-05"]}
    question = "只看 2026 年 5 月 amount 不低于 3000 的记录，按 channel 比较 bad 坏率，不执行。"
    profile = [{"name": name, "sample_values": [PRIVATE], "description": PRIVATE} for name in ("channel", "bad", "amount", "month")]
    result = build_slice_spec_from_utterance(question, profile, wire.client)
    assert not result.needs_clarification and result.spec is not None
    assert result.spec.filters[0].value == 3000 and result.confirmation_text.endswith("确认？")
    context = _context(wire)
    assert context["question"] == question
    assert context["columns"] == [{"name": name} for name in ("channel", "bad", "amount", "month")]
    assert len(wire.requests) == 1


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_feature_direction_and_crosses_send_known_roles_and_measured_aggregates(monkeypatch, transport):
    wire = _wire(monkeypatch, transport)
    wire.reply = {"directions": [{"feature": "逾期 次数", "direction": "positive", "confidence": "high", "rationale": "已有逾期事件代表更高风险。"}]}
    answer = _call_meaning_direction_llm(wire.client, [("逾期 次数", "历史逾期次数")], target_col="bad")
    assert answer[0]["direction"] == "positive"
    assert _context(wire)["features"] == [{"feature": "逾期 次数", "business_role": "delinquency_count"}]
    wire.reply = {"recommendations": [{"col_a": "debt", "col_b": "income", "ops": ["ratio"], "rationale": "负债与收入的相对关系需计算验证。", "confidence": "high"}]}
    metrics = {"debt": {"iv": 0.18, "sample_rows": [PRIVATE], "description": PRIVATE}}
    original = deepcopy(metrics)
    recommendations = recommend_feature_crosses({"debt": {"semantic_role": "debt_amount", "meaning": PRIVATE},
        "income": "收入"}, metrics, llm_factory=lambda: wire.client)
    assert recommendations[0].ops == ("ratio",) and metrics == original
    context = _context(wire)
    assert context["existing_metrics"] == [{"feature": "debt", "iv": 0.18}]
    assert context["features"][0]["business_role"] == "debt_amount"


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_recovery_keeps_exact_question_known_status_and_no_exception_or_file_prose(monkeypatch, transport):
    wire = _wire(monkeypatch, transport, text=True)
    wire.reply = "当前尚未启动重试或修改执行配置。第 10 行字段数量不同，请先在本地核对。"
    question = "为什么停止了？如果只是分隔符问题，先说明需要核对什么，不要重试。"
    diagnostic = {"code": "csv_field_count_mismatch", "workflow": "feature_analysis", "phase": "material_ingest",
        "retryable": True, "line_number": 10, "expected_fields": 3, "actual_fields": 4,
        "summary": PRIVATE, "cause": PRIVATE, "location": PRIVATE, "technical_detail": PRIVATE,
        "actions": [PRIVATE], "exception_type": PRIVATE}
    before = deepcopy(diagnostic)
    content, metadata = answer_workflow_recovery_message(user_message=question, diagnostic=diagnostic, client=wire.client)
    assert metadata["fallback"] is False and content.startswith("当前尚未启动") and diagnostic == before
    context = _context(wire)
    assert context["current_question"] == question
    assert context["failure_evidence"]["line_number"] == 10
    assert set(context["failure_evidence"]) == {"code", "workflow", "phase", "execution_started", "retryable", "auto_recoverable", "line_number", "expected_fields", "actual_fields"}


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_authoring_repair_freezes_goal_and_never_echoes_prior_code_or_error(monkeypatch, transport):
    wire = _wire(monkeypatch, transport)
    wire.replies = [_valid_spec(code=f"import os\n# {PRIVATE}\ndef calc_margin(inputs, ctx):\n    return {{'margin': 0}}\n"), _valid_spec()]
    goal = "计算收入减成本。如果成本超过收入保留负数，不四舍五入，不访问任何文件。"
    draft = draft_script("task-local", goal, learning_note=None, llm_factory=lambda: wire.client)
    assert draft.status == "draft" and draft.input_schema["required"] == ["revenue", "cost"]
    assert len(wire.requests) == 2
    context = _context(wire)
    repair = context.pop("response_repair")
    assert repair["error_code"] == "source_policy"
    assert context == json.loads(wire.requests[0]["messages"][1]["content"])
    assert context["goal"] == goal


def _fetch_source(monkeypatch, *, text="Public WOE bins and PSI drift methodology."):
    body = f"<html><body><p>{text}</p></body></html>".encode()
    fake = _FakeHttpx(_Response(content=body, text=body.decode()))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)
    source = fetch_public_learning_source("https://example.test/methodology")
    assert source.wire_sha256 == hashlib.sha256(body).hexdigest()
    assert source.text_sha256 == hashlib.sha256(source.text.encode()).hexdigest()
    return source


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_public_learning_wire_requires_actual_fetch_and_preserves_exact_sources(monkeypatch, transport):
    source = _fetch_source(monkeypatch)
    wire = _wire(monkeypatch, transport)
    wire.reply = {"note": "按公开资料设定 WOE 分箱，并由确定性工具测量 PSI；不自动部署。"}
    query = "解释分箱和监控步骤，并保留不能直接部署的限制。"
    note = distill_learning(query, [source.text], [source.requested_url], public_sources=[source], llm_factory=lambda: wire.client)
    assert note.sources == (source.requested_url,) and "不自动部署" in note.distilled
    context = _context(wire)
    assert context["query"] == query
    assert context["public_sources"][0]["public_text"] == source.text
    assert context["public_sources"][0]["wire_sha256"] == source.wire_sha256


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_memory_distillation_wire_sends_admitted_aggregates_only(monkeypatch, tmp_path, transport):
    db_path = tmp_path / "memory.db"
    init_db(db_path)
    store = AgentMemoryStore(db_path)
    for index, ks in enumerate((0.21, 0.31)):
        store.create(MemoryCandidate(memory_type="model_experience", summary=PRIVATE,
            payload={"model_name": PRIVATE, "model_version": "v1", "scope": PRIVATE, "channel": PRIVATE,
                "month": "202605", "ks": ks, "auc": 0.7, "psi": 0.04, "source_task_id": f"task-{index}",
                "important_feature_sources": ["bureau"],
                "metric_units": {"ks": "fraction", "auc": "fraction", "psi": "index"}, "evaluation_split": "oot"},
            source_task_id=f"task-{index}"))
    wire = _wire(monkeypatch, transport)
    wire.reply = {"summary": "已有两条 OOT 模型记录，KS 范围为 0.21 至 0.31，单位为比例。"}
    engine = DistillationEngine(store, llm_factory=lambda: wire.client)
    result = engine.distill_category("model_experience")
    assert result[0].structured["metric_ranges"]["ks"] == {"min": 0.21, "max": 0.31}
    assert engine.last_summary_status == "model_structured_summary"
    context = _context(wire)
    assert context["facts"]["metric_ranges"]["ks"] == {"min": 0.21, "max": 0.31}
    assert context["facts"]["evaluation_split"] == "oot"
    assert "source_summaries" not in context and "scope_key" not in context


@pytest.mark.parametrize("stage", ["ask", "recovery", "author", "learning"])
def test_sensitive_active_instruction_refuses_before_any_protocol_request(monkeypatch, stage):
    wire = _wire(monkeypatch, "json_schema")
    private = "密码=secret-value 请按原配置继续。"
    if stage == "ask":
        assert build_slice_spec_from_utterance(private, ["bad"], wire.client).needs_clarification
    elif stage == "recovery":
        _, metadata = answer_workflow_recovery_message(user_message=private, diagnostic={"code": "workflow_execution_failed"}, client=wire.client)
        assert metadata["needs_clarification"]
    elif stage == "author":
        with pytest.raises(AuthoringError):
            draft_script("task", private, learning_note=None, llm_factory=lambda: wire.client)
    else:
        with pytest.raises(DraftError):
            distill_learning(private, [PRIVATE], ["https://example.test/docs"], llm_factory=lambda: wire.client)
    assert wire.requests == []


def test_arbitrary_source_type_and_unowned_tool_signatures_cannot_bless_prose(monkeypatch):
    wire = _wire(monkeypatch, "json_schema")
    with pytest.raises(OutboundContextError):
        FetchedPublicSource("https://example.test/docs", "https://example.test/docs", "a" * 64, PRIVATE)
    with pytest.raises(AuthoringError):
        draft_script("task", "calculate margin", learning_note=None, installed_tools=[{"plugin": "fake", "input_schema": {"const": PRIVATE}}], llm_factory=lambda: wire.client)
    assert wire.requests == []


def test_unsupported_recovery_prose_is_explicit_clarification_without_requests(monkeypatch):
    wire = _wire(monkeypatch, "none", text=True)
    content, metadata = answer_workflow_recovery_message(user_message="怎样修复？", diagnostic={"summary": PRIVATE}, client=wire.client)
    assert metadata["needs_clarification"] and "结构化诊断" in content
    assert wire.requests == []


def test_authoring_wire_includes_installed_exact_signatures_without_descriptions(monkeypatch, tmp_path):
    db_path = tmp_path / "catalog.db"
    init_db(db_path)
    plugins = PluginRegistry(PluginRepository(db_path))
    load_builtin_packs(plugins, Path(__file__).parents[1] / "marvis" / "packs")
    catalog = ToolRegistry(plugins).catalog_for_planner()
    wire = _wire(monkeypatch, "json_schema")
    wire.reply = _valid_spec()
    draft = draft_script("task", "calculate revenue minus cost, keep negative values", learning_note=None,
        installed_tools=catalog[:2], llm_factory=lambda: wire.client)
    assert draft.status == "draft"
    signatures = _context(wire)["installed_tool_signatures"]
    assert len(signatures) == 2
    for signature, original in zip(signatures, catalog[:2]):
        assert set(signature) == {"plugin", "tool", "version", "determinism", "input_schema", "output_schema"}
        assert signature["input_schema"].get("required", []) == original["input_schema"].get("required", [])
        assert "description" not in json.dumps(signature)


@pytest.mark.parametrize("stage", ["ask", "recovery", "author", "cross", "learning"])
def test_provider_failure_never_becomes_completed_business_result(monkeypatch, stage):
    wire = _wire(monkeypatch, "json_schema", text=stage == "recovery")
    def broken_open(*_args, **_kwargs):
        raise OSError("provider unavailable " + PRIVATE)

    monkeypatch.setattr("marvis.llm_client.urlopen", broken_open)
    if stage == "ask":
        assert build_slice_spec_from_utterance("count by channel", ["channel"], wire.client).needs_clarification
    elif stage == "recovery":
        _, metadata = answer_workflow_recovery_message(user_message="why did preparation stop?",
            diagnostic={"code": "workflow_execution_failed", "retryable": True}, client=wire.client)
        assert metadata["fallback"]
    elif stage == "author":
        with pytest.raises(LLMClientError):
            draft_script("task", "calculate revenue minus cost", learning_note=None, llm_factory=lambda: wire.client)
    elif stage == "cross":
        from marvis.feature.errors import FeatureError

        with pytest.raises(FeatureError, match="未完成"):
            recommend_feature_crosses({"debt": "负债", "income": "收入"}, {}, llm_factory=lambda: wire.client)
    else:
        source = _fetch_source(monkeypatch)
        with pytest.raises(LLMClientError):
            distill_learning("methodology", [source.text], [source.requested_url], public_sources=[source], llm_factory=lambda: wire.client)


def test_public_source_query_credentials_are_rejected_before_network(monkeypatch):
    calls = []
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: calls.append(True))
    with pytest.raises(OutboundContextError):
        fetch_public_learning_source("https://example.test/docs?api_key=secret")
    assert calls == []
