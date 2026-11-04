"""D-10 (batch 1b): request-scoped UI locale, English labels and reply language."""

from __future__ import annotations

import json
import logging

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.testclient import TestClient

from marvis import ui_locale
from marvis.llm_client import OpenAICompatibleLLMClient
from marvis.llm_outbound import outbound_prompt
from marvis.llm_prompts import AGENT_SYSTEM_PROMPT, WORD_CONCLUSION_SYSTEM_PROMPT
from marvis.ui_locale import (
    UiLocaleMiddleware,
    agent_text_locale,
    current_reply_language,
    current_ui_locale,
    detect_text_language,
    localize_payload,
    normalize_locale,
    reply_language_instruction,
    ui_label,
    use_reply_language,
    use_ui_locale,
    with_reply_language,
)
from marvis.ui_locale_en import EN_CELL_LABELS, EN_LABELS


def test_locale_contextvars_default_to_chinese_and_nest():
    assert current_ui_locale() == "zh"
    assert current_reply_language() is None
    assert agent_text_locale() == "zh"
    with use_ui_locale("en-US"):
        assert current_ui_locale() == "en"
        assert agent_text_locale() == "en"
        with use_reply_language("zh"):
            assert agent_text_locale() == "zh"
        assert agent_text_locale() == "en"
    assert current_ui_locale() == "zh"
    assert normalize_locale("fr") == "zh"
    assert normalize_locale(b"EN") == "en"


def test_detect_text_language_matches_the_frontend_rule():
    assert detect_text_language("帮我看下 KS") == "zh"
    assert detect_text_language("check PSI 0.1") == "en"
    assert detect_text_language("0.30") is None
    assert detect_text_language("") is None


def test_ui_label_translates_only_in_english_and_falls_back_with_one_log(caplog):
    assert ui_label("特征") == "特征"
    with use_ui_locale("en"):
        assert ui_label("特征") == EN_LABELS["特征"]
        assert ui_label("命中率") == "Match rate"
        assert ui_label(3.14) == 3.14
        with caplog.at_level(logging.DEBUG, logger="marvis.ui_locale"):
            assert ui_label("一个没有英文的平台词") == "一个没有英文的平台词"
            assert ui_label("一个没有英文的平台词") == "一个没有英文的平台词"
    misses = [record for record in caplog.records if "一个没有英文的平台词" in record.getMessage()]
    assert len(misses) == 1


def _message():
    return {
        "id": "m1",
        "role": "assistant",
        "content": "**特征筛选完成**：保留 23 个特征",
        "metadata": {
            "tables": [{
                "title": "Agent 特征建议",
                "columns": ["特征", "Agent建议", "IV"],
                "rows": [["年龄", "推荐", 0.31], ["通过", "慎用", 0.02]],
            }],
            "quick_replies": [{"label": "继续", "command": "继续"}],
            "confirmation_snapshot": {"expected_plan_fingerprint": "特征"},
            "draft_values": {"TEXT:final_validation_conclusion": "特征"},
        },
    }


def test_localize_payload_rewrites_labels_but_not_values_content_or_snapshots():
    payload = {"messages": [_message()], "incremental": False}
    assert localize_payload(payload) is payload  # Chinese requests pass through.

    english = localize_payload(payload, locale="en")
    message = english["messages"][0]
    table = message["metadata"]["tables"][0]
    assert table["title"] == "Agent feature suggestions"
    assert table["columns"] == ["Feature", "Agent suggestion", "IV"]
    # Cells: curated status words only; user field names and numbers stay.
    assert table["rows"][0] == ["年龄", EN_CELL_LABELS["推荐"], 0.31]
    assert table["rows"][1][0] == "通过"  # a user field name, not a status
    assert table["rows"][1][1] == EN_CELL_LABELS["慎用"]
    assert table["rows"][1][2] == 0.02
    assert message["metadata"]["quick_replies"][0] == {"label": "Continue", "command": "继续"}
    assert message["content"] == "**特征筛选完成**：保留 23 个特征"
    assert message["metadata"]["confirmation_snapshot"] == {"expected_plan_fingerprint": "特征"}
    assert message["metadata"]["draft_values"] == {"TEXT:final_validation_conclusion": "特征"}
    # The stored payload is never mutated.
    assert payload["messages"][0]["metadata"]["tables"][0]["title"] == "Agent 特征建议"


def test_platform_table_enums_require_column_semantics_and_keep_raw_names():
    payload = _message()
    payload["metadata"]["tables"] = [
        {
            "columns": ["名称", "状态", "文件名", "业务备注"],
            "rows": [["通过", "通过", "推荐", "推荐"]],
        },
        {
            "columns": ["字段", "自定义标签"],
            "column_specs": [{"kind": "enum"}, {"kind": "status"}],
            "rows": [["通过", "警告"]],
        },
        {"rows": [["通过", "推荐"]]},
    ]
    tables = localize_payload(payload, locale="en")["metadata"]["tables"]
    assert tables[0]["rows"] == [["通过", "Pass", "推荐", "推荐"]]
    assert tables[1]["rows"] == [["通过", "Warning"]]
    assert tables[2]["rows"] == [["通过", "推荐"]]
    assert payload["metadata"]["tables"][0]["rows"][0][1] == "通过"


def test_preview_placeholder_follows_ui_language_and_escapes_details():
    from marvis.api_report_helpers import report_preview_unavailable_html

    with use_ui_locale("en"), use_reply_language("zh"):
        html = report_preview_unavailable_html("报告尚未生成，请先确认报告结论并等待生成完成。")
        assert '<html lang="en">' in html
        assert "Report preview unavailable" in html
        assert "Confirm the report conclusions" in html
        assert not any("\u3400" <= char <= "\u9fff" for char in html)
        assert "A &lt; B &amp; C" in report_preview_unavailable_html("A < B & C")
    with use_ui_locale("zh"), use_reply_language("en"):
        html = report_preview_unavailable_html("报告尚未生成。")
        assert '<html lang="zh-CN">' in html
        assert "报告尚未生成。" in html


def test_preview_iframe_locale_query_is_scoped_and_header_takes_precedence():
    from fastapi.responses import HTMLResponse
    from marvis.api_report_helpers import report_preview_unavailable_html

    app = FastAPI()

    @app.get("/api/tasks/t1/report/preview")
    def preview():
        return HTMLResponse(report_preview_unavailable_html(None))

    @app.get("/api/locale")
    def locale_probe():
        return {"locale": current_ui_locale()}

    app.add_middleware(UiLocaleMiddleware)
    client = TestClient(app)
    english = client.get("/api/tasks/t1/report/preview?ui_locale=en")
    assert english.status_code == 200
    assert '<html lang="en">' in english.text
    chinese = client.get(
        "/api/tasks/t1/report/preview?ui_locale=en", headers={"X-MARVIS-UI-Locale": "zh"},
    )
    assert '<html lang="zh-CN">' in chinese.text
    assert client.get("/api/locale?ui_locale=en").json() == {"locale": "zh"}


def test_localize_payload_translates_plan_step_titles_only():
    plan = {"plan": {"goal": "数据拼接", "steps": [{
        "id": "s1", "title": "拼接诊断", "tool_ref": {"tool": "propose_join"},
        "inputs": {"title": "拼接诊断"}, "confirmation_snapshot": {"title": "拼接诊断"},
    }]}}
    english = localize_payload(plan, locale="en")
    step = english["plan"]["steps"][0]
    assert step["title"] == "Join diagnostics"
    assert english["plan"]["goal"] == "数据拼接"  # the user's goal is data
    assert step["inputs"] == {"title": "拼接诊断"}
    assert step["confirmation_snapshot"] == {"title": "拼接诊断"}


def _probe_app():
    app = FastAPI()
    seen: dict[str, object] = {}

    def record_background():
        seen["background"] = (current_ui_locale(), current_reply_language())

    @app.post("/api/tasks/t1/agent/messages")
    def post_message(request: Request, background_tasks: BackgroundTasks) -> dict:
        seen["endpoint"] = (current_ui_locale(), current_reply_language())
        background_tasks.add_task(record_background)
        return {"messages": [_message()]}

    @app.get("/api/tasks/t1/agent/messages")
    def get_messages() -> dict:
        return {"messages": [_message()]}

    app.add_middleware(UiLocaleMiddleware)
    return app, seen


def test_middleware_sets_locale_from_header_and_reply_language_from_body():
    app, seen = _probe_app()
    client = TestClient(app)
    response = client.post(
        "/api/tasks/t1/agent/messages",
        json={"content": "Please continue with the join"},
        headers={"X-MARVIS-UI-Locale": "zh"},
    )
    assert response.status_code == 200
    assert seen["endpoint"] == ("zh", "en")
    assert seen["background"] == ("zh", "en")
    # Chinese UI: response untouched.
    assert response.json()["messages"][0]["metadata"]["tables"][0]["title"] == "Agent 特征建议"


def test_middleware_typed_text_beats_the_hint_and_english_responses_are_localized():
    app, seen = _probe_app()
    client = TestClient(app)
    # A Chinese message in the English UI gets a Chinese reply, whatever the
    # (possibly stale or global) hint says.
    response = client.post(
        "/api/tasks/t1/agent/messages",
        json={"content": "帮我把这两张表拼起来"},
        headers={"X-MARVIS-UI-Locale": "en", "X-MARVIS-Reply-Language": "en"},
    )
    assert seen["endpoint"] == ("en", "zh")
    assert seen["background"] == ("en", "zh")
    table = response.json()["messages"][0]["metadata"]["tables"][0]
    assert table["columns"] == ["Feature", "Agent suggestion", "IV"]
    assert int(response.headers["content-length"]) == len(response.content)

    plain = client.get("/api/tasks/t1/agent/messages")
    assert plain.json()["messages"][0]["metadata"]["tables"][0]["title"] == "Agent 特征建议"


def test_middleware_hint_fills_in_for_command_chips_and_bodies_without_language():
    app, seen = _probe_app()
    client = TestClient(app)
    # A quick-reply chip sends a Chinese platform command: the hint stays.
    client.post(
        "/api/tasks/t1/agent/messages",
        json={"content": "继续"},
        headers={"X-MARVIS-UI-Locale": "en", "X-MARVIS-Reply-Language": "en", "X-MARVIS-Reply-Source": "quick-reply"},
    )
    assert seen["endpoint"] == ("en", "en")
    # Nothing detectable in the body (digits only): the hint decides.
    client.post("/api/tasks/t1/agent/messages", json={"content": "3"}, headers={"X-MARVIS-Reply-Language": "zh"})
    assert seen["endpoint"] == ("zh", "zh")
    # No hint and nothing detectable: unknown.
    client.post("/api/tasks/t1/agent/messages", json={"content": "3"})
    assert seen["endpoint"] == ("zh", None)
    # Without the chip marker, the same Chinese command reads as Chinese.
    client.post(
        "/api/tasks/t1/agent/messages", json={"content": "继续"},
        headers={"X-MARVIS-UI-Locale": "en", "X-MARVIS-Reply-Language": "en"},
    )
    assert seen["endpoint"] == ("en", "zh")


def test_canonical_ui_response_keeps_labels_without_changing_language_context():
    app, seen = _probe_app()
    client = TestClient(app)
    response = client.post(
        "/api/tasks/t1/agent/messages",
        json={"content": "Please continue"},
        headers={
            "X-MARVIS-UI-Locale": "en",
            "X-MARVIS-Reply-Language": "zh",
            "X-MARVIS-UI-Canonical": "1",
        },
    )
    assert response.status_code == 200
    assert seen["endpoint"] == ("en", "en")  # the typed English beats the stale hint
    assert seen["background"] == ("en", "en")
    assert response.json()["messages"][0] == _message()
    assert int(response.headers["content-length"]) == len(response.content)
    localized = client.get(
        "/api/tasks/t1/agent/messages",
        headers={"X-MARVIS-UI-Locale": "en", "X-MARVIS-UI-Canonical": "0"},
    )
    table = localized.json()["messages"][0]["metadata"]["tables"][0]
    assert table["title"] == "Agent feature suggestions"
    assert table["columns"] == ["Feature", "Agent suggestion", "IV"]
    assert table["rows"][0][2] == 0.31


def test_reply_language_rule_only_for_conversational_prompts_with_known_language():
    base = AGENT_SYSTEM_PROMPT.text
    assert with_reply_language(base, "AGENT_SYSTEM_PROMPT") == base
    with use_reply_language("en"):
        ruled = with_reply_language(base, "AGENT_SYSTEM_PROMPT")
        assert ruled.startswith(base)
        assert ruled.endswith(reply_language_instruction("en"))
        assert "English" in ruled and "overrides any earlier instruction" in ruled
        assert with_reply_language(base, None) == ruled  # matched by text
        report = WORD_CONCLUSION_SYSTEM_PROMPT.text
        assert with_reply_language(report, "WORD_CONCLUSION_SYSTEM_PROMPT") == report
    assert reply_language_instruction(None) == ""


class _JsonResponse:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        payload = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        return iter([json.dumps(payload).encode("utf-8")])

    def read(self):
        return b"".join(self)


def test_llm_client_appends_reply_language_to_the_governed_system_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _JsonResponse()

    monkeypatch.setattr("marvis.llm_client.urlopen", fake_urlopen)
    client = OpenAICompatibleLLMClient({
        "api_base_url": "https://llm.example.test", "model_name": "m", "api_key": "k",
    })
    with use_reply_language("en"):
        client.complete(
            system_prompt=AGENT_SYSTEM_PROMPT.text,
            user_prompt=outbound_prompt({}),
            prompt_name="AGENT_SYSTEM_PROMPT",
            stream=False,
        )
    system = captured["payload"]["messages"][0]["content"]
    assert system.startswith(AGENT_SYSTEM_PROMPT.text)
    assert system.endswith(reply_language_instruction("en"))

    client.complete(
        system_prompt=AGENT_SYSTEM_PROMPT.text,
        user_prompt=outbound_prompt({}),
        prompt_name="AGENT_SYSTEM_PROMPT",
        stream=False,
    )
    assert captured["payload"]["messages"][0]["content"] == AGENT_SYSTEM_PROMPT.text


def test_english_label_tables_have_no_chinese_values():
    for table in (EN_LABELS, EN_CELL_LABELS):
        for key, value in table.items():
            assert value and not ui_locale._CJK_RE.search(value), key
