"""English display of platform-templated Agent prose (D-10 follow-up).

Guard: the main journeys' templated messages, rendered by the real presenters
or copied from the platform's own templates, contain no Chinese in the English
UI outside user data (code spans, quoted user values). LLM prose, user turns
and Chinese-mode responses stay as written.
"""

from __future__ import annotations

import re

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marvis.agent import renderers
from marvis.ui_locale import UiLocaleMiddleware, localize_payload, localize_prose_payload, use_ui_locale
from marvis.ui_locale_prose import translate_prose, translate_prose_line

CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
CODE_SPAN = re.compile(r"`[^`]*`")


def _platform_chinese(text: str, user_values: tuple[str, ...] = ()) -> list[str]:
    """Lines that still show Chinese the platform authored."""

    bad = []
    for line in text.split("\n"):
        visible = CODE_SPAN.sub("", line)  # file names, field names, commands
        for value in user_values:
            visible = visible.replace(value, "")
        if CJK.search(visible):
            bad.append(line)
    return bad


def _rendered(name: str, output: dict) -> str:
    text, _tables = getattr(renderers, name)(output)
    return text


RENDERED_JOURNEYS = {
    "join": [
        ("_render_propose_join", {"joins": []}),
        ("_render_execute_join", {"anchor_rows": 100, "rows": 100, "dataset_id": "ds1"}),
    ],
    "feature": [
        ("_render_screen", {
            "selected": ["age", "income"], "leakage": [["bad_flag", 0.91, "x"]],
            "suspected": [["score_v1", 0.52, "y"]], "unusable": [["const_col", ""]],
            "excluded_categorical": [{"column": "city", "cardinality": 30}],
            "nan_labels_dropped": 3, "n_screened": 10,
        }),
        ("_render_select", {"selected": ["age", "income"], "n_in": 5, "dropped_iv": ["a"], "dropped_corr": ["b"]}),
        ("_render_feature_metrics", {"features": ["a", "b"], "metrics": ["iv", "ks"]}),
    ],
    "modeling": [
        ("_render_make_split", {"rows": 100}),
        ("_render_tune", {"n_trials": 10}),
        ("_render_train", {}),
    ],
    "vintage": [("_render_vintage_curve", {"cohorts": ["2026-01", "2026-02"], "curves": {}})],
    "portfolio": [
        ("_render_flow_rate", {"pairs": 3}),
        ("_render_el_estimate", {"loss_state": "charged_off", "total_el": 1234.5}),
        ("_render_portfolio_gate_summary", {"red_flags": []}),
        ("_render_monitor_run", {"overall": "green", "red_flags": [], "yellow_flags": []}),
    ],
    "data": [
        ("_render_export_dataset", {"rows": 10, "columns": 3}),
        ("_render_transform_dataset", {}),
    ],
}

# Lines exactly as the platform writes them (captured from the journeys' turn
# handlers, setup modules and presenters); user values are Latin here.
TEMPLATED_LINES = {
    "join": [
        "我发现 2 个数据文件，先确认每张的**角色与目标列**（样本是锚，只贴列不改行，**1:1**）:",
        "- 提议**样本主表** = `sample.parquet`（目标列 `long_y`）",
        "- 提议**特征表** = `feature_table.parquet`",
        "**拼接执行完成**:已生成拼接结果数据集，锚行 100 → 拼接后 100 行（1:1 保持 ✓）",
        "收到角色与目标列确认，开始生成执行计划。",
    ],
    "feature": [
        "**特征分析**:特征指标 → 可选分箱分析 → 生成特征分析报告",
        "单变量分析已完成。是否有特征需要进一步做分箱分析？可以多选特征并设置 3–20 箱，也可以跳过；确认后才会生成最终报告。",
        "已在「特征指标」里加上按月 PSI（时间列 apply_month）（首月基准、逐月环比），对全部 3 个特征重新计算；特征集合和其他设置不变。",
        "- 本次完成 3 个特征的单变量分析。",
    ],
    "modeling": [
        "开始建模:样本 `sample.parquet`，目标列 `long_y`（坏率 25.67%），切分 `model_flag` train/test/oot=150/60/90，候选特征 3 个。先做泄漏感知特征筛选，随后请确认特征集。",
        "**建模**:配置调参 → 调参 → 训练模型 → 对比实验 → 选择实验",
        "**已选择最终实验**:LightGBM 多分类 实验；平台推荐候选（按 测试集 Macro-AUC 选优），用户已确认。",
        "- 平台标记了过拟合风险（训练/测试 KS 相对差距 0.2954，超过 0.10；训练/OOT KS 相差 0.0690，超过 0.05），交付前需复核模型稳健性。",
        "已按你的要求调整（IV 下限改为 0），并重算「精选特征」。",
    ],
    "validation": [
        "是否继续执行【模型效果&稳定性验证】？点「继续」开始，也可以先接着提问。",
        "字段合同无歧义，已按唯一识别结果自动确认，继续验证。",
        "报告结论已生成，正在直接生成最终 Word 报告。",
    ],
    "strategy": [
        "**单变量候选分析完成**：已分析 4 个字段，得到 6 个可用字段/分箱方法组合。候选证据 `ev-1` 仅处于 `development / unvalidated`，不代表独立验证、采纳或上线。",
        "**规则集评估完成**:合计拒绝率 12.00%，拒绝客群坏率 30.00%；残余通过率 88.00%，通过客群坏率 4.00%。",
        "**策略已在本地采纳**：`strategy-1` v2，资产状态 adopted_local（兼容状态 adopted），退役 1 个旧版本，生成 3 份交付物。本地采纳不代表生产环境已上线。",
    ],
    "vintage": [
        "目标已确认：**标准 Vintage**。请提供 Vintage panel：",
        "参考 MOB 下资产质量最差的是 2026-01（8.00%），最好的是 2026-03（2.00%）。",
    ],
    "portfolio": [
        "已确认桶顺序：current → M1 → charged_off。开始并行分析（流量/迁徙/细分），随后汇总确认。",
        "**组合分析**:流量分析 → 迁徙热力 → 细分画像 → 损失估计 → 组合分析汇总",
        "**组合分析汇总**:0 项数据质量红旗，请确认后生成报告。",
    ],
}


def test_rendered_journey_messages_have_no_platform_chinese_in_english():
    for journey, cases in RENDERED_JOURNEYS.items():
        for name, output in cases:
            english = translate_prose(_rendered(name, output))
            assert _platform_chinese(english) == [], (journey, name, english)


def test_templated_journey_lines_have_no_platform_chinese_in_english():
    for journey, lines in TEMPLATED_LINES.items():
        for line in lines:
            english = translate_prose_line(line)
            assert english is not None, (journey, line)
            assert _platform_chinese(english) == [], (journey, line, english)


def test_values_pass_through_and_untemplated_chinese_stays():
    # Raw captured values keep their exact text (a Chinese business meaning too).
    assert translate_prose_line("age（含义:客户年龄）") == "age (meaning: 客户年龄)"
    assert "bad_flag" in translate_prose_line("- 剔除疑似**泄漏** 1 个（如 bad_flag）")
    # LLM-style prose, short words chained, and partial matches stay Chinese.
    for line in ("建议继续使用模型，同时监控缺失率较高的数据源。", "风险高", "特征通过", "普通结论。"):
        assert translate_prose_line(line) is None
    text = "**特征筛选完成**:从 3 个候选中提议保留 **2** 个特征。\n这是模型自己写的一句话。"
    assert translate_prose(text).split("\n")[1] == "这是模型自己写的一句话。"


def _repair_message() -> dict:
    return {
        "id": "m2",
        "role": "assistant",
        "content": "「特征指标」失败:boom",
        "metadata": {
            "error_diagnostic": {
                "title": "特征 city 是文本（类别）列，数值指标算不了。",
                "summary": "特征 city 是文本（类别）列，数值指标算不了。",
                "cause": "IV、KS、分箱等指标按数值计算；文本列需要先编码，平台不会自行编码。",
                "impact": "特征指标还没有算出来，后续分箱和报告都还没开始。",
                "actions": ["要用文本列，先在数据处理里做编码，再重新发起特征分析。"],
                "recovery_actions": [{"label": "这次先不算文本列 city", "command": "这次先不算文本列 city"}],
                "technical_detail": "ValueError: 无法转换",
            },
            "repair_report": {
                "finding": "拼接键 cust_id 一侧是文本、一侧是数值，平台需要你确认是同一个编号后才能拼接。",
                "action": "按你的确认，把文本和数值编号视为同一标识重新确认拼接；样本和特征表都没有改动。",
                "verification": "已还原。",
                "undo_command": "撤销",
            },
        },
    }


def test_repair_card_prose_translates_but_commands_and_details_do_not():
    with use_ui_locale("en"):
        message = localize_prose_payload({"messages": [_repair_message()]})["messages"][0]
    diagnostic = message["metadata"]["error_diagnostic"]
    for key in ("title", "summary", "cause", "impact"):
        assert _platform_chinese(diagnostic[key]) == [], diagnostic[key]
    assert _platform_chinese(diagnostic["actions"][0]) == []
    assert diagnostic["recovery_actions"][0]["label"] == "Skip text column city this time"
    assert diagnostic["recovery_actions"][0]["command"] == "这次先不算文本列 city"
    assert diagnostic["technical_detail"] == "ValueError: 无法转换"
    report = message["metadata"]["repair_report"]
    assert _platform_chinese(report["finding"] + report["action"] + report["verification"]) == []
    assert report["undo_command"] == "撤销"
    assert message["content"] == "“Feature metrics” failed: boom"


def test_user_turns_and_chinese_requests_are_untouched():
    user = {"id": "u1", "role": "user", "content": "收到确认，开始执行下一步。", "metadata": {}}
    with use_ui_locale("en"):
        assert localize_prose_payload([user])[0] == user
    payload = {"messages": [_repair_message()]}
    assert localize_payload(payload) is payload


def _app():
    app = FastAPI()

    @app.get("/api/tasks/t1/agent/messages")
    def messages():
        return {"messages": [
            {"id": "m1", "role": "assistant", "content": "**训练完成**。",
             "metadata": {"tables": [{"title": "Agent 特征建议", "columns": ["特征"], "rows": [["年龄"]]}]}},
            {"id": "u1", "role": "user", "content": "继续", "metadata": {}},
        ]}

    app.add_middleware(UiLocaleMiddleware)
    return app


def test_canonical_english_response_keeps_labels_and_serves_english_prose():
    client = TestClient(_app())
    headers = {"X-MARVIS-UI-Locale": "en", "X-MARVIS-UI-Canonical": "1"}
    body = client.get("/api/tasks/t1/agent/messages", headers=headers).json()
    assistant, user = body["messages"]
    assert assistant["content"] == "**Training done**."
    assert assistant["metadata"]["tables"][0]["title"] == "Agent 特征建议"  # canonical label
    assert user["content"] == "继续"
    chinese = client.get("/api/tasks/t1/agent/messages", headers={"X-MARVIS-UI-Canonical": "1"}).json()
    assert chinese["messages"][0]["content"] == "**训练完成**。"


def test_field_names_are_never_translated_even_when_they_equal_a_label():
    # 状态 and 收入 are platform labels elsewhere, but here they are field names.
    assert translate_prose_line("生成 2 个字段：状态、收入") == "Derived 2 fields: 状态, 收入"
    assert translate_prose_line("重命名 1 个：状态") == "Renamed 1: 状态"
    assert translate_prose_line("状态（含义:客户状态）") == "状态 (meaning: 客户状态)"
    assert translate_prose_line("- 剔除疑似**泄漏** 1 个（如 状态）") == "- Removed suspected **leakage**: 1 (e.g. 状态)"
    # The group-by slot is a field; only the metric slot is marked {1|t}.
    assert translate_prose_line("将按〔状态〕统计〔数量〕，确认？") == "Compute [Count] by [状态]. Confirm?"


def test_only_marked_slots_translate_platform_values():
    # Step titles and phase names sit in {x|t} slots.
    assert translate_prose_line("「特征指标」失败:boom") == "“Feature metrics” failed: boom"
    assert translate_prose_line("**特征**:切分样本 → 选择建模规格") == "**Feature**: Split sample → Choose modeling spec"
    # An unmarked open slot never takes a translated label in; the line stays.
    from marvis.ui_locale_prose import _compile

    template = _compile("删除 {0} 个：{1}", "Dropped {0}: {1}")
    assert template.nested == (False, False)
    assert _compile("**{0}**:{1}", "**{0}**: {1}") is None  # layout slots must be marked


def test_dataset_columns_keep_user_dataset_names_and_raw_headers_are_shared():
    from marvis import ui_locale

    table = {"columns": ["数据集", "状态"], "rows": [["训练集", "通过"]]}
    rows = ui_locale._localize_table_rows(table["rows"], table)
    assert rows == [["训练集", ui_locale.EN_CELL_LABELS["通过"]]]
    typed = {**table, "column_specs": [{"kind": "split-badge"}, {}]}
    assert ui_locale._localize_table_rows(typed["rows"], typed)[0][0] != "训练集"
    # One raw-header list for server and browser (static/js/raw-columns.js).
    for header in ("特征名", "特征名称", "字段名", "验证字段", "feature name", "column"):
        assert header.casefold() in ui_locale._RAW_CELL_COLUMNS
    named = {"columns": ["特征名称", "状态"], "rows": [["通过", "通过"]]}
    assert ui_locale._localize_table_rows(named["rows"], named)[0][0] == "通过"


def test_english_rewrite_skips_bodies_without_chinese():
    from marvis.ui_locale import _localized_body

    body = b'{"items":[{"id":"x","status":"done"}]}'
    assert _localized_body(body) is body
    assert _localized_body(body, prose_only=True) is body
    escaped = b'{"messages":[{"role":"assistant","content":"**\\u8bad\\u7ec3\\u5b8c\\u6210**\\u3002","metadata":{}}]}'
    assert _localized_body(escaped, prose_only=True) != escaped


def test_validation_stage_summaries_keep_the_report_language(monkeypatch):
    """Stage summaries feed the Word conclusion draft, so an English message
    must not switch them (and through them the report) to English."""

    from types import SimpleNamespace

    from marvis.agent import service
    from marvis.ui_locale import REPLY_LANGUAGE, use_reply_language

    seen = []

    class Client:
        def complete(self, **_kwargs):
            seen.append(REPLY_LANGUAGE.get())
            return "材料扫描完成。"

    monkeypatch.setattr(service, "_client", lambda _profile: Client())
    monkeypatch.setattr("marvis.llm_outbound_scan.scan_prompt", lambda evidence, **_kw: {"scan": evidence})
    task = SimpleNamespace(validation_workflow_version=2, id="t1")
    with use_reply_language("en"):
        content, _metadata = service.summarize_stage(
            task=task, stage="scan", evidence={}, model_profile={"model_id": "m"}, fallback="fallback",
        )
        assert REPLY_LANGUAGE.get() == "en"  # restored for the rest of the request
    assert seen == [None]
    assert content
