import json
import subprocess
from pathlib import Path


STATIC_DIR = Path(__file__).resolve().parents[1] / "marvis" / "static"


def _read_static(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


def _run_report_draft_module(script_body: str) -> None:
    module_url = (STATIC_DIR / "js" / "report-draft-table.js").as_uri()
    script = f"""
import assert from "node:assert/strict";
const draft = await import({json.dumps(module_url)});
{script_body}
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout or "node test failed")


def test_app_wires_editable_report_draft_table_and_confirm_endpoint():
    app_js = _read_static("app.js")
    css = _read_static("css/agent-conversation.css")

    assert 'from "./js/report-draft-table.js"' in app_js
    assert "reportDraftTableHtml" in app_js
    assert "agent/report-draft/confirm" in app_js
    assert "确认并生成报告" in app_js
    assert "data-report-draft-confirm" in _read_static("js/report-draft-table.js")
    assert ".report-draft-table" in css
    assert ".report-draft-input" in css


def test_report_draft_table_pairs_placeholders_with_editable_values():
    _run_report_draft_module(
        r"""
const values = {
  "TEXT:pressure_test_summary": "压力测试显示模型整体稳定。",
  "TEXT:pressure_impact_recommendation": "建议继续监测缺失率较高的数据源。",
  "TEXT:final_validation_conclusion": "模型整体满足验证要求。",
  "TEXT:model_overview": "自营通用T卡为支用环节XGBoost模型。",
  "TEXT:model_scope": "支用环节（支用申请阶段）。",
  "TEXT:sample_audience": "申请支用的用户",
  "TEXT:bad_sample_definition": "MOB6 逾期 >= 30 天（默认假设）",
  "TEXT:good_sample_definition": "MOB6 未逾期",
  "TEXT:model_training_description": "本模型采用浅树 LightGBM，max_depth=1。",
};
const html = draft.reportDraftTableHtml(values, {
  editable: true,
  revision: 0,
  messageId: "draft-1",
});
assert.match(html, /data-report-draft-table="true"/);
assert.match(html, /data-report-draft-editable="true"/);
assert.match(html, /确认并生成报告/);
assert.match(html, /data-report-draft-confirm/);
assert.match(html, /压力测试总结/);
assert.match(html, /压力影响建议/);
assert.match(html, /最终验证结论/);
assert.match(html, /模型概述/);
assert.match(html, /适用范围/);
assert.match(html, /样本人群/);
assert.match(html, /坏样本定义/);
assert.match(html, /好样本定义/);
assert.match(html, /模型训练说明/);
assert.match(html, /TEXT:model_training_description/);
assert.match(html, /浅树 LightGBM/);
assert.match(html, /TEXT:bad_sample_definition/);
assert.match(html, /MOB6 逾期 &gt;= 30 天（默认假设）/);
assert.match(html, /申请支用的用户/);
assert.match(html, /textarea/);
assert.match(html, /type="text"/);
assert.doesNotMatch(html, /TEXT:bad_sample_definition<\/td>/);
const readonly = draft.reportDraftTableHtml(values, { editable: false });
assert.doesNotMatch(readonly, /data-report-draft-confirm/);
assert.match(readonly, /report-draft-readonly/);
assert.equal(
  draft.latestPendingReportDraftMessageId([
    { id: "old", role: "assistant", stage: "word_conclusion_draft", metadata: { draft_values: values } },
    { id: "confirmed", role: "assistant", stage: "word_conclusion_confirmed", metadata: {} },
    { id: "new", role: "assistant", stage: "word_conclusion_draft", metadata: { draft_values: values } },
  ]),
  "new",
);
assert.equal(
  draft.latestPendingReportDraftMessageId([
    { id: "old", role: "assistant", stage: "word_conclusion_draft", metadata: { draft_values: values } },
    { id: "confirmed", role: "assistant", stage: "word_conclusion_confirmed", metadata: {} },
  ]),
  "",
);
"""
    )
