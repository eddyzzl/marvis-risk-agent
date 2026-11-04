"""Result and report cards (2026-10-14 redesign, batch 6a: D12, D13, D8).

Evidence is a card in the conversation instead of a separate 证据 view: the
card only rearranges the platform's metric table sections (values copied
verbatim, statuses only from thresholds the payload carries) and keeps every
original table under 全部指标明细. The report draft is a card under its
message with the same data attributes as the old report view, so F01
autosave/conflict and F02 confirmation run through unchanged handlers.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"

# Shape of a real validation payload (report-fields metric_table_sections).
SECTIONS = [
    {"title": "样本情况", "tables": [{"title": "样本总体分布", "layout": "table", "headers": ["数据集", "样本量"], "rows": [["train", "5,082"]]}]},
    {
        "title": "整体效果&稳定性",
        "tables": [{
            "title": "整体效果&稳定性",
            "layout": "kpi_cards",
            "headers": ["数据集", "时间范围", "样本量", "逾期率", "坏样本量", "KS(%)", "AUC(%)", "5%头部lift", "5%尾部lift", "PSI"],
            "column_specs": [{"kind": "split-badge"}, {"kind": "period"}, {"kind": "databar"}, {"kind": "percent-heat"}, {"kind": "databar"}, {"kind": "databar-primary"}, {"kind": "databar"}, {"kind": "databar"}, {"kind": "databar"}, {"kind": "psi", "thresholds": [0.02, 0.1]}],
            "rows": [
                ["train", "20250102-20250930", "5,082", "2.56%", "130", "38.5", "75.7", "0.00", "5.54", "BASE"],
                ["test", "20250102-20250930", "2,180", "2.66%", "58", "35.8", "69.7", "0.00", "2.76", "0.010"],
                ["oot", "20251001-20260310", "4,818", "3.45%", "166", "35.4", "71.9", "0.00", "1.69", "0.042"],
            ],
        }],
    },
    {
        "title": "分月效果&稳定性",
        "tables": [{
            "title": "分月效果&稳定性",
            "layout": "trend_table",
            "headers": ["月份", "样本量", "KS(%)", "PSI(首月基准)", "PSI(尾月基准)", "PSI参考月"],
            "column_specs": [{"kind": "text"}, {"kind": "databar"}, {"kind": "databar-primary"}, {"kind": "psi", "thresholds": [0.02, 0.1]}, {"kind": "psi", "thresholds": [0.02, 0.1]}, {"kind": "text"}],
            "rows": [
                ["202501", "458", "97.3", "BASE", "0.031", "-"],
                ["202502", "460", "40.5", "0.120", "0.015", "202501"],
                ["202503", "470", "35.0", "0.050", "BASE", "202501"],
            ],
        }],
    },
    # marvis/metric_tables.py: 分箱排序性 holds the train-binned tables; the
    # independent deciles come as their own 独立10等分分箱 section.
    {
        "title": "分箱排序性",
        "tables": [
            {"title": "Train(按照train分箱)", "layout": "table", "headers": ["Train(按照train分箱)", "样本总数", "累计占比", "逾期数量", "逾期率", "累计逾期率"], "rows": [["[-inf,0.011]", "430", "8.46%", "0", "0.00%", "0.00%"], ["(0.011,0.02]", "500", "18%", "5", "1.00%", "0.50%"]]},
            {"title": "OOT(按照train分箱)", "layout": "table", "headers": ["OOT(按照train分箱)", "样本总数", "累计占比", "逾期数量", "逾期率", "累计逾期率"], "rows": [["[-inf,0.011]", "456", "9.46%", "1", "0.22%", "0.22%"], ["(0.011,0.02]", "480", "19%", "9", "1.88%", "1.05%"]]},
        ],
    },
    {
        "title": "独立10等分分箱",
        "tables": [
            {"title": "Train(独立10等分)", "layout": "table", "headers": ["Train(独立10等分)", "样本总数", "累计占比", "逾期数量", "逾期率", "累计逾期率"], "rows": [["[0,0.012]", "508", "10%", "1", "0.20%", "0.20%"]]},
            {"title": "OOT(独立10等分)", "layout": "table", "headers": ["OOT(独立10等分)", "样本总数", "累计占比", "逾期数量", "逾期率", "累计逾期率"], "rows": [["[0,0.01]", "482", "10%", "2", "0.41%", "0.41%"]]},
        ],
    },
    {
        "title": "压力测试",
        "tables": [{
            "title": "压力测试",
            "layout": "table",
            "headers": ["类别", "状态", "KS_baseline", "KS_after", "KS_delta", "PSI vs baseline"],
            "column_specs": [{"kind": "text"}, {"kind": "text"}, {"kind": "text"}, {"kind": "databar"}, {"kind": "psi", "thresholds": [0.01, 0.03]}, {"kind": "psi", "thresholds": [0.02, 0.1]}],
            "rows": [
                ["内部特征", "完成", "0.3539", "0.2740", "-0.0799", "4.9903"],
                ["睿智", "完成", "0.3539", "0.3797", "0.0258", "0.2150"],
                ["腾讯云", "完成", "0.3539", "0.3540", "0.0001", "0.0049"],
            ],
        }],
    },
]


def _run_module(script: str):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _card_model():
    return _run_module(
        f"""
import {{ validationMetricCardModel }} from "./marvis/static/js/result-cards.js";
process.stdout.write(JSON.stringify(validationMetricCardModel({json.dumps(SECTIONS, ensure_ascii=False)})));
"""
    )


def test_tiles_copy_platform_values_and_only_psi_gets_a_status():
    model = _card_model()
    tiles = {tile["key"]: tile for tile in model["tiles"]}
    assert [tile["key"] for tile in model["tiles"]] == ["ks", "auc", "psi"]
    assert (tiles["ks"]["label"], tiles["ks"]["value"], tiles["ks"]["sub"]) == ("KS(%) · OOT", "35.4", "训练 38.5 · 测试 35.8")
    assert (tiles["auc"]["value"], tiles["auc"]["sub"]) == ("71.9", "训练 75.7 · 测试 69.7")
    # KS and AUC carry no threshold in the payload, so no verdict is invented.
    assert "status" not in tiles["ks"] and "status" not in tiles["auc"]
    # PSI 0.042 sits between the payload's 0.02 and 0.1: the platform's 可接受.
    assert (tiles["psi"]["value"], tiles["psi"]["status"], tiles["psi"]["tone"]) == ("0.042", "可接受", "warn")
    assert tiles["psi"]["sub"] == "逐月最高 0.120 · 202502"


def test_monthly_and_stress_rows_follow_the_payload_thresholds():
    model = _card_model()
    monthly = model["monthly"]
    # "PSI参考月" is typed text in the payload: a month label, not a PSI.
    assert monthly["psiHeaders"] == ["PSI(首月基准)", "PSI(尾月基准)"]
    assert (monthly["critical"], monthly["warn"], monthly["criticalAt"]) == (1, 2, 0.1)
    assert [[entry["value"], entry["tier"]] for entry in monthly["items"][1]["psi"]] == [["0.120", "critical"], ["0.015", "stable"]]
    stress = model["stress"]
    assert stress["largestDrop"]["name"] == "内部特征"
    assert [(item["name"], item["deltaTier"], item["psiTier"]) for item in stress["items"]] == [
        ("内部特征", "critical", "critical"),
        ("睿智", "warn", "critical"),
        ("腾讯云", "stable", "stable"),
    ]


def test_binning_chart_keeps_each_binning_mode_and_the_bin_bad_rate():
    model = _card_model()
    modes = model["binning"]
    assert [mode["label"] for mode in modes] == ["按照train分箱", "独立10等分"]
    assert [(series["name"], series["raw"]) for series in modes[0]["series"]] == [
        ("Train", ["0.00%", "1.00%"]),
        ("OOT", ["0.22%", "1.88%"]),
    ]
    # Plotted as fractions (parseNumeric), labelled with the payload's own text.
    assert [round(value, 6) for value in modes[0]["series"][1]["values"]] == [0.0022, 0.0188]
    # Independent deciles: each split keeps its own interval labels.
    assert [(series["name"], series["labels"]) for series in modes[1]["series"]] == [
        ("Train", ["[0,0.012]"]),
        ("OOT", ["[0,0.01]"]),
    ]


def test_card_markup_keeps_every_original_table_and_marks_itself_a_result():
    data = _run_module(
        f"""
import {{ scoreEvidenceTile, validationMetricCardHtml, validationMetricCardModel }} from "./marvis/static/js/result-cards.js";
const model = validationMetricCardModel({json.dumps(SECTIONS, ensure_ascii=False)});
const details = '<section class="metric-table-section" data-original="all-sections"></section>';
const html = validationMetricCardHtml(model, {{
  title: "<A卡>",
  scoreTile: scoreEvidenceTile({{ kind: "pmml", data: {{ status: "pass", input_row_count: 12080, success_count: 12080, failure_count: 0, null_count: 0 }} }}),
  detailsHtml: details,
}});
process.stdout.write(JSON.stringify({{ html, empty: validationMetricCardHtml(validationMetricCardModel([]), {{}}) }}));
"""
    )
    html = data["html"]
    assert 'data-agent-result="true"' in html
    assert "&lt;A卡&gt; 的效果与稳定性" in html and "<A卡>" not in html
    assert '<section class="metric-table-section" data-original="all-sections"></section>' in html
    # 2026-10-19 user feedback: everything open and nothing twice -- no
    # 全部指标明细 disclosure, no folded sections, no duplicate bin table view
    # (the full bin tables follow the card as their own sections).
    assert "全部指标明细" not in html
    assert "<details" not in html
    assert 'data-result-view="table"' not in html and "result-chart-table" not in html
    assert 'data-result-bin-mode="1"' in html
    assert "逐月稳定性" in html and "压力测试，逐个剔除数据源" in html
    assert "PMML 打分，成功行数" in html and "12,080" in html and "通过" in html
    # No invented KS/AUC verdict words anywhere on the card.
    for word in ("达标", "高于 0.30", "不达标"):
        assert word not in html
    assert "result-tiles" not in data["empty"]


def test_score_tile_uses_the_platform_status():
    data = _run_module(
        """
import { scoreEvidenceTile } from "./marvis/static/js/result-cards.js";
process.stdout.write(JSON.stringify([
  scoreEvidenceTile({ kind: "pmml", data: { status: "fail", input_row_count: 10, success_count: 7, failure_count: 3, null_count: 0 } }),
  scoreEvidenceTile({ kind: "pmml", data: { status: "failed", input_row_count: 10, success_count: 7, failure_count: 3, null_count: 0 } }),
  scoreEvidenceTile({ kind: "reproducibility", data: { sample_size: 2000, summary: { status: "pass", mismatch_count: 0 } } }),
  scoreEvidenceTile({ kind: "pmml", data: {} }),
  scoreEvidenceTile(null),
  scoreEvidenceTile({ kind: "reproducibility", data: { summary: { status: "review", mismatch_count: 60 } } }),
  scoreEvidenceTile({ kind: "reproducibility", data: { summary: { status: "fail", mismatch_count: 900 } } }),
  scoreEvidenceTile({ kind: "reproducibility", data: { summary: { status: "unknown" } } }),
  scoreEvidenceTile({ kind: "pmml", data: { status: "pass", input_row_count: null, success_count: null } }),
  scoreEvidenceTile({ kind: "pmml", data: { status: "running", input_row_count: 5, success_count: 5 } }),
]));
"""
    )
    # PMML scoring is pass | failed (marvis/validation/results.py); other words get no status.
    assert (data[0]["status"], data[0]["tone"], data[0]["value"]) == ("", "", "7")
    assert (data[1]["status"], data[1]["tone"], data[1]["value"]) == ("未通过", "bad", "7")
    assert (data[2]["label"], data[2]["value"], data[2]["status"]) == ("分数一致性", "0 条不一致", "通过")
    assert data[3] is None and data[4] is None
    # Consistency is pass | review | fail: review is 需复核, never 未通过.
    assert (data[5]["status"], data[5]["tone"]) == ("需复核", "warn")
    assert (data[6]["status"], data[6]["tone"]) == ("未通过", "bad")
    assert data[7] is None
    # Missing counts give no tile rather than "共 0 行".
    assert data[8] is None
    assert (data[9]["status"], data[9]["tone"]) == ("", "")


def test_report_card_opens_the_required_items_and_keeps_every_field():
    data = _run_module(
        """
import { REPORT_DRAFT_FIELDS, reportDraftCardHtml, reportDraftPreviewHtml } from "./marvis/static/js/report-draft-table.js";
const values = {
  "TEXT:final_validation_conclusion": "区分度良好。!!2 月 PSI 偏高!!，<需关注>",
  "TEXT:pressure_test_summary": "内部特征剔除后 KS 下降。",
  "TEXT:model_overview": "支用环节模型。",
};
const editable = reportDraftCardHtml(values, { editable: true, taskId: "t1", messageId: "m1", revision: 3, title: "T卡", state: { status: "已保存草稿" } });
const readonly = reportDraftCardHtml(values, { editable: false, taskId: "t1", title: "T卡" });
process.stdout.write(JSON.stringify({ editable, readonly, keys: REPORT_DRAFT_FIELDS.map((field) => field.key), preview: reportDraftPreviewHtml(values["TEXT:final_validation_conclusion"]), empty: reportDraftPreviewHtml("  ") }));
"""
    )
    html = data["editable"]
    for attr in (
        'data-report-draft-table="true"',
        'data-report-draft-task-id="t1"',
        'data-report-draft-message-id="m1"',
        'data-report-revision="3"',
        'data-report-draft-editable="true"',
        "data-report-draft-save-status",
        "data-report-draft-feedback",
        "data-report-draft-save",
        "data-report-draft-confirm",
        "data-report-draft-expand",
    ):
        assert attr in html, attr
    # Every field stays in the DOM so collectReportDraftValues sends the full set.
    for key in data["keys"]:
        assert f'data-report-draft-key="{key}"' in html, key
    # The three required items come first, the other six fold into one line.
    order = [html.index(f'data-report-card-row="{key}"') for key in (
        "TEXT:final_validation_conclusion", "TEXT:pressure_test_summary", "TEXT:pressure_impact_recommendation",
    )]
    assert order == sorted(order)
    assert html.index('data-report-card-row="TEXT:pressure_impact_recommendation"') < html.index("data-report-card-more-toggle")
    assert "其余 6 项：模型概述、适用范围、样本人群、坏样本定义、好样本定义、模型训练说明" in html
    assert "5 项还没填" in html
    assert "确认只表示同意生成文档，不代表模型通过验证" in html
    assert "T卡 的报告文案" in html
    # Failures render red-bold in the preview; everything else is escaped.
    assert data["preview"] == '<p>区分度良好。<strong class="agent-finding-fail">2 月 PSI 偏高</strong>，&lt;需关注&gt;</p>'
    assert data["empty"] == '<span class="report-card-empty">待补充</span>'
    readonly = data["readonly"]
    assert 'data-report-draft-editable="false"' in readonly
    assert "data-report-draft-key" not in readonly and "data-report-draft-confirm" not in readonly
    assert "5 项留空" in readonly


def test_validation_tabs_are_gone_and_the_card_is_slotted_into_the_conversation():
    index_html = (STATIC / 'index.html').read_text(encoding='utf-8')
    assert 'validationWorkspaceNav' not in index_html
    assert 'data-validation-view' not in index_html


def test_monthly_summary_needs_a_computed_psi_and_a_payload_threshold():
    sections = [
        {"title": "分月效果&稳定性", "tables": [{
            "title": "分月效果&稳定性",
            "layout": "trend_table",
            "headers": ["月份", "KS(%)", "PSI(首月基准)", "PSI(尾月基准)"],
            "column_specs": [{"kind": "text"}, {"kind": "databar"}, {"kind": "psi", "thresholds": [0.02, 0.1]}, {"kind": "psi", "thresholds": [0.02, 0.1]}],
            "rows": [["202501", "40.1", "BASE", "BASE"]],
        }]},
    ]
    no_thresholds = [
        {"title": "分月效果&稳定性", "tables": [{
            "title": "分月效果&稳定性",
            "layout": "trend_table",
            "headers": ["月份", "KS(%)", "PSI(首月基准)"],
            "column_specs": [{"kind": "text"}, {"kind": "databar"}, {"kind": "psi"}],
            "rows": [["202501", "40.1", "0.300"]],
        }]},
    ]
    data = _run_module(
        f"""
import {{ validationMetricCardHtml, validationMetricCardModel }} from "./marvis/static/js/result-cards.js";
const html = (sections) => validationMetricCardHtml(validationMetricCardModel(sections), {{}});
process.stdout.write(JSON.stringify({{ base: html({json.dumps(sections, ensure_ascii=False)}), bare: html({json.dumps(no_thresholds, ensure_ascii=False)}) }}));
"""
    )
    for html in data.values():
        assert "逐月稳定性" in html
        # No computed PSI, or no threshold from the payload: no verdict at all.
        for word in ("各月稳定", "漂移显著", "可接受区间", "PSI 阈值来自平台"):
            assert word not in html
    assert 'data-tone="bad"' not in data["bare"]


def _tables_harness(cases: str) -> dict:
    from tests.javascript_source import slice_function as _slice
    app_js = (STATIC / "app.js").read_text(encoding="utf-8")
    metric_tables_js = (STATIC / "js" / "metric-tables.js").read_text(encoding="utf-8")
    parts = [
        'import { renderVintageCurvesCard } from "./marvis/static/js/metric-tables.js";',
        f"import {{ escapeHtml }} from {json.dumps((STATIC / 'js' / 'ui-utils.js').as_uri())};",
        f"import {{ platformLabelSource }} from {json.dumps((STATIC / 'js' / 'platform-labels.js').as_uri())};",
        f"import {{ renderWorkflowDataWidget }} from {json.dumps((STATIC / 'js' / 'v2' / 'workflow_widgets.js').as_uri())};",
        "import { columnFractions, parseNumeric, psiTier, psiTooltipText } from "
        + json.dumps((STATIC / "js" / "render-metrics.js").as_uri()) + ";",
        _slice(app_js, "function driverColumnKindFromHeader"),
        _slice(app_js, "function driverTableCellHtml"),
        _slice(metric_tables_js, "export function metricHeaderShouldRightAlign"),
        _slice(metric_tables_js, "export function renderCalibrationCard(chart)"),
        _slice(metric_tables_js, "export function renderScoreBandCard(chart)"),
        _slice(app_js, "function driverTableChartHtml"),
        _slice(app_js, "function agentMessageTablesHtml"),
        cases,
    ]
    return _run_module("\n".join(parts))


def test_result_tables_become_one_card_with_the_first_table_open():
    data = _tables_harness(
        """
const table = (title, rows) => ({ title, columns: ["特征", "IV"], rows });
const many = { metadata: { tables: [
  table("特征指标", [["f_income", "0.0127"], ["f_age", "0.0059"]]),
  table("数据质量", [["f_income", "1.0000"]]),
  table("唯一值检查<b>", [["f_age", "40"]]),
] } };
const one = { metadata: { tables: [table("核心指标", [["count", "2"]])] } };
process.stdout.write(JSON.stringify({ many: agentMessageTablesHtml(many), one: agentMessageTablesHtml(one), none: agentMessageTablesHtml({ metadata: {} }) }));
"""
    )
    many = data["many"]
    assert many.startswith('<div class="agent-message-tables result-tables-card">')
    # The first (platform-ordered) table is open; the others fold into one line each.
    assert many.index("特征指标") < many.index('<div class="result-tables-more"')
    assert many.count('<details class="result-more result-table-more"') == 2
    assert "数据质量</span><span class=\"result-row-value\">1 行 · 2 列</span>" in many
    assert "唯一值检查&lt;b&gt;" in many and "唯一值检查<b>" not in many
    # Same values as before: every cell is still rendered.
    for value in ("0.0127", "0.0059", "1.0000", "40"):
        assert value in many
    assert "result-tables-more" not in data["one"] and "核心指标" in data["one"]
    assert data["none"] == ""


def test_gate_evidence_and_risk_tables_are_never_folded():
    data = _tables_harness(
        """
const openResultTableFolds = new Set(["m2:2"]);
const table = (title) => ({ title, columns: ["项", "值"], rows: [["a", "1"]] });
const gate = { id: "g1", metadata: { kind: "gate", tables: [table("关键数字"), table("红旗 checklist")] } };
const step = { id: "m2", metadata: { tables: [table("特征指标"), table("疑似泄漏（KS≥阈值）"), table("数据质量"), table("分布统计")] } };
process.stdout.write(JSON.stringify({ gate: agentMessageTablesHtml(gate), step: agentMessageTablesHtml(step) }));
"""
    )
    # The portfolio-style gate: the red-flag checklist being confirmed stays open.
    assert "result-tables-more" not in data["gate"] and "红旗 checklist" in data["gate"]
    step = data["step"]
    more = step.index('<div class="result-tables-more"')
    assert step.index("疑似泄漏") < more
    assert step.count('<details class="result-more result-table-more"') == 2
    # A fold the user opened is restored on the next render.
    assert '<details class="result-more result-table-more" data-fold-key="m2:2" open>' in step
    assert '<details class="result-more result-table-more" data-fold-key="m2:3">' in step
