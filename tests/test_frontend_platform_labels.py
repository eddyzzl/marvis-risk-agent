"""Fixed table labels follow the UI locale while identifiers stay canonical."""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _node(body: str) -> None:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", body],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_dynamic_system_titles_translate_their_fixed_captures_and_keep_raw_names():
    _node(r'''
import assert from "node:assert/strict";
import { translateText } from "./marvis/static/js/i18n.js";
const titles = [
  "数据处理执行中断", "特征分析未开始", "模型开发尚未就绪", "策略分析未完成",
  "组合分析准备步骤需要重新执行", "疑似泄漏（KS≥阈值，共3）",
  "疑似模型输出/评分列（共3）", "剔除·不可用（常量/稀疏，共3）",
  "最终清单（前20，共3）", "淘汰清单（前30，共3）",
  "最终模型交付能力（所选模型）", "评分分段（OOT）", "数据质量红旗 · 细分画像",
  "目录里的样本数据", "独立回放额度指标", "独立回放定价分布",
  "独立回放分群经济指标", "独立回放逐月额度证据", "坏率 5.00%",
];
for (const title of titles) {
  const translated = translateText(title);
  assert.ok(translated && !/[\u3400-\u4dbf\u4e00-\u9fff]/.test(translated), `${title} => ${translated}`);
}
// These captures are data identifiers, even when they match a platform word.
assert.equal(translateText("数据字典 通过.csv"), "Data dictionary 通过.csv");
assert.equal(translateText("按「通过」分布（逐划分）"), "Distribution by “通过” (per split)");
assert.equal(translateText("通过 特殊值策略"), "通过 special-value policy");
assert.equal(translateText("通过 的拼接键"), "Join keys for 通过");
''')


def test_fixed_english_headers_follow_locale_without_mutating_table_values():
    _node(r'''
import assert from "node:assert/strict";
import { __setI18nStateForTests } from "./marvis/static/js/i18n.js";
import { localizePlatformLabel, platformLabelSource } from "./marvis/static/js/platform-labels.js";
import { renderMetricTable } from "./marvis/static/js/metric-tables.js";
const table = {headers: ["Rank", "Feature", "Category", "KS_baseline"], rows: [[1, "通过", "征信", 0.31]]};
const original = JSON.stringify(table);
__setI18nStateForTests({locale: "zh"});
assert.equal(localizePlatformLabel("Feature"), "特征");
assert.equal(localizePlatformLabel("KS_baseline"), "基准 KS");
const html = renderMetricTable(table);
assert.match(html, /<th[^>]*>特征<\/th>/);
assert.match(html, /translate="no">通过<\/td>/);
assert.equal(JSON.stringify(table), original);
__setI18nStateForTests({locale: "en"});
assert.equal(localizePlatformLabel("Feature"), "Feature");
assert.equal(localizePlatformLabel("KS_baseline"), "Baseline KS");
assert.equal(platformLabelSource("用户定义列"), "用户定义列");
assert.equal(platformLabelSource("custom_feature"), "custom_feature");
''')


def test_dynamic_stage_and_material_labels_keep_raw_file_names():
    _node(r'''
import assert from "node:assert/strict";
import {translateText} from "./marvis/static/js/i18n.js";
assert.equal(translateText("阶段任务 · 3/3 完成"), "Stage tasks · 3/3 done");
const recognized = translateText("已识别： 模型样本.csv");
assert.ok(recognized.startsWith("Recognized"), recognized);
assert.ok(recognized.endsWith("模型样本.csv"), recognized);
assert.equal(translateText("样本总数"), "Sample count");
assert.ok(!/[\u3400-\u9fff]/.test(translateText("OOT(独立10等分)")));
''')


def test_field_lists_survive_english_api_headers_and_preserve_user_field_names():
    _node(r'''
import assert from "node:assert/strict";
import {__setI18nStateForTests} from "./marvis/static/js/i18n.js";
import {fieldListModel, fieldListHtml} from "./marvis/static/js/table-presentation.js";
__setI18nStateForTests({locale: "en"});
const model = fieldListModel({columns: ["Validation fields", "Value"], rows: [["target_col", "通过"], ["通过", "原始取值"]]});
assert.equal(model.items[0].label, "Target field");
assert.equal(model.items[0].value, "通过");
assert.equal(model.items[1].label, "通过");
assert.match(fieldListHtml(model), /translate="no"/);
__setI18nStateForTests({locale: "zh"});
const restored = fieldListModel({columns: ["验证字段", "值"], rows: [["target_col", "通过"], ["通过", "原始取值"]]});
assert.equal(restored.items[0].label, "目标字段");
assert.equal(restored.items[0].value, "通过");
''')


def test_composed_rail_status_labels_follow_locale_with_counts_unchanged():
    _node(r'''
import assert from "node:assert/strict";
import {__setI18nStateForTests} from "./marvis/static/js/i18n.js";
import {railProgressLabel} from "./marvis/static/js/workbench-chrome.js";
const summary = {status: "needs_review", done: 4, total: 4};
__setI18nStateForTests({locale: "en"});
assert.ok(!/[\u3400-\u9fff]/.test(railProgressLabel(summary)));
assert.ok(railProgressLabel(summary).endsWith("4/4"));
assert.ok(!/[\u3400-\u9fff]/.test(railProgressLabel({status: "failed", failedAt: 2})));
__setI18nStateForTests({locale: "zh"});
assert.equal(railProgressLabel(summary), "需复核 · 4/4");
''')
