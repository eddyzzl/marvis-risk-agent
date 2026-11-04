"""Result-card headline (2026-10-19): tiles and bars copied from platform tables."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_tiles_and_bars_copy_cells_verbatim_from_registered_tables() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { resultSummaryParts, resultSummaryHtml, numericCell } from "./marvis/static/js/result-summary.js";

        const strategy = [
          { title: "策略回测摘要", columns: ["指标", "值"], rows: [["拒绝人数", "48"], ["审批率", "80.0%"], ["通过人数", "192"], ["通过坏率", "40.6%"], ["备注", "很长很长很长很长很长的一句说明文字"]] },
          { title: "按决策分组", columns: ["决策", "样本数", "占比", "有标签数", "坏样本", "坏率"], rows: [["approve", "192", "80.0%", "192", "78", "40.6%"], ["reject", "48", "20.0%", "48", "1", "2.1%"], ["review", "0", "0.0%", "0", "0", "n/a"]] },
        ];
        const parts = resultSummaryParts(strategy);
        // Registry order for tiles; values are the cells' own text.
        assert.deepEqual(parts.tiles.map((t) => [t.label, t.value]), [["审批率", "80.0%"], ["通过坏率", "40.6%"], ["通过人数", "192"], ["拒绝人数", "48"]]);
        assert.equal(parts.bars.title, "按决策分组");
        assert.deepEqual(parts.bars.items.map((i) => [i.label, i.text, i.note]), [["approve", "192", "40.6%"], ["reject", "48", "2.1%"], ["review", "0", "n/a"]]);
        assert.equal(parts.bars.items[0].ratio, 1);
        assert.equal(parts.bars.items[1].ratio, 48 / 192);
        assert.equal(parts.bars.items[2].ratio, 0);
        const html = resultSummaryHtml(strategy);
        assert.ok(html.includes('class="result-headline"'));
        assert.ok(html.includes('<b class="num" title="80.0%">80.0%</b>'));

        // Final model metrics win over trial metrics; labels name the split.
        const modeling = [
          { title: "最优 trial 指标", columns: ["指标", "值"], rows: [["oot_ks", "0.30"]] },
          { title: "最终模型指标", columns: ["指标", "值"], rows: [["train_ks", "0.41"], ["oot_ks", "0.354"], ["oot_auc", "0.719"]] },
          { title: "模型指标", columns: ["算法", "指标"], rows: [["lgb", "{'train_ks': 0.4}"]] },
        ];
        assert.deepEqual(resultSummaryParts(modeling).tiles.map((t) => [t.label, t.value]), [["KS · OOT", "0.354"], ["AUC · OOT", "0.719"], ["KS · 训练", "0.41"]]);

        // A record-shaped cell never becomes a tile; one bar never makes a chart.
        assert.deepEqual(resultSummaryParts([modeling[2]]).tiles, []);
        assert.equal(resultSummaryParts([{ title: "各特征表贡献", columns: ["特征表", "命中率"], rows: [["ds_1", "1.0000"]] }]).bars, null);
        // All-unparseable values give no bars; unknown tables give nothing.
        assert.equal(resultSummaryParts([{ title: "入选特征（前20）", columns: ["特征", "KS", "IV"], rows: [["a", "n/a", "n/a"], ["b", "n/a", "n/a"]] }]).bars, null);
        assert.equal(resultSummaryHtml([{ title: "建模规格", columns: ["项目", "值"], rows: [["算法", "lgb"]] }]), "");

        assert.equal(numericCell("6.0%"), 6);
        assert.equal(numericCell("1,234"), 1234);
        assert.equal(numericCell("n/a"), null);
        assert.equal(numericCell(90), 90);
        """
    )


def test_result_card_leads_with_the_summary_and_folds_tables_outside_gates() -> None:
    css = (ROOT / 'marvis/static/css/workbench-shell.css').read_text(encoding='utf-8')
    assert '.result-headline-bar i' in css




def test_rail_summary_only_claims_what_the_steps_say() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { railSummaryView } from "./marvis/static/js/workbench-chrome.js";

        const step = (cls, title = "") => ({
          classList: { contains: (name) => cls.split(" ").includes(name) },
          parentElement: { closest: () => null },
          querySelector: () => ({ textContent: title }),
        });
        const stepper = (classes, status) => ({
          dataset: { railStatus: status },
          querySelectorAll: () => classes.map(([cls, title]) => step(cls, title)),
        });

        const failed = railSummaryView(stepper([["step succeeded"], ["step failed", "PMML"], ["step pending"]], "failed"));
        assert.equal(failed.title, "卡在第 2 步");
        assert.equal(failed.detail, "处理后从这一步接着跑");
        // Reconciliation never promises a resume.
        const reconcile = railSummaryView(stepper([["step succeeded"], ["step failed"]], "needs_reconciliation"));
        assert.notEqual(reconcile.detail, "处理后从这一步接着跑");
        const running = railSummaryView(stepper([["step succeeded"], ["step running", "特征"], ["step pending"]], "running"));
        assert.equal(running.title, "已完成 1/3 步");
        assert.equal(running.detail, "正在：特征");
        const done = railSummaryView(stepper([["step succeeded"], ["step skipped"]], "done"));
        assert.equal(done.detail, "2 步完成（含 1 步跳过）");
        """
    )


def test_batch_chip_carries_its_progress_as_text_and_tooltip() -> None:
    css = (ROOT / 'marvis/static/css/workbench-shell.css').read_text(encoding='utf-8')
    progress_rule = css.split('.task-hero-top .validation-batch-switcher-progress {', 1)[1].split('}', 1)[0]
    assert 'clip-path' not in progress_rule
