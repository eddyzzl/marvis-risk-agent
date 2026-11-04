"""B-line extension (2026-10-23 spec §5.4–5.5): workflow repair cards and batch records."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def run_node(script: str) -> None:
    subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )


def test_data_action_report_shows_who_when_what_and_undo_in_the_log() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { repairReportCardHtml } from "./marvis/static/js/v2/workflow_error_card.js";

        const report = {
          schema_version: "workflow_repair.v1", repair_id: "r1", verified: true,
          finding: "目标有空值", action: "剔除目标为空的 18 行", verification: "工具只剔除了 18 行",
          undo_command: "撤销这次自动处理", log: ["剔除 18 行"],
          audit: { actor: { kind: "user", label: "操作员<b>" }, at: "2026-10-28T10:00:00+00:00",
                   changed: "本次分析剔除 18 行", undo: "点「撤销」恢复原设置" },
        };
        const html = repairReportCardHtml(report);
        assert.ok(html.includes("操作人：操作员&lt;b&gt;，时间：2026-10-28T10:00:00+00:00"));
        assert.ok(html.includes("<li>改动：本次分析剔除 18 行</li>"));
        assert.ok(html.includes("撤销方式：点「撤销」恢复原设置"));
        assert.ok(html.includes('data-workflow-recovery-command="撤销这次自动处理"'));
        // The backend says undo no longer applies: no button.
        const stale = repairReportCardHtml({ ...report, undo_available: false });
        assert.ok(!stale.includes("data-workflow-recovery-command"));
        """
    )


def test_timeline_respects_undo_available_and_sample_design_card_renders() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { repairTimelineState, workflowMessageContentHtml } from "./marvis/static/js/v2/workflow_error_card.js";

        const message = (available) => ({ id: "m", role: "assistant", stage: "chat", metadata: { repair_report: {
          repair_id: "a", verified: true, undo_command: "撤销这次自动处理", undo_available: available,
          finding: "f", action: "a", verification: "v" } } });
        assert.equal(repairTimelineState([message(true)]).undoableId, "a");
        assert.equal(repairTimelineState([message(false)]).undoableId, "");

        const clarification = { role: "assistant", content: "field_bindings.time_field 使用了数据集中不存在的列「APPLY_DATE」。",
          metadata: { kind: "clarification", sample_design_diagnostic: {
            title: "样本设计里的字段在样本中找不到", summary: "时间字段写的是 APPLY_DATE，当前样本没有这一列。",
            evidence: [{ label: "样本里相近的列", value: "apply_date" }] } } };
        const html = workflowMessageContentHtml(clarification);
        assert.ok(html.includes('role="status"'));
        assert.ok(html.includes("样本设计里的字段在样本中找不到"));
        assert.ok(!html.includes("field_bindings"));
        """
    )


def test_batch_rows_show_each_child_repair_record() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { batchItemRepairHtml, normalizeValidationBatchPayload, renderValidationBatchOverview }
          from "./marvis/static/js/validation-batch.js";

        const payload = { batch: { parent_task_id: "p", status: "running", item_count: 3 }, items: [
          { id: "i1", child_task_id: "c1", ordinal: 1, model_name: "A", status: "running", repair_count: 1,
            repair_records: [{ finding: "x1 找不到", action: "补上字段映射：X1 → x1", verified: true, undone: false }] },
          { id: "i2", child_task_id: "c2", ordinal: 2, model_name: "B", status: "failed", repair_count: 1,
            repair_records: [{ finding: "时间列有坏值", action: "试过", verified: false, undone: false }] },
          { id: "i3", child_task_id: "c3", ordinal: 3, model_name: "C", status: "running", repair_count: 2,
            repair_records: [{ finding: "f", action: "a", verified: true, undone: false },
                             { finding: "撤销的映射", action: "b", verified: true, undone: true }] },
        ] };
        const batch = normalizeValidationBatchPayload(payload);
        assert.ok(batchItemRepairHtml(batch.items[0]).includes("已处理：补上字段映射：X1 → x1"));
        assert.ok(batchItemRepairHtml(batch.items[1]).includes("处理没成功，已还原：时间列有坏值"));
        assert.ok(batchItemRepairHtml(batch.items[2]).includes("已撤销 2 项：撤销的映射"));
        const html = renderValidationBatchOverview(batch);
        assert.equal((html.match(/validation-batch-repair/g) || []).length, 3);
        assert.equal(batchItemRepairHtml({ repairRecords: [] }), "");
        """
    )
