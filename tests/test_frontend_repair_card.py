"""B-line (2026-10-16 spec): the "已处理好" card and repair option tiers."""

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


def test_repair_report_card_reads_finding_action_check_with_undo_and_log() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          repairReportCardHtml,
          workflowMessageContentHtml,
        } from "./marvis/static/js/v2/workflow_error_card.js";

        const report = {
          repair_id: "r1",
          verified: true,
          finding: "PMML 需要的输入 x1 在样本里找不到同名列。",
          action: "在验证合同里补上字段映射：X1 → x1<script>",
          verification: "重新生成的验证合同没有冲突。",
          undo_command: "撤销这次自动处理",
          log: ["近似匹配：X1 → x1", "核对：通过"],
        };
        const html = repairReportCardHtml(report);
        assert.ok(html.includes('class="workflow-repair-card"'));
        assert.ok(html.includes('role="status"'));
        assert.ok(html.includes("遇到问题，已处理好"));
        for (const label of ["发现", "处理", "核对"]) assert.ok(html.includes(`<dt>${label}</dt>`));
        assert.ok(html.includes("X1 → x1&lt;script&gt;"));
        assert.ok(html.includes('data-workflow-recovery-command="撤销这次自动处理"'));
        assert.ok(html.includes("<summary>查看日志</summary>"));
        assert.ok(html.includes("<li>核对：通过</li>"));

        // Undone: no undo button, the title says so.
        const undone = repairReportCardHtml(report, { undone: true });
        assert.ok(undone.includes("已撤销这次处理"));
        assert.ok(!undone.includes("data-workflow-recovery-command"));

        // A repair that failed its check never claims success.
        const failed = repairReportCardHtml({ ...report, verified: false, undo_command: "" });
        assert.ok(failed.includes("处理没成功，已还原"));
        assert.ok(!failed.includes("已处理好"));
        assert.ok(failed.includes("is-unverified"));

        // The message renderer routes repair metadata to the card, and the
        // full-timeline undone set wins over a stale report.
        const message = { content: "**遇到问题，已处理好**", metadata: { repair_report: report } };
        assert.ok(workflowMessageContentHtml(message).includes("workflow-repair-card"));
        assert.ok(
          workflowMessageContentHtml(message, undefined, { repairState: { undone: new Set(["r1"]), undoableId: "" } })
            .includes("已撤销这次处理"),
        );
        const undoMessage = { content: "x", metadata: { repair_undo: { repair_id: "r1", finding: "已撤销字段映射：X1 → x1" } } };
        assert.ok(workflowMessageContentHtml(undoMessage).includes("workflow-repair-undo"));
        """
    )


def test_auto_tier_option_is_marked_reversible_and_keeps_its_consequence() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { workflowErrorDiagnosticHtml } from "./marvis/static/js/v2/workflow_error_card.js";

        const html = workflowErrorDiagnosticHtml({
          title: "样本缺少模型需要的字段",
          summary: "x1 找不到",
          recovery_actions: [
            { label: "按近似列名补上字段映射", command: "按近似列名补上字段映射并重新扫描", tier: "auto", consequence: "样本文件不变；可以撤销。" },
            { label: "重新选择验证材料", command: "重新选择验证材料", tier: "one_click", consequence: "之前的识别结果作废。" },
          ],
        });
        assert.ok(html.includes('data-repair-tier="auto"'));
        assert.equal((html.match(/workflow-error-card__tier/g) || []).length, 1);
        assert.ok(html.includes("<small>样本文件不变；可以撤销。</small>"));
        assert.ok(html.includes('data-repair-tier="one_click"'));
        """
    )


def test_confirmation_failure_refreshes_the_agent_conversation() -> None:
    css = (ROOT / 'marvis/static/css/result-cards.css').read_text(encoding='utf-8')
    block = css.split('.workflow-repair-card {', 1)[1].split('}', 1)[0]
    assert 'border: 0' in block and 'box-shadow: none' in block


def test_streaming_fast_path_never_flattens_structured_cards() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { updateAgentMessageContentsInPlace } from "./marvis/static/js/agent-conversation-mount.js";

        const nodes = {};
        const article = (id) => {
          const content = { innerHTML: id === "card" ? "<section class=\\"workflow-repair-card\\"></section>" : "old", dataset: {} };
          nodes[id] = content;
          return { querySelector: (selector) => selector === ".agent-message-content" ? content : null };
        };
        const articles = { card: article("card"), text: article("text") };
        const scrollContent = {
          querySelector: (selector) => articles[selector.match(/"([^"]+)"/)[1]] || null,
        };
        const ok = updateAgentMessageContentsInPlace(
          [
            { id: "card", role: "assistant", content: "**遇到问题，已处理好**", metadata: { repair_report: {} } },
            { id: "text", role: "assistant", content: "新的文字", metadata: {} },
          ],
          {
            scrollContent,
            formatMessageContent: (text) => `<p>${text}</p>`,
            hasStructuredContent: (message) => Boolean(message.metadata?.repair_report),
          },
        );
        assert.equal(ok, true);
        assert.ok(nodes.card.innerHTML.includes("workflow-repair-card"));
        assert.equal(nodes.text.innerHTML, "<p>新的文字</p>");
        """
    )
    source = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    assert "hasStructuredContent: agentMessageHasStructuredContent" in source
    assert "repair_report: metadata.repair_report || null" in source


def test_recovery_quick_reply_carries_the_consequence_as_a_tooltip() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { agentQuickReplies, agentQuickRepliesHtml } from "./marvis/static/js/agent-quick-replies.js";

        const replies = agentQuickReplies([{ role: "assistant", metadata: { error_diagnostic: {
          recovery_actions: [{ label: "改用 apply_date 作为时间列", command: "改用 apply_date 作为时间列并确认", consequence: "月份划分按 apply_date 计算。" }],
        } } }]);
        assert.equal(replies[0].consequence, "月份划分按 apply_date 计算。");
        assert.ok(agentQuickRepliesHtml(replies).includes('title="月份划分按 apply_date 计算。"'));
        """
    )


def test_only_the_newest_report_keeps_undo_and_only_before_later_stages() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import { repairTimelineState, workflowMessageContentHtml } from "./marvis/static/js/v2/workflow_error_card.js";

        const report = (id) => ({ id: `m${id}`, role: "assistant", stage: "scan", metadata: { repair_report: {
          repair_id: id, verified: true, undo_command: "撤销这次自动处理", finding: "f", action: "a", verification: "v" } } });
        const two = [report("a"), report("b")];
        const state = repairTimelineState(two);
        assert.equal(state.undoableId, "b");
        const html = (message, s) => workflowMessageContentHtml(message, undefined, { repairState: s });
        assert.ok(!html(two[0], state).includes("data-workflow-recovery-command"));
        assert.ok(html(two[1], state).includes("data-workflow-recovery-command"));

        // Once PMML scoring has started, undo would discard results: no button.
        const later = [...two, { role: "assistant", stage: "reproducibility", metadata: {} }];
        assert.equal(repairTimelineState(later).undoableId, "");
        assert.ok(!html(later[1], repairTimelineState(later)).includes("data-workflow-recovery-command"));
        """
    )
