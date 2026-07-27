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


def test_workflow_error_card_is_structured_accessible_and_escaped() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          workflowErrorDiagnosticHtml,
        } from "./marvis/static/js/v2/workflow_error_card.js";

        const html = workflowErrorDiagnosticHtml({
          code: 'csv<code>"',
          workflow: "数据处理<script>",
          title: "数据拼接<img src=x onerror=1>",
          summary: "第 10 行字段数不一致</p><script>",
          cause: "分隔符混用 & 引号未闭合",
          location: "sample.csv:10<11",
          evidence: [{ label: "预期列数<th>", value: "1 & 实际 2<td>" }],
          actions: ["检查第 10 行<li>", "统一分隔符 & 保存"],
          agent_prompt: "是否由 Agent 继续？",
          recovery_actions: [{ label: "由 Agent 重试", command: "重试当前步骤" }],
          technical_detail: "Error tokenizing data\\n<trace>",
          retryable: true,
        });

        assert.ok(html.includes('class="workflow-error-card"'));
        assert.ok(html.includes('role="alert"'));
        assert.ok(html.includes('data-retryable="true"'));
        assert.ok(html.includes("workflow-error-card__summary"));
        assert.ok(html.includes("问题原因"));
        assert.ok(html.includes("出错位置"));
        assert.ok(html.includes("workflow-error-card__facts"));
        assert.ok(html.includes('<ol class="workflow-error-card__actions">'));
        assert.ok(html.includes('data-workflow-recovery-command="重试当前步骤"'));
        assert.ok(html.includes("是否由 Agent 继续？"));
        assert.ok(html.includes('<details class="workflow-error-card__technical">'));
        assert.ok(html.includes("技术信息"));
        assert.ok(html.includes("csv&lt;code&gt;&quot;"));
        assert.ok(html.includes("数据拼接&lt;img src=x onerror=1&gt;"));
        assert.ok(html.includes("sample.csv:10&lt;11"));
        assert.ok(html.includes("1 &amp; 实际 2&lt;td&gt;"));
        assert.ok(html.includes("Error tokenizing data\\n&lt;trace&gt;"));
        assert.equal(html.includes("<script>"), false);
        assert.equal(html.includes("<img"), false);
        assert.equal(html.includes("<trace>"), false);
        """
    )


def test_ingest_notice_is_yellow_status_metadata_and_preserves_body() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          workflowMessageContentHtml,
        } from "./marvis/static/js/v2/workflow_error_card.js";

        const html = workflowMessageContentHtml({
          content: "已识别目标列 y，继续分析。",
          metadata: {
            ingest_notices: [{
              code: 'extension<code>"',
              severity: "warning<svg>",
              file: "sample<bad>.csv",
              declared_format: "csv<script>",
              detected_format: "xlsx<img>",
              message: "扩展名不匹配，已按 Excel 读取</p><script>",
            }],
          },
        });

        assert.ok(html.includes('class="workflow-ingest-notice"'));
        assert.ok(html.includes('role="status"'));
        assert.ok(html.includes("已自动处理"));
        assert.ok(html.includes("已识别目标列 y，继续分析。"));
        assert.ok(html.includes("sample&lt;bad&gt;.csv"));
        assert.ok(html.includes("csv&lt;script&gt;"));
        assert.ok(html.includes("xlsx&lt;img&gt;"));
        assert.ok(html.includes("extension&lt;code&gt;&quot;"));
        assert.ok(html.includes("warning&lt;svg&gt;"));
        assert.equal(html.includes("<script>"), false);
        assert.equal(html.includes("<svg>"), false);
        assert.equal(html.includes("<img>"), false);
        """
    )


def test_workflow_message_legacy_fallback_is_unchanged() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          hasWorkflowErrorDiagnostic,
          ingestNoticesHtml,
          workflowErrorDiagnosticHtml,
          workflowMessageContentHtml,
        } from "./marvis/static/js/v2/workflow_error_card.js";

        const renderLegacy = (value) => `<div class="legacy">${value}</div>`;
        const html = workflowMessageContentHtml(
          { content: "原有消息正文", metadata: { error: true } },
          renderLegacy,
        );

        assert.equal(html, '<div class="legacy">原有消息正文</div>');
        assert.equal(hasWorkflowErrorDiagnostic({ error: true }), false);
        assert.equal(workflowErrorDiagnosticHtml(null), "");
        assert.equal(ingestNoticesHtml([]), "");
        """
    )


def test_manual_driver_reuses_structured_cards_and_agent_mode_is_wired() -> None:
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          driverManualAnalysisHtml,
        } from "./marvis/static/js/v2/driver_manual_analysis.js";

        const html = driverManualAnalysisHtml([{
          id: "error-1",
          role: "assistant",
          content: "RAW_EXCEPTION_SHOULD_NOT_RENDER",
          metadata: {
            error: true,
            error_diagnostic: {
              code: "csv_parse_error",
              workflow: "feature_analysis",
              title: "材料解析失败",
              summary: "第 10 行多出一列",
              cause: "分隔符不一致",
              location: "sample.csv:10",
              evidence: [{ label: "实际列数", value: "2" }],
              actions: ["修正第 10 行"],
              technical_detail: "parser trace",
              retryable: true,
            },
          },
        }], {
          renderAgentMarkdown: (value) => `<p>${value}</p>`,
        });

        assert.ok(html.includes("driver-analysis-section is-error has-workflow-error"));
        assert.ok(html.includes('role="alert"'));
        assert.ok(html.includes("材料解析失败"));
        assert.equal(html.includes("RAW_EXCEPTION_SHOULD_NOT_RENDER"), false);

        const noticeOnlyHtml = driverManualAnalysisHtml([{
          id: "notice-1",
          role: "assistant",
          content: "",
          metadata: { ingest_notices: [{
            code: "extension_content_mismatch",
            severity: "warning",
            file: "sample.csv",
            declared_format: "csv",
            detected_format: "xlsx",
            message: "已自动恢复读取",
          }] },
        }], {
          renderAgentMarkdown: (value) => `<p>${value}</p>`,
        });
        assert.ok(noticeOnlyHtml.includes("workflow-ingest-notice"));
        assert.ok(noticeOnlyHtml.includes("已自动恢复读取"));
        """
    )

    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    start = app_js.index("function agentMessageHtml")
    end = app_js.index("function agentThinkingHtml", start)
    render_source = app_js[start:end]
    assert "hasWorkflowErrorDiagnostic" in render_source
    assert "workflowMessageContentHtml" in render_source
    assert "has-workflow-error" in render_source
    assert "workflow_presentation" in app_js


def test_workflow_cards_use_border_first_responsive_theme_styles() -> None:
    styles = (ROOT / "marvis/static/styles.css").read_text(encoding="utf-8")
    start = styles.index(".workflow-error-card,")
    end = styles.index("/* Historical input-confirmation", start)
    card_styles = styles[start:end]

    assert "border-radius: 8px" in card_styles
    assert "border-left: 3px solid var(--danger)" in card_styles
    assert "border-left: 3px solid var(--warning)" in card_styles
    assert "box-shadow: none" in card_styles
    assert 'body[data-theme="dark"] .workflow-error-card' in card_styles
    assert 'body[data-theme="dark"] .workflow-ingest-notice' in card_styles
    assert "@media (max-width: 640px)" in card_styles
