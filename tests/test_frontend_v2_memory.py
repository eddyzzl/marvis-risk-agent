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


def test_memory_manager_html_renders_distillations_and_escapes_payloads():
    run_node(
        """
        import assert from "node:assert/strict";
        import { memoryDistillationsHtml } from "./marvis/static/js/v2/memory_manager.js";

        const html = memoryDistillationsHtml({
          items: [
            {
              id: "distill-1",
              category: "field_convention",
              summary: "目标字段 <bad>",
              confidence: "high",
              support_count: 4,
              status: "active",
            },
          ],
        });

        assert.ok(html.includes('data-memory-category'));
        assert.ok(html.includes('data-consolidate-memory'));
        assert.ok(html.includes('data-memory-distillation-id="distill-1"'));
        assert.ok(html.includes('data-rollback-memory-distillation="distill-1"'));
        assert.ok(html.includes("字段口径"));
        assert.ok(html.includes("支持证据 4"));
        assert.equal(html.includes("目标字段 <bad>"), false);
        assert.ok(html.includes("目标字段 &lt;bad&gt;"));
        """
    )


def test_memory_detail_html_shows_sources_audit_and_restored_predecessor():
    run_node(
        """
        import assert from "node:assert/strict";
        import { memoryDistillationDetailHtml } from "./marvis/static/js/v2/memory_manager.js";

        const html = memoryDistillationDetailHtml({
          distillation: {
            id: "distill-2",
            category: "validation_pitfall",
            summary: "PSI 分箱 <risk>",
            confidence: "medium",
            support_count: 2,
            superseded_by: "distill-3",
            status: "rolled_back",
          },
          source_memories: [
            { id: "mem-1", memory_type: "validation_pitfall", summary: "source <x>" },
          ],
          events: [
            { event_type: "rollback", created_at: "2026-06-20T00:00:00Z" },
          ],
          restored: { id: "distill-1", summary: "restored <old>" },
        });

        assert.equal(html.includes("PSI 分箱 <risk>"), false);
        assert.equal(html.includes("source <x>"), false);
        assert.equal(html.includes("restored <old>"), false);
        assert.ok(html.includes("PSI 分箱 &lt;risk&gt;"));
        assert.ok(html.includes("source &lt;x&gt;"));
        assert.ok(html.includes("restored &lt;old&gt;"));
        assert.ok(html.includes("被 distill-3 替代"));
        assert.ok(html.includes("rollback"));
        """
    )


def test_memory_handlers_load_detail_rollback_consolidate_and_filter():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachMemoryHandlers } from "./marvis/static/js/v2/memory_manager.js";

        const calls = [];
        const messages = [];
        const detailSlot = { innerHTML: "" };
        const categoryFilter = { value: "field_convention" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            if (selector === "[data-memory-detail]") return detailSlot;
            if (selector === "[data-memory-category]") return categoryFilter;
            return null;
          },
        };

        const detach = attachMemoryHandlers(root, {
          getMemoryDistillation: async (id) => {
            calls.push(["getMemoryDistillation", id]);
            return { distillation: { id, summary: "detail", category: "field_convention" }, events: [] };
          },
          rollbackMemoryDistillation: async (id) => {
            calls.push(["rollbackMemoryDistillation", id]);
            return { distillation: { id, summary: "rolled", status: "rolled_back" } };
          },
          consolidateMemory: async (category) => {
            calls.push(["consolidateMemory", category]);
            return { consolidated: { [category]: 1 } };
          },
          refreshMemories: async (query) => calls.push(["refreshMemories", query]),
          showMessage: (message) => messages.push(message),
          showError: (message) => messages.push(`error:${message}`),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-memory-distillation-id]"
                ? { dataset: { memoryDistillationId: "distill-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-rollback-memory-distillation]"
                ? { dataset: { rollbackMemoryDistillation: "distill-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-consolidate-memory]" ? this : null;
            },
          },
          preventDefault() {},
        });

        await listeners.change({
          target: {
            closest(selector) {
              return selector === "[data-memory-category]" ? categoryFilter : null;
            },
          },
        });

        assert.deepEqual(calls, [
          ["getMemoryDistillation", "distill-1"],
          ["rollbackMemoryDistillation", "distill-1"],
          ["refreshMemories", { category: "field_convention" }],
          ["consolidateMemory", "field_convention"],
          ["refreshMemories", { category: "field_convention" }],
          ["refreshMemories", { category: "field_convention" }],
        ]);
        assert.ok(detailSlot.innerHTML.includes("detail"));
        assert.ok(messages.some((message) => message.includes("已合并 1 条记忆沉淀")));

        detach();
        assert.equal(listeners.click, undefined);
        assert.equal(listeners.change, undefined);
        """
    )
