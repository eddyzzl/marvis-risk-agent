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


def test_draft_manager_html_escapes_list_and_detail_payloads():
    run_node(
        """
        import assert from "node:assert/strict";
        import { draftDetailHtml, draftManagerHtml } from "./marvis/static/js/v2/draft_manager.js";

        const listHtml = draftManagerHtml({
          drafts: [
            {
              id: "draft-1",
              name: "calc<script>",
              summary: "Run <margin>",
              source: "web_learning",
              status: "draft",
              task_id: "task-1",
            },
          ],
        });

        assert.ok(listHtml.includes('data-draft-id="draft-1"'));
        assert.equal(listHtml.includes("calc<script>"), false);
        assert.ok(listHtml.includes("calc&lt;script&gt;"));
        assert.equal(listHtml.includes("Run <margin>"), false);
        assert.ok(listHtml.includes("Run &lt;margin&gt;"));
        assert.ok(listHtml.includes("data-draft-status"));

        const detailHtml = draftDetailHtml({
          draft: {
            id: "draft-1",
            name: "calc_margin",
            summary: "Summary <unsafe>",
            code: "def calc_margin():\\n    return '<bad>'",
            input_schema: { title: "<input>" },
            output_schema: { title: "<output>" },
            status: "tested",
          },
          learning_note: {
            sources: ["https://example.test/a?<x>"],
            distilled: "Use revenue < cost.",
          },
          runs: [{ ok: false, error: "boom <script>", at: "now" }],
        });

        assert.equal(detailHtml.includes("Summary <unsafe>"), false);
        assert.equal(detailHtml.includes("'<bad>'"), false);
        assert.ok(detailHtml.includes("Summary &lt;unsafe&gt;"));
        assert.ok(detailHtml.includes("&lt;input&gt;"));
        assert.ok(detailHtml.includes("boom &lt;script&gt;"));
        assert.ok(detailHtml.includes("data-run-draft"));
        assert.ok(detailHtml.includes("data-promote-draft"));
        assert.ok(detailHtml.includes("data-reject-draft"));
        """
    )


def test_draft_handlers_load_run_promote_reject_and_validate_json_inputs():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachDraftHandlers } from "./marvis/static/js/v2/draft_manager.js";

        const calls = [];
        const messages = [];
        const detailSlot = { innerHTML: "" };
        const runInputs = { value: '{"revenue":10,"cost":3}' };
        const promotionTests = { value: '[{"inputs":{"revenue":10,"cost":3},"expect":{"margin":7}}]' };
        const statusFilter = { value: "draft" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            if (selector === "[data-draft-detail]") return detailSlot;
            if (selector === "[data-draft-run-inputs]") return runInputs;
            if (selector === "[data-draft-promotion-tests]") return promotionTests;
            if (selector === "[data-draft-status]") return statusFilter;
            return null;
          },
        };

        const detach = attachDraftHandlers(root, {
          getDraft: async (id) => {
            calls.push(["getDraft", id]);
            return { draft: { id, name: "calc_margin", status: "draft" }, runs: [] };
          },
          runDraft: async (id, inputs) => {
            calls.push(["runDraft", id, inputs]);
            return { ok: true, output: { margin: 7 }, error: null };
          },
          promoteDraft: async (id, testCases) => {
            calls.push(["promoteDraft", id, testCases]);
            return { plugin: { name: "draft_calc_margin" } };
          },
          rejectDraft: async (id, reason) => calls.push(["rejectDraft", id, reason]),
          refreshDrafts: async (query) => calls.push(["refreshDrafts", query]),
          confirmReject: () => "not useful",
          showError: (message) => messages.push(message),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-draft-id]" ? { dataset: { draftId: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });
        assert.equal(detailSlot.innerHTML.includes("calc_margin"), true);

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-run-draft]" ? { dataset: { runDraft: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-promote-draft]" ? { dataset: { promoteDraft: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-reject-draft]" ? { dataset: { rejectDraft: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });

        await listeners.change({
          target: {
            closest(selector) {
              return selector === "[data-draft-status]" ? statusFilter : null;
            },
          },
        });

        assert.deepEqual(calls, [
          ["getDraft", "draft-1"],
          ["runDraft", "draft-1", { revenue: 10, cost: 3 }],
          ["getDraft", "draft-1"],
          ["promoteDraft", "draft-1", [{ inputs: { revenue: 10, cost: 3 }, expect: { margin: 7 } }]],
          ["refreshDrafts", { status: "draft" }],
          ["rejectDraft", "draft-1", "not useful"],
          ["refreshDrafts", { status: "draft" }],
          ["refreshDrafts", { status: "draft" }],
        ]);

        runInputs.value = "{bad";
        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-run-draft]" ? { dataset: { runDraft: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });
        assert.ok(messages.at(-1).includes("Invalid JSON"));

        detach();
        assert.equal(listeners.click, undefined);
        assert.equal(listeners.change, undefined);
        """
    )


def test_draft_handlers_surface_refresh_errors_without_bubbling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachDraftHandlers } from "./marvis/static/js/v2/draft_manager.js";

        const listeners = {};
        const messages = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector() {
            return { value: "draft" };
          },
        };
        attachDraftHandlers(root, {
          refreshDrafts: async (query) => {
            assert.deepEqual(query, { status: "draft" });
            throw new Error("draft list failed");
          },
          showError: (message) => messages.push(message),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-refresh-drafts]" ? this : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(messages, ["draft list failed"]);
        """
    )


def test_draft_handlers_require_secondary_confirmation_before_promotion():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachDraftHandlers } from "./marvis/static/js/v2/draft_manager.js";

        const calls = [];
        const promotionTests = { value: '[{"inputs":{},"expect":{}}]' };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector(selector) {
            if (selector === "[data-draft-promotion-tests]") return promotionTests;
            if (selector === "[data-draft-status]") return { value: "tested" };
            return null;
          },
        };

        attachDraftHandlers(root, {
          confirmPromote: (id, testCases) => {
            calls.push(["confirmPromote", id, testCases]);
            return false;
          },
          promoteDraft: async (id, testCases) => calls.push(["promoteDraft", id, testCases]),
          refreshDrafts: async (query) => calls.push(["refreshDrafts", query]),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-promote-draft]" ? { dataset: { promoteDraft: "draft-1" } } : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(calls, [
          ["confirmPromote", "draft-1", [{ inputs: {}, expect: {} }]],
        ]);
        """
    )


def test_render_draft_manager_preserves_status_filter_on_refresh():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderDraftManager } from "./marvis/static/js/v2/draft_manager.js";

        const calls = [];
        const statusControl = { value: "tested" };
        const container = {
          innerHTML: "",
          dataset: {},
          querySelector(selector) {
            return selector === "[data-draft-status]" ? statusControl : null;
          },
        };

        await renderDraftManager(container, {
          listDrafts: async (query) => {
            calls.push(query);
            return { drafts: [{ id: "draft-1", name: "Tested Draft", status: "tested" }] };
          },
        });

        assert.deepEqual(calls, [{ status: "tested" }]);
        assert.ok(container.innerHTML.includes('value="tested" selected'));
        assert.ok(container.innerHTML.includes("Tested Draft"));
        """
    )
