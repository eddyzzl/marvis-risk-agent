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


def test_join_review_renders_diagnostics_warnings_and_forces_dedup():
    run_node(
        """
        import assert from "node:assert/strict";
        import { joinReviewHtml } from "./marvis/static/js/v2/join_review.js";

        const html = joinReviewHtml({
          id: "join-1",
          anchor_dataset_id: "sample<img>",
          joins: [
            {
              feature_dataset_id: "feature<bad>",
              confirmed: false,
              key_pairs: [
                {
                  anchor_col: "cust_id",
                  feature_col: "cust_md5",
                  match_method: "hash:md5",
                  match_rate: 0.98,
                },
              ],
              diagnostics: {
                anchor_rows: 100,
                matched_rows: 48,
                match_rate: 0.48,
                feature_key_unique: false,
                fan_out_detected: true,
                shrink_detected: true,
                joined_rows_preview: 145,
                new_columns: 12,
                new_columns_null_rate: 0.52,
              },
            },
          ],
        });

        assert.equal(html.includes("<img>"), false);
        assert.equal(html.includes("feature<bad>"), false);
        assert.ok(html.includes("sample&lt;img&gt;"));
        assert.ok(html.includes("feature&lt;bad&gt;"));
        assert.ok(html.includes("join-warning fan-out"));
        assert.ok(html.includes("145"));
        assert.ok(html.includes("100"));
        assert.ok(html.includes("join-warning shrink"));
        assert.ok(html.includes("48.0%"));
        assert.ok(html.includes("cust_id"));
        assert.ok(html.includes("cust_md5"));
        assert.ok(html.includes("hash:md5"));
        assert.ok(html.includes("98.0%"));
        assert.ok(html.includes('data-dedup="feature&lt;bad&gt;"'));
        assert.ok(html.includes("合成聚合行"));
        assert.ok(html.includes("聚合去重会基于同键冲突生成合成特征行"));
        assert.ok(html.includes('data-confirm-join="feature&lt;bad&gt;"'));
        assert.ok(html.includes('data-exec-join="join-1" disabled'));
        """
    )


def test_join_review_enables_execute_after_all_specs_are_confirmed():
    run_node(
        """
        import assert from "node:assert/strict";
        import { joinReviewHtml } from "./marvis/static/js/v2/join_review.js";

        const html = joinReviewHtml({
          id: "join-1",
          anchor_dataset_id: "sample",
          joins: [
            {
              feature_dataset_id: "feature-1",
              confirmed: true,
              key_pairs: [],
              diagnostics: {
                anchor_rows: 100,
                matched_rows: 100,
                match_rate: 1,
                feature_key_unique: true,
                fan_out_detected: false,
                shrink_detected: false,
                joined_rows_preview: 100,
                new_columns: 3,
                new_columns_null_rate: 0,
              },
            },
          ],
        });

        assert.ok(html.includes("join-confirmed"));
        assert.equal(html.includes('data-exec-join="join-1" disabled'), false);
        assert.ok(html.includes('data-exec-join="join-1"'));
        """
    )


def test_join_review_accepts_backend_join_plan_payload_shape():
    run_node(
        """
        import assert from "node:assert/strict";
        import { joinReviewHtml } from "./marvis/static/js/v2/join_review.js";

        const html = joinReviewHtml({
          join_plan_id: "join-backend-1",
          anchor_dataset_id: "anchor-backend",
          status: "draft",
          joins: [
            {
              feature_id: "feature-backend",
              confirmed: false,
              key_pairs: [],
              diagnostics: {
                anchor_rows: 10,
                matched_rows: 9,
                match_rate: 0.9,
                feature_key_unique: true,
                fan_out_detected: false,
                shrink_detected: false,
                joined_rows_preview: 10,
                new_columns: 2,
                new_columns_null_rate: 0.1,
              },
            },
          ],
        });

        assert.ok(html.includes('data-join-id="join-backend-1"'));
        assert.ok(html.includes('data-feature-dataset="feature-backend"'));
        assert.ok(html.includes('data-confirm-join="feature-backend"'));
        assert.ok(html.includes('data-exec-join="join-backend-1" disabled'));
        assert.ok(html.includes("主表：anchor-backend"));
        assert.ok(html.includes("feature-backend"));
        """
    )


def test_render_join_review_with_explicit_plan_renders_immediately():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderJoinReview } from "./marvis/static/js/v2/join_review.js";
        import { resetV2State } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        const container = { dataset: {}, innerHTML: "" };
        const cleanup = renderJoinReview(container, {
          id: "join-explicit",
          anchor_dataset_id: "sample",
          joins: [],
        });

        assert.equal(container.dataset.v2JoinReview, "true");
        assert.ok(container.innerHTML.includes('data-join-id="join-explicit"'));
        cleanup();
        """
    )


def test_join_proposal_renders_dataset_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import { joinProposalHtml } from "./marvis/static/js/v2/join_review.js";

        const html = joinProposalHtml([
          {
            id: "anchor-1",
            source_name: "sample<img>.csv",
            role: "sample",
            row_count: 100,
          },
          {
            id: "feature-1",
            source_name: "feature.csv",
            role: "feature",
            row_count: 50,
          },
        ]);

        assert.equal(html.includes("<img>"), false);
        assert.ok(html.includes("sample&lt;img&gt;.csv"));
        assert.ok(html.includes('data-refresh-datasets'));
        assert.ok(html.includes('data-join-anchor'));
        assert.ok(html.includes('data-join-features'));
        assert.ok(html.includes("multiple"));
        assert.ok(html.includes('value="anchor-1"'));
        assert.ok(html.includes('value="feature-1"'));
        assert.ok(html.includes("sample | 100 行"));
        assert.ok(html.includes("feature | 50 行"));
        assert.ok(html.includes('data-propose-join'));
        """
    )


def test_join_handlers_refresh_datasets_and_propose_join_from_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const calls = [];
        const problemSlot = { innerHTML: "stale" };
        const anchorControl = { value: "anchor-1" };
        const featureControl = {
          selectedOptions: [{ value: "feature-1" }, { value: "anchor-1" }],
        };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            if (selector === "[data-join-problems]") return problemSlot;
            if (selector === "[data-join-anchor]") return anchorControl;
            if (selector === "[data-join-features]") return featureControl;
            return null;
          },
        };

        const detach = attachJoinHandlers(root, () => "task-1", {
          listDatasets: async (taskId) => {
            calls.push(["listDatasets", taskId]);
            return {
              datasets: [
                { id: "anchor-1", source_name: "sample.csv", role: "sample" },
                { id: "feature-1", source_name: "feature.csv", role: "feature" },
              ],
            };
          },
          proposeJoin: async (taskId, body) => {
            calls.push(["proposeJoin", taskId, body]);
            return { join_plan_id: "join-new", joins: [] };
          },
          setDatasets: (datasets) => calls.push(["setDatasets", datasets.map((item) => item.id)]),
          setCurrentJoin: (joinPlan) => calls.push(["setCurrentJoin", joinPlan.join_plan_id]),
          showError: (message) => calls.push(["showError", message]),
        });

        const refreshTarget = {
          closest(selector) {
            return selector === "[data-refresh-datasets]" ? { dataset: { refreshDatasets: "" } } : null;
          },
        };
        await listeners.click({ target: refreshTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          ["listDatasets", "task-1"],
          ["setDatasets", ["anchor-1", "feature-1"]],
        ]);
        assert.equal(problemSlot.innerHTML, "");

        const proposeTarget = {
          closest(selector) {
            return selector === "[data-propose-join]" ? { dataset: { proposeJoin: "" } } : null;
          },
        };
        await listeners.click({ target: proposeTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          [
            "proposeJoin",
            "task-1",
            { anchor_dataset_id: "anchor-1", feature_dataset_ids: ["feature-1"] },
          ],
          ["setCurrentJoin", "join-new"],
        ]);
        assert.equal(problemSlot.innerHTML, "");

        detach();
        assert.equal(listeners.click, undefined);
        """
    )


def test_join_handlers_require_task_before_dataset_actions():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const calls = [];
        const problemSlot = { innerHTML: "" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            return selector === "[data-join-problems]" ? problemSlot : null;
          },
        };

        attachJoinHandlers(root, () => "", {
          listDatasets: async () => calls.push(["listDatasets"]),
          proposeJoin: async () => calls.push(["proposeJoin"]),
          showError: (message) => calls.push(["showError", message]),
        });

        const refreshTarget = {
          closest(selector) {
            return selector === "[data-refresh-datasets]" ? { dataset: { refreshDatasets: "" } } : null;
          },
        };
        await listeners.click({ target: refreshTarget, preventDefault() {} });

        assert.deepEqual(calls, []);
        assert.ok(problemSlot.innerHTML.includes("请先选择或创建任务"));
        """
    )


def test_join_handlers_require_dedup_before_confirming_and_surface_execute_errors():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";
        import { resetV2State, setCurrentJoin } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setCurrentJoin({ id: "join-1" });
        const calls = [];
        const messages = [];
        const dedupControl = { value: "" };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
          querySelector(selector) {
            return selector === '[data-dedup="feature-1"]' ? dedupControl : null;
          },
        };

        const detach = attachJoinHandlers(root, {
          confirmJoinSpec: async (joinId, body) => calls.push(["confirmJoinSpec", joinId, body]),
          executeJoin: async (joinId) => {
            calls.push(["executeJoin", joinId]);
            return { fan_out: true };
          },
          refreshJoin: async () => calls.push(["refreshJoin"]),
          showError: (message) => messages.push(message),
        });

        const confirmTarget = {
          closest(selector) {
            return selector === "[data-confirm-join]"
              ? { dataset: { confirmJoin: "feature-1" } }
              : null;
          },
        };
        await listeners.click({ target: confirmTarget, preventDefault() {} });
        assert.deepEqual(calls, []);
        assert.ok(messages.at(-1).includes("去重策略"));

        dedupControl.value = "first";
        await listeners.click({ target: confirmTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [
          [
            "confirmJoinSpec",
            "join-1",
            {
              feature_id: "feature-1",
              feature_dataset_id: "feature-1",
              dedup_strategy: "first",
            },
          ],
          ["refreshJoin"],
        ]);

        const executeTarget = {
          closest(selector) {
            return selector === "[data-exec-join]"
              ? { dataset: { execJoin: "join-1" } }
              : null;
          },
        };
        await listeners.click({ target: executeTarget, preventDefault() {} });
        assert.deepEqual(calls.splice(0), [["executeJoin", "join-1"]]);
        assert.ok(messages.at(-1).includes("fan-out"));

        detach();
        assert.equal(listeners.click, undefined);
        """
    )


def test_join_handlers_poll_after_async_execute_acceptance():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const listeners = {};
        const calls = [];
        const results = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachJoinHandlers(root, "task-1", {
          executeJoin: async (joinId) => {
            calls.push(["executeJoin", joinId]);
            return { status: "accepted", job_id: "job-1", join_plan_id: joinId };
          },
          pollJoinExecution: async ({ joinId, taskId, accepted, refreshJoin }) => {
            calls.push([
              "pollJoinExecution",
              joinId,
              taskId,
              accepted.job_id,
              typeof refreshJoin,
            ]);
            return { status: "executed", result_dataset_id: "joined-1" };
          },
          showResult: (payload) => results.push(payload),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-exec-join]"
                ? { dataset: { execJoin: "join-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(calls, [
          ["executeJoin", "join-1"],
          ["pollJoinExecution", "join-1", "task-1", "job-1", "function"],
        ]);
        assert.deepEqual(results, [
          { status: "accepted", job_id: "job-1", join_plan_id: "join-1" },
          { status: "executed", result_dataset_id: "joined-1" },
        ]);
        """
    )


def test_join_handlers_surface_async_job_poll_errors():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const listeners = {};
        const messages = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachJoinHandlers(root, "task-1", {
          executeJoin: async () => ({ status: "accepted", job_id: "job-1" }),
          pollJoinExecution: async () => {
            throw new Error("join produced 12 > anchor 10 rows");
          },
          showError: (message) => messages.push(message),
          showResult: () => {},
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-exec-join]"
                ? { dataset: { execJoin: "join-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(messages, ["join produced 12 > anchor 10 rows"]);
        """
    )


def test_join_handlers_ignore_latest_job_from_different_async_execute():
    run_node(
        """
        import assert from "node:assert/strict";
        import { pollJoinExecution } from "./marvis/static/js/v2/join_review.js";

        await assert.rejects(
          pollJoinExecution({
            accepted: { job_id: "job-original" },
            joinId: "join-1",
            taskId: "task-1",
            maxAttempts: 1,
            refreshJoin: async () => ({ id: "join-1", status: "draft" }),
            getLatestTaskJob: async () => ({
              job: {
                id: "job-retry",
                status: "failed",
                error_value: "newer retry failed",
              },
            }),
            sleepFn: async () => {},
          }),
          /后台拼接任务已结束/,
        );

        await assert.rejects(
          pollJoinExecution({
            accepted: { job_id: "job-original" },
            joinId: "join-1",
            taskId: "task-1",
            maxAttempts: 1,
            refreshJoin: async () => ({ id: "join-1", status: "draft" }),
            getLatestTaskJob: async () => ({
              job: {
                id: "job-original",
                status: "failed",
                error_value: "accepted job failed",
              },
            }),
            sleepFn: async () => {},
          }),
          /accepted job failed/,
        );
        """
    )


def test_join_handlers_surface_execute_api_errors_without_bubbling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const listeners = {};
        const messages = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachJoinHandlers(root, {
          executeJoin: async (joinId) => {
            assert.equal(joinId, "join-1");
            throw new Error("join produced 12 > anchor 10 rows");
          },
          showError: (message) => messages.push(message),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-exec-join]"
                ? { dataset: { execJoin: "join-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(messages, ["join produced 12 > anchor 10 rows"]);
        """
    )


def test_join_handlers_render_fan_out_message_for_conflict_errors():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";

        const listeners = {};
        const messages = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
        };
        attachJoinHandlers(root, {
          executeJoin: async () => {
            const error = new Error("conflict");
            error.status = 409;
            throw error;
          },
          showError: (message) => messages.push(message),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-exec-join]"
                ? { dataset: { execJoin: "join-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(messages, ["检测到 fan-out 风险，已停止执行拼接。"]);
        """
    )


def test_join_handlers_surface_confirm_api_errors_without_bubbling():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachJoinHandlers } from "./marvis/static/js/v2/join_review.js";
        import { resetV2State, setCurrentJoin } from "./marvis/static/js/v2/state_v2.js";

        resetV2State();
        setCurrentJoin({ id: "join-1" });
        const listeners = {};
        const messages = [];
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener() {},
          querySelector() { return null; },
        };
        attachJoinHandlers(root, {
          confirmJoinSpec: async () => {
            throw new Error("join plan or feature not found");
          },
          showError: (message) => messages.push(message),
        });

        await listeners.click({
          target: {
            closest(selector) {
              return selector === "[data-confirm-join]"
                ? { dataset: { confirmJoin: "feature-1" } }
                : null;
            },
          },
          preventDefault() {},
        });

        assert.deepEqual(messages, ["join plan or feature not found"]);
        """
    )
