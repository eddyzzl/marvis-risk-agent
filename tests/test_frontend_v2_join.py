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
        assert.ok(html.includes("feature-backend"));
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
        assert.ok(messages.at(-1).includes("dedup"));

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
