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


def test_candidate_lab_api_uses_task_owned_get_and_existing_agent_message_envelope():
    run_node(
        """
        import assert from "node:assert/strict";

        globalThis.document = {
          body: { dataset: { marvisLocalToken: "local-token" } },
        };
        const {
          getStrategyCandidateLab,
          submitStrategyCandidateLabRequest,
        } = await import("./marvis/static/js/v2/api_v2.js");

        const calls = [];
        globalThis.fetch = async (url, options = {}) => {
          calls.push({ url, options });
          return {
            ok: true,
            status: 200,
            headers: { get: () => "application/json" },
            json: async () => ({ task_id: "task / 1", messages: [] }),
            text: async () => "",
          };
        };

        await getStrategyCandidateLab("task / 1");
        assert.equal(
          calls.at(-1).url,
          "/api/tasks/task%20%2F%201/strategy-candidate-lab",
        );
        assert.equal(calls.at(-1).options.method, "GET");

        const strategyRequest = {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_analysis",
          workflow_inputs: { features: ["score"] },
        };
        await submitStrategyCandidateLabRequest(
          "task / 1",
          strategyRequest,
          "启动单变量候选分析",
        );
        const post = calls.at(-1);
        assert.equal(post.url, "/api/tasks/task%20%2F%201/agent/messages");
        assert.equal(post.options.method, "POST");
        assert.equal(post.options.headers["X-Marvis-Token"], "local-token");
        assert.deepEqual(JSON.parse(post.options.body), {
          content: "启动单变量候选分析",
          strategy_request: strategyRequest,
        });
        assert.equal("model_id" in JSON.parse(post.options.body), false);
        """
    )


def test_primary_launchers_emit_only_user_owned_inputs_and_omit_empty_methods():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(workflow, values = {}, checked = {}) {
          const fields = new Map(
            Object.entries(values).map(([key, value]) => [key, { value }]),
          );
          return {
            dataset: { candidateLabWorkflow: workflow },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields.get(match[1]) || null : null;
            },
            querySelectorAll(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match
                ? (checked[match[1]] || []).map((value) => ({ value }))
                : [];
            },
          };
        }

        const univariate = collectStrategyCandidateLabRequest(makeForm(
          "univariate_candidate_analysis",
          {
            features: "score，age",
            bin_count: "5",
            min_bin_pct: "0.02",
            loan_amount_col: "loan_amount",
            overdue_amount_col: "",
            sentinel_values: "number:-9999, text:UNKNOWN",
            manual_breakpoints: "",
          },
          { methods: [] },
        ));
        assert.deepEqual(univariate, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_analysis",
          workflow_inputs: {
            features: ["score", "age"],
            bin_count: 5,
            min_bin_pct: 0.02,
            loan_amount_col: "loan_amount",
            sentinel_values: [-9999, "UNKNOWN"],
          },
        });
        assert.equal("methods" in univariate.workflow_inputs, false);

        const cross = collectStrategyCandidateLabRequest(makeForm(
          "cross_matrix_analysis",
          {
            x_feature: "age",
            x_method: "equal_frequency",
            y_feature: "score",
            y_method: "equal_width",
            bin_count: "4",
            min_bin_pct: "0.03",
            x_manual_breakpoints: "",
            y_manual_breakpoints: "",
            loan_amount_col: "",
            overdue_amount_col: "",
            sentinel_values: "",
          },
        ));
        assert.deepEqual(cross.workflow_inputs, {
          x_feature: "age",
          x_method: "equal_frequency",
          y_feature: "score",
          y_method: "equal_width",
          bin_count: 4,
          min_bin_pct: 0.03,
        });
        assert.equal("features" in cross.workflow_inputs, false);
        assert.equal("methods" in cross.workflow_inputs, false);

        const tree = collectStrategyCandidateLabRequest(makeForm(
          "automatic_tree_candidate_build",
          {
            features: "score;income",
            directions: "score=decreasing；income=unordered",
            max_depth: "3",
            min_leaf_count: "20",
            min_weight_fraction_leaf: "",
            seed: "42",
            sample_weight_col: "",
            loan_amount_col: "loan_amount",
            overdue_amount_col: "",
          },
        ));
        assert.deepEqual(tree.workflow_inputs, {
          features: ["score", "income"],
          directions: { score: "decreasing", income: "unordered" },
          max_depth: 3,
          min_leaf_count: 20,
          seed: 42,
          loan_amount_col: "loan_amount",
        });

        const platformOwned = new Set([
          "artifact_id",
          "content_hash",
          "dataset_id",
          "expected_content_hash",
          "sample_design_ref",
          "target_col",
          "workspace_revision",
        ]);
        for (const request of [univariate, cross, tree]) {
          for (const key of Object.keys(request.workflow_inputs)) {
            assert.equal(platformOwned.has(key), false, key);
            assert.equal(key.startsWith("expected_"), false, key);
          }
        }
        """
    )


def test_sentinel_types_are_explicit_and_preserve_textual_leading_zeroes():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function formWithSentinels(sentinelValues) {
          const values = {
            features: "score",
            bin_count: "",
            min_bin_pct: "",
            loan_amount_col: "",
            overdue_amount_col: "",
            sentinel_values: sentinelValues,
            manual_breakpoints: "",
          };
          return {
            dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? { value: values[match[1]] || "" } : null;
            },
            querySelectorAll() { return []; },
          };
        }

        const request = collectStrategyCandidateLabRequest(
          formWithSentinels(
            "text:001, number:1, text:1, number:-9999",
          ),
        );
        assert.deepEqual(request.workflow_inputs.sentinel_values, [
          "001",
          1,
          "1",
          -9999,
        ]);
        assert.equal(typeof request.workflow_inputs.sentinel_values[0], "string");
        assert.equal(typeof request.workflow_inputs.sentinel_values[1], "number");

        assert.throws(
          () => collectStrategyCandidateLabRequest(formWithSentinels("001, 1")),
          /必须显式标注类型/,
        );
        """
    )


def test_refinement_fresh_and_existing_payloads_match_manual_api_contract():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          collectStrategyCandidateLabRequest,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        function makeForm(values, fields = {}) {
          return {
            dataset: {
              candidateLabWorkflow: "univariate_candidate_refinement",
            },
            querySelector(selector) {
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              if (!match) return null;
              return fields[match[1]] || { value: values[match[1]] || "" };
            },
            querySelectorAll() { return []; },
          };
        }

        const fresh = collectStrategyCandidateLabRequest(makeForm({
          refinement_mode: "fresh",
          refinement_feature: "score",
          refinement_method: "manual",
          risk_operator: ">=",
          risk_value: "0.5",
          bin_count: "5",
          min_bin_pct: "0.02",
          loan_amount_col: "loan_amount",
          overdue_amount_col: "",
          sentinel_values: "text:001, number:-9999",
          refinement_manual_breakpoints: "500, 700",
          selection_reason: "保留高风险区间",
        }));
        assert.deepEqual(fresh, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_refinement",
          workflow_inputs: {
            feature: "score",
            method: "manual",
            selection: {
              risk_threshold: { operator: ">=", value: 0.5 },
            },
            bin_count: 5,
            min_bin_pct: 0.02,
            loan_amount_col: "loan_amount",
            sentinel_values: ["001", -9999],
            manual_breakpoints: { score: [500, 700] },
            selection_reason: "保留高风险区间",
          },
        });

        const candidateId = `candidate-${"a".repeat(32)}`;
        const sourceOption = {
          value: candidateId,
          dataset: { candidateLabProjection: "1" },
        };
        const pairOption = {
          value: "pair-0",
          dataset: {
            candidateLabProjection: "1",
            sourceCandidateId: candidateId,
            feature: "score",
            method: "manual",
          },
        };
        const bin = (value, selected = true) => ({
          value,
          selected,
          dataset: {
            candidateLabProjection: "1",
            sourceCandidateId: candidateId,
            feature: "score",
            method: "manual",
          },
        });
        const bins = [bin("regular:0"), bin("regular:1"), bin("regular:2", false)];
        const fields = {
          source_candidate_id: {
            value: candidateId,
            selectedOptions: [sourceOption],
            options: [sourceOption],
          },
          source_feature_method: {
            value: "pair-0",
            selectedOptions: [pairOption],
            options: [pairOption],
          },
          source_bin_ids: {
            selectedOptions: bins.filter((option) => option.selected),
            options: bins,
          },
        };
        const existing = collectStrategyCandidateLabRequest(makeForm({
          refinement_mode: "existing",
          merge_groups: "regular:0+regular:1",
          selection_reason: "合并并保留两个风险箱",
        }, fields));
        assert.deepEqual(existing, {
          request_kind: "standard_workflow",
          workflow: "univariate_candidate_refinement",
          workflow_inputs: {
            feature: "score",
            method: "manual",
            source_candidate_id: candidateId,
            selection: {
              source_bin_ids: ["regular:0", "regular:1"],
            },
            merge_groups: [["regular:0", "regular:1"]],
            selection_reason: "合并并保留两个风险箱",
          },
        });
        for (const forbidden of [
          "artifact_id",
          "content_hash",
          "evidence_hash",
          "expected_content_hash",
        ]) {
          assert.equal(forbidden in existing.workflow_inputs, false);
        }

        const forgedSource = {
          ...fields,
          source_candidate_id: {
            value: candidateId,
            selectedOptions: [{ value: candidateId, dataset: {} }],
            options: [],
          },
        };
        assert.throws(
          () => collectStrategyCandidateLabRequest(makeForm({
            refinement_mode: "existing",
          }, forgedSource)),
          /受认证投影/,
        );
        """
    )


def test_candidate_lab_renders_authenticated_collections_without_raw_json_or_paths():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const artifact = (suffix) => ({
          artifact_id: `task-artifact-${suffix}`,
          content_hash: suffix.repeat(64).slice(0, 64),
          created_at: "2026-07-24T10:00:00Z",
          download_url: `/api/tasks/task-1/task-artifacts/${suffix}/download`,
          path: "/private/workspace/never-render-this.json",
        });
        const collection = (item, truncated = false) => ({
          latest: item,
          all: [item],
          total: 1,
          truncated,
        });
        const candidate = (kind, pointerName, pointer, detail) => ({
          kind,
          artifact: artifact(kind.slice(0, 1)),
          candidate_id: `candidate-${kind}`,
          evidence_hash: `evidence-${kind}`,
          lifecycle: {
            candidate_stage: "development",
            validation_status: "unvalidated",
          },
          detail,
          risks: { red_flags: ["仅开发样本"], report_info_gaps: [] },
          pointers: { [pointerName]: [pointer] },
          total: 3,
          truncated: true,
        });
        const univariate = candidate(
          "univariate",
          "bins",
          { feature: "score", method: "equal_width", bin_id: "bin-1" },
          { metrics: { iv: 0.21, ks: 0.33 }, rankings: ["score"] },
        );
        const cross = candidate(
          "cross_matrix",
          "cells",
          {
            cell_id: "cross-cell-1",
            row_bin_id: "age-bin-1",
            column_bin_id: "score-bin-1",
            effect: { bad_rate: 0.08 },
          },
          {
            asset_id: "candidate-asset-cross",
            asset_hash: "cross-hash",
            axes: ["age", "score"],
            summary: { count: 120 },
          },
        );
        const tree = candidate(
          "automatic_tree",
          "leaves",
          {
            leaf_id: "leaf-1",
            fragment_id: "fragment-1",
            rule_id: "rule-1",
            effect_id: "effect-1",
            condition: { op: "lt", field: "score", value: 600 },
            metrics: { bad_rate: 0.12 },
          },
          {
            asset_id: "candidate-asset-tree",
            asset_hash: "tree-hash",
            tree_id: "tree-1",
            tree_result_hash: "tree-result-hash",
            summary: { count: 120 },
          },
        );
        const pool = {
          pool_id: "pool-1",
          strategy_type: "approval",
          revision: 2,
          revision_id: "pool-revision-2",
          snapshot_hash: "pool-hash",
          status: "draft",
          validation_status: "unvalidated",
          default_action: { type: "approval" },
          entries: [{
            position: 0,
            rule_id: "rule-1",
            source: { asset_id: "candidate-asset-tree" },
            action: { type: "reject" },
            execution: { condition: "score < 600" },
            enabled: true,
          }],
          artifact: artifact("p"),
          total: 4,
          truncated: true,
        };
        const html = strategyCandidateLabResultsHtml({
          candidates: {
            univariate: collection(univariate, true),
            cross_matrix: collection(cross),
            automatic_tree: collection(tree),
          },
          pools: collection(pool),
        });

        for (const expected of [
          "单变量候选",
          "Cross Matrix",
          "自动规则树",
          "Strategy Pool",
          "candidate-univariate",
          "evidence-cross_matrix",
          "tree-result-hash",
          "cross-cell-1",
          "leaf-1",
          "pool-revision-2",
          "下载受认证产物",
          "已由服务端截断",
        ]) {
          assert.ok(html.includes(expected), expected);
        }
        assert.equal(html.includes("/private/workspace"), false);
        assert.equal(html.includes('{"'), false);
        assert.equal(html.includes("undefined"), false);
        """
    )


def test_controller_drops_stale_task_response_and_uses_server_blocking_state():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        class ClassList {
          constructor() { this.values = new Set(["hidden"]); }
          toggle(name, force) {
            if (force) this.values.add(name);
            else this.values.delete(name);
          }
        }
        const controls = [{ disabled: false }, { disabled: false }];
        const panel = {
          classList: new ClassList(),
          dataset: {},
          attrs: {},
          setAttribute(name, value) { this.attrs[name] = value; },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [];
            if (selector === "[data-candidate-lab-retry]") return [];
            return controls;
          },
        };
        const results = { innerHTML: "" };
        const status = { textContent: "", dataset: {} };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: results,
          strategyCandidateLabStatus: status,
        };
        const deferred = new Map();
        let currentTask = { id: "task-1", task_type: "strategy" };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => currentTask,
          getSelectedTaskId: () => currentTask?.id || "",
          getStrategyCandidateLab: (taskId) => new Promise((resolve) => {
            deferred.set(taskId, resolve);
          }),
        });

        const taskOneLoad = controller.selectTask(currentTask);
        currentTask = { id: "task-2", task_type: "strategy" };
        const taskTwoLoad = controller.selectTask(currentTask);
        deferred.get("task-1")({
          task_id: "task-1",
          can_start: true,
          candidates: {},
          pools: {},
        });
        await taskOneLoad;
        assert.equal(controller.getState().taskId, "task-2");
        assert.equal(controller.getState().payload, null);

        deferred.get("task-2")({
          task_id: "task-2",
          can_start: false,
          blocked_reason: "open_gate",
          candidates: {},
          pools: {},
        });
        await taskTwoLoad;
        assert.equal(controller.getState().payload.task_id, "task-2");
        assert.equal(panel.dataset.candidateLabBlockedReason, "open_gate");
        assert.ok(controls.every((control) => control.disabled));
        assert.ok(status.textContent.includes("待处理确认门"));

        currentTask = { id: "task-3", task_type: "validation" };
        await controller.selectTask(currentTask);
        assert.equal(controller.getState().taskId, "");
        assert.ok(panel.classList.values.has("hidden"));
        """
    )


def test_failed_typed_submit_preserves_inputs_and_never_requires_llm_controls():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const featureField = { value: "score, age", disabled: false };
        const fields = new Map([
          ["features", featureField],
          ["bin_count", { value: "5", disabled: false }],
          ["min_bin_pct", { value: "0.02", disabled: false }],
          ["loan_amount_col", { value: "", disabled: false }],
          ["overdue_amount_col", { value: "", disabled: false }],
          ["sentinel_values", { value: "", disabled: false }],
          ["manual_breakpoints", { value: "", disabled: false }],
        ]);
        const errorTarget = { textContent: "" };
        const form = {
          dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
          resetCount: 0,
          reset() { this.resetCount += 1; },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") return errorTarget;
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? fields.get(match[1]) || null : null;
          },
          querySelectorAll(selector) {
            if (selector.includes(":checked")) return [];
            return [];
          },
        };
        const controls = Array.from(fields.values());
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [form];
            return controls;
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const calls = [];
        const statuses = [];
        let currentTask = { id: "strategy-1", task_type: "strategy" };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => currentTask,
          getSelectedTaskId: () => currentTask.id,
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => ({
            task_id: "strategy-1",
            can_start: true,
            blocked_reason: null,
            candidates: {},
            pools: {},
          }),
          submitStrategyCandidateLabRequest: async (...args) => {
            calls.push(args);
            throw new Error("server rejected request");
          },
          setActionStatus: (...args) => statuses.push(args),
        });
        await controller.selectTask(currentTask);
        // reset on task selection is expected; the operator fills the form after it.
        featureField.value = "score, age";
        const resetCountBeforeSubmit = form.resetCount;
        await controller.submit(form);

        assert.equal(calls.length, 1);
        assert.equal(calls[0][0], "strategy-1");
        assert.equal(calls[0][1].workflow, "univariate_candidate_analysis");
        assert.equal(calls[0][1].workflow_inputs.methods, undefined);
        assert.equal(featureField.value, "score, age");
        assert.equal(form.resetCount, resetCountBeforeSubmit);
        assert.equal(errorTarget.textContent, "server rejected request");
        assert.ok(statuses.some(([message, tone]) => (
          message === "server rejected request" && tone === "error"
        )));
        """
    )


def test_clarification_response_is_not_success_and_keeps_form_for_retry():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const featureField = { value: "score", disabled: false };
        const values = new Map([
          ["features", featureField],
          ["bin_count", { value: "5", disabled: false }],
          ["min_bin_pct", { value: "0.02", disabled: false }],
          ["loan_amount_col", { value: "", disabled: false }],
          ["overdue_amount_col", { value: "", disabled: false }],
          ["sentinel_values", { value: "text:001", disabled: false }],
          ["manual_breakpoints", { value: "", disabled: false }],
        ]);
        const errorTarget = { textContent: "" };
        const form = {
          dataset: { candidateLabWorkflow: "univariate_candidate_analysis" },
          resetCount: 0,
          reset() { this.resetCount += 1; },
          querySelector(selector) {
            if (selector === "[data-candidate-lab-form-error]") return errorTarget;
            const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
            return match ? values.get(match[1]) || null : null;
          },
          querySelectorAll(selector) {
            if (selector.includes(":checked")) return [];
            return [];
          },
        };
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll(selector) {
            if (selector === "[data-candidate-lab-form]") return [form];
            return Array.from(values.values());
          },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const statuses = [];
        const messagesSeen = [];
        let responseStatus = "clarification_required";
        const payload = {
          task_id: "strategy-1",
          can_start: true,
          blocked_reason: null,
          candidates: {},
          pools: {},
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => ({ id: "strategy-1", task_type: "strategy" }),
          getSelectedTaskId: () => "strategy-1",
          getBlockedReason: () => "",
          getStrategyCandidateLab: async () => payload,
          submitStrategyCandidateLabRequest: async () => (
            responseStatus === "clarification_required"
              ? {
                  status: "clarification_required",
                  code: "strategy_workspace_required",
                  messages: [{
                    role: "assistant",
                    content: "请先选择活动数据集并确认目标列。",
                  }],
                }
              : { status: "accepted", messages: [] }
          ),
          setActionStatus: (...args) => statuses.push(args),
          setAgentMessages: (messages) => messagesSeen.push(messages),
        });
        await controller.selectTask({ id: "strategy-1", task_type: "strategy" });
        featureField.value = "score";
        const resetCountBefore = form.resetCount;

        const clarified = await controller.submit(form);
        assert.equal(clarified.status, "clarification_required");
        assert.equal(featureField.value, "score");
        assert.equal(form.resetCount, resetCountBefore);
        assert.equal(errorTarget.textContent, "请先选择活动数据集并确认目标列。");
        assert.ok(statuses.some(([message, tone]) => (
          message === "请先选择活动数据集并确认目标列。" && tone === "info"
        )));
        assert.equal(statuses.some(([, tone]) => tone === "success"), false);
        assert.equal(messagesSeen.length, 1);

        responseStatus = "accepted";
        const accepted = await controller.submit(form);
        assert.equal(accepted.status, "accepted");
        assert.ok(statuses.some(([, tone]) => tone === "success"));
        """
    )


def test_candidate_lab_refresh_is_single_flight_and_aborts_on_task_switch():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          createStrategyCandidateLabController,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        let selectedTask = { id: "strategy-1", task_type: "strategy" };
        const calls = [];
        const pending = new Map();
        const panel = {
          classList: { toggle() {} },
          dataset: {},
          setAttribute() {},
          querySelector() { return null; },
          querySelectorAll() { return []; },
        };
        const ids = {
          strategyCandidateLabPanel: panel,
          strategyCandidateLabResults: { innerHTML: "" },
          strategyCandidateLabStatus: { textContent: "", dataset: {} },
        };
        const controller = createStrategyCandidateLabController({
          $: (id) => ids[id] || null,
          getSelectedTask: () => selectedTask,
          getSelectedTaskId: () => selectedTask.id,
          getBlockedReason: () => "",
          getStrategyCandidateLab(taskId, { signal } = {}) {
            calls.push({ taskId, signal });
            return new Promise((resolve, reject) => {
              pending.set(taskId, { resolve, reject });
              signal?.addEventListener("abort", () => {
                const error = new Error("aborted");
                error.name = "AbortError";
                reject(error);
              }, { once: true });
            });
          },
        });

        const first = controller.selectTask(selectedTask);
        const duplicate = controller.refresh("strategy-1");
        assert.equal(calls.length, 1);

        selectedTask = { id: "strategy-2", task_type: "strategy" };
        const second = controller.selectTask(selectedTask);
        assert.equal(calls.length, 2);
        assert.equal(calls[0].signal.aborted, true);

        pending.get("strategy-2").resolve({
          task_id: "strategy-2",
          can_start: true,
          blocked_reason: null,
          candidates: {},
          pools: {},
        });
        await Promise.all([first, duplicate, second]);
        assert.equal(controller.getState().taskId, "strategy-2");
        assert.equal(controller.getState().error, "");
        """
    )


def test_candidate_lab_refreshes_after_settle_once_but_never_per_poll_tick():
    app_js = (ROOT / "marvis/static/app.js").read_text(encoding="utf-8")
    index_html = (ROOT / "marvis/static/index.html").read_text(encoding="utf-8")
    workbench_css = (
        ROOT / "marvis/static/css/v2-workbench.css"
    ).read_text(encoding="utf-8")

    assert (
        'from "./js/v2/strategy_candidate_lab_controller.js"' in app_js
    )
    assert "strategyCandidateLabController.selectTask(task)" in app_js
    assert (
        'strategyCandidateLabController.refresh(taskId, { silent: true })'
        in app_js
    )
    poll_body = app_js[
        app_js.index("async function pollValidationProgress(") :
        app_js.index("async function validateCurrentTask(")
    ]
    assert "strategyCandidateLabController.refresh(" not in poll_body
    assert poll_body.count(
        "refreshStrategyCandidateLabAfterSettled(taskId, polledTask)"
    ) == 1
    settled_refresh_body = app_js[
        app_js.index("async function refreshStrategyCandidateLabAfterSettled(") :
        app_js.index("async function pollValidationProgress(")
    ]
    assert settled_refresh_body.count(
        "strategyCandidateLabController.refresh(taskId, { silent: true })"
    ) == 1
    assert "try {" in settled_refresh_body
    assert "catch (_error)" in settled_refresh_body
    assert "strategyCandidateLabController.bind(document)" in app_js
    assert 'id="strategyCandidateLabPanel"' in index_html
    assert 'id="strategyCandidateLabResults"' in index_html
    for workflow in (
        "univariate_candidate_analysis",
        "univariate_candidate_refinement",
        "cross_matrix_analysis",
        "automatic_tree_candidate_build",
    ):
        assert f'data-candidate-lab-workflow="{workflow}"' in index_html
    refinement_start = index_html.index(
        'data-candidate-lab-workflow="univariate_candidate_refinement"'
    )
    refinement_end = index_html.index("</form>", refinement_start)
    refinement_html = index_html[refinement_start:refinement_end]
    assert 'data-candidate-lab-field="source_candidate_id"' in refinement_html
    assert '<select data-candidate-lab-field="source_candidate_id">' in refinement_html
    for forbidden in ("artifact_id", "content_hash", "evidence_hash"):
        assert forbidden not in refinement_html
    assert "text:001" in index_html
    assert "number:-9999" in index_html
    assert ".candidate-lab-layout" in workbench_css
    assert (
        "grid-template-columns: minmax(260px, 0.72fr) minmax(0, 1.55fr)"
        in workbench_css
    )
    assert "@media" not in workbench_css[
        workbench_css.index("/* Candidate Lab") :
        workbench_css.index("/* Modeling setup gate controls.")
    ]
