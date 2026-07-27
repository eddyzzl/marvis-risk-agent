"""Wide-desktop Candidate Lab controls for interactive-tree frontier OR groups."""

from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap

from tests.test_frontend_strategy_candidate_lab_interactive_tree import (
    NODE_HARNESS as TREE_NODE_HARNESS,
)


ROOT = Path(__file__).parents[1]


def run_node(body: str) -> None:
    script = f"""
        import assert from "node:assert/strict";
        import {{ readFileSync }} from "node:fs";
        import {{
          STRATEGY_CANDIDATE_LAB_WORKFLOWS,
          collectStrategyCandidateLabRequest,
          createStrategyCandidateLabController,
        }} from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        {TREE_NODE_HARNESS}

        {textwrap.dedent(body)}
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Node contract failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_frontier_group_form_collects_only_two_to_fifty_authenticated_members():
    run_node(
        r"""
        const revisionId = `interactive-tree-revision-${"a".repeat(32)}`;
        const nodeA = `node-${"1".repeat(20)}`;
        const nodeB = `leaf-${"2".repeat(20)}`;
        const option = (value, dataset) => ({ value, dataset });
        const revisionSelect = {
          value: revisionId,
          selectedOptions: [
            option(revisionId, {
              candidateLabProjection: "1",
              revisionId,
            }),
          ],
        };
        const nodeSelect = {
          selectedOptions: [
            option(nodeA, {
              candidateLabProjection: "1",
              revisionId,
              sourceNodeId: nodeA,
            }),
            option(nodeB, {
              candidateLabProjection: "1",
              revisionId,
              sourceNodeId: nodeB,
            }),
          ],
        };
        const reason = { value: "  Policy owner approved OR semantics.  " };
        const fields = new Map([
          ["interactive_tree_frontier_group_revision_id", revisionSelect],
          ["interactive_tree_frontier_group_source_node_ids", nodeSelect],
          ["interactive_tree_frontier_group_selection_reason", reason],
        ]);
        const form = {
          dataset: {
            candidateLabWorkflow:
              "interactive_tree_frontier_group_materialization",
          },
          querySelector(selector) {
            const match = selector.match(
              /^\[data-candidate-lab-field="([^"]+)"\]$/,
            );
            return match ? fields.get(match[1]) || null : null;
          },
        };

        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        for (const marker of [
          'data-candidate-lab-workflow="interactive_tree_frontier_group_materialization"',
          "data-candidate-lab-interactive-tree-frontier-group-operation",
          'data-candidate-lab-field="interactive_tree_frontier_group_revision_id"',
          'data-candidate-lab-field="interactive_tree_frontier_group_source_node_ids"',
          'data-candidate-lab-field="interactive_tree_frontier_group_selection_reason"',
        ]) {
          assert.ok(indexHtml.includes(marker), marker);
        }
        assert.ok(
          STRATEGY_CANDIDATE_LAB_WORKFLOWS.includes(
            "interactive_tree_frontier_group_materialization",
          ),
        );
        assert.deepEqual(
          collectStrategyCandidateLabRequest(form),
          {
            request_kind: "standard_workflow",
            workflow: "interactive_tree_frontier_group_materialization",
            workflow_inputs: {
              revision_id: revisionId,
              source_node_ids: [nodeA, nodeB],
              selection_reason: "Policy owner approved OR semantics.",
            },
          },
        );

        for (const invalidOptions of [
          nodeSelect.selectedOptions.slice(0, 1),
          [nodeSelect.selectedOptions[0], nodeSelect.selectedOptions[0]],
          Array.from({ length: 51 }, (_, index) => {
            const value = `node-${index.toString(16).padStart(20, "0")}`;
            return option(value, {
              candidateLabProjection: "1",
              revisionId,
              sourceNodeId: value,
            });
          }),
          [
            option(nodeA, {
              candidateLabProjection: "1",
              revisionId: `interactive-tree-revision-${"b".repeat(32)}`,
              sourceNodeId: nodeA,
            }),
            nodeSelect.selectedOptions[1],
          ],
        ]) {
          nodeSelect.selectedOptions = invalidOptions;
          assert.throws(
            () => collectStrategyCandidateLabRequest(form),
            /2.*50|重复|受认证|revision/,
          );
        }
        """
    )


def test_frontier_group_reuses_projection_blocking_and_single_flight_submit():
    run_node(
        r"""
        class FakeMultiSelect extends FakeSelect {
          set innerHTML(html) {
            super.innerHTML = html;
            for (const option of this.options) option.selected = false;
            this._value = "";
          }

          get innerHTML() {
            return "";
          }
        }

        function groupPayload(taskId, blockedReason = null) {
          const payload = payloadFor(taskId, { blockedReason });
          const item = payload.candidates.interactive_tree_revision.latest;
          const secondLeaf = `leaf-${"6".repeat(20)}`;
          item.pointers.nodes.push(node(secondLeaf, {
            kind: "leaf",
            depth: 1,
            frontier: true,
            condition: { field: "income", operator: ">", value: 5000 },
          }));
          item.pointers.frontier_node_ids.push(secondLeaf);
          item.pointers.frontier.push({
            source_node_id: secondLeaf,
            leaf_id: "leaf-c",
            fragment_id: "fragment-c",
            rule_id: "rule-c",
            effect_id: "effect-c",
            condition: { field: "income", operator: ">", value: 5000 },
            metrics: { count: 8, bad_rate: 0.3 },
          });
          return payload;
        }

        function makeGroupHarness({
          blockedReason = null,
          submitCandidateLab,
        } = {}) {
          const revisionSelect = new FakeSelect(
            "interactive_tree_frontier_group_revision_id",
          );
          const nodeSelect = new FakeMultiSelect(
            "interactive_tree_frontier_group_source_node_ids",
          );
          const reason = {
            value: "",
            disabled: false,
            closest() { return null; },
          };
          const submitButton = {
            disabled: false,
            closest() { return null; },
          };
          const error = { textContent: "" };
          const help = { textContent: "" };
          const fields = new Map([
            ["interactive_tree_frontier_group_revision_id", revisionSelect],
            ["interactive_tree_frontier_group_source_node_ids", nodeSelect],
            ["interactive_tree_frontier_group_selection_reason", reason],
          ]);
          const form = {
            dataset: {
              candidateLabWorkflow:
                "interactive_tree_frontier_group_materialization",
            },
            querySelector(selector) {
              if (selector === "[data-candidate-lab-form-error]") return error;
              if (
                selector
                === "[data-candidate-lab-interactive-tree-frontier-group-help]"
              ) return help;
              const match = selector.match(/data-candidate-lab-field="([^"]+)"/);
              return match ? fields.get(match[1]) || null : null;
            },
            querySelectorAll() { return []; },
            reset() {
              revisionSelect.value = "";
              for (const option of nodeSelect.options) option.selected = false;
              reason.value = "";
              error.textContent = "";
            },
            closest() { return null; },
          };
          revisionSelect.form = form;
          nodeSelect.form = form;
          const controls = [revisionSelect, nodeSelect, reason, submitButton];
          const panel = {
            classList: { toggle() {} },
            dataset: {},
            setAttribute() {},
            querySelector(selector) {
              return selector
                === '[data-candidate-lab-workflow="interactive_tree_frontier_group_materialization"]'
                ? form
                : null;
            },
            querySelectorAll(selector) {
              if (selector === "[data-candidate-lab-form]") return [form];
              if (selector === "[data-candidate-lab-retry]") return [];
              return controls;
            },
          };
          const results = { innerHTML: "" };
          const status = { textContent: "", dataset: {} };
          const submitCalls = [];
          const controller = createStrategyCandidateLabController({
            $: (id) => ({
              strategyCandidateLabPanel: panel,
              strategyCandidateLabResults: results,
              strategyCandidateLabStatus: status,
            })[id] || null,
            getSelectedTask: () => ({
              id: "strategy-a",
              task_type: "strategy",
            }),
            getSelectedTaskId: () => "strategy-a",
            getStrategyCandidateLab: async () => (
              groupPayload("strategy-a", blockedReason)
            ),
            submitStrategyCandidateLabRequest: async (...args) => {
              submitCalls.push(args);
              return submitCandidateLab
                ? submitCandidateLab(...args)
                : { status: "accepted", messages: [] };
            },
            pollAgentMessagesUntilSettled: async () => {},
            setActionStatus() {},
          });
          return {
            controller,
            error,
            form,
            help,
            nodeSelect,
            reason,
            revisionSelect,
            submitButton,
            submitCalls,
          };
        }

        let resolveSubmission;
        const pending = new Promise((resolve) => {
          resolveSubmission = resolve;
        });
        const harness = makeGroupHarness({
          submitCandidateLab: () => pending,
        });
        await harness.controller.selectTask({
          id: "strategy-a",
          task_type: "strategy",
        });
        assert.equal(harness.revisionSelect.value, sourceB);
        assert.equal(harness.nodeSelect.options.length, 2);
        assert.equal(harness.nodeSelect.selectedOptions.length, 0);
        assert.ok(harness.help.textContent.includes("2–50"));
        harness.nodeSelect.options[0].selected = true;
        harness.nodeSelect.options[1].selected = true;
        harness.reason.value = "  Approved by policy owner.  ";

        const first = harness.controller.submit(harness.form);
        await Promise.resolve();
        const second = await harness.controller.submit(harness.form);
        assert.equal(second, null);
        assert.equal(harness.submitCalls.length, 1);
        for (const control of [
          harness.revisionSelect,
          harness.nodeSelect,
          harness.reason,
          harness.submitButton,
        ]) {
          assert.equal(control.disabled, true);
        }
        assert.deepEqual(harness.submitCalls[0], [
          "strategy-a",
          {
            request_kind: "standard_workflow",
            workflow: "interactive_tree_frontier_group_materialization",
            workflow_inputs: {
              revision_id: sourceB,
              source_node_ids: [leafB, `leaf-${"6".repeat(20)}`],
              selection_reason: "Approved by policy owner.",
            },
          },
          "物化交互树前沿 OR 分组",
        ]);
        resolveSubmission({ status: "accepted", messages: [] });
        await first;
        assert.equal(harness.controller.getState().submitting, false);

        for (const blockedReason of ["active_plan", "open_gate"]) {
          const blocked = makeGroupHarness({ blockedReason });
          await blocked.controller.selectTask({
            id: "strategy-a",
            task_type: "strategy",
          });
          assert.equal(blocked.revisionSelect.disabled, true, blockedReason);
          assert.equal(blocked.nodeSelect.disabled, true, blockedReason);
          assert.equal(blocked.reason.disabled, true, blockedReason);
          assert.equal(blocked.submitButton.disabled, true, blockedReason);
        }
        """
    )
