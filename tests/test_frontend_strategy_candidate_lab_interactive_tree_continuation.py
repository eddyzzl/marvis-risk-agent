from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _run_node(body: str) -> None:
    script = f"""
        import assert from "node:assert/strict";
        import {{ readFileSync }} from "node:fs";
        import {{
          collectStrategyCandidateLabRequest,
          strategyCandidateLabResultsHtml,
        }} from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";
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


def test_continuation_form_collects_exact_candidate_and_all_limits() -> None:
    _run_node(
        r"""
        const searchId = `interactive-tree-split-search-${"a".repeat(32)}`;
        const candidateId = (
          `interactive-tree-split-candidate-${"b".repeat(32)}`
        );
        const fields = {
          interactive_tree_continuation_search_id: { value: searchId },
          interactive_tree_continuation_candidate_id: { value: candidateId },
          interactive_tree_continuation_max_depth: { value: "3" },
          interactive_tree_continuation_min_gain: { value: "0.01" },
          interactive_tree_continuation_max_nodes: { value: "31" },
          interactive_tree_continuation_max_thresholds: { value: "10" },
          interactive_tree_continuation_max_row_evaluations: {
            value: "2000000",
          },
          interactive_tree_continuation_objective: {
            value: "max_gini_gain",
          },
          interactive_tree_continuation_tie_break: {
            value: "eligible_gain_feature_threshold_candidate_id",
          },
          interactive_tree_continuation_reason: {
            value: "Reviewed seed candidate.",
          },
        };
        const form = {
          dataset: {
            candidateLabWorkflow: "interactive_tree_auto_continuation",
          },
          querySelector(selector) {
            const match = selector.match(
              /data-candidate-lab-field="([^"]+)"/,
            );
            return match ? fields[match[1]] || null : null;
          },
        };
        assert.deepEqual(collectStrategyCandidateLabRequest(form), {
          request_kind: "standard_workflow",
          workflow: "interactive_tree_auto_continuation",
          workflow_inputs: {
            search_id: searchId,
            candidate_id: candidateId,
            max_additional_depth: 3,
            min_gini_gain: 0.01,
            max_generated_nodes: 31,
            max_thresholds_per_feature: 10,
            max_row_evaluations: 2000000,
            objective: "max_gini_gain",
            tie_break: "eligible_gain_feature_threshold_candidate_id",
            reason: "Reviewed seed candidate.",
          },
        });
        fields.interactive_tree_continuation_max_nodes.value = "128";
        assert.throws(
          () => collectStrategyCandidateLabRequest(form),
          /最大生成节点数/,
        );
        """
    )


def test_frontier_search_exposes_prefill_only_and_form_contract() -> None:
    _run_node(
        r"""
        const sourceId = `interactive-tree-revision-${"c".repeat(32)}`;
        const nodeId = `node-${"1".repeat(20)}`;
        const searchId = `interactive-tree-split-search-${"d".repeat(32)}`;
        const candidateId = (
          `interactive-tree-split-candidate-${"e".repeat(32)}`
        );
        const item = {
          kind: "interactive_tree_split_search",
          search_id: searchId,
          search_hash: "f".repeat(64),
          source_tree_id: sourceId,
          node_id: nodeId,
          node_kind: "split",
          source_node: {
            node_id: nodeId,
            kind: "split",
            is_visible: true,
            is_frontier: true,
            can_prune: false,
            feature: "score",
            threshold: 600,
          },
          mode: "all_features",
          features: ["score"],
          candidates: [{
            candidate_id: candidateId,
            rank: 1,
            feature: "score",
            threshold: 620,
            missing_child: "left",
            eligible: true,
            failures: [],
            gain: 0.02,
            parent: { count: 20 },
            left: { count: 10 },
            right: { count: 10 },
            direction: {},
          }],
          budget: {},
          population: {},
          claims: {},
          artifact: {},
          total: 1,
          truncated: false,
        };
        const payload = {
          candidates: {
            interactive_tree_split_search: {
              latest: item,
              all: [item],
              total: 1,
              truncated: false,
            },
          },
          pools: { latest: null, all: [], total: 0, truncated: false },
          strategies: {
            latest: null,
            all: [],
            total: 0,
            truncated: false,
            current_local_champions: [],
          },
        };
        const html = strategyCandidateLabResultsHtml(payload);
        assert.ok(html.includes(
          'data-candidate-lab-interactive-tree-auto-continuation="1"',
        ));
        assert.ok(html.includes("带入受控续建"));
        assert.equal(html.includes("自动提交"), false);

        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        for (const marker of [
          'data-candidate-lab-workflow="interactive_tree_auto_continuation"',
          'data-candidate-lab-field="interactive_tree_continuation_search_id"',
          'data-candidate-lab-field="interactive_tree_continuation_candidate_id"',
          'data-candidate-lab-field="interactive_tree_continuation_max_depth"',
          'data-candidate-lab-field="interactive_tree_continuation_min_gain"',
          'data-candidate-lab-field="interactive_tree_continuation_max_nodes"',
          'data-candidate-lab-field="interactive_tree_continuation_max_thresholds"',
          'data-candidate-lab-field="interactive_tree_continuation_max_row_evaluations"',
        ]) assert.ok(indexHtml.includes(marker), marker);
        """
    )
