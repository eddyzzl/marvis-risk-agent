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


def test_split_search_form_collects_only_exact_visible_user_controls() -> None:
    _run_node(
        r"""
        const sourceId = `candidate-asset-${"a".repeat(32)}`;
        const nodeId = `node-${"1".repeat(20)}`;
        const fields = {
          interactive_tree_search_source_id: {
            value: sourceId,
            selectedOptions: [{
              value: sourceId,
              dataset: {
                candidateLabProjection: "1",
                sourceTreeId: sourceId,
                featureUniverse: "income\u001fscore",
              },
            }],
          },
          interactive_tree_search_node_id: {
            value: nodeId,
            selectedOptions: [{
              value: nodeId,
              dataset: {
                candidateLabProjection: "1",
                sourceTreeId: sourceId,
                nodeId,
              },
            }],
          },
          interactive_tree_search_mode: { value: "selected_features" },
          interactive_tree_search_features: { value: "score, income" },
          interactive_tree_search_max_thresholds: { value: "8" },
          interactive_tree_search_max_row_evaluations: { value: "500000" },
        };
        const form = {
          dataset: {
            candidateLabWorkflow: "interactive_tree_split_search",
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
          workflow: "interactive_tree_split_search",
          workflow_inputs: {
            source_tree_id: sourceId,
            node_id: nodeId,
            mode: "selected_features",
            features: ["score", "income"],
            max_thresholds_per_feature: 8,
            max_row_evaluations: 500000,
          },
        });

        fields.interactive_tree_search_features.value = "score, forged";
        assert.throws(
          () => collectStrategyCandidateLabRequest(form),
          /认证特征全集/,
        );
        """
    )


def test_split_search_result_is_aggregate_safe_and_prefills_only_by_button() -> (
    None
):
    _run_node(
        r"""
        const sourceId = `candidate-asset-${"a".repeat(32)}`;
        const nodeId = `node-${"1".repeat(20)}`;
        const searchId = `interactive-tree-split-search-${"b".repeat(32)}`;
        const candidateId = (
          `interactive-tree-split-candidate-${"c".repeat(32)}`
        );
        const payload = {
          candidates: {
            interactive_tree_split_search: {
              latest: null,
              all: [{
                kind: "interactive_tree_split_search",
                search_id: searchId,
                search_hash: "d".repeat(64),
                source_tree_id: sourceId,
                node_id: nodeId,
                node_kind: "split",
                source_node: {
                  node_id: nodeId,
                  kind: "split",
                  is_visible: true,
                  is_frontier: false,
                  can_prune: true,
                  feature: "score",
                  threshold: 600,
                },
                mode: "all_features",
                features: ["income", "score"],
                budget: {
                  evaluated_candidates: 1,
                  row_evaluations: 200,
                  truncated: false,
                },
                population: { count: 20, bad: 4, bad_rate: 0.2 },
                claims: {
                  rank_is_navigation_only: true,
                  winner_selected: false,
                  tree_modified: false,
                },
                candidates: [{
                  candidate_id: candidateId,
                  rank: 1,
                  feature: 'income<script>alert("x")</script>',
                  threshold: 500,
                  missing_child: "left",
                  eligible: true,
                  failures: [],
                  gain: 0.12,
                  parent: { count: 20, bad_rate: 0.2 },
                  left: { count: 10, bad_rate: 0.1 },
                  right: { count: 10, bad_rate: 0.3 },
                  direction: {
                    expected: "increasing",
                    status: "consistent",
                  },
                }],
                artifact: {
                  artifact_id: "artifact-1",
                  created_at: "2026-07-27T00:00:00Z",
                  filename: "search.json",
                  download_url: "/download",
                },
                total: 1,
                truncated: false,
              }],
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
        payload.candidates.interactive_tree_split_search.latest = (
          payload.candidates.interactive_tree_split_search.all[0]
        );
        const html = strategyCandidateLabResultsHtml(payload);

        assert.ok(html.includes("树节点分裂候选"));
        assert.ok(html.includes("排名仅用于浏览"));
        assert.ok(html.includes('data-candidate-lab-interactive-tree-split-candidate="1"'));
        assert.ok(html.includes(`data-search-id="${searchId}"`));
        assert.ok(html.includes(`data-candidate-id="${candidateId}"`));
        assert.equal(html.includes("<script"), false);
        assert.ok(html.includes("income&lt;script&gt;"));

        const indexHtml = readFileSync("./marvis/static/index.html", "utf8");
        for (const marker of [
          'data-candidate-lab-workflow="interactive_tree_split_search"',
          'data-candidate-lab-field="interactive_tree_search_source_id"',
          'data-candidate-lab-field="interactive_tree_search_node_id"',
          'data-candidate-lab-field="interactive_tree_search_mode"',
          'data-candidate-lab-field="interactive_tree_search_max_thresholds"',
          'data-candidate-lab-field="interactive_tree_search_max_row_evaluations"',
        ]) assert.ok(indexHtml.includes(marker), marker);
        """
    )
