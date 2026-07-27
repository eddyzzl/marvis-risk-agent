from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_drawer_renders_authenticated_bindings_and_memory_refs() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const artifactId = "a".repeat(64);
        const contentHash = "b".repeat(64);
        const provenanceHash = "c".repeat(64);
        const inputHash = "d".repeat(64);
        const html = strategyCandidateLabResultsHtml({
          evidence_drawer: {
            artifacts: {
              all: [{
                artifact_id: artifactId,
                kind: "strategy_candidate_json",
                origin_tool: "strategy.analyze_univariate_candidates",
                artifact_schema_version: "strategy.univariate-candidate-artifact.v1",
                producer_version: "strategy.univariate-candidate/1",
                content_hash: contentHash,
                provenance_hash: provenanceHash,
                input_binding_hash: inputHash,
                input_binding_status: "derived_from_provenance",
                explicit_input_hashes: [],
                datasets: [{
                  dataset_id: "dataset-1",
                  content_hash: contentHash,
                  role: "dataset_id",
                }],
                created_at: "2026-07-27T00:00:00+00:00",
                download_url: `/api/tasks/task-1/task-artifacts/${artifactId}/download`,
              }],
              total: 1,
              linked_total: 1,
              truncated: false,
            },
            datasets: {
              all: [{
                dataset_id: "dataset-1",
                content_hash: contentHash,
                artifact_ids: [artifactId],
              }],
              total: 1,
              truncated: false,
            },
            red_flags: {
              all: [{
                code: "sample_warning",
                level: "warn",
                message: "样本尚未完全成熟",
              }],
              total: 1,
              truncated: false,
            },
            memory_references: {
              all: [{
                id: "memory-1",
                kind: "raw",
                memory_type: "strategy_pitfall",
                source_task_id: "source-task-1",
                confidence: "high",
                use_reason: "提醒样本成熟度",
                support_count: 2,
                source_memory_count: 1,
              }],
              total: 1,
              omitted: 0,
              truncated: false,
            },
          },
        });

        assert.match(html, /Evidence Drawer/);
        assert.match(html, /strategy\\.analyze_univariate_candidates/);
        assert.match(html, new RegExp(contentHash));
        assert.match(html, new RegExp(provenanceHash));
        assert.match(html, new RegExp(inputHash));
        assert.match(html, /未冒充原始 Tool 输入 hash/);
        assert.match(html, /dataset-1/);
        assert.match(html, /样本尚未完全成熟/);
        assert.match(html, /memory-1/);
        assert.match(html, /提醒样本成熟度/);
        assert.doesNotMatch(html, /\\/private\\/raw\\/customers\\.csv/);
        """
    )
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_evidence_drawer_empty_state_does_not_infer_missing_lineage() -> None:
    script = textwrap.dedent(
        """
        import assert from "node:assert/strict";
        import {
          strategyCandidateLabResultsHtml,
        } from "./marvis/static/js/v2/strategy_candidate_lab_controller.js";

        const html = strategyCandidateLabResultsHtml({
          evidence_drawer: {
            artifacts: { all: [], total: 0, linked_total: 0, truncated: false },
            datasets: { all: [], total: 0, truncated: false },
            red_flags: { all: [], total: 0, truncated: false },
            memory_references: {
              all: [],
              total: 0,
              omitted: 0,
              truncated: false,
            },
          },
        });

        assert.match(html, /当前页面尚无受认证产物/);
        assert.match(html, /不从文件内容猜测|没有可展示的数据集指针/);
        assert.match(html, /未引用受治理记忆/);
        """
    )
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
