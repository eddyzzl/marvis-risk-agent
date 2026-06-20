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


def test_dataset_table_html_escapes_columns_profiles_and_rows():
    run_node(
        """
        import assert from "node:assert/strict";
        import { datasetTableHtml } from "./marvis/static/js/v2/artifact_view.js";

        const html = datasetTableHtml({
          columns: ["id", "score<script>"],
          column_profiles: [
            { name: "id", semantic_role: "key", null_rate: 0 },
            { name: "score<script>", semantic_role: "feature<img>", null_rate: 0.25 },
          ],
          rows: [
            { id: "u1", "score<script>": "<bad>" },
          ],
          truncated: true,
        });

        assert.equal(html.includes("<script>"), false);
        assert.equal(html.includes("<bad>"), false);
        assert.ok(html.includes("score&lt;script&gt;"));
        assert.ok(html.includes("feature&lt;img&gt;"));
        assert.ok(html.includes("25.0%"));
        assert.ok(html.includes("&lt;bad&gt;"));
        assert.ok(html.includes("dataset-truncated"));
        """
    )


def test_render_artifact_fetches_dataset_preview_and_writes_container():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderArtifact } from "./marvis/static/js/v2/artifact_view.js";

        const calls = [];
        const container = { innerHTML: "", dataset: {} };
        await renderArtifact(container, "dataset:dataset-1", {
          previewDataset: async (datasetId, rows) => {
            calls.push(["previewDataset", datasetId, rows]);
            return { columns: ["id"], rows: [{ id: "u1" }], truncated: false };
          },
        });

        assert.deepEqual(calls, [["previewDataset", "dataset-1", 50]]);
        assert.equal(container.dataset.v2ArtifactView, "true");
        assert.ok(container.innerHTML.includes("u1"));
        """
    )


def test_artifact_handlers_open_result_refs_into_preview_panel():
    run_node(
        """
        import assert from "node:assert/strict";
        import { attachArtifactHandlers } from "./marvis/static/js/v2/artifact_view.js";

        const calls = [];
        const container = { innerHTML: "", dataset: {} };
        const listeners = {};
        const root = {
          addEventListener(type, fn) { listeners[type] = fn; },
          removeEventListener(type, fn) {
            if (listeners[type] === fn) delete listeners[type];
          },
        };

        const detach = attachArtifactHandlers(root, () => container, {
          renderArtifact: async (targetContainer, artifactRef) => {
            calls.push(["renderArtifact", targetContainer === container, artifactRef]);
            targetContainer.innerHTML = "opened";
          },
          showError: (message) => calls.push(["showError", message]),
        });

        const artifactTarget = {
          closest(selector) {
            return selector === "[data-artifact]"
              ? { dataset: { artifact: "dataset:dataset-1" } }
              : null;
          },
        };
        await listeners.click({ target: artifactTarget, preventDefault() {} });

        assert.deepEqual(calls, [["renderArtifact", true, "dataset:dataset-1"]]);
        assert.equal(container.innerHTML, "opened");

        detach();
        assert.equal(listeners.click, undefined);
        """
    )


def test_artifact_ref_html_handles_value_metrics_and_file_refs_safely():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          artifactFileHtml,
          metricsHtml,
          valueHtml,
        } from "./marvis/static/js/v2/artifact_view.js";

        const value = valueHtml("<img onerror=alert(1)>");
        assert.equal(value.includes("<img onerror"), false);
        assert.ok(value.includes("&lt;img onerror=alert(1)&gt;"));

        const metrics = metricsHtml({ auc: 0.77, label: "<bad>" });
        assert.equal(metrics.includes("<bad>"), false);
        assert.ok(metrics.includes("&lt;bad&gt;"));

        const artifact = artifactFileHtml("report<script>.docx");
        assert.equal(artifact.includes("<script>"), false);
        assert.ok(artifact.includes("report&lt;script&gt;.docx"));
        assert.ok(artifact.includes("data-artifact-download"));
        """
    )
