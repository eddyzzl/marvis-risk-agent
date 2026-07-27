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
        assert.ok(html.includes("类型：未识别"));
        assert.ok(html.includes("语义角色：feature&lt;img&gt;"));
        assert.ok(html.includes("缺失率：25.0%"));
        assert.ok(html.includes("&lt;bad&gt;"));
        assert.equal(html.includes("dataset-truncated"), false);
        assert.equal(html.includes('class="dataset-column-role"'), false);
        assert.equal(html.includes('class="dataset-column-null"'), false);
        assert.ok(html.includes('class="dataset-column-tooltip"'));
        assert.ok(html.includes('class="dataset-preview-table"'));
        assert.ok(html.includes('data-dataset-column-index="1"'));
        """
    )


def test_dataset_preview_styles_support_metadata_tooltip_and_crosshair_hover():
    shared_css = (ROOT / "marvis/static/styles.css").read_text(encoding="utf-8")
    workbench_css = (ROOT / "marvis/static/css/v2-workbench.css").read_text(encoding="utf-8")

    assert ".dataset-column-info:hover + .dataset-column-tooltip" in shared_css
    assert ".dataset-column-info:focus-visible + .dataset-column-tooltip" in shared_css
    assert "tbody tr:nth-child(even) td" in workbench_css
    assert ".dataset-preview .is-column-hovered" in workbench_css
    assert "tbody tr:hover td.is-column-hovered" in workbench_css


def test_dataset_preview_pointer_handlers_highlight_and_clear_a_column():
    run_node(
        """
        import assert from "node:assert/strict";
        import {
          handleDatasetTablePointerOut,
          handleDatasetTablePointerOver,
        } from "./marvis/static/js/v2/artifact_view.js";

        const makeClassList = () => {
          const values = new Set();
          return {
            add(value) { values.add(value); },
            remove(value) { values.delete(value); },
            contains(value) { return values.has(value); },
          };
        };
        const table = {
          cells: [],
          querySelectorAll(selector) {
            if (selector === ".is-column-hovered") {
              return this.cells.filter((cell) => cell.classList.contains("is-column-hovered"));
            }
            const match = selector.match(/data-dataset-column-index="(\\d+)"/);
            return this.cells.filter((cell) => cell.index === match?.[1]);
          },
          contains(target) { return this.cells.includes(target); },
        };
        const makeCell = (index) => ({
          index: String(index),
          classList: makeClassList(),
          getAttribute(name) { return name === "data-dataset-column-index" ? this.index : null; },
          closest(selector) {
            if (selector === "[data-dataset-column-index]") return this;
            if (selector === ".dataset-preview-table") return table;
            return null;
          },
        });
        table.cells = [makeCell(0), makeCell(1), makeCell(0), makeCell(1)];

        assert.equal(handleDatasetTablePointerOver({ target: table.cells[1] }), true);
        assert.deepEqual(
          table.cells.map((cell) => cell.classList.contains("is-column-hovered")),
          [false, true, false, true],
        );
        assert.equal(handleDatasetTablePointerOut({ target: table.cells[1], relatedTarget: null }), true);
        assert.equal(table.cells.some((cell) => cell.classList.contains("is-column-hovered")), false);
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

        const refs = metricsHtml({
          validation_results_ref: "artifact:validation_results.json",
          joined_dataset: "dataset:joined<script>",
          note: "not:a-ref <img>",
        });
        assert.equal(refs.includes("joined<script>"), false);
        assert.equal(refs.includes("<img>"), false);
        assert.ok(refs.includes('data-artifact="artifact:validation_results.json"'));
        assert.ok(refs.includes('data-artifact="dataset:joined&lt;script&gt;"'));
        assert.ok(refs.includes("not:a-ref &lt;img&gt;"));

        const artifact = artifactFileHtml("report<script>.docx");
        assert.equal(artifact.includes("<script>"), false);
        assert.ok(artifact.includes("report&lt;script&gt;.docx"));
        assert.ok(artifact.includes("data-artifact-download"));
        assert.ok(artifact.includes("data-artifact-preview"));

        const image = artifactFileHtml("chart<script>.png");
        assert.equal(image.includes("<script>"), false);
        assert.ok(image.includes("chart&lt;script&gt;.png"));
        assert.ok(image.includes("data-artifact-image"));
        assert.ok(image.includes("<img"));
        assert.ok(image.includes("/api/artifacts/chart%3Cscript%3E.png"));
        """
    )


def test_dedup_conflict_picker_uses_cards_and_explains_protected_values():
    run_node(
        """
        import assert from "node:assert/strict";
        import { renderDedupPicker } from "./marvis/static/js/v2/join_gate_controller.js";

        const html = renderDedupPicker({
          id: "gate-1",
          metadata: {
            step_id: "step-1",
            dedup: {
              strategies: ["first", "last"],
              features: [{
                feature_id: "ds-1",
                feature_name: "vars.csv",
                conflict_keys: 2,
                conflict_columns: ["idcard_md5", "x6"],
                examples: [{
                  key: "date=2026-01-01, mobile=a",
                  values: { idcard_md5: ["[REDACTED]"], x6: [1, 13] },
                }],
              }],
            },
          },
        });

        assert.ok(html.includes('class="dedup-feature-card"'));
        assert.equal(html.includes('class="dedup-table"'), false);
        assert.ok(html.includes("vars.csv"));
        assert.ok(html.includes("敏感字段，示例值已保护"));
        assert.equal(html.includes("idcard_md5 两行分别为 [REDACTED]"), false);
        assert.ok(html.includes("x6"));
        assert.ok(html.includes("1"));
        assert.ok(html.includes("13"));
        """
    )
