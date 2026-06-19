import { apiGet } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import { previewDataset as previewDatasetApi } from "./api_v2.js";

function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function cellText(value) {
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value ?? "");
}

function profileByName(preview) {
  const profiles = preview.column_profiles || preview.profiles || [];
  return new Map(profiles.map((profile) => [String(profile.name), profile]));
}

function parseArtifactRef(artifactRef) {
  const raw = String(artifactRef || "");
  const index = raw.indexOf(":");
  if (index < 1) {
    return { kind: "value", value: raw };
  }
  return {
    kind: raw.slice(0, index),
    value: raw.slice(index + 1),
  };
}

export function datasetTableHtml(preview = {}) {
  const columns = preview.columns?.length
    ? preview.columns
    : Object.keys(preview.rows?.[0] || {});
  const profiles = profileByName(preview);
  const header = columns.map((column) => {
    const profile = profiles.get(String(column)) || {};
    return `<th>
      <span class="dataset-column-name">${escapeHtml(column)}</span>
      <span class="dataset-column-role">${escapeHtml(profile.semantic_role || "")}</span>
      <span class="dataset-column-null">${pct(profile.null_rate)}</span>
    </th>`;
  }).join("");
  const rows = (preview.rows || []).map((row) => {
    const cells = columns
      .map((column) => `<td>${escapeHtml(cellText(row?.[column]))}</td>`)
      .join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  const truncated = preview.truncated
    ? '<div class="dataset-truncated">Preview truncated</div>'
    : "";
  return `<section class="dataset-preview">
    ${truncated}
    <table>
      <thead><tr>${header}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </section>`;
}

export function metricsHtml(metrics = {}) {
  const rows = Object.entries(metrics)
    .map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(cellText(value))}</td></tr>`)
    .join("");
  return `<table class="metrics-preview"><tbody>${rows}</tbody></table>`;
}

export function artifactFileHtml(artifactId) {
  const id = String(artifactId || "");
  const encoded = encodeURIComponent(id);
  return `<section class="artifact-file">
    <span>${escapeHtml(id)}</span>
    <a data-artifact-download href="/api/artifacts/${encoded}">Download</a>
  </section>`;
}

export function valueHtml(value) {
  return `<pre class="value-preview"><code>${escapeHtml(cellText(value))}</code></pre>`;
}

export async function renderArtifact(container, artifactRef, deps = {}) {
  if (!container) {
    throw new Error("renderArtifact requires a container");
  }
  if (container.dataset) {
    container.dataset.v2ArtifactView = "true";
  }
  const actions = {
    fetchMetrics: (id) => apiGet(`/api/step-outputs/${encodeURIComponent(id)}`),
    previewDataset: previewDatasetApi,
    ...deps,
  };
  const ref = parseArtifactRef(artifactRef);
  if (ref.kind === "dataset") {
    const preview = await actions.previewDataset(ref.value, 50);
    container.innerHTML = datasetTableHtml(preview);
    return preview;
  }
  if (ref.kind === "metrics") {
    const metrics = await actions.fetchMetrics(ref.value);
    container.innerHTML = metricsHtml(metrics);
    return metrics;
  }
  if (ref.kind === "artifact") {
    container.innerHTML = artifactFileHtml(ref.value);
    return ref;
  }
  container.innerHTML = valueHtml(ref.value);
  return ref;
}
