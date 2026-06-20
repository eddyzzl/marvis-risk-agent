import { apiGet } from "../api.js";
import { escapeHtml } from "../ui-utils.js";
import { previewDataset as previewDatasetApi } from "./api_v2.js";

function closest(target, selector) {
  return typeof target?.closest === "function" ? target.closest(selector) : null;
}

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

function outputRefKind(value) {
  const text = String(value || "");
  const index = text.indexOf(":");
  if (index < 1) {
    return "";
  }
  const kind = text.slice(0, index);
  return new Set(["dataset", "metrics", "artifact", "value"]).has(kind) ? kind : "";
}

function metricsCellHtml(value) {
  const text = cellText(value);
  const kind = outputRefKind(text);
  if (kind) {
    return `<button type="button" data-artifact="${escapeHtml(text)}">Open ${escapeHtml(kind)}</button>`;
  }
  return escapeHtml(text);
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

function extensionOf(value) {
  const match = String(value || "").toLowerCase().match(/\.([a-z0-9]+)(?:[?#].*)?$/);
  return match ? match[1] : "";
}

function artifactUrl(encodedId) {
  return `/api/artifacts/${encodedId}`;
}

function isImageArtifact(id) {
  return new Set(["png", "jpg", "jpeg", "gif", "webp"]).has(extensionOf(id));
}

function isPreviewableReportArtifact(id) {
  return new Set(["docx", "pdf", "html", "htm"]).has(extensionOf(id));
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
    .map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${metricsCellHtml(value)}</td></tr>`)
    .join("");
  return `<table class="metrics-preview"><tbody>${rows}</tbody></table>`;
}

export function artifactFileHtml(artifactId) {
  const id = String(artifactId || "");
  const encoded = encodeURIComponent(id);
  const url = artifactUrl(encoded);
  if (isImageArtifact(id)) {
    return `<section class="artifact-file artifact-image">
      <span>${escapeHtml(id)}</span>
      <img data-artifact-image src="${url}" alt="${escapeHtml(id)}">
      <a data-artifact-download href="${url}">Download</a>
    </section>`;
  }
  const preview = isPreviewableReportArtifact(id)
    ? `<a data-artifact-preview href="${url}/preview">Preview</a>`
    : "";
  return `<section class="artifact-file">
    <span>${escapeHtml(id)}</span>
    ${preview}
    <a data-artifact-download href="${url}">Download</a>
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

function defaultShowError(message) {
  if (typeof alert === "function") {
    alert(message);
    return;
  }
  console.error(message);
}

function resolvePreviewContainer(container) {
  return typeof container === "function" ? container() : container;
}

export function attachArtifactHandlers(root, container = null, deps = {}) {
  if (!root || typeof root.addEventListener !== "function") {
    throw new Error("attachArtifactHandlers requires a stable event root");
  }
  const actions = {
    renderArtifact,
    showError: defaultShowError,
    ...deps,
  };

  const handler = async (event) => {
    const artifactButton = closest(event.target, "[data-artifact]");
    if (!artifactButton?.dataset?.artifact) {
      return;
    }
    event.preventDefault?.();
    const previewContainer = resolvePreviewContainer(container)
      || root.querySelector?.("#artifactPanel");
    if (!previewContainer) {
      actions.showError("artifact preview panel unavailable");
      return;
    }
    try {
      await actions.renderArtifact(previewContainer, artifactButton.dataset.artifact);
    } catch (error) {
      actions.showError(error?.message || "artifact preview failed");
    }
  };

  root.addEventListener("click", handler);
  return () => root.removeEventListener?.("click", handler);
}
