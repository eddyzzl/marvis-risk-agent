export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.payload = payload;
  }
}

function errorDetailText(error) {
  const candidates = [
    error?.detail,
    error?.payload?.detail,
    error?.message,
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) return candidate.trim();
  }
  return "";
}

export function apiConflictKind(error) {
  if (Number(error?.status) !== 409) return "not_conflict";
  const detail = errorDetailText(error);
  if (
    detail.includes("该任务正在执行上一步，请等待完成")
    || detail === "task already has an active stage"
  ) {
    return "active_driver_job";
  }
  if (
    /(?:当前待确认步骤|该确认按钮对应的步骤|该操作对应的计划|确认快照).*已变化/.test(detail)
    || /^(?:stale gate|confirmation snapshot changed)$/i.test(detail)
    || /(?:plan|plan step|step).*(?:changed|revision|snapshot|fingerprint).*confirm/i.test(detail)
    || /(?:plan|plan step|step) (?:revision|snapshot|fingerprint) changed/i.test(detail)
  ) {
    return "confirmation_snapshot_stale";
  }
  return "other_conflict";
}

export function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return detail || "请求失败";
}

export async function readErrorPayload(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json();
    const detail = payload.detail || payload;
    return {
      detail,
      message: formatErrorDetail(detail),
      payload,
    };
  }
  const message = (await response.text()) || "请求失败";
  return { detail: message, message, payload: null };
}

function isFormDataBody(body) {
  return typeof FormData !== "undefined" && body instanceof FormData;
}

function hasContentType(headers) {
  return Object.keys(headers || {}).some((name) => name.toLowerCase() === "content-type");
}

function requestBodyOptions(body, headers = {}) {
  if (body === undefined) {
    return { headers };
  }
  if (isFormDataBody(body)) {
    return { body, headers };
  }
  const nextHeaders = hasContentType(headers)
    ? { ...headers }
    : { "Content-Type": "application/json", ...headers };
  return {
    body: typeof body === "string" ? body : JSON.stringify(body),
    headers: nextHeaders,
  };
}

// GAP-5: when the server is started with MARVIS_LOCAL_TOKEN set, the index
// page embeds it into <body data-marvis-local-token>. The shared-host guard
// requires the credential for every private local API read, and requires this
// explicit header (rather than browser-cached Basic credentials) for writes.
// It is never attached to an off-origin URL.
export function localToken() {
  return typeof document !== "undefined" ? document.body?.dataset?.marvisLocalToken || "" : "";
}

function hasHeader(headers, expectedName) {
  return Object.keys(headers).some((name) => name.toLowerCase() === expectedName.toLowerCase());
}

function applicationBaseUrl() {
  if (typeof document !== "undefined" && typeof document.baseURI === "string" && document.baseURI) {
    return document.baseURI;
  }
  if (typeof location !== "undefined" && typeof location.href === "string" && location.href) {
    return location.href;
  }
  return "";
}

function normalizedApiEndpoint(endpoint) {
  const value = String(endpoint || "").trim();
  if (!value) throw new ApiError("API endpoint is required");
  if (/^https?:\/\//i.test(value)) return value;
  // A protocol-relative or non-HTTP URL would escape the mounted application
  // prefix and must never inherit its credentials.
  if (value.startsWith("//") || /^[a-z][a-z0-9+.-]*:/i.test(value)) {
    throw new ApiError("invalid API endpoint");
  }
  // API callers historically used both `/api/...` and `api/...`. Resolve both
  // below the document base so a JupyterHub `/proxy/<port>/` deployment keeps
  // requests inside the application instead of jumping to the origin root.
  const relative = value.replace(/^\/+/, "");
  const base = applicationBaseUrl();
  // Unit-test/SSR callers do not have a document location. Preserve the
  // historic origin-root form there; browsers always take the mounted path.
  return base ? new URL(relative, base).toString() : `/${relative}`;
}

function isSameOriginEndpoint(endpoint) {
  const base = applicationBaseUrl() || "http://marvis.local/";
  try {
    return new URL(endpoint, base).origin === new URL(base).origin;
  } catch (_error) {
    return false;
  }
}

export async function api(endpoint, options = {}) {
  const normalizedEndpoint = normalizedApiEndpoint(endpoint);
  const body = options.body;
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  const headers = { ...(options.headers || {}) };
  if (body !== undefined && !isFormData && !hasContentType(headers)) {
    headers["Content-Type"] = "application/json";
  }
  const method = (options.method || "GET").toUpperCase();
  const token = localToken();
  if (token && isSameOriginEndpoint(normalizedEndpoint) && !hasHeader(headers, "X-Marvis-Token")) {
    headers["X-Marvis-Token"] = token;
  }
  const response = await fetch(normalizedEndpoint, {
    ...options,
    headers,
  });
  if (!response.ok) {
    const error = await readErrorPayload(response);
    throw new ApiError(error.message, {
      status: response.status,
      detail: error.detail,
      payload: error.payload,
    });
  }
  if (response.status === 204) {
    return null;
  }
  return response.json();
}

export function apiGet(endpoint, options = {}) {
  return api(endpoint, {
    ...options,
    method: "GET",
  });
}

export function apiPost(endpoint, body = {}, options = {}) {
  const headers = { ...(options.headers || {}) };
  const bodyOptions = requestBodyOptions(body, headers);
  return api(endpoint, {
    ...options,
    method: "POST",
    ...bodyOptions,
  });
}

export function apiPut(endpoint, body = {}, options = {}) {
  const headers = { ...(options.headers || {}) };
  const bodyOptions = requestBodyOptions(body, headers);
  return api(endpoint, {
    ...options,
    method: "PUT",
    ...bodyOptions,
  });
}

export function apiDelete(endpoint, options = {}) {
  return api(endpoint, {
    ...options,
    method: "DELETE",
  });
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
