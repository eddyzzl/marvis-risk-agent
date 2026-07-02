export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.payload = payload;
  }
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

export async function readErrorMessage(response) {
  return (await readErrorPayload(response)).message;
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
// page embeds it into <body data-marvis-local-token>. Non-GET requests must
// echo it back via X-Marvis-Token or the shared-host access guard rejects
// them (see marvis/app.py _local_access_guard). Left blank (the default,
// MARVIS_LOCAL_TOKEN unset), this header is simply omitted and behavior is
// unchanged.
function localToken() {
  return typeof document !== "undefined" ? document.body?.dataset?.marvisLocalToken || "" : "";
}

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);

export async function api(endpoint, options = {}) {
  const normalizedEndpoint = endpoint.startsWith("/") || endpoint.startsWith("http")
    ? endpoint
    : `/${endpoint}`;
  const body = options.body;
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  const headers = { ...(options.headers || {}) };
  if (body !== undefined && !isFormData && !hasContentType(headers)) {
    headers["Content-Type"] = "application/json";
  }
  const method = (options.method || "GET").toUpperCase();
  const token = localToken();
  if (token && !SAFE_METHODS.has(method) && !("X-Marvis-Token" in headers)) {
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

export function apiDelete(endpoint, options = {}) {
  return api(endpoint, {
    ...options,
    method: "DELETE",
  });
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
