export function formatErrorDetail(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  if (detail && typeof detail === "object") {
    return JSON.stringify(detail);
  }
  return detail || "请求失败";
}

export async function readErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return formatErrorDetail(payload.detail || payload);
  }
  return (await response.text()) || "请求失败";
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

export async function api(endpoint, options = {}) {
  const normalizedEndpoint = endpoint.startsWith("/") || endpoint.startsWith("http")
    ? endpoint
    : `/${endpoint}`;
  const headers = isFormDataBody(options.body)
    ? { ...(options.headers || {}) }
    : {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      };
  const response = await fetch(normalizedEndpoint, {
    ...options,
    headers,
  });
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
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
