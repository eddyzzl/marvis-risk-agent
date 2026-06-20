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

export async function api(endpoint, options = {}) {
  const normalizedEndpoint = endpoint.startsWith("/") || endpoint.startsWith("http")
    ? endpoint
    : `/${endpoint}`;
  const body = options.body;
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;
  const headers = { ...(options.headers || {}) };
  const hasContentType = Object.keys(headers).some(
    (name) => name.toLowerCase() === "content-type",
  );
  if (body !== undefined && !isFormData && !hasContentType) {
    headers["Content-Type"] = "application/json";
  }
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

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
