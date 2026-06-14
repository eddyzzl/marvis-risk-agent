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
  const headers = {
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

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
