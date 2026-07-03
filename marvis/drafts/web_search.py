from __future__ import annotations

import html
import os
import re

from marvis.drafts.errors import FetchError, OfflineError


DEFAULT_PROBE_URL = "https://example.com"
DEFAULT_SEARCH_ENDPOINT = "https://api.duckduckgo.com/"
OFFLINE_GUIDANCE = "无网络：请在有网环境产出工具后，通过插件上传导入。"


def network_available() -> bool:
    client = _httpx()
    if client is None:
        return False
    try:
        response = client.head(_probe_url(), timeout=2)
    except Exception:
        return False
    return int(getattr(response, "status_code", 599)) < 500


def web_search(query: str, *, max_results: int = 5) -> list[dict]:
    if not network_available():
        raise OfflineError(OFFLINE_GUIDANCE)
    client = _httpx()
    if client is None:
        raise OfflineError("httpx 未安装；请改用外部产出工具后通过插件上传导入。")
    response = client.get(
        _search_endpoint(),
        params={"q": query, "n": int(max_results)},
        timeout=15,
    )
    if int(getattr(response, "status_code", 200)) >= 400:
        raise FetchError(f"HTTP {response.status_code}")
    return _parse_search_results(response.json())[: int(max_results)]


def fetch_url(url: str, *, max_bytes: int = 500_000) -> str:
    if not network_available():
        raise OfflineError(OFFLINE_GUIDANCE)
    client = _httpx()
    if client is None:
        raise OfflineError("httpx 未安装；请改用外部产出工具后通过插件上传导入。")
    response = client.get(url, timeout=20, follow_redirects=True)
    if int(getattr(response, "status_code", 200)) >= 400:
        raise FetchError(f"HTTP {response.status_code}")
    content = getattr(response, "content", b"") or b""
    if len(content) > int(max_bytes):
        raise FetchError("response body is too large")
    return _extract_main_text(str(getattr(response, "text", "")))[: int(max_bytes)]


def _parse_search_results(payload: dict) -> list[dict]:
    raw_results = payload.get("results") or payload.get("items") or []
    results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        snippet = str(item.get("snippet") or item.get("summary") or "").strip()
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _extract_main_text(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    return " ".join(text.split())


def _probe_url() -> str:
    return os.getenv("MARVIS_PROBE_URL", DEFAULT_PROBE_URL)


def _search_endpoint() -> str:
    return os.getenv("MARVIS_SEARCH_ENDPOINT", DEFAULT_SEARCH_ENDPOINT)


def _httpx():
    try:
        import httpx
    except ImportError:
        return None
    return httpx


__all__ = ["fetch_url", "network_available", "web_search"]
