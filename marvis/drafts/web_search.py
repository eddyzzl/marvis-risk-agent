from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

from marvis.drafts.errors import FetchError, OfflineError


DEFAULT_PROBE_URL = "https://example.com"
DEFAULT_SEARCH_ENDPOINT = "https://api.duckduckgo.com/"
OFFLINE_GUIDANCE = "无网络：请在有网环境产出工具后，通过插件上传导入。"
_MAX_REDIRECTS = 5
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_STREAM_CHUNK_BYTES = 64 * 1024
_URL_NOT_ALLOWED = "URL is not allowed"
_PINNED_ADDRESSES_EXTENSION = "marvis_pinned_addresses"
_MISSING = object()


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
    limit = int(max_bytes)
    if limit < 0:
        raise FetchError("response body is too large")

    current_url = str(url)
    transport = _PinnedHTTPTransport(client)
    with client.Client(
        timeout=20,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as http_client:
        for redirect_count in range(_MAX_REDIRECTS + 1):
            current_url, pinned_addresses = _validated_public_url(current_url)
            with http_client.stream(
                "GET",
                current_url,
                # fetch_url is a bounded text extractor, not a general browser.
                # Ask for wire bytes without a content decoder; a server that
                # ignores this header is rejected below before HTTPX can expand
                # a tiny compressed response into an unbounded allocation.
                headers={"Accept-Encoding": "identity"},
                extensions={_PINNED_ADDRESSES_EXTENSION: pinned_addresses},
            ) as response:
                status_code = int(getattr(response, "status_code", 200))
                if status_code >= 400:
                    raise FetchError(f"HTTP {response.status_code}")

                location = (getattr(response, "headers", {}) or {}).get("location")
                if status_code in _REDIRECT_STATUS_CODES and location:
                    if redirect_count >= _MAX_REDIRECTS:
                        raise FetchError("too many redirects")
                    current_url = urljoin(current_url, str(location))
                    continue

                _reject_content_encoded_response(response)
                content = _read_bounded_body(response, limit)
                return _extract_main_text(_decode_response_body(response, content))[:limit]

    raise FetchError("too many redirects")


def _validated_public_url(url: str) -> tuple[str, tuple[str, ...]]:
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
        has_credentials = parsed.username is not None or parsed.password is not None
    except (TypeError, ValueError):
        raise FetchError(_URL_NOT_ALLOWED) from None

    if scheme not in {"http", "https"} or not hostname or has_credentials:
        raise FetchError(_URL_NOT_ALLOWED)
    if port is not None and not 1 <= port <= 65535:
        raise FetchError(_URL_NOT_ALLOWED)

    addresses = _require_public_addresses(hostname, port or (443 if scheme == "https" else 80))
    return parsed._replace(scheme=scheme, fragment="").geturl(), addresses


def _require_public_addresses(hostname: str, port: int) -> tuple[str, ...]:
    literal = _ip_address(hostname)
    if literal is not None:
        addresses = [literal]
    else:
        try:
            resolved = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError):
            raise FetchError(_URL_NOT_ALLOWED) from None
        addresses = [_ip_address(result[4][0]) for result in resolved]

    if not addresses or any(address is None or not _is_public_address(address) for address in addresses):
        raise FetchError(_URL_NOT_ALLOWED)
    return tuple(dict.fromkeys(str(address) for address in addresses))


def _ip_address(value: str):
    address = str(value).split("%", 1)[0]
    try:
        return ipaddress.ip_address(address)
    except ValueError:
        return None


def _is_public_address(address) -> bool:
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return bool(
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
    )


class _PinnedHTTPTransport:
    """Connect to validated numeric addresses without changing HTTP/TLS origin semantics."""

    def __init__(self, httpx_module):
        self._httpx = httpx_module
        self._transports = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        for transport in self._transports.values():
            transport.close()
        self._transports.clear()

    def handle_request(self, request):
        pinned_addresses = tuple(request.extensions.pop(_PINNED_ADDRESSES_EXTENSION, ()))
        parsed_addresses = [_ip_address(str(address)) for address in pinned_addresses]
        if not pinned_addresses or any(
            address is None or not _is_public_address(address) for address in parsed_addresses
        ):
            raise FetchError(_URL_NOT_ALLOWED)
        pinned_addresses = tuple(str(address) for address in parsed_addresses)

        original_url = request.url
        previous_sni = request.extensions.get("sni_hostname", _MISSING)
        # HTTPX has already built Host from the original URL. The SNI override
        # keeps certificate hostname verification on that origin after pinning.
        request.extensions["sni_hostname"] = original_url.raw_host.decode("ascii")
        origin = (original_url.raw_scheme, original_url.raw_host, original_url.port)
        transport = self._transports.get(origin)
        if transport is None:
            transport = self._httpx.HTTPTransport(trust_env=False)
            self._transports[origin] = transport

        last_error = None
        try:
            for address in pinned_addresses:
                request.url = original_url.copy_with(host=str(address))
                try:
                    return transport.handle_request(request)
                except (self._httpx.ConnectError, self._httpx.ConnectTimeout) as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            raise FetchError(_URL_NOT_ALLOWED)
        finally:
            request.url = original_url
            request.extensions[_PINNED_ADDRESSES_EXTENSION] = pinned_addresses
            if previous_sni is _MISSING:
                request.extensions.pop("sni_hostname", None)
            else:
                request.extensions["sni_hostname"] = previous_sni


def _read_bounded_body(response, limit: int) -> bytes:
    headers = getattr(response, "headers", {}) or {}
    declared_length = headers.get("content-length")
    if declared_length is not None:
        try:
            parsed_length = int(declared_length)
        except (TypeError, ValueError):
            parsed_length = None
        if parsed_length is not None and parsed_length > limit:
            raise FetchError("response body is too large")

    content = bytearray()
    chunk_size = max(1, min(_STREAM_CHUNK_BYTES, limit + 1))
    for chunk in response.iter_bytes(chunk_size=chunk_size):
        if len(content) + len(chunk) > limit:
            raise FetchError("response body is too large")
        content.extend(chunk)
    return bytes(content)


def _reject_content_encoded_response(response) -> None:
    """Fail before HTTPX's decoded iterator can expand a compressed body.

    The byte budget is a safety contract, so silently accepting gzip/deflate/
    brotli would be wrong: ``Response.iter_bytes`` decodes content before this
    module can account for the resulting allocation. ``fetch_url`` requests
    identity bytes and treats servers that ignore that request as unsupported.
    """

    headers = getattr(response, "headers", {}) or {}
    content_encoding = str(headers.get("content-encoding", "")).strip().lower()
    if content_encoding and content_encoding != "identity":
        raise FetchError("compressed response bodies are not supported")


def _decode_response_body(response, content: bytes) -> str:
    encoding = getattr(response, "encoding", None) or "utf-8"
    try:
        return content.decode(str(encoding), errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


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
