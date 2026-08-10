import gzip
import socket

import pytest

from marvis.drafts import FetchError, OfflineError
from marvis.drafts.web_search import fetch_url, web_search


class _Response:
    def __init__(
        self,
        *,
        status_code=200,
        payload=None,
        text="",
        content=None,
        headers=None,
        chunks=None,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {}
        self._chunks = list(chunks) if chunks is not None else None
        self.content = (
            content
            if content is not None
            else b"".join(self._chunks) if self._chunks is not None else text.encode("utf-8")
        )
        self.iterated_chunks = 0

    def json(self):
        return self._payload

    def iter_bytes(self, chunk_size=None):
        del chunk_size
        chunks = self._chunks if self._chunks is not None else [self.content]
        for chunk in chunks:
            self.iterated_chunks += 1
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeClient:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def stream(self, method, url, **kwargs):
        self.owner.calls.append(("stream", method, url, kwargs))
        return self.owner._next_response()


class _FakeHttpx:
    def __init__(self, response):
        self.responses = list(response) if isinstance(response, (list, tuple)) else [response]
        self.calls = []

    def head(self, url, timeout):
        self.calls.append(("head", url, timeout))
        return self.responses[0]

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.responses[-1] if kwargs.get("follow_redirects") else self.responses[0]

    def Client(self, **kwargs):
        self.calls.append(("client", kwargs))
        return _FakeClient(self)

    def _next_response(self):
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def _install_public_dns(monkeypatch, **addresses):
    calls = []

    def fake_getaddrinfo(host, port, *args, **kwargs):
        del args, kwargs
        calls.append((host, port))
        address = addresses.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return calls


def test_web_search_offline_raises_guided_error(monkeypatch):
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: False)

    with pytest.raises(OfflineError, match="上传"):
        web_search("new scorecard method")


def test_web_search_httpx_missing_degrades(monkeypatch):
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: None)

    with pytest.raises(OfflineError, match="httpx"):
        web_search("new scorecard method")


def test_web_search_parses_bounded_results(monkeypatch):
    fake = _FakeHttpx(
        _Response(
            payload={
                "results": [
                    {"title": "A", "url": "https://example.test/a", "snippet": "first"},
                    {"title": "B", "url": "https://example.test/b", "snippet": "second"},
                    {"title": "C", "url": "https://example.test/c", "snippet": "third"},
                ]
            }
        )
    )
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)

    results = web_search("risk strategy", max_results=2)

    assert results == [
        {"title": "A", "url": "https://example.test/a", "snippet": "first"},
        {"title": "B", "url": "https://example.test/b", "snippet": "second"},
    ]
    assert fake.calls[0][0] == "get"


def test_fetch_url_offline_and_http_errors(monkeypatch):
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: False)
    with pytest.raises(OfflineError, match="上传"):
        fetch_url("https://example.test/a")

    fake = _FakeHttpx(_Response(status_code=404, text="missing"))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_url("https://example.test/a")


def test_fetch_url_rejects_oversized_content_and_extracts_text(monkeypatch):
    oversized = _FakeHttpx(_Response(text="abcdef", content=b"abcdef"))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: oversized)
    _install_public_dns(monkeypatch)
    with pytest.raises(FetchError, match="too large"):
        fetch_url("https://example.test/a", max_bytes=3)

    html = _FakeHttpx(
        _Response(
            text="<html><head><title>T</title><script>x()</script></head><body><h1>Title</h1><p>Hello&nbsp;world</p></body></html>"
        )
    )
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: html)

    assert fetch_url("https://example.test/a") == "T Title Hello world"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.test/file",
        "https://user:secret@example.test/private",
        "http://127.0.0.1/admin",
        "http://[::1]/admin",
        "http://[::ffff:127.0.0.1]/admin",
        "http://2130706433/admin",
        "http://0x7f000001/admin",
        "http://100.64.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://192.0.2.1/reserved",
        "http://224.0.0.1/multicast",
    ],
)
def test_fetch_url_rejects_unsafe_url_targets(monkeypatch, url):
    fake = _FakeHttpx(_Response(text="must not be returned"))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)

    with pytest.raises(FetchError, match="not allowed"):
        fetch_url(url)

    assert not any(call[0] == "stream" for call in fake.calls)


def test_fetch_url_rejects_hostname_with_any_non_public_resolution(monkeypatch):
    fake = _FakeHttpx(_Response(text="must not be returned"))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)

    def mixed_getaddrinfo(host, port, *args, **kwargs):
        del host, args, kwargs
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.4", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mixed_getaddrinfo)

    with pytest.raises(FetchError, match="not allowed"):
        fetch_url("https://mixed.example.test/article")

    assert not any(call[0] == "stream" for call in fake.calls)


def test_fetch_url_revalidates_redirects_and_rejects_private_target(monkeypatch):
    redirect = _Response(
        status_code=302,
        headers={"location": "http://127.0.0.1/admin"},
    )
    fake = _FakeHttpx([redirect, _Response(text="private response")])
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)

    with pytest.raises(FetchError, match="not allowed"):
        fetch_url("https://public.example.test/article")

    stream_calls = [call for call in fake.calls if call[0] == "stream"]
    assert [call[2] for call in stream_calls] == ["https://public.example.test/article"]


def test_fetch_url_allows_public_redirect_and_disables_environment_proxy(monkeypatch):
    fake = _FakeHttpx(
        [
            _Response(status_code=301, headers={"location": "https://cdn.example.test/article"}),
            _Response(text="<main>Public article</main>"),
        ]
    )
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    dns_calls = _install_public_dns(monkeypatch)

    assert fetch_url("https://public.example.test/start") == "Public article"

    client_call = next(call for call in fake.calls if call[0] == "client")
    assert client_call[1]["trust_env"] is False
    assert client_call[1]["follow_redirects"] is False
    assert [call[2] for call in fake.calls if call[0] == "stream"] == [
        "https://public.example.test/start",
        "https://cdn.example.test/article",
    ]
    assert [host for host, _port in dns_calls] == [
        "public.example.test",
        "cdn.example.test",
    ]


def test_fetch_url_preserves_public_fetch_with_real_httpx_stream(monkeypatch):
    import httpx

    requests = []

    def handler(request):
        requests.append(
            {
                "url": str(request.url),
                "host": request.headers["host"],
                "sni_hostname": request.extensions["sni_hostname"],
            }
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<main>Real HTTPX public control</main>",
        )

    def transport_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return httpx.MockTransport(handler)

    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr(httpx, "HTTPTransport", transport_factory)
    _install_public_dns(monkeypatch)

    assert fetch_url("https://public.example.test/article") == "Real HTTPX public control"
    assert requests == [
        {
            "url": "https://93.184.216.34/article",
            "host": "public.example.test",
            "sni_hostname": "public.example.test",
        }
    ]


def test_fetch_url_pins_validated_address_against_dns_rebinding(monkeypatch):
    import httpx

    real_client = httpx.Client
    hostname_queries = []
    connected_addresses = []

    def rebinding_getaddrinfo(host, port, *args, **kwargs):
        del args, kwargs
        if host == "rebind.example.test":
            hostname_queries.append(host)
            address = "93.184.216.34" if len(hostname_queries) == 1 else "127.0.0.1"
        else:
            address = str(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    def handler(request):
        resolved = socket.getaddrinfo(request.url.host, request.url.port, type=socket.SOCK_STREAM)
        connected_address = resolved[0][4][0]
        connected_addresses.append(connected_address)
        body = (
            b"<main>Private service response</main>"
            if connected_address == "127.0.0.1"
            else b"<main>Public service response</main>"
        )
        return httpx.Response(200, content=body)

    def client_factory(**kwargs):
        assert kwargs["trust_env"] is False
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_client(**kwargs)

    def transport_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return httpx.MockTransport(handler)

    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)
    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(httpx, "HTTPTransport", transport_factory)

    assert fetch_url("https://rebind.example.test/article") == "Public service response"
    assert hostname_queries == ["rebind.example.test"]
    assert connected_addresses == ["93.184.216.34"]


def test_fetch_url_tries_next_validated_address_after_connect_failure(monkeypatch):
    import httpx

    attempts = []

    def public_getaddrinfo(host, port, *args, **kwargs):
        del host, args, kwargs
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", port)),
        ]

    def handler(request):
        attempts.append(str(request.url))
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("first address unavailable", request=request)
        return httpx.Response(200, content=b"<main>Fallback public response</main>")

    def transport_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return httpx.MockTransport(handler)

    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr(socket, "getaddrinfo", public_getaddrinfo)
    monkeypatch.setattr(httpx, "HTTPTransport", transport_factory)

    assert fetch_url("https://public.example.test/article") == "Fallback public response"
    assert attempts == [
        "https://93.184.216.34/article",
        "https://1.1.1.1/article",
    ]


def test_fetch_url_stops_after_bounded_redirect_count(monkeypatch):
    redirects = [
        _Response(status_code=302, headers={"location": f"/redirect/{index}"})
        for index in range(7)
    ]
    fake = _FakeHttpx(redirects)
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)

    with pytest.raises(FetchError, match="too many redirects"):
        fetch_url("https://example.test/start")

    assert len([call for call in fake.calls if call[0] == "stream"]) == 6


def test_fetch_url_stops_streaming_when_chunked_body_exceeds_limit(monkeypatch):
    response = _Response(
        headers={"transfer-encoding": "chunked"},
        chunks=[b"abc", b"def", b"must-not-be-read"],
    )
    fake = _FakeHttpx(response)
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)

    with pytest.raises(FetchError, match="too large"):
        fetch_url("https://example.test/chunked", max_bytes=5)

    assert response.iterated_chunks == 2


def test_fetch_url_rejects_oversized_content_length_before_reading(monkeypatch):
    response = _Response(
        headers={"content-length": "6"},
        chunks=[b"abcdef"],
    )
    fake = _FakeHttpx(response)
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: fake)
    _install_public_dns(monkeypatch)

    with pytest.raises(FetchError, match="too large"):
        fetch_url("https://example.test/declared-large", max_bytes=5)

    assert response.iterated_chunks == 0


def test_fetch_url_rejects_compressed_body_before_httpx_can_decode_it(monkeypatch):
    import httpx

    requests = []

    def handler(request):
        requests.append(request)
        # The compressed payload is intentionally much smaller than its
        # decoded form; fetch_url must reject it from headers without asking
        # HTTPX to decode it first.
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            content=gzip.compress(b"x" * (1024 * 1024)),
        )

    def transport_factory(**kwargs):
        assert kwargs["trust_env"] is False
        return httpx.MockTransport(handler)

    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr(httpx, "HTTPTransport", transport_factory)
    _install_public_dns(monkeypatch)

    with pytest.raises(FetchError, match="compressed response"):
        fetch_url("https://public.example.test/compressed")

    assert requests[0].headers["accept-encoding"] == "identity"
