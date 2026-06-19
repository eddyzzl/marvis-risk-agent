import pytest

from marvis.drafts import FetchError, OfflineError
from marvis.drafts.web_search import fetch_url, web_search


class _Response:
    def __init__(self, *, status_code=200, payload=None, text="", content=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")

    def json(self):
        return self._payload


class _FakeHttpx:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def head(self, url, timeout):
        self.calls.append(("head", url, timeout))
        return self.response

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.response


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
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_url("https://example.test/a")


def test_fetch_url_rejects_oversized_content_and_extracts_text(monkeypatch):
    oversized = _FakeHttpx(_Response(text="abcdef", content=b"abcdef"))
    monkeypatch.setattr("marvis.drafts.web_search.network_available", lambda: True)
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: oversized)
    with pytest.raises(FetchError, match="too large"):
        fetch_url("https://example.test/a", max_bytes=3)

    html = _FakeHttpx(
        _Response(
            text="<html><head><title>T</title><script>x()</script></head><body><h1>Title</h1><p>Hello&nbsp;world</p></body></html>"
        )
    )
    monkeypatch.setattr("marvis.drafts.web_search._httpx", lambda: html)

    assert fetch_url("https://example.test/a") == "T Title Hello world"
