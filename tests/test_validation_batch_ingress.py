import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from marvis.app import create_app
from marvis.routers.validation_batches import _MAX_MATERIAL_UPLOAD_FILES
from marvis.settings import build_settings


def _client_with_ingress_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    limit_bytes: int,
) -> TestClient:
    monkeypatch.setattr(
        "marvis.validation_batch_ingress._MULTIPART_OVERHEAD_BYTES",
        0,
    )
    settings = build_settings(tmp_path)
    object.__setattr__(settings, "max_csv_upload_bytes", limit_bytes)
    return TestClient(create_app(settings))


def test_batch_material_upload_rejects_content_length_before_multipart_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client_with_ingress_limit(
        tmp_path,
        monkeypatch,
        limit_bytes=64,
    )

    response = client.post(
        "/api/validation-batches/material-uploads",
        content=b"x" * 65,
        headers={
            "content-length": "65",
            "content-type": "multipart/form-data; boundary=not-present",
        },
    )

    assert response.status_code == 413, response.text
    assert response.json() == {
        "detail": "validation batch material upload request exceeds "
        "ingress limit: limit_bytes=64"
    }
    assert response.headers["connection"] == "close"


def test_batch_material_upload_rejects_chunked_body_during_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client_with_ingress_limit(
        tmp_path,
        monkeypatch,
        limit_bytes=64,
    )
    encoded = httpx.Request(
        "POST",
        "http://testserver/irrelevant",
        files={"files": ("model.csv", b"small-file", "text/csv")},
        data={"relative_paths": "model.csv"},
    )
    body = encoded.read()

    def chunks():
        for offset in range(0, len(body), 11):
            yield body[offset : offset + 11]

    request = client.build_request(
        "POST",
        "/api/validation-batches/material-uploads",
        content=chunks(),
        headers={"content-type": encoded.headers["content-type"]},
    )
    assert "content-length" not in request.headers

    response = client.send(request)

    assert response.status_code == 413, response.text
    assert response.json() == {
        "detail": "validation batch material upload request exceeds "
        "ingress limit: limit_bytes=64"
    }
    assert response.headers["connection"] == "close"


def test_batch_streaming_limit_stops_before_consuming_remaining_request_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "marvis.validation_batch_ingress._MULTIPART_OVERHEAD_BYTES",
        0,
    )
    settings = build_settings(tmp_path)
    object.__setattr__(settings, "max_csv_upload_bytes", 256)
    app = create_app(settings)
    encoded = httpx.Request(
        "POST",
        "http://testserver/irrelevant",
        files={"files": ("model.csv", b"x" * 512, "text/csv")},
        data={"relative_paths": "model.csv"},
    )
    body = encoded.read()
    chunks = [body[offset : offset + 64] for offset in range(0, len(body), 64)]
    sent: list[dict] = []
    receive_calls = 0

    async def exchange() -> None:
        nonlocal receive_calls

        async def receive() -> dict:
            nonlocal receive_calls
            receive_calls += 1
            if receive_calls <= len(chunks):
                return {
                    "type": "http.request",
                    "body": chunks[receive_calls - 1],
                    "more_body": receive_calls < len(chunks),
                }
            return {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)

        await app(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/api/validation-batches/material-uploads",
                "raw_path": b"/api/validation-batches/material-uploads",
                "root_path": "",
                "query_string": b"",
                "headers": [
                    (b"host", b"testserver"),
                    (
                        b"content-type",
                        encoded.headers["content-type"].encode("latin-1"),
                    ),
                ],
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
            },
            receive,
            send,
        )

    asyncio.run(exchange())

    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    headers = dict(start["headers"])
    assert start["status"] == 413
    assert json.loads(response_body) == {
        "detail": "validation batch material upload request exceeds "
        "ingress limit: limit_bytes=256"
    }
    assert headers[b"connection"] == b"close"
    assert receive_calls < len(chunks)


def test_batch_ingress_limit_does_not_apply_to_other_post_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client_with_ingress_limit(
        tmp_path,
        monkeypatch,
        limit_bytes=64,
    )

    response = client.post("/api/tasks", json={"padding": "x" * 200})

    assert response.status_code == 422
    assert response.headers.get("connection") != "close"


def test_batch_ingress_limit_preserves_outer_access_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = _client_with_ingress_limit(
        tmp_path,
        monkeypatch,
        limit_bytes=64,
    )

    response = client.post(
        "/api/validation-batches/material-uploads",
        content=b"x" * 65,
        headers={
            "content-length": "65",
            "content-type": "multipart/form-data; boundary=not-present",
            "x-forwarded-for": "203.0.113.7",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "unsafe API methods are limited to local clients"
    )
    assert response.headers.get("connection") != "close"


def test_batch_ingress_allows_multipart_framing_above_material_byte_limit(
    tmp_path: Path,
):
    settings = build_settings(tmp_path)
    object.__setattr__(settings, "max_csv_upload_bytes", 64)
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/validation-batches/material-uploads",
        files={"files": ("model.csv", b"small-file", "text/csv")},
        data={"relative_paths": "model.csv"},
    )

    assert response.status_code == 201, response.text
    assert response.json()["files"] == [
        {"relative_path": "model.csv", "size_bytes": 10}
    ]


def test_batch_file_count_contract_matches_starlette_parser_limit(tmp_path: Path):
    assert _MAX_MATERIAL_UPLOAD_FILES == 1000
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/validation-batches/material-uploads",
        files=[
            ("files", (f"material-{index}.csv", b"", "text/csv"))
            for index in range(1001)
        ],
    )

    assert response.status_code == 400
    assert "files" in response.json()["detail"].lower()
    assert "1000" in response.json()["detail"]
