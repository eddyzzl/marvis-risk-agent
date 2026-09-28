"""ASGI receive boundaries: reject oversize before consuming the remaining body."""

import asyncio

from fastapi import HTTPException
import pytest
from starlette.requests import Request

from marvis.collection.router import body
from marvis.collection.service import CollectionMaterial


@pytest.mark.parametrize("content_length", [None, b"1", b"16000000"])
def test_collection_body_stops_at_limit_even_without_honest_length(content_length):
    seen = []
    chunk = b"x" * 8_000_000

    async def receive():
        seen.append(len(seen))
        if len(seen) <= 2:
            return {"type": "http.request", "body": chunk, "more_body": True}
        if len(seen) == 3:
            return {"type": "http.request", "body": b"x", "more_body": True}
        raise AssertionError("must not consume later chunks after detecting overflow")

    headers = [] if content_length is None else [(b"content-length", content_length)]
    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": headers},
        receive=receive,
    )
    with pytest.raises(HTTPException) as caught:
        asyncio.run(body(request, CollectionMaterial))
    assert caught.value.status_code == 413
    assert caught.value.detail["code"] == "collection_request_too_large"
    assert len(seen) == 3
    assert not hasattr(request, "_body")


def test_collection_body_accepts_bounded_json_across_chunks():
    chunks = iter(
        [
            b'{"material_id":"small",',
            b'"description":"declared",',
            b'"declarations":{}}',
        ]
    )

    async def receive():
        chunk = next(chunks, b"")
        return {"type": "http.request", "body": chunk, "more_body": bool(chunk)}

    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": []}, receive=receive
    )
    parsed = asyncio.run(body(request, CollectionMaterial))
    assert parsed.material_id == "small"
