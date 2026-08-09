from __future__ import annotations

from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from marvis.settings import Settings


_MATERIAL_UPLOAD_PATH = "/api/validation-batches/material-uploads"
MATERIAL_UPLOAD_MAX_FILES = 1000
# Keep raw multipart framing from reducing the configured material-byte budget.
# This bounds parser-owned temporary storage while allowing normal metadata for
# the parser's 1,000-file contract.
_MULTIPART_OVERHEAD_BYTES = (MATERIAL_UPLOAD_MAX_FILES * 8 * 1024) + (64 * 1024)


class _IngressLimitExceeded(MultiPartException):
    """Trigger Starlette's parser cleanup before the middleware emits 413."""


class ValidationBatchIngressLimitMiddleware:
    """Cap the batch material multipart body before Starlette parses it."""

    def __init__(self, app: ASGIApp, *, settings: Settings) -> None:
        self._app = app
        self._settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not _is_material_upload(scope):
            await self._app(scope, receive, send)
            return

        limit_bytes = (
            self._settings.max_csv_upload_bytes + _MULTIPART_OVERHEAD_BYTES
        )
        content_length = _content_length(scope)
        if content_length is not None and content_length > limit_bytes:
            await _send_too_large(scope, receive, send, limit_bytes=limit_bytes)
            return

        received_bytes = 0
        ingress_limit_exceeded = False

        async def limited_receive() -> Message:
            nonlocal ingress_limit_exceeded, received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit_bytes:
                    ingress_limit_exceeded = True
                    raise _IngressLimitExceeded(_error_detail(limit_bytes))
            return message

        async def limited_send(message: Message) -> None:
            # FastAPI converts MultiPartException into its own 400 response.
            # Suppress that response after our sentinel fires so this outer
            # boundary can preserve the intended 413 contract.
            if not ingress_limit_exceeded:
                await send(message)

        try:
            await self._app(scope, limited_receive, limited_send)
        except _IngressLimitExceeded:
            pass
        if ingress_limit_exceeded:
            await _send_too_large(scope, receive, send, limit_bytes=limit_bytes)


def _is_material_upload(scope: Scope) -> bool:
    return (
        scope["type"] == "http"
        and scope.get("method", "").upper() == "POST"
        and scope.get("path") == _MATERIAL_UPLOAD_PATH
    )


def _content_length(scope: Scope) -> int | None:
    lengths: list[int] = []
    for name, raw_value in scope.get("headers", ()):
        if name.lower() != b"content-length":
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value >= 0:
            lengths.append(value)
    return max(lengths, default=None)


async def _send_too_large(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    limit_bytes: int,
) -> None:
    headers = None
    if _should_close_connection(scope):
        # The request body may be intentionally left unread. Prevent a reused
        # HTTP/1 connection from interpreting those bytes as another request.
        headers = {"Connection": "close"}
    response = JSONResponse(
        status_code=413,
        content={"detail": _error_detail(limit_bytes)},
        headers=headers,
    )
    await response(scope, receive, send)


def _should_close_connection(scope: Scope) -> bool:
    return scope.get("http_version") in {"1.0", "1.1"}


def _error_detail(limit_bytes: int) -> str:
    return (
        "validation batch material upload request exceeds "
        f"ingress limit: limit_bytes={limit_bytes}"
    )
