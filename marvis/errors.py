"""Centralised error taxonomy for MARVIS (ARCH-7).

Two concerns live here, both *convergence only* -- no behaviour changes:

1. :class:`ErrorKind` -- the machine-readable ``error_kind`` / ``to_detail()["kind"]``
   vocabulary. Historically these were bare string literals scattered across the
   plugin runner, the data layer, and the pack typed-errors. It is now *defined*
   in the zero-dependency leaf module :mod:`marvis.error_kinds` and re-exported
   here so ``from marvis.errors import ErrorKind`` keeps working on the host
   side. The pack typed-errors import it from the leaf module directly, because
   they load inside the tool worker's execution environment where ``fastapi``
   may be absent; importing this module (which pulls in ``fastapi`` below) would
   crash there. The *string values* are load-bearing (tests and the subprocess
   protocol assert on them) and live untouched in the leaf module.

2. HTTP error factories (:func:`not_found`, :func:`conflict`, ...) -- thin
   wrappers over :class:`fastapi.HTTPException` that pin the status code and pass
   the ``detail`` through verbatim. They exist so routers stop hand-writing
   ``raise HTTPException(status_code=404, ...)`` at 130+ call sites; the resulting
   response (status + detail) is identical.

Nothing here parses free text or reshapes a payload -- callers keep full control
of the ``detail`` string so the many exact-match test assertions stay green.
"""

from __future__ import annotations

from fastapi import HTTPException

from marvis.error_kinds import ErrorKind


def not_found(detail: str) -> HTTPException:
    """404 -- requested resource does not exist. ``detail`` is passed verbatim."""
    return HTTPException(status_code=404, detail=detail)


def conflict(detail: str) -> HTTPException:
    """409 -- request conflicts with current resource state. ``detail`` verbatim."""
    return HTTPException(status_code=409, detail=detail)


def unprocessable(detail: str) -> HTTPException:
    """422 -- syntactically valid but semantically invalid input. ``detail`` verbatim."""
    return HTTPException(status_code=422, detail=detail)


def bad_request(detail: str) -> HTTPException:
    """400 -- malformed request. ``detail`` is passed verbatim."""
    return HTTPException(status_code=400, detail=detail)


def payload_too_large(detail: str) -> HTTPException:
    """413 -- upload/payload exceeds a configured guardrail. ``detail`` verbatim."""
    return HTTPException(status_code=413, detail=detail)


def forbidden(detail: str) -> HTTPException:
    """403 -- caller is not permitted to perform the action. ``detail`` verbatim."""
    return HTTPException(status_code=403, detail=detail)


def bad_gateway(detail: str) -> HTTPException:
    """502 -- an upstream dependency failed. ``detail`` is passed verbatim."""
    return HTTPException(status_code=502, detail=detail)


def server_error(detail: str) -> HTTPException:
    """500 -- an internal invariant failed. ``detail`` is passed verbatim."""
    return HTTPException(status_code=500, detail=detail)


def precondition_required(detail: str) -> HTTPException:
    """428 -- the request must be conditional (e.g. missing If-Match). ``detail`` verbatim."""
    return HTTPException(status_code=428, detail=detail)


def not_implemented(detail: str) -> HTTPException:
    """501 -- the requested capability is not wired up yet. ``detail`` verbatim."""
    return HTTPException(status_code=501, detail=detail)


__all__ = [
    "ErrorKind",
    "not_found",
    "conflict",
    "unprocessable",
    "bad_request",
    "payload_too_large",
    "forbidden",
    "bad_gateway",
    "server_error",
    "precondition_required",
    "not_implemented",
]
