"""Centralised error taxonomy for MARVIS (ARCH-7).

Two concerns live here, both *convergence only* -- no behaviour changes:

1. :class:`ErrorKind` -- the machine-readable ``error_kind`` / ``to_detail()["kind"]``
   vocabulary. Historically these were bare string literals scattered across the
   plugin runner, the data layer, and the pack typed-errors. The *string values*
   are load-bearing (tests and the subprocess protocol assert on them), so every
   constant below is byte-for-byte identical to the literal it replaces. This
   class is the single place to look up "what error kinds exist".

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


class ErrorKind:
    """Canonical ``error_kind`` string values (the structured error taxonomy).

    These are *not* an ``enum.Enum``: the raw ``str`` values cross the subprocess
    protocol boundary and are asserted on directly by tests, so plain string
    constants keep the wire format and comparisons unchanged. Grouped by origin.
    """

    # --- Plugin subprocess runner / hook dispatcher (marvis/plugins) ---
    RESOURCE_LIMIT = "resource_limit"
    PROTOCOL = "protocol"
    PROTOCOL_VERSION_MISMATCH = "protocol_version_mismatch"
    SCHEMA = "schema"
    PERMISSION = "permission"
    EXECUTION = "execution"
    HOOK = "hook"
    AUDIT = "audit"

    # --- Data layer typed errors (marvis/data/errors.py to_detail kinds) ---
    NAN_LABEL_NOT_CONFIRMED = "nan_label_not_confirmed"
    SCORE_DIRECTION_CONFLICT = "score_direction_conflict"
    PERFORMANCE_FRAME_INVALID = "performance_frame_invalid"
    DATASET_TOO_LARGE = "dataset_too_large"

    # --- Pack typed errors (marvis/packs/*/errors.py to_detail kinds) ---
    STRATEGY_NOT_ADOPTED = "strategy_not_adopted"
    MISSING_BASELINE = "missing_baseline"
    REPORT_SCORE_MISSING = "report_score_missing"


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
]
