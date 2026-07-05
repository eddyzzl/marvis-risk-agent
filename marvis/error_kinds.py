"""Zero-dependency leaf module for the ``error_kind`` string taxonomy (ARCH-7).

:class:`ErrorKind` lives here, apart from :mod:`marvis.errors`, because the
constant values cross the host<->worker subprocess protocol boundary: the pack
typed-errors (``marvis/packs/*/errors.py``) import ``ErrorKind`` to build their
``to_detail()["kind"]`` payloads, and those pack modules load inside the tool
worker's execution environment, which may not have the server-only dependency
stack (``fastapi`` et al.) installed. Keeping ``ErrorKind`` in a leaf module with
zero internal marvis dependencies -- mirroring :mod:`marvis.plugins.contracts`'s
"worker entrypoint import must stay dependency-free" rule -- means importing it
never pulls in ``fastapi``.

:mod:`marvis.errors` re-exports :class:`ErrorKind` from here, so every existing
``from marvis.errors import ErrorKind`` on the host side keeps working; the HTTP
error factories and the ``fastapi`` import stay in :mod:`marvis.errors`.

The *string values* are load-bearing (tests and the subprocess protocol assert on
them), so every constant below is byte-for-byte identical to the literal it
replaces.
"""

from __future__ import annotations


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
    LABEL_SEMANTICS_NOT_DECLARED = "label_semantics_not_declared"
    SCORE_DIRECTION_CONFLICT = "score_direction_conflict"
    PERFORMANCE_FRAME_INVALID = "performance_frame_invalid"
    DATASET_TOO_LARGE = "dataset_too_large"

    # --- Pack typed errors (marvis/packs/*/errors.py to_detail kinds) ---
    STRATEGY_NOT_ADOPTED = "strategy_not_adopted"
    MISSING_BASELINE = "missing_baseline"
    REPORT_SCORE_MISSING = "report_score_missing"


__all__ = ["ErrorKind"]
