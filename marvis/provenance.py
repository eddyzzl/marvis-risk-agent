"""T3-2: minimal lineage tuple for a headline decision number.

Every number the platform puts in front of a human for a decision should be able
to answer "where did you come from?". :class:`NumberProvenance` is the smallest
tuple that pins a number to its inputs so two runs producing the same number can
be shown to have used the same data + code + parameters + seed:

- ``dataset_fingerprint`` -- content identity of the source dataset(s), reusing
  the registry's sha256 ``content_hash`` (see marvis/data/registry.py).
- ``code_version`` -- the app version string (marvis.__version__), the same
  version the static-asset cache buster keys off.
- ``params_digest`` -- a stable sha256 of the tool inputs (canonical JSON),
  reusing the codebase's existing input-hash convention.
- ``seed`` -- the deterministic seed the computation used, if any.

This is intentionally NOT a lineage graph and NOT a cross-report query surface
(both explicitly deferred by the T3 plan). It rides on the existing evidence
envelope / gate payload; a renderer expands it into a "数字溯源" detail block.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

from marvis import __version__


def code_version() -> str:
    """The app version string used as the lineage's ``code_version`` -- the same
    ``marvis.__version__`` that drives the static-asset cache buster in app.py."""
    return __version__


def params_digest(inputs: Any) -> str:
    """Stable ``sha256:`` digest of a computation's inputs, canonicalized to JSON.

    Reuses the codebase's canonical-hash convention (sorted keys, UTF-8 preserved,
    ``default=str`` for non-JSON scalars) so the same inputs always digest the same
    across processes -- the identity half of "same number => same params". Matches
    marvis/orchestrator/executor.py:_payload_hash so the digest is comparable to
    the evidence envelope's ``input_hash``.
    """
    encoded = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(content_hashes: Iterable[str | None]) -> str:
    """Combine one-or-more source-dataset ``content_hash`` values into a single
    dataset fingerprint string.

    A single dataset fingerprints to its own ``sha256:<hash>``; multiple datasets
    (e.g. a join's anchor + feature) fingerprint to a stable sha256 over the
    ordered list, so the tuple identifies the exact combination of inputs. Missing
    hashes (datasets registered before content_hash existed) are recorded as the
    literal ``"unknown"`` so the fingerprint stays deterministic and the gap is
    visible rather than silently dropped.
    """
    ordered = [str(item) if item else "unknown" for item in content_hashes]
    if not ordered:
        return "sha256:" + hashlib.sha256(b"[]").hexdigest()
    if len(ordered) == 1 and ordered[0] != "unknown":
        return f"sha256:{ordered[0]}"
    encoded = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class NumberProvenance:
    """The minimal ``(dataset_fingerprint, code_version, params_digest, seed)``
    lineage tuple attached to a headline gate number."""

    dataset_fingerprint: str
    code_version: str
    params_digest: str
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "code_version": self.code_version,
            "params_digest": self.params_digest,
            "seed": self.seed,
        }

    @classmethod
    def build(
        cls,
        *,
        content_hashes: Iterable[str | None],
        params: Any,
        seed: int | None = None,
    ) -> "NumberProvenance":
        """Assemble a provenance tuple from raw ingredients: the source datasets'
        content hashes, the tool inputs to digest, and the seed. ``code_version``
        is filled from the app version automatically."""
        return cls(
            dataset_fingerprint=dataset_fingerprint(content_hashes),
            code_version=code_version(),
            params_digest=params_digest(params),
            seed=seed,
        )


__all__ = [
    "NumberProvenance",
    "code_version",
    "dataset_fingerprint",
    "params_digest",
]
