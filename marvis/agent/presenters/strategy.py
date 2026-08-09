"""Composed Strategy presenter registry.

Concrete presenters are grouped by change reason so candidate UX, governed
evidence, Pool replay, and lifecycle analytics can evolve independently.
"""

from __future__ import annotations

from . import strategy_candidates, strategy_evidence, strategy_lifecycle, strategy_pool


_DOMAIN_MODULES = (
    strategy_candidates,
    strategy_evidence,
    strategy_lifecycle,
    strategy_pool,
)
_DOMAIN_REGISTRIES = (
    strategy_candidates.CANDIDATE_PRESENTERS,
    strategy_evidence.EVIDENCE_PRESENTERS,
    strategy_lifecycle.LIFECYCLE_PRESENTERS,
    strategy_pool.POOL_PRESENTERS,
)

STRATEGY_RENDERERS = {}
_INTEGRITY_FAILURES = {}
for _module, _registry in zip(_DOMAIN_MODULES, _DOMAIN_REGISTRIES, strict=True):
    overlap = set(STRATEGY_RENDERERS).intersection(_registry)
    if overlap:
        raise RuntimeError(f"duplicate Strategy presenter refs: {sorted(overlap)}")
    STRATEGY_RENDERERS.update(_registry)
    failures = _module.INTEGRITY_FAILURES
    failure_overlap = set(_INTEGRITY_FAILURES).intersection(failures)
    if failure_overlap:
        raise RuntimeError(
            f"duplicate Strategy integrity fallbacks: {sorted(failure_overlap)}"
        )
    _INTEGRITY_FAILURES.update(failures)


def strategy_integrity_failure(tool: str) -> tuple[str, list[dict]] | None:
    failure = _INTEGRITY_FAILURES.get(tool)
    return failure() if failure is not None else None


def __getattr__(name: str):
    """Resolve historical private imports from their new domain owner."""

    for module in _DOMAIN_MODULES:
        try:
            return getattr(module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["STRATEGY_RENDERERS", "strategy_integrity_failure"]
