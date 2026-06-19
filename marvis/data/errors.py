from __future__ import annotations


class DataLayerError(RuntimeError):
    """Base error for deterministic data-layer operations."""


class DataBackendError(DataLayerError):
    """Raised when the tabular backend cannot complete a data operation."""


class DataIngestError(DataLayerError):
    """Raised when a source file cannot be normalized into a dataset."""


class DedupRequiredError(DataLayerError):
    """Raised when a non-unique feature key needs a user-selected dedup strategy."""


class FanOutError(DataLayerError):
    """Raised when a join would expand beyond the anchor row count."""


class JoinNotConfirmedError(DataLayerError):
    """Raised when executing a join plan before every join is confirmed."""


class DataSecurityError(DataLayerError):
    """Raised when untrusted input would enter a backend query."""


__all__ = [
    "DataBackendError",
    "DataIngestError",
    "DataLayerError",
    "DataSecurityError",
    "DedupRequiredError",
    "FanOutError",
    "JoinNotConfirmedError",
]
