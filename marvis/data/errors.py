from __future__ import annotations


class DataLayerError(RuntimeError):
    """Base error for deterministic data-layer operations."""


class DataBackendError(DataLayerError):
    """Raised when the tabular backend cannot complete a data operation."""


class DataIngestError(DataLayerError):
    """Raised when a source file cannot be normalized into a dataset."""


class DataSecurityError(DataLayerError):
    """Raised when untrusted input would enter a backend query."""


__all__ = ["DataBackendError", "DataIngestError", "DataLayerError", "DataSecurityError"]
