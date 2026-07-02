"""Shared exception types for the validation pipeline and its submodules.

Split out from marvis/pipeline.py (ARCH-6) so pipeline_io.py, pipeline_cellgen.py,
and pipeline_memory.py can raise/catch the same PipelineError/PipelineCancelled
types as marvis.pipeline itself without a circular import. marvis.pipeline
re-exports both names for backward compatibility with the existing import
surface (tests and callers import them from marvis.pipeline).
"""
from __future__ import annotations

from marvis.domain import TaskStatus


class PipelineError(Exception):
    pass


class PipelineCancelled(PipelineError):
    def __init__(self, message: str, resume_status: TaskStatus) -> None:
        super().__init__(message)
        self.resume_status = resume_status
