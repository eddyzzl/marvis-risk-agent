"""Compatibility exports for the Draft Language v1 policy.

The implementation intentionally lives at :mod:`marvis.draft_language` so
the subprocess worker can import the policy without importing the eager
``marvis.drafts`` package (which owns authoring, persistence, and LLM-facing
dependencies).  Keep this module for callers that adopted the original Draft
namespace during the v1 rollout.
"""

from marvis.draft_language import (
    ALLOWED_BUILTINS,
    ALLOWED_IMPORT_ROOTS,
    DRAFT_EXECUTION_PROFILE,
    DRAFT_LANGUAGE_VERSION,
    DraftCapabilityBinding,
    DraftLanguageError,
    ValidatedDraft,
    validate_draft_source,
)


__all__ = [
    "ALLOWED_BUILTINS",
    "ALLOWED_IMPORT_ROOTS",
    "DRAFT_EXECUTION_PROFILE",
    "DRAFT_LANGUAGE_VERSION",
    "DraftCapabilityBinding",
    "DraftLanguageError",
    "ValidatedDraft",
    "validate_draft_source",
]
