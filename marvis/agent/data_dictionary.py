"""Shared data-dictionary resolution helpers (GAP-4).

A "data dictionary" is a CSV/Excel material whose filename contains 字典/dictionary
(``marvis.files.classify_file`` → ``FileRole.DATA_DICTIONARY``), with columns naming
each data column plus its business meaning (e.g. 特征名/含义). Historically only the
modeling setup flow (``marvis.agent.modeling_setup._resolve_feature_dictionary_id``)
registered such a file as a dataset; this module generalizes that same
detect-once-register-as-dataset pattern so every V2 setup flow (join/feature/vintage/
modeling) picks it up, and adds a small read helper for consumers (screen gate, JOIN
gate, LLM gate prompt) that only need a compact ``{column: business_name}`` map rather
than the full dictionary frame.

Read-only / best-effort throughout: any failure to read or parse a dictionary file
never blocks the caller's own setup flow — callers get back "" / {} on failure, exactly
as if no dictionary were present (INV-1: presentation/context only, never changes what
gets computed).
"""

from __future__ import annotations

from pathlib import Path

from marvis.domain import FileRole
from marvis.files import scan_source_dir

# Historical role string used by the modeling flow's proposal slots kept as the
# canonical registration role; FileRole.DATA_DICTIONARY.value is accepted too since
# the V1 validation flow's scanner may have already registered one under that name.
DICTIONARY_ROLE = "feature_dictionary"
_DICTIONARY_ROLES = frozenset({DICTIONARY_ROLE, FileRole.DATA_DICTIONARY.value})

# Column-name conventions accepted for the "which data column" and "business name"
# fields, mirroring marvis.packs.modeling.report_compute.build_feature_dictionary.
_COLUMN_NAME_HEADERS = ("特征名", "字段名", "列名", "feature", "feature_name", "column")
_BUSINESS_NAME_HEADERS = ("业务名称", "含义", "meaning", "description", "业务含义")
# Cap the map passed into any single gate/table/prompt payload — a dictionary can run
# to thousands of rows for third-party bureau data; only the columns actually present
# in the caller's table need a lookup, but this bounds the worst case (e.g. a raw JOIN
# gate that hasn't narrowed columns yet) so metadata size stays predictable.
MAX_DICTIONARY_ENTRIES = 2000


def first_data_dictionary_id(datasets) -> str:
    """The first already-registered dictionary dataset's id, or ``""``."""
    for dataset in datasets:
        if str(getattr(dataset, "role", "")) in _DICTIONARY_ROLES:
            return str(dataset.id)
    return ""


def resolve_data_dictionary_id(registry, task_id: str, source_dir) -> str:
    """Return a registered data-dictionary dataset id for ``task_id``, registering one
    from ``source_dir`` on first use if a dictionary-named file is present. Generalizes
    the modeling-only ``_resolve_feature_dictionary_id`` so every V2 setup flow that
    scans ``source_dir`` (join/feature/vintage/modeling) shares one detection path."""
    existing = first_data_dictionary_id(registry.list_for_task(task_id))
    if existing:
        return existing
    if source_dir is None:
        return ""
    try:
        artifacts = scan_source_dir(Path(source_dir))
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return ""
    for artifact in artifacts:
        if artifact.role == FileRole.DATA_DICTIONARY:
            dataset = registry.register_from_upload(
                task_id,
                Path(artifact.path),
                role=DICTIONARY_ROLE,
            )
            return dataset.id
    return ""


def load_business_names(backend, registry, dictionary_dataset_id: str) -> dict[str, str]:
    """Read a registered dictionary dataset into a compact ``{column: business_name}``
    map. Returns ``{}`` for a missing id, an unreadable file, or a file with neither a
    recognized column-name header nor a recognized business-name header — never raises,
    since this is optional presentation context (INV-1)."""
    if not dictionary_dataset_id:
        return {}
    try:
        dataset = registry.get(str(dictionary_dataset_id))
        frame = backend.read_frame(registry.resolve_path(dataset.id))
    except Exception:
        return {}
    name_col = _first_existing(frame, _COLUMN_NAME_HEADERS)
    meaning_col = _first_existing(frame, _BUSINESS_NAME_HEADERS)
    if not name_col or not meaning_col:
        return {}
    out: dict[str, str] = {}
    for _, row in frame.iterrows():
        if len(out) >= MAX_DICTIONARY_ENTRIES:
            break
        key = row.get(name_col)
        if key is None or (isinstance(key, float) and key != key):  # NaN
            continue
        value = row.get(meaning_col)
        text = "" if value is None or (isinstance(value, float) and value != value) else str(value).strip()
        if text:
            out[str(key)] = text
    return out


def _first_existing(frame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


__all__ = [
    "DICTIONARY_ROLE",
    "MAX_DICTIONARY_ENTRIES",
    "first_data_dictionary_id",
    "load_business_names",
    "resolve_data_dictionary_id",
]
