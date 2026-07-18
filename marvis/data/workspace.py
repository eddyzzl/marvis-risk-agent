"""Canonical data-workspace state shared by the API and repository layers.

The workspace is intentionally small: it stores navigation and semantic choices,
not computed analysis results.  ``analysis_generation`` is server-owned and lets
later slices key cached analysis to the exact active dataset generation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import re
from types import MappingProxyType
from typing import Any


DATA_WORKSPACE_SCHEMA_VERSION = "data-workspace.v1"
DATA_WORKSPACE_PAGES = (
    "overview",
    "fields",
    "semantics",
    "history",
    "statistics",
)
DATA_FIELD_ROLES = (
    "phone",
    "idcard",
    "id",
    "date",
    "target",
    "score",
    "amount",
    "name",
    "feature",
    "numeric",
    "categorical",
    "loan_amount",
    "overdue_amount",
    "month",
    "rule_node",
    "segment",
    "weight",
    "ignore",
)

_PAGE_SET = frozenset(DATA_WORKSPACE_PAGES)
_FIELD_ROLE_SET = frozenset(DATA_FIELD_ROLES)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_MAPPING_KEYS = frozenset({"target_col", "field_roles", "business_names"})
_DRAFT_KEYS = frozenset({
    "active_dataset_id",
    "active_dataset_content_hash",
    "page",
    "selected_field",
    "semantic_mapping",
})
_SNAPSHOT_KEYS = frozenset({
    "schema_version",
    "task_id",
    "revision",
    "active_dataset_id",
    "active_dataset_content_hash",
    "analysis_generation",
    "page",
    "selected_field",
    "semantic_mapping",
    "updated_at",
})


@dataclass(frozen=True)
class DataSemanticMapping:
    target_col: str | None = None
    field_roles: Mapping[str, str] = field(default_factory=dict)
    business_names: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_col = _optional_text(self.target_col, field_name="target_col")
        field_roles = _text_mapping(self.field_roles, field_name="field_roles")
        business_names = _text_mapping(
            self.business_names,
            field_name="business_names",
        )
        invalid_roles = sorted(set(field_roles.values()) - _FIELD_ROLE_SET)
        if invalid_roles:
            raise ValueError(
                "field_roles contains unsupported role(s): " + ", ".join(invalid_roles)
            )
        target_fields = [
            column for column, role in field_roles.items() if role == "target"
        ]
        if len(target_fields) > 1:
            raise ValueError("field_roles may mark at most one target column")
        if target_fields and target_fields[0] != target_col:
            raise ValueError("field_roles target must match target_col")
        object.__setattr__(self, "target_col", target_col)
        object.__setattr__(self, "field_roles", MappingProxyType(field_roles))
        object.__setattr__(
            self,
            "business_names",
            MappingProxyType(business_names),
        )


@dataclass(frozen=True)
class DataWorkspaceDraft:
    active_dataset_id: str | None = None
    active_dataset_content_hash: str | None = None
    page: str = "overview"
    selected_field: str | None = None
    semantic_mapping: DataSemanticMapping = field(default_factory=DataSemanticMapping)

    def __post_init__(self) -> None:
        dataset_id = _optional_text(
            self.active_dataset_id,
            field_name="active_dataset_id",
        )
        dataset_hash = _optional_hash(
            self.active_dataset_content_hash,
            field_name="active_dataset_content_hash",
        )
        if (dataset_id is None) != (dataset_hash is None):
            raise ValueError(
                "active_dataset_id and active_dataset_content_hash must both be null or non-null"
            )
        page = _page(self.page)
        selected_field = _optional_text(
            self.selected_field,
            field_name="selected_field",
        )
        if not isinstance(self.semantic_mapping, DataSemanticMapping):
            raise TypeError("semantic_mapping must be a DataSemanticMapping")
        object.__setattr__(self, "active_dataset_id", dataset_id)
        object.__setattr__(self, "active_dataset_content_hash", dataset_hash)
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "selected_field", selected_field)


@dataclass(frozen=True)
class DataWorkspaceSnapshot:
    task_id: str
    updated_at: str
    schema_version: str = DATA_WORKSPACE_SCHEMA_VERSION
    revision: int = 0
    active_dataset_id: str | None = None
    active_dataset_content_hash: str | None = None
    analysis_generation: int = 0
    page: str = "overview"
    selected_field: str | None = None
    semantic_mapping: DataSemanticMapping = field(default_factory=DataSemanticMapping)

    def __post_init__(self) -> None:
        if self.schema_version != DATA_WORKSPACE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {DATA_WORKSPACE_SCHEMA_VERSION}"
            )
        task_id = _required_text(self.task_id, field_name="task_id")
        revision = _non_negative_int(self.revision, field_name="revision")
        generation = _non_negative_int(
            self.analysis_generation,
            field_name="analysis_generation",
        )
        updated_at = _timestamp(self.updated_at)
        draft = DataWorkspaceDraft(
            active_dataset_id=self.active_dataset_id,
            active_dataset_content_hash=self.active_dataset_content_hash,
            page=self.page,
            selected_field=self.selected_field,
            semantic_mapping=self.semantic_mapping,
        )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "analysis_generation", generation)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "active_dataset_id", draft.active_dataset_id)
        object.__setattr__(
            self,
            "active_dataset_content_hash",
            draft.active_dataset_content_hash,
        )
        object.__setattr__(self, "page", draft.page)
        object.__setattr__(self, "selected_field", draft.selected_field)


def data_semantic_mapping_to_dict(mapping: DataSemanticMapping) -> dict[str, Any]:
    if not isinstance(mapping, DataSemanticMapping):
        raise TypeError("mapping must be a DataSemanticMapping")
    return {
        "target_col": mapping.target_col,
        "field_roles": dict(mapping.field_roles),
        "business_names": dict(mapping.business_names),
    }


def data_semantic_mapping_from_dict(payload: object) -> DataSemanticMapping:
    value = _exact_object(
        payload,
        expected_keys=_SEMANTIC_MAPPING_KEYS,
        field_name="semantic_mapping",
    )
    return DataSemanticMapping(
        target_col=value["target_col"],
        field_roles=value["field_roles"],
        business_names=value["business_names"],
    )


def data_workspace_draft_to_dict(draft: DataWorkspaceDraft) -> dict[str, Any]:
    if not isinstance(draft, DataWorkspaceDraft):
        raise TypeError("draft must be a DataWorkspaceDraft")
    return {
        "active_dataset_id": draft.active_dataset_id,
        "active_dataset_content_hash": draft.active_dataset_content_hash,
        "page": draft.page,
        "selected_field": draft.selected_field,
        "semantic_mapping": data_semantic_mapping_to_dict(draft.semantic_mapping),
    }


def data_workspace_draft_from_dict(payload: object) -> DataWorkspaceDraft:
    value = _exact_object(
        payload,
        expected_keys=_DRAFT_KEYS,
        field_name="data_workspace_draft",
    )
    return DataWorkspaceDraft(
        active_dataset_id=value["active_dataset_id"],
        active_dataset_content_hash=value["active_dataset_content_hash"],
        page=value["page"],
        selected_field=value["selected_field"],
        semantic_mapping=data_semantic_mapping_from_dict(value["semantic_mapping"]),
    )


def data_workspace_snapshot_to_dict(snapshot: DataWorkspaceSnapshot) -> dict[str, Any]:
    if not isinstance(snapshot, DataWorkspaceSnapshot):
        raise TypeError("snapshot must be a DataWorkspaceSnapshot")
    return {
        "schema_version": snapshot.schema_version,
        "task_id": snapshot.task_id,
        "revision": snapshot.revision,
        "active_dataset_id": snapshot.active_dataset_id,
        "active_dataset_content_hash": snapshot.active_dataset_content_hash,
        "analysis_generation": snapshot.analysis_generation,
        "page": snapshot.page,
        "selected_field": snapshot.selected_field,
        "semantic_mapping": data_semantic_mapping_to_dict(snapshot.semantic_mapping),
        "updated_at": snapshot.updated_at,
    }


def data_workspace_snapshot_from_dict(payload: object) -> DataWorkspaceSnapshot:
    value = _exact_object(
        payload,
        expected_keys=_SNAPSHOT_KEYS,
        field_name="data_workspace_snapshot",
    )
    return DataWorkspaceSnapshot(
        schema_version=value["schema_version"],
        task_id=value["task_id"],
        revision=value["revision"],
        active_dataset_id=value["active_dataset_id"],
        active_dataset_content_hash=value["active_dataset_content_hash"],
        analysis_generation=value["analysis_generation"],
        page=value["page"],
        selected_field=value["selected_field"],
        semantic_mapping=data_semantic_mapping_from_dict(value["semantic_mapping"]),
        updated_at=value["updated_at"],
    )


def _exact_object(
    value: object,
    *,
    expected_keys: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        raise ValueError(f"{field_name} has invalid keys ({'; '.join(detail)})")
    return value


def _page(value: object) -> str:
    if not isinstance(value, str) or value not in _PAGE_SET:
        raise ValueError("page must be one of: " + ", ".join(DATA_WORKSPACE_PAGES))
    return value


def _text_mapping(value: object, *, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = _required_text(key, field_name=f"{field_name} key")
        normalized_value = _required_text(
            item,
            field_name=f"{field_name}[{normalized_key}]",
        )
        normalized[normalized_key] = normalized_value
    return dict(sorted(normalized.items()))


def _required_text(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or value == ""
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name=field_name)


def _optional_hash(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _timestamp(value: object) -> str:
    raw = _required_text(value, field_name="updated_at")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("updated_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("updated_at must include a timezone")
    return raw


__all__ = [
    "DATA_FIELD_ROLES",
    "DATA_WORKSPACE_PAGES",
    "DATA_WORKSPACE_SCHEMA_VERSION",
    "DataSemanticMapping",
    "DataWorkspaceDraft",
    "DataWorkspaceSnapshot",
    "data_semantic_mapping_from_dict",
    "data_semantic_mapping_to_dict",
    "data_workspace_draft_from_dict",
    "data_workspace_draft_to_dict",
    "data_workspace_snapshot_from_dict",
    "data_workspace_snapshot_to_dict",
]
