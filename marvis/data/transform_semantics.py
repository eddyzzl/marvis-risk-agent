"""Deterministic semantic migration for immutable derived datasets.

Data transforms create a new physical dataset instead of mutating the source.
This module carries forward only semantics that remain structurally true.  It
does not infer new business meaning: derived columns stay unclassified until a
user or a later governed semantic-mapping action assigns them a role.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from marvis.data.workspace import DataSemanticMapping


_SUPPORTED_OPERATIONS = frozenset(
    {
        "rename_columns",
        "drop_columns",
        "cast_columns",
        "fill_missing",
        "filter_rows",
        "derive_columns",
        "deduplicate",
    }
)
_PROTECTED_DROP_ROLES = frozenset({"target", "id", "phone", "idcard"})


class TransformSemanticError(ValueError):
    """The operation stream and semantic/schema evidence disagree."""


class ProtectedSemanticDropError(TransformSemanticError):
    """A target or key-like semantic field needs explicit confirmation."""


@dataclass(frozen=True)
class TransformSemanticMigration:
    semantic_mapping: DataSemanticMapping
    selected_field: str | None
    renamed_fields: Mapping[str, str]
    dropped_fields: tuple[str, ...]
    dropped_protected_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "renamed_fields",
            MappingProxyType(dict(self.renamed_fields)),
        )


def effective_transform_semantic_mapping(
    dataset,
    workspace_mapping: DataSemanticMapping,
    *,
    source_columns: Sequence[str],
) -> DataSemanticMapping:
    """Merge trusted registry target/key evidence into workspace semantics.

    Dataset registration can identify a target or identifier before a user has
    saved an explicit workspace mapping.  Both the Agent clarification layer
    and the Tool execution layer use this same function so protected fields do
    not become visible only after execution has already started.
    """

    if not isinstance(workspace_mapping, DataSemanticMapping):
        raise TransformSemanticError("workspace_mapping must be DataSemanticMapping")
    available = set(_canonical_columns(source_columns, field="source schema"))
    roles = dict(workspace_mapping.field_roles)
    for column in getattr(dataset, "columns", ()):
        name = str(getattr(column, "name", ""))
        role = str(getattr(column, "semantic_role", "") or "")
        if (
            name in available
            and name not in roles
            and role in _PROTECTED_DROP_ROLES
        ):
            roles[name] = role
    target = workspace_mapping.target_col
    registered_target = getattr(dataset, "target_col", None)
    if (
        target is None
        and isinstance(registered_target, str)
        and registered_target in available
    ):
        target = registered_target
    if target is not None:
        roles.setdefault(target, "target")
    return DataSemanticMapping(
        target_col=target,
        field_roles=roles,
        business_names=dict(workspace_mapping.business_names),
    )


def migrate_transform_semantics(
    semantic_mapping: DataSemanticMapping,
    *,
    selected_field: str | None,
    operations: Sequence[Mapping[str, object]],
    source_columns: Sequence[str],
    result_columns: Sequence[str],
    confirm_protected_drop: bool = False,
) -> TransformSemanticMigration:
    """Carry confirmed semantics through ordered rename/drop operations.

    The transform kernel remains authoritative for the physical schema.  This
    function independently simulates the two operations that can change field
    identity, then requires its predicted schema to equal the kernel result.
    That cross-check prevents a future operation implementation from silently
    invalidating semantics without updating this boundary.
    """

    if not isinstance(semantic_mapping, DataSemanticMapping):
        raise TransformSemanticError("semantic_mapping must be DataSemanticMapping")
    source = _canonical_columns(source_columns, field="source schema")
    result = _canonical_columns(result_columns, field="result schema")
    if selected_field is not None and selected_field not in source:
        raise TransformSemanticError(
            f"selected field is not present in source schema: {selected_field}"
        )
    semantic_fields = (
        set(semantic_mapping.field_roles)
        | set(semantic_mapping.business_names)
        | ({semantic_mapping.target_col} if semantic_mapping.target_col else set())
    )
    missing_semantics = sorted(semantic_fields - set(source))
    if missing_semantics:
        raise TransformSemanticError(
            "source schema is missing semantic field(s): "
            + ", ".join(missing_semantics)
        )
    if not isinstance(operations, Sequence) or isinstance(
        operations, (str, bytes, bytearray)
    ):
        raise TransformSemanticError("operations must be an ordered array")

    current_columns = list(source)
    roles = dict(semantic_mapping.field_roles)
    names = dict(semantic_mapping.business_names)
    target = semantic_mapping.target_col
    selected = selected_field
    original_to_current = {column: column for column in source}
    dropped: list[str] = []
    protected_dropped: list[str] = []

    for index, raw_operation in enumerate(operations):
        if not isinstance(raw_operation, Mapping):
            raise TransformSemanticError(f"operation {index} must be an object")
        op = raw_operation.get("op")
        if not isinstance(op, str) or op not in _SUPPORTED_OPERATIONS:
            raise TransformSemanticError(
                f"unsupported operation at index {index}: {op!r}"
            )
        if op == "rename_columns":
            mapping = _rename_mapping(raw_operation, index=index)
            unknown = sorted(set(mapping) - set(current_columns))
            if unknown:
                raise TransformSemanticError(
                    "rename operation references unknown field(s): "
                    + ", ".join(unknown)
                )
            current_columns = [mapping.get(column, column) for column in current_columns]
            if len(current_columns) != len(set(current_columns)):
                raise TransformSemanticError("rename operation creates duplicate fields")
            roles = _rename_keys(roles, mapping)
            names = _rename_keys(names, mapping)
            target = mapping.get(target, target) if target is not None else None
            selected = mapping.get(selected, selected) if selected is not None else None
            for original, current in list(original_to_current.items()):
                original_to_current[original] = mapping.get(current, current)
            continue
        if op == "drop_columns":
            columns = _drop_columns(raw_operation, index=index)
            unknown = sorted(set(columns) - set(current_columns))
            if unknown:
                raise TransformSemanticError(
                    "drop operation references unknown field(s): "
                    + ", ".join(unknown)
                )
            protected = [
                column
                for column in columns
                if column == target or roles.get(column) in _PROTECTED_DROP_ROLES
            ]
            if protected and not confirm_protected_drop:
                raise ProtectedSemanticDropError(
                    "dropping protected target/key field(s) requires explicit confirmation: "
                    + ", ".join(protected)
                )
            drop_set = set(columns)
            current_columns = [
                column for column in current_columns if column not in drop_set
            ]
            for column in columns:
                roles.pop(column, None)
                names.pop(column, None)
                if column == target:
                    target = None
                if column == selected:
                    selected = None
                if column not in dropped:
                    dropped.append(column)
                if column in protected and column not in protected_dropped:
                    protected_dropped.append(column)
            continue

        # Cast/fill/filter/deduplicate preserve names; derive appends names.  The
        # kernel result is checked below, so this boundary need not reimplement
        # expression/type validation or invent semantics for derived fields.
        if op == "derive_columns":
            derivations = raw_operation.get("derivations")
            if not isinstance(derivations, Sequence) or isinstance(
                derivations, (str, bytes, bytearray)
            ):
                raise TransformSemanticError(
                    f"derive operation {index} must contain a derivations array"
                )
            for derivation in derivations:
                if not isinstance(derivation, Mapping):
                    raise TransformSemanticError("derivation must be an object")
                name = derivation.get("name")
                if not isinstance(name, str) or not name or name != name.strip():
                    raise TransformSemanticError("derived field name must be canonical text")
                if name in current_columns:
                    raise TransformSemanticError(
                        f"derived field already exists: {name}"
                    )
                current_columns.append(name)

    if tuple(current_columns) != result:
        raise TransformSemanticError(
            "result schema does not match the ordered transform operations"
        )
    result_set = set(result)
    dangling = sorted((set(roles) | set(names)) - result_set)
    if target is not None and target not in result_set:
        dangling.append(target)
    if selected is not None and selected not in result_set:
        dangling.append(selected)
    if dangling:
        raise TransformSemanticError(
            "result schema is missing migrated semantic field(s): "
            + ", ".join(sorted(set(dangling)))
        )

    renamed = {
        original: current
        for original, current in original_to_current.items()
        if original != current and current in result_set
    }
    return TransformSemanticMigration(
        semantic_mapping=DataSemanticMapping(
            target_col=target,
            field_roles=roles,
            business_names=names,
        ),
        selected_field=selected,
        renamed_fields=renamed,
        dropped_fields=tuple(dropped),
        dropped_protected_fields=tuple(protected_dropped),
    )


def _canonical_columns(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(
        values, (str, bytes, bytearray)
    ):
        raise TransformSemanticError(f"{field} must be an ordered array")
    columns: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise TransformSemanticError(f"{field} contains invalid field name")
        columns.append(value)
    if len(columns) != len(set(columns)):
        raise TransformSemanticError(f"{field} contains duplicate field names")
    return tuple(columns)


def _rename_mapping(
    operation: Mapping[str, object],
    *,
    index: int,
) -> dict[str, str]:
    if set(operation) != {"op", "mapping"}:
        raise TransformSemanticError(
            f"rename operation {index} has unexpected or missing fields"
        )
    raw = operation.get("mapping")
    if not isinstance(raw, Mapping) or not raw:
        raise TransformSemanticError("rename mapping must be a non-empty object")
    mapping: dict[str, str] = {}
    for source, target in raw.items():
        if not isinstance(source, str) or not source or source != source.strip():
            raise TransformSemanticError("rename source must be canonical text")
        if not isinstance(target, str) or not target or target != target.strip():
            raise TransformSemanticError("rename target must be canonical text")
        if source == target:
            raise TransformSemanticError("rename source and target must differ")
        mapping[source] = target
    return mapping


def _drop_columns(
    operation: Mapping[str, object],
    *,
    index: int,
) -> tuple[str, ...]:
    if set(operation) != {"op", "columns"}:
        raise TransformSemanticError(
            f"drop operation {index} has unexpected or missing fields"
        )
    raw = operation.get("columns")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise TransformSemanticError("drop columns must be a non-empty array")
    columns = _canonical_columns(raw, field="drop columns")
    if not columns:
        raise TransformSemanticError("drop columns must be a non-empty array")
    return columns


def _rename_keys(values: dict[str, str], mapping: Mapping[str, str]) -> dict[str, str]:
    renamed = {mapping.get(key, key): value for key, value in values.items()}
    if len(renamed) != len(values):
        raise TransformSemanticError("rename operation collides semantic fields")
    return renamed


__all__ = [
    "ProtectedSemanticDropError",
    "TransformSemanticError",
    "TransformSemanticMigration",
    "effective_transform_semantic_mapping",
    "migrate_transform_semantics",
]
