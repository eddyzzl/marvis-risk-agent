from __future__ import annotations

import pytest

from marvis.data.transform_semantics import (
    ProtectedSemanticDropError,
    TransformSemanticError,
    migrate_transform_semantics,
)
from marvis.data.workspace import DataSemanticMapping


def _mapping() -> DataSemanticMapping:
    return DataSemanticMapping(
        target_col="bad",
        field_roles={
            "customer_id": "id",
            "bad": "target",
            "score": "score",
            "amount": "amount",
        },
        business_names={
            "customer_id": "客户编号",
            "bad": "风险标签",
            "score": "模型分",
            "amount": "申请金额",
        },
    )


def test_rename_columns_migrates_target_roles_business_names_and_selection():
    migrated = migrate_transform_semantics(
        _mapping(),
        selected_field="score",
        operations=[
            {
                "op": "rename_columns",
                "mapping": {"bad": "label", "score": "score_v2"},
            },
            {
                "op": "derive_columns",
                "derivations": [
                    {
                        "name": "ratio",
                        "expression": {
                            "op": "divide",
                            "left": {"column": "amount"},
                            "right": {"literal": 100},
                        },
                    }
                ],
            },
        ],
        source_columns=("customer_id", "bad", "score", "amount"),
        result_columns=("customer_id", "label", "score_v2", "amount", "ratio"),
    )

    assert migrated.selected_field == "score_v2"
    assert migrated.semantic_mapping.target_col == "label"
    assert dict(migrated.semantic_mapping.field_roles) == {
        "amount": "amount",
        "customer_id": "id",
        "label": "target",
        "score_v2": "score",
    }
    assert dict(migrated.semantic_mapping.business_names) == {
        "amount": "申请金额",
        "customer_id": "客户编号",
        "label": "风险标签",
        "score_v2": "模型分",
    }
    assert migrated.renamed_fields == {"bad": "label", "score": "score_v2"}
    assert migrated.dropped_fields == ()
    assert migrated.dropped_protected_fields == ()


def test_drop_ordinary_columns_prunes_semantics_and_selected_field():
    migrated = migrate_transform_semantics(
        _mapping(),
        selected_field="amount",
        operations=[{"op": "drop_columns", "columns": ["amount"]}],
        source_columns=("customer_id", "bad", "score", "amount"),
        result_columns=("customer_id", "bad", "score"),
    )

    assert migrated.selected_field is None
    assert "amount" not in migrated.semantic_mapping.field_roles
    assert "amount" not in migrated.semantic_mapping.business_names
    assert migrated.dropped_fields == ("amount",)


@pytest.mark.parametrize("field", ["customer_id", "bad"])
def test_drop_protected_target_or_key_requires_explicit_confirmation(field):
    remaining = tuple(
        column
        for column in ("customer_id", "bad", "score", "amount")
        if column != field
    )

    with pytest.raises(ProtectedSemanticDropError, match=field):
        migrate_transform_semantics(
            _mapping(),
            selected_field=None,
            operations=[{"op": "drop_columns", "columns": [field]}],
            source_columns=("customer_id", "bad", "score", "amount"),
            result_columns=remaining,
        )
    migrated = migrate_transform_semantics(
        _mapping(),
        selected_field=None,
        operations=[{"op": "drop_columns", "columns": [field]}],
        source_columns=("customer_id", "bad", "score", "amount"),
        result_columns=remaining,
        confirm_protected_drop=True,
    )
    assert field not in migrated.semantic_mapping.field_roles
    assert migrated.dropped_protected_fields == (field,)
    if field == "bad":
        assert migrated.semantic_mapping.target_col is None


def test_chained_renames_follow_operation_order():
    migrated = migrate_transform_semantics(
        _mapping(),
        selected_field="score",
        operations=[
            {"op": "rename_columns", "mapping": {"score": "model_score"}},
            {"op": "rename_columns", "mapping": {"model_score": "score_v2"}},
        ],
        source_columns=("customer_id", "bad", "score", "amount"),
        result_columns=("customer_id", "bad", "score_v2", "amount"),
    )

    assert migrated.selected_field == "score_v2"
    assert migrated.renamed_fields == {"score": "score_v2"}
    assert migrated.semantic_mapping.field_roles["score_v2"] == "score"


def test_semantic_migration_fails_closed_on_schema_or_operation_drift():
    with pytest.raises(TransformSemanticError, match="source schema"):
        migrate_transform_semantics(
            _mapping(),
            selected_field=None,
            operations=[{"op": "drop_columns", "columns": ["amount"]}],
            source_columns=("bad", "score", "amount"),
            result_columns=("bad", "score"),
        )

    with pytest.raises(TransformSemanticError, match="result schema"):
        migrate_transform_semantics(
            _mapping(),
            selected_field=None,
            operations=[{"op": "drop_columns", "columns": ["amount"]}],
            source_columns=("customer_id", "bad", "score", "amount"),
            result_columns=("customer_id", "bad", "score", "unexpected"),
        )

    with pytest.raises(TransformSemanticError, match="unsupported operation"):
        migrate_transform_semantics(
            _mapping(),
            selected_field=None,
            operations=[{"op": "raw_sql", "sql": "DROP TABLE datasets"}],
            source_columns=("customer_id", "bad", "score", "amount"),
            result_columns=("customer_id", "bad", "score", "amount"),
        )
