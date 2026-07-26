"""Pure execution-path selection for StrategySampleDesign V2."""

from __future__ import annotations

from copy import deepcopy

import pytest

from marvis.agent.turn_handlers import (
    _native_sample_design_v2_context_relation,
    _strategy_sample_design_v2_template_id,
)


def _eq(column: str, value: object) -> dict:
    return {
        "op": "eq",
        "left": {"column": column},
        "right": {"literal": value},
    }


def _inputs() -> dict:
    return {
        "relationship": "nested_same_cohort",
        "approval_population": {"inclusion": None, "exclusion": None},
        "risk_population": {"inclusion": None, "exclusion": None},
        "field_bindings": {
            "entity_field": "customer_id",
            "time_field": "apply_date",
            "group_field": None,
            "month_field": "apply_month",
            "weight_field": "weight",
            "loan_amount_field": "loan_amount",
            "overdue_amount_field": "overdue_amount",
        },
        "partitioning": {
            "method": "predicate_ast",
            "selectors": {
                "development": _eq("sample_role", "dev"),
                "validation": _eq("sample_role", "valid"),
                "oot": _eq("sample_role", "oot"),
            },
        },
    }


def test_v2_selector_uses_legacy_only_for_exact_lossless_shape() -> None:
    assert (
        _strategy_sample_design_v2_template_id(_inputs())
        == "strategy_sample_design_v2"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"relationship": "parallel_time_cohorts"}),
        lambda value: value["approval_population"].update(
            {"inclusion": _eq("channel", "app")}
        ),
        lambda value: value["risk_population"].update(
            {"exclusion": _eq("channel", "blocked")}
        ),
        lambda value: value.update(
            {
                "partitioning": {
                    "method": "time_ranges",
                    "column": "apply_date",
                    "ranges": {
                        "development": {
                            "start": "2026-01-01",
                            "end": "2026-02-28",
                        },
                        "validation": {
                            "start": "2026-03-01",
                            "end": "2026-03-31",
                        },
                        "oot": {
                            "start": "2026-04-01",
                            "end": "2026-04-30",
                        },
                    },
                }
            }
        ),
        lambda value: value["partitioning"]["selectors"].update(
            {"validation": _eq("channel", "valid")}
        ),
        lambda value: value["partitioning"]["selectors"].update(
            {
                "development": {
                    "op": "and",
                    "args": [
                        _eq("sample_role", "dev"),
                        _eq("channel", "app"),
                    ],
                }
            }
        ),
        lambda value: value["field_bindings"].update(
            {"month_field": "sample_role"}
        ),
        lambda value: value["field_bindings"].update(
            {"weight_field": "apply_month"}
        ),
    ],
)
def test_v2_selector_routes_every_nonlegacy_semantic_to_native(mutate) -> None:
    inputs = deepcopy(_inputs())
    mutate(inputs)

    assert (
        _strategy_sample_design_v2_template_id(inputs)
        == "strategy_sample_design_v2_native"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"dataset_id": "dataset-other"}),
        lambda value: value.update({"workspace_revision": 4}),
    ],
)
def test_authenticated_native_from_other_data_or_workspace_is_unrelated(
    mutate,
) -> None:
    expected = {
        "task_id": "task-1",
        "dataset_id": "dataset-current",
        "dataset_content_hash": "1" * 64,
        "workspace_revision": 3,
        "workspace_generation": 2,
        "semantic_mapping_hash": "2" * 64,
        "target_col": "bad",
    }
    authenticated_source = {
        **expected,
        "drop_nan_labels": False,
    }
    mutate(authenticated_source)

    assert (
        _native_sample_design_v2_context_relation(
            authenticated_source,
            expected=expected,
            drop_nan_labels=False,
        )
        == "other"
    )


def test_authenticated_native_same_dataset_with_wrong_hash_is_invalid() -> None:
    expected = {
        "task_id": "task-1",
        "dataset_id": "dataset-current",
        "dataset_content_hash": "1" * 64,
        "workspace_revision": 3,
        "workspace_generation": 2,
        "semantic_mapping_hash": "2" * 64,
        "target_col": "bad",
    }
    authenticated_source = {
        **expected,
        "dataset_content_hash": "3" * 64,
        "drop_nan_labels": False,
    }

    assert (
        _native_sample_design_v2_context_relation(
            authenticated_source,
            expected=expected,
            drop_nan_labels=False,
        )
        == "invalid"
    )
