from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    validate_strategy_request,
)


def _validate(inputs: dict, *, columns=None):
    return validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": inputs,
        },
        allowed_columns=columns or ["income", "score", "loan_amount", "overdue_amount"],
        target_col="bad",
    )


def test_univariate_request_validates_exact_user_inputs_and_echoes_scope():
    result = _validate(
        {
            "features": ["income", "score"],
            "methods": ["tree", "equal_frequency"],
            "bin_count": 5,
            "min_bin_pct": 0.05,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "sentinel_values": [-9999, "UNKNOWN"],
        }
    )

    assert result.draft is not None
    assert result.draft.workflow == "univariate_candidate_analysis"
    assert result.draft.to_dict()["workflow_inputs"] == {
        "features": ["income", "score"],
        "methods": ["tree", "equal_frequency"],
        "bin_count": 5,
        "min_bin_pct": 0.05,
        "sentinel_values": [-9999, "UNKNOWN"],
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
    }
    assert "单变量候选分析" in str(result.confirmation)
    assert "income、score" in str(result.confirmation)
    assert "只生成 development/unvalidated" in str(result.confirmation)
    assert "univariate_candidate_analysis" in STANDARD_STRATEGY_WORKFLOWS


def test_univariate_request_defaults_are_platform_bounded_and_do_not_guess_fields():
    result = _validate({})

    assert result.draft is not None
    inputs = result.draft.workflow_inputs
    assert inputs["features"] == ()
    assert inputs["methods"] == ()
    assert inputs["bin_count"] == 10
    assert inputs["min_bin_pct"] == 0.02
    assert inputs["sentinel_values"] == ()
    assert "全部候选字段" in str(result.confirmation)
    assert "类别字段使用等值箱" in str(result.confirmation)


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"features": ["ghost"]}, "ghost"),
        ({"features": ["income", "income"]}, "重复"),
        ({"methods": ["quantile_magic"]}, "quantile_magic"),
        ({"methods": []}, "1 到 4"),
        ({"methods": ["tree", "tree"]}, "重复"),
        ({"bin_count": 1}, "bin_count"),
        ({"min_bin_pct": 0.9}, "min_bin_pct"),
        (
            {
                "loan_amount_col": "loan_amount",
                "overdue_amount_col": "loan_amount",
            },
            "不同字段",
        ),
        ({"sentinel_values": [float("nan")]}, "有限数字"),
        ({"sentinel_values": [True]}, "文本或有限数字"),
        ({"metrics": {"iv": 0.8}}, "不支持的字段"),
    ],
)
def test_univariate_request_fails_closed_on_unsafe_or_result_fields(
    inputs: dict,
    message: str,
):
    result = _validate(inputs)

    assert result.draft is None
    assert message in str(result.clarification)


def test_univariate_request_never_accepts_the_observed_target_as_a_feature():
    result = _validate({"features": ["bad"]}, columns=["income", "bad"])

    assert result.draft is None
    assert "目标列" in str(result.clarification)
