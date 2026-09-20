import pytest

from marvis.agent.strategy_workflows._input_validation import (
    bounded_number,
    column,
    reject_fields,
    reject_fields_with_metadata,
)
from marvis.agent.strategy_workflows.contracts import StrategyWorkflowValidationError


@pytest.mark.parametrize(
    "validator,expected_fields",
    [(reject_fields, ()), (reject_fields_with_metadata, ("a", "z"))],
)
def test_unknown_workflow_fields_preserve_error_metadata_contract(validator, expected_fields):
    with pytest.raises(StrategyWorkflowValidationError) as caught:
        validator({"z": 1, "a": 2, "known": 3}, {"known"}, workflow="example")
    assert str(caught.value) == "example workflow_inputs 包含不支持的字段：a、z。"
    assert caught.value.code == "invalid_strategy_request"
    assert caught.value.fields == expected_fields


@pytest.mark.parametrize("validator", [reject_fields, reject_fields_with_metadata])
def test_nontext_workflow_fields_fail_before_sorting_mixed_types(validator):
    with pytest.raises(StrategyWorkflowValidationError) as caught:
        validator({1: "invalid", "known": 3}, {"known"}, workflow="example")
    assert str(caught.value) == "example workflow_inputs 字段名必须是文本。"
    assert caught.value.fields == ()
    assert validator({"known": 3}, {"known"}, workflow="example") is None


@pytest.mark.parametrize("value", [True, "1", None, -1, float("nan"), float("inf"), 1.01])
def test_bounded_workflow_numbers_reject_type_and_range_violations(value):
    with pytest.raises(StrategyWorkflowValidationError):
        bounded_number(value, name="rate", maximum=1)


def test_bounded_workflow_numbers_preserve_inclusive_bounds_and_error_text():
    assert bounded_number(0, name="rate", maximum=1) == 0.0
    assert bounded_number(1, name="rate", maximum=1) == 1.0
    assert bounded_number(2, name="amount") == 2.0
    with pytest.raises(StrategyWorkflowValidationError, match="rate 必须是 0 到 1 之间的有限数字。"):
        bounded_number(-1, name="rate", maximum=1)
    with pytest.raises(StrategyWorkflowValidationError, match="amount 必须是大于等于 0 的有限数字。"):
        bounded_number(-1, name="amount")


def test_workflow_column_is_trimmed_but_never_inferred_or_case_folded():
    assert column(" score ", name="feature", whitelist=["score"]) == "score"
    for value in ("Score", "missing"):
        with pytest.raises(StrategyWorkflowValidationError, match="不存在的列"):
            column(value, name="feature", whitelist=["score"])
    with pytest.raises(StrategyWorkflowValidationError, match="非空文本"):
        column(" ", name="feature", whitelist=["score"])
