"""Natural-language compiler contract for explicit 2D Cross matrices."""

from __future__ import annotations

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)


class _FakeLLM:
    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _payload(**overrides: object) -> dict:
    inputs: dict[str, object] = {
        "x_feature": "age",
        "x_method": "equal_frequency",
        "y_feature": "score",
        "y_method": "equal_width",
        "bin_count": 5,
        "min_bin_pct": 0.02,
        "sentinel_values": [],
    }
    inputs.update(overrides)
    return {
        "request_kind": "standard_workflow",
        "workflow": "cross_matrix_analysis",
        "workflow_inputs": inputs,
    }


def test_cross_matrix_validates_only_human_controls() -> None:
    result = validate_strategy_request(
        _payload(),
        allowed_columns=("age", "score"),
        target_col="bad",
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["features"] == ["age", "score"]
    assert inputs["methods"] == ["equal_frequency", "equal_width"]
    assert inputs["x_feature"] == "age"
    assert inputs["y_feature"] == "score"
    assert "cross_matrix_analysis" in STANDARD_STRATEGY_WORKFLOWS
    assert "二维 Cross Matrix" in result.confirmation
    assert "不会选择格子" in result.confirmation
    assert "不会" in result.confirmation and "入池" in result.confirmation

    replay = validate_strategy_request(
        result.draft.to_dict(),
        allowed_columns=("age", "score"),
        target_col="bad",
    )
    assert replay.draft is not None
    assert replay.draft.to_dict() == result.draft.to_dict()


@pytest.mark.parametrize(
    "forged",
    [
        {"features": ["score", "age"]},
        {"features": ["age"]},
        {"methods": ["tree"]},
    ],
)
def test_cross_matrix_rejects_forged_derived_controls(forged: dict) -> None:
    result = validate_strategy_request(
        _payload(**forged),
        allowed_columns=("age", "score"),
        target_col="bad",
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


@pytest.mark.parametrize(
    "forbidden",
    [
        "dataset_id",
        "expected_content_hash",
        "workspace_revision",
        "analysis_generation",
        "semantic_mapping_hash",
        "target_col",
        "source_artifact_id",
        "expected_candidate_id",
        "cell_metrics",
        "condition",
        "action",
        "max_cells",
    ],
)
def test_cross_matrix_rejects_platform_owned_fields(forbidden: str) -> None:
    result = validate_strategy_request(
        _payload(**{forbidden: "forged"}),
        allowed_columns=("age", "score"),
        target_col="bad",
    )

    assert result.draft is None
    assert forbidden in result.clarification


@pytest.mark.parametrize(
    "overrides",
    [
        {"x_feature": "age", "y_feature": "age"},
        {"x_feature": "unknown"},
        {"x_method": "magic"},
        {"y_method": "magic"},
        {"bin_count": 2},
        {"bin_count": 21},
        {"min_bin_pct": 0.8},
    ],
)
def test_cross_matrix_rejects_invalid_axes_or_analysis_controls(
    overrides: dict[str, object],
) -> None:
    result = validate_strategy_request(
        _payload(**overrides),
        allowed_columns=("age", "score"),
        target_col="bad",
    )

    assert result.draft is None
    assert result.clarification_code == "invalid_strategy_request"


def test_cross_matrix_compiles_exact_axes_and_methods() -> None:
    llm = _FakeLLM(_payload())
    utterance = "构建 age 等频 5 箱 × score 等距 5 箱的二维交叉矩阵"

    result = compile_strategy_request(
        utterance,
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["x_feature"] == "age"
    assert inputs["x_method"] == "equal_frequency"
    assert inputs["y_feature"] == "score"
    assert inputs["y_method"] == "equal_width"
    assert llm.calls[0]["prompt_version"] == 43
    assert "cross_matrix_analysis" in llm.calls[0]["system_prompt"]


def test_cross_matrix_compiles_mixed_manual_and_automatic_axes() -> None:
    result = compile_strategy_request(
        "构建 age manual 切点 [30, 50] × score 等频 5 箱的二维 Cross Matrix",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="manual",
                y_method="equal_frequency",
                manual_breakpoints={"age": [30, 50]},
            )
        ),
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["methods"] == ["manual", "equal_frequency"]
    assert inputs["manual_breakpoints"] == {"age": [30.0, 50.0]}
    assert "手工轴切点：age=[30、50]" in result.confirmation


@pytest.mark.parametrize(
    "manual_breakpoints",
    [
        {"age": [30, 60]},
        {"age": [50, 30]},
        {"score": [30, 50]},
        {"age": [30, 50], "score": [300]},
    ],
)
def test_cross_matrix_rejects_ungrounded_or_misbound_manual_breakpoints(
    manual_breakpoints: dict[str, list[int]],
) -> None:
    result = compile_strategy_request(
        "构建 age manual 切点 [30, 50] × score 等频 5 箱的二维 Cross Matrix",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="manual",
                y_method="equal_frequency",
                manual_breakpoints=manual_breakpoints,
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code in {
        "invalid_strategy_request",
        "cross_matrix_analysis_controls_not_grounded",
    }


@pytest.mark.parametrize(
    "reply",
    [
        {"operation": "analyze", "strategy_type": "approval"},
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_analysis",
            "workflow_inputs": {
                "features": ["age", "score"],
                "methods": ["equal_frequency"],
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
            },
        },
    ],
)
def test_explicit_cross_matrix_cannot_route_to_another_workflow(reply: dict) -> None:
    result = compile_strategy_request(
        "构建 age 与 score 的二维 Cross Matrix，两个轴都用等频 5 箱",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(reply),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_workflow_required"
    assert result.clarification_fields == ("workflow",)


@pytest.mark.parametrize(
    ("utterance", "code"),
    [
        (
            "查看 age 与 score 的二维交叉矩阵现状，两个轴都用等频 5 箱",
            "cross_matrix_build_intent_required",
        ),
        (
            "构建 age 的二维交叉矩阵，两个轴都用等频 5 箱",
            "cross_matrix_axes_not_grounded",
        ),
        (
            "构建 age 与 score 的二维交叉矩阵",
            "cross_matrix_methods_not_grounded",
        ),
    ],
)
def test_cross_matrix_requires_positive_explicit_axes_and_methods(
    utterance: str,
    code: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(_payload(x_method="equal_frequency", y_method="equal_frequency")),
    )

    assert result.draft is None
    assert result.clarification_code == code


@pytest.mark.parametrize(
    "utterance",
    [
        "如果以后要构建 age 和 score 的二维交叉矩阵，两个轴都用决策树，应该怎么做？",
        "请说明如何构建 age 和 score 的二维交叉矩阵，两个轴都用决策树",
        "昨天构建 age 和 score 的二维交叉矩阵时，两个轴都用了决策树",
        "构建 age 和 score 的二维交叉矩阵，两个轴都用决策树。算了，先不做了",
    ],
)
def test_cross_matrix_rejects_non_immediate_or_cancelled_commands(
    utterance: str,
) -> None:
    result = compile_strategy_request(
        utterance,
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="tree",
                y_method="tree",
                bin_count=10,
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_positive_command_required"


def test_cross_matrix_rejects_negated_or_replaced_method_controls() -> None:
    result = compile_strategy_request(
        "构建 age 和 score 的二维交叉矩阵，不要用决策树，改用等频 5 箱",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="tree",
                y_method="tree",
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_controls_rewritten"


def test_cross_matrix_rejects_multiple_axis_pairs_in_one_request() -> None:
    result = compile_strategy_request(
        "构建 age/score 和 income/tenure 两个二维交叉矩阵，都用决策树 5 箱",
        allowed_columns=("age", "score", "income", "tenure"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="tree",
                y_method="tree",
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_axes_not_unique"


def test_cross_matrix_binds_axis_orientation_to_mention_order() -> None:
    result = compile_strategy_request(
        "构建 age 等频 5 箱 × score 等距 5 箱的二维交叉矩阵",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_feature="score",
                x_method="equal_width",
                y_feature="age",
                y_method="equal_frequency",
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_axis_order_not_grounded"


def test_cross_matrix_rejects_ungrounded_nondefault_analysis_controls() -> None:
    result = compile_strategy_request(
        "构建 age 与 score 的二维交叉矩阵，两个轴都用等频",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="equal_frequency",
                y_method="equal_frequency",
                bin_count=7,
                min_bin_pct=0.03,
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_analysis_controls_not_grounded"


def test_cross_matrix_rejects_sentinel_from_historical_clause() -> None:
    result = compile_strategy_request(
        "历史材料的哨兵是 UNKNOWN。构建 age 与 score 的二维交叉矩阵，两个轴都用等频 5 箱",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(
            _payload(
                x_method="equal_frequency",
                y_method="equal_frequency",
                sentinel_values=["UNKNOWN"],
            )
        ),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_analysis_controls_not_grounded"
    assert result.clarification_fields == ("sentinel_values",)


@pytest.mark.parametrize("sentinel_values", [[-9999], []])
def test_cross_matrix_rejects_omitted_explicit_sentinel(
    sentinel_values: list[object],
) -> None:
    result = compile_strategy_request(
        "构建 age 等频 5 箱 × score 等距 5 箱的二维交叉矩阵，"
        "哨兵值 -9999 和 UNKNOWN",
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(_payload(sentinel_values=sentinel_values)),
    )

    assert result.draft is None
    assert result.clarification_code == "cross_matrix_analysis_controls_not_grounded"
    assert result.clarification_fields == ("sentinel_values",)


def test_cross_matrix_accepts_exact_typed_sentinel_set() -> None:
    result = compile_strategy_request(
        "构建 age 等频 5 箱 × score 等距 5 箱的二维交叉矩阵，"
        '哨兵值 -9999 和 "UNKNOWN"',
        allowed_columns=("age", "score"),
        target_col="bad",
        llm=_FakeLLM(_payload(sentinel_values=["UNKNOWN", -9999])),
    )

    assert result.draft is not None
    assert set(result.draft.to_dict()["workflow_inputs"]["sentinel_values"]) == {
        -9999,
        "UNKNOWN",
    }
