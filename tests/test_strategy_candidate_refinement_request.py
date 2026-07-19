from __future__ import annotations

import json

import pytest

from marvis.agent.strategy_request_compiler import (
    STANDARD_STRATEGY_WORKFLOWS,
    compile_strategy_request,
    validate_strategy_request,
)

_SOURCE_CANDIDATE_ID = "candidate-" + "a" * 32


def _validate(inputs: dict, *, columns=None, bind_source: bool = True):
    normalized_inputs = dict(inputs)
    selection = normalized_inputs.get("selection")
    if bind_source and (
        normalized_inputs.get("merge_groups")
        or (isinstance(selection, dict) and "source_bin_ids" in selection)
    ):
        normalized_inputs.setdefault("source_candidate_id", _SOURCE_CANDIDATE_ID)
    return validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": normalized_inputs,
        },
        allowed_columns=columns or ["income", "score", "loan_amount", "overdue_amount"],
        target_col="bad",
    )


def test_refinement_request_normalizes_analysis_and_threshold_selection():
    result = _validate(
        {
            "feature": "score",
            "method": "tree",
            "features": ["income", "score"],
            "methods": ["equal_frequency", "tree"],
            "bin_count": 5,
            "min_bin_pct": 0.05,
            "loan_amount_col": "loan_amount",
            "overdue_amount_col": "overdue_amount",
            "sentinel_values": [-9999],
            "selection": {"risk_threshold": {"operator": ">=", "value": 0.2}},
            "selection_reason": "保留高风险区间",
        }
    )

    assert result.draft is not None
    assert result.draft.workflow == "univariate_candidate_refinement"
    assert result.draft.to_dict()["workflow_inputs"] == {
        "features": ["income", "score"],
        "methods": ["equal_frequency", "tree"],
        "bin_count": 5,
        "min_bin_pct": 0.05,
        "sentinel_values": [-9999],
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
        "feature": "score",
        "method": "tree",
        "merge_groups": [],
        "selection": {"risk_threshold": {"operator": ">=", "value": 0.2}},
        "selection_reason": "保留高风险区间",
    }
    assert "候选选择与合并" in str(result.confirmation)
    assert "观测坏率 >= 20.00%" in str(result.confirmation)
    assert "development/unvalidated" in str(result.confirmation)
    assert "univariate_candidate_refinement" in STANDARD_STRATEGY_WORKFLOWS


def test_refinement_request_defaults_analysis_to_the_selected_feature_and_method():
    result = _validate(
        {
            "feature": "score",
            "method": "equal_width",
            "selection": {"risk_threshold": {"operator": ">=", "value": 0.2}},
        }
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs["features"] == ["score"]
    assert inputs["methods"] == ["equal_width"]
    assert inputs["merge_groups"] == []
    assert inputs["selection"] == {"risk_threshold": {"operator": ">=", "value": 0.2}}
    assert "观测坏率 >= 20.00%" in str(result.confirmation)


def test_categorical_refinement_uses_type_aware_analysis_defaults():
    result = _validate(
        {
            "feature": "income",
            "method": "categorical",
            "selection": {"source_bin_ids": ["category:0"]},
        }
    )

    assert result.draft is not None
    inputs = result.draft.to_dict()["workflow_inputs"]
    assert inputs == {
        "feature": "income",
        "method": "categorical",
        "merge_groups": [],
        "selection": {"source_bin_ids": ["category:0"]},
        "source_candidate_id": _SOURCE_CANDIDATE_ID,
    }


def test_source_bin_controls_require_the_candidate_evidence_the_user_inspected():
    result = _validate(
        {
            "feature": "score",
            "method": "equal_width",
            "selection": {"source_bin_ids": ["regular:0"]},
        },
        bind_source=False,
    )

    assert result.draft is None
    assert "source_candidate_id" in str(result.clarification)


def test_existing_candidate_refinement_rejects_silently_ignored_reanalysis_inputs():
    result = _validate(
        {
            "feature": "score",
            "method": "equal_width",
            "bin_count": 7,
            "selection": {"source_bin_ids": ["regular:0"]},
        }
    )

    assert result.draft is None
    assert "不能重设分析参数" in str(result.clarification)
    assert "bin_count" in str(result.clarification)


class _OneReplyLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False)


def test_compiler_rejects_an_llm_invented_threshold_for_pick_the_best():
    llm = _OneReplyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": {
                "feature": "score",
                "method": "equal_width",
                "selection": {"risk_threshold": {"operator": ">=", "value": 0.9}},
            },
        }
    )

    result = compile_strategy_request(
        "请从 score 分箱里选最好的候选规则",
        allowed_columns=["score"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_refinement_controls_not_grounded"
    assert "不会根据“最好”" in str(result.clarification)
    assert len(llm.calls) == 1


def test_compiler_accepts_only_literal_candidate_bins_and_merge_controls():
    llm = _OneReplyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": {
                "feature": "score",
                "method": "equal_width",
                "source_candidate_id": _SOURCE_CANDIDATE_ID,
                "merge_groups": [["regular:1", "regular:2"]],
                "selection": {"source_bin_ids": ["regular:1", "regular:2"]},
            },
        }
    )

    result = compile_strategy_request(
        f"在 {_SOURCE_CANDIDATE_ID} 中合并 regular:1 和 regular:2，并选择它们",
        allowed_columns=["score"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is not None
    assert result.draft.workflow_inputs["source_candidate_id"] == (_SOURCE_CANDIDATE_ID)


@pytest.mark.parametrize(
    ("utterance", "workflow_inputs"),
    [
        (
            f"在 {_SOURCE_CANDIDATE_ID} 中选择 regular:10 的候选箱",
            {
                "feature": "score",
                "method": "equal_width",
                "source_candidate_id": _SOURCE_CANDIDATE_ID,
                "selection": {"source_bin_ids": ["regular:1"]},
            },
        ),
        (
            "选择 score 中坏率大于等于 50% 的候选箱",
            {
                "feature": "score",
                "method": "equal_width",
                "selection": {"risk_threshold": {"operator": ">", "value": 0.5}},
            },
        ),
        (
            "选择 score 中坏率超过 10%，min_bin_pct 20% 的候选箱",
            {
                "feature": "score",
                "method": "equal_width",
                "min_bin_pct": 0.2,
                "selection": {"risk_threshold": {"operator": ">", "value": 0.2}},
            },
        ),
    ],
)
def test_compiler_rejects_substring_operator_or_unbound_percentage_grounding(
    utterance: str,
    workflow_inputs: dict,
):
    llm = _OneReplyLLM(
        {
            "request_kind": "standard_workflow",
            "workflow": "univariate_candidate_refinement",
            "workflow_inputs": workflow_inputs,
        }
    )

    result = compile_strategy_request(
        utterance,
        allowed_columns=["score"],
        target_col="bad",
        llm=llm,
    )

    assert result.draft is None
    assert result.clarification_code == "strategy_refinement_controls_not_grounded"


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"method": "tree", "selection": {"source_bin_ids": ["regular:0"]}}, "feature"),
        (
            {"feature": "score", "selection": {"source_bin_ids": ["regular:0"]}},
            "method",
        ),
        ({"feature": "score", "method": "tree"}, "selection"),
        (
            {
                "feature": "score",
                "method": "magic",
                "selection": {"source_bin_ids": ["regular:0"]},
            },
            "magic",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "features": ["income"],
                "selection": {"risk_threshold": {"operator": ">=", "value": 0.2}},
            },
            "候选字段",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "methods": ["equal_width"],
                "selection": {"risk_threshold": {"operator": ">=", "value": 0.2}},
            },
            "分箱方法",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "merge_groups": [
                    ["regular:0", "regular:1"],
                    ["regular:1", "regular:2"],
                ],
                "selection": {"source_bin_ids": ["regular:0"]},
            },
            "重复",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "selection": {
                    "source_bin_ids": ["regular:0"],
                    "risk_threshold": {"operator": ">=", "value": 0.2},
                },
            },
            "二选一",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "selection": {"risk_threshold": {"operator": "=", "value": 0.2}},
            },
            "operator",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "selection": {"risk_threshold": {"operator": ">=", "value": 1.2}},
            },
            "value",
        ),
        (
            {
                "feature": "score",
                "method": "tree",
                "selection": {"source_bin_ids": ["regular:0"]},
                "metrics": {"iv": 0.8},
            },
            "不支持的字段",
        ),
        (
            {
                "feature": "bad",
                "method": "tree",
                "selection": {"source_bin_ids": ["regular:0"]},
            },
            "目标列",
        ),
    ],
)
def test_refinement_request_fails_closed(inputs: dict, message: str):
    result = _validate(inputs, columns=["income", "score", "bad"])

    assert result.draft is None
    assert message in str(result.clarification)
