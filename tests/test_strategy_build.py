import pandas as pd
import pytest

import marvis.packs.strategy as strategy_pack
from marvis.data.errors import ScoreDirectionConflictError
from marvis.packs.strategy import StrategyError, apply_strategy, build_strategy
from marvis.packs.strategy.strategy import infer_strategy_rule_direction


def test_build_strategy_applies_rules_in_order_with_default_decision():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "score >= 720", "decision": "approve"},
        ],
        score_col="score",
        default_decision="approve",
        description="baseline cutoff",
    )
    frame = pd.DataFrame({"score": [580, 650, 750]})

    decisions = apply_strategy(frame, strategy)

    assert strategy.strategy_type == "approval"
    assert strategy.score_col == "score"
    assert strategy.description == "baseline cutoff"
    assert decisions.tolist() == ["reject", "approve", "approve"]


def test_apply_strategy_uses_first_matching_rule():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score >= 600", "decision": "approve"},
            {"condition": "score >= 700", "decision": "reject"},
        ],
        score_col="score",
        default_decision="reject",
    )

    decisions = apply_strategy(pd.DataFrame({"score": [750]}), strategy)

    assert decisions.tolist() == ["approve"]


def test_apply_strategy_supports_in_conditions_and_rule_values():
    strategy = build_strategy(
        "segmentation",
        [
            {"condition": "grade in ['A', 'B']", "decision": "segment", "value": "prime"},
            {"condition": "grade == 'C' or score >= 700", "decision": "segment", "value": "watch"},
        ],
        score_col="score",
        default_decision="other",
    )
    frame = pd.DataFrame({"grade": ["A", "C", "D"], "score": [610, 650, 720]})

    assert apply_strategy(frame, strategy).tolist() == ["prime", "watch", "watch"]


def test_build_strategy_rejects_decision_mismatch_and_unknown_type():
    with pytest.raises(StrategyError, match="decision"):
        build_strategy(
            "approval",
            [{"condition": "score < 600", "decision": "limit", "value": 1000}],
            score_col="score",
            default_decision="approve",
        )
    with pytest.raises(StrategyError, match="strategy_type"):
        build_strategy(
            "unknown",
            [{"condition": "score < 600", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )


def test_safe_condition_rejects_calls_attributes_and_unknown_columns():
    with pytest.raises(StrategyError, match="unsupported condition"):
        build_strategy(
            "approval",
            [{"condition": "__import__('os').system('touch /tmp/marvis_pwned') == 0", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )
    with pytest.raises(StrategyError, match="unsupported condition"):
        build_strategy(
            "approval",
            [{"condition": "score.__class__ == int", "decision": "reject"}],
            score_col="score",
            default_decision="approve",
        )

    strategy = build_strategy(
        "approval",
        [{"condition": "score < 600", "decision": "reject"}],
        score_col="score",
        default_decision="approve",
    )
    with pytest.raises(StrategyError, match="unknown field"):
        apply_strategy(pd.DataFrame({"model_score": [500]}), strategy)


def test_strategy_package_exports_build_surface():
    assert strategy_pack.StrategyError is StrategyError
    assert strategy_pack.build_strategy is build_strategy
    assert strategy_pack.apply_strategy is apply_strategy


def test_build_strategy_raises_on_inconsistent_rule_directions():
    """Two rules land on the SAME decision via opposite comparison styles against
    score_col -- no single score direction can explain "always reject" whether the
    score is low or high, unlike a coherent banded cutoff strategy."""
    with pytest.raises(ScoreDirectionConflictError, match="score_direction conflict"):
        build_strategy(
            "approval",
            [
                {"condition": "score < 500", "decision": "reject"},
                {"condition": "score >= 900", "decision": "reject"},
            ],
            score_col="score",
            default_decision="approve",
        )


def test_build_strategy_single_direction_rules_pass_and_report_inferred_direction():
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score >= 600", "decision": "approve"},
            {"condition": "score >= 750", "decision": "approve"},
        ],
        score_col="score",
        default_decision="reject",
    )

    assert infer_strategy_rule_direction(list(strategy.rules), strategy.score_col) == "higher_is_better"


def test_build_strategy_banded_cutoff_does_not_raise_despite_opposite_operator_styles():
    """score < 600 -> reject and score >= 720 -> approve use opposite comparison
    styles but agree that a higher score is better (reject the low band, approve the
    high band) -- a coherent monotonic cutoff strategy, not a direction conflict."""
    strategy = build_strategy(
        "approval",
        [
            {"condition": "score < 600", "decision": "reject"},
            {"condition": "score >= 720", "decision": "approve"},
        ],
        score_col="score",
        default_decision="approve",
    )

    assert strategy.score_col == "score"
