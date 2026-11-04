"""Edge contracts for the independently written optional feature calculations."""
import json

import numpy as np
import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_feature import _optional_metrics
from marvis.orchestrator.eval.runtime_feature_extended_reference import head_tail_lift_reference, meaning_consistency_reference
from marvis.orchestrator.eval.runtime_feature_reference import feature_recommendation_reference


@pytest.mark.parametrize("reverse", [False, True])
def test_head_tail_uses_risk_direction_stable_ties_and_exact_floor_slices(reverse):
    values = [float(i // 2) * (-1 if reverse else 1) for i in range(40)] + [float("nan")]
    labels = [0.0] * 30 + [1.0] * 10 + [1.0]
    result = head_tail_lift_reference(values, labels)
    assert result == {"lift_head_5": 4.0, "lift_tail_5": 0.0, "lift_head_10": 4.0, "lift_tail_10": 0.0, "lift_reason": ""}


@pytest.mark.parametrize("direction", ["positive", "negative", "uncertain", "u_shape"])
def test_meaning_direction_is_a_check_against_frozen_decision_not_semantic_truth(direction):
    x = np.arange(140, dtype=float)
    y = ((x < 30) | (x > 110)).astype(float)
    result = meaning_consistency_reference(x, y, feature="age", meaning="年龄", decision={"expected_direction": direction})
    assert result["expected_direction"] == direction
    if direction == "u_shape":
        from marvis.packs.feature.tools import _actual_u_shape_direction
        actual, evidence, reason = _actual_u_shape_direction(x, y, feature="age")
        assert result["actual_direction"] == actual == "u_shape"
        assert result["u_shape_evidence"] == evidence
        assert result["meaning_consistency_reason"] == reason
        assert result["meaning_consistency"] == "一致"
    elif direction == "uncertain":
        assert result["meaning_consistency"] == "需人工看"


def test_split_discovery_checks_roles_and_can_skip_an_unusable_first_candidate():
    frame = pd.DataFrame({"x": range(60), "y": [0, 1] * 30, "split": ["unknown"] * 60,
                          "sample_split": ["train", "test", "oot"] * 20})
    source = {"columns_json": json.dumps([{"name": c} for c in frame.columns])}
    result = _optional_metrics(frame, source, "x", "y", {"psi_split"}, {}, {})
    assert [r["compare"] for r in result["psi_split_series"]] == ["test", "oot"]
    missing = _optional_metrics(frame.drop(columns="sample_split"), source, "x", "y", {"psi_split"}, {}, {})
    assert missing["psi_split_reason"]["code"] == "missing_dependency"


def test_meaning_inconsistency_and_stability_affect_recommendations():
    assert feature_recommendation_reference({"ks": .2, "psi_month_first": .05})["recommendation"] == "推荐"
    assert feature_recommendation_reference({"ks": .2, "psi_month_first": .3})["recommendation"] == "不推荐"
    assert feature_recommendation_reference({"ks": .2, "meaning_consistency": "不一致"})["recommendation"] == "不推荐"
    assert feature_recommendation_reference({"ks": None})["recommendation"] == "待评估"
