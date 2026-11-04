"""One PSI band convention across feature advice, delivery cards and LLM text.

Stable below 0.10, attention from 0.10 up to below 0.25, drift at 0.25 and
above. Feature advice used ``> 0.10`` / ``> 0.25`` while the delivery card used
``>= 0.10`` / ``>= 0.25``, so a feature at exactly 0.10 was 推荐 in one place
and 关注 in the other.
"""
import json

import pytest

from marvis.agent.gate_payloads import _stability_label
from marvis.llm_outbound_workflows import workflow_insight_prompt
from marvis.packs.feature.tools import _feature_recommendation

_SIGNAL = {"valid_count": 480, "iv": 0.3, "ks": 0.2, "auc": 0.65}


@pytest.mark.parametrize(("psi", "feature_state", "card_state"), [
    (0.0999, "推荐", "稳定"),
    (0.10, "谨慎", "关注"),
    (0.2499, "谨慎", "关注"),
    (0.25, "不推荐", "高风险"),
    (0.31, "不推荐", "高风险"),
])
def test_feature_advice_and_delivery_card_share_psi_bands(psi, feature_state, card_state):
    state, reason = _feature_recommendation({**_SIGNAL, "psi": psi})

    assert state == feature_state
    assert f"PSI={psi:.3f}" in reason or feature_state == "推荐"
    assert _stability_label(psi, None, None, False) == card_state
    assert _stability_label(None, psi, None, False) == card_state


def test_split_and_monthly_psi_views_use_the_same_inclusive_bands():
    assert _feature_recommendation({**_SIGNAL, "psi_split": 0.10})[0] == "谨慎"
    assert _feature_recommendation({**_SIGNAL, "psi_month_first": 0.25})[0] == "不推荐"


@pytest.mark.parametrize("milestone", ["feature_analyzed", "model_completed"])
def test_interpretation_instruction_states_inclusive_boundaries(milestone):
    evidence = ({"feature_count": 1, "features": [{"feature": "age", "psi": 0.10}]}
                if milestone.startswith("feature_") else {"metrics": {"oot_ks": 0.31}})
    prompt = workflow_insight_prompt({"milestone": milestone, "measured_evidence": evidence})
    instruction = json.loads(prompt)["instruction"]

    assert "低于 0.10 为稳定" in instruction
    assert "0.10 及以上、低于 0.25 需关注" in instruction
    assert "0.25 及以上为明显漂移" in instruction
    assert "恰好 0.10 算需关注" in instruction and "恰好 0.25 算明显漂移" in instruction


def test_join_interpretation_does_not_carry_psi_bands():
    prompt = workflow_insight_prompt({"milestone": "join_completed",
                                      "measured_evidence": {"anchor_rows": 10, "joined_rows": 10}})
    assert "PSI" not in json.loads(prompt)["instruction"]


@pytest.mark.parametrize(("psi", "state"), [(0.2499, "谨慎"), (0.25, "不推荐"), (0.40, "不推荐")])
def test_suspected_leakage_never_softens_a_drift_verdict(psi, state):
    leaky = {**_SIGNAL, "iv": 1.6, "ks": 0.46, "auc": 0.8, "psi": psi}
    verdict, reason = _feature_recommendation(leaky)

    assert verdict == state
    assert "疑似目标泄漏" in reason
    if state == "不推荐":
        assert f"PSI={psi:.3f}" in reason
