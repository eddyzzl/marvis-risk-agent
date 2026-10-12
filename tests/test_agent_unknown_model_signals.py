from marvis.agent.gate_payloads import _business_signals


def test_explicit_unknown_model_overfit_signal_remains_unknown_at_agent_boundary():
    result = _business_signals({"metrics": {"train_auc": 0.8, "test_auc": None,
                                            "oot_auc": None, "overfit_flag": None}})
    assert result["overfit_flag"] is None
    assert result["stability"] == "待评估"
    assert result["generalization_gap"] is None
    assert _business_signals({"metrics": {}})["overfit_flag"] is False
    assert _business_signals({"metrics": {"overfit_flag": False}})["overfit_flag"] is False
    assert _business_signals({"metrics": {"overfit_flag": True}})["overfit_flag"] is True
