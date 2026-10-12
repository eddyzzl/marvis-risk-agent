"""Independent arithmetic compared with actual native typed kernel outputs."""
import json

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_domain import numbers
from marvis.orchestrator.eval.runtime_archive_strategy import _bad_target
from marvis.orchestrator.eval.runtime_strategy_reference import native_economics_bindings, typed_impact_reference
from marvis.packs.strategy.dsl import canonical_strategy_json
from marvis.packs.strategy.typed_backtest import run_typed_backtest
from tests.test_strategy_typed_backtest import _decision_spec, _limit_spec, _pricing_spec, _segmentation_spec


@pytest.mark.parametrize("kind", ["approval", "reject", "segmentation", "limit", "pricing"])
def test_five_native_typed_profiles_have_independent_population_transitions_and_economics(kind):
    frame = pd.DataFrame({"x": [0, 1, 2, 3], "score": [800, 650, 400, 600], "y": [1, None, 0, 1],
                          "pd": [.1, .2, .3, .4], "ead": [1000, 0, 2000, 4000]})
    builders = {
        "approval": lambda: _decision_spec(), "reject": lambda: _decision_spec("reject"),
        "segmentation": _segmentation_spec, "limit": lambda: _limit_spec((2000, 1000, 2000, 0)),
        "pricing": lambda: _pricing_spec((.12, .18)),
    }
    spec = builders[kind]()
    baseline = _limit_spec((1500, 1000, 2500, 0)) if kind == "limit" else _pricing_spec((.10, .16)) if kind == "pricing" else spec
    native_inputs = {"pd": frame.pd, "lgd": .5, "utilization": .7} if kind == "limit" else {
        "ead": frame.ead, "pd": frame.pd, "lgd": .5, "funding_rate": .03, "term_months": 12,
        "operating_cost_per_loan": 10} if kind == "pricing" else None
    tool_inputs = {"pd_col": "pd", "lgd_value": .5, "utilization_value": .7} if kind == "limit" else {
        "ead_col": "ead", "pd_col": "pd", "lgd_value": .5, "funding_rate_value": .03, "term_months_value": 12,
        "operating_cost_per_loan_value": 10} if kind == "pricing" else None
    observed = run_typed_backtest(frame, spec, target_col="y", baseline=baseline, economics_inputs=native_inputs).to_dict()
    independent = typed_impact_reference(frame.to_dict("records"), json.loads(canonical_strategy_json(spec)), "y",
        baseline=json.loads(canonical_strategy_json(baseline)), economics=native_economics_bindings(tool_inputs, kind, frame.columns))
    numbers(observed, independent)
    assert independent["population_count"] == 4 and independent["labeled_count"] == 3
    if kind == "limit":
        assert [row["assigned_limit"] for row in independent["breakdown"]] == [0, 1000, 2000]
    if kind in {"limit", "pricing"}:
        assert independent["economics"]


@pytest.mark.parametrize("payload", [
    {"pd_value": .1}, {"pd_value": .1, "pd_col": "pd", "lgd_value": .5, "utilization_value": .8},
    {"pd_col": "absent", "lgd_value": .5, "utilization_value": .8},
    {"pd_value": True, "lgd_value": .5, "utilization_value": .8},
])
def test_native_economic_binding_requires_complete_exact_authenticated_fields(payload):
    with pytest.raises(ValueError):
        native_economics_bindings(payload, "limit", ["pd"])


@pytest.mark.parametrize("bad", [0, 1])
def test_strategy_bad_label_comes_from_frozen_native_request(bad):
    request = {"target_col": "y", "target_bad_value": bad}
    output = {"normalized_input": {"target_encoding": {"bad": bad}}}
    assert _bad_target({"target_col": "y"}, output, request) == bad
    output["normalized_input"]["target_encoding"]["bad"] = 1 - bad
    with pytest.raises(ValueError, match="bad-label direction"):
        _bad_target({"target_col": "y"}, output, request)
    with pytest.raises(ValueError, match="target differs"):
        _bad_target({"target_col": "other"}, output, request)


def test_strategy_without_sample_contract_uses_installed_binary_default():
    assert _bad_target({"target_col": "y"}, {"normalized_input": {"target_encoding": {"bad": 1}}}, None) == 1
    with pytest.raises(ValueError, match="bad-label direction"):
        _bad_target({"target_col": "y"}, {"normalized_input": {"target_encoding": {"bad": 0}}}, None)
