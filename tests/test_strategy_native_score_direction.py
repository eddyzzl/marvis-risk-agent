"""The score direction comes from the exact authenticated native sample contract."""

import json

import pytest

from marvis.packs.strategy import tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design_v2_native_tools import (
    run_materialize_sample_design_v2_native,
)
from test_strategy_sample_design_v2_native_tool import _setup_native


def _native_case(tmp_path):
    fx = _setup_native(tmp_path)
    output = run_materialize_sample_design_v2_native(
        fx["request"], fx["ctx"], fx["runtime"]
    )
    record = next(
        record
        for record in fx["runtime"].task_artifacts.list_for_task(fx["task"].id)
        if record["kind"] == "strategy_sample_design_v2_json"
    )
    inputs = {
        "dataset_id": fx["dataset"].id,
        "target_col": "bad",
        "score_col": "legacy_score",
        "drop_nan_labels": True,
        "sample_design_ref": {
            "artifact_id": record["id"],
            "artifact_content_hash": output["artifacts"]["bundle"]["content_hash"],
            "sample_design_id": output["sample_design_id"],
            "sample_design_content_hash": output["sample_design_content_hash"],
            "partition": "risk/development",
        },
        "objective": "max_approval",
        "max_bad_rate": 0.5,
        "min_approval_rate": 0.4,
    }
    return fx, inputs


@pytest.mark.parametrize(
    "tool", [tools.tool_tradeoff_view, tools.tool_design_cutoff_bands]
)
def test_native_declared_direction_is_consumed_without_an_optional_slot(tmp_path, tool):
    fx, inputs = _native_case(tmp_path)
    output = tool(inputs, fx["ctx"])
    assert output["score_direction"] == "higher_is_riskier"


@pytest.mark.parametrize(
    "tool", [tools.tool_tradeoff_view, tools.tool_design_cutoff_bands]
)
def test_explicit_opposite_direction_cannot_override_frozen_contract(tmp_path, tool):
    fx, inputs = _native_case(tmp_path)
    with pytest.raises(StrategyError, match="frozen historical score direction"):
        tool(
            {
                **inputs,
                "score_direction": "higher_is_better",
                "confirm_direction_conflict": True,
            },
            fx["ctx"],
        )


def test_authenticated_native_direction_is_bound_to_the_exact_column(tmp_path):
    fx, inputs = _native_case(tmp_path)
    from marvis.packs.strategy.sample_design_execution import (
        load_historical_strategy_risk_development_execution_binding_from_ref,
    )

    binding = load_historical_strategy_risk_development_execution_binding_from_ref(
        fx["runtime"],
        task_id=fx["task"].id,
        sample_design_ref=inputs["sample_design_ref"],
    )
    assert binding.score_direction_for("legacy_score") == "higher_is_riskier"
    assert binding.score_direction_for("unused_feature") is None


def test_changed_native_direction_bytes_are_not_new_authority(tmp_path):
    fx, inputs = _native_case(tmp_path)
    record = fx["runtime"].task_artifacts.get_for_task(
        fx["task"].id, inputs["sample_design_ref"]["artifact_id"]
    )
    path = fx["settings"].workspace / record["path"]
    payload = json.loads(path.read_text())
    payload["historical_score"]["direction"] = "lower_is_riskier"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, RuntimeError)):
        tools.tool_tradeoff_view(inputs, fx["ctx"])


def test_native_high_risk_direction_crosses_real_http_and_toolrunner(tmp_path):
    import hashlib
    import pandas as pd
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_normal_strategy import _strategy_protocol

    paths = write_synthetic_suite(tmp_path / "suite", normal_strategy_only=True)
    suite = json.loads(paths["cases"].read_text())
    case = suite["cases"][0]
    material = case["materials"][0]
    path = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(path)
    frame["credit_score"] = 1 - frame["credit_score"]
    frame.to_parquet(path, index=False)
    material["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    case["business_constraints_source"] += (
        " This independent regression uses an explicitly declared higher-is-riskier score."
    )
    for action in case["actions"]:
        action["content"] = action["content"].replace("越低越风险", "越高越风险")
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    expected = json.loads(paths["expected"].read_text())
    for tool in ("tradeoff_view", "design_cutoff_bands"):
        expected["cases"][case["id"]]["assertions"].append(
            {
                "kind": "output_equals",
                "tool": "strategy." + tool,
                "path": ["score_direction"],
                "value": "higher_is_riskier",
            }
        )
    paths["expected"].write_text(json.dumps(expected))

    def high_risk_protocol(request, answer, payload):
        result = _strategy_protocol(request, answer, payload)
        if result.get("workflow") == "strategy_sample_design_v2":
            result["workflow_inputs"]["historical_score"]["direction"] = (
                "higher_is_riskier"
            )
        return result

    with fixture_model(answer_factory=high_risk_protocol) as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"],
            expected_path=paths["expected"],
            dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "runs",
            model=model,
            model_source="fixture_model",
        )
    assert report["all_passed"], json.dumps(report, ensure_ascii=False, indent=2)
