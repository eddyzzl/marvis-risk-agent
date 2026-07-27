from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import marvis.packs.strategy.sample_design as sample_design_module

from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.sample_design import (
    METRIC_OBSERVATION_STATUSES,
    STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION,
    build_strategy_sample_design_bundle,
    canonical_strategy_sample_design_bundle_json,
    strategy_sample_design_bundle_from_json,
    validate_strategy_sample_design_bundle,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target": [0, 1, np.nan, 0, 1, 1],
            "sample_split": ["dev", "dev", "val", "val", "oot", "oot"],
            "month": ["2026-01", "2026-01", "2026-02", "2026-02", "2026-03", "2026-03"],
            "weight": [1.0, 3.0, 2.0, np.nan, 4.0, 0.0],
            "loan_amount": [100.0, 200.0, np.nan, 50.0, 300.0, 100.0],
            "overdue_amount": [0.0, 50.0, 20.0, 0.0, 150.0, np.nan],
        }
    )


def _build(**overrides):
    params = {
        "frame": _frame(),
        "task_id": "task-strategy",
        "dataset_id": "dataset-active",
        "dataset_content_hash": "a" * 64,
        "workspace_revision": 4,
        "workspace_generation": 7,
        "semantic_mapping_hash": "b" * 64,
        "target_col": "target",
        "target_bad_value": 1,
        "drop_nan_labels": True,
        "performance_window": {"status": "provided", "days": 30},
        "observation_window": {
            "status": "provided",
            "start": "2026-01-01",
            "end": "2026-03-31",
        },
        "split_definition": {
            "status": "available",
            "column": "sample_split",
            "development_values": ["dev"],
            "validation_values": ["val"],
            "oot_values": ["oot"],
        },
        "maturity": "confirmed_matured",
        "month_col": "month",
        "weight_col": "weight",
        "loan_amount_col": "loan_amount",
        "overdue_amount_col": "overdue_amount",
    }
    params.update(overrides)
    return build_strategy_sample_design_bundle(**params)


def _observations_by_dimension(bundle, kind: str, value: str):
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    return {
        definitions[item["metric_definition_ref"]["metric_definition_id"]]: item
        for item in bundle["metric_observations"]
        if item["dimension"] == {"kind": kind, "value": value}
    }


def test_builds_deterministic_content_addressed_bundle_with_all_dimensions():
    first = _build()
    second = _build()

    assert first == second
    assert first["schema_version"] == STRATEGY_SAMPLE_DESIGN_BUNDLE_SCHEMA_VERSION
    assert first["bundle_id"].startswith("strategy-sample-design-bundle-")
    assert first["sample_design"]["scope"] == "strategy_development"
    assert first["sample_design"]["lifecycle"] == {
        "candidate_stage": "development",
        "validation_status": "unvalidated",
        "oot_validation_claimed": False,
        "creates_strategy": False,
        "adopted": False,
        "deployed": False,
    }
    assert len(first["metric_definitions"]) == 13
    assert len(first["metric_observations"]) == 52
    assert all(len(item["content_hash"]) == 64 for item in first["metric_definitions"])
    assert all(len(item["content_hash"]) == 64 for item in first["metric_observations"])

    overall = _observations_by_dimension(first, "overall", "overall")
    assert overall["population_count"]["value"] == 6
    assert overall["labeled_count"]["value"] == 5
    assert overall["good_count"]["value"] == 2
    assert overall["bad_count"]["value"] == 3
    assert overall["bad_rate"]["value"] == pytest.approx(3 / 5)
    assert overall["label_coverage"]["value"] == pytest.approx(5 / 6)
    assert overall["loan_amount_coverage"]["value"] == pytest.approx(5 / 6)
    assert overall["loan_amount_sum"]["value"] == 750.0
    assert overall["overdue_amount_coverage"]["value"] == pytest.approx(5 / 6)
    assert overall["overdue_amount_sum"]["value"] == 220.0
    assert overall["weight_coverage"]["value"] == pytest.approx(5 / 6)
    assert overall["weight_sum"]["value"] == 10.0
    assert overall["weighted_bad_rate"]["value"] == pytest.approx(7 / 8)

    development = _observations_by_dimension(first, "split", "development")
    assert development["population_count"]["value"] == 2
    assert development["bad_rate"]["value"] == 0.5
    assert development["weighted_bad_rate"]["value"] == 0.75
    validation = _observations_by_dimension(first, "split", "validation")
    assert validation["population_count"]["value"] == 2
    assert validation["labeled_count"]["value"] == 1
    oot = _observations_by_dimension(first, "split", "oot")
    assert oot["bad_rate"]["value"] == 1.0


def test_canonical_roundtrip_is_byte_stable_and_rejects_duplicate_keys():
    bundle = _build()
    raw = canonical_strategy_sample_design_bundle_json(bundle)

    assert strategy_sample_design_bundle_from_json(raw) == bundle
    assert canonical_strategy_sample_design_bundle_json(
        strategy_sample_design_bundle_from_json(raw)
    ) == raw
    with pytest.raises(StrategyError, match="duplicate key"):
        strategy_sample_design_bundle_from_json(
            '{"schema_version":"x","schema_version":"y"}'
        )


def test_missing_labels_require_explicit_drop_but_remain_in_population():
    with pytest.raises(StrategyError, match="drop_nan_labels=true"):
        _build(drop_nan_labels=False)

    bundle = _build(drop_nan_labels=True)
    flags = bundle["sample_design"]["red_flags"]
    assert any(
        item["code"] == "missing_labels_excluded_from_risk:1" for item in flags
    )
    overall = _observations_by_dimension(bundle, "overall", "overall")
    assert overall["population_count"]["value"] == 6
    assert overall["labeled_count"]["value"] == 5


def test_explicit_reverse_target_polarity_drives_metrics_and_definitions():
    bundle = _build(target_bad_value=0)

    assert bundle["sample_design"]["target_definition"] == {
        "column": "target",
        "good_value": 1,
        "bad_value": 0,
        "drop_nan_labels": True,
    }
    overall = _observations_by_dimension(bundle, "overall", "overall")
    assert overall["good_count"]["value"] == 3
    assert overall["bad_count"]["value"] == 2
    assert overall["bad_rate"]["value"] == pytest.approx(2 / 5)
    assert overall["weighted_bad_rate"]["value"] == pytest.approx(1 / 8)
    semantics = {
        item["label_semantics"]
        for item in bundle["metric_definitions"]
        if item["label_semantics"] is not None
    }
    assert semantics == {
        "target 1=good, 0=bad; missing is never treated as good"
    }


@pytest.mark.parametrize("invalid", [True, False, -1, 2, 0.5, float("nan"), "1"])
def test_target_bad_value_must_be_explicit_integer_binary(invalid):
    with pytest.raises(StrategyError, match="target_bad_value"):
        _build(target_bad_value=invalid)


def test_json_integral_target_polarity_is_canonicalized_to_integer():
    assert _build(target_bad_value=1.0) == _build(target_bad_value=1)


@pytest.mark.parametrize("invalid", [2, -1, "1", True, np.inf])
def test_target_accepts_only_numeric_zero_one_or_missing(invalid):
    frame = _frame()
    frame["target"] = frame["target"].astype("object")
    frame.loc[0, "target"] = invalid
    with pytest.raises(StrategyError, match="target row"):
        _build(frame=frame)


def test_available_split_must_be_disjoint_and_cover_every_row_exactly():
    overlap = {
        "status": "available",
        "column": "sample_split",
        "development_values": ["dev", "oot"],
        "validation_values": ["val"],
        "oot_values": ["oot"],
    }
    with pytest.raises(StrategyError, match="overlap"):
        _build(split_definition=overlap)

    unknown = _frame()
    unknown.loc[0, "sample_split"] = "holdout"
    with pytest.raises(StrategyError, match="unknown value"):
        _build(frame=unknown)

    missing = _frame()
    missing.loc[0, "sample_split"] = None
    with pytest.raises(StrategyError, match="missing value"):
        _build(frame=missing)

    no_development = {
        "status": "available",
        "column": "sample_split",
        "development_values": [],
        "validation_values": ["dev", "val"],
        "oot_values": ["oot"],
    }
    with pytest.raises(StrategyError, match="requires development_values"):
        _build(split_definition=no_development)


def test_numeric_split_values_have_one_canonical_json_identity():
    frame = _frame()
    frame["sample_split"] = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    integer_request = {
        "status": "available",
        "column": "sample_split",
        "development_values": [1],
        "validation_values": [2],
        "oot_values": [3],
    }
    float_request = {
        **integer_request,
        "development_values": [1.0],
        "validation_values": [2.0],
        "oot_values": [3.0],
    }

    first = _build(frame=frame, split_definition=integer_request)
    second = _build(frame=frame, split_definition=float_request)

    assert first == second
    assert first["sample_design"]["split_definition"]["development_values"] == [1]

    overlap = {
        **integer_request,
        "validation_values": [1.0],
    }
    with pytest.raises(StrategyError, match="overlap"):
        _build(frame=frame, split_definition=overlap)


@pytest.mark.parametrize("invalid", [-0.1, np.inf, "3"])
def test_weight_must_be_non_negative_and_finite_when_present(invalid):
    frame = _frame()
    frame["weight"] = frame["weight"].astype("object")
    frame.loc[0, "weight"] = invalid
    with pytest.raises(StrategyError, match="weight row"):
        _build(frame=frame)


def test_unavailable_optional_fields_emit_explicit_unavailable_observations():
    bundle = _build(
        frame=_frame().drop(columns=["weight", "loan_amount", "overdue_amount"]),
        weight_col=None,
        loan_amount_col=None,
        overdue_amount_col=None,
    )
    overall = _observations_by_dimension(bundle, "overall", "overall")
    for metric_key in (
        "loan_amount_coverage",
        "loan_amount_sum",
        "overdue_amount_coverage",
        "overdue_amount_sum",
        "weight_coverage",
        "weight_sum",
        "weighted_bad_rate",
    ):
        assert overall[metric_key]["status"] == "unavailable"
        assert overall[metric_key]["value"] is None


@pytest.mark.parametrize(
    ("performance_window", "maturity", "flag_code"),
    [
        (
            {"status": "unavailable", "days": None},
            "confirmed_matured",
            "performance_window_unavailable",
        ),
        ({"status": "provided", "days": 30}, "not_matured", "sample_not_matured"),
        ({"status": "provided", "days": 30}, "unknown", "sample_maturity_unknown"),
    ],
)
def test_missing_performance_window_or_unconfirmed_maturity_forces_exploration(
    performance_window, maturity, flag_code
):
    bundle = _build(performance_window=performance_window, maturity=maturity)
    assert bundle["sample_design"]["scope"] == "exploration_only"
    assert any(
        item["code"] == flag_code
        for item in bundle["sample_design"]["red_flags"]
    )
    assert all(
        item["stage"] == "development"
        and item["scope"] == "exploration_only"
        and item["maturity"] == maturity
        for item in bundle["metric_observations"]
    )


def test_unavailable_split_and_window_are_explicit_and_only_overall_is_emitted():
    bundle = _build(
        observation_window={"status": "unavailable", "start": None, "end": None},
        split_definition={
            "status": "unavailable",
            "column": None,
            "development_values": [],
            "validation_values": [],
            "oot_values": [],
        },
    )
    assert len(bundle["metric_observations"]) == 13
    assert {
        (item["dimension"]["kind"], item["dimension"]["value"])
        for item in bundle["metric_observations"]
    } == {("overall", "overall")}
    codes = {item["code"] for item in bundle["sample_design"]["red_flags"]}
    assert {"observation_window_unavailable", "split_unavailable"} <= codes


def test_zero_population_split_uses_insufficient_data_not_fake_zero_rate():
    split = {
        "status": "available",
        "column": "sample_split",
        "development_values": ["dev"],
        "validation_values": ["val"],
        "oot_values": ["oot", "future_oot"],
    }
    # The OOT dimension is not empty here, so use an empty validation assignment instead.
    frame = _frame()
    frame["sample_split"] = ["dev", "dev", "dev", "dev", "oot", "oot"]
    bundle = _build(frame=frame, split_definition=split)
    validation = _observations_by_dimension(bundle, "split", "validation")
    assert validation["population_count"]["value"] == 0
    assert validation["bad_rate"]["status"] == "insufficient_data"
    assert validation["bad_rate"]["value"] is None
    assert validation["label_coverage"]["status"] == "insufficient_data"
    assert any(
        flag["code"] == "validation_split_empty"
        for flag in bundle["sample_design"]["red_flags"]
    )


def test_undefined_validation_and_oot_splits_emit_no_fake_zero_observations():
    frame = _frame()
    frame["sample_split"] = ["dev"] * len(frame)
    bundle = _build(
        frame=frame,
        split_definition={
            "status": "available",
            "column": "sample_split",
            "development_values": ["dev"],
            "validation_values": [],
            "oot_values": [],
        },
    )

    dimensions = {
        (item["dimension"]["kind"], item["dimension"]["value"])
        for item in bundle["metric_observations"]
    }
    assert dimensions == {("overall", "overall"), ("split", "development")}
    assert len(bundle["metric_observations"]) == 26
    assert bundle["sample_design"]["split_population_counts"] == {
        "development": 6,
        "validation": 0,
        "oot": 0,
    }
    codes = {item["code"] for item in bundle["sample_design"]["red_flags"]}
    assert {"validation_split_unavailable", "oot_split_unavailable"} <= codes


@pytest.mark.parametrize("field", ["loan_amount", "overdue_amount", "weight"])
def test_bound_all_missing_amount_or_weight_sum_is_not_reported_as_zero(field):
    frame = _frame()
    frame[field] = np.nan
    bundle = _build(frame=frame)
    overall = _observations_by_dimension(bundle, "overall", "overall")

    total = overall[f"{field}_sum"]
    assert total["status"] == "insufficient_data"
    assert total["value"] is None
    assert overall[f"{field}_coverage"]["value"] == 0.0


def test_available_split_requires_actual_development_rows():
    frame = _frame()
    frame["sample_split"] = ["val", "val", "val", "val", "oot", "oot"]
    with pytest.raises(StrategyError, match="development row"):
        _build(frame=frame)


def test_unavailable_observation_window_forces_exploration_and_hides_risk_metrics():
    bundle = _build(
        observation_window={"status": "unavailable", "start": None, "end": None}
    )
    assert bundle["sample_design"]["scope"] == "exploration_only"
    overall = _observations_by_dimension(bundle, "overall", "overall")
    for key in (
        "good_count",
        "bad_count",
        "bad_rate",
        "overdue_amount_coverage",
        "overdue_amount_sum",
        "weighted_bad_rate",
    ):
        assert overall[key]["status"] == "unavailable"
        assert overall[key]["value"] is None


def test_unconfirmed_maturity_hides_risk_metrics_as_not_matured():
    bundle = _build(maturity="not_matured")
    overall = _observations_by_dimension(bundle, "overall", "overall")
    for key in (
        "good_count",
        "bad_count",
        "bad_rate",
        "overdue_amount_coverage",
        "overdue_amount_sum",
        "weighted_bad_rate",
    ):
        assert overall[key]["status"] == "not_matured"
        assert overall[key]["value"] is None


def test_exact_fields_and_content_hash_drift_fail_closed():
    bundle = _build()
    unknown = deepcopy(bundle)
    unknown["unexpected"] = True
    with pytest.raises(StrategyError, match="fields are invalid"):
        validate_strategy_sample_design_bundle(unknown)

    drifted = deepcopy(bundle)
    drifted["sample_design"]["identity"]["task_id"] = "another-task"
    with pytest.raises(StrategyError, match="does not match content"):
        validate_strategy_sample_design_bundle(drifted)

    observation_drift = deepcopy(bundle)
    observation_drift["metric_observations"][0]["value"] = 999
    with pytest.raises(StrategyError, match="does not match content"):
        validate_strategy_sample_design_bundle(observation_drift)


def test_invalid_windows_and_noncanonical_unavailable_shapes_fail_closed():
    with pytest.raises(StrategyError, match="must not be after"):
        _build(
            observation_window={
                "status": "provided",
                "start": "2026-04-01",
                "end": "2026-03-31",
            }
        )
    with pytest.raises(StrategyError, match="null days"):
        _build(performance_window={"status": "unavailable", "days": 30})
    with pytest.raises(StrategyError, match="null column"):
        _build(
            split_definition={
                "status": "unavailable",
                "column": "sample_split",
                "development_values": [],
                "validation_values": [],
                "oot_values": [],
            }
        )


def test_bounded_json_depth_and_nodes_fail_closed(monkeypatch):
    import marvis.packs.strategy.sample_design as module

    bundle = _build()
    deep = deepcopy(bundle)
    nested = deep
    for _ in range(40):
        nested["x"] = {}
        nested = nested["x"]
    with pytest.raises(StrategyError, match="depth budget"):
        validate_strategy_sample_design_bundle(deep)

    monkeypatch.setattr(module, "MAX_SAMPLE_DESIGN_JSON_NODES", 20)
    with pytest.raises(StrategyError, match="node budget"):
        validate_strategy_sample_design_bundle(bundle)


def test_bounded_json_rejects_cycles_and_oversized_split_controls():
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(StrategyError, match="repeated or cyclic"):
        validate_strategy_sample_design_bundle(cyclic)

    too_many = {
        "status": "available",
        "column": "sample_split",
        "development_values": [f"dev-{index}" for index in range(101)],
        "validation_values": [],
        "oot_values": [],
    }
    with pytest.raises(StrategyError, match="item budget"):
        _build(split_definition=too_many)

    too_long = deepcopy(too_many)
    too_long["development_values"] = ["x" * 257]
    with pytest.raises(StrategyError, match="string length budget"):
        _build(split_definition=too_long)


def test_validator_rejects_rehashed_fractional_coverage_operands():
    bundle = deepcopy(_build())
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    index = next(
        index
        for index, item in enumerate(bundle["metric_observations"])
        if item["dimension"] == {"kind": "overall", "value": "overall"}
        and definitions[item["metric_definition_ref"]["metric_definition_id"]]
        == "loan_amount_coverage"
    )
    observation = bundle["metric_observations"][index]
    observation_body = {
        key: value
        for key, value in observation.items()
        if key not in {"observation_id", "content_hash"}
    }
    observation_body["numerator"] = 1.5
    observation_body["value"] = 1.5 / observation_body["denominator"]
    bundle["metric_observations"][index] = sample_design_module._address_object(
        observation_body,
        id_field="observation_id",
        id_prefix="metric-observation-",
    )
    bundle_body = {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_id", "content_hash"}
    }
    rehashed = sample_design_module._address_object(
        bundle_body,
        id_field="bundle_id",
        id_prefix="strategy-sample-design-bundle-",
    )
    with pytest.raises(StrategyError, match="coverage operands"):
        validate_strategy_sample_design_bundle(rehashed)


def test_validator_rejects_rehashed_operands_on_sum_metric():
    bundle = deepcopy(_build())
    definitions = {
        item["metric_definition_id"]: item["metric_key"]
        for item in bundle["metric_definitions"]
    }
    index = next(
        index
        for index, item in enumerate(bundle["metric_observations"])
        if item["dimension"] == {"kind": "overall", "value": "overall"}
        and definitions[item["metric_definition_ref"]["metric_definition_id"]]
        == "loan_amount_sum"
    )
    observation = bundle["metric_observations"][index]
    observation_body = {
        key: value
        for key, value in observation.items()
        if key not in {"observation_id", "content_hash"}
    }
    observation_body["numerator"] = 123
    observation_body["denominator"] = 456
    bundle["metric_observations"][index] = sample_design_module._address_object(
        observation_body,
        id_field="observation_id",
        id_prefix="metric-observation-",
    )
    bundle_body = {
        key: value
        for key, value in bundle.items()
        if key not in {"bundle_id", "content_hash"}
    }
    rehashed = sample_design_module._address_object(
        bundle_body,
        id_field="bundle_id",
        id_prefix="strategy-sample-design-bundle-",
    )
    with pytest.raises(StrategyError, match="sum is invalid"):
        validate_strategy_sample_design_bundle(rehashed)


def test_loader_converts_parser_recursion_to_domain_error():
    raw = "[" * 20_000 + "0" + "]" * 20_000
    with pytest.raises(StrategyError, match="bounded JSON"):
        strategy_sample_design_bundle_from_json(raw)


def test_status_contract_includes_all_required_explicit_states():
    assert METRIC_OBSERVATION_STATUSES == {
        "present",
        "unavailable",
        "not_applicable",
        "not_matured",
        "insufficient_data",
    }


def test_canonical_loader_rejects_nan_even_when_python_json_accepts_it():
    bundle = _build()
    raw = json.loads(canonical_strategy_sample_design_bundle_json(bundle))
    raw["metric_observations"][0]["value"] = float("nan")
    with pytest.raises(StrategyError, match="non-finite"):
        validate_strategy_sample_design_bundle(raw)
