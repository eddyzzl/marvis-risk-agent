from __future__ import annotations

import json
import types
from copy import deepcopy
import hashlib

import duckdb
import pandas as pd
import pytest

from marvis.packs.strategy.dsl_delivery import (
    StrategyDeliveryError,
    generate_strategy_duckdb_sql_source,
    generate_strategy_python_source,
    validate_strategy_delivery_equivalence,
    validate_strategy_duckdb_input_frame,
    verify_strategy_delivery_equivalence,
)


def _load_generated_python(source: str) -> types.ModuleType:
    module = types.ModuleType("generated_strategy_delivery")
    exec(compile(source, "<generated-strategy>", "exec"), module.__dict__)
    return module


def _rehash_equivalence(evidence: dict) -> dict:
    body = {
        key: value
        for key, value in evidence.items()
        if key not in {"equivalence_id", "content_hash"}
    }
    body_hash = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence["equivalence_id"] = (
        "strategy-dsl-equivalence-" + body_hash[:24]
    )
    without_hash = {
        key: value for key, value in evidence.items() if key != "content_hash"
    }
    evidence["content_hash"] = hashlib.sha256(
        json.dumps(
            without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return evidence


def _approval_spec() -> dict:
    return {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "approval",
        "match_policy": "first_match",
        "default_action": {
            "type": "approval",
            "value": "approve",
            "reason_code": "DEFAULT",
            "stop": True,
        },
        "rules": [
            {
                "rule_id": "risk-review",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "risk_score",
                    "operator": ">=",
                    "value": 600,
                    "missing": "no_match",
                },
                "action": {
                    "type": "review",
                    "value": "review",
                    "reason_code": "HIGH_RISK_REVIEW",
                    "stop": True,
                },
            },
            {
                "rule_id": "risk-reject",
                "priority": 20,
                "condition": {
                    "op": "compare",
                    "field": "risk_score",
                    "operator": ">=",
                    "value": 700,
                    "missing": "no_match",
                },
                "action": {
                    "type": "reject",
                    "value": "reject",
                    "reason_code": "VERY_HIGH_RISK",
                    "stop": True,
                },
            },
        ],
        "metadata": {"lineage": {"source": "test"}},
    }


def test_generated_python_applies_generic_strategy_dsl_with_first_match_actions() -> None:
    generated = _load_generated_python(
        generate_strategy_python_source(_approval_spec())
    )

    assert generated.apply_rows(
        pd.DataFrame({"risk_score": [500, 650, 750, None]})
    ) == [
        {
            "matched_rule_id": None,
            "action_type": "approval",
            "action_value": "approve",
            "decision": "approve",
            "reason_code": "DEFAULT",
        },
        {
            "matched_rule_id": "risk-review",
            "action_type": "review",
            "action_value": "review",
            "decision": "review",
            "reason_code": "HIGH_RISK_REVIEW",
        },
        {
            "matched_rule_id": "risk-review",
            "action_type": "review",
            "action_value": "review",
            "decision": "review",
            "reason_code": "HIGH_RISK_REVIEW",
        },
        {
            "matched_rule_id": None,
            "action_type": "approval",
            "action_value": "approve",
            "decision": "approve",
            "reason_code": "DEFAULT",
        },
    ]


def test_generated_duckdb_sql_applies_the_same_first_match_action_contract() -> None:
    frame = pd.DataFrame({"risk_score": [500, 650, 750, None]})
    assert validate_strategy_duckdb_input_frame(
        frame,
        _approval_spec(),
    ) is frame

    with duckdb.connect() as connection:
        connection.register("input_rows", frame)
        rows = connection.sql(
            generate_strategy_duckdb_sql_source(_approval_spec())
        ).fetchall()

    assert [
        {
            "row": row[0],
            "matched_rule_id": row[1],
            "action_type": row[2],
            "action_value": json.loads(row[3]),
            "decision": json.loads(row[4]),
            "reason_code": row[5],
        }
        for row in rows
    ] == [
        {
            "row": 0,
            "matched_rule_id": None,
            "action_type": "approval",
            "action_value": "approve",
            "decision": "approve",
            "reason_code": "DEFAULT",
        },
        {
            "row": 1,
            "matched_rule_id": "risk-review",
            "action_type": "review",
            "action_value": "review",
            "decision": "review",
            "reason_code": "HIGH_RISK_REVIEW",
        },
        {
            "row": 2,
            "matched_rule_id": "risk-review",
            "action_type": "review",
            "action_value": "review",
            "decision": "review",
            "reason_code": "HIGH_RISK_REVIEW",
        },
        {
            "row": 3,
            "matched_rule_id": None,
            "action_type": "approval",
            "action_value": "approve",
            "decision": "approve",
            "reason_code": "DEFAULT",
        },
    ]


def test_delivery_equivalence_is_content_addressed_and_reconciles_three_engines() -> None:
    frame = pd.DataFrame({"risk_score": [500, 650, 750, None]})

    first = verify_strategy_delivery_equivalence(_approval_spec(), frame)
    second = verify_strategy_delivery_equivalence(_approval_spec(), frame)

    assert first == second
    assert first["schema_version"] == "strategy.dsl-delivery-equivalence.v1"
    assert first["source_row_count"] == 4
    assert first["sample_count"] == 4
    assert first["matched"] is True
    assert first["engines"] == ["marvis_evaluator", "python", "duckdb_sql"]
    assert set(first["result_hashes"]) == {
        "marvis_evaluator",
        "python",
        "duckdb_sql",
    }
    assert len(set(first["result_hashes"].values())) == 1
    assert first["equivalence_id"].startswith("strategy-dsl-equivalence-")
    assert len(first["content_hash"]) == 64


def test_default_only_typed_strategy_is_deliverable_without_dummy_columns() -> None:
    spec = {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "limit",
        "match_policy": "first_match",
        "default_action": {
            "type": "limit",
            "value": 12000,
            "output_value": 11800,
            "reason_code": "BASE_LIMIT",
            "stop": True,
        },
        "rules": [],
        "metadata": {"lineage": {}},
    }
    frame = pd.DataFrame({"unused": ["a", "b"]})

    evidence = verify_strategy_delivery_equivalence(spec, frame)

    assert evidence["matched"] is True
    generated = _load_generated_python(generate_strategy_python_source(spec))
    assert generated.apply_rows(frame) == [
        {
            "matched_rule_id": None,
            "action_type": "limit",
            "action_value": 12000,
            "decision": 11800,
            "reason_code": "BASE_LIMIT",
        },
        {
            "matched_rule_id": None,
            "action_type": "limit",
            "action_value": 12000,
            "decision": 11800,
            "reason_code": "BASE_LIMIT",
        },
    ]


def test_equivalence_validator_rejects_reauthored_or_incomplete_evidence() -> None:
    evidence = verify_strategy_delivery_equivalence(
        _approval_spec(),
        pd.DataFrame({"risk_score": [500, 750]}),
    )
    trusted = {
        "expected_strategy_spec_hash": evidence["strategy_spec_hash"],
        "expected_sample_hash": evidence["sample_hash"],
        "expected_content_hash": evidence["content_hash"],
    }

    assert validate_strategy_delivery_equivalence(evidence, **trusted) == evidence
    reauthored = deepcopy(evidence)
    reauthored["sample_count"] = 1
    with pytest.raises(StrategyDeliveryError, match="content_hash"):
        validate_strategy_delivery_equivalence(reauthored, **trusted)
    incomplete = deepcopy(evidence)
    del incomplete["result_hashes"]["python"]
    with pytest.raises(StrategyDeliveryError, match="result_hashes"):
        validate_strategy_delivery_equivalence(incomplete, **trusted)
    over_budget = deepcopy(evidence)
    over_budget["source_row_count"] = 4097
    over_budget["sample_count"] = 4097
    _rehash_equivalence(over_budget)
    with pytest.raises(StrategyDeliveryError, match="sample_count.*budget"):
        validate_strategy_delivery_equivalence(over_budget, **trusted)

    fabricated = deepcopy(evidence)
    fabricated["strategy_spec_hash"] = "0" * 64
    fabricated["sample_hash"] = "0" * 64
    fabricated["result_hashes"] = {
        engine: "0" * 64 for engine in fabricated["engines"]
    }
    _rehash_equivalence(fabricated)
    with pytest.raises(StrategyDeliveryError, match="trusted binding"):
        validate_strategy_delivery_equivalence(fabricated, **trusted)


def test_delivery_preflight_rejects_unsafe_auto_numeric_text() -> None:
    spec = {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "approval",
        "match_policy": "first_match",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "unsafe-double",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "x",
                    "operator": "==",
                    "value": 9007199254740992,
                    "missing": "no_match",
                },
                "action": {"type": "reject"},
            }
        ],
        "metadata": {"lineage": {}},
    }
    frame = pd.DataFrame(
        {"x": ["0"] * 4096 + ["9007199254740993"]}
    )

    with pytest.raises(StrategyDeliveryError, match="exact comparison range"):
        verify_strategy_delivery_equivalence(spec, frame)


def test_bounded_equivalence_sample_always_binds_the_final_source_row() -> None:
    spec = _approval_spec()
    baseline = pd.DataFrame({"risk_score": [0] * 4097})
    changed_tail = baseline.copy()
    changed_tail.loc[4096, "risk_score"] = 750

    first = verify_strategy_delivery_equivalence(spec, baseline)
    second = verify_strategy_delivery_equivalence(spec, changed_tail)

    assert first["bounded"] is True
    assert first["sample_count"] == 4096
    assert first["sample_hash"] != second["sample_hash"]


def test_auto_not_equal_null_has_row_and_frame_delivery_parity() -> None:
    spec = {
        "schema_version": "strategy.dsl.v1",
        "strategy_type": "approval",
        "match_policy": "first_match",
        "default_action": {"type": "approval"},
        "rules": [
            {
                "rule_id": "present",
                "priority": 10,
                "condition": {
                    "op": "compare",
                    "field": "x",
                    "operator": "!=",
                    "value": None,
                    "missing": "no_match",
                },
                "action": {"type": "reject"},
            }
        ],
        "metadata": {"lineage": {}},
    }

    evidence = verify_strategy_delivery_equivalence(
        spec,
        pd.DataFrame(
            {"x": pd.Series(["present", None], dtype="string")}
        ),
    )

    assert evidence["matched"] is True
