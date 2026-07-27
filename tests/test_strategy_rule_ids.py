from __future__ import annotations

import pandas as pd

from marvis.packs.strategy.rules import evaluate_rule_set, mine_rules


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [10, 20, 30, 40, 50, 60, 70, 80],
            "f2": [1, 1, 0, 0, 1, 0, 0, 0],
            "bad": [1, 1, 1, 0, 0, 0, 0, 0],
        }
    )


def _mine(*, top_k: int):
    return mine_rules(
        _frame(),
        feature_cols=["f1", "f2"],
        target_col="bad",
        max_depth=2,
        min_support=0.1,
        min_lift=1.2,
        top_k=top_k,
    )


def test_mined_rule_id_is_stable_when_top_k_changes() -> None:
    short = {rule.condition: rule.rule_id for rule in _mine(top_k=2)}
    long = {rule.condition: rule.rule_id for rule in _mine(top_k=20)}

    assert short
    assert set(short) <= set(long)
    assert short == {condition: long[condition] for condition in short}
    assert all(rule_id.startswith("rule-") for rule_id in short.values())
    assert not any(rule_id in {"rule_1", "rule_2"} for rule_id in short.values())


def test_mined_rule_id_excludes_observation_and_display_metadata() -> None:
    candidate = _mine(top_k=20)[0]
    condition = candidate.condition

    first = evaluate_rule_set(
        _frame(),
        [
            {
                "condition": condition,
                "support": candidate.support,
                "hit_count": candidate.hit_count,
                "lift": candidate.lift,
                "source": candidate.source,
            }
        ],
        target_col="bad",
    )
    changed_metadata = evaluate_rule_set(
        _frame(),
        [
            {
                "condition": condition,
                "support": 0.999,
                "hit_count": 999,
                "lift": 999.0,
                "source": "manual",
                "display_name": "renamed",
            }
        ],
        target_col="bad",
    )

    assert first["waterfall"][0]["rule_id"] == candidate.rule_id
    assert changed_metadata["waterfall"][0]["rule_id"] == candidate.rule_id


def test_evaluate_rule_set_prefers_input_id_and_uses_semantic_fallback() -> None:
    result = evaluate_rule_set(
        _frame(),
        [
            {"rule_id": "manually-curated", "condition": "f1 < 35"},
            {"condition": "f2 >= 1"},
        ],
        target_col="bad",
    )
    equivalent = evaluate_rule_set(
        _frame(),
        [{"condition": "  f2   >=   1  "}],
        target_col="bad",
    )

    assert result["waterfall"][0]["rule_id"] == "manually-curated"
    assert result["waterfall"][1]["rule_id"].startswith("rule-")
    assert (
        result["waterfall"][1]["rule_id"]
        == equivalent["waterfall"][0]["rule_id"]
    )


def test_waterfall_preserves_mined_ids_when_rules_are_reordered() -> None:
    candidates = _mine(top_k=20)[:2]
    assert len(candidates) == 2
    selected = [candidate.as_dict() for candidate in reversed(candidates)]

    result = evaluate_rule_set(_frame(), selected, target_col="bad")

    assert [row["rule_id"] for row in result["waterfall"]] == [
        candidate.rule_id for candidate in reversed(candidates)
    ]
