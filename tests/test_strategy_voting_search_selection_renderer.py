"""Agent projection for a Voting candidate built from exact search pointers."""

from __future__ import annotations

from copy import deepcopy

from marvis.agent.renderers import render_tool_output
from tests.test_strategy_voting_renderer import _output as _voting_candidate_output


SEARCH_ID = "voting-search-" + "a" * 32
COMBO_ID = "voting-combo-" + "b" * 32


def _trusted_inputs() -> dict:
    return {
        "search_id": SEARCH_ID,
        "combo_id": COMBO_ID,
        "strategy_type": "approval",
    }


def _output(*, eligible: bool) -> dict:
    candidate = _voting_candidate_output()
    member_rule_ids = [
        "candidate-rule-" + "c" * 32,
        "candidate-rule-" + "d" * 32,
        "candidate-rule-" + "e" * 32,
    ]
    candidate.update(
        {
            "schema_version": "strategy.build-voting-candidate-tool.v2",
            "sample_design_ref": {
                "artifact_id": "8" * 64,
                "artifact_content_hash": "9" * 64,
                "sample_design_id": "sample-design-renderer",
                "sample_design_content_hash": "a" * 64,
                "partition": "development",
            },
        }
    )
    candidate["effect"] = {
        "schema_version": "strategy.voting-effect.v1",
        "effect_id": candidate["effect_id"],
        "effect_hash": candidate["effect_hash"],
        **candidate["effect"],
    }
    candidate["metrics"] = {
        "schema_version": "strategy.voting-metrics.v1",
        "metrics_hash": candidate["metrics"]["metrics_hash"],
        **{
            field: candidate["effect"][field]
            for field in (
                "population_count",
                "labeled_count",
                "matched_count",
                "matched_rate",
                "matched_bad_count",
                "matched_bad_rate",
                "unmatched_count",
                "unmatched_bad_count",
                "unmatched_bad_rate",
                "bad_capture_rate",
                "lift",
            )
        },
    }
    while len(candidate["metric_observations"]) < 27:
        candidate["metric_observations"].append(
            deepcopy(candidate["metric_observations"][-1])
        )
    for item, rule_id in zip(candidate["selected_entries"], member_rule_ids):
        item["rule_id"] = rule_id
    return {
        "schema_version": "strategy.build-voting-candidate-from-search-tool.v1",
        "source_search_selection": {
            "search_id": SEARCH_ID,
            "combo_id": COMBO_ID,
            "strategy_type": "approval",
            "rank": 3,
            "member_rule_ids": member_rule_ids,
            "n": candidate["n"],
            "eligible": eligible,
            "constraint_failures": (
                []
                if eligible
                else [
                    {
                        "metric": "hit_share",
                        "operator": "gte",
                        "threshold": 0.2,
                        "actual": 0.1,
                    }
                ]
            ),
        },
        "voting_candidate": candidate,
        "not_mutated_pool": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def test_voting_search_selection_renderer_shows_exact_source_without_winner_label() -> None:
    output = _output(eligible=True)

    text, _tables = render_tool_output(
        "build_voting_candidate_from_search",
        output,
        trusted_inputs=_trusted_inputs(),
    )

    assert SEARCH_ID in text
    assert COMBO_ID in text
    assert output["voting_candidate"]["asset_id"] in text
    assert "精确点名" in text
    assert "满足搜索时的资格约束" in text
    assert "winner" not in text.casefold()
    assert "champion" not in text.casefold()
    assert "冠军" not in text
    assert "不代表平台自动选择或推荐" in text
    assert "尚未入池" in text


def test_voting_search_selection_renderer_warns_when_constraints_failed() -> None:
    text, _tables = render_tool_output(
        "build_voting_candidate_from_search",
        _output(eligible=False),
        trusted_inputs=_trusted_inputs(),
    )

    assert "未满足搜索时的资格约束" in text
    assert "hit_share gte 0.2" in text
    assert "actual 0.1" in text
    assert "不代表平台推荐" in text
    assert "winner" not in text.casefold()
    assert "champion" not in text.casefold()


def test_voting_search_selection_renderer_rejects_untrusted_source_claims() -> None:
    output = _output(eligible=True)
    output["source_search_selection"]["winner"] = True

    text, tables = render_tool_output(
        "build_voting_candidate_from_search",
        output,
        trusted_inputs=_trusted_inputs(),
    )

    assert "完整性校验失败" in text
    assert SEARCH_ID not in text
    assert COMBO_ID not in text
    assert tables == []


def test_voting_search_selection_renderer_rejects_malformed_nested_candidate() -> None:
    output = _output(eligible=True)
    del output["voting_candidate"]["evidence_hash"]

    text, tables = render_tool_output(
        "build_voting_candidate_from_search",
        output,
        trusted_inputs=_trusted_inputs(),
    )

    assert "完整性校验失败" in text
    assert output["voting_candidate"]["asset_id"] not in text
    assert tables == []


def test_voting_search_selection_renderer_rejects_pointer_candidate_mismatch() -> None:
    for mutation in ("n", "members", "lifecycle"):
        output = deepcopy(_output(eligible=True))
        if mutation == "n":
            output["source_search_selection"]["n"] = 1
        elif mutation == "members":
            output["source_search_selection"]["member_rule_ids"][0] = (
                "candidate-rule-" + "f" * 32
            )
        else:
            output["voting_candidate"]["not_admitted"] = False

        text, tables = render_tool_output(
            "build_voting_candidate_from_search",
            output,
            trusted_inputs=_trusted_inputs(),
        )

        assert "完整性校验失败" in text
        assert tables == []


def test_voting_search_selection_renderer_binds_pointer_to_trusted_inputs() -> None:
    output = _output(eligible=True)
    output["source_search_selection"]["search_id"] = (
        "voting-search-" + "1" * 32
    )
    output["source_search_selection"]["combo_id"] = "voting-combo-" + "2" * 32

    text, tables = render_tool_output(
        "build_voting_candidate_from_search",
        output,
        trusted_inputs=_trusted_inputs(),
    )

    assert "完整性校验失败" in text
    assert output["source_search_selection"]["search_id"] not in text
    assert output["source_search_selection"]["combo_id"] not in text
    assert tables == []
