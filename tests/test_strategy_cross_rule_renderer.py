from __future__ import annotations

from marvis.agent.renderers import render_tool_output
from marvis.packs.strategy.cross_rule_candidate import (
    build_cross_rule_candidate,
)
from marvis.packs.strategy.cross_rule_search import (
    search_cross_threshold_rules,
)
from tests.test_strategy_cross_rule_search import _request


def test_cross_rule_search_renderer_never_claims_automatic_selection() -> None:
    result = search_cross_threshold_rules(_request())

    text, tables = render_tool_output(
        "search_cross_threshold_rules",
        {
            "search_result": result,
            "evaluated": result["evaluated"],
            "not_selected": True,
        },
    )

    assert "没有自动选择" in text
    assert result["search_id"] in text
    assert tables[0]["rows"][0][1] == result["rules"][0]["rule_id"]


def test_cross_rule_candidate_renderer_preserves_lifecycle_boundary() -> None:
    result = search_cross_threshold_rules(_request())
    rule = result["rules"][0]
    candidate = build_cross_rule_candidate(
        result,
        search_artifact_ref={
            "artifact_id": "a" * 64,
            "artifact_content_hash": "b" * 64,
        },
        rule_id=rule["rule_id"],
    )

    text, tables = render_tool_output(
        "build_cross_rule_candidate_from_search",
        {
            "source_search_selection": {
                "search_id": result["search_id"],
                "rule_id": rule["rule_id"],
                "rank": rule["rank"],
                "eligible": rule["eligible"],
            },
            "candidate": candidate,
        },
    )

    assert "未独立验证、未入池、未应用、未采纳、未部署" in text
    assert tables[0]["rows"][1][1] == rule["rule_id"]
