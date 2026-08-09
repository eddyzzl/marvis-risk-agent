from __future__ import annotations

import json

from marvis.packs.strategy.cross_rule_search import (
    search_cross_threshold_rules,
)
from marvis.packs.strategy.report_bundle_adapters import (
    _cross_rule_search_report_projection,
)
from tests.test_strategy_cross_rule_search import _request


def test_cross_rule_report_projection_is_aggregate_and_never_selects() -> None:
    result = search_cross_threshold_rules(_request())
    source_ref = {
        "kind": "cross_rule_search",
        "ref_id": "a" * 64,
        "content_hash": "b" * 64,
    }

    projected = _cross_rule_search_report_projection(
        result,
        source_ref=source_ref,
    )

    table = projected["table"]
    assert table["table_id"] == "cross_threshold_rule_search"
    assert table["sheet_key"] == "appendix_cross_rules"
    assert table["effect_stage"] == "backtested"
    assert len(table["rows"]) == result["evaluated"]
    assert table["rows"][0]["row_id"] == result["rules"][0]["rule_id"]
    rendered = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    assert "winner" not in rendered
    assert "champion" not in rendered
    assert "selected_rule" not in rendered
    assert "Cross阈值规则已评估数" in rendered


def test_cross_rule_report_projection_marks_missing_amount_lift_not_applicable() -> None:
    request = _request()
    request["population"]["loan_amount_sum"] = None
    request["population"]["overdue_amount_sum"] = None
    request["constraints"]["min_amount_lift"] = None
    for trial in request["trials"]:
        trial["loan_amount_sum"] = None
        trial["overdue_amount_sum"] = None
    result = search_cross_threshold_rules(request)
    source_ref = {
        "kind": "cross_rule_search",
        "ref_id": "a" * 64,
        "content_hash": "b" * 64,
    }

    projected = _cross_rule_search_report_projection(
        result,
        source_ref=source_ref,
    )

    for row in projected["table"]["rows"]:
        amount_lift = row["cells"]["amount_lift"]
        assert amount_lift["availability"] == "not_applicable"
        assert amount_lift["value"] is None
        assert amount_lift["source_refs"] == []
