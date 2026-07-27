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
