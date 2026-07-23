from marvis.agent.adjust_specs import adjust_param_error, has_join_key_adjust
from marvis.agent.gate_payloads import build_join_keys_payload


def test_join_key_payload_nests_alternatives_under_each_feature_table():
    payload = build_join_keys_payload({
        "join_plan_id": "plan-1",
        "joins": [
            {
                "feature_id": "feature-a",
                "feature_name": "vars_a.parquet",
                "key_pairs": [
                    {"anchor_col": "applydt", "feature_col": "huisudate", "match_method": "exact"},
                    {"anchor_col": "phone", "feature_col": "mobile", "match_method": "exact"},
                    {"anchor_col": "usedate", "feature_col": "huisudate", "match_method": "exact"},
                ],
                "diagnostics": {
                    "match_rate": 0.0,
                    "feature_key_unique": False,
                    "fan_out_detected": False,
                    "key_alternatives": [
                        {"key_pairs": [["applydt", "huisudate"], ["phone", "mobile"]], "dropped": "usedate", "match_rate": 1.0, "feature_key_unique": False, "fan_out_detected": True},
                        {"key_pairs": [["phone", "mobile"], ["usedate", "huisudate"]], "dropped": "applydt", "match_rate": 0.2857, "feature_key_unique": True, "fan_out_detected": False},
                    ],
                },
            },
            {
                "feature_id": "feature-b",
                "feature_name": "vars_b.parquet",
                "key_pairs": [{"anchor_col": "phone", "feature_col": "mobile_md5", "match_method": "hash:md5"}],
                "diagnostics": {"match_rate": 1.0, "key_alternatives": []},
            },
        ],
    })

    assert payload is not None
    assert len(payload["features"]) == 2
    assert len(payload["features"][0]["alternatives"]) == 2
    assert payload["features"][0]["alternatives"][0]["anchor_cols"] == ["applydt", "phone"]
    assert payload["features"][1]["feature_name"] == "vars_b.parquet"


def test_join_key_adjust_requires_nonempty_unique_keys_per_feature():
    assert has_join_key_adjust({"key_overrides": {"feature-a": ["phone"]}})
    assert adjust_param_error({"key_overrides": {"feature-a": ["phone"]}}) is None
    assert "至少选择一个" in adjust_param_error({"key_overrides": {"feature-a": []}})
    assert "不能重复" in adjust_param_error({"key_overrides": {"feature-a": ["phone", "phone"]}})
