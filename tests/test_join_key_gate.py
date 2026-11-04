import json
import subprocess
import textwrap
from pathlib import Path

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


def _s6_propose_output(selected: str = "cust_id") -> dict:
    """propose_join output for eval S6 (cust_id preferred, hashed phone switchable)."""
    pairs = {
        "cust_id": {"anchor_col": "cust_id", "feature_col": "cust_id", "match_method": "exact",
                    "transform_side": "both", "match_rate": 1.0},
        "mobile": {"anchor_col": "mobile", "feature_col": "mobile_md5", "match_method": "hash:md5",
                   "transform_side": "anchor", "match_rate": 1.0},
    }
    other = "mobile" if selected == "cust_id" else "cust_id"
    return {
        "join_plan_id": "plan-s6",
        "joins": [{
            "feature_id": "ds-bureau",
            "feature_name": "bureau_features.csv",
            "key_pairs": [pairs[selected]],
            "diagnostics": {
                "match_rate": 1.0,
                "feature_key_unique": True,
                "fan_out_detected": False,
                "key_choice": "customer_id" if selected == "cust_id" else "user",
                "key_candidates": [pairs["cust_id"], pairs["mobile"]],
                "key_alternatives": [{
                    "key_pairs": [[pairs[other]["anchor_col"], pairs[other]["feature_col"]]],
                    "dropped": selected,
                    "match_rate": 1.0,
                    "feature_key_unique": True,
                    "fan_out_detected": False,
                    "kind": "switch",
                }],
            },
        }],
    }


def test_join_key_payload_offers_customer_id_and_hashed_phone_as_switchable_options():
    payload = build_join_keys_payload(_s6_propose_output())

    assert payload is not None
    [feature] = payload["features"]
    assert feature["selected_anchor_cols"] == ["cust_id"]
    assert [(option["anchor_col"], option["feature_col"]) for option in feature["key_options"]] == [
        ("cust_id", "cust_id"), ("mobile", "mobile_md5"),
    ]
    [alternative] = feature["alternatives"]
    assert alternative["kind"] == "switch"
    assert alternative["anchor_cols"] == ["mobile"]
    assert alternative["feature_key_unique"] is True

    # after switching to the phone, cust_id is the switch-back option
    switched = build_join_keys_payload(_s6_propose_output(selected="mobile"))["features"][0]
    assert switched["selected_anchor_cols"] == ["mobile"]
    assert switched["alternatives"][0]["anchor_cols"] == ["cust_id"]
    assert {option["anchor_col"] for option in switched["key_options"]} == {"cust_id", "mobile"}


def test_join_key_reason_explains_customer_id_choice_and_hashed_switch_in_business_chinese():
    from marvis.agent.renderers import _render_propose_join
    from marvis.packs.data_ops.tools import _key_choice_reason

    output = _s6_propose_output()
    reason = _key_choice_reason(output["joins"][0])
    assert "选用客户号 cust_id = cust_id" in reason
    assert "两表都有客户号" in reason and "原值完全一致" in reason
    assert "命中率 100.00%" in reason and "键唯一" in reason
    assert "其他候选键：mobile = mobile_md5" in reason

    switched = _key_choice_reason(_s6_propose_output(selected="mobile")["joins"][0])
    assert "已按你的选择改用 mobile = mobile_md5" in switched
    assert "原文对 MD5 哈希" in switched and "主表 mobile 为原文" in switched

    output["joins"][0]["key_choice_reason"] = reason
    text, _tables = _render_propose_join(output)
    assert "**拼接键选择**" in text
    assert reason in text


def test_join_key_reason_explains_hashed_phone_when_there_is_no_customer_id():
    from marvis.packs.data_ops.tools import _key_choice_reason

    reason = _key_choice_reason({
        "feature_name": "features.parquet",
        "key_pairs": [{"anchor_col": "mobile", "feature_col": "phone_md5", "match_method": "hash:md5",
                       "transform_side": "anchor", "match_rate": 1.0}],
        "diagnostics": {"match_rate": 1.0, "feature_key_unique": True, "key_choice": "dictionary"},
    })

    assert "两表没有可直接对上的客户号" in reason
    assert "原文对 MD5 哈希" in reason
    # legacy plans (no key_choice) get no reason line rather than a guessed one
    assert _key_choice_reason({"key_pairs": [{"anchor_col": "a", "feature_col": "b"}], "diagnostics": {}}) == ""


def test_join_key_picker_renders_every_candidate_with_current_key_checked():
    payload = build_join_keys_payload(_s6_propose_output())
    script = textwrap.dedent(f"""
        import assert from "node:assert/strict";
        import {{ renderJoinKeyPicker }} from "./marvis/static/js/v2/join_gate_controller.js";
        const html = renderJoinKeyPicker({{ id: "m1", metadata: {{ step_id: "gate-1", join_keys: {json.dumps(payload)} }} }});
        assert.match(html, /value="cust_id" checked/);
        assert.match(html, /value="mobile"(?! checked)/);
        assert.equal(html.includes("改用 mobile=mobile_md5"), true);
        assert.equal(html.includes('data-join-key-columns="[&quot;mobile&quot;]"'), true);
        process.stdout.write("ok");
    """)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "ok"


def _run_node(script: str) -> str:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_gate_with_key_choice_and_dedup_keeps_the_dedup_confirm():
    # Review H1: a non-unique key with a switch alternative carried both
    # join_keys and dedup; only the key picker rendered, so manual mode had
    # no way to pick a dedup strategy or confirm.
    from marvis.agent.gate_payloads import build_dedup_payload

    propose = _s6_propose_output()
    payload = build_join_keys_payload(propose)
    feature_id = propose["joins"][0]["feature_id"]
    dedup = build_dedup_payload({"needs_dedup": [feature_id]}, propose)
    assert dedup and dedup["features"]
    assert _run_node(f"""
        import assert from "node:assert/strict";
        import {{ driverManualAnalysisHtml, driverGateBodyHtml }} from "./marvis/static/js/v2/driver_manual_analysis.js";
        import {{ renderJoinKeyPicker, renderDedupPicker }} from "./marvis/static/js/v2/join_gate_controller.js";
        import {{ renderDriverGateButton }} from "./marvis/static/js/v2/driver_gate_confirm.js";
        const message = {{ id: "m1", role: "assistant", content: "诊断完成", metadata: {{
          kind: "gate", plan_id: "p", step_id: "gate-1", join_keys: {json.dumps(payload)}, dedup: {json.dumps(dedup)} }} }};
        const renderers = {{ renderJoinKeyPicker, renderDedupPicker, renderTables: () => "",
          renderAgentMarkdown: (v) => String(v || ""), renderGateConfirm: (m) => renderDriverGateButton(m) }};
        for (const html of [driverManualAnalysisHtml([message], renderers), driverGateBodyHtml(message, renderers)]) {{
          assert.equal(html.includes('class="join-key-picker"'), true);
          assert.equal(html.includes("data-dedup-confirm="), true);
          // The dedup picker owns the confirm; no second plain confirm.
          assert.equal(html.includes("data-driver-confirm"), false);
        }}
        // Without dedup the key-only gate keeps the plain confirm (eval S6).
        const keysOnly = {{ ...message, metadata: {{ ...message.metadata, dedup: undefined }} }};
        assert.equal(driverManualAnalysisHtml([keysOnly], renderers).includes("data-driver-confirm"), true);
        process.stdout.write("ok");
    """) == "ok"


def test_confirm_waits_while_a_ticked_key_is_not_applied():
    # Review M1: ticking another key and pressing 「确认并执行拼接」 ran the
    # diagnosed key instead of the ticked one.
    assert _run_node("""
        import assert from "node:assert/strict";
        import { joinKeySelectionPending, syncJoinKeyConfirmState } from "./marvis/static/js/v2/join_gate_controller.js";
        const inputs = [{ value: "cust_id", checked: true }, { value: "mobile", checked: false }];
        const card = {
          getAttribute: (name) => name === "data-join-key-current" ? '["cust_id"]' : null,
          querySelectorAll: (selector) => selector.endsWith(":checked") ? inputs.filter((i) => i.checked) : inputs,
        };
        const confirm = { disabled: false, dataset: {}, title: "", removeAttribute(name) { if (name === "title") this.title = ""; } };
        const gate = { querySelectorAll: (selector) => selector === "[data-driver-confirm]" ? [confirm] : [] };
        const form = { dataset: {}, querySelectorAll: () => [card], closest: () => gate };
        assert.equal(joinKeySelectionPending(form), false);
        inputs[0].checked = false; inputs[1].checked = true;
        assert.equal(syncJoinKeyConfirmState(form), true);
        assert.equal(confirm.disabled, true);
        assert.equal(form.dataset.joinKeyPending, "true");
        assert.match(confirm.title, /应用拼接键并重新诊断/);
        inputs[0].checked = true; inputs[1].checked = false;
        assert.equal(syncJoinKeyConfirmState(form), false);
        assert.equal(confirm.disabled, false);
        assert.equal(form.dataset.joinKeyPending, "false");
        process.stdout.write("ok");
    """) == "ok"


def test_lower_match_reason_compares_with_the_running_key():
    # Review N1: the reason named a date pair as the rival; it now compares the
    # customer id with the diagnosed rate of the key that runs.
    from marvis.packs.data_ops.tools import _key_choice_reason

    reason = _key_choice_reason({
        "feature_name": "bureau.csv",
        "key_pairs": [
            {"anchor_col": "apply_date", "feature_col": "apply_date", "match_method": "exact", "match_rate": 0.9},
            {"anchor_col": "mobile", "feature_col": "mobile_md5", "match_method": "hash:md5",
             "transform_side": "anchor", "match_rate": 1.0},
        ],
        "diagnostics": {
            "match_rate": 0.86, "feature_key_unique": True, "key_choice": "customer_id_lower_match",
            "key_candidates": [
                {"anchor_col": "cust_id", "feature_col": "cust_id", "match_rate": 0.7},
                {"anchor_col": "apply_date", "feature_col": "apply_date", "match_rate": 0.9},
            ],
        },
    })

    assert "客户号 cust_id = cust_id 抽样命中率 70.00%，低于当前键的 86.00%" in reason
    assert "apply_date = apply_date 的" not in reason
