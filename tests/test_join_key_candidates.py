"""One anchor column, several feature-column join-key candidates.

The aligner used to keep only the best feature column per anchor column, so a
feature table carrying both ``phone`` and ``phone_md5`` offered no way to pick the
other one. Every matching feature column now stays a candidate and is diagnosed as
its own switchable key; the proposed key, the anchor-first left join, the 1:1 row
assertion and the forced confirmation are unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import textwrap

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.gate_payloads import build_join_keys_payload
from marvis.app import create_app
from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import key_selection_token
from marvis.data.errors import JoinNotConfirmedError
from marvis.data.join_engine import KeySelectionError, resolve_key_selection
from tests.test_align import _write_csv
from tests.test_data_join_api import _last_assistant, _typed_gate_action_payload
from tests.test_join_engine import _engine, _write_dataset

_N = 240
_REPO = Path(__file__).resolve().parents[1]


def _run_node(script: str) -> str:
    return subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script)],
        cwd=_REPO, check=True, capture_output=True, text=True,
    ).stdout


def _md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def _frames():
    """申请表 (mobile) + 特征表 (phone raw for every row, phone_md5 for 80% of rows)."""
    phones = [f"138{i:08d}" for i in range(_N)]
    order = list(reversed(range(_N)))
    anchor = pd.DataFrame({"mobile": phones, "bad_flag": [i % 2 for i in range(_N)]})
    feature = pd.DataFrame({
        "phone": [phones[i] for i in order],
        "phone_md5": [_md5(phones[i]) if i < 192 else _md5(f"x{i}") for i in order],
        "sig1": [float(i) for i in order],
    })
    return anchor, feature


def _cols(pairs):
    return [(pair.anchor_col, pair.feature_col) for pair in pairs]


def test_aligner_keeps_every_matching_feature_column_and_offers_each_as_a_key(tmp_path):
    anchor_frame, feature_frame = _frames()
    anchor_path, anchor = _write_csv(tmp_path, "anchor.csv", anchor_frame)
    feature_path, feature = _write_csv(tmp_path, "feature.csv", feature_frame)

    proposal = ColumnAligner(DataBackend(tmp_path)).propose_keys(anchor, anchor_path, feature, feature_path)

    # the best feature column still forms the proposed key
    assert _cols(proposal.key_pairs) == [("mobile", "phone")]
    assert proposal.key_pairs[0].match_method == "exact"
    assert proposal.choice == "dictionary"
    # the second feature column of the same anchor column stays a candidate ...
    by_cols = {(pair.anchor_col, pair.feature_col): pair for pair in proposal.candidates}
    assert set(by_cols) == {("mobile", "phone"), ("mobile", "phone_md5")}
    assert by_cols[("mobile", "phone_md5")].match_method == "hash:md5"
    assert by_cols[("mobile", "phone_md5")].match_rate == pytest.approx(0.8)
    # ... and a whole key the user can switch to
    assert [_cols(key) for key in proposal.alternatives] == [[("mobile", "phone_md5")]]


def test_engine_diagnoses_each_candidate_and_switches_to_the_second_feature_column(tmp_path):
    engine, registry, repo = _engine(tmp_path)
    anchor_frame, feature_frame = _frames()
    anchor = _write_dataset(registry, tmp_path, "applications", anchor_frame)
    feature = _write_dataset(registry, tmp_path, "bureau", feature_frame)

    plan = engine.propose_join_plan(anchor.id, [feature.id], "task-1")
    spec = plan.joins[0]
    assert _cols(spec.key_pairs) == [("mobile", "phone")]
    [switch] = spec.diagnostics.key_alternatives
    assert switch.kind == "switch"
    assert switch.key_pairs == (("mobile", "phone_md5"),)
    assert switch.match_rate == pytest.approx(0.8)
    assert switch.feature_key_unique is True
    assert switch.fan_out_detected is False

    engine.select_key_pairs(
        anchor, registry.resolve_path(anchor.id), feature, registry.resolve_path(feature.id),
        spec, [key_selection_token("mobile", "phone_md5")], seed=0,
    )
    repo.update_join_spec(plan.id, spec)

    assert [(p.anchor_col, p.feature_col, p.match_method, p.resolved_by) for p in spec.key_pairs] == [
        ("mobile", "phone_md5", "hash:md5", "user"),
    ]
    assert spec.diagnostics.key_choice == "user"
    assert spec.diagnostics.match_rate == pytest.approx(0.8)
    # the replaced pair is the switch-back option, re-measured on its own
    assert [(alt.kind, alt.key_pairs, alt.match_rate) for alt in spec.diagnostics.key_alternatives] == [
        ("switch", (("mobile", "phone"),), 1.0),
    ]
    # the switched key still needs confirmation before it can execute
    assert spec.confirmed is False
    with pytest.raises(JoinNotConfirmedError):
        engine.execute_join_plan(plan.id, out_dir=tmp_path / "unconfirmed")

    engine.confirm_join_spec(plan.id, feature.id, dedup_strategy=None)
    result = engine.execute_join_plan(plan.id, out_dir=tmp_path / "joined")
    assert result.row_count == anchor.row_count == _N
    joined = pd.read_parquet(registry.resolve_path(result.id))
    assert sorted(joined["mobile"]) == sorted(anchor_frame["mobile"])
    # 80% of the anchor rows find their own feature row on the hashed phone; the rest stay
    matched = joined[joined["sig1"].notna()]
    assert len(matched) == 192
    assert (matched["sig1"] == matched["mobile"].str[3:].astype(int)).all()


def test_bare_anchor_column_keeps_its_current_pair():
    pairs = _pairs()
    current = [pairs["phone_md5"]]

    assert resolve_key_selection(list(pairs.values()), current, ["mobile"]) == [pairs["phone_md5"]]
    # without a current pair the best-matching candidate is used
    assert resolve_key_selection(list(pairs.values()), [], ["mobile"]) == [pairs["phone"]]


def test_key_selection_rejects_unknown_pairs_and_two_columns_for_one_anchor_column():
    pairs = _pairs()
    candidates = list(pairs.values())

    with pytest.raises(KeySelectionError, match="不在候选范围内") as unknown:
        resolve_key_selection(candidates, [], [key_selection_token("mobile", "sig1")])
    assert unknown.value.unknown == ("mobile=sig1",)
    with pytest.raises(KeySelectionError, match="同一主表列只能选一个"):
        resolve_key_selection(candidates, [], [
            key_selection_token("mobile", "phone"), key_selection_token("mobile", "phone_md5"),
        ])


def _pairs():
    from marvis.data.contracts import KeyPair

    return {
        "phone": KeyPair("mobile", "phone", "exact", "both", 1.0, "empirical"),
        "phone_md5": KeyPair("mobile", "phone_md5", "hash:md5", "anchor", 0.8, "empirical"),
    }


def _propose_output(selected: str = "phone") -> dict:
    pairs = {
        "phone": {"anchor_col": "mobile", "feature_col": "phone", "match_method": "exact", "match_rate": 1.0},
        "phone_md5": {"anchor_col": "mobile", "feature_col": "phone_md5", "match_method": "hash:md5",
                      "match_rate": 0.8},
        "applydt": {"anchor_col": "applydt", "feature_col": "report_date", "match_method": "date_norm",
                    "match_rate": 1.0},
    }
    other = "phone_md5" if selected == "phone" else "phone"
    return {
        "join_plan_id": "plan-1",
        "joins": [{
            "feature_id": "ds-bureau",
            "feature_name": "bureau.csv",
            "key_pairs": [pairs["applydt"], pairs[selected]],
            "diagnostics": {
                "match_rate": 1.0, "feature_key_unique": True, "fan_out_detected": False,
                "key_candidates": [pairs["applydt"], pairs["phone"], pairs["phone_md5"]],
                "key_alternatives": [{
                    "key_pairs": [["applydt", "report_date"], ["mobile", pairs[other]["feature_col"]]],
                    "dropped": "", "match_rate": 0.8, "feature_key_unique": True,
                    "fan_out_detected": False, "kind": "switch",
                }],
            },
        }],
    }


def test_key_picker_lists_each_feature_column_of_one_anchor_column_as_its_own_option():
    [feature] = build_join_keys_payload(_propose_output())["features"]

    options = [(o["anchor_col"], o["feature_col"], o["value"]) for o in feature["key_options"]]
    assert options == [
        ("applydt", "report_date", "applydt"),
        ("mobile", "phone", "mobile=phone"),
        ("mobile", "phone_md5", "mobile=phone_md5"),
    ]
    assert feature["selected_anchor_cols"] == ["applydt", "mobile"]
    assert feature["selected_values"] == ["applydt", "mobile=phone"]
    [alternative] = feature["alternatives"]
    assert alternative["values"] == ["applydt", "mobile=phone_md5"]


def test_key_picker_keeps_bare_anchor_values_when_each_anchor_column_has_one_candidate():
    output = _propose_output()
    diagnostics = output["joins"][0]["diagnostics"]
    diagnostics["key_candidates"] = diagnostics["key_candidates"][:2]
    diagnostics["key_alternatives"][0]["key_pairs"] = [["applydt", "report_date"]]

    [feature] = build_join_keys_payload(output)["features"]
    assert [o["value"] for o in feature["key_options"]] == ["applydt", "mobile"]
    assert feature["selected_values"] == feature["selected_anchor_cols"] == ["applydt", "mobile"]
    assert feature["alternatives"][0]["values"] == ["applydt"]


def _join_dir(root: Path) -> Path:
    src = root / "join_multi"
    src.mkdir(parents=True, exist_ok=True)
    anchor, feature = _frames()
    anchor.to_csv(src / "applications.csv", index=False)
    feature.to_csv(src / "bureau_features.csv", index=False)
    return src


def test_c2_gate_switches_to_the_second_feature_column_and_executes_one_to_one(tmp_path):
    client = TestClient(create_app(tmp_path))
    task_id = client.post("/api/tasks", json={
        "model_name": "多候选拼接键", "validator": "qa", "source_dir": str(_join_dir(tmp_path)),
        "task_type": "data_join", "run_mode": "manual",
    }).json()["id"]
    client.post(f"/api/tasks/{task_id}/agent/start", json={})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认", "ui_action": "confirm_roles"})
    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "开始"})

    gate = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    [feature] = gate["metadata"]["join_keys"]["features"]
    assert feature["selected_values"] == ["mobile=phone"]
    assert [o["value"] for o in feature["key_options"]] == ["mobile=phone", "mobile=phone_md5"]

    switched = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json=_typed_gate_action_payload(
            gate, action="apply_join_keys", content="重新诊断拼接键",
            adjust_params={"key_overrides": {feature["feature_id"]: ["mobile=phone_md5"]}},
        ),
    )
    assert switched.status_code == 202, switched.text
    gate2 = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "已按你的选择改用 mobile = phone_md5" in gate2["content"]
    [feature2] = gate2["metadata"]["join_keys"]["features"]
    assert feature2["selected_values"] == ["mobile=phone_md5"]
    assert [alt["values"] for alt in feature2["alternatives"]] == [["mobile=phone"]]

    client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": "确认"})
    done = _last_assistant(client.get(f"/api/tasks/{task_id}/agent/messages").json()["messages"])
    assert "拼接执行完成" in done["content"]
    assert "1:1 保持" in done["content"]
    plan_repo = client.app.state.plan_repo
    plan = plan_repo.load_plan(done["metadata"]["result_dataset"]["plan_id"])
    steps = {step.tool_ref.tool: step for step in plan.steps}
    executed = plan_repo.load_step_output(steps["execute_join"].id)
    assert executed["anchor_rows"] == executed["joined_rows"] == _N
    proposed = plan_repo.load_step_output(steps["propose_join"].id)
    assert [(p["anchor_col"], p["feature_col"]) for p in proposed["joins"][0]["key_pairs"]] == [
        ("mobile", "phone_md5"),
    ]


def test_key_picker_submits_pair_values_and_one_feature_column_per_anchor_column():
    output = _run_node(
        """
        import assert from "node:assert/strict";
        import { renderJoinKeyPicker, handleJoinKeyOptionChange } from "./marvis/static/js/v2/join_gate_controller.js";
        const message = { id: "m1", metadata: { step_id: "gate-1", join_keys: { features: [{
          feature_id: "f1", feature_name: "bureau.csv", current_match_rate: 1,
          key_options: [
            { anchor_col: "mobile", feature_col: "phone", value: "mobile=phone", match_rate: 1 },
            { anchor_col: "mobile", feature_col: "phone_md5", value: "mobile=phone_md5", match_rate: 0.8 },
          ],
          selected_anchor_cols: ["mobile"], selected_values: ["mobile=phone"],
          alternatives: [{ anchor_cols: ["mobile"], key_pairs: [["mobile", "phone_md5"]], values: ["mobile=phone_md5"],
                           match_rate: 0.8, feature_key_unique: true, fan_out_detected: false, kind: "switch" }],
        }] } } };
        const html = renderJoinKeyPicker(message);
        assert.equal(html.includes('value="mobile=phone" checked'), true);
        assert.equal(html.includes('value="mobile=phone_md5">'), true);
        assert.equal(html.includes('data-join-key-anchor="mobile"'), true);
        assert.equal(html.includes('data-join-key-columns="[&quot;mobile=phone_md5&quot;]"'), true);
        assert.equal(html.includes('data-join-key-current="[&quot;mobile=phone&quot;]"'), true);

        const attrs = (feature, anchor) => ({ "data-join-key-feature": feature, "data-join-key-anchor": anchor });
        const inputs = [];
        const form = { dataset: {}, querySelectorAll: (sel) => (sel === "[data-join-key-feature]" ? inputs : []), closest: () => null, parentElement: null };
        const make = (value, anchor, checked) => {
          const a = attrs("f1", anchor);
          const input = { value, checked, getAttribute: (k) => a[k], closest: (sel) => (sel === ".join-key-picker" ? form : input) };
          inputs.push(input);
          return input;
        };
        const phone = make("mobile=phone", "mobile", true);
        const hashed = make("mobile=phone_md5", "mobile", true);
        handleJoinKeyOptionChange({ target: hashed });
        assert.equal(phone.checked, false);
        assert.equal(hashed.checked, true);
        process.stdout.write("ok");
        """
    )
    assert output == "ok"
