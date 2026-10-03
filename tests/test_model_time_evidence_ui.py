"""Time evidence is visible without promoting local artifacts to business approval."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from marvis.agent.gate_payloads import build_model_delivery_payload
from marvis.db_schema import connect
from marvis.plugins.manifest import ToolRef
from marvis.repositories.modeling import ModelingRepository
from tests.test_operations_api import _claim_role
from tests.test_reference_decision import packaged as packaged


ROOT = Path(__file__).resolve().parents[1]


def _node(body, data):
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { readinessHtml, collectPackage, packageSignature } from './marvis/static/js/production-package-form.js';
import { renderModelDeliveryPanel } from './marvis/static/js/v2/model_delivery_panel.js';
const data = JSON.parse(fs.readFileSync(0, 'utf8'));
"""
            + body,
        ],
        input=json.dumps(data),
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def _delivery(card):
    return build_model_delivery_payload(
        {
            "artifact_id": "native-artifact",
            "native_model_path": "model.pkl",
            "model_card_path": "model.model_card.json",
            "model_card": card,
            "capabilities": {
                "native_model_supported": True,
                "pmml_supported": True,
                "handoff_supported": True,
            },
        },
        SimpleNamespace(tool_ref=ToolRef("modeling", "post_training_action")),
    )


@pytest.mark.parametrize(
    "training",
    [
        None,
        {},
        {
            "feature_time_evidence": {"assurance": "pass"},
            "preprocessing_evidence": {"assurance": "verified"},
        },
        {
            "feature_time_evidence": {"assurance": []},
            "preprocessing_evidence": {"assurance": {}},
        },
    ],
)
def test_legacy_or_unknown_delivery_is_visible_even_when_all_artifacts_ready(training):
    delivery = _delivery({"training": training})
    evidence = [
        item for item in delivery["readiness"] if item["id"].endswith("_evidence")
    ]
    assert len(evidence) == 3
    assert {item["status"] for item in evidence} == {"unknown"}
    _node(
        r"""
const html = renderModelDeliveryPanel({ metadata: { model_delivery: data } });
assert.match(html, /3 项证据未知/);
assert.match(html, /业务验收未在此确认/);
assert.doesNotMatch(html, /交付项已就绪|交付产物已就绪/);
for (const label of ['字段时点', '预处理范围', '参数历史时间']) assert.match(html, new RegExp(label));
assert.equal((html.match(/data-readiness-kind="warning"/g) || []).length, 3);
// Old persisted messages never passed through the new payload builder.
data.readiness = data.readiness.filter(item => !item.id.endsWith('_evidence'));
const legacy = renderModelDeliveryPanel({ metadata: { model_delivery: data } });
assert.match(legacy, /3 项证据未知/);
assert.match(legacy, /旧交付记录未提供此项证据/);
""",
        delivery,
    )


def test_distinct_assurances_never_promote_parameter_time_or_cv_scope():
    delivery = _delivery(
        {
            "training": {
                "feature_time_evidence": {"assurance": "verified"},
                "preprocessing_evidence": {"assurance": "training_only"},
                "parameter_time_evidence": {"assurance": "verified"},
            }
        }
    )
    cards = {item["id"]: item for item in delivery["readiness"]}
    assert cards["feature_time_evidence"]["status"] == "verified"
    assert cards["preprocessing_evidence"]["status"] == "training_only"
    assert cards["parameter_time_evidence"]["status"] == "unknown"
    _node(
        r"""
const html = renderModelDeliveryPanel({ metadata: { model_delivery: data } });
assert.match(html, /1 项证据未知/);
assert.match(html, /已核验记录时点/);
assert.match(html, /仅外层训练成员/);
assert.match(html, /不证明交叉验证折内独立拟合/);
assert.doesNotMatch(html, /交付项已就绪|交付产物已就绪/);
""",
        delivery,
    )


def test_local_package_shows_unknown_but_keeps_existing_declaration_flow(packaged):
    app, _, request, _, _, artifact, *_ = packaged
    with TestClient(app) as client:
        _claim_role(app, client, "maker")
        response = client.get(
            "/api/reference-decision/readiness",
            params={
                "model_artifact_id": artifact.id,
                "strategy_id": request.strategy_id,
                "strategy_version": request.strategy_version,
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["state"] == "authenticated"
    assert result["parameter_time_evidence"]["assurance"] == "unknown"
    _node(
        r"""
const html = readinessHtml(data);
assert.match(html, /仅用于本地参考构包/);
assert.equal((html.match(/<strong>未知<\/strong>/g) || []).length, 3);
assert.match(html, /业务验收与生产认证尚未由此确认/);
assert.match(html, /确认合同并构建冻结包/);
const strategy = { strategy_id: data.strategy.id, version: data.strategy.version };
const values = { package_kind: 'model', model_artifact_id: data.model.id,
  strategy_id: strategy.strategy_id, score_product: 'raw_pd', score_field: 'pd',
  decision_node: 'underwriting', timeout_seconds: '5', failure_action: 'review' };
data.raw_requirements.forEach((item, index) => {
  values[`raw_type_${index}`] = 'number'; values[`raw_nullable_${index}`] = 'false';
});
const contract = collectPackage(values, { readiness: data, strategies: [strategy], checkedSignature: packageSignature(values, strategy) });
assert.equal(contract.model_artifact_id, data.model.id);
assert.equal(contract.raw_schema.length, 2);
// A caller-supplied claim cannot make historical parameter time appear verified.
data.feature_time_evidence.assurance = 'verified';
data.preprocessing.state = 'training_only';
data.parameter_time_evidence.assurance = 'verified';
const bounded = readinessHtml(data);
assert.match(bounded, /已核验记录时点/);
assert.match(bounded, /仅外层训练成员/);
assert.match(bounded, /参数历史时间<\/legend><strong>未知/);
data.feature_time_evidence.assurance = '__proto__';
data.preprocessing.state = 'constructor';
assert.equal((readinessHtml(data).match(/<strong>未知<\/strong>/g) || []).length, 3);
""",
        result,
    )


def test_self_reported_parameter_evidence_cannot_bypass_real_readiness_api(packaged):
    app, _, _, _, _, artifact, *_ = packaged
    with connect(app.state.settings.db_path) as conn:
        original = conn.execute(
            "SELECT params_json FROM model_artifacts WHERE id=?", (artifact.id,)
        ).fetchone()["params_json"]
        params = json.loads(original)
        params["parameter_time_evidence"] = {"assurance": "verified"}
        conn.execute(
            "UPDATE model_artifacts SET params_json=? WHERE id=?",
            (json.dumps(params), artifact.id),
        )
    try:
        with TestClient(app) as client:
            _claim_role(app, client, "maker")
            body = client.get(
                "/api/reference-decision/readiness",
                params={"model_artifact_id": artifact.id},
            ).json()
        assert body["state"] == "blocked"
        assert body["parameter_time_evidence"]["assurance"] == "unknown"
    finally:
        with connect(app.state.settings.db_path) as conn:
            conn.execute(
                "UPDATE model_artifacts SET params_json=? WHERE id=?",
                (original, artifact.id),
            )


def test_rule_only_readiness_has_no_model_problem_or_model_lookup(
    packaged, monkeypatch
):
    app, _, request, *_ = packaged
    monkeypatch.setattr(
        ModelingRepository,
        "get_model_artifact",
        lambda *args: pytest.fail("rule queried model"),
    )
    with TestClient(app) as client:
        _claim_role(app, client, "maker")
        result = client.get(
            "/api/reference-decision/readiness",
            params={
                "package_kind": "rule_only",
                "strategy_id": request.strategy_id,
                "strategy_version": request.strategy_version,
            },
        ).json()
    _node(
        r"""
const html = readinessHtml(data, 'rule_only');
assert.match(html, /模型证据不适用（纯规则包，无模型）/);
assert.doesNotMatch(html, /<strong>未知|模型分数字段/);
assert.match(html, /确认合同并构建冻结包/);
const candidate = renderModelDeliveryPanel({ metadata: { model_delivery: { source_tool: 'compare_experiments', model_card: {}, readiness: [] } } });
assert.doesNotMatch(candidate, /字段时点|参数历史时间|证据未知/);
""",
        result,
    )
