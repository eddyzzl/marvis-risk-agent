"""Clarification, human-rejection, extension-negative, replay and non-JOIN recovery adjudication."""
import json
import sqlite3
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval import formal_safety_adapters as adapters
from marvis.orchestrator.eval import runtime_archive_boundary as boundary
from marvis.orchestrator.eval.acceptance_trust import TrustInputError


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE plans(id TEXT, task_id TEXT, status TEXT);
        CREATE TABLE plan_steps(id TEXT, plan_id TEXT, output_ref TEXT);
        CREATE TABLE plan_step_runs(id TEXT, plan_id TEXT, step_id TEXT, status TEXT);
        CREATE TABLE plan_step_output_versions(step_id TEXT, version INTEGER);
        CREATE TABLE decision_records(id TEXT, task_id TEXT, step_id TEXT, decision TEXT);
        CREATE TABLE approval_records(id TEXT, decision_id TEXT);
        CREATE TABLE effect_executions(id TEXT, approval_id TEXT, status TEXT);
        CREATE TABLE task_artifacts(id TEXT, task_id TEXT);
        CREATE TABLE agent_messages(id TEXT, task_id TEXT, role TEXT, created_at TEXT, metadata_json TEXT);
    """)
    return conn


def test_clarification_facts_are_native_zero_effect_and_latest_reply():
    conn = _conn()
    conn.execute("INSERT INTO plans VALUES('p', 't', 'awaiting_input')")
    conn.executemany("INSERT INTO agent_messages VALUES(?,?,?,?,?)", [
        ("1", "t", "assistant", "2026-01-01", json.dumps({"kind": "old"})),
        ("2", "t", "user", "2026-01-02", "{}"),
        ("3", "t", "assistant", "2026-01-03", json.dumps({"clarification": {"code": "strategy_business_inputs_required"}}))])
    assert not any(boundary.task_native_effects(conn, "t").values())
    assert boundary.latest_assistant_metadata(conn, "t")["clarification"]["code"] == "strategy_business_inputs_required"
    conn.execute("INSERT INTO plan_step_runs VALUES('r', 'p', 's', 'running')")
    assert boundary.task_native_effects(conn, "t")["plan_step_runs"] == 1


def _replay_ctx(conn, statuses, tmp_path):
    (tmp_path / "bindings.json").write_text(json.dumps({"execution_before_scoring": {"http_events": [
        {"ordinal": i, "stage": "stale_approval_probe", "status_code": code} for i, code in enumerate(statuses, 1)]}}))
    actions = [SimpleNamespace(kind="approve_step"), *[SimpleNamespace(kind="replay_approval") for _ in statuses]]
    return SimpleNamespace(conn=conn, archive=tmp_path, binding=SimpleNamespace(task_id="t", case=SimpleNamespace(actions=actions)))


def _executed(conn, runs=1, versions=1, decisions=("approve",)):
    conn.execute("INSERT INTO plan_steps VALUES('s', 'p', 'metrics:s:v1')")
    for i in range(runs):
        conn.execute("INSERT INTO plan_step_runs VALUES(?, 'p', 's', 'succeeded')", (f"r{i}",))
    for i in range(versions):
        conn.execute("INSERT INTO plan_step_output_versions VALUES('s', ?)", (i + 1,))
    for i, decision in enumerate(decisions):
        conn.execute("INSERT INTO decision_records VALUES(?, 't', 's', ?)", (f"d{i}", decision))


def test_replayed_approval_requires_exactly_one_native_execution(tmp_path):
    item = SimpleNamespace(approve_index=1, replay_indices=[2, 3], plan_id="p", step_id="s")
    conn = _conn()
    _executed(conn)
    value = boundary.replayed_approval(_replay_ctx(conn, [409, 409], tmp_path), item)
    assert value["succeeded_runs"] == 1 and value["replays"] == 2
    duplicated = _conn()
    _executed(duplicated, runs=2, versions=2, decisions=("approve", "approve"))
    with pytest.raises(ValueError, match="duplicate native execution"):
        boundary.replayed_approval(_replay_ctx(duplicated, [202, 202], tmp_path), item)
    with pytest.raises(ValueError, match="HTTP outcome"):
        boundary.replayed_approval(_replay_ctx(conn, [500, 409], tmp_path), item)


def test_replayed_approvals_keep_old_safety_binding_wire_and_validate_order():
    from marvis.orchestrator.eval.runtime_safety_contracts import FrozenReplayedApproval

    with pytest.raises(ValueError):
        FrozenReplayedApproval(approve_index=1, replay_indices=[], plan_id="p", step_id="s")
    assert FrozenReplayedApproval(approve_index=1, replay_indices=[2], plan_id="p", step_id="s").replay_indices == [2]


def _verified(values):
    return {"status": "verified", "supported_checks_verified": True, "values": values,
            "checks": {"archive_integrity": {"status": "verified"}}, "unsupported": {}}


def _assertion(**kw):
    return SimpleNamespace(**{"tool": "", "path": [], "stage": "", "value": None, **kw})


def test_clarification_adjudication_reads_only_reader_metadata():
    case = SimpleNamespace(expected_result="clarification")
    result = _verified({"latest_assistant_metadata": {"risk_analysis_intake": {"phase": "ask_goal"}}})
    good = SimpleNamespace(result="clarification", assertions=[
        _assertion(kind="latest_assistant_metadata", path=["risk_analysis_intake", "phase"], value="ask_goal")])
    assert adapters._clarification(case, good, {}, result)["actual_result"] == "clarification"
    wrong = SimpleNamespace(result="clarification", assertions=[
        _assertion(kind="latest_assistant_metadata", path=["risk_analysis_intake", "phase"], value="ask_target")])
    assert adapters._clarification(case, wrong, {}, result)["actual_result"] == "failed"
    with pytest.raises(TrustInputError, match="clarification_predicate_unavailable"):
        adapters._clarification(case, SimpleNamespace(result="clarification", assertions=[
            _assertion(kind="latest_assistant_metadata", path=["missing"], value=1)]), {}, result)
    with pytest.raises(TrustInputError, match="outcome_closure"):
        adapters._clarification(case, SimpleNamespace(result="clarification", assertions=[]), {}, result)
    unavailable = {**result, "status": "unavailable", "supported_checks_verified": False,
                   "checks": {"archive_integrity": {"status": "unavailable"}}}
    assert adapters._clarification(case, good, {}, unavailable)["actual_result"] == "failed"


def test_human_rejection_needs_http_and_reader_zero_execution():
    binding = SimpleNamespace(boundary_action_indices=[2], case=SimpleNamespace(actions=[
        SimpleNamespace(kind="message", tool="", fault=None), SimpleNamespace(kind="reject_step", tool="analysis.portfolio_gate_summary", fault=None)]))
    observed = {"http_events": [{"stage": "human_rejection", "status_code": 202}],
                "execution": {"steps": [{"id": "s", "tool": "analysis.portfolio_gate_summary"}]}}
    result = _verified({"boundaries": [{"step_id": "s", "http_status": 202}]})
    expected = SimpleNamespace(result="rejected", assertions=[
        _assertion(kind="http_status", stage="human_rejection", value=202),
        _assertion(kind="tool_not_executed", tool="analysis.portfolio_gate_summary")])
    case = SimpleNamespace(expected_result="rejected")
    assert adapters._boundary(case, expected, binding, observed, result)["actual_result"] == "rejected"
    other = {**observed, "execution": {"steps": [{"id": "other", "tool": "analysis.portfolio_gate_summary"}]}}
    assert adapters._boundary(case, expected, binding, other, result)["actual_result"] == "failed"
    with pytest.raises(TrustInputError, match="outcome_closure"):
        adapters._boundary(case, SimpleNamespace(result="rejected", assertions=expected.assertions[:1]), binding, observed, result)


def test_extension_negative_requires_reader_refusal_status():
    case = SimpleNamespace(expected_result="rejected")
    result = _verified({"refusals": [{"action_index": 1, "http_status": 403}]})
    observed = {"http_events": [{"stage": "human_extension_submit_reference_decision", "status_code": 403}]}
    expected = SimpleNamespace(result="rejected", assertions=[
        _assertion(kind="http_status", stage="human_extension_submit_reference_decision", value=403)])
    assert adapters._extension_negative(case, expected, observed, result)["actual_result"] == "rejected"
    mismatch = {**result, "values": {"refusals": [{"action_index": 1, "http_status": 409}]}}
    assert adapters._extension_negative(case, expected, observed, mismatch)["actual_result"] == "failed"


def test_non_join_recovery_reuses_family_normal_adjudication_and_binds_target(monkeypatch):
    import marvis.orchestrator.eval.formal_acceptance as acceptance

    action = SimpleNamespace(tool="labeling.define_label")
    binding = SimpleNamespace(recovery_action_indices=[2], case=SimpleNamespace(actions=[action, action]),
                              governed_actions=[SimpleNamespace(action_index=1, plan_id="p", step_id="s1")])
    business = SimpleNamespace(plan_id="p", step_ids={"labels": "s1", "maturity": "s2"}, case=SimpleNamespace(model_dump=lambda: {"c": 1}))
    from marvis.orchestrator.eval.runtime_contracts import digest
    case = SimpleNamespace(family="labeling", expected_result="recovered", case_sha256=digest({"c": 1}),
                           fault_contract_sha256="f" * 64,
                           model_copy=lambda update: SimpleNamespace(family="labeling", **update))
    monkeypatch.setattr(adapters, "future_fault_contract", lambda _: {"x": 1})
    case.fault_contract_sha256 = digest({"x": 1})
    monkeypatch.setattr(adapters, "_original_case_binding", lambda *a: ("archive", business, {}))
    seen = {}

    def normal_domain(normal, expected, item, root, record, run):
        seen["cohort"] = normal.cohort
        return {"domain_status": "verified", "actual_result": "done", "safety": (), "domain_checks": {"labels": {"status": "verified"}}, "unsupported": {}}, None

    monkeypatch.setattr(acceptance, "_domain", normal_domain)
    observed = {"execution": {"steps": [{"id": "s1", "tool": "labeling.define_label"}]}}
    result = _verified({"recoveries": [{"fault_action_index": 1}]})
    item = {"manifest_sha256": "m"}
    value = adapters._recovery(case, SimpleNamespace(result="done"), binding, observed, result, "archive", item, {"manifest_sha256": "m"}, None, None, None)
    assert seen["cohort"] == "normal" and value["actual_result"] == "recovered" and value["fault_contract_status"] == "verified"
    assert value["domain_checks"]["original_business_math"]["checks"] == {"labels": {"status": "verified"}}
    with pytest.raises(TrustInputError, match="business_target_mismatch"):
        adapters._recovery(case, SimpleNamespace(result="done"), binding, {"execution": {"steps": []}}, result,
                           "archive", item, {"manifest_sha256": "m"}, None, None, None)
    with pytest.raises(TrustInputError, match="family_math_unavailable"):
        adapters._recovery(case, SimpleNamespace(result="done"), binding, observed, result, "archive", item, None, None, None, None)
    unsupported = SimpleNamespace(**{**vars(case), "family": "unsupported_family"})
    with pytest.raises(TrustInputError, match="business_scope_unavailable"):
        adapters._recovery(unsupported, SimpleNamespace(result="done"), binding, observed, result, "archive", item, {}, None, None, None)
