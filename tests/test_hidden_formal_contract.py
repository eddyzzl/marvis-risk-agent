"""Materialization/preflight only: fixtures never call a model or claim history."""
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from marvis.orchestrator.eval import hidden_benchmark as proposal
from marvis.orchestrator.eval import hidden_formal_contract as contract
from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract
from marvis.orchestrator.eval.perf_gate import load_profile
from marvis.orchestrator.eval.runtime_contracts import ModelConnection, RuntimeCase, RuntimeSuite, digest
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase, ExpectedSuite



@pytest.fixture(scope="session")
def hidden_source_checkout(tmp_path_factory):
    """Real bounded inventory carrier, outside the developer's evidence tree.

    These are contract/coordinator fixtures, not an attestation of the running
    checkout. Copy the actual runtime package bytes; retain the production
    source_inventory implementation, its Git/untracked walk and alias limits.
    """
    source = Path(contract.__file__).resolve().parents[3]
    checkout = tmp_path_factory.mktemp("hidden-contract-source").resolve()
    shutil.copytree(source / "marvis", checkout / "marvis",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"))
    shutil.copy2(source / "pyproject.toml", checkout / "pyproject.toml")
    (checkout / ".gitignore").write_text("ignored-notes.txt\n")
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", ".gitignore"], check=True)
    return checkout


@pytest.fixture(autouse=True)
def isolated_hidden_source(monkeypatch, hidden_source_checkout):
    # Only redirect this fixture's checkout location; no inventory, hash,
    # trust limit, provider or execution function is replaced.
    monkeypatch.setattr(contract, "__file__", str(hidden_source_checkout / "marvis/orchestrator/eval/hidden_formal_contract.py"))


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return {"path": path.name, "sha256": digest(path.read_bytes())}


def _materialized(tmp_path, *, supported_faults=True, native_faults=True):
    """Synthetic schema fixtures; declared source labels are not attestation."""
    held = tmp_path / "held"
    if supported_faults and not native_faults:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(proposal, "RECOVERY_FAULTS", ("response_lost_after_commit",))
            proposal.freeze_case_set(held, "unit-proposal", salt=b"x" * 32, commit=contract._source()["commit"])
    else:
        proposal.freeze_case_set(held, "unit-proposal", salt=b"x" * 32, commit=contract._source()["commit"])
    freeze_dir = held / "unit-proposal"
    frozen = proposal.load_frozen_case_set(freeze_dir)
    root = tmp_path / "inputs"
    data = root / "data"
    data.mkdir(parents=True)
    contents = {"sample.csv": b"x,y\n1,0\n2,1\n", "model.ipynb": b"{}", "model.pmml": b"<PMML/>", "dictionary.csv": b"field,meaning\nx,test\n"}
    if native_faults:
        contents["monitoring.csv"] = b"x,y\n3,0\n4,1\n"
    for name, raw in contents.items():
        (data / name).write_bytes(raw)
    materials = [{"path": name, "sha256": digest(raw)} for name, raw in contents.items()]
    budget = {"wall_seconds": 60, "max_llm_attempts": 2, "max_http_requests": 20,
              "max_output_tokens_per_attempt": 32, "max_total_tokens": 10000, "max_cost": .1, "currency": "TEST"}
    cases, expectations = [], {}
    tasks = {"data_processing": "data_join", "labeling": "data_join", "feature": "feature_analysis", "modeling": "modeling",
             "validation": "validation", "strategy": "strategy", "risk_portfolio": "portfolio", "monitoring": "modeling",
             "online_decision": "strategy", "connector_antifraud": "strategy", "collections": "strategy"}
    for source in frozen["cases"]:
        scenario = {"normal": "normal", "extension": "normal", "recovery": "recovery"}.get(source["cohort"],
                    "clarification" if source["expected_result"] == "clarification" else "rejection")
        mat = [{**materials[0], "role": "sample", "source_kind": "deidentified_historical"}]
        task, actions = {"task_type": tasks[source["family"]]}, []
        if source["family"] == "validation":
            task["algorithm"] = "lr"
            mat += [{**item, "role": role} for item, role in zip(materials[1:4], ("notebook", "pmml", "dictionary"), strict=True)]
            actions = [{"kind": "start_validation_agent", "content": "Start local schema fixture"},
                       {"kind": "confirm_current_validation_report", "content": "Confirm local schema fixture"}]
        if source["cohort"] == "recovery":
            if source["family"] == "validation":
                actions[1]["fault"] = {"kind": "discard_success_response_once"}
                actions.append({"kind": "recover_validation_report", "content": "Read committed result", "fault_source_action": 2})
            else:
                actions = [{"kind": "approve_step", "tool": "feature.analyze", "content": "Approve fixed fixture",
                            "fault": {"kind": "discard_success_response_once"}},
                           {"kind": "recover_governed_decision", "tool": "feature.analyze", "content": "Read committed result", "fault_source_action": 1}]
        if native_faults and source["cohort"] == "recovery":
            fault_name = source["spec"]["fault_kind"]
            kind = contract.SUPPORTED_FAULT_BINDINGS[fault_name]
            if source["family"] == "validation":
                if fault_name == "stale_cas_write":
                    actions[1]["fault"] = {"kind": "stale_report_revision_once"}
                else:
                    actions = [{"kind": "start_validation_agent", "content": "Start exact native entry", "fault": {"kind": kind}},
                        {"kind": "recover_validation_execution", "content": "Explicit native retry", "fault_source_action": 1},
                        {"kind": "confirm_current_validation_report", "content": "Confirm current report"}]
            elif source["family"] == "monitoring":
                mat.append({**materials[4], "role": "unknown", "source_kind": "deidentified_historical"})
                kind = "stale_monitoring_revision_once" if fault_name == "stale_cas_write" else kind
                actions = [{"kind": "approve_step", "tool": "modeling.configure_tuning", "content": "Prior modeling gate"},
                    {"kind": "submit_model_monitoring_request", "content": "Separate monitoring request",
                     "monitoring_request": {"material_path": "monitoring.csv", "target_col": None}},
                    {"kind": "confirm_model_monitoring_plan", "content": "Confirm native monitoring", "fault": {"kind": kind}},
                    {"kind": "recover_monitoring_execution", "content": "Explicit native monitoring recovery", "fault_source_action": 3}]
            else:
                tool = {"data_processing": "data_ops.execute_join", "labeling": "labeling.define_label",
                        "feature": "feature.analyze_feature_bins", "modeling": "modeling.configure_tuning",
                        "risk_portfolio": "analysis.portfolio_gate_summary", "strategy": "strategy.backtest_strategy"}[source["family"]]
                fault = {"kind": kind}
                if fault_name == "worker_crash_after_commit" and source["family"] in {"modeling", "risk_portfolio"}:
                    fault["downstream_tool"] = "modeling.train_models" if source["family"] == "modeling" else "analysis.portfolio_report"
                actions = [{"kind": "approve_step", "tool": tool, "content": "Approve exact native boundary", "fault": fault},
                    {"kind": "recover_governed_decision", "tool": tool, "content": "Explicit recovery", "fault_source_action": 1}]
        case = RuntimeCase.model_validate({"id": source["id"], "revision": "test-materialization-v1", "family": source["family"],
            "case_set": "independently_held_hidden_acceptance", "scenario": scenario, "task": task, "materials": mat,
            "actions": actions, "business_constraints_source": "Synthetic contract-only test; no independent historical claim", "budget": budget})
        cases.append(case)
        expectations[case.id] = ExpectedCase.model_validate({"result": "done" if source["expected_result"] == "recovered" else source["expected_result"],
            "assertions": [{"kind": "http_status", "stage": "create_task", "value": 200}]})
    suite, expected = RuntimeSuite(cases=cases), ExpectedSuite(cases=expectations)
    files = {"cases": _write(root / "cases.json", suite.model_dump()), "expected": _write(root / "expected.json", expected.model_dump())}
    model = ModelConnection(model_id="test-only", model_name="test-only", api_base_url="http://127.0.0.1:1/v1", api_key_env="MARVIS_HIDDEN_TEST_KEY", context_window=4096, max_output_tokens=32)
    files["model"] = _write(root / "model.json", model.model_dump())
    files["price_book"] = _write(root / "prices.json", {"model_name": model.model_name, "provider": "unit-test", "currency": "TEST", "source": "local synthetic test only", "effective_date": "2026-11-04", "prompt_per_million": 0., "completion_per_million": 0.})
    document = root / "provider-terms.txt"
    document.write_text("Local test double only; no external provider guarantee or independent signature.\n")
    approval_ref = _write(root / "provider-approval.json", {
        "schema": "marvis.provider-budget-approval.v1", "protocol": "single_text_utf8_bound_v1", "provider": "unit-test",
        "model_name": model.model_name, "approved_by": "local-test-only", "approved_on": "2026-11-04", "source_scope": "local_test_fixture",
        "source_document": {"path": document.name, "sha256": digest(document.read_bytes())},
        "price_book_sha256": files["price_book"]["sha256"],
        "context_window": model.context_window, "max_output_tokens": model.max_output_tokens, "framing_token_allowance": 32,
        "input_token_accounting": "at_most_utf8_request_bytes_plus_framing", "output_token_accounting": "max_tokens_includes_all_billed_completion",
        "extra_charges": "none_for_allowed_payload"})
    provider_limits = {"protocol": "single_text_utf8_bound_v1", "context_window": model.context_window,
                       "max_output_tokens": model.max_output_tokens, "framing_token_allowance": 32,
                       "price_book_sha256": files["price_book"]["sha256"], "retain_full_reservations": True, "approval": approval_ref}
    profile = copy.deepcopy(load_profile())
    profile.update(status="frozen", approved_by="test-only-not-an-approval", approved_on="2026-11-04")
    profile["load"] = {"data_rows": 2, "concurrent_users": 2, "sustained_seconds": 1}
    profile["slo"] = {mode: {"minimum_observations": 1, "p95_seconds": 60., "max_error_rate": .01} for mode in ("cold_start", "steady", "concurrent", "recovery")}
    load_policy = {"concurrent_workers": 2, "warmup": {"endpoints": ["/api/tasks"], "requests": 1, "minimum_seconds": 0.}}
    environment = contract.observed_environment(profile, load_policy)
    profile["hardware"] = {"description": "test-only current host", "fingerprint_sha256": digest(environment["hardware"])}
    files["environment"] = _write(root / "environment.json", environment)
    source = contract._source()
    inventory = contract.source_inventory(Path(contract.__file__).resolve().parents[3])
    files["source_inventory"] = _write(root / "source-inventory.json", inventory)
    benchmark = proposal.to_formal_benchmark(frozen, benchmark_id="contract-unit", commit=source["commit"], source_inventory_sha256=digest(inventory),
        model={"profile": model.model_id, "model_name": model.model_name, "connection_sha256": digest(model.model_dump()), "source": "real_model", "evaluation_mode": "blind"},
        environment={f"{key}_sha256": digest(value) for key, value in environment.items()}, budgets={case.id: budget for case in cases},
        slo={mode: {"minimum_observations": 1, "p95_wall_seconds": 60.} for mode in profile["slo"]}).model_dump(by_alias=True)
    bindings = []
    for original, actual, plan in zip(frozen["cases"], cases, benchmark["cases"], strict=True):
        plan["case_sha256"] = digest(actual.model_dump())
        plan["expected_sha256"] = digest(expectations[actual.id].model_dump())
        plan["fault_contract_sha256"] = digest(future_fault_contract(actual)) if plan["cohort"] == "recovery" else None
        bindings.append({"case_id": actual.id, "scenario_sha256": original["case_sha256"], "case_sha256": plan["case_sha256"],
                         "expected_sha256": plan["expected_sha256"], "proposal_fault_sha256": original["fault_contract_sha256"],
                         "actual_fault_sha256": plan["fault_contract_sha256"]})
    binding = {"schema": contract.SCHEMA, "proposal_manifest_sha256": digest((freeze_dir / "freeze-manifest.json").read_bytes()),
               "benchmark": benchmark, "files": files, "dataset_root": "data", "materials": materials, "case_bindings": bindings,
               "load_policy": load_policy, "provider_limits": provider_limits}
    _write(root / "binding.json", binding)
    _write(root / "profile.json", profile)
    _write(root / "ledger.json", {"agent_benchmark_plan": {"cost_cap": {"status": "decided", "max_cost": 19.3, "currency": "TEST", "price_book": "prices.json",
                                                                        "provider_approval_sha256": approval_ref["sha256"]}}})
    return {"freeze_dir": freeze_dir, "binding_path": root / "binding.json", "ledger_path": root / "ledger.json", "profile_path": root / "profile.json",
            "binding": binding, "profile": profile, "root": root}


def _prepare(fixture, **kwargs):
    return contract.prepare_hidden_execution(fixture["freeze_dir"], fixture["binding_path"], ledger_path=fixture["ledger_path"],
        profile_path=fixture["profile_path"], secret_env={"MARVIS_HIDDEN_TEST_KEY": "local-fixture-placeholder"}, **kwargs)


def test_actual_case_hashes_are_distinct_from_scenario_templates_and_all_repeat_costs_are_counted(tmp_path):
    fixture = _materialized(tmp_path)
    prepared = _prepare(fixture)
    assert len(prepared.suite.cases) == 133 and len(prepared.benchmark.attempts) == 193
    assert prepared.total_cost_bound == 19.3
    assert all(c.case_sha256 != c.scenario_sha256 for c in prepared.binding.case_bindings)
    assert all(p.case_sha256 == digest(c.model_dump()) for p, c in zip(prepared.benchmark.cases, prepared.suite.cases, strict=True))
    assert prepared.model.api_base_url == "http://127.0.0.1:1/v1"  # no server exists and no request is attempted
    inventory = json.loads((fixture["root"] / "source-inventory.json").read_bytes())
    assert digest(inventory) == prepared.benchmark.source_inventory_sha256
    assert prepared.benchmark.source_inventory_sha256 != prepared.source_identity["source_sha256"]
    assert prepared.ledger_sha256 == digest(prepared.ledger_path.read_bytes())
    assert prepared.profile_sha256 == digest(prepared.profile_path.read_bytes())


def test_original_six_fault_proposal_cannot_reclassify_every_fault_as_response_loss(tmp_path):
    fixture = _materialized(tmp_path, supported_faults=False, native_faults=False)
    with pytest.raises(ValueError, match="hidden_runtime_fault_reclassified"):
        _prepare(fixture)


@pytest.mark.parametrize("mutation,reason", [
    ("budget", "slots_exceed"), ("pending", "pending_user"), ("material", "input_hash"),
    ("scenario_hash", "materialization_hash"), ("actual_hash", "materialization_hash"),
    ("expected", "input_hash"), ("model", "input_hash"), ("environment", "input_hash"),
    ("warmup_budget", "warmup_exhausts"), ("price", "price_book_or_currency"),
    ("source", "source_identity"), ("slot_id", "slot_inventory"),
    ("provider_approval", "input_hash"), ("provider_terms", "input_hash"), ("provider_cap", "ceiling_or_prices"),
    ("authority", "not_bound_by_cost_authority"),
])
def test_preflight_rejects_tampering_and_incomplete_inputs_without_creating_outputs(tmp_path, mutation, reason):
    fixture = _materialized(tmp_path)
    binding = fixture["binding"]
    root = fixture["root"]
    if mutation in {"budget", "pending", "price", "authority"}:
        ledger = json.loads(fixture["ledger_path"].read_text())
        key, value = {"budget": ("max_cost", 13.3), "pending": ("status", "pending_user_decision"), "price": ("currency", "USD"),
                      "authority": ("provider_approval_sha256", "a" * 64)}[mutation]
        ledger["agent_benchmark_plan"]["cost_cap"][key] = value
        _write(fixture["ledger_path"], ledger)
    elif mutation == "material":
        (root / "data/sample.csv").write_text("changed")
    elif mutation in {"expected", "model", "environment"}:
        (root / binding["files"][mutation]["path"]).write_text("{}")
    elif mutation == "scenario_hash":
        binding["case_bindings"][0]["scenario_sha256"] = "a" * 64
    elif mutation == "actual_hash":
        binding["benchmark"]["cases"][0]["case_sha256"] = binding["case_bindings"][0]["scenario_sha256"]
    elif mutation == "warmup_budget":
        binding["load_policy"]["warmup"]["requests"] = 20
    elif mutation == "source":
        binding["benchmark"]["source_inventory_sha256"] = "b" * 64
    elif mutation == "slot_id":
        binding["benchmark"]["attempts"][0]["id"] = "other-slot"
    elif mutation == "provider_approval":
        (root / "provider-approval.json").write_text("{}")
    elif mutation == "provider_terms":
        (root / "provider-terms.txt").write_text("unbound terms")
    elif mutation == "provider_cap":
        binding["provider_limits"]["context_window"] = 8192
    _write(fixture["binding_path"], binding)
    before = set(root.rglob("*"))
    with pytest.raises(ValueError, match=reason):
        _prepare(fixture)
    assert set(root.rglob("*")) == before


def test_load_mode_can_be_explicitly_reassigned_without_mutating_proposal(tmp_path):
    fixture = _materialized(tmp_path)
    before = (fixture["freeze_dir"] / "attempts.json").read_bytes()
    attempts = fixture["binding"]["benchmark"]["attempts"]
    attempts[0]["load_mode"], attempts[1]["load_mode"] = attempts[1]["load_mode"], attempts[0]["load_mode"]
    _write(fixture["binding_path"], fixture["binding"])
    assert _prepare(fixture).benchmark.attempts[0].load_mode == attempts[0]["load_mode"]
    assert (fixture["freeze_dir"] / "attempts.json").read_bytes() == before


@pytest.mark.parametrize("endpoints,requests", [(["/api/tasks", "/api/tasks"], 2), (["/api/tasks", "/"], 1), (["/api/tasks/delete"], 1), (["/api/tasks"], 1001)])
def test_warmup_is_fixed_read_only_and_covers_each_endpoint(endpoints, requests):
    with pytest.raises(ValueError):
        contract.WarmupPolicy(endpoints=endpoints, requests=requests, minimum_seconds=0.)


def test_concurrent_policy_requires_at_least_two_workers():
    with pytest.raises(ValueError):
        contract.LoadPolicy(concurrent_workers=1, warmup={"endpoints": ["/api/tasks"], "requests": 1, "minimum_seconds": 0.})


def test_error_rate_approval_is_part_of_the_frozen_load_environment(tmp_path):
    fixture = _materialized(tmp_path)
    profile = fixture["profile"]
    profile["slo"]["steady"]["max_error_rate"] = .5
    _write(fixture["profile_path"], profile)
    with pytest.raises(ValueError, match="environment_changed"):
        _prepare(fixture)


def test_replacing_prices_and_rehashing_binding_does_not_replace_budget_authority(tmp_path):
    fixture = _materialized(tmp_path)
    price_path = fixture["root"] / "prices.json"
    price = json.loads(price_path.read_text())
    price["prompt_per_million"] = 9.
    ref = _write(price_path, price)
    fixture["binding"]["files"]["price_book"] = ref
    fixture["binding"]["provider_limits"]["price_book_sha256"] = ref["sha256"]
    _write(fixture["binding_path"], fixture["binding"])
    with pytest.raises(ValueError, match="provider_budget_approval_mismatch"):
        _prepare(fixture)


@pytest.mark.parametrize("field,value,reason", [
    ("enable_thinking", True, "ceiling_or_prices"),
    ("api_base_url", "https://provider.invalid/v1", "approval_mismatch"),
])
def test_local_provider_approval_cannot_authorize_reasoning_or_external_requests(tmp_path, field, value, reason):
    fixture = _materialized(tmp_path)
    model_path = fixture["root"] / "model.json"
    model = json.loads(model_path.read_text())
    model[field] = value
    fixture["binding"]["files"]["model"] = _write(model_path, model)
    fixture["binding"]["benchmark"]["model"]["connection_sha256"] = digest(model)
    _write(fixture["binding_path"], fixture["binding"])
    with pytest.raises(ValueError, match=reason):
        _prepare(fixture)


def test_original_16_fault_slots_materialize_without_reclassifying_families_or_denominator(tmp_path):
    # Schema-only preflight. Actual family carriers have separate real HTTP tests.
    fixture = _materialized(tmp_path, native_faults=True)
    prepared = _prepare(fixture)
    original = proposal.load_frozen_case_set(fixture["freeze_dir"])
    recovery = [row for row in original["cases"] if row["cohort"] == "recovery"]
    assert len(recovery) == 16 and len(prepared.benchmark.attempts) == 193
    actual = {case.id: case for case in prepared.suite.cases}
    for row in recovery:
        case = actual[row["id"]]
        assert case.family == row["family"] and case.scenario == "recovery"
        action = next(a for a in case.actions if a.fault is not None)
        kind = contract.SUPPORTED_FAULT_BINDINGS[row["spec"]["fault_kind"]]
        if row["spec"]["fault_kind"] == "stale_cas_write":
            kind = {"validation": "stale_report_revision_once", "monitoring": "stale_monitoring_revision_once"}.get(row["family"], kind)
        assert action.fault.kind == kind


@pytest.mark.parametrize("mutation,reason", [
    ("ignored_bytes", "hidden_source_identity_changed"), ("alias", "source_alias_rejected"),
])
def test_isolated_fixture_still_uses_actual_untracked_inventory_and_alias_boundary(tmp_path, hidden_source_checkout, mutation, reason):
    fixture = _materialized(tmp_path)
    extra = hidden_source_checkout / ("ignored-notes.txt" if mutation == "ignored_bytes" else "untrusted-alias")
    try:
        if mutation == "ignored_bytes":
            extra.write_text("A later ignored file is still part of the actual checkout.\n")
        else:
            extra.symlink_to(hidden_source_checkout / ".gitignore")
        with pytest.raises(ValueError, match=reason):
            _prepare(fixture)
    finally:
        extra.unlink(missing_ok=True)
