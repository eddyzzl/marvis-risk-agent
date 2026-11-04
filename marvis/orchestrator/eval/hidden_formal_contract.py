"""Materialized hidden execution inputs, checked before any runtime starts.

Scenario-template hashes remain distinct from actual RuntimeCase/ExpectedCase
hashes. A prepared input is a runnable candidate, never an independent approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .formal_benchmark import FrozenBenchmark, FrozenModel
from .formal_safety_adapters import future_fault_contract
from .hidden_benchmark import load_frozen_case_set
from .perf_gate import _machine, formal_blockers, load_profile
from .runtime_contracts import ModelConnection, RuntimeSuite, digest
from .runtime_scoring import ExpectedSuite, PriceBook
from .runtime_formal_budget import ProviderApproval, ProviderLimits
from .acceptance_trust import source_inventory

SCHEMA = "marvis.hidden-execution-binding.v1"
_HASH = r"^[0-9a-f]{64}$"
# Admission follows implemented fault execution, not merely a newly added enum.
SUPPORTED_FAULT_BINDINGS = {
    "response_lost_after_commit": "discard_success_response_once",
    "stale_cas_write": "stale_step_cas_once",
    "worker_crash_after_commit": "worker_crash_after_commit_once",
    "tool_timeout": "tool_timeout_once",
    "service_restart_mid_step": "service_restart_mid_step_once",
    "cancel_then_resume": "cancel_then_resume_once",
}


def _relative(value):
    path = Path(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value
            or path.as_posix() != value or value == "."):
        raise ValueError("hidden_binding_path_must_be_canonical_relative")
    return value


class FileBinding(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HASH)

    @model_validator(mode="after")
    def canonical_path(self):
        _relative(self.path)
        return self


class ExecutionFiles(FrozenModel):
    cases: FileBinding
    expected: FileBinding
    model: FileBinding
    price_book: FileBinding
    environment: FileBinding
    source_inventory: FileBinding


class WarmupPolicy(FrozenModel):
    endpoints: list[Literal["/api/tasks", "/api/branding", "/"]] = Field(min_length=1, max_length=3)
    requests: int = Field(ge=1, le=1000)
    minimum_seconds: float = Field(ge=0, le=300, allow_inf_nan=False)

    @model_validator(mode="after")
    def coverage(self):
        if len(set(self.endpoints)) != len(self.endpoints) or self.requests < len(self.endpoints):
            raise ValueError("warmup_must_visit_each_unique_read_endpoint")
        return self


class LoadPolicy(FrozenModel):
    concurrent_workers: int = Field(ge=2, le=64)
    warmup: WarmupPolicy


class MaterializedCaseBinding(FrozenModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    scenario_sha256: str = Field(pattern=_HASH)
    case_sha256: str = Field(pattern=_HASH)
    expected_sha256: str = Field(pattern=_HASH)
    proposal_fault_sha256: str | None = Field(pattern=_HASH)
    actual_fault_sha256: str | None = Field(pattern=_HASH)


class FrozenHiddenExecution(FrozenModel):
    schema_version: Literal["marvis.hidden-execution-binding.v1"] = Field(alias="schema")
    proposal_manifest_sha256: str = Field(pattern=_HASH)
    benchmark: FrozenBenchmark
    files: ExecutionFiles
    dataset_root: str = Field(min_length=1)
    materials: list[FileBinding] = Field(min_length=1, max_length=5000)
    case_bindings: list[MaterializedCaseBinding] = Field(min_length=133, max_length=133)
    load_policy: LoadPolicy
    provider_limits: ProviderLimits

    @model_validator(mode="after")
    def distinct(self):
        _relative(self.dataset_root)
        if len({m.path for m in self.materials}) != len(self.materials):
            raise ValueError("hidden_material_paths_must_be_unique")
        if len({c.case_id for c in self.case_bindings}) != len(self.case_bindings):
            raise ValueError("hidden_case_bindings_must_be_unique")
        return self


@dataclass(frozen=True)
class PreparedHiddenExecution:
    binding: FrozenHiddenExecution
    benchmark: FrozenBenchmark
    suite: RuntimeSuite
    expected: ExpectedSuite
    model: ModelConnection
    price_book: PriceBook
    provider_limits: ProviderLimits
    environment: dict
    cases_path: Path
    expected_path: Path
    dataset_root: Path
    price_book_path: Path
    load_policy: LoadPolicy
    binding_sha256: str
    total_cost_bound: float
    source_identity: dict
    ledger_path: Path
    ledger_sha256: str
    profile_path: Path
    profile_sha256: str


def _source():
    from .runtime_runner import _source_identity
    return _source_identity()


def observed_environment(profile, load_policy):
    """Observed local facts for a caller to inspect before freezing approval."""
    source = _source()
    policy = load_policy.model_dump() if isinstance(load_policy, LoadPolicy) else load_policy
    return {"hardware": {k: v for k, v in _machine().items() if k != "scope"},
            "load": {"profile_load": profile["load"], "profile_slo": profile["slo"], "load_policy": policy},
            "runtime_identity": {"python": source["python"], "platform": source["platform"],
                                 "package_source_sha256": source["source_sha256"]},
            "dependency_inventory": source["packages"]}


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_hidden_json_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_hidden_json")))


def _path(root, relative):
    _relative(relative)
    path = root / relative
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
        raise ValueError("hidden_input_link_or_escape")
    return path


def _read(root, reference, *, maximum=32 * 1024**2):
    path = _path(root, reference.path)
    info = path.stat()
    if not path.is_file() or info.st_nlink != 1 or info.st_size > maximum:
        raise ValueError("hidden_input_not_regular_or_bounded")
    raw = path.read_bytes()
    if digest(raw) != reference.sha256:
        raise ValueError("hidden_input_hash_mismatch")
    return path, raw


def _check_cases(frozen, binding, suite, expected):
    planned = binding.benchmark.cases
    ids = [case["id"] for case in frozen["cases"]]
    if ([case.id for case in planned] != ids or [case.id for case in suite.cases] != ids
            or [item.case_id for item in binding.case_bindings] != ids or set(expected.cases) != set(ids)):
        raise ValueError("hidden_materialized_case_inventory_mismatch")
    source_slots = [(a["id"], a["case_id"], a["repetition"]) for a in frozen["attempts"]]
    actual_slots = [(a.id, a.case_id, a.repetition) for a in binding.benchmark.attempts]
    if source_slots != actual_slots:
        raise ValueError("hidden_materialized_slot_inventory_mismatch")
    material_refs = {}
    for original, plan, definition, mapping in zip(frozen["cases"], planned, suite.cases, binding.case_bindings, strict=True):
        expectation = expected.cases[plan.id]
        if any(getattr(plan, key) != original[key] for key in ("family", "cohort", "expected_result", "critical")):
            raise ValueError("hidden_proposal_objective_reclassified")
        if (mapping.scenario_sha256 != original["case_sha256"]
                or mapping.proposal_fault_sha256 != original["fault_contract_sha256"]
                or mapping.case_sha256 != plan.case_sha256 or mapping.expected_sha256 != plan.expected_sha256
                or mapping.actual_fault_sha256 != plan.fault_contract_sha256
                or digest(definition.model_dump()) != plan.case_sha256
                or digest(expectation.model_dump()) != plan.expected_sha256):
            raise ValueError("hidden_scenario_materialization_hash_mismatch")
        scenario = {"normal": "normal", "extension": "normal", "recovery": "recovery"}.get(plan.cohort)
        scenario = scenario or ("clarification" if plan.expected_result == "clarification" else "rejection")
        if (definition.family != plan.family or definition.case_set != "independently_held_hidden_acceptance"
                or definition.scenario != scenario or expectation.result != ("done" if plan.expected_result == "recovered" else plan.expected_result)
                or definition.budget.model_dump() != plan.budget.model_dump()):
            raise ValueError("hidden_runtime_definition_changed")
        data = [m for m in definition.materials if m.role in {"sample", "feature", "unknown"}]
        if (any(m.source_kind != "deidentified_historical" for m in data)
                or not data and plan.cohort not in {"boundary", "extension_safety"}):
            raise ValueError("formal_historical_materials_unavailable")
        for material in definition.materials:
            if material.path in material_refs and material_refs[material.path] != material.sha256:
                raise ValueError("hidden_shared_material_hash_conflict")
            material_refs[material.path] = material.sha256
        if plan.cohort == "recovery":
            expected_fault = SUPPORTED_FAULT_BINDINGS.get(original["spec"]["fault_kind"])
            if plan.family == "validation" and original["spec"]["fault_kind"] == "stale_cas_write":
                expected_fault = "stale_report_revision_once"
            if plan.family == "monitoring" and original["spec"]["fault_kind"] == "stale_cas_write":
                expected_fault = "stale_monitoring_revision_once"
            if expected_fault is None:
                raise ValueError("unsupported_fault_binding")
            actions = [action for action in definition.actions if action.fault is not None]
            faults = [action.fault.kind for action in actions]
            if len(faults) != 1 or faults[0] != expected_fault:
                raise ValueError("hidden_runtime_fault_reclassified")
            fault_action = actions[0]
            if plan.family == "validation":
                expected_action = "start_validation_agent" if expected_fault == "tool_timeout_once" else "confirm_current_validation_report"
                if fault_action.kind != expected_action or fault_action.tool:
                    raise ValueError("hidden_validation_fault_carrier_unavailable")
            elif plan.family == "monitoring":
                if fault_action.kind != "confirm_model_monitoring_plan" or fault_action.tool:
                    raise ValueError("hidden_monitoring_fault_carrier_unavailable")
            elif fault_action.kind != "approve_step" or not fault_action.tool:
                raise ValueError("hidden_native_approval_fault_carrier_unavailable")
            if expected_fault == "worker_crash_after_commit_once":
                producers = {"data_processing": {"data_ops.clean_format", "data_ops.execute_join"},
                             "modeling": {"modeling.configure_tuning"},
                             "risk_portfolio": {"analysis.portfolio_gate_summary"}}
                if (fault_action.tool not in producers.get(plan.family, set())
                        or (plan.family == "risk_portfolio" and fault_action.fault.downstream_tool != "analysis.portfolio_report")
                        or (plan.family == "modeling" and fault_action.fault.downstream_tool != "modeling.train_models")):
                    raise ValueError("hidden_after_commit_family_carrier_unavailable")
            if plan.fault_contract_sha256 != digest(future_fault_contract(definition)):
                raise ValueError("hidden_actual_fault_contract_mismatch")
        if binding.load_policy.warmup.requests >= plan.budget.max_http_requests:
            raise ValueError("hidden_warmup_exhausts_http_budget")
    if material_refs != {m.path: m.sha256 for m in binding.materials}:
        raise ValueError("hidden_material_inventory_mismatch")


def prepare_hidden_execution(freeze_dir: Path, binding_path: Path, *, ledger_path: Path,
                             profile_path: Path, secret_env: dict | None = None) -> PreparedHiddenExecution:
    """Read and authenticate all inputs without creating tasks, outputs or calls."""
    frozen = load_frozen_case_set(Path(freeze_dir))
    profile_path = Path(profile_path).resolve()
    ledger_path = Path(ledger_path).resolve()
    profile_raw = profile_path.read_bytes()
    ledger_raw = ledger_path.read_bytes()
    profile = load_profile(profile_path)
    if profile_path.read_bytes() != profile_raw:
        raise ValueError("hidden_profile_changed_during_preflight")
    if formal_blockers(profile):
        raise ValueError("performance_profile_pending_user_freeze")
    binding_path = Path(binding_path).resolve()
    if binding_path.stat().st_size > 32 * 1024**2:
        raise ValueError("hidden_binding_too_large")
    raw_binding = binding_path.read_bytes()
    binding = FrozenHiddenExecution.model_validate(_json(raw_binding))
    if digest((Path(freeze_dir) / "freeze-manifest.json").read_bytes()) != binding.proposal_manifest_sha256:
        raise ValueError("hidden_proposal_manifest_changed")
    root = binding_path.parent
    loaded = {name: _read(root, getattr(binding.files, name)) for name in type(binding.files).model_fields}
    suite = RuntimeSuite.model_validate(_json(loaded["cases"][1]))
    expected = ExpectedSuite.model_validate(_json(loaded["expected"][1]))
    model = ModelConnection.model_validate(_json(loaded["model"][1]))
    price = PriceBook.model_validate(_json(loaded["price_book"][1]))
    environment = _json(loaded["environment"][1])
    dataset_root = _path(root, binding.dataset_root)
    if not dataset_root.is_dir() or loaded["expected"][0].resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("hidden_expected_must_be_outside_dataset_root")
    if len({path.resolve() for path, _ in loaded.values()}) != len(loaded):
        raise ValueError("hidden_input_roles_must_be_separate")
    from .runtime_runner import _validate_package_isolation
    _validate_package_isolation(Path(__file__).resolve().parents[3] / "marvis", tuple(path for path, _ in loaded.values()))
    _check_cases(frozen, binding, suite, expected)
    for material in binding.materials:
        _read(dataset_root, material, maximum=512 * 1024**2)
    source = _source()
    inventory = _json(loaded["source_inventory"][1])
    package = {name: sha for name, sha in inventory.items() if name.startswith("marvis/")
               and "__pycache__" not in Path(name).parts and Path(name).suffix != ".pyc"}
    if (inventory != source_inventory(Path(__file__).resolve().parents[3])
            or digest(inventory) != binding.benchmark.source_inventory_sha256
            or digest(package) != source["source_sha256"] or source["commit"] != binding.benchmark.commit):
        raise ValueError("hidden_source_identity_changed")
    observed = observed_environment(profile, binding.load_policy)
    if environment != observed:
        raise ValueError("hidden_environment_changed")
    hashes = {f"{key}_sha256": digest(value) for key, value in environment.items()}
    if hashes != binding.benchmark.environment.model_dump() or hashes["hardware_sha256"] != profile["hardware"]["fingerprint_sha256"]:
        raise ValueError("hidden_environment_binding_mismatch")
    if binding.load_policy.concurrent_workers != profile["load"]["concurrent_users"]:
        raise ValueError("hidden_concurrent_load_changed")
    for mode, slo in binding.benchmark.slo.items():
        approved = profile["slo"][mode]
        if slo.minimum_observations != approved["minimum_observations"] or slo.p95_wall_seconds != approved["p95_seconds"]:
            raise ValueError("hidden_slo_changed")
    model.profile("real_model")
    frozen_model = binding.benchmark.model
    if (frozen_model.profile != model.model_id or frozen_model.model_name != model.model_name
            or frozen_model.connection_sha256 != digest(model.model_dump())):
        raise ValueError("hidden_model_connection_changed")
    limits = binding.provider_limits
    if (model.enable_thinking or limits.context_window != model.context_window
            or limits.max_output_tokens != model.max_output_tokens or limits.price_book_sha256 != binding.files.price_book.sha256):
        raise ValueError("hidden_provider_ceiling_or_prices_changed")
    _, approval_raw = _read(root, FileBinding.model_validate(limits.approval.model_dump()))
    approval = ProviderApproval.model_validate(_json(approval_raw))
    _, documentation = _read(root, FileBinding.model_validate(approval.source_document.model_dump()))
    if not documentation.strip():
        raise ValueError("provider_budget_source_document_empty")
    if (approval.model_name != model.model_name or approval.provider != price.provider
            or approval.price_book_sha256 != limits.price_book_sha256
            or any(getattr(approval, key) != getattr(limits, key) for key in ("protocol", "context_window", "max_output_tokens", "framing_token_allowance"))
            or approval.source_scope == "local_test_fixture" and urlsplit(model.api_base_url).hostname not in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("provider_budget_approval_mismatch")
    if not (secret_env if secret_env is not None else os.environ).get(model.api_key_env):
        raise ValueError("missing_explicit_model_credential")
    cap = _json(ledger_raw).get("agent_benchmark_plan", {}).get("cost_cap")
    if (type(cap) is not dict or cap.get("status") != "decided" or type(cap.get("max_cost")) not in {int, float}
            or not math.isfinite(cap["max_cost"]) or cap["max_cost"] < 0
            or type(cap.get("currency")) is not str or not cap["currency"].strip()
            or type(cap.get("price_book")) is not str or not cap["price_book"].strip()):
        raise ValueError("cost_cap_pending_user_decision")
    approved_price = Path(cap["price_book"])
    approved_price = approved_price if approved_price.is_absolute() else Path(ledger_path).parent / approved_price
    if (approved_price.resolve() != loaded["price_book"][0].resolve() or price.model_name != model.model_name
            or price.currency != cap["currency"]
            or any(case.budget.currency != cap["currency"] for case in binding.benchmark.cases)):
        raise ValueError("hidden_price_book_or_currency_changed")
    if cap.get("provider_approval_sha256") != limits.approval.sha256:
        raise ValueError("provider_budget_approval_not_bound_by_cost_authority")
    budgets = {case.id: case.budget for case in binding.benchmark.cases}
    total = sum((Decimal(str(budgets[a.case_id].max_cost)) for a in binding.benchmark.attempts), Decimal(0))
    if total > Decimal(str(cap["max_cost"])):
        raise ValueError("hidden_planned_slots_exceed_approved_total_cost")
    return PreparedHiddenExecution(binding, binding.benchmark, suite, expected, model, price, limits, environment,
        loaded["cases"][0], loaded["expected"][0], dataset_root, loaded["price_book"][0], binding.load_policy,
        digest(raw_binding), float(total), source, ledger_path, digest(ledger_raw), profile_path, digest(profile_raw))
