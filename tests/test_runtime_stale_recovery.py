"""Real local HTTP stale refusal followed by a separately declared current approval."""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_archive_validation_recovery import revalidate_validation_recovery_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from test_runtime_agent_benchmark import fixture_model
from test_runtime_safety_archives import repin_private
from test_runtime_validation_agent import _v2_protocol
from test_runtime_workflow_families import _business_protocol


@pytest.fixture(
    scope="module",
    params=["labeling", pytest.param("validation", marks=pytest.mark.pmml_runtime)],
)
def stale_originals(tmp_path_factory, request):
    family = request.param
    root = tmp_path_factory.mktemp("stale-" + family)
    paths = write_synthetic_suite(root / "suite", **{"normal_" + ("validation_agent" if family == "validation" else "labeling") + "_only": True})
    suite = json.loads(paths["cases"].read_bytes())
    raw = suite["cases"][0]
    raw.update(family=family, scenario="recovery")
    raw["budget"].update(max_total_tokens=100000, max_cost=1.0, currency="USD")
    source = 2 if family == "validation" else next(i for i, a in enumerate(raw["actions"], 1) if a.get("tool") == "labeling.define_label")
    raw["actions"][source - 1]["fault"] = {"kind": "stale_report_revision_once" if family == "validation" else "stale_step_cas_once"}
    followup = {"kind": "recover_validation_report" if family == "validation" else "recover_governed_decision",
                "fault_source_action": source, "content": "已核对当前版本，明确确认本次操作并读取实际结果。"}
    if family == "labeling":
        followup["tool"] = "labeling.define_label"
    raw["actions"].insert(source, followup)
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    case = RuntimeCase.model_validate(raw)
    price = root / "price.json"
    price.write_text(json.dumps({"model_name": "fixture", "provider": "local-protocol-substitute", "currency": "USD",
        "source": "synthetic local rates; no provider billing", "effective_date": "2026-11-04",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}))
    with fixture_model(answer_factory=_v2_protocol if family == "validation" else _business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model", price_book_path=price)
    row = report["cases"][0]
    assert report["all_passed"], row
    archive = root / "custody" / report["run_id"] / case.id
    bindings = json.loads((archive / "bindings.json").read_bytes())
    manifest = json.loads((archive / "manifest.json").read_bytes())
    common = dict(case=case, task_id=row["execution"]["task_id"],
        **{k: bindings[k] for k in ("run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")})
    governed = []
    if family == "labeling":
        with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
            conn.row_factory = sqlite3.Row
            native = dict(conn.execute("SELECT * FROM decision_records WHERE reason=?", (followup["content"],)).fetchone())
        governed = [{"action_index": source + 1, "plan_id": native["plan_id"], "step_id": native["step_id"],
                     "decision_id": native["id"], "decision_record_sha256": digest(native)}]
        from marvis.orchestrator.eval.runtime_archive_labeling import FrozenLabelingBinding
        steps = {"labels" if s["tool"] == "labeling.define_label" else "maturity": s for s in row["execution"]["steps"]
                 if s["tool"] in {"labeling.define_label", "labeling.check_cohort_maturity"}}
        business = FrozenLabelingBinding(**common, plan_id=native["plan_id"], step_ids={k: v["id"] for k, v in steps.items()},
            output_sha256={k: v["output_sha256"] for k, v in steps.items()},
            download_sha256={kind: next(e["sha256"] for e in row["http_events"] if e["stage"] == "download_labeling_" + kind) for kind in ("dataset", "evidence")})
    else:
        from marvis.orchestrator.eval.runtime_archive_validation import FrozenValidationBinding
        pipeline = row["execution"]["validation_pipeline"]
        business = FrozenValidationBinding(**common, input_contract_sha256=pipeline["input_contract_sha256"],
            confirmed_draft_sha256=pipeline["confirmed_draft_sha256"], report_revision=pipeline["report_revision"],
            report_sha256={r["kind"]: r["sha256"] for r in pipeline["report_files"]}, bin_count=10)
    safety = FrozenSafetyBinding(**common, recovery_action_indices=[source + 1], governed_actions=governed,
        before_manifest_sha256=manifest["safety_before"]["manifest_sha256"])
    return archive, digest((archive / "manifest.json").read_bytes()), safety, business, paths


def _read(archive, sha, binding):
    reader = revalidate_validation_recovery_archive if binding.case.family == "validation" else revalidate_recovery_archive
    return reader(archive, expected_manifest_sha256=sha, frozen_binding=binding)


@pytest.mark.e2e
def test_stale_refusal_then_actual_current_approval_is_verified(stale_originals):
    archive, sha, safety, _, _ = stale_originals
    result = _read(archive, sha, safety)
    assert result["status"] == "verified", result


@pytest.mark.e2e
def test_formal_stale_recovery_requires_actual_family_math(stale_originals, tmp_path):
    from marvis.orchestrator.eval.formal_benchmark import FormalCase
    from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract, retained_safety
    from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
    from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
    from test_acceptance_trust import put
    from test_formal_benchmark import frozen_fixture
    archive, sha, safety, business, paths = stale_originals
    shutil.copytree(archive, tmp_path / "archive")
    safety_ref, business_ref = put(tmp_path, "safety.json", safety.model_dump()), put(tmp_path, "business.json", business.model_dump())
    public = archive.parents[2] / "public" / safety.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json(paths["expected"].read_bytes()).cases[safety.case.id]
    template = next(c for c in frozen_fixture().cases if c.family == safety.case.family and c.cohort == "recovery")
    case = FormalCase.model_validate({**template.model_dump(), "id": safety.case.id, "budget": safety.case.budget.model_dump(),
        "case_sha256": digest(safety.case.model_dump()), "expected_sha256": digest(expected.model_dump()),
        "fault_contract_sha256": digest(future_fault_contract(safety.case))})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": safety_ref}
    record = {"input_hashes": {r["path"]: r["sha256"] for r in (safety_ref, business_ref)}}
    result = retained_safety(case, expected, item, tmp_path, record, run, domain_item={**item, "binding": business_ref})
    assert result["domain"]["domain_status"] == "verified", result
    assert result["domain"]["actual_result"] == "recovered", result


@pytest.mark.e2e
@pytest.mark.parametrize("damage", ["missing_recovery", "stale_effect", "approval_timeline", "other_action", "other_target", "repeat_post"])
def test_repin_cannot_hide_stale_recovery_original_mismatch(stale_originals, tmp_path, damage):
    from marvis.orchestrator.eval.runtime_native_json import decode_native
    archive, _, safety, _, _ = stale_originals
    target = tmp_path / "archive"
    shutil.copytree(archive, target)
    private = decode_native(json.loads((target / "private.json").read_bytes()))
    validation = safety.case.family == "validation"
    fault = private["validation_faults" if validation else "runtime_faults"][0]
    recovery = fault["recovery"]
    if damage == "missing_recovery":
        del fault["recovery"]
    elif damage == "stale_effect":
        if validation:
            fault["after_response"]["state"]["task"]["report_values_revision"] += 1
        else:
            fault["after_response"]["target"]["confirmed"] = 1
    elif damage == "approval_timeline":
        recovery["before_confirmation" if validation else "before_approval"]["started_elapsed_ns"] = 0
    elif damage == "other_action":
        recovery["action_index"] -= 1
    elif damage == "other_target":
        if validation:
            recovery["confirmation"]["request"]["draft_message_id"] = "wrong-draft"
        else:
            private["human_decisions"][-1]["request"]["expected_step_fingerprint"] = "0" * 64
    else:
        binding = json.loads((target / "bindings.json").read_bytes())
        events = binding["execution_before_scoring"]["http_events"]
        event = next(e for e in events if e["ordinal"] > fault["http_ordinal"] and e["method"] == "GET")
        event["method"] = "POST"
        (target / "bindings.json").write_text(json.dumps(binding))
    sha = repin_private(target, private)
    result = _read(target, sha, safety)
    assert result["status"] in {"mismatch", "unavailable"}, result
