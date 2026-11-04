"""Whole HTTP feature journeys retain enough originals for optional adjudication."""
from contextlib import closing
import json
import sqlite3

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_feature import FrozenFeatureBinding, _TOOLS, revalidate_feature_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model


def _build(root, *, dictionary, recovery=False):
    paths = write_synthetic_suite(root / "suite")
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_feature")
    case["task"]["metrics"] = ["iv", "ks", "auc", "coverage", "head_tail_lift", "meaning_consistency",
        "psi_month_first", "psi_month_last", "psi_month_previous", "psi_split"]
    material = case["materials"][0]
    path = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(path)
    frame["apply_month"] = ["2026-01", "2026-02", "2026-03"] * (len(frame) // 3) + ["2026-03"] * (len(frame) % 3)
    frame["split"] = ["train", "test", "oot"] * (len(frame) // 3) + ["oot"] * (len(frame) % 3)
    frame.to_parquet(path, index=False)
    material["sha256"] = digest(path.read_bytes())
    if dictionary:
        path = paths["dataset_root"] / "data_dictionary.csv"
        pd.DataFrame({"feature": ["balance", "age"], "meaning": ["贷款余额", "年龄"], "importance": [1, 1]}).to_csv(path, index=False)
        case["materials"].append({"path": path.name, "sha256": digest(path.read_bytes()), "role": "dictionary"})
    if recovery:
        index = next(i for i, action in enumerate(case["actions"]) if action["kind"] == "approve_step")
        case["actions"][index]["fault"] = {"kind": "discard_success_response_once"}
        case["actions"].insert(index + 1, {"kind": "recover_governed_decision", "tool": case["actions"][index]["tool"],
            "content": "读取同一次已确认特征分箱的执行结果，不重复提交。", "fault_source_action": index + 1})
        case["scenario"] = "recovery"
    case["actions"].append({"kind": "download_feature_report", "content": "下载特征分析报告。"})
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    with fixture_model() as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report["cases"][0], ensure_ascii=False)
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case["id"]
    steps = {key: next(s for s in observed["execution"]["steps"] if s["tool"] == tool) for key, tool in _TOOLS.items()}
    extra = {}
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        dataset_id, sha = conn.execute("SELECT id,content_hash FROM datasets WHERE role='sample'").fetchone()
        inputs = json.loads(conn.execute("SELECT input_json FROM plan_step_runs WHERE step_id=? AND status='succeeded'",
                                        (steps["metrics"]["id"],)).fetchone()[0])
        extra["meaning_directions"] = inputs["meaning_directions"]
        if dictionary:
            extra["dictionary_id"], extra["dictionary_sha256"] = conn.execute(
                "SELECT id,content_hash FROM datasets WHERE role IN ('feature_dictionary','data_dictionary')").fetchone()
    binding = FrozenFeatureBinding(case=RuntimeCase.model_validate_json((archive / "case.json").read_bytes()),
        run_id=report["run_id"], task_id=observed["execution"]["task_id"], plan_id=observed["execution"]["plans"][0]["id"],
        dataset_id=dataset_id, dataset_sha256=sha, step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        download_sha256=next(e["sha256"] for e in observed["http_events"] if e["stage"] == "download_feature_report"),
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], **extra)
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


@pytest.fixture(scope="module", params=[False, True], ids=["no-dictionary", "with-dictionary"])
def feature_extended_archive(tmp_path_factory, request):
    return _build(tmp_path_factory.mktemp("feature-extended"), dictionary=request.param)


def test_optional_metrics_and_report_from_actual_http_originals(feature_extended_archive, monkeypatch):
    import marvis.feature.metrics as production
    import marvis.packs.feature.tools as tools

    def forbidden(*a, **kw):
        pytest.fail("producer computation called by independent adjudicator")

    for module, names in ((production, ("head_tail_lift", "feature_psi")),
                          (tools, ("_feature_psi_series", "_attach_meaning_consistency", "_feature_recommendation"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    archive, sha, binding = feature_extended_archive
    result = revalidate_feature_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["checks"]["workbook_cells"]["sheets"] == 2
    row = result["values"]["metrics"][0]
    assert len(row["psi_split_series"]) == 2
    assert row["lift_head_5"] is not None
    assert "meaning_consistency" in row


def test_changed_independent_lift_is_detected(feature_extended_archive, monkeypatch):
    import marvis.orchestrator.eval.runtime_feature_extended_reference as reference

    original = reference.head_tail_lift_reference
    monkeypatch.setattr(reference, "head_tail_lift_reference", lambda *a, **k: {**original(*a, **k), "lift_head_5": 999.0})
    archive, sha, binding = feature_extended_archive
    result = revalidate_feature_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["checks"]["independent_metrics"]["status"] == "mismatch"


def test_feature_dictionary_role_is_scoped_to_feature_or_validation():
    from pydantic import ValidationError

    raw = {"id": "dictionary-scope", "revision": "1", "family": "feature_analysis",
           "task": {"task_type": "feature_analysis"}, "business_constraints_source": "explicit test",
           "materials": [{"path": "data_dictionary.csv", "sha256": "a" * 64, "role": "dictionary"}]}
    assert RuntimeCase.model_validate(raw).materials[0].role == "dictionary"
    for kind in ("portfolio", "vintage", "data_join", "strategy", "modeling"):
        with pytest.raises(ValidationError):
            RuntimeCase.model_validate({**raw, "task": {"task_type": kind}})
    for role in ("notebook", "pmml"):
        with pytest.raises(ValidationError):
            RuntimeCase.model_validate({**raw, "materials": [{**raw["materials"][0], "role": role}]})


def test_feature_discard_response_recovers_from_actual_http_originals(tmp_path):
    from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
    from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding

    archive, sha, binding = _build(tmp_path, dictionary=False, recovery=True)
    private = json.loads((archive / "private.json").read_bytes())
    manifest = json.loads((archive / "manifest.json").read_bytes())
    governed = []
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        for capture in private["human_decisions"]:
            native = dict(conn.execute("SELECT * FROM decision_records WHERE id=?", (capture["response"]["decision_id"],)).fetchone())
            governed.append({"action_index": capture["case_action_index"], "plan_id": native["plan_id"],
                "step_id": native["step_id"], "decision_id": native["id"], "decision_record_sha256": digest(native)})
    fields = {key: getattr(binding, key) for key in ("case", "task_id", "run_id", "cases_sha256", "expected_sha256", "source", "model_connection_sha256")}
    safety = FrozenSafetyBinding(**fields, governed_actions=governed,
        before_manifest_sha256=manifest["safety_before"]["manifest_sha256"],
        recovery_action_indices=[i for i, action in enumerate(binding.case.actions, 1) if action.kind == "recover_governed_decision"])
    result = revalidate_recovery_archive(archive, expected_manifest_sha256=sha, frozen_binding=safety)
    assert result["status"] == "verified", result
    report = revalidate_feature_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert report["supported_checks_verified"], report["checks"]
    (tmp_path / "feature-recovery-revalidation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
