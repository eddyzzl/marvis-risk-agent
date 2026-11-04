"""Standard Vintage source custody and explicitly confirmed missing-label policy."""
from contextlib import closing
import json
import sqlite3

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_vintage import FrozenVintageBinding, revalidate_vintage_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


@pytest.fixture(scope="module")
def missing_label_vintage(tmp_path_factory):
    root = tmp_path_factory.mktemp("vintage-missing-label-original")
    paths = write_synthetic_suite(root / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_vintage")
    path = paths["dataset_root"] / case["materials"][0]["path"]
    frame = pd.read_parquet(path)
    frame.loc[0, "bad"] = float("nan")
    frame.to_parquet(path, index=False)
    case["materials"][0]["sha256"] = digest(path.read_bytes())
    case["actions"].append({"kind": "retry_step", "tool": "strategy.vintage_curve",
        "drop_nan_labels": True, "content": "明确剔除目标 bad 为空的 1 行后重试，保留原样本。"})
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], json.dumps(report["cases"][0], ensure_ascii=False)
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / case["id"]
    step = next(s for s in observed["execution"]["steps"] if s["tool"] == "strategy.vintage_curve")
    with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True)) as conn:
        dataset, sha = conn.execute("SELECT id, content_hash FROM datasets WHERE role='sample'").fetchone()
    binding = FrozenVintageBinding(case=RuntimeCase.model_validate(case), run_id=report["run_id"],
        task_id=observed["execution"]["task_id"], plan_id=observed["execution"]["plans"][0]["id"],
        step_ids={"vintage": step["id"]}, output_sha256={"vintage": step["output_sha256"]},
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], dataset_id=dataset, dataset_sha256=sha)
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


def test_missing_label_exclusion_and_full_expected_values_are_independent(missing_label_vintage):
    archive, sha, binding = missing_label_vintage
    result = revalidate_vintage_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["values"]["nan_labels_dropped"] == 1
    assert set(result["values"]) == {"cohorts", "mob_axis", "curves", "counts", "summary", "label_semantics", "nan_labels_dropped", "warnings"}
    assert result["checks"]["source_material"]["status"] == "verified"


def test_explicit_retry_label_policy_and_legacy_identity():
    from pydantic import ValidationError
    from marvis.orchestrator.eval.runtime_contracts import RuntimeAction

    legacy = {"kind": "retry_step", "tool": "strategy.vintage_curve", "content": "重试"}
    assert RuntimeAction.model_validate(legacy).model_dump() == legacy
    assert RuntimeAction.model_validate({**legacy, "drop_nan_labels": True}).drop_nan_labels is True
    for changed in ({"kind": "message"}, {"content": ""}, {"drop_nan_labels": "true"}, {"drop_nan_labels": 1}):
        with pytest.raises(ValidationError):
            RuntimeAction.model_validate({**legacy, "drop_nan_labels": True, **changed})
