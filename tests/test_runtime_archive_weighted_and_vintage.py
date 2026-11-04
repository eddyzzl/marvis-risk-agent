"""Weighted modeling and standard Vintage originals are adjudicated, not marked unsupported."""
import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_modeling import revalidate_modeling_archive
from marvis.orchestrator.eval.runtime_archive_vintage import FrozenVintageBinding, revalidate_vintage_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def weighted_archive(tmp_path_factory):
    import test_runtime_domain_archives as dom

    root = tmp_path_factory.mktemp("weighted-modeling-custody")
    paths = write_synthetic_suite(root / "suite", normal_modeling_only=True)
    suite = json.loads(paths["cases"].read_text())
    case = suite["cases"][0]
    material = case["materials"][0]
    path = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(path)
    frame["w"] = [1.0 + (i % 3) * 0.5 for i in range(len(frame))]
    frame.to_parquet(path, index=False)
    material["sha256"] = _sha(path)
    case["initial_message"] += "使用 w 列作为样本权重。"
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))

    def protocol(request, answer, payload):
        out = dom._protocol(request, answer, payload)
        if isinstance(out, dict) and out.get("action") == "adjust" and "recipes" in (out.get("params") or {}):
            out = {**out, "params": {**out["params"], "sample_weight_col": "w"}}
        return out

    with fixture_model(answer_factory=protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"], report["cases"][0].get("error_code")
    archive = root / "custody" / report["run_id"] / report["cases"][0]["case_id"]
    binding = dom._model_binding(archive, json.loads((archive / "case.json").read_text()))
    return archive, _sha(archive / "manifest.json"), binding


def test_weighted_linear_fit_and_weighted_metrics_are_recomputed(weighted_archive, monkeypatch):
    import marvis.feature.metrics as metrics

    for name in ("weighted_feature_ks", "weighted_feature_auc", "weighted_feature_psi"):
        monkeypatch.setattr(metrics, name, lambda *a, **k: pytest.fail("producer weighted metric used"))
    archive, sha, binding = weighted_archive
    result = revalidate_modeling_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    values = result["values"]["metrics"]
    assert {key for key in values if key.startswith("weighted_")} == {
        "weighted_train_ks", "weighted_test_ks", "weighted_oot_ks", "weighted_train_auc", "weighted_test_auc",
        "weighted_oot_auc", "weighted_psi_test_vs_train", "weighted_psi_oot_vs_train"}
    assert "weighted_calibrated" not in result["unsupported"]["model_profiles"]


def test_weighted_metric_disagreement_is_a_mismatch(weighted_archive, monkeypatch):
    import marvis.orchestrator.eval.runtime_modeling_extended_reference as reference

    original = reference.weighted_binary_reference
    monkeypatch.setattr(reference, "weighted_binary_reference",
                        lambda *a: {**original(*a), "ks": original(*a)["ks"] + 0.01})
    archive, sha, binding = weighted_archive
    result = revalidate_modeling_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["checks"]["model_closure_fit_and_metrics"]["status"] == "mismatch"


@pytest.fixture(scope="module")
def vintage_archive(tmp_path_factory):
    from test_runtime_workflow_families import _business_protocol

    root = tmp_path_factory.mktemp("vintage-custody")
    paths = write_synthetic_suite(root / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_text())
    suite["cases"] = [c for c in suite["cases"] if c["id"] == "synthetic_vintage"]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
            model=model, model_source="fixture_model")
    assert report["all_passed"]
    observed = report["cases"][0]
    archive = root / "custody" / report["run_id"] / observed["case_id"]
    step = next(s for s in observed["execution"]["steps"] if s["tool"] == "strategy.vintage_curve")
    with sqlite3.connect(archive / "workspace/marvis.sqlite") as conn:
        dataset_id, sha = conn.execute("SELECT id, content_hash FROM datasets WHERE role='sample'").fetchone()
    binding = FrozenVintageBinding(case=RuntimeCase.model_validate_json((archive / "case.json").read_bytes()),
        run_id=report["run_id"], task_id=observed["execution"]["task_id"], plan_id=observed["execution"]["plans"][0]["id"],
        step_ids={"vintage": step["id"]}, output_sha256={"vintage": step["output_sha256"]},
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], dataset_id=dataset_id, dataset_sha256=sha)
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


def test_standard_vintage_is_recomputed_without_the_producer(vintage_archive, monkeypatch):
    import marvis.packs.strategy.vintage as producer
    import marvis.validation.vintage as kernel

    monkeypatch.setattr(producer, "vintage_curve", lambda *a, **k: pytest.fail("producer Vintage used"))
    monkeypatch.setattr(kernel, "compute_vintage_curve", lambda *a, **k: pytest.fail("producer Vintage used"))
    archive, sha, binding = vintage_archive
    result = revalidate_vintage_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["checks"]["independent_vintage"]["cohorts"] == 2
    assert result["checks"]["vintage_contract"]["label_semantics"] == "incremental"


def test_vintage_curve_disagreement_is_a_mismatch(vintage_archive, monkeypatch):
    import marvis.orchestrator.eval.runtime_archive_vintage as reader

    original = reader.vintage_reference
    monkeypatch.setattr(reader, "vintage_reference", lambda rows, **kw: original(rows, **{**kw, "mob_max": 1}))
    archive, sha, binding = vintage_archive
    result = revalidate_vintage_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["checks"]["independent_vintage"]["status"] == "mismatch"
