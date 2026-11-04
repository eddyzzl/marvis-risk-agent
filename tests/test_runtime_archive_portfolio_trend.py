"""Actual HTTP trend production against a caller-frozen retained baseline.

The fixture registers an explicitly supplied historical distribution before the
journey. It does not fabricate trend outputs or claim to audit model training.
"""
from contextlib import closing, contextmanager
import json
import sqlite3

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_archive_portfolio import (
    FrozenPortfolioBaseline, FrozenPortfolioBinding, _TOOLS, revalidate_portfolio_archive,
)
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from test_runtime_agent_benchmark import fixture_model
from test_runtime_workflow_families import _business_protocol


@pytest.fixture(scope="module")
def trend_archive(tmp_path_factory):
    import marvis.orchestrator.eval.runtime_runner as runner
    from marvis.domain import TaskCreate
    from marvis.packs.modeling.contracts import Experiment, ModelArtifact, TrainConfig
    from marvis.repositories.modeling import ModelingRepository
    from marvis.repositories.tasks import TaskRepository

    root = tmp_path_factory.mktemp("portfolio-trend-original")
    paths = write_synthetic_suite(root / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_portfolio")
    case["actions"][0]["portfolio_request"].update(experiment_id="frozen_experiment", score_col="score")
    case["actions"].append({"kind": "download_portfolio_report", "content": "下载组合报告。"})
    material = case["materials"][0]
    path = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(path)
    frame["score"] = [0.1 if i % 3 else 0.9 for i in range(len(frame))]
    frame.to_parquet(path, index=False)
    material["sha256"] = digest(path.read_bytes())
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    original = runner._application
    frozen = {}

    @contextmanager
    def with_retained_baseline(*args, **kwargs):
        with original(*args, **kwargs) as app:
            client, workspace, deadline = app
            db = workspace / "marvis.sqlite"
            task = TaskRepository(db).create_task(TaskCreate(model_name="historical-baseline", model_version="1",
                validator="synthetic", source_dir="", task_type="modeling", run_mode="manual"))
            repo = ModelingRepository(db)
            config = TrainConfig(dataset_id="historical", features=("score",), target_col="y", split_col="split",
                split_values={"train": "train"}, params={}, seed=1, early_stopping_rounds=None)
            repo.create_experiment(Experiment(id="frozen_experiment", task_id=task.id, recipe_id="lr", config=config,
                metrics=None, artifact_id="frozen_baseline", status="succeeded", created_at="2026-01-01T00:00:00Z"))
            repo.create_model_artifact(ModelArtifact(id="frozen_baseline", experiment_id="frozen_experiment", algorithm="lr",
                model_path="", pmml_path=None, feature_list=("score",), params={}, woe_maps=None,
                created_at="2026-01-01T00:00:00Z", baseline_distributions={
                    "score_edges": [float("-inf"), 0.5, float("inf")],
                    "score_distribution": {"train": {"bin_proportions": [0.5, 0.5]}}}))
            with closing(sqlite3.connect(db)) as conn:
                raw = conn.execute("SELECT baseline_distributions_json FROM model_artifacts WHERE id='frozen_baseline'").fetchone()[0]
            frozen.update(experiment_id="frozen_experiment", artifact_id="frozen_baseline", task_id=task.id,
                          baseline_sha256=digest(raw.encode()))
            yield client, workspace, deadline

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_application", with_retained_baseline)
        with fixture_model(answer_factory=_business_protocol) as (model, _):
            report = runner.run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
                dataset_root=paths["dataset_root"], output_dir=root / "public", evidence_custody_dir=root / "custody",
                model=model, model_source="fixture_model")
    assert report["all_passed"], report["cases"][0]
    observed = report["cases"][0]
    tools = {**_TOOLS, "trend": "analysis.score_stability_trend"}
    steps = {key: next(s for s in observed["execution"]["steps"] if s["tool"] == tool) for key, tool in tools.items()}
    archive = root / "custody" / report["run_id"] / case["id"]
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        source = json.loads(conn.execute("SELECT input_json FROM plan_step_runs WHERE step_id=? AND status='succeeded'",
                                        (steps["flow"]["id"],)).fetchone()[0])
    binding = FrozenPortfolioBinding(case=RuntimeCase.model_validate(case), run_id=report["run_id"],
        task_id=observed["execution"]["task_id"], plan_id=observed["execution"]["plans"][0]["id"],
        dataset_id=source["dataset_id"], dataset_sha256=source["expected_content_hash"],
        state_action_index=1, states=["current", "M1", "charged_off"],
        step_ids={key: step["id"] for key, step in steps.items()},
        output_sha256={key: step["output_sha256"] for key, step in steps.items()},
        download_sha256=next(e["sha256"] for e in observed["http_events"] if e["stage"] == "download_portfolio_report"),
        cases_sha256=report["cases_sha256"], expected_sha256=report["expected_sha256"], source=report["source"],
        model_connection_sha256=report["model_connection_sha256"], trend_baseline=FrozenPortfolioBaseline(**frozen))
    return archive, observed["evidence_custody"]["manifest_sha256"], binding


def test_score_trend_and_workbook_recomputed_without_producer(trend_archive, monkeypatch):
    import marvis.packs.analysis.trend as producer
    import marvis.validation.binning as bins
    for module, name in ((producer, "score_stability_trend"), (bins, "bin_distribution"), (bins, "compute_psi")):
        monkeypatch.setattr(module, name, lambda *a, **k: pytest.fail("producer used by independent adjudicator"))
    archive, sha, binding = trend_archive
    result = revalidate_portfolio_archive(archive, expected_manifest_sha256=sha, frozen_binding=binding)
    assert result["supported_checks_verified"], result["checks"]
    assert result["checks"]["native_steps"]["steps"] == 7
    assert len(result["values"]["trend"]["trend"]) >= 2
    assert result["checks"]["workbook_cells"]["sheets"] == 7


def test_changed_external_baseline_identity_fails_closed(trend_archive):
    archive, sha, binding = trend_archive
    changed = binding.model_copy(update={"trend_baseline": binding.trend_baseline.model_copy(update={"baseline_sha256": "f" * 64})})
    result = revalidate_portfolio_archive(archive, expected_manifest_sha256=sha, frozen_binding=changed)
    assert result["checks"]["trend_baseline"]["status"] == "mismatch"
