from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import nbformat

from marvis.notebook_contract import RuntimeContract
from marvis.notebooks import NotebookExecutionSession
from marvis.pipeline_cellgen import _build_metrics_cell_sources
from marvis.pipeline_io import _metrics_cancel_marker_path
from marvis.validation import effectiveness
from marvis.validation_services.metrics import write_platform_validation_metrics


@pytest.fixture
def notebook_metrics(tmp_path):
    sample = pd.DataFrame(
        {
            "x": [0.1, 0.4, 0.2, 0.8] * 3,
            "y": [0, 1, 0, 1] * 3,
            "split": [split for split in ("train", "test", "oot") for _ in range(4)],
            "month": ["202601"] * 4 + ["202602"] * 4 + ["202603"] * 4,
        },
        index=range(100, 112),
    )
    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame({"feature": ["x"], "category": ["local"]}).to_csv(
        dictionary_path, index=False
    )
    code_scores_path = tmp_path / "scores.csv"
    pd.DataFrame({"row_index": sample.index, "code_model_score": sample.x}).to_csv(
        code_scores_path, index=False
    )
    model_meta_path = tmp_path / "model_meta.json"
    model_meta_path.write_text(json.dumps({"hyperparameters": {}, "feature_importance": []}))
    reproducibility_path = tmp_path / "reproducibility.json"
    reproducibility_path.write_text(json.dumps({
        "sample_size": 12, "seed": 42, "rows": [],
        "summary": {"match_count": 12, "mismatch_count": 0, "max_abs_diff": 0, "status": "pass"},
    }))
    cells = _build_metrics_cell_sources(
        package_root=Path(__file__).resolve().parents[1],
        task_dir=tmp_path,
        task=SimpleNamespace(model_name="fixture", model_version="v1", split_col="split", time_col="month"),
        settings=SimpleNamespace(bin_count=2, random_sample_size=12, random_seed=42,
                                 data_dict_feature_col="feature", data_dict_category_col="category"),
        dictionary_path=dictionary_path,
        input_pmml_path=tmp_path / "unused.pmml",
        contract=RuntimeContract(
            target_col="y", split_col="split", time_col="month",
            pmml_output_field="probability_1", score_decimal_places=6,
            code_model_scores_path=code_scores_path, feature_importance_path=None,
            model_params_path=None, algorithm="lgb",
        ),
        model_meta_path=model_meta_path,
        reproducibility_json_path=reproducibility_path,
        results_json_path=tmp_path / "results.json",
        excel_path=tmp_path / "validation.xlsx",
    )
    scorer_inputs = []

    def score(frame):
        scorer_inputs.append(frame.copy())
        return frame.x

    return cells, {"RMC_SAMPLE_DF": sample, "RMC_SCORE_FN": score}, scorer_inputs


@pytest.mark.parametrize(("stage", "kernel"), [
    ("metrics-ks", "compute_overall_ks"),
    ("metrics-psi", "compute_overall_psi"),
    ("metrics-binning", "compute_bin_tables"),
])
def test_notebook_cancellation_stops_inside_effectiveness_stage(
    notebook_metrics, tmp_path, monkeypatch, stage, kernel
):
    """A cancellation arriving after a subcalculation stops that same cell."""
    cells, namespace, _ = notebook_metrics
    original = getattr(effectiveness, kernel)

    def request_cancel(**kwargs):
        result = original(**kwargs)
        marker = _metrics_cancel_marker_path(tmp_path)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("cancel")
        return result

    monkeypatch.setattr(effectiveness, kernel, request_cancel)
    for name, source in cells:
        if name == stage:
            with pytest.raises(KeyboardInterrupt, match="metrics cancelled"):
                exec(compile(source, f"<{name}>", "exec"), namespace)
            break
        exec(compile(source, f"<{name}>", "exec"), namespace)
    assert "_rmc_effectiveness" not in namespace
    assert not (tmp_path / "results.json").exists()


def test_notebook_effectiveness_uses_live_frame_scorer_and_preserves_progress(notebook_metrics):
    cells, namespace, scorer_inputs = notebook_metrics
    assert [name for name, _ in cells] == [
        "metrics-prepare", "metrics-score", "metrics-basic", "metrics-ks",
        "metrics-psi", "metrics-binning", "metrics-stress", "metrics-output",
    ]
    original_sample = namespace["RMC_SAMPLE_DF"].copy(deep=True)
    for name, source in cells:
        if name == "metrics-stress":
            break
        exec(compile(source, f"<{name}>", "exec"), namespace)
    expected = effectiveness.run_effectiveness(
        sample=namespace["_rmc_sample_scored"], config=namespace["_rmc_config"]
    )
    assert asdict(namespace["_rmc_effectiveness"]) == asdict(expected)
    assert len(scorer_inputs) == 1
    pd.testing.assert_frame_equal(scorer_inputs[0], original_sample)
    pd.testing.assert_frame_equal(namespace["RMC_SAMPLE_DF"], original_sample)


def test_legacy_metrics_resolve_conflicts_before_loading_stress_artifact(notebook_metrics, tmp_path):
    """Moving artifact reads to the service must preserve validation order."""
    _, namespace, _ = notebook_metrics
    sample = namespace["RMC_SAMPLE_DF"].reset_index(drop=True)
    sample_path = tmp_path / "sample.csv"
    sample.to_csv(sample_path, index=False)
    pd.DataFrame({"row_index": sample.index, "code_model_score": sample.x}).to_csv(
        tmp_path / "scores.csv", index=False
    )
    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame({"feature": ["x", "x"], "category": ["local", "other"]}).to_csv(
        dictionary_path, index=False
    )
    stress_path = tmp_path / "broken-stress.json"
    stress_path.write_text("invalid JSON")
    with pytest.raises(ValueError, match="stress category conflict for x"):
        write_platform_validation_metrics(
            task=SimpleNamespace(model_name="fixture", model_version="v1", split_col="split", time_col="month"),
            settings=SimpleNamespace(bin_count=2, random_sample_size=12, random_seed=42,
                                     data_dict_feature_col="feature", data_dict_category_col="category"),
            contract=RuntimeContract(
                target_col="y", split_col="split", time_col="month",
                pmml_output_field="probability_1", score_decimal_places=6,
                code_model_scores_path=tmp_path / "scores.csv", feature_importance_path=None,
                model_params_path=None, algorithm="lgb",
            ),
            fallback_sample_path=sample_path,
            dictionary_path=dictionary_path,
            model_meta_path=tmp_path / "model_meta.json",
            reproducibility_json_path=tmp_path / "reproducibility.json",
            results_json_path=tmp_path / "results.json",
            excel_path=tmp_path / "validation.xlsx",
            stress_scores_path=stress_path,
        )
    assert not (tmp_path / "results.json").exists()


def test_live_kernel_can_repeat_existing_effectiveness_cells(notebook_metrics, tmp_path):
    cells, namespace, _ = notebook_metrics
    sample = namespace["RMC_SAMPLE_DF"]
    sources = [
        "import pandas as pd\n"
        f"RMC_SAMPLE_DF = pd.DataFrame({sample.to_dict(orient='list')!r}, index={sample.index.tolist()!r})\n"
        "score_calls = 0\n"
        "def RMC_SCORE_FN(frame):\n"
        "    global score_calls\n"
        "    score_calls += 1\n"
        "    return frame.x\n"
    ]
    indexes = {}
    for name, source in cells:
        if name == "metrics-stress":
            break
        indexes[name] = len(sources)
        sources.append(source)
    check_index = len(sources)
    sources.append(
        "from marvis.validation.effectiveness import run_effectiveness\n"
        "assert _rmc_effectiveness == run_effectiveness(sample=_rmc_sample_scored, config=_rmc_config)\n"
        "assert score_calls == 1\n"
    )
    notebook_path = tmp_path / "repeat-metrics.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(s) for s in sources]), notebook_path)
    session = NotebookExecutionSession(
        notebook_path=notebook_path,
        executed_path=tmp_path / "executed.ipynb",
        log_path=tmp_path / "kernel.log",
        timeout=60,
    )
    try:
        result = session.execute_notebook(keep_alive=True)
        assert result.succeeded, result
        for repeated in ("metrics-psi", "metrics-ks", "metrics-binning"):
            result = session.execute_existing_code_cell(indexes[repeated])
            assert result.succeeded, result
            if repeated != "metrics-binning":
                session.notebook.cells[check_index].source = "assert '_rmc_effectiveness' not in globals()"
                assert session.execute_existing_code_cell(check_index).succeeded
                session.notebook.cells[check_index].source = sources[check_index]
            if repeated == "metrics-ks":
                assert session.execute_existing_code_cell(indexes["metrics-psi"]).succeeded
            if repeated != "metrics-binning":
                assert session.execute_existing_code_cell(indexes["metrics-binning"]).succeeded
            assert session.execute_existing_code_cell(check_index).succeeded
    finally:
        session.close()


def test_repeated_notebook_psi_cancellation_clears_old_result_and_can_resume(
    notebook_metrics, tmp_path, monkeypatch
):
    cells, namespace, _ = notebook_metrics
    by_name = dict(cells)
    for name, source in cells:
        if name == "metrics-stress":
            break
        exec(compile(source, f"<{name}>", "exec"), namespace)
    expected = namespace["_rmc_effectiveness"]
    original = effectiveness.compute_overall_psi
    marker = _metrics_cancel_marker_path(tmp_path)

    def request_cancel(**kwargs):
        result = original(**kwargs)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("cancel")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(effectiveness, "compute_overall_psi", request_cancel)
        with pytest.raises(KeyboardInterrupt, match="metrics cancelled"):
            exec(compile(by_name["metrics-psi"], "<metrics-psi>", "exec"), namespace)
    assert "_rmc_effectiveness" not in namespace
    marker.unlink()
    for name in ("metrics-psi", "metrics-binning"):
        exec(compile(by_name[name], f"<{name}>", "exec"), namespace)
    assert namespace["_rmc_effectiveness"] == expected
