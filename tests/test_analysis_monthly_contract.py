"""Calendar duration must survive adjacent-snapshot portfolio analysis."""

import json
import sqlite3

import pandas as pd
import pytest

from marvis.data.errors import PerformanceFrameError
from marvis.packs.analysis.flow import flow_rate, bucket_migration
from marvis.packs.analysis.loss import expected_loss_estimate
from marvis.plugins.manifest import ToolRef
from test_analysis_pack import _runtime, _register


def _panel(months, buckets):
    return pd.DataFrame({"loan": ["A"] * len(months), "month": months,
                         "bucket": buckets, "balance": [100.0] * len(months)})


def _inputs():
    return {"id_col": "loan", "snapshot_col": "month", "bucket_col": "bucket",
            "balance_col": "balance", "states": ["current", "bad"]}


def test_two_month_transition_is_visible_but_never_claimed_as_monthly_loss_probability():
    frame = _panel(["2025-01", "2025-03"], ["current", "bad"])
    flow = flow_rate(frame, **_inputs())
    assert flow.transitions[0].to_month == "2025-03"
    assert flow.transitions[0].from_to_matrix[0] == [0.0, 1.0, 0.0]
    gaps = [flag for flag in flow.red_flags if flag["kind"] == "month_gap"]
    assert len(gaps) == 1
    assert (gaps[0]["month"], gaps[0]["to_month"], gaps[0]["interval_months"]) == ("2025-01", "2025-03", 2)
    migration = bucket_migration(frame, **_inputs())
    assert gaps[0] in migration.red_flags
    with pytest.raises(PerformanceFrameError) as error:
        expected_loss_estimate(frame, **_inputs(), loss_state="bad", lgd=0.5, horizon_months=1)
    assert error.value.problem == "non_monthly_migration"


def test_explicit_matrix_window_can_use_only_actual_monthly_intervals():
    frame = _panel(["2025-01", "2025-03", "2025-04"], ["bad", "current", "current"])
    with pytest.raises(PerformanceFrameError, match="月度"):
        expected_loss_estimate(frame, **_inputs(), window=["2025-01"], horizon_months=1)
    result = expected_loss_estimate(frame, **_inputs(), window=["2025-03"], horizon_months=1)
    assert result.assumptions["matrix_window"] == ["2025-03"]
    assert result.total_el == 0.0
    assert {row.from_state: row.p_to_loss for row in result.chain} == {"current": 0.0, "bad": 1.0}


def test_calendar_year_boundary_is_one_month():
    frame = _panel(["2025-12", "2026-01"], ["current", "bad"])
    assert not any(flag["kind"] == "month_gap" for flag in flow_rate(frame, **_inputs()).red_flags)
    result = expected_loss_estimate(frame, **_inputs(), lgd=0.5, horizon_months=1)
    assert result.total_el == 50.0


@pytest.mark.parametrize("window", [["2025-02"], ["2025-01", "2025-02"]])
def test_explicit_unavailable_migration_window_is_not_silently_dropped(window):
    frame = _panel(["2025-01", "2025-02"], ["current", "bad"])
    # February is a snapshot, but it has no following snapshot and therefore
    # cannot supply a February-origin transition probability.
    with pytest.raises(PerformanceFrameError) as error:
        expected_loss_estimate(frame, **_inputs(), window=window, horizon_months=1)
    assert error.value.problem == "unavailable_migration_window"


def test_actual_portfolio_tool_rejects_monthly_el_from_gapped_snapshots(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, task.id,
                        _panel(["2025-01", "2025-03"], ["current", "bad"]), "gapped")
    result = runner.invoke(ToolRef("analysis", "expected_loss_estimate"),
        {**_inputs(), "dataset_id": dataset.id, "expected_content_hash": dataset.content_hash,
         "loss_state": "bad", "lgd": 0.5, "horizon_months": 1}, task_id=task.id)
    assert not result.ok
    assert result.error_kind == "performance_frame_invalid"
    assert result.error_detail["problem"] == "non_monthly_migration"


def test_http_portfolio_with_missing_month_cannot_finish_or_publish_el_report(tmp_path):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_contracts import digest
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_workflow_families import _business_protocol

    paths = write_synthetic_suite(tmp_path / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_bytes())
    case = next(case for case in suite["cases"] if case["id"] == "synthetic_portfolio")
    material = case["materials"][0]
    source = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(source)
    frame["snapshot_month"] = frame["snapshot_month"].replace({"2025-02": "2025-03"})
    frame.to_parquet(source, index=False)
    material["sha256"] = digest(source.read_bytes())
    suite["cases"] = [case]
    expected = json.loads(paths["expected"].read_bytes())
    expected["cases"] = {case["id"]: expected["cases"][case["id"]]}
    paths["cases"].write_text(json.dumps(suite, ensure_ascii=False))
    paths["expected"].write_text(json.dumps(expected, ensure_ascii=False))
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "public",
            evidence_custody_dir=tmp_path / "custody", model=model, model_source="fixture_model")
    assert not report["all_passed"]  # The original successful-case denominator is retained.
    observed = report["cases"][0]
    steps = observed["execution"]["steps"]
    loss = next(step for step in steps if step["tool"] == "analysis.expected_loss_estimate")
    assert loss["status"] == "failed", steps
    publication = next(step for step in steps if step["tool"] == "analysis.portfolio_report")
    assert publication["status"] != "done" and not publication["runs"]
    archive = tmp_path / "custody" / report["run_id"] / case["id"]
    with sqlite3.connect(f"file:{archive}/workspace/marvis.sqlite?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT count(*) FROM task_artifacts WHERE kind='portfolio_report_xlsx'").fetchone()[0] == 0
        persisted = conn.execute("SELECT error FROM plan_steps WHERE id=?", (loss["id"],)).fetchone()[0]
        assert "月度" in persisted
