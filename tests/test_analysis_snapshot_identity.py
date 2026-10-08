"""No silent record overwrite or fabricated exposure in portfolio snapshots."""
import json
import sqlite3

import pandas as pd
import pytest

from marvis.data.errors import PerformanceFrameError
from marvis.data.performance import parse_snapshot_month
from marvis.packs.analysis.flow import flow_rate, bucket_migration
from marvis.packs.analysis.loss import expected_loss_estimate
from marvis.plugins.manifest import ToolRef
from test_analysis_pack import _runtime, _register


def _panel():
    return pd.DataFrame({"loan": ["A", "A", "B", "B"],
                         "month": ["2025-01", "2025-02", "2025-01", "2025-02"],
                         "bucket": ["current", "bad", "current", "current"],
                         "balance": [100.0, 90.0, 50.0, 40.0]})


def _inputs():
    return {"id_col": "loan", "snapshot_col": "month", "bucket_col": "bucket",
            "balance_col": "balance", "states": ["current", "bad"]}


@pytest.mark.parametrize("calculator", [flow_rate, bucket_migration, expected_loss_estimate])
@pytest.mark.parametrize("alias", ["2025-01", "202501", "2025-01-21"])
def test_duplicate_canonical_loan_month_is_never_last_write_wins(calculator, alias):
    frame = _panel()
    duplicate = frame.iloc[[0]].copy()
    duplicate["month"], duplicate["bucket"], duplicate["balance"] = alias, "bad", 999
    frame = pd.concat([frame, duplicate], ignore_index=True)
    with pytest.raises(PerformanceFrameError) as exc:
        calculator(frame, **_inputs())
    assert exc.value.problem == "duplicate_loan_snapshot"


@pytest.mark.parametrize("bad_id", [None, "", "   ", float("nan"), True])
def test_invalid_identity_cannot_be_coerced_into_a_shared_loan(bad_id):
    frame = _panel().astype({"loan": object})
    frame.loc[0, "loan"] = bad_id
    with pytest.raises(PerformanceFrameError) as exc:
        flow_rate(frame, **_inputs())
    assert exc.value.problem == "bad_loan_id"


def test_numeric_and_text_id_aliases_do_not_create_an_invented_transition():
    frame = _panel().iloc[:2].copy()
    frame["loan"] = pd.Series([1, "1"], dtype=object)
    with pytest.raises(PerformanceFrameError) as exc:
        flow_rate(frame, **_inputs())
    assert exc.value.problem == "mixed_loan_id_types"


@pytest.mark.parametrize("bad_balance", [None, float("nan"), float("inf"), -1.0, True])
def test_missing_nonfinite_negative_and_boolean_balances_are_not_zero_or_money(bad_balance):
    frame = _panel().astype({"balance": object})
    frame.loc[0, "balance"] = bad_balance
    with pytest.raises(PerformanceFrameError) as exc:
        expected_loss_estimate(frame, **_inputs())
    assert exc.value.problem == "bad_balance"


def test_missing_bucket_cannot_become_a_declared_literal_nan_state():
    frame = _panel()
    frame.loc[0, "bucket"] = float("nan")
    with pytest.raises(PerformanceFrameError) as exc:
        flow_rate(frame, **{**_inputs(), "states": ["current", "bad", "nan"]})
    assert exc.value.problem == "missing_bucket"


@pytest.mark.parametrize("state", ["exited", "from"])
def test_state_names_cannot_overwrite_reserved_output_columns(state):
    frame = _panel()
    with pytest.raises(PerformanceFrameError) as exc:
        bucket_migration(frame, **{**_inputs(), "states": ["current", "bad", state]})
    assert exc.value.problem == "reserved_state"


@pytest.mark.parametrize("snapshot", ["2025-02-31", "2025-01-trailing-junk", ["2025-01", "2025-02"], "NaT", "nan"])
def test_snapshot_parser_does_not_accept_an_invalid_suffix_or_container(snapshot):
    assert parse_snapshot_month(snapshot) is None
    frame = _panel()
    frame["month"] = pd.Series([snapshot, *frame["month"].tolist()[1:]], dtype=object)
    with pytest.raises(PerformanceFrameError) as exc:
        flow_rate(frame, **_inputs())
    assert exc.value.problem == "bad_snapshot"


def test_separate_loans_and_zero_exposure_remain_valid():
    frame = _panel()
    frame.loc[0, "balance"] = 0.0
    result = flow_rate(frame, **_inputs())
    assert result.transitions[0].pair_count == 2
    assert result.transitions[0].base["current"] == 50.0
    assert result.transitions[0].from_to_matrix[0] == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("identities", [[0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0], ["001", "001", "1", "1"]])
def test_consistent_id_types_preserve_separate_loans_and_leading_zeros(identities):
    frame = _panel()
    frame["loan"] = identities
    result = flow_rate(frame, **_inputs())
    assert result.transitions[0].pair_count == 2
    assert result.transitions[0].base["current"] == 150.0
    assert result.transitions[0].from_to_matrix[0] == pytest.approx([1 / 3, 2 / 3, 0.0])


def test_duplicate_error_survives_actual_tool_process_without_loan_values(tmp_path):
    runner, _, registry, task = _runtime(tmp_path)
    frame = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    frame.loc[frame["loan"] == "A", "loan"] = "private-loan-identifier"
    dataset = _register(registry, tmp_path, task.id, frame, "duplicate")
    result = runner.invoke(ToolRef("analysis", "flow_rate"), {
        **_inputs(), "dataset_id": dataset.id, "expected_content_hash": dataset.content_hash,
    }, task_id=task.id)
    assert not result.ok
    assert result.error_kind == "performance_frame_invalid"
    assert result.error_detail["problem"] == "duplicate_loan_snapshot"
    assert "private-loan-identifier" not in json.dumps(result.error_detail)


def test_duplicate_snapshot_in_real_http_journey_cannot_produce_a_report(tmp_path):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
    from marvis.orchestrator.eval.runtime_contracts import digest
    from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
    from test_runtime_agent_benchmark import fixture_model
    from test_runtime_workflow_families import _business_protocol

    paths = write_synthetic_suite(tmp_path / "suite", include_workflow_families=True)
    suite = json.loads(paths["cases"].read_text())
    case = next(c for c in suite["cases"] if c["id"] == "synthetic_portfolio")
    material = case["materials"][0]
    path = paths["dataset_root"] / material["path"]
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    material["sha256"] = digest(path.read_bytes())
    suite["cases"] = [case]
    paths["cases"].write_text(json.dumps(suite))
    with fixture_model(answer_factory=_business_protocol) as (model, _):
        report = run_runtime_suite(cases_path=paths["cases"], expected_path=paths["expected"],
            dataset_root=paths["dataset_root"], output_dir=tmp_path / "public", evidence_custody_dir=tmp_path / "custody",
            model=model, model_source="fixture_model")
    assert not report["all_passed"]  # Keep the original normal-case expectations.
    steps = report["cases"][0]["execution"]["steps"]
    failed = [s for s in steps if s["tool"] in {"analysis.flow_rate", "analysis.bucket_migration"} and s["status"] == "failed"]
    assert failed, steps
    publication = next(s for s in steps if s["tool"] == "analysis.portfolio_report")
    assert publication["status"] != "done" and not publication["runs"]
    archive = tmp_path / "custody" / report["run_id"] / case["id"]
    with sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True) as conn:
        assert conn.execute("SELECT count(*) FROM task_artifacts WHERE kind='portfolio_report_xlsx'").fetchone()[0] == 0
