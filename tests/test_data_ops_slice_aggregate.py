"""S6 Commit 1: data_ops slice_aggregate tool.

Whitelisted deterministic group-by aggregate: hand-calculated numbers, an injected
column name rejected before any SQL runs, top_k truncation + empty-result red flags,
between/in filter boundaries, and the data.slice_aggregate audit row.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, init_db
from marvis.db_schema import connect
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    return runner, registry, settings


def _register(registry, tmp_path, frame: pd.DataFrame):
    path = tmp_path / "slice.csv"
    frame.to_csv(path, index=False)
    return registry.register_from_upload("task-1", path, role="sample")


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "channel": ["A", "A", "B", "B", "B", "A"],
        "month": ["2026-05", "2026-05", "2026-05", "2026-06", "2026-05", "2026-06"],
        "bad": [1, 0, 1, 0, 1, 0],
        "decision": ["approve", "reject", "approve", "approve", "reject", "approve"],
        "amount": [100, 200, 300, 400, 500, 600],
    })


def test_slice_aggregate_hand_calculated_group_metrics(tmp_path):
    runner, registry, _settings = _runtime(tmp_path)
    ds = _register(registry, tmp_path, _frame())

    result = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {
            "dataset_id": ds.id,
            "group_by": ["channel"],
            "metrics": [
                {"op": "count"},
                {"op": "bad_rate", "col": "bad"},
                {"op": "approval_rate", "col": "decision"},
                {"op": "mean", "col": "amount"},
            ],
            "month_col": "month",
            "months": ["2026-05"],
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    rows = {row["channel"]: row for row in result.output["rows"]}
    # May-only: channel A = rows [bad1/approve/100, bad0/reject/200]
    assert rows["A"]["count"] == 2
    assert rows["A"]["bad_rate_bad"] == 0.5
    assert rows["A"]["approval_rate_decision"] == 0.5
    assert rows["A"]["mean_amount"] == 150.0
    # May-only: channel B = rows [bad1/approve/300, bad1/reject/500]
    assert rows["B"]["count"] == 2
    assert rows["B"]["bad_rate_bad"] == 1.0
    assert rows["B"]["approval_rate_decision"] == 0.5
    assert rows["B"]["mean_amount"] == 400.0
    assert result.output["n_rows_scanned"] == 4
    assert result.output["red_flags"] == []
    # spec_echo mirrors the口径 verbatim (all filters/months/ops echoed)
    assert result.output["spec_echo"]["months"] == ["2026-05"]
    assert result.output["columns"][0] == "channel"


def test_slice_aggregate_rejects_injected_column_name(tmp_path):
    runner, registry, _settings = _runtime(tmp_path)
    ds = _register(registry, tmp_path, _frame())

    injected = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {"dataset_id": ds.id, "group_by": ["channel; DROP TABLE audit"], "metrics": [{"op": "count"}]},
        task_id="task-1",
    )

    assert injected.ok is False
    # A whitelist miss surfaces as a typed error before any SQL is compiled.
    assert "channel; DROP TABLE audit" in (injected.error or "")


def test_slice_aggregate_hallucinated_metric_column_rejected(tmp_path):
    runner, registry, _settings = _runtime(tmp_path)
    ds = _register(registry, tmp_path, _frame())

    result = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {"dataset_id": ds.id, "metrics": [{"op": "mean", "col": "nonexistent_col"}]},
        task_id="task-1",
    )

    assert result.ok is False
    assert "nonexistent_col" in (result.error or "")


def test_slice_aggregate_truncated_and_between_in_filters(tmp_path):
    runner, registry, _settings = _runtime(tmp_path)
    ds = _register(registry, tmp_path, _frame())

    # top_k=1 over two channels -> truncated red flag (one extra row detected then dropped).
    truncated = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {"dataset_id": ds.id, "group_by": ["channel"], "metrics": [{"op": "count"}], "top_k": 1},
        task_id="task-1",
    )
    assert truncated.ok is True, truncated.error
    assert len(truncated.output["rows"]) == 1
    assert any(flag["code"] == "truncated" for flag in truncated.output["red_flags"])

    # between boundary: amount BETWEEN 200 AND 400 keeps rows 200,300,400 (inclusive).
    between = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {
            "dataset_id": ds.id,
            "metrics": [{"op": "count"}],
            "filters": [{"col": "amount", "op": "between", "value": [200, 400]}],
        },
        task_id="task-1",
    )
    assert between.ok is True, between.error
    assert between.output["rows"][0]["count"] == 3

    # in filter over channels; empty selection triggers the empty_result flag.
    empty = runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {
            "dataset_id": ds.id,
            "group_by": ["channel"],
            "metrics": [{"op": "count"}],
            "filters": [{"col": "channel", "op": "in", "value": ["Z"]}],
        },
        task_id="task-1",
    )
    assert empty.ok is True, empty.error
    assert empty.output["rows"] == []
    assert any(flag["code"] == "empty_result" for flag in empty.output["red_flags"])


def test_slice_aggregate_writes_audit_row(tmp_path):
    runner, registry, settings = _runtime(tmp_path)
    ds = _register(registry, tmp_path, _frame())

    runner.invoke(
        ToolRef("data_ops", "slice_aggregate"),
        {"dataset_id": ds.id, "group_by": ["channel"], "metrics": [{"op": "count"}]},
        task_id="task-1",
    )

    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT target_ref, detail_json FROM audit WHERE kind = 'data.slice_aggregate'"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["target_ref"] == ds.id
    assert '"task_id":"task-1"' in rows[0]["detail_json"]
