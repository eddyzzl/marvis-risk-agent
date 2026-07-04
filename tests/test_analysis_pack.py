"""S3 组合分析套件 pack tests (Commit 2 部分：flow/migration/segment/EL).

Covers: manifest 工具注册；契约 typed error 经 runner 的 error_kind；flow/migration
4 贷款手算矩阵逐值断言 + exited 伪状态 + sparse red flag；segment HHI/top1 手算 +
「其他」归并；EL 3 状态吸收链手算 + 非吸收示警。趋势/报告在后续 commit。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings

_STATES = ["current", "M1", "bad"]


def _runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    packs_root = Path(__file__).parents[1] / "marvis" / "packs"
    load_builtin_packs(plugin_registry, packs_root)
    runner = ToolRunner(
        ToolRegistry(plugin_registry),
        plugin_repo,
        python_executable=sys.executable,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(settings.datasets_dir)
    registry = DatasetRegistry(data_repo, backend, settings.datasets_dir)
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="组合分析样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
        )
    )
    return runner, plugin_registry, registry, task


def _register(registry, tmp_path, task_id, frame, name):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="performance")


def _four_loan_frame() -> pd.DataFrame:
    # loan A: current->current->M1 ; B: current->M1->bad ; C: current->current->current ;
    # D: current->M1 (no month 3 -> exited from 2025-02)
    seqs = {
        "A": [("2025-01", "current"), ("2025-02", "current"), ("2025-03", "M1")],
        "B": [("2025-01", "current"), ("2025-02", "M1"), ("2025-03", "bad")],
        "C": [("2025-01", "current"), ("2025-02", "current"), ("2025-03", "current")],
        "D": [("2025-01", "current"), ("2025-02", "M1")],
    }
    rows = []
    for loan_id, seq in seqs.items():
        for month, bucket in seq:
            rows.append({"loan_id": loan_id, "snapshot_month": month, "bucket": bucket, "balance": 1000.0})
    return pd.DataFrame(rows)


def test_analysis_manifest_registers_expected_tools(tmp_path):
    _runner, plugin_registry, _registry, _task = _runtime(tmp_path)
    manifest = plugin_registry.get("analysis")
    names = {tool.name for tool in manifest.tools}
    assert {"flow_rate", "bucket_migration", "segment_profile", "expected_loss_estimate"} <= names
    assert "write:dataset" in manifest.permissions
    assert "read:experiment" in manifest.permissions
    flow = next(tool for tool in manifest.tools if tool.name == "flow_rate")
    assert flow.determinism == "deterministic"
    assert "read:dataset" in flow.side_effects


@pytest.mark.slow
def test_flow_rate_hand_computed_matrix_and_exited(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, task.id, _four_loan_frame(), "perf")
    result = runner.invoke(
        ToolRef("analysis", "flow_rate"),
        {
            "dataset_id": dataset.id,
            "id_col": "loan_id",
            "snapshot_col": "snapshot_month",
            "bucket_col": "bucket",
            "states": _STATES,
        },
        task_id=task.id,
    )
    assert result.ok is True, result.error
    assert result.output["months"] == ["2025-01", "2025-02"]
    assert result.output["to_states"] == ["current", "M1", "bad", "exited"]
    by_month = {row["month"]: row for row in result.output["matrix_by_month"]}
    # 2025-02 -> 2025-03: current base {A,C}=2 -> A M1, C current => [0.5,0.5,0,0]
    feb = by_month["2025-02"]
    assert feb["from_to_matrix"][0] == pytest.approx([0.5, 0.5, 0.0, 0.0])
    # M1 base {B,D}=2 -> B bad, D exited => [0,0,0.5,0.5]  (exited 伪状态显式)
    assert feb["from_to_matrix"][1] == pytest.approx([0.0, 0.0, 0.5, 0.5])
    # net into_bad in 2025-02 = 1 (B: M1->bad)
    net = {row["month"]: row for row in result.output["net_flows"]}
    assert net["2025-02"]["into_bad"] == pytest.approx(1.0)


@pytest.mark.slow
def test_flow_rate_sparse_month_red_flag(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, task.id, _four_loan_frame(), "perf")
    result = runner.invoke(
        ToolRef("analysis", "flow_rate"),
        {
            "dataset_id": dataset.id,
            "id_col": "loan_id",
            "snapshot_col": "snapshot_month",
            "bucket_col": "bucket",
            "states": _STATES,
        },
        task_id=task.id,
    )
    assert result.ok is True, result.error
    kinds = {flag["kind"] for flag in result.output["red_flags"]}
    # every month has <100 aligned pairs -> sparse_month raised
    assert "sparse_month" in kinds


@pytest.mark.slow
def test_flow_rate_unknown_bucket_typed_error(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    frame = _four_loan_frame()
    frame.loc[0, "bucket"] = "mystery"
    dataset = _register(registry, tmp_path, task.id, frame, "perf")
    result = runner.invoke(
        ToolRef("analysis", "flow_rate"),
        {
            "dataset_id": dataset.id,
            "id_col": "loan_id",
            "snapshot_col": "snapshot_month",
            "bucket_col": "bucket",
            "states": _STATES,
        },
        task_id=task.id,
    )
    assert result.ok is False
    assert result.error_kind == "performance_frame_invalid"
    assert result.error_detail["problem"] == "unknown_bucket"


@pytest.mark.slow
def test_bucket_migration_hand_computed_avg(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, task.id, _four_loan_frame(), "perf")
    result = runner.invoke(
        ToolRef("analysis", "bucket_migration"),
        {
            "dataset_id": dataset.id,
            "id_col": "loan_id",
            "snapshot_col": "snapshot_month",
            "bucket_col": "bucket",
            "states": _STATES,
        },
        task_id=task.id,
    )
    assert result.ok is True, result.error
    # M1 row avg across the two months: month1 M1 base 0 -> row all 0; month2 M1 -> [0,0,0.5,0.5]
    # avg = ([0,0,0,0] + [0,0,0.5,0.5]) / 2 = [0,0,0.25,0.25]
    assert result.output["avg_matrix"][1] == pytest.approx([0.0, 0.0, 0.25, 0.25])
    # heat_table row shape
    assert result.output["heat_table"][1]["from"] == "M1"
    assert result.output["heat_table"][1]["exited"] == pytest.approx(0.25)


@pytest.mark.slow
def test_segment_profile_hand_computed_hhi_and_other_merge(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    # segments A(6) B(3) C(1) -> shares .6/.3/.1 ; HHI = .36+.09+.01 = .46 ; top1 .6
    frame = pd.DataFrame(
        {
            "seg": ["A"] * 6 + ["B"] * 3 + ["C"] * 1,
            "y": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        }
    )
    dataset = _register(registry, tmp_path, task.id, frame, "seg")
    result = runner.invoke(
        ToolRef("analysis", "segment_profile"),
        {"dataset_id": dataset.id, "segment_col": "seg", "target_col": "y"},
        task_id=task.id,
    )
    assert result.ok is True, result.error
    conc = result.output["concentration"]
    assert conc["top1_pct"] == pytest.approx(0.6)
    assert conc["hhi"] == pytest.approx(0.46)
    kinds = {flag["kind"] for flag in result.output["red_flags"]}
    assert "high_concentration" in kinds  # top1 .6 > .40

    # top_k=1 -> B,C merged into 「其他」
    merged = runner.invoke(
        ToolRef("analysis", "segment_profile"),
        {"dataset_id": dataset.id, "segment_col": "seg", "target_col": "y", "top_k": 1},
        task_id=task.id,
    )
    assert merged.ok is True, merged.error
    seg_names = {row["segment"] for row in merged.output["segments"]}
    assert "其他" in seg_names
    other = next(row for row in merged.output["segments"] if row["segment"] == "其他")
    assert other["count"] == 4  # B(3)+C(1)
    assert "sparse_segment" in {flag["kind"] for flag in merged.output["red_flags"]}


def _absorbing_hand_frame() -> pd.DataFrame:
    """A frame whose count-based avg migration matrix is a clean 3-state chain.

    Constructed so that over the aligned month pairs:
      current -> {0.5 current, 0.5 M1}
      M1      -> {0.5 M1, 0.5 bad}
      bad     -> bad (absorbing)
    We realize this with 4 pairs per from-state across two month steps.
    """
    rows = []
    # We need many loans to hit the exact 0.5 splits deterministically.
    # current sources: 4 loans, 2 stay current, 2 go M1
    plans = []
    # loan set to produce current->0.5/0.5 in month pair 01->02
    plans += [("c1", [("2025-01", "current"), ("2025-02", "current")])]
    plans += [("c2", [("2025-01", "current"), ("2025-02", "current")])]
    plans += [("c3", [("2025-01", "current"), ("2025-02", "M1")])]
    plans += [("c4", [("2025-01", "current"), ("2025-02", "M1")])]
    # M1 sources in 01->02: 4 loans, 2 stay M1, 2 go bad
    plans += [("m1", [("2025-01", "M1"), ("2025-02", "M1")])]
    plans += [("m2", [("2025-01", "M1"), ("2025-02", "M1")])]
    plans += [("m3", [("2025-01", "M1"), ("2025-02", "bad")])]
    plans += [("m4", [("2025-01", "M1"), ("2025-02", "bad")])]
    # bad sources: absorbing
    plans += [("b1", [("2025-01", "bad"), ("2025-02", "bad")])]
    plans += [("b2", [("2025-01", "bad"), ("2025-02", "bad")])]
    for loan_id, seq in plans:
        for month, bucket in seq:
            rows.append({"loan_id": loan_id, "snapshot_month": month, "bucket": bucket, "balance": 1000.0})
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_expected_loss_absorbing_chain_hand_computed(tmp_path):
    runner, _pr, registry, task = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, task.id, _absorbing_hand_frame(), "perf")
    result = runner.invoke(
        ToolRef("analysis", "expected_loss_estimate"),
        {
            "dataset_id": dataset.id,
            "id_col": "loan_id",
            "snapshot_col": "snapshot_month",
            "bucket_col": "bucket",
            "states": _STATES,
            "balance_col": "balance",
            "loss_state": "bad",
            "lgd": 0.5,
            "horizon_months": 2,
        },
        task_id=task.id,
    )
    assert result.ok is True, result.error
    chain = {row["from_state"]: row["p_to_loss"] for row in result.output["chain"]}
    # T = [[.5,.5,0],[0,.5,.5],[0,0,1]]
    # T^2 = [[.25,.5,.25],[0,.25,.75],[0,0,1]]
    # P(bad in 2 steps): current .25, M1 .75, bad 1
    assert chain["current"] == pytest.approx(0.25)
    assert chain["M1"] == pytest.approx(0.75)
    assert chain["bad"] == pytest.approx(1.0)
    # bad is genuinely absorbing here, so no matrix_not_absorbing flag; the
    # 2-month hand frame does legitimately trip short_history (<3 months).
    kinds = {flag["kind"] for flag in result.output["red_flags"]}
    assert "matrix_not_absorbing" not in kinds


def test_expected_loss_matrix_not_absorbing_warns():
    """Kernel-level: a loss state with observed out-flow raises matrix_not_absorbing."""
    from marvis.packs.analysis.loss import expected_loss_estimate

    # bad -> current in month 01->02, so bad is NOT absorbing in the data
    rows = []
    plans = [
        ("b1", [("2025-01", "bad"), ("2025-02", "current")]),
        ("b2", [("2025-01", "bad"), ("2025-02", "current")]),
        ("c1", [("2025-01", "current"), ("2025-02", "current")]),
    ]
    for loan_id, seq in plans:
        for month, bucket in seq:
            rows.append({"loan_id": loan_id, "snapshot_month": month, "bucket": bucket, "balance": 1000.0})
    frame = pd.DataFrame(rows)
    result = expected_loss_estimate(
        frame,
        id_col="loan_id",
        snapshot_col="snapshot_month",
        bucket_col="bucket",
        states=_STATES,
        balance_col="balance",
        loss_state="bad",
    )
    kinds = {flag["kind"] for flag in result.red_flags}
    assert "matrix_not_absorbing" in kinds


def _stable_panel_frame(months: list[str]) -> pd.DataFrame:
    """跨月稳定组合面板：每个快照月的桶分布/余额完全相同 -> 各月 EL 相同。

    每月都含 4 current / 4 M1 / 2 bad（余额均 1000），loan_id 在各月保持同一桶，
    故 groupby-month EL 恒定。另加两个每月都出现且发生迁徙的探针 loan（用于让
    bucket_migration 观测到 current->M1 与 M1->bad 的迁出），矩阵可算。跨月求和会
    ~N× 虚高，参考快照口径应回到单月 EL。
    """
    # fixed-bucket loans: identical (bucket, balance) in every month -> stable distribution
    fixed = [
        ("c1", "current"), ("c2", "current"), ("c3", "current"), ("c4", "current"),
        ("m1", "M1"), ("m2", "M1"), ("m3", "M1"), ("m4", "M1"),
        ("b1", "bad"), ("b2", "bad"),
    ]
    rows = []
    for month in months:
        for loan_id, bucket in fixed:
            rows.append({"loan_id": loan_id, "snapshot_month": month, "bucket": bucket, "balance": 1000.0})
    # migration probes: present every month, migrating current->M1->bad so the
    # transition matrix has observed out-flow from current and M1 (zero balance so
    # they do not perturb the per-month EL headline balances/distribution weights).
    ladder = {"current": "M1", "M1": "bad", "bad": "bad"}
    probe_state = "current"
    for month in months:
        rows.append({"loan_id": "probe", "snapshot_month": month, "bucket": probe_state, "balance": 0.0})
        probe_state = ladder[probe_state]
    frame = pd.DataFrame(rows).reset_index(drop=True)
    return frame


def _el_kernel(frame: pd.DataFrame):
    from marvis.packs.analysis.loss import expected_loss_estimate

    return expected_loss_estimate(
        frame,
        id_col="loan_id",
        snapshot_col="snapshot_month",
        bucket_col="bucket",
        states=_STATES,
        balance_col="balance",
        loss_state="bad",
        lgd=0.5,
        horizon_months=2,
    )


def test_expected_loss_total_el_is_reference_snapshot_not_sum():
    """A2 anti-inflation lock: total_el == latest month's EL, NOT the cross-month sum."""
    months = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06"]
    result = _el_kernel(_stable_panel_frame(months))
    per_month = {row.month: row.expected_loss for row in result.el_by_month}
    latest = max(per_month)
    assert result.total_el == pytest.approx(per_month[latest])
    # more than one month present -> the naive sum would be materially larger
    assert len(result.el_by_month) > 1
    naive_sum = sum(per_month.values())
    assert result.total_el < naive_sum
    assert naive_sum == pytest.approx(result.total_el * len(per_month))


def test_expected_loss_assumptions_annotate_basis():
    """口径 machine-readable: assumptions carries basis + reference snapshot month."""
    months = ["2025-01", "2025-02", "2025-03"]
    result = _el_kernel(_stable_panel_frame(months))
    assert result.assumptions["total_el_basis"] == "reference_snapshot"
    assert result.assumptions["reference_snapshot"] == "2025-03"


def test_expected_loss_per_month_rows_unchanged():
    """Only the sum changed: per-month rows keep one row per snapshot month with per-month EL."""
    months = ["2025-01", "2025-02", "2025-03", "2025-04"]
    frame = _stable_panel_frame(months)
    result = _el_kernel(frame)
    present = sorted({str(m) for m in frame["snapshot_month"].tolist()})
    assert [row.month for row in result.el_by_month] == present
    # each per-month EL is positive and equal across the stable panel
    els = [row.expected_loss for row in result.el_by_month]
    assert all(el > 0 for el in els)
    assert all(el == pytest.approx(els[0]) for el in els)
    # exactly one row flagged as the reference (latest) month
    ref_rows = [row for row in result.el_by_month if row.is_reference]
    assert len(ref_rows) == 1
    assert ref_rows[0].month == max(row.month for row in result.el_by_month)


def test_expected_loss_single_month_total_equals_month():
    """Edge: single-snapshot dataset -> total_el == that month's EL, reference == that month."""
    frame = _stable_panel_frame(["2025-07", "2025-08"])
    frame = frame[frame["snapshot_month"] == "2025-08"].reset_index(drop=True)
    result = _el_kernel(frame)
    assert len(result.el_by_month) == 1
    assert result.total_el == pytest.approx(result.el_by_month[0].expected_loss)
    assert result.assumptions["reference_snapshot"] == "2025-08"
