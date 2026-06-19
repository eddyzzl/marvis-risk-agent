import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
from marvis.domain import TaskCreate
from marvis.output.model_report import ModelReportPayload, render_model_report
from marvis.packs.modeling.report_compute import (
    BusinessColumns,
    compute_sample_analysis,
    compute_vintage_report,
    resolve_report_sections,
    stress_low_pricing,
)
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


def _business_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "loan_month": ["2026-01", "2026-01", "2026-02", "2026-02"],
        "rate": [0.12, 0.18, 0.10, 0.20],
        "amount": [1000.0, 2000.0, 1500.0, 2500.0],
        "term": [6, 12, 6, 12],
        "drawdown": [800.0, 1500.0, 1200.0, 2000.0],
        "limit": [2000.0, 3000.0, 2000.0, 4000.0],
        "mob1": [0, 1, 0, 0],
        "mob2": [0, 1, 0, 1],
        "mob3": [1, 1, 0, 1],
        "y": [0, 1, 0, 1],
        "score": [0.1, 0.8, 0.2, 0.7],
    })


def test_model_report_compute_functions_are_deterministic(tmp_path):
    path = tmp_path / "sample.parquet"
    _business_frame().to_parquet(path, index=False)
    backend = DataBackend(tmp_path)
    business = BusinessColumns(
        loan_month_col="loan_month",
        interest_rate_col="rate",
        loan_amount_col="amount",
        term_col="term",
        drawdown_amount_col="drawdown",
        credit_limit_col="limit",
        mob_observe_cols=("mob1", "mob2", "mob3"),
    )

    sample = compute_sample_analysis(
        backend,
        path,
        loan_month_col="loan_month",
        target_col="y",
        business=business,
        mob_cols=business.mob_observe_cols,
    )
    vintage = compute_vintage_report(
        backend,
        path,
        loan_month_col="loan_month",
        mob_observe_cols=business.mob_observe_cols,
        amount_col="amount",
    )
    low_pricing = stress_low_pricing(
        backend,
        path,
        score_col="score",
        target_col="y",
        interest_rate_col="rate",
        low_pricing_threshold=None,
        ratios=(0.25, 0.5),
    )

    assert sample[0]["放款月"] == "2026-01"
    assert sample[0]["放款笔数"] == 2
    assert sample[0]["平均利率"] == 0.15
    assert sample[0]["Mob3逾期率"] == 1.0
    assert vintage["headers"] == ["mob1", "mob2", "mob3"]
    assert vintage["curves"]["2026-01"] == sorted(vintage["curves"]["2026-01"])
    assert low_pricing == stress_low_pricing(
        backend,
        path,
        score_col="score",
        target_col="y",
        interest_rate_col="rate",
        low_pricing_threshold=None,
        ratios=(0.25, 0.5),
    )
    assert set(low_pricing["by_ratio"]) == {"0.25", "0.5"}


def test_resolve_sections_and_render_model_report_degrades_missing_business_data(tmp_path):
    statuses = resolve_report_sections(BusinessColumns(), dictionary_id=None)
    output = tmp_path / "model_report.xlsx"
    render_model_report(
        ModelReportPayload(
            project_meta={"项目名称": "建模报告"},
            dataset_split=[{"split": "train", "ks": 0.3}],
            stability=[{"metric": "psi", "value": 0.01}],
            sample_analysis=None,
            vintage=None,
            feature_importance=[{"feature": "x1", "importance": 0.7}],
            univariate=[{"feature": "x1", "iv": 0.2, "ks": 0.3}],
            oot_bin_table=[{"bin": 1, "bad_rate": 0.1}],
            stress_product_removal={"baseline": []},
            stress_low_pricing=None,
            narratives={"summary": "受控模板文本"},
            section_status=statuses,
        ),
        output,
    )

    workbook = load_workbook(output)
    assert workbook.sheetnames == [
        "汇总",
        "样本分析",
        "Vintage",
        "特征重要性",
        "oot分箱评估_十分箱",
        "单变量分析",
        "压力测试",
    ]
    assert workbook["样本分析"]["A1"].value.startswith("无业务数据")
    assert any(status.section == "product_list" and not status.available for status in statuses)


def test_generate_model_report_tool_round_trips_via_runner(tmp_path):
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
    task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="报告样例",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="y",
            split_col="split",
            feature_columns=["x1", "x2"],
        )
    )
    frame = pd.concat([_business_frame().assign(split="train", x1=[0.1, 0.2, 0.3, 0.4], x2=[0.4, 0.3, 0.2, 0.1])] * 40, ignore_index=True)
    frame.loc[80:119, "split"] = "test"
    frame.loc[120:, "split"] = "oot"
    path = tmp_path / "report_sample.parquet"
    frame.to_parquet(path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path),
        DataBackend(settings.datasets_dir),
        settings.datasets_dir,
    )
    dataset = registry.register_existing(path, task_id=task.id, role="modeling_sample")
    trained = runner.invoke(
        ToolRef("modeling", "train_model"),
        {
            "dataset_id": dataset.id,
            "recipe": "lr",
            "features": ["x1", "x2"],
            "target_col": "y",
            "split_col": "split",
            "split_values": {"train": "train", "test": "test", "oot": "oot"},
            "params": {"max_iter": 200},
            "seed": 7,
        },
        task_id=task.id,
    )
    assert trained.ok is True, trained.error

    report = runner.invoke(
        ToolRef("modeling", "generate_model_report"),
        {
            "experiment_id": trained.output["experiment_id"],
            "dataset_id": dataset.id,
            "business_columns": {
                "loan_month_col": "loan_month",
                "interest_rate_col": "rate",
                "loan_amount_col": "amount",
                "term_col": "term",
                "drawdown_amount_col": "drawdown",
                "credit_limit_col": "limit",
                "mob_observe_cols": ["mob1", "mob2", "mob3"],
            },
            "project_meta": {"项目名称": "报告样例"},
        },
        task_id=task.id,
    )

    assert report.ok is True, report.error
    assert Path(report.output["report_path"]).exists()
    assert len(report.output["section_status"]) == 5
