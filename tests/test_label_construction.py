"""C1 标签构造与成熟度工具 — 内核 + 桥接 + 工具边界成熟度门测试.

内核测试是纯函数（无子进程），随 fast tier 跑；工具边界的成熟度确认门用
ToolRunner 走子进程，标 @pytest.mark.slow。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.label_construction import (
    BadDefinition,
    check_cohort_maturity,
    construct_label,
    suggest_bad_definition,
)


# ---------------------------------------------------------------------------
# construct_label: 0/1 target + 定坏口径元数据 (dpd 口径, 手算).
# ---------------------------------------------------------------------------
def _dpd_long_frame() -> pd.DataFrame:
    # 3 loans观测 mob 0..6. A 在 mob4 逾期 95 天 (命中 90+); B 全程正常; C 逾期 35/40
    # 天但从不到 90 天. 全部观测到 mob6.
    return pd.DataFrame({
        "loan_id": ["A"] * 7 + ["B"] * 7 + ["C"] * 7,
        "mob": list(range(7)) * 3,
        "dpd": [0, 0, 0, 0, 95, 120, 150, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 35, 40, 0, 0],
        "cohort": ["2026-01"] * 14 + ["2026-02"] * 7,
    })


def test_construct_label_dpd_threshold_hand_computed():
    result = construct_label(
        _dpd_long_frame(),
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
        cohort_col="cohort",
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    assert records == {"A": 1.0, "B": 0.0, "C": 0.0}
    assert result.n_loans == 3
    assert result.n_bad == 1
    assert result.n_good == 2
    assert result.n_unmatured == 0
    assert result.definition.label_expression() == "DPD90+@mob6 (obs=0, perf=6)"


def test_construct_label_carries_bad_definition_metadata():
    result = construct_label(
        _dpd_long_frame(),
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    definition = result.definition
    assert isinstance(definition, BadDefinition)
    assert definition.threshold_kind == "dpd"
    assert definition.threshold == 90.0
    assert definition.observation_window == 0
    assert definition.performance_window == 6
    assert definition.at_mob == 6
    payload = definition.to_dict()
    assert payload["label"] == "DPD90+@mob6 (obs=0, perf=6)"
    assert payload["at_mob"] == 6


def test_construct_label_at_mob_overrides_window_end():
    # 90+@mob3: 判定点在 mob3, A 的命中在 mob4 -> 落在 (0, 3] 窗口外 -> A 变好客户.
    result = construct_label(
        _dpd_long_frame(),
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        at_mob=3,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    assert records == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert result.definition.at_mob == 3


def test_construct_label_status_bucket_threshold():
    # status 口径: states 由好到坏 C < M1 < M2 < M3+. 命中 = 达到或坏于 M2.
    frame = pd.DataFrame({
        "loan_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "mob": [0, 1, 2, 0, 1, 2, 0, 1, 2],
        "status": ["C", "M1", "M2", "C", "C", "M1", "C", "M3+", "M3+"],
    })
    result = construct_label(
        frame,
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=2,
        status_col="status",
        threshold_status="M2",
        states=["C", "M1", "M2", "M3+"],
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    # A 到 M2 (命中); B 最坏 M1 (未命中); C 到 M3+ (坏于 M2 -> 命中).
    assert records == {"A": 1.0, "B": 0.0, "C": 1.0}
    assert result.definition.threshold_kind == "status"
    assert result.definition.threshold == "M2"


def test_construct_label_unmatured_loan_is_nan():
    # C 只观测到 mob3, 判定点 mob6 -> 表现期未闭合 -> 标签 NaN (不当好客户).
    frame = _dpd_long_frame()
    frame = frame[~((frame["loan_id"] == "C") & (frame["mob"] > 3))]
    result = construct_label(
        frame,
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    assert records["A"] == 1.0
    assert records["B"] == 0.0
    assert pd.isna(records["C"])
    assert result.n_unmatured == 1


def test_construct_label_matured_when_at_mob_row_missing_but_observed_beyond():
    # C1 缺陷复现: loan D 观测 mob [0,1,2,3,4,5,7] (恰好缺 at_mob=6 那一期),
    # 从不逾期. 它在 mob6 前后都有观测 -> 表现期已闭合到 mob6+ -> 应为 good=0.0,
    # 不能因 mob6 那一行恰好缺失就被判未成熟当成 NaN.
    frame = pd.DataFrame({
        "loan_id": ["D"] * 7,
        "mob": [0, 1, 2, 3, 4, 5, 7],
        "dpd": [0, 0, 0, 0, 0, 0, 0],
    })
    result = construct_label(
        frame,
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    assert records["D"] == 0.0
    assert result.n_good == 1
    assert result.n_unmatured == 0

    # 对照: dense 0..6 (含 mob6) 仍为 good=0.0 — 标签不该在缺/不缺 mob6 那一行间翻转.
    dense = pd.DataFrame({
        "loan_id": ["D"] * 7,
        "mob": [0, 1, 2, 3, 4, 5, 6],
        "dpd": [0, 0, 0, 0, 0, 0, 0],
    })
    dense_result = construct_label(
        dense,
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    dense_records = {row["loan_id"]: row["target"] for row in dense_result.frame.to_dict("records")}
    assert dense_records["D"] == 0.0
    assert dense_result.n_good == 1
    assert dense_result.n_unmatured == 0


def test_construct_label_truly_unmatured_when_max_mob_below_at_mob():
    # 对照: loan E 最大观测 mob=4 < at_mob=6 -> 表现期真未闭合 -> 仍应判 NaN/unmatured.
    frame = pd.DataFrame({
        "loan_id": ["E"] * 5,
        "mob": [0, 1, 2, 3, 4],
        "dpd": [0, 0, 0, 0, 0],
    })
    result = construct_label(
        frame,
        id_col="loan_id",
        mob_col="mob",
        observation_window=0,
        performance_window=6,
        dpd_col="dpd",
        threshold_dpd=90,
    )
    records = {row["loan_id"]: row["target"] for row in result.frame.to_dict("records")}
    assert pd.isna(records["E"])
    assert result.n_unmatured == 1
    assert result.n_good == 0


def test_construct_label_is_deterministic():
    frame = _dpd_long_frame()
    kwargs = dict(
        id_col="loan_id", mob_col="mob", observation_window=0,
        performance_window=6, dpd_col="dpd", threshold_dpd=90, cohort_col="cohort",
    )
    a = construct_label(frame, **kwargs)
    b = construct_label(frame, **kwargs)
    assert a.frame.to_dict("records") == b.frame.to_dict("records")
    assert a.definition.to_dict() == b.definition.to_dict()


def test_construct_label_rejects_both_and_neither_threshold_kinds():
    frame = _dpd_long_frame()
    with pytest.raises(ValueError, match="二选一"):
        construct_label(
            frame, id_col="loan_id", mob_col="mob", observation_window=0,
            performance_window=6, dpd_col="dpd", threshold_dpd=90,
            status_col="dpd", threshold_status="X", states=["X"],
        )
    with pytest.raises(ValueError, match="逾期强度口径"):
        construct_label(
            frame, id_col="loan_id", mob_col="mob", observation_window=0,
            performance_window=6,
        )


def test_construct_label_rejects_bad_windows_and_status_threshold():
    frame = _dpd_long_frame()
    with pytest.raises(ValueError, match="performance_window"):
        construct_label(
            frame, id_col="loan_id", mob_col="mob", observation_window=0,
            performance_window=0, dpd_col="dpd", threshold_dpd=90,
        )
    with pytest.raises(ValueError, match="not in states|不在 states"):
        construct_label(
            frame, id_col="loan_id", mob_col="mob", observation_window=0,
            performance_window=2, status_col="dpd", threshold_status="ZZZ",
            states=["C", "M1"],
        )


# ---------------------------------------------------------------------------
# check_cohort_maturity: 表现期闭合判定.
# ---------------------------------------------------------------------------
def test_check_cohort_maturity_flags_immature_cohorts():
    frame = _dpd_long_frame()
    # C (cohort 2026-02) 只观测到 mob3; 定坏需 mob6 -> 2026-02 未成熟.
    frame = frame[~((frame["loan_id"] == "C") & (frame["mob"] > 3))]
    report = check_cohort_maturity(
        frame, id_col="loan_id", mob_col="mob", cohort_col="cohort", required_mob=6,
    )
    assert report.required_mob == 6
    assert report.immature_cohorts == ("2026-02",)
    assert report.all_matured is False
    by_cohort = {c.cohort: c for c in report.cohorts}
    assert by_cohort["2026-01"].matured is True
    assert by_cohort["2026-01"].max_observed_mob == 6
    assert by_cohort["2026-02"].matured is False
    assert by_cohort["2026-02"].max_observed_mob == 3


def test_check_cohort_maturity_all_matured():
    report = check_cohort_maturity(
        _dpd_long_frame(), id_col="loan_id", mob_col="mob", cohort_col="cohort", required_mob=6,
    )
    assert report.immature_cohorts == ()
    assert report.all_matured is True


# ---------------------------------------------------------------------------
# suggest_bad_definition: roll_rate -> 定坏口径桥接.
# ---------------------------------------------------------------------------
def test_suggest_bad_definition_from_roll_rate_matrix():
    # states 由好到坏; M2 回滚率 (回到 C/M1) = 0.05+0.03 = 0.08 < 0.10 -> 建议 M2.
    # M1 回滚率 = 0.30 (回 C) >= 0.10 -> 不选 M1 (更靠前但不稳定).
    states = ["C", "M1", "M2", "M3+"]
    matrix = [
        [0.90, 0.10, 0.00, 0.00],
        [0.30, 0.40, 0.30, 0.00],
        [0.05, 0.03, 0.50, 0.42],
        [0.00, 0.00, 0.00, 1.00],
    ]
    suggestion = suggest_bad_definition(states=states, matrix=matrix, at_mob=6)
    assert suggestion is not None
    assert suggestion.threshold_status == "M2"
    assert suggestion.at_mob == 6
    assert round(suggestion.roll_back_rate, 4) == 0.08
    assert "M2@mob6" in suggestion.rationale


def test_suggest_bad_definition_picks_earliest_stable_bucket():
    # M1 回滚率 0.05 < 0.10 -> 选最靠前的稳定逾期桶 M1 (越早定坏越扩样本).
    states = ["C", "M1", "M2"]
    matrix = [
        [0.90, 0.10, 0.00],
        [0.05, 0.60, 0.35],
        [0.00, 0.00, 1.00],
    ]
    suggestion = suggest_bad_definition(states=states, matrix=matrix, at_mob=4)
    assert suggestion.threshold_status == "M1"
    assert suggestion.at_mob == 4


def test_suggest_bad_definition_returns_none_when_no_stable_bucket():
    # 所有逾期桶回滚率都 >= 阈值 -> 无建议.
    states = ["C", "M1"]
    matrix = [
        [0.90, 0.10],
        [0.50, 0.50],
    ]
    assert suggest_bad_definition(states=states, matrix=matrix, at_mob=6) is None


def test_suggest_bad_definition_is_deterministic_and_validates_shape():
    states = ["C", "M1", "M2"]
    matrix = [[0.9, 0.1, 0.0], [0.05, 0.6, 0.35], [0.0, 0.0, 1.0]]
    a = suggest_bad_definition(states=states, matrix=matrix, at_mob=4)
    b = suggest_bad_definition(states=states, matrix=matrix, at_mob=4)
    assert a.to_dict() == b.to_dict()
    with pytest.raises(ValueError, match="matrix shape"):
        suggest_bad_definition(states=states, matrix=[[0.9, 0.1]], at_mob=4)


# ---------------------------------------------------------------------------
# Tool boundary: cohort maturity 强制确认门 (失败形状), via the runner.
# ---------------------------------------------------------------------------
def _runtime(tmp_path):
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.db import DatasetRepository, PluginRepository, TaskRepository, init_db
    from marvis.domain import TaskCreate
    from marvis.plugins.loader import load_builtin_packs
    from marvis.plugins.registry import PluginRegistry, ToolRegistry
    from marvis.plugins.runner import ToolRunner
    from marvis.settings import build_settings

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
            model_name="C1 标签构造",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path / "source"),
            algorithm="lr",
            run_mode="agent",
            target_col="target",
            score_col="score",
        )
    )
    return runner, registry, task, backend


def _register(registry, tmp_path, frame: pd.DataFrame, name: str, task_id: str):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return registry.register_existing(path, task_id=task_id, role="labeling_sample")


@pytest.mark.slow
def test_tool_define_label_gates_immature_cohorts(tmp_path):
    from marvis.plugins.manifest import ToolRef

    runner, registry, task, _backend = _runtime(tmp_path)
    frame = _dpd_long_frame()
    # cohort 2026-02 (loan C) 只观测到 mob3; 定坏需 mob6 -> 未成熟 -> 应触发确认门.
    frame = frame[~((frame["loan_id"] == "C") & (frame["mob"] > 3))]
    dataset = _register(registry, tmp_path, frame, "immature", task.id)
    base = {
        "dataset_id": dataset.id, "id_col": "loan_id", "mob_col": "mob",
        "cohort_col": "cohort", "observation_window": 0, "performance_window": 6,
        "dpd_col": "dpd", "threshold_dpd": 90,
    }
    blocked = runner.invoke(ToolRef("labeling", "define_label"), dict(base), task_id=task.id)
    assert blocked.ok is False
    assert blocked.error_kind == "cohort_maturity_not_confirmed"

    confirmed = runner.invoke(
        ToolRef("labeling", "define_label"),
        {**base, "confirm_immature_cohorts": True},
        task_id=task.id,
    )
    assert confirmed.ok is True, confirmed.error
    assert confirmed.output["target_col"] == "target"
    assert confirmed.output["bad_definition"]["at_mob"] == 6
    assert confirmed.output["maturity"]["immature_cohorts"] == ["2026-02"]


@pytest.mark.slow
def test_tool_define_label_round_trip_writes_labeled_dataset(tmp_path):
    from marvis.plugins.manifest import ToolRef

    runner, registry, task, backend = _runtime(tmp_path)
    dataset = _register(registry, tmp_path, _dpd_long_frame(), "mature", task.id)
    out = runner.invoke(
        ToolRef("labeling", "define_label"),
        {
            "dataset_id": dataset.id, "id_col": "loan_id", "mob_col": "mob",
            "cohort_col": "cohort", "observation_window": 0, "performance_window": 6,
            "dpd_col": "dpd", "threshold_dpd": 90,
        },
        task_id=task.id,
    )
    assert out.ok is True, out.error
    assert out.output["n_bad"] == 1
    assert out.output["n_good"] == 2
    assert out.output["maturity"]["all_matured"] is True
    # 衍生数据集应可读回, 且带 target 列.
    result_id = out.output["result_dataset_id"]
    labeled = registry.get(result_id)
    frame = backend.read_frame(registry.resolve_path(labeled.id))
    assert "target" in frame.columns
    assert set(frame["target"].dropna().unique()) <= {0.0, 1.0}


@pytest.mark.slow
def test_tool_suggest_bad_definition_bridges_roll_rate(tmp_path):
    from marvis.plugins.manifest import ToolRef

    runner, _registry, task, _backend = _runtime(tmp_path)
    out = runner.invoke(
        ToolRef("labeling", "suggest_bad_definition"),
        {
            "states": ["C", "M1", "M2", "M3+"],
            "matrix": [
                [0.90, 0.10, 0.00, 0.00],
                [0.30, 0.40, 0.30, 0.00],
                [0.05, 0.03, 0.50, 0.42],
                [0.00, 0.00, 0.00, 1.00],
            ],
            "at_mob": 6,
        },
        task_id=task.id,
    )
    assert out.ok is True, out.error
    assert out.output["suggestion"]["threshold_status"] == "M2"
    assert out.output["suggestion"]["at_mob"] == 6
