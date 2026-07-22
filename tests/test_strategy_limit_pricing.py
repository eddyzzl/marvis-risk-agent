"""S6 Commit 2: strategy limit x pricing matrix (A3).

Hand-checks every cell of a 2-band x 2-limit x 2-rate grid against the profit
formula, asserts the PD-proxy red flag when no PD column is supplied, and verifies
the limit_pricing_csv deliverable is written ONLY after the matrix confirmation gate.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from pathlib import Path
import threading

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import (
    DatasetRepository,
    PluginRepository,
    StrategyRepository,
    TaskRepository,
    init_db,
)
from marvis.domain import TaskCreate
from marvis.db_schema import connect
from marvis.files import sha256_file
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.pricing import PricingParams, limit_pricing_matrix
from marvis.plugins.contracts import ToolContext
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.settings import build_settings
from tests.strategy_tool_sample_design_support import (
    materialize_strategy_tool_sample_design,
)


def _params() -> PricingParams:
    return PricingParams(lgd=0.5, funding_rate=0.02, term_months=12, cost_per_loan=1.0, el_ead_max=0.5)


def _cell(result, band, limit, rate):
    for cell in result.matrix:
        if cell.band == band and cell.limit == limit and cell.rate == rate:
            return cell
    raise AssertionError(f"cell not found: {band} {limit} {rate}")


def test_limit_pricing_matrix_hand_calculated_2x2x2():
    # score 1,1,3,3 with band_edges [1,2,3] -> band [1,2) has bad=[1,0] (proxy PD 0.5),
    # band [2,3] has bad=[0,0] (proxy PD 0.0). term_factor = 12/12 = 1.
    df = pd.DataFrame({"score": [1, 1, 3, 3], "bad": [1, 0, 0, 0]})
    params = _params()

    result = limit_pricing_matrix(
        df,
        score_col="score",
        limit_grid=[100.0, 200.0],
        rate_grid=[0.10, 0.20],
        params=params,
        target_col="bad",
        band_edges=[1.0, 2.0, 3.0],
    )

    # Band [1,2), PD proxy 0.5. profit/loan = EAD*rate - EAD*PD*LGD - EAD*fund - cost.
    low = _cell(result, "[1,2)", 100.0, 0.10)
    assert low.pd == 0.5
    assert low.el == pytest.approx(50.0)  # 100*0.5*0.5 * 2 loans
    assert low.ead == pytest.approx(200.0)
    assert low.expected_profit == pytest.approx((100 * 0.10 - 100 * 0.5 * 0.5 - 100 * 0.02 - 1) * 2)
    assert low.expected_profit == pytest.approx(-36.0)
    assert low.roa == pytest.approx(-0.18)
    assert low.feasible is False

    low_hi_rate = _cell(result, "[1,2)", 200.0, 0.20)
    assert low_hi_rate.expected_profit == pytest.approx((200 * 0.20 - 200 * 0.5 * 0.5 - 200 * 0.02 - 1) * 2)
    assert low_hi_rate.expected_profit == pytest.approx(-30.0)

    # Band [2,3], PD 0.0 -> no EL, all feasible.
    hi = _cell(result, "[2,3)", 200.0, 0.20)
    assert hi.pd == 0.0
    assert hi.el == pytest.approx(0.0)
    assert hi.expected_profit == pytest.approx((200 * 0.20 - 0 - 200 * 0.02 - 1) * 2)
    assert hi.expected_profit == pytest.approx(70.0)
    assert hi.roa == pytest.approx(0.175)
    assert hi.feasible is True

    # recommended = per band max-profit feasible cell; only band [2,3) has any.
    assert result.recommended == ({"band": "[2,3)", "limit": 200.0, "rate": 0.20},)


def test_limit_pricing_pd_proxy_and_negative_profit_flags():
    df = pd.DataFrame({"score": [1, 1, 3, 3], "bad": [1, 0, 0, 0]})
    result = limit_pricing_matrix(
        df, score_col="score", limit_grid=[100.0], rate_grid=[0.10],
        params=_params(), target_col="bad", band_edges=[1.0, 2.0, 3.0],
    )
    codes = [flag["code"] for flag in result.red_flags]
    # PD proxy flag first (no pd_col), plus band [1,2) has no feasible cell.
    assert "pd_proxy_used" in codes
    assert "negative_profit_band" in codes


def test_limit_pricing_target_proxy_respects_zero_as_bad_semantics():
    df = pd.DataFrame({"score": [1, 1, 3, 3], "bad": [0, 1, 1, 1]})

    result = limit_pricing_matrix(
        df,
        score_col="score",
        limit_grid=[100.0],
        rate_grid=[0.20],
        params=_params(),
        target_col="bad",
        target_bad_value=0,
        band_edges=[1.0, 2.0, 3.0],
    )

    assert _cell(result, "[1,2)", 100.0, 0.20).pd == pytest.approx(0.5)
    assert _cell(result, "[2,3)", 100.0, 0.20).pd == pytest.approx(0.0)


def test_limit_pricing_uses_pd_col_when_supplied_no_proxy_flag():
    df = pd.DataFrame({"score": [1, 1, 3, 3], "pd": [0.1, 0.1, 0.01, 0.01]})
    result = limit_pricing_matrix(
        df, score_col="score", limit_grid=[100.0], rate_grid=[0.20],
        params=_params(), pd_col="pd", band_edges=[1.0, 2.0, 3.0],
    )
    assert all(flag["code"] != "pd_proxy_used" for flag in result.red_flags)
    low = _cell(result, "[1,2)", 100.0, 0.20)
    assert low.pd == pytest.approx(0.1)  # mean of the band's pd_col


def _tool_runtime(tmp_path):
    settings = build_settings(tmp_path / "workspace")
    init_db(settings.db_path)
    created_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="额度定价测试",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            task_type="strategy",
        )
    )
    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET id = 'task-1' WHERE id = ?",
            (created_task.id,),
        )
    plugin_repo = PluginRepository(settings.db_path)
    plugin_registry = PluginRegistry(plugin_repo)
    load_builtin_packs(plugin_registry, Path(__file__).parents[1] / "marvis" / "packs")
    runner = ToolRunner(
        ToolRegistry(plugin_registry), plugin_repo, python_executable=sys.executable,
        datasets_root=settings.datasets_dir, workspace=settings.workspace,
    )
    registry = DatasetRegistry(
        DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir
    )
    return runner, registry, settings


def _sample_design_ref(settings, dataset, **kwargs) -> dict[str, str]:
    task = TaskRepository(settings.db_path).get_task("task-1")
    assert task is not None
    return materialize_strategy_tool_sample_design(
        settings,
        task,
        dataset,
        **kwargs,
    )


def test_limit_pricing_tool_uses_exact_development_sample_and_target_polarity(
    tmp_path,
):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame(
        {
            "score": [1, 1, 3, 3],
            "bad": [0, 1, 0, 1],
            "split": ["dev", "validation", "dev", "oot"],
        }
    ).to_csv(path, index=False)
    dataset = registry.register_from_upload("task-1", path, role="sample")
    sample_design_ref = _sample_design_ref(
        settings,
        dataset,
        target_bad_value=0,
        split_col="split",
        development_values=["dev"],
        validation_values=["validation"],
        oot_values=["oot"],
    )

    result = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            "dataset_id": dataset.id,
            "score_col": "score",
            "target_col": "bad",
            "sample_design_ref": sample_design_ref,
            "band_edges": [1, 2, 3],
            "limit_grid": [100],
            "rate_grid": [0.5],
            "funding_rate": 0.0,
            "term_months": 12,
            "cost_per_loan": 0.0,
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["sample_design_ref"] == sample_design_ref
    assert result.output["source_evidence"]["sample_design_ref"] == sample_design_ref
    assert result.output["source_evidence"]["analyzed_row_count"] == 2
    assert result.output["assumptions"]["target_bad_value"] == 0
    assert {cell["count"] for cell in result.output["matrix"]} == {1}
    assert {cell["pd"] for cell in result.output["matrix"]} == {1.0}


def test_limit_pricing_pd_mode_still_uses_development_sample_scope(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame(
        {
            "score": [1, 1, 3, 3],
            "bad": [0, 0, 1, 1],
            "pd": [0.1, 0.9, 0.2, 0.8],
            "split": ["dev", "validation", "dev", "oot"],
        }
    ).to_csv(path, index=False)
    dataset = registry.register_from_upload("task-1", path, role="sample")
    sample_design_ref = _sample_design_ref(
        settings,
        dataset,
        split_col="split",
        development_values=["dev"],
        validation_values=["validation"],
        oot_values=["oot"],
    )

    result = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            "dataset_id": dataset.id,
            "score_col": "score",
            "pd_col": "pd",
            "sample_design_ref": sample_design_ref,
            "band_edges": [1, 2, 3],
            "limit_grid": [100],
            "rate_grid": [0.5],
            "funding_rate": 0.0,
            "term_months": 12,
            "cost_per_loan": 0.0,
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert result.output["sample_design_ref"] == sample_design_ref
    assert result.output["assumptions"]["population_scope"] == "development"
    assert [cell["pd"] for cell in result.output["matrix"]] == [0.1, 0.2]
    assert {cell["count"] for cell in result.output["matrix"]} == {1}


def test_limit_pricing_matrix_artifact_only_after_confirm(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    df = pd.DataFrame({"score": [1, 1, 3, 3], "bad": [1, 0, 0, 0]})
    path = tmp_path / "pricing.csv"
    df.to_csv(path, index=False)
    ds = registry.register_from_upload("task-1", path, role="sample")
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "pricing", "rules": [{"condition": "score < 2", "decision": "price", "value": 0.25}],
         "score_col": "score", "default_decision": 0.15},
        task_id="task-1",
    )
    strategy_id = built.output["strategy_id"]
    base = {
        "dataset_id": ds.id, "score_col": "score", "target_col": "bad",
        "sample_design_ref": _sample_design_ref(settings, ds),
        "limit_grid": [100, 200], "rate_grid": [0.1, 0.2],
        "funding_rate": 0.02, "term_months": 12, "cost_per_loan": 1.0, "lgd": 0.5,
        "el_ead_max": 0.5, "strategy_id": strategy_id,
    }
    strategies = StrategyRepository(settings.db_path)

    # Unconfirmed: matrix returned, but NO artifact written.
    unconfirmed = runner.invoke(ToolRef("strategy", "limit_pricing_matrix"), base, task_id="task-1")
    assert unconfirmed.ok is True, unconfirmed.error
    assert unconfirmed.output["artifacts"] == []
    assert strategies.list_strategy_artifacts(strategy_id) == []

    # Confirmed: the limit_pricing_csv deliverable is written and registered + audited.
    confirmed = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"), {**base, "confirm": True}, task_id="task-1"
    )
    assert confirmed.ok is True, confirmed.error
    assert [a["kind"] for a in confirmed.output["artifacts"]] == [
        "limit_pricing_csv",
        "limit_pricing_markdown",
    ]
    assert all("path" not in item for item in confirmed.output["artifacts"])
    task_records = TaskArtifactRepository(settings.db_path).list_for_task("task-1")
    pricing_records = [
        record for record in task_records if record["kind"].startswith("limit_pricing_")
    ]
    assert {record["kind"] for record in pricing_records} == {
        "limit_pricing_csv",
        "limit_pricing_markdown",
    }
    assert all(
        record["provenance"]["assumptions"]["sample_design_ref"]
        == base["sample_design_ref"]
        for record in pricing_records
    )
    kinds = [a["kind"] for a in strategies.list_strategy_artifacts(strategy_id)]
    assert kinds == ["limit_pricing_csv", "limit_pricing_markdown"]
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT detail_json FROM audit WHERE kind='strategy.artifact' "
            "AND detail_json LIKE '%limit_pricing_%'"
        ).fetchall()
    assert len(rows) == 2


def test_limit_pricing_confirmed_export_is_task_owned_without_strategy(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame(
        {"score": [1, 2, 3], "pd": [0.1, 0.05, 0.01], "bad": [0, 0, 1]}
    ).to_csv(path, index=False)
    ds = registry.register_from_upload("task-1", path, role="sample")
    sample_design_ref = _sample_design_ref(settings, ds)

    result = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            "dataset_id": ds.id,
            "score_col": "score",
            "pd_col": "pd",
            "sample_design_ref": sample_design_ref,
            "limit_grid": [100, 200],
            "rate_grid": [0.1, 0.2],
            "funding_rate": 0.02,
            "term_months": 12,
            "cost_per_loan": 1.0,
            "confirm": True,
        },
        task_id="task-1",
    )

    assert result.ok is True, result.error
    assert {item["kind"] for item in result.output["artifacts"]} == {
        "limit_pricing_csv",
        "limit_pricing_markdown",
    }
    assert result.output["source_evidence"]["dataset_content_hash"]
    assert all("path" not in artifact for artifact in result.output["artifacts"])
    repository = TaskArtifactRepository(settings.db_path)
    records = [
        record
        for record in repository.list_for_task("task-1")
        if record["kind"].startswith("limit_pricing_")
    ]
    assert {record["id"] for record in records} == {
        artifact["artifact_id"] for artifact in result.output["artifacts"]
    }
    for record in records:
        artifact_path = Path(record["path"])
        assert artifact_path.is_file()
        artifact_path.relative_to(settings.tasks_dir / "task-1")

    replay = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            "dataset_id": ds.id,
            "score_col": "score",
            "pd_col": "pd",
            "sample_design_ref": sample_design_ref,
            "limit_grid": [100, 200],
            "rate_grid": [0.1, 0.2],
            "funding_rate": 0.02,
            "term_months": 12,
            "cost_per_loan": 1.0,
            "confirm": True,
        },
        task_id="task-1",
    )
    assert replay.ok is True, replay.error
    assert [item["artifact_id"] for item in replay.output["artifacts"]] == [
        item["artifact_id"] for item in result.output["artifacts"]
    ]
    assert len(
        [
            record
            for record in repository.list_for_task("task-1")
            if record["kind"].startswith("limit_pricing_")
        ]
    ) == 2


def test_failed_limit_pricing_registration_rolls_back_before_identical_peer_promotes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    _runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame(
        {"score": [1, 2, 3], "pd": [0.1, 0.05, 0.01], "bad": [0, 0, 1]}
    ).to_csv(path, index=False)
    dataset = registry.register_from_upload("task-1", path, role="sample")
    inputs = {
        "dataset_id": dataset.id,
        "score_col": "score",
        "pd_col": "pd",
        "sample_design_ref": _sample_design_ref(settings, dataset),
        "limit_grid": [100, 200],
        "rate_grid": [0.1, 0.2],
        "funding_rate": 0.02,
        "term_months": 12,
        "cost_per_loan": 1.0,
        "confirm": True,
    }
    context = ToolContext(
        task_id="task-1",
        seed=None,
        datasets_root=settings.datasets_dir,
        workspace=settings.workspace,
    )
    failing_db_exited = threading.Event()
    release_failing_writer = threading.Event()
    original_register = TaskArtifactRepository.register_on_connection
    original_transaction = TaskArtifactRepository.transaction
    failing_registration_calls = 0

    def fail_second_for_first_writer(self, conn, **kwargs):
        nonlocal failing_registration_calls
        if threading.current_thread().name == "failing-writer":
            failing_registration_calls += 1
            if failing_registration_calls == 2:
                raise RuntimeError("injected post-promotion registration failure")
        return original_register(self, conn, **kwargs)

    @contextmanager
    def pause_failed_writer_after_db_exit(self):
        try:
            with original_transaction(self) as conn:
                yield conn
        finally:
            if threading.current_thread().name == "failing-writer":
                failing_db_exited.set()
                if not release_failing_writer.wait(timeout=10):
                    raise RuntimeError("timed out waiting to release failed writer")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "register_on_connection",
        fail_second_for_first_writer,
    )
    monkeypatch.setattr(
        TaskArtifactRepository,
        "transaction",
        pause_failed_writer_after_db_exit,
    )
    failures: dict[str, BaseException] = {}
    outputs: dict[str, dict] = {}

    def invoke(name: str) -> None:
        try:
            outputs[name] = strategy_tools.tool_limit_pricing_matrix(inputs, context)
        except BaseException as exc:  # captured for assertions in the main thread
            failures[name] = exc

    failing = threading.Thread(
        target=invoke,
        args=("failing",),
        name="failing-writer",
    )
    peer = threading.Thread(target=invoke, args=("peer",), name="peer-writer")
    failing.start()
    assert failing_db_exited.wait(timeout=10)
    peer.start()
    peer.join(timeout=10)
    assert not peer.is_alive()
    assert "peer" not in failures
    assert len(outputs["peer"]["artifacts"]) == 2

    # Resume the failed writer only after the peer has committed the same
    # deterministic final paths. A rollback delayed until after DB exit would
    # delete the peer's durable reports.
    release_failing_writer.set()
    failing.join(timeout=10)

    assert not failing.is_alive()
    assert isinstance(failures.get("failing"), RuntimeError)
    records = [
        record
        for record in TaskArtifactRepository(settings.db_path).list_for_task("task-1")
        if record["kind"].startswith("limit_pricing_")
    ]
    assert len(records) == 2
    for record in records:
        artifact_path = Path(record["path"])
        assert artifact_path.is_file()
        assert sha256_file(artifact_path) == record["content_hash"]


def test_limit_pricing_export_rejects_source_changed_since_preview(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame(
        {"score": [1, 2, 3], "pd": [0.1, 0.05, 0.01], "bad": [0, 0, 1]}
    ).to_csv(path, index=False)
    ds = registry.register_from_upload("task-1", path, role="sample")
    base = {
        "dataset_id": ds.id,
        "score_col": "score",
        "pd_col": "pd",
        "sample_design_ref": _sample_design_ref(settings, ds),
        "limit_grid": [100],
        "rate_grid": [0.2],
        "funding_rate": 0.02,
        "term_months": 12,
        "cost_per_loan": 1.0,
    }
    preview = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"), base, task_id="task-1"
    )
    assert preview.ok is True, preview.error
    expected_hash = preview.output["source_evidence"]["dataset_content_hash"]
    pd.DataFrame({"score": [1, 2, 3], "pd": [0.9, 0.8, 0.7]}).to_parquet(
        registry.resolve_path(ds.id), index=False
    )

    export = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {**base, "confirm": True, "expected_source_hash": expected_hash},
        task_id="task-1",
    )

    assert export.ok is False
    assert "immutable hash" in str(export.error)


@pytest.mark.parametrize(
    "risk_inputs",
    [{}, {"target_col": "bad", "pd_col": "pd"}],
)
def test_limit_pricing_requires_exactly_one_risk_source(tmp_path, risk_inputs):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame({"score": [1, 2], "bad": [0, 1], "pd": [0.1, 0.2]}).to_csv(path, index=False)
    ds = registry.register_from_upload("task-1", path, role="sample")
    sample_design_ref = _sample_design_ref(settings, ds)
    result = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            "dataset_id": ds.id,
            "score_col": "score",
            "sample_design_ref": sample_design_ref,
            "limit_grid": [100],
            "rate_grid": [0.1],
            "funding_rate": 0.02,
            "term_months": 12,
            "cost_per_loan": 1.0,
            **risk_inputs,
        },
        task_id="task-1",
    )
    assert result.ok is False
    assert "exactly one" in str(result.error)


def test_limit_pricing_rejects_excessive_grid_and_invalid_pd(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame({"score": [1, 2], "pd": [0.1, 1.2], "bad": [0, 1]}).to_csv(
        path, index=False
    )
    ds = registry.register_from_upload("task-1", path, role="sample")
    base = {
        "dataset_id": ds.id,
        "score_col": "score",
        "pd_col": "pd",
        "sample_design_ref": _sample_design_ref(settings, ds),
        "funding_rate": 0.02,
        "term_months": 12,
        "cost_per_loan": 1.0,
    }
    too_large = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {
            **base,
            "limit_grid": list(range(1, 101)),
            "rate_grid": [index / 100 for index in range(100)],
        },
        task_id="task-1",
    )
    assert too_large.ok is False
    assert "too large" in str(too_large.error)

    invalid_pd = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {**base, "limit_grid": [100], "rate_grid": [0.1]},
        task_id="task-1",
    )
    assert invalid_pd.ok is False
    assert "pd_col values" in str(invalid_pd.error)


def test_limit_pricing_rejects_wrong_type_and_cross_task_strategy_attachment(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    path = tmp_path / "pricing.csv"
    pd.DataFrame({"score": [1, 2], "pd": [0.1, 0.2], "bad": [0, 1]}).to_csv(
        path, index=False
    )
    ds = registry.register_from_upload("task-1", path, role="sample")
    common = {
        "dataset_id": ds.id,
        "score_col": "score",
        "pd_col": "pd",
        "sample_design_ref": _sample_design_ref(settings, ds),
        "limit_grid": [100],
        "rate_grid": [0.1],
        "funding_rate": 0.02,
        "term_months": 12,
        "cost_per_loan": 1.0,
        "confirm": True,
    }
    wrong_type = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval", "rules": [{"condition": "score < 2", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id="task-1",
    )
    wrong = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {**common, "strategy_id": wrong_type.output["strategy_id"]},
        task_id="task-1",
    )
    assert wrong.ok is False
    assert "limit or pricing" in str(wrong.error)

    other_task = TaskRepository(settings.db_path).create_task(
        TaskCreate(
            model_name="other",
            model_version="dev",
            validator="qa",
            source_dir=str(tmp_path),
            algorithm="lr",
            run_mode="agent",
            target_col="bad",
            score_col="score",
            split_col="split",
            time_col="month",
            feature_columns=["score"],
        )
    )
    cross = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "pricing", "rules": [{"condition": "score < 2", "decision": "price", "value": 0.2}],
         "score_col": "score", "default_decision": 0.1},
        task_id=other_task.id,
    )
    rejected = runner.invoke(
        ToolRef("strategy", "limit_pricing_matrix"),
        {**common, "strategy_id": cross.output["strategy_id"]},
        task_id="task-1",
    )
    assert rejected.ok is False
    assert "strategy not found" in str(rejected.error)
