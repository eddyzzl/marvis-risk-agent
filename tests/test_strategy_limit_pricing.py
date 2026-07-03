"""S6 Commit 2: strategy limit x pricing matrix (A3).

Hand-checks every cell of a 2-band x 2-limit x 2-rate grid against the profit
formula, asserts the PD-proxy red flag when no PD column is supplied, and verifies
the limit_pricing_csv deliverable is written ONLY after the matrix confirmation gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, PluginRepository, StrategyRepository, init_db
from marvis.db_schema import connect
from marvis.packs.strategy.pricing import PricingParams, limit_pricing_matrix
from marvis.plugins.loader import load_builtin_packs
from marvis.plugins.manifest import ToolRef
from marvis.plugins.registry import PluginRegistry, ToolRegistry
from marvis.plugins.runner import ToolRunner
from marvis.settings import build_settings


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


def test_limit_pricing_matrix_artifact_only_after_confirm(tmp_path):
    runner, registry, settings = _tool_runtime(tmp_path)
    df = pd.DataFrame({"score": [1, 1, 3, 3], "bad": [1, 0, 0, 0]})
    path = tmp_path / "pricing.csv"
    df.to_csv(path, index=False)
    ds = registry.register_from_upload("task-1", path, role="sample")
    built = runner.invoke(
        ToolRef("strategy", "build_strategy"),
        {"strategy_type": "approval", "rules": [{"condition": "score < 2", "decision": "reject"}],
         "score_col": "score", "default_decision": "approve"},
        task_id="task-1",
    )
    strategy_id = built.output["strategy_id"]
    base = {
        "dataset_id": ds.id, "score_col": "score", "target_col": "bad",
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
    assert [a["kind"] for a in confirmed.output["artifacts"]] == ["limit_pricing_csv"]
    kinds = [a["kind"] for a in strategies.list_strategy_artifacts(strategy_id)]
    assert kinds == ["limit_pricing_csv"]
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT detail_json FROM audit WHERE kind='strategy.artifact' "
            "AND detail_json LIKE '%limit_pricing_csv%'"
        ).fetchall()
    assert len(rows) == 1
