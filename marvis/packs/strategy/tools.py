from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from marvis.data.backend import DataBackend
from marvis.data.direction import check_score_direction, normalize_score_direction
from marvis.data.labels import resolve_labeled_frame
from marvis.data.registry import DatasetRegistry
from marvis.db import DatasetRepository, StrategyRepository
from marvis.packs.strategy.backtest import backtest_strategy
from marvis.packs.strategy.bands import design_cutoff_bands
from marvis.packs.strategy.compare import compare_strategies
from marvis.packs.strategy.contracts import BacktestResult, Strategy
from marvis.packs.strategy.deliverables import decision_table_csv
from marvis.packs.strategy.doc import render_strategy_doc_markdown
from marvis.packs.strategy.monitor_tools import (  # noqa: F401
    tool_render_monitoring_report,
    tool_run_strategy_monitoring,
)
from marvis.packs.strategy.monitoring_plan import (
    build_monitoring_plan,
    save_monitoring_plan,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pricing import (
    LimitPricingResult,
    PricingParams,
    limit_pricing_matrix,
)
from marvis.packs.strategy.profit import ProfitParams, profit_calc
from marvis.packs.strategy.roll_rate import roll_rate_matrix
from marvis.packs.strategy.rules import (
    DEFAULT_MINE_SEED,
    evaluate_rule_set,
    mine_rules,
)
from marvis.packs.strategy.strategy import build_strategy, infer_strategy_rule_direction
from marvis.packs.strategy.tradeoff import (
    recommend_operating_point,
    tradeoff_feasible_flags,
    tradeoff_view,
)
from marvis.packs.strategy.vintage import vintage_curve, vintage_summary
from marvis.settings import build_settings


def tool_vintage_curve(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        columns=[str(inputs["cohort_col"]), str(inputs["mob_col"]), str(inputs["bad_col"])],
    )
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame,
        str(inputs["bad_col"]),
        drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    curve = vintage_curve(
        frame,
        cohort_col=str(inputs["cohort_col"]),
        mob_col=str(inputs["mob_col"]),
        bad_col=str(inputs["bad_col"]),
        mob_max=int(inputs.get("mob_max", 12)),
    )
    return {
        "cohorts": list(curve.cohorts),
        "mob_axis": list(curve.mob_axis),
        "curves": _jsonable(curve.curves),
        "counts": _jsonable(curve.counts),
        "summary": vintage_summary(curve, ref_mob=int(inputs.get("ref_mob", 6))),
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_roll_rate(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(
        runtime,
        str(inputs["dataset_id"]),
        columns=[str(inputs["id_col"]), str(inputs["time_col"]), str(inputs["status_col"])],
    )
    matrix = roll_rate_matrix(
        frame,
        id_col=str(inputs["id_col"]),
        time_col=str(inputs["time_col"]),
        status_col=str(inputs["status_col"]),
        states=[str(item) for item in inputs["states"]],
    )
    return {
        "states": list(matrix.states),
        "matrix": [list(row) for row in matrix.matrix],
        "base_counts": dict(matrix.base_counts),
    }


def tool_profit_calc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    segment_col = _optional_str(inputs.get("segment_col"))
    columns = _unique([segment_col, str(inputs["ead_col"]), str(inputs["pd_col"])])
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]), columns=columns)
    results = profit_calc(
        frame,
        segment_col=segment_col,
        ead_col=str(inputs["ead_col"]),
        pd_col=str(inputs["pd_col"]),
        params=_profit_params(inputs["params"]),
    )
    return {"results": [_jsonable(result) for result in results]}


def tool_build_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy = build_strategy(
        str(inputs["strategy_type"]),
        list(inputs["rules"]),
        score_col=_optional_str(inputs.get("score_col")),
        default_decision=inputs.get("default_decision"),
        description=str(inputs.get("description") or ""),
    )
    if runtime.strategies.get_strategy(strategy.id) is None:
        runtime.strategies.create_strategy_with_audit(
            ctx.task_id,
            strategy,
            audit={
                "kind": "strategy.create",
                "target_ref": strategy.id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_type": strategy.strategy_type,
                    "rule_count": len(strategy.rules),
                },
            },
        )
    return {
        "strategy_id": strategy.id,
        "strategy_type": strategy.strategy_type,
        "score_col": strategy.score_col,
        "default_decision": strategy.default_decision,
        "description": strategy.description,
        "rules": [_jsonable(rule) for rule in strategy.rules],
        "inferred_score_direction": infer_strategy_rule_direction(list(strategy.rules), strategy.score_col),
    }


def tool_backtest_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy = _strategy(runtime, str(inputs["strategy_id"]))
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    baseline = _strategy(runtime, baseline_id) if baseline_id else None
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = backtest_strategy(
        frame,
        strategy,
        target_col=str(inputs["target_col"]),
        baseline=baseline,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    backtest_id = _backtest_id(str(inputs["dataset_id"]), result)
    if runtime.strategies.get_backtest(backtest_id) is None:
        runtime.strategies.save_backtest_with_audit(
            backtest_id,
            strategy.id,
            str(inputs["dataset_id"]),
            result,
            audit={
                "kind": "strategy.backtest",
                "target_ref": backtest_id,
                "outcome": "succeeded",
                "detail": {
                    "task_id": str(ctx.task_id),
                    "strategy_id": strategy.id,
                    "dataset_id": str(inputs["dataset_id"]),
                    "approval_rate": result.approval_rate,
                    "expected_profit": result.expected_profit,
                },
            },
        )
    payload = _jsonable(result)
    payload["backtest_id"] = backtest_id
    payload["nan_labels_dropped"] = nan_labels_dropped
    return payload


def tool_tradeoff_view(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_col = str(inputs["score_col"])
    target_col = str(inputs["target_col"])
    score_direction = normalize_score_direction(_optional_str(inputs.get("score_direction")))
    effective_direction = score_direction or "higher_is_better"
    points = tradeoff_view(
        frame,
        score_col=score_col,
        target_col=target_col,
        cutoffs=[float(item) for item in inputs["cutoffs"]] if inputs.get("cutoffs") is not None else None,
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
        score_direction=score_direction,
        confirm_direction_conflict=bool(inputs.get("confirm_direction_conflict")),
    )
    max_bad_rate = _optional_float(inputs.get("max_bad_rate"))
    min_approval_rate = _optional_float(inputs.get("min_approval_rate"))
    feasible_flags = tradeoff_feasible_flags(
        points, max_bad_rate=max_bad_rate, min_approval_rate=min_approval_rate
    )
    red_flags: list[dict] = []
    recommended = None
    if points and any(feasible_flags):
        recommended = recommend_operating_point(
            [point for point, ok in zip(points, feasible_flags, strict=True) if ok],
            objective=str(inputs.get("objective") or "max_profit"),
            max_bad_rate=max_bad_rate,
        )
    elif points and (max_bad_rate is not None or min_approval_rate is not None):
        red_flags.append(
            {
                "code": "infeasible_constraints",
                "level": "red",
                "message": "在给定 max_bad_rate/min_approval_rate 约束下没有可行 cutoff。",
            }
        )
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    point_rows = []
    for point, feasible in zip(points, feasible_flags, strict=True):
        row = _jsonable(point)
        row["feasible"] = bool(feasible)
        point_rows.append(row)
    result = {
        "points": point_rows,
        "recommended": _jsonable(recommended),
        "nan_labels_dropped": nan_labels_dropped,
        "score_direction": effective_direction,
        "red_flags": red_flags,
    }
    if direction_check.status != "skipped":
        result["direction_diagnostics"] = _jsonable(direction_check)
    return result


def tool_design_cutoff_bands(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    score_col = str(inputs["score_col"])
    target_col = str(inputs["target_col"])
    score_direction = normalize_score_direction(_optional_str(inputs.get("score_direction")))
    effective_direction = score_direction or "higher_is_better"
    red_flags: list[dict] = []
    # Direction self-check (S1a): a conflict is a red flag and blocks unless the
    # caller confirms, mirroring tradeoff_view's confirm_direction_conflict gate.
    direction_check = check_score_direction(
        pd.to_numeric(frame[score_col], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=float),
        declared_direction=effective_direction,
    )
    if direction_check.status == "conflict" and not bool(inputs.get("confirm_direction_conflict")):
        from marvis.data.errors import ScoreDirectionConflictError

        raise ScoreDirectionConflictError(
            tool="design_cutoff_bands",
            score_col=score_col,
            target_col=target_col,
            declared_direction=effective_direction,
            implied_direction=direction_check.implied_direction,
            corr=direction_check.corr,
            n_labeled=direction_check.n,
        )
    if direction_check.status == "conflict":
        red_flags.append(
            {
                "code": "direction_conflict",
                "level": "red",
                "message": (
                    f"分数方向自检冲突：声明 {effective_direction}，数据隐含 "
                    f"{direction_check.implied_direction}（corr={direction_check.corr:.3f}）。"
                ),
            }
        )
    result = design_cutoff_bands(
        frame,
        score_col=score_col,
        target_col=target_col,
        score_direction=effective_direction,
        n_bands=int(inputs.get("n_bands", 5)),
        band_edges=[float(edge) for edge in inputs["band_edges"]]
        if inputs.get("band_edges") is not None
        else None,
        objective=str(inputs.get("objective") or "max_profit"),
        max_bad_rate=_optional_float(inputs.get("max_bad_rate")),
        min_approval_rate=_optional_float(inputs.get("min_approval_rate")),
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    red_flags.extend(_jsonable(flag) for flag in result.red_flags)
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return {
        "bands": [_jsonable(band) for band in result.bands],
        "band_edges": [float(edge) for edge in result.band_edges],
        "recommended_rules": [dict(rule) for rule in result.recommended_rules],
        "red_flags": red_flags,
        "score_direction": effective_direction,
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_compare_strategies(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    baseline_id = _optional_str(inputs.get("baseline_strategy_id"))
    if baseline_id is None:
        # No baseline supplied (e.g. the template's optional compare step ran
        # without a baseline_strategy_id slot): degrade to a no-op result
        # instead of failing the plan -- the step is informational, not gating.
        return {
            "matrix_2x2": {
                cell: {"count": 0, "bad_rate": 0.0}
                for cell in ("both_approve", "only_new", "only_baseline", "both_decline")
            },
            "deltas": {"approval_rate": 0.0, "approved_bad_rate": 0.0, "expected_profit": 0.0},
            "summary_text": "未提供基线策略，跳过对比。",
            "red_flags": [],
            "nan_labels_dropped": 0,
        }
    strategy = _strategy(runtime, str(inputs["strategy_id"]))
    baseline = _strategy(runtime, baseline_id)
    frame = _dataset_frame(runtime, str(inputs["dataset_id"]))
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, str(inputs["target_col"]), drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = compare_strategies(
        frame,
        strategy,
        baseline,
        target_col=str(inputs["target_col"]),
        profit_params=_optional_profit_params(inputs.get("profit_params")),
        ead_col=_optional_str(inputs.get("ead_col")),
        pd_col=_optional_str(inputs.get("pd_col")),
    )
    payload = _jsonable(result)
    payload["nan_labels_dropped"] = nan_labels_dropped
    return payload


def tool_limit_pricing_matrix(inputs: dict, ctx) -> dict:
    """S6 (A3): a band x limit x rate expected-profit grid with an EL simulation.

    Always computes and returns the full matrix + per-band recommended feasible cell.
    The strategy_artifacts(kind='limit_pricing_csv') deliverable is written ONLY when
    ``confirm`` is true -- the driver flips it after the matrix confirmation gate
    (矩阵确认门后才落 artifact), mirroring adopt_strategy's forced-gate precedent. The
    CSV is attached to ``strategy_id`` so it rides the same per-strategy artifact list.
    """
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    score_col = str(inputs["score_col"])
    target_col = _optional_str(inputs.get("target_col"))
    pd_col = _optional_str(inputs.get("pd_col"))
    columns = _unique([score_col, target_col, pd_col])
    frame = _dataset_frame(runtime, dataset_id, columns=columns)
    if target_col:
        frame, nan_labels_dropped = resolve_labeled_frame(
            frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
        )
    else:
        nan_labels_dropped = 0

    params = PricingParams(
        lgd=float(inputs.get("lgd", 0.6)),
        funding_rate=float(inputs["funding_rate"]),
        term_months=int(inputs["term_months"]),
        cost_per_loan=float(inputs["cost_per_loan"]),
        el_ead_max=float(inputs.get("el_ead_max", 0.20)),
    )
    result = limit_pricing_matrix(
        frame,
        score_col=score_col,
        limit_grid=[float(item) for item in inputs["limit_grid"]],
        rate_grid=[float(item) for item in inputs["rate_grid"]],
        params=params,
        target_col=target_col,
        pd_col=pd_col,
        band_edges=[float(edge) for edge in inputs["band_edges"]]
        if inputs.get("band_edges") is not None
        else None,
        n_bands=int(inputs.get("n_bands", 5)),
    )
    red_flags = [dict(flag) for flag in result.red_flags]
    if nan_labels_dropped:
        red_flags.append({
            "code": "nan_labels_dropped",
            "level": "amber",
            "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
        })

    assumptions = {
        "dataset_id": dataset_id,
        "score_col": score_col,
        "target_col": target_col,
        "pd_col": pd_col,
        "lgd": params.lgd,
        "funding_rate": params.funding_rate,
        "term_months": params.term_months,
        "cost_per_loan": params.cost_per_loan,
        "el_ead_max": params.el_ead_max,
        "limit_grid": [float(item) for item in inputs["limit_grid"]],
        "rate_grid": [float(item) for item in inputs["rate_grid"]],
        "band_edges": [float(edge) for edge in result.band_edges],
        "n_bands": int(inputs.get("n_bands", 5)),
    }

    payload = {
        "matrix": [_jsonable(cell) for cell in result.matrix],
        "recommended": [dict(item) for item in result.recommended],
        "band_edges": [float(edge) for edge in result.band_edges],
        "assumptions": assumptions,
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
    }

    strategy_id = _optional_str(inputs.get("strategy_id"))
    artifacts: list[dict] = []
    # 矩阵确认门后才落 artifact: only after the user confirms the matrix does the CSV
    # deliverable get written and registered (adopt_strategy forced-gate precedent).
    if bool(inputs.get("confirm")) and strategy_id:
        strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        csv_path = strategy_dir / f"limit_pricing_{strategy_id}.csv"
        csv_path.write_text(_limit_pricing_csv(result), encoding="utf-8")
        runtime.strategies.save_strategy_artifact(
            strategy_id, kind="limit_pricing_csv", path=str(csv_path)
        )
        _write_strategy_artifact_audit(runtime, ctx, strategy_id, "limit_pricing_csv", csv_path)
        artifacts.append({"kind": "limit_pricing_csv", "path": str(csv_path)})
    payload["artifacts"] = artifacts
    return payload


def _limit_pricing_csv(result: LimitPricingResult) -> str:
    header = "band,limit,rate,count,pd,el,ead,expected_profit,roa,feasible,recommended"
    recommended = {
        (item["band"], float(item["limit"]), float(item["rate"])) for item in result.recommended
    }
    lines = [header]
    for cell in result.matrix:
        is_reco = (cell.band, float(cell.limit), float(cell.rate)) in recommended
        lines.append(
            ",".join([
                cell.band,
                _csv_num(cell.limit),
                _csv_num(cell.rate),
                str(cell.count),
                _csv_num(cell.pd),
                _csv_num(cell.el),
                _csv_num(cell.ead),
                _csv_num(cell.expected_profit),
                _csv_num(cell.roa),
                "1" if cell.feasible else "0",
                "1" if is_reco else "0",
            ])
        )
    return "\n".join(lines) + "\n"


def _csv_num(value) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.6f}"


def tool_adopt_strategy(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id)
    backtest_id = str(inputs["backtest_id"])
    backtest = runtime.strategies.get_backtest(backtest_id)
    if backtest is None or backtest.strategy_id != strategy_id:
        raise StrategyError(
            f"backtest {backtest_id} does not belong to strategy {strategy_id}"
        )
    adoption_reason = str(inputs["adoption_reason"])
    adopt_result = runtime.strategies.adopt_strategy_with_audit(
        strategy_id,
        reason=adoption_reason,
        audit={
            "kind": "strategy.adopt",
            "target_ref": strategy_id,
            "outcome": "succeeded",
            "detail": {
                "task_id": str(ctx.task_id),
                "backtest_id": backtest_id,
                "approval_rate": backtest.approval_rate,
                "approved_bad_rate": backtest.approved_bad_rate,
                "expected_profit": backtest.expected_profit,
            },
        },
    )
    version = int(adopt_result["version"])
    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{strategy_id}_v{version}"

    band_stats = _band_stats_from_inputs(inputs.get("band_stats"))
    rules = [_jsonable(rule) for rule in strategy.rules]
    csv_text = decision_table_csv(rules, band_stats)
    csv_path = strategy_dir / f"decision_table_{stem}.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    monitoring_plan = build_monitoring_plan(
        strategy_id=strategy_id,
        version=version,
        approved_bad_rate=backtest.approved_bad_rate,
        approval_rate=backtest.approval_rate,
        experiment_id=_optional_str(inputs.get("experiment_id")),
        source_backtest_id=backtest_id,
    )
    json_path = strategy_dir / f"monitoring_plan_{stem}.json"
    save_monitoring_plan(json_path, monitoring_plan)

    artifacts = []
    for kind, path in (
        ("decision_table_csv", csv_path),
        ("monitoring_plan_json", json_path),
    ):
        runtime.strategies.save_strategy_artifact(strategy_id, kind=kind, path=str(path))
        _write_strategy_artifact_audit(runtime, ctx, strategy_id, kind, path)
        artifacts.append({"kind": kind, "path": str(path)})

    return {
        "strategy_id": strategy_id,
        "version": version,
        "status": "adopted",
        "retired_strategy_ids": list(adopt_result["retired_strategy_ids"]),
        "artifacts": artifacts,
    }


def tool_render_challenger_report(inputs: dict, ctx) -> dict:
    """S6 Commit 3: assemble a challenger-vs-champion Markdown report from the compare
    output + both backtests + the adoption status, register it as
    strategy_artifacts(kind='challenger_report_md'), and audit it.

    Graceful degradation (compare_strategies precedent): with no champion/baseline the
    report is a no-op — it returns status='no_baseline' + a 「未提供基线」 markdown and
    writes NO artifact, so an optional template step that ran without a champion slot
    does not fail the plan. Every number in the report comes straight from the passed-in
    compare/backtest tool outputs (INV-1: presentation only, report follows tool output).
    """
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    champion_id = _optional_str(inputs.get("champion_strategy_id"))
    compare = _as_dict(inputs.get("compare"))
    # A compare that itself degraded to the no-baseline no-op carries this text; treat it
    # as "no champion" too so the report degrades in lockstep with compare_strategies.
    compare_degraded = str(compare.get("summary_text") or "").startswith("未提供基线")

    if not champion_id or compare_degraded:
        markdown = "# 挑战者对比报告\n\n未提供基线（champion）策略，跳过对比报告。\n"
        return {
            "status": "no_baseline",
            "report_md": markdown,
            "artifacts": [],
        }

    challenger_backtest = _as_dict(inputs.get("challenger_backtest"))
    champion_backtest = _as_dict(inputs.get("champion_backtest"))
    adopted = bool(inputs.get("adopted"))
    markdown = _challenger_report_markdown(
        strategy_id=strategy_id,
        champion_id=champion_id,
        compare=compare,
        challenger_backtest=challenger_backtest,
        champion_backtest=champion_backtest,
        adopted=adopted,
    )

    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    report_path = strategy_dir / f"challenger_report_{strategy_id}.md"
    report_path.write_text(markdown, encoding="utf-8")
    runtime.strategies.save_strategy_artifact(
        strategy_id, kind="challenger_report_md", path=str(report_path)
    )
    _write_strategy_artifact_audit(runtime, ctx, strategy_id, "challenger_report_md", report_path)
    return {
        "status": "rendered",
        "report_md": markdown,
        "report_path": str(report_path),
        "artifacts": [{"kind": "challenger_report_md", "path": str(report_path)}],
    }


def _challenger_report_markdown(
    *,
    strategy_id: str,
    champion_id: str,
    compare: dict,
    challenger_backtest: dict,
    champion_backtest: dict,
    adopted: bool,
) -> str:
    deltas = _as_dict(compare.get("deltas"))
    lines = [
        "# 挑战者对比报告",
        "",
        f"- 挑战者策略：`{strategy_id}`",
        f"- 基线（champion）策略：`{champion_id}`",
        f"- 采纳状态：{'已采纳挑战者' if adopted else '未采纳（仍以基线为准）'}",
        "",
        "## 关键指标并排",
        "",
        "| 指标 | 挑战者 | 基线 | 挑战者−基线 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (
        ("审批率", "approval_rate"),
        ("通过客群坏率", "approved_bad_rate"),
        ("预期利润", "expected_profit"),
    ):
        lines.append(
            f"| {label} | {_report_num(challenger_backtest.get(key))} | "
            f"{_report_num(champion_backtest.get(key))} | {_report_num(deltas.get(key))} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        str(compare.get("summary_text") or ""),
        "",
    ])
    red_flags = [flag for flag in (compare.get("red_flags") or []) if isinstance(flag, dict)]
    if red_flags:
        lines.append("## 红旗")
        lines.append("")
        for flag in red_flags:
            lines.append(f"- [{flag.get('level', '')}] {flag.get('code', '')}: {flag.get('message', '')}")
        lines.append("")
    return "\n".join(lines)


def _report_num(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _as_dict(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def tool_render_strategy_doc(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    strategy_id = str(inputs["strategy_id"])
    strategy = _strategy(runtime, strategy_id)
    meta = runtime.strategies.get_strategy_meta(strategy_id)
    backtests = [_jsonable(result) for result in runtime.strategies.list_backtests(strategy_id)]
    artifacts = runtime.strategies.list_strategy_artifacts(strategy_id)
    band_stats = _band_stats_from_inputs(inputs.get("band_stats"))
    markdown, sections = render_strategy_doc_markdown(
        strategy=_jsonable(strategy),
        meta=meta or {},
        backtests=backtests,
        artifacts=artifacts,
        band_stats=band_stats,
    )
    strategy_dir = Path(runtime.settings.tasks_dir) / str(ctx.task_id) / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    version = int((meta or {}).get("version", 1))
    doc_path = strategy_dir / f"strategy_doc_{strategy_id}_v{version}.md"
    doc_path.write_text(markdown, encoding="utf-8")
    runtime.strategies.save_strategy_artifact(
        strategy_id, kind="strategy_doc_md", path=str(doc_path)
    )
    _write_strategy_artifact_audit(runtime, ctx, strategy_id, "strategy_doc_md", doc_path)
    return {"doc_path": str(doc_path), "sections": list(sections)}


# ---------------------------------------------------------------------------
# S4 rule strategy: mining, evaluation, and the rule-set selection gate helper.
# ---------------------------------------------------------------------------
# A single-rule lift this high (or a hit bad rate this high) usually means a
# leakage/near-target feature slipped into the candidate set, not a genuine
# reject rule -- surfaced so a reviewer can drop it before adoption.
_SUSPECT_LEAKAGE_LIFT = 10.0
_SUSPECT_LEAKAGE_BAD_RATE = 0.9
# Two rules co-hitting more than this share (Jaccard) are largely redundant.
_HIGH_OVERLAP_THRESHOLD = 0.8
# An included rule whose population share is below this fixed floor is flagged
# low_support (mirrors bands.py's _SPARSE_BAND_THRESHOLD). Distinct from the
# caller's min_support MINING filter: a caller may mine at a looser min_support
# (e.g. 0.01) yet still want a warning on any sub-2% rule before adoption.
_LOW_SUPPORT_FLOOR = 0.02


def tool_mine_rules(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    feature_cols = _optional_str_list(inputs.get("feature_cols"))
    columns = _unique([*(feature_cols or []), target_col]) if feature_cols else None
    frame = _dataset_frame(runtime, dataset_id, columns=columns)
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    resolved_features = feature_cols or _default_feature_cols(frame, target_col)
    min_support = _float_or(inputs.get("min_support"), 0.02)
    min_lift = _float_or(inputs.get("min_lift"), 1.5)
    candidates = mine_rules(
        frame,
        feature_cols=resolved_features,
        target_col=target_col,
        max_depth=int(inputs.get("max_depth", 3)),
        min_support=min_support,
        min_lift=min_lift,
        top_k=int(inputs.get("top_k", 20)),
        seed=int(inputs.get("seed", DEFAULT_MINE_SEED)),
    )
    candidate_rules = [rule.as_dict() for rule in candidates]
    red_flags = _mine_red_flags(candidate_rules, nan_labels_dropped)
    return {
        "candidate_rules": candidate_rules,
        "n_rows": int(len(frame)),
        "feature_cols": list(resolved_features),
        "red_flags": red_flags,
        "nan_labels_dropped": nan_labels_dropped,
    }


def tool_evaluate_rule_set(inputs: dict, ctx) -> dict:
    runtime = _runtime(ctx)
    dataset_id = str(inputs["dataset_id"])
    target_col = str(inputs["target_col"])
    rules_ordered = [dict(rule) for rule in (inputs.get("rules") or []) if isinstance(rule, dict)]
    frame = _dataset_frame(runtime, dataset_id)
    frame, nan_labels_dropped = resolve_labeled_frame(
        frame, target_col, drop_nan_labels=bool(inputs.get("drop_nan_labels")),
    )
    result = evaluate_rule_set(
        frame,
        rules_ordered,
        target_col=target_col,
        decision=str(inputs.get("decision") or "reject"),
    )
    red_flags = _evaluate_red_flags(result, rules_ordered, nan_labels_dropped)
    result["red_flags"] = red_flags
    result["nan_labels_dropped"] = nan_labels_dropped
    return result


def tool_select_rule_set(inputs: dict, ctx) -> dict:
    """Lightweight rule-set selection gate helper (S4).

    Assembles the user-selected ordered subset of the mined candidate rules into
    a gate payload and passes it through unchanged. ``selection`` is a literal
    ``None`` default in the template step's inputs so the generic apply_adjust
    gate-override channel (agent/gate_execution_adapter.py) can overwrite it with
    the parsed 「选 1,3,5」/「全选」/「去掉 2」 instruction -- exactly the band_edges
    precedent. A ``None`` selection means "keep all candidates" (no filter yet).
    """
    candidate_rules = [dict(rule) for rule in (inputs.get("candidate_rules") or []) if isinstance(rule, dict)]
    selection = inputs.get("selection")
    decision = str(inputs.get("decision") or "reject")
    selected = [_build_ready_rule(rule, decision) for rule in _apply_rule_selection(candidate_rules, selection)]
    return {
        "selected_rules": selected,
        "selected_count": len(selected),
        "candidate_count": len(candidate_rules),
    }


def _build_ready_rule(rule: dict, decision: str) -> dict:
    """Shape a mined candidate into a build_strategy-ready rule dict.

    build_strategy needs {condition, decision(, value)}; a mined CandidateRule
    carries only condition + display stats (lift/support/source/hit_bad_rate).
    Attach the reject decision and keep the display fields (build_strategy reads
    only condition/decision/value and ignores the rest, so they ride along for
    the renderer/waterfall without affecting the strategy)."""
    ready = dict(rule)
    ready["condition"] = str(rule.get("condition", ""))
    ready["decision"] = decision
    return ready


def _apply_rule_selection(candidate_rules: list[dict], selection) -> list[dict]:
    """Resolve a parsed selection into an ordered subset of candidate_rules.

    ``selection`` is None (keep all) or a list of 1-based indices in the display
    order the user chose (e.g. [1, 3, 5]); the returned order follows the
    selection order, not the candidate order, so the user can also reorder.
    Out-of-range/duplicate indices are dropped defensively -- the gate reply
    parser already validated them, this is belt-and-braces."""
    if selection is None:
        return [dict(rule) for rule in candidate_rules]
    ordered: list[dict] = []
    seen: set[int] = set()
    for raw in selection:
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index < 1 or index > len(candidate_rules) or index in seen:
            continue
        seen.add(index)
        ordered.append(dict(candidate_rules[index - 1]))
    return ordered


def _mine_red_flags(candidate_rules: list[dict], nan_labels_dropped: int) -> list[dict]:
    red_flags: list[dict] = []
    for rule in candidate_rules:
        lift = _finite(rule.get("lift"))
        hit_bad_rate = _finite(rule.get("hit_bad_rate"))
        if (lift is not None and lift > _SUSPECT_LEAKAGE_LIFT) or (
            hit_bad_rate is not None and hit_bad_rate > _SUSPECT_LEAKAGE_BAD_RATE
        ):
            red_flags.append(
                {
                    "code": "suspect_leakage",
                    "level": "red",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）lift="
                        f"{_fmt_num(lift)}、命中坏率={_fmt_pct(hit_bad_rate)}，疑似泄漏/近目标特征入选，请核查。"
                    ),
                }
            )
        support = _finite(rule.get("support"))
        if support is not None and support < _LOW_SUPPORT_FLOOR:
            red_flags.append(
                {
                    "code": "low_support",
                    "level": "amber",
                    "message": (
                        f"规则 {rule.get('rule_id')}（{rule.get('condition')}）支持度 "
                        f"{_fmt_pct(support)} 低于 {_fmt_pct(_LOW_SUPPORT_FLOOR)} 底线，样本量偏小。"
                    ),
                }
            )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags


def _evaluate_red_flags(result: dict, rules_ordered: list[dict], nan_labels_dropped: int) -> list[dict]:
    red_flags: list[dict] = []
    waterfall = result.get("waterfall") or []
    for row in waterfall:
        if int(row.get("incremental_hits") or 0) == 0:
            red_flags.append(
                {
                    "code": "rule_shadowed",
                    "level": "amber",
                    "message": (
                        f"规则 {row.get('rule_id')} 在瀑布中零增量命中（被前序规则完全覆盖），可考虑移除。"
                    ),
                }
            )
    overlap = result.get("overlap_matrix") or []
    for i in range(len(overlap)):
        for j in range(i + 1, len(overlap)):
            share = _finite(overlap[i][j])
            if share is not None and share > _HIGH_OVERLAP_THRESHOLD:
                red_flags.append(
                    {
                        "code": "high_overlap",
                        "level": "amber",
                        "message": (
                            f"规则 {_rule_id_at(i)} 与 {_rule_id_at(j)} 重叠 {_fmt_pct(share)} "
                            f"(>{_fmt_pct(_HIGH_OVERLAP_THRESHOLD)})，高度冗余。"
                        ),
                    }
                )
    if nan_labels_dropped:
        red_flags.append(
            {
                "code": "nan_labels_dropped",
                "level": "amber",
                "message": f"已按确认丢弃 {nan_labels_dropped} 行 NaN 标签样本。",
            }
        )
    return red_flags


def _rule_id_at(index: int) -> str:
    return f"rule_{index + 1}"


def _default_feature_cols(frame: pd.DataFrame, target_col: str) -> list[str]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return [column for column in numeric if column != target_col]


def _optional_str_list(value) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        cleaned = [str(item) for item in value if str(item).strip()]
        return cleaned or None
    return None


def _float_or(value, default: float) -> float:
    number = _optional_float(value)
    return default if number is None else number


def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _fmt_num(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.2f}"


def _fmt_pct(value) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number * 100:.1f}%"


def _write_strategy_artifact_audit(runtime, ctx, strategy_id: str, kind: str, path) -> None:
    from marvis.repositories.strategy import _write_audit_row

    from marvis.db_schema import connect

    with connect(runtime.settings.db_path) as conn:
        _write_audit_row(
            conn,
            kind="strategy.artifact",
            target_ref=strategy_id,
            outcome="succeeded",
            detail={"task_id": str(ctx.task_id), "kind": kind, "path": str(path)},
        )


def _band_stats_from_inputs(value) -> list[dict]:
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        bands = value.get("bands")
        if isinstance(bands, list):
            return [dict(band) for band in bands if isinstance(band, dict)]
        return []
    if isinstance(value, list):
        return [dict(band) for band in value if isinstance(band, dict)]
    return []


class _Runtime:
    def __init__(self, ctx):
        self.settings = build_settings(ctx.workspace)
        self.datasets_root = Path(ctx.datasets_root)
        self.repo = DatasetRepository(self.settings.db_path)
        self.backend = DataBackend(self.datasets_root)
        self.registry = DatasetRegistry(self.repo, self.backend, self.datasets_root)
        self.strategies = StrategyRepository(self.settings.db_path)


def _runtime(ctx) -> _Runtime:
    return _Runtime(ctx)


def _dataset_frame(runtime: _Runtime, dataset_id: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    dataset = runtime.registry.get(dataset_id)
    return runtime.backend.read_frame(runtime.registry.resolve_path(dataset.id), columns=columns)


def _strategy(runtime: _Runtime, strategy_id: str) -> Strategy:
    strategy = runtime.strategies.get_strategy(strategy_id)
    if strategy is None:
        raise StrategyError(f"strategy not found: {strategy_id}")
    return strategy


def _profit_params(payload: dict) -> ProfitParams:
    return ProfitParams(
        annual_rate=float(payload["annual_rate"]),
        funding_rate=float(payload["funding_rate"]),
        lgd=float(payload["lgd"]),
        operating_cost_per_loan=float(payload["operating_cost_per_loan"]),
        term_months=int(payload["term_months"]),
    )


def _optional_profit_params(payload) -> ProfitParams | None:
    return None if payload in (None, "") else _profit_params(dict(payload))


def _backtest_id(dataset_id: str, result: BacktestResult) -> str:
    payload = {"dataset_id": dataset_id, "result": _jsonable(result)}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return f"backtest-{digest[:12]}"


def _jsonable(value):
    if value is None:
        return None
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _unique(values: list[str | None]) -> list[str]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out
