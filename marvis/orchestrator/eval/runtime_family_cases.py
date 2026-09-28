"""Additional public development journeys using normal product inputs.

These cases exercise declared business choices, not independently held answers.
They are opt-in so the original two-case corpus and archived identities survive.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd


def workflow_frames() -> dict:
    return {
        "modeling.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "signal": list(range(120)),
                    "label_sqandzy": [0, 1] * 60,
                    "label_sqandzy_new": [1, 0] * 60,
                    "split_tag": ["train"] * 72 + ["test"] * 24 + ["oot"] * 24,
                }
            ),
        ),
        "portfolio.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "loan_id": ["a", "a", "b", "b"],
                    "snapshot_month": ["2025-01", "2025-02", "2025-01", "2025-02"],
                    "bucket": ["current", "M1", "M1", "charged_off"],
                    "balance": [100.0, 90.0, 200.0, 180.0],
                    "segment": ["new", "new", "existing", "existing"],
                }
            ),
        ),
        "vintage.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "cohort": ["2025-01", "2025-01", "2025-02", "2025-02"],
                    "mob": [0, 1, 0, 1],
                    "bad": [0, 1, 0, 0],
                }
            ),
        ),
    }


def workflow_cases(materials: dict) -> tuple[list[dict], dict]:
    cases = []
    expected = {}

    def add(
        case_id,
        family,
        scenario,
        *,
        task,
        actions,
        result,
        assertions,
        source,
        material_name=None,
    ):
        cases.append(
            {
                "id": case_id,
                "revision": "1",
                "family": family,
                "case_set": "development",
                "scenario": scenario,
                "task": {"task_type": family, **task},
                "materials": [materials[material_name or f"{family}.parquet"]],
                "actions": actions,
                "business_constraints_source": source,
                "budget": {
                    "wall_seconds": 180,
                    "max_llm_attempts": 30,
                    "max_http_requests": 150,
                    "max_output_tokens_per_attempt": 2048,
                },
            }
        )
        expected[case_id] = {"result": result, "assertions": assertions}

    add(
        "synthetic_portfolio",
        "portfolio",
        "normal",
        task={},
        source="Public synthetic two-loan panel; ordered states current/M1/charged_off; business-declared LGD 0.45 and horizon 18 months, no model trend requested",
        actions=[
            {
                "kind": "message",
                "content": "使用已确认的组合口径。",
                "portfolio_request": {
                    "id_col": "loan_id",
                    "snapshot_col": "snapshot_month",
                    "bucket_col": "bucket",
                    "balance_col": "balance",
                    "segment_col": "segment",
                    "loss_state": "charged_off",
                    "lgd": 0.45,
                    "horizon_months": 18,
                },
            },
            {"kind": "message", "content": "current,M1,charged_off"},
            {"kind": "message", "content": "开始"},
            {
                "kind": "approve_step",
                "tool": "analysis.portfolio_gate_summary",
                "content": "按上述显式假设保留合成数据结果并生成组合报告。",
            },
        ],
        result="done",
        assertions=[
            {"kind": "tool_succeeded", "tool": f"analysis.{tool}"}
            for tool in (
                "flow_rate",
                "bucket_migration",
                "segment_profile",
                "expected_loss_estimate",
                "portfolio_report",
            )
        ]
        + [
            {
                "kind": "output_length",
                "tool": "analysis.segment_profile",
                "path": ["segments"],
                "value": 2,
            },
            {
                "kind": "artifact_exists",
                "tool": "analysis.portfolio_report",
                "path": ["report_path"],
            },
        ],
    )
    add(
        "synthetic_portfolio_missing_economics",
        "portfolio",
        "clarification",
        task={},
        actions=[],
        result="clarification",
        source="Same synthetic panel; LGD, horizon and absorbing loss-state semantics intentionally withheld; Agent must request them before creating a plan",
        assertions=[
            {
                "kind": "latest_assistant_metadata",
                "path": ["kind"],
                "value": "portfolio_setup_required",
            }
        ],
    )
    add(
        "synthetic_vintage",
        "vintage",
        "normal",
        task={"target_col": "bad", "time_col": "cohort"},
        source="Public synthetic two-cohort panel, all rows in scope; bad explicitly means incremental new bads at each MOB, never a cumulative snapshot",
        actions=[
            {
                "kind": "message",
                "content": "做标准 Vintage，bad 是 incremental，不是 snapshot。",
            },
            {
                "kind": "message",
                "content": "材料已上传，覆盖全部 cohort 和 MOB；bad 是 incremental 当期新增，不是 snapshot。",
            },
        ],
        result="done",
        assertions=[
            {"kind": "tool_succeeded", "tool": "strategy.vintage_curve"},
            {
                "kind": "output_length",
                "tool": "strategy.vintage_curve",
                "path": ["cohorts"],
                "value": 2,
            },
        ],
    )
    add(
        "synthetic_vintage_missing_goal",
        "vintage",
        "clarification",
        task={"target_col": "bad", "time_col": "cohort"},
        actions=[],
        result="clarification",
        source="Synthetic panel supplied without selecting standard Vintage, terminal risk or profitability; Agent must ask which analysis is intended",
        assertions=[
            {
                "kind": "latest_assistant_metadata",
                "path": ["risk_analysis_intake", "phase"],
                "value": "ask_goal",
            }
        ],
    )
    add(
        "synthetic_modeling_ambiguous_target",
        "modeling",
        "clarification",
        task={},
        actions=[],
        result="clarification",
        source="Synthetic 120-row panel has two distinct plausible targets label_sqandzy and label_sqandzy_new; no business choice supplied, so training must wait for explicit target selection",
        assertions=[
            {
                "kind": "latest_assistant_metadata",
                "path": ["join_c1", "target_col"],
                "value": None,
            }
        ],
    )
    add(
        "synthetic_strategy_missing_objective",
        "strategy",
        "clarification",
        task={"strategy_input": {"entry_mode": "strategy_development"}},
        material_name="analysis.parquet",
        actions=[],
        result="clarification",
        source="Synthetic data supplied for full strategy development without business objective or risk/approval bounds; Agent must not invent those inputs",
        assertions=[
            {
                "kind": "latest_assistant_metadata",
                "path": ["clarification", "code"],
                "value": "strategy_business_inputs_required",
            }
        ],
    )
    rejection = deepcopy(cases[0])
    rejection.update(id="synthetic_portfolio_reject_report", scenario="rejection")
    rejection["business_constraints_source"] += (
        "; operator explicitly refuses report publication at the result gate"
    )
    rejection["actions"][-1] = {
        "kind": "reject_step",
        "tool": "analysis.portfolio_gate_summary",
        "content": "拒绝本次结果发布，停止生成组合报告。",
    }
    cases.append(rejection)
    expected[rejection["id"]] = {
        "result": "cancelled",
        "assertions": [
            {"kind": "http_status", "stage": "human_rejection", "value": 202},
            {"kind": "tool_not_executed", "tool": "analysis.portfolio_report"},
        ],
    }
    return cases, expected
