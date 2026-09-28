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


def normal_modeling_frames() -> dict:
    """Declared synthetic statistical signal, never customer or outcome evidence."""
    import numpy as np

    rng = np.random.default_rng(20260928)
    n = 600
    signal, affordability, noise = rng.normal(size=(3, n))
    probability = 1 / (1 + np.exp(-(0.8 * signal - 0.6 * affordability)))
    return {
        "normal_modeling.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "signal": signal,
                    "affordability": affordability,
                    "noise": noise,
                    "y": (rng.random(n) < probability).astype(int),
                    "split": ["train"] * 360 + ["test"] * 120 + ["oot"] * 120,
                }
            ),
        )
    }


def normal_modeling_cases(materials: dict) -> tuple[list[dict], dict]:
    """A separate opt-in denominator; no changes to the archived nine cases."""
    case = {
        "id": "synthetic_normal_modeling",
        "revision": "1",
        "family": "modeling",
        "case_set": "development",
        "scenario": "normal",
        "task": {
            "task_type": "modeling",
            "target_col": "y",
            "split_col": "split",
            "feature_columns": ["signal", "affordability", "noise"],
        },
        "materials": [materials["normal_modeling.parquet"]],
        "initial_message": "请按已上传的合成样本建立二分类模型，目标列 y，使用现有 split 的 train/test/oot，特征为 signal、affordability、noise。只训练逻辑回归 lr，调参 1 轮。",
        "business_constraints_source": "Public synthetic 600-row fixed-seed logistic sample, 360/120/120 train/test/OOT. Human explicitly chooses the platform-displayed recommended experiment and authorizes local reports/delivery. No real customers, loan economics, MOB maturity, feature dictionary, production Champion or business acceptance threshold supplied; previous_selected_experiment is only the current synthetic task's pre-refit candidate, never a production Champion; missing evidence must remain visible. Execution completion is not production or business approval.",
        "actions": [
            {
                "kind": "approve_step",
                "tool": "modeling.screen_features",
                "content": "确认当前展示的 360/120/120 合成 train/test/oot 切分与仅 lr、1 轮规格，开始筛选特征。",
            },
            {
                "kind": "approve_step",
                "tool": "modeling.select_features",
                "content": "采用平台展示的训练集特征筛选设置并保留筛选记录，继续精选特征。",
            },
            {
                "kind": "approve_step",
                "tool": "modeling.configure_tuning",
                "content": "确认仅逻辑回归 lr、1 轮调参，不扩大搜索或改变训练口径。",
            },
            {
                "kind": "approve_step",
                "tool": "modeling.tune_hyperparameters",
                "content": "确认当前已展示的特征和单轮调参配置，执行真实训练。",
            },
            {
                "kind": "select_recommended_experiment",
                "tool": "modeling.select_experiment",
                "content": "我已审阅当前展示的候选和限制，明确采用平台展示的推荐实验，仅用于本次合成流程验证。",
            },
            {
                "kind": "approve_step",
                "tool": "modeling.generate_model_reports",
                "content": "生成本地模型开发报告，保留所有缺少业务列、字典和成熟度证据的说明，不宣称业务验收通过。",
            },
            {
                "kind": "approve_step",
                "tool": "modeling.post_training_action",
                "content": "批准本地模型交付产物与验证移交，保留业务证据限制；仅本任务合成实验内部对照，不当作生产 Champion，不进行生产发布。",
            },
        ],
        "budget": {
            "wall_seconds": 300,
            "max_llm_attempts": 40,
            "max_http_requests": 240,
            "max_output_tokens_per_attempt": 2048,
        },
    }
    assertions = [
        {"kind": "tool_succeeded", "tool": "modeling." + tool}
        for tool in (
            "train_models",
            "select_experiment",
            "generate_model_reports",
            "post_training_action",
        )
    ]
    assertions += [
        {"kind": "http_status", "stage": "human_recommended_selection", "value": 202},
        {
            "kind": "output_equals",
            "tool": "modeling.select_experiment",
            "path": ["policy_decision", "explicit_selection"],
            "value": True,
        },
        {
            "kind": "output_equals",
            "tool": "modeling.select_experiment",
            "path": ["recipe"],
            "value": "lr",
        },
        {
            "kind": "artifact_exists",
            "tool": "modeling.generate_model_reports",
            "path": ["report_path"],
        },
        {
            "kind": "output_equals",
            "tool": "modeling.generate_model_reports",
            "path": ["section_status", 0, "available"],
            "value": False,
        },
        {
            "kind": "output_equals",
            "tool": "modeling.generate_model_reports",
            "path": ["section_status", 1, "available"],
            "value": False,
        },
        {
            "kind": "artifact_exists",
            "tool": "modeling.post_training_action",
            "path": ["model_card_path"],
        },
        {
            "kind": "artifact_exists",
            "tool": "modeling.post_training_action",
            "path": ["approval_package_path"],
        },
        {
            "kind": "output_equals",
            "tool": "modeling.post_training_action",
            "path": ["challenger_comparison", "champion", "label"],
            "value": "previous_selected_experiment",
        },
    ]
    return [case], {case["id"]: {"result": "done", "assertions": assertions}}


def normal_strategy_frames() -> dict:
    """Public artificial labels and cohorts, with no financial outcome evidence."""
    n = 120
    return {
        "normal_strategy.parquet": (
            "sample",
            pd.DataFrame(
                {
                    "synthetic_id": [f"synthetic-strategy-{i:04d}" for i in range(n)],
                    "y": [int(i % 10 >= 7) for i in range(n)],
                    "credit_score": [1 - (i % 10 + 0.5) / 10 for i in range(n)],
                    "apply_date": ["2026-01-15"] * 60
                    + ["2026-02-15"] * 30
                    + ["2026-03-15"] * 30,
                    "apply_month": ["2026-01"] * 60
                    + ["2026-02"] * 30
                    + ["2026-03"] * 30,
                }
            ),
        ),
    }


def normal_strategy_cases(materials: dict) -> tuple[list[dict], dict]:
    case = {
        "id": "synthetic_normal_strategy",
        "revision": "1",
        "family": "strategy",
        "case_set": "development",
        "scenario": "normal",
        "task": {
            "task_type": "strategy",
            "target_col": "y",
            "score_col": "credit_score",
            "strategy_input": {
                "entry_mode": "strategy_development",
                "strategy_type": "approval",
                "objective": "max_approval",
                "max_bad_rate": 0.25,
                "min_approval_rate": 0.4,
            },
        },
        "materials": [materials["normal_strategy.parquet"]],
        "business_constraints_source": (
            "Public synthetic 120-row strategy scenario with artificial credit_score/y and three disjoint cohorts. "
            "Human explicitly binds the sole uploaded sample and field roles. The declared 30-day maturity is "
            "a synthetic scenario assumption, never verified real historical maturity or point-in-time availability. "
            "Objective max_approval, approved bad-rate ceiling 0.25, approval floor 0.4. "
            "No production baseline, financial inputs, realized cashflows or production deployment authority. "
            "Only local adoption and report delivery are requested; profit and business readiness remain unestablished."
        ),
        "actions": [
            {
                "kind": "bind_single_strategy_sample",
                "content": "人工选择本次唯一上传的合成样本，确认 y 为目标、credit_score 为信用分及下列字段角色。",
                "semantic_mapping": {
                    "target_col": "y",
                    "business_names": {},
                    "field_roles": {
                        "synthetic_id": "id",
                        "y": "target",
                        "credit_score": "score",
                        "apply_date": "date",
                        "apply_month": "month",
                    },
                },
            },
            {
                "kind": "message",
                "content": (
                    "固化 V2 策略样本设计；1 代表坏样本；不丢弃缺失标签；"
                    "审批总体与风险总体是同批 cohort 的嵌套关系；审批总体无纳排条件；风险总体无纳排条件；"
                    "按时间范围切分，时间列 apply_date；开发范围 2026-01-01 至 2026-01-31；"
                    "验证范围 2026-02-01 至 2026-02-28；OOT 范围 2026-03-01 至 2026-03-31；"
                    "表现窗 30 天；观察窗 2026-01-01 至 2026-04-30；成熟度已确认成熟；"
                    "成熟表现窗 30 天；成熟度截止日 2026-04-30；实体字段 synthetic_id；"
                    "时间字段 apply_date；分组字段暂无；月份字段 apply_month；权重字段暂无；"
                    "放款金额字段暂无；逾期金额字段暂无；历史分 credit_score，越低越风险。"
                    "上述成熟口径仅为本次公开合成场景声明，不构成真实历史认证或收益证据。"
                ),
            },
            {
                "kind": "message",
                "content": "做完整审批策略开发，目标 max_approval，获批坏账率不超过 0.25，通过率至少 0.4。无财务数据，不评价收益；无生产基线。",
            },
            {
                "kind": "approve_step",
                "tool": "strategy.adopt_strategy",
                "content": "依据当前展示的合成样本回测与限制，批准本地采纳此候选并生成策略文档。缺少真实财务、生产基线及真实成熟度证据，不能证明收益或生产可用性。",
            },
        ],
        "budget": {
            "wall_seconds": 300,
            "max_llm_attempts": 40,
            "max_http_requests": 240,
            "max_output_tokens_per_attempt": 2048,
        },
    }
    assertions = [
        {"kind": "tool_succeeded", "tool": f"strategy.{tool}"}
        for tool in (
            "materialize_sample_design_v2_native",
            "tradeoff_view",
            "design_cutoff_bands",
            "build_strategy",
            "backtest_strategy",
            "adopt_strategy",
            "render_strategy_doc",
        )
    ] + [
        {"kind": "http_status", "stage": "human_strategy_sample_binding", "value": 200},
        {"kind": "http_status", "stage": "human_approval", "value": 202},
        {
            "kind": "output_length",
            "tool": "strategy.backtest_strategy",
            "path": ["economics"],
            "value": 0,
        },
        {
            "kind": "output_equals",
            "tool": "strategy.adopt_strategy",
            "path": ["asset_status"],
            "value": "adopted_local",
        },
        {
            "kind": "output_equals",
            "tool": "strategy.render_challenger_report",
            "path": ["status"],
            "value": "no_baseline",
        },
        {
            "kind": "artifact_exists",
            "tool": "strategy.render_strategy_doc",
            "path": ["doc_path"],
        },
    ]
    return [case], {case["id"]: {"result": "done", "assertions": assertions}}
