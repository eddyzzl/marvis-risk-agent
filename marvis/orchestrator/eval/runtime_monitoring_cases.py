"""Separate public development journeys; these are not held-out acceptance."""
from __future__ import annotations

from .runtime_family_cases import normal_modeling_cases, normal_modeling_frames


def normal_monitoring_frames():
    import numpy as np
    import pandas as pd

    frames = normal_modeling_frames()
    rng = np.random.default_rng(20261003)
    frames["new_monitoring.parquet"] = ("unknown", pd.DataFrame({
        "signal": rng.normal(4, 1, 180),
        "affordability": rng.normal(0, 1, 180),
        "noise": rng.normal(0, 1, 180),
    }))
    return frames


def normal_monitoring_cases(materials, *, manual=False):
    cases, expected = normal_modeling_cases(materials)
    case = cases[0]
    previous = case["id"]
    case["revision"] = "2"
    case["id"] = "synthetic_normal_monitoring" + ("_manual_workflow" if manual else "_agent")
    case["family"] = "monitoring"
    case["materials"].append(materials["new_monitoring.parquet"])
    case["business_constraints_source"] += (
        " After real modeling and explicit experiment selection, the user uploads a separate "
        "180-row unlabeled new-period sample with a declared synthetic feature shift. "
        "The monitoring request uses the current selected experiment and current dataset/hash. "
        "Platform default technical thresholds are not business targets. Missing labels imply "
        "drift-only monitoring; maturity and business acceptance are unknown. "
        + ("Monitoring plan is created through the manual generic workflow API; the Agent task still uses its required typed start-plan confirmation, not the monitoring message intake." if manual else
           "Monitoring entry is a typed Agent message and its existing human plan-overview gate.")
    )
    case["actions"].extend([
        {"kind": "submit_model_monitoring_request" if not manual else "start_model_monitoring_workflow",
         "content": "使用当前已选实验监控这份新数据，仅检查漂移；平台技术阈值不代表业务达标。",
         "monitoring_request": {"material_path": "new_monitoring.parquet", "target_col": None}},
        {"kind": "confirm_model_monitoring_plan", "content": "已审阅模型、数据和限制，确认本次监控计划。"},
    ])
    case["budget"] = {"wall_seconds": 420, "max_llm_attempts": 40,
                      "max_http_requests": 280, "max_output_tokens_per_attempt": 2048}
    result = expected.pop(previous)
    result["assertions"].extend([
        {"kind": "tool_succeeded", "tool": "modeling.score_dataset"},
        {"kind": "tool_succeeded", "tool": "modeling.monitor_run"},
        {"kind": "dataset_rows", "tool": "modeling.score_dataset", "value": 180},
        {"kind": "output_equals", "tool": "modeling.monitor_run", "path": ["overall_level"], "value": "red"},
        {"kind": "monitoring_evidence", "tool": "modeling.monitor_run",
         "value": "manual_monitoring_workflow" if manual else "standard_model_monitoring_agent"},
        {"kind": "http_status", "stage": "human_monitoring_plan_start", "value": 202},
    ])
    return [case], {case["id"]: result}
