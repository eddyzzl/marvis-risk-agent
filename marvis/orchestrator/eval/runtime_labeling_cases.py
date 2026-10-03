"""Independent public development fixture for the normal label-construction path."""

import pandas as pd

from .runtime_contracts import digest


def normal_labeling_frames():
    rows = []
    for entity, dpds in (
        ("0001", [0, 0, 90, 0]),
        ("0002", [120, 0, 0, 0]),
        ("0004", [0, 0, 0, 0]),
    ):
        for mob, dpd in enumerate(dpds):
            rows.append([entity, "2026-01", mob, f"2026-{mob + 1:02d}-28", dpd])
    rows.extend(
        [
            ["0003", "2026-03", 0, "2026-03-28", 0],
            ["0003", "2026-03", 1, "2026-04-28", 0],
            ["0003", "2026-03", 3, "2026-06-28", 120],
            ["0004", "2026-01", 4, "2026-05-28", 120],
        ]
    )
    return {
        "normal_labeling.parquet": (
            "sample",
            pd.DataFrame(
                rows, columns=["loan_id", "cohort", "mob", "snapshot_date", "dpd"]
            ),
        )
    }


def normal_labeling_cases(materials):
    case_id = "synthetic_normal_labeling"
    case = {
        "id": case_id,
        "revision": "1",
        "family": "label_construction",
        "case_set": "development",
        "scenario": "normal",
        "task": {"task_type": "data_join", "target_col": "bad_90_mob3"},
        "materials": [materials["normal_labeling.parquet"]],
        "business_constraints_source": (
            "Public synthetic four-loan repayment panel. Explicit business rule: DPD>=90 ever in MOB (0,3], "
            "as-of 2026-04-30; retain unobserved immature non-bad labels as null. "
            "Observation-window DPD and records after the cutoff must not define the label. "
            "This development example does not establish external outcome truth or hidden acceptance."
        ),
        "actions": [
            {
                "kind": "submit_labeling_request",
                "content": "提交完整标签口径，请先展示提案和成熟度，未成熟且未定坏的贷款保留空标签。",
                "labeling_request": {
                    "id_col": "loan_id",
                    "mob_col": "mob",
                    "cohort_col": "cohort",
                    "date_col": "snapshot_date",
                    "as_of_date": "2026-04-30",
                    "target_col": "bad_90_mob3",
                    "observation_window": 0,
                    "performance_window": 3,
                    "at_mob": 3,
                    "rule_kind": "dpd",
                    "dpd_col": "dpd",
                    "threshold_dpd": 90,
                },
            },
            {"kind": "message", "content": "确认"},
            {"kind": "message", "content": "开始"},
            {
                "kind": "approve_step",
                "tool": "labeling.define_label",
                "content": "确认按照已审核的截止日、DPD90 和 MOB3 口径写入标签；未成熟空标签保留，不补为0。",
            },
            {
                "kind": "download_labeling_results",
                "content": "下载本次标签明细和质量证据。",
            },
        ],
        "budget": {
            "wall_seconds": 240,
            "max_llm_attempts": 30,
            "max_http_requests": 180,
            "max_output_tokens_per_attempt": 2048,
        },
    }
    expected = {
        "result": "done",
        "assertions": [
            {"kind": "tool_succeeded", "tool": "labeling.check_cohort_maturity"},
            {"kind": "tool_succeeded", "tool": "labeling.define_label"},
            *[
                {
                    "kind": "output_equals",
                    "tool": "labeling.define_label",
                    "path": [key],
                    "value": value,
                }
                for key, value in (
                    ("n_loans", 4),
                    ("n_bad", 1),
                    ("n_good", 2),
                    ("n_unmatured", 1),
                )
            ],
            {
                "kind": "output_equals",
                "tool": "labeling.check_cohort_maturity",
                "path": ["all_matured"],
                "value": False,
            },
            {
                "kind": "output_equals",
                "tool": "labeling.define_label",
                "path": ["workspace", "active_dataset_changed"],
                "value": False,
            },
            {
                "kind": "output_close",
                "tool": "labeling.define_label",
                "path": ["bad_rate"],
                "value": 1 / 3,
                "tolerance": 1e-12,
            },
            {
                "kind": "labeling_evidence",
                "tool": "labeling.define_label",
                "value": digest(
                    [
                        ["0001", "2026-01", 1.0],
                        ["0002", "2026-01", 0.0],
                        ["0003", "2026-03", None],
                        ["0004", "2026-01", 0.0],
                    ]
                ),
            },
        ],
    }
    return [case], {case_id: expected}
