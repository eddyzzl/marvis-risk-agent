"""Measured validation evidence for external narrative generation.

Each adapter follows a validation producer's result shape. Report text,
individual score comparisons, fitted model parameters, category values and
errors are deliberately absent. The original local result is never changed.
"""
from __future__ import annotations

from marvis.llm_outbound import OutboundContextError, _DROP, _identifier, _mint_prompt
from marvis.llm_outbound_reports import _feature, _labels, _numeric, _period, _rows

_SPLITS = {"train", "test", "oot", "valid"}
_COUNTS = ("sample_count", "bad_count", "bad_rate")
_PERFORMANCE = (*_COUNTS, "ks", "auc", "psi_vs_train", "head_lift_5pct", "tail_lift_5pct")
_STRENGTH = {"unknown", "fail", "weak", "ok", "strong"}
_STRESS_STATUS = {"completed", "skipped", "error", "failed", "partial"}
_CHECK_STATUS = {"pass", "fail", "not_available"}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _dates(row, fields):
    return {key: value for key in fields if (value := _period(row.get(key))) is not None}


def _hyperparameters(value):
    row = _mapping(value)
    out = _numeric(row, (
        "C", "max_iter", "tol", "random_state", "seed", "n_estimators", "max_depth",
        "learning_rate", "num_leaves", "min_child_samples", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "scale_pos_weight",
        "num_boost_round", "early_stopping_rounds", "n_jobs", "num_threads",
        "iterations", "depth", "l2_leaf_reg", "bagging_fraction", "feature_fraction",
    ))
    out.update(_labels(row, {
        "penalty": {"l1", "l2", "elasticnet", "none"},
        "solver": {"lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"},
        "objective": {"binary", "binary:logistic", "Logloss", "CrossEntropy"},
        "boosting_type": {"gbdt", "dart", "rf", "goss"},
        "class_weight": {"balanced"},
    }))
    return out


def _basic_info(value):
    row = _mapping(value)
    out = {}
    period = row.get("sample_period")
    if isinstance(period, (list, tuple)) and len(period) == 2 and all(_period(item) for item in period):
        out["sample_period"] = list(period)
    if "split_summary" in row:
        out["split_summary"] = [
            {**_labels(item, {"split": _SPLITS}), **_numeric(item, _COUNTS),
             **_dates(item, ("period_start", "period_end"))}
            for item in _rows(row["split_summary"]) if item.get("split") in _SPLITS
        ]
    if "monthly_distribution" in row:
        out["monthly_distribution"] = [
            {**_dates(item, ("month",)), **_numeric(item, _COUNTS)}
            for item in _rows(row["monthly_distribution"]) if _period(item.get("month"))
        ]
    if "feature_importance" in row:
        out["feature_importance"] = [
            {**_feature(item), **_numeric(item, ("rank", "importance"))}
            for item in _rows(row["feature_importance"]) if _feature(item)
        ]
    if "hyperparameters" in row:
        out["hyperparameters"] = _hyperparameters(row["hyperparameters"])
    return out


def _ranking_group(value, source):
    if value is None:
        return None
    row = _mapping(value)
    out = {"source": source, **_numeric(row, (
        "bin_count", "head_group_lift", "tail_group_lift", "head_group_bad_rate",
        "tail_group_bad_rate", "bad_rate_inversions", "group_lift_spread",
    )), **_labels(row, {
        "head_group_strength": _STRENGTH, "tail_group_strength": _STRENGTH,
        "bad_rate_monotonicity": {"unknown", "monotonic", "mostly_monotonic", "broken"},
    })}
    notes = row.get("notes")
    if isinstance(notes, list):
        allowed = {"head_lift_near_one", "tail_lift_near_one", "head_not_better_than_average",
                   "tail_not_worse_than_average", "flat_discrimination"}
        out["notes"] = [note for note in notes if isinstance(note, str) and note in allowed]
    return out


def _ranking(value):
    from marvis.validation.lift_ranking import GUIDANCE

    row = _mapping(value)
    out = {**_labels(row, {"order": {"good_to_bad"}}), "guidance": GUIDANCE, "splits": []}
    for item in _rows(row.get("splits")):
        if item.get("split") not in _SPLITS:
            continue
        clean = {**_labels(item, {"split": _SPLITS, "head_5pct_strength": _STRENGTH,
                                  "tail_5pct_strength": _STRENGTH}),
                 **_numeric(item, ("head_lift_5pct", "tail_lift_5pct"))}
        for source in ("train_aligned", "independent_quantile"):
            if source in item:
                clean[source] = _ranking_group(item[source], source)
        out["splits"].append(clean)
    return out


def _effectiveness(value):
    row = _mapping(value)
    out = {}
    if "overall" in row:
        out["overall"] = [{**_labels(item, {"split": _SPLITS}), **_numeric(item, _PERFORMANCE)}
                          for item in _rows(row["overall"]) if item.get("split") in _SPLITS]
    for key, fields in (
        ("monthly_ks", _PERFORMANCE),
        ("monthly_psi", ("psi_vs_train", "psi_first_month", "psi_last_month", "psi_mom")),
    ):
        if key not in row:
            continue
        out[key] = []
        for item in _rows(row[key]):
            if not _period(item.get("month")):
                continue
            clean = {**_dates(item, ("month", "psi_mom_reference_month")), **_numeric(item, fields)}
            if type(item.get("psi_mom_has_calendar_gap")) is bool:
                clean["psi_mom_has_calendar_gap"] = item["psi_mom_has_calendar_gap"]
            out[key].append(clean)
    if "psi_stability_table" in row:
        # Bin labels may contain data values. Keep ordinal identity and counts.
        out["psi_stability_table"] = [
            {"bin_index": index, **_numeric(item, (
                "expected_count", "expected_pct", "actual_count", "actual_pct", "psi",
            ))} for index, item in enumerate(_rows(row["psi_stability_table"]), start=1)
        ]
    if "bin_tables" in row or "independent_quantile_bin_tables" in row:
        from marvis.validation.lift_ranking import assess_lift_ranking

        # Native result files contain bin tables, whereas the old prompt
        # compactor added this assessment. Derive it locally before withholding
        # bin boundaries; otherwise the new outbound path loses ranking evidence.
        native = {"overall": out.get("overall", [])}
        for key in ("bin_tables", "independent_quantile_bin_tables"):
            native[key] = {split: [_numeric(item, ("lift", "bad_rate")) for item in _rows(rows)]
                           for split, rows in _mapping(row.get(key)).items() if split in _SPLITS}
        out["lift_ranking_assessment"] = _ranking(assess_lift_ranking(native))
    elif "lift_ranking_assessment" in row:
        out["lift_ranking_assessment"] = _ranking(row["lift_ranking_assessment"])
    return out


def _stress(value):
    row = _mapping(value)
    out = _labels(row, {"status": _STRESS_STATUS})
    if "baseline" in row:
        out["baseline"] = _numeric(_mapping(row["baseline"]), ("ks", "sample_count"))
    if isinstance(row.get("unclassified_features"), list):
        out["unclassified_feature_count"] = len(row["unclassified_features"])
    if "per_category" in row:
        out["per_category"] = []
        for index, item in enumerate(_rows(row["per_category"]), start=1):
            clean = {"category_index": index, **_numeric(item, ("ks_after", "ks_delta", "psi_vs_baseline")),
                     **_labels(item, {"status": _STRESS_STATUS})}
            dropped = item.get("dropped_features")
            if isinstance(dropped, list):
                clean["dropped_feature_count"] = len(dropped)
            out["per_category"].append(clean)
    return out


def validation_metrics(value):
    """Project native ValidationResults plus deterministic assessment adapters."""
    row = _mapping(value)
    out = _labels(row, {
        "algorithm": {"logistic", "logistic_regression", "lr", "lightgbm", "lgb", "xgboost", "xgb",
                      "catboost", "scorecard", "random_forest", "mlp", "LogisticRegression",
                      "LGBMClassifier", "XGBClassifier", "CatBoostClassifier"},
        "target_type": {"binary"},
    })
    for key, adapter in (("basic_info", _basic_info), ("effectiveness", _effectiveness), ("stress_test", _stress)):
        if key in row:
            out[key] = adapter(row[key])
    if "overfitting_check" in row:
        check = _mapping(row["overfitting_check"])
        out["overfitting_check"] = {**_numeric(check, (
            "train_ks", "test_ks", "oot_ks", "train_test_relative_diff", "train_test_threshold",
            "train_oot_abs_diff", "train_oot_threshold",
        )), **_labels(check, {"metric": {"ks"}, "status": _CHECK_STATUS,
                             "train_test_status": _CHECK_STATUS, "train_oot_status": _CHECK_STATUS})}
    return out


def validation_metrics_prompt(value):
    """Build a metrics-only narrative request without local report context."""
    return _mint_prompt({"evidence": {"validation_results": validation_metrics(value)}, "instruction": (
        "Explain only these measured validation aggregates. Missing evidence is unknown. "
        "The platform owns metrics and pass/fail results. Preserve sample splits, temporal gaps, "
        "and the distinction between train-aligned and independently binned assessments. "
        "Category indexes identify groups only within this result; do not invent category names. "
        "Individual scores, model bodies, historical comparisons and local report text are withheld."
    )})


def validation_metrics_summary_prompt(evidence, *, missing_sections=None, history=None):
    """V1 stage narrative and its repair share the same measured context."""
    from marvis.agent.service import (
        METRICS_SUMMARY_REQUIRED_SECTIONS, _llm_task_reference_instruction, _stage_instructions,
    )
    from marvis.llm_outbound_memory import add_model_history

    clean = {
        "stage": "metrics", "validation_workflow_version": 1,
        "evidence": {"validation_results": validation_metrics(_mapping(evidence).get("validation_results"))},
        "history_comparison_available": False,
        "instructions": _stage_instructions("metrics", validation_workflow_version=1).replace(
            _llm_task_reference_instruction(),
            "称呼当前模型，不猜测未提供的模型名称、客户或机构名称。",
        ) + (
            "本次指标只使用平台计算的聚合证据，未提供已有报告文字或逐行评分。"
            "历史比较仅依据明确提供的 historical_models；压力测试类别用本次序号引用，不猜测数据源名称。"
            "缺失指标和风险分层明确未知；保留时间断档及按 train 分箱与独立分箱的区别。"
        ),
    }
    if missing_sections is not None:
        if (not isinstance(missing_sections, list) or not missing_sections
                or any(section not in METRICS_SUMMARY_REQUIRED_SECTIONS for section in missing_sections)):
            raise OutboundContextError("未知指标解读修复章节，未发送请求。")
        clean["repair_request"] = {
            "reason": "stage summary is missing required sections",
            "missing_sections": list(missing_sections),
            "required_sections": list(METRICS_SUMMARY_REQUIRED_SECTIONS),
            "instruction": (
                "请完整重写当前阶段分析，逐一使用上述五个标题；每节基于同一份 evidence 给出完整判断。"
                "不得编造指标，不得只补标题，不得引用上一轮未通过格式检查的文字。只输出完整分析正文。"
            ),
        }
    return _mint_prompt(add_model_history(clean, history))


def validation_start_prompt(*, workflow_version):
    return _mint_prompt({
        "stage": "agent_start", "validation_workflow_version": 2 if workflow_version == 2 else 1,
        "next_tool": "scan_materials",
        "instructions": (
            "用最多两句中文说明将启动本次模型验证，下一步调用材料识别工具检查材料完备性。"
            "称呼当前模型，不猜测模型、客户或机构名称。材料尚未扫描，不能声称已完成检查、"
            "材料齐全、指标通过或已生成报告，不提前给出验证结论。"
        ),
    })


def pmml_scoring_prompt(evidence):
    """Only persisted scoring aggregates; no individual scores or file labels."""
    row = _mapping(evidence)
    nested = _mapping(row.get("validation_results"))
    clean = {}
    for candidate in (row.get("pmml_scoring"), nested.get("pmml_scoring")):
        scoring = _mapping(candidate)
        projected = {**_numeric(scoring, (
            "input_row_count", "success_count", "failure_count", "null_count", "non_finite_count",
            "elapsed_seconds", "rows_per_second", "chunk_size", "required_input_count",
        )), **_labels(scoring, {"status": {"pass", "failed"},
                               "engine": {"pypmml-pmml4s-batch", "jpmml-batch"}})}
        for source, count in (("missing_inputs", "missing_input_count"), ("bounded_errors", "error_count")):
            if isinstance(scoring.get(source), list):
                projected[count] = len(scoring[source])
        if projected:
            clean = projected
            break
    return _mint_prompt({
        "stage": "pmml_scoring", "evidence": {"pmml_scoring": clean},
        "history_comparison_available": False,
        "instructions": (
            "只解读当前模型的 PMML 全量打分证据，按结论、评分覆盖、异常情况、性能、建议组织中文摘要。"
            "平台提供的计数与状态为准，缺失字段明确未知；异常文本和输入字段名称未发送，"
            "仅能引用异常数和缺失字段数，不猜测具体错误、样本值或文件名。"
            "V2 未执行 Notebook 模型，也不进行代码模型与 PMML 分数一致性验证，"
            "不得声称已经执行或建议后续补做这些检查。"
            "打分成功只证明当前评分覆盖，不证明 KS/AUC/PSI 达标、模型可用或可投产。"
            "此阶段不作历史比较或最终报告判断。"
        ),
    })


def notebook_execution_context(raw_notebook):
    steps = _rows(raw_notebook.get("steps")) if isinstance(raw_notebook, dict) else _rows(raw_notebook)
    cells = _rows(raw_notebook.get("cells")) if isinstance(raw_notebook, dict) else []
    positions = {step["id"]: index for index, step in enumerate(steps, 1) if isinstance(step.get("id"), str)}
    statuses = {"pending", "running", "succeeded", "failed", "skipped", "cancelled"}
    notebook = {"steps": [], "failed_cells": []}
    for index, step in enumerate(steps, 1):
        item = {"step_index": index, **_numeric(step, ("step_order", "cell_count", "elapsed_seconds")),
                **_labels(step, {"status": statuses})}
        if type(step.get("system")) is bool:
            item["system"] = step["system"]
        notebook["steps"].append(item)
    exceptions = {"FileNotFoundError", "PermissionError", "ModuleNotFoundError", "ImportError",
                  "KeyError", "ValueError", "TypeError", "NameError", "AttributeError",
                  "MemoryError", "TimeoutError", "RuntimeError", "SyntaxError", "ZeroDivisionError"}
    for cell in cells:
        if cell.get("status") != "failed":
            continue
        item = {"status": "failed", **_numeric(cell, ("cell_index",)),
                **_labels(cell, {"exception_name": exceptions})}
        if isinstance(cell.get("step_id"), str) and cell["step_id"] in positions:
            item["step_index"] = positions[cell["step_id"]]
        notebook["failed_cells"].append(item)
    return notebook


def reproducibility_context(evidence):
    """V1 comparison aggregates and execution positions, never code or rows."""
    row = _mapping(evidence)
    comparison = {}
    for candidate in (row.get("reproducibility"), _mapping(row.get("validation_results")).get("reproducibility")):
        candidate = _mapping(candidate)
        summary = _mapping(candidate.get("summary"))
        projected = _numeric(candidate, ("sample_size", "seed"))
        verdict = {**_numeric(summary, ("match_count", "mismatch_count", "max_abs_diff")),
                   **_labels(summary, {"status": {"pass", "review", "fail"}})}
        if verdict:
            projected["summary"] = verdict
        rows = candidate.get("rows")
        if isinstance(rows, list) and all(isinstance(item, dict) and "score_submitted_pmml" in item for item in rows):
            # Count missing submissions locally; never expose scores or row ids.
            projected["missing_submitted_score_count"] = sum(item["score_submitted_pmml"] is None for item in rows)
        if projected:
            comparison = projected
            break
    raw_contract = _mapping(row.get("contract"))
    contract = {**validation_metrics({"algorithm": raw_contract.get("algorithm")}),
                **_numeric(raw_contract, ("score_decimal_places",))}
    for key in ("target_col", "score_col", "split_col", "time_col"):
        field = _identifier(raw_contract.get(key))
        if field is not _DROP:
            contract[key] = field
    if isinstance(raw_contract.get("feature_columns"), list):
        contract["feature_count"] = len(raw_contract["feature_columns"])
    notebook = notebook_execution_context(row.get("notebook_steps"))
    return {"reproducibility": comparison, "contract": contract, "notebook_steps": notebook}


def reproducibility_prompt(evidence):
    return _mint_prompt({
        "stage": "reproducibility", "validation_workflow_version": 1,
        "evidence": reproducibility_context(evidence),
        "history_comparison_available": False,
        "instructions": (
            "本阶段是 V1 Notebook 内存模型分与提交 PMML 分的一致性比较。"
            "按结论、证据、风险含义、建议组织中文摘要，只使用本次摘要与执行状态。"
            "summary.status 是平台判定，匹配数、偏差与样本量不得重算或推断缺失值。"
            "max_abs_diff 只涉及可比较分数，缺失提交分数可使平台判为失败，不能因最大偏差为零而改写为通过。"
            "步骤序号从 1 起；cell_index 是平台原始索引，从 0 起。失败步骤不表示已有一致性结果。"
            "源码、错误原文、文件名和逐笔分数未提供；只引用位置、状态及已知异常类型，"
            "不能猜测缺失文件或具体代码，建议查看本地该 cell 的诊断。"
            "不得扩展为 KS/AUC/PSI、模型效果、压力测试、整体通过或投产判断；不作历史比较。"
        ),
    })


def validation_v1_report_prompt(evidence, *, history=None):
    """Fresh V1 candidate prose from measured results and execution status."""
    from marvis.agent.service import _llm_task_reference_instruction, _stage_instructions
    from marvis.llm_outbound_scan import scan_context
    from marvis.llm_outbound_memory import add_model_history

    row = _mapping(evidence)
    clean = {
        "stage": "word_conclusion_draft", "validation_workflow_version": 1,
        "evidence": {
            **scan_context(row, workflow_version=1), **reproducibility_context(row),
            "validation_results": validation_metrics(row.get("validation_results")),
        },
        "history_comparison_available": False, "report_output_status_available": False,
        "instructions": _stage_instructions("word_conclusion_draft", validation_workflow_version=1).replace(
            _llm_task_reference_instruction(), "称呼当前模型，不猜测模型、客户或机构名称。",
        ) + (
            "本次为首次候选文字生成，只返回指定的四项候选文字，不改写其他业务定义字段。"
            "未提供阶段总结或已有报告文字；历史比较仅依据明确提供的 historical_models。"
            "未提供报告产物状态，不能声称 Word/Excel 已生成、可预览或可下载。"
            "压力类别用本次序号引用，不猜测数据源名称；未知风险分层明确未知。"
            "分箱排序性须分别解释按 train 分箱和独立分箱的幅度、单调性、头尾区分。"
            "scan_interpretation 中缺少或未满足的检查不等同于文件一定缺失。"
            "一致性 summary.status 是平台结论，缺失提交分数仍可导致失败；不能因 max_abs_diff 为零写成通过。"
            "源码、样本行、逐笔分数、文件名和模型参数正文未提供，不猜测其内容。"
        ),
    }
    return _mint_prompt(add_model_history(clean, history))


def validation_metric_section_prompt(value, *, topic, repair_reason=None, history=None):
    return _validation_section_prompt(value, topic=topic, kind="metrics", repair_reason=repair_reason, history=history)


def validation_report_section_prompt(value, *, topic, repair_reason=None, history=None):
    return _validation_section_prompt(value, topic=topic, kind="report", repair_reason=repair_reason, history=history)


def _validation_section_prompt(value, *, topic, kind, repair_reason, history=None):
    from marvis.agent.validation_narrative import _REPORT_SECTIONS, _TOPICS
    from marvis.llm_outbound_memory import add_model_history

    fields = dict(_REPORT_SECTIONS)
    if topic not in fields or (kind == "metrics" and topic == "training"):
        raise OutboundContextError("未知指标解读主题，未发送请求。")
    clean = {
        "stage": "metrics" if kind == "metrics" else "word_conclusion_draft",
        "narrative_topic": topic,
        "requested_fields": ["summary"] if kind == "metrics" else [fields[topic][0]],
        "evidence": {"validation_results": validation_metrics(value)},
        "history_comparison_available": False,
        "instructions": (
            "仅返回 JSON 对象，requested_fields 中的首个字段必填，使用完整中文段落，约 150 至 350 字。"
            "本次只负责当前主题，其他主题由独立调用完整合并；不要重复整份结论。"
            "未要求修改的可选业务叙事字段必须省略，由平台保留本地内容。"
            "指标与通过判断以本次平台证据为准，缺失值明确未知，不推算未测指标。"
            "保留时间断档和按 train 分箱/独立分箱的区别。"
            "未提供本地报告文字；只有 overall 主题可依据明确提供的 historical_models 作克制历史对照。"
            "压力测试类别仅用本次序号引用，不猜测被隐藏的数据源名称；缺失分层明确未知。"
            + _TOPICS[topic]
        ),
    }
    if repair_reason is not None:
        if repair_reason not in {"json_text_contract", "forbidden_process_narration"}:
            raise OutboundContextError("未知响应修复原因，未发送请求。")
        clean["response_repair"] = {
            "reason": repair_reason,
            "instruction": (
                "这是唯一一次修复。返回符合相同 schema 的完整 JSON，不输出推导过程；保留风险与证据限制。"
                "最终结论只解释模型效果与风险，不写材料检查、验证输入契约、PMML打分过程、"
                "报告生成或定稿状态、投产前审阅安排，不声称可直接部署或投产。"
            ),
        }
    return _mint_prompt(add_model_history(clean, history) if topic == "overall" else clean)
