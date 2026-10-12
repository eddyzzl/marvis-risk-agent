"""Outbound DTO for workflow insight producers; no rendered table prose."""
from marvis.llm_outbound import OutboundContextError, _mint_prompt
from marvis.llm_outbound_reports import _feature, _labels, _numeric, _rows
from marvis.llm_outbound_validation import _hyperparameters


def workflow_insight_prompt(context):
    milestone = context.get("milestone")
    if milestone not in {"join_completed", "join_diagnosed", "feature_reported", "feature_analyzed",
                         "model_completed", "model_review"}:
        raise OutboundContextError("工作流分析缺少已知阶段，未发送请求。")
    source = context.get("measured_evidence")
    if not isinstance(source, dict) or not source:
        raise OutboundContextError("工作流分析缺少结构化度量，未将报告文字作为证据发送。")
    clean = {}
    if milestone.startswith("join_"):
        clean.update(_numeric(source, ("anchor_rows", "joined_rows")))
        clean["join_tables"] = [{**_numeric(row, ("table_index", "match_rate", "missing_rate")),
                                **{key: row[key] for key in ("key_unique", "row_inflation")
                                   if type(row.get(key)) is bool}}
                               for row in _rows(source.get("join_tables"))]
    elif milestone.startswith("feature_"):
        clean.update(_numeric(source, ("feature_count",)))
        clean["features"] = [{**_feature(row), **_labels(row, {"state": {
            "recommended", "candidate", "not_recommended", "caution", "unevaluated",
        }})} for row in _rows(source.get("features")) if _feature(row)]
        clean["quality"] = [{**_feature(row), **_numeric(row, ("missing_rate", "single_value_rate", "zero_rate"))}
                            for row in _rows(source.get("quality")) if _feature(row)]
    else:
        clean.update(_labels(source, {"recipe": {"lr", "lgb", "xgb", "catboost", "scorecard", "mlp",
            "lgb_regressor", "xgb_regressor", "lr_regressor", "mlp_regressor", "lgb_multiclass",
            "xgb_multiclass", "lr_multiclass", "mlp_multiclass"}}))
        metrics = source.get("metrics")
        clean["metrics"] = _numeric(metrics if isinstance(metrics, dict) else {}, (
            "train_ks", "test_ks", "oot_ks", "test_auc", "oot_auc", "psi_oot_vs_train",
        ))
        clean["parameters"] = _hyperparameters(source.get("parameters"))
        flags = source.get("risk_flags")
        clean["risk_flags"] = [item for item in flags if isinstance(item, str) and item in {
            "train_test_ks_gap", "test_oot_ks_gap", "oot_ks_unavailable",
        }] if isinstance(flags, list) else []
    return _mint_prompt({"milestone": milestone, "measured_evidence": clean, "instruction": (
        "只解释已给出的平台度量。缺失值代表未知，不能推定为通过。"
        "拼接表序号仅用于本次结果，不要推测文件名或机构名。"
        "特征建议状态由平台提供；区分推荐、候选、不推荐、慎用和未评估，不能把未评估写成推荐。"
        "解释低命中率、缺失率、重复键、行数膨胀和样本外表现的风险。"
        "报告原文、原始记录与历史记忆文字未提供，不能声称已据此比较历史。"
        "返回 JSON，含 summary、findings、risks、recommendations。"
    )})
