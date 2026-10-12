"""Typed current controls and native diagnostics for gate decisions."""
import re

from marvis.llm_outbound import OutboundContextError, _DROP, _gate_parameters, _identifier, _mint_prompt, structural_summary
from marvis.llm_outbound_reports import _labels, _numeric, _rows

_METRIC_POLICIES = {
    "test_ks", "oot_ks", "weighted_test_ks", "cv_ks", "test_rmse", "test_auc", "test_logloss",
    "lower OOT RMSE, fallback lower test RMSE",
    "higher OOT macro-AUC, fallback higher test macro-AUC then lower logloss",
    "higher overfit-penalized test KS; OOT reported only, not used for selection",
}
_RECIPES = {"lr", "lgb", "xgb", "catboost", "scorecard", "mlp", "lgb_regressor", "xgb_regressor",
            "lr_regressor", "mlp_regressor", "lgb_multiclass", "xgb_multiclass", "lr_multiclass", "mlp_multiclass"}


def _risk_evidence(value):
    codes = {"small_sample", "missing_oot", "many_features", "train_test_ks_gap",
             "thin_champion_margin", "weighted_champion_mismatch", "failed_candidates"}
    out = []
    for row in _rows(value):
        if row.get("code") not in codes:
            continue
        item = {"code": row["code"], **_numeric(row, ("sample_count", "feature_count", "threshold", "ks_gap", "count")),
                **_labels(row, {"weighted_recipe": _RECIPES, "unweighted_recipe": _RECIPES})}
        recipes = row.get("recipes")
        if isinstance(recipes, list):
            item["recipes"] = [recipe for recipe in recipes if isinstance(recipe, str) and recipe in _RECIPES]
        out.append(item)
    return out


def _join_diagnostics(tables):
    """Project only displayed aggregate rate/flag cells, never row labels."""
    out = []
    for table in _rows(tables):
        columns = table.get("columns")
        rows = table.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        if len(rows) > 10000:
            raise OutboundContextError("拼接诊断过大，未截断风险证据或发送请求。")
        for row in rows:
            if not isinstance(row, (list, tuple)):
                continue
            item = {}
            for column, value in zip(columns, row):
                if not isinstance(column, str):
                    continue
                cell = str(value).strip()
                if "命中率" in column and re.fullmatch(r"(?:0|1|0?\.\d+|1\.0+|\d{1,2}(?:\.\d+)?%|100(?:\.0+)?%)", cell):
                    rate = float(cell[:-1]) / 100 if cell.endswith("%") else float(cell)
                    item.update(match_rate=rate, low_match_rate=rate < .2)
                if "膨胀" in column and cell in {"是", "true", "True", "1", "⚠是", "⚠️是", "否", "false", "False", "0"}:
                    item["row_inflation"] = cell in {"是", "true", "True", "1", "⚠是", "⚠️是"}
                if "去重" in column:
                    match = re.fullmatch(r"安全(\d+)/[⚠️]*冲突(\d+)", cell)
                    if match:
                        item.update(safe_key_count=int(match[1]), conflict_key_count=int(match[2]))
            if item:
                out.append({"table_row_index": len(out) + 1, **item})
    return out


def split_configuration(value):
    if not isinstance(value, dict):
        return {}
    if value.get("rules"):
        raise OutboundContextError("当前切分含规则条件，尚无法完整生成可发送的上下文，未请求模型决策。")
    from marvis.agent.adjust_specs import adjust_param_error, normalize_adjust_params
    from marvis.llm_outbound_instructions import column_identifiers
    from marvis.modeling_split import SplitConfigurationError, validate_split_configuration

    controls = {key: item for key, item in value.items() if key not in {"notes", "description", "summary", "reason"}}
    normalized = normalize_adjust_params({"split_config": controls})
    if adjust_param_error(normalized):
        raise OutboundContextError("当前切分配置无效或存在冲突，未发送请求。")
    try:
        out = validate_split_configuration(normalized["split_config"])
    except SplitConfigurationError:
        raise OutboundContextError("当前切分配置不是已安装的结构合同，未发送请求。") from None
    if out.get("oot_by_time") is not None:
        column_identifiers([out["oot_by_time"]])
    if "group_cols" in out:
        column_identifiers(out["group_cols"])
    return out


def _weight_diagnostics(value):
    reasons = {"": "none", "全为空或非数值": "no_numeric_values", "存在空值或非数值": "missing_or_non_numeric",
               "存在非正权重": "non_positive_weights", "总权重不为正": "non_positive_weight_sum"}
    out = []
    for row in _rows(value):
        field = _identifier(row.get("column"))
        if field is _DROP:
            continue
        item = {"column": field, **_numeric(row, (
            "rows_sampled", "non_missing", "missing_rate", "min", "max", "mean", "target_correlation",
        )), **_labels(row, {"leakage_risk": {"high", "low", "unknown"}})}
        for key in ("valid", "excluded_from_features"):
            if type(row.get(key)) is bool:
                item[key] = row[key]
        reason = row.get("reason")
        if isinstance(reason, str) and reason in reasons:
            item["reason_code"] = reasons[reason]
        out.append(item)
    return out


def _controls(value):
    out = []
    for control in _rows(value):
        name = control.get("id")
        if _identifier(name) is _DROP:
            continue
        schema = control.get("schema") if isinstance(control.get("schema"), dict) else {}
        kind = control.get("kind")
        projected = _gate_parameters([{
            "name": name, "current": control.get("default"), "bounds": control.get("bounds"),
            "enum": schema.get("enum", []), "type": {
                "number": "number", "list": "array", "multi_select": "array", "map": "object",
                "select": "string", "input": "string",
            }.get(kind, schema.get("type")),
        }])[0]
        item = {"id": name, **_labels(control, {"kind": {"number", "list", "multi_select", "map", "select", "input"}}),
                "required": control.get("required") is True,
                "schema": structural_summary({"schema": schema}).get("schema", {})}
        if "bounds" in projected:
            item["bounds"] = projected["bounds"]
        if "current" in projected:
            item["default"] = projected["current"]
        elif "default" in control:
            item["current_value_withheld"] = True
        if name in {"sample_weight_col", "features", "selection"} and isinstance(schema.get("enum"), list):
            item["schema"]["enum"] = [entry for entry in schema["enum"]
                                      if entry == "" or _identifier(entry) is not _DROP]
        if name == "split_config":
            from marvis.modeling_split_schema import split_configuration_schema, split_configuration_semantics

            item["schema"] = split_configuration_schema()
            item["semantics"] = split_configuration_semantics()
        out.append(item)
    return out


def _screen(value):
    out = structural_summary({key: value[key] for key in ("selected",) if key in value})
    out.update(_numeric(value, ("n_screened",)))
    thresholds = value.get("thresholds")
    out["thresholds"] = _numeric(thresholds if isinstance(thresholds, dict) else {}, (
        "leakage_ks", "max_missing_rate", "max_ks_decay", "max_feature_psi",
    ))
    for name, metric in (
        ("ranked", "ks"), ("leakage", "ks"), ("suspected", "ks"), ("unusable", None),
        ("leakage_watch", "ks"), ("ks_decay_watch", "ks_decay"),
        ("psi_watch", "psi_split"), ("split_shift", "ks_shift"),
    ):
        entries = value.get(name)
        if not isinstance(entries, (list, tuple)):
            continue
        if len(entries) > 10000:
            raise OutboundContextError("筛选证据过大，未截断风险列表或发送请求。")
        out[name] = []
        for row in entries:
            if not isinstance(row, (list, tuple)) or not row or _identifier(row[0]) is _DROP:
                continue
            clean = {"feature": row[0]}
            if metric and len(row) > 1:
                clean.update(_numeric({metric: row[1]}, (metric,)))
            out[name].append(clean)
    scores = value.get("scores")
    if isinstance(scores, dict):
        if len(scores) > 10000:
            raise OutboundContextError("筛选指标过大，未截断或发送请求。")
        out["feature_metrics"] = [{"feature": field, **_numeric(row, (
            "ks", "iv", "missing_rate", "unique_count", "ks_train", "ks_test", "ks_decay", "psi_split",
            "assoc_score", "deterministic_outcome_fit_a_count", "deterministic_outcome_fit_b_count",
        ))} for field, row in scores.items() if _identifier(field) is not _DROP and isinstance(row, dict)]
    return out


def auto_gate_prompt(envelope, metadata, *, red_flag_count, plan_overview=None, role_assignment=None):
    from marvis.agent.gates.contracts import _HIGH_RISK_GATE_SOURCE_TOOLS, _HIGH_RISK_STEP_ID_TOKENS

    meta = metadata if isinstance(metadata, dict) else {}
    raw_gate = envelope.to_dict()
    clean = structural_summary({"gate": raw_gate, "metadata": meta, "red_flag_count": red_flag_count})
    known_flags = {flag for group in (*_HIGH_RISK_GATE_SOURCE_TOOLS.values(), *_HIGH_RISK_STEP_ID_TOKENS.values())
                   for flag in group} | {"model_delivery_champion_handoff"}
    clean["gate"]["risk_flags"] = [flag for flag in envelope.risk_flags if flag in known_flags]
    clean["gate"]["unrecognized_risk_flag_count"] = sum(flag not in known_flags for flag in envelope.risk_flags)
    # The local safety policy still checks the complete original envelope.
    # An extension's arbitrary label is not thereby approved outbound text.
    clean["metadata"].pop("risk_flags", None)
    if plan_overview is not None:
        from marvis.llm_outbound_plan_overview import ApprovedPlanOverview

        if envelope.kind != "plan_overview" or not isinstance(plan_overview, ApprovedPlanOverview):
            raise OutboundContextError("执行计划概览来源或当前节点无效，未发送请求。")
        clean["plan_overview"] = plan_overview.payload()
    if role_assignment is not None:
        from marvis.llm_outbound_role_assignment import ApprovedRoleAssignment

        if not isinstance(meta.get("join_c1"), dict) or not isinstance(role_assignment, ApprovedRoleAssignment):
            raise OutboundContextError("文件角色节点来源无效，未发送请求。")
        clean["role_assignment"] = role_assignment.payload()
    clean["gate"]["controls"] = _controls(raw_gate.get("controls"))
    clean["metadata"]["red_flag_evidence"] = _risk_evidence(meta.get("red_flag_evidence"))
    clean["metadata"]["join_diagnostics"] = _join_diagnostics(meta.get("tables"))
    if isinstance(meta.get("screen"), dict):
        clean["metadata"]["screen"] = _screen(meta["screen"])
    setup = meta.get("modeling_setup")
    if isinstance(setup, dict):
        target = clean["metadata"].setdefault("modeling_setup", {})
        if setup.get("sample_weight_col") == "":
            target["sample_weight_col"] = ""
        target["sample_weight_diagnostics"] = _weight_diagnostics(setup.get("sample_weight_diagnostics"))
        target.update(_labels(setup, {"metric_policy": _METRIC_POLICIES}))
        if isinstance(setup.get("split_config"), dict):
            target["split_config"] = split_configuration(setup["split_config"])
        summary = setup.get("split_summary")
        if isinstance(summary, dict):
            target.setdefault("split_summary", {}).update(_numeric(summary, ("total_rows",)))
            known_warnings = {
                "缺少 train 样本。": "missing_train", "缺少 test 样本。": "missing_test",
                "缺少 OOT 样本，上线前建议补充时间外验证。": "missing_oot",
                "OOT 占比低于 5%，稳定性结论需谨慎。": "small_oot_share",
            }
            warnings = summary.get("warnings")
            target["split_summary"]["warning_codes"] = [known_warnings[message.replace(",", "，")]
                for message in warnings if isinstance(message, str) and message.replace(",", "，") in known_warnings
            ] if isinstance(warnings, list) else []
            if isinstance(summary.get("split_config"), dict):
                target["split_summary"]["split_config"] = split_configuration(summary["split_config"])
    clean["instruction"] = (
        "按当前节点声明的动作和控件审阅平台结果。当前值未提供或被保留在本地时不要推测；"
        "screen.leakage 是平台硬泄漏标记，suspected 是需确认的疑似模型输出，unusable 是不可用特征；"
        "各 watch 列表是平台确定的风险观察项，不能当作无风险，也不能自行改写平台筛选结果。"
        "red_flag_evidence 是与本地风险文字同源的确定性检查；保留其中的风险代码、阈值和已测差距。"
        "权重有效性、与标签的相关性和泄漏风险必须一并考虑，不能只因权重为正就忽略标签泄漏。"
        "未提供的样本、自由错误和报告文字不是可用证据。"
    )
    return _mint_prompt(clean)
