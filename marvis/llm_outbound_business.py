"""Measured business acceptance for external explanation, without private contracts."""
from datetime import date
import re

from marvis.llm_outbound import OutboundContextError, _METRIC, _mint_prompt, structural_summary
from marvis.llm_outbound_reports import _labels, _numeric
from marvis.orchestrator.evidence import payload_hash

_STATUSES = {"passed", "failed", "insufficient_evidence", "not_configured", "not_applicable"}
_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_UNITS = {"ratio", "percent", "count", "currency", "days"}
_DENOMINATORS = {"mature_loans", "all_applicants", "all_rows", "approved_loans", "disbursed_loans"}
_METRICS = {"expected_profit", "net_profit", "approval_rate", "bad_rate", "loss_rate", "population"}
_REASONS = {
    "未配置业务验收标准；流程完成不代表业务达标。": "objective_not_configured",
    "缺少实际采用对象的认证证据。": "adopted_evidence_missing",
    "采用对象或版本未绑定。": "target_binding_missing",
    "证据不属于指定采用对象。": "target_mismatch",
    "采用对象版本不匹配。": "target_version_mismatch",
    "缺少认证证据引用或内容哈希。": "evidence_authentication_missing",
    "证据效果阶段不足。": "effect_stage_insufficient",
    "缺少成熟的真实观察标签；假设或拒绝推断不能代替。": "mature_observed_labels_missing",
    "缺少因果识别条件；观察差异不能认定为增量收益。": "causal_identification_missing",
    "旧成功标准缺少业务口径、时期与采用对象绑定；不能据候选最大值认定达标。": "legacy_objective_unbound",
    **{label + "未绑定或不匹配。": key + "_mismatch" for label, key in (
        ("业务线", "business_line"), ("决策节点", "decision_node"), ("目标客群", "population"),
        ("时期起点", "period_start"), ("时期终点", "period_end"), ("币种", "currency"),
    )},
}
_CRITERION_REASONS = {
    "指标单位未绑定或不匹配": "unit_mismatch",
    "指标分母未绑定或不匹配": "denominator_mismatch",
    "基线未认证或不匹配": "baseline_unbound",
    "基线客群、时期、单位或分母不匹配": "baseline_context_mismatch",
    "基线指标单位或分母不匹配": "baseline_measurement_mismatch",
    "baseline must be a finite number": "baseline_not_finite",
    "delta must be a finite number": "delta_not_finite",
}


def _hashes(value, fields):
    return {key: value[key] for key in fields
            if isinstance(value.get(key), str) and _HASH.fullmatch(value[key])}


def business_acceptance_context(result):
    """Preserve every criterion and native verdict; alias private object labels."""
    if not isinstance(result, dict) or result.get("status") not in _STATUSES:
        raise OutboundContextError("业务验收状态不可识别，未发送解释请求。")
    criteria = result.get("criteria")
    if not isinstance(criteria, list) or len(criteria) > 10000:
        raise OutboundContextError("业务验收标准不可完整传递，未截断或发送请求。")
    out = {"status": result["status"], "criteria_count": len(criteria), "criteria": [],
           **_hashes(result, ("objective_hash", "execution_binding"))}
    target = result.get("target")
    if isinstance(target, dict):
        # Raw ids/versions may contain a business name. Bind the complete adopted
        # identity locally and send its digest, never a guessed candidate label.
        identity = {key: target.get(key) for key in ("kind", "id", "version")}
        out["target"] = {"role": "adopted_target", "binding_hash": payload_hash(identity),
                         **_labels(target, {"kind": {"model", "strategy", "workflow"}})}
    else:
        out["target"] = None
    evidence = result.get("evidence")
    if isinstance(evidence, dict):
        clean = {**_hashes(evidence, ("source_hash", "baseline_binding_hash")),
                 **_labels(evidence, {
                     "effect_stage": {"estimated", "backtested", "oot_validated", "post_launch_observed"},
                     "label_origin": {"observed", "assumed", "reject_inferred"},
                     "claim": {"observational", "causal"},
                 })}
        if evidence.get("labels_mature") is None or type(evidence.get("labels_mature")) is bool:
            clean["labels_mature"] = evidence.get("labels_mature")
        for key in ("period_start", "period_end"):
            value = evidence.get(key)
            if value is None:
                clean[key] = None
            elif isinstance(value, str):
                try:
                    if date.fromisoformat(value).isoformat() == value:
                        clean[key] = value
                except ValueError:
                    pass
        out["evidence"] = clean
    else:
        out["evidence"] = None
    for index, row in enumerate(criteria, 1):
        if not isinstance(row, dict) or row.get("status") not in _STATUSES:
            raise OutboundContextError("业务验收标准不可识别，未发送解释请求。")
        item = {"criterion_index": index, "status": row["status"],
                **_numeric(row, ("minimum", "maximum", "value", "baseline_value")),
                **_labels(row, {"comparison": {"absolute", "delta"},
                                "unit": _UNITS, "denominator": _DENOMINATORS})}
        metric = row.get("metric")
        if isinstance(metric, str) and (metric in _METRICS or _METRIC.fullmatch(metric)):
            item["metric"] = metric
        else:
            item["metric_name_withheld"] = True
        reason = row.get("reason")
        if isinstance(reason, str) and reason:
            code = _CRITERION_REASONS.get(reason)
            if isinstance(metric, str):
                if reason == metric + " 未达到约定阈值。":
                    code = "threshold_not_met"
                elif reason == metric + " must be a finite number":
                    code = "metric_not_finite"
            item["reason_code"] = code or "details_withheld"
        for key in ("unit", "denominator"):
            if key not in item:
                item[key + "_withheld"] = True
        out["criteria"].append(item)
    out["reason_count"] = len(result.get("reasons") or [])
    out["reason_text_withheld"] = bool(out["reason_count"])
    out["reason_codes"] = [_REASONS[reason] for reason in result.get("reasons", [])
                           if isinstance(reason, str) and reason in _REASONS]
    return out


def business_review_prompt(*, goal, plan_id, step_count, outputs, result):
    clean = structural_summary({"goal": goal, "plan_id": plan_id,
                                "step_count": step_count, "outputs": outputs})
    clean["business_acceptance"] = business_acceptance_context(result)
    clean["acceptance_authority"] = (
        "业务验收由平台确定性判定。解释必须遵守此status、objective_hash和采用对象绑定，"
        "不得用候选指标改写结论；流程完成不代表业务达标。"
        "整体status独立于单项status；delta的value已经是差值，不要再次减去baseline_value。"
        "minimum和maximum均含边界。名称或原因已隐藏时按criterion_index引用，不能猜测。"
    )
    return _mint_prompt(clean)
