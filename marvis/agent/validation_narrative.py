"""Generate bounded V2 narrative sections before exposing one complete draft."""
from __future__ import annotations

from copy import deepcopy
import json
import re

from marvis.llm_prompts import WORD_CONCLUSION_V2_SYSTEM_PROMPT
from marvis.repositories.tasks import AGENT_REPORT_NARRATIVE_KEYS


_FINAL = "TEXT:final_validation_conclusion"
_FINAL_FORBIDDEN_PATTERNS = (
    re.compile(r"材料(?:扫描|识别|完备)"),
    re.compile(r"验证输入契约"),
    re.compile(r"PMML\s*(?:全量)?(?:打分|评分)(?:测试|完成|通过|成功|覆盖|样本|耗时)?", re.IGNORECASE),
    re.compile(r"报告(?:已|进入|生成|产出|定稿)"),
    re.compile(r"最终定稿|建议(?:在)?投产前审阅|建议审阅(?:报告|结论)|确认\s*Word", re.IGNORECASE),
    re.compile(r"可直接(?:部署|投产)"),
)
_TOPICS = {
    "pressure": "解释高、中、低风险数据源及基线和剔除后 KS/PSI；缺失分层明确未知。",
    "recommendation": "围绕已测出的风险给出监控、替代、降级和使用限制建议。",
    "performance": "逐一引用已有 Train/Test/OOT KS、AUC，解释区分效果及平台过拟合检查。",
    "stability": "引用已有各样本 PSI 和逐期证据，解释样本外稳定性、变化及限制。",
    "ranking": "只按 lift_ranking_assessment 分别解释按 train 分箱和独立分箱的单调性、头尾幅度及区分度。",
    "overall": "综合效果、稳定性和压力测试的关键风险作审慎判断，不重复各项明细；克制对比可比历史记忆，没有则明确本次未见可比历史模型。",
    "training": "介绍实际算法和给定关键超参；仅在用户要求修改时返回可选业务叙事字段。",
}
_REPORT_SECTIONS = (
    ("pressure", ("TEXT:pressure_test_summary",)),
    ("recommendation", ("TEXT:pressure_impact_recommendation",)),
    *((topic, (_FINAL,)) for topic in ("performance", "stability", "ranking", "overall")),
    ("training", ("TEXT:model_training_description", *sorted(AGENT_REPORT_NARRATIVE_KEYS))),
)


def _topic_metrics(metrics, topic):
    """Project existing evidence, never recompute or infer missing metrics."""
    if not isinstance(metrics, dict):
        return metrics
    scoped = {key: metrics[key] for key in (
        "model_name", "model_version", "algorithm", "target_type", "basic_info",
    ) if key in metrics}
    if topic == "training":
        return scoped
    basic = scoped.get("basic_info")
    if isinstance(basic, dict):
        scoped["basic_info"] = {key: basic[key] for key in ("sample_period", "split_summary") if key in basic}
    effectiveness = metrics.get("effectiveness")
    if isinstance(effectiveness, dict):
        keys = ["overall"]
        # Overall judgements and recommendations must see adverse periods even
        # when aggregate split metrics hide them. Other generated prose is never
        # substituted for these deterministic observations.
        if topic in {"stability", "overall", "recommendation"}:
            keys += ["monthly_ks", "monthly_psi", "psi_stability_table"]
        if topic in {"ranking", "overall", "recommendation"}:
            keys += ["lift_ranking_assessment", "bin_tables", "independent_quantile_bin_tables"]
        scoped["effectiveness"] = {key: effectiveness[key] for key in keys if key in effectiveness}
    if topic in {"performance", "overall", "recommendation"} and "overfitting_check" in metrics:
        scoped["overfitting_check"] = metrics["overfitting_check"]
    if topic in {"pressure", "recommendation", "overall"} and "stress_test" in metrics:
        scoped["stress_test"] = metrics["stress_test"]
    return scoped


def has_report_revision_context(evidence, user_instruction=None, *, algorithm=None):
    """Existing prose needs the revision contract, even with no new instruction."""
    if user_instruction:
        return True
    if not isinstance(evidence, dict):
        return False
    if evidence.get("report_draft"):
        return True
    saved = evidence.get("saved_report_narrative")
    # The native stage always supplies this envelope, including before the
    # first draft. Only that exact uninitialized state has no revision prose.
    # A saved empty edit, positive revision, or unknown context stays protected.
    if (isinstance(saved, dict) and set(saved) == {"report_revision", "text_values"}
            and type(saved["report_revision"]) is int
            and saved["report_revision"] == 0):
        if saved["text_values"] == {}:
            return False
        if algorithm is not None:
            from marvis.model_algorithms import PENDING_MODEL_TRAINING_DESCRIPTION, model_training_report_text
            from marvis.validation_report_copy import narrative_report_values

            try:
                training = model_training_report_text(algorithm) if str(algorithm).strip() else PENDING_MODEL_TRAINING_DESCRIPTION
            except ValueError:
                return True
            initial_values = {**narrative_report_values(""), "TEXT:model_training_description": training}
            if saved["text_values"] == initial_values:
                return False
    return bool(saved)


def _section_payload(original, topic, keys, *, repair_reason=None):
    if original.get("stage") == "metrics":
        from marvis.llm_outbound import OutboundContextError
        from marvis.llm_outbound_validation import validation_metric_section_prompt

        if original.get("user_instruction"):
            raise OutboundContextError("指标解读修订指令尚未完成出站适配，未发送请求。")
        return validation_metric_section_prompt(
            _topic_metrics(original.get("evidence", {}).get("validation_results"), topic),
            topic=topic, repair_reason=repair_reason,
            history=original.get("model_history"),
        )
    if original.get("report_revision_request") is None and not has_report_revision_context(
        original.get("evidence"), original.get("user_instruction"),
    ):
        from marvis.llm_outbound_validation import validation_report_section_prompt

        return validation_report_section_prompt(
            _topic_metrics(original.get("evidence", {}).get("validation_results"), topic),
            topic=topic, repair_reason=repair_reason,
            history=original.get("model_history"),
        )
    from marvis.llm_outbound_revision import ReportRevisionError, report_revision_prompt
    from marvis.llm_outbound_validation import validation_report_section_prompt

    request = original.get("report_revision_request")
    if request is None:
        raise ReportRevisionError("报告修订需明确已核对的字段操作，未发送请求。")
    measured = validation_report_section_prompt(
        _topic_metrics(original.get("evidence", {}).get("validation_results"), topic),
        topic=topic, repair_reason=repair_reason, history=original.get("model_history"),
    )
    return report_revision_prompt(
        measured, request=request, provenance=original["report_provenance"], requested_fields=[keys[0]],
    )


def _generate_section(client, original, topic, keys, *, truncated=False):
    spec = WORD_CONCLUSION_V2_SYSTEM_PROMPT
    if original.get("stage") != "metrics":
        # Measured narratives cannot overwrite local business definitions that
        # were not supplied to the model or authorized for this operation.
        keys = (keys[0],)
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": [keys[0]],
        "additionalProperties": False,
    }
    kind = "metrics" if original.get("stage") == "metrics" else "report"
    repair_reason = None
    for attempt in range(2):
        repair = attempt == 1
        content = client.complete(
            system_prompt=spec.text,
            user_prompt=_section_payload(original, topic, keys, repair_reason=repair_reason),
            json_schema={"name": f"validation_{kind}_{topic}", "strict": False, "schema": schema},
            temperature=0.1 if repair else 0.2, stream=False, max_tokens=2048, truncated=truncated,
            caller=f"validation_{kind}_{topic}" + ("_repair" if repair else ""),
            prompt_name=spec.name, prompt_version=spec.version,
        )
        try:
            repair_reason = "json_text_contract"
            section = _parse_section(content, keys, kind=kind, topic=topic)
            repair_reason = "forbidden_process_narration"
            validate_v2_word_conclusions(section)
            return section
        except ValueError as exc:
            if repair:
                raise ValueError(f"{kind} topic {topic}: {exc}") from exc
    raise AssertionError("unreachable")


def _parse_section(content, keys, *, kind, topic):
    try:
        section = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{kind} topic {topic} is not valid JSON") from exc
    if (not isinstance(section, dict) or not set(section) <= set(keys)
            or not isinstance(section.get(keys[0]), str) or not section[keys[0]].strip()
            or any(not isinstance(value, str) for value in section.values())):
        raise ValueError(f"{kind} topic {topic} violates its text contract")
    return {key: value.strip() for key, value in section.items()}


def validate_v2_word_conclusions(values):
    """One unchanged guard for individual topics and the assembled report."""
    conclusion = values.get(_FINAL, "")
    for pattern in _FINAL_FORBIDDEN_PATTERNS:
        if pattern.search(conclusion):
            raise ValueError("V2 final validation conclusion contains forbidden process narration")


def generate_v2_sections(client, prompt):
    """Return all required sections or raise; never publish partial text."""
    original = deepcopy(prompt) if isinstance(prompt, dict) else json.loads(prompt)
    original["stage"] = "word_conclusion_draft"
    values, conclusion = {}, []
    for topic, keys in _REPORT_SECTIONS:
        section = _generate_section(client, original, topic, keys)
        if keys == (_FINAL,):
            conclusion.append(section[_FINAL])
        else:
            values.update(section)
    values[_FINAL] = "\n\n".join(conclusion)
    return values


def generate_v2_revision_sections(client, evidence, *, request, provenance, fields, history=None):
    """Generate only authorized fields; the saved report remains with the caller."""
    from marvis.llm_outbound_revision import ReportRevisionError, ReportRevisionRequest

    allowed = {keys[0] for _topic, keys in _REPORT_SECTIONS}
    if (not isinstance(request, ReportRevisionRequest) or request.operation != "regenerate_from_evidence"
            or not fields or not set(fields) <= allowed or not set(fields) <= set(request.target_fields)):
        raise ReportRevisionError("报告修订章节超出已核对的字段操作，未发送请求。")
    original = {
        "stage": "word_conclusion_draft", "evidence": evidence,
        "report_revision_request": request, "report_provenance": provenance,
        "model_history": history,
    }
    values, conclusion = {}, []
    for topic, keys in _REPORT_SECTIONS:
        if keys[0] not in fields:
            continue
        section = _generate_section(client, original, topic, (keys[0],))
        if keys[0] == _FINAL:
            conclusion.append(section[_FINAL])
        else:
            values.update(section)
    if conclusion:
        values[_FINAL] = "\n\n".join(conclusion)
    return values


def generate_v2_metrics_summary(client, prompt, *, truncated=False):
    """Use the same evidence projections for complete, bounded stage analysis."""
    original = deepcopy(prompt) if isinstance(prompt, dict) else json.loads(prompt)
    original["stage"] = "metrics"
    sections = []
    for heading, topics in (
        ("总体判断", ("overall",)),
        ("效果表现", ("performance", "ranking")),
        ("稳定性表现", ("stability",)),
        ("压力测试风险", ("pressure",)),
        ("建议", ("recommendation",)),
    ):
        parts = [_generate_section(client, original, topic, ("summary",), truncated=truncated)["summary"]
                 for topic in topics]
        sections.append(heading + "\n" + "\n\n".join(parts))
    return "\n\n".join(sections)
