"""Generate bounded V2 narrative sections before exposing one complete draft."""
from __future__ import annotations

from copy import deepcopy
import json

from marvis.llm_prompts import WORD_CONCLUSION_V2_SYSTEM_PROMPT
from marvis.repositories.tasks import AGENT_REPORT_NARRATIVE_KEYS


_FINAL = "TEXT:final_validation_conclusion"
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
            keys += ["lift_ranking_assessment"]
        scoped["effectiveness"] = {key: effectiveness[key] for key in keys if key in effectiveness}
    if topic in {"performance", "overall", "recommendation"} and "overfitting_check" in metrics:
        scoped["overfitting_check"] = metrics["overfitting_check"]
    if topic in {"pressure", "recommendation", "overall"} and "stress_test" in metrics:
        scoped["stress_test"] = metrics["stress_test"]
    return scoped


def _section_payload(original, topic, keys):
    payload = deepcopy(original)
    payload["requested_fields"] = list(keys)
    payload["narrative_topic"] = topic
    payload["instructions"] = (
        "本次只生成 requested_fields 的报告文字，其他段落由独立调用完成。"
        "保持本模型专属的证据解读和完整句子，直接输出 JSON，不输出推导过程。"
        "指标和通过判断只依据平台 evidence.validation_results；已保存报告文字、当前草稿及历史记忆仅辅助修订和比较。"
        "evidence.saved_report_narrative 是已有报告文字，report_draft 中的当前编辑优先；二者均不能替代本次指标证据。"
        "如有 user_instruction，以当前草稿为起点修改指定内容，保留未要求改写的事实及主动清空字段。"
        "已有报告或草稿中的可选字段只在用户明确要求修改时返回；未修改字段请省略，平台会原样保留。"
        "本次仅负责一个主题，其他主题会完整合并；不要在本次重复整份结论。"
        "使用完整、精炼的中文段落，通常 150 至 350 字；证据限制必须保留，不输出标题或推导过程。"
        + _TOPICS[topic]
    )
    evidence = payload.get("evidence", {})
    # Stage narratives and execution bookkeeping are not additional measured
    # evidence. Keep the deterministic metrics and independently cited memory.
    for name in ("scan", "pmml_scoring", "visible_stage_summaries", "report_fields"):
        evidence.pop(name, None)
    # Keep the whole current draft as read-only context: revisions can refer to
    # another paragraph. The response schema alone restricts the writable keys.
    if "validation_results" in evidence:
        evidence["validation_results"] = _topic_metrics(evidence["validation_results"], topic)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _generate_section(client, original, topic, keys, *, truncated=False):
    spec = WORD_CONCLUSION_V2_SYSTEM_PROMPT
    schema = {
        "type": "object",
        "properties": {key: {"type": "string"} for key in keys},
        "required": [keys[0]],
        "additionalProperties": False,
    }
    kind = "metrics" if original.get("stage") == "metrics" else "report"
    content = client.complete(
        system_prompt=spec.text,
        user_prompt=_section_payload(original, topic, keys),
        json_schema={"name": f"validation_{kind}_{topic}", "strict": False, "schema": schema},
        temperature=0.2, stream=False, max_tokens=2048, truncated=truncated,
        caller=f"validation_{kind}_{topic}", prompt_name=spec.name, prompt_version=spec.version,
    )
    try:
        section = json.loads(content)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{kind} topic {topic} is not valid JSON") from exc
    if (not isinstance(section, dict) or not set(section) <= set(keys)
            or not isinstance(section.get(keys[0]), str) or not section[keys[0]].strip()
            or any(not isinstance(value, str) for value in section.values())):
        raise ValueError(f"{kind} topic {topic} violates its text contract")
    return {key: value.strip() for key, value in section.items()}


def generate_v2_sections(client, prompt):
    """Return all required sections or raise; never publish partial text."""
    original = json.loads(prompt)
    values, conclusion = {}, []
    for topic, keys in _REPORT_SECTIONS:
        section = _generate_section(client, original, topic, keys)
        if keys == (_FINAL,):
            conclusion.append(section[_FINAL])
        else:
            values.update(section)
    values[_FINAL] = "\n\n".join(conclusion)
    return values


def generate_v2_metrics_summary(client, prompt, *, truncated=False):
    """Use the same evidence projections for complete, bounded stage analysis."""
    original = json.loads(prompt)
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
