"""Summarize admitted structured memory facts; prose and identity stay local."""
from marvis.llm_outbound import OutboundContextError, _finite, _mint_prompt
from marvis.llm_outbound_analysis import field_names


def distillation_prompt(category, members, structured):
    from marvis.agent_memory.models import MemoryCandidate, MODEL_EXPERIENCE_METRIC_UNITS, MODEL_EXPERIENCE_EVALUATION_SPLITS
    from marvis.agent_memory.policy import classify_memory_candidate
    from marvis.agent_memory.store import MemoryEntry

    if not members or len(members) > 2000 or not isinstance(structured, dict):
        raise OutboundContextError("记忆压缩缺少已准入的结构化事实，保留本地确定性摘要。")
    for member in members:
        if not isinstance(member, MemoryEntry) or member.status != "active" or member.memory_type != category:
            raise OutboundContextError("未准入记忆不能进入外部压缩请求，保留本地摘要。")
        candidate = MemoryCandidate(memory_type=category, summary=member.summary, payload=member.payload,
            source_task_id=member.source_task_id, confidence=member.confidence)
        if not classify_memory_candidate(candidate).allowed:
            raise OutboundContextError("记忆未通过敏感内容校验，保留本地摘要。")
    support = structured.get("support")
    if type(support) is not int or not 0 <= support <= len(members):
        raise OutboundContextError("记忆支持计数无效，保留本地摘要。")
    facts = {"support": support}
    if category == "model_experience":
        metrics = {}
        for key in ("ks", "auc", "psi"):
            row = (structured.get("metric_ranges") or {}).get(key)
            if isinstance(row, dict) and all(_finite(row.get(part)) for part in ("min", "max")) and row["min"] <= row["max"]:
                metrics[key] = {part: row[part] for part in ("min", "max")}
        if not metrics:
            raise OutboundContextError("记忆没有可外发的模型聚合指标，保留本地摘要。")
        facts["metric_ranges"] = metrics
        units = structured.get("metric_units")
        if units is not None and (not isinstance(units, dict) or any(MODEL_EXPERIENCE_METRIC_UNITS.get(key) != unit for key, unit in units.items())):
            raise OutboundContextError("模型记忆单位契约无效，保留本地摘要。")
        facts["metric_units"] = {key: unit for key, unit in (units or {}).items() if key in metrics}
        split = structured.get("evaluation_split")
        if split is not None and split not in MODEL_EXPERIENCE_EVALUATION_SPLITS:
            raise OutboundContextError("模型记忆评估分层未知，保留本地摘要。")
        if split:
            facts["evaluation_split"] = split
        facts.update(metric_units_known=set(metrics) <= facts["metric_units"].keys(), evaluation_split_known=bool(split))
    elif category == "validation_pitfall":
        kind = structured.get("pitfall_type")
        if kind not in {"notebook", "pmml", "score_consistency", "sample", "target", "general", "feature", "stability", "deployment"}:
            raise OutboundContextError("记忆坑点类别尚不支持结构外发，保留本地摘要。")
        facts["pitfall_type"] = kind
    elif category == "field_convention":
        fields = {}
        for key in ("target_col", "time_col", "id_col", "score_col", "sample_weight_col", "split_col"):
            values = (structured.get("fields") or {}).get(key)
            if values:
                fields[key] = field_names(values)
        if not fields:
            raise OutboundContextError("记忆字段口径缺少受支持的标识结构，保留本地摘要。")
        facts["fields"] = fields
    elif category == "task_experience":
        tags = structured.get("outcome_tags") or {}
        allowed = {"completed", "failed", "succeeded", "cancelled", "halted", "general", "blocked", "done"}
        if not tags or any(key not in allowed or type(value) is not int or not 0 <= value <= len(members) for key, value in tags.items()):
            raise OutboundContextError("任务记忆缺少受支持的结果类别，保留本地摘要。")
        facts["outcome_tags"] = tags
    else:
        raise OutboundContextError("该记忆类别尚无已准入的结构出站契约，保留本地确定性摘要。")
    return _mint_prompt({"stage": "memory_distillation", "category": category, "facts": facts,
        "instruction": "只将结构事实改写为一句经验。未提供原记忆叙事、身份或作用域标签，不得推测它们；不新增数字、因果或建议。缺少单位或评估分层时必须说明未知，不能默认可直接比较。"})
