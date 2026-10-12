"""Historical model aggregates with local-only identity and citation metadata."""
from __future__ import annotations

from dataclasses import dataclass
import json

from marvis.llm_outbound import OutboundContextError
from marvis.llm_outbound_reports import _labels, _numeric, _period, _rows


_HISTORY_TOKEN = object()


@dataclass(frozen=True, init=False)
class ModelHistory:
    # Both strings are immutable. Only the first may enter an LLM request.
    payload_json: str
    references_json: str

    def __init__(self, payload_json, references_json, *, _token=None):
        if _token is not _HISTORY_TOKEN:
            raise OutboundContextError("历史模型上下文必须由受信任结构适配器创建。")
        object.__setattr__(self, "payload_json", payload_json)
        object.__setattr__(self, "references_json", references_json)

    def metadata(self, *, use_reason):
        references = json.loads(self.references_json)
        result = {"history_comparison_available": bool(references)}
        if references:
            result["memory_references"] = [{**item, "use_reason": use_reason} for item in references]
        return result


def _known(value):
    return isinstance(value, str) and value.strip().casefold() not in {"", "未标注", "未知", "unknown", "n/a"}


def _same(left, right):
    return _known(left) and _known(right) and left.strip().casefold() == right.strip().casefold()


def model_history_context(*, task, evidence, memory_context):
    """Project recalled raw model experience; never parse its summary as facts.

    The caller owns retrieval policy. Comparability here is an additional local
    check, not a claim that model family alone proves equivalent populations.
    Distillations and free-form preference/pitfall memories need other adapters.
    """
    from marvis.agent_memory.api_support import agent_memory_dimensions_from_validation_results
    from marvis.agent_memory.retrieval import normalize_model_family
    from marvis.agent_memory.models import MODEL_EXPERIENCE_EVALUATION_SPLITS, MODEL_EXPERIENCE_METRIC_UNITS

    evidence = evidence if isinstance(evidence, dict) else {}
    current = agent_memory_dimensions_from_validation_results(evidence.get("validation_results"))
    current["model_name"] = task.model_name or current.get("model_name")
    current["model_version"] = task.model_version or current.get("model_version")
    family = normalize_model_family(current["model_name"])
    context = memory_context if isinstance(memory_context, dict) else {}
    records, references, seen = [], [], set()
    for item in _rows(context.get("memories")):
        confidence = _labels(item, {"confidence": {"high", "medium"}}).get("confidence")
        if (item.get("kind", "raw") != "raw" or item.get("memory_type") != "model_experience"
                or item.get("status", "active") != "active" or not confidence):
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        identity, source = item.get("id"), item.get("source_task_id")
        if (not isinstance(identity, str) or not identity or identity in seen
                or not isinstance(source, str) or not source or source == str(task.id)
                or ("source_task_id" in payload and payload["source_task_id"] != source)):
            continue
        if any(not isinstance(payload.get(key), str) or not payload[key].strip()
               for key in ("model_name", "model_version", "scope", "channel", "month")):
            continue
        metrics = _numeric(payload, ("ks", "auc", "psi", "head_lift_5pct", "tail_lift_5pct"))
        if not any(metrics.get(key) is not None for key in ("ks", "auc", "psi")):
            continue
        same_model = _same(current.get("model_name"), payload["model_name"])
        same_family = bool(family and family == normalize_model_family(payload["model_name"]))
        same_scope = _same(current.get("scope"), payload["scope"])
        if not (same_model or same_family or same_scope):
            continue
        index = len(records) + 1
        relation = {"same_model_name": same_model, "same_model_family": same_family,
                    "same_scope": same_scope,
                    "scope_known_for_current": _known(current.get("scope")),
                    "channel_known_for_current": _known(current.get("channel")),
                    "same_channel": _same(current.get("channel"), payload["channel"]),
                    "same_model_version": _same(current.get("model_version"), payload["model_version"])}
        declared_units = payload.get("metric_units")
        units = {}
        if isinstance(declared_units, dict):
            for metric, expected_unit in MODEL_EXPERIENCE_METRIC_UNITS.items():
                value = metrics.get(metric)
                if (declared_units.get(metric) == expected_unit and value is not None and value >= 0
                        and (expected_unit != "fraction" or value <= 1)):
                    units[metric] = expected_unit
        split = _labels(payload, {"evaluation_split": MODEL_EXPERIENCE_EVALUATION_SPLITS})
        core_metrics = {key for key in ("ks", "auc", "psi") if metrics.get(key) is not None}
        record = {"history_index": index, "confidence": item["confidence"], "comparison_basis": relation,
                  "metrics": {**metrics, **_labels(payload, {"overfitting_status": {"pass", "fail", "not_available"}})},
                  "metric_units_known": bool(core_metrics) and core_metrics <= units.keys(),
                  "evaluation_split_known": bool(split), "metric_units": units, **split}
        month = _period(payload["month"])
        if month:
            record["month"] = month
        if type(item.get("age_days")) is int and item["age_days"] >= 0:
            record["age_days"] = item["age_days"]
        records.append(record)
        references.append({"kind": "raw", "id": identity, "memory_type": "model_experience",
                           "source_task_id": source, "confidence": item["confidence"], "history_index": index})
        seen.add(identity)
    return ModelHistory(json.dumps({"records": records}, ensure_ascii=False, allow_nan=False),
                        json.dumps(references, ensure_ascii=False, allow_nan=False), _token=_HISTORY_TOKEN)


def add_model_history(payload, history):
    """Add only the internal projection, never a caller's raw memory mapping."""
    if history is not None and not isinstance(history, ModelHistory):
        raise OutboundContextError("历史模型上下文尚未完成结构化投影，未发送请求。")
    context = json.loads(history.payload_json) if history else {"records": []}
    payload["history_comparison_available"] = bool(context["records"])
    if context["records"]:
        payload["historical_models"] = context
        payload["instructions"] += (
            "historical_models 为历史记忆中的结构化指标，只能辅助对比和风险提醒，不能改变本次确定性结果。"
            "引用时使用「历史记录 N」及其月份与置信度，N 为 history_index；来源由本地展开引用提供。"
            "比较条件以 comparison_basis 为准，同类模型不等于同渠道或同客群，未知条件必须说明。"
            "历史记录的单位仅以 metric_units 为准，评估分层仅以 evaluation_split 为准；"
            "两者缺失时明确未知，不默认已与本次指标一致。只有单位与样本分层一致的指标才可作直接数值对照。"
            "原值照录，不自行换算百分比，"
            "不能把历史 KS=20 直接与本次 KS=0.3 作数值增减比较，也不能默认历史值就是 OOT。"
            "只做有依据的克制对照，不据此声称模型升级有效或当前指标通过。"
        )
    else:
        payload["instructions"] += "本次未提供可用的结构化历史模型记录，明确本次未见可比历史模型，不编造历史比较。"
    return payload
