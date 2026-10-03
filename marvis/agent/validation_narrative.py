"""Generate bounded V2 narrative sections before exposing one complete draft."""
from __future__ import annotations

from copy import deepcopy
import json

from marvis.llm_prompts import WORD_CONCLUSION_V2_SYSTEM_PROMPT
from marvis.repositories.tasks import AGENT_REPORT_NARRATIVE_KEYS


_SECTIONS = (
    ("TEXT:pressure_test_summary",),
    ("TEXT:pressure_impact_recommendation",),
    ("TEXT:final_validation_conclusion",),
    ("TEXT:model_training_description", *sorted(AGENT_REPORT_NARRATIVE_KEYS)),
)


def _section_payload(original, keys):
    payload = deepcopy(original)
    payload["requested_fields"] = list(keys)
    payload["instructions"] = (
        "本次只生成 requested_fields 的报告文字，其他段落由独立调用完成。"
        "保持本模型专属的证据解读和完整句子，直接输出 JSON，不输出推导过程。"
        "所有数值和判断依据来自平台 evidence；当前草稿及历史记忆仅辅助修订和比较。"
        "如有 user_instruction，以当前草稿为起点修改指定内容，保留未要求改写的事实及主动清空字段。"
        "已有草稿中的可选字段只在用户明确要求修改时返回；未修改字段请省略，平台会原样保留。"
    )
    evidence = payload.get("evidence", {})
    # Stage narratives and execution bookkeeping are not additional measured
    # evidence. Keep the deterministic metrics and independently cited memory.
    for name in ("scan", "pmml_scoring", "visible_stage_summaries", "report_fields"):
        evidence.pop(name, None)
    # Keep the whole current draft as read-only context: revisions can refer to
    # another paragraph. The response schema alone restricts the writable keys.
    metrics = evidence.get("validation_results")
    if isinstance(metrics, dict) and keys[0] == "TEXT:model_training_description":
        evidence["validation_results"] = {
            key: metrics[key] for key in ("model_name", "model_version", "algorithm", "target_type", "basic_info")
            if key in metrics
        }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_v2_sections(client, prompt):
    """Return all required sections or raise; never publish partial text."""
    original = json.loads(prompt)
    values = {}
    spec = WORD_CONCLUSION_V2_SYSTEM_PROMPT
    for index, keys in enumerate(_SECTIONS):
        schema = {
            "type": "object",
            "properties": {key: {"type": "string"} for key in keys},
            "required": [keys[0]],
            "additionalProperties": False,
        }
        content = client.complete(
            system_prompt=spec.text,
            user_prompt=_section_payload(original, keys),
            json_schema={
                "name": f"validation_report_section_{index + 1}", "strict": False, "schema": schema,
            },
            temperature=0.2, stream=False, max_tokens=2048,
            caller="validation_report_section", prompt_name=spec.name, prompt_version=spec.version,
        )
        try:
            section = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"report section {index + 1} is not valid JSON") from exc
        if (not isinstance(section, dict) or not set(section) <= set(keys)
                or not isinstance(section.get(keys[0]), str) or not section[keys[0]].strip()
                or any(not isinstance(value, str) for value in section.values())):
            raise ValueError(f"report section {index + 1} violates its text contract")
        values.update({key: value.strip() for key, value in section.items()})
    return values
