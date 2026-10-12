"""Bounded report revisions without exporting the locally saved report.

A revision is an operation on named fields, not a new classification of report
prose as safe. Paragraph-preserving edits need that prose and remain unsupported
until a separate approved data contract exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from marvis.llm_outbound import OutboundContextError, SafeInstruction, _mint_prompt
from marvis.repositories.tasks import AGENT_REPORT_CONCLUSION_KEYS, AGENT_REPORT_WRITABLE_KEYS


_FINAL = "TEXT:final_validation_conclusion"
_REGENERATABLE = AGENT_REPORT_CONCLUSION_KEYS | {"TEXT:model_training_description"}
_REQUEST_TOKEN = object()


class ReportRevisionError(OutboundContextError):
    def __init__(self, message, *, code="unsupported_report_instruction"):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, init=False)
class ReportRevisionRequest:
    instruction: SafeInstruction | None
    operation: str
    target_fields: tuple[str, ...]
    local_text: str | None
    local_old_text: str | None

    def __init__(self, instruction, operation, target_fields, *, local_text=None, local_old_text=None, _token=None):
        if _token is not _REQUEST_TOKEN:
            raise ReportRevisionError("报告修订必须来自已核对的字段操作，未发送请求。")
        object.__setattr__(self, "instruction", instruction)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "target_fields", tuple(target_fields))
        object.__setattr__(self, "local_text", local_text)
        object.__setattr__(self, "local_old_text", local_old_text)


def report_revision_request(instruction: SafeInstruction | None = None):
    """Parse a whole approved instruction without dropping any constraints.

    No new lexical allowance is granted here. Even a SafeInstruction can contain
    a condition or refer to text that is unavailable to the external model.
    Only exact field operations below have a deterministic semantic mapping.
    """
    if instruction is None:
        return ReportRevisionRequest(None, "regenerate_from_evidence", sorted(_REGENERATABLE), _token=_REQUEST_TOKEN)
    if not isinstance(instruction, SafeInstruction):
        raise ReportRevisionError("报告修订指令尚未获准外发，请使用报告字段编辑控件。")
    compact = re.sub(r"[\s，,。.!！]+", "", instruction)
    if compact in {"重新生成报告", "再生成报告", "请重新生成报告", "请再生成报告"}:
        return ReportRevisionRequest(
            instruction, "regenerate_from_evidence", sorted(AGENT_REPORT_CONCLUSION_KEYS),
            _token=_REQUEST_TOKEN,
        )
    match = re.fullmatch(r"(?:请)?按(?:本次|当前)验证结果(?:改写|修改)(结论|报告)(?:其余不变|其他不变)?", compact)
    if match:
        fields = (_FINAL,) if match[1] == "结论" else sorted(AGENT_REPORT_CONCLUSION_KEYS)
        return ReportRevisionRequest(instruction, "regenerate_from_evidence", fields, _token=_REQUEST_TOKEN)
    match = re.fullmatch(
        r"(?:请)?(?:按|根据)(?:本次|当前)(?:平台)?验证结果(?:整段)?(?:重写|重新撰写|重新生成)"
        r"((?:最终(?:验证)?)?结论|报告)(?:(?:其余|其他)(?:字段|段落)?(?:保持)?(?:不变|原样))?", compact,
    )
    if match:
        fields = sorted(AGENT_REPORT_CONCLUSION_KEYS) if match[1] == "报告" else (_FINAL,)
        return ReportRevisionRequest(instruction, "regenerate_from_evidence", fields, _token=_REQUEST_TOKEN)
    if compact in {"删除结论", "删除结论其余不变", "把结论删掉", "把结论删掉其余不变"}:
        return ReportRevisionRequest(instruction, "clear", (_FINAL,), _token=_REQUEST_TOKEN)
    # An explicit business-field replacement is applied locally. The supplied
    # literal never becomes external model evidence or permission to edit metrics.
    aliases = {
        "模型概述": "TEXT:model_overview", "概述": "TEXT:model_overview",
        "模型适用范围": "TEXT:model_scope", "适用范围": "TEXT:model_scope", "模型范围": "TEXT:model_scope",
        "坏样本定义": "TEXT:bad_sample_definition", "坏样本": "TEXT:bad_sample_definition",
        "好样本定义": "TEXT:good_sample_definition", "好样本": "TEXT:good_sample_definition",
        "样本客群": "TEXT:sample_audience", "样本来源": "TEXT:sample_audience", "样本对象": "TEXT:sample_audience",
    }
    replacement_aliases = {**aliases,
        "最终验证结论": _FINAL, "最终结论": _FINAL, "结论": _FINAL,
        "压力测试总结": "TEXT:pressure_test_summary", "压力测试建议": "TEXT:pressure_impact_recommendation",
        "训练描述": "TEXT:model_training_description",
    }
    replacement = re.fullmatch(
        r"\s*(?:请)?(?:把)?(" + "|".join(replacement_aliases) + r")(?:中|里)(?:的)?\s*"
        r"[“\"]([^“”\"\n]+)[”\"]\s*(?:改成|改为|替换为|换成)\s*[“\"]([^“”\"\n]*)[”\"]"
        r"(?:[，,；;]\s*(?:其余|其他)(?:字段|段落)?(?:保持)?不变)?[。.!！]?\s*", instruction,
    )
    if replacement:
        field, old, new = replacement_aliases[replacement[1]], replacement[2], replacement[3]
        metric_text = re.compile(
            r"\d|\b(?:KS|AUC|PSI)\b|分数一致性|(?:百分之|千分之|点)[零〇一二三四五六七八九十百千万两]+"
            r"|^[零〇一二三四五六七八九十百千万两]+$", re.IGNORECASE,
        )
        if field in _REGENERATABLE and (metric_text.search(old) or metric_text.search(new)):
            raise ReportRevisionError(
                "指标及数字不能按文字替换修订，请查看本次平台证据；可在业务叙事字段编辑样本定义。",
                code="metric_rewrite_refused",
            )
        return ReportRevisionRequest(
            instruction, "replace_local_text", (field,), local_text=new, local_old_text=old, _token=_REQUEST_TOKEN,
        )
    clear_match = re.fullmatch(
        r"(?:请)?(?:把)?(" + "|".join(aliases) + r")(?:清空|删掉|删除)(?:其余不变|其他不变)?"
        r"|(?:请)?(?:清空|删除)(" + "|".join(aliases) + r")(?:其余不变|其他不变)?", compact,
    )
    if clear_match:
        field = clear_match[1] or clear_match[2]
        return ReportRevisionRequest(instruction, "clear", (aliases[field],), _token=_REQUEST_TOKEN)
    match = re.fullmatch(
        r"\s*(?:请)?(?:把)?(" + "|".join(aliases) + r")\s*(?:改成|改为|修改为|写成|设置为)\s*"
        r"(?:[“\"]([^“”\"\n]+)[”\"]|([^，,；;。!?！？\n]+?))"
        r"(?:[，,；;]\s*(?:其余|其他)(?:字段|段落)?(?:保持)?不变)?[。.!！]?\s*", instruction,
    )
    if match:
        literal = (match[2] or match[3]).strip()
        if literal and not re.search(r"如果|除非|只有|先.*再|不要|不得|其余|其他字段|保持不变", literal):
            return ReportRevisionRequest(
                instruction, "set_local_text", (aliases[match[1]],), local_text=literal, _token=_REQUEST_TOKEN,
            )
    raise ReportRevisionError(
        "当前支持按本次验证结果改写指定结论，或在报告表格直接编辑文字。"
        "此指令包含尚不支持的修订语义，未发送请求；请明确要整段重写的字段，"
        "涉及保留原段落事实或措辞的修改请使用本地字段编辑。"
    )


def report_revision_baseline(evidence, *, report_revision):
    """Validate local provenance; retain every saved field verbatim locally."""
    if type(report_revision) is not int or report_revision < 0:
        raise ReportRevisionError("报告版本无法核对，请刷新当前报告草稿。", code="invalid_report_provenance")
    row = evidence if isinstance(evidence, dict) else {}
    baseline, provenance = {}, {"report_revision": report_revision}
    for name in ("saved_report_narrative", "report_draft"):
        context = row.get(name)
        if not context:
            continue
        allowed = {"report_revision", "text_values"}
        if name == "report_draft":
            allowed |= {"message_id", "draft_edit_revision"}
        if (not isinstance(context, dict) or set(context) - allowed
                or type(context.get("report_revision")) is not int
                or context["report_revision"] != report_revision
                or not isinstance(context.get("text_values"), dict)):
            raise ReportRevisionError("报告草稿来源或版本无法核对，请刷新后重试。", code="invalid_report_provenance")
        values = context["text_values"]
        if (set(values) - AGENT_REPORT_WRITABLE_KEYS
                or any(not isinstance(value, str) for value in values.values())):
            raise ReportRevisionError("报告草稿字段不符合编辑契约，请在报告表格中核对。", code="invalid_report_provenance")
        baseline.update(values)
        if name == "report_draft":
            edit_revision = context.get("draft_edit_revision", 0)
            if type(edit_revision) is not int or edit_revision < 0:
                raise ReportRevisionError("草稿编辑版本无法核对，请刷新后重试。", code="invalid_report_provenance")
            provenance["draft_edit_revision"] = edit_revision
    return baseline, provenance


def revision_fields(request, baseline):
    if not isinstance(request, ReportRevisionRequest):
        raise ReportRevisionError("报告修订缺少已核对的操作，未发送请求。")
    if request.operation in {"clear", "set_local_text", "replace_local_text"}:
        return request.target_fields
    # Deliberate blanks remain local blanks through a generic measured redraft.
    # An explicit request to regenerate a field authorizes replacing its blank.
    return tuple(key for key in request.target_fields
                 if (request.instruction is not None or baseline.get(key) != "")
                 and not (request.instruction is None and key == "TEXT:model_training_description" and key in baseline))


def report_revision_prompt(prompt, *, request, provenance, requested_fields):
    """Extend an already projected measured prompt with a bounded operation."""
    if (not isinstance(request, ReportRevisionRequest) or request.operation != "regenerate_from_evidence"
            or not requested_fields or not set(requested_fields) <= set(request.target_fields)
            or not set(requested_fields) <= _REGENERATABLE):
        raise ReportRevisionError("报告修订字段操作未获准，未发送请求。")
    # This receives a minted structural prompt, never arbitrary serialized text.
    from marvis.llm_outbound import StructuralPrompt

    if not isinstance(prompt, StructuralPrompt):
        raise ReportRevisionError("报告修订证据必须通过结构投影，未发送请求。")
    if (not isinstance(provenance, dict) or set(provenance) - {"report_revision", "draft_edit_revision"}
            or "report_revision" not in provenance
            or any(type(value) is not int or value < 0 for value in provenance.values())):
        raise ReportRevisionError("报告修订来源版本无效，未发送请求。", code="invalid_report_provenance")
    clean = json.loads(prompt)
    if clean.get("validation_workflow_version") == 1:
        clean["instructions"] = clean["instructions"].replace(
            "本次为首次候选文字生成，只返回指定的四项候选文字，不改写其他业务定义字段。",
            "本次修订只返回 requested_fields 授权的文字，不改写其他业务定义字段。",
        )
    clean["report_revision_operation"] = {
        "operation": request.operation, "target_fields": list(request.target_fields),
        "provenance": dict(provenance),
    }
    if request.instruction is not None:
        clean["report_revision_operation"]["instruction"] = request.instruction
    clean["requested_fields"] = list(requested_fields)
    clean["instructions"] += (
        "本次仅按已有确定性证据整段重写 requested_fields，不读取或猜测本地旧段落。"
        "其他报告字段由平台原样保留，不得返回额外字段。来源版本只用于绑定本次修订，"
        "不构成模型效果、报告确认或投产证据。"
    )
    return _mint_prompt(clean)


def parse_report_revision_response(content, fields):
    try:
        values = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise ValueError("报告修订返回不是有效 JSON") from exc
    if (not isinstance(values, dict) or set(values) != set(fields)
            or any(not isinstance(value, str) or not value.strip() for value in values.values())):
        raise ValueError("报告修订返回字段超出授权或缺少完整文字")
    return {key: value.strip() for key, value in values.items()}
