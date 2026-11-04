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
# Operations whose new prose comes from measured evidence (never local text).
EVIDENCE_OPERATIONS = frozenset({"regenerate_from_evidence", "append_from_evidence", "replace_and_append"})
APPEND_OPERATIONS = frozenset({"append_from_evidence", "replace_and_append"})
LOCAL_OPERATIONS = frozenset({"clear", "set_local_text", "replace_local_text"})
_BUSINESS_ALIASES = {
    "模型概述": "TEXT:model_overview", "概述": "TEXT:model_overview",
    "模型适用范围": "TEXT:model_scope", "适用范围": "TEXT:model_scope", "模型范围": "TEXT:model_scope",
    "坏样本定义": "TEXT:bad_sample_definition", "坏样本": "TEXT:bad_sample_definition",
    "好样本定义": "TEXT:good_sample_definition", "好样本": "TEXT:good_sample_definition",
    "样本客群": "TEXT:sample_audience", "样本来源": "TEXT:sample_audience", "样本对象": "TEXT:sample_audience",
    "样本人群": "TEXT:sample_audience",
}
_FIELD_ALIASES = {**_BUSINESS_ALIASES,
    "最终验证结论": _FINAL, "最终结论": _FINAL, "结论": _FINAL,
    "压力测试总结": "TEXT:pressure_test_summary", "压力测试建议": "TEXT:pressure_impact_recommendation",
    "压力影响建议": "TEXT:pressure_impact_recommendation",
    "训练描述": "TEXT:model_training_description", "模型训练说明": "TEXT:model_training_description",
}
# Longest alias first so "最终验证结论" never matches as "结论".
_FIELD_ALTERNATION = "|".join(sorted(_FIELD_ALIASES, key=len, reverse=True))
# The draft card's「请 Agent 修订」button pre-fills exactly this field-bound prefix.
REVISION_BUTTON_PREFIX = re.compile(r"\s*请根据当前证据修订当前模型报告的「([^」\n]{1,16})」\s*[：:]\s*(.*)", re.S)
_APPEND_VERBS = r"(?:补充|补上|补进|加上|写上|增加|添加|加入)"
_CONDITIONAL = re.compile(r"如果|假如|除非|只有.{0,20}才|不要继续|先.{0,20}再")
_METRIC_TEXT = re.compile(
    r"\d|\b(?:KS|AUC|PSI)\b|分数一致性|(?:百分之|千分之|点)[零〇一二三四五六七八九十百千万两]+"
    r"|^[零〇一二三四五六七八九十百千万两]+$", re.IGNORECASE,
)
UNSUPPORTED_REVISION_GUIDANCE = (
    "这句修改无法确定地对应到报告字段，未发送请求，报告内容没有改动。可以这样说："
    "①在报告表格对应字段点「请 Agent 修订」，在冒号后写要求；"
    "②把最终验证结论里的“原文”改成“新文”（可接“，并补充……”）；"
    "③在最终验证结论里补充……；④把适用范围改成“……”；⑤重新生成报告。"
)


def report_revision_button_field(text) -> str | None:
    """Report key named by the revise button's prefix, or None."""
    match = REVISION_BUTTON_PREFIX.fullmatch(str(text or ""))
    return _FIELD_ALIASES.get(match[1].strip()) if match else None


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
    button = REVISION_BUTTON_PREFIX.fullmatch(instruction)
    if button:
        return _button_revision_request(instruction, button[1].strip(), button[2].strip())
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
    aliases = _BUSINESS_ALIASES
    replacement = re.fullmatch(
        r"\s*(?:请)?(?:把)?(" + _FIELD_ALTERNATION + r")(?:中|里)(?:的)?\s*"
        r"[“\"]([^“”\"\n]+)[”\"]\s*(?:改成|改为|替换为|换成)\s*[“\"]([^“”\"\n]*)[”\"]"
        r"(?:[，,；;]\s*(?:其余|其他)(?:字段|段落)?(?:保持)?不变)?[。.!！]?\s*", instruction,
    )
    if replacement:
        return _replacement_request(
            instruction, _FIELD_ALIASES[replacement[1]], replacement[2], replacement[3],
        )
    # "把<字段>里的“A”改成“B”，并补充……": the replacement stays local; the
    # addendum is written from measured evidence and appended locally.
    compound = re.fullmatch(
        r"\s*(?:请)?(?:把)?(" + _FIELD_ALTERNATION + r")(?:中|里)(?:的)?\s*"
        r"[“\"]([^“”\"\n]+)[”\"]\s*(?:改成|改为|替换为|换成)\s*[“\"]([^“”\"\n]*)[”\"]"
        r"\s*[，,；;]?\s*(?:并且|并|同时|再|还要|也要)?\s*(?:在(?:结论|该段|此段)?(?:中|里))?\s*"
        + _APPEND_VERBS + r"\s*(?:一下)?\s*(.+?)[。.!！]?\s*", instruction, re.S,
    )
    if compound:
        return _replacement_request(
            instruction, _FIELD_ALIASES[compound[1]], compound[2], compound[3], append_point=compound[4],
        )
    append = re.fullmatch(
        r"\s*(?:请)?(?:在)?(" + _FIELD_ALTERNATION + r")(?:中|里)?\s*(?:再|也|还)?(?:要)?\s*"
        + _APPEND_VERBS + r"\s*(?:一下)?\s*(.+?)[。.!！]?\s*", instruction, re.S,
    )
    if append:
        return _append_request(instruction, _FIELD_ALIASES[append[1]], append[2])
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
    raise ReportRevisionError(UNSUPPORTED_REVISION_GUIDANCE)


def _replacement_request(instruction, field, old, new, *, append_point=None):
    if field in _REGENERATABLE and (_METRIC_TEXT.search(old) or _METRIC_TEXT.search(new)):
        raise ReportRevisionError(
            "指标及数字不能按文字替换修订，请查看本次平台证据；可在业务叙事字段编辑样本定义。",
            code="metric_rewrite_refused",
        )
    if append_point is None:
        return ReportRevisionRequest(
            instruction, "replace_local_text", (field,), local_text=new, local_old_text=old, _token=_REQUEST_TOKEN,
        )
    _require_appendable(field, append_point)
    return ReportRevisionRequest(
        instruction, "replace_and_append", (field,), local_text=new, local_old_text=old, _token=_REQUEST_TOKEN,
    )


def _append_request(instruction, field, point):
    _require_appendable(field, point)
    return ReportRevisionRequest(instruction, "append_from_evidence", (field,), _token=_REQUEST_TOKEN)


def _require_appendable(field, point):
    if field not in _REGENERATABLE:
        raise ReportRevisionError(
            "业务口径字段（概述、适用范围、样本定义）只按你给出的原文填写，平台不会代写补充内容；"
            "请写成：把适用范围改成“……”。报告内容没有改动。",
        )
    if not str(point or "").strip() or _CONDITIONAL.search(point):
        raise ReportRevisionError(UNSUPPORTED_REVISION_GUIDANCE)


def _button_revision_request(instruction, label, rest):
    """The revise button names one field; the text after the colon is guidance."""
    field = _FIELD_ALIASES.get(label)
    if field is None:
        raise ReportRevisionError(f"未识别要修订的报告字段「{label[:16]}」，报告内容没有改动。")
    if not rest:
        if field in _REGENERATABLE:
            return ReportRevisionRequest(instruction, "regenerate_from_evidence", (field,), _token=_REQUEST_TOKEN)
        raise ReportRevisionError(
            f"「{label}」是业务口径字段，平台不会代写；请在冒号后写出要填写的内容。报告内容没有改动。",
        )
    replacement = re.fullmatch(
        r"(?:请)?(?:把)?\s*[“\"]([^“”\"\n]+)[”\"]\s*(?:改成|改为|替换为|换成)\s*[“\"]([^“”\"\n]*)[”\"]"
        r"(?:\s*[，,；;]?\s*(?:并且|并|同时|再|还要|也要)?\s*" + _APPEND_VERBS + r"\s*(?:一下)?\s*(.+?))?[。.!！]?\s*",
        rest, re.S,
    )
    if replacement:
        return _replacement_request(
            instruction, field, replacement[1], replacement[2], append_point=replacement[3],
        )
    append = re.fullmatch(r"(?:请)?(?:再|也|还)?(?:要)?\s*" + _APPEND_VERBS + r"\s*(?:一下)?\s*(.+?)[。.!！]?\s*", rest, re.S)
    if append and field in _REGENERATABLE:
        return _append_request(instruction, field, append[1])
    if field not in _REGENERATABLE:
        literal = re.fullmatch(
            r"(?:请)?(?:改成|改为|修改为|写成|设置为|填写为|填成|填为)?\s*"
            r"(?:[“\"]([^“”\"\n]+)[”\"]|([^\n]+?))[。.!！]?\s*", rest,
        )
        text = (literal[1] or literal[2]).strip() if literal else ""
        if text and not re.search(r"如果|除非|只有|不要|不得|帮我|根据|补充|生成|重写|其余|其他字段|保持不变", text):
            return ReportRevisionRequest(
                instruction, "set_local_text", (field,), local_text=text, _token=_REQUEST_TOKEN,
            )
        raise ReportRevisionError(
            f"「{label}」是业务口径字段，平台不会代写；请在冒号后直接写出要填写的内容，"
            "例如：改成“贷中存量客户”。报告内容没有改动。",
        )
    if _CONDITIONAL.search(rest):
        raise ReportRevisionError(UNSUPPORTED_REVISION_GUIDANCE)
    # Guidance for regenerating one measured field. The model still receives
    # only measured evidence; metrics remain platform values.
    return ReportRevisionRequest(instruction, "regenerate_from_evidence", (field,), _token=_REQUEST_TOKEN)


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
    if request.operation in LOCAL_OPERATIONS | APPEND_OPERATIONS:
        return request.target_fields
    # Deliberate blanks remain local blanks through a generic measured redraft.
    # An explicit request to regenerate a field authorizes replacing its blank.
    return tuple(key for key in request.target_fields
                 if (request.instruction is not None or baseline.get(key) != "")
                 and not (request.instruction is None and key == "TEXT:model_training_description" and key in baseline))


def report_revision_prompt(prompt, *, request, provenance, requested_fields):
    """Extend an already projected measured prompt with a bounded operation."""
    if (not isinstance(request, ReportRevisionRequest) or request.operation not in EVIDENCE_OPERATIONS
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
    if request.operation in APPEND_OPERATIONS:
        clean["instructions"] += (
            "本次只为 requested_fields 撰写一段补充内容（约 80 至 200 字），聚焦"
            " report_revision_operation.instruction 中要求补充的要点，依据已有确定性证据；"
            "平台会把这段补充附加到本地原段落之后，因此不要复述整段结论或其他主题，"
            "不读取或猜测本地旧段落。其他报告字段由平台原样保留，不得返回额外字段。"
            "来源版本只用于绑定本次修订，不构成模型效果、报告确认或投产证据。"
        )
    else:
        clean["instructions"] += (
            "本次仅按已有确定性证据整段重写 requested_fields，不读取或猜测本地旧段落。"
            "report_revision_operation.instruction 是用户对口径和侧重点的要求，应在证据范围内遵循；"
            "其中出现的指标数字不能替代平台证据。"
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
