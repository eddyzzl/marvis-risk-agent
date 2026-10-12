"""User-authored control text, separate from data-bearing local artifacts.

This boundary is for the current user's instruction only. It must never be used
to certify retrieved documents, old assistant messages, report prose or files.
It preserves the complete text or refuses it; it does not classify intent.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata

from marvis.llm_outbound import OutboundContextError, SafeInstruction, _PROMPT_TOKEN


MAX_INSTRUCTION_CHARS = 16384
# These are content boundaries, not a vocabulary of permitted business tasks.
# Field names, numeric thresholds, dates, negations and conditions remain intact.
_SENSITIVE_PATTERNS = (
    ("credential", r"(?i)(?:\b(?:[a-z0-9]+_)*(?:api[_ -]?key|password|passwd|pwd|secret|access[_ -]?(?:key|token)|authorization|connection[_ -]?string|private[_ -]?key)\b|密码|密钥|令牌)[\"']?\s*(?:[:=：]|是|为|\bis\b|\bequals\b)\s*\S+"),
    ("credential", r"(?i)\b(?:bearer\s+[A-Za-z0-9._~-]{8,}|sk-[A-Za-z0-9_-]{12,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"),
    ("local_path", r"(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9]|/(?:Users|home|tmp|var|etc|private|opt|mnt|workspace)/|(?:^|\s)\.{1,2}/)"),
    ("url_or_connection", r"(?i)\b(?:https?|postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|jdbc|file)://\S+"),
    ("personal_identifier", r"(?<![A-Za-z0-9_])(?:1[3-9]\d{9}|\d{17}[\dXx]|\d{15,19})(?![A-Za-z0-9_])"),
    ("personal_identifier", r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    ("personal_record", r"(?:客户姓名|客户名称|身份证号|手机号码|银行卡号|客户地址|姓名)[\"']?\s*(?:[:=：]|是|为)\s*\S+"),
    ("personal_record", r"(?:客户|账户|账号|客户ID)\s+(?:\d{6,}|[A-Za-z0-9_-]+\s+(?:分数|余额|金额)\s*\d+)"),
    ("source_literal", r"(?:文件正文|报告全文|样本明细|Notebook源码|模型文件内容)\s*[:：=]\s*\S+"),
    ("raw_records", r"(?i)(?:客户(?:记录|明细)|样本(?:数据|明细|行)|records|sample_rows)\s*[:：=]?\s*[\[{]"),
    ("source_literal", r"\{\s*[\"'][^\n\"']{1,128}[\"']\s*:"),
    ("source_code", r"(?im)^\s*(?:import\s+[A-Za-z_]|from\s+[A-Za-z_.]+\s+import|def\s+\w+\s*\(|class\s+\w+\s*[:(]|RMC_\w+\s*=)"),
    ("source_code", r"```|~~~|(?im:^\s*(?:print|exec|eval|open)\s*\()"),
    ("model_or_notebook", r"(?i)(?:<\??(?:xml|PMML)\b|\"(?:cells|nbformat|model_bytes|sample_rows|records|preview_rows)\"\s*:|\b(?:gAS[A-Za-z0-9+/=]{24,}))"),
    ("tabular_records", r"(?m)^\s*\|[^\n]*\|\s*$"),
)
_COMPILED_PATTERNS = tuple((reason, re.compile(pattern)) for reason, pattern in _SENSITIVE_PATTERNS)


class InstructionLengthError(OutboundContextError):
    """A complete instruction exceeded the admission budget; never trim it."""


def user_instruction(value: str, *, allowed_identifiers=()) -> SafeInstruction:
    """Approve a complete user-authored utterance, without semantic rewriting.

    The caller owns provenance: only an actual interactive user input or its
    unchanged locally persisted goal may use this constructor. Identifier
    bindings are metadata, not permission to bypass any content boundary.
    """
    del allowed_identifiers
    if not isinstance(value, str) or not value.strip():
        raise OutboundContextError("用户指令为空，未发送模型请求。")
    if len(value) > MAX_INSTRUCTION_CHARS:
        raise InstructionLengthError("用户指令超过长度上限，未截断约束或发送请求；请拆分任务。")
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(char) in {"Cc", "Cf"} and char not in "\n\r\t" for char in normalized):
        raise OutboundContextError("用户指令包含不可见控制字符，未发送请求；请重新输入。")
    for reason, pattern in _COMPILED_PATTERNS:
        if pattern.search(normalized):
            # Never include the matched sensitive value in a UI error or audit.
            raise OutboundContextError(
                f"用户指令含禁止外发内容（{reason}），未发送请求；请移除明细、凭据、路径或文件正文，使用已导入材料的结构化入口。"
            )
    starts = [match.start() for match in re.finditer(r"[\[{]", normalized)]
    if len(starts) > 100:
        raise OutboundContextError("指令结构过多，未截断约束或发送请求；请使用结构化控件。")
    decoder = json.JSONDecoder()
    for offset in starts:
        try:
            literal, _ = decoder.raw_decode(normalized, offset)
        except (ValueError, RecursionError):
            continue
        # Explicit cutpoints/band edges are user-authored numeric controls.
        # A bare array, nested rows, strings or a document object still cannot
        # be admitted as an instruction. Keep the original text verbatim.
        numeric_control = (
            isinstance(literal, list) and 0 < len(literal) <= 64
            and all(type(item) in (int, float) and math.isfinite(item)
                    and abs(item) < 10**15 for item in literal)
            and re.search(r"(?:切点|分带边界|分箱边界|输出值|breakpoints|band_edges)\s*[:：=]?\s*$",
                          normalized[:offset], re.IGNORECASE)
        )
        if isinstance(literal, (list, dict)) and not numeric_control:
            raise OutboundContextError("用户指令含记录或结构正文（source_literal），未发送；请使用对应的结构化入口。")
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    for delimiter in (",", "\t"):
        for previous, line in zip(lines, lines[1:]):
            fields = line.split(delimiter)
            if (len(fields) >= 2 and len(previous.split(delimiter)) == len(fields)
                    and any(re.fullmatch(r"-?\d+(?:\.\d+)?", item.strip()) for item in fields)
                    and not any("=" in item for item in fields)):
                raise OutboundContextError("用户指令含逐行表格内容（tabular_records），未发送；请使用数据导入入口。")
    # A sequence of YAML-style key/value lines is a document body, even without
    # fences or braces. Interactive prose and one standalone task heading remain
    # available; structured settings have their own provenance-bound controls.
    key_value = re.compile(r"^[\w\u4e00-\u9fff][\w\u4e00-\u9fff .-]{0,127}\s*[:：]\s*\S.*")
    if any(key_value.fullmatch(left) and key_value.fullmatch(right)
           for left, right in zip(lines, lines[1:])):
        raise OutboundContextError("用户指令含结构正文（source_literal），未发送；请使用结构化入口。")
    return SafeInstruction(value, _token=_PROMPT_TOKEN)


def column_identifiers(values) -> list[str]:
    """Bound field-name metadata; no descriptions or observed cell values."""
    if not isinstance(values, (list, tuple)) or len(values) > 10000:
        raise OutboundContextError("字段目录过大或无效，未截断契约或发送请求。")
    out = []
    for value in values:
        if (not isinstance(value, str) or not value or len(value) > 128
                or any(unicodedata.category(char)[0] not in {"L", "M", "N"}
                       and char not in "_ -()." for char in value)):
            raise OutboundContextError("字段名称不符合结构出站规则，请先在数据工作台设置字段别名。")
        # Field names carry provenance as schema metadata, not a user message.
        for reason, pattern in _COMPILED_PATTERNS:
            if pattern.search(unicodedata.normalize("NFKC", value)):
                raise OutboundContextError(f"字段名称含禁止外发内容（{reason}），请设置字段别名。")
        out.append(value)
    return out
