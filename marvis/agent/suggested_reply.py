"""One suggested next reply per LLM-authored Agent turn (UI-6).

The suggestion is display-only message metadata (``suggested_reply``): the
composer shows it as a placeholder and the user may accept it with the
right-arrow key. Nothing on the backend reads or acts on it. Anything that is
not a short, plain, user-language sentence is dropped rather than repaired:
an absent suggestion is always safe.
"""
from __future__ import annotations

import re

MAX_SUGGESTED_REPLY_CHARS = 40
METADATA_KEY = "suggested_reply"
# Shared by every adapter that asks the model for a suggestion.
SUGGESTED_REPLY_INSTRUCTION = (
    "另给出一条用户下一句可能直接发送的建议回复：使用与用户最新消息相同的语言，"
    f"不超过 {MAX_SUGGESTED_REPLY_CHARS} 个字；若平台正在等待用户确认，给出自然的确认答复"
    "（如「确认」或简短的参数调整），否则给出自然的下一步问题或指令。"
    "建议回复不得包含任何指标数值或结果数字，不得包含编号、工具名或英文字段代码。"
)

# Result metrics: a suggestion must never quote or imply a measured value.
_METRIC_WORDS = (
    r"ks|auc|psi|csi|iv|gini|lift|f1|rmse|mae|r2|logloss|brier|"
    r"分数|评分|得分|score|坏账率|逾期率|坏样本率|命中率|缺失率|通过率|准确率|召回率|精确率|"
    r"区分度|稳定性指标|相关系数|vif"
)
_METRIC_NUMBER = re.compile(
    rf"(?:(?:{_METRIC_WORDS})\s*(?:值|为|是|=|:|：|>|<|≥|≤|>=|<=|约|达到|仅)?\s*[-+]?\d)"
    rf"|(?:\d(?:[\d.,]*)\s*%?\s*(?:的)?\s*(?:{_METRIC_WORDS}))",
    re.IGNORECASE,
)
# "KS达到零点四二": a metric word followed by a number spelled in Chinese numerals.
_METRIC_CHINESE_NUMBER = re.compile(
    rf"(?:{_METRIC_WORDS})\s*(?:值|为|是|=|:|：|>|<|≥|≤|>=|<=|约|达到|仅|只有|有)?\s*[零〇一二两三四五六七八九十百点]",
    re.IGNORECASE,
)
# Decimals and percentages read as measured results, whatever they are next to.
_MEASURED_NUMBER = re.compile(r"\d+\.\d+|\d\s*%|百分之|\d\s*分(?![钟析箱别类组层布])")
# Internal identifiers, tool names and codes the user should never be asked to type.
# Python treats CJK characters as word characters, so ``\b`` misses a code glued
# to Chinese text ("删除任务task_3f2a吗"); the boundaries are ASCII lookarounds.
_ASCII_START = r"(?<![A-Za-z0-9_])"
_ASCII_END = r"(?![A-Za-z0-9_])"
_INTERNAL_ID = re.compile(
    _ASCII_START + r"(?:ds|plan|artifact|task|exp|step|run|job|mem|draft|gate|tool|anchor|dataset)[_-][A-Za-z0-9]",
    re.IGNORECASE,
)
# Any run of 8+ hex characters is an id or digest, whatever surrounds it.
_HEX_OR_UUID = re.compile(r"[0-9a-f]{8,}|[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)
_DOTTED_TOOL = re.compile(_ASCII_START + r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*" + _ASCII_END)
_SNAKE_CODE = re.compile(_ASCII_START + r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+" + _ASCII_END)
# Approval words. Never offered when the gate needs a human decision: the
# user has to read the evidence, not accept a placeholder with one key; an
# absent suggestion is always safe.
_APPROVAL_WORD = re.compile(r"确认|同意|批准|通过(?!率)|继续|执行|confirm|approve|proceed|continue|lgtm", re.IGNORECASE)
_MARKUP = re.compile(r"[`<>{}\[\]$|\\]|\*\*|https?://|/\w+/", re.IGNORECASE)
_WRAPPERS = "\"'“”‘’「」『』《》()（）*_ "
_REPLY_LINE = re.compile(
    r"^\s*(?:[-*>]\s*)?(?:\*\*)?(?:建议回复|推荐回复|建议用户回复|suggested\s+reply)(?:\*\*)?\s*[:：]\s*(?P<text>.*)$",
    re.IGNORECASE,
)


def sanitize_suggested_reply(
    value, *, user_message: str = "", human_decision_required: bool = False
) -> str | None:
    """Return a safe suggestion or ``None``; never raises."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.replace("\r", "\n").split("\n")[0].split())
    text = text.strip(_WRAPPERS).strip()
    if not text or len(text) > MAX_SUGGESTED_REPLY_CHARS:
        return None
    if _METRIC_NUMBER.search(text) or _METRIC_CHINESE_NUMBER.search(text) or _MEASURED_NUMBER.search(text):
        return None
    if human_decision_required and _APPROVAL_WORD.search(text):
        return None
    if _MARKUP.search(text) or _INTERNAL_ID.search(text) or _HEX_OR_UUID.search(text):
        return None
    typed = str(user_message or "")
    for pattern in (_SNAKE_CODE, _DOTTED_TOOL):
        if any(match.group(0) not in typed for match in pattern.finditer(text)):
            return None
    return text


def split_suggested_reply(content: str) -> tuple[str, str | None]:
    """Remove a trailing ``建议回复：…`` line from model prose.

    Returns the prose without that line and the raw (unsanitised) suggestion.
    Only the last non-empty line is considered, so a quoted phrase in the
    body is never mistaken for the suggestion.
    """
    if not isinstance(content, str) or not content.strip():
        return content, None
    lines = content.rstrip().split("\n")
    match = _REPLY_LINE.match(lines[-1])
    if match is None:
        return content, None
    return "\n".join(lines[:-1]).rstrip(), match.group("text")


def attach_suggested_reply(
    metadata: dict, value, *, user_message: str = "", human_decision_required: bool | None = None
) -> dict:
    """Set ``suggested_reply`` only when the sanitised value is usable.

    ``human_decision_required`` defaults to the message's own gate policy
    (``human_decision_gate == "required"``).
    """
    if human_decision_required is None:
        human_decision_required = str(metadata.get("human_decision_gate") or "").strip().lower() == "required"
    safe = sanitize_suggested_reply(
        value, user_message=user_message, human_decision_required=human_decision_required
    )
    if safe is None:
        metadata.pop(METADATA_KEY, None)
    else:
        metadata[METADATA_KEY] = safe
    return metadata
