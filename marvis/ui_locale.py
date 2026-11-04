"""Request-scoped UI locale and Agent reply language (D-10, batch 1b).

Two independent languages travel with a request:

- the **UI locale** (``zh`` | ``en``) chosen in the sidebar. It decides the
  language of platform labels: card titles, table headers, gate labels, step
  titles and status names;
- the **reply language**: the language of what the user last typed. It decides
  the language of Agent prose, both LLM replies and platform-templated text that
  speaks as the Agent. It does not follow the UI toggle.

Stored messages, plans and gate payloads stay canonical (Chinese), because
platform code parses them again (memory bridge, auto drive, workflow insights).
English is applied only to the HTTP response of an English request, and only to
structured label fields, by exact lookup in ``marvis.ui_locale_en``.
Deterministic values, field names and user data are never rewritten.

See docs/superpowers/specs/2026-10-28-ui-language-switch.md.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs

from marvis.ui_locale_en import EN_CELL_LABELS, EN_LABELS
from marvis.ui_locale_prose import translate_prose

logger = logging.getLogger("marvis.ui_locale")

UI_LOCALE_HEADER = "x-marvis-ui-locale"
REPLY_LANGUAGE_HEADER = "x-marvis-reply-language"
UI_CANONICAL_HEADER = "x-marvis-ui-canonical"
# "quick-reply": the body text is a platform command chip, not the user's own
# words, so it says nothing about the reply language; the hint stays.
REPLY_SOURCE_HEADER = "x-marvis-reply-source"
UI_LOCALES = ("zh", "en")

UI_LOCALE: ContextVar[str] = ContextVar("marvis_ui_locale", default="zh")
REPLY_LANGUAGE: ContextVar[str | None] = ContextVar("marvis_reply_language", default=None)

_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_MISSING_LOGGED: set[str] = set()
_MISSING_LOG_LIMIT = 2000


def normalize_locale(value: Any) -> str:
    text = value.decode("latin-1") if isinstance(value, bytes) else str(value or "")
    return "en" if re.match(r"^\s*en\b", text, re.IGNORECASE) else "zh"


def normalize_reply_language(value: Any) -> str | None:
    text = value.decode("latin-1") if isinstance(value, bytes) else str(value or "")
    text = text.strip().lower()
    if text.startswith("en"):
        return "en"
    if text.startswith("zh"):
        return "zh"
    return None


def detect_text_language(text: Any) -> str | None:
    """``zh`` for any CJK ideograph, ``en`` for Latin letters without CJK."""

    value = str(text or "")
    if _CJK_RE.search(value):
        return "zh"
    if _LATIN_RE.search(value):
        return "en"
    return None


def current_ui_locale() -> str:
    return UI_LOCALE.get()


def current_reply_language() -> str | None:
    return REPLY_LANGUAGE.get()


def agent_text_locale() -> str:
    """Locale for platform-templated text that speaks as the Agent."""

    return REPLY_LANGUAGE.get() or UI_LOCALE.get()


@contextmanager
def use_ui_locale(locale: Any) -> Iterator[str]:
    token = UI_LOCALE.set(normalize_locale(locale))
    try:
        yield UI_LOCALE.get()
    finally:
        UI_LOCALE.reset(token)


@contextmanager
def use_reply_language(language: Any) -> Iterator[str | None]:
    token = REPLY_LANGUAGE.set(normalize_reply_language(language))
    try:
        yield REPLY_LANGUAGE.get()
    finally:
        REPLY_LANGUAGE.reset(token)


def _record_missing(text: str) -> None:
    if text in _MISSING_LOGGED or len(_MISSING_LOGGED) >= _MISSING_LOG_LIMIT:
        return
    _MISSING_LOGGED.add(text)
    logger.debug("ui label has no English entry: %s", text)


def ui_label(text: Any, *, locale: str | None = None) -> Any:
    """Exact-match English label in the English UI locale; Chinese otherwise.

    A miss returns the Chinese source and is logged once. Never raises.
    """

    if not isinstance(text, str):
        return text
    if (locale or UI_LOCALE.get()) != "en" or not _CJK_RE.search(text):
        return text
    key = text.strip()
    translated = EN_LABELS.get(key)
    if translated is None:
        _record_missing(key)
        return text
    return text.replace(key, translated, 1)


def _cell_label(value: Any) -> Any:
    if isinstance(value, str):
        return EN_CELL_LABELS.get(value.strip(), value)
    return value


# A curated enum word is still user data when it occurs in a field/name
# column. Translate cells only where the table declares platform semantics.
def _load_raw_columns() -> frozenset[str]:
    """The browser's raw-header list (static/js/raw-columns.js), one source for both."""

    source = (Path(__file__).parent / "static" / "js" / "raw-columns.js").read_text(encoding="utf-8")
    match = re.search(r"/\* raw-columns:start \*/(.*?)/\* raw-columns:end \*/", source, re.DOTALL)
    if match is None:
        raise RuntimeError("raw-columns.js lost its raw-columns markers")
    return frozenset(str(label).strip().casefold() for label in json.loads(match.group(1)))


_RAW_CELL_COLUMNS = _load_raw_columns()
_PLATFORM_CELL_COLUMNS = frozenset(label.casefold() for label in {
    "状态", "建议", "Agent建议", "Agent 建议", "判断", "结论", "风险等级", "严重度", "等级",
    "是否通过", "单调性", "可用性", "算法", "模型类型", "目标类型", "搜索阶段",
    "status", "suggestion", "agent suggestion", "judgment", "decision", "conclusion", "risk level",
    "severity", "level", "passed", "monotonicity", "availability", "algorithm", "model type",
    "target type", "search stage", "split",
})
# 数据集/样本集/dataset columns usually hold user dataset names; their cells
# translate only when the table declares a platform kind (split-badge, enum).
_PLATFORM_CELL_KINDS = frozenset({"enum", "status", "tag", "platform-label", "split-badge"})


def _localize_table_rows(rows: list[Any], table: Mapping) -> list[Any]:
    columns = table.get("columns", table.get("headers", []))
    specs = table.get("column_specs", [])
    columns = columns if isinstance(columns, list) else []
    specs = specs if isinstance(specs, list) else []
    translated_indexes = set()
    for index in range(max(len(columns), len(specs))):
        label = str(columns[index]).strip().casefold() if index < len(columns) else ""
        if label in _RAW_CELL_COLUMNS:
            continue
        spec = specs[index] if index < len(specs) and isinstance(specs[index], dict) else {}
        if label in _PLATFORM_CELL_COLUMNS or spec.get("kind") in _PLATFORM_CELL_KINDS:
            translated_indexes.add(index)
    return [
        [_cell_label(cell) if index in translated_indexes else cell for index, cell in enumerate(row)]
        if isinstance(row, list) else row
        for row in rows
    ]


# --------------------------------------------------------------- presentation
_LABEL_KEYS = frozenset({
    "label", "title", "prompt", "hint", "reply_hint", "button_label", "confirm_label",
    "cancel_label", "status_label", "stage_label", "step_title", "section_title",
})
# Dicts whose every string value is a platform label (gate_payloads._policy_signals).
_LABEL_MAP_KEYS = frozenset({"policy_signals"})
# Subtrees that are fingerprints, user data or deterministic values.
_SKIP_KEYS = frozenset({
    "confirmation_snapshot", "draft_values", "content", "command", "names", "inputs",
    "values", "params", "parameters", "fields", "columns_meta", "dataset", "data",
    "metrics", "evidence", "output", "raw", "memory_references", "sample_keys",
    "sample_conflicts", "rows_preview",
})


def _localize_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return value
    if isinstance(value, list):
        return [_localize_metadata(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SKIP_KEYS:
            out[key] = item
        elif key == "rows" and isinstance(item, list):
            out[key] = _localize_table_rows(item, value)
        elif key == "columns" and isinstance(item, list):
            out[key] = [ui_label(column) if isinstance(column, str) else column for column in item]
        elif isinstance(item, str) and (key in _LABEL_KEYS or key.endswith("_label")):
            out[key] = ui_label(item)
        elif key in _LABEL_MAP_KEYS and isinstance(item, dict):
            out[key] = {name: ui_label(text) for name, text in item.items()}
        else:
            out[key] = _localize_metadata(item, depth + 1)
    return out


def _is_message(value: Mapping) -> bool:
    return "role" in value and "metadata" in value and isinstance(value.get("metadata"), dict)


def _is_plan_step(value: Mapping) -> bool:
    return "tool_ref" in value and isinstance(value.get("title"), str)


def localize_payload(payload: Any, *, locale: str | None = None, depth: int = 0) -> Any:
    """Return ``payload`` with platform labels in English for an ``en`` request.

    Only structured label fields change: message ``metadata`` labels and table
    headers (cells only for curated status words), and plan step titles.
    ``confirmation_snapshot``, message content and every value are untouched.
    """

    if (locale or UI_LOCALE.get()) != "en" or depth > 16:
        return payload
    if UI_LOCALE.get() != "en":
        with use_ui_locale("en"):
            return localize_payload(payload, locale="en", depth=depth)
    if isinstance(payload, list):
        return [localize_payload(item, locale="en", depth=depth + 1) for item in payload]
    if not isinstance(payload, dict):
        return payload
    if _is_message(payload):
        out = dict(payload)
        out["metadata"] = _localize_metadata(payload["metadata"])
        return localize_message_prose(out)
    out = {}
    for key, item in payload.items():
        if key == "confirmation_snapshot":
            out[key] = item
        elif key in {"title", "phase"} and _is_plan_step(payload):
            out[key] = ui_label(item, locale="en")
        elif key in {"status_label", "stage_label"} and isinstance(item, str):
            out[key] = ui_label(item, locale="en")
        else:
            out[key] = localize_payload(item, locale="en", depth=depth + 1)
    return out


# ------------------------------------------------------- platform prose
# Platform-templated prose of an Agent message: the message body and the text
# fields of the structured failure / repair cards. ``ui_locale_prose`` changes
# a line only when platform templates explain all of its Chinese, so LLM prose
# and user text pass through. ``command`` and other machine fields never change.
_PROSE_CONTAINERS = frozenset({
    "error_diagnostic", "sample_design_diagnostic", "repair_report", "repair_undo",
})
_PROSE_FIELDS = frozenset({
    "title", "summary", "cause", "location", "impact", "actions", "finding", "action",
    "verification", "log", "label", "consequence", "value", "note", "description",
    "message", "changed", "undo", "text_note", "numeric_note",
})
_PROSE_SKIP = frozenset({
    "command", "undo_command", "technical_detail", "fingerprint", "before", "after",
    "params", "code", "schema_version", "repair_option_id", "option_id",
})


def _localize_prose_tree(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return value
    if isinstance(value, list):
        return [_localize_prose_tree(item, depth + 1) for item in value]
    if not isinstance(value, dict):
        return value
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in _PROSE_SKIP:
            out[key] = item
        elif key in _PROSE_FIELDS and isinstance(item, str):
            out[key] = translate_prose(item)
        elif key in _PROSE_FIELDS and isinstance(item, list):
            out[key] = [translate_prose(entry) if isinstance(entry, str) else _localize_prose_tree(entry, depth + 1)
                        for entry in item]
        else:
            out[key] = _localize_prose_tree(item, depth + 1)
    return out


def localize_message_prose(message: Mapping) -> Any:
    """English platform prose for one served Agent message; user turns untouched."""

    if not isinstance(message, Mapping) or message.get("role") == "user":
        return message
    out = dict(message)
    if isinstance(out.get("content"), str):
        out["content"] = translate_prose(out["content"])
    metadata = out.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
        for key in _PROSE_CONTAINERS & metadata.keys():
            metadata[key] = _localize_prose_tree(metadata[key])
        if isinstance(metadata.get("reply_hint"), str):
            metadata["reply_hint"] = translate_prose(metadata["reply_hint"])
        out["metadata"] = metadata
    return out


def localize_prose_payload(payload: Any, depth: int = 0) -> Any:
    """Apply :func:`localize_message_prose` to every message in ``payload``.

    Used for canonical-label responses: the browser keeps Chinese labels for
    its structured logic and translates them in the DOM, but Agent prose is
    rendered as Markdown that the DOM translator leaves alone.
    """

    if depth > 16:
        return payload
    if isinstance(payload, list):
        return [localize_prose_payload(item, depth + 1) for item in payload]
    if not isinstance(payload, dict):
        return payload
    if _is_message(payload):
        return localize_message_prose(payload)
    return {
        key: item if key == "confirmation_snapshot" else localize_prose_payload(item, depth + 1)
        for key, item in payload.items()
    }


# ------------------------------------------------------------- reply language
# Prompts whose output a person reads in the conversation. Report writers keep
# their language (out of scope), memory distillation stays canonical, and plan
# or code generators produce machine contracts.
CONVERSATIONAL_PROMPTS = frozenset({
    "AGENT_SYSTEM_PROMPT",
    "WORKFLOW_INSIGHT_SYS",
    "CRITIC_SYS",
    "GATE_SYSTEM_TEMPLATE",
    "GATE_INSTRUCTION_ROUTER_SYS",
    "GATE_SEMANTIC_AUTHORIZATION_REVIEW_SYS",
    "TOP_LEVEL_INTENT_ROUTER_SYS",
    "TOP_LEVEL_INTENT_REVIEW_SYS",
    "TOP_LEVEL_INTENT_REPAIR_SYS",
    "STRATEGY_REQUEST_COMPILER_SYS",
    "SAMPLE_DESIGN_V2_CORRECTION_SYS",
    "SQL_SELECT_COMPILER_SYS",
    "SLICE_SPEC_SYS",
})

_LANGUAGE_NAMES = {"zh": "Simplified Chinese", "en": "English"}


def reply_language_instruction(language: str | None) -> str:
    """The rule appended to a conversational system prompt, or ``""``."""

    name = _LANGUAGE_NAMES.get(language or "")
    if not name:
        return ""
    return (
        "\n\nReply language: the user's latest message is in "
        f"{name}. Write every user-facing natural-language field (replies, summaries, "
        f"reasons, findings, risks, recommendations, clarifying questions) in {name}. "
        "This overrides any earlier instruction about the output language. Keep JSON keys, "
        "enum codes, tool names, field and column names, file names and every metric value "
        "exactly as supplied."
    )


def conversational_prompt_name(system_prompt: str, prompt_name: str | None) -> str | None:
    """Name of the registered conversational prompt behind ``system_prompt``."""

    if prompt_name:
        return prompt_name if prompt_name in CONVERSATIONAL_PROMPTS else None
    from marvis.llm_prompts import ALL_PROMPTS, GATE_SYSTEM_TEMPLATE

    # Report prompts extend the chat prompt's text, so the longest registered
    # prefix decides which prompt this is.
    text = str(system_prompt or "")
    best_name, best_length = None, 0
    for spec in ALL_PROMPTS:
        prefix = spec.text.split("{allowed_actions}")[0] if spec == GATE_SYSTEM_TEMPLATE else spec.text
        if prefix and text.startswith(prefix) and len(prefix) > best_length:
            best_name, best_length = spec.name, len(prefix)
    return best_name if best_name in CONVERSATIONAL_PROMPTS else None


def with_reply_language(system_prompt: str, prompt_name: str | None = None) -> str:
    """Append the reply-language rule when the prompt is conversational and the
    language of the user's latest message is known."""

    language = REPLY_LANGUAGE.get()
    if not language or conversational_prompt_name(system_prompt, prompt_name) is None:
        return system_prompt
    return system_prompt + reply_language_instruction(language)


# ------------------------------------------------------------------ middleware
_BODY_TEXT_FIELDS = ("content", "goal", "message", "user_message", "instruction")
_MAX_SNIFF_BYTES = 1024 * 1024
_MAX_LOCALIZE_BYTES = 32 * 1024 * 1024


def _user_text(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for field in _BODY_TEXT_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


class UiLocaleMiddleware:
    """Pure ASGI middleware, registered outermost.

    - Sets ``UI_LOCALE`` from ``X-MARVIS-UI-Locale``. ``REPLY_LANGUAGE`` is the
      language of the user text in a small JSON ``POST`` body; the
      ``X-MARVIS-Reply-Language`` hint (the open task's last typed language) is
      used only when the body has none, or when ``X-MARVIS-Reply-Source:
      quick-reply`` marks the body as a platform command. Being pure ASGI and
      outermost, both values reach sync endpoints and ``BackgroundTasks``.
    - In the English UI locale, rewrites JSON ``/api/`` responses through
      ``localize_payload``. Chinese requests pass through byte for byte.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {name.lower(): value for name, value in scope.get("headers") or []}
        locale_value = headers.get(UI_LOCALE_HEADER.encode(), b"")
        # An iframe navigation cannot attach the API's custom locale header.
        # Only the platform report-preview page accepts its explicit locale
        # query; this does not change the language of the report artifact.
        if not locale_value and str(scope.get("path") or "").endswith("/report/preview"):
            query = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="replace"))
            locale_value = query.get("ui_locale", [""])[0]
        locale = normalize_locale(locale_value)
        hint = normalize_reply_language(headers.get(REPLY_LANGUAGE_HEADER.encode(), b""))
        ui_token = UI_LOCALE.set(locale)
        reply_token = REPLY_LANGUAGE.set(hint)
        try:
            receive = await self._sniff_reply_language(scope, headers, receive, hint)
            canonical = headers.get(UI_CANONICAL_HEADER.encode(), b"").strip() == b"1"
            if locale == "en" and str(scope.get("path") or "").startswith("/api/"):
                # Canonical clients keep Chinese labels; their Agent prose is
                # still served in English (it bypasses the DOM translator).
                send = self._localizing_send(send, prose_only=canonical)
            await self.app(scope, receive, send)
        finally:
            REPLY_LANGUAGE.reset(reply_token)
            UI_LOCALE.reset(ui_token)

    async def _sniff_reply_language(self, scope, headers, receive, hint):
        # What the user just wrote wins; the header hint only fills in when
        # the body carries no detectable language (or is a command chip).
        if scope.get("method") != "POST":
            return receive
        if headers.get(REPLY_SOURCE_HEADER.encode(), b"").strip().lower() == b"quick-reply":
            return receive
        content_type = headers.get(b"content-type", b"").decode("latin-1").lower()
        try:
            length = int(headers.get(b"content-length", b"0") or 0)
        except ValueError:
            length = 0
        if "application/json" not in content_type or not 0 < length <= _MAX_SNIFF_BYTES:
            return receive
        messages = []
        body = b""
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") != "http.request":
                break
            body += message.get("body", b"")
            if not message.get("more_body") or len(body) > _MAX_SNIFF_BYTES:
                break
        language = detect_text_language(_user_text(body))
        if language:
            REPLY_LANGUAGE.set(language)
        pending = list(messages)

        async def replay():
            if pending:
                return pending.pop(0)
            return await receive()

        return replay

    @staticmethod
    def _localizing_send(send, *, prose_only: bool = False):
        state: dict[str, Any] = {"start": None, "chunks": [], "size": 0, "passthrough": False}

        async def wrapped(message):
            kind = message.get("type")
            if kind == "http.response.start":
                headers = {name.lower(): value for name, value in message.get("headers") or []}
                content_type = headers.get(b"content-type", b"").decode("latin-1").lower()
                if "application/json" not in content_type or b"content-encoding" in headers:
                    state["passthrough"] = True
                    await send(message)
                    return
                state["start"] = message
                return
            if kind != "http.response.body" or state["passthrough"]:
                await send(message)
                return
            state["chunks"].append(message.get("body", b""))
            state["size"] += len(state["chunks"][-1])
            if message.get("more_body") and state["size"] <= _MAX_LOCALIZE_BYTES:
                return
            body = b"".join(state["chunks"])
            start = dict(state["start"])
            if not message.get("more_body"):
                # Complete body: localize it and fix the length to match.
                body = _localized_body(body, prose_only=prose_only)
                start["headers"] = [
                    (name, str(len(body)).encode() if name.lower() == b"content-length" else value)
                    for name, value in start.get("headers") or []
                ]
            await send(start)
            await send({"type": "http.response.body", "body": body, "more_body": bool(message.get("more_body"))})
            state["passthrough"] = True

        return wrapped


def _localized_body(body: bytes, *, prose_only: bool = False) -> bytes:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body
    # Nothing Chinese (raw or \u-escaped): nothing to localize, skip the parse.
    if not _CJK_RE.search(text) and "\\u" not in text:
        return body
    try:
        payload = json.loads(text)
    except ValueError:
        return body
    if prose_only:
        with use_ui_locale("en"):
            localized = localize_prose_payload(payload)
    else:
        localized = localize_payload(payload, locale="en")
    if localized == payload:
        return body
    try:
        return json.dumps(localized, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        return body


__all__ = [
    "CONVERSATIONAL_PROMPTS",
    "REPLY_LANGUAGE",
    "UI_LOCALE",
    "UiLocaleMiddleware",
    "agent_text_locale",
    "conversational_prompt_name",
    "current_reply_language",
    "current_ui_locale",
    "detect_text_language",
    "localize_message_prose",
    "localize_payload",
    "localize_prose_payload",
    "normalize_locale",
    "normalize_reply_language",
    "reply_language_instruction",
    "ui_label",
    "use_reply_language",
    "use_ui_locale",
    "with_reply_language",
]
