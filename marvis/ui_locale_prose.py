"""English display of platform-templated Agent prose (D-10 follow-up).

Platform code writes many Agent messages from fixed Chinese templates
(``**特征筛选完成**：从 {n} 个候选中…``). They are stored in Chinese because
memory, auto drive and workflow insights read them again, so English is applied
only when an English request is served.

A line is translated only when **all** of it is accounted for:

- by templates from ``marvis.ui_locale_messages_en`` (the literal Chinese
  pieces of the platform's f-strings, ``{0}`` marking an interpolated value),
  or exact labels from ``marvis.ui_locale_en``;
- or by runs without Chinese ideographs (numbers, field names, punctuation),
  which pass through unchanged.

Any Chinese that no template explains (LLM prose, user text, a template that
has no English yet) leaves the whole line in Chinese. There is no substring
replacement inside text the platform did not author. Captured values pass
through raw. Only a slot the template marks ``{name|t}`` (a status word or a
step title interpolated into a sentence, mirroring the frontend dictionary)
translates its value, and only when that value is itself exactly platform
text. A field name in an unmarked slot is never translated, even when it
happens to equal a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from marvis.ui_locale_en import EN_CELL_LABELS, EN_LABELS
from marvis.ui_locale_messages_en import EN_MESSAGE_TEMPLATES

_IDEOGRAPH = "㐀-䶿一-鿿豈-﫿"
_CJK_RE = re.compile(f"[{_IDEOGRAPH}]")
_PLACEHOLDER_RE = re.compile(r"\{(\w+)(?:\|t)?\}")
_KEY_SLOT_RE = re.compile(r"\{(\w+)(\|t)?\}")
_STRICT_VALUE = f"[^{_IDEOGRAPH}\\n]*?"
_LENIENT_VALUE = ".*?"
_MAX_LINE = 2000
# A Chinese full stop or comma inside an English sentence reads as a typo.
_PUNCTUATION = (
    ("：", ": "), ("；", "; "), ("，", ", "), ("、", ", "), ("。", ". "),
    ("！", "! "), ("？", "? "), ("（", " ("), ("）", ") "), ("「", "“"), ("」", "”"),
    ("『", "“"), ("』", "”"), ("【", "["), ("】", "] "), ("《", "“"), ("》", "”"),
    ("～", "~"), ("　", " "),
)


_BRACKETS = {
    opening: pair for pair in ("（）", "()", "〔〕", "「」", "【】", "“”", "[]", "《》")
    for opening in pair
}
_OPENING = frozenset(pair[0] for pair in set(_BRACKETS.values()))
_CLOSING = frozenset(pair[1] for pair in set(_BRACKETS.values()))


@dataclass(frozen=True)
class _Template:
    key: str
    english: str
    names: tuple[str, ...]
    strict: re.Pattern
    lenient: re.Pattern
    trailing_value: bool
    open_values: tuple[int, ...]
    nested: tuple[bool, ...]
    raw_open: tuple[bool, ...]


def _compile(key: str, english: str) -> _Template | None:
    parts = _KEY_SLOT_RE.split(key)
    literals, names = parts[0::3], tuple(parts[1::3])
    nested = tuple(bool(flag) for flag in parts[2::3])
    if not _CJK_RE.search("".join(literals)):
        if not names or not all(nested):
            return None
        # A pure layout template ("**{0|t}**:{1|t}"): every value must be platform text.
        open_values = tuple(range(len(names)))
    else:
        open_values = tuple(
            index for index in range(len(names))
            if (index == 0 and not literals[0]) or (index == len(names) - 1 and not literals[-1])
        )

    def pattern(value: str) -> re.Pattern:
        # A value enclosed by literals may hold raw Chinese (a business name in
        # "{0}（含义:{1}）"). An open-ended first or last value may hold Chinese
        # only when that Chinese is itself platform text (checked in _segment),
        # or it would swallow neighbouring text that no template explains.
        out = []
        for index, literal in enumerate(literals):
            out.append(r"\s*".join(re.escape(chunk) for chunk in re.split(r"\s+", literal)))
            if index < len(names):
                trailing = index == len(names) - 1 and not literals[-1]
                # An optional fragment ("{8}{9}", "`{1}`{2}，") may be empty; a value
                # set off by spaces ("{0} 个") or opening the template may not.
                after = literals[index + 1]
                optional = (
                    not (index == 0 and not literal)
                    and not (literal and literal[-1].isspace())
                    and not (after and after[0].isspace())
                )
                if trailing:
                    body = "." if value == _LENIENT_VALUE else f"[^{_IDEOGRAPH}\\n]"
                    out.append(f"({body}{'*' if optional else '+'})")
                else:
                    # A value never spans its own delimiters: "（{2}）" stops at "）".
                    opened = literal[-1:] if literal[-1:] in _OPENING else ""
                    closed = after[:1] if after[:1] in _CLOSING else ""
                    pair = _BRACKETS.get(opened) or _BRACKETS.get(closed)
                    fences = re.escape("".join(sorted(pair))) if pair else ""
                    body = (f"[^{_IDEOGRAPH}{fences}\\n]" if value == _STRICT_VALUE
                            else f"[^{fences}\\n]" if fences else ".")
                    out.append(f"({body}{'*?' if optional else '+?'})")
        return re.compile("".join(out), re.DOTALL)

    return _Template(
        key=key,
        english=english,
        names=names,
        strict=pattern(_STRICT_VALUE),
        lenient=pattern(_LENIENT_VALUE),
        trailing_value=bool(names) and not literals[-1],
        open_values=open_values,
        nested=nested,
        # A raw name may still sit in an open slot where the template says so:
        # after a list colon ("生成 {0} 个字段：{1}") or before its annotation
        # ("{0}（含义:{1}）").
        raw_open=tuple(
            (index == len(names) - 1 and not literals[-1] and literals[index][-1:] in "：:")
            or (index == 0 and not literals[0] and literals[1][:1] in "（(")
            for index in range(len(names))
        ),
    )


def _first_char(key: str) -> str:
    stripped = _PLACEHOLDER_RE.sub("\0", key).lstrip()
    return "" if not stripped or stripped[0] == "\0" else stripped[0]


# A fixed phrase this short may only translate a whole line or a whole value:
# chained word by word it could "explain" a Chinese sentence nobody templated.
_MIN_SEGMENT_IDEOGRAPHS = 4


@lru_cache(maxsize=1)
def _exact() -> dict[str, str]:
    entries: dict[str, str] = {}
    for table in (EN_CELL_LABELS, EN_LABELS, EN_MESSAGE_TEMPLATES):
        for key, english in table.items():
            key = key.strip()
            if key and isinstance(english, str) and not _PLACEHOLDER_RE.search(key):
                entries[key] = english
    return entries


@lru_cache(maxsize=1)
def _index() -> tuple[dict[str, list[_Template]], list[_Template]]:
    by_first: dict[str, list[_Template]] = {}
    leading_value: list[_Template] = []
    entries: dict[str, str] = {}
    for table in (EN_CELL_LABELS, EN_LABELS, EN_MESSAGE_TEMPLATES):
        for key, english in table.items():
            key = key.strip()
            if not key or not isinstance(english, str):
                continue
            fixed = not _PLACEHOLDER_RE.search(key)
            # A fragment that opens with punctuation ("（1:1 保持 ✓）") only ever
            # continues a platform sentence, so it may be a segment too.
            if fixed and (table is not EN_MESSAGE_TEMPLATES or (
                    len(_CJK_RE.findall(key)) < _MIN_SEGMENT_IDEOGRAPHS and key[0] not in "（，。；：、(")):
                continue
            entries[key] = english
    for key, english in entries.items():
        template = _compile(key, english)
        if template is None:
            continue
        first = _first_char(key)
        (by_first.setdefault(first, []) if first else leading_value).append(template)
    for bucket in (*by_first.values(), leading_value):
        # Longest literal first: the most specific template wins.
        bucket.sort(key=lambda item: len(_PLACEHOLDER_RE.sub("", item.key)), reverse=True)
    return by_first, leading_value


def _join(left: str, right: str) -> str:
    if left and right and right[0].isalnum() and (left[-1].isalnum() or left[-1] in ".;!?"):
        return f"{left} {right}"
    return left + right


def _polish(text: str) -> str:
    for source, target in _PUNCTUATION:
        text = text.replace(source, target)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"([;:])(?=[A-Za-z(*“])", r"\1 ", text)
    return text


# Platform lists inside a value: step chains, enumerations.
_LIST_SEPARATORS = ((" → ", " → "), ("→", " → "), ("、", ", "), ("，", ", "), ("；", "; "), (" / ", " / "))


def _platform_value(text: str, depth: int) -> str | None:
    """English for a value that is itself platform text (or a list of it)."""

    stripped = text.strip()
    if not _CJK_RE.search(stripped):
        return text
    if depth > 3:
        return None
    translated = None
    for separator, joiner in _LIST_SEPARATORS[:3]:
        # A platform list reads item by item before any sentence template.
        if separator in stripped:
            parts = [_translate(part.strip(), depth + 1) for part in stripped.split(separator) if part.strip()]
            if parts and all(part is not None for part in parts):
                translated = joiner.join(parts)
                break
    if translated is None:
        translated = _translate(stripped, depth + 1)
    if translated is None:
        for separator, joiner in _LIST_SEPARATORS:
            if separator not in stripped:
                continue
            parts = [_translate(part.strip(), depth + 1) if part.strip() else "" for part in stripped.split(separator)]
            if all(part is not None for part in parts):
                translated = joiner.join(parts)
                break
    if translated is None:
        return None
    lead = text[: len(text) - len(text.lstrip())]
    tail = text[len(text.rstrip()):]
    return f"{lead}{translated}{tail}"


def _value(text: str, depth: int) -> str:
    """A captured value stays raw unless it is itself platform text."""

    translated = _platform_value(text, depth)
    return text if translated is None else translated


def _render(template: _Template, captures: tuple[str, ...], depth: int) -> str:
    values = dict(zip(template.names, captures, strict=False))
    nested = dict(zip(template.names, template.nested, strict=False))

    def fill(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        return _value(values[name], depth) if nested.get(name) else values[name]

    return _PLACEHOLDER_RE.sub(fill, template.english)


def _open_value_ok(template: _Template, index: int, value: str, depth: int) -> bool:
    if template.nested[index] and _platform_value(value, depth) is not None:
        return True
    # A raw name list, never a sentence.
    return template.raw_open[index] and not re.search(r"[。；！？]", value)


def _segment(line: str, lenient: bool, depth: int) -> str | None:
    by_first, leading_value = _index()
    length = len(line)
    memo: dict[int, str | None] = {}

    def at(position: int) -> str | None:
        if position == length:
            return ""
        if position in memo:
            return memo[position]
        memo[position] = None
        candidates = by_first.get(line[position], []) + leading_value
        for template in candidates:
            regex = template.lenient if lenient else template.strict
            match = regex.match(line, position)
            if match is None or match.end() == position:
                continue
            ends = [match.end()]
            if template.trailing_value:
                # The last value may stop earlier, where the next template starts.
                ends = list(range(match.end(), match.start(len(template.names)) - 1, -1))
            for end in ends:
                captures = match.groups()
                if end != match.end():
                    captures = (*captures[:-1], line[match.start(len(template.names)):end])
                layout = len(template.open_values) == len(template.names) and not _CJK_RE.search(
                    _PLACEHOLDER_RE.sub("", template.key))
                if layout and not any(_CJK_RE.search(value) for value in captures):
                    continue
                # An open-ended value may hold Chinese only in a {x|t} slot and
                # only when that Chinese is platform text.
                if (lenient or layout) and any(
                    _CJK_RE.search(captures[index]) and not _open_value_ok(template, index, captures[index], depth)
                    for index in template.open_values
                ):
                    continue
                rest = at(end)
                if rest is not None:
                    memo[position] = _join(_render(template, captures, depth), rest)
                    return memo[position]
        # A run without ideographs passes through as it is.
        stop = position
        while stop < length and not _CJK_RE.match(line[stop]):
            stop += 1
        for end in range(stop, position, -1):
            rest = at(end)
            if rest is not None:
                memo[position] = _join(line[position:end], rest)
                return memo[position]
        return None

    return at(0)


@lru_cache(maxsize=8192)
def _translate(line: str, depth: int = 0) -> str | None:
    if not _CJK_RE.search(line):
        return line
    exact = _exact().get(line)
    if exact is not None:
        return exact
    if len(line) > _MAX_LINE:
        return None
    for lenient in (False, True):
        translated = _segment(line, lenient, depth)
        if translated is not None:
            return translated
    return None


def translate_prose_line(line: str) -> str | None:
    """English for one fully templated line, or ``None``."""

    stripped = line.strip()
    if not stripped:
        return line
    translated = _translate(stripped)
    if translated is None:
        return None
    lead = line[: len(line) - len(line.lstrip())]
    return lead + _polish(translated).strip()


def translate_prose(text: Any) -> Any:
    """Translate each fully templated line of ``text``; other lines stay as written."""

    if not isinstance(text, str) or not _CJK_RE.search(text):
        return text
    lines = text.split("\n")
    out = []
    for line in lines:
        translated = translate_prose_line(line) if _CJK_RE.search(line) else line
        out.append(line if translated is None else translated)
    return "\n".join(out)


__all__ = ["translate_prose", "translate_prose_line"]
