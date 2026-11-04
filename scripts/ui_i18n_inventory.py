"""Inventory platform Chinese UI strings and report the ones without English.

Dev check for the UI language switch (docs/superpowers/specs/2026-10-28-ui-language-switch.md).
It scans ``marvis/static/index.html`` (text nodes, text attributes, inline scripts)
and every string/template literal in ``marvis/static/**/*.js``. Template
expressions become ``{0}``, ``{1}`` … and HTML tags split a literal into the
text segments the browser renders as separate text nodes.

    python scripts/ui_i18n_inventory.py                # summary of untranslated keys
    python scripts/ui_i18n_inventory.py --json out.json  # full inventory
    python scripts/ui_i18n_inventory.py --missing        # list untranslated keys
"""

from __future__ import annotations

import argparse
import html.parser
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "marvis" / "static"
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
TEXT_ATTRS = {
    "placeholder", "title", "aria-label", "alt", "aria-description", "data-tooltip",
    "data-panel-title", "data-panel-description", "data-tip",
}
REGEX_PREV = set("(,=:[!&|?{};+-*%<>~^") | {""}
KEYWORDS_BEFORE_REGEX = {
    "return", "typeof", "case", "do", "else", "in", "of", "new", "delete", "void", "throw", "yield", "await",
}
TAG_RE = re.compile(r"<[^<>]*>")
ATTR_RE = re.compile(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)')""")
HTML_EXPR = re.compile(r"(Html|HTML|render|\.join\(|\.map\(|<|Markup|markup|svg|Svg|Icon\b|Badge|Button\()")
SAFE_TEXT_EXPR = re.compile(r"^(escapeHtml|esc|escapeAttr)\(")
TRANSLATE_EXPR = re.compile(
    r"(['\"`][^'\"`]*[一-鿿]|label|Label|status|Status|state|State|kind|Kind|stage|Stage|phase|Phase|"
    r"verdict|Verdict|level|Level|tone|Tone)"
)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class Inventory:
    def __init__(self) -> None:
        self.entries: dict[str, set[str]] = {}

    def add(self, key: str, where: str) -> None:
        key = norm(key)
        if key and CJK.search(key):
            self.entries.setdefault(key, set()).add(where)


class _HtmlCollector(html.parser.HTMLParser):
    def __init__(self, inventory: Inventory, where: str) -> None:
        super().__init__(convert_charrefs=True)
        self.inventory = inventory
        self.where = where
        self.stack: list[str] = []
        self.scripts: list[str] = []

    def _attrs(self, attrs) -> None:
        for name, value in attrs:
            if name in TEXT_ATTRS and value:
                self.inventory.add(value, self.where)

    def handle_starttag(self, tag, attrs):
        if tag not in {"br", "img", "input", "meta", "link", "hr", "source", "path", "circle", "rect"}:
            self.stack.append(tag)
        self._attrs(attrs)

    def handle_startendtag(self, tag, attrs):
        self._attrs(attrs)

    def handle_endtag(self, tag):
        while self.stack and self.stack.pop() != tag:
            pass

    def handle_data(self, data):
        top = self.stack[-1] if self.stack else ""
        if top == "script":
            self.scripts.append(data)
        elif top != "style":
            self.inventory.add(data, self.where)


def _read_string(src: str, i: int, quote: str) -> tuple[str, int]:
    n = len(src)
    buf: list[str] = []
    i += 1
    escapes = {"n": "\n", "t": "\t", "r": "", "\\": "\\", "'": "'", '"': '"', "`": "`", "$": "$"}
    while i < n:
        c = src[i]
        if c == "\\":
            nxt = src[i + 1] if i + 1 < n else ""
            if nxt == "u" and src[i + 2:i + 3] == "{":
                j = src.index("}", i)
                buf.append(chr(int(src[i + 3:j], 16)))
                i = j + 1
                continue
            if nxt == "u":
                buf.append(chr(int(src[i + 2:i + 6], 16)))
                i += 6
                continue
            if nxt == "x":
                buf.append(chr(int(src[i + 2:i + 4], 16)))
                i += 4
                continue
            if nxt == "\n":
                i += 2
                continue
            buf.append(escapes.get(nxt, nxt))
            i += 2
            continue
        if c == quote:
            return "".join(buf), i + 1
        buf.append(c)
        i += 1
    return "".join(buf), i


def tokenize_js(src: str) -> list[tuple[str, object]]:
    """('str', value) for quoted strings, ('tpl', parts) for template literals."""
    out: list[tuple[str, object]] = []
    n = len(src)

    def read_template(i: int) -> tuple[list[tuple[str, str]], int]:
        parts: list[tuple[str, str]] = []
        buf: list[str] = []
        i += 1
        while i < n:
            c = src[i]
            if c == "\\":
                nxt = src[i + 1] if i + 1 < n else ""
                if nxt == "u":
                    if src[i + 2:i + 3] == "{":
                        j = src.index("}", i)
                        buf.append(chr(int(src[i + 3:j], 16)))
                        i = j + 1
                        continue
                    buf.append(chr(int(src[i + 2:i + 6], 16)))
                    i += 6
                    continue
                buf.append({"n": "\n", "t": "\t"}.get(nxt, nxt))
                i += 2
                continue
            if c == "`":
                parts.append(("lit", "".join(buf)))
                return parts, i + 1
            if c == "$" and i + 1 < n and src[i + 1] == "{":
                parts.append(("lit", "".join(buf)))
                buf = []
                depth = 1
                j = start = i + 2
                while j < n and depth:
                    cj = src[j]
                    if cj in "'\"":
                        _, j = _read_string(src, j, cj)
                        continue
                    if cj == "`":
                        sub, j = read_template(j)
                        out.append(("tpl", sub))
                        continue
                    if cj == "{":
                        depth += 1
                    elif cj == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                expr = src[start:j]
                out.extend(item for item in tokenize_js(expr) if item[0] == "str")
                parts.append(("expr", expr))
                i = j + 1
                continue
            buf.append(c)
            i += 1
        parts.append(("lit", "".join(buf)))
        return parts, i

    i = 0
    last_sig = ""
    last_word = ""
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "'\"":
            value, i = _read_string(src, i, c)
            out.append(("str", value))
            last_sig, last_word = "s", ""
            continue
        if c == "`":
            parts, i = read_template(i)
            out.append(("tpl", parts))
            last_sig, last_word = "s", ""
            continue
        if c == "/" and (last_sig in REGEX_PREV or last_word in KEYWORDS_BEFORE_REGEX):
            j = i + 1
            in_class = False
            while j < n:
                cj = src[j]
                if cj == "\\":
                    j += 2
                    continue
                if cj == "[":
                    in_class = True
                elif cj == "]":
                    in_class = False
                elif (cj == "/" and not in_class) or cj == "\n":
                    break
                j += 1
            i = j + 1
            while i < n and src[i].isalpha():
                i += 1
            last_sig = "r"
            continue
        if c.isspace():
            i += 1
            continue
        if c.isalnum() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            last_word, last_sig = src[i:j], "w"
            i = j
            continue
        last_sig, last_word = c, ""
        i += 1
    return out


def _attr_values(markup: str):
    for tag in TAG_RE.findall(markup):
        for match in ATTR_RE.finditer(tag):
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if match.group(1) in TEXT_ATTRS and CJK.search(value):
                yield value


def _template_key(chunk: str) -> str:
    counter = 0

    def placeholder(match: re.Match) -> str:
        nonlocal counter
        token = "{%d|t}" % counter if match.group(1) == "T" else "{%d}" % counter
        counter += 1
        return token

    return re.sub(r"\x02([TR])\x02", placeholder, chunk)


def add_template(inventory: Inventory, parts, where: str) -> None:
    pieces = []
    for kind, value in parts:
        if kind == "lit":
            pieces.append(value)
            continue
        expr = value.strip()
        if HTML_EXPR.search(expr) and not SAFE_TEXT_EXPR.match(expr):
            pieces.append("\x01")
        else:
            pieces.append("\x02T\x02" if TRANSLATE_EXPR.search(expr) else "\x02R\x02")
    joined = "".join(pieces)
    for value in _attr_values(joined):
        inventory.add(_template_key(value), where)
    for chunk in re.split(r"<[^<>]*>|\x01", joined):
        inventory.add(_template_key(chunk), where)


def add_string(inventory: Inventory, value: str, where: str) -> None:
    if not CJK.search(value):
        return
    if TAG_RE.search(value):
        for attr in _attr_values(value):
            inventory.add(attr, where)
        for chunk in TAG_RE.split(value):
            inventory.add(chunk, where)
        return
    if "\n" in value:
        for line in value.split("\n"):
            inventory.add(line, where)
    inventory.add(value, where)


def build_inventory() -> Inventory:
    inventory = Inventory()
    collector = _HtmlCollector(inventory, "index.html")
    collector.feed((STATIC / "index.html").read_text(encoding="utf-8"))
    sources = [("index.html", script) for script in collector.scripts]
    for path in sorted(STATIC.rglob("*.js")):
        rel = str(path.relative_to(STATIC))
        if not rel.startswith("js/i18n"):
            sources.append((rel, path.read_text(encoding="utf-8")))
    for where, source in sources:
        for kind, value in tokenize_js(source):
            if kind == "str":
                add_string(inventory, value, where)
            else:
                add_template(inventory, value, where)
    return inventory


def translated_keys() -> set[str]:
    script = (
        'import { EN_UI } from "./marvis/static/js/i18n-en.js";'
        "process.stdout.write(JSON.stringify(Object.keys(EN_UI)));"
    )
    raw = subprocess.run(["node", "--input-type=module", "-e", script], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    return {norm(key) for key in json.loads(raw)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the full inventory as JSON")
    parser.add_argument("--missing", action="store_true", help="print keys without an English entry")
    args = parser.parse_args(argv)
    inventory = build_inventory()
    if args.json:
        payload = {key: sorted(files) for key, files in sorted(inventory.entries.items())}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    known = translated_keys() if (STATIC / "js" / "i18n-en.js").exists() else set()
    missing = sorted(key for key in inventory.entries if key not in known)
    if args.missing:
        for key in missing:
            print(f"{key}\t{','.join(sorted(inventory.entries[key]))}")
    print(f"inventory={len(inventory.entries)} translated={len(inventory.entries) - len(missing)} "
          f"missing={len(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
