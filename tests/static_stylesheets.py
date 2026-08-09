"""Read the stylesheets a browser loads from the MARVIS entry document."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class _StylesheetLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "link":
            return
        attributes = {name: value for name, value in attrs}
        relations = str(attributes.get("rel") or "").split()
        href = attributes.get("href")
        if "stylesheet" in relations and isinstance(href, str):
            self.hrefs.append(href)


def browser_stylesheet_hrefs(static_dir: Path) -> tuple[str, ...]:
    parser = _StylesheetLinkParser()
    parser.feed((static_dir / "index.html").read_text(encoding="utf-8"))
    return tuple(parser.hrefs)


def browser_stylesheet_paths(static_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    static_root = static_dir.resolve()
    for href in browser_stylesheet_hrefs(static_dir):
        path = urlparse(href).path
        if not path.startswith("static/"):
            raise AssertionError(f"stylesheet is outside /static: {href}")
        candidate = (static_root / path.removeprefix("static/")).resolve()
        if not candidate.is_relative_to(static_root):
            raise AssertionError(f"stylesheet escapes static root: {href}")
        paths.append(candidate)
    return tuple(paths)


def read_browser_stylesheets(static_dir: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in browser_stylesheet_paths(static_dir)
    )
