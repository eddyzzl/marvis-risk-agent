"""Every model call site in marvis/ is listed in the outbound inventory.

The inventory (docs/development/llm-outbound-inventory.md) names each
``.complete(`` call by file and enclosing function, with the adapter that
mints its prompt. Only file + function + count are checked here, so moving a
caller onto another adapter (the strategy compiler, plan 5.2) only needs the
adapter column updated.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re

_ROOT = Path(__file__).resolve().parents[1]
_INVENTORY = _ROOT / "docs" / "development" / "llm-outbound-inventory.md"
_ROW = re.compile(r"^\|\s*`(marvis/[^`]+\.py)`\s*\|\s*`([^`]+)`\s*\|")


def _call_sites(root: Path = _ROOT) -> Counter:
    """(file, innermost enclosing function) -> number of ``.complete(`` calls."""
    sites: Counter = Counter()
    for path in sorted((root / "marvis").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".complete(" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        functions = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "complete"):
                continue
            enclosing = [
                fn for fn in functions
                if fn.lineno <= node.lineno <= (fn.end_lineno or fn.lineno)
            ]
            # innermost function: the one that starts last
            name = max(enclosing, key=lambda fn: fn.lineno).name if enclosing else "<module>"
            sites[(path.relative_to(root).as_posix(), name)] += 1
    return sites


def _listed() -> Counter:
    rows = Counter()
    for line in _INVENTORY.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match:
            rows[(match.group(1), match.group(2))] += 1
    return rows


def test_every_complete_call_site_is_in_the_outbound_inventory():
    found, listed = _call_sites(), _listed()

    unlisted = found - listed
    stale = listed - found
    assert not unlisted, (
        "新的 .complete( 调用点没有登记到 docs/development/llm-outbound-inventory.md："
        f"{sorted(unlisted.elements())}"
    )
    assert not stale, f"清单里的调用点在代码里已不存在：{sorted(stale.elements())}"


def test_inventory_total_matches_the_listed_rows():
    total = sum(_listed().values())
    text = _INVENTORY.read_text(encoding="utf-8")
    assert f"共 {total} 处" in text


def test_scanner_counts_nested_async_and_repeated_calls_but_not_comments(tmp_path):
    package = tmp_path / "marvis" / "agent"
    package.mkdir(parents=True)
    (package / "x.py").write_text(
        "# client.complete( in a comment is not a call\n"
        "def outer(client):\n"
        "    def inner():\n"
        "        return client.complete(user_prompt=p)\n"
        "    return inner\n"
        "async def later(client):\n"
        "    first = await client.complete(user_prompt=p)\n"
        "    return first, await client.complete(user_prompt=q)\n",
        encoding="utf-8",
    )

    assert _call_sites(tmp_path) == Counter({
        ("marvis/agent/x.py", "inner"): 1,
        ("marvis/agent/x.py", "later"): 2,
    })
