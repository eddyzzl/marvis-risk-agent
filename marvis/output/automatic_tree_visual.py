"""Deterministic SVG and PNG projections of automatic rule-tree assets.

The automatic-tree asset and weighted-tree kernel own topology and metrics.  This
module validates that committed contract, derives only drawing coordinates from
its topology, and projects the already-measured labels without recalculation.
SVG and PNG deliberately consume the same private layout so their node and edge
placement cannot drift.

PNG byte determinism is scoped to ``AUTOMATIC_TREE_PNG_RENDERER_VERSION`` and
the recorded font digest. Unicode labels use an available covering font, with
Matplotlib's bundled DejaVu Sans as the portable baseline. Display rounding and
abbreviated identifiers never alter the full evidence embedded in either file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from matplotlib import font_manager, get_data_path
from matplotlib.ft2font import FT2Font
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION
from PIL.PngImagePlugin import PngInfo

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)


AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION = "strategy.automatic-tree-visual.v1"
AUTOMATIC_TREE_PNG_RENDERER_VERSION = (
    "strategy.automatic-tree-png/3;"
    f"pillow/{PILLOW_VERSION};"
    "font/sha256-in-metadata;labels/unicode"
)

_MIN_CANVAS_WIDTH = 960
_MAX_CANVAS_WIDTH = 4096
_HORIZONTAL_MARGIN = 56
_VERTICAL_MARGIN = 48
_LEAF_WIDTH = 280
_NODE_MAX_WIDTH = 256
_NODE_MIN_WIDTH = 180
_NODE_HEIGHT = 154
_LEVEL_PITCH = 214
_TEXT_LINE_HEIGHT = 18
_TEXT_PADDING = 12
_FONT_SIZE = 12
_MIN_NODE_GAP = 4
_DENSE_NODE_MAX_WIDTH = 10
_DENSE_NODE_HEIGHT = 10
_DENSE_LEVEL_PITCH = 56
_DENSE_BANNER_HEIGHT = 42
_DENSE_NOTICE = "dense overview/details in Nodes and Leaf Rules"
_FONT_NOTICE = "Font fallback: field aliases shown; full names remain in node evidence."

_BACKGROUND = "#f7f8fa"
_EDGE = "#7b8491"
_EDGE_LABEL = "#515967"
_SPLIT_FILL = "#e8eef6"
_SPLIT_STROKE = "#556987"
_LEAF_FILL = "#edf3e9"
_LEAF_STROKE = "#63785b"
_TEXT = "#20242b"


@dataclass(frozen=True)
class _LayoutNode:
    node_id: str
    kind: str
    x: int
    y: int
    width: int
    height: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _LayoutEdge:
    parent_id: str
    child_id: str
    branch: str


@dataclass(frozen=True)
class _TreeLayout:
    width: int
    height: int
    dense_overview: bool
    nodes: tuple[_LayoutNode, ...]
    edges: tuple[_LayoutEdge, ...]
    font: ImageFont.FreeTypeFont
    font_digest: str
    field_aliases: Mapping[str, str]


def render_automatic_tree_svg(asset: Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 SVG bytes for one strict tree asset."""

    canonical = validate_automatic_tree_asset(asset)
    layout = _build_layout(canonical)
    node_by_id = {node.node_id: node for node in layout.nodes}
    evidence_node_by_id = {
        node["node_id"]: node for node in canonical["tree_result"]["tree"]["nodes"]
    }
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{layout.width}" height="{layout.height}" '
            f'viewBox="0 0 {layout.width} {layout.height}" role="img" '
            f'data-layout-mode="{_layout_mode(layout)}" '
            'aria-label="Automatic rule tree">'
        ),
        "<metadata>"
        + _svg_text(
            f"{AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION} "
            f"asset_id={canonical['asset_id']} asset_hash={canonical['asset_hash']} "
            f"font_sha256={layout.font_digest} renderer={_renderer_identity(layout)}"
        )
        + "</metadata>",
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',"
        "'Noto Sans CJK SC','Microsoft YaHei','DejaVu Sans',sans-serif;"
        f"font-size:{_FONT_SIZE}px;fill:#20242b}}"
        ".edge-label{font-size:11px;font-weight:600;fill:#515967}"
        ".node-title{font-weight:600}"
        "</style>",
        f'<rect width="{layout.width}" height="{layout.height}" fill="{_BACKGROUND}"/>',
    ]
    if layout.dense_overview or layout.field_aliases:
        notice = _DENSE_NOTICE if layout.dense_overview else _FONT_NOTICE
        parts.extend(
            (
                f'<rect x="12" y="8" width="{layout.width - 24}" height="30" '
                'rx="6" fill="#fff4d6" stroke="#a66a00"/>',
                f'<text x="24" y="28" fill="#714700">{notice}</text>',
            )
        )
    parts.append('<g id="edges">')
    for edge in layout.edges:
        parent = node_by_id[edge.parent_id]
        child = node_by_id[edge.child_id]
        start_x, start_y, end_x, end_y = _edge_endpoints(parent, child, edge.branch)
        middle_y = (start_y + end_y) // 2
        path = (
            f"M {start_x} {start_y} L {end_x} {end_y}"
            if layout.dense_overview
            else (
                f"M {start_x} {start_y} C {start_x} {middle_y}, "
                f"{end_x} {middle_y}, {end_x} {end_y}"
            )
        )
        parts.append(
            '<path fill="none" '
            f'stroke="{_EDGE}" stroke-width="2" '
            f'data-parent-id="{_svg_attribute(edge.parent_id)}" '
            f'data-child-id="{_svg_attribute(edge.child_id)}" '
            f'data-branch="{edge.branch}" '
            f'd="{path}"/>'
        )
        if not layout.dense_overview:
            label_x = (start_x + end_x) // 2
            label_y = middle_y - 4
            parts.append(
                f'<text class="edge-label" x="{label_x}" y="{label_y}" '
                'text-anchor="middle">'
                + ("L" if edge.branch == "left" else "R")
                + "</text>"
            )
    parts.extend(("</g>", '<g id="nodes">'))
    for node in layout.nodes:
        left = node.x - node.width // 2
        top = node.y - node.height // 2
        fill = _SPLIT_FILL if node.kind == "split" else _LEAF_FILL
        stroke = _SPLIT_STROKE if node.kind == "split" else _LEAF_STROKE
        parts.append(
            f'<g class="node {node.kind}" '
            f'data-node-id="{_svg_attribute(node.node_id)}">'
        )
        parts.append(
            f'<rect x="{left}" y="{top}" width="{node.width}" '
            f'height="{node.height}" rx="8" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"/>'
        )
        parts.append(
            "<title>"
            + _svg_text(_accessible_node_text(evidence_node_by_id[node.node_id]))
            + "</title>"
        )
        text_x = left + _TEXT_PADDING
        first_y = top + _TEXT_PADDING + _FONT_SIZE
        if not layout.dense_overview:
            parts.append(f'<text x="{text_x}" y="{first_y}">')
            for index, line in enumerate(node.lines):
                class_attribute = ' class="node-title"' if index == 0 else ""
                dy = "0" if index == 0 else str(_TEXT_LINE_HEIGHT)
                text_width = round(layout.font.getlength(line), 2)
                parts.append(
                    f'<tspan x="{text_x}" dy="{dy}"{class_attribute} '
                    f'textLength="{text_width}" lengthAdjust="spacingAndGlyphs">'
                    + _svg_text(line)
                    + "</tspan>"
                )
            parts.append("</text>")
        parts.append("</g>")
    parts.extend(("</g>", "</svg>"))
    return ("\n".join(parts) + "\n").encode("utf-8")


def render_automatic_tree_png(asset: Mapping[str, Any]) -> bytes:
    """Return deterministic PNG bytes using the same layout as the SVG."""

    canonical = validate_automatic_tree_asset(asset)
    layout = _build_layout(canonical)
    font = layout.font
    image = Image.new("RGB", (layout.width, layout.height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    node_by_id = {node.node_id: node for node in layout.nodes}

    if layout.dense_overview or layout.field_aliases:
        notice = _DENSE_NOTICE if layout.dense_overview else _FONT_NOTICE
        draw.rounded_rectangle(
            (12, 8, layout.width - 12, 38),
            radius=6,
            fill="#fff4d6",
            outline="#a66a00",
            width=1,
        )
        _draw_png_text(draw, (24, 16), notice, font=font, fill="#714700")

    for edge in layout.edges:
        parent = node_by_id[edge.parent_id]
        child = node_by_id[edge.child_id]
        start_x, start_y, end_x, end_y = _edge_endpoints(parent, child, edge.branch)
        if layout.dense_overview:
            draw.line((start_x, start_y, end_x, end_y), fill=_EDGE, width=1)
        else:
            middle_y = (start_y + end_y) // 2
            points = (
                (start_x, start_y),
                (start_x, middle_y),
                (end_x, middle_y),
                (end_x, end_y),
            )
            draw.line(points, fill=_EDGE, width=2, joint="curve")
            draw.polygon(
                (
                    (end_x, end_y),
                    (end_x - 4, end_y - 7),
                    (end_x + 4, end_y - 7),
                ),
                fill=_EDGE,
            )
            branch_label = "L" if edge.branch == "left" else "R"
            label_position = ((start_x + end_x) // 2, middle_y - 13)
            _draw_centered_text(
                draw,
                label_position,
                branch_label,
                font=font,
                fill=_EDGE_LABEL,
            )

    for node in layout.nodes:
        left = node.x - node.width // 2
        top = node.y - node.height // 2
        right = left + node.width
        bottom = top + node.height
        fill = _SPLIT_FILL if node.kind == "split" else _LEAF_FILL
        outline = _SPLIT_STROKE if node.kind == "split" else _LEAF_STROKE
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=8,
            fill=fill,
            outline=outline,
            width=2,
        )
        if not layout.dense_overview:
            text_x = left + _TEXT_PADDING
            text_y = top + _TEXT_PADDING
            for index, line in enumerate(node.lines):
                _draw_png_text(
                    draw,
                    (text_x, text_y + index * _TEXT_LINE_HEIGHT),
                    line,
                    font=font,
                    fill=_TEXT,
                )

    output = BytesIO()
    png_metadata = PngInfo()
    png_metadata.add_text("MARVIS Renderer", _renderer_identity(layout))
    png_metadata.add_text("MARVIS Font SHA256", layout.font_digest)
    png_metadata.add_itxt(
        "MARVIS Evidence",
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        zip=True,
    )
    png_metadata.add_itxt(
        "MARVIS Field Aliases",
        json.dumps(dict(layout.field_aliases), ensure_ascii=False, sort_keys=True),
    )
    try:
        image.save(
            output,
            format="PNG",
            optimize=False,
            compress_level=9,
            pnginfo=png_metadata,
        )
    finally:
        image.close()
    return output.getvalue()


def _build_layout(asset: Mapping[str, Any]) -> _TreeLayout:
    tree_result = asset["tree_result"]
    tree = tree_result["tree"]
    raw_nodes = tree["nodes"]
    node_by_id = {node["node_id"]: node for node in raw_nodes}
    display_ids = _display_ids(
        [node["node_id"] for node in raw_nodes]
        + [node["rule_id"] for node in raw_nodes if node["kind"] == "leaf"]
    )
    labels = {node["node_id"]: _node_lines(node, display_ids) for node in raw_nodes}
    characters = "".join(sorted(set("…" + "".join(line for lines in labels.values() for line in lines))))
    font, font_digest, missing = _label_font(characters)
    field_aliases = {}
    if missing:
        features = sorted({node["feature"] for node in raw_nodes if node["kind"] == "split"})
        field_aliases = {
            feature: f"Field {index + 1}"
            for index, feature in enumerate(features)
            if any(ord(character) in missing for character in feature)
        }
        labels = {
            node["node_id"]: _node_lines(node, display_ids, field_aliases)
            for node in raw_nodes
        }
    leaf_positions: dict[str, int] = {}

    def assign_leaf_position(node_id: str) -> float:
        node = node_by_id[node_id]
        if node["kind"] == "leaf":
            position = len(leaf_positions)
            leaf_positions[node_id] = position
            return float(position)
        left_position = assign_leaf_position(node["left_child_id"])
        right_position = assign_leaf_position(node["right_child_id"])
        return (left_position + right_position) / 2.0

    abstract_x: dict[str, float] = {}

    def record_positions(node_id: str) -> float:
        node = node_by_id[node_id]
        if node["kind"] == "leaf":
            position = float(leaf_positions[node_id])
        else:
            left_position = record_positions(node["left_child_id"])
            right_position = record_positions(node["right_child_id"])
            position = (left_position + right_position) / 2.0
        abstract_x[node_id] = position
        return position

    assign_leaf_position(tree["root_node_id"])
    record_positions(tree["root_node_id"])
    leaf_count = tree["leaf_count"]
    width = min(
        _MAX_CANVAS_WIDTH,
        max(_MIN_CANVAS_WIDTH, leaf_count * _LEAF_WIDTH + 2 * _HORIZONTAL_MARGIN),
    )
    usable_width = width - 2 * _HORIZONTAL_MARGIN
    leaf_spacing = usable_width / leaf_count
    dense_overview = leaf_spacing < _NODE_MIN_WIDTH + _MIN_NODE_GAP
    if dense_overview:
        node_width = max(
            2,
            min(_DENSE_NODE_MAX_WIDTH, int(leaf_spacing - _MIN_NODE_GAP)),
        )
        node_height = _DENSE_NODE_HEIGHT
        level_pitch = _DENSE_LEVEL_PITCH
        banner_height = _DENSE_BANNER_HEIGHT
        display_lines = {node_id: () for node_id in labels}
    else:
        node_width = max(
            _NODE_MIN_WIDTH,
            min(_NODE_MAX_WIDTH, int(leaf_spacing * 0.82)),
        )
        display_lines = {
            node_id: tuple(
                part
                for index, line in enumerate(lines)
                for part in (
                    (_fit_label(line, font, node_width - 2 * _TEXT_PADDING),)
                    if index < 2
                    else _wrap_label(line, font, node_width - 2 * _TEXT_PADDING)
                )
            )
            for node_id, lines in labels.items()
        }
        node_height = max(
            _NODE_HEIGHT,
            max(map(len, display_lines.values())) * _TEXT_LINE_HEIGHT + 2 * _TEXT_PADDING,
        )
        level_pitch = node_height + (_LEVEL_PITCH - _NODE_HEIGHT)
        banner_height = _DENSE_BANNER_HEIGHT if field_aliases else 0
    max_depth = max(node["depth"] for node in raw_nodes)
    height = (
        banner_height + 2 * _VERTICAL_MARGIN + node_height + max_depth * level_pitch
    )

    layout_nodes = []
    for node in raw_nodes:
        abstract = abstract_x[node["node_id"]]
        x = int(round(_HORIZONTAL_MARGIN + (abstract + 0.5) * leaf_spacing))
        y = (
            banner_height
            + _VERTICAL_MARGIN
            + node_height // 2
            + node["depth"] * level_pitch
        )
        layout_nodes.append(
            _LayoutNode(
                node_id=node["node_id"],
                kind=node["kind"],
                x=x,
                y=y,
                width=node_width,
                height=node_height,
                lines=display_lines[node["node_id"]],
            )
        )

    edges = []
    for node in raw_nodes:
        if node["kind"] != "split":
            continue
        edges.extend(
            (
                _LayoutEdge(node["node_id"], node["left_child_id"], "left"),
                _LayoutEdge(node["node_id"], node["right_child_id"], "right"),
            )
        )
    return _TreeLayout(
        width=width,
        height=height,
        dense_overview=dense_overview,
        nodes=tuple(layout_nodes),
        edges=tuple(edges),
        font=font,
        font_digest=font_digest,
        field_aliases=field_aliases,
    )


def _node_lines(
    node: Mapping[str, Any],
    display_ids: Mapping[str, str],
    field_aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    metrics = node["metrics"]
    unweighted = metrics["unweighted"]
    common = [
        f"Samples {_count_text(unweighted['total'])} · bad {_count_text(unweighted['bad'])}",
        (
            f"Bad rate {_percent_text(unweighted['bad_rate'])} · "
            f"lift {_metric_text(unweighted['lift'])}"
        ),
        (
            f"Share {_percent_text(unweighted['share'])} · "
            f"capture {_percent_text(unweighted['bad_capture'])}"
        ),
    ]
    weighted = metrics["weighted"]
    if weighted["status"] == "available":
        common.append(
            f"Weighted {_count_text(weighted['total'])} · "
            f"bad {_percent_text(weighted['bad_rate'])}"
        )
    if node["kind"] == "split":
        diagnostic = node["direction_diagnostic"]
        lines = [
            f"Split · {display_ids[node['node_id']]}",
            f"{(field_aliases or {}).get(node['feature'], node['feature'])} ≤ {_value_text(node['threshold'])}",
            f"Missing → {node['missing_child']} · {diagnostic['status']}",
            *common,
        ]
    else:
        lines = [
            f"Leaf · {display_ids[node['node_id']]}",
            f"Rule · {display_ids[node['rule_id']]}",
            *common,
        ]
    return tuple(_display_safe_text(line) for line in lines[:7])


def _accessible_node_text(node: Mapping[str, Any]) -> str:
    return json.dumps(
        node,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _value_text(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _count_text(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


def _percent_text(value: object) -> str:
    return f"{value:.1%}" if isinstance(value, int | float) else "N/A"


def _metric_text(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, int | float) else "N/A"


def _display_ids(identifiers: list[str]) -> dict[str, str]:
    """Choose collision-free prefixes; full identifiers remain in file evidence."""
    ids = set(identifiers)
    suffixes = {value: value.rsplit("-", 1)[-1] for value in ids}
    result = {}
    for value in sorted(ids):
        suffix = suffixes[value]
        length = 8
        while length < len(suffix) and any(
            other != value and text[:length] == suffix[:length]
            for other, text in suffixes.items()
        ):
            length += 1
        result[value] = (
            value
            if any(other != value and text == suffix for other, text in suffixes.items())
            else suffix[:length]
        )
    return result


@lru_cache(maxsize=32)
def _label_font(characters: str) -> tuple[ImageFont.FreeTypeFont, str, frozenset[int]]:
    baseline = Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    required = {ord(character) for character in characters}
    baseline_map = FT2Font(str(baseline)).get_charmap()
    selected = baseline
    coverage = len(required.intersection(baseline_map))
    if coverage < len(required):
        # Explicit coverage, stable ordering, and a recorded digest prevent font
        # names or FONTCONFIG settings from silently changing evidence identity.
        paths = sorted({entry.fname for entry in font_manager.fontManager.ttflist})
        for path in paths:
            try:
                candidate = FT2Font(path)
                header = candidate.get_sfnt_table("head")
                # OpenType head.flags bit 14 denotes generic range placeholders,
                # not real glyph coverage (e.g. Matplotlib's LastResortHE font).
                if header is not None and header["flags"] & (1 << 14):
                    continue
                covered = len(required.intersection(candidate.get_charmap()))
            except (OSError, RuntimeError):
                continue
            if covered > coverage:
                selected, coverage = Path(path), covered
            if coverage == len(required):
                break
    missing = frozenset()
    if coverage < len(required):
        # Never silently substitute tofu. Unsupported field names receive
        # explicit aliases, and their full names remain in both evidence files.
        selected = baseline
        missing = frozenset(required.difference(baseline_map))
    payload = selected.read_bytes()
    return (
        ImageFont.truetype(BytesIO(payload), size=_FONT_SIZE),
        hashlib.sha256(payload).hexdigest(),
        missing,
    )


def automatic_tree_renderer_identity(asset: Mapping[str, Any]) -> str:
    """Return the version and actual font identity used by all tree deliveries."""
    return _renderer_identity(_build_layout(validate_automatic_tree_asset(asset)))


def _renderer_identity(layout: _TreeLayout) -> str:
    return f"{AUTOMATIC_TREE_PNG_RENDERER_VERSION};font-sha256/{layout.font_digest}"


def _fit_label(value: str, font: ImageFont.FreeTypeFont, width: int) -> str:
    safe = _display_safe_text(value).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if font.getlength(safe) <= width:
        return safe
    if " ≤ " in safe:
        field, threshold = safe.rsplit(" ≤ ", 1)
        suffix = "… ≤ " + threshold
        if font.getlength(suffix) <= width:
            while field and font.getlength(field + suffix) > width:
                field = field[:-1]
            return field + suffix
    while safe and font.getlength(safe + "…") > width:
        safe = safe[:-1]
    return safe + "…" if font.getlength("…") <= width else ""


def _wrap_label(value: str, font: ImageFont.FreeTypeFont, width: int) -> tuple[str, ...]:
    """Keep measured metrics whole where possible; never replace them with ellipses."""
    if font.getlength(value) <= width:
        return (value,)
    lines = []
    for metric in value.split(" · "):
        current = ""
        for word in metric.split():
            combined = f"{current} {word}" if current else word
            if font.getlength(combined) <= width:
                current = combined
                continue
            if current:
                lines.append(current)
            current = ""
            # An unusually large count can itself exceed the available width.
            # Character wrapping preserves every digit instead of truncating it.
            for character in word:
                if current and font.getlength(current + character) > width:
                    lines.append(current)
                    current = ""
                current += character
        if current:
            lines.append(current)
    return tuple(lines)


def _display_safe_text(value: object) -> str:
    text = str(value)
    pieces = []
    for character in text:
        codepoint = ord(character)
        if (
            character in "\t\n\r"
            or 0x20 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        ):
            pieces.append(character)
        else:
            pieces.append(f"\\u{codepoint:04x}")
    return "".join(pieces)


def _svg_text(value: object) -> str:
    return escape(_display_safe_text(value), quote=False)


def _svg_attribute(value: object) -> str:
    return escape(_display_safe_text(value), quote=True)


def _layout_mode(layout: _TreeLayout) -> str:
    return "dense-overview" if layout.dense_overview else "detail"


def _edge_endpoints(
    parent: _LayoutNode,
    child: _LayoutNode,
    branch: str,
) -> tuple[int, int, int, int]:
    offset = max(1, min(4, parent.width // 3))
    start_x = parent.x - offset if branch == "left" else parent.x + offset
    return (
        start_x,
        parent.y + parent.height // 2,
        child.x,
        child.y - child.height // 2,
    )


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    _draw_png_text(
        draw,
        (position[0] - width // 2, position[1]),
        text,
        font=font,
        fill=fill,
    )


def _draw_png_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: str,
) -> None:
    draw.text(position, text, font=font, fill=fill)


__all__ = [
    "AUTOMATIC_TREE_PNG_RENDERER_VERSION",
    "AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION",
    "automatic_tree_renderer_identity",
    "render_automatic_tree_png",
    "render_automatic_tree_svg",
]
