"""Deterministic SVG and PNG projections of automatic rule-tree assets.

The automatic-tree asset and weighted-tree kernel own topology and metrics.  This
module validates that committed contract, derives only drawing coordinates from
its topology, and projects the already-measured labels without recalculation.
SVG and PNG deliberately consume the same private layout so their node and edge
placement cannot drift.

PNG byte determinism is scoped to ``AUTOMATIC_TREE_PNG_RENDERER_VERSION``: the
renderer identifier includes the active Pillow version, uses only Pillow's
bundled default font, and converts every non-ASCII raster label to deterministic
``\\uXXXX`` ASCII escapes.  No operating-system font discovery participates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from io import BytesIO
import json
from typing import Any

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION
from PIL.PngImagePlugin import PngInfo

from marvis.packs.strategy.automatic_tree_asset import (
    validate_automatic_tree_asset,
)


AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION = "strategy.automatic-tree-visual.v1"
AUTOMATIC_TREE_PNG_RENDERER_VERSION = (
    "strategy.automatic-tree-png/1;"
    f"pillow/{PILLOW_VERSION};"
    "font/pillow-bundled-default;labels/ascii-unicode-escapes"
)

_MIN_CANVAS_WIDTH = 960
_MAX_CANVAS_WIDTH = 4096
_HORIZONTAL_MARGIN = 56
_VERTICAL_MARGIN = 48
_LEAF_WIDTH = 224
_NODE_MAX_WIDTH = 216
_NODE_MIN_WIDTH = 88
_NODE_HEIGHT = 132
_LEVEL_PITCH = 190
_TEXT_LINE_HEIGHT = 16
_TEXT_PADDING = 10
_MAX_LABEL_CHARACTERS = 30
_MIN_NODE_GAP = 4
_DENSE_NODE_MAX_WIDTH = 10
_DENSE_NODE_HEIGHT = 10
_DENSE_LEVEL_PITCH = 56
_DENSE_BANNER_HEIGHT = 42
_DENSE_NOTICE = "dense overview/details in Nodes and Leaf Rules"

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
            f"asset_id={canonical['asset_id']} asset_hash={canonical['asset_hash']}"
        )
        + "</metadata>",
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',"
        "'Noto Sans CJK SC','Microsoft YaHei','DejaVu Sans',sans-serif;"
        "font-size:12px;fill:#20242b}"
        ".edge-label{font-size:11px;font-weight:600;fill:#515967}"
        ".node-title{font-weight:600}"
        "</style>",
        f'<rect width="{layout.width}" height="{layout.height}" fill="{_BACKGROUND}"/>',
    ]
    if layout.dense_overview:
        parts.extend(
            (
                f'<rect x="12" y="8" width="{layout.width - 24}" height="30" '
                'rx="6" fill="#fff4d6" stroke="#a66a00"/>',
                f'<text x="24" y="28" fill="#714700">{_DENSE_NOTICE}</text>',
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
        first_y = top + _TEXT_PADDING + 12
        if not layout.dense_overview:
            parts.append(f'<text x="{text_x}" y="{first_y}">')
            for index, line in enumerate(node.lines):
                class_attribute = ' class="node-title"' if index == 0 else ""
                dy = "0" if index == 0 else str(_TEXT_LINE_HEIGHT)
                parts.append(
                    f'<tspan x="{text_x}" dy="{dy}"{class_attribute}>'
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
    font = ImageFont.load_default()
    image = Image.new("RGB", (layout.width, layout.height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    node_by_id = {node.node_id: node for node in layout.nodes}

    if layout.dense_overview:
        draw.rounded_rectangle(
            (12, 8, layout.width - 12, 38),
            radius=6,
            fill="#fff4d6",
            outline="#a66a00",
            width=1,
        )
        _draw_png_text(draw, (24, 16), _DENSE_NOTICE, font=font, fill="#714700")

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
    png_metadata.add_text("MARVIS Renderer", AUTOMATIC_TREE_PNG_RENDERER_VERSION)
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
    else:
        node_width = max(
            _NODE_MIN_WIDTH,
            min(_NODE_MAX_WIDTH, int(leaf_spacing * 0.82)),
        )
        node_height = _NODE_HEIGHT
        level_pitch = _LEVEL_PITCH
        banner_height = 0
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
                lines=_node_lines(node),
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
    )


def _node_lines(node: Mapping[str, Any]) -> tuple[str, ...]:
    metrics = node["metrics"]
    unweighted = metrics["unweighted"]
    common = [
        f"n={_value_text(unweighted['total'])} bad={_value_text(unweighted['bad'])}",
        (
            f"bad_rate={_value_text(unweighted['bad_rate'])} "
            f"lift={_value_text(unweighted['lift'])}"
        ),
        (
            f"share={_value_text(unweighted['share'])} "
            f"capture={_value_text(unweighted['bad_capture'])}"
        ),
    ]
    weighted = metrics["weighted"]
    if weighted["status"] == "available":
        common.append(
            f"weighted n={_value_text(weighted['total'])} "
            f"rate={_value_text(weighted['bad_rate'])}"
        )
    if node["kind"] == "split":
        diagnostic = node["direction_diagnostic"]
        lines = [
            f"Split · {_clip_label(node['node_id'])}",
            _clip_label(f"{node['feature']} ≤ {_value_text(node['threshold'])}"),
            _clip_label(f"missing → {node['missing_child']} · {diagnostic['status']}"),
            *common,
        ]
    else:
        lines = [
            f"Leaf · {_clip_label(node['node_id'])}",
            _clip_label(str(node["rule_id"])),
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


def _clip_label(value: str) -> str:
    safe = _display_safe_text(value)
    if len(safe) <= _MAX_LABEL_CHARACTERS:
        return safe
    return safe[: _MAX_LABEL_CHARACTERS - 1] + "…"


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
    ascii_text = _png_ascii_text(text)
    bounds = draw.textbbox((0, 0), ascii_text, font=font)
    width = bounds[2] - bounds[0]
    _draw_png_text(
        draw,
        (position[0] - width // 2, position[1]),
        ascii_text,
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
    draw.text(position, _png_ascii_text(text), font=font, fill=fill)


def _png_ascii_text(value: object) -> str:
    pieces = []
    for character in str(value):
        codepoint = ord(character)
        if 0x20 <= codepoint <= 0x7E:
            pieces.append(character)
        elif codepoint <= 0xFFFF:
            pieces.append(f"\\u{codepoint:04x}")
        else:
            offset = codepoint - 0x10000
            high = 0xD800 + (offset >> 10)
            low = 0xDC00 + (offset & 0x3FF)
            pieces.append(f"\\u{high:04x}\\u{low:04x}")
    return "".join(pieces)


__all__ = [
    "AUTOMATIC_TREE_PNG_RENDERER_VERSION",
    "AUTOMATIC_TREE_VISUAL_SCHEMA_VERSION",
    "render_automatic_tree_png",
    "render_automatic_tree_svg",
]
