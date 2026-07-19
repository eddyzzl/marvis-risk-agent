from __future__ import annotations

import copy
from io import BytesIO
import json
import os
from xml.etree import ElementTree

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pytest

from marvis.feature.weighted_rule_tree import build_weighted_rule_tree
from marvis.output.automatic_tree_visual import (
    AUTOMATIC_TREE_PNG_RENDERER_VERSION,
    render_automatic_tree_png,
    render_automatic_tree_svg,
)
from marvis.packs.strategy.automatic_tree_asset import (
    AutomaticTreeAssetError,
    build_automatic_tree_asset,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _reverse_mapping_order(value):
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(item)
            for key, item in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


def _segments_intersect(first, second) -> bool:
    start_a, end_a = first
    start_b, end_b = second

    def orientation(left, middle, right):
        return (middle[1] - left[1]) * (right[0] - middle[0]) - (
            middle[0] - left[0]
        ) * (right[1] - middle[1])

    def on_segment(left, middle, right):
        return min(left[0], right[0]) <= middle[0] <= max(left[0], right[0]) and min(
            left[1], right[1]
        ) <= middle[1] <= max(left[1], right[1])

    orientations = (
        orientation(start_a, end_a, start_b),
        orientation(start_a, end_a, end_b),
        orientation(start_b, end_b, start_a),
        orientation(start_b, end_b, end_a),
    )
    if (orientations[0] > 0) != (orientations[1] > 0) and (orientations[2] > 0) != (
        orientations[3] > 0
    ):
        return True
    return any(
        orientation_value == 0 and on_segment(left, middle, right)
        for orientation_value, left, middle, right in (
            (orientations[0], start_a, start_b, end_a),
            (orientations[1], start_a, end_b, end_a),
            (orientations[2], start_b, start_a, end_b),
            (orientations[3], start_b, end_a, end_b),
        )
    )


def _segment_intersects_box(segment, box) -> bool:
    start, end = segment
    left, top, right, bottom = box
    if any(
        left <= point[0] <= right and top <= point[1] <= bottom
        for point in (start, end)
    ):
        return True
    corners = (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    )
    return any(
        _segments_intersect(segment, (corners[index], corners[(index + 1) % 4]))
        for index in range(4)
    )


def _asset() -> dict:
    feature = '风险&<变量>"'
    frame = pd.DataFrame(
        {
            feature: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "bad": [0, 0, 0, 1, 1, 1],
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=[feature],
        target_col="bad",
        directions={feature: "unordered"},
        max_depth=1,
        min_leaf_count=1,
    )
    return build_automatic_tree_asset(
        tree,
        task_id="task-automatic-tree-visual",
        dataset_id="dataset-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=["dataset:dataset-labelled", "workspace:visual:3"],
    )


def _dense_asset() -> dict:
    features = ["风险&<位0>", *(f"bit_{index}" for index in range(1, 8))]
    row_ids = list(range(256))
    frame = pd.DataFrame(
        {
            **{
                feature: [float((row_id >> index) & 1) for row_id in row_ids]
                for index, feature in enumerate(features)
            },
            "bad": [row_id.bit_count() % 2 for row_id in row_ids],
        }
    )
    tree = build_weighted_rule_tree(
        frame,
        feature_cols=features,
        target_col="bad",
        directions={feature: "unordered" for feature in features},
        max_depth=8,
        min_leaf_count=1,
    )
    assert tree["tree"]["node_count"] == 511
    assert tree["tree"]["leaf_count"] == 256
    return build_automatic_tree_asset(
        tree,
        task_id="task-automatic-tree-dense",
        dataset_id="dataset-dense-labelled",
        dataset_content_hash=HASH_A,
        workspace_revision=3,
        workspace_generation=7,
        semantic_mapping_hash=HASH_B,
        registry_metadata_hash=HASH_C,
        sample_context_hash=HASH_D,
        source_refs=["dataset:dataset-dense-labelled", "workspace:dense:3"],
    )


def test_svg_is_valid_xml_with_exact_topology_and_escaped_cjk_labels() -> None:
    asset = _asset()

    svg = render_automatic_tree_svg(asset)
    root = ElementTree.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    node_groups = root.findall(".//svg:g[@data-node-id]", namespace)
    edges = root.findall(".//svg:path[@data-parent-id]", namespace)

    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert len(node_groups) == asset["tree_result"]["tree"]["node_count"]
    assert len(edges) == asset["tree_result"]["tree"]["node_count"] - 1
    assert {item.attrib["data-node-id"] for item in node_groups} == {
        item["node_id"] for item in asset["tree_result"]["tree"]["nodes"]
    }
    text = "".join(root.itertext())
    assert '风险&<变量>"' in text
    root_metrics = asset["tree_result"]["tree"]["nodes"][0]["metrics"]["unweighted"]
    assert f"bad_rate={root_metrics['bad_rate']!r}" in text
    assert f"lift={root_metrics['lift']!r}" in text
    assert b"&amp;" in svg
    assert b"&lt;" in svg


def test_svg_and_png_repeat_byte_for_byte_and_png_has_valid_signature() -> None:
    asset = _asset()
    reordered = _reverse_mapping_order(asset)

    first_svg = render_automatic_tree_svg(asset)
    second_svg = render_automatic_tree_svg(asset)
    first_png = render_automatic_tree_png(asset)
    second_png = render_automatic_tree_png(asset)

    assert first_svg == second_svg
    assert first_svg == render_automatic_tree_svg(reordered)
    assert first_png == second_png
    assert first_png == render_automatic_tree_png(reordered)
    assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(first_png) > 100

    with Image.open(BytesIO(first_png)) as image:
        assert image.info["MARVIS Renderer"] == AUTOMATIC_TREE_PNG_RENDERER_VERSION


def test_dense_depth_eight_tree_uses_non_overlapping_overview_with_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _dense_asset()
    original_text = ImageDraw.ImageDraw.text
    drawn_text: list[str] = []

    def capture_text(draw, xy, text, *args, **kwargs):
        drawn_text.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)

    svg = render_automatic_tree_svg(asset)
    png = render_automatic_tree_png(asset)
    root = ElementTree.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    node_groups = root.findall(".//svg:g[@data-node-id]", namespace)
    boxes_by_node_id = {}
    for group in node_groups:
        rect = group.find("svg:rect", namespace)
        assert rect is not None
        left = int(rect.attrib["x"])
        top = int(rect.attrib["y"])
        width = int(rect.attrib["width"])
        height = int(rect.attrib["height"])
        node_id = group.attrib["data-node-id"]
        boxes_by_node_id[node_id] = (left, top, left + width, top + height)
        title = group.find("svg:title", namespace)
        assert title is not None
        evidence_node = next(
            node
            for node in asset["tree_result"]["tree"]["nodes"]
            if node["node_id"] == node_id
        )
        assert title.text == json.dumps(
            evidence_node,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    assert root.attrib["data-layout-mode"] == "dense-overview"
    assert "dense overview/details in Nodes and Leaf Rules" in "".join(root.itertext())
    assert "风险&<位0>" in "".join(root.itertext())
    boxes = list(boxes_by_node_id.values())
    assert len(boxes_by_node_id) == 511
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert (
                first[2] <= second[0]
                or second[2] <= first[0]
                or first[3] <= second[1]
                or second[3] <= first[1]
            )
    paths = root.findall(".//svg:path[@data-parent-id]", namespace)
    assert len({path.attrib["d"] for path in paths}) == len(paths)
    edges = []
    for path in paths:
        command, start_x, start_y, line, end_x, end_y = path.attrib["d"].split()
        assert (command, line) == ("M", "L")
        edges.append(
            (
                path.attrib["data-parent-id"],
                path.attrib["data-child-id"],
                ((int(start_x), int(start_y)), (int(end_x), int(end_y))),
            )
        )
    segments = [segment for _, _, segment in edges]
    for index, first in enumerate(segments):
        assert all(
            not _segments_intersect(first, second) for second in segments[index + 1 :]
        )
    for parent_id, child_id, segment in edges:
        assert all(
            not _segment_intersects_box(segment, box)
            for node_id, box in boxes_by_node_id.items()
            if node_id not in {parent_id, child_id}
        )
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert "dense overview/details in Nodes and Leaf Rules" in drawn_text


def test_png_uses_only_bundled_font_and_ascii_escaped_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _asset()
    baseline = render_automatic_tree_png(asset)
    original_truetype = ImageFont.truetype
    original_text = ImageDraw.ImageDraw.text
    drawn_text: list[str] = []

    def guarded_truetype(font, *args, **kwargs):
        if isinstance(font, str | bytes | os.PathLike):
            raise AssertionError("PNG renderer attempted to load an OS font path")
        return original_truetype(font, *args, **kwargs)

    def capture_text(draw, xy, text, *args, **kwargs):
        drawn_text.append(text)
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", guarded_truetype)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    monkeypatch.setenv("FONTCONFIG_FILE", "/nonexistent/marvis-fonts.conf")
    monkeypatch.setenv("FONTCONFIG_PATH", "/nonexistent/marvis-fonts")

    rendered = render_automatic_tree_png(asset)

    assert rendered == baseline
    assert drawn_text
    assert all(text.isascii() for text in drawn_text)
    assert any("\\u98ce\\u9669" in text for text in drawn_text)


@pytest.mark.parametrize(
    "renderer", [render_automatic_tree_svg, render_automatic_tree_png]
)
def test_visual_renderers_fail_closed_on_tampered_asset(renderer) -> None:
    tampered = copy.deepcopy(_asset())
    tampered["tree_result"]["tree"]["node_count"] += 1

    with pytest.raises(AutomaticTreeAssetError):
        renderer(tampered)
