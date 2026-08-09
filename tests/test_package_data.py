import hashlib
import json
from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from docx import Document

from marvis.plugins.manifest import parse_manifest
from scripts.build_default_report_template import build_template
from tests.static_stylesheets import browser_stylesheet_paths


_WORDPROCESSING_NAMESPACE = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
)
_VOLATILE_CORE_PROPERTIES = {
    "{http://purl.org/dc/terms/}created",
    "{http://purl.org/dc/terms/}modified",
}


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_xml_node(
    element: ET.Element,
    *,
    part_name: str,
) -> tuple | None:
    if part_name == "docProps/core.xml" and element.tag in _VOLATILE_CORE_PROPERTIES:
        return None

    children = tuple(
        normalized
        for child in element
        if (normalized := _normalized_xml_node(child, part_name=part_name)) is not None
    )
    if part_name.endswith(".rels"):
        children = tuple(sorted(children))

    text = element.text or ""
    if not text.strip():
        text = ""
    return element.tag, tuple(sorted(element.attrib.items())), text, children


def _visible_text_digest(path: Path) -> str:
    with ZipFile(path) as archive:
        visible_parts = ["word/document.xml"]
        visible_parts.extend(
            name
            for name in sorted(archive.namelist())
            if name.startswith(("word/header", "word/footer"))
            and name.endswith(".xml")
        )
        visible_text = []
        text_tag = f"{{{_WORDPROCESSING_NAMESPACE}}}t"
        for part_name in visible_parts:
            root = ET.fromstring(archive.read(part_name))
            visible_text.extend(element.text or "" for element in root.iter(text_tag))
    return _digest(visible_text)


def _package_structure_digest(path: Path) -> str:
    with ZipFile(path) as archive:
        normalized_parts = []
        for part_name in sorted(archive.namelist()):
            payload = archive.read(part_name)
            if part_name.endswith((".xml", ".rels")):
                content_digest = _digest(
                    _normalized_xml_node(
                        ET.fromstring(payload),
                        part_name=part_name,
                    )
                )
                part_kind = "xml"
            else:
                content_digest = hashlib.sha256(payload).hexdigest()
                part_kind = "media" if part_name.startswith("word/media/") else "binary"
            normalized_parts.append((part_name, part_kind, content_digest))
    return _digest(normalized_parts)


def _assert_docx_matches(actual: Path, canonical: Path) -> None:
    assert _visible_text_digest(actual) == _visible_text_digest(canonical), (
        "visible report text differs from the canonical neutral template"
    )
    assert _package_structure_digest(actual) == _package_structure_digest(canonical), (
        "normalized DOCX package differs from the canonical neutral template"
    )


def test_static_es_modules_are_declared_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["marvis"]

    assert "static/js/*" in package_data
    assert "static/css/*" in package_data
    assert "packs/*/manifest.json" in package_data


def test_default_validation_report_template_is_declared_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["marvis"]

    assert "report_templates/*.docx" in package_data
    assert Path("marvis/report_templates/default.docx").is_file()


def test_default_validation_report_template_matches_canonical_builder(
    tmp_path: Path,
) -> None:
    packaged_template = Path("marvis/report_templates/default.docx")
    canonical_template = build_template(tmp_path / "canonical.docx")
    _assert_docx_matches(packaged_template, canonical_template)

    packaged_round_trip = tmp_path / "packaged-round-trip.docx"
    canonical_round_trip = tmp_path / "canonical-round-trip.docx"
    Document(packaged_template).save(packaged_round_trip)
    Document(canonical_template).save(canonical_round_trip)
    _assert_docx_matches(packaged_round_trip, canonical_round_trip)


def test_package_discovery_is_limited_to_marvis_runtime_package():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert package_finder["include"] == ["marvis*"]


def test_static_es_module_files_exist_for_declared_imports():
    static_js = Path("marvis/static/js")

    for module_name in (
        "api.js",
        "agent-memory-panel.js",
        "branding.js",
        "dialogs.js",
        "draft-tools-panel.js",
        "focus-ring.js",
        "layout-resize.js",
        "metric-tables.js",
        "platform-confirm.js",
        "polling.js",
        "precision-consistency.js",
        "render-agent.js",
        "render-metrics.js",
        "state.js",
        "step-checker.js",
        "task-search.js",
        "toast.js",
        "ui-utils.js",
    ):
        assert (static_js / module_name).is_file()


def test_static_css_module_files_exist_for_declared_links():
    static_dir = Path("marvis/static")
    stylesheet_paths = browser_stylesheet_paths(static_dir)

    assert stylesheet_paths
    assert all(path.is_file() for path in stylesheet_paths)


def test_browser_app_manifest_and_icons_exist():
    manifest_path = Path("marvis/static/manifest.webmanifest")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["display"] == "standalone"
    assert manifest["theme_color"] == "#181818"
    assert manifest["background_color"] == "#181818"
    icon_sources = {icon["src"] for icon in manifest["icons"]}
    assert icon_sources == {
        "brand/marvis-app-icon-192.png",
        "brand/marvis-app-icon-512.png",
    }
    for source in icon_sources:
        assert (Path("marvis/static") / source).is_file()


def test_builtin_stochastic_tool_manifests_declare_seed_inputs():
    for manifest_path in sorted(Path("marvis/packs").glob("*/manifest.json")):
        manifest = parse_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            builtin=True,
        )
        for tool in manifest.tools:
            if tool.determinism == "stochastic":
                assert "seed" in tool.input_schema["properties"], f"{manifest.name}.{tool.name}"
