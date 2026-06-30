import json
from pathlib import Path
import tomllib

from marvis.plugins.manifest import parse_manifest


def test_static_es_modules_are_declared_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]["marvis"]

    assert "static/js/*" in package_data
    assert "static/css/*" in package_data
    assert "packs/*/manifest.json" in package_data


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
        "polling.js",
        "render-agent.js",
        "render-metrics.js",
        "state.js",
        "ui-utils.js",
    ):
        assert (static_js / module_name).is_file()


def test_static_css_module_files_exist_for_declared_links():
    assert Path("marvis/static/css/welcome.css").is_file()


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
