from pathlib import Path
import tomllib


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
        "branding.js",
        "dialogs.js",
        "polling.js",
        "render-agent.js",
        "render-metrics.js",
        "state.js",
        "ui-utils.js",
    ):
        assert (static_js / module_name).is_file()


def test_static_css_module_files_exist_for_declared_links():
    assert Path("marvis/static/css/welcome.css").is_file()
