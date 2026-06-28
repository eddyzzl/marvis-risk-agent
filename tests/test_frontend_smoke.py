from __future__ import annotations

from html.parser import HTMLParser
import posixpath
import re
from urllib.parse import urljoin, urlparse

from fastapi.testclient import TestClient

from marvis import __version__
from marvis.app import create_app


_STATIC_IMPORT_RE = re.compile(
    r"""(?:import|export)\s+(?:[^'"]*?\s+from\s+)?["']([^"']+)["']"""
)


class _ModuleScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.module_srcs: list[str] = []
        self.stylesheet_hrefs: list[str] = []
        self.governance_extension_mount_attrs: dict[str, str | None] = {}
        self.governance_settings_dialog_attrs: dict[str, str | None] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = {name: value for name, value in attrs}
        if tag == "dialog" and attr_map.get("id") == "governanceSettingsDialog":
            self.governance_settings_dialog_attrs = attr_map
        if tag == "div" and attr_map.get("id") == "governanceExtensionMount":
            self.governance_extension_mount_attrs = attr_map
        if tag != "script":
            if tag == "link" and attr_map.get("rel") == "stylesheet":
                self.stylesheet_hrefs.append(str(attr_map.get("href") or ""))
            return
        if attr_map.get("type") == "module" and attr_map.get("src"):
            self.module_srcs.append(str(attr_map["src"]))


def test_frontend_entrypoint_serves_declared_es_modules(tmp_path):
    client = TestClient(create_app(tmp_path))
    index_response = client.get("/")
    assert index_response.status_code == 200
    assert "模型开发" in index_response.text
    assert "模型验证" in index_response.text
    assert "策略开发" in index_response.text

    parser = _ModuleScriptParser()
    parser.feed(index_response.text)
    assert len(parser.module_srcs) == 1
    assert parser.module_srcs[0].startswith(f"static/app.js?v={__version__}-")
    assert parser.governance_settings_dialog_attrs["aria-labelledby"] == "governanceSettingsTitle"
    assert "hidden" not in parser.governance_extension_mount_attrs
    assert parser.governance_extension_mount_attrs["aria-label"] == "扩展设置"
    assert 'id="openGovernanceSettingsButton"' in index_response.text
    assert 'id="closeGovernanceSettingsButton"' in index_response.text
    # The old standalone V2 workspace dialog was retired: plugins / workflows /
    # capabilities now share a single extension settings mount and a single
    # context-aware refresh button driven by the governance nav (plugins / workflows /
    # capabilities). Assert the current IA, not the removed button ids / dialog funcs.
    assert 'id="governanceRefreshButton"' in index_response.text
    assert 'data-governance-nav="plugins"' in index_response.text
    assert 'data-governance-nav="workflows"' in index_response.text
    assert 'data-governance-nav="capabilities"' in index_response.text
    app_response = client.get("/" + parser.module_srcs[0])
    assert app_response.status_code == 200
    assert "function refreshActiveGovernancePanel" in app_response.text
    assert "runGovernanceExtensionAction(refreshGovernancePlugins)" in app_response.text
    assert "runGovernanceExtensionAction(refreshGovernanceSkills)" in app_response.text
    assert "runGovernanceExtensionAction(refreshGovernanceCapability)" in app_response.text
    assert "async function refreshGovernancePlugins" in app_response.text
    assert "async function refreshGovernanceSkills" in app_response.text
    assert "async function refreshGovernanceCapability" in app_response.text
    assert "mountGovernanceExtensionPanels(root, governanceExtensionActions())" in app_response.text
    assert '$("openGovernanceSettingsButton").addEventListener("pointerdown", handleGovernanceSettingsPointerDown, true);' in app_response.text
    assert '$("openGovernanceSettingsButton").onclick' in app_response.text
    assert '$("closeGovernanceSettingsButton").onclick = closeGovernanceSettingsDialog;' in app_response.text
    assert '$("governanceRefreshButton").onclick = refreshActiveGovernancePanel;' in app_response.text
    assert len(parser.stylesheet_hrefs) == 3
    assert parser.stylesheet_hrefs[0].startswith(f"static/styles.css?v={__version__}-")
    assert parser.stylesheet_hrefs[1].startswith(f"static/css/welcome.css?v={__version__}-")
    assert parser.stylesheet_hrefs[2].startswith(f"static/css/v2-workbench.css?v={__version__}-")
    for href in parser.stylesheet_hrefs:
        response = client.get("/" + href)
        assert response.status_code == 200, href

    visited: set[str] = set()
    pending = ["/" + parser.module_srcs[0]]
    loaded_modules: list[str] = []
    while pending:
        raw_url = pending.pop()
        path = urlparse(raw_url).path
        if path in visited:
            continue
        visited.add(path)
        response = client.get(raw_url)
        assert response.status_code == 200, raw_url
        loaded_modules.append(path)
        for import_specifier in _STATIC_IMPORT_RE.findall(response.text):
            if not import_specifier.startswith("."):
                continue
            module_url = _resolve_relative_module(path, import_specifier)
            pending.append(module_url)

    assert "/static/app.js" in loaded_modules
    assert "/static/js/api.js" in loaded_modules
    assert "/static/js/branding.js" in loaded_modules
    assert "/static/js/dialogs.js" in loaded_modules
    assert "/static/js/render-agent.js" in loaded_modules
    assert "/static/js/state.js" in loaded_modules
    assert "/static/js/ui-utils.js" in loaded_modules
    assert "/static/js/v2/governance_extensions.js" in loaded_modules
    assert "/static/js/v2/plan_view.js" not in loaded_modules
    assert "/static/js/v2/subagent_view.js" not in loaded_modules


def _resolve_relative_module(base_path: str, specifier: str) -> str:
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(base_path), specifier)
    )
    return urljoin("/", normalized)
