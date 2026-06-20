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
        self.v2_runtime_mount_attrs: dict[str, str | None] = {}
        self.v2_workspace_dialog_attrs: dict[str, str | None] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = {name: value for name, value in attrs}
        if tag == "dialog" and attr_map.get("id") == "v2WorkspaceDialog":
            self.v2_workspace_dialog_attrs = attr_map
        if tag == "div" and attr_map.get("id") == "v2RuntimeMount":
            self.v2_runtime_mount_attrs = attr_map
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
    assert parser.module_srcs == [f"static/app.js?v={__version__}"]
    assert parser.v2_workspace_dialog_attrs["aria-labelledby"] == "v2WorkspaceTitle"
    assert "hidden" not in parser.v2_runtime_mount_attrs
    assert parser.v2_runtime_mount_attrs["aria-label"] == "V2 工作台"
    assert 'id="openV2WorkspaceButton"' in index_response.text
    assert 'id="closeV2WorkspaceButton"' in index_response.text
    assert 'id="refreshV2PluginsButton"' in index_response.text
    assert 'id="refreshV2SkillsButton"' in index_response.text
    assert 'id="refreshV2CapabilityButton"' in index_response.text
    app_response = client.get(f"/static/app.js?v={__version__}")
    assert app_response.status_code == 200
    assert "function openV2WorkspaceDialog" in app_response.text
    assert "function closeV2WorkspaceDialog" in app_response.text
    assert "async function refreshV2Plugins" in app_response.text
    assert "async function refreshV2Skills" in app_response.text
    assert "async function refreshV2Capability" in app_response.text
    assert "mountV2(root, { taskId: () => selectedTaskId })" in app_response.text
    assert '$("openV2WorkspaceButton").onclick = openV2WorkspaceDialog;' in app_response.text
    assert '$("closeV2WorkspaceButton").onclick = closeV2WorkspaceDialog;' in app_response.text
    assert parser.stylesheet_hrefs == [
        f"static/styles.css?v={__version__}",
        f"static/css/welcome.css?v={__version__}",
        f"static/css/v2-workbench.css?v={__version__}",
    ]
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
    assert "/static/js/v2/main_v2.js" in loaded_modules
    assert "/static/js/v2/plan_view.js" in loaded_modules
    assert "/static/js/v2/subagent_view.js" in loaded_modules


def _resolve_relative_module(base_path: str, specifier: str) -> str:
    normalized = posixpath.normpath(
        posixpath.join(posixpath.dirname(base_path), specifier)
    )
    return urljoin("/", normalized)
