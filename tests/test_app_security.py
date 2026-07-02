import json
import os
from pathlib import Path
import re

from fastapi.testclient import TestClient

from marvis import __version__
from marvis.app import _is_local_client, _static_asset_version, create_app
from marvis.data.backend import DUCKDB_TEMP_DIR_NAME
from marvis.db import PluginRepository, connect, init_db


def test_create_app_refreshes_stale_builtin_manifest_before_plugin_registry_load(tmp_path):
    db_path = tmp_path / "marvis.sqlite"
    init_db(db_path)
    manifest_path = Path(__file__).parents[1] / "marvis" / "packs" / "feature" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "write:artifact" in manifest["permissions"]
    manifest["permissions"] = [
        permission for permission in manifest["permissions"] if permission != "write:artifact"
    ]

    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plugins(
                name, version, display_name, description, module,
                manifest_json, checksum, builtin, enabled, installed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                manifest["name"],
                manifest["version"],
                manifest["display_name"],
                manifest["description"],
                manifest["module"],
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
                "",
                1,
                1,
                "2026-06-29T00:00:00Z",
            ),
        )

    create_app(tmp_path)

    refreshed = PluginRepository(db_path).get_plugin("feature")
    assert refreshed is not None
    refreshed_manifest = json.loads(refreshed["manifest_json"])
    assert "write:artifact" in refreshed_manifest["permissions"]


def test_remote_read_does_not_leak_validator_aliases_via_branding(tmp_path, monkeypatch):
    # /api/branding carries private workspace branding incl. validator aliases
    # (real names). Even with remote read enabled, a remote client must not read it
    # (the branding asset files are already local-only).
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "brand.json").write_text(
        json.dumps({"validator_aliases": {"张三": "审核员A"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    app = create_app(tmp_path)

    remote = TestClient(app, client=("203.0.113.9", 5555))
    response = remote.get("/api/branding")

    assert response.status_code == 403
    assert "审核员A" not in response.text
    assert "张三" not in response.text
    # local clients still get full branding (incl. aliases) for the agent display name
    local_payload = TestClient(app).get("/api/branding").json()
    assert local_payload["validatorAliases"] == {"张三": "审核员A"}


def test_is_local_client_accepts_ipv6_loopback_and_mapped_forms():
    assert _is_local_client("127.0.0.1") is True
    assert _is_local_client("::1") is True
    # IPv4-mapped IPv6 loopback must unwrap to a loopback verdict.
    assert _is_local_client("::ffff:127.0.0.1") is True
    assert _is_local_client("localhost") is True
    assert _is_local_client("203.0.113.7") is False
    assert _is_local_client("not-an-ip") is False
    assert _is_local_client(None) is False


def test_remote_client_cannot_read_task_data_by_default(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app, client=("192.168.1.20", 43210))

    response = client.get("/api/tasks")

    assert response.status_code == 403
    assert response.json()["detail"] == "API access is limited to local clients"


def test_remote_client_can_read_health_check(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app, client=("192.168.1.20", 43210))

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["sqlite_journal_mode"] in {"wal", "memory"}
    assert payload["sqlite_wal_degraded"] is False
    assert isinstance(payload["sqlite_busy_timeout_ms"], int)


def test_health_check_reports_llm_configured_false_when_unset(tmp_path):
    app = create_app(tmp_path)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is False


def test_health_check_reports_llm_configured_true_when_enabled_model_saved(tmp_path):
    from marvis.llm_settings import save_llm_settings

    save_llm_settings(
        tmp_path,
        {
            "default_model_id": "model-a",
            "models": [
                {
                    "model_id": "model-a",
                    "enabled": True,
                    "api_base_url": "https://example.test/v1",
                    "model_name": "gpt",
                    "api_key": "secret",
                }
            ],
        },
    )
    app = create_app(tmp_path)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is True


def test_health_check_surfaces_duckdb_runtime_config(tmp_path):
    """PERF-8: /api/health must expose the DuckDB memory_limit / threads /
    temp_directory actually applied, so an operator can confirm the JOIN engine
    is not still running on DuckDB's own unconfigured (all-cores, ~80%-RAM)
    defaults."""
    app = create_app(tmp_path)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["duckdb_memory_limit"]
    assert payload["duckdb_threads"]
    assert payload["duckdb_temp_directory"] == str(tmp_path / DUCKDB_TEMP_DIR_NAME)


def test_health_check_surfaces_sqlite_wal_degradation(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "marvis.app.sqlite_health",
        lambda _db_path: {
            "sqlite_journal_mode": "delete",
            "sqlite_wal_degraded": True,
            "sqlite_busy_timeout_ms": 5000,
        },
    )
    app = create_app(tmp_path)

    response = TestClient(app).get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    # This test only cares that sqlite_health()'s output is surfaced verbatim;
    # duckdb_health() (PERF-8) contributes its own independent keys, asserted in
    # test_health_check_surfaces_duckdb_runtime_config below.
    assert payload["status"] == "ok"
    assert payload["stuck_jobs"] == 0
    assert payload["sqlite_journal_mode"] == "delete"
    assert payload["sqlite_wal_degraded"] is True
    assert payload["sqlite_busy_timeout_ms"] == 5000


def test_remote_client_can_read_api_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    client = TestClient(app, client=("192.168.1.20", 43210))

    response = client.get("/api/tasks")

    assert response.status_code == 200
    assert response.json() == []


def test_spoofed_forwarded_header_is_ignored_without_trusted_proxy(tmp_path):
    # A remote client connecting directly (no configured proxy) cannot forge
    # locality with X-Forwarded-For — forwarded headers are only honored from an
    # explicitly trusted proxy.
    app = create_app(tmp_path)
    client = TestClient(app, client=("192.168.1.20", 43210))

    response = client.get("/api/tasks", headers={"X-Forwarded-For": "127.0.0.1"})

    assert response.status_code == 403
    assert response.json()["detail"] == "API access is limited to local clients"


def test_untrusted_loopback_proxy_does_not_inherit_local_access(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 43210))

    response = client.get("/api/tasks", headers={"X-Forwarded-For": "203.0.113.7"})

    assert response.status_code == 403
    assert response.json()["detail"] == "API access is limited to local clients"


def test_trusted_proxy_forwards_remote_client_and_guard_still_applies(tmp_path, monkeypatch):
    # Behind a same-host reverse proxy the TCP peer is the proxy's loopback IP.
    # With the proxy trusted, the real (remote) client is read from the forwarded
    # header so the local-only guard is not bypassed.
    monkeypatch.setenv("MARVIS_TRUSTED_PROXY_HOSTS", "127.0.0.1")
    app = create_app(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 43210))

    blocked_read = client.get("/api/tasks", headers={"X-Forwarded-For": "203.0.113.7"})
    blocked_write = client.post("/api/tasks", headers={"X-Forwarded-For": "203.0.113.7"}, json={})

    assert blocked_read.status_code == 403
    assert blocked_read.json()["detail"] == "API access is limited to local clients"
    assert blocked_write.status_code == 403
    assert blocked_write.json()["detail"] == "unsafe API methods are limited to local clients"


def test_trusted_proxy_forwards_local_client_as_local(tmp_path, monkeypatch):
    monkeypatch.setenv("MARVIS_TRUSTED_PROXY_HOSTS", "127.0.0.1")
    app = create_app(tmp_path)
    client = TestClient(app, client=("127.0.0.1", 43210))

    response = client.get("/api/tasks", headers={"X-Forwarded-For": "127.0.0.1"})

    assert response.status_code == 200
    assert response.json() == []


def test_remote_read_does_not_expose_settings_or_branding(tmp_path, monkeypatch):
    # MARVIS_ALLOW_REMOTE_READ opens read-only data APIs but never system settings
    # or private branding assets, which can leak local paths / configuration.
    monkeypatch.setenv("MARVIS_ALLOW_REMOTE_READ", "1")
    app = create_app(tmp_path)
    client = TestClient(app, client=("192.168.1.20", 43210))

    settings_response = client.get("/api/settings")
    branding_response = client.get("/branding/assets/logo.png")

    assert settings_response.status_code == 403
    assert settings_response.json()["detail"] == "this endpoint is limited to local clients"
    assert branding_response.status_code == 403
    assert branding_response.json()["detail"] == "this endpoint is limited to local clients"


def test_remote_index_uses_default_branding_instead_of_workspace_branding(tmp_path):
    branding_dir = tmp_path / "branding"
    branding_dir.mkdir()
    (branding_dir / "brand.json").write_text(
        json.dumps(
            {
                "platform_name": "私有机构风控平台",
                "browser_title": "私有机构",
                "primary_color": "#123456",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    app = create_app(tmp_path)

    response = TestClient(app, client=("203.0.113.9", 5555)).get("/")

    assert response.status_code == 200
    assert "私有机构风控平台" not in response.text
    assert "私有机构" not in response.text
    assert "MARVIS-全能风控智能体" in response.text


def test_index_replaces_static_asset_version_placeholder(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert f"static/app.js?v={__version__}-" in response.text
    assert "__MARVIS_STATIC_VERSION__" not in response.text


def test_static_asset_version_changes_when_any_v2_module_changes(tmp_path):
    # PERF-9 regression: the version hash used to be derived from a
    # hardcoded 4-file allowlist (app.js/styles.css/welcome.css/
    # v2-workbench.css), so editing any js/v2/*.js controller left the
    # version string -- and therefore the cache-busted URL -- unchanged,
    # and browsers could keep running the stale module indefinitely.
    static_dir = tmp_path / "static"
    (static_dir / "js" / "v2").mkdir(parents=True)
    (static_dir / "app.js").write_text("// app", encoding="utf-8")
    (static_dir / "js" / "v2" / "screen_gate_controller.js").write_text(
        "// original", encoding="utf-8"
    )

    before = _static_asset_version(static_dir)

    edited = static_dir / "js" / "v2" / "screen_gate_controller.js"
    edited.write_text("// edited", encoding="utf-8")
    os.utime(edited, (edited.stat().st_atime, edited.stat().st_mtime + 5))

    after = _static_asset_version(static_dir)

    assert before != after


def test_static_import_map_covers_every_js_module_and_is_valid_json(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    response = client.get("/")
    html = response.text

    assert "__MARVIS_STATIC_IMPORT_MAP__" not in html
    start = html.index('<script type="importmap">') + len('<script type="importmap">')
    end = html.index("</script>", start)
    import_map = json.loads(html[start : end])

    real_static_dir = Path(__file__).resolve().parents[1] / "marvis" / "static"
    js_files = sorted(p.name for p in (real_static_dir / "js").glob("*.js"))
    v2_files = sorted(p.name for p in (real_static_dir / "js" / "v2").glob("*.js"))

    for name in js_files:
        assert f"./js/{name}" in import_map["imports"]
        assert import_map["imports"][f"./js/{name}"].startswith(f"static/js/{name}?v=")
    for name in v2_files:
        assert f"./js/v2/{name}" in import_map["imports"]
        assert import_map["scopes"]["static/js/v2/"][f"./{name}"].startswith(
            f"static/js/v2/{name}?v="
        )


def test_static_response_cache_control_depends_on_version_query_param(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)

    unversioned = client.get("/static/app.js")
    assert unversioned.status_code == 200
    assert unversioned.headers["cache-control"] == "no-cache"

    index_html = client.get("/").text
    m = re.search(r'src="(static/app\.js\?v=[^"]+)"', index_html)
    assert m is not None
    versioned = client.get("/" + m.group(1))
    assert versioned.status_code == 200
    assert versioned.headers["cache-control"] == "public, max-age=31536000, immutable"
