import asyncio
import json
import zipfile

from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.plugins.manifest import ToolRef
from marvis.routers.plugins import _read_plugin_upload


PLUGIN_ADMIN_HEADERS = {"X-MARVIS-Plugin-Admin": "local-dev"}


class FakeUploadRequest:
    def __init__(self, body: bytes, boundary: bytes):
        self.headers = {"content-type": f"multipart/form-data; boundary={boundary.decode()}"}
        self._body = body

    async def body(self) -> bytes:
        return self._body


def _manifest(name: str = "uploaded_pack", *, bad: bool = False) -> dict:
    schema = {"type": "not-a-json-schema-type"} if bad else {
        "type": "object",
        "properties": {},
        "required": [],
    }
    return {
        "name": name,
        "version": "0.1.0",
        "display_name": "Uploaded Pack",
        "description": "Plugin API test pack",
        "module": f"{name}.tools",
        "tools": [
            {
                "name": "noop",
                "summary": "No-op",
                "input_schema": schema,
                "output_schema": schema,
                "determinism": "deterministic",
                "timeout_seconds": 10,
                "failure_policy": "fail",
                "entrypoint": "tool_noop",
            }
        ],
        "hooks": [],
        "permissions": [],
    }


def _zip_bytes(manifest: dict, *, comment: bytes = b"") -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("uploaded_pack/manifest.json", json.dumps(manifest))
        archive.writestr("uploaded_pack/tools.py", "def tool_noop(inputs, ctx):\n    return {}\n")
        archive.comment = comment
    return buffer.getvalue()


def test_app_startup_registers_sample_plugin(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/plugins")

    assert response.status_code == 200
    payload = response.json()
    assert payload["plugins"][0]["name"] == "_sample"
    assert payload["plugins"][0]["enabled"] is True
    assert payload["plugins"][0]["builtin"] is True


def test_plugin_tools_endpoint_exposes_schema_without_entrypoints(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/plugins/_sample/tools")

    assert response.status_code == 200
    tools = response.json()["tools"]
    assert {tool["name"] for tool in tools} >= {"echo", "sleep"}
    assert "input_schema" in tools[0]
    assert "entrypoint" not in tools[0]


def test_builtin_plugin_can_be_disabled_and_enabled(tmp_path):
    client = TestClient(create_app(tmp_path))

    disabled = client.post("/api/plugins/_sample/disable", headers=PLUGIN_ADMIN_HEADERS)
    listed = client.get("/api/plugins?include_disabled=true")
    enabled = client.post("/api/plugins/_sample/enable", headers=PLUGIN_ADMIN_HEADERS)

    assert disabled.status_code == 200
    assert listed.json()["plugins"][0]["enabled"] is False
    assert enabled.status_code == 200
    assert client.get("/api/plugins").json()["plugins"][0]["enabled"] is True


def test_builtin_plugin_delete_is_rejected(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.delete("/api/plugins/_sample", headers=PLUGIN_ADMIN_HEADERS)

    assert response.status_code == 400
    assert "builtin" in response.json()["detail"]


def test_upload_plugin_zip_installs_and_rejects_duplicate(tmp_path):
    client = TestClient(create_app(tmp_path))
    archive = _zip_bytes(_manifest())

    created = client.post(
        "/api/plugins",
        files={"file": ("uploaded_pack.zip", archive, "application/zip")},
        headers=PLUGIN_ADMIN_HEADERS,
    )
    duplicate = client.post(
        "/api/plugins",
        files={"file": ("uploaded_pack.zip", archive, "application/zip")},
        headers=PLUGIN_ADMIN_HEADERS,
    )

    assert created.status_code == 201
    assert created.json()["name"] == "uploaded_pack"
    assert duplicate.status_code == 409

    result = client.app.state.tool_runner.invoke(
        ToolRef("uploaded_pack", "noop"),
        {},
        task_id="task-1",
    )
    assert result.ok is True
    assert result.output == {}


def test_upload_bad_manifest_returns_422(tmp_path):
    client = TestClient(create_app(tmp_path))

    response = client.post(
        "/api/plugins",
        files={"file": ("bad.zip", _zip_bytes(_manifest(bad=True)), "application/zip")},
        headers=PLUGIN_ADMIN_HEADERS,
    )

    assert response.status_code == 422


def test_upload_plugin_preserves_zip_trailing_bytes(tmp_path):
    client = TestClient(create_app(tmp_path))
    archive = _zip_bytes(_manifest(), comment=b"\r\n")

    response = client.post(
        "/api/plugins",
        files={"file": ("uploaded_pack.zip", archive, "application/zip")},
        headers=PLUGIN_ADMIN_HEADERS,
    )

    assert response.status_code == 201


def test_read_plugin_upload_preserves_payload_trailing_crlf_bytes():
    boundary = b"MARVISBOUNDARY"
    payload = b"zip payload bytes ending in crlf\r\n"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="plugin.zip"\r\n'
        b"Content-Type: application/zip\r\n\r\n"
        + payload
        + b"\r\n--" + boundary + b"--\r\n"
    )

    filename, content = asyncio.run(_read_plugin_upload(FakeUploadRequest(body, boundary)))

    assert filename == "plugin.zip"
    assert content == payload


def test_plugin_mutations_require_admin_header(tmp_path):
    client = TestClient(create_app(tmp_path))

    upload = client.post(
        "/api/plugins",
        files={"file": ("uploaded_pack.zip", _zip_bytes(_manifest()), "application/zip")},
    )
    disable = client.post("/api/plugins/_sample/disable")
    delete = client.delete("/api/plugins/_sample")

    assert upload.status_code == 403
    assert disable.status_code == 403
    assert delete.status_code == 403
