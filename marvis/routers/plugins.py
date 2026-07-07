from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
import re
import tempfile

from fastapi import APIRouter, HTTPException, Request
from marvis.errors import bad_request, conflict, forbidden, not_found, unprocessable

from marvis.plugins.errors import (
    DuplicatePluginError,
    ManifestError,
    PluginError,
    PluginNotFoundError,
)
from marvis.plugins.loader import install_plugin
from marvis.plugins.manifest import PluginManifest, ToolSpec


router = APIRouter(prefix="/api/plugins", tags=["plugins"])
PLUGIN_ADMIN_HEADER = "x-marvis-plugin-admin"


def ensure_plugin_admin_token(token_path: Path) -> str:
    """Read the per-workspace plugin-admin token, generating it on first use.

    Plugin install/enable/disable/delete (and draft promote/reject) mutate the
    running tool library, so gating them on a fixed magic header ("local-dev")
    meant any process that could reach loopback and knew that string could run
    arbitrary code. Instead the secret is a random token minted into the
    workspace on first startup and stored in a 0600 file: the file's owner-only
    permission is the single-user isolation boundary (consistent with the LT-9
    threat model). Callers compare the presented header against this value with
    hmac.compare_digest. Regenerating simply means deleting the file.
    """
    try:
        existing = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    token = secrets.token_hex(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    # Create with 0600 from the start (never a wider window), then write.
    fd = os.open(str(token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, token.encode("utf-8"))
    finally:
        os.close(fd)
    # Defensively re-assert 0600 in case a prior umask-created file existed.
    os.chmod(token_path, 0o600)
    return token


@router.get("")
def list_plugins(request: Request, include_disabled: bool = False) -> dict:
    registry = request.app.state.plugin_registry
    return {
        "plugins": [
            _public_plugin(manifest, registry.is_enabled(manifest.name))
            for manifest in registry.list(include_disabled=include_disabled)
        ]
    }


@router.post("", status_code=201)
async def upload_plugin(request: Request) -> dict:
    _require_plugin_admin(request)
    settings = request.app.state.settings
    registry = request.app.state.plugin_registry
    filename, content = await _read_plugin_upload(request)
    with tempfile.TemporaryDirectory(prefix="marvis-plugin-upload-") as temp_name:
        upload_path = Path(temp_name) / filename
        upload_path.write_bytes(content)
        try:
            manifest = install_plugin(upload_path, settings.plugins_dir, registry)
        except DuplicatePluginError as exc:
            raise conflict(str(exc)) from exc
        except ManifestError as exc:
            raise unprocessable(str(exc)) from exc
        except PluginError as exc:
            detail = str(exc)
            status_code = 422 if "invalid json schema" in detail else 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {
        "name": manifest.name,
        "version": manifest.version,
        "tool_count": len(manifest.tools),
    }


@router.post("/{name}/enable")
def enable_plugin(request: Request, name: str) -> dict:
    _require_plugin_admin(request)
    return _set_enabled(request, name, True)


@router.post("/{name}/disable")
def disable_plugin(request: Request, name: str) -> dict:
    _require_plugin_admin(request)
    return _set_enabled(request, name, False)


@router.delete("/{name}")
def remove_plugin(request: Request, name: str) -> dict:
    _require_plugin_admin(request)
    try:
        request.app.state.plugin_registry.remove(name)
    except PluginNotFoundError as exc:
        raise not_found(str(exc)) from exc
    except ValueError as exc:
        raise bad_request(str(exc)) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {"ok": True}


@router.get("/{name}/tools")
def list_plugin_tools(request: Request, name: str) -> dict:
    try:
        manifest = request.app.state.plugin_registry.get(name)
    except PluginNotFoundError as exc:
        raise not_found(str(exc)) from exc
    return {
        "module": manifest.module,
        "permissions": list(manifest.permissions),
        "hooks": [{"event": hook.event, "tool": hook.tool} for hook in manifest.hooks],
        "tools": [_public_tool(tool) for tool in manifest.tools],
    }


def _set_enabled(request: Request, name: str, enabled: bool) -> dict:
    try:
        request.app.state.plugin_registry.set_enabled(name, enabled)
    except PluginNotFoundError as exc:
        raise not_found(str(exc)) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {"ok": True}


def _require_plugin_admin(request: Request) -> None:
    expected = getattr(request.app.state, "plugin_admin_token", "") or ""
    presented = request.headers.get(PLUGIN_ADMIN_HEADER, "")
    if not expected or not hmac.compare_digest(presented, expected):
        raise forbidden("plugin admin confirmation required")


def _public_plugin(manifest: PluginManifest, enabled: bool) -> dict:
    return {
        "name": manifest.name,
        "version": manifest.version,
        "display_name": manifest.display_name,
        "description": manifest.description,
        "enabled": enabled,
        "builtin": manifest.builtin,
        "tool_count": len(manifest.tools),
    }


def _public_tool(tool: ToolSpec) -> dict:
    return {
        "name": tool.name,
        "summary": tool.summary,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
        "determinism": tool.determinism,
        "timeout_seconds": tool.timeout_seconds,
        "failure_policy": tool.failure_policy,
        "entrypoint": tool.entrypoint,
        "memory_limit_mb": tool.memory_limit_mb,
        "side_effects": list(tool.side_effects),
    }


async def _read_plugin_upload(request: Request) -> tuple[str, bytes]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if not content_type.startswith("multipart/form-data"):
        return "plugin.zip", body

    boundary_match = re.search(r"boundary=([^;]+)", content_type)
    if boundary_match is None:
        raise bad_request("multipart boundary is missing")
    boundary = boundary_match.group(1).strip().strip('"').encode("utf-8")
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        if b"\r\n\r\n" not in part:
            continue
        header_blob, payload = part.split(b"\r\n\r\n", 1)
        if b'name="file"' not in header_blob:
            continue
        filename = _filename_from_content_disposition(header_blob) or "plugin.zip"
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        if payload.endswith(b"--"):
            payload = payload[:-2]
            if payload.endswith(b"\r\n"):
                payload = payload[:-2]
        return filename, payload
    raise bad_request("file field is required")


def _filename_from_content_disposition(header_blob: bytes) -> str | None:
    text = header_blob.decode("utf-8", errors="replace")
    match = re.search(r'filename="([^"]+)"', text)
    if match is None:
        return None
    return Path(match.group(1)).name
