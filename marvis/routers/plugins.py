from __future__ import annotations

from pathlib import Path
import re
import tempfile

from fastapi import APIRouter, HTTPException, Request

from marvis.plugins.errors import (
    DuplicatePluginError,
    ManifestError,
    PluginError,
    PluginNotFoundError,
)
from marvis.plugins.loader import install_plugin
from marvis.plugins.manifest import PluginManifest, ToolSpec


router = APIRouter(prefix="/api/plugins", tags=["plugins"])


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
    settings = request.app.state.settings
    registry = request.app.state.plugin_registry
    filename, content = await _read_plugin_upload(request)
    with tempfile.TemporaryDirectory(prefix="marvis-plugin-upload-") as temp_name:
        upload_path = Path(temp_name) / filename
        upload_path.write_bytes(content)
        try:
            manifest = install_plugin(upload_path, settings.plugins_dir, registry)
        except DuplicatePluginError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ManifestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
    return _set_enabled(request, name, True)


@router.post("/{name}/disable")
def disable_plugin(request: Request, name: str) -> dict:
    return _set_enabled(request, name, False)


@router.delete("/{name}")
def remove_plugin(request: Request, name: str) -> dict:
    try:
        request.app.state.plugin_registry.remove(name)
    except PluginNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {"ok": True}


@router.get("/{name}/tools")
def list_plugin_tools(request: Request, name: str) -> dict:
    try:
        manifest = request.app.state.plugin_registry.get(name)
    except PluginNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"tools": [_public_tool(tool) for tool in manifest.tools]}


def _set_enabled(request: Request, name: str, enabled: bool) -> dict:
    try:
        request.app.state.plugin_registry.set_enabled(name, enabled)
    except PluginNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.hook_dispatcher.rebuild_index()
    return {"ok": True}


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
        "side_effects": list(tool.side_effects),
    }


async def _read_plugin_upload(request: Request) -> tuple[str, bytes]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if not content_type.startswith("multipart/form-data"):
        return "plugin.zip", body

    boundary_match = re.search(r"boundary=([^;]+)", content_type)
    if boundary_match is None:
        raise HTTPException(status_code=400, detail="multipart boundary is missing")
    boundary = boundary_match.group(1).strip().strip('"').encode("utf-8")
    delimiter = b"--" + boundary
    for part in body.split(delimiter):
        if b"\r\n\r\n" not in part:
            continue
        header_blob, payload = part.split(b"\r\n\r\n", 1)
        if b'name="file"' not in header_blob:
            continue
        filename = _filename_from_content_disposition(header_blob) or "plugin.zip"
        payload = payload.rstrip(b"\r\n")
        if payload.endswith(b"--"):
            payload = payload[:-2].rstrip(b"\r\n")
        return filename, payload
    raise HTTPException(status_code=400, detail="file field is required")


def _filename_from_content_disposition(header_blob: bytes) -> str | None:
    text = header_blob.decode("utf-8", errors="replace")
    match = re.search(r'filename="([^"]+)"', text)
    if match is None:
        return None
    return Path(match.group(1)).name
