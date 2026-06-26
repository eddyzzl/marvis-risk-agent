from __future__ import annotations

from html import escape
from pathlib import Path
import json
import re
from typing import Any
from urllib.parse import quote, unquote


DEFAULT_BRANDING: dict[str, Any] = {
    "platformName": "MARVIS-全能风控智能体",
    "browserTitle": "MARVIS-全能风控智能体",
    "primaryColor": "#303034",
    "logoUrl": "static/brand/marvis-logo.png?v=20260624-gauge",
    "workspaceLogoUrl": "static/brand/marvis-workspace-logo.png?v=20260624-gauge",
    "faviconUrl": "static/brand/marvis-favicon.png?v=20260624-gauge",
    # Optional per-workspace map of real validator name -> display alias, used for
    # the agent's display name. Kept out of the public static JS so real names live
    # only in the deployment's own brand.json, never in the shipped bundle.
    "validatorAliases": {},
    "source": "default",
}

BRANDING_DIR_NAME = "branding"
BRANDING_CONFIG_NAME = "brand.json"
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def load_branding(workspace: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace)
    config_path = branding_config_path(workspace_path)
    config = _load_config(config_path)
    if config is None:
        return dict(DEFAULT_BRANDING)

    result = dict(DEFAULT_BRANDING)
    result["source"] = "workspace"
    result["validatorAliases"] = _validator_aliases(
        _first(config, "validator_aliases", "validatorAliases")
    )
    result["platformName"] = _text_field(
        config, "platform_name", "platformName", fallback=result["platformName"]
    )
    result["browserTitle"] = _text_field(
        config, "browser_title", "browserTitle", fallback=result["browserTitle"]
    )
    result["primaryColor"] = _color_field(
        config, "primary_color", "primaryColor", fallback=result["primaryColor"]
    )
    logo_config = _first(config, "logo", "logo_path", "logoPath")
    result["logoUrl"] = _asset_url(
        workspace_path,
        logo_config,
        fallback=result["logoUrl"],
    )
    workspace_logo_config = _first(
        config,
        "workspace_logo",
        "workspaceLogo",
        "workspace_logo_path",
        "workspaceLogoPath",
    )
    result["workspaceLogoUrl"] = _asset_url(
        workspace_path,
        workspace_logo_config,
        fallback=(
            result["logoUrl"]
            if result["logoUrl"] != DEFAULT_BRANDING["logoUrl"]
            else result["workspaceLogoUrl"]
        ),
    )
    result["faviconUrl"] = _asset_url(
        workspace_path,
        _first(config, "favicon", "favicon_path", "faviconPath"),
        fallback=result["faviconUrl"],
    )
    return result


def render_branded_index_html(index_html: str, branding: dict[str, str]) -> str:
    current = _normalized_branding(branding)
    platform_name = escape(current["platformName"], quote=True)
    browser_title = escape(current["browserTitle"], quote=True)
    logo_url = escape(current["logoUrl"], quote=True)
    workspace_logo_url = escape(current["workspaceLogoUrl"], quote=True)
    favicon_url = escape(current["faviconUrl"], quote=True)
    favicon_type = escape(_image_mime_type(current["faviconUrl"]), quote=True)
    primary_color = current["primaryColor"]
    primary_hover = _brand_hover_color(primary_color)

    html = index_html.replace(
        '<html lang="zh-CN">',
        (
            '<html lang="zh-CN" '
            f'style="--brand-primary: {primary_color}; '
            f'--brand-primary-hover: {primary_hover};">'
        ),
        1,
    )
    for title_placeholder in (
        "<title>MARVIS-全能风控智能体</title>",
        "<title>MARVIS-Agent</title>",
    ):
        html = html.replace(title_placeholder, f"<title>{browser_title}</title>", 1)
    html = html.replace(
        'type="image/png" href="static/brand/marvis-favicon.png"',
        f'type="{favicon_type}" href="{favicon_url}"',
        1,
    )
    html = html.replace(
        'id="brandLogo"\n              class="brand-mark"\n              src="static/brand/marvis-logo.png"',
        f'id="brandLogo"\n              class="brand-mark"\n              src="{logo_url}"',
        1,
    )
    html = html.replace(
        'id="workspaceBrandLogo"\n              class="workspace-brand-logo"\n              src="static/brand/marvis-workspace-logo.png"',
        f'id="workspaceBrandLogo"\n              class="workspace-brand-logo"\n              src="{workspace_logo_url}"',
        1,
    )
    for alt_placeholder in (
        'alt="MARVIS-全能风控智能体 logo"',
        'alt="MARVIS-Agent logo"',
    ):
        html = html.replace(alt_placeholder, f'alt="{platform_name} logo"')
    for name_placeholder in (
        '<h1 id="platformName">MARVIS-全能风控智能体</h1>',
        '<h1 id="platformName">MARVIS-Agent</h1>',
    ):
        html = html.replace(name_placeholder, f'<h1 id="platformName">{platform_name}</h1>', 1)
    return html


def branding_config_path(workspace: str | Path) -> Path:
    return Path(workspace) / BRANDING_DIR_NAME / BRANDING_CONFIG_NAME


def resolve_branding_asset(workspace: str | Path, asset_path: str) -> Path | None:
    branding_dir = (Path(workspace) / BRANDING_DIR_NAME).resolve()
    try:
        candidate = (branding_dir / unquote(asset_path)).resolve()
        candidate.relative_to(branding_dir)
    except (ValueError, OSError):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _load_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _first(config: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return None


def _text_field(config: dict[str, Any], *keys: str, fallback: str) -> str:
    value = _first(config, *keys)
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 120:
        return fallback
    return cleaned


def _validator_aliases(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    aliases: dict[str, str] = {}
    for name, alias in value.items():
        if not isinstance(name, str) or not isinstance(alias, str):
            continue
        cleaned_name = name.strip()
        cleaned_alias = alias.strip()
        if not cleaned_name or not cleaned_alias:
            continue
        if len(cleaned_name) > 120 or len(cleaned_alias) > 120:
            continue
        aliases[cleaned_name] = cleaned_alias
        if len(aliases) >= 200:
            break
    return aliases


def _color_field(config: dict[str, Any], *keys: str, fallback: str) -> str:
    value = _first(config, *keys)
    if not isinstance(value, str):
        return fallback
    cleaned = value.strip()
    if not _HEX_COLOR_RE.match(cleaned):
        return fallback
    return cleaned.lower()


def _asset_url(workspace: Path, value: Any, *, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    asset = resolve_branding_asset(workspace, value.strip())
    if asset is None:
        return fallback
    branding_dir = (workspace / BRANDING_DIR_NAME).resolve()
    relative = asset.relative_to(branding_dir).as_posix()
    stat = asset.stat()
    version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
    return f"branding/assets/{quote(relative)}?v={version}"


def _normalized_branding(branding: dict[str, str]) -> dict[str, str]:
    current = dict(DEFAULT_BRANDING)
    current.update(
        {
            key: value
            for key, value in branding.items()
            if key in current and isinstance(value, str) and value.strip()
        }
    )
    if not _HEX_COLOR_RE.match(current["primaryColor"]):
        current["primaryColor"] = DEFAULT_BRANDING["primaryColor"]
    current["primaryColor"] = current["primaryColor"].lower()
    return current


def _brand_hover_color(color: str) -> str:
    if color == "#000000":
        return "#1f1f1f"
    if color == "#1f1f1f":
        return "#303034"
    if color == "#2b2b2d":
        return "#343438"
    if color == "#303034":
        return "#3b3b42"
    if color == "#343438":
        return "#3f3f46"
    channels = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
    return "#" + "".join(f"{max(0, int(channel * 0.86 + 0.5)):02x}" for channel in channels)


def _image_mime_type(url: str) -> str:
    path = url.split("?")[0].lower()
    if path.endswith(".svg"):
        return "image/svg+xml"
    if path.endswith(".ico"):
        return "image/x-icon"
    if path.endswith(".webp"):
        return "image/webp"
    return "image/png"
