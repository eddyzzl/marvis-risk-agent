"""Plugin/Tool runtime primitives for MARVIS V2."""

from marvis.plugins.manifest import (
    HookSpec,
    PluginManifest,
    ToolRef,
    ToolSpec,
    manifest_to_dict,
    parse_manifest,
)

__all__ = [
    "HookSpec",
    "PluginManifest",
    "ToolRef",
    "ToolSpec",
    "manifest_to_dict",
    "parse_manifest",
]
