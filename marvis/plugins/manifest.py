from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from marvis.plugins.errors import ManifestError


DETERMINISM_CHOICES = frozenset({"deterministic", "stochastic"})
FAILURE_POLICY_CHOICES = frozenset({"fail", "retry", "skip"})
PLATFORM_HOOK_EVENTS = frozenset({
    "task.created",
    "task.scanned",
    "notebook.completed",
    "validation.completed",
    "report.before_generate",
    "report.after_generate",
    "memory.before_save",
    "memory.after_save",
    "workflow.completed",
    "step.completed",
})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_PYTHON_REQUIRES_RE = re.compile(
    r"^(?:>=|>|<=|<|==|~=|!=)\s*\d+(?:\.\d+){0,2}(?:\.\*)?"
    r"(?:\s*,\s*(?:>=|>|<=|<|==|~=|!=)\s*\d+(?:\.\d+){0,2}(?:\.\*)?)*$"
)


@dataclass(frozen=True)
class ToolRef:
    plugin: str
    tool: str
    version: str = ""

    def label(self) -> str:
        return f"{self.plugin}.{self.tool}"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    determinism: str
    timeout_seconds: int
    failure_policy: str
    side_effects: tuple[str, ...]
    entrypoint: str
    memory_limit_mb: int = 2048


@dataclass(frozen=True)
class HookSpec:
    event: str
    tool: str


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    display_name: str
    description: str
    module: str
    python_requires: str
    tools: tuple[ToolSpec, ...]
    hooks: tuple[HookSpec, ...] = ()
    permissions: tuple[str, ...] = ()
    builtin: bool = False
    checksum: str = ""


def parse_manifest(data: dict[str, Any], *, builtin: bool = False) -> PluginManifest:
    if not isinstance(data, dict):
        raise ManifestError("manifest must be an object")

    name = _required_text(data, "name")
    version = _required_text(data, "version")
    display_name = _optional_text(data, "display_name", name)
    description = _optional_text(data, "description", "")
    module = _required_text(data, "module")
    python_requires = _optional_text(data, "python_requires", "")
    _validate_identifier(name, "name")
    _validate_semver(version)
    _validate_python_requires(python_requires)

    tools_data = data.get("tools")
    if not isinstance(tools_data, list) or not tools_data:
        raise ManifestError("tools must be a non-empty list")

    tools: list[ToolSpec] = []
    seen_tools: set[str] = set()
    for index, item in enumerate(tools_data):
        tool = _parse_tool(item, index)
        if tool.name in seen_tools:
            raise ManifestError(f"duplicate tool name: {tool.name}")
        seen_tools.add(tool.name)
        tools.append(tool)

    hooks = tuple(_parse_hooks(data.get("hooks", []), seen_tools))
    permissions = tuple(_parse_string_list(data.get("permissions", []), "permissions"))
    checksum = "" if builtin else str(data.get("checksum") or "")

    return PluginManifest(
        name=name,
        version=version,
        display_name=display_name,
        description=description,
        module=module,
        python_requires=python_requires,
        tools=tuple(tools),
        hooks=hooks,
        permissions=permissions,
        builtin=bool(builtin),
        checksum=checksum,
    )


def manifest_to_dict(manifest: PluginManifest) -> dict[str, Any]:
    return {
        "name": manifest.name,
        "version": manifest.version,
        "display_name": manifest.display_name,
        "description": manifest.description,
        "module": manifest.module,
        "python_requires": manifest.python_requires,
        "tools": [
            {
                "name": tool.name,
                "summary": tool.summary,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "determinism": tool.determinism,
                "timeout_seconds": tool.timeout_seconds,
                "failure_policy": tool.failure_policy,
                "side_effects": list(tool.side_effects),
                "entrypoint": tool.entrypoint,
                "memory_limit_mb": tool.memory_limit_mb,
            }
            for tool in manifest.tools
        ],
        "hooks": [
            {"event": hook.event, "tool": hook.tool}
            for hook in manifest.hooks
        ],
        "permissions": list(manifest.permissions),
        "builtin": manifest.builtin,
        "checksum": manifest.checksum,
    }


def _parse_tool(item: Any, index: int) -> ToolSpec:
    if not isinstance(item, dict):
        raise ManifestError(f"tool[{index}] must be an object")
    name = _required_text(item, "name", context=f"tool[{index}]")
    _validate_identifier(name, f"tool[{index}].name")
    summary = _required_text(item, "summary", context=f"tool[{index}]")
    input_schema = _required_schema(item, "input_schema", context=f"tool {name}")
    output_schema = _required_schema(item, "output_schema", context=f"tool {name}")
    determinism = _required_text(item, "determinism", context=f"tool {name}")
    if determinism not in DETERMINISM_CHOICES:
        raise ManifestError(f"tool {name} determinism must be deterministic or stochastic")
    failure_policy = _required_text(item, "failure_policy", context=f"tool {name}")
    if failure_policy not in FAILURE_POLICY_CHOICES:
        raise ManifestError(f"tool {name} failure_policy must be fail, retry, or skip")
    timeout_seconds = _positive_int(item.get("timeout_seconds"), f"tool {name} timeout_seconds")
    memory_limit_mb = _positive_int(
        item.get("memory_limit_mb", 2048),
        f"tool {name} memory_limit_mb",
    )
    entrypoint = _required_text(item, "entrypoint", context=f"tool {name}")
    side_effects = tuple(_parse_string_list(item.get("side_effects", []), f"tool {name} side_effects"))
    return ToolSpec(
        name=name,
        summary=summary,
        input_schema=input_schema,
        output_schema=output_schema,
        determinism=determinism,
        timeout_seconds=timeout_seconds,
        failure_policy=failure_policy,
        side_effects=side_effects,
        entrypoint=entrypoint,
        memory_limit_mb=memory_limit_mb,
    )


def _parse_hooks(raw_hooks: Any, tool_names: set[str]) -> list[HookSpec]:
    if raw_hooks is None:
        return []
    if not isinstance(raw_hooks, list):
        raise ManifestError("hooks must be a list")
    hooks: list[HookSpec] = []
    for index, raw in enumerate(raw_hooks):
        if not isinstance(raw, dict):
            raise ManifestError(f"hook[{index}] must be an object")
        event = _required_text(raw, "event", context=f"hook[{index}]")
        if event not in PLATFORM_HOOK_EVENTS:
            raise ManifestError(f"unknown hook event: {event}")
        tool = _required_text(raw, "tool", context=f"hook[{index}]")
        if tool not in tool_names:
            raise ManifestError(f"hook tool not found: {tool}")
        hooks.append(HookSpec(event=event, tool=tool))
    return hooks


def _required_text(data: dict[str, Any], field: str, *, context: str = "manifest") -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{field} is required")
    return value.strip()


def _optional_text(data: dict[str, Any], field: str, default: str) -> str:
    value = data.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ManifestError(f"manifest.{field} must be a string")
    return value.strip()


def _required_schema(data: dict[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = data.get(field)
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{context} {field} must be a non-empty object")
    return value


def _parse_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ManifestError(f"{label}[{index}] must be a string")
        result.append(item)
    return result


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _validate_identifier(value: str, label: str) -> None:
    if not _IDENTIFIER_RE.match(value):
        raise ManifestError(f"{label} must be an identifier")


def _validate_semver(version: str) -> None:
    if not _SEMVER_RE.fullmatch(version):
        raise ManifestError("version must be a semantic version like 1.2.3")


def _validate_python_requires(value: str) -> None:
    if value and not _PYTHON_REQUIRES_RE.fullmatch(value):
        raise ManifestError("python_requires must be a Python version specifier")
