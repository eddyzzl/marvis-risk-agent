from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import sys
from typing import Any

from marvis.plugins.errors import ManifestError


DETERMINISM_CHOICES = frozenset({"deterministic", "stochastic"})
FAILURE_POLICY_CHOICES = frozenset({"fail", "retry", "skip"})
GOVERNANCE_POLICY_SCHEMA_VERSION = "tool-policy.v1"
GOVERNANCE_REQUIREMENT_CHOICES = frozenset({"none", "required"})
PERMISSION_CHOICES = frozenset({
    "llm",
    "network:optional",
    "process:spawn",
    "read:artifacts",
    "read:dataset",
    "read:draft",
    "read:experiment",
    "read:input",
    "read:join_plan",
    "read:materials",
    "read:model",
    "read:strategy",
    "read:task",
    "write:artifact",
    "write:artifacts",
    "write:backtest",
    "write:dataset",
    "write:draft",
    "write:draft_run",
    "write:experiment",
    "write:join_plan",
    "write:learning_note",
    "write:model",
    "write:report",
    "write:strategy",
    "write:task",
})
PLATFORM_HOOK_EVENTS = frozenset({
    "task.created",
    "task.scanned",
    "dataset.registered",
    "join.confirmed",
    "notebook.completed",
    "validation.completed",
    "feature.computed",
    "plan.confirmed",
    "report.before_generate",
    "report.after_generate",
    "memory.before_save",
    "memory.after_save",
    "workflow.completed",
    "step.completed",
    "plan.replanned",
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
class EffectTargetPolicy:
    """How a protected tool's state transition is bound and reconciled.

    Phase 0B intentionally supports only targets whose current state can be
    verified by the platform.  New target kinds must add a repository verifier
    before a manifest may declare them.
    """

    kind: str
    id_input: str
    expected_statuses: tuple[str, ...]
    result_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id_input": self.id_input,
            "expected_statuses": list(self.expected_statuses),
            "result_status": self.result_status,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "EffectTargetPolicy":
        if not isinstance(value, dict):
            raise ManifestError("tool policy.effect_target must be an object")
        kind = _policy_text(value.get("kind"), "effect_target.kind")
        id_input = _policy_text(value.get("id_input"), "effect_target.id_input")
        result_status = _policy_text(
            value.get("result_status"), "effect_target.result_status"
        )
        raw_statuses = value.get("expected_statuses")
        if not isinstance(raw_statuses, list) or not raw_statuses:
            raise ManifestError(
                "tool policy.effect_target.expected_statuses must be a non-empty list"
            )
        expected_statuses = tuple(
            _policy_text(item, "effect_target.expected_statuses")
            for item in raw_statuses
        )
        return cls(
            kind=kind,
            id_input=id_input,
            expected_statuses=expected_statuses,
            result_status=result_status,
        )


@dataclass(frozen=True)
class GovernancePolicy:
    """Orthogonal human-decision and side-effect authorization requirements."""

    schema_version: str = GOVERNANCE_POLICY_SCHEMA_VERSION
    human_decision_gate: str = "none"
    effect_authorization: str = "none"
    effect_target: EffectTargetPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "human_decision_gate": self.human_decision_gate,
            "effect_authorization": self.effect_authorization,
        }
        if self.effect_target is not None:
            payload["effect_target"] = self.effect_target.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> "GovernancePolicy":
        if value in (None, {}):
            return cls()
        if not isinstance(value, dict):
            raise ManifestError("tool policy must be an object")
        schema_version = str(
            value.get("schema_version") or GOVERNANCE_POLICY_SCHEMA_VERSION
        ).strip()
        if schema_version != GOVERNANCE_POLICY_SCHEMA_VERSION:
            raise ManifestError(
                f"tool policy.schema_version must be {GOVERNANCE_POLICY_SCHEMA_VERSION}"
            )
        human = str(value.get("human_decision_gate") or "none").strip().lower()
        effect = str(value.get("effect_authorization") or "none").strip().lower()
        if human not in GOVERNANCE_REQUIREMENT_CHOICES:
            raise ManifestError(
                "tool policy.human_decision_gate must be none or required"
            )
        if effect not in GOVERNANCE_REQUIREMENT_CHOICES:
            raise ManifestError(
                "tool policy.effect_authorization must be none or required"
            )
        target_value = value.get("effect_target")
        target = (
            EffectTargetPolicy.from_dict(target_value)
            if target_value is not None
            else None
        )
        if effect == "required" and human != "required":
            raise ManifestError(
                "tool policy.effect_authorization=required also requires "
                "human_decision_gate=required"
            )
        if effect == "required" and target is None:
            raise ManifestError(
                "tool policy.effect_authorization=required requires effect_target"
            )
        if effect == "none" and target is not None:
            raise ManifestError(
                "tool policy.effect_target is only valid when effect_authorization=required"
            )
        return cls(
            schema_version=schema_version,
            human_decision_gate=human,
            effect_authorization=effect,
            effect_target=target,
        )


def merge_governance_policies(*policies: GovernancePolicy) -> GovernancePolicy:
    """Return the strongest compatible policy; callers can raise, never lower."""

    human = "required" if any(
        item.human_decision_gate == "required" for item in policies
    ) else "none"
    effect = "required" if any(
        item.effect_authorization == "required" for item in policies
    ) else "none"
    targets = [item.effect_target for item in policies if item.effect_target is not None]
    target = targets[0] if targets else None
    if any(item != target for item in targets[1:]):
        raise ManifestError("tool policy.effect_target declarations conflict")
    if effect == "required" and target is None:
        raise ManifestError("tool policy.effect_authorization=required requires effect_target")
    if effect == "required":
        human = "required"
    return GovernancePolicy(
        human_decision_gate=human,
        effect_authorization=effect,
        effect_target=target,
    )


def governance_policy_hash(policy: GovernancePolicy) -> str:
    encoded = json.dumps(
        policy.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    policy: GovernancePolicy = field(default_factory=GovernancePolicy)


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

    permissions = tuple(_parse_string_list(data.get("permissions", []), "permissions"))
    _validate_known_permissions(permissions, label="permissions")
    _validate_tool_permissions(tools, permissions)
    hooks = tuple(_parse_hooks(data.get("hooks", []), seen_tools))
    _validate_hook_governance(hooks, tools)
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
                "policy": tool.policy.to_dict(),
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
    if determinism == "stochastic":
        _validate_stochastic_seed(input_schema, name)
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
    _validate_known_permissions(side_effects, label=f"tool {name} side_effects")
    policy = GovernancePolicy.from_dict(item.get("policy"))
    if policy.effect_target is not None:
        properties = input_schema.get("properties")
        if (
            not isinstance(properties, dict)
            or policy.effect_target.id_input not in properties
        ):
            raise ManifestError(
                f"tool {name} policy.effect_target.id_input must name an input_schema property"
            )
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
        policy=policy,
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


def _validate_tool_permissions(tools: list[ToolSpec], permissions: tuple[str, ...]) -> None:
    allowed = set(permissions)
    for tool in tools:
        missing = [effect for effect in tool.side_effects if effect not in allowed]
        if missing:
            raise ManifestError(
                f"tool {tool.name} side_effects not declared in permissions: {', '.join(missing)}"
            )


def _validate_hook_governance(
    hooks: tuple[HookSpec, ...],
    tools: list[ToolSpec],
) -> None:
    by_name = {tool.name: tool for tool in tools}
    for hook in hooks:
        tool = by_name[hook.tool]
        if tool.policy.effect_authorization == "required":
            raise ManifestError(
                f"effect-authorized tool {tool.name} cannot be registered as a hook"
            )
        if tool.policy.human_decision_gate == "required":
            raise ManifestError(
                f"human-decision-gated tool {tool.name} cannot be registered as a hook"
            )


def _validate_known_permissions(values: tuple[str, ...], *, label: str) -> None:
    unknown = sorted({value for value in values if value not in PERMISSION_CHOICES})
    if unknown:
        raise ManifestError(f"{label} contains unknown permission: {', '.join(unknown)}")


def _required_text(data: dict[str, Any], field: str, *, context: str = "manifest") -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{context}.{field} is required")
    return value.strip()


def _policy_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"tool policy.{field} must be a non-empty string")
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


def _validate_stochastic_seed(input_schema: dict[str, Any], tool_name: str) -> None:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict) or "seed" not in properties:
        raise ManifestError(f"tool {tool_name} stochastic tools must declare integer seed input")
    seed_schema = properties["seed"]
    if not isinstance(seed_schema, dict) or seed_schema.get("type") != "integer":
        raise ManifestError(f"tool {tool_name} seed input must be an integer schema")


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


def python_requires_satisfied(value: str, version: tuple[int, int, int] | None = None) -> bool:
    if not value:
        return True
    current = version or tuple(sys.version_info[:3])
    return all(_version_spec_satisfied(part.strip(), current) for part in value.split(",") if part.strip())


def _version_spec_satisfied(spec: str, current: tuple[int, int, int]) -> bool:
    match = re.fullmatch(r"(>=|>|<=|<|==|~=|!=)\s*(\d+(?:\.\d+){0,2})(?:\.\*)?", spec)
    if match is None:
        return False
    op, raw_version = match.groups()
    wildcard = spec.strip().endswith(".*")
    target = _version_tuple(raw_version)
    if wildcard and op in {"==", "!="}:
        prefix_len = len(raw_version.split("."))
        equal = current[:prefix_len] == target[:prefix_len]
        return equal if op == "==" else not equal
    if op == "~=":
        return current >= target and current < _compatible_upper_bound(raw_version)
    if op == ">=":
        return current >= target
    if op == ">":
        return current > target
    if op == "<=":
        return current <= target
    if op == "<":
        return current < target
    if op == "==":
        return current == target
    if op == "!=":
        return current != target
    return False


def _version_tuple(raw: str) -> tuple[int, int, int]:
    parts = [int(part) for part in raw.split(".")]
    return tuple([*parts, 0, 0][:3])


def _compatible_upper_bound(raw: str) -> tuple[int, int, int]:
    parts = [int(part) for part in raw.split(".")]
    if len(parts) <= 2:
        return (parts[0] + 1, 0, 0)
    return (parts[0], parts[1] + 1, 0)
