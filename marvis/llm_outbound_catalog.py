"""Installed tool contracts, carried through planner compaction and budgeting.

Catalog dictionaries remain usable locally. Only registry-resolved, immutable
schema snapshots can enter a prompt with the complete JSON Schema vocabulary.
Plain dictionaries, including JSON round trips, carry no such authority.
"""

from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from marvis.llm_outbound import (
    ApplicationSchema,
    OutboundContextError,
    StructuralPrompt,
    _check_output_schema,
    _check_schema_limits,
    _mint_application_schema,
    safe_schema,
)


_CATALOG_TOKEN = object()
_SCHEMA_FIELDS = ("input_schema", "output_schema")
_ANNOTATIONS = frozenset({
    "description", "examples", "default", "title", "$comment", "deprecated", "readOnly", "writeOnly",
})
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_CHILDREN = frozenset({
    "items", "additionalItems", "additionalProperties", "contains", "propertyNames", "not",
    "if", "then", "else", "unevaluatedItems", "unevaluatedProperties", "contentSchema",
})
_SCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_DYNAMIC_SCHEMA_KEYWORDS = (_SCHEMA_MAPS | _SCHEMA_CHILDREN | _SCHEMA_LISTS | _ANNOTATIONS | {
    "$schema", "$id", "$anchor", "$dynamicAnchor", "$ref", "$dynamicRef", "$vocabulary",
    "type", "enum", "const", "required", "dependentRequired",
    "minProperties", "maxProperties", "minItems", "maxItems", "minContains", "maxContains", "uniqueItems",
    "format", "pattern", "contentMediaType", "contentEncoding", "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
})


class _CatalogEntry(dict):
    def __init__(self, values, contracts, *, _token=None):
        if _token is not _CATALOG_TOKEN:
            raise OutboundContextError("工具契约需要已安装目录来源，未发送请求。")
        super().__init__(values)
        self._identity = _identity(values)
        self._contracts = tuple((key, contract) for key, contract in contracts.items())


class _CompactCatalogEntry(_CatalogEntry):
    pass


def installed_catalog_entry(registry, ref, values):
    """Attach provenance to an entry made by the actual installed ToolRegistry.

Builtin contracts must match the package on disk. Dynamic contracts must match
an enabled repository row and its successful installation/promotion audit. A
caller-supplied manifest or builtin flag alone never grants schema authority.
An unproven local tool can still be listed with the ordinary structural summary.
"""
    from marvis.plugins.registry import ToolRegistry

    if not isinstance(registry, ToolRegistry):
        return dict(values)
    manifest, tool = registry.resolve_with_manifest(ref)
    if manifest.builtin:
        installed = _builtin_manifest(manifest.name)
        if installed is None or (installed.name, installed.version, installed.module) != (
            manifest.name, manifest.version, manifest.module,
        ):
            return dict(values)
        source = next((entry for entry in installed.tools if entry.name == tool.name), None)
        if source is None or any(getattr(source, field) != getattr(tool, field) for field in _SCHEMA_FIELDS):
            return dict(values)
        contracts = dict(_builtin_contracts(manifest.name, tool.name))
    elif _dynamic_installation_is_authenticated(registry, manifest):
        for field in _SCHEMA_FIELDS:
            _check_schema_limits(getattr(tool, field))
            _check_output_schema(getattr(tool, field))
        schemas = {field: _without_annotations(getattr(tool, field)) for field in _SCHEMA_FIELDS}
        contracts = {
            field: _mint_application_schema("tool_catalog", schema)
            for field, schema in schemas.items()
        }
    else:
        return dict(values)
    return _CatalogEntry(values, contracts, _token=_CATALOG_TOKEN)


@lru_cache(maxsize=256)
def _builtin_manifest(name):
    from marvis.plugins.errors import ManifestError
    from marvis.plugins.loader import load_manifest
    from marvis.llm_outbound import _DROP, _identifier

    if _identifier(name) is _DROP:
        return None
    directory = Path(__file__).parent / "packs" / name
    if not (directory / "manifest.json").is_file():
        return None
    try:
        return load_manifest(directory, builtin=True)
    except (ManifestError, OSError, ValueError):
        return None


@lru_cache(maxsize=1024)
def _builtin_contracts(plugin, tool):
    source = next(entry for entry in _builtin_manifest(plugin).tools if entry.name == tool)
    return tuple(
        (field, _mint_application_schema("tool_catalog", getattr(source, field)))
        for field in _SCHEMA_FIELDS
    )


def _dynamic_installation_is_authenticated(registry, manifest):
    from marvis.plugins.manifest import manifest_to_dict
    from marvis.repositories.plugins import PluginRepository

    repository = registry._plugins._repo
    if not isinstance(repository, PluginRepository):
        return False
    checksum = manifest.checksum
    if not isinstance(checksum, str) or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        return False
    try:
        row = repository.get_plugin(manifest.name)
        if not row or not row.get("enabled") or row.get("builtin"):
            return False
        if row.get("version") != manifest.version or row.get("checksum") != checksum:
            return False
        if json.loads(row["manifest_json"]) != manifest_to_dict(manifest):
            return False
        registered = any(
            audit.get("target_ref") == manifest.name
            and audit.get("outcome") == "succeeded"
            and audit.get("detail", {}).get("version") == manifest.version
            and audit.get("detail", {}).get("builtin") is False
            and audit.get("detail", {}).get("enabled") is True
            for audit in repository.list_audit(kind="plugin.register", limit=None)
        )
        if not registered:
            return False
        receipt = manifest.draft_promotion
        if receipt is None:
            return True
        return any(
            audit.get("target_ref") == receipt.draft_id
            and audit.get("outcome") == "succeeded"
            and audit.get("detail", {}).get("draft_promotion") == receipt.to_dict()
            and audit.get("detail", {}).get("tests", {}).get("passed") is True
            for audit in repository.list_audit(kind="draft.promote", limit=None)
        )
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error):
        return False


def _without_annotations(schema):
    """Remove dynamic free prose at schema nodes, not properties named title."""
    if isinstance(schema, bool):
        return schema
    clean = {}
    for key, value in schema.items():
        if key not in _DYNAMIC_SCHEMA_KEYWORDS:
            raise OutboundContextError("动态工具契约包含未经支持的注释或扩展，未发送请求。")
        if key == "$vocabulary":
            raise OutboundContextError("动态工具契约包含未经支持的词汇表，未发送请求。")
        if key == "$schema" and value not in {
            "https://json-schema.org/draft/2020-12/schema",
        }:
            raise OutboundContextError("动态工具契约包含未知元模式，未发送请求。")
        if key in _ANNOTATIONS:
            continue
        if key in {"enum", "const"}:
            _inspect_dynamic_literals(value)
        if key == "format":
            from jsonschema import Draft202012Validator
            if value not in Draft202012Validator.FORMAT_CHECKER.checkers:
                raise OutboundContextError("动态工具契约包含未知格式注释，未发送请求。")
        if key in {"pattern", "contentMediaType", "contentEncoding"}:
            _inspect_dynamic_literals(value)
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            clean[key] = {name: _without_annotations(child) for name, child in value.items()}
        elif key in _SCHEMA_CHILDREN and isinstance(value, (dict, bool)):
            clean[key] = _without_annotations(value)
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            clean[key] = [_without_annotations(child) for child in value]
        else:
            clean[key] = value
    return clean


def _inspect_dynamic_literals(value):
    from marvis.llm_outbound_instructions import user_instruction
    if isinstance(value, str):
        if value:
            user_instruction(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            user_instruction(key)
            _inspect_dynamic_literals(child)
    elif isinstance(value, list):
        for child in value:
            _inspect_dynamic_literals(child)


def _identity(item):
    return tuple(item.get(key) for key in ("plugin", "tool", "version"))


def catalog_schema_contracts(item) -> dict[str, ApplicationSchema]:
    """Return detached immutable capabilities; ordinary dictionaries yield none."""
    if not isinstance(item, _CatalogEntry):
        return {}
    if _identity(item) != item._identity:
        raise OutboundContextError("工具契约身份发生变化，未发送请求。")
    return dict(item._contracts)


def compact_catalog_contracts(source, compact):
    contracts = catalog_schema_contracts(source)
    if not contracts:
        return compact
    clean = {**compact, **{field: safe_schema(contract) for field, contract in contracts.items()}}
    return _CompactCatalogEntry(clean, contracts, _token=_CATALOG_TOKEN)


def project_planning_catalog(payload):
    """Withhold complete schema bytes until their capabilities are checked."""
    tools = payload.get("available_tools")
    if not isinstance(tools, list):
        return payload
    projected = dict(payload)
    projected["available_tools"] = [
        {key: value for key, value in item.items() if key not in _SCHEMA_FIELDS}
        if isinstance(item, dict) else item
        for item in tools
    ]
    return projected


def attach_planning_catalog(clean, payload):
    source = payload.get("available_tools")
    target = clean.get("available_tools")
    if not isinstance(source, list) or not isinstance(target, list):
        return ()
    contracts_by_identity = {}
    for item in source:
        contracts = catalog_schema_contracts(item)
        if contracts:
            identity = _identity(item)
            if identity in contracts_by_identity:
                raise OutboundContextError("工具目录包含重复契约身份，未发送请求。")
            contracts_by_identity[identity] = tuple(contracts.items())
    for item in target:
        for field, contract in contracts_by_identity.get(_identity(item), ()):
            item[field] = safe_schema(contract)
    return tuple(contracts_by_identity.items())


def restore_planning_catalog(payload, source_prompt):
    """Restore authority only from the original prompt during local budgeting."""
    if not isinstance(source_prompt, StructuralPrompt):
        raise OutboundContextError("目录预算调整缺少原始结构契约，未发送请求。")
    contracts_by_identity = dict(getattr(source_prompt, "_catalog_contracts", ()))
    tools = payload.get("available_tools")
    if not isinstance(tools, list):
        return payload
    restored = dict(payload)
    restored_tools = []
    for item in tools:
        if not isinstance(item, dict):
            restored_tools.append(item)
            continue
        contracts = dict(contracts_by_identity.get(_identity(item), ()))
        if contracts:
            if any(item.get(field) != safe_schema(contract) for field, contract in contracts.items()):
                raise OutboundContextError("工具目录约束在预算调整时发生变化，未发送请求。")
            item = _CompactCatalogEntry(item, contracts, _token=_CATALOG_TOKEN)
        elif any(field in item for field in _SCHEMA_FIELDS):
            raise OutboundContextError("工具目录缺少原始契约来源，未发送请求。")
        restored_tools.append(item)
    restored["available_tools"] = restored_tools
    from marvis.llm_outbound_planning import restore_planning_controls
    restored = restore_planning_controls(restored, source_prompt)
    return restored
