from __future__ import annotations

from datetime import datetime
import json

from marvis.repositories.plugins import PluginRepository
from marvis.plugins.errors import (
    DuplicatePluginError,
    ManifestError,
    PluginNotFoundError,
    ToolNotFoundError,
)
from marvis.plugins.manifest import (
    DraftPromotionReceipt,
    EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED,
    EXECUTION_PROFILE_DRAFT_RESTRICTED_V1,
    EXECUTION_PROFILE_STANDARD,
    PluginManifest,
    ToolRef,
    draft_promotion_receipt_from_dict,
    ToolSpec,
    parse_manifest,
    python_requires_satisfied,
)


class PluginRegistry:
    def __init__(self, repo: PluginRepository):
        self._repo = repo
        self._plugins: dict[str, tuple[PluginManifest, bool]] = {}

    def load_from_db(self) -> None:
        self._plugins.clear()
        rows = self._repo.list_plugins(include_disabled=True)
        raw_manifests = [json.loads(row["manifest_json"]) for row in rows]
        promotion_audits = (
            _load_draft_promotion_audits(self._repo)
            if any(_has_missing_execution_profile(data) for data in raw_manifests)
            else ()
        )
        for row, data in zip(rows, raw_manifests, strict=True):
            _assert_manifest_row_identity(data, row)
            changed = False
            if _has_missing_execution_profile(data):
                changed = _migrate_legacy_execution_profile(
                    data,
                    row=row,
                    promotion_audits=promotion_audits,
                )
            manifest = parse_manifest(data, builtin=bool(row["builtin"]))
            if changed:
                # The profile and exact receipt become manifest-owned after one
                # proven migration. Future starts no longer depend on audit
                # availability or a same-name plugin history.
                self._repo.upsert_plugin(manifest, enabled=bool(row["enabled"]))
            self._plugins[manifest.name] = (manifest, bool(row["enabled"]))

    def register(self, manifest: PluginManifest, *, enabled: bool = True) -> None:
        existing = self._plugins.get(manifest.name)
        if existing is not None and existing[0].version == manifest.version:
            raise DuplicatePluginError(f"{manifest.name}@{manifest.version} already registered")
        audit = {
            "kind": "plugin.register",
            "target_ref": manifest.name,
            "outcome": "succeeded",
            "detail": {
                "version": manifest.version,
                "builtin": manifest.builtin,
                "enabled": bool(enabled),
            },
        }
        self._repo.upsert_plugin_with_audit(manifest, enabled=enabled, audit=audit)
        self._plugins[manifest.name] = (manifest, bool(enabled))

    def remove(self, name: str) -> None:
        manifest, _enabled = self._require(name)
        if manifest.builtin:
            raise ValueError("cannot remove builtin plugin")
        audit = {
            "kind": "plugin.remove",
            "target_ref": name,
            "outcome": "succeeded",
            "detail": {"version": manifest.version},
        }
        self._repo.delete_plugin_with_audit(name, audit=audit)
        del self._plugins[name]

    def set_enabled(self, name: str, enabled: bool) -> None:
        manifest, _current = self._require(name)
        audit = {
            "kind": "plugin.enable" if enabled else "plugin.disable",
            "target_ref": name,
            "outcome": "succeeded",
            "detail": {"version": manifest.version, "enabled": bool(enabled)},
        }
        self._repo.set_enabled_with_audit(name, enabled, audit=audit)
        self._plugins[name] = (manifest, bool(enabled))

    def get(self, name: str) -> PluginManifest:
        return self._require(name)[0]

    def is_enabled(self, name: str) -> bool:
        return self._require(name)[1]

    def list(self, *, include_disabled: bool = False) -> list[PluginManifest]:
        items = sorted(self._plugins.items(), key=lambda item: item[0])
        return [
            manifest
            for _name, (manifest, enabled) in items
            if include_disabled or enabled
        ]

    def _require(self, name: str) -> tuple[PluginManifest, bool]:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise PluginNotFoundError(name) from exc


class ToolRegistry:
    def __init__(self, plugin_registry: PluginRegistry):
        self._plugins = plugin_registry

    def resolve(self, ref: ToolRef) -> ToolSpec:
        _manifest, tool = self.resolve_with_manifest(ref)
        return tool

    def resolve_with_manifest(self, ref: ToolRef) -> tuple[PluginManifest, ToolSpec]:
        try:
            manifest = self._plugins.get(ref.plugin)
        except PluginNotFoundError:
            raise
        if not self._plugins.is_enabled(ref.plugin):
            raise PluginNotFoundError(ref.plugin)
        if not python_requires_satisfied(manifest.python_requires):
            raise ToolNotFoundError(
                f"{ref.label()} requires Python {manifest.python_requires}"
            )
        if ref.version and ref.version != manifest.version:
            raise ToolNotFoundError(
                f"{ref.label()} version {ref.version} does not match {manifest.version}"
            )
        for tool in manifest.tools:
            if tool.name == ref.tool:
                if tool.execution_profile == EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED:
                    raise ToolNotFoundError(
                        f"{ref.label()} is unavailable until its legacy Draft is re-promoted"
                    )
                return manifest, tool
        raise ToolNotFoundError(ref.label())

    def manifests(self) -> tuple[PluginManifest, ...]:
        """Return all manifests for runtime boundary checks."""

        return tuple(self._plugins.list(include_disabled=True))

    def catalog_for_planner(self) -> list[dict]:
        catalog: list[dict] = []
        for manifest in self._plugins.list():
            if not python_requires_satisfied(manifest.python_requires):
                continue
            for tool in manifest.tools:
                if tool.execution_profile == EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED:
                    # An ambiguous legacy Draft is deliberately unavailable to
                    # planners as well as to the runner. It must be promoted
                    # again to receive a concrete identity receipt.
                    continue
                catalog.append({
                    "plugin": manifest.name,
                    "tool": tool.name,
                    "version": manifest.version,
                    "summary": tool.summary,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "determinism": tool.determinism,
                    "policy": tool.policy.to_dict(),
                })
        return catalog


def _load_draft_promotion_audits(repo) -> tuple[dict, ...]:
    """Load promotion evidence or stop; an unavailable audit is not an empty one."""

    list_audit = getattr(repo, "list_audit", None)
    if not callable(list_audit):
        raise ManifestError("cannot read Draft promotion provenance")
    try:
        rows = list_audit(kind="draft.promote", limit=None)
    except Exception as exc:
        raise ManifestError("cannot read Draft promotion provenance") from exc
    if not isinstance(rows, list):
        raise ManifestError("Draft promotion provenance is invalid")
    return tuple(row for row in rows if isinstance(row, dict))


def _has_missing_execution_profile(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    tools = data.get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and "execution_profile" not in tool
        for tool in tools
    )


def _assert_manifest_row_identity(data: object, row: dict) -> None:
    if not isinstance(data, dict):
        raise ManifestError("stored manifest must be an object")
    for field in ("name", "version", "checksum"):
        manifest_value = str(data.get(field) or "")
        row_value = str(row.get(field) or "")
        if manifest_value != row_value:
            raise ManifestError(f"stored manifest {field} does not match plugin identity")


def _migrate_legacy_execution_profile(
    data: dict,
    *,
    row: dict,
    promotion_audits: tuple[dict, ...],
) -> bool:
    """Persist a profile for an old manifest without name-only attribution.

    A modern receipt binds name/version/checksum/tool/profile directly.  The
    earliest promoted Drafts lack that receipt, so their only migration proof is
    a platform ``draft.promote`` event that occurred after this exact installed
    artifact was recorded.  A later same-name install has a later
    ``plugins.installed_at`` and is therefore normalized to standard instead of
    inheriting an unrelated Draft restriction.
    """

    receipt = _manifest_receipt(data, row=row)
    if receipt is not None:
        _apply_receipt(data, receipt)
        return True

    exact_receipts = _matching_audit_receipts(row, promotion_audits)
    if exact_receipts:
        if len({item.tool_name for item in exact_receipts}) != 1:
            _mark_repromotion_required(data)
        else:
            _apply_receipt(data, exact_receipts[-1])
        return True

    legacy_audits = _legacy_audits_for_current_artifact(row, promotion_audits)
    if legacy_audits is None:
        _mark_repromotion_required(data)
        return True
    if legacy_audits:
        receipt = _legacy_receipt(data, row=row, audit=legacy_audits[-1])
        if receipt is None:
            _mark_repromotion_required(data)
        else:
            _apply_receipt(data, receipt)
        return True

    if _looks_like_legacy_draft_artifact(data, row=row):
        # A missing historical audit cannot prove this old Draft's identity.
        # Do not let the parser's standard default re-open it; this is an
        # explicit, non-runnable state until the artifact is re-promoted.
        _mark_repromotion_required(data)
        return True

    _set_missing_profiles(data, EXECUTION_PROFILE_STANDARD)
    return True


def _manifest_receipt(data: dict, *, row: dict) -> DraftPromotionReceipt | None:
    raw = data.get("draft_promotion")
    if raw is None:
        return None
    receipt = draft_promotion_receipt_from_dict(raw)
    if not _receipt_matches_row(receipt, row):
        raise ManifestError("draft_promotion does not match stored plugin identity")
    return receipt


def _matching_audit_receipts(
    row: dict,
    audits: tuple[dict, ...],
) -> list[DraftPromotionReceipt]:
    receipts: list[DraftPromotionReceipt] = []
    for audit in audits:
        detail = audit.get("detail")
        raw = detail.get("draft_promotion") if isinstance(detail, dict) else None
        if raw is None:
            continue
        receipt = draft_promotion_receipt_from_dict(raw)
        if _receipt_matches_row(receipt, row):
            receipts.append(receipt)
    return receipts


def _legacy_audits_for_current_artifact(
    row: dict,
    audits: tuple[dict, ...],
) -> list[dict] | None:
    """Return proven legacy events, [] for a later ordinary install, or None for ambiguity."""

    installed_at = _parse_utc_timestamp(row.get("installed_at"))
    if installed_at is None:
        return None
    matching: list[dict] = []
    for audit in audits:
        detail = audit.get("detail")
        if not isinstance(detail, dict) or detail.get("plugin") != row.get("name"):
            continue
        if detail.get("draft_promotion") is not None:
            # Modern receipts were handled above. A same-name but different
            # identity is only relevant when its event overlaps this artifact.
            audit_at = _parse_utc_timestamp(audit.get("at"))
            if audit_at is None:
                return None
            if audit_at >= installed_at:
                return None
            continue
        audit_at = _parse_utc_timestamp(audit.get("at"))
        if audit_at is None:
            return None
        if audit_at >= installed_at:
            matching.append(audit)
    return matching


def _legacy_receipt(
    data: dict,
    *,
    row: dict,
    audit: dict,
) -> DraftPromotionReceipt | None:
    if not _looks_like_legacy_draft_artifact(data, row=row):
        return None
    tools = data["tools"]
    tool_name = tools[0].get("name")
    draft_id = audit.get("target_ref")
    if not isinstance(tool_name, str) or not tool_name or not isinstance(draft_id, str) or not draft_id:
        return None
    return DraftPromotionReceipt(
        draft_id=draft_id,
        plugin_name=str(row["name"]),
        plugin_version=str(row["version"]),
        plugin_checksum=str(row["checksum"]),
        tool_name=tool_name,
        execution_profile=EXECUTION_PROFILE_DRAFT_RESTRICTED_V1,
    )


def _looks_like_legacy_draft_artifact(data: dict, *, row: dict) -> bool:
    """Recognize the old promotion envelope without treating an audit name as identity."""

    tools = data.get("tools")
    return (
        str(row.get("name") or "").startswith("draft_")
        and str(row.get("version") or "") == "0.1.0"
        and str(data.get("module") or "") == f"{row.get('name')}.tools"
        and isinstance(tools, list)
        and len(tools) == 1
        and isinstance(tools[0], dict)
        and isinstance(tools[0].get("name"), str)
        and bool(tools[0]["name"])
    )


def _apply_receipt(data: dict, receipt: DraftPromotionReceipt) -> None:
    tools = data.get("tools")
    if not isinstance(tools, list):
        raise ManifestError("stored manifest tools must be a list")
    matched = False
    for tool in tools:
        if not isinstance(tool, dict):
            raise ManifestError("stored manifest tool must be an object")
        if tool.get("name") == receipt.tool_name:
            existing = tool.get("execution_profile")
            if existing not in (None, receipt.execution_profile):
                raise ManifestError("draft_promotion conflicts with tool execution profile")
            tool["execution_profile"] = receipt.execution_profile
            matched = True
    if not matched:
        raise ManifestError("draft_promotion tool is not present in stored manifest")
    data["draft_promotion"] = receipt.to_dict()


def _mark_repromotion_required(data: dict) -> None:
    _set_missing_profiles(data, EXECUTION_PROFILE_DRAFT_REPROMOTION_REQUIRED)


def _set_missing_profiles(data: dict, profile: str) -> None:
    tools = data.get("tools")
    if not isinstance(tools, list):
        raise ManifestError("stored manifest tools must be a list")
    for tool in tools:
        if not isinstance(tool, dict):
            raise ManifestError("stored manifest tool must be an object")
        tool.setdefault("execution_profile", profile)


def _receipt_matches_row(receipt: DraftPromotionReceipt, row: dict) -> bool:
    return (
        receipt.plugin_name == str(row.get("name") or "")
        and receipt.plugin_version == str(row.get("version") or "")
        and receipt.plugin_checksum == str(row.get("checksum") or "")
    )


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None
