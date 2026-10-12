"""Signed install proposals reuse the existing Plugin and Workflow runtimes."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from marvis.db_schema import connect
from marvis.plugins.capability_versions import (
    CapabilityStore,
    capability_mode,
    governed_head,
)
from marvis.plugins.loader import _assert_manifest_schemas, _safe_extract_zip
from marvis.plugins.manifest import ToolRef, manifest_to_dict, parse_manifest
from marvis.plugins.publisher_trust import (
    SIGNATURE_FILE,
    digest,
    read_json,
    require,
    tree_hash,
)


def _declaration_root(extracted: Path) -> Path:
    candidates = (
        [extracted]
        if (extracted / SIGNATURE_FILE).is_file()
        else [
            p
            for p in extracted.iterdir()
            if p.is_dir() and (p / SIGNATURE_FILE).is_file()
        ]
    )
    require(len(candidates) == 1, "publisher_signature_missing", status=422)
    return candidates[0]


def _ref_key(ref):
    return ref.plugin, ref.tool, ref.version


def _no_inline_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            require(
                str(key).casefold()
                not in {
                    "api_key",
                    "apikey",
                    "password",
                    "secret",
                    "access_token",
                    "refresh_token",
                    "authorization",
                    "connection_string",
                },
                "workflow_inline_credentials_forbidden",
                status=422,
            )
            _no_inline_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _no_inline_credentials(item)


def validate_workflow_declaration(declaration, tools, validator):
    from marvis.orchestrator.templates.skills import (
        parse_skill_template,
        validate_skill_template,
    )

    require(
        isinstance(declaration, dict)
        and set(declaration) == {"workflow", "tool_bindings", "permissions"},
        "workflow_signed_declaration_shape",
        status=422,
    )
    _no_inline_credentials(declaration["workflow"])
    template = parse_skill_template(declaration["workflow"])
    refs = [
        ref for step in template.steps for ref in (step.tool_ref, *step.granted_tools)
    ]
    require(
        all(ref.version for ref in refs), "workflow_tool_version_required", status=422
    )
    bindings = []
    permissions = set()
    for key in sorted(set(_ref_key(ref) for ref in refs)):
        manifest, _tool = tools.resolve_with_manifest(ToolRef(*key))
        bindings.append(
            {
                "plugin": key[0],
                "tool": key[1],
                "version": key[2],
                "declaration_hash": digest(manifest_to_dict(manifest)),
            }
        )
        permissions.update(manifest.permissions)
    require(
        declaration["tool_bindings"] == bindings, "workflow_tool_declaration_changed"
    )
    require(
        declaration["permissions"] == sorted(permissions),
        "workflow_permissions_changed",
    )
    require(
        not validate_skill_template(template, tools, validator),
        "workflow_plan_validation_failed",
        status=422,
    )
    return template


def validate_declaration(kind, declaration, registry, tools, validator):
    if kind == "workflow":
        return validate_workflow_declaration(declaration, tools, validator)
    manifest = parse_manifest(declaration)
    _assert_manifest_schemas(manifest)
    require(
        not manifest.builtin and not manifest.draft_promotion,
        "governed_plugin_builtin_or_draft_forbidden",
        status=422,
    )
    row = registry._repo.get_plugin(manifest.name)
    require(
        row is None or not row["builtin"],
        "governed_builtin_shadow_forbidden",
        status=422,
    )
    require(
        manifest.name
        not in set(sys.stdlib_module_names)
        | {"marvis", "tests", "sitecustomize", "usercustomize"},
        "governed_module_shadow_forbidden",
        status=422,
    )
    require(
        manifest.module.startswith(manifest.name + "."),
        "governed_module_namespace_required",
        status=422,
    )
    require(
        all(part.isidentifier() for part in manifest.module.split(".")),
        "governed_module_namespace_required",
        status=422,
    )
    return manifest


def propose_archive(
    zip_path,
    *,
    db_path,
    registry,
    tools,
    validator,
    actor,
    expected_revision,
    kind=None,
    expected_name=None,
):
    store = CapabilityStore(db_path)
    require(
        Path(zip_path).is_file() and Path(zip_path).stat().st_size <= 50 * 1024 * 1024,
        "capability_archive_budget",
        status=422,
    )
    with tempfile.TemporaryDirectory(prefix="marvis-governed-") as temp:
        extracted = Path(temp)
        _safe_extract_zip(Path(zip_path), extracted)
        root = _declaration_root(extracted)
        envelope = read_json(root / SIGNATURE_FILE)
        body = store.verify(envelope)
        require(
            expected_name is None or body["name"] == expected_name,
            "capability_target_changed",
            status=422,
        )
        require(
            kind is None or body["kind"] == kind, "capability_kind_changed", status=422
        )
        content_hash = tree_hash(root)
        require(
            body["content_hash"] == content_hash,
            "capability_signed_content_changed",
            status=422,
        )
        if body["kind"] == "plugin":
            manifest = parse_manifest(read_json(root / "manifest.json"))
            declaration = manifest_to_dict(
                replace(manifest, checksum="", builtin=False)
            )
            permissions = sorted(manifest.permissions)
        else:
            declaration = read_json(root / "workflow.json")
            permissions = declaration.get("permissions")
        require(
            body["declaration_hash"] == digest(declaration),
            "capability_signed_content_changed",
            status=422,
        )
        validated = validate_declaration(
            body["kind"], declaration, registry, tools, validator
        )
        name = validated.name if body["kind"] == "plugin" else validated.id
        require(
            body["name"] == name
            and (body["kind"] != "plugin" or body["version"] == validated.version)
            and body["content_hash"] == content_hash
            and body["declaration_hash"] == digest(declaration),
            "capability_signed_content_changed",
            status=422,
        )
        if body["kind"] == "plugin":
            relative = Path(*validated.module.split(".")[1:])
            require(
                (root / relative.with_suffix(".py")).is_file()
                or (root / relative / "__init__.py").is_file(),
                "governed_module_file_missing",
                status=422,
            )
        require(
            isinstance(permissions, list)
            and all(isinstance(p, str) for p in permissions)
            and len(set(permissions)) == len(permissions),
            "capability_permissions_invalid",
            status=422,
        )
        destination = store.root / body["kind"] / content_hash / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists():
            require(
                tree_hash(destination) == content_hash
                and read_json(destination / SIGNATURE_FILE) == envelope,
                "capability_retained_material_changed",
            )
        else:
            temporary = destination.parent / ("staging-" + os.urandom(12).hex())
            shutil.copytree(root, temporary)
            try:
                temporary.rename(destination)
            except OSError:
                shutil.rmtree(temporary)
                require(
                    destination.is_dir() and tree_hash(destination) == content_hash,
                    "capability_retained_material_changed",
                )
            for path in destination.rglob("*"):
                path.chmod(0o500 if path.is_dir() else 0o400)
            destination.chmod(0o500)
        return store.propose(
            body=body,
            envelope=envelope,
            declaration=declaration,
            permissions=permissions,
            material_path=destination,
            expected_revision=expected_revision,
            actor=actor,
        )


def confirm_proposal(db_path, registry, tools, validator, proposal_id, **confirmation):
    store = CapabilityStore(db_path)
    result = store.confirm(
        proposal_id,
        validator=lambda kind, declaration: validate_declaration(
            kind, declaration, registry, tools, validator
        ),
        **confirmation,
    )
    registry.load_from_db()
    return result


def load_governed_workflows(workspace, tools, validator, report):
    from marvis.orchestrator.templates import register_user_template

    store = CapabilityStore(Path(workspace) / "marvis.sqlite")
    if not store.db_path.exists():
        return
    with connect(store.db_path) as conn:
        if (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='capability_heads'"
            ).fetchone()
            is None
        ):
            return
        heads = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM capability_heads WHERE kind='workflow'"
            )
        ]
    for head in heads:
        if not head["enabled"]:
            report.disabled.append(head["name"])
            continue
        try:
            row = store.resolve("workflow", head["name"])
            template = validate_workflow_declaration(
                json.loads(row["declaration_json"]), tools, validator
            )
        except Exception as exc:
            report.rejected.append((head["name"], [str(exc)]))
            continue
        register_user_template(template)
        report.active.append(template.id)


def bind_workflow_plan(conn, plan, db_path):
    from marvis.orchestrator.templates import builtin_template_ids

    head = (
        governed_head(conn, "workflow", plan.template_id) if plan.template_id else None
    )
    if not head:
        if capability_mode(conn) == "governed" and plan.template_id:
            require(
                plan.template_id in builtin_template_ids(),
                "signed_workflow_required",
                status=403,
            )
        return
    require(head["enabled"], "capability_disabled", status=403)
    row = CapabilityStore(db_path)._version(conn, head["version_id"])
    _require_plan_refs(plan, json.loads(row["declaration_json"]))
    conn.execute(
        "INSERT INTO capability_plan_bindings VALUES(?,?)", (plan.id, row["id"])
    )


def _require_plan_refs(plan, declaration):
    from marvis.orchestrator.templates.skills import parse_skill_template

    template = parse_skill_template(declaration["workflow"])
    expected = [
        (_ref_key(step.tool_ref), sorted(_ref_key(r) for r in step.granted_tools))
        for step in template.steps
    ]
    actual = [
        (_ref_key(step.tool_ref), sorted(_ref_key(r) for r in step.granted_tools))
        for step in plan.steps
    ]
    require(actual == expected, "workflow_plan_tools_changed", status=403)


def validate_plan_capabilities(db_path, plan):
    store = CapabilityStore(db_path)
    with connect(db_path) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='capability_plan_bindings'"
        ).fetchone():
            return
        row = conn.execute(
            "SELECT workflow_version_id FROM capability_plan_bindings WHERE plan_id=?",
            (plan.id,),
        ).fetchone()
        if row:
            version = store._version(conn, row["workflow_version_id"])
            head = governed_head(conn, "workflow", version["name"])
            require(
                head is not None and head["enabled"],
                "workflow_dependency_disabled",
                status=403,
            )
            declaration = json.loads(version["declaration_json"])
            _require_plan_refs(plan, declaration)
            for binding in declaration["tool_bindings"]:
                plugin_head = governed_head(conn, "plugin", binding["plugin"])
                if plugin_head:
                    dependency = conn.execute(
                        "SELECT id FROM capability_versions WHERE kind=? AND name=? AND version=?",
                        ("plugin", binding["plugin"], binding["version"]),
                    ).fetchone()
                    require(
                        dependency is not None and plugin_head["enabled"],
                        "workflow_dependency_disabled",
                        status=403,
                    )
                    dependency_version = store._version(conn, dependency["id"])
                    manifest = replace(
                        parse_manifest(
                            json.loads(dependency_version["declaration_json"])
                        ),
                        checksum=dependency_version["content_hash"],
                    )
                    require(
                        digest(manifest_to_dict(manifest))
                        == binding["declaration_hash"],
                        "workflow_tool_declaration_changed",
                    )
                else:
                    plugin = conn.execute(
                        "SELECT manifest_json,enabled FROM plugins WHERE name=?",
                        (binding["plugin"],),
                    ).fetchone()
                    require(
                        plugin is not None and plugin["enabled"],
                        "workflow_dependency_disabled",
                        status=403,
                    )
                    require(
                        digest(json.loads(plugin["manifest_json"]))
                        == binding["declaration_hash"],
                        "workflow_tool_declaration_changed",
                    )
        elif capability_mode(conn) == "governed" and plan.template_id:
            from marvis.orchestrator.templates import builtin_template_ids

            require(
                plan.template_id in builtin_template_ids(),
                "signed_workflow_required",
                status=403,
            )
