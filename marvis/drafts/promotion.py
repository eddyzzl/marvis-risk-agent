from __future__ import annotations

from pathlib import Path
import re

from marvis.artifacts import TransactionalDirectoryStore
from marvis.drafts.contracts import DraftTool, PromotionCheck
from marvis.drafts.errors import PromotionError
from marvis.plugins.errors import DuplicatePluginError, PluginNotFoundError
from marvis.plugins.loader import compute_checksum
from marvis.plugins.manifest import PluginManifest, ToolSpec, manifest_to_dict


def validate_for_promotion(
    draft: DraftTool,
    *,
    sandbox,
    test_cases: list[dict],
) -> PromotionCheck:
    problems = []
    if not draft.input_schema or not draft.output_schema:
        problems.append("missing schema")
    if draft.determinism not in {"deterministic", "stochastic"}:
        problems.append("determinism not declared")
    if not test_cases:
        problems.append("at least one test case required")
    for index, test_case in enumerate(test_cases, start=1):
        if not isinstance(test_case, dict) or not isinstance(test_case.get("inputs"), dict):
            problems.append(f"test case {index} inputs must be an object")

    test_result = None
    if not problems:
        results = []
        for test_case in test_cases:
            run = sandbox.run_draft(draft.id, dict(test_case["inputs"]), task_id=draft.task_id)
            ok = run.ok
            if ok and "expect" in test_case:
                ok = _matches(run.output, test_case["expect"])
            results.append(bool(ok))
        test_result = {"passed": all(results), "n": len(results)}
        if not all(results):
            problems.append("test cases failed")

    return PromotionCheck(
        passed=not problems,
        problems=tuple(problems),
        test_result=test_result,
    )


def promote_draft(
    draft: DraftTool,
    *,
    registry,
    drafts,
    plugins_dir: Path,
    check: PromotionCheck,
) -> PluginManifest:
    if not check.passed:
        raise PromotionError(f"cannot promote: {', '.join(check.problems)}")
    plugin_name = _plugin_name(draft)
    plugins_root = Path(plugins_dir)
    staged = TransactionalDirectoryStore(plugins_root).stage(plugin_name)
    staged.path.mkdir(parents=True, exist_ok=False)
    (staged.path / "__init__.py").write_text("", encoding="utf-8")
    (staged.path / "tools.py").write_text(draft.code, encoding="utf-8")
    manifest = _manifest_from_draft(draft, plugin_name, checksum=compute_checksum(staged.path))
    _raise_if_duplicate_same_version(registry, manifest)
    (staged.path / "manifest.json").write_text(
        _manifest_json(manifest),
        encoding="utf-8",
    )
    try:
        staged.activate()
        _register_promoted_draft(
            registry=registry,
            drafts=drafts,
            manifest=manifest,
            draft=draft,
            check=check,
        )
        staged.commit()
    except Exception:
        staged.rollback()
        raise
    return manifest


def _register_promoted_draft(
    *,
    registry,
    drafts,
    manifest: PluginManifest,
    draft: DraftTool,
    check: PromotionCheck,
) -> None:
    plugin_audit = {
        "kind": "plugin.register",
        "target_ref": manifest.name,
        "outcome": "succeeded",
        "detail": {
            "version": manifest.version,
            "builtin": manifest.builtin,
            "enabled": True,
        },
    }
    audit = {
        "kind": "draft.promote",
        "target_ref": draft.id,
        "outcome": "succeeded",
        "detail": {"plugin": manifest.name, "tests": check.test_result},
    }
    registry._repo.promote_draft_with_plugin_audits(
        manifest,
        enabled=True,
        draft_id=draft.id,
        plugin_audit=plugin_audit,
        draft_audit=audit,
    )
    registry._plugins[manifest.name] = (manifest, True)


def reject_draft(draft: DraftTool, *, drafts, reason: str) -> None:
    audit = {
        "kind": "draft.reject",
        "target_ref": draft.id,
        "outcome": "succeeded",
        "detail": {"reason": str(reason)},
    }
    drafts.set_status_with_audit(draft.id, "rejected", audit=audit)


def _manifest_from_draft(draft: DraftTool, plugin_name: str, *, checksum: str) -> PluginManifest:
    return PluginManifest(
        name=plugin_name,
        version="0.1.0",
        display_name=f"Promoted Draft: {draft.name}",
        description=draft.summary,
        module=f"{plugin_name}.tools",
        python_requires=">=3.10,<3.14",
        tools=(
            ToolSpec(
                name=draft.name,
                summary=draft.summary,
                input_schema=draft.input_schema,
                output_schema=draft.output_schema,
                determinism=draft.determinism,
                timeout_seconds=60,
                failure_policy="fail",
                side_effects=(),
                entrypoint=draft.name,
                memory_limit_mb=2048,
            ),
        ),
        hooks=(),
        permissions=(),
        builtin=False,
        checksum=checksum,
    )


def _manifest_json(manifest: PluginManifest) -> str:
    import json

    return json.dumps(manifest_to_dict(manifest), ensure_ascii=False, indent=2) + "\n"


def _plugin_name(draft: DraftTool) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", f"draft_{draft.name}").strip("_")
    if not name or not re.match(r"^[A-Za-z_]", name):
        name = f"draft_{name}"
    return name[:64]


def _raise_if_duplicate_same_version(registry, manifest: PluginManifest) -> None:
    try:
        existing = registry.get(manifest.name)
    except PluginNotFoundError:
        return
    if existing.version == manifest.version:
        raise DuplicatePluginError(f"{manifest.name}@{manifest.version} already registered")


def _matches(output, expected) -> bool:
    return output == expected


__all__ = ["PromotionError", "promote_draft", "reject_draft", "validate_for_promotion"]
