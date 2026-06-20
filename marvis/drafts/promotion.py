from __future__ import annotations

from pathlib import Path
import re

from marvis.drafts.contracts import DraftTool, PromotionCheck
from marvis.drafts.errors import PromotionError
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
    dest = Path(plugins_dir) / plugin_name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "__init__.py").write_text("", encoding="utf-8")
    (dest / "tools.py").write_text(draft.code, encoding="utf-8")
    manifest = _manifest_from_draft(draft, plugin_name, checksum=compute_checksum(dest))
    (dest / "manifest.json").write_text(
        _manifest_json(manifest),
        encoding="utf-8",
    )
    registry.register(manifest, enabled=True)
    drafts.set_status(draft.id, "promoted")
    if hasattr(registry, "_repo"):
        registry._repo.write_audit(
            kind="draft.promote",
            target_ref=draft.id,
            outcome="succeeded",
            detail={"plugin": manifest.name, "tests": check.test_result},
        )
    return manifest


def reject_draft(draft: DraftTool, *, drafts, reason: str, audit_repo=None) -> None:
    drafts.set_status(draft.id, "rejected")
    if audit_repo is not None:
        audit_repo.write_audit(
            kind="draft.reject",
            target_ref=draft.id,
            outcome="succeeded",
            detail={"reason": str(reason)},
        )


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


def _matches(output, expected) -> bool:
    return output == expected


__all__ = ["PromotionError", "promote_draft", "reject_draft", "validate_for_promotion"]
