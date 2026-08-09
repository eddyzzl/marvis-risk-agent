from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from marvis.governance.errors import ApprovalBindingError
from marvis.governance.service import GovernanceService
from marvis.orchestrator.references import parse_step_output_ref
from marvis.repositories.task_artifacts import stable_task_artifact_id


def test_shared_step_output_ref_parser_is_a_dependency_leaf() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "marvis/orchestrator/references.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
        )
    ]

    assert imports == []


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("$ref:step-1.output", ("step-1", "")),
        ("$ref:step-1.output.value", ("step-1", "value")),
        (
            "$ref:step-1.output.items.0.message",
            ("step-1", "items.0.message"),
        ),
    ],
)
def test_shared_step_output_ref_parser_preserves_supported_forms(
    reference: str,
    expected: tuple[str, str],
) -> None:
    assert parse_step_output_ref(reference) == expected


@pytest.mark.parametrize(
    "reference",
    [
        "step-1.output.value",
        "$ref:",
        "$ref:.output.value",
        "$ref:step-1",
        "$ref:step-1.output.",
        "$ref:step-1.outputs.value",
    ],
)
def test_shared_step_output_ref_parser_rejects_ambiguous_forms(
    reference: str,
) -> None:
    with pytest.raises(ValueError, match="invalid ref"):
        parse_step_output_ref(reference)


def test_governance_preserves_approval_binding_error_for_malformed_ref() -> None:
    service = GovernanceService(
        plan_repo=None,
        tool_registry=None,
        strategy_repo=None,
        governance_repo=None,
    )

    with pytest.raises(
        ApprovalBindingError,
        match=r"invalid output reference: \$ref:step-1\.output\.",
    ):
        service._resolve_inputs({"message": "$ref:step-1.output."})


def test_canonical_task_artifact_id_preserves_legacy_digest() -> None:
    task_id = "task-1"
    kind = "strategy_candidate_monthly_stability_json"
    path = "/tmp/tasks/task-1/strategy_candidate_stability/result.json"
    legacy_identity = json.dumps(
        [task_id, kind, path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    legacy_digest = hashlib.sha256(
        f"marvis.task_artifact.v1:{legacy_identity}".encode("utf-8")
    ).hexdigest()

    assert stable_task_artifact_id(
        task_id=task_id,
        kind=kind,
        path=path,
    ) == legacy_digest


def test_task_artifact_consumers_do_not_reimplement_identity_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical_path = root / "marvis/repositories/task_artifacts.py"
    expected_consumers = {
        "marvis/agent/presenters/strategy_pool.py",
        "marvis/packs/strategy/candidate_lab_projection.py",
        "marvis/packs/strategy/candidate_stability_tools.py",
        "marvis/packs/strategy/dsl_delivery_tools.py",
        "marvis/packs/strategy/impact_cube_tools.py",
        "marvis/packs/strategy/interactive_tree_frontier_group_tools.py",
        "marvis/packs/strategy/interactive_tree_frontier_tools.py",
        "marvis/packs/strategy/interactive_tree_split_search_tools.py",
        "marvis/packs/strategy/interactive_tree_tools.py",
        "marvis/packs/strategy/pool_apply_tools.py",
        "marvis/packs/strategy/pool_stability_tools.py",
        "marvis/packs/strategy/pool_validation_tools.py",
        "marvis/packs/strategy/project_context_tools.py",
        "marvis/packs/strategy/report_bundle_adapters.py",
    }
    discovered_consumers: set[str] = set()

    for path in sorted((root / "marvis").rglob("*.py")):
        relative_path = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        private_identity_helpers = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in {"_stable_artifact_id", "_stable_task_artifact_id"}
        }
        assert not private_identity_helpers, relative_path
        private_names = {"_stable_artifact_id", "_stable_task_artifact_id"}
        private_identity_references = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in private_names
        }
        private_identity_references.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in private_names
        )
        private_identity_references.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in private_names
        )
        assert not private_identity_references, relative_path
        if path != canonical_path:
            assert "marvis.task_artifact.v1" not in source, relative_path

        public_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "stable_task_artifact_id"
        ]
        if not public_calls or path == canonical_path:
            continue
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "marvis.repositories.task_artifacts"
            for alias in node.names
        }
        assert "stable_task_artifact_id" in imported_names, relative_path
        discovered_consumers.add(relative_path)

    assert expected_consumers <= discovered_consumers
