from __future__ import annotations

import ast
from pathlib import Path

import pytest

from marvis.agent import renderers
from marvis.agent.presenters.strategy import (
    STRATEGY_RENDERERS,
    strategy_integrity_failure,
)
from marvis.agent.presenters.strategy_candidates import CANDIDATE_PRESENTERS
from marvis.agent.presenters.strategy_evidence import EVIDENCE_PRESENTERS
from marvis.agent.presenters.strategy_lifecycle import LIFECYCLE_PRESENTERS
from marvis.agent.presenters.strategy_pool import POOL_PRESENTERS
from marvis.agent.strategy_workflows._catalog import build_catalog


_WORKFLOW_PRESENTER_REFS = tuple(
    (spec.workflow_id, presenter_ref)
    for spec in build_catalog()
    if spec.migrated
    for presenter_ref in spec.presenter_refs
)


def test_strategy_presenter_registry_is_the_live_dispatch_source(monkeypatch) -> None:
    sentinel = ("domain presenter", [{"title": "sentinel", "columns": [], "rows": []}])

    monkeypatch.setitem(
        STRATEGY_RENDERERS,
        "build_strategy",
        lambda _output: sentinel,
    )

    assert renderers.render_tool_output("build_strategy", {}) == sentinel


def test_strategy_integrity_fallback_is_owned_by_domain_presenter() -> None:
    text, tables = strategy_integrity_failure("materialize_model_evidence_v2")

    assert "完整性校验失败" in text
    assert tables == []
    assert strategy_integrity_failure("unknown_tool") is None


def test_strategy_registry_covers_governed_candidate_and_pool_presenters() -> None:
    assert {
        "build_automatic_tree_candidate",
        "search_voting_candidates",
        "build_cross_matrix_candidate",
        "compile_strategy_pool",
        "apply_strategy_pool",
        "measure_strategy_pool_validation",
        "export_strategy_delivery",
    } <= set(STRATEGY_RENDERERS)
    assert all(renderers.has_tool_presenter(ref) for ref in STRATEGY_RENDERERS)
    assert not renderers.has_tool_presenter("unknown_tool")


def test_strategy_registry_is_composed_from_change_reason_domains() -> None:
    domain_keys = [
        set(CANDIDATE_PRESENTERS),
        set(EVIDENCE_PRESENTERS),
        set(LIFECYCLE_PRESENTERS),
        set(POOL_PRESENTERS),
    ]

    assert all(domain_keys)
    assert sum((len(keys) for keys in domain_keys), start=0) == len(
        set().union(*domain_keys)
    )
    assert set().union(*domain_keys) == set(STRATEGY_RENDERERS)


def test_strategy_presenter_domains_do_not_reach_into_sibling_implementations() -> None:
    domain_modules = {
        "strategy_candidates",
        "strategy_evidence",
        "strategy_lifecycle",
        "strategy_pool",
    }
    presenter_root = Path(__file__).resolve().parents[1] / "marvis/agent/presenters"

    for module_name in domain_modules:
        tree = ast.parse(
            (presenter_root / f"{module_name}.py").read_text(encoding="utf-8")
        )
        sibling_imports = {
            node.module.rsplit(".", 1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and isinstance(node.module, str)
            and node.module.rsplit(".", 1)[-1] in domain_modules
        }
        assert not sibling_imports, (module_name, sibling_imports)


@pytest.mark.parametrize(
    ("workflow_id", "presenter_ref"),
    _WORKFLOW_PRESENTER_REFS,
    ids=(
        f"{workflow_id}:{presenter_ref}"
        for workflow_id, presenter_ref in _WORKFLOW_PRESENTER_REFS
    ),
)
def test_each_canonical_workflow_template_tool_has_a_presenter(
    workflow_id: str,
    presenter_ref: str,
) -> None:
    assert renderers.has_tool_presenter(presenter_ref), (
        workflow_id,
        presenter_ref,
    )
