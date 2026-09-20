"""Compiler modules must work through normal imports and resolvable types."""

import ast
import importlib
import inspect
import typing
from graphlib import TopologicalSorter
from pathlib import Path

import pytest

from marvis.agent import strategy_request_compiler as compiler


@pytest.mark.parametrize(
    "name",
    [
        "core",
        "cross",
        "impact",
        "model_evidence",
        "pool",
        "refinement_report",
        "sample_design",
        "scorecard",
        "tree",
        "voting",
        "contracts",
        "grounding",
    ],
)
def test_grammar_functions_have_their_own_importable_module(name):
    module = importlib.import_module(f"{compiler.__name__}.{name}")
    functions = [
        value
        for value in vars(module).values()
        if inspect.isfunction(value) and value.__module__ == module.__name__
    ]
    assert functions
    for function in functions:
        assert inspect.getmodule(function) is module
        assert function.__globals__ is vars(module)
        assert function.__globals__ is not vars(compiler)
        # An unresolved sibling name can otherwise survive imports and only
        # fail when a schema/introspection caller evaluates its annotation.
        typing.get_type_hints(function)


def test_public_request_types_keep_one_identity_across_grammar_modules():
    from marvis.agent.strategy_request_compiler import contracts

    result = compiler.validate_strategy_request(
        {
            "request_kind": "standard_workflow",
            "workflow": "cross_matrix_analysis",
            "workflow_inputs": {
                "x_feature": "age",
                "x_method": "equal_frequency",
                "y_feature": "score",
                "y_method": "equal_width",
                "bin_count": 5,
                "min_bin_pct": 0.02,
                "sentinel_values": [],
            },
        },
        allowed_columns=("age", "score"),
        target_col="bad",
    )
    assert isinstance(result, compiler.StrategyRequestCompilation)
    assert isinstance(result.draft, compiler.StandardWorkflowRequestDraft)
    assert type(result.draft) is contracts.StandardWorkflowRequestDraft
    assert type(result) is contracts.StrategyRequestCompilation
    assert compiler.StrategyRequestDraft is contracts.StrategyRequestDraft
    assert (
        compiler.CompiledStrategyRequestDraft is contracts.CompiledStrategyRequestDraft
    )
    for request_type in (
        contracts.StrategyRequestDraft,
        contracts.StandardWorkflowRequestDraft,
        contracts.StrategyRequestCompilation,
    ):
        typing.get_type_hints(request_type)
    assert result.draft.to_dict()["workflow_inputs"]["features"] == ["age", "score"]


def test_compiler_modules_have_an_acyclic_import_graph():
    package = Path(inspect.getfile(compiler)).parent
    sources = {
        path.stem: ast.parse(path.read_text(encoding="utf-8"))
        for path in package.glob("*.py")
    }
    dependencies = {name: set() for name in sources}
    prefix = compiler.__name__ + "."

    def add_dependency(source, imported):
        if imported == compiler.__name__:
            dependencies[source].add("__init__")
        elif imported.startswith(prefix):
            destination = imported.removeprefix(prefix).split(".")[0]
            assert destination in sources, imported
            dependencies[source].add(destination)

    for name, tree in sources.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add_dependency(name, alias.name)
            elif isinstance(node, ast.ImportFrom):
                imported = node.module or ""
                if node.level:
                    imported = importlib.util.resolve_name(
                        "." * node.level + imported, compiler.__name__
                    )
                if imported == compiler.__name__:
                    for alias in node.names:
                        destination = (
                            prefix + alias.name
                            if alias.name in sources
                            else compiler.__name__
                        )
                        add_dependency(name, destination)
                else:
                    add_dependency(name, imported)

    # This checks module dependencies, including annotations and imports nested
    # in functions; a symbol DAG alone can still hide a module-level cycle.
    assert set(TopologicalSorter(dependencies).static_order()) == set(sources)
    for name in sources.keys() - {"__init__", "core"}:
        assert dependencies[name].isdisjoint({"__init__", "core"}), name
    for name in ("contracts", "grounding", "identifiers"):
        assert dependencies[name] == set(), name


def test_private_compatibility_exports_keep_their_implementation_identity():
    from marvis.agent.strategy_request_compiler import core, impact, sample_design

    assert compiler._SYSTEM == core._SYSTEM
    assert (
        compiler.utterance_targets_strategy_impact_cube
        is impact.utterance_targets_strategy_impact_cube
    )
    for name in (
        "_sample_design_v2_has_chained_operation",
        "_sample_v2_predicate_grounded",
        "_sample_v2_predicate_semantics_grounded",
    ):
        assert getattr(compiler, name) is getattr(sample_design, name)


def test_grammar_definitions_do_not_use_runtime_namespace_bridges():
    package = Path(inspect.getfile(compiler)).parent
    forbidden_calls = {
        "exec",
        "eval",
        "__import__",
        "import_module",
        "ModuleType",
        "globals",
    }
    for path in package.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name != "__getattr__", path.name
            if isinstance(node, ast.Call):
                called = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else None
                )
                assert called not in forbidden_calls, (path.name, node.lineno, called)
