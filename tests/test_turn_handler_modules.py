"""Turn lanes are real modules with explicit acyclic dependencies."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

from marvis.agent import turn_handlers
from marvis.agent.turn_handlers import _registry


PACKAGE = "marvis.agent.turn_handlers"
ROOT = Path(turn_handlers.__file__).parent


def test_lane_imports_form_an_explicit_acyclic_graph():
    modules = {path.stem: path for path in ROOT.glob("*.py") if path.stem != "__init__"}
    graph = {}
    for module, path in modules.items():
        tree = ast.parse(path.read_text())
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {
                    "exec",
                    "eval",
                    "compile",
                    "globals",
                    "locals",
                }, path
            if isinstance(node, ast.If):
                assert not (
                    isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
                ), path
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    dependencies.add(node.module.split(".")[0])
                else:
                    dependencies.update(alias.name for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module == PACKAGE:
                raise AssertionError(f"{path} imports the compatibility facade")
        assert dependencies <= modules.keys(), (path, dependencies)
        graph[module] = dependencies

    visited = set()
    active = []

    def visit(module):
        assert module not in active, " -> ".join([*active, module])
        if module in visited:
            return
        active.append(module)
        for dependency in graph[module]:
            visit(dependency)
        active.pop()
        visited.add(module)

    for module in graph:
        visit(module)


def test_exports_keep_their_real_module_identity_and_globals():
    namespace_ids = set()
    for path in ROOT.glob("*.py"):
        if path.stem == "__init__":
            continue
        module = importlib.import_module(f"{PACKAGE}.{path.stem}")
        namespace_ids.add(id(vars(module)))
    assert len(namespace_ids) == len(list(ROOT.glob("*.py"))) - 1

    for name in turn_handlers.__all__:
        value = getattr(turn_handlers, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            assert value.__module__.startswith(PACKAGE + "."), name
            owner = importlib.import_module(value.__module__)
            assert getattr(owner, name) is value
            if inspect.isfunction(value):
                assert value.__globals__ is vars(owner)
                assert value.__globals__ is not vars(turn_handlers)


def test_registry_uses_real_lane_entries_with_the_existing_call_contract():
    expected = {
        "modeling": "modeling",
        "data_join": "join",
        "feature_analysis": "feature",
        "strategy": "strategy_turns",
        "vintage": "vintage",
        "portfolio": "portfolio",
    }
    assert set(_registry.DRIVER_TURN_FUNCS) == set(expected)
    parameters = [
        "runtime",
        "repo",
        "task",
        "user_text",
        "selection",
        "dedup_strategies",
        "adjust_params",
        "expected_step_id",
        "expected_plan_id",
        "expected_plan_status",
        "expected_plan_revision",
        "expected_plan_fingerprint",
        "expected_step_fingerprint",
        "confirmation_source",
        "ui_action",
    ]
    for task_type, entry in _registry.DRIVER_TURN_FUNCS.items():
        assert entry.__module__ == f"{PACKAGE}.{expected[task_type]}"
        signature = inspect.signature(entry)
        assert list(signature.parameters) == parameters
        assert signature.parameters["user_text"].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["confirmation_source"].default == "human"
