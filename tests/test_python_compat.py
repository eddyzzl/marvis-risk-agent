import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kernel_imported_modules_do_not_require_stdlib_strenum():
    for relative_path in [
        "marvis/domain.py",
        "marvis/validation/results.py",
    ]:
        tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
        imports = {
            (node.module, alias.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert ("enum", "StrEnum") not in imports
        assert ("marvis.compat", "StrEnum") in imports


def test_kernel_imported_modules_defer_annotations_at_runtime():
    for relative_path in [
        "marvis/domain.py",
        "marvis/output/excel.py",
        "marvis/output/styles.py",
        "marvis/validation/results.py",
        "marvis/validation/reproducibility.py",
        "marvis/validation/stress_test.py",
    ]:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert source.startswith("from __future__ import annotations\n")


def test_kernel_runtime_code_avoids_pep604_type_union_expressions():
    tree = ast.parse((PROJECT_ROOT / "marvis/report_texts.py").read_text(encoding="utf-8"))
    runtime_type_checks = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"isinstance", "issubclass"}
        and len(node.args) >= 2
    ]
    assert runtime_type_checks, "The report formatting module must retain runtime type checks"
    for check in runtime_type_checks:
        assert not any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            for node in ast.walk(check.args[1])
        ), f"Python 3.7 cannot evaluate a type union at line {check.lineno}"
