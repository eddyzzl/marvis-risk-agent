"""Keep the extracted calculation boundary free of application side effects."""
import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["effectiveness", "platform_metrics", "suggested_confirmation"])
def test_calculation_modules_do_not_own_task_or_artifact_io(module):
    path = Path(__file__).resolve().parents[2] / "marvis" / "validation" / f"{module}.py"
    tree = ast.parse(path.read_text())
    forbidden = (
        "marvis.repositories", "marvis.db", "marvis.domain", "marvis.pipeline",
        "marvis.output", "marvis.validation_services", "fastapi", "pathlib", "sqlite3",
    )
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", f"{module} opens a file"
    assert not [name for name in imports if name.startswith(forbidden)]
