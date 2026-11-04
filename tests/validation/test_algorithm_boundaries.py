"""Keep the extracted calculation boundary free of application side effects."""
import ast
from pathlib import Path

import pytest

VALIDATION_DIR = Path(__file__).resolve().parents[2] / "marvis" / "validation"

# validation/ owns deterministic algorithms only (AGENTS.md "模块边界"). File,
# cache, lock and DB side effects live in validation_services/ or pipeline.py.
_FORBIDDEN_IMPORTS = (
    "marvis.repositories", "marvis.db", "marvis.domain", "marvis.pipeline",
    "marvis.output", "marvis.validation_services", "marvis.agent", "marvis.api",
    "fastapi", "sqlite3", "shutil", "filelock", "tempfile",
)
# Reason-carrying exceptions: pmml_manifest snapshots an untrusted PMML into an
# anonymous SpooledTemporaryFile while sanitising it; nothing is persisted.
_ALLOWED_IMPORTS = {("pmml_manifest", "tempfile")}
_WRITE_METHODS = frozenset({
    "write_text", "write_bytes", "mkdir", "makedirs", "unlink", "rmdir", "touch",
    "symlink_to", "hardlink_to", "to_parquet", "to_csv", "to_excel", "to_pickle",
    "to_feather", "write_table", "ParquetWriter",
})
_OS_WRITE_CALLS = frozenset({
    "remove", "unlink", "rename", "replace", "makedirs", "mkdir", "rmdir",
    "link", "symlink", "truncate",
})
VALIDATION_MODULES = sorted(path.stem for path in VALIDATION_DIR.glob("*.py"))


def _imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def _open_mode(call: ast.Call, *, positional_index: int) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    if len(call.args) > positional_index and isinstance(call.args[positional_index], ast.Constant):
        return str(call.args[positional_index].value)
    return None


def _write_side_effects(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            mode = _open_mode(node, positional_index=1) or "r"
            if set(mode) & set("wax+"):
                found.append(f"open(mode={mode!r}) line {node.lineno}")
        elif isinstance(func, ast.Attribute):
            if func.attr in _WRITE_METHODS:
                found.append(f".{func.attr}() line {node.lineno}")
            elif func.attr == "open":
                mode = _open_mode(node, positional_index=0) or "r"
                if set(mode) & set("wax+"):
                    found.append(f".open(mode={mode!r}) line {node.lineno}")
            elif (
                isinstance(func.value, ast.Name)
                and func.value.id == "os"
                and func.attr in _OS_WRITE_CALLS
            ):
                found.append(f"os.{func.attr}() line {node.lineno}")
    return found


@pytest.mark.parametrize("module", ["effectiveness", "platform_metrics", "suggested_confirmation"])
def test_calculation_modules_do_not_own_task_or_artifact_io(module):
    path = VALIDATION_DIR / f"{module}.py"
    tree = ast.parse(path.read_text())
    forbidden = (*_FORBIDDEN_IMPORTS, "pathlib")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "open", f"{module} opens a file"
    assert not [name for name in _imports(tree) if name.startswith(forbidden)]


@pytest.mark.parametrize("module", VALIDATION_MODULES)
def test_validation_modules_import_no_service_or_storage_layer(module):
    tree = ast.parse((VALIDATION_DIR / f"{module}.py").read_text())
    offending = [
        name
        for name in _imports(tree)
        if name.startswith(_FORBIDDEN_IMPORTS)
        and (module, name.split(".")[0]) not in _ALLOWED_IMPORTS
    ]
    assert offending == []


@pytest.mark.parametrize("module", VALIDATION_MODULES)
def test_validation_modules_do_not_write_the_filesystem(module):
    tree = ast.parse((VALIDATION_DIR / f"{module}.py").read_text())
    assert _write_side_effects(tree) == []


def test_side_effect_scanner_detects_writes():
    """Guard the guard: each forbidden pattern is actually recognised."""
    source = (
        "open(p, 'w')\nopen(p, mode='ab')\np.open('x')\np.write_text('')\n"
        "p.mkdir()\nos.replace(a, b)\nframe.to_parquet(p)\nopen(p)\np.open('rb')\n"
        "text.replace('a', 'b')\n"
    )
    found = _write_side_effects(ast.parse(source))
    assert len(found) == 7


def test_importing_validation_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    """Importing every validation module must not create files in cwd."""
    import importlib

    monkeypatch.chdir(tmp_path)
    for module in VALIDATION_MODULES:
        if module != "__init__":
            importlib.import_module(f"marvis.validation.{module}")
    assert list(tmp_path.iterdir()) == []
