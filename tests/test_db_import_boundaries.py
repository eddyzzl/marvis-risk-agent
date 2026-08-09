from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _production_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _format_import_violation(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}"


def test_internal_production_modules_do_not_import_through_db_facade():
    violations: list[str] = []
    marvis_root = _REPOSITORY_ROOT / "marvis"

    for path in _production_python_files(marvis_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imports_facade = (
                isinstance(node, ast.ImportFrom) and node.module == "marvis.db"
            ) or (
                isinstance(node, ast.Import)
                and any(alias.name == "marvis.db" for alias in node.names)
            )
            if imports_facade:
                violations.append(_format_import_violation(path, node))

    assert violations == [], (
        "internal production modules must import db_schema or an exact repository "
        f"module, not marvis.db: {violations}"
    )


def test_strategy_modules_do_not_import_siblings_via_package_at_module_scope():
    violations: list[str] = []
    strategy_root = _REPOSITORY_ROOT / "marvis" / "packs" / "strategy"

    for path in _production_python_files(strategy_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "marvis.packs.strategy"
            ):
                violations.append(_format_import_violation(path, node))

    assert violations == [], (
        "strategy siblings must be imported from their exact submodule, not the "
        f"strategy package top level: {violations}"
    )


def test_schema_initialization_does_not_load_db_facade_or_memory_store(tmp_path):
    script = """
import sys
from pathlib import Path
import marvis.db_schema as schema

schema.init_db(Path(sys.argv[1]))
unexpected = [
    name
    for name in ("marvis.db", "marvis.agent_memory.store")
    if name in sys.modules
]
if unexpected:
    raise SystemExit("unexpected imports: " + ", ".join(unexpected))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "app.sqlite")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_schema_initialization_does_not_load_strategy_runtime(tmp_path):
    script = """
import sys
from pathlib import Path
import marvis.db_schema as schema

schema.init_db(Path(sys.argv[1]))
loaded = sorted(
    name
    for name in sys.modules
    if name == "marvis.packs.strategy"
    or name.startswith("marvis.packs.strategy.")
)
if loaded:
    raise SystemExit("unexpected strategy imports: " + ", ".join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "fresh.sqlite")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_strategy_dsl_migration_loads_only_the_dsl_leaf(tmp_path):
    script = r'''
import sqlite3
import sys
from pathlib import Path
import marvis.db_schema as schema

db_path = Path(sys.argv[1])
schema.init_db(db_path)
canonical_dsl = (
    '{"default_action":{"reason_code":null,"stop":true,"type":"approval",'
    '"value":"approve"},"match_policy":"first_match","metadata":'
    '{"description":"baseline cutoff","lineage":{"source":"build_strategy"}},'
    '"rules":[{"action":{"reason_code":null,"stop":true,"type":"reject",'
    '"value":"reject"},"condition":{"field":"score","missing":"no_match",'
    '"op":"compare","operator":"<","value":600},"priority":10,'
    '"rule_id":"rule-legacy-ab9233c11e308faa"}],"schema_version":'
    '"strategy.dsl.v1","strategy_type":"approval"}'
)
with sqlite3.connect(db_path) as conn:
    conn.execute(
        """
        INSERT INTO strategies(
            id, task_id, strategy_type, rules_json, score_col,
            default_decision_json, description, created_at,
            dsl_json, dsl_schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "strategy-import-boundary",
            "task-import-boundary",
            "approval",
            "[]",
            "score",
            '"approve"',
            "migration import boundary",
            "2026-08-01T00:00:00Z",
            canonical_dsl,
            "strategy.dsl.v1",
        ),
    )
    conn.execute("DROP TABLE strategy_pool_materializations")
    conn.execute("ALTER TABLE strategies DROP COLUMN dsl_content_hash")
    conn.execute("PRAGMA user_version = 17")

schema.init_db(db_path)
loaded = sorted(
    name
    for name in sys.modules
    if name == "marvis.packs.strategy"
    or name.startswith("marvis.packs.strategy.")
)
allowed = {
    "marvis.packs.strategy",
    "marvis.packs.strategy.dsl",
    "marvis.packs.strategy.errors",
}
unexpected = sorted(set(loaded) - allowed)
if unexpected:
    raise SystemExit("unexpected strategy imports: " + ", ".join(unexpected))
with sqlite3.connect(db_path) as conn:
    content_hash = conn.execute(
        "SELECT dsl_content_hash FROM strategies WHERE id = ?",
        ("strategy-import-boundary",),
    ).fetchone()[0]
if content_hash != "7d797899a2048bc559b0900891003216fea1a933aa9c541e11a412284e805c49":
    raise SystemExit("migration changed the canonical Strategy DSL hash")
'''

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "upgrade.sqlite")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_agent_memory_keeps_schema_installer_public_reexports():
    from marvis.agent_memory import ensure_agent_memory_schema as package_export
    from marvis.agent_memory.store import (
        ensure_agent_memory_schema as store_export,
    )
    from marvis.agent_memory_schema import ensure_agent_memory_schema

    assert package_export is ensure_agent_memory_schema
    assert store_export is ensure_agent_memory_schema
