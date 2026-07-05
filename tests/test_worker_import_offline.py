"""Guard: the tool worker's import chain must not need server-only deps.

The plugin worker (``marvis/plugins/subprocess_worker.py``) runs inside the
user-selected execution_environment (its own conda env / python_executable),
which is expected to carry the data-science stack (pandas/numpy/sklearn) but
NOT the server-only stack (fastapi/uvicorn/starlette/jinja2). Loading a builtin
pack's tool module must therefore never transitively import fastapi et al.

This regression pins the fix for the "No module named 'fastapi'" crash on the
切分样本 (modeling.prepare) path: ``ErrorKind`` was pulled from ``marvis.errors``
(which imports fastapi) purely for its string constants, so it now lives in the
zero-dependency leaf module ``marvis.error_kinds``.

The block is done with a ``sys.meta_path`` finder, which only bites if the target
module has NOT already been imported. Since this pytest process itself imports
fastapi, the check runs in a fresh ``sys.executable`` subprocess that installs the
blocker before importing anything under ``marvis`` -- deterministic, offline, and
needing no extra venv.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Server-only third-party deps that the execution environment need not have.
# pandas/numpy/sklearn are intentionally NOT blocked: they belong to the
# execution environment and blocking them would test the wrong thing.
_BLOCKED = ("fastapi", "uvicorn", "starlette", "jinja2")

# Every module the worker actually imports to run a builtin tool: the worker
# entrypoint + its leaf contracts, plus each builtin pack's registered `module`
# (see marvis/packs/*/manifest.json) and the modeling prepare entrypoint that
# hosts 切分样本 (marvis/orchestrator/templates/modeling.py).
_WORKER_CHAIN_MODULES = (
    "marvis.plugins.subprocess_worker",
    "marvis.plugins.contracts",
    "marvis.error_kinds",
    "marvis.data.errors",
    "marvis.packs.data_ops.tools",
    "marvis.packs.analysis.tools",
    "marvis.packs.modeling.tools",
    "marvis.packs.modeling.prepare_tools",
    "marvis.packs.strategy.tools",
    "marvis.packs.v1_compat.tools",
)

# Self-contained probe: install a meta_path finder that raises for the blocked
# roots, then import each target and report OK/FAIL per module on stdout.
_PROBE = """
import importlib.abc
import sys

BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        root = name.split(".", 1)[0]
        if root in BLOCKED:
            raise ModuleNotFoundError("No module named %r (blocked by test)" % root)
        return None


sys.meta_path.insert(0, _Blocker())

failures = []
for module in {modules!r}:
    try:
        __import__(module)
    except ModuleNotFoundError as exc:
        failures.append("%s -> %s" % (module, exc))

for line in failures:
    print("FAIL " + line)
sys.exit(1 if failures else 0)
"""


def test_worker_import_chain_has_no_server_only_deps() -> None:
    script = _PROBE.format(blocked=_BLOCKED, modules=_WORKER_CHAIN_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "builtin pack tool modules must import without the server-only "
        f"dependency stack {_BLOCKED}; failures:\n{result.stdout}{result.stderr}"
    )


def test_blocker_actually_bites_on_a_server_only_import() -> None:
    """Negative control: prove the meta_path blocker really does block fastapi,
    so a green worker-chain check above is meaningful and not a no-op finder."""
    probe = _PROBE.format(blocked=_BLOCKED, modules=("fastapi",))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, (
        "blocker should have made `import fastapi` fail but it did not; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "fastapi" in result.stdout
