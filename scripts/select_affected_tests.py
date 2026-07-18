#!/usr/bin/env python3
"""Select a conservative pytest subset for ``scripts/check --affected``.

The selector follows local Python imports transitively.  It only returns a
targeted test list when every changed runtime file can be mapped.  Unknown
files, parse failures, and source modules without a dependent test request the
caller's fast-tier fallback instead of guessing.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal


FAST_FALLBACK_EXIT = 2
SelectionMode = Literal["targeted", "none", "fallback"]

# These trees are outside the Python/runtime surfaces validated by
# ``scripts/check``.  Keeping an unrelated local website or generated build
# output untracked must not turn every small backend edit into the whole fast
# suite.  Unknown untracked roots remain conservative and still request the
# fallback tier.
NON_PYTEST_UNTRACKED_ROOTS = frozenset({"build", "dist", "website"})
CURATED_DYNAMIC_PACKS = frozenset({"strategy"})
PLUGIN_CONTRACT_TEST_NAMES = frozenset(
    {
        "test_frontend_v2_plugin.py",
        "test_package_data.py",
        "test_windows_packaging.py",
        "test_worker_import_offline.py",
    }
)
STRATEGY_PACK_SHARED_TEST_NAMES = frozenset(
    {
        "test_agent_autodrive.py",
        "test_agent_gate_contracts.py",
        "test_agent_task_routing.py",
        "test_governance_api.py",
        "test_governance_policy.py",
        "test_orch_templates.py",
        "test_plan_driver.py",
        "test_plugin_api.py",
        "test_plugin_loader.py",
        "test_plugin_manifest.py",
        "test_plugin_registry.py",
        "test_plugin_schema.py",
        "test_v2_memory_wiring.py",
    }
)


@dataclass(frozen=True)
class Selection:
    mode: SelectionMode
    tests: tuple[str, ...] = ()
    reason: str = ""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _range_base(diff_range: str) -> str:
    if "..." in diff_range:
        return diff_range.split("...", 1)[0]
    if ".." in diff_range:
        return diff_range.split("..", 1)[0]
    return diff_range


def discover_changed_files(root: Path, diff_range: str | None) -> Selection | list[str]:
    """Return repo-relative changed paths, or a fallback request on git errors."""

    if diff_range:
        base = _range_base(diff_range)
        verified = _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
        if verified.returncode != 0:
            return Selection("fallback", reason=f"diff base not found: {base}")
        changed = _git(
            root,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMRD",
            diff_range,
        )
        if changed.returncode != 0:
            detail = changed.stderr.decode(errors="replace").strip()
            return Selection("fallback", reason=f"cannot inspect diff range {diff_range}: {detail}")
        payloads = [changed.stdout]
    else:
        changed = _git(
            root,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACMRD",
            "HEAD",
        )
        untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        if changed.returncode != 0 or untracked.returncode != 0:
            return Selection("fallback", reason="cannot inspect local git changes")
        payloads = [changed.stdout]
        untracked_paths = [
            item.decode(errors="surrogateescape")
            for item in untracked.stdout.split(b"\0")
            if item
        ]
        payloads.append(
            b"\0".join(
                path.encode(errors="surrogateescape")
                for path in untracked_paths
                if not (
                    PurePosixPath(path).parts
                    and PurePosixPath(path).parts[0] in NON_PYTEST_UNTRACKED_ROOTS
                )
            )
        )

    paths: set[str] = set()
    for payload in payloads:
        paths.update(
            item.decode(errors="surrogateescape")
            for item in payload.split(b"\0")
            if item
        )
    return sorted(paths)


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative_import(
    current_module: str,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    keep = max(0, len(package_parts) - (level - 1))
    parts = package_parts[:keep]
    if imported_module:
        parts.extend(imported_module.split("."))
    return ".".join(parts)


def _import_names(tree: ast.AST, current_module: str, is_package: bool) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value == "marvis" or value.startswith("marvis."):
                names.add(value)
            continue
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = _resolve_relative_import(
                current_module,
                is_package,
                node.level,
                node.module,
            )
        else:
            base = node.module or ""
        if base:
            names.add(base)
        for alias in node.names:
            if alias.name != "*" and base:
                names.add(f"{base}.{alias.name}")
    return names


def _local_dependencies(imports: Iterable[str], modules: set[str]) -> set[str]:
    dependencies: set[str] = set()
    for imported in imports:
        candidate = imported
        while candidate:
            if candidate in modules:
                dependencies.add(candidate)
                break
            candidate = candidate.rpartition(".")[0]
    return dependencies


def _parse_dependencies(
    root: Path,
    paths: Iterable[Path],
    modules: set[str],
) -> tuple[dict[Path, set[str]], str | None]:
    result: dict[Path, set[str]] = {}
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            return {}, f"cannot parse {path.relative_to(root)}: {exc}"
        module = _module_name(root, path)
        imports = _import_names(tree, module, path.name == "__init__.py")
        result[path] = _local_dependencies(imports, modules)
    return result, None


def _depends_on(
    start: Iterable[str],
    changed_modules: set[str],
    module_dependencies: dict[str, set[str]],
) -> bool:
    pending = list(start)
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        if module in changed_modules:
            return True
        pending.extend(module_dependencies.get(module, ()))
    return False


def _documentation_only(path: PurePosixPath) -> bool:
    if path.parts and path.parts[0] == "docs":
        return True
    if len(path.parts) == 1:
        if path.name.startswith(("README", "CHANGELOG", "LICENSE")):
            return True
        return path.suffix.lower() in {".md", ".rst"}
    return False


def _test_aliases(root: Path, path: Path) -> set[str]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    aliases = {".".join(parts), parts[-1]}
    if parts and parts[0] == "tests":
        aliases.add(".".join(parts[1:]))
    return aliases


def _expand_dependent_tests(
    root: Path,
    selected: set[str],
) -> tuple[set[str], str | None]:
    if not selected:
        return selected, None
    missing = sorted(path for path in selected if not (root / path).is_file())
    if missing:
        return set(), f"mapped test is missing: {', '.join(missing)}"
    test_paths = sorted((root / "tests").rglob("test_*.py"))
    aliases: dict[str, set[Path]] = {}
    for path in test_paths:
        for alias in _test_aliases(root, path):
            aliases.setdefault(alias, set()).add(path)

    dependencies: dict[Path, set[Path]] = {}
    for path in test_paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            return set(), f"cannot parse {path.relative_to(root)}: {exc}"
        imports = _import_names(
            tree,
            _module_name(root, path),
            path.name == "__init__.py",
        )
        imported_tests: set[Path] = set()
        for imported in imports:
            candidate = imported
            while candidate:
                imported_tests.update(aliases.get(candidate, ()))
                if candidate in aliases:
                    break
                candidate = candidate.rpartition(".")[0]
        dependencies[path] = imported_tests

    expanded = {root / path for path in selected if (root / path).is_file()}
    changed = True
    while changed:
        changed = False
        for path, imported_tests in dependencies.items():
            if path not in expanded and imported_tests & expanded:
                expanded.add(path)
                changed = True
    return {path.relative_to(root).as_posix() for path in expanded}, None


def _frontend_tests(root: Path) -> set[str]:
    selected: set[str] = set()
    for path in (root / "tests").rglob("test_*.py"):
        relative = path.relative_to(root).as_posix()
        if "frontend" in path.name:
            selected.add(relative)
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if "marvis/static" in source or "static/app.js" in source:
            selected.add(relative)
    return selected


def _references_dynamic_pack(tree: ast.AST, pack_name: str) -> bool:
    """Detect structured Tool/Workflow references to a dynamically loaded pack."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(f"{pack_name}."):
                return True
            continue
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if (
                func_name == "ToolRef"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == pack_name
            ):
                return True
            continue
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value in {"pack", "plugin", "plugin_id"}
                and isinstance(value, ast.Constant)
                and value.value == pack_name
            ):
                return True
    return False


def _curated_dynamic_pack_tests(root: Path, pack_name: str) -> set[str]:
    """Return the maintained blast-radius group for a dynamic built-in pack.

    Dynamic Tool references are strings, so Python import analysis alone cannot
    discover their consumers.  A pack is eligible only after it has an explicit
    curated group here.  For Strategy, select every test that names the product
    surface plus the generic Plugin contract/packaging tests that enumerate all
    built-in manifests.
    """

    if pack_name not in CURATED_DYNAMIC_PACKS:
        return set()
    selected: set[str] = set()
    for path in (root / "tests").rglob("test_*.py"):
        relative = path.relative_to(root).as_posix()
        if (
            pack_name in path.name.lower()
            or path.name in STRATEGY_PACK_SHARED_TEST_NAMES
            or path.name in PLUGIN_CONTRACT_TEST_NAMES
        ):
            selected.add(relative)
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            # The common dependency-expansion pass below reports parse errors
            # as a conservative fallback, so do not silently bless this pack.
            continue
        imports = _import_names(
            tree,
            _module_name(root, path),
            path.name == "__init__.py",
        )
        pack_module = f"marvis.packs.{pack_name}"
        if any(
            imported == pack_module or imported.startswith(f"{pack_module}.")
            for imported in imports
        ) or _references_dynamic_pack(tree, pack_name):
            selected.add(relative)
    return selected


def select_affected_tests(root: Path, changed_files: Iterable[str]) -> Selection:
    root = root.resolve()
    changed = sorted(set(changed_files))
    if not changed:
        return Selection("none", reason="no changed files found")

    direct_tests: set[str] = set()
    changed_modules: set[str] = set()
    changed_dynamic_packs: set[str] = set()
    include_frontend = False
    saw_runtime_change = False

    for raw_path in changed:
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts:
            return Selection("fallback", reason=f"unsafe changed path: {raw_path}")
        if _documentation_only(path):
            continue
        if path.parts and path.parts[0] == "tests":
            if path.suffix == ".py" and path.name.startswith("test_"):
                direct_tests.add(path.as_posix())
                saw_runtime_change = True
                continue
            return Selection("fallback", reason=f"shared test support changed: {raw_path}")
        if path.as_posix() in {"scripts/check", "scripts/select_affected_tests.py"}:
            direct_tests.add("tests/test_check_affected.py")
            saw_runtime_change = True
            continue
        if len(path.parts) >= 2 and path.parts[:2] == ("marvis", "static"):
            include_frontend = True
            saw_runtime_change = True
            continue
        if len(path.parts) >= 2 and path.parts[:2] == ("marvis", "packs"):
            # Built-in packs are discovered from manifests and imported by
            # string at runtime.  Only packs with an explicit maintained test
            # group may use targeted selection; every other pack stays on the
            # conservative fallback.
            if len(path.parts) >= 3 and path.parts[2] in CURATED_DYNAMIC_PACKS:
                changed_dynamic_packs.add(path.parts[2])
                saw_runtime_change = True
                continue
            return Selection(
                "fallback",
                reason=f"dynamically loaded plugin pack changed: {raw_path}",
            )
        if path.parts and path.parts[0] == "marvis" and path.suffix == ".py":
            if path.name == "__init__.py":
                return Selection("fallback", reason=f"package initializer changed: {raw_path}")
            module_path = Path(path.as_posix())
            changed_modules.add(_module_name(Path("."), module_path))
            saw_runtime_change = True
            continue
        return Selection("fallback", reason=f"no safe test mapping for: {raw_path}")

    for pack_name in sorted(changed_dynamic_packs):
        pack_tests = _curated_dynamic_pack_tests(root, pack_name)
        if not pack_tests:
            return Selection(
                "fallback",
                reason=f"curated plugin pack has no mapped tests: {pack_name}",
            )
        direct_tests.update(pack_tests)

    if include_frontend:
        frontend = _frontend_tests(root)
        if not frontend:
            return Selection("fallback", reason="frontend change has no mapped tests")
        direct_tests.update(frontend)

    if changed_modules:
        module_paths = sorted((root / "marvis").rglob("*.py"))
        modules_by_path = {path: _module_name(root, path) for path in module_paths}
        local_modules = set(modules_by_path.values())

        parsed_modules, error = _parse_dependencies(root, module_paths, local_modules)
        if error:
            return Selection("fallback", reason=error)
        module_dependencies = {
            modules_by_path[path]: dependencies
            for path, dependencies in parsed_modules.items()
        }

        test_paths = sorted((root / "tests").rglob("test_*.py"))
        parsed_tests, error = _parse_dependencies(root, test_paths, local_modules)
        if error:
            return Selection("fallback", reason=error)

        conftest_paths = sorted((root / "tests").rglob("conftest.py"))
        parsed_conftests, error = _parse_dependencies(root, conftest_paths, local_modules)
        if error:
            return Selection("fallback", reason=error)
        shared_dependencies: set[str] = set()
        for dependencies in parsed_conftests.values():
            shared_dependencies.update(dependencies)

        mapped_by_module = {module: 0 for module in changed_modules}
        for test_path, dependencies in parsed_tests.items():
            combined = dependencies | shared_dependencies
            matched = {
                module
                for module in changed_modules
                if _depends_on(combined, {module}, module_dependencies)
            }
            if not matched:
                continue
            direct_tests.add(test_path.relative_to(root).as_posix())
            for module in matched:
                mapped_by_module[module] += 1

        unmapped = sorted(module for module, count in mapped_by_module.items() if count == 0)
        if unmapped:
            return Selection(
                "fallback",
                reason=f"source module has no dependent tests: {', '.join(unmapped)}",
            )

    # Tests may import helper functions from another test module.  Expand only
    # after every runtime-module mapping has been added, otherwise a source
    # change can select the helper while silently omitting its consumers.
    direct_tests, error = _expand_dependent_tests(root, direct_tests)
    if error:
        return Selection("fallback", reason=error)

    existing_tests = tuple(
        sorted(path for path in direct_tests if (root / path).is_file())
    )
    if direct_tests and len(existing_tests) != len(direct_tests):
        missing = sorted(direct_tests - set(existing_tests))
        return Selection("fallback", reason=f"mapped test is missing: {', '.join(missing)}")
    if existing_tests:
        return Selection("targeted", tests=existing_tests)
    if saw_runtime_change:
        return Selection("fallback", reason="runtime change has no mapped tests")
    return Selection("none", reason="documentation-only changes")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--diff-range", help="git diff range, for example origin/main...HEAD")
    parser.add_argument(
        "--null",
        action="store_true",
        help="terminate selected test paths with NUL for safe shell transport",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root.resolve()
    discovered = discover_changed_files(root, args.diff_range)
    if isinstance(discovered, Selection):
        selection = discovered
    else:
        selection = select_affected_tests(root, discovered)

    if selection.mode == "fallback":
        print(
            f"affected-test selection: {selection.reason}; fallback required",
            file=sys.stderr,
        )
        return FAST_FALLBACK_EXIT
    if selection.mode == "none":
        print(f"affected-test selection: {selection.reason}; pytest not needed", file=sys.stderr)
        return 0
    print(
        f"affected-test selection: {len(selection.tests)} test file(s)",
        file=sys.stderr,
    )
    if args.null:
        sys.stdout.buffer.write(
            b"".join(path.encode(errors="surrogateescape") + b"\0" for path in selection.tests)
        )
    else:
        print("\n".join(selection.tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
