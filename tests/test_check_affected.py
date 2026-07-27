from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.select_affected_tests import (
    Selection,
    discover_changed_files,
    select_affected_tests,
)


ROOT = Path(__file__).resolve().parents[1]


def _isolated_check_env(**updates: str) -> dict[str, str]:
    env = os.environ.copy()
    # Temporary repositories own their Git history. Do not let the outer CI
    # checkout force its commit range onto their affected-test selection.
    env.pop("CHECK_DIFF_RANGE", None)
    env.update(updates)
    return env


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _minimal_project(tmp_path: Path) -> Path:
    _write(tmp_path / "marvis" / "__init__.py")
    _write(tmp_path / "tests" / "test_placeholder.py", "def test_placeholder():\n    pass\n")
    return tmp_path


def test_selector_follows_transitive_local_imports(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "core.py", "def value():\n    return 1\n")
    _write(root / "marvis" / "service.py", "from marvis.core import value\n")
    _write(root / "marvis" / "other.py", "OTHER = True\n")
    _write(root / "tests" / "test_service.py", "import marvis.service\n")
    _write(root / "tests" / "test_other.py", "import marvis.other\n")

    selection = select_affected_tests(root, ["marvis/core.py"])

    assert selection.mode == "targeted"
    assert selection.tests == ("tests/test_service.py",)


def test_selector_expands_test_helpers_after_source_mapping(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    _write(
        root / "tests" / "test_helper.py",
        "import marvis.core\n\ndef helper():\n    return marvis.core.VALUE\n",
    )
    _write(
        root / "tests" / "test_consumer.py",
        "from test_helper import helper\n\ndef test_consumer():\n    assert helper() == 1\n",
    )

    selection = select_affected_tests(root, ["marvis/core.py"])

    assert selection.mode == "targeted"
    assert selection.tests == (
        "tests/test_consumer.py",
        "tests/test_helper.py",
    )


def test_selector_includes_changed_test_file_directly(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "tests" / "test_changed.py", "def test_changed():\n    pass\n")

    selection = select_affected_tests(root, ["tests/test_changed.py"])

    assert selection == Selection("targeted", tests=("tests/test_changed.py",))


def test_selector_includes_tests_that_import_changed_test_helper(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "tests" / "test_helper.py", "def helper():\n    return 1\n")
    _write(
        root / "tests" / "test_consumer.py",
        "from test_helper import helper\n\ndef test_consumer():\n    assert helper() == 1\n",
    )

    selection = select_affected_tests(root, ["tests/test_helper.py"])

    assert selection.mode == "targeted"
    assert selection.tests == (
        "tests/test_consumer.py",
        "tests/test_helper.py",
    )


def test_selector_maps_dotted_monkeypatch_target(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "dynamic.py", "VALUE = 1\n")
    _write(
        root / "tests" / "test_dynamic.py",
        'TARGET = "marvis.dynamic.VALUE"\n',
    )

    selection = select_affected_tests(root, ["marvis/dynamic.py"])

    assert selection.mode == "targeted"
    assert selection.tests == ("tests/test_dynamic.py",)


def test_selector_skips_pytest_for_documentation_only_changes(tmp_path: Path):
    root = _minimal_project(tmp_path)

    selection = select_affected_tests(root, ["README.md", "docs/runbook.md"])

    assert selection.mode == "none"
    assert selection.tests == ()


def test_selector_skips_pytest_when_worktree_has_no_changes(tmp_path: Path):
    root = _minimal_project(tmp_path)

    selection = select_affected_tests(root, [])

    assert selection.mode == "none"
    assert "no changed files" in selection.reason


def test_selector_falls_back_for_unknown_or_shared_support_changes(tmp_path: Path):
    root = _minimal_project(tmp_path)

    unknown = select_affected_tests(root, ["pyproject.toml"])
    conftest = select_affected_tests(root, ["tests/conftest.py"])

    assert unknown.mode == "fallback"
    assert "no safe test mapping" in unknown.reason
    assert conftest.mode == "fallback"
    assert "shared test support" in conftest.reason


def test_selector_does_not_treat_runtime_markdown_as_documentation(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "agent" / "prompt.md", "runtime prompt\n")

    selection = select_affected_tests(root, ["marvis/agent/prompt.md"])

    assert selection.mode == "fallback"
    assert "no safe test mapping" in selection.reason


def test_selector_falls_back_when_changed_module_has_no_dependent_test(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "orphan.py", "VALUE = 1\n")

    selection = select_affected_tests(root, ["marvis/orphan.py"])

    assert selection.mode == "fallback"
    assert "no dependent tests" in selection.reason


def test_selector_maps_curated_strategy_pack_to_strategy_and_plugin_tests(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "packs" / "strategy" / "tools.py", "VALUE = 1\n")
    _write(
        root / "tests" / "test_strategy_flow.py",
        'TOOL_ID = "strategy.build_strategy"\n',
    )
    _write(
        root / "tests" / "test_direct_consumer.py",
        "from marvis.packs.strategy import tools\n",
    )
    _write(root / "tests" / "test_plugin_registry.py", "def test_registry():\n    pass\n")
    _write(root / "tests" / "test_unrelated.py", "def test_unrelated():\n    pass\n")

    selection = select_affected_tests(
        root,
        ["marvis/packs/strategy/tools.py"],
    )

    assert selection == Selection(
        "targeted",
        tests=(
            "tests/test_direct_consumer.py",
            "tests/test_plugin_registry.py",
            "tests/test_strategy_flow.py",
        ),
    )


def test_real_strategy_pack_mapping_covers_direct_import_consumers():
    selection = select_affected_tests(
        ROOT,
        ["marvis/packs/strategy/contracts.py"],
    )
    assert selection.mode == "targeted"
    selected = set(selection.tests)
    direct_consumers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if "marvis.packs.strategy" in path.read_text(encoding="utf-8")
    }

    assert direct_consumers
    assert direct_consumers <= selected
    assert {
        "tests/test_orch_api.py",
        "tests/test_orch_eval.py",
        "tests/test_orch_planner.py",
        "tests/test_orch_skills.py",
    } <= selected


def test_selector_falls_back_for_unmapped_dynamic_plugin_pack(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "packs" / "modeling" / "tools.py", "VALUE = 1\n")

    selection = select_affected_tests(root, ["marvis/packs/modeling/tools.py"])

    assert selection.mode == "fallback"
    assert "dynamically loaded plugin pack" in selection.reason


def test_discover_changed_files_uses_explicit_diff_range(tmp_path: Path):
    root = _minimal_project(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "add core",
        ],
        cwd=root,
        check=True,
    )

    changed = discover_changed_files(root, f"{base}...HEAD")

    assert changed == ["marvis/core.py"]


def test_discover_changed_files_ignores_untracked_non_runtime_tree(tmp_path: Path):
    root = _minimal_project(tmp_path)
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "marvis" / "core.py", "VALUE = 2\n")
    _write(root / "website" / "PUBLIC_LAUNCH_CHECKLIST.md", "user-owned\n")

    changed = discover_changed_files(root, None)

    assert changed == ["marvis/core.py"]


def test_discover_changed_files_keeps_untracked_runtime_and_test_files(tmp_path: Path):
    root = _minimal_project(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "marvis" / "new_runtime.py", "VALUE = 1\n")
    _write(root / "tests" / "test_new_runtime.py", "import marvis.new_runtime\n")

    changed = discover_changed_files(root, None)

    assert changed == ["marvis/new_runtime.py", "tests/test_new_runtime.py"]


def test_check_fast_excludes_heavy_and_pmml_runtime_tests(tmp_path: Path):
    capture = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    _write(
        fake_python,
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(ROOT / "scripts" / "check"),
            "--fast",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-q",
        "-m",
        "not slow and not e2e and not llm and not pmml_runtime",
    ]


def test_check_help_documents_affected_mode_and_diff_range():
    completed = subprocess.run(
        [str(ROOT / "scripts" / "check"), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0
    assert "--affected" in completed.stdout
    assert "--affected-full" in completed.stdout
    assert "--profile" in completed.stdout
    assert "CHECK_DIFF_RANGE" in completed.stdout
    assert "not slow and not e2e and not llm and not pmml_runtime" in completed.stdout


def test_ci_runs_pmml_runtime_separately_from_fast_gate():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    pmml_job = workflow.split("\n  pmml_runtime:\n", maxsplit=1)[1].split(
        "\n  security:\n", maxsplit=1
    )[0]

    assert "uv run scripts/check --fast" in workflow
    assert "if: github.event_name != 'workflow_dispatch'" in pmml_job
    assert "uses: actions/setup-java@v4" in pmml_job
    assert "uv run python -m pytest -q -m pmml_runtime" in pmml_job


def test_ci_runs_strategy_smoke_separately_without_jvm_or_manual_duplication():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    strategy_job = workflow.split("\n  strategy_smoke:\n", maxsplit=1)[1].split(
        "\n  pmml_runtime:\n", maxsplit=1
    )[0]

    assert "if: github.event_name != 'workflow_dispatch'" in strategy_job
    assert "uses: actions/setup-java@" not in strategy_job
    assert "uv run python -m pytest -q" in strategy_job
    assert (
        "tests/test_strategy_candidate_request_e2e.py::"
        "test_natural_language_candidate_auto_runs_to_only_adoption_gate_and_rerenders_doc"
        in strategy_job
    )
    assert (
        "tests/test_strategy_standard_workflows_e2e.py::"
        "test_natural_language_standard_workflow_executes_and_exports"
        in strategy_job
    )
    assert (
        "tests/test_strategy_monitoring_e2e.py::"
        "test_strategy_monitoring_e2e_red_then_new_version_then_report"
        in strategy_job
    )


def test_check_fast_keeps_marker_filter_with_custom_pytest_args(tmp_path: Path):
    capture = tmp_path / "python-args.txt"
    fake_python = tmp_path / "python"
    _write(
        fake_python,
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(ROOT / "scripts" / "check"),
            "--fast",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
            "--",
            "-x",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-m",
        "not slow and not e2e and not llm and not pmml_runtime",
        "-x",
    ]


def test_check_affected_runs_only_mapped_test_file(tmp_path: Path):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    _write(root / "tests" / "test_core.py", "import marvis.core\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "marvis" / "core.py", "VALUE = 2\n")

    capture = tmp_path / "affected-python-args.txt"
    fake_python = tmp_path / "affected-python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'if [ "$1" = "scripts/select_affected_tests.py" ]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        REAL_PYTHON=sys.executable,
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "affected-test selection: 1 test file(s)" in completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "tests/test_core.py",
        "-q",
        "-m",
        "not slow and not e2e and not llm and not pmml_runtime",
    ]


def test_check_affected_full_runs_all_tiers_in_mapped_test_file(tmp_path: Path):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    _write(root / "tests" / "test_core.py", "import marvis.core\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "marvis" / "core.py", "VALUE = 2\n")

    capture = tmp_path / "affected-full-python-args.txt"
    fake_python = tmp_path / "affected-full-python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'if [ "$1" = "scripts/select_affected_tests.py" ]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        REAL_PYTHON=sys.executable,
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected-full",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "affected-test selection: 1 test file(s)" in completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "tests/test_core.py",
        "-q",
    ]


def test_check_affected_succeeds_when_mapped_file_has_no_fast_tests(tmp_path: Path):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(
        root / "pyproject.toml",
        "[tool.pytest.ini_options]\nmarkers = ['slow: slow test']\n",
    )
    _write(root / ".gitignore", ".pytest_cache/\n__pycache__/\n")
    _write(
        root / "tests" / "test_only_slow.py",
        "import pytest\n\n@pytest.mark.slow\ndef test_only_slow():\n    pass\n",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(
        root / "tests" / "test_only_slow.py",
        "import pytest\n\n@pytest.mark.slow\ndef test_only_slow():\n    assert True\n",
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=root,
        env=_isolated_check_env(PYTHON=sys.executable),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no fast tests selected" in completed.stderr

    narrowed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
            "--",
            "-q",
            "-k",
            "__definitely_not_a_test_name__",
        ],
        cwd=root,
        env=_isolated_check_env(PYTHON=sys.executable),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert narrowed.returncode == 5
    assert "no fast tests selected" not in narrowed.stderr


def test_check_affected_fallback_keeps_one_fast_marker_with_custom_args(tmp_path: Path):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(root / "pyproject.toml", "[project]\nname = 'before'\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "pyproject.toml", "[project]\nname = 'after'\n")

    capture = tmp_path / "fallback-python-args.txt"
    fake_python = tmp_path / "fallback-python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'if [ "$1" = "scripts/select_affected_tests.py" ]; then\n'
        '  "$REAL_PYTHON" "$@"\n'
        "  exit $?\n"
        "fi\n"
        'printf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        REAL_PYTHON=sys.executable,
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
            "--",
            "-x",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "running fast tier" in completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-m",
        "not slow and not e2e and not llm and not pmml_runtime",
        "-x",
    ]


@pytest.mark.parametrize(
    ("mode", "expected_prefix"),
    [
        (
            "--fast",
            [
                "-m",
                "pytest",
                "-q",
                "-m",
                "not slow and not e2e and not llm and not pmml_runtime",
            ],
        ),
        ("--affected-full", ["-m", "pytest", "tests/test_core.py", "-q"]),
    ],
)
def test_check_profile_appends_duration_reporting(
    tmp_path: Path,
    mode: str,
    expected_prefix: list[str],
):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(root / "marvis" / "core.py", "VALUE = 1\n")
    _write(root / "tests" / "test_core.py", "import marvis.core\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "marvis" / "core.py", "VALUE = 2\n")

    capture = tmp_path / f"{mode.removeprefix('--')}-profile-args.txt"
    fake_python = tmp_path / f"{mode.removeprefix('--')}-profile-python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'if [ "$1" = "scripts/select_affected_tests.py" ]; then\n'
        '  exec "$REAL_PYTHON" "$@"\n'
        "fi\n"
        'printf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        REAL_PYTHON=sys.executable,
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            mode,
            "--profile",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        *expected_prefix,
        "--durations=50",
        "--durations-min=0.5",
    ]


def test_check_affected_full_fallback_runs_all_tiers(tmp_path: Path):
    root = _minimal_project(tmp_path / "repo")
    shutil.copytree(ROOT / "scripts", root / "scripts", dirs_exist_ok=True)
    _write(root / "pyproject.toml", "[project]\nname = 'before'\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MARVIS Tests",
            "-c",
            "user.email=tests@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=root,
        check=True,
    )
    _write(root / "pyproject.toml", "[project]\nname = 'after'\n")

    capture = tmp_path / "fallback-full-python-args.txt"
    fake_python = tmp_path / "fallback-full-python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'if [ "$1" = "scripts/select_affected_tests.py" ]; then\n'
        '  "$REAL_PYTHON" "$@"\n'
        "  exit $?\n"
        "fi\n"
        'printf "%s\\n" "$@" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)
    env = _isolated_check_env(
        PYTHON=str(fake_python),
        REAL_PYTHON=sys.executable,
        CHECK_CAPTURE=str(capture),
    )

    completed = subprocess.run(
        [
            str(root / "scripts" / "check"),
            "--affected-full",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "running all test tiers" in completed.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "pytest",
        "-q",
    ]


@pytest.mark.parametrize("affected_flag", ["--affected", "--affected-full"])
def test_check_rejects_fast_and_affected_together(affected_flag: str):
    completed = subprocess.run(
        [str(ROOT / "scripts" / "check"), "--fast", affected_flag],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "cannot be used together" in completed.stderr


def test_check_rejects_custom_marker_with_affected():
    completed = subprocess.run(
        [
            str(ROOT / "scripts" / "check"),
            "--affected",
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
            "--",
            "-m",
            "llm",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "custom pytest -m" in completed.stderr
