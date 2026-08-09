import ast
from pathlib import Path

import pytest

from marvis import __main__ as cli


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_degraded_output_touchpoint_corpus_is_test_only():
    production_corpus = REPO_ROOT / "marvis/orchestrator/eval/touchpoint_cases.py"
    test_corpus = REPO_ROOT / "tests/eval/touchpoint_cases.py"

    assert not production_corpus.exists()
    assert test_corpus.is_file()


def test_production_eval_cli_does_not_import_touchpoint_test_corpus():
    cli_path = REPO_ROOT / "marvis/orchestrator/eval/cli.py"
    tree = ast.parse(cli_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )

    assert "marvis.orchestrator.eval.cases" in imported_modules
    assert not any(module.startswith("tests") for module in imported_modules)
    assert not any(module.endswith("touchpoint_cases") for module in imported_modules)


def test_eval_llm_help_names_planning_and_guardrail_suite(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])

    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "orchestrator planning/guardrail eval suite" in help_text
    assert "LLM-touchpoint eval suite" not in help_text
