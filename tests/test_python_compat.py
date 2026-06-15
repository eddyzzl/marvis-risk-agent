from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_kernel_imported_modules_do_not_require_stdlib_strenum():
    for relative_path in [
        "marvis/domain.py",
        "marvis/validation/results.py",
    ]:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert "from enum import StrEnum" not in source
        assert "marvis.compat import StrEnum" in source


def test_kernel_imported_modules_defer_annotations_at_runtime():
    for relative_path in [
        "marvis/domain.py",
        "marvis/output/excel.py",
        "marvis/output/styles.py",
        "marvis/validation/results.py",
        "marvis/validation/reproducibility.py",
        "marvis/validation/engine.py",
    ]:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")

        assert source.startswith("from __future__ import annotations\n")


def test_kernel_runtime_code_avoids_pep604_type_union_expressions():
    source = (PROJECT_ROOT / "marvis/report_texts.py").read_text(encoding="utf-8")

    assert "isinstance(sample_period, list | tuple)" not in source
    assert "isinstance(sample_period, (list, tuple))" in source
