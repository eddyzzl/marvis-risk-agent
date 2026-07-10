from pathlib import Path

import pytest

from marvis import notebook_io
from marvis.notebook_io import read_notebook


def _notebook_text(source: str) -> str:
    return (
        '{"cells":[{"cell_type":"markdown","id":"markdown-1","metadata":{},'
        f'"source":[{source!r}]}}],'
        '"metadata":{},"nbformat":4,"nbformat_minor":5}'
    ).replace("'", '"')


def test_read_notebook_preserves_unambiguous_cp1252_text(tmp_path: Path):
    notebook_path = tmp_path / "cp1252.ipynb"
    notebook_path.write_bytes(_notebook_text("score · label").encode("cp1252"))

    notebook = read_notebook(notebook_path)

    assert notebook.cells[0].source == "score · label"


def test_read_notebook_preserves_unambiguous_gb18030_text(tmp_path: Path):
    notebook_path = tmp_path / "gb18030.ipynb"
    notebook_path.write_bytes(_notebook_text("丂模型").encode("gb18030"))

    notebook = read_notebook(notebook_path)

    assert notebook.cells[0].source == "丂模型"


def test_read_notebook_rejects_ambiguous_legacy_encoding(tmp_path: Path):
    notebook_path = tmp_path / "ambiguous.ipynb"
    notebook_path.write_bytes(_notebook_text("caféx").encode("cp1252"))

    with pytest.raises(ValueError, match="ambiguous notebook encoding"):
        read_notebook(notebook_path)


def test_read_notebook_does_not_reparse_identical_gb18030_and_gbk_text(
    tmp_path: Path,
    monkeypatch,
):
    notebook_path = tmp_path / "gbk-subset.ipynb"
    notebook_path.write_bytes(_notebook_text("模型").encode("gbk"))
    original_reads = notebook_io.nbformat.reads
    parse_calls = {"count": 0}

    def counting_reads(*args, **kwargs):
        parse_calls["count"] += 1
        return original_reads(*args, **kwargs)

    monkeypatch.setattr(notebook_io.nbformat, "reads", counting_reads)

    with pytest.raises(ValueError, match="ambiguous notebook encoding"):
        read_notebook(notebook_path)

    assert parse_calls["count"] == 2
