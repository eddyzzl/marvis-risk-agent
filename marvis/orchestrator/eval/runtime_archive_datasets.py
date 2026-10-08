"""Bounded source and registered-data reads for independent archive checks."""
from pathlib import Path
import zipfile

import pandas as pd
import pyarrow.parquet as pq

from .runtime_archive_reader import _Unsupported, _file, _json, _original_path
from .runtime_contracts import digest

_MAX_ROWS = 100_000
_MAX_CELLS = 2_000_000


def _equal(actual, expected):
    if digest(actual) != digest(expected):
        raise ValueError("source evidence mismatch")


def _parquet(path):
    metadata = pq.read_metadata(path)
    if (metadata.num_rows > _MAX_ROWS or metadata.num_rows * metadata.num_columns > _MAX_CELLS
            or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) > 256 * 1024**2):
        raise _Unsupported("dataset recomputation row cell or expansion limit")
    return pd.read_parquet(path)


def _dataset(conn, archive, task_id, dataset_id, expected_hash):
    row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if (row is None or row["task_id"] != task_id or row["content_hash"] != expected_hash
            or row["format"] != "parquet" or Path(row["source_path"]).is_absolute()):
        raise ValueError("registered dataset binding mismatch")
    path = _file(archive, Path("workspace/datasets") / row["source_path"])
    if digest(path.read_bytes()) != expected_hash:
        raise ValueError("registered dataset bytes mismatch")
    frame = _parquet(path)
    if type(row["row_count"]) is not int or row["row_count"] != len(frame):
        raise ValueError("registered row count mismatch")
    return dict(row), frame


def _source_input(archive, manifest, binding, source, source_frame, scratch, material_index=0):
    from marvis.data.csv_ingest import read_csv_with_fallback_encoding
    from marvis.data.excel_ingest import ingest_sheet, detect_excel_container_format

    materials = binding.case.materials
    expected_materials = [{"original_relative_path": item.path,
                           "archive_path": f"inputs/{index:03d}{Path(item.path).suffix}",
                           "sha256": item.sha256, "role": item.role}
                          for index, item in enumerate(materials)]
    _equal(manifest["materials"], expected_materials)
    if type(material_index) is not int or not 0 <= material_index < len(materials):
        raise ValueError("invalid source material index")
    material, expected = materials[material_index], expected_materials[material_index]
    path = _file(archive, expected["archive_path"])
    if digest(path.read_bytes()) != material.sha256:
        raise ValueError("source material changed")
    identity = _json(archive, Path("workspace/datasets") / binding.task_id / ".source-identities" / f"{source['id']}.json")
    if identity.get("sha256") != material.sha256:
        raise ValueError("registered source differs from frozen material")
    uploaded = _original_path(archive, manifest["original_workspace"], identity["resolved_path"])
    if digest(uploaded.read_bytes()) != material.sha256:
        raise ValueError("original upload differs from frozen material")
    if path.suffix.lower() == ".parquet":
        original = _parquet(path)
    elif detect_excel_container_format(path) is not None:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as book:
                members = book.infolist()
                if len(members) > 10000 or sum(member.file_size for member in members) > 256 * 1024**2:
                    raise _Unsupported("source workbook expansion limit")
        if not source["sheet"]:
            raise ValueError("original workbook sheet absent")
        parquet, _ = ingest_sheet(path, source["sheet"], scratch / "ingested", max_rows=_MAX_ROWS)
        original = _parquet(parquet)
    elif path.suffix.lower() == ".csv":
        original, _ = read_csv_with_fallback_encoding(path, nrows=_MAX_ROWS + 1)
        if len(original) > _MAX_ROWS or original.size > _MAX_CELLS:
            raise _Unsupported("source row or cell limit")
    else:
        raise _Unsupported("source file format is not supported by archive reader")
    pd.testing.assert_frame_equal(original, source_frame, check_dtype=False, check_exact=True)
