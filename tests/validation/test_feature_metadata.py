from __future__ import annotations

import base64
import io
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marvis.validation import feature_metadata
from marvis.validation.feature_categories import (
    feature_category_resolution_from_metadata,
)
from marvis.validation.feature_metadata import (
    DEFAULT_MAX_METADATA_BYTES,
    MAX_DIAGNOSTIC_CHARS,
    MAX_DIAGNOSTICS,
    FeatureMetadataSelection,
    inspect_feature_metadata,
    normalize_feature_metadata,
)
from marvis.validation.input_contracts import (
    PMML_INPUT_MANIFEST_SCHEMA,
    PmmlInputManifest,
    StressUnit,
)


def _manifest(
    *features: str,
    stress_units: tuple[StressUnit, ...] | None = None,
    unsupported_derivations: tuple[str, ...] = (),
) -> PmmlInputManifest:
    units = stress_units or tuple(StressUnit(name, (name,), ()) for name in features)
    raw_fields = tuple(
        dict.fromkeys(field for unit in units for field in unit.raw_input_fields)
    )
    return PmmlInputManifest(
        schema_version=PMML_INPUT_MANIFEST_SCHEMA,
        raw_required_fields=raw_fields,
        derived_fields=(),
        model_features=tuple(features),
        stress_units=units,
        unsupported_derivations=unsupported_derivations,
        output_candidates=("probability_1",),
        algorithm="xgb",
    )


def test_gb18030_metadata_accepts_zero_importance_and_alias_columns(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "数据字典.csv"
    path.write_bytes(
        "指标英文,source,feature_importance\nx1,征信,1.5\nx2,内部,0\n".encode(
            "gb18030"
        )
    )

    inspection = inspect_feature_metadata(path, direct_manifest)
    selection = inspection.only_valid_selection()
    resolution = normalize_feature_metadata(
        path, selection=selection, manifest=direct_manifest
    )

    assert resolution.coverage.feature == 1.0
    assert resolution.coverage.category == 1.0
    assert resolution.coverage.importance == 1.0
    assert resolution.rows[1].importance == 0.0


def test_conflicting_duplicate_importance_blocks(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\nx1,征信,1\nx1,征信,2\nx2,内部,0\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("conflicting feature metadata for x1" in error for error in inspection.blocking_errors)
    with pytest.raises(ValueError, match="conflicting feature metadata for x1"):
        inspection.only_valid_selection()


def test_missing_importance_is_blocking_not_selection_ambiguity(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\nx1,征信,1\nx2,内部,\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("importance" in value for value in inspection.blocking_errors)
    with pytest.raises(ValueError, match="importance"):
        inspection.only_valid_selection()


def test_missing_low_importance_pmml_feature_is_supplemented_from_namespace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\n"
        "ppdi_known,朴道v4,12.5\n"
        "bh_known,百行V2,8.0\n"
        "qrorg_known,人行,3.0\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        "ppdi_known", "ppdi_missing", "bh_known", "qrid_missing"
    )

    selection = inspect_feature_metadata(path, manifest).only_valid_selection()
    resolution = normalize_feature_metadata(
        path, selection=selection, manifest=manifest
    )

    by_feature = {row.feature: row for row in resolution.rows}
    assert by_feature["ppdi_missing"].category == "朴道v4"
    assert by_feature["ppdi_missing"].importance == 0.0
    assert by_feature["ppdi_missing"].in_pmml is True
    assert by_feature["qrid_missing"].category == "人行"
    assert resolution.coverage.feature == 1.0


def test_missing_pmml_feature_with_ambiguous_namespace_stays_blocking(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "feature,category,importance\n"
        "qrorg_known,人行,3.0\n"
        "qrfoo_known,其他,2.0\n",
        encoding="utf-8",
    )
    manifest = _manifest("qrorg_known", "qrfoo_known", "qrid_missing")

    inspection = inspect_feature_metadata(path, manifest)

    assert inspection.selections == ()
    assert "qrid_missing" in " ".join(inspection.blocking_errors)


@pytest.mark.parametrize(
    "category_alias",
    ["source", "特征分类", "产品名称", "product", "特征产品", "特征信源"],
)
def test_real_corpus_category_aliases_are_exact_candidates(
    tmp_path: Path, direct_manifest, category_alias: str
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        f"feature,{category_alias},importance\nx1,征信,1\nx2,内部,-0.25\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)
    selection = inspection.only_valid_selection()

    assert selection.category_col == category_alias
    resolution = normalize_feature_metadata(
        path, selection=selection, manifest=direct_manifest
    )
    assert resolution.rows[1].importance == -0.25


@pytest.mark.parametrize("feature_alias", ["特征英文名", "var"])
@pytest.mark.parametrize("importance_alias", ["特征重要性", "score"])
def test_inventory_feature_and_importance_aliases_are_supported(
    tmp_path: Path,
    feature_alias: str,
    importance_alias: str,
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        f"{feature_alias},category,{importance_alias}\n001,内部,0\n",
        encoding="utf-8",
    )
    manifest = _manifest("001")

    selection = inspect_feature_metadata(path, manifest).only_valid_selection()

    assert selection.feature_col == feature_alias
    assert selection.importance_col == importance_alias


def test_feature_identifiers_remain_exact_strings_with_leading_zeroes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,特征分类,importance\n001,征信,1\n",
        encoding="utf-8",
    )
    manifest = _manifest("001")

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, manifest).only_valid_selection(),
        manifest=manifest,
    )

    assert resolution.rows[0].feature == "001"


def test_identical_duplicate_rows_merge_and_extras_keep_source_order(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\n"
        "extra_b,额外,3\n"
        "x2,征信,0\n"
        "x1,内部,1\n"
        "x1,内部,1.0\n"
        "extra_a,额外,2\n",
        encoding="utf-8",
    )

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(
            path, direct_manifest
        ).only_valid_selection(),
        manifest=direct_manifest,
    )

    assert [row.feature for row in resolution.rows] == [
        "x1",
        "x2",
        "extra_b",
        "extra_a",
    ]
    assert [row.in_pmml for row in resolution.rows] == [True, True, False, False]
    assert resolution.extra_features == ("extra_b", "extra_a")


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ("x1,内部,1\nx1,征信,1\nx2,征信,0\n", "x1"),
        ("x1,内部,1\nx1,内部,2\nx2,征信,0\n", "x1"),
    ],
)
def test_duplicate_category_or_importance_disagreement_blocks(
    tmp_path: Path, direct_manifest, rows: str, expected: str
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text("feature,category,importance\n" + rows, encoding="utf-8")

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any(expected in item for item in inspection.blocking_errors)


def test_multiple_alias_columns_require_user_confirmation(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,指标英文,category,importance\n"
        "x1,x1,内部,1\n"
        "x2,x2,征信,0\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert [selection.feature_col for selection in inspection.selections] == [
        "feature",
        "指标英文",
    ]
    assert inspection.blocking_errors == ()
    with pytest.raises(ValueError, match="user confirmation"):
        inspection.only_valid_selection()


def test_multiple_valid_excel_sheets_preserve_workbook_order_and_require_confirmation(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["内部", "征信"], "importance": [1, 0]}
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="second_name_but_first", index=False)
        frame.to_excel(writer, sheet_name="first_name_but_second", index=False)

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert [selection.sheet_name for selection in inspection.selections] == [
        "second_name_but_first",
        "first_name_but_second",
    ]
    with pytest.raises(ValueError, match="user confirmation"):
        inspection.only_valid_selection()


def test_manifest_features_are_normalized_once_per_inspection(
    tmp_path: Path, direct_manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,特征名,category,importance\n"
        "x1,x1,内部,1\nx2,x2,征信,0\n",
        encoding="utf-8",
    )
    original = feature_metadata._normalized_manifest_features
    calls = 0

    def counted_normalize(features):
        nonlocal calls
        calls += 1
        return original(features)

    monkeypatch.setattr(
        feature_metadata, "_normalized_manifest_features", counted_normalize
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert len(inspection.selections) == 2
    assert calls == 1


def test_inspection_work_budget_is_global_across_candidate_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {
            "feature": ["x1", "x2", "x3"],
            "category": ["内部", "内部", "征信"],
            "importance": [1, 2, 3],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="one", index=False)
        frame.to_excel(writer, sheet_name="two", index=False)
    monkeypatch.setattr(feature_metadata, "MAX_INSPECTION_WORK_UNITS", 20, raising=False)

    with pytest.raises(ValueError, match="inspection work limit"):
        inspect_feature_metadata(path, _manifest("x1", "x2", "x3"))


def test_selection_budget_is_global_across_excel_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {"feature": ["x1"], "category": ["内部"], "importance": [1]}
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="one", index=False)
        frame.to_excel(writer, sheet_name="two", index=False)
    monkeypatch.setattr(feature_metadata, "MAX_SELECTIONS", 1)

    with pytest.raises(ValueError, match="selection limit"):
        inspect_feature_metadata(path, _manifest("x1"))


def test_declared_merged_excel_category_cells_are_expanded_without_generic_ffill(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["征信", None], "importance": [1, 0]}
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
        worksheet = writer.book.active
        worksheet.merge_cells("B2:B3")

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(
            path, direct_manifest
        ).only_valid_selection(),
        manifest=direct_manifest,
    )

    assert [row.category for row in resolution.rows] == ["征信", "征信"]


def test_merge_expansion_only_writes_selected_columns_and_charges_intersection() -> None:
    raw_header = ["feature", "importance", "category"]
    raw_rows = [[f"x{index}", index, None] for index in range(2_000)]
    merged = feature_metadata._MergedRange(3, 2, 512, 2_001)
    ledger = feature_metadata._InspectionLedger(
        work_limit=1,
        merged_cell_limit=102_400_000,
    )

    feature_metadata._expand_declared_merges(
        raw_header=raw_header,
        raw_rows=raw_rows,
        merged_ranges=(merged,),
        selected_columns=frozenset({3}),
        ledger=ledger,
    )

    assert max(map(len, raw_rows)) == 3
    assert ledger.expanded_cells == 2_000


def test_merge_expansion_budget_is_global_across_xlsx_sheets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {
            "feature": [f"x{index}" for index in range(100)],
            "importance": list(range(100)),
            "category": ["征信", *([None] * 99)],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in ("one", "two"):
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            writer.book[sheet_name].merge_cells("C2:SR101")
    monkeypatch.setattr(
        feature_metadata, "MAX_MERGED_EXPANDED_CELLS", 150, raising=False
    )

    with pytest.raises(ValueError, match="merged expansion limit"):
        inspect_feature_metadata(
            path, _manifest(*(f"x{index}" for index in range(100)))
        )


def test_unmerged_blank_excel_category_is_not_forward_filled(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["征信", None], "importance": [1, 0]}
    ).to_excel(path, index=False)

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("category" in item for item in inspection.blocking_errors)


def test_feature_matching_trims_only_surrounding_whitespace_and_is_case_sensitive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\n X1 , 内部 ,1\nx1,征信,2\n",
        encoding="utf-8",
    )
    manifest = _manifest("X1", "x1")

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, manifest).only_valid_selection(),
        manifest=manifest,
    )

    assert [(row.feature, row.category) for row in resolution.rows] == [
        ("X1", "内部"),
        ("x1", "征信"),
    ]


@pytest.mark.parametrize("bad_importance", ["", "NaN", "inf", "-inf", "abc"])
def test_blank_nonfinite_and_nonnumeric_importance_block(
    tmp_path: Path, direct_manifest, bad_importance: str
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\n"
        f"x1,内部,1\nx2,征信,{bad_importance}\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("importance" in item for item in inspection.blocking_errors)


def test_blank_category_blocks_complete_coverage(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx1,内部,1\nx2,  ,0\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("category" in item for item in inspection.blocking_errors)


def test_missing_many_pmml_features_with_source_extras_is_hard_block_and_bounded(
    tmp_path: Path,
) -> None:
    manifest = _manifest(*(f"x{index}" for index in range(50)))
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx0,内部,1\nextra_a,额外,2\nextra_b,额外,3\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, manifest)

    assert inspection.selections == ()
    assert inspection.blocking_errors
    assert len(inspection.blocking_errors) <= MAX_DIAGNOSTICS
    assert all(len(item) <= MAX_DIAGNOSTIC_CHARS for item in inspection.blocking_errors)
    assert any("missing" in item and "x1" in item for item in inspection.blocking_errors)


def test_legacy_dictionary_without_importance_column_is_hard_block(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text(
        "feature,category\nx1,内部\nx2,征信\n", encoding="utf-8"
    )

    inspection = inspect_feature_metadata(path, direct_manifest)

    assert inspection.selections == ()
    assert any("importance column" in item for item in inspection.blocking_errors)


@pytest.mark.parametrize(
    "header", ["feature,feature,category,importance", "feature,,category,importance"]
)
def test_csv_rejects_raw_duplicate_or_blank_headers_before_pandas_mangling(
    tmp_path: Path, direct_manifest, header: str
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        f"{header}\nx1,x1,内部,1\nx2,x2,征信,0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="column name"):
        inspect_feature_metadata(path, direct_manifest)


def test_metadata_csv_byte_row_and_column_limits_precede_full_materialization(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx1,内部,1\nx2,征信,0\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="byte limit"):
        inspect_feature_metadata(path, direct_manifest, max_bytes=8)
    with pytest.raises(ValueError, match="row limit"):
        inspect_feature_metadata(path, direct_manifest, max_rows=1)
    with pytest.raises(ValueError, match="column limit"):
        inspect_feature_metadata(path, direct_manifest, max_columns=2)


def test_csv_reader_streams_without_stringio_full_text_copy(
    tmp_path: Path, direct_manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx1,内部,1\nx2,征信,0\n", encoding="utf-8"
    )

    def fail_stringio(*_args, **_kwargs):
        raise AssertionError("CSV reader must not create a full decoded StringIO copy")

    monkeypatch.setattr(io, "StringIO", fail_stringio)

    assert inspect_feature_metadata(path, direct_manifest).only_valid_selection()


def test_csv_cell_and_total_decoded_character_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nlong_feature,内部,1\n", encoding="utf-8"
    )
    manifest = _manifest("long_feature")
    monkeypatch.setattr(feature_metadata, "MAX_METADATA_CELL_CHARS", 10, raising=False)

    with pytest.raises(ValueError, match="cell length"):
        inspect_feature_metadata(path, manifest)

    monkeypatch.setattr(
        feature_metadata, "MAX_METADATA_CELL_CHARS", 100, raising=False
    )
    monkeypatch.setattr(
        feature_metadata, "MAX_METADATA_DECODED_CHARS", 20, raising=False
    )
    with pytest.raises(ValueError, match="decoded character"):
        inspect_feature_metadata(path, manifest)


def test_alias_cartesian_product_normalizes_each_importance_column_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    headers = [
        *feature_metadata.FEATURE_ALIASES,
        *feature_metadata.CATEGORY_ALIASES,
        *feature_metadata.IMPORTANCE_ALIASES,
    ]
    lines = [",".join(headers)]
    for index in range(100):
        lines.append(
            ",".join(
                [f"x{index}"] * len(feature_metadata.FEATURE_ALIASES)
                + ["内部"] * len(feature_metadata.CATEGORY_ALIASES)
                + [str(index)] * len(feature_metadata.IMPORTANCE_ALIASES)
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = _manifest(*(f"x{index}" for index in range(100)))
    original = pd.to_numeric
    calls = 0

    def counted_to_numeric(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "to_numeric", counted_to_numeric)
    started = time.monotonic()

    inspection = inspect_feature_metadata(path, manifest)

    elapsed = time.monotonic() - started
    assert len(inspection.selections) == 588
    assert calls <= len(feature_metadata.IMPORTANCE_ALIASES)
    assert elapsed < 2.0


def test_default_metadata_byte_cap_covers_verified_large_workbook_inventory() -> None:
    assert DEFAULT_MAX_METADATA_BYTES >= 64 * 1024 * 1024


def test_xlsx_archive_limits_are_checked_before_openpyxl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(
        {"feature": ["x1"], "category": ["内部"], "importance": [1]}
    ).to_excel(path, index=False)
    monkeypatch.setattr(
        feature_metadata, "MAX_XLSX_ARCHIVE_ENTRIES", 1, raising=False
    )

    with pytest.raises(ValueError, match="archive entry limit"):
        inspect_feature_metadata(path, _manifest("x1"))


def test_xlsx_archive_rejects_excessive_compression_ratio_before_xml_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xlsx"
    monkeypatch.setattr(
        feature_metadata, "MAX_XLSX_COMPRESSION_RATIO", 5.0, raising=False
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"x" * 20_000)

    with pytest.raises(ValueError, match="compression ratio"):
        inspect_feature_metadata(path, _manifest("x1"))


def test_xlsx_merge_xml_rejects_doctype_and_entity_declarations(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.xlsx"
    frame = pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["征信", None], "importance": [1, 0]}
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False)
        writer.book.active.merge_cells("B2:B3")
    rewritten = tmp_path / "unsafe.xlsx"
    with ZipFile(path, "r") as source, ZipFile(
        rewritten, "w", compression=ZIP_DEFLATED
    ) as destination:
        for item in source.infolist():
            payload = source.read(item.filename)
            if item.filename == "xl/worksheets/sheet1.xml":
                payload = payload.replace(
                    b"<worksheet ",
                    b'<!DOCTYPE worksheet [<!ENTITY boom "boom">]><worksheet ',
                    1,
                )
            destination.writestr(item, payload)
    rewritten.replace(path)

    with pytest.raises(ValueError, match="unsafe.*XML|DOCTYPE|ENTITY"):
        inspect_feature_metadata(path, direct_manifest)


def test_xlsx_archive_thresholds_cover_verified_large_real_workbook() -> None:
    assert feature_metadata.MAX_XLSX_MEMBER_UNCOMPRESSED_BYTES >= 256 * 1024 * 1024
    assert feature_metadata.MAX_XLSX_TOTAL_UNCOMPRESSED_BYTES >= 512 * 1024 * 1024


def test_excel_row_and_column_limits_are_checked_per_sheet(
    tmp_path: Path, direct_manifest
) -> None:
    row_path = tmp_path / "rows.xlsx"
    pd.DataFrame(
        {"feature": ["x1", "x2"], "category": ["内部", "征信"], "importance": [1, 0]}
    ).to_excel(row_path, index=False)
    with pytest.raises(ValueError, match="row limit"):
        inspect_feature_metadata(row_path, direct_manifest, max_rows=1)

    column_path = tmp_path / "columns.xlsx"
    pd.DataFrame(
        {"feature": ["x1"], "category": ["内部"], "extra": [0], "importance": [1]}
    ).to_excel(column_path, index=False)
    with pytest.raises(ValueError, match="column limit"):
        inspect_feature_metadata(column_path, _manifest("x1"), max_columns=3)


def test_oversized_non_candidate_excel_sheet_is_skipped_before_row_scan(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"metric": [1, 2, 3], "value": [4, 5, 6]}).to_excel(
            writer, sheet_name="单变量分析", index=False
        )
        pd.DataFrame(
            {"feature": ["x1"], "category": ["内部"], "importance": [1]}
        ).to_excel(writer, sheet_name="特征重要性", index=False)

    inspection = inspect_feature_metadata(path, _manifest("x1"), max_rows=1)

    assert inspection.only_valid_selection().sheet_name == "特征重要性"


def test_candidate_excel_ignores_unselected_blank_headers_and_pivot_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            {"feature": ["x1"], "category": ["内部"], "importance": [1]}
        ).to_excel(writer, index=False)
        worksheet = writer.book.active
        worksheet.cell(1, 4, None)
        worksheet.cell(2, 4, "unrelated")
        worksheet.cell(1, 5, "pivot")
        worksheet.cell(2, 5, 123)

    inspection = inspect_feature_metadata(path, _manifest("x1"))

    assert inspection.only_valid_selection().feature_col == "feature"


def test_feature_product_and_source_columns_remain_category_confirmation_candidates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(
        {
            "feature": ["x1"],
            "特征产品": ["产品A"],
            "特征信源": ["信源B"],
            "importance": [1],
        }
    ).to_excel(path, index=False)

    inspection = inspect_feature_metadata(path, _manifest("x1"))

    assert [selection.category_col for selection in inspection.selections] == [
        "特征产品",
        "特征信源",
    ]
    with pytest.raises(ValueError, match="user confirmation"):
        inspection.only_valid_selection()


def test_xls_candidate_and_declared_merged_category_are_supported(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.xls"
    compressed_fixture = (
        "eNrtWE1oE0EU/mby0yS0aVJToRVKKFi19lK9CNKuCtqTtXpRRKjbdFtKa1LWCNaD"
        "ttYcBcGT4qXQi5eqF39QRG8ehIoeBEFIKp48CQoe2qxvXnbT1ObQgBaV+ZZ58+"
        "bNfDOzM2/e/rxZjOfnHrQW8At64UPRCSNYYROUwl4hBqp3HKV6eYiSo/FPIRyij"
        "QwG8LThdZ3aQ7XfBUjc978kCSxROoNJ9GfSVnITcYjnYAo1hx6SAnfIEkULz6qJ"
        "ZYrlFpb3uOUzlgfYcp1lD7XNi9NYNPo797lefEq2c10Uqt9HzPnAlm4045Xy4is"
        "3RKltAAftMXPi76xo89djHrRvfVbass2JPBK0gfP47iSBb95JfZHU9s21C5D9x1"
        "p7XRX7TekHpuGcZQfPUUxd8pUC6YhlZi/Y1vll7OcjqRLts2umQJsys9Zoxp6KAG"
        "PnJjN21kynLHLhi91K7KH+ZgYLxyIqYvMJj6054Q3s+fUkh9HIepz9P0ZDL9/9+"
        "vbo0IAxyJZpjuql2L9dTRcOZhSDyFGu8ZVlJzN2s7zKvW5jvZVlgryW8o6BZlc5"
        "MsttrnFtB42zl/HO2FGh7yQ99+X447bcJ2MX6Qt9hUuJhffGHNrpWTRMfHXNokt0"
        "idu3FJ4YXi7cOPGRZcu6mBGSMXfujvuAa8QKIqzGWZZKanVEuSTdtVJsUYUtmO1"
        "z2wtmB6h0WaqSYgddtqzClsxW6/yc7la6YyuuLI/82e1XsuyVTXiozBT3VhGBho"
        "aGhoaGhoaGxnoIfqMtvb+rt87A6scG/9dZoVTUv0n+W5xAhq4sfZgeRppyG1M1+c"
        "9WBITXl9ggx/tfqHCSRrcxjiGex3jN/ktfeKLyfjZMjP2+I1Tr+MVa5vmHx/8JL/"
        "PSBw=="
    )
    path.write_bytes(zlib.decompress(base64.b64decode(compressed_fixture)))

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(
            path, direct_manifest
        ).only_valid_selection(),
        manifest=direct_manifest,
    )

    assert [row.category for row in resolution.rows] == ["征信", "征信"]


def test_xls_uses_a_lower_early_file_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.xls"
    path.write_bytes(b"not-an-xls")
    monkeypatch.setattr(feature_metadata, "MAX_XLS_METADATA_BYTES", 8, raising=False)

    with pytest.raises(ValueError, match="XLS byte limit"):
        inspect_feature_metadata(path, _manifest("x1"), max_bytes=100)


def test_parquet_metadata_is_bounded_and_keeps_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "metadata.parquet"
    frame = pd.DataFrame(
        {"feature": ["001"], "category": ["内部"], "importance": [0.0]}
    )
    frame.to_parquet(path, index=False)
    manifest = _manifest("001")

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, manifest).only_valid_selection(),
        manifest=manifest,
    )

    assert resolution.rows[0].feature == "001"
    assert resolution.rows[0].importance == 0.0
    with pytest.raises(ValueError, match="row limit"):
        inspect_feature_metadata(path, manifest, max_rows=0)


@pytest.mark.parametrize(
    ("bad_role", "bad_array"),
    [
        ("feature", pa.array([["x1"]], type=pa.list_(pa.string()))),
        ("feature", pa.array([b"x1"], type=pa.binary())),
        ("category", pa.array([{"name": "内部"}])),
        ("importance", pa.array([[1.0]], type=pa.list_(pa.float64()))),
    ],
)
def test_parquet_rejects_nested_or_binary_role_columns(
    tmp_path: Path, bad_role: str, bad_array: pa.Array
) -> None:
    columns = {
        "feature": pa.array(["x1"]),
        "category": pa.array(["内部"]),
        "importance": pa.array([1.0]),
    }
    columns[bad_role] = bad_array
    path = tmp_path / f"bad-{bad_role}.parquet"
    pq.write_table(pa.table(columns), path)

    with pytest.raises(ValueError, match=f"Parquet {bad_role} column type"):
        inspect_feature_metadata(path, _manifest("x1"))


def test_parquet_accepts_bounded_string_importance(tmp_path: Path) -> None:
    path = tmp_path / "string-importance.parquet"
    pq.write_table(
        pa.table(
            {
                "feature": pa.array(["001"]),
                "category": pa.array(["内部"]),
                "importance": pa.array(["0.25"]),
            }
        ),
        path,
    )
    manifest = _manifest("001")

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, manifest).only_valid_selection(),
        manifest=manifest,
    )

    assert resolution.rows[0].importance == 0.25


def test_feather_metadata_is_rejected_with_conversion_guidance(tmp_path: Path) -> None:
    path = tmp_path / "metadata.feather"
    pd.DataFrame(
        {"feature": ["001"], "category": ["内部"], "importance": [0.0]}
    ).to_feather(path)

    with pytest.raises(ValueError, match="CSV or Parquet"):
        inspect_feature_metadata(path, _manifest("001"))


def test_derived_features_expand_to_raw_stress_fields_in_first_seen_order(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        "derived_b",
        "direct",
        "derived_a",
        stress_units=(
            StressUnit("derived_b", ("raw_2", "raw_1"), ("derived_b <- raw",)),
            StressUnit("direct", ("direct",), ()),
            StressUnit("derived_a", ("raw_1", "raw_3"), ("derived_a <- raw",)),
        ),
    )
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\n"
        "derived_b,征信,0.4\n"
        "direct,内部,0.3\n"
        "derived_a,征信,0.3\n",
        encoding="utf-8",
    )

    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(path, manifest).only_valid_selection(),
        manifest=manifest,
    )

    assert list(resolution.per_category_raw_fields) == ["征信", "内部"]
    assert resolution.per_category_raw_fields == {
        "征信": ("raw_2", "raw_1", "raw_3"),
        "内部": ("direct",),
    }


def test_same_raw_stress_field_cannot_belong_to_two_categories(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        "derived_a",
        "derived_b",
        stress_units=(
            StressUnit("derived_a", ("raw",), ("derived_a <- raw",)),
            StressUnit("derived_b", ("raw",), ("derived_b <- raw",)),
        ),
    )
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nderived_a,征信,0.5\nderived_b,内部,0.5\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, manifest)

    assert inspection.selections == ()
    assert any("raw" in item and "category conflict" in item for item in inspection.blocking_errors)


def test_normalize_bounds_feature_category_and_raw_stress_field_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feature_metadata, "MAX_METADATA_CELL_CHARS", 12, raising=False)
    long_raw = "raw_" + ("x" * 100)
    manifest = _manifest(
        "x1", stress_units=(StressUnit("x1", (long_raw,), ()),)
    )
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx1,内部,1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cell length") as captured:
        normalize_feature_metadata(
            path,
            selection=FeatureMetadataSelection(
                None, "feature", "category", "importance"
            ),
            manifest=manifest,
        )

    assert len(str(captured.value)) <= MAX_DIAGNOSTIC_CHARS


@pytest.mark.parametrize(
    ("stress_units", "unsupported", "match"),
    [
        ((StressUnit("x1", ("x1",), ()),), (), "x2"),
        (
            (StressUnit("x1", ("x1",), ()), StressUnit("x2", (), ())),
            (),
            "empty",
        ),
        (
            (StressUnit("x1", ("x1",), ()),),
            ("x2: unsupported derivation Apply for x2",),
            "unsupported",
        ),
    ],
)
def test_missing_empty_or_unsupported_stress_unit_blocks(
    tmp_path: Path,
    stress_units: tuple[StressUnit, ...],
    unsupported: tuple[str, ...],
    match: str,
) -> None:
    manifest = _manifest(
        "x1",
        "x2",
        stress_units=stress_units,
        unsupported_derivations=unsupported,
    )
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nx1,内部,1\nx2,征信,0\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(path, manifest)

    assert inspection.selections == ()
    assert any(match in item for item in inspection.blocking_errors)


def test_feature_category_adapter_uses_confirmed_raw_stress_mapping(
    tmp_path: Path, derived_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\nage_bucket,征信,0.6\nincome,内部,0.4\n",
        encoding="utf-8",
    )
    resolution = normalize_feature_metadata(
        path,
        selection=inspect_feature_metadata(
            path, derived_manifest
        ).only_valid_selection(),
        manifest=derived_manifest,
    )

    adapted = feature_category_resolution_from_metadata(resolution)

    assert adapted.per_category == {"征信": ["age"], "内部": ["income"]}
    assert adapted.unclassified_features == []
    assert adapted.conflicts == []
    assert adapted.source_counts == {
        "notebook": 0,
        "dictionary": 2,
        "unresolved": 0,
    }


def test_alias_diagnostics_are_count_and_length_bounded(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,特征名,特征名称,指标英文,category,类别,分类,importance,gain,权重\n"
        "missing,missing,missing,missing,内部,内部,内部,1,1,1\n",
        encoding="utf-8",
    )

    inspection = inspect_feature_metadata(
        path,
        direct_manifest,
        max_diagnostics=3,
        max_diagnostic_chars=40,
    )

    assert inspection.selections == ()
    assert 1 <= len(inspection.blocking_errors) <= 3
    assert all(len(item) <= 40 for item in inspection.blocking_errors)


def test_rejection_strings_are_not_generated_after_diagnostic_budget(
    tmp_path: Path, direct_manifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,特征名,指标英文,category,类别,importance,gain\n"
        "missing,missing,missing,内部,内部,1,1\n",
        encoding="utf-8",
    )
    original = feature_metadata._append_rejection
    calls = 0

    def counted_append(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(feature_metadata, "_append_rejection", counted_append)

    inspection = inspect_feature_metadata(path, direct_manifest, max_diagnostics=2)

    assert len(inspection.blocking_errors) == 2
    assert calls == 2


def test_alias_row_work_budget_blocks_large_cartesian_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,特征名,category,importance\n"
        "x1,x1,内部,1\nx2,x2,内部,1\nx3,x3,内部,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(feature_metadata, "MAX_ALIAS_ROW_EVALUATIONS", 5)

    inspection = inspect_feature_metadata(path, _manifest("x1", "x2", "x3"))

    assert inspection.selections == ()
    assert any("row evaluation limit" in item for item in inspection.blocking_errors)


def test_manual_selection_still_requires_every_pmml_feature_after_exact_trim(
    tmp_path: Path, direct_manifest
) -> None:
    path = tmp_path / "metadata.csv"
    path.write_text(
        "feature,category,importance\n x1 ,内部,1\nextra,额外,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing.*x2"):
        normalize_feature_metadata(
            path,
            selection=FeatureMetadataSelection(None, "feature", "category", "importance"),
            manifest=direct_manifest,
        )


def test_public_entry_bounds_duplicate_header_error_text(
    tmp_path: Path, direct_manifest
) -> None:
    names = [f"column_{index}_" + ("x" * 980) for index in range(20)]
    path = tmp_path / "metadata.csv"
    path.write_text(
        ",".join([*names, *names]) + "\n" + ",".join(["1"] * 40) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as captured:
        inspect_feature_metadata(path, direct_manifest)

    assert "duplicate column names" in str(captured.value)
    assert len(str(captured.value)) <= MAX_DIAGNOSTIC_CHARS
