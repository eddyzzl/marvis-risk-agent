from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from marvis.output import image_render
from marvis.output.image_render import get_matplotlib_font, render_all_images
from marvis.output.styles import CJK_FONT_CANDIDATES
from marvis.validation.results import FeatureImportanceRow, RocKsCurve
from tests.output.test_excel import _make_results


def test_renders_required_image_keys(tmp_path: Path):
    image_paths = render_all_images(_make_results(), tmp_path)
    required = {
        "IMAGE:sample_overall_distribution",
        "IMAGE:sample_month_distribution",
        "IMAGE:top20_feature_ranking",
        "IMAGE:model_parameters",
        "IMAGE:ranking_table",
        "IMAGE:ranking_table_train",
        "IMAGE:ranking_table_test",
        "IMAGE:ranking_table_oot",
        "IMAGE:roc_ks_graph_train",
        "IMAGE:roc_ks_graph_test",
        "IMAGE:roc_ks_graph_oot",
        "IMAGE:overall_model_effect",
        "IMAGE:dataset_model_effect",
        "IMAGE:loan_month_effect",
        "IMAGE:psi_stability_table",
        "IMAGE:ks_discrimination_table",
        "IMAGE:pressure_ks_table",
        "IMAGE:pressure_psi_table",
        "IMAGE:pressure_score_shift",
        "IMAGE:pressure_score_shift_1",
        "IMAGE:pressure_score_shift_7",
    }
    assert required.issubset(image_paths.keys())
    assert isinstance(image_paths["IMAGE:pressure_score_shift"], list)
    assert len(image_paths["IMAGE:pressure_score_shift"]) == 1


def test_pressure_score_shift_placeholder_collects_all_actual_category_images(
    tmp_path: Path,
):
    results = _make_results()
    base_category = results.stress_test.per_category[0]
    categories = ["征信", "身份", "交易", "行为", "设备", "地域", "联系人", "其他"]
    results = replace(
        results,
        stress_test=replace(
            results.stress_test,
            per_category=[
                replace(base_category, category=category)
                for category in categories
            ],
        ),
    )

    image_paths = render_all_images(results, tmp_path)

    pressure_images = image_paths["IMAGE:pressure_score_shift"]
    assert isinstance(pressure_images, list)
    assert len(pressure_images) == len(categories)
    assert [path.name for path in pressure_images] == [
        f"pressure_score_shift_{index}.png"
        for index in range(1, len(categories) + 1)
    ]
    assert "IMAGE:pressure_score_shift_8" in image_paths


def test_pressure_score_shift_placeholder_gets_fallback_image_when_no_categories(
    tmp_path: Path,
):
    results = _make_results()
    results = replace(
        results,
        stress_test=replace(results.stress_test, per_category=[]),
    )

    image_paths = render_all_images(results, tmp_path)

    pressure_images = image_paths["IMAGE:pressure_score_shift"]
    assert isinstance(pressure_images, list)
    assert [path.name for path in pressure_images] == ["pressure_score_shift.png"]
    assert pressure_images[0].exists()


def test_rendered_files_are_non_empty_pngs(tmp_path: Path):
    image_paths = render_all_images(_make_results(), tmp_path)
    for key, value in image_paths.items():
        paths = value if isinstance(value, list) else [value]
        for path in paths:
            assert path.exists(), key
            assert path.suffix == ".png", key
            assert path.stat().st_size > 200, key
            with path.open("rb") as fh:
                assert fh.read(8) == b"\x89PNG\r\n\x1a\n", key


def test_matplotlib_font_uses_candidate_fallback_list():
    font = get_matplotlib_font()

    assert "Microsoft YaHei" in CJK_FONT_CANDIDATES
    assert len(CJK_FONT_CANDIDATES) > 1
    assert font.get_file() or font.get_family()[0] in CJK_FONT_CANDIDATES


def test_word_image_tables_use_reference_model_analysis_headers(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_render_table(output_path: Path, *, header: list[str], rows: list[tuple], **kwargs) -> Path:
        captured[output_path.name] = {"header": header, "rows": rows, "kwargs": kwargs}
        return output_path

    monkeypatch.setattr(image_render, "_render_table", fake_render_table)

    image_render.render_all_images(_make_results(), tmp_path)

    assert captured["sample_overall_distribution.png"]["header"] == [
        "数据集", "时间范围", "样本量", "样本占比", "坏样本量", "逾期率",
    ]
    assert captured["overall_model_effect.png"]["header"] == [
        "数据集", "时间范围", "样本量", "逾期率", "坏样本量",
        "KS(%)", "AUC(%)", "5%头部lift", "5%尾部lift", "PSI",
    ]
    assert captured["loan_month_effect.png"]["header"] == [
        "月份", "样本量", "逾期率", "坏样本量", "KS(%)", "AUC(%)",
        "5%头部lift", "5%尾部lift", "PSI(首月基准)", "PSI(尾月基准)",
        "PSI(较上一有样本月)", "PSI参考月",
    ]
    assert captured["ks_discrimination_table.png"]["header"] == [
        "分箱", "样本总数", "累计占比", "逾期数量", "逾期率",
        "累计逾期率", "单组lift", "累计lift", "ks",
    ]
    assert captured["ranking_table_train.png"]["header"] == [
        "train(独立分箱)", "样本总数", "累计占比", "逾期数量", "逾期率",
        "累计逾期率", "单组lift", "累计lift", "ks",
    ]
    assert captured["ranking_table_test.png"]["header"][0] == "test(独立分箱)"
    assert captured["ranking_table_oot.png"]["header"][0] == "oot(独立分箱)"
    assert captured["psi_stability_table.png"]["header"] == [
        "分箱", "train+test样本数", "train+test占比", "oot样本数", "oot占比", "PSI",
    ]
    assert captured["pressure_score_shift_1.png"]["header"][0] == "征信"


def test_word_image_tables_use_excel_conditional_formatting_rules(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_render_table(output_path: Path, *, header: list[str], rows: list[tuple], **kwargs) -> Path:
        captured[output_path.name] = {"header": header, "rows": rows, **kwargs}
        return output_path

    monkeypatch.setattr(image_render, "_render_table", fake_render_table)

    image_render.render_all_images(_make_results(), tmp_path)

    assert captured["sample_overall_distribution.png"]["data_bar_columns"] == {
        2: "5A8AC6",
        5: "F8696B",
    }
    assert captured["sample_month_distribution.png"]["data_bar_columns"] == {
        1: "5A8AC6",
        4: "F8696B",
    }
    assert captured["overall_model_effect.png"]["data_bar_columns"] == {5: "63BE7B"}
    assert captured["loan_month_effect.png"]["data_bar_columns"] == {4: "63BE7B"}
    assert captured["psi_stability_table.png"]["data_bar_columns"] == {
        1: "5A8AC6",
        3: "5A8AC6",
    }
    assert captured["psi_stability_table.png"]["color_scale_columns"] == {5}
    assert captured["ranking_table_train.png"]["color_scale_columns"] == {4}
    assert captured["ranking_table_train.png"]["data_bar_columns"] == {7: "63BE7B"}
    assert captured["pressure_score_shift_1.png"]["color_scale_columns"] == {4}
    assert captured["pressure_score_shift_1.png"]["data_bar_columns"] == {7: "63BE7B"}


def test_render_table_draws_data_bars_and_color_scales(tmp_path: Path):
    output = image_render._render_table(
        tmp_path / "conditional.png",
        header=["name", "count", "rate"],
        rows=[("low", 1, "10.00%"), ("high", 10, "90.00%")],
        data_bar_columns={1: "5A8AC6"},
        color_scale_columns={2},
    )

    image = Image.open(output).convert("RGB")
    colors = image.getcolors(maxcolors=1_000_000) or []
    rgbs = {rgb for _, rgb in colors}

    assert (90, 138, 198) in rgbs
    assert (248, 105, 107) in rgbs


def test_data_bar_fractions_keep_positive_minimum_values_visible():
    rows = [
        ("train", 94810, "5.06%"),
        ("test", 40634, "5.33%"),
        ("oot", 73801, "6.45%"),
    ]

    fractions = image_render._data_bar_fractions(rows, {1: "5A8AC6", 2: "F8696B"})

    assert fractions[(1, 1)] == pytest.approx(40634 / 94810)
    assert fractions[(0, 2)] == pytest.approx(0.0506 / 0.0645)
    assert fractions[(1, 1)] > 0
    assert fractions[(0, 2)] > 0


def test_render_table_expands_columns_for_long_values(tmp_path: Path):
    header = ["数据集", "时间范围", "样本量", "样本占比", "坏样本量", "逾期率"]
    compact = image_render._render_table(
        tmp_path / "compact.png",
        header=header,
        rows=[("train", "202503", 94810, "45.31%", 4802, "5.06%")],
    )
    expanded = image_render._render_table(
        tmp_path / "expanded.png",
        header=header,
        rows=[("train", "20250301-20250630", 94810, "45.31%", 4802, "5.06%")],
    )

    compact_width = Image.open(compact).width
    expanded_width = Image.open(expanded).width

    assert expanded_width > compact_width + 120


def test_feature_importance_table_caps_feature_column_width(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_render_table(output_path: Path, *, header: list[str], rows: list[tuple], **kwargs) -> Path:
        captured[output_path.name] = {"header": header, "rows": rows, **kwargs}
        return output_path

    monkeypatch.setattr(image_render, "_render_table", fake_render_table)
    results = _make_results()
    results = replace(
        results,
        basic_info=replace(
            results.basic_info,
            feature_importance=[
                FeatureImportanceRow(1, "very_long_feature_name_" * 8, 0.8, category="征信"),
                FeatureImportanceRow(2, "x2", 0.2, category="行为"),
            ],
        ),
    )

    image_render.render_all_images(results, tmp_path)

    assert captured["top20_feature_ranking.png"]["max_column_widths"][1] <= 4.0
    assert captured["top20_feature_ranking.png"]["header"] == ["排名", "特征", "类别", "重要性"]
    assert captured["top20_feature_ranking.png"]["rows"][0][2] == "征信"


def test_fpr_at_ks_anchors_marker_on_fpr_axis_not_population():
    # |ks_curve| peaks at index 2, where fpr=0.10. The ROC KS marker must use that
    # FPR, NOT population_at_ks=0.55 — they live on different axes and diverge on
    # imbalanced credit data, which is the misplacement bug this guards against.
    curve = RocKsCurve(
        split="train",
        fpr=[0.0, 0.05, 0.10, 0.40, 1.0],
        tpr=[0.0, 0.30, 0.70, 0.85, 1.0],
        ks_curve=[0.0, 0.25, 0.60, 0.45, 0.0],
        ks=0.60,
        population_at_ks=0.55,
    )

    assert image_render._fpr_at_ks(curve) == 0.10


def test_fpr_at_ks_handles_degenerate_curve():
    curve = RocKsCurve(
        split="train",
        fpr=[0.0, 1.0],
        tpr=[0.0, 1.0],
        ks_curve=[0.0, 0.0],
        ks=0.0,
        population_at_ks=0.0,
    )

    assert image_render._fpr_at_ks(curve) == 0.0


def test_roc_ks_graph_closes_figure_when_savefig_fails(monkeypatch, tmp_path: Path):
    closed = []

    def fail_savefig(self, *args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(image_render.matplotlib.figure.Figure, "savefig", fail_savefig)
    monkeypatch.setattr(image_render.plt, "close", lambda fig: closed.append(fig))

    with pytest.raises(RuntimeError, match="save failed"):
        image_render.render_roc_ks_graph(
            RocKsCurve(
                split="train",
                fpr=[0.0, 1.0],
                tpr=[0.0, 1.0],
                ks_curve=[0.0, 0.0],
                ks=0.0,
                population_at_ks=0.0,
            ),
            tmp_path / "roc.png",
        )

    assert closed


def test_cjk_font_path_must_match_candidate_family_name():
    assert image_render._font_path_matches_family(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "Noto Sans CJK SC",
    )
    assert not image_render._font_path_matches_family(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Microsoft YaHei",
    )


def test_model_effect_rows_preserves_zero_bad_count():
    """A split with a genuine zero bad_count must render 0, not an estimate.

    Mirrors test_excel.test_effectiveness_sheet_preserves_zero_bad_count so the
    PNG/Word report stays consistent with the Excel report (the `or round(...)`
    falsy trap previously estimated a non-zero value when bad_count was 0).
    """
    results = _make_results()
    overall = list(results.effectiveness.overall)
    overall[0] = replace(overall[0], bad_count=0, bad_rate=0.004)
    results = replace(
        results,
        effectiveness=replace(results.effectiveness, overall=overall),
    )

    rows = image_render._model_effect_rows(results)

    # column layout: split, period, sample_count, bad_rate, bad_count, ks, auc, ...
    assert rows[0][4] == 0
