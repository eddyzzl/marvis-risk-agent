from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — headless backend, required before pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

from riskmodel_checker.output.styles import (
    BRAND_HEADER_FILL,
    BRAND_HEADER_FONT_COLOR,
    CJK_FONT_CANDIDATES,
    FONT_NAME,
    FONT_SIZE_PT,
    ks_delta_cell_color,
)
from riskmodel_checker.validation.results import RocKsCurve, ValidationResults


_FONT = None
RenderedImageValue = Union[Path, list[Path]]
FEATURE_IMPORTANCE_FEATURE_COLUMN_MAX_WIDTH = 3.8
MIN_TABLE_COLUMN_WIDTH = 0.95
TABLE_COLUMN_PADDING_WIDTH = 0.55
TABLE_WIDTH_PER_DISPLAY_UNIT = 0.15


def get_matplotlib_font() -> FontProperties:
    global _FONT
    if _FONT is None:
        _FONT = _resolve_matplotlib_font()
    return _FONT


def render_all_images(results: ValidationResults, output_dir: Path) -> dict[str, RenderedImageValue]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images: dict[str, RenderedImageValue] = {}

    images["IMAGE:sample_overall_distribution"] = _render_table(
        output_dir / "sample_overall_distribution.png",
        header=["数据集", "时间范围", "样本量", "样本占比", "坏样本量", "逾期率"],
        rows=_sample_overall_rows(results),
        data_bar_columns={2: "5A8AC6", 5: "F8696B"},
    )
    images["IMAGE:sample_month_distribution"] = _render_table(
        output_dir / "sample_month_distribution.png",
        header=["月份", "样本量", "样本占比", "坏样本量", "逾期率"],
        rows=_sample_month_rows(results),
        data_bar_columns={1: "5A8AC6", 4: "F8696B"},
    )
    images["IMAGE:top20_feature_ranking"] = _render_table(
        output_dir / "top20_feature_ranking.png",
        header=["排名", "特征", "类别", "重要性"],
        rows=[(row.rank, row.feature, row.category, f"{row.importance:.4f}")
              for row in results.basic_info.feature_importance[:20]],
        max_column_widths={1: FEATURE_IMPORTANCE_FEATURE_COLUMN_MAX_WIDTH},
    )
    images["IMAGE:ranking_table"] = images["IMAGE:top20_feature_ranking"]
    for split in ("train", "test", "oot"):
        images[f"IMAGE:roc_ks_graph_{split}"] = render_roc_ks_graph(
            results.effectiveness.roc_ks_curves.get(split),
            output_dir / f"roc_ks_graph_{split}.png",
            title_prefix=split,
        )
        images[f"IMAGE:ranking_table_{split}"] = _render_table(
            output_dir / f"ranking_table_{split}.png",
            header=[
                f"{split}(独立分箱)", "样本总数", "累计占比", "逾期数量", "逾期率",
                "累计逾期率", "单组lift", "累计lift", "ks",
            ],
            rows=_reference_bin_rows(results.effectiveness.bin_tables.get(split, [])),
            color_scale_columns={4},
            data_bar_columns={7: "63BE7B"},
        )
    images["IMAGE:model_parameters"] = _render_table(
        output_dir / "model_parameters.png",
        header=["参数", "取值"],
        rows=[(key, value) for key, value in results.basic_info.hyperparameters.items()]
        or [("REVIEW REQUIRED", "no model parameters found")],
    )
    images["IMAGE:overall_model_effect"] = _render_table(
        output_dir / "overall_model_effect.png",
        header=[
            "数据集", "时间范围", "样本量", "逾期率", "坏样本量",
            "KS(%)", "AUC(%)", "5%头部lift", "5%尾部lift", "PSI",
        ],
        rows=_model_effect_rows(results),
        data_bar_columns={5: "63BE7B"},
    )
    images["IMAGE:dataset_model_effect"] = images["IMAGE:overall_model_effect"]
    images["IMAGE:loan_month_effect"] = _render_table(
        output_dir / "loan_month_effect.png",
        header=[
            "月份", "样本量", "逾期率", "坏样本量", "KS(%)", "AUC(%)",
            "5%头部lift", "5%尾部lift", "PSI(首月基准)", "PSI(尾月基准)",
            "PSI(较上一有样本月)", "PSI参考月",
        ],
        rows=_monthly_effect_rows(results),
        data_bar_columns={4: "63BE7B"},
    )
    images["IMAGE:psi_stability_table"] = _render_table(
        output_dir / "psi_stability_table.png",
        header=["分箱", "train+test样本数", "train+test占比", "oot样本数", "oot占比", "PSI"],
        rows=[
            (
                row.bin_label,
                row.expected_count,
                f"{row.expected_pct:.2%}",
                row.actual_count,
                f"{row.actual_pct:.2%}",
                f"{row.psi:.4f}",
            )
            for row in results.effectiveness.psi_stability_table
        ],
        data_bar_columns={1: "5A8AC6", 3: "5A8AC6"},
        color_scale_columns={5},
    )
    images["IMAGE:ks_discrimination_table"] = _render_table(
        output_dir / "ks_discrimination_table.png",
        header=[
            "分箱", "样本总数", "累计占比", "逾期数量", "逾期率",
            "累计逾期率", "单组lift", "累计lift", "ks",
        ],
        rows=_reference_bin_rows(results.effectiveness.bin_tables.get("oot", [])),
        color_scale_columns={4},
        data_bar_columns={7: "63BE7B"},
    )
    pressure_ks_rows = [
        (item.category, f"{results.stress_test.baseline.ks:.4f}",
         _opt_decimal(item.ks_after), _opt_decimal(item.ks_delta))
        for item in results.stress_test.per_category
    ]
    images["IMAGE:pressure_ks_table"] = _render_table(
        output_dir / "pressure_ks_table.png",
        header=["类别", "KS_baseline", "KS_after", "KS_delta"],
        rows=pressure_ks_rows,
        cell_fill_colors={
            (row_index, 3): color
            for row_index, item in enumerate(results.stress_test.per_category)
            if item.ks_delta is not None
            if (color := ks_delta_cell_color(item.ks_delta))
        },
    )
    images["IMAGE:pressure_psi_table"] = _render_table(
        output_dir / "pressure_psi_table.png",
        header=["类别", "PSI vs baseline"],
        rows=[(item.category, _opt_decimal(item.psi_vs_baseline))
              for item in results.stress_test.per_category],
    )
    pressure_shift_paths: list[Path] = []
    for index, item in enumerate(results.stress_test.per_category, start=1):
        key = f"IMAGE:pressure_score_shift_{index}"
        image_path = _render_table(
            output_dir / f"pressure_score_shift_{index}.png",
            header=[
                item.category, "样本总数", "累计占比", "逾期数量", "逾期率",
                "累计逾期率", "单组lift", "累计lift", "ks",
            ],
            rows=_reference_bin_rows(item.bin_table) or [("(无数据)", "", "", "", "", "", "", "", "")],
            color_scale_columns={4},
            data_bar_columns={7: "63BE7B"},
        )
        images[key] = image_path
        pressure_shift_paths.append(image_path)
    if not pressure_shift_paths:
        fallback_path = _render_table(
            output_dir / "pressure_score_shift.png",
            header=["状态", "说明"],
            rows=[("REVIEW REQUIRED", "no pressure category data")],
        )
        pressure_shift_paths.append(fallback_path)
    images["IMAGE:pressure_score_shift"] = pressure_shift_paths
    for index in range(len(results.stress_test.per_category) + 1, 8):
        key = f"IMAGE:pressure_score_shift_{index}"
        images[key] = _render_table(
            output_dir / f"pressure_score_shift_{index}.png",
            header=["状态", "说明"],
            rows=[("REVIEW REQUIRED", f"no pressure category data for slot {index}")],
        )
    return images


def _fpr_at_ks(curve: RocKsCurve) -> float:
    """FPR coordinate of the KS-maximizing threshold.

    The KS marker is drawn on the ROC x-axis (False Positive Rate), so it must be
    anchored at the FPR where |TPR-FPR| peaks — i.e. fpr[argmax(|ks_curve|)].
    ``population_at_ks`` lives on a different (cumulative-population) axis and would
    misplace the line on imbalanced credit data; it is only used for the text label.
    """
    if not curve.ks_curve or not curve.fpr:
        return 0.0
    ks_index = max(range(len(curve.ks_curve)), key=lambda i: abs(curve.ks_curve[i]))
    if ks_index >= len(curve.fpr):
        ks_index = len(curve.fpr) - 1
    return curve.fpr[ks_index]


def render_roc_ks_graph(
    curve: RocKsCurve | None,
    output_path: Path,
    *,
    title_prefix: str = "",
) -> Path:
    if curve is None or not curve.fpr or not curve.tpr:
        return _render_table(
            output_path,
            header=["状态", "说明"],
            rows=[("REVIEW REQUIRED", f"no ROC/KS data for {title_prefix or 'split'}")],
        )

    font = get_matplotlib_font()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(curve.fpr, curve.tpr, "r-", label="True Positive Rate", linewidth=2)
    ax.plot(curve.fpr, curve.fpr, "b-", label="Random Baseline", linewidth=2)
    ax.plot(curve.fpr, curve.ks_curve, "g-", label="KS Curve", linewidth=2)
    ax.axvline(x=_fpr_at_ks(curve), color="g", linestyle="--", alpha=0.5)
    ax.text(
        0.02,
        0.95,
        f"KS={curve.ks:.4f} at pop={curve.population_at_ks:.2f}",
        transform=ax.transAxes,
        fontsize=28,
        fontproperties=font,
    )
    ax.set_xlabel("False Positive Rate", fontsize=32, fontproperties=font)
    ax.set_ylabel("True Positive Rate", fontsize=32, fontproperties=font)
    title = f"{title_prefix}ROC曲线和KS曲线" if title_prefix else "ROC曲线和KS曲线"
    ax.set_title(title, fontsize=36, fontproperties=font)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right", fontsize=28)
    ax.tick_params(axis="both", which="major", labelsize=28)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    try:
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def _sample_overall_rows(results: ValidationResults) -> list[tuple]:
    total_count = sum(row.sample_count for row in results.basic_info.split_summary)
    return [
        (
            row.split,
            _period_text(row.period_start, row.period_end, default="-"),
            row.sample_count,
            f"{_ratio(row.sample_count, total_count):.2%}",
            row.bad_count,
            f"{row.bad_rate:.2%}",
        )
        for row in results.basic_info.split_summary
    ]


def _sample_month_rows(results: ValidationResults) -> list[tuple]:
    total_count = sum(row.sample_count for row in results.basic_info.monthly_distribution)
    return [
        (
            row.month,
            row.sample_count,
            f"{_ratio(row.sample_count, total_count):.2%}",
            row.bad_count,
            f"{row.bad_rate:.2%}",
        )
        for row in results.basic_info.monthly_distribution
    ]


def _model_effect_rows(results: ValidationResults) -> list[tuple]:
    split_summary = {row.split: row for row in results.basic_info.split_summary}
    rows: list[tuple] = []
    for row in results.effectiveness.overall:
        split_row = split_summary.get(row.split)
        rows.append((
            row.split,
            _period_text(
                split_row.period_start if split_row else "",
                split_row.period_end if split_row else "",
                default="-",
            ),
            row.sample_count,
            f"{row.bad_rate:.2%}",
            int(row.bad_count),
            f"{row.ks * 100:.1f}",
            f"{row.auc * 100:.1f}",
            _opt_float(row.head_lift_5pct, digits=2),
            _opt_float(row.tail_lift_5pct, digits=2),
            "BASE" if row.split == "train" else _opt_float(row.psi_vs_train, digits=3),
        ))
    return rows


def _monthly_effect_rows(results: ValidationResults) -> list[tuple]:
    by_month: dict[str, dict[str, object]] = {}
    for row in results.effectiveness.monthly_ks:
        by_month.setdefault(row.month, {}).update({
            "sample_count": row.sample_count,
            "bad_rate": row.bad_rate,
            "bad_count": row.bad_count,
            "ks": row.ks,
            "auc": row.auc,
            "head_lift_5pct": row.head_lift_5pct,
            "tail_lift_5pct": row.tail_lift_5pct,
        })
    for row in results.effectiveness.monthly_psi:
        by_month.setdefault(row.month, {}).update({
            "psi_first_month": row.psi_first_month,
            "psi_last_month": row.psi_last_month,
            "psi_mom": row.psi_mom,
            "psi_mom_reference_month": row.psi_mom_reference_month,
            "psi_mom_has_calendar_gap": row.psi_mom_has_calendar_gap,
        })
    months = sorted(by_month)
    first_month = months[0] if months else ""
    last_month = months[-1] if months else ""
    rows: list[tuple] = []
    for month in months:
        data = by_month[month]
        rows.append((
            month,
            data.get("sample_count", 0),
            f"{float(data.get('bad_rate', 0.0)):.2%}",
            data.get("bad_count", 0),
            f"{float(data.get('ks', 0.0)) * 100:.1f}",
            f"{float(data.get('auc', 0.0)) * 100:.1f}",
            _opt_float(data.get("head_lift_5pct"), digits=2),
            _opt_float(data.get("tail_lift_5pct"), digits=2),
            "BASE" if month == first_month else _opt_float(data.get("psi_first_month"), digits=3),
            "BASE" if month == last_month else _opt_float(data.get("psi_last_month"), digits=3),
            "-" if month == first_month else _opt_float(data.get("psi_mom"), digits=3),
            "-" if month == first_month else _psi_reference_month_text(
                str(data.get("psi_mom_reference_month") or ""),
                has_calendar_gap=bool(data.get("psi_mom_has_calendar_gap")),
            ),
        ))
    return rows


def _reference_bin_rows(bins) -> list[tuple]:
    total = sum(row.sample_count for row in bins)
    total_bad = sum(row.bad_count for row in bins)
    overall_bad_rate = _ratio(total_bad, total)
    cumulative_count = 0
    cumulative_bad = 0
    rows: list[tuple] = []
    for row in bins:
        cumulative_count += row.sample_count
        cumulative_bad += row.bad_count
        cumulative_bad_rate = _ratio(cumulative_bad, cumulative_count)
        rows.append((
            _score_interval(row.score_lower, row.score_upper),
            row.sample_count,
            f"{_ratio(cumulative_count, total):.2%}",
            row.bad_count,
            f"{row.bad_rate:.2%}",
            f"{cumulative_bad_rate:.2%}",
            f"{row.lift:.2f}",
            f"{_ratio(cumulative_bad_rate, overall_bad_rate):.2f}",
            f"{row.ks:.4f}",
        ))
    return rows


def _opt_decimal(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else ""


def _opt_float(value, *, digits: int) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _psi_reference_month_text(month: str, *, has_calendar_gap: bool) -> str:
    if not month:
        return ""
    return f"{month}(跨月)" if has_calendar_gap else month


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _period_text(start: str, end: str, *, default: str) -> str:
    if not start and not end:
        return default
    if not start:
        return str(end)
    if not end:
        return str(start)
    return str(start) if start == end else f"{start}-{end}"


def _score_interval(lower: float, upper: float) -> str:
    return f"[{_compact_number(lower)},{_compact_number(upper)}]"


def _compact_number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _render_table(
    output_path: Path,
    *,
    header: list[str],
    rows: list[tuple],
    data_bar_columns: dict[int, str] | None = None,
    color_scale_columns: set[int] | None = None,
    cell_fill_colors: dict[tuple[int, int], str] | None = None,
    max_column_widths: dict[int, float] | None = None,
) -> Path:
    font = get_matplotlib_font()
    data_bar_columns = data_bar_columns or {}
    color_scale_columns = color_scale_columns or set()
    cell_fill_colors = cell_fill_colors or {}
    max_column_widths = max_column_widths or {}
    column_count = len(header)
    body_rows = [tuple(row) for row in rows] or [tuple("" for _ in range(column_count))]
    row_count = len(body_rows) + 1
    figure_height = max(1.0, 0.35 * row_count)
    column_widths = _table_column_widths(
        header,
        body_rows,
        max_column_widths=max_column_widths,
    )
    column_starts = _column_starts(column_widths)
    table_width = sum(column_widths)
    figure_width = max(4.0, table_width)
    data_bar_fractions = _data_bar_fractions(body_rows, data_bar_columns)
    color_scale_fills = _color_scale_fills(body_rows, color_scale_columns)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=180)
    ax.axis("off")
    ax.set_xlim(0, table_width)
    ax.set_ylim(0, row_count)

    for row_index in range(row_count):
        y = row_count - row_index - 1
        is_header = row_index == 0
        values = header if is_header else body_rows[row_index - 1]
        for column_index in range(column_count):
            x = column_starts[column_index]
            width = column_widths[column_index]
            value = values[column_index] if column_index < len(values) else ""
            if is_header:
                fill_color = f"#{BRAND_HEADER_FILL}"
            else:
                body_row_index = row_index - 1
                fill_color = (
                    f"#{cell_fill_colors[(body_row_index, column_index)]}"
                    if (body_row_index, column_index) in cell_fill_colors
                    else color_scale_fills.get((body_row_index, column_index), "#FFFFFF")
                )
            cell_background = Rectangle(
                (x, y),
                width,
                1,
                facecolor=fill_color,
                edgecolor="none",
            )
            ax.add_patch(cell_background)
            if not is_header and column_index in data_bar_columns:
                fraction = data_bar_fractions.get((row_index - 1, column_index), 0.0)
                if fraction > 0:
                    horizontal_padding = min(0.08, width * 0.06)
                    ax.add_patch(
                        Rectangle(
                            (x + horizontal_padding, y + 0.2),
                            max(0.0, (width - 2 * horizontal_padding) * fraction),
                            0.6,
                            facecolor=f"#{data_bar_columns[column_index]}",
                            edgecolor="none",
                        )
                    )
            ax.add_patch(
                Rectangle(
                    (x, y),
                    width,
                    1,
                    facecolor="none",
                    edgecolor="#D9D9D9",
                    linewidth=0.6,
                )
            )
            text = ax.text(
                x + width / 2,
                y + 0.5,
                str(value),
                ha="center",
                va="center",
                fontsize=FONT_SIZE_PT,
                color=f"#{BRAND_HEADER_FONT_COLOR}" if is_header else "#000000",
                fontproperties=font,
                fontweight="bold" if is_header else "normal",
            )
            text.set_clip_path(cell_background)
    try:
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def _table_column_widths(
    header: list[str],
    rows: list[tuple],
    *,
    max_column_widths: dict[int, float],
) -> list[float]:
    column_count = len(header)
    widths: list[float] = []
    for column_index in range(column_count):
        values = [header[column_index]]
        values.extend(
            row[column_index]
            for row in rows
            if column_index < len(row)
        )
        display_units = max(_display_width(value) for value in values)
        width = max(
            MIN_TABLE_COLUMN_WIDTH,
            TABLE_COLUMN_PADDING_WIDTH + display_units * TABLE_WIDTH_PER_DISPLAY_UNIT,
        )
        if column_index in max_column_widths:
            width = min(width, max_column_widths[column_index])
        widths.append(width)
    return widths


def _column_starts(widths: list[float]) -> list[float]:
    starts: list[float] = []
    cursor = 0.0
    for width in widths:
        starts.append(cursor)
        cursor += width
    return starts


def _display_width(value) -> float:
    total = 0.0
    for char in str(value):
        codepoint = ord(char)
        if codepoint < 128:
            total += 0.35 if char in " -_.,:%/[]" else 0.6
        else:
            total += 1.0
    return total


def _data_bar_fractions(
    rows: list[tuple],
    data_bar_columns: dict[int, str],
) -> dict[tuple[int, int], float]:
    fractions: dict[tuple[int, int], float] = {}
    for column_index in data_bar_columns:
        indexed_values = [
            (row_index, value)
            for row_index, row in enumerate(rows)
            if column_index < len(row)
            if (value := _numeric_value(row[column_index])) is not None
        ]
        if not indexed_values:
            continue
        values = [value for _, value in indexed_values]
        minimum = min(values)
        maximum = max(values)
        for row_index, value in indexed_values:
            if maximum == minimum:
                fraction = 1.0 if value > 0 else 0.0
            elif minimum >= 0:
                fraction = value / maximum if maximum > 0 else 0.0
            else:
                fraction = (value - minimum) / (maximum - minimum)
            fractions[(row_index, column_index)] = max(0.0, min(1.0, fraction))
    return fractions


def _color_scale_fills(
    rows: list[tuple],
    color_scale_columns: set[int],
) -> dict[tuple[int, int], str]:
    fills: dict[tuple[int, int], str] = {}
    for column_index in color_scale_columns:
        indexed_values = [
            (row_index, value)
            for row_index, row in enumerate(rows)
            if column_index < len(row)
            if (value := _numeric_value(row[column_index])) is not None
        ]
        if not indexed_values:
            continue
        values = [value for _, value in indexed_values]
        minimum = min(values)
        midpoint = _median(values)
        maximum = max(values)
        for row_index, value in indexed_values:
            fills[(row_index, column_index)] = f"#{_color_scale_color(value, minimum, midpoint, maximum)}"
    return fills


def _numeric_value(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100
        return float(text)
    except ValueError:
        return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _color_scale_color(value: float, minimum: float, midpoint: float, maximum: float) -> str:
    green = (0x63, 0xBE, 0x7B)
    yellow = (0xFF, 0xEB, 0x84)
    red = (0xF8, 0x69, 0x6B)
    if maximum == minimum:
        return _rgb_to_hex(yellow)
    if value <= midpoint:
        fraction = 1.0 if midpoint == minimum else (value - minimum) / (midpoint - minimum)
        return _rgb_to_hex(_interpolate_rgb(green, yellow, fraction))
    fraction = 1.0 if maximum == midpoint else (value - midpoint) / (maximum - midpoint)
    return _rgb_to_hex(_interpolate_rgb(yellow, red, fraction))


def _interpolate_rgb(start: tuple[int, int, int], end: tuple[int, int, int], fraction: float) -> tuple[int, int, int]:
    fraction = max(0.0, min(1.0, fraction))
    return tuple(
        round(start[channel] + (end[channel] - start[channel]) * fraction)
        for channel in range(3)
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "".join(f"{channel:02X}" for channel in rgb)


def _resolve_matplotlib_font() -> FontProperties:
    for family in CJK_FONT_CANDIDATES:
        try:
            font_path = font_manager.findfont(
                FontProperties(family=family),
                fallback_to_default=False,
            )
        except ValueError:
            continue
        if font_path and _font_path_matches_family(font_path, family):
            return FontProperties(fname=font_path)
    logging.warning(
        "No CJK font found among %s; Chinese text in rendered PNGs may show as "
        "blank boxes. Install one (e.g. `apt install fonts-noto-cjk`) on the server.",
        ", ".join(CJK_FONT_CANDIDATES),
    )
    return FontProperties(family=FONT_NAME)


def _font_path_matches_family(font_path: str, family: str) -> bool:
    normalized_path = Path(font_path).stem.lower().replace(" ", "").replace("-", "")
    normalized_family = family.lower().replace(" ", "").replace("-", "")
    family_tokens = [token for token in re.split(r"[\s,-]+", family.lower()) if token]
    return normalized_family in normalized_path or any(token in normalized_path for token in family_tokens)
