from __future__ import annotations

from pathlib import Path

from marvis.output.image_render import render_all_images
from marvis.report_texts import report_text_values_from_results
from marvis.template_reports import (
    TemplateReportPayload,
    TemplateReportResult,
    render_template_report,
)
from marvis.validation.results import ValidationResults


def write_validation_word(
    results: ValidationResults,
    *,
    template_path: Path,
    output_path: Path,
    image_output_dir: Path,
    report_values: dict[str, str] | None = None,
    manual_values: dict[str, str] | None = None,
) -> TemplateReportResult:
    text_values = report_text_values_from_results(
        results,
        report_values=report_values,
        manual_values=manual_values,
    )
    image_paths = render_all_images(results, image_output_dir)
    # template_reports.render_template_report keys text_values and image_values by
    # the inner identifier INCLUDING the "TEXT:" / "IMAGE:" prefix
    # (it replaces {{KEY}} → text_values[KEY]); both producers already prefix, but
    # we normalise defensively here.
    payload = TemplateReportPayload(
        template_path=template_path,
        output_path=output_path,
        text_values=_with_prefix(text_values, "TEXT:"),
        image_values=_with_prefix(image_paths, "IMAGE:"),
    )
    return render_template_report(payload)


def _with_prefix(values, prefix: str):
    return {(key if key.startswith(prefix) else f"{prefix}{key}"): value
            for key, value in values.items()}
