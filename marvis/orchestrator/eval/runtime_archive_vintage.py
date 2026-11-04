"""Independent revalidation of a retained standard Vintage analysis.

The native ``strategy.vintage_curve`` output is recomputed from the registered
source rows with ``runtime_portfolio_trend_reference``; the producer
(``marvis.validation.vintage`` / ``marvis.packs.strategy.vintage``) never runs.
"""
from pathlib import Path
import math
import tempfile

from pydantic import Field, model_validator

from .runtime_archive_datasets import _dataset, _source_input
from .runtime_archive_domain import HASH, ID, DomainArchive, FrozenDomainBinding, numbers, one_step
from .runtime_archive_reader import _Unsupported
from .runtime_portfolio_trend_reference import vintage_reference, vintage_summary_reference

_TOOL = "strategy.vintage_curve"


class FrozenVintageBinding(FrozenDomainBinding):
    dataset_id: str = Field(pattern=ID)
    dataset_sha256: str = Field(pattern=HASH)

    @model_validator(mode="after")
    def vintage_scope(self):
        if len(self.step_ids) != 1 or len(self.case.materials) != 1 or self.case.materials[0].role != "sample":
            raise ValueError("standard Vintage binds one native step and one sample")
        return self


def revalidate_vintage_archive(archive: Path, *, expected_manifest_sha256: str, frozen_binding: FrozenVintageBinding):
    if not isinstance(frozen_binding, FrozenVintageBinding):
        raise ValueError("an explicit FrozenVintageBinding is required")
    binding = FrozenVintageBinding.model_validate(frozen_binding.model_dump())
    with DomainArchive(archive, expected_manifest_sha256, binding, family="vintage",
                       checks=("vintage_contract", "registered_dataset", "source_material", "independent_vintage")) as ctx:
        ctx.result["unsupported"].update(label_maturity_and_cohort_completeness="not_established_by_vintage_arithmetic")
        ctx.open()
        ctx.current = "vintage_contract"
        plan = ctx.conn.execute("SELECT template_id FROM plans WHERE id=?", (binding.plan_id,)).fetchone()
        if plan is None or plan["template_id"] != "vintage_analysis":
            raise _Unsupported("Vintage revalidation supports the standard vintage_analysis template only")
        step = one_step(ctx.steps, _TOOL)
        inputs = step["input"]
        if inputs.get("dataset_id") != binding.dataset_id or type(inputs.get("drop_nan_labels")) is not bool:
            raise ValueError("Vintage dataset or explicit label policy differs")
        if inputs.get("label_semantics") not in {"incremental", "snapshot"}:
            raise ValueError("Vintage label semantics undeclared")
        ctx.mark("vintage_contract", label_semantics=inputs["label_semantics"])
        ctx.current = "registered_dataset"
        source, frame = _dataset(ctx.conn, ctx.archive, binding.task_id, binding.dataset_id, binding.dataset_sha256)
        ctx.mark("registered_dataset", rows=len(frame))
        ctx.current = "source_material"
        scratch = Path(ctx.stack.enter_context(tempfile.TemporaryDirectory(prefix="marvis-vintage-input-")))
        _source_input(ctx.archive, ctx.manifest, binding, source, frame, scratch)
        ctx.mark("source_material", scope="frozen_upload_and_registered_source_values")
        ctx.current = "independent_vintage"
        columns = [inputs["cohort_col"], inputs["mob_col"], inputs["bad_col"]]
        valid_labels = [math.isfinite(float(value)) if value is not None else False
                        for value in frame[inputs["bad_col"]]]
        dropped = len(valid_labels) - sum(valid_labels)
        if dropped:
            if not inputs["drop_nan_labels"]:
                raise ValueError("Vintage missing labels were not authorized for exclusion")
            frame = frame.loc[valid_labels]
        rows = frame[columns].to_dict("records")
        try:
            curve = vintage_reference(rows, cohort_col=columns[0], mob_col=columns[1], bad_col=columns[2],
                                      mob_max=int(inputs.get("mob_max", 12)), label_semantics=inputs["label_semantics"])
        except ValueError as exc:
            raise _Unsupported(f"Vintage source outside independent subset: {exc}") from None
        expected = {"cohorts": curve["cohorts"], "mob_axis": curve["mob_axis"], "curves": curve["curves"],
                    "counts": curve["counts"], "summary": vintage_summary_reference(curve, ref_mob=int(inputs.get("ref_mob", 6))),
                    "label_semantics": inputs["label_semantics"], "nan_labels_dropped": dropped, "warnings": curve["warnings"]}
        numbers(step["output"], expected)
        if set(step["output"]) != set(expected):
            raise ValueError("Vintage output fields differ")
        ctx.mark("independent_vintage", cohorts=len(curve["cohorts"]), points=len(curve["points"]),
                 scope="cohort_mob_counts_fixed_denominator_cumulation_wide_curve_summary_and_warnings")
        ctx.result["values"] = expected
    return ctx.result
