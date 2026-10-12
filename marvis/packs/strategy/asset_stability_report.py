"""Same frozen asset metrics in JSON, Markdown, Excel and Word reports."""

from copy import deepcopy

from marvis.packs.strategy.asset_stability import validate_evidence
from marvis.packs.strategy.report_bundle import (
    StrategyReportBundleError,
    build_strategy_report_section,
    build_strategy_report_table,
)
from marvis.packs.strategy.report_bundle_adapters import (
    _artifact_ref,
    _column,
    _named,
    _present_field,
    _nullable_metric_field,
    _absent_field,
)


def add_asset_stability_report_evidence(
    source_inputs, *, bindings, sample_design, task_id
):
    if not bindings:
        return source_inputs
    result = deepcopy(source_inputs)
    section = next(s for s in result["sections"] if s["key"] == "univariate_and_models")
    tables = list(section["tables"])
    summary = list(section["summary_fields"])
    refs = list(section["source_refs"])
    design = sample_design.bundle["sample_design"]
    for index, binding in enumerate(bindings):
        evidence = validate_evidence(binding.evidence)
        frozen = evidence["freeze"]
        sample = frozen["sample_design_ref"]
        if (
            binding.task_id != task_id
            or frozen["task_id"] != task_id
            or sample["bundle_artifact_id"] != sample_design.bundle_artifact_id
            or sample["expected_bundle_artifact_content_hash"]
            != sample_design.bundle_artifact_content_hash
            or sample["expected_sample_design_id"] != design["sample_design_id"]
            or sample["expected_sample_design_content_hash"] != design["content_hash"]
            or sample["membership_artifact_id"] != sample_design.membership_artifact_id
            or sample["expected_membership_artifact_content_hash"]
            != sample_design.membership_artifact_content_hash
        ):
            raise StrategyReportBundleError(
                "independent asset stability does not bind exact report SampleDesign V2"
            )
        ref = _artifact_ref(
            "task_artifact", binding.artifact_id, binding.artifact_content_hash
        )
        frozen_ref = _artifact_ref(
            "task_artifact",
            evidence["frozen_ref"]["artifact_id"],
            evidence["frozen_ref"]["artifact_content_hash"],
        )
        source_ref = _artifact_ref(
            "strategy_candidate_asset",
            frozen["source_ref"]["artifact_id"],
            frozen["source_ref"]["artifact_content_hash"],
        )
        sample_ref = _artifact_ref(
            "sample_design",
            sample["bundle_artifact_id"],
            sample["expected_bundle_artifact_content_hash"],
        )
        prefix = f"asset_stability_{index}"
        for key, value in [
            ("asset_id", frozen["source_ref"]["asset_id"]),
            ("asset_hash", frozen["source_ref"]["asset_hash"]),
            ("source_kind", frozen["source_kind"]),
            ("freeze_id", frozen["stability_id"]),
            ("sample_design_id", sample["expected_sample_design_id"]),
            ("baseline_policy", frozen["baseline_policy"]),
            ("maturity", frozen["maturity"]),
            ("target", frozen["target"]["column"]),
            ("target_bad_value", frozen["target"]["bad_value"]),
            ("development_count", evidence["baseline"]["sample_count"]),
            (
                "evidence_scope",
                "independent partition distribution; candidate state unchanged",
            ),
        ]:
            summary.append(_named(f"{prefix}_{key}", key, _present_field(value, ref)))
        rows = []
        for partition, metrics in [
            ("development", evidence["baseline"]),
            *[(p["partition"], p) for p in evidence["partitions"]],
        ]:
            for ordinal, bucket in enumerate(metrics["buckets"]):
                cells = {
                    key: _nullable_metric_field(value, ref)
                    for key, value in {
                        "partition": partition,
                        "bucket_id": bucket["bucket_id"],
                        "count": bucket["count"],
                        "share": bucket["share"],
                        "labeled_count": bucket["labeled_count"],
                        "bad_count": bucket["bad_count"],
                        "bad_rate": bucket["bad_rate"],
                        "psi_vs_development": metrics.get("psi_vs_development"),
                        "psi_contribution": bucket.get("psi_contribution"),
                    }.items()
                }
                cells["performance_status"] = _present_field(
                    metrics["performance"]["status"], ref
                )
                for metric in ("auc", "ks"):
                    performance = metrics["performance"]
                    cells[metric] = (
                        _present_field(performance[metric], ref)
                        if performance[metric] is not None
                        else _absent_field(
                            "unavailable"
                            if performance["status"] == "unavailable"
                            else "not_applicable",
                            note=performance["reason"],
                        )
                    )
                rows.append({"row_id": f"{partition}-{ordinal}", "cells": cells})
        columns = [
            _column("partition", "分区"),
            _column("bucket_id", "固定资产分桶"),
            _column("count", "样本数", precision=0),
            _column("share", "占比", unit="%", precision=6),
            _column("labeled_count", "有标签数", precision=0),
            _column("bad_count", "坏样本数", precision=0),
            _column("bad_rate", "坏率", unit="%", precision=6),
            _column("psi_vs_development", "相对development PSI", precision=6),
            _column("psi_contribution", "PSI贡献", precision=6),
            _column("performance_status", "评分区分度状态"),
            _column("auc", "原始PD AUC", precision=6),
            _column("ks", "原始PD KS", precision=6),
        ]
        tables.append(
            build_strategy_report_table(
                table_id=prefix,
                title=f"独立候选 {frozen['source_ref']['asset_id']} 分区稳定性",
                sheet_key=f"appendix_asset_stab_{index}",
                granularity="aggregate",
                content_class="bin_summary",
                columns=columns,
                rows=rows,
                source_refs=[ref, frozen_ref, source_ref, sample_ref],
            )
        )
        month_rows = []
        for part in evidence["partitions"]:
            for month in part["monthly"]:
                cells = {
                    key: _present_field(value, ref)
                    for key, value in {
                        "partition": part["partition"],
                        "month": month["month"],
                        "sample_count": month["sample_count"],
                        "labeled_count": month["labeled_count"],
                        "bad_count": month["bad_count"],
                        "psi_vs_development": month["psi_vs_development"],
                        "small_sample": month["small_sample"],
                    }.items()
                }
                month_rows.append(
                    {"row_id": f"{part['partition']}-{month['month']}", "cells": cells}
                )
        tables.append(
            build_strategy_report_table(
                table_id=f"{prefix}_monthly",
                title="独立分区逐月稳定性（development固定基线）",
                sheet_key=f"appendix_asset_month_{index}",
                granularity="aggregate",
                content_class="monthly_summary",
                columns=[
                    _column("partition", "分区"),
                    _column("month", "月份"),
                    _column("sample_count", "样本数", precision=0),
                    _column("labeled_count", "有标签数", precision=0),
                    _column("bad_count", "坏样本数", precision=0),
                    _column("psi_vs_development", "相对development PSI", precision=6),
                    _column("small_sample", "少量样本"),
                ],
                rows=month_rows,
                source_refs=[ref, frozen_ref, source_ref, sample_ref],
            )
        )
        refs.extend([ref, frozen_ref, source_ref, sample_ref])
        result["strategy_artifact_refs"].extend([ref, frozen_ref, source_ref])
    amended = build_strategy_report_section(
        key=section["key"],
        title=section["title"],
        availability="present",
        summary_fields=summary,
        tables=tables,
        stage_evidence=section["stage_evidence"],
        red_flags=section["red_flags"],
        source_refs=refs,
    )
    result["sections"] = [
        amended if s["key"] == section["key"] else s for s in result["sections"]
    ]
    # Keep stable deduplicated source inventories without upgrading report effect stages.
    result["strategy_artifact_refs"] = [
        dict(x)
        for x in {
            tuple(sorted(r.items())): r for r in result["strategy_artifact_refs"]
        }.values()
    ]
    return result
