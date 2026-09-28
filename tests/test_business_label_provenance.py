from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from marvis.business_context import bind_business_context, model_business_measurement
from marvis.data.workspace import DataWorkspaceDraft
from marvis.packs.labeling.contracts import LabelingRequest
from marvis.packs.labeling.tools import tool_define_label
from marvis.packs.modeling.evidence_tools import build_training_evidence_ref
from marvis.packs.modeling.select_tools import tool_select_experiment
from marvis.packs.strategy import tools as strategy_tools
from marvis.packs.strategy.sample_design_v2_native_tools import (
    run_materialize_sample_design_v2_native,
)
from marvis.repositories.data_workspace import DataWorkspaceRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_business_context_producers import declaration
from tests.test_modeling_training_evidence_tool import _native_fixture, _run, _binding


def _observed_fixture(tmp_path, *, flaw=None):
    fx = _native_fixture(tmp_path)
    registry = fx["runtime"].registry
    sample = registry.read_authenticated_parquet_snapshot(fx["dataset"].id)
    rows = []
    for _, row in sample.iterrows():
        for mob in (0, 1):
            if flaw == "missing_member" and row.customer_id == "risk-14":
                continue
            if flaw == "immature" and row.customer_id == "risk-14" and mob == 1:
                continue
            rows.append(
                {
                    "customer_id": row.customer_id,
                    "cohort": "202602"
                    if flaw == "wrong_cohort" and row.customer_id == "risk-14"
                    else row.apply_month,
                    "snapshot_date": (
                        pd.Timestamp(row.apply_date) + pd.DateOffset(months=mob)
                    )
                    .date()
                    .isoformat(),
                    "mob": mob,
                    "dpd": 90
                    if row.bad
                    or (flaw == "wrong_target" and row.customer_id == "risk-14")
                    else 0,
                }
            )
    if flaw == "ambiguous_mob":
        rows.append(
            next(
                dict(row)
                for row in rows
                if row["customer_id"] == "risk-14" and row["mob"] == 1
            )
        )
    if flaw == "missing_observation":
        next(
            row for row in rows if row["customer_id"] == "risk-14" and row["mob"] == 1
        )["dpd"] = float("nan")
    if flaw == "ambiguous_cohort":
        extra = next(
            dict(row)
            for row in rows
            if row["customer_id"] == "risk-14" and row["mob"] == 1
        )
        extra.update(cohort="202602", mob=2)
        rows.append(extra)
    path = tmp_path / "observed-history.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    source = registry.register_existing(path, task_id=fx["task"].id, role="sample")
    workspaces = DataWorkspaceRepository(fx["settings"].db_path)
    active = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=source.id,
            active_dataset_content_hash=source.content_hash,
        ),
        expected_revision=fx["workspace"].revision,
    )
    request = LabelingRequest(
        dataset_id=source.id,
        expected_content_hash=source.content_hash,
        workspace_revision=active.revision,
        analysis_generation=active.analysis_generation,
        id_col="customer_id",
        cohort_col="cohort",
        mob_col="mob",
        date_col="snapshot_date",
        as_of_date="2026-05-31" if flaw == "future_cutoff" else "2026-04-30",
        target_col="bad",
        observation_window=0,
        performance_window=1,
        at_mob=1,
        rule_kind="dpd",
        dpd_col="dpd",
        threshold_dpd=90,
    )
    produced = tool_define_label(
        {
            **request.to_dict(),
            "proposal_hash": request.contract_hash,
            "confirm_immature_cohorts": False,
        },
        fx["ctx"],
    )
    active = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=fx["dataset"].id,
            active_dataset_content_hash=fx["dataset"].content_hash,
        ),
        expected_revision=active.revision,
    )
    active = workspaces.save(
        fx["task"].id,
        DataWorkspaceDraft(
            active_dataset_id=fx["dataset"].id,
            active_dataset_content_hash=fx["dataset"].content_hash,
            semantic_mapping=fx["workspace"].semantic_mapping,
        ),
        expected_revision=active.revision,
    )
    native_request = deepcopy(fx["sample_request"])
    native_request.pop("legacy_sample_design_ref")
    native_request.update(
        source_mode="native_active_dataset",
        dataset_id=fx["dataset"].id,
        expected_dataset_content_hash=fx["dataset"].content_hash,
        workspace_revision=active.revision,
        workspace_generation=active.analysis_generation,
        semantic_mapping_hash=strategy_tools.data_semantic_mapping_hash(
            active.semantic_mapping
        ),
        target_col="bad",
        target_bad_value=0 if flaw == "target_definition" else 1,
        drop_nan_labels=False,
    )
    native = run_materialize_sample_design_v2_native(
        native_request, fx["ctx"], strategy_tools._runtime(fx["ctx"])
    )
    records = TaskArtifactRepository(fx["settings"].db_path).list_for_task(
        fx["task"].id
    )
    artifacts = {
        name: next(
            record
            for record in records
            if Path(record["path"]).name == value["filename"]
        )
        for name, value in native["artifacts"].items()
    }
    ref = {
        "membership_artifact_id": artifacts["membership"]["id"],
        "expected_membership_artifact_content_hash": artifacts["membership"][
            "content_hash"
        ],
        "bundle_artifact_id": artifacts["bundle"]["id"],
        "expected_bundle_artifact_content_hash": artifacts["bundle"]["content_hash"],
        "expected_bundle_id": native["bundle_id"],
        "expected_sample_design_id": native["sample_design_id"],
        "expected_sample_design_content_hash": native["sample_design_content_hash"],
    }
    fx.update(sample=native, sample_ref=ref, workspace=active)
    fx["inputs"]["sample_design_ref"] = ref
    fx["labeling_ref"] = {
        "artifact_id": produced["evidence_artifact_id"],
        "content_hash": produced["evidence_content_hash"],
    }
    fx["label_source"] = source
    fx["label_result"] = produced
    return fx


def _context(fx):
    return bind_business_context(
        {
            "sample_design_ref": fx["sample_ref"],
            "declaration": declaration(),
            "labeling_evidence_ref": fx["labeling_ref"],
        },
        fx["ctx"],
        fx["runtime"],
    )


def _measured(fx):
    context = _context(fx)
    trained = _run(fx)
    ref = build_training_evidence_ref(_binding(fx, trained))
    inputs = {
        "experiment_ids": [trained["experiment_id"]],
        "selected_experiment_id": trained["experiment_id"],
        "refit_on_train_plus_test": False,
        "training_evidence_ref": ref,
        "business_context_ref": context["business_context_ref"],
    }
    return fx, context, trained, ref, inputs


def test_observed_labels_require_replayed_source_and_complete_actual_oot_membership(
    tmp_path,
):
    fx, context, trained, ref, inputs = _measured(_observed_fixture(tmp_path))
    output = tool_select_experiment(inputs, fx["ctx"])
    value = output["business_measurement"]
    assert value["label_origin"] == "observed"
    assert value["labels_mature"] is True
    proof = value["label_provenance"]
    assert proof["labeling_evidence_ref"] == fx["labeling_ref"]
    assert proof["covered_member_count"] == proof["required_member_count"] == 6
    assert proof["coverage"] == 1.0
    assert proof["sample_dataset_id"] == fx["dataset"].id
    assert proof["scope"] == "agreement_with_registered_label_producer"
    # After adoption, even historical context must replay the actual label source.
    path = fx["runtime"].registry.resolve_path(fx["label_source"].id)
    path.write_bytes(path.read_bytes() + b"changed")
    from marvis.data.errors import DataLayerError

    with pytest.raises((ValueError, DataLayerError)):
        model_business_measurement(
            fx["runtime"],
            fx["task"].id,
            context_ref=context["business_context_ref"],
            training_ref=ref,
            experiment_id=trained["experiment_id"],
            artifact_id=trained["model_artifact_id"],
        )


@pytest.mark.parametrize(
    "flaw,expected",
    [
        ("missing_member", "cover every"),
        ("wrong_cohort", "cohorts differ"),
        ("wrong_target", "target differs"),
        ("immature", "target differs|not mature"),
        ("ambiguous_mob", "ambiguous MOB"),
        ("missing_observation", "unobserved performance"),
        ("ambiguous_cohort", "ambiguous cohorts"),
        ("target_definition", "target definition differs"),
        ("future_cutoff", "cutoff exceeds"),
    ],
)
def test_incomplete_ambiguous_or_different_labels_cannot_certify_observed(
    tmp_path, flaw, expected
):
    fx = _observed_fixture(tmp_path, flaw=flaw)
    with pytest.raises(ValueError, match=expected):
        _context(fx)


def test_label_receipt_cross_task_and_file_drift_are_rejected(tmp_path):
    fx = _observed_fixture(tmp_path)
    from marvis.packs.labeling.evidence import _replay_labeling_evidence

    with pytest.raises(ValueError, match="task-owned"):
        _replay_labeling_evidence(fx["runtime"], "other-task", fx["labeling_ref"])
    record = TaskArtifactRepository(fx["settings"].db_path).get_for_task(
        fx["task"].id, fx["labeling_ref"]["artifact_id"]
    )
    Path(record["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact drifted"):
        _context(fx)


def test_registered_executor_with_mature_observed_requirement_passes_only_replayed_labels(
    tmp_path,
):
    from tests.test_business_context_producers import _exercise_registered_business_flow

    _exercise_registered_business_flow(
        _measured(_observed_fixture(tmp_path)), "metrics", observed=True
    )
