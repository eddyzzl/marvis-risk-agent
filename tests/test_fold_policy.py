"""The runtime plan cannot substitute a globally pruned candidate list."""

from copy import deepcopy

import numpy as np
import pytest

from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.fold_policy import build_fold_selection_plan, selection_session
from marvis.plugins.manifest import ToolRef
from tests.test_selection_evidence import _source, _run


def _plan_inputs(tmp_path, *, mask=False):
    runner, registry, _, source = _source(tmp_path)
    if mask:
        frame = registry.read_authenticated_parquet_snapshot(source.id)
        frame.loc[::8, "x"] = -999.0
        path = tmp_path / "sentinels.parquet"
        frame.to_parquet(path, index=False)
        source = registry.register_from_upload("task-feature", path, role="sample")
    screen = _run(runner, source, leakage_ks=1.0, top_k=1)
    assert screen.ok, screen.error
    current = source
    governance = {}
    if mask:
        resolved = runner.invoke(ToolRef("modeling", "resolve_special_values"), {
            "dataset_id": source.id, "features": ["x", "z"],
            "sentinel_columns": {"x": [[-999.0, .1]]},
            "decisions": {"x": {"action": "mask"}},
        }, task_id="task-feature")
        assert resolved.ok, resolved.error
        current = registry.get(resolved.output["result_dataset_id"])
        governance = resolved.output["governance"]
    selected = _run(runner, current, "select_features", iv_min=0.0, top_k=1)
    assert selected.ok, selected.error
    return registry, {
        "dataset_id": current.id, "features": selected.output["selected"],
        "target_col": "y", "split_col": "split",
        "split_values": {"train": "train", "test": "test", "oot": "oot"},
        "special_value_governance": governance,
        "selection_evidence_refs": [screen.output["selection_evidence_ref"], selected.output["selection_evidence_ref"]],
        "fold_selection": {"source_dataset_id": source.id, "candidates": ["x", "z"]},
    }


@pytest.mark.parametrize("mask", [False, True])
def test_native_plan_retains_all_candidates_and_exact_source_policy(tmp_path, mask):
    registry, inputs = _plan_inputs(tmp_path, mask=mask)
    plan = build_fold_selection_plan(registry, "task-feature", inputs)
    assert len(inputs["features"]) == 1
    assert plan["candidates"] == ["x", "z"]
    assert plan["upstream_candidate_selection"] == "unknown"
    assert bool(plan["special_value_policy"]) is mask
    with selection_session(registry, plan, target_type="binary") as session:
        controls = session.training_controls(split_col="split", train_value="train")
        result = session.prepare_fold(fit_positions=controls.index[:20].to_numpy(),
            train_positions=controls.index[:20].to_numpy(), valid_positions=np.arange(20, 40),
            test_positions=np.arange(20, 40))
    assert result.evidence["candidates"] == ["x", "z"]


@pytest.mark.parametrize("change", ["pruned", "reordered", "receipt", "mask"])
def test_changed_plan_cannot_reuse_recorded_identity(tmp_path, change):
    registry, inputs = _plan_inputs(tmp_path, mask=True)
    original = build_fold_selection_plan(registry, "task-feature", inputs)
    altered = deepcopy(inputs)
    if change == "pruned":
        altered["fold_selection"]["candidates"] = ["x"]
    elif change == "reordered":
        altered["fold_selection"]["candidates"].reverse()
    elif change == "receipt":
        altered["selection_evidence_refs"].pop()
    else:
        altered["special_value_governance"]["x"]["detected_values"] = [-998.0]
    with pytest.raises(ModelingError):
        build_fold_selection_plan(registry, "task-feature", altered)
    assert build_fold_selection_plan(registry, "task-feature", inputs)["sha256"] == original["sha256"]
