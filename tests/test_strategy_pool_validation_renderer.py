"""Renderer contract for independent Strategy Pool replay evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.packs.strategy.pool_validation import (
    canonical_strategy_pool_validation_json,
)
from marvis.packs.strategy.pool_validation_tools import (
    POOL_VALIDATION_ARTIFACT_KIND,
    POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION,
    POOL_VALIDATION_ORIGIN_TOOL,
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.repositories.task_artifacts import stable_task_artifact_id
from marvis.repositories.task_artifacts import TaskArtifactRepository
from marvis.plugins.manifest import ToolRef
from tests.test_strategy_pool_tools import (
    _add_inputs as _pool_add_inputs,
    _setup as _pool_setup,
)
from tests.test_strategy_pool_validation_tools import (
    _native_validation_request,
)
from tests.test_strategy_pool_validation import _build


def _output(*, artifact_id: str = "1" * 64) -> dict:
    evidence = _build(partition="validation")
    canonical = canonical_strategy_pool_validation_json(evidence).encode("utf-8")
    return {
        "schema_version": "strategy.measure-pool-validation-tool.v1",
        "evidence_id": evidence["evidence_id"],
        "content_hash": evidence["content_hash"],
        "pool_id": evidence["identity"]["pool_id"],
        "pool_revision": evidence["identity"]["revision"],
        "pool_snapshot_hash": evidence["identity"]["snapshot_hash"],
        "partition": "validation",
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": "validation",
        "validation_status": "independent_evidence",
        "population_count": 10,
        "labeled_count": 9,
        "unlabeled_count": 1,
        "evidence": evidence,
        "warnings": [
            "1 population rows are excluded from risk denominators",
        ],
        "artifact": {
            "artifact_id": artifact_id,
            "kind": "strategy_pool_validation_json",
            "format": "json",
            "filename": f"{evidence['evidence_id']}.json",
            "content_hash": hashlib.sha256(canonical).hexdigest(),
            "download_url": (
                f"/api/tasks/task-1/task-artifacts/{artifact_id}/download"
            ),
        },
        "not_mutated_pool": True,
        "not_created_strategy": True,
        "not_adopted": True,
        "not_promoted": True,
        "not_deployed": True,
    }


def _provenance(output: dict) -> dict:
    evidence = output["evidence"]
    identity = evidence["identity"]
    sources = evidence["source_bindings"]
    sample = sources["sample_design_v2"]
    return {
        "schema_version": POOL_VALIDATION_ARTIFACT_SCHEMA_VERSION,
        "producer_version": evidence["producer_version"],
        "task_id": identity["task_id"],
        "evidence_id": evidence["evidence_id"],
        "evidence_content_hash": evidence["content_hash"],
        "pool_ref": {
            "artifact_id": sources["pool_artifact"]["artifact_id"],
            "expected_artifact_content_hash": sources["pool_artifact"][
                "artifact_content_hash"
            ],
            "expected_pool_id": identity["pool_id"],
            "expected_revision": identity["revision"],
            "expected_revision_id": identity["revision_id"],
            "expected_snapshot_hash": identity["snapshot_hash"],
            "pool_id": identity["pool_id"],
            "revision_id": identity["revision_id"],
        },
        "sample_design_ref": {
            "membership_artifact_id": sample["membership_artifact_id"],
            "expected_membership_artifact_content_hash": sample[
                "membership_artifact_content_hash"
            ],
            "bundle_artifact_id": sample["bundle_artifact_id"],
            "expected_bundle_artifact_content_hash": sample[
                "bundle_artifact_content_hash"
            ],
            "expected_bundle_id": sample["bundle_id"],
            "expected_sample_design_id": sample["sample_design_id"],
            "expected_sample_design_content_hash": sample[
                "sample_design_content_hash"
            ],
        },
        "dataset_binding": sources["dataset"],
        "target_binding": sources["target"],
        "field_bindings": sources["fields"],
        "partition": evidence["partition"],
        "population": "risk",
        "comparison_mode": "absolute",
        "lifecycle_stage": evidence["partition"],
        "validation_status": "independent_evidence",
    }


def _registered_output(tmp_path: Path) -> tuple[dict, dict, Path]:
    tasks_root = tmp_path / "tasks"
    seed = _output()
    evidence = seed["evidence"]
    path = (
        tasks_root
        / "task-1"
        / "strategy_pool_validations"
        / f"{evidence['evidence_id']}.json"
    )
    path.parent.mkdir(parents=True)
    canonical = canonical_strategy_pool_validation_json(evidence).encode("utf-8")
    path.write_bytes(canonical)
    artifact_id = stable_task_artifact_id(
        task_id="task-1",
        kind=POOL_VALIDATION_ARTIFACT_KIND,
        path=str(path),
    )
    output = _output(artifact_id=artifact_id)
    record = {
        "id": artifact_id,
        "task_id": "task-1",
        "kind": POOL_VALIDATION_ARTIFACT_KIND,
        "path": str(path),
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "origin_tool": POOL_VALIDATION_ORIGIN_TOOL,
        "provenance": _provenance(output),
        "created_at": "2026-07-25T00:00:00+00:00",
    }
    return output, record, tasks_root


def _trusted_artifacts(record: dict, tasks_root: Path) -> dict:
    return {
        "pool_validation": {
            "record": record,
            "tasks_root": str(tasks_root),
        }
    }


def _trusted_inputs(output: dict) -> dict:
    evidence = output["evidence"]
    identity = evidence["identity"]
    sources = evidence["source_bindings"]
    sample = sources["sample_design_v2"]
    return {
        "strategy_type": identity["strategy_type"],
        "partition": evidence["partition"],
        "pool_ref": {
            "artifact_id": sources["pool_artifact"]["artifact_id"],
            "expected_artifact_content_hash": sources["pool_artifact"][
                "artifact_content_hash"
            ],
            "expected_pool_id": identity["pool_id"],
            "expected_revision": identity["revision"],
            "expected_revision_id": identity["revision_id"],
            "expected_snapshot_hash": identity["snapshot_hash"],
        },
        "sample_design_ref": {
            "membership_artifact_id": sample["membership_artifact_id"],
            "expected_membership_artifact_content_hash": sample[
                "membership_artifact_content_hash"
            ],
            "bundle_artifact_id": sample["bundle_artifact_id"],
            "expected_bundle_artifact_content_hash": sample[
                "bundle_artifact_content_hash"
            ],
            "expected_bundle_id": sample["bundle_id"],
            "expected_sample_design_id": sample["sample_design_id"],
            "expected_sample_design_content_hash": sample[
                "sample_design_content_hash"
            ],
        },
        "population": "risk",
        "comparison_mode": "absolute",
    }


def test_pool_validation_renderer_surfaces_independent_replay_not_psi(
    tmp_path: Path,
) -> None:
    output, record, tasks_root = _registered_output(tmp_path)
    text, tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id="task-1",
        trusted_inputs=_trusted_inputs(output),
        trusted_artifacts=_trusted_artifacts(record, tasks_root),
    )

    assert "独立样本回放验证完成" in text
    assert "independent replay evidence" in text
    assert "`validation`" in text
    assert "**10**" in text
    assert "**2** 个月" in text
    assert "不会修改 Pool" in text
    assert "不晋级、不采纳、不部署" in text
    assert output["artifact"]["download_url"] in text
    rendered = text + repr(tables)
    assert "PSI" not in rendered
    assert "稳定性" not in rendered

    summary = next(
        table for table in tables if table["title"] == "独立回放总体风险与金额"
    )
    assert [
        "overall",
        "10",
        "9",
        "55.6%",
        "1400.0000",
        "45.0000",
        "3.8%",
    ] in summary["rows"]
    actions = next(
        table for table in tables if table["title"] == "独立回放总体动作"
    )
    assert ["approve", "3", "30.0%", "1", "33.3%"] in actions["rows"]
    monthly = next(
        table for table in tables if table["title"] == "独立回放逐月证据"
    )
    assert len(monthly["rows"]) == 2


@pytest.mark.parametrize(
    ("strategy_type", "default_action", "action", "table_title"),
    [
        (
            "limit",
            {"type": "limit", "value": 1_000},
            {"type": "limit", "value": 2_000},
            "独立回放额度分布",
        ),
        (
            "pricing",
            {"type": "pricing", "value": 0.10},
            {"type": "pricing", "value": 0.20},
            "独立回放定价分布",
        ),
        (
            "segmentation",
            {"type": "segment", "value": "standard"},
            {"type": "segment", "value": "priority"},
            "独立回放分群分布",
        ),
    ],
)
def test_pool_validation_renderer_surfaces_native_typed_distribution(
    tmp_path: Path,
    strategy_type: str,
    default_action: dict,
    action: dict,
    table_title: str,
) -> None:
    fx = _pool_setup(tmp_path / strategy_type, native_sample=True)
    added_inputs = _pool_add_inputs(
        fx["first"],
        expected_revision=0,
        expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
    )
    added_inputs.update(
        {
            "strategy_type": strategy_type,
            "default_action": default_action,
            "action": action,
        }
    )
    added = run_add_candidate_to_pool(
        added_inputs,
        fx["ctx"],
        fx["runtime"],
    )
    request = _native_validation_request(
        fx,
        added,
        strategy_type=strategy_type,
    )
    output = run_measure_strategy_pool_validation(
        request,
        fx["ctx"],
        fx["runtime"],
    )
    record = next(
        item
        for item in TaskArtifactRepository(
            fx["settings"].db_path
        ).list_for_task(fx["task"].id)
        if item["id"] == output["artifact"]["artifact_id"]
    )

    text, tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id=fx["task"].id,
        trusted_inputs=request,
        trusted_artifacts=_trusted_artifacts(
            record,
            fx["settings"].tasks_dir,
        ),
    )

    assert "独立样本回放验证完成" in text
    assert strategy_type in text
    assert "结果完整性校验失败" not in text
    assert any(table["title"] == table_title for table in tables)


def test_pool_validation_renderer_fails_closed_on_task_or_plan_input_drift(
    tmp_path: Path,
) -> None:
    output, record, tasks_root = _registered_output(tmp_path)
    trusted = _trusted_inputs(output)
    trusted["partition"] = "oot"

    text, tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id="task-1",
        trusted_inputs=trusted,
        trusted_artifacts=_trusted_artifacts(record, tasks_root),
    )

    assert "独立样本回放验证结果完整性校验失败" in text
    assert "55.56%" not in text
    assert tables == []


def test_pool_validation_renderer_requires_registered_canonical_artifact(
    tmp_path: Path,
) -> None:
    output, record, tasks_root = _registered_output(tmp_path)

    missing_text, missing_tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id="task-1",
        trusted_inputs=_trusted_inputs(output),
    )
    forged = deepcopy(record)
    forged["origin_tool"] = "strategy.forged"
    forged_text, forged_tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id="task-1",
        trusted_inputs=_trusted_inputs(output),
        trusted_artifacts=_trusted_artifacts(forged, tasks_root),
    )

    assert "结果完整性校验失败" in missing_text
    assert output["artifact"]["download_url"] not in missing_text
    assert missing_tables == []
    assert "结果完整性校验失败" in forged_text
    assert output["artifact"]["download_url"] not in forged_text
    assert forged_tables == []


def test_pool_validation_renderer_rejects_registered_file_byte_drift(
    tmp_path: Path,
) -> None:
    output, record, tasks_root = _registered_output(tmp_path)
    Path(record["path"]).write_text("{}", encoding="utf-8")

    text, tables = render_tool_output(
        "measure_strategy_pool_validation",
        output,
        trusted_task_id="task-1",
        trusted_inputs=_trusted_inputs(output),
        trusted_artifacts=_trusted_artifacts(record, tasks_root),
    )

    assert "结果完整性校验失败" in text
    assert output["artifact"]["download_url"] not in text
    assert tables == []


def test_plan_composer_loads_registered_pool_validation_artifact(
    tmp_path: Path,
) -> None:
    output, record, tasks_root = _registered_output(tmp_path)
    calls: list[tuple[str, str]] = []
    step = PlanStep(
        id="pool-validation-step",
        plan_id="plan-1",
        index=0,
        title="measure independent Pool replay",
        tool_ref=ToolRef("strategy", "measure_strategy_pool_validation"),
        inputs=_trusted_inputs(output),
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
        output_ref="artifact:pool-validation",
    )
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="measure independent Pool replay",
        source="agent",
        template_id=None,
        steps=[step],
        autonomy_level=1,
    )

    def load_task_artifact(task_id: str, artifact_id: str):
        calls.append((task_id, artifact_id))
        return record

    message = PlanMessageComposer(
        load_output=lambda _step_id: output,
        load_task_artifact=load_task_artifact,
        tasks_root=tasks_root,
    ).done_message(plan, run_seq=1)

    assert calls == [("task-1", output["artifact"]["artifact_id"])]
    assert "独立样本回放验证完成" in message.content
    assert output["artifact"]["download_url"] in message.content
