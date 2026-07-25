"""Fail-closed rendering for candidate monthly stability evidence."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from marvis.agent.plan_message_composer import PlanMessageComposer
from marvis.agent.renderers import render_tool_output
from marvis.orchestrator.contracts import Plan, PlanStep, StepStatus
from marvis.packs.strategy.candidate_stability import (
    build_candidate_stability_artifact,
    canonical_candidate_stability_artifact_json,
)
from marvis.packs.strategy.candidate_stability_tools import (
    ARTIFACT_KIND,
    ARTIFACT_SCHEMA_VERSION,
    ORIGIN_TOOL,
    TOOL_SCHEMA_VERSION,
)
from marvis.packs.strategy.pool import strategy_pool_id
from marvis.plugins.manifest import ToolRef


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64


def _identity(*, task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "dataset_id": "dataset-1",
        "dataset_content_hash": HASH_A,
        "workspace_revision": 3,
        "workspace_generation": 2,
        "semantic_mapping_hash": HASH_B,
        "sample_context_hash": HASH_C,
    }


def _sample_ref() -> dict:
    return {
        "artifact_id": HASH_D,
        "artifact_content_hash": HASH_E,
        "sample_design_id": "strategy-sample-design-" + "1" * 24,
        "sample_design_content_hash": HASH_F,
        "partition": "development",
    }


def _asset_source() -> dict:
    return {
        "source_kind": "univariate_asset",
        "artifact_id": "1" * 64,
        "artifact_content_hash": HASH_D,
        "asset_id": "candidate-asset-1",
        "asset_hash": HASH_E,
        "rule_id": "candidate-rule-1",
    }


def _pool_source(*, task_id: str = "task-1") -> dict:
    return {
        "source_kind": "pool_entry",
        "artifact_id": "pool-artifact-1",
        "artifact_content_hash": HASH_D,
        "pool_id": strategy_pool_id(task_id, "approval"),
        "revision": 4,
        "revision_id": "strategy-pool-revision-4",
        "snapshot_hash": HASH_E,
        "entry_id": "strategy-pool-entry-1",
        "rule_id": "candidate-rule-1",
    }


def _stability(
    *,
    pool_entry: bool = False,
    task_id: str = "task-1",
) -> dict:
    frame = pd.DataFrame(
        {
            "month": [
                "202603",
                "202601",
                "202602",
                "202601",
                "202603",
                "202602",
                "202601",
                "202603",
                "202602",
                "202601",
            ],
            "bad": [0, 0, 0, 1, 1, 1, None, 1, None, 1],
        }
    )
    return build_candidate_stability_artifact(
        frame=frame,
        month_col="month",
        target_col="bad",
        hit_mask=[
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
        ],
        basis=(
            "pool_entry_incremental_first_match"
            if pool_entry
            else "asset_rule_hit"
        ),
        identity=_identity(task_id=task_id),
        source_ref=(
            _pool_source(task_id=task_id) if pool_entry else _asset_source()
        ),
        sample_design_ref=_sample_ref(),
    )


def _stable_artifact_id(*, task_id: str, path: str) -> str:
    identity = json.dumps(
        [task_id, ARTIFACT_KIND, path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        f"marvis.task_artifact.v1:{identity}".encode("utf-8")
    ).hexdigest()


def _artifact_path(stability: dict) -> Path:
    return (
        Path("/tmp/marvis-renderer-tests")
        / stability["identity"]["task_id"]
        / "strategy_candidate_stability"
        / f"{stability['stability_id']}.json"
    )


def _output(
    *,
    pool_entry: bool = False,
    task_id: str = "task-1",
) -> dict:
    stability = _stability(pool_entry=pool_entry, task_id=task_id)
    artifact_hash = hashlib.sha256(
        canonical_candidate_stability_artifact_json(stability).encode("utf-8")
    ).hexdigest()
    path = _artifact_path(stability)
    artifact_id = _stable_artifact_id(task_id=task_id, path=str(path))
    return {
        "schema_version": TOOL_SCHEMA_VERSION,
        "stability_id": stability["stability_id"],
        "content_hash": stability["content_hash"],
        "basis": stability["basis"],
        "source_kind": stability["source_ref"]["source_kind"],
        "month_col": stability["bindings"]["month_col"],
        "population_count": stability["summary"]["population_count"],
        "month_count": stability["summary"]["month_count"],
        "max_psi": stability["summary"]["max_psi"],
        "stability": stability,
        "warnings": [
            (
                f"month {flag['month']} has {flag['observed_rows']} rows, "
                f"below minimum {flag['minimum_rows']}"
            )
            for flag in stability["red_flags"]
        ],
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "kind": ARTIFACT_KIND,
                "format": "json",
                "filename": f"{stability['stability_id']}.json",
                "content_hash": artifact_hash,
                "download_url": (
                    f"/api/tasks/{task_id}/task-artifacts/{artifact_id}/download"
                ),
            }
        ],
        "not_created_strategy": True,
        "not_adopted": True,
        "not_deployed": True,
    }


def _inputs(output: dict) -> dict:
    source = output["stability"]["source_ref"]
    if source["source_kind"] == "univariate_asset":
        return {
            "source_kind": "univariate_asset",
            "source_artifact_id": source["artifact_id"],
            "expected_artifact_content_hash": source[
                "artifact_content_hash"
            ],
            "expected_asset_id": source["asset_id"],
            "expected_asset_hash": source["asset_hash"],
        }
    return {
        "source_kind": "pool_entry",
        "strategy_type": "approval",
        "expected_pool_revision": source["revision"],
        "expected_pool_snapshot_hash": source["snapshot_hash"],
        "entry_id": source["entry_id"],
    }


def _artifact_provenance(output: dict) -> dict:
    stability = output["stability"]
    identity = stability["identity"]
    source = stability["source_ref"]
    bindings = stability["bindings"]
    sample_ref = stability["sample_design_ref"]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "producer_version": stability["producer_version"],
        "task_id": identity["task_id"],
        "stability_id": stability["stability_id"],
        "stability_content_hash": stability["content_hash"],
        "basis": stability["basis"],
        "source_kind": source["source_kind"],
        "source_artifact_id": source["artifact_id"],
        "source_artifact_content_hash": source["artifact_content_hash"],
        "source_id": (
            source["asset_id"]
            if source["source_kind"] == "univariate_asset"
            else source["pool_id"]
        ),
        "source_hash": (
            source["asset_hash"]
            if source["source_kind"] == "univariate_asset"
            else source["snapshot_hash"]
        ),
        "rule_id": source["rule_id"],
        "entry_id": source.get("entry_id"),
        "pool_id": source.get("pool_id"),
        "pool_revision": source.get("revision"),
        "pool_revision_id": source.get("revision_id"),
        "dataset_id": identity["dataset_id"],
        "dataset_content_hash": identity["dataset_content_hash"],
        "workspace_revision": identity["workspace_revision"],
        "workspace_generation": identity["workspace_generation"],
        "semantic_mapping_hash": identity["semantic_mapping_hash"],
        "target_col": bindings["target_col"],
        "month_col": bindings["month_col"],
        "sample_design_ref": sample_ref,
        "sample_context_hash": identity["sample_context_hash"],
        "sample_partition": sample_ref["partition"],
    }


def _registry_record(output: dict) -> dict:
    stability = output["stability"]
    artifact = output["artifacts"][0]
    return {
        "id": artifact["artifact_id"],
        "task_id": stability["identity"]["task_id"],
        "kind": ARTIFACT_KIND,
        "path": str(_artifact_path(stability)),
        "content_hash": artifact["content_hash"],
        "origin_tool": ORIGIN_TOOL,
        "provenance": _artifact_provenance(output),
        "created_at": "2026-07-24T00:00:00+00:00",
    }


def _render(
    output: dict,
    *,
    trusted_task_id: str = "task-1",
    trusted_inputs: dict | None = None,
    record: dict | None = None,
):
    return render_tool_output(
        "measure_candidate_monthly_stability",
        output,
        trusted_task_id=trusted_task_id,
        trusted_inputs=trusted_inputs or _inputs(output),
        trusted_artifacts={
            "stability": record or _registry_record(output),
        },
    )


def test_renderer_surfaces_authenticated_asset_metrics_low_sample_and_download() -> None:
    output = _output()
    stability = output["stability"]

    text, tables = _render(output)

    assert "单变量候选资产 `candidate-asset-1`" in text
    assert "直接规则命中/未命中分布" in text
    assert "完整 development 样本 **10** 行，共 **3** 个月" in text
    assert f"**{stability['summary']['max_psi']:.4f}**" in text
    assert f"`{stability['summary']['max_psi_month']}`" in text
    assert "**低样本提醒**" in text
    assert "`202601` 4 行 < minimum_rows=30" in text
    assert "低样本只标记证据强度" in text
    assert "高风险" not in text
    assert "development / backtested / unvalidated" in text
    assert "未创建策略、未修改 Strategy Pool、未采纳、未部署" in text
    assert output["artifacts"][0]["download_url"] in text
    assert output["artifacts"][0]["content_hash"] in text

    monthly = tables[0]
    assert monthly["title"] == "候选逐月稳定性（完整 development 基线）"
    assert monthly["rows"][0][:4] == ["202601", "4", "2", "2"]
    assert monthly["rows"][0][-1] == (
        f"{stability['monthly'][0]['psi_vs_development']:.4f}"
    )
    low_sample = tables[1]
    assert low_sample["title"] == "候选逐月稳定性低样本提醒"
    assert low_sample["rows"] == [
        ["202601", "4", "30"],
        ["202602", "3", "30"],
        ["202603", "3", "30"],
    ]


def test_renderer_names_exact_pool_entry_and_incremental_first_match_basis() -> None:
    output = _output(pool_entry=True)

    text, tables = _render(output)

    assert f"Strategy Pool `{output['stability']['source_ref']['pool_id']}` revision 4" in text
    assert "当前顺序条目 `strategy-pool-entry-1`" in text
    assert "增量 first-match 命中/未命中分布" in text
    assert tables[0]["rows"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__(
            "population_count", value["population_count"] + 1
        ),
        lambda value: value.__setitem__("max_psi", value["max_psi"] + 0.5),
        lambda value: value.__setitem__("warnings", []),
        lambda value: value["artifacts"][0].__setitem__(
            "content_hash", "0" * 64
        ),
        lambda value: value.__setitem__("caller_metric", 0.99),
    ],
)
def test_renderer_fails_closed_on_outer_or_artifact_drift(mutate) -> None:
    output = _output()
    forged = deepcopy(output)
    mutate(forged)

    text, tables = _render(
        forged,
        trusted_inputs=_inputs(output),
        record=_registry_record(output),
    )

    assert "结果完整性校验失败" in text
    assert output["stability_id"] not in text
    assert output["artifacts"][0]["download_url"] not in text
    assert tables == []


def test_renderer_fails_closed_on_inner_evidence_drift() -> None:
    output = _output()
    forged = deepcopy(output)
    forged["stability"]["summary"]["max_psi"] += 0.5

    text, tables = _render(
        forged,
        trusted_inputs=_inputs(output),
        record=_registry_record(output),
    )

    assert "结果完整性校验失败" in text
    assert output["artifacts"][0]["download_url"] not in text
    assert tables == []


def test_renderer_keeps_dedicated_failure_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "marvis.agent.renderers._validate_candidate_stability_tool_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unexpected")
        ),
    )

    text, tables = render_tool_output(
        "measure_candidate_monthly_stability",
        {"population_count": 999999},
    )

    assert "候选逐月稳定性结果完整性校验失败" in text
    assert "999999" not in text
    assert tables == []


def test_renderer_rejects_self_consistent_cross_task_stability_output() -> None:
    foreign = _output(task_id="task-2")

    text, tables = _render(
        foreign,
        trusted_task_id="task-1",
        record=_registry_record(foreign),
    )

    assert "结果完整性校验失败" in text
    assert foreign["stability_id"] not in text
    assert foreign["artifacts"][0]["download_url"] not in text
    assert tables == []


@pytest.mark.parametrize("pool_entry", [False, True])
def test_renderer_rejects_terminal_source_input_drift(pool_entry: bool) -> None:
    output = _output(pool_entry=pool_entry)
    forged_inputs = deepcopy(_inputs(output))
    if pool_entry:
        forged_inputs["strategy_type"] = "reject"
    else:
        forged_inputs["expected_asset_hash"] = "0" * 64

    text, tables = _render(output, trusted_inputs=forged_inputs)

    assert "结果完整性校验失败" in text
    assert output["stability_id"] not in text
    assert output["artifacts"][0]["download_url"] not in text
    assert tables == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("id", "0" * 64),
        lambda value: value.__setitem__("task_id", "task-2"),
        lambda value: value.__setitem__("kind", "forged_kind"),
        lambda value: value.__setitem__("origin_tool", "forged.tool"),
        lambda value: value.__setitem__(
            "path",
            str(Path(value["path"]).with_name("forged.json")),
        ),
        lambda value: value.__setitem__("content_hash", "0" * 64),
        lambda value: value["provenance"].__setitem__(
            "stability_content_hash",
            "0" * 64,
        ),
    ],
)
def test_renderer_rejects_registry_identity_path_hash_or_provenance_drift(
    mutate,
) -> None:
    output = _output()
    record = deepcopy(_registry_record(output))
    mutate(record)

    text, tables = _render(output, record=record)

    assert "结果完整性校验失败" in text
    assert output["stability_id"] not in text
    assert output["artifacts"][0]["download_url"] not in text
    assert tables == []


def test_plan_composer_loads_registered_stability_artifact_for_renderer() -> None:
    output = _output()
    inputs = _inputs(output)
    record = _registry_record(output)
    calls: list[tuple[str, str]] = []
    step = PlanStep(
        id="stability-step",
        plan_id="plan-1",
        index=0,
        title="measure stability",
        tool_ref=ToolRef(
            "strategy",
            "measure_candidate_monthly_stability",
        ),
        inputs=inputs,
        depends_on=[],
        post_checks=[],
        status=StepStatus.DONE,
        output_ref="artifact:stability",
    )
    plan = Plan(
        id="plan-1",
        task_id="task-1",
        goal="measure candidate monthly stability",
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
    ).done_message(plan, run_seq=1)

    assert calls == [("task-1", output["artifacts"][0]["artifact_id"])]
    assert "候选逐月稳定性测算完成" in message.content
    assert output["artifacts"][0]["download_url"] in message.content
