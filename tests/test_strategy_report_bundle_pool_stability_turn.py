from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import marvis.agent.turn_handlers as turn_handlers
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import (
    _StrategyV2EvidenceSetupError,
    _strategy_report_bundle_v2_plan_slots,
)
from marvis.orchestrator.templates.strategy import STRATEGY_REPORT_BUNDLE_V2
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.pool_stability_tools import (
    POOL_STABILITY_ARTIFACT_KIND,
    run_measure_strategy_pool_stability,
)
from marvis.repositories.task_artifacts import TaskArtifactRepository
from test_strategy_report_bundle_tools import (
    _setup as _setup_legacy_report,
    _setup_impact_cube_report,
)


def _draft() -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_report_bundle_v2",
        workflow_inputs={
            "title": "策略池稳定性评审报告",
            "status": "partial",
        },
    )


def _runtime(fixture: dict) -> SimpleNamespace:
    return SimpleNamespace(settings=fixture["settings"])


def _pool_stability_ref(output: dict) -> dict[str, str]:
    return {
        "artifact_id": output["artifact"]["artifact_id"],
        "expected_artifact_content_hash": output["artifact"]["content_hash"],
        "expected_stability_id": output["stability_id"],
        "expected_stability_content_hash": output["content_hash"],
    }


def test_report_turn_freezes_exact_matching_pool_stability_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["request"]["impact_cube_ref"],
        fixture["ctx"],
        fixture["runtime"],
    )
    observed: dict[str, object] = {}
    original_preflight = (
        turn_handlers.build_strategy_report_bundle_source_inputs
    )

    def capture_preflight(**kwargs):
        observed["pool_stability"] = kwargs["pool_stability"]
        return original_preflight(**kwargs)

    monkeypatch.setattr(
        turn_handlers,
        "build_strategy_report_bundle_source_inputs",
        capture_preflight,
    )

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )

    assert slots["pool_stability_ref"] == _pool_stability_ref(output)
    assert observed["pool_stability"].artifact_id == (
        output["artifact"]["artifact_id"]
    )


def test_report_turn_without_pool_stability_freezes_none(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )

    assert slots["impact_cube_ref"] == fixture["request"]["impact_cube_ref"]
    assert slots["pool_stability_ref"] is None


def test_report_turn_skips_newer_authenticated_unrelated_pool_stability(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    newer_request = deepcopy(fixture["impact_request"])
    newer_request["partitions"] = ["development", "validation", "oot"]
    selected_cube = run_measure_strategy_impact_cube(
        newer_request,
        fixture["ctx"],
        fixture["runtime"],
    )
    selected_cube_ref = {
        "artifact_id": selected_cube["artifact"]["artifact_id"],
        "expected_artifact_content_hash": selected_cube["artifact"][
            "content_hash"
        ],
        "expected_cube_id": selected_cube["cube_id"],
        "expected_cube_content_hash": selected_cube["content_hash"],
    }
    matching = run_measure_strategy_pool_stability(
        selected_cube_ref,
        fixture["ctx"],
        fixture["runtime"],
    )
    unrelated = run_measure_strategy_pool_stability(
        fixture["request"]["impact_cube_ref"],
        fixture["ctx"],
        fixture["runtime"],
    )
    records, total = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_recent_for_task_kind_with_count(
        fixture["task"].id,
        "strategy_pool_stability_json",
        limit=64,
    )
    assert total == 2
    assert [record["id"] for record in records] == [
        unrelated["artifact"]["artifact_id"],
        matching["artifact"]["artifact_id"],
    ]

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )

    assert slots["impact_cube_ref"] == selected_cube_ref
    assert slots["pool_stability_ref"] == _pool_stability_ref(matching)


def test_report_turn_corrupt_latest_pool_stability_blocks_fallback(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    output = run_measure_strategy_pool_stability(
        fixture["request"]["impact_cube_ref"],
        fixture["ctx"],
        fixture["runtime"],
    )
    record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        output["artifact"]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_text("{}", encoding="utf-8")

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_bundle_v2_plan_slots(
            _runtime(fixture),
            fixture["task"],
            _draft(),
            source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
        )

    assert (
        raised.value.code
        == "strategy_report_bundle_v2_pool_stability_invalid"
    )


def test_report_turn_pool_stability_selection_window_exhaustion_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    records = [
        {
            "id": f"{index:064x}",
            "kind": POOL_STABILITY_ARTIFACT_KIND,
            "content_hash": f"{index + 100:064x}",
            "provenance": {
                "stability_id": (
                    "strategy-pool-stability-" + f"{index:024x}"
                ),
                "stability_content_hash": f"{index + 200:064x}",
            },
        }
        for index in range(64)
    ]
    unrelated_ref = {
        **fixture["request"]["impact_cube_ref"],
        "artifact_id": "f" * 64,
    }
    calls: list[str] = []
    original_recent = (
        TaskArtifactRepository.list_recent_for_task_kind_with_count
    )

    def bounded_recent(self, task_id, kind, *, limit):
        if kind == POOL_STABILITY_ARTIFACT_KIND:
            assert limit == 64
            return records, 65
        return original_recent(
            self,
            task_id,
            kind,
            limit=limit,
        )

    def authenticate(_runtime, *, artifact_id, **_kwargs):
        calls.append(artifact_id)
        return SimpleNamespace(
            stability={
                "source_bindings": {"impact_cube": unrelated_ref},
            }
        )

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_recent_for_task_kind_with_count",
        bounded_recent,
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_strategy_pool_stability_artifact",
        authenticate,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_bundle_v2_plan_slots(
            _runtime(fixture),
            fixture["task"],
            _draft(),
            source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_pool_stability_"
        "selection_window_exhausted"
    )
    assert calls == [record["id"] for record in records]


def test_report_turn_legacy_pool_impact_never_selects_pool_stability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_legacy_report(tmp_path)

    def reject_selection(*_args, **_kwargs):
        raise AssertionError(
            "legacy PoolImpact report must not select PoolStability"
        )

    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_latest_pool_stability_binding",
        reject_selection,
    )

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )

    assert slots["impact_cube_ref"] is None
    assert slots["pool_impact_ref"] is not None
    assert slots["pool_stability_ref"] is None


def test_report_turn_pool_stability_ref_does_not_rebind_after_planning(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    initial = run_measure_strategy_pool_stability(
        fixture["request"]["impact_cube_ref"],
        fixture["ctx"],
        fixture["runtime"],
    )
    planned = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )
    frozen = deepcopy(planned)

    later_request = deepcopy(fixture["impact_request"])
    later_request["partitions"] = ["development", "validation", "oot"]
    later_cube = run_measure_strategy_impact_cube(
        later_request,
        fixture["ctx"],
        fixture["runtime"],
    )
    later_ref = {
        "artifact_id": later_cube["artifact"]["artifact_id"],
        "expected_artifact_content_hash": later_cube["artifact"][
            "content_hash"
        ],
        "expected_cube_id": later_cube["cube_id"],
        "expected_cube_content_hash": later_cube["content_hash"],
    }
    later = run_measure_strategy_pool_stability(
        later_ref,
        fixture["ctx"],
        fixture["runtime"],
    )
    refreshed = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略池稳定性评审报告。"},
    )

    assert planned == frozen
    assert planned["pool_stability_ref"] == _pool_stability_ref(initial)
    assert refreshed["pool_stability_ref"] == _pool_stability_ref(later)


def test_report_template_exposes_optional_pool_stability_task_context() -> None:
    slot = next(
        item
        for item in STRATEGY_REPORT_BUNDLE_V2.slots
        if item.name == "pool_stability_ref"
    )

    assert slot.required is False
    assert slot.source == "task_context"
    assert (
        STRATEGY_REPORT_BUNDLE_V2.steps[0].inputs_template[
            "pool_stability_ref"
        ]
        == "{slot:pool_stability_ref}"
    )
