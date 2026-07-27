"""Turn-boundary binding for natural-language StrategyReportBundle V2."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import marvis.agent.turn_handlers as turn_handlers
from marvis.agent.strategy_request_compiler import StandardWorkflowRequestDraft
from marvis.agent.turn_handlers import (
    _StrategyV2EvidenceSetupError,
    _strategy_report_bundle_v2_plan_slots,
    _strategy_report_current_pool_binding,
    _strategy_report_identity,
    _strategy_report_latest_impact_cube_binding,
    _strategy_report_latest_pool_impact_binding,
    _strategy_report_latest_sample_binding,
    _strategy_report_optional_model_evidence,
    _strategy_report_optional_score_evidence,
    _strategy_report_read_runtime,
    _strategy_report_requested_pool_type,
)
from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.packs.strategy.candidate_stability_tools import (
    resolve_candidate_monthly_stability_inputs,
    run_measure_candidate_monthly_stability,
)
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.impact_cube_tools import (
    run_measure_strategy_impact_cube,
)
from marvis.packs.strategy.model_evidence_tools import (
    MODEL_EVIDENCE_V2_ARTIFACT_KIND,
)
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import run_add_candidate_to_pool
from marvis.packs.strategy.pool_stability_tools import (
    run_measure_strategy_pool_stability,
)
from marvis.packs.strategy.pool_validation_tools import (
    run_measure_strategy_pool_validation,
)
from marvis.packs.strategy.project_context_tools import (
    run_materialize_project_context,
)
from marvis.packs.strategy.sample_design_v2_native_tools import (
    SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND,
    SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
)
from marvis.packs.strategy.sample_design_v2_tools import (
    SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND,
)
from marvis.packs.strategy.report_bundle_tools import (
    run_build_strategy_report_bundle_v2,
    validate_build_strategy_report_bundle_v2_tool_output,
)
from marvis.packs.strategy.sample_membership import decode_sample_membership
from marvis.plugins.manifest import ToolRef
from marvis.repositories.strategy_reports import StrategyReportRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from test_strategy_report_bundle_tools import (
    _attach_candidate_stability,
    _run,
    _setup,
    _setup_impact_cube_report,
)
from test_strategy_candidate_stability_tools import (
    _pool_add_inputs as _candidate_stability_pool_add_inputs,
    _setup as _candidate_stability_setup,
)
from test_strategy_request_turn import (
    _FakeLLM as _StoredStrategyLLM,
    _install_llm as _install_stored_strategy_llm,
    _saved_strategy,
    _strategy_request_plans,
    _task as _strategy_task,
)


class _ReportLLM:
    def __init__(self, workflow_inputs: dict | None = None) -> None:
        self.workflow_inputs = workflow_inputs or {}
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps(
            {
                "request_kind": "standard_workflow",
                "workflow": "strategy_report_bundle_v2",
                "workflow_inputs": self.workflow_inputs,
            },
            ensure_ascii=False,
        )


def _draft(
    *,
    title: str = "策略迭代评审报告",
    status: str = "partial",
) -> StandardWorkflowRequestDraft:
    return StandardWorkflowRequestDraft(
        workflow="strategy_report_bundle_v2",
        workflow_inputs={"title": title, "status": status},
    )


def _runtime(fixture: dict) -> SimpleNamespace:
    return SimpleNamespace(settings=fixture["settings"])


def _setup_native_parallel_report(tmp_path: Path) -> dict:
    fixture = _candidate_stability_setup(
        tmp_path,
        native_sample=True,
        target_bad_value=0,
    )
    pool_output = run_add_candidate_to_pool(
        _candidate_stability_pool_add_inputs(
            fixture["first"],
            expected_revision=0,
            expected_hash=ABSENT_POOL_SNAPSHOT_HASH,
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    records = TaskArtifactRepository(
        fixture["settings"].db_path
    ).list_for_task(fixture["task"].id)
    membership_record = next(
        record
        for record in records
        if record["kind"] == SAMPLE_DESIGN_V2_NATIVE_MEMBERSHIP_ARTIFACT_KIND
        and record["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    )
    bundle_record = next(
        record
        for record in records
        if record["kind"] == SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
        and record["origin_tool"] == SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL
    )
    bundle = json.loads(Path(bundle_record["path"]).read_text("utf-8"))
    design = bundle["sample_design"]
    sample_design_ref = {
        "membership_artifact_id": membership_record["id"],
        "expected_membership_artifact_content_hash": membership_record[
            "content_hash"
        ],
        "bundle_artifact_id": bundle_record["id"],
        "expected_bundle_artifact_content_hash": bundle_record["content_hash"],
        "expected_bundle_id": bundle["bundle_id"],
        "expected_sample_design_id": design["sample_design_id"],
        "expected_sample_design_content_hash": design["content_hash"],
    }
    pool_artifact = pool_output["artifacts"][0]
    pool_ref = {
        "artifact_id": pool_artifact["artifact_id"],
        "expected_artifact_content_hash": pool_artifact["content_hash"],
        "expected_pool_id": pool_output["pool_id"],
        "expected_revision": pool_output["revision"],
        "expected_revision_id": pool_output["pool"]["revision_id"],
        "expected_snapshot_hash": pool_output["snapshot_hash"],
    }
    impact = run_measure_strategy_impact_cube(
        {
            "strategy_type": "approval",
            "pool_ref": pool_ref,
            "sample_design_ref": sample_design_ref,
            "partitions": ["development", "validation", "oot"],
            "population": "risk",
            "dimension_bindings": {
                "month_col": "month",
                "group_col": None,
                "segment_col": None,
            },
            "current_strategy_ref": None,
            "economics_inputs": None,
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    validations = {
        partition: run_measure_strategy_pool_validation(
            {
                "strategy_type": "approval",
                "pool_ref": pool_ref,
                "sample_design_ref": sample_design_ref,
                "partition": partition,
                "population": "risk",
                "comparison_mode": "absolute",
            },
            fixture["ctx"],
            fixture["runtime"],
        )
        for partition in ("validation", "oot")
    }
    [entry] = pool_output["entries"]
    candidate_stability = run_measure_candidate_monthly_stability(
        resolve_candidate_monthly_stability_inputs(
            fixture["runtime"],
            task_id=fixture["task"].id,
            user_pointer={
                "source_kind": "pool_entry",
                "strategy_type": "approval",
                "entry_id": entry["entry_id"],
            },
        ),
        fixture["ctx"],
        fixture["runtime"],
    )
    impact_ref = {
        "artifact_id": impact["artifact"]["artifact_id"],
        "expected_artifact_content_hash": impact["artifact"]["content_hash"],
        "expected_cube_id": impact["cube_id"],
        "expected_cube_content_hash": impact["content_hash"],
    }
    pool_stability = run_measure_strategy_pool_stability(
        impact_ref,
        fixture["ctx"],
        fixture["runtime"],
    )
    pool_stability_ref = {
        "artifact_id": pool_stability["artifact"]["artifact_id"],
        "expected_artifact_content_hash": pool_stability["artifact"][
            "content_hash"
        ],
        "expected_stability_id": pool_stability["stability_id"],
        "expected_stability_content_hash": pool_stability["content_hash"],
    }
    message = TaskRepository(fixture["settings"].db_path).add_agent_message(
        fixture["task"].id,
        role="user",
        stage="chat",
        content="生成原生平行双样本准入策略七步报告。",
    )
    run_materialize_project_context(
        {
            "expected_revision": 0,
            "expected_revision_id": None,
            "expected_state_hash": None,
            "user_message_ref": {
                "message_id": message["id"],
                "content_hash": hashlib.sha256(
                    message["content"].encode("utf-8")
                ).hexdigest(),
            },
            "as_of": "2026-07-27",
            "scope": "原生平行双样本准入策略",
            "business_context": {"project.goal": "验证七步报告原生主链"},
            "explicit_unavailable": ["historical_strategy_reviews"],
            "external_report_filenames": [],
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    membership = decode_sample_membership(
        Path(membership_record["path"]).read_bytes()
    )
    return {
        **fixture,
        "bundle": bundle,
        "membership": membership,
        "sample_design_ref": sample_design_ref,
        "pool_output": pool_output,
        "pool_ref": pool_ref,
        "impact": impact,
        "impact_ref": impact_ref,
        "pool_stability": pool_stability,
        "pool_stability_ref": pool_stability_ref,
        "validations": validations,
        "candidate_stability": candidate_stability,
    }


class _ArtifactRepositoryStub:
    def __init__(
        self,
        *records: dict,
        totals: dict[str, int] | None = None,
    ) -> None:
        self.records = tuple(records)
        self.totals = dict(totals or {})

    def list_recent_for_task_kind_with_count(
        self,
        task_id: str,
        kind: str,
        *,
        limit: int,
    ):
        del task_id
        matching = tuple(
            record for record in self.records if record.get("kind") == kind
        )
        return list(matching[:limit]), self.totals.get(kind, len(matching))

    def get_for_task(self, task_id: str, artifact_id: object):
        del task_id
        return next(
            (
                record
                for record in self.records
                if record.get("id") == artifact_id
            ),
            None,
        )

    def list_for_task(self, *_args, **_kwargs):
        raise AssertionError("report selector must not scan full history")


def _window_runtime(
    *records: dict,
    totals: dict[str, int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        task_artifacts=_ArtifactRepositoryStub(*records, totals=totals)
    )


def _pool_validation_ref(
    seed: str,
    *,
    partition: str = "validation",
) -> dict[str, str]:
    return {
        "partition": partition,
        "artifact_id": seed * 64,
        "expected_artifact_content_hash": chr(ord(seed) + 1) * 64,
        "expected_evidence_id": "strategy-pool-validation-" + seed * 24,
        "expected_evidence_content_hash": chr(ord(seed) + 2) * 64,
    }


def test_report_turn_binds_exact_current_sources_and_first_head(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert slots["title"] == "策略迭代评审报告"
    assert slots["status"] == "partial"
    for field in (
        "project_context_ref",
        "sample_design_ref",
        "candidate_pool_ref",
        "pool_impact_ref",
    ):
        assert slots[field] == fixture["request"][field]
    assert slots["impact_cube_ref"] is None
    assert slots["candidate_stability_ref"] is None
    assert slots["voting_candidate_search_ref"] is None
    assert slots["pool_validation_refs"] == []
    assert slots["report_revision"] == 1
    assert slots["previous_report_id"] is None
    assert slots["previous_report_content_hash"] is None
    assert slots["strategy_identity"] is None
    assert slots["model_evidence_ref"] is None
    assert slots["training_evidence_ref"] is None
    assert slots["score_evidence_ref"] is None
    generated_at = datetime.fromisoformat(slots["generated_at"])
    assert generated_at.utcoffset() == timedelta(0)


def test_report_turn_never_scans_full_artifact_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)

    def reject_unbounded_history(*_args, **_kwargs):
        raise AssertionError("Strategy report plan must not call list_for_task")

    monkeypatch.setattr(
        TaskArtifactRepository,
        "list_for_task",
        reject_unbounded_history,
    )

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert slots["sample_design_ref"] == fixture["request"]["sample_design_ref"]
    assert slots["pool_impact_ref"] == fixture["request"]["pool_impact_ref"]


def test_report_turn_passes_exact_selected_voting_search_to_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    search = SimpleNamespace(
        artifact_id="a" * 64,
        artifact_content_hash="b" * 64,
        result={
            "search_id": "voting-search-" + ("1" * 32),
            "content_hash": "c" * 64,
        },
    )
    observed = {}
    original_adapter = (
        turn_handlers.build_strategy_report_bundle_source_inputs
    )

    def capture_preflight(**kwargs):
        observed["search"] = kwargs.pop("voting_candidate_search")
        return original_adapter(**kwargs)

    monkeypatch.setattr(
        turn_handlers,
        "_strategy_report_latest_voting_search_binding",
        lambda *args, **kwargs: search,
    )
    monkeypatch.setattr(
        turn_handlers,
        "build_strategy_report_bundle_source_inputs",
        capture_preflight,
    )

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert observed["search"] is search
    assert slots["voting_candidate_search_ref"] == {
        "artifact_id": search.artifact_id,
        "expected_artifact_content_hash": search.artifact_content_hash,
        "expected_search_id": search.result["search_id"],
        "expected_search_content_hash": search.result["content_hash"],
    }


def test_report_turn_binds_exact_pool_validation_refs_and_preflights_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    validation_ref = _pool_validation_ref("1")
    validation_binding = object()
    observed: dict[str, object] = {}
    original_adapter = (
        turn_handlers.build_strategy_report_bundle_source_inputs
    )

    def select_refs(runtime, *, task_id, candidate_pool, sample_design):
        observed["select"] = (
            runtime,
            task_id,
            candidate_pool,
            sample_design,
        )
        return (validation_ref,)

    def load_refs(
        runtime,
        *,
        task_id,
        refs,
        candidate_pool,
        sample_design,
    ):
        observed["load"] = (
            runtime,
            task_id,
            refs,
            candidate_pool,
            sample_design,
        )
        return (validation_binding,)

    def capture_preflight(**kwargs):
        observed["preflight"] = kwargs.pop("pool_validations")
        return original_adapter(**kwargs)

    monkeypatch.setattr(
        turn_handlers,
        "select_latest_strategy_pool_validation_refs",
        select_refs,
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_strategy_pool_validation_artifacts",
        load_refs,
    )
    monkeypatch.setattr(
        turn_handlers,
        "build_strategy_report_bundle_source_inputs",
        capture_preflight,
    )

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    read_runtime, task_id, candidate_pool, sample_design = observed["select"]
    assert task_id == fixture["task"].id
    assert observed["load"] == (
        read_runtime,
        task_id,
        (validation_ref,),
        candidate_pool,
        sample_design,
    )
    assert observed["preflight"] == (validation_binding,)
    assert slots["pool_validation_refs"] == [validation_ref]


def test_report_turn_pool_validation_selection_is_frozen_in_each_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    validation_ref = _pool_validation_ref("1")
    later_oot_ref = _pool_validation_ref("4", partition="oot")
    selected = [(validation_ref,), (validation_ref, later_oot_ref)]

    monkeypatch.setattr(
        turn_handlers,
        "select_latest_strategy_pool_validation_refs",
        lambda *args, **kwargs: selected.pop(0),
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_strategy_pool_validation_artifacts",
        lambda *args, **kwargs: (),
    )

    planned = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )
    frozen = deepcopy(planned)
    later = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert planned == frozen
    assert planned["pool_validation_refs"] == [validation_ref]
    assert later["pool_validation_refs"] == [validation_ref, later_oot_ref]


def test_report_turn_pool_validation_adapter_incompatibility_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    validation_ref = _pool_validation_ref("1")
    validation_binding = object()

    monkeypatch.setattr(
        turn_handlers,
        "select_latest_strategy_pool_validation_refs",
        lambda *args, **kwargs: (validation_ref,),
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_strategy_pool_validation_artifacts",
        lambda *args, **kwargs: (validation_binding,),
    )

    def reject_incompatible(**kwargs):
        assert kwargs["pool_validations"] == (validation_binding,)
        raise StrategyError("incompatible Pool validation evidence")

    monkeypatch.setattr(
        turn_handlers,
        "build_strategy_report_bundle_source_inputs",
        reject_incompatible,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_bundle_v2_plan_slots(
            _runtime(fixture),
            fixture["task"],
            _draft(),
            source_message={"content": "请生成当前审批策略迭代评审报告。"},
        )

    assert raised.value.code == "strategy_report_bundle_v2_source_incompatible"


def test_report_turn_selects_latest_compatible_candidate_stability(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    stability = _attach_candidate_stability(fixture)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert slots["candidate_stability_ref"] == {
        "artifact_id": stability["artifacts"][0]["artifact_id"],
        "expected_artifact_content_hash": stability["artifacts"][0][
            "content_hash"
        ],
        "expected_stability_id": stability["stability_id"],
        "expected_stability_content_hash": stability["content_hash"],
    }


def test_report_turn_corrupt_candidate_stability_fails_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    stability = _attach_candidate_stability(fixture)
    record = fixture["runtime"].task_artifacts.get_for_task(
        fixture["task"].id,
        stability["artifacts"][0]["artifact_id"],
    )
    assert record is not None
    Path(record["path"]).write_bytes(b"{}")

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_bundle_v2_plan_slots(
            _runtime(fixture),
            fixture["task"],
            _draft(),
            source_message={"content": "请生成当前审批策略迭代评审报告。"},
        )

    assert (
        raised.value.code
        == "strategy_report_bundle_v2_candidate_stability_invalid"
    )


def test_report_turn_reloads_next_exact_report_head(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    first = _run(fixture)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(status="final"),
        source_message={"content": "请再生成当前审批策略评审报告，状态 final。"},
    )

    assert slots["report_revision"] == 2
    assert slots["previous_report_id"] == first["report_id"]
    assert slots["previous_report_content_hash"] == first["content_hash"]


def test_report_planned_refs_do_not_rebind_after_head_drift(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    planned = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略评审报告。"},
    )
    frozen = deepcopy(planned)
    _run(fixture)

    assert planned == frozen
    with pytest.raises(StrategyError):
        run_build_strategy_report_bundle_v2(
            planned,
            fixture["ctx"],
            fixture["runtime"],
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("生成审批策略评审报告", "approval"),
        ("生成拒绝策略评审报告", "reject"),
        ("生成额度策略评审报告", "limit"),
        ("生成定价策略评审报告", "pricing"),
        ("生成分群策略评审报告", "segmentation"),
        ("generate approval strategy review report", "approval"),
        ("generate reject strategy review report", "reject"),
        ("generate limit strategy review report", "limit"),
        ("generate pricing strategy review report", "pricing"),
        ("generate segmentation strategy review report", "segmentation"),
    ],
)
def test_report_turn_uses_only_explicit_pool_type_selection(
    message: str,
    expected: str,
) -> None:
    assert _strategy_report_requested_pool_type({"content": message}) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "生成审批策略评审报告，标题为《拒绝 Pool 复盘》",
            "approval",
        ),
        (
            "昨天的拒绝 Pool 报告已归档，现在生成审批策略评审报告",
            "approval",
        ),
        (
            "不要使用拒绝 Pool，请生成审批策略评审报告",
            "approval",
        ),
        (
            "不是拒绝 Pool，而是审批 Pool，请生成当前策略评审报告",
            "approval",
        ),
        (
            "请生成当前策略评审报告，策略类型为 reject",
            "reject",
        ),
        (
            "生成当前策略评审报告，标题为《拒绝 Pool 复盘》",
            None,
        ),
    ],
)
def test_report_turn_ignores_title_history_and_negated_pool_type_mentions(
    message: str,
    expected: str | None,
) -> None:
    assert _strategy_report_requested_pool_type({"content": message}) == expected


def test_report_turn_rejects_only_negated_pool_type_selection() -> None:
    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_requested_pool_type(
            {"content": "请生成当前策略评审报告，不要使用拒绝 Pool。"}
        )

    assert raised.value.code == "strategy_report_bundle_v2_pool_type_required"


def test_report_turn_clarifies_when_both_nonempty_pool_types_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DualPoolRepository:
        def __init__(self, db_path: Path) -> None:
            pass

        def get_current(self, task_id: str, strategy_type: str) -> dict:
            return {"entries": [{"entry_id": strategy_type}]}

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        _DualPoolRepository,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_current_pool_binding(
            SimpleNamespace(settings=SimpleNamespace(db_path=tmp_path / "db.sqlite")),
            task_id="task-1",
            requested_type=None,
        )

    assert raised.value.code == "strategy_report_bundle_v2_pool_type_required"


def test_report_turn_explicit_pool_type_ignores_unrelated_corrupt_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    selected = {
        "entries": [{"entry_id": "approval-entry"}],
        "revision": 3,
    }
    binding = object()

    class _Repository:
        def __init__(self, db_path: Path) -> None:
            pass

        def get_current(self, task_id: str, strategy_type: str):
            calls.append(strategy_type)
            if strategy_type == "approval":
                return selected
            raise RuntimeError(f"unrelated corrupt {strategy_type} pool")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        _Repository,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.strategy_pool_snapshot_hash",
        lambda pool: "a" * 64,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_candidate_pool_artifact",
        lambda *args, **kwargs: binding,
    )

    result = _strategy_report_current_pool_binding(
        SimpleNamespace(
            settings=SimpleNamespace(db_path=tmp_path / "db.sqlite")
        ),
        task_id="task-1",
        requested_type="approval",
    )

    assert result is binding
    assert calls == ["approval"]


def test_report_turn_clarifies_for_missing_context_sample_pool_or_impact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    runtime = _runtime(fixture)
    read_runtime = _strategy_report_read_runtime(runtime)

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_current_strategy_project_context_artifact",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_context:
        _strategy_report_bundle_v2_plan_slots(
            runtime,
            fixture["task"],
            _draft(),
            source_message={"content": "生成审批策略评审报告"},
        )
    assert missing_context.value.code == (
        "strategy_report_bundle_v2_project_context_required"
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_sample:
        _strategy_report_latest_sample_binding(
            _window_runtime(),
            task_id=fixture["task"].id,
        )
    assert missing_sample.value.code == "strategy_report_bundle_v2_sample_required"

    class _EmptyPoolRepository:
        def __init__(self, db_path: Path) -> None:
            pass

        def get_current(self, task_id: str, strategy_type: str):
            return None

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyCandidatePoolRepository",
        _EmptyPoolRepository,
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_pool:
        _strategy_report_current_pool_binding(
            read_runtime,
            task_id=fixture["task"].id,
            requested_type=None,
        )
    assert missing_pool.value.code == "strategy_report_bundle_v2_pool_required"

    monkeypatch.undo()
    pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=fixture["task"].id,
        requested_type="approval",
    )
    with pytest.raises(_StrategyV2EvidenceSetupError) as missing_impact:
        _strategy_report_latest_pool_impact_binding(
            _window_runtime(),
            task_id=fixture["task"].id,
            pool=pool,
        )
    assert missing_impact.value.code == (
        "strategy_report_bundle_v2_pool_impact_required"
    )


def test_report_turn_corrupt_latest_native_sample_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kind = SAMPLE_DESIGN_V2_BUNDLE_ARTIFACT_KIND
    newest = {
        "kind": kind,
        "origin_tool": SAMPLE_DESIGN_V2_NATIVE_ORIGIN_TOOL,
        "id": "1" * 64,
        "content_hash": "2" * 64,
        "provenance": {
            "membership_artifact_id": "3" * 64,
            "membership_artifact_content_hash": "4" * 64,
            "bundle_id": "strategy-sample-design-bundle-" + "5" * 24,
            "sample_design_id": "strategy-sample-design-" + "6" * 24,
            "sample_design_content_hash": "7" * 64,
        },
    }
    older = {
        "kind": kind,
        "origin_tool": turn_handlers.SAMPLE_DESIGN_V2_ORIGIN_TOOL,
        "id": "8" * 64,
        "content_hash": "9" * 64,
        "provenance": {
            "membership_artifact_id": "a" * 64,
            "membership_artifact_content_hash": "b" * 64,
            "bundle_id": "strategy-sample-design-bundle-" + "c" * 24,
            "sample_design_id": "strategy-sample-design-" + "d" * 24,
            "sample_design_content_hash": "e" * 64,
        },
    }
    calls: list[str] = []

    def reject_latest(*_args, **kwargs):
        calls.append(kwargs["bundle_artifact_id"])
        raise StrategyError("native membership bytes changed")

    monkeypatch.setattr(
        turn_handlers,
        "load_any_strategy_sample_design_v2_artifacts",
        reject_latest,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_latest_sample_binding(
            _window_runtime(
                newest,
                older,
                totals={kind: 2},
            ),
            task_id="task-native-report",
        )

    assert raised.value.code == "strategy_report_bundle_v2_sample_invalid"
    assert calls == [newest["id"]]


def test_report_turn_authenticates_newest_pool_impact_before_binding_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SimpleNamespace(
        artifact_id="1" * 64,
        artifact_content_hash="2" * 64,
        pool={
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 4,
            "snapshot_hash": "5" * 64,
        },
    )
    newest = {
        "kind": turn_handlers.POOL_IMPACT_ARTIFACT_KIND,
        "id": "6" * 64,
        "content_hash": "7" * 64,
        "provenance": {
            "pool_id": "strategy-pool-" + "8" * 32,
            "pool_revision": 9,
            "pool_snapshot_hash": "9" * 64,
            "assessment_id": "assessment-newest",
            "assessment_content_hash": "a" * 64,
        },
    }
    older = {
        "kind": turn_handlers.POOL_IMPACT_ARTIFACT_KIND,
        "id": "b" * 64,
        "content_hash": "c" * 64,
        "provenance": {
            "pool_id": pool.pool["pool_id"],
            "pool_revision": pool.pool["revision"],
            "pool_snapshot_hash": pool.pool["snapshot_hash"],
            "assessment_id": "assessment-older",
            "assessment_content_hash": "d" * 64,
        },
    }
    calls: list[str] = []

    def authenticate(*_args, **kwargs):
        artifact_id = kwargs["artifact_id"]
        calls.append(artifact_id)
        if artifact_id == newest["id"]:
            raise StrategyError("registry provenance drift")
        return SimpleNamespace(
            stage="development_backtest",
            pool=SimpleNamespace(
                artifact_id=pool.artifact_id,
                artifact_content_hash=pool.artifact_content_hash,
            ),
        )

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_historical_strategy_pool_impact_artifact",
        authenticate,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_latest_pool_impact_binding(
            _window_runtime(newest, older),
            task_id="task-1",
            pool=pool,
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_pool_impact_invalid"
    )
    assert calls == [newest["id"]]


def test_report_turn_skips_authenticated_unrelated_pool_impact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SimpleNamespace(
        artifact_id="1" * 64,
        artifact_content_hash="2" * 64,
        pool={
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 4,
            "snapshot_hash": "5" * 64,
        },
    )
    newest = {
        "kind": turn_handlers.POOL_IMPACT_ARTIFACT_KIND,
        "id": "6" * 64,
        "content_hash": "7" * 64,
        "provenance": {
            "assessment_id": "assessment-newest",
            "assessment_content_hash": "8" * 64,
        },
    }
    older = {
        "kind": turn_handlers.POOL_IMPACT_ARTIFACT_KIND,
        "id": "9" * 64,
        "content_hash": "a" * 64,
        "provenance": {
            "assessment_id": "assessment-older",
            "assessment_content_hash": "b" * 64,
        },
    }
    unrelated = SimpleNamespace(
        stage="development_backtest",
        pool=SimpleNamespace(
            artifact_id="c" * 64,
            artifact_content_hash="d" * 64,
            pool={
                "pool_id": "strategy-pool-" + "e" * 32,
                "revision": 1,
                "snapshot_hash": "f" * 64,
            },
        ),
    )
    exact = SimpleNamespace(
        stage="development_backtest",
        pool=SimpleNamespace(
            artifact_id=pool.artifact_id,
            artifact_content_hash=pool.artifact_content_hash,
            pool=pool.pool,
        ),
    )
    calls: list[str] = []

    def authenticate(*_args, **kwargs):
        artifact_id = kwargs["artifact_id"]
        calls.append(artifact_id)
        return unrelated if artifact_id == newest["id"] else exact

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_historical_strategy_pool_impact_artifact",
        authenticate,
    )

    selected = _strategy_report_latest_pool_impact_binding(
        _window_runtime(newest, older),
        task_id="task-1",
        pool=pool,
    )

    assert selected is exact
    assert calls == [newest["id"], older["id"]]


def test_report_turn_prefers_latest_exact_impact_cube(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)

    slots = _strategy_report_bundle_v2_plan_slots(
        _runtime(fixture),
        fixture["task"],
        _draft(),
        source_message={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert slots["impact_cube_ref"] == fixture["request"]["impact_cube_ref"]
    assert slots["pool_impact_ref"] is None


def test_report_turn_selects_latest_exact_cube_not_latest_other_cube(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SimpleNamespace(
        artifact_id="1" * 64,
        artifact_content_hash="2" * 64,
        pool={
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 4,
            "revision_id": "strategy-pool-revision-" + "5" * 32,
            "snapshot_hash": "6" * 64,
        },
    )
    sample_ref = {
        "membership_artifact_id": "7" * 64,
        "expected_membership_artifact_content_hash": "8" * 64,
        "bundle_artifact_id": "9" * 64,
        "expected_bundle_artifact_content_hash": "a" * 64,
        "expected_bundle_id": "strategy-sample-design-bundle-" + "b" * 24,
        "expected_sample_design_id": "strategy-sample-design-" + "c" * 24,
        "expected_sample_design_content_hash": "d" * 64,
    }
    pool_ref = {
        "artifact_id": pool.artifact_id,
        "expected_artifact_content_hash": pool.artifact_content_hash,
        "expected_pool_id": pool.pool["pool_id"],
        "expected_revision": pool.pool["revision"],
        "expected_revision_id": pool.pool["revision_id"],
        "expected_snapshot_hash": pool.pool["snapshot_hash"],
    }

    def _record(seed: str, *, exact: bool) -> dict:
        return {
            "kind": "strategy_impact_cube_json",
            "id": seed * 64,
            "content_hash": seed.upper().lower() * 64,
            "provenance": {
                "pool_ref": (
                    pool_ref
                    if exact
                    else {**pool_ref, "expected_revision": 99}
                ),
                "sample_design_ref": sample_ref,
                "cube_id": "strategy-impact-cube-" + seed * 24,
                "cube_content_hash": seed * 64,
            },
        }

    older_exact = _record("a", exact=True)
    latest_other = _record("c", exact=False)
    calls: list[dict] = []

    def _binding(record: dict) -> SimpleNamespace:
        record_pool_ref = record["provenance"]["pool_ref"]
        record_sample_ref = record["provenance"]["sample_design_ref"]
        return SimpleNamespace(
            artifact_id=record["id"],
            artifact_content_hash=record["content_hash"],
            cube={
                "cube_id": record["provenance"]["cube_id"],
                "content_hash": record["provenance"]["cube_content_hash"],
                "identity": {
                    "pool_id": record_pool_ref["expected_pool_id"],
                    "revision": record_pool_ref["expected_revision"],
                    "revision_id": record_pool_ref[
                        "expected_revision_id"
                    ],
                    "snapshot_hash": record_pool_ref[
                        "expected_snapshot_hash"
                    ],
                },
                "source_bindings": {
                    "pool_artifact": {
                        "artifact_id": record_pool_ref["artifact_id"],
                        "artifact_content_hash": record_pool_ref[
                            "expected_artifact_content_hash"
                        ],
                    },
                    "sample_design_v2": {
                        "membership_artifact_id": record_sample_ref[
                            "membership_artifact_id"
                        ],
                        "membership_artifact_content_hash": (
                            record_sample_ref[
                                "expected_membership_artifact_content_hash"
                            ]
                        ),
                        "bundle_artifact_id": record_sample_ref[
                            "bundle_artifact_id"
                        ],
                        "bundle_artifact_content_hash": record_sample_ref[
                            "expected_bundle_artifact_content_hash"
                        ],
                        "bundle_id": record_sample_ref[
                            "expected_bundle_id"
                        ],
                        "sample_design_id": record_sample_ref[
                            "expected_sample_design_id"
                        ],
                        "sample_design_content_hash": record_sample_ref[
                            "expected_sample_design_content_hash"
                        ],
                    },
                },
            },
        )

    bindings = {
        record["id"]: _binding(record)
        for record in (older_exact, latest_other)
    }

    def _load(*args, **kwargs):
        calls.append(kwargs)
        return bindings[kwargs["artifact_id"]]

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_strategy_impact_cube_artifact",
        _load,
    )

    binding = _strategy_report_latest_impact_cube_binding(
        _window_runtime(latest_other, older_exact),
        task_id="task-1",
        pool=pool,
        sample_ref=sample_ref,
    )

    assert binding is bindings[older_exact["id"]]
    assert calls == [
        {
            "task_id": "task-1",
            "artifact_id": latest_other["id"],
            "expected_artifact_content_hash": latest_other["content_hash"],
            "expected_cube_id": latest_other["provenance"]["cube_id"],
            "expected_cube_content_hash": latest_other["provenance"][
                "cube_content_hash"
            ],
        },
        {
            "task_id": "task-1",
            "artifact_id": older_exact["id"],
            "expected_artifact_content_hash": older_exact["content_hash"],
            "expected_cube_id": older_exact["provenance"]["cube_id"],
            "expected_cube_content_hash": older_exact["provenance"][
                "cube_content_hash"
            ],
        }
    ]


def test_report_turn_impact_cube_selection_window_exhaustion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = SimpleNamespace(
        artifact_id="1" * 64,
        artifact_content_hash="2" * 64,
        pool={
            "pool_id": "strategy-pool-" + "3" * 32,
            "revision": 4,
            "revision_id": "strategy-pool-revision-" + "5" * 32,
            "snapshot_hash": "6" * 64,
        },
    )
    sample_ref = {
        "membership_artifact_id": "7" * 64,
        "expected_membership_artifact_content_hash": "8" * 64,
        "bundle_artifact_id": "9" * 64,
        "expected_bundle_artifact_content_hash": "a" * 64,
        "expected_bundle_id": "strategy-sample-design-bundle-" + "b" * 24,
        "expected_sample_design_id": "strategy-sample-design-" + "c" * 24,
        "expected_sample_design_content_hash": "d" * 64,
    }
    replay_limit = turn_handlers._STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
    records = tuple(
        {
            "kind": "strategy_impact_cube_json",
            "id": f"{index:064x}",
            "content_hash": f"{index + replay_limit:064x}",
            "provenance": {
                "cube_id": "strategy-impact-cube-" + f"{index:024x}",
                "cube_content_hash": f"{index + replay_limit * 2:064x}",
            },
        }
        for index in range(replay_limit)
    )

    class _ArtifactWindow:
        def list_recent_for_task_kind_with_count(
            self,
            task_id: str,
            kind: str,
            *,
            limit: int,
        ):
            assert (task_id, kind, limit) == (
                "task-1",
                "strategy_impact_cube_json",
                replay_limit,
            )
            return list(records), replay_limit + 1

    unrelated_pool_ref = {
        "artifact_id": "e" * 64,
        "artifact_content_hash": "f" * 64,
    }

    def load_unrelated(*_args, **kwargs):
        return SimpleNamespace(
            artifact_id=kwargs["artifact_id"],
            artifact_content_hash=kwargs[
                "expected_artifact_content_hash"
            ],
            cube={
                "identity": {
                    "pool_id": "strategy-pool-" + "0" * 32,
                    "revision": 99,
                    "revision_id": "strategy-pool-revision-" + "0" * 32,
                    "snapshot_hash": "0" * 64,
                },
                "source_bindings": {
                    "pool_artifact": unrelated_pool_ref,
                    "sample_design_v2": {
                        "membership_artifact_id": sample_ref[
                            "membership_artifact_id"
                        ],
                        "membership_artifact_content_hash": sample_ref[
                            "expected_membership_artifact_content_hash"
                        ],
                        "bundle_artifact_id": sample_ref["bundle_artifact_id"],
                        "bundle_artifact_content_hash": sample_ref[
                            "expected_bundle_artifact_content_hash"
                        ],
                        "bundle_id": sample_ref["expected_bundle_id"],
                        "sample_design_id": sample_ref[
                            "expected_sample_design_id"
                        ],
                        "sample_design_content_hash": sample_ref[
                            "expected_sample_design_content_hash"
                        ],
                    },
                },
            },
        )

    monkeypatch.setattr(
        turn_handlers,
        "load_strategy_impact_cube_artifact",
        load_unrelated,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_latest_impact_cube_binding(
            SimpleNamespace(task_artifacts=_ArtifactWindow()),
            task_id="task-1",
            pool=pool,
            sample_ref=sample_ref,
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_impact_cube_selection_window_exhausted"
    )


def test_nonlegacy_report_type_without_exact_cube_never_uses_pool_impact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    read_runtime = _strategy_report_read_runtime(_runtime(fixture))
    approval_pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=fixture["task"].id,
        requested_type="approval",
    )
    limit_pool = SimpleNamespace(
        strategy_type="limit",
        pool=approval_pool.pool,
        artifact_id=approval_pool.artifact_id,
        artifact_content_hash=approval_pool.artifact_content_hash,
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers._strategy_report_current_pool_binding",
        lambda *args, **kwargs: limit_pool,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_bundle_v2_plan_slots(
            _runtime(fixture),
            fixture["task"],
            _draft(),
            source_message={"content": "请生成当前额度策略迭代评审报告。"},
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_impact_cube_required"
    )


def test_report_turn_corrupt_exact_impact_cube_fails_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    newer_request = deepcopy(fixture["impact_request"])
    newer_request["partitions"] = ["development"]
    newer = run_measure_strategy_impact_cube(
        newer_request,
        fixture["ctx"],
        fixture["runtime"],
    )
    read_runtime = _strategy_report_read_runtime(_runtime(fixture))
    cube_record = read_runtime.task_artifacts.get_for_task(
        fixture["task"].id,
        newer["artifact"]["artifact_id"],
    )
    _, cube_total = (
        read_runtime.task_artifacts.list_recent_for_task_kind_with_count(
            fixture["task"].id,
            "strategy_impact_cube_json",
            limit=64,
        )
    )
    pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=fixture["task"].id,
        requested_type="approval",
    )
    assert cube_record is not None
    assert cube_total == 2
    Path(cube_record["path"]).write_text("{}", encoding="utf-8")

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_latest_impact_cube_binding(
            read_runtime,
            task_id=fixture["task"].id,
            pool=pool,
            sample_ref=fixture["request"]["sample_design_ref"],
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_impact_cube_invalid"
    )


def test_report_turn_disguised_latest_exact_cube_fails_without_fallback(
    tmp_path: Path,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    newer_request = deepcopy(fixture["impact_request"])
    newer_request["partitions"] = ["development"]
    newer = run_measure_strategy_impact_cube(
        newer_request,
        fixture["ctx"],
        fixture["runtime"],
    )
    with sqlite3.connect(fixture["settings"].db_path) as conn:
        row = conn.execute(
            "SELECT provenance_json FROM task_artifacts WHERE id = ?",
            (newer["artifact"]["artifact_id"],),
        ).fetchone()
        assert row is not None
        provenance = json.loads(row[0])
        provenance["pool_ref"]["expected_revision"] += 1
        conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
        conn.execute(
            "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
            (
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                newer["artifact"]["artifact_id"],
            ),
        )
        conn.commit()

    read_runtime = _strategy_report_read_runtime(_runtime(fixture))
    pool = _strategy_report_current_pool_binding(
        read_runtime,
        task_id=fixture["task"].id,
        requested_type="approval",
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_latest_impact_cube_binding(
            read_runtime,
            task_id=fixture["task"].id,
            pool=pool,
            sample_ref=fixture["request"]["sample_design_ref"],
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_impact_cube_invalid"
    )


def test_report_turn_latest_optional_evidence_is_compatible_or_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_ref = {
        "membership_artifact_id": "1" * 64,
        "expected_membership_artifact_content_hash": "2" * 64,
        "bundle_artifact_id": "3" * 64,
        "expected_bundle_artifact_content_hash": "4" * 64,
        "expected_bundle_id": "strategy-sample-design-bundle-" + "5" * 24,
        "expected_sample_design_id": "strategy-sample-design-" + "6" * 24,
        "expected_sample_design_content_hash": "7" * 64,
    }

    def _sample_binding(ref: dict) -> SimpleNamespace:
        return SimpleNamespace(
            membership_artifact_id=ref["membership_artifact_id"],
            membership_artifact_content_hash=ref[
                "expected_membership_artifact_content_hash"
            ],
            bundle_artifact_id=ref["bundle_artifact_id"],
            bundle_artifact_content_hash=ref[
                "expected_bundle_artifact_content_hash"
            ],
            bundle={
                "bundle_id": ref["expected_bundle_id"],
                "sample_design": {
                    "sample_design_id": ref["expected_sample_design_id"],
                    "content_hash": ref[
                        "expected_sample_design_content_hash"
                    ],
                },
            },
        )

    model_binding = SimpleNamespace(
        artifact_id="8" * 64,
        artifact_content_hash="9" * 64,
        bundle={"bundle_id": "model-bundle", "content_hash": "a" * 64},
        sample_design_binding=_sample_binding(sample_ref),
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_strategy_model_evidence_v2_artifact",
        lambda *args, **kwargs: model_binding,
    )
    records = (
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": model_binding.artifact_id,
            "content_hash": model_binding.artifact_content_hash,
            "provenance": {
                "bundle_id": "model-bundle",
                "bundle_content_hash": "a" * 64,
                "sample_design_ref": sample_ref,
            },
        },
    )

    _, reference = _strategy_report_optional_model_evidence(
        _window_runtime(*records),
        task_id="task-1",
        sample_ref=sample_ref,
    )
    assert reference == {
        "artifact_id": "8" * 64,
        "expected_artifact_content_hash": "9" * 64,
        "expected_bundle_id": "model-bundle",
        "expected_bundle_content_hash": "a" * 64,
    }

    incompatible_ref = {**sample_ref, "expected_bundle_id": "other-bundle"}
    model_binding.sample_design_binding = _sample_binding(incompatible_ref)
    binding, reference = _strategy_report_optional_model_evidence(
        _window_runtime(*records),
        task_id="task-1",
        sample_ref=sample_ref,
    )
    assert binding is None
    assert reference is None


def test_report_turn_compatible_score_can_supply_its_own_training_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_ref = {"expected_sample_design_id": "sample-1"}
    score_binding = SimpleNamespace(
        training=object(),
        evidence_record={"id": "a" * 64, "content_hash": "b" * 64},
        vector_record={"id": "c" * 64, "content_hash": "d" * 64},
    )
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_model_score_evidence_artifacts",
        lambda *args, **kwargs: score_binding,
    )
    score_training_ref = {
        "sample_design_ref": sample_ref,
        "expected_experiment_id": "experiment-1",
    }
    monkeypatch.setattr(
        "marvis.agent.turn_handlers.build_training_evidence_ref",
        lambda binding: score_training_ref,
    )
    records = (
        {
            "kind": "model_score_evidence_json",
            "id": "a" * 64,
            "content_hash": "b" * 64,
            "provenance": {
                "score_vector_artifact_id": "c" * 64,
                "score_vector_artifact_content_hash": "d" * 64,
            },
        },
    )

    binding, reference = _strategy_report_optional_score_evidence(
        _window_runtime(*records),
        task_id="task-1",
        sample_ref=sample_ref,
        training_ref=None,
    )

    assert binding is score_binding
    assert reference == {
        "evidence_artifact_id": "a" * 64,
        "expected_evidence_artifact_content_hash": "b" * 64,
        "score_vector_artifact_id": "c" * 64,
        "expected_score_vector_artifact_content_hash": "d" * 64,
    }


def test_report_turn_corrupt_latest_same_kind_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": "1" * 64,
            "content_hash": "2" * 64,
            "provenance": {"older": "otherwise-valid"},
        },
        {
            "kind": MODEL_EVIDENCE_V2_ARTIFACT_KIND,
            "id": "3" * 64,
            "content_hash": "4" * 64,
            "provenance": None,
        },
    )

    def _must_not_fallback(*args, **kwargs):
        pytest.fail("corrupt newest same-kind evidence must stop selection")

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.load_strategy_model_evidence_v2_artifact",
        _must_not_fallback,
    )

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        _strategy_report_optional_model_evidence(
            _window_runtime(records[1], records[0]),
            task_id="task-1",
            sample_ref={},
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_optional_evidence_invalid"
    )


def test_report_turn_strategy_identity_requires_one_exact_same_type_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StrategyRepository:
        matches = [
            {
                "id": "strategy-1",
                "task_id": "task-1",
                "strategy_type": "approval",
                "version": 3,
            }
        ]

        def __init__(self, db_path: Path) -> None:
            pass

        def list_meta_for_task(self, task_id: str) -> list[dict]:
            return list(self.matches)

        def get_strategy(self, strategy_id: str):
            return SimpleNamespace(
                spec={"schema_version": "strategy.v2"},
                strategy_type="approval",
            )

        def get_strategy_spec_hash(self, strategy_id: str) -> str:
            return "a" * 64

    monkeypatch.setattr(
        "marvis.agent.turn_handlers.StrategyRepository",
        _StrategyRepository,
    )
    runtime = SimpleNamespace(
        settings=SimpleNamespace(db_path=tmp_path / "db.sqlite")
    )

    assert _strategy_report_identity(
        runtime,
        task_id="task-1",
        strategy_type="approval",
    ) == {
        "strategy_id": "strategy-1",
        "strategy_version": "3",
        "strategy_type": "approval",
    }
    _StrategyRepository.matches.append(
        {
            "id": "strategy-2",
            "task_id": "task-1",
            "strategy_type": "approval",
            "version": 1,
        }
    )
    assert (
        _strategy_report_identity(
            runtime,
            task_id="task-1",
            strategy_type="approval",
        )
        is None
    )


def test_report_command_autostarts_exact_one_step_without_dataset_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _ReportLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    def _unexpected_preview(*args, **kwargs):
        pytest.fail("report workflow must not require an active dataset preview")

    for name in (
        "_strategy_dataset_preview",
        "_strategy_sample_design_dataset_preview",
        "_strategy_pool_impact_dataset_preview",
    ):
        monkeypatch.setattr(f"marvis.agent.turn_handlers.{name}", _unexpected_preview)

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"]
    assert len(plans) == 1, response.json()
    assert plans[0]["template_id"] == "strategy_report_bundle_v2"
    assert plans[0]["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plans[0]["id"])
    assert len(stored.steps) == 1
    step = stored.steps[0]
    assert step.tool_ref == ToolRef("strategy", "build_report_bundle_v2")
    assert step.needs_confirmation is False
    assert step.inputs["title"] == "策略迭代评审报告"
    assert step.inputs["status"] == "partial"
    assert "previous_report_id" not in step.inputs
    assert "previous_report_content_hash" not in step.inputs
    assert step.inputs["project_context_ref"] == fixture["request"][
        "project_context_ref"
    ]
    output = client.app.state.plan_repo.load_step_output(step.id)
    validate_build_strategy_report_bundle_v2_tool_output(output)
    rendered_messages = "\n".join(
        str(message.get("content") or "")
        for message in response.json()["messages"]
        if message.get("role") == "assistant"
    )
    assert output["report_id"] in rendered_messages
    assert "未创建策略、未采纳、未部署或上线" in rendered_messages
    assert rendered_messages.count("](/api/tasks/") == 4
    assert len(llm.calls) == 1


def test_report_command_autostarts_with_exact_impact_cube(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_impact_cube_report(tmp_path)
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _ReportLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert response.status_code == 202, response.text
    plan = client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"][0]
    assert plan["template_id"] == "strategy_report_bundle_v2"
    assert plan["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    step = stored.steps[0]
    assert step.inputs["impact_cube_ref"] == fixture["request"][
        "impact_cube_ref"
    ]
    assert "pool_impact_ref" not in step.inputs
    output = client.app.state.plan_repo.load_step_output(step.id)
    validate_build_strategy_report_bundle_v2_tool_output(output)
    assert output["bundle"]["strategy_type"] == "approval"
    assert len(llm.calls) == 1


def test_native_parallel_bad_zero_report_turn_publishes_exact_seven_step_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup_native_parallel_report(tmp_path)
    masks = fixture["membership"]["masks"]
    approval = (
        masks["approval/development"]
        | masks["approval/validation"]
        | masks["approval/oot"]
    )
    risk = (
        masks["risk/development"]
        | masks["risk/validation"]
        | masks["risk/oot"]
    )
    assert not (approval & risk).any()
    assert fixture["bundle"]["sample_design"]["relationship"] == (
        "parallel_time_cohorts"
    )
    assert fixture["bundle"]["sample_design"]["target_selector"]["bad_value"] == 0
    assert fixture["impact"]["cube"]["source_bindings"]["target"]["bad_value"] == 0
    assert {
        output["evidence"]["source_bindings"]["target"]["bad_value"]
        for output in fixture["validations"].values()
    } == {0}
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _ReportLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "请生成当前审批策略迭代评审报告。"},
    )

    assert response.status_code == 202, response.text
    plans = client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"]
    assert len(plans) == 1, response.json()
    [plan] = plans
    assert plan["template_id"] == "strategy_report_bundle_v2"
    assert plan["status"] == "done"
    stored = client.app.state.plan_repo.load_plan(plan["id"])
    [step] = stored.steps
    assert step.inputs["sample_design_ref"] == fixture["sample_design_ref"]
    assert step.inputs["impact_cube_ref"] == fixture["impact_ref"]
    assert {
        item["partition"] for item in step.inputs["pool_validation_refs"]
    } == {"validation", "oot"}
    candidate_artifact = fixture["candidate_stability"]["artifacts"][0]
    assert step.inputs["candidate_stability_ref"] == {
        "artifact_id": candidate_artifact["artifact_id"],
        "expected_artifact_content_hash": candidate_artifact["content_hash"],
        "expected_stability_id": fixture["candidate_stability"]["stability_id"],
        "expected_stability_content_hash": fixture["candidate_stability"][
            "content_hash"
        ],
    }
    assert step.inputs["pool_stability_ref"] == fixture["pool_stability_ref"]

    output = client.app.state.plan_repo.load_step_output(step.id)
    validate_build_strategy_report_bundle_v2_tool_output(output)
    bundle = output["bundle"]
    assert [section["key"] for section in bundle["sections"]] == [
        "current_project",
        "historical_versions",
        "sample_design",
        "univariate_and_models",
        "candidate_combinations",
        "impact_assessment",
        "final_document",
    ]
    sample_summary = {
        item["field_id"]: item["field"]["value"]
        for item in bundle["sections"][2]["summary_fields"]
    }
    counts = fixture["membership"]["header"]["counts"]
    assert sample_summary["sample_relationship"] == "parallel_time_cohorts"
    assert sample_summary["analysis_universe_count"] == counts[
        "analysis_universe"
    ]
    assert sample_summary["approval_population_count"] == counts["approval"][
        "total"
    ]
    assert sample_summary["risk_population_count"] == counts["risk"]["total"]
    source_kinds = {
        ref["kind"] for ref in bundle["strategy_artifact_refs"]
    }
    assert {
        "strategy_impact",
        "strategy_validation",
        "backtest",
    } <= source_kinds
    assert bundle["strategy_id"] is None
    assert len(llm.calls) == 1


def test_canonical_stored_strategy_report_stays_on_legacy_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _strategy_task(client, tmp_path)
    strategy_id = _saved_strategy(client, task_id)
    llm = _StoredStrategyLLM(
        {
            "request_kind": "strategy_lifecycle",
            "operation": "report",
            "strategy_type": "approval",
            "strategy_id": strategy_id,
        }
    )
    _install_stored_strategy_llm(monkeypatch, llm)

    response = client.post(
        f"/api/tasks/{task_id}/agent/messages",
        json={
            "content": (
                f"请为 strategy_id={strategy_id} 生成审批策略评审报告。"
            )
        },
    )

    assert response.status_code == 202, response.text
    plans = _strategy_request_plans(client, task_id)
    assert [plan["template_id"] for plan in plans] == [
        "stored_strategy_report"
    ]
    assert plans[0]["status"] == "done"
    assert len(llm.calls) == 1


def test_viewing_past_report_does_not_create_a_new_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _setup(tmp_path)
    first = _run(fixture)
    client = TestClient(create_app(fixture["settings"].workspace))
    llm = _ReportLLM()
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )

    response = client.post(
        f"/api/tasks/{fixture['task'].id}/agent/messages",
        json={"content": "现在查看昨天生成的策略评审报告。"},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "clarification_required"
    assert response.json()["code"] == (
        "strategy_report_bundle_v2_positive_command_required"
    )
    assert client.get(
        f"/api/tasks/{fixture['task'].id}/plans"
    ).json()["plans"] == []
    head = StrategyReportRepository(
        fixture["settings"].db_path
    ).get_head(
        task_id=fixture["task"].id,
        strategy_id=None,
    )
    assert head["current_revision"] == 1
    assert head["current_report_id"] == first["report_id"]
    assert head["current_content_hash"] == first["content_hash"]
