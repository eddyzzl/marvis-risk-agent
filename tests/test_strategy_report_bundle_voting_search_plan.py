"""Voting-search evidence selection for StrategyReportBundle V2 plans."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import marvis.agent.turn_handlers as turn_handlers
from marvis.agent.turn_handlers import _StrategyV2EvidenceSetupError
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.voting_candidate_search_tools import (
    VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
)


@pytest.fixture(autouse=True)
def _derive_stub_execution_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        turn_handlers,
        "derive_strategy_model_evidence_candidate_execution_ref",
        lambda binding: dict(binding.execution_ref),
    )


class _ArtifactWindow:
    def __init__(
        self,
        records: tuple[dict[str, object], ...],
        *,
        total: int,
    ) -> None:
        self.records = records
        self.total = total
        self.calls: list[tuple[str, str, int]] = []

    def list_recent_for_task_kind_with_count(
        self,
        task_id: str,
        kind: str,
        *,
        limit: int,
    ) -> tuple[list[dict[str, object]], int]:
        self.calls.append((task_id, kind, limit))
        return list(self.records[:limit]), self.total

    def list_for_task(self, *_args, **_kwargs):
        raise AssertionError("Voting search selection must not scan full history")


def _runtime(
    *records: dict[str, object],
    total: int | None = None,
) -> tuple[SimpleNamespace, _ArtifactWindow]:
    window = _ArtifactWindow(
        tuple(records),
        total=len(records) if total is None else total,
    )
    return SimpleNamespace(task_artifacts=window), window


def _sources() -> tuple[object, dict[str, object], object, object, object]:
    sample_ref = {
        "membership_artifact_id": "1" * 64,
        "expected_membership_artifact_content_hash": "2" * 64,
        "bundle_artifact_id": "3" * 64,
        "expected_bundle_artifact_content_hash": "4" * 64,
        "expected_bundle_id": "sample-design-bundle-current",
        "expected_sample_design_id": "sample-design-current",
        "expected_sample_design_content_hash": "5" * 64,
    }
    sample = SimpleNamespace(
        task_id="task-report",
        membership_artifact_id=sample_ref["membership_artifact_id"],
        membership_artifact_content_hash=sample_ref[
            "expected_membership_artifact_content_hash"
        ],
        bundle_artifact_id=sample_ref["bundle_artifact_id"],
        bundle_artifact_content_hash=sample_ref[
            "expected_bundle_artifact_content_hash"
        ],
        bundle={
            "bundle_id": sample_ref["expected_bundle_id"],
            "sample_design": {
                "sample_design_id": sample_ref[
                    "expected_sample_design_id"
                ],
                "content_hash": sample_ref[
                    "expected_sample_design_content_hash"
                ],
            },
        },
    )
    pool_payload = {
        "pool_id": "strategy-pool-current",
        "strategy_type": "approval",
        "revision": 3,
        "revision_id": "strategy-pool-revision-current",
        "snapshot_hash": "6" * 64,
        "entries": [
            {
                "rule_id": "rule-a",
                "enabled": True,
                "source": {
                    "asset_type": "univariate_rule",
                    "fragment_id": "fragment-a",
                },
                "execution": {"requirements": []},
            },
            {
                "rule_id": "rule-b",
                "enabled": True,
                "source": {
                    "asset_type": "automatic_tree_leaf",
                    "fragment_id": "fragment-b",
                },
                "execution": {"requirements": []},
            },
        ],
    }
    pool = SimpleNamespace(
        task_id="task-report",
        artifact_id="7" * 64,
        artifact_content_hash="8" * 64,
        pool=pool_payload,
    )
    legacy_ref = {
        "artifact_id": "9" * 64,
        "artifact_content_hash": "a" * 64,
        "sample_design_id": "legacy-sample-current",
        "sample_design_content_hash": "b" * 64,
        "partition": "development",
    }
    sample.execution_ref = legacy_ref
    execution_sample = SimpleNamespace(
        reference=SimpleNamespace(partition="development"),
        to_ref_dict=lambda: dict(legacy_ref),
        workspace_revision=2,
        workspace_generation=1,
        semantic_mapping_hash="c" * 64,
        target_col="bad",
        target_bad_value=1,
        drop_nan_labels=True,
        development_population_count=100,
        weight_col="weight",
        loan_amount_col="loan_amount",
    )
    dataset = SimpleNamespace(
        task_id="task-report",
        dataset_id="dataset-current",
        source_path="/governed/dataset.parquet",
        content_hash="d" * 64,
        registry_metadata_hash="e" * 64,
    )
    development = SimpleNamespace(
        task_id="task-report",
        pool=pool,
        dataset=dataset,
        sample_design=execution_sample,
        sample_design_v2=sample,
        evidence_identity={"sample_context_hash": "f" * 64},
    )
    provenance = {
        "task_id": "task-report",
        "pool_ref": {
            "artifact_id": pool.artifact_id,
            "artifact_content_hash": pool.artifact_content_hash,
            "pool_id": pool_payload["pool_id"],
            "strategy_type": pool_payload["strategy_type"],
            "revision": pool_payload["revision"],
            "revision_id": pool_payload["revision_id"],
            "snapshot_hash": pool_payload["snapshot_hash"],
        },
        "dataset_binding": {
            "task_id": "task-report",
            "dataset_id": dataset.dataset_id,
            "dataset_source_path": dataset.source_path,
            "dataset_content_hash": dataset.content_hash,
            "dataset_registry_metadata_hash": dataset.registry_metadata_hash,
            "workspace_revision": execution_sample.workspace_revision,
            "workspace_generation": execution_sample.workspace_generation,
            "semantic_mapping_hash": execution_sample.semantic_mapping_hash,
        },
        "sample_design_ref": legacy_ref,
        "sample_context_hash": development.evidence_identity[
            "sample_context_hash"
        ],
        "target_binding": {
            "column": "bad",
            "raw_bad_value": 1,
            "normalized_bad_value": 1,
            "drop_nan_labels": True,
            "nan_labels_dropped": 2,
            "labeled_count": 98,
            "sample_partition": "development",
        },
        "observation_bindings": {
            "weight_col": "weight",
            "amount_col": "loan_amount",
        },
        "requirement_bindings": None,
    }
    binding = SimpleNamespace(
        task_id="task-report",
        artifact_id="0" * 64,
        artifact_content_hash="1" * 64,
        artifact_provenance=provenance,
        pool_development=development,
        result={
            "search_id": "voting-search-" + ("2" * 32),
            "content_hash": "3" * 64,
            "configuration": {"candidate_ids": ["rule-a", "rule-b"]},
        },
    )
    return sample, sample_ref, pool, development, binding


def _native_sources() -> tuple[object, dict[str, object], object, object, object]:
    sample, sample_ref, pool, development, binding = _sources()
    native_ref = {
        "artifact_id": sample_ref["bundle_artifact_id"],
        "artifact_content_hash": sample_ref[
            "expected_bundle_artifact_content_hash"
        ],
        "sample_design_id": sample_ref["expected_sample_design_id"],
        "sample_design_content_hash": sample_ref[
            "expected_sample_design_content_hash"
        ],
        "partition": "risk/development",
    }
    sample.execution_ref = native_ref
    execution_sample = SimpleNamespace(
        **{
            **vars(development.sample_design),
            "reference": SimpleNamespace(partition="risk/development"),
            "to_ref_dict": lambda: dict(native_ref),
            "target_bad_value": 0,
        }
    )
    native_development = SimpleNamespace(
        **{
            **vars(development),
            "sample_design": execution_sample,
            "sample_design_v2": None,
        }
    )
    provenance = {
        **binding.artifact_provenance,
        "sample_design_ref": native_ref,
        "target_binding": {
            **binding.artifact_provenance["target_binding"],
            "raw_bad_value": 0,
            "sample_partition": "risk/development",
        },
    }
    native_binding = SimpleNamespace(
        **{
            **vars(binding),
            "artifact_provenance": provenance,
            "pool_development": native_development,
        }
    )
    return sample, sample_ref, pool, native_development, native_binding


def _record(binding: object) -> dict[str, object]:
    return {
        "id": binding.artifact_id,
        "kind": VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        "content_hash": binding.artifact_content_hash,
    }


def _unrelated(binding: object) -> object:
    provenance = {
        **binding.artifact_provenance,
        "pool_ref": {
            **binding.artifact_provenance["pool_ref"],
            "snapshot_hash": "4" * 64,
        },
    }
    other_pool = SimpleNamespace(
        **{
            **vars(binding.pool_development.pool),
            "artifact_id": "5" * 64,
        }
    )
    development = SimpleNamespace(
        **{
            **vars(binding.pool_development),
            "pool": other_pool,
        }
    )
    return SimpleNamespace(
        **{
            **vars(binding),
            "artifact_id": "6" * 64,
            "artifact_content_hash": "7" * 64,
            "artifact_provenance": provenance,
            "pool_development": development,
        }
    )


def test_report_voting_search_selects_first_exact_after_newer_valid_unrelated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _sources()
    unrelated = _unrelated(exact)
    calls = []
    by_id = {
        exact.artifact_id: exact,
        unrelated.artifact_id: unrelated,
    }

    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )

    def load_historical(runtime, **kwargs):
        calls.append(kwargs["artifact_id"])
        return by_id[kwargs["artifact_id"]]

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        load_historical,
        raising=False,
    )
    runtime, window = _runtime(_record(unrelated), _record(exact))

    selected = turn_handlers._strategy_report_latest_voting_search_binding(
        runtime,
        task_id="task-report",
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )

    assert selected is exact
    assert calls == [unrelated.artifact_id, exact.artifact_id]
    assert window.calls == [
        (
            "task-report",
            VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            turn_handlers._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT,
        )
    ]


def test_report_voting_search_selects_exact_native_execution_without_legacy_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _native_sources()
    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        lambda runtime, **kwargs: exact,
        raising=False,
    )
    runtime, _window = _runtime(_record(exact))

    selected = turn_handlers._strategy_report_latest_voting_search_binding(
        runtime,
        task_id="task-report",
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )

    assert selected is exact


def test_report_voting_search_corrupt_newest_fails_closed_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _sources()
    corrupt = {
        "id": "8" * 64,
        "kind": VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
        "content_hash": "9" * 64,
    }
    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )

    def load_historical(runtime, **kwargs):
        if kwargs["artifact_id"] == corrupt["id"]:
            raise StrategyError("corrupt search artifact")
        return exact

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        load_historical,
        raising=False,
    )
    runtime, _window = _runtime(corrupt, _record(exact))

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        turn_handlers._strategy_report_latest_voting_search_binding(
            runtime,
            task_id="task-report",
            sample=sample,
            sample_ref=sample_ref,
            pool=pool,
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_voting_candidate_search_invalid"
    )


def test_report_voting_search_returns_none_when_no_authenticated_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _sources()
    unrelated = _unrelated(exact)
    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )
    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        lambda runtime, **kwargs: unrelated,
        raising=False,
    )
    runtime, _window = _runtime(_record(unrelated))

    selected = turn_handlers._strategy_report_latest_voting_search_binding(
        runtime,
        task_id="task-report",
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )

    assert selected is None


def test_report_voting_search_absent_returns_none_without_source_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, _development, _exact = _sources()

    def unexpected(*args, **kwargs):
        raise AssertionError("no source binding should be loaded")

    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        unexpected,
        raising=False,
    )
    runtime, window = _runtime()

    selected = turn_handlers._strategy_report_latest_voting_search_binding(
        runtime,
        task_id="task-report",
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )

    assert selected is None
    assert window.calls == [
        (
            "task-report",
            VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            turn_handlers._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT,
        )
    ]


def test_report_voting_search_selection_window_exhaustion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _sources()
    unrelated = _unrelated(exact)
    calls: list[str] = []
    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )

    def load_historical(runtime, **kwargs):
        artifact_id = kwargs["artifact_id"]
        calls.append(artifact_id)
        return unrelated

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        load_historical,
        raising=False,
    )
    replay_limit = turn_handlers._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT
    recent_records = tuple(
        {
            "id": f"{index:064x}",
            "kind": VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            "content_hash": f"{index + replay_limit:064x}",
        }
        for index in range(replay_limit)
    )
    runtime, window = _runtime(*recent_records, total=replay_limit + 1)

    with pytest.raises(_StrategyV2EvidenceSetupError) as raised:
        turn_handlers._strategy_report_latest_voting_search_binding(
            runtime,
            task_id="task-report",
            sample=sample,
            sample_ref=sample_ref,
            pool=pool,
        )

    assert raised.value.code == (
        "strategy_report_bundle_v2_voting_candidate_search_"
        "selection_window_exhausted"
    )
    assert calls == [str(record["id"]) for record in recent_records]
    assert window.calls == [
        (
            "task-report",
            VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            replay_limit,
        )
    ]


def test_report_voting_search_returns_exact_inside_truncated_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, sample_ref, pool, development, exact = _sources()
    unrelated = _unrelated(exact)
    replay_limit = turn_handlers._STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT
    recent_records = tuple(
        {
            "id": f"{index + 10:064x}",
            "kind": VOTING_CANDIDATE_SEARCH_ARTIFACT_KIND,
            "content_hash": f"{index + 100:064x}",
        }
        for index in range(replay_limit - 1)
    )
    exact_record = _record(exact)
    by_id = {
        **{str(record["id"]): unrelated for record in recent_records},
        exact.artifact_id: exact,
    }
    calls: list[str] = []
    monkeypatch.setattr(
        turn_handlers,
        "bind_strategy_pool_development_execution",
        lambda runtime, actual_pool: development,
        raising=False,
    )

    def load_historical(runtime, **kwargs):
        artifact_id = kwargs["artifact_id"]
        calls.append(artifact_id)
        return by_id[artifact_id]

    monkeypatch.setattr(
        turn_handlers,
        "load_historical_voting_candidate_search_artifact",
        load_historical,
        raising=False,
    )
    runtime, _window = _runtime(
        *recent_records,
        exact_record,
        total=replay_limit + 20,
    )

    selected = turn_handlers._strategy_report_latest_voting_search_binding(
        runtime,
        task_id="task-report",
        sample=sample,
        sample_ref=sample_ref,
        pool=pool,
    )

    assert selected is exact
    assert calls == [
        *(str(record["id"]) for record in recent_records),
        exact.artifact_id,
    ]
