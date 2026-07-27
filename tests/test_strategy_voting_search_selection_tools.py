from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import marvis.packs.strategy.tools as strategy_tools
import marvis.packs.strategy.voting_candidate_search_tools as search_tools
from marvis.packs.strategy.errors import StrategyError
from marvis.packs.strategy.pool import ABSENT_POOL_SNAPSHOT_HASH
from marvis.packs.strategy.pool_tools import (
    run_add_candidate_to_pool,
    run_set_pool_entry_action,
)
from marvis.packs.strategy.voting_candidate_search_tools import (
    resolve_voting_candidate_search_inputs,
    run_search_voting_candidates,
)
from marvis.packs.strategy.voting_candidate_tools import (
    VOTING_CANDIDATE_ARTIFACT_KIND,
    run_build_voting_candidate,
)
from marvis.plugins.schema_validation import validate_against_schema
from marvis.repositories.strategy_pool import StrategyCandidatePoolRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_strategy_candidate_stability_tools import _pool_add_inputs
from tests.test_strategy_voting_candidate_search_tools import _search_fixture


def _searched_fixture(tmp_path: Path) -> dict:
    fixture = _search_fixture(tmp_path)
    search_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls=fixture["controls"],
    )
    search = run_search_voting_candidates(
        search_inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    return {**fixture, "search": search}


def test_resolver_recovers_exact_evaluated_combo_from_current_pool(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    before_pool = StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval")

    binding = search_tools.resolve_voting_candidate_search_selection(
        fixture["runtime"],
        task_id=fixture["task"].id,
        search_id=fixture["search"]["search_id"],
        combo_id=combo["combo_id"],
    )

    entries_by_rule = {
        entry["rule_id"]: entry
        for entry in fixture["pool"]["entries"]
        if entry["enabled"]
        and entry["source"]["asset_type"] != "voting_n_of_k"
    }
    assert binding.search_id == fixture["search"]["search_id"]
    assert binding.combo_id == combo["combo_id"]
    assert binding.strategy_type == "approval"
    assert binding.member_rule_ids == tuple(combo["member_ids"])
    assert binding.selected_entry_ids == tuple(
        entries_by_rule[rule_id]["entry_id"] for rule_id in combo["member_ids"]
    )
    assert binding.n == combo["n"]
    assert binding.eligible is combo["eligible"]
    assert binding.constraint_failures == tuple(combo["constraint_failures"])
    assert binding.rank == combo["rank"]
    assert (
        StrategyCandidatePoolRepository(
            fixture["settings"].db_path
        ).get_current(fixture["task"].id, "approval")
        == before_pool
    )


def test_build_from_search_uses_only_authenticated_pointer_and_leaves_pool_unchanged(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    before_pool = StrategyCandidatePoolRepository(
        fixture["settings"].db_path
    ).get_current(fixture["task"].id, "approval")

    output = search_tools.run_build_voting_candidate_from_search(
        {
            "search_id": fixture["search"]["search_id"],
            "combo_id": combo["combo_id"],
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    assert (
        output["schema_version"]
        == "strategy.build-voting-candidate-from-search-tool.v1"
    )
    candidate = output["voting_candidate"]
    assert candidate["schema_version"] == "strategy.build-voting-candidate-tool.v2"
    assert candidate["n"] == combo["n"]
    assert {
        entry["rule_id"] for entry in candidate["selected_entries"]
    } == set(combo["member_ids"])
    assert output["source_search_selection"] == {
        "search_id": fixture["search"]["search_id"],
        "combo_id": combo["combo_id"],
        "strategy_type": "approval",
        "rank": combo["rank"],
        "member_rule_ids": combo["member_ids"],
        "n": combo["n"],
        "eligible": combo["eligible"],
        "constraint_failures": combo["constraint_failures"],
    }
    assert output["not_admitted"] is True
    assert output["not_applied"] is True
    assert output["not_adopted"] is True
    assert output["not_deployed"] is True
    manifest = json.loads(
        (
            Path(__file__).parents[1]
            / "marvis"
            / "packs"
            / "strategy"
            / "manifest.json"
        ).read_text("utf-8")
    )
    tool = next(
        item
        for item in manifest["tools"]
        if item["name"] == "build_voting_candidate_from_search"
    )
    validate_against_schema(
        output,
        tool["output_schema"],
        label="build Voting candidate from search output",
    )
    assert (
        StrategyCandidatePoolRepository(
            fixture["settings"].db_path
        ).get_current(fixture["task"].id, "approval")
        == before_pool
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_id", "a" * 64),
        ("artifact_hash", "b" * 64),
        ("rule_ids", ["rule-forged"]),
        ("selected_entry_ids", ["entry-forged"]),
        ("n", 1),
        ("rank", 1),
        ("winner", True),
        ("champion", True),
        ("dataset_id", "dataset-forged"),
    ],
)
def test_build_from_search_rejects_all_non_pointer_user_inputs(
    field: str,
    value: object,
) -> None:
    with pytest.raises(StrategyError, match="unsupported"):
        search_tools.run_build_voting_candidate_from_search(
            {
                "search_id": "voting-search-" + "a" * 32,
                "combo_id": "voting-combo-" + "b" * 32,
                field: value,
            },
            SimpleNamespace(task_id="task-owned"),
            None,
        )


def test_build_from_search_rejects_explicit_null_strategy_type() -> None:
    with pytest.raises(StrategyError, match="strategy_type"):
        search_tools.run_build_voting_candidate_from_search(
            {
                "search_id": "voting-search-" + "a" * 32,
                "combo_id": "voting-combo-" + "b" * 32,
                "strategy_type": None,
            },
            SimpleNamespace(task_id="task-owned"),
            None,
        )


def test_resolver_rejects_search_after_current_pool_revision_changes(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    changed = run_set_pool_entry_action(
        {
            "strategy_type": "approval",
            "expected_pool_revision": fixture["pool"]["revision"],
            "expected_pool_snapshot_hash": fixture["pool"]["snapshot_hash"],
            "rule_id": fixture["pool"]["entries"][0]["rule_id"],
            "action": {
                "type": "review",
                "value": "review",
                "reason_code": "POOL_DRIFT",
                "stop": True,
            },
        },
        fixture["ctx"],
        fixture["runtime"],
    )
    assert changed["snapshot_hash"] != fixture["pool"]["snapshot_hash"]

    with pytest.raises(StrategyError, match="current Strategy Pool|search again"):
        search_tools.resolve_voting_candidate_search_selection(
            fixture["runtime"],
            task_id=fixture["task"].id,
            search_id=fixture["search"]["search_id"],
            combo_id=combo["combo_id"],
            strategy_type="approval",
        )


@pytest.mark.parametrize("tamper", ["artifact_bytes", "provenance"])
def test_resolver_rejects_search_artifact_or_provenance_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    descriptor = fixture["search"]["artifacts"][0]
    repository = TaskArtifactRepository(fixture["settings"].db_path)
    record = repository.get_for_task(
        fixture["task"].id,
        descriptor["artifact_id"],
    )
    assert record is not None
    if tamper == "artifact_bytes":
        Path(record["path"]).write_bytes(b"{}")
    else:
        provenance = dict(record["provenance"])
        provenance["search_id"] = "voting-search-" + "f" * 32
        with repository.transaction() as conn:
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
                    descriptor["artifact_id"],
                ),
            )

    with pytest.raises(StrategyError, match="content|provenance|identity"):
        search_tools.resolve_voting_candidate_search_selection(
            fixture["runtime"],
            task_id=fixture["task"].id,
            search_id=fixture["search"]["search_id"],
            combo_id=combo["combo_id"],
            strategy_type="approval",
        )


def test_truncated_search_allows_evaluated_combo_and_rejects_unevaluated_combo(
    tmp_path: Path,
) -> None:
    fixture = _search_fixture(tmp_path)
    truncated_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls={**fixture["controls"], "max_combinations": 1},
    )
    truncated = run_search_voting_candidates(
        truncated_inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    full_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls=fixture["controls"],
    )
    full = run_search_voting_candidates(
        full_inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    [evaluated] = truncated["search_result"]["combinations"]
    unevaluated = next(
        item
        for item in full["search_result"]["combinations"]
        if item["combo_id"] != evaluated["combo_id"]
    )
    assert truncated["truncated"] is True

    binding = search_tools.resolve_voting_candidate_search_selection(
        fixture["runtime"],
        task_id=fixture["task"].id,
        search_id=truncated["search_id"],
        combo_id=evaluated["combo_id"],
        strategy_type="approval",
    )
    assert binding.combo_id == evaluated["combo_id"]
    with pytest.raises(StrategyError, match="evaluated"):
        search_tools.resolve_voting_candidate_search_selection(
            fixture["runtime"],
            task_id=fixture["task"].id,
            search_id=truncated["search_id"],
            combo_id=unevaluated["combo_id"],
            strategy_type="approval",
        )


def test_ineligible_evaluated_combo_remains_buildable_with_failures_exposed(
    tmp_path: Path,
) -> None:
    fixture = _search_fixture(tmp_path)
    inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls={
            **fixture["controls"],
            "constraints": [
                {"metric": "hit_count", "operator": "gte", "value": 121}
            ],
        },
    )
    search = run_search_voting_candidates(
        inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    combo = search["search_result"]["combinations"][0]
    assert combo["eligible"] is False
    assert combo["constraint_failures"]

    output = search_tools.run_build_voting_candidate_from_search(
        {
            "search_id": search["search_id"],
            "combo_id": combo["combo_id"],
            "strategy_type": "approval",
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    selection = output["source_search_selection"]
    assert selection["eligible"] is False
    assert selection["constraint_failures"] == combo["constraint_failures"]


def test_resolver_requires_strategy_type_when_same_search_matches_two_pool_types(
    tmp_path: Path,
) -> None:
    fixture = _searched_fixture(tmp_path)
    reject_pool = None
    for candidate in (
        fixture["first"],
        fixture["refine"](1),
        fixture["refine"](2),
    ):
        inputs = _pool_add_inputs(
            candidate,
            expected_revision=(
                0 if reject_pool is None else reject_pool["revision"]
            ),
            expected_hash=(
                ABSENT_POOL_SNAPSHOT_HASH
                if reject_pool is None
                else reject_pool["snapshot_hash"]
            ),
        )
        inputs["strategy_type"] = "reject"
        added = run_add_candidate_to_pool(
            inputs,
            fixture["ctx"],
            fixture["runtime"],
        )
        reject_pool = added["pool"]
    reject_search_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["task"].id,
        user_controls={**fixture["controls"], "strategy_type": "reject"},
    )
    reject_search = run_search_voting_candidates(
        reject_search_inputs,
        fixture["ctx"],
        fixture["runtime"],
    )
    assert reject_search["search_id"] == fixture["search"]["search_id"]
    combo = fixture["search"]["search_result"]["combinations"][0]

    with pytest.raises(StrategyError, match="multiple.*strategy_type"):
        search_tools.resolve_voting_candidate_search_selection(
            fixture["runtime"],
            task_id=fixture["task"].id,
            search_id=fixture["search"]["search_id"],
            combo_id=combo["combo_id"],
        )
    approval = search_tools.resolve_voting_candidate_search_selection(
        fixture["runtime"],
        task_id=fixture["task"].id,
        search_id=fixture["search"]["search_id"],
        combo_id=combo["combo_id"],
        strategy_type="approval",
    )
    reject = search_tools.resolve_voting_candidate_search_selection(
        fixture["runtime"],
        task_id=fixture["task"].id,
        search_id=fixture["search"]["search_id"],
        combo_id=combo["combo_id"],
        strategy_type="reject",
    )
    assert approval.strategy_type == "approval"
    assert reject.strategy_type == "reject"
    assert approval.selected_entry_ids != reject.selected_entry_ids


def test_builder_pool_cas_rejects_race_after_search_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marvis.packs.strategy.voting_candidate_tools as voting_tools

    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    original_builder = (
        voting_tools._run_build_voting_candidate_with_registration_guard
    )

    def mutate_pool_then_build(
        inputs,
        ctx,
        runtime,
        *,
        registration_guard,
    ):
        run_set_pool_entry_action(
            {
                "strategy_type": "approval",
                "expected_pool_revision": fixture["pool"]["revision"],
                "expected_pool_snapshot_hash": fixture["pool"]["snapshot_hash"],
                "rule_id": fixture["pool"]["entries"][0]["rule_id"],
                "action": {
                    "type": "review",
                    "value": "review",
                    "reason_code": "RACE",
                    "stop": True,
                },
            },
            ctx,
            runtime,
        )
        return original_builder(
            inputs,
            ctx,
            runtime,
            registration_guard=registration_guard,
        )

    monkeypatch.setattr(
        voting_tools,
        "_run_build_voting_candidate_with_registration_guard",
        mutate_pool_then_build,
    )

    with pytest.raises(StrategyError, match="stale.*Pool|revision|snapshot"):
        search_tools.run_build_voting_candidate_from_search(
            {
                "search_id": fixture["search"]["search_id"],
                "combo_id": combo["combo_id"],
                "strategy_type": "approval",
            },
            fixture["ctx"],
            fixture["runtime"],
        )
    assert not [
        item
        for item in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if item["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]


def test_builder_rechecks_search_bytes_before_candidate_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marvis.packs.strategy.voting_candidate_tools as voting_tools

    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    original_resolver = search_tools.resolve_voting_candidate_search_selection
    original_promote = voting_tools.ArtifactUnitOfWork.promote_all
    search_path: Path | None = None
    search_bytes: bytes | None = None
    promotion_calls = 0

    def resolve_then_tamper(*args, **kwargs):
        nonlocal search_path, search_bytes
        selection = original_resolver(*args, **kwargs)
        search_path = selection.artifact_binding.artifact_path
        search_bytes = search_path.read_bytes()
        search_path.write_bytes(search_bytes + b" ")
        return selection

    def track_promotion(self):
        nonlocal promotion_calls
        promotion_calls += 1
        return original_promote(self)

    monkeypatch.setattr(
        search_tools,
        "resolve_voting_candidate_search_selection",
        resolve_then_tamper,
    )
    monkeypatch.setattr(
        voting_tools.ArtifactUnitOfWork,
        "promote_all",
        track_promotion,
    )
    try:
        with pytest.raises(StrategyError, match="search artifact.*content|content hash"):
            search_tools.run_build_voting_candidate_from_search(
                {
                    "search_id": fixture["search"]["search_id"],
                    "combo_id": combo["combo_id"],
                    "strategy_type": "approval",
                },
                fixture["ctx"],
                fixture["runtime"],
            )
    finally:
        if search_path is not None and search_bytes is not None:
            search_path.write_bytes(search_bytes)

    assert promotion_calls == 0
    assert not [
        item
        for item in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if item["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]


def test_builder_rechecks_search_bytes_before_candidate_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    descriptor = fixture["search"]["artifacts"][0]
    search_record = TaskArtifactRepository(
        fixture["settings"].db_path
    ).get_for_task(
        fixture["task"].id,
        descriptor["artifact_id"],
    )
    assert search_record is not None
    search_path = Path(search_record["path"])
    search_bytes = search_path.read_bytes()
    original_register = fixture["runtime"].task_artifacts.register_on_connection

    def tamper_search_after_candidate_registration(conn, **kwargs):
        record = original_register(conn, **kwargs)
        search_path.write_bytes(search_bytes + b" ")
        return record

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        tamper_search_after_candidate_registration,
    )
    try:
        with pytest.raises(StrategyError, match="search artifact.*content|content hash"):
            search_tools.run_build_voting_candidate_from_search(
                {
                    "search_id": fixture["search"]["search_id"],
                    "combo_id": combo["combo_id"],
                    "strategy_type": "approval",
                },
                fixture["ctx"],
                fixture["runtime"],
            )
    finally:
        search_path.write_bytes(search_bytes)

    assert not [
        item
        for item in TaskArtifactRepository(
            fixture["settings"].db_path
        ).list_for_task(fixture["task"].id)
        if item["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]
    candidate_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_voting_candidates"
    )
    assert not list(candidate_dir.glob("*.json"))


@pytest.mark.parametrize("tamper", ["provenance", "deletion"])
def test_builder_rechecks_search_registry_before_candidate_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    descriptor = fixture["search"]["artifacts"][0]
    artifact_repository = TaskArtifactRepository(fixture["settings"].db_path)
    search_record = artifact_repository.get_for_task(
        fixture["task"].id,
        descriptor["artifact_id"],
    )
    assert search_record is not None
    original_register = fixture["runtime"].task_artifacts.register_on_connection

    def drift_search_after_candidate_registration(conn, **kwargs):
        record = original_register(conn, **kwargs)
        if tamper == "provenance":
            conn.execute("DROP TRIGGER trg_task_artifacts_immutable_update")
            conn.execute(
                "UPDATE task_artifacts SET provenance_json = ? WHERE id = ?",
                ("{}", descriptor["artifact_id"]),
            )
        else:
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (descriptor["artifact_id"],),
            )
        return record

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        drift_search_after_candidate_registration,
    )

    with pytest.raises(StrategyError, match="registry|registered"):
        search_tools.run_build_voting_candidate_from_search(
            {
                "search_id": fixture["search"]["search_id"],
                "combo_id": combo["combo_id"],
                "strategy_type": "approval",
            },
            fixture["ctx"],
            fixture["runtime"],
        )

    assert artifact_repository.get_for_task(
        fixture["task"].id,
        descriptor["artifact_id"],
    ) == search_record
    assert not [
        item
        for item in artifact_repository.list_for_task(fixture["task"].id)
        if item["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]
    candidate_dir = (
        Path(fixture["settings"].tasks_dir)
        / fixture["task"].id
        / "strategy_voting_candidates"
    )
    assert not list(candidate_dir.glob("*.json"))


@pytest.mark.slow
def test_builder_rechecks_nonselected_pool_requirement_before_candidate_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marvis.packs.strategy.scorecard_candidate_tools import (
        run_build_scorecard_band_asset,
    )
    from tests.test_model_score_evidence_tool import _run_score
    from tests.test_modeling_training_evidence_tool import (
        _run as run_training,
    )
    from tests.test_strategy_pool_scorecard import (
        _add_inputs as scorecard_add_inputs,
        _selection as scorecard_selection,
    )
    from tests.test_strategy_voting_scorecard import (
        _two_scorecard_pool_entries,
    )

    fixture = _two_scorecard_pool_entries(tmp_path)
    fixture["fx"]["inputs"]["seed"] += 1
    training = run_training(fixture["fx"])
    scored = _run_score(fixture["fx"], training)
    score_artifacts = scored["artifacts"]
    third_band = run_build_scorecard_band_asset(
        {
            "score_evidence_ref": {
                "evidence_artifact_id": score_artifacts["score_evidence"][
                    "artifact_id"
                ],
                "expected_evidence_artifact_content_hash": score_artifacts[
                    "score_evidence"
                ]["content_hash"],
                "score_vector_artifact_id": score_artifacts["score_vector"][
                    "artifact_id"
                ],
                "expected_score_vector_artifact_content_hash": score_artifacts[
                    "score_vector"
                ]["content_hash"],
            },
            "sample_design_ref": fixture["fx"]["sample_ref"],
            "banding": {"method": "equal_frequency", "bin_count": 3},
        },
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    third_selection = scorecard_selection(
        {**fixture, "band": third_band},
        ordinal=0,
    )
    added = run_add_candidate_to_pool(
        scorecard_add_inputs(
            third_selection,
            expected_revision=fixture["pool"]["revision"],
            expected_snapshot_hash=fixture["pool"]["snapshot_hash"],
        ),
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    pool = added["pool"]
    third_entry = next(
        entry
        for entry in pool["entries"]
        if entry["source"]["artifact_id"]
        == third_selection["artifacts"][0]["artifact_id"]
    )
    [third_requirement] = third_entry["execution"]["requirements"]

    search_inputs = resolve_voting_candidate_search_inputs(
        fixture["runtime"],
        task_id=fixture["fx"]["task"].id,
        user_controls={
            "strategy_type": "approval",
            "member_count": 2,
            "n": 1,
            "objective": {
                "metric": "bad_capture_rate",
                "direction": "maximize",
            },
            "constraints": [],
            "include_rule_ids": [],
            "exclude_rule_ids": [],
            "max_combinations": 10,
        },
    )
    search = run_search_voting_candidates(
        search_inputs,
        fixture["fx"]["ctx"],
        fixture["runtime"],
    )
    combo = next(
        item
        for item in search["search_result"]["combinations"]
        if third_entry["rule_id"] not in item["member_ids"]
    )
    selected_entries = [
        entry
        for entry in pool["entries"]
        if entry["rule_id"] in combo["member_ids"]
    ]
    selected_evidence_ids = {
        requirement["score_evidence_artifact_id"]
        for entry in selected_entries
        for requirement in entry["execution"]["requirements"]
    }
    assert (
        third_requirement["score_evidence_artifact_id"]
        not in selected_evidence_ids
    )

    artifact_repository = TaskArtifactRepository(
        fixture["fx"]["settings"].db_path
    )
    requirement_record = artifact_repository.get_for_task(
        fixture["fx"]["task"].id,
        third_requirement["score_evidence_artifact_id"],
    )
    assert requirement_record is not None
    requirement_path = Path(requirement_record["path"])
    requirement_bytes = requirement_path.read_bytes()
    original_register = fixture["runtime"].task_artifacts.register_on_connection

    def tamper_nonselected_requirement_after_candidate_registration(
        conn,
        **kwargs,
    ):
        record = original_register(conn, **kwargs)
        requirement_path.write_bytes(requirement_bytes + b" ")
        return record

    monkeypatch.setattr(
        fixture["runtime"].task_artifacts,
        "register_on_connection",
        tamper_nonselected_requirement_after_candidate_registration,
    )
    try:
        with pytest.raises(
            StrategyError,
            match="model score evidence|artifact|content|hash",
        ):
            search_tools.run_build_voting_candidate_from_search(
                {
                    "search_id": search["search_id"],
                    "combo_id": combo["combo_id"],
                    "strategy_type": "approval",
                },
                fixture["fx"]["ctx"],
                fixture["runtime"],
            )
    finally:
        requirement_path.write_bytes(requirement_bytes)

    assert not [
        item
        for item in artifact_repository.list_for_task(
            fixture["fx"]["task"].id
        )
        if item["kind"] == VOTING_CANDIDATE_ARTIFACT_KIND
    ]
    candidate_dir = (
        Path(fixture["fx"]["settings"].tasks_dir)
        / fixture["fx"]["task"].id
        / "strategy_voting_candidates"
    )
    assert not list(candidate_dir.glob("*.json"))


def test_existing_explicit_voting_builder_remains_compatible(tmp_path: Path) -> None:
    fixture = _searched_fixture(tmp_path)
    combo = fixture["search"]["search_result"]["combinations"][0]
    binding = search_tools.resolve_voting_candidate_search_selection(
        fixture["runtime"],
        task_id=fixture["task"].id,
        search_id=fixture["search"]["search_id"],
        combo_id=combo["combo_id"],
        strategy_type="approval",
    )

    explicit = run_build_voting_candidate(
        {
            "strategy_type": binding.strategy_type,
            "expected_pool_revision": binding.pool_revision,
            "expected_pool_snapshot_hash": binding.pool_snapshot_hash,
            "selected_entry_ids": list(binding.selected_entry_ids),
            "n": binding.n,
        },
        fixture["ctx"],
        fixture["runtime"],
    )

    assert explicit["schema_version"] == "strategy.build-voting-candidate-tool.v2"
    assert "source_search_selection" not in explicit


def test_strategy_pack_registers_pointer_only_voting_search_builder() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "marvis" / "packs" / "strategy" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    [tool] = [
        item
        for item in manifest["tools"]
        if item["name"] == "build_voting_candidate_from_search"
    ]

    assert tool["entrypoint"] == "tool_build_voting_candidate_from_search"
    assert tool["determinism"] == "deterministic"
    assert tool["input_schema"]["required"] == ["search_id", "combo_id"]
    assert set(tool["input_schema"]["properties"]) == {
        "search_id",
        "combo_id",
        "strategy_type",
    }
    assert tool["input_schema"]["additionalProperties"] is False
    output = tool["output_schema"]
    assert output["properties"]["schema_version"] == {
        "const": "strategy.build-voting-candidate-from-search-tool.v1"
    }
    assert output["properties"]["not_mutated_pool"] == {"const": True}
    selection = output["$defs"]["source_search_selection"]
    assert "eligible" in selection["required"]
    assert "constraint_failures" in selection["required"]
    assert output["additionalProperties"] is False


def test_strategy_tools_entrypoint_forwards_pointer_without_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = object()
    runtime = object()
    inputs = {
        "search_id": "voting-search-" + "a" * 32,
        "combo_id": "voting-combo-" + "b" * 32,
    }
    calls: list[tuple[object, object, object]] = []

    monkeypatch.setattr(strategy_tools, "_runtime", lambda actual: runtime)

    def fake_runner(actual_inputs, actual_ctx, actual_runtime):
        calls.append((actual_inputs, actual_ctx, actual_runtime))
        return {"schema_version": "fake"}

    monkeypatch.setattr(
        strategy_tools,
        "run_build_voting_candidate_from_search",
        fake_runner,
    )

    assert strategy_tools.tool_build_voting_candidate_from_search(inputs, ctx) == {
        "schema_version": "fake"
    }
    assert calls == [(inputs, ctx, runtime)]
