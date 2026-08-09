"""Authenticated presenters for governed Strategy exploration outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import quote

from marvis.agent.presenters.modeling_evidence import (
    CanonicalPresenterIntegrityError,
)
from marvis.packs.strategy import candidate_asset_tools
from marvis.packs.strategy import cross_matrix_candidate_tools as cross_matrix_tools
from marvis.packs.strategy.cross_candidate_search_tools import (
    CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
    CROSS_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION,
    CROSS_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION,
    load_cross_candidate_search_artifact,
    resolve_cross_candidate_search_pair,
    _validate_search_inputs,
    _validate_selection_inputs,
)
from marvis.packs.strategy.cross_matrix_candidate import (
    CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION,
    parse_cross_matrix_candidate_asset_json,
    rebuild_cross_matrix_candidate_asset,
)
from marvis.packs.strategy.impact_cube_tools import (
    IMPACT_CUBE_ARTIFACT_KIND,
    IMPACT_CUBE_ORIGIN_TOOL,
    validate_measure_strategy_impact_cube_tool_output,
    validate_impact_cube_producer_run,
)
from marvis.packs.strategy.interactive_tree_revision import (
    INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION,
)
from marvis.packs.strategy.interactive_tree_split_search_tools import (
    INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
    INTERACTIVE_TREE_SPLIT_SEARCH_TOOL_SCHEMA_VERSION,
    load_verified_interactive_tree_split_search,
    _validate_inputs as _validate_split_search_inputs,
)
from marvis.packs.strategy.interactive_tree_tools import (
    AUTO_CONTINUE_TOOL_SCHEMA_VERSION,
    INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
    TOOL_SCHEMA_VERSION,
    TOOL_SCHEMA_VERSION_V2,
    _ResolvedRevisionSource,
    _build_revision,
    _validate_auto_continuation_inputs,
    _validate_inputs as _validate_revision_inputs,
    load_verified_interactive_tree_revision,
)


_INTEGRITY_ERROR = (
    "canonical Tool output authentication failed; presentation stopped"
)
_ARTIFACT_FIELDS = {
    "artifact_id",
    "kind",
    "format",
    "filename",
    "content_hash",
    "download_url",
}
_CROSS_SEARCH_FIELDS = {
    "schema_version",
    "search_id",
    "request_hash",
    "content_hash",
    "source_artifact_id",
    "candidate_id",
    "evidence_hash",
    "population_count",
    "search_space",
    "evaluated",
    "truncated",
    "eligible",
    "search_result",
    "artifacts",
    "not_selected",
    "not_admitted",
    "not_applied",
    "not_adopted",
    "not_deployed",
}
_CROSS_SELECTION_FIELDS = {
    "schema_version",
    "source_search_selection",
    "cross_matrix_candidate",
    "not_selected",
    "not_admitted",
    "not_applied",
    "not_adopted",
    "not_deployed",
}
_CROSS_CANDIDATE_FIELDS = {
    "schema_version",
    "asset_id",
    "asset_hash",
    "parent_candidate_id",
    "parent_evidence_hash",
    "candidate_id",
    "evidence_hash",
    "dataset_id",
    "target_col",
    "population_count",
    "labeled_count",
    "drop_nan_labels",
    "nan_labels_dropped",
    "row_axis",
    "column_axis",
    "cell_count",
    "candidate_stage",
    "observation_stage",
    "validation_status",
    "cross_matrix_candidate",
    "artifacts",
    "not_selected",
    "not_admitted",
    "not_applied",
    "not_adopted",
    "not_deployed",
}
_SPLIT_SEARCH_FIELDS = {
    "schema_version",
    "search_id",
    "search_hash",
    "source_tree_id",
    "node_id",
    "mode",
    "feature_count",
    "evaluated_candidates",
    "eligible_candidates",
    "truncated",
    "search_result",
    "artifacts",
    "winner_selected",
    "tree_modified",
}
_REVISION_FIELDS = {
    "schema_version",
    "revision_id",
    "revision_hash",
    "semantic_tree_id",
    "tree_hash",
    "source_tree_id",
    "base_asset_id",
    "parent_revision_id",
    "edit",
    "visible_node_count",
    "frontier_node_count",
    "replay",
    "artifacts",
}
_AUTO_CONTINUE_FIELDS = _REVISION_FIELDS | {
    "search_id",
    "search_hash",
    "candidate_id",
    "automatic_winner_selection",
    "pool_modified",
}


def _presenter(function: Callable[..., tuple[str, list[dict[str, Any]]]]):
    @wraps(function)
    def fail_closed(
        output: object,
        *,
        runtime: object,
        task_id: str,
        trusted_inputs: Mapping[str, Any] | None = None,
        trusted_artifacts: Mapping[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        try:
            if runtime is None:
                raise ValueError("runtime is missing")
            trusted_task_id = _text(task_id, "task_id")
            if not isinstance(trusted_inputs, Mapping):
                raise ValueError("trusted step inputs are missing")
            return function(
                output,
                runtime=runtime,
                task_id=trusted_task_id,
                trusted_inputs=trusted_inputs,
                trusted_artifacts=trusted_artifacts,
            )
        except CanonicalPresenterIntegrityError:
            raise
        except Exception as exc:
            raise CanonicalPresenterIntegrityError(_INTEGRITY_ERROR) from exc

    return fail_closed


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} is invalid")
    return value.strip()


def _exact_object(
    value: object,
    fields: set[str],
    name: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def _one_artifact(value: object, name: str) -> dict[str, Any]:
    if type(value) is not list or len(value) != 1:
        raise ValueError(f"{name} must contain one artifact")
    return _exact_object(value[0], _ARTIFACT_FIELDS, f"{name}[0]")


def _artifact_output(
    *,
    task_id: str,
    artifact_id: str,
    kind: str,
    filename: str,
    content_hash: str,
    include_expected_hash: bool = False,
) -> dict[str, str]:
    url = (
        f"/api/tasks/{quote(task_id, safe='')}"
        f"/task-artifacts/{quote(artifact_id, safe='')}/download"
    )
    if include_expected_hash:
        url += f"?expected_content_hash={quote(content_hash, safe='')}"
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "format": "json",
        "filename": filename,
        "content_hash": content_hash,
        "download_url": url,
    }


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        detail = ""
        if isinstance(actual, Mapping) and isinstance(expected, Mapping):
            keys = sorted(
                str(key)
                for key in set(actual) | set(expected)
                if actual.get(key) != expected.get(key)
            )
            detail = ": " + ", ".join(keys)
        raise ValueError(
            f"{name} drifted from authenticated evidence{detail}"
        )


def _identity_table(title: str, rows: list[list[object]]) -> dict[str, Any]:
    return {
        "title": title,
        "columns": ["字段", "值"],
        "rows": [[str(label), str(value)] for label, value in rows],
    }


def _artifact_table(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return _identity_table(
        "任务内不可变产物",
        [
            ["Kind", artifact["kind"]],
            ["Artifact ID", artifact["artifact_id"]],
            ["Content Hash", artifact["content_hash"]],
            ["Filename", artifact["filename"]],
        ],
    )


def _validated_cross_search(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
):
    value = _exact_object(output, _CROSS_SEARCH_FIELDS, "Cross search output")
    artifact = _one_artifact(value["artifacts"], "Cross search artifacts")
    binding = load_cross_candidate_search_artifact(
        runtime,
        task_id=task_id,
        artifact_id=artifact["artifact_id"],
        expected_artifact_content_hash=artifact["content_hash"],
        expected_search_id=value["search_id"],
        expected_search_content_hash=value["content_hash"],
    )
    result = binding.result
    request = _validate_search_inputs(trusted_inputs)
    provenance = binding.artifact_provenance
    if (
        provenance["source_artifact_id"] != request["source_artifact_id"]
        or provenance["source_artifact_content_hash"]
        != request["expected_artifact_content_hash"]
        or provenance["candidate_id"] != request["expected_candidate_id"]
        or provenance["evidence_hash"] != request["expected_evidence_hash"]
        or sorted(item["feature"] for item in provenance["features"])
        != request["features"]
        or provenance["max_pairs"] != request["max_pairs"]
    ):
        raise ValueError("Cross search producer inputs drifted")
    expected_artifact = _artifact_output(
        task_id=task_id,
        artifact_id=binding.artifact_id,
        kind=CROSS_CANDIDATE_SEARCH_ARTIFACT_KIND,
        filename=binding.artifact_path.name,
        content_hash=binding.artifact_content_hash,
    )
    expected = {
        "schema_version": CROSS_CANDIDATE_SEARCH_TOOL_SCHEMA_VERSION,
        "search_id": result["search_id"],
        "request_hash": result["request_hash"],
        "content_hash": result["content_hash"],
        "source_artifact_id": binding.source.artifact_id,
        "candidate_id": binding.evidence["candidate_id"],
        "evidence_hash": binding.evidence["evidence_hash"],
        "population_count": result["population"]["row_count"],
        "search_space": result["search_space"],
        "evaluated": result["evaluated"],
        "truncated": result["truncated"],
        "eligible": result["eligible"],
        "search_result": result,
        "artifacts": [expected_artifact],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    _require_equal(value, expected, "Cross search output")
    return value, binding, expected_artifact


def _validated_cross_candidate(
    value: object,
    *,
    runtime: object,
    task_id: str,
    search_binding,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _exact_object(
        value,
        _CROSS_CANDIDATE_FIELDS,
        "Cross candidate output",
    )
    artifact = _one_artifact(candidate["artifacts"], "Cross candidate artifacts")
    record = runtime.task_artifacts.get_for_task(task_id, artifact["artifact_id"])
    if (
        not isinstance(record, Mapping)
        or record.get("id") != artifact["artifact_id"]
        or record.get("task_id") != task_id
        or record.get("kind") != cross_matrix_tools.ASSET_ARTIFACT_KIND
        or record.get("origin_tool") != cross_matrix_tools.ORIGIN_TOOL
        or record.get("content_hash") != artifact["content_hash"]
    ):
        raise ValueError("Cross candidate artifact registry binding drifted")
    path = Path(_text(record.get("path"), "Cross candidate artifact path"))
    tasks_root = Path(runtime.settings.tasks_dir).absolute()
    candidate_asset_tools._require_regular_artifact_path(path, root=tasks_root)
    candidate_asset_tools._require_file_content_hash(
        path,
        artifact["content_hash"],
        "Cross candidate artifact content hash changed",
    )
    persisted = parse_cross_matrix_candidate_asset_json(path.read_bytes())
    persisted = rebuild_cross_matrix_candidate_asset(
        persisted,
        search_binding.evidence,
    )
    _require_equal(
        candidate["cross_matrix_candidate"],
        persisted,
        "Cross candidate embedded asset",
    )
    provenance = record.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != cross_matrix_tools._ASSET_PROVENANCE_FIELDS
    ):
        raise ValueError("Cross candidate artifact provenance is invalid")
    parameters = search_binding.evidence["generation"]["parameters"]
    lifecycle = cross_matrix_tools._asset_lifecycle(persisted)
    expected_artifact = _artifact_output(
        task_id=task_id,
        artifact_id=str(record["id"]),
        kind=cross_matrix_tools.ASSET_ARTIFACT_KIND,
        filename=path.name,
        content_hash=str(record["content_hash"]),
    )
    expected = {
        "schema_version": (
            cross_matrix_tools.TOOL_V2_SCHEMA_VERSION
            if persisted["schema_version"]
            == CROSS_MATRIX_CANDIDATE_ASSET_V2_SCHEMA_VERSION
            else cross_matrix_tools.TOOL_SCHEMA_VERSION
        ),
        "asset_id": persisted["asset_id"],
        "asset_hash": persisted["asset_hash"],
        "parent_candidate_id": search_binding.evidence["candidate_id"],
        "parent_evidence_hash": search_binding.evidence["evidence_hash"],
        "candidate_id": persisted["candidate_evidence"]["candidate_id"],
        "evidence_hash": persisted["candidate_evidence"]["evidence_hash"],
        "dataset_id": search_binding.dataset.dataset_id,
        "target_col": persisted["sample_identity"]["target_col"],
        "population_count": search_binding.dataset.row_count,
        "labeled_count": persisted["sample_identity"]["row_count"],
        "drop_nan_labels": parameters["drop_nan_labels"],
        "nan_labels_dropped": parameters["nan_labels_dropped"],
        "row_axis": cross_matrix_tools._axis_projection(persisted["axes"][0]),
        "column_axis": cross_matrix_tools._axis_projection(persisted["axes"][1]),
        "cell_count": len(persisted["matrix"]["cells"]),
        "candidate_stage": lifecycle["candidate_stage"],
        "observation_stage": lifecycle["observation_stage"],
        "validation_status": lifecycle["validation_status"],
        "cross_matrix_candidate": persisted,
        "artifacts": [expected_artifact],
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    _require_equal(candidate, expected, "Cross candidate output")
    return candidate, expected_artifact


def _validated_cross_selection(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
):
    value = _exact_object(
        output,
        _CROSS_SELECTION_FIELDS,
        "Cross selection output",
    )
    selection = _exact_object(
        value["source_search_selection"],
        {
            "search_id",
            "pair_id",
            "rank",
            "x_feature",
            "x_method",
            "y_feature",
            "y_method",
            "eligible",
        },
        "Cross source search selection",
    )
    request = _validate_selection_inputs(trusted_inputs)
    if (
        selection["search_id"] != request["search_id"]
        or selection["pair_id"] != request["pair_id"]
    ):
        raise ValueError("Cross selection producer inputs drifted")
    binding, pair = resolve_cross_candidate_search_pair(
        runtime,
        task_id=task_id,
        search_id=selection["search_id"],
        pair_id=selection["pair_id"],
    )
    expected_selection = {
        "search_id": binding.result["search_id"],
        "pair_id": pair["pair_id"],
        "rank": pair["rank"],
        "x_feature": pair["x_feature"],
        "x_method": pair["x_method"],
        "y_feature": pair["y_feature"],
        "y_method": pair["y_method"],
        "eligible": pair["eligible"],
    }
    _require_equal(selection, expected_selection, "Cross source search selection")
    candidate, artifact = _validated_cross_candidate(
        value["cross_matrix_candidate"],
        runtime=runtime,
        task_id=task_id,
        search_binding=binding,
    )
    expected = {
        "schema_version": CROSS_CANDIDATE_SEARCH_SELECTION_TOOL_SCHEMA_VERSION,
        "source_search_selection": expected_selection,
        "cross_matrix_candidate": candidate,
        "not_selected": True,
        "not_admitted": True,
        "not_applied": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    _require_equal(value, expected, "Cross selection output")
    return value, artifact


def _validated_split_search(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
):
    value = _exact_object(
        output,
        _SPLIT_SEARCH_FIELDS,
        "interactive-tree split search output",
    )
    artifact = _one_artifact(
        value["artifacts"],
        "interactive-tree split search artifacts",
    )
    binding = load_verified_interactive_tree_split_search(
        runtime,
        task_id=task_id,
        search_id=value["search_id"],
    )
    request = _validate_split_search_inputs(trusted_inputs)
    provenance = binding.provenance
    if (
        provenance["source_tree_id"] != request["source_tree_id"]
        or provenance["node_id"] != request["node_id"]
        or provenance["mode"] != request["mode"]
        or provenance["max_thresholds_per_feature"]
        != request["max_thresholds_per_feature"]
        or provenance["max_row_evaluations"]
        != request["max_row_evaluations"]
        or (
            request["features"] is not None
            and provenance["features"] != request["features"]
        )
    ):
        raise ValueError("interactive-tree split search producer inputs drifted")
    result = binding.result
    expected_artifact = _artifact_output(
        task_id=task_id,
        artifact_id=binding.artifact_id,
        kind=INTERACTIVE_TREE_SPLIT_SEARCH_ARTIFACT_KIND,
        filename=binding.path.name,
        content_hash=binding.content_hash,
    )
    expected = {
        "schema_version": INTERACTIVE_TREE_SPLIT_SEARCH_TOOL_SCHEMA_VERSION,
        "search_id": result["search_id"],
        "search_hash": result["search_hash"],
        "source_tree_id": result["source"]["source_tree_id"],
        "node_id": result["source"]["node_id"],
        "mode": binding.provenance["mode"],
        "feature_count": result["budget"]["feature_count"],
        "evaluated_candidates": result["budget"]["evaluated_candidates"],
        "eligible_candidates": sum(
            1 for candidate in result["candidates"] if candidate["eligible"]
        ),
        "truncated": result["budget"]["truncated"],
        "search_result": result,
        "artifacts": [expected_artifact],
        "winner_selected": False,
        "tree_modified": False,
    }
    _require_equal(value, expected, "interactive-tree split search output")
    _require_equal(artifact, expected_artifact, "split search artifact")
    return value, binding, expected_artifact


def _validated_revision(
    output: object,
    *,
    runtime: object,
    task_id: str,
    auto_continue: bool,
    trusted_inputs: Mapping[str, Any],
):
    fields = _AUTO_CONTINUE_FIELDS if auto_continue else _REVISION_FIELDS
    value = _exact_object(output, fields, "interactive-tree revision output")
    artifact = _one_artifact(
        value["artifacts"],
        "interactive-tree revision artifacts",
    )
    binding = load_verified_interactive_tree_revision(
        runtime,
        task_id=task_id,
        revision_id=value["revision_id"],
    )
    revision = binding.revision
    parent_revision = (
        None if not binding.ancestor_revisions else binding.ancestor_revisions[0]
    )
    source = _ResolvedRevisionSource(
        automatic_source=binding.automatic_source,
        parent_revision=parent_revision,
        ancestor_revisions=(
            () if parent_revision is None else binding.ancestor_revisions[1:]
        ),
        source_tree_id=binding.provenance["source_tree_id"],
    )
    edit = revision["edit"]
    request = {
        "source_tree_id": source.source_tree_id,
        "node_id": edit["node_id"],
        "operation": edit["operation"],
        "reason": edit.get("reason"),
    }
    if edit["operation"] in {
        "adjust_split_threshold",
        "replace_split_feature",
    }:
        request["threshold"] = edit["threshold"]
    if edit["operation"] == "replace_split_feature":
        request["feature"] = edit["feature"]
    if edit["operation"] == "auto_continue_subtree":
        request.update(
            {
                "search_id": edit["search_id"],
                "candidate_id": edit["candidate_id"],
                **edit["controls"],
                "objective": edit["objective"],
                "tie_break": edit["tie_break"],
            }
        )
    if auto_continue:
        expected_request = _validate_auto_continuation_inputs(trusted_inputs)
        expected_request.setdefault("reason", None)
        actual_request = {
            key: request.get(key)
            for key in expected_request
        }
    else:
        expected_request = _validate_revision_inputs(trusted_inputs)
        expected_request.setdefault("reason", None)
        actual_request = dict(request)
    if actual_request != expected_request:
        raise ValueError("interactive-tree revision producer inputs drifted")
    rebuilt_revision, replay_binding = _build_revision(
        source,
        request=request,
        runtime=runtime,
        task_id=task_id,
    )
    _require_equal(revision, rebuilt_revision, "interactive-tree revision")
    replay = replay_binding.evidence
    expected_artifact = _artifact_output(
        task_id=task_id,
        artifact_id=binding.artifact_id,
        kind=INTERACTIVE_TREE_REVISION_ARTIFACT_KIND,
        filename=binding.path.name,
        content_hash=binding.content_hash,
    )
    expected = {
        "schema_version": (
            TOOL_SCHEMA_VERSION_V2
            if revision["schema_version"]
            == INTERACTIVE_TREE_REVISION_V2_SCHEMA_VERSION
            else TOOL_SCHEMA_VERSION
        ),
        "revision_id": revision["revision_id"],
        "revision_hash": revision["revision_hash"],
        "semantic_tree_id": revision["semantic_tree_id"],
        "tree_hash": revision["tree"]["tree_hash"],
        "source_tree_id": binding.provenance["source_tree_id"],
        "base_asset_id": revision["base_tree"]["asset_id"],
        "parent_revision_id": (
            None
            if revision["parent_revision"] is None
            else revision["parent_revision"]["revision_id"]
        ),
        "edit": revision["edit"],
        "visible_node_count": len(revision["tree"]["visible_node_ids"]),
        "frontier_node_count": len(revision["tree"]["frontier_node_ids"]),
        "replay": replay,
        "artifacts": [expected_artifact],
    }
    if auto_continue:
        search = load_verified_interactive_tree_split_search(
            runtime,
            task_id=task_id,
            search_id=value["search_id"],
        )
        candidate = next(
            (
                item
                for item in search.result["candidates"]
                if item["candidate_id"] == value["candidate_id"]
            ),
            None,
        )
        if candidate is None or candidate["eligible"] is not True:
            raise ValueError("automatic continuation candidate is not authenticated")
        expected.update(
            {
                "schema_version": AUTO_CONTINUE_TOOL_SCHEMA_VERSION,
                "search_id": search.result["search_id"],
                "search_hash": search.result["search_hash"],
                "candidate_id": candidate["candidate_id"],
                "automatic_winner_selection": False,
                "pool_modified": False,
            }
        )
    _require_equal(value, expected, "interactive-tree revision output")
    _require_equal(artifact, expected_artifact, "revision artifact")
    return value, expected_artifact


def _validated_impact_cube(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    trusted = _exact_object(
        trusted_artifacts,
        {"impact_cube"},
        "trusted ImpactCube artifacts",
    )
    container = _exact_object(
        trusted["impact_cube"],
        {"record"},
        "trusted ImpactCube artifact",
    )
    record = container["record"]
    if not isinstance(record, Mapping):
        raise ValueError("trusted ImpactCube record is invalid")
    provenance = record.get("provenance")
    if (
        record.get("task_id") != task_id
        or record.get("kind") != IMPACT_CUBE_ARTIFACT_KIND
        or record.get("origin_tool") != IMPACT_CUBE_ORIGIN_TOOL
        or not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("producer_run"), Mapping)
    ):
        raise ValueError("trusted ImpactCube registry binding is invalid")
    producer_run = provenance["producer_run"]
    value = validate_measure_strategy_impact_cube_tool_output(
        output,
        trusted_task_id=task_id,
        trusted_artifact_id=record["id"],
        trusted_artifact_content_hash=record["content_hash"],
        trusted_producer_run_id=producer_run["run_id"],
        trusted_producer_run_content_hash=producer_run["content_hash"],
    )
    live = runtime.task_artifacts.get_for_task(task_id, record["id"])
    _require_equal(dict(record), dict(live), "trusted ImpactCube registry record")
    validate_impact_cube_producer_run(
        producer_run,
        expected_task_id=task_id,
        expected_request=trusted_inputs,
        expected_cube_id=value["cube_id"],
        expected_cube_content_hash=value["content_hash"],
        expected_artifact_id=record["id"],
        expected_artifact_filename=value["artifact"]["filename"],
        expected_artifact_content_hash=record["content_hash"],
    )
    return value


@_presenter
def _render_search_cross_matrix_candidates(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    del trusted_artifacts
    value, _binding, artifact = _validated_cross_search(
        output,
        runtime=runtime,
        task_id=task_id,
        trusted_inputs=trusted_inputs,
    )
    text = (
        "**Cross Matrix 候选搜索证据已认证。**"
        f"搜索 `{value['search_id']}` 共评估 {value['evaluated']} 个特征对，"
        f"其中 {value['eligible']} 个满足搜索约束。"
        "排名只用于证据导航，没有自动选择或形成策略结论；"
        "未入池、未采纳、未部署。"
    )
    return text, [
        _identity_table(
            "Cross Matrix 搜索身份",
            [
                ["Search ID", value["search_id"]],
                ["Search Content Hash", value["content_hash"]],
                ["Request Hash", value["request_hash"]],
                ["Source Artifact ID", value["source_artifact_id"]],
                ["Candidate ID", value["candidate_id"]],
                ["Evidence Hash", value["evidence_hash"]],
                ["Search Space", value["search_space"]],
                ["Evaluated", value["evaluated"]],
                ["Eligible", value["eligible"]],
                ["Truncated", value["truncated"]],
            ],
        ),
        _artifact_table(artifact),
    ]


@_presenter
def _render_build_cross_matrix_candidate_from_search(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    del trusted_artifacts
    value, artifact = _validated_cross_selection(
        output,
        runtime=runtime,
        task_id=task_id,
        trusted_inputs=trusted_inputs,
    )
    selection = value["source_search_selection"]
    candidate = value["cross_matrix_candidate"]
    text = (
        "**Cross Matrix 搜索候选已精确物化并认证。**"
        f"候选 `{candidate['asset_id']}` 绑定搜索 `{selection['search_id']}` "
        f"中的 pair `{selection['pair_id']}`。"
        "来源 rank 与 eligible 仅是确定性搜索证据，不构成自动推荐；"
        "未入池、未采纳、未部署。"
    )
    return text, [
        _identity_table(
            "Cross Matrix 搜索候选身份",
            [
                ["Search ID", selection["search_id"]],
                ["Pair ID", selection["pair_id"]],
                ["Rank", selection["rank"]],
                ["Eligible", selection["eligible"]],
                ["Asset ID", candidate["asset_id"]],
                ["Asset Hash", candidate["asset_hash"]],
                ["Candidate ID", candidate["candidate_id"]],
                ["Evidence Hash", candidate["evidence_hash"]],
            ],
        ),
        _artifact_table(artifact),
    ]


@_presenter
def _render_search_interactive_tree_split_candidates(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    del trusted_artifacts
    value, _binding, artifact = _validated_split_search(
        output,
        runtime=runtime,
        task_id=task_id,
        trusted_inputs=trusted_inputs,
    )
    text = (
        "**交互式决策树切分搜索证据已认证。**"
        f"搜索 `{value['search_id']}` 在节点 `{value['node_id']}` 评估 "
        f"{value['evaluated_candidates']} 个候选，其中 "
        f"{value['eligible_candidates']} 个满足约束。"
        "搜索没有选择赢家、没有修改树；未入池、未采纳、未部署。"
    )
    return text, [
        _identity_table(
            "交互式决策树切分搜索身份",
            [
                ["Search ID", value["search_id"]],
                ["Search Hash", value["search_hash"]],
                ["Source Tree ID", value["source_tree_id"]],
                ["Node ID", value["node_id"]],
                ["Mode", value["mode"]],
                ["Feature Count", value["feature_count"]],
                ["Evaluated Candidates", value["evaluated_candidates"]],
                ["Eligible Candidates", value["eligible_candidates"]],
                ["Truncated", value["truncated"]],
            ],
        ),
        _artifact_table(artifact),
    ]


def _revision_tables(
    value: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    auto_continue: bool,
) -> list[dict[str, Any]]:
    rows: list[list[object]] = [
        ["Revision ID", value["revision_id"]],
        ["Revision Hash", value["revision_hash"]],
        ["Semantic Tree ID", value["semantic_tree_id"]],
        ["Tree Hash", value["tree_hash"]],
        ["Source Tree ID", value["source_tree_id"]],
        ["Base Asset ID", value["base_asset_id"]],
        ["Parent Revision ID", value["parent_revision_id"]],
        ["Edit Operation", value["edit"]["operation"]],
        ["Edit Node ID", value["edit"]["node_id"]],
        ["Visible Node Count", value["visible_node_count"]],
        ["Frontier Node Count", value["frontier_node_count"]],
        ["Replay Result Hash", value["replay"]["result_hash"]],
    ]
    if auto_continue:
        rows.extend(
            [
                ["Search ID", value["search_id"]],
                ["Search Hash", value["search_hash"]],
                ["Candidate ID", value["candidate_id"]],
            ]
        )
    return [
        _identity_table("交互式决策树修订身份", rows),
        _artifact_table(artifact),
    ]


@_presenter
def _render_revise_interactive_tree(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    del trusted_artifacts
    value, artifact = _validated_revision(
        output,
        runtime=runtime,
        task_id=task_id,
        auto_continue=False,
        trusted_inputs=trusted_inputs,
    )
    text = (
        "**交互式决策树修订已认证。**"
        f"修订 `{value['revision_id']}` 已对开发样本完成确定性重放，"
        f"操作为 `{value['edit']['operation']}`。"
        "该修订仍是候选探索证据；未入池、未采纳、未部署。"
    )
    return text, _revision_tables(value, artifact, auto_continue=False)


@_presenter
def _render_auto_continue_interactive_tree(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    del trusted_artifacts
    value, artifact = _validated_revision(
        output,
        runtime=runtime,
        task_id=task_id,
        auto_continue=True,
        trusted_inputs=trusted_inputs,
    )
    text = (
        "**交互式决策树受控续建修订已认证。**"
        f"修订 `{value['revision_id']}` 精确绑定搜索 `{value['search_id']}` "
        f"和候选 `{value['candidate_id']}`，并已完成确定性重放。"
        "平台未自动选择赢家，也未修改策略池；未入池、未采纳、未部署。"
    )
    return text, _revision_tables(value, artifact, auto_continue=True)


@_presenter
def _render_measure_strategy_impact_cube(
    output: object,
    *,
    runtime: object,
    task_id: str,
    trusted_inputs: Mapping[str, Any],
    trusted_artifacts: Mapping[str, Any] | None,
):
    value = _validated_impact_cube(
        output,
        runtime=runtime,
        task_id=task_id,
        trusted_inputs=trusted_inputs,
        trusted_artifacts=trusted_artifacts,
    )
    artifact = value["artifact"]
    text = (
        "**策略 ImpactCube 已认证。**"
        f"Cube `{value['cube_id']}` 固化了 {value['slice_count']} 个确定性切片，"
        "只用于后续稳定性与影响分析。"
        "本步骤未修改策略池、未创建或晋升策略；未入池、未采纳、未部署。"
    )
    return text, [
        _identity_table(
            "ImpactCube 身份",
            [
                ["Cube ID", value["cube_id"]],
                ["Cube Content Hash", value["content_hash"]],
                ["Pool ID", value["pool_id"]],
                ["Pool Revision", value["pool_revision"]],
                ["Pool Snapshot Hash", value["pool_snapshot_hash"]],
                ["Strategy Type", value["strategy_type"]],
                ["Partitions", "、".join(value["partitions"])],
                ["Slice Count", value["slice_count"]],
                ["Producer Run ID", value["producer_run_ref"]["ref_id"]],
                [
                    "Producer Run Content Hash",
                    value["producer_run_ref"]["content_hash"],
                ],
            ],
        ),
        _artifact_table(artifact),
    ]


STRATEGY_EXPLORATION_PRESENTERS = {
    "search_cross_matrix_candidates": _render_search_cross_matrix_candidates,
    "build_cross_matrix_candidate_from_search": (
        _render_build_cross_matrix_candidate_from_search
    ),
    "search_interactive_tree_split_candidates": (
        _render_search_interactive_tree_split_candidates
    ),
    "auto_continue_interactive_tree": _render_auto_continue_interactive_tree,
    "revise_interactive_tree": _render_revise_interactive_tree,
    "measure_strategy_impact_cube": _render_measure_strategy_impact_cube,
}


__all__ = ["STRATEGY_EXPLORATION_PRESENTERS"]
