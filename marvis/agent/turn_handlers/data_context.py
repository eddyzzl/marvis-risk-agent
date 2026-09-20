"""Data context for governed Agent turns."""

from __future__ import annotations

from marvis.agent.instruction_router import route_instruction
from marvis.agent.modeling_setup import supported_modeling_recipes
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.domain import TaskRecord
from marvis.llm_client import LLMClientError
from marvis.repositories.datasets import DatasetRepository
from . import contracts as contracts_lane


def _modeling_data_runtime(settings):
    datasets_root = getattr(settings, "datasets_dir", settings.workspace / "datasets")
    data_repo = DatasetRepository(settings.db_path)
    backend = DataBackend(datasets_root)
    registry = DatasetRegistry(data_repo, backend, datasets_root)
    return backend, registry


def _modeling_recipes(task: TaskRecord) -> list[str] | None:
    recipes = [
        str(item).strip()
        for item in (getattr(task, "recipes", None) or [])
        if str(item).strip()
    ]
    return recipes or None


def _modeling_intake_param_schema(task: TaskRecord) -> list[dict[str, object]]:
    return [
        {
            "name": "target_type",
            "type": "string",
            "current": _modeling_target_type(task) or "",
            "bounds": {
                "enum": ["binary", "continuous", "multiclass"],
            },
        },
        {
            "name": "recipes",
            "type": "array",
            "current": _modeling_recipes(task) or [],
            "bounds": {"enum": supported_modeling_recipes()},
        },
        {
            "name": "split_config",
            "type": "object",
            "current": {},
        },
        {
            "name": "n_trials",
            "type": "integer",
            "current": 1,
            "bounds": {"min": 1, "max": 200},
        },
        {
            "name": "sample_weight_col",
            "type": "string",
            "current": getattr(task, "sample_weight_col", "") or "",
        },
    ]


def _modeling_intake_params(
    runtime: contracts_lane.DriverTurnRuntime,
    task: TaskRecord,
    user_text: str | None,
) -> dict[str, object]:
    """Let the configured LLM bind first-turn modeling controls before planning.

    This reuses the same structured, declared-control router used at later
    confirmation gates. The deterministic modeling setup still validates target
    family, recipe ids, columns, and split behavior before a plan is created.
    """

    text = str(user_text or "").strip()
    if (
        str(getattr(task, "run_mode", "") or "") != "agent"
        or runtime.llm_client is None
        or not text
    ):
        return {}
    if runtime.workflow_intake_route is not None:
        route = dict(runtime.workflow_intake_route)
    else:
        try:
            route = route_instruction(
                runtime.llm_client,
                gate_context=(
                    "首轮建模规格收集：在创建计划前理解用户完整要求，只抽取当前"
                    "声明的建模控件；不新增、删除或重排工作流步骤。"
                ),
                instruction=text,
                param_schema=_modeling_intake_param_schema(task),
            )
        except LLMClientError:
            return {}
    if route.get("action") != "adjust":
        return {}
    params = route.get("params")
    if not isinstance(params, dict):
        return {}
    return {
        key: value
        for key, value in params.items()
        if key in contracts_lane._MODELING_INTAKE_PARAM_NAMES
    }


def _modeling_target_type(task: TaskRecord) -> str | None:
    target_type = str(getattr(task, "target_type", "") or "").strip()
    return target_type or None


def _modeling_success_criteria(task: TaskRecord) -> list[dict] | None:
    """AGT-4: turn the task's optional oot_ks_min into a deterministic success
    criterion final_review can evaluate. None/absent oot_ks_min (the default) means
    no criterion is injected — the platform never hard-codes a threshold; only a
    value the user (or AUTO, once wired) explicitly set produces one."""
    oot_ks_min = getattr(task, "oot_ks_min", None)
    if oot_ks_min is None:
        return None
    return [
        {
            "metric": "oot_ks",
            "min": float(oot_ks_min),
            "aggregate": "max",
            "label": "OOT KS",
            "target_type": "binary",
        }
    ]


def _modeling_field_hint_keywords(task: TaskRecord, c1_proposal) -> tuple[str, ...]:
    # MEM-4: scope the field_convention lookup to this task's own dataset file
    # names (+ model name) so a hint only ever comes from prior tasks that look
    # like they touched the same data, never an unrelated model's column names.
    values = [getattr(task, "model_name", None)]
    values.extend(
        getattr(item, "name", None)
        for item in getattr(c1_proposal, "files", None) or ()
    )
    return tuple(
        dict.fromkeys(
            str(value).strip() for value in values if str(value or "").strip()
        )
    )


def _modeling_project_meta(task: TaskRecord) -> dict[str, str]:
    meta: dict[str, str] = {}
    for key, value in (
        ("模型名称", getattr(task, "model_name", "")),
        ("模型版本", getattr(task, "model_version", "")),
        ("验证人", getattr(task, "validator", "")),
    ):
        text = str(value or "").strip()
        if text:
            meta[key] = text
    return meta
