"""Registered file schema and role bindings for the initial AUTO gate."""

from dataclasses import dataclass
import json

from marvis.llm_outbound import OutboundContextError, _DROP, _identifier
from marvis.llm_outbound_instructions import column_identifiers

_TOKEN = object()
MAX_ROLE_FILES = 1000


@dataclass(frozen=True)
class ApprovedRoleAssignment:
    binding_json: str
    _token: object

    def payload(self):
        if self._token is not _TOKEN:
            raise OutboundContextError("文件角色节点来源无效，未发送请求。")
        return json.loads(self.binding_json)


def approved_role_assignment(repository, settings, *, task_id, gate_metadata):
    from marvis.repositories.tasks import TaskRepository
    from marvis.repositories.datasets import DatasetRepository
    from marvis.data.registry import DatasetRegistry
    from marvis.data.backend import DataBackend
    from marvis.data.errors import DataLayerError
    from marvis.agent.join_setup import _proposal_columns, _strong_target_candidates

    if (
        not isinstance(repository, TaskRepository)
        or _identifier(task_id, opaque=True) is _DROP
    ):
        raise OutboundContextError("文件角色需要当前任务的注册材料来源，未发送请求。")
    c1 = gate_metadata.get("join_c1") if isinstance(gate_metadata, dict) else None
    cards = [
        message.get("metadata", {}).get("join_c1")
        for message in repository.list_agent_messages(task_id)
        if message.get("role") == "assistant"
        and isinstance(message.get("metadata", {}).get("join_c1"), dict)
    ]
    if not isinstance(c1, dict) or not cards or c1 != cards[-1]:
        raise OutboundContextError("文件角色卡片已变化，未发送请求。")
    files = c1.get("files")
    if not isinstance(files, list) or not 0 < len(files) <= MAX_ROLE_FILES:
        raise OutboundContextError("文件角色材料数量为空或超限，未发送请求。")
    root = settings.datasets_dir
    registry = DatasetRegistry(
        DatasetRepository(repository.db_path), DataBackend(root), root
    )
    schemas = []
    ids = set()
    for item in files:
        identity = item.get("dataset_id") if isinstance(item, dict) else None
        if _identifier(identity, opaque=True) is _DROP or identity in ids:
            raise OutboundContextError("文件角色材料身份无效，未发送请求。")
        try:
            dataset = registry.get(identity)
        except (DataLayerError, KeyError) as exc:
            raise OutboundContextError("文件角色注册材料不可用，未发送请求。") from exc
        columns = column_identifiers([column.name for column in dataset.columns])
        expected_candidates = _strong_target_candidates(dataset)
        if (
            dataset.task_id != task_id
            or dataset.content_hash != item.get("content_hash")
            or dataset.row_count != item.get("row_count")
            or len(columns) != item.get("n_cols")
            or _proposal_columns(dataset, expected_candidates) != item.get("columns")
            or (bool(expected_candidates) or dataset.has_target)
            != item.get("has_target")
        ):
            raise OutboundContextError("文件角色注册材料或字段已变化，未发送请求。")
        try:
            registry.resolve_verified_path(identity)
        except (DataLayerError, OSError, ValueError, KeyError) as exc:
            raise OutboundContextError(
                "文件角色注册材料内容已变化，未发送请求。"
            ) from exc
        candidates = column_identifiers(item.get("target_candidates", []))
        if candidates != expected_candidates:
            raise OutboundContextError(
                "文件角色目标列候选不属于已注册字段，未发送请求。"
            )
        ids.add(identity)
        schemas.append(
            {
                "dataset_id": identity,
                "content_hash": dataset.content_hash,
                "row_count": dataset.row_count,
                "column_count": len(columns),
                "columns": columns,
                "has_target": dataset.has_target,
                "target_candidates": candidates,
            }
        )
    anchor = c1.get("anchor_id")
    features = c1.get("feature_ids")
    target = c1.get("target_col")
    if (
        _identifier(anchor, opaque=True) is _DROP
        or anchor not in ids
        or not isinstance(features, list)
        or any(_identifier(identity, opaque=True) is _DROP for identity in features)
        or len(set(features)) != len(features)
        or any(identity not in ids or identity == anchor for identity in features)
    ):
        raise OutboundContextError("文件角色绑定身份无效，未发送请求。")
    anchor_columns = next(
        row["columns"] for row in schemas if row["dataset_id"] == anchor
    )
    if target is not None and column_identifiers([target])[0] not in anchor_columns:
        raise OutboundContextError("文件角色目标列不属于当前锚样本，未发送请求。")
    return ApprovedRoleAssignment(
        json.dumps(
            {
                "purpose": "accept_registered_file_roles",
                "application_contract": (
                    "当前节点尚未计算建模、权重、泄漏或拼接效果，只确认平台已展示的注册文件角色与目标列。"
                    "请审阅明确绑定的 anchor_id、feature_ids、target_col 和注册字段结构，决定是否接受该方案；"
                    "不能因尚无后续计算结果而要求补造指标。若模型目标列不唯一或角色仍需猜测，应停止澄清。"
                    "confirm 只采用当前已绑定角色，不新增字段、不换材料，也不授权后续人工决策或副作用。"
                ),
                "task_id": task_id,
                "anchor_id": anchor,
                "feature_ids": features,
                "target_col": target,
                "registered_files": schemas,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        _TOKEN,
    )
