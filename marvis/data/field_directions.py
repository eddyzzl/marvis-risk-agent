"""Explicit field meaning declarations, separate from measured risk direction."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import re


FIELD_RISK_DIRECTIONS = (
    "unknown", "higher_is_riskier", "lower_is_riskier", "non_monotonic",
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = object()


@dataclass(frozen=True)
class ApprovedFieldDirections:
    binding_json: str
    _token: object

    def payload(self):
        from marvis.llm_outbound import OutboundContextError
        if self._token is not _TOKEN:
            raise OutboundContextError("字段方向声明来源无效，未发送请求。")
        return json.loads(self.binding_json)


def approved_field_directions(registry, task_id):
    """Mint outbound metadata only from a reauthenticated active workspace."""
    from marvis.data.registry import DatasetRegistry
    from marvis.data.workspace import data_semantic_mapping_hash
    from marvis.llm_outbound import OutboundContextError
    from marvis.llm_outbound_instructions import column_identifiers
    from marvis.repositories.data_workspace import DataWorkspaceRepository
    if not isinstance(registry, DatasetRegistry):
        raise OutboundContextError("字段方向需要已登记的任务数据来源，未发送请求。")
    snapshot = DataWorkspaceRepository(registry._repo.db_path).get_or_default(task_id)
    if not snapshot.active_dataset_id:
        return None
    dataset = registry.get(snapshot.active_dataset_id)
    if dataset.task_id != task_id or dataset.content_hash != snapshot.active_dataset_content_hash:
        raise OutboundContextError("字段方向活动来源已变化，未发送请求。")
    registry.resolve_verified_path(dataset.id)
    rows = confirmed_field_directions(snapshot, dataset)
    column_identifiers([row["column"] for row in rows])
    return ApprovedFieldDirections(json.dumps({"fields": rows, "dataset_id": dataset.id,
        "dataset_content_hash": dataset.content_hash, "workspace_revision": snapshot.revision,
        "semantic_mapping_hash": data_semantic_mapping_hash(snapshot.semantic_mapping)},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False), _TOKEN)


@dataclass(frozen=True)
class FieldDirectionDeclaration:
    direction: str
    source: str
    actor_id: str
    confirmed_at: str
    dataset_id: str
    dataset_content_hash: str
    source_workspace_revision: int
    source_semantic_mapping_hash: str

    def __post_init__(self):
        if self.direction not in FIELD_RISK_DIRECTIONS or self.source != "user_confirmed":
            raise ValueError("field direction needs an explicit supported user declaration")
        for key in ("actor_id", "dataset_id", "confirmed_at"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or value != value.strip() or len(value) > 256:
                raise ValueError("field direction declaration identity is invalid")
        for key in ("dataset_content_hash", "source_semantic_mapping_hash"):
            if not isinstance(getattr(self, key), str) or not _SHA.fullmatch(getattr(self, key)):
                raise ValueError("field direction declaration digest is invalid")
        if type(self.source_workspace_revision) is not int or self.source_workspace_revision < 0:
            raise ValueError("field direction source revision is invalid")
        try:
            stamp = datetime.fromisoformat(self.confirmed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("field direction confirmation time is invalid") from exc
        if stamp.tzinfo is None:
            raise ValueError("field direction confirmation time needs a timezone")


def normalize_field_directions(value):
    if not isinstance(value, Mapping) or len(value) > 10000:
        raise ValueError("field directions must be a bounded field mapping")
    result = {}
    for column, declaration in value.items():
        if not isinstance(column, str) or not column or column != column.strip() or len(column) > 256:
            raise ValueError("field direction needs a physical column name")
        if isinstance(declaration, Mapping):
            declaration = FieldDirectionDeclaration(**declaration)
        if not isinstance(declaration, FieldDirectionDeclaration):
            raise ValueError("field direction declaration is invalid")
        result[column] = declaration
    return result


def field_directions_to_dict(value):
    return {column: asdict(declaration) for column, declaration in value.items()}


def confirmation_hash(declaration):
    return hashlib.sha256(json.dumps(asdict(declaration), sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def confirmed_field_directions(snapshot, dataset):
    """Only exact physical fields and bounded confirmation facts may be consumed.

    A declaration copied through a derived dataset keeps its provenance locally,
    but the new dataset needs explicit confirmation. Names never establish meaning.
    """
    result = []
    for column in dataset.columns:
        name = column.name
        declaration = snapshot.semantic_mapping.risk_directions.get(name)
        current = declaration is not None and (
            declaration.dataset_id == dataset.id
            and declaration.dataset_content_hash == dataset.content_hash
            and snapshot.active_dataset_id == dataset.id
            and snapshot.active_dataset_content_hash == dataset.content_hash
        )
        row = {"column": name, "direction": declaration.direction if current else "unknown",
            "status": "confirmed" if current else "requires_confirmation" if declaration else "undeclared",
            "source": "user_confirmed" if current else "unknown", "actor_confirmed": bool(current)}
        if current:
            row.update(confirmation_hash=confirmation_hash(declaration), dataset_id=dataset.id,
                dataset_content_hash=dataset.content_hash,
                source_workspace_revision=declaration.source_workspace_revision)
        result.append(row)
    return result


def confirm_field_directions(settings, *, task_id, actor_id, expected_revision,
    expected_semantic_mapping_hash, dataset_id, dataset_content_hash, directions):
    """Confirm choices under one workspace transaction and source-byte CAS."""
    from marvis.data.backend import DataBackend
    from marvis.data.registry import DatasetRegistry
    from marvis.data.workspace import DataWorkspaceDraft, data_semantic_mapping_hash
    from marvis.db_schema import connect
    from marvis.repositories.datasets import DatasetRepository
    from marvis.repositories.data_workspace import DataWorkspaceRepository, DataWorkspaceRevisionConflict

    if not isinstance(directions, Mapping) or not 0 < len(directions) <= 128:
        raise ValueError("明确选择 1 至 128 个字段后确认风险方向。")
    if any(value not in FIELD_RISK_DIRECTIONS for value in directions.values()):
        raise ValueError("风险方向必须明确选择，不能从字段名称推断。")
    repository = DataWorkspaceRepository(settings.db_path)
    registry = DatasetRegistry(DatasetRepository(settings.db_path), DataBackend(settings.datasets_dir), settings.datasets_dir)
    snapshot = repository.get_or_default(task_id)
    if snapshot.revision != expected_revision or data_semantic_mapping_hash(snapshot.semantic_mapping) != expected_semantic_mapping_hash:
        raise DataWorkspaceRevisionConflict("字段语义版本已变化，请刷新后重新确认。")
    if (snapshot.active_dataset_id, snapshot.active_dataset_content_hash) != (dataset_id, dataset_content_hash):
        raise DataWorkspaceRevisionConflict("活动数据来源已变化，请重新确认字段方向。")
    dataset = registry.get(dataset_id)
    if dataset.task_id != task_id or dataset.content_hash != dataset_content_hash:
        raise ValueError("字段方向来源不属于当前任务或内容已变化。")
    columns = {column.name for column in dataset.columns}
    if set(directions) - columns:
        raise ValueError("请使用活动数据的物理字段；业务别名不能替代字段身份。")
    declarations = dict(snapshot.semantic_mapping.risk_directions)
    # A current same-actor replay retains the original confirmation identity.
    def already_confirmed(name, direction):
        old = declarations.get(name)
        return old is not None and (old.direction, old.actor_id, old.dataset_id,
            old.dataset_content_hash) == (direction, actor_id, dataset_id, dataset_content_hash)
    changed = {name: direction for name, direction in directions.items() if not already_confirmed(name, direction)}
    stamp = datetime.now(timezone.utc).isoformat()
    for name, direction in changed.items():
        declarations[name] = FieldDirectionDeclaration(direction, "user_confirmed", actor_id,
            stamp, dataset_id, dataset_content_hash, expected_revision, expected_semantic_mapping_hash)
    draft = DataWorkspaceDraft(active_dataset_id=dataset_id, active_dataset_content_hash=dataset_content_hash,
        page=snapshot.page, selected_field=snapshot.selected_field,
        semantic_mapping=replace(snapshot.semantic_mapping, risk_directions=declarations))
    def verify():
        live = registry.get(dataset_id)
        if live.task_id != task_id or live.content_hash != dataset_content_hash:
            raise ValueError("字段方向来源已变化，未保存确认。")
        registry.resolve_verified_path(dataset_id)
    with connect(settings.db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        return repository.save_on_connection(connection, task_id, draft, expected_revision,
            dataset_authenticator=verify, audit={"actor": actor_id, "detail": {
                "action": "field_direction_confirm", "fields": sorted(changed),
                "source_semantic_mapping_hash": expected_semantic_mapping_hash}})
