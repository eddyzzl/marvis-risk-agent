"""The existing Plugin/Workflow surfaces share one signed admission service."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import APIRouter, Query, Request
from marvis.errors import domain_error, unprocessable
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from marvis.plugins.capability_versions import CapabilityStore
from marvis.plugins.errors import PluginError
from marvis.orchestrator.templates.skills import SkillTemplateError
from marvis.plugins.governed_install import confirm_proposal, propose_archive
from marvis.plugins.publisher_trust import CapabilityError, require


class PublisherRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=96)
    display_name: str = Field(min_length=1, max_length=120)
    public_key: str = Field(min_length=1, max_length=128)


class ReasonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: str = Field(min_length=1, max_length=2000)


class ModeRequest(ReasonRequest):
    mode: Literal["trusted_local", "governed"]
    expected_revision: StrictInt = Field(ge=0)


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    proposal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_revision: StrictInt = Field(ge=0)
    approved_permissions: list[str]


class RollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["plugin", "workflow"]
    name: str = Field(min_length=1, max_length=96)
    version: str = Field(min_length=1, max_length=96)
    expected_revision: StrictInt = Field(ge=1)


class EnabledRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["plugin", "workflow"]
    name: str = Field(min_length=1, max_length=96)
    enabled: bool
    expected_revision: StrictInt = Field(ge=1)


def _store(request):
    return CapabilityStore(request.app.state.settings.db_path)


def _actor(request):
    from marvis.routers.plugins import _require_plugin_admin

    _require_plugin_admin(request)
    value = str(getattr(getattr(request.state, "local_principal", None), "id", ""))
    require(bool(value), "server_session_principal_required", status=401)
    return value


def _refresh(request):
    from marvis.orchestrator.templates import (
        clear_user_templates,
        load_builtin_templates,
    )
    from marvis.orchestrator.templates.skills import load_user_skill_templates

    request.app.state.plugin_registry.load_from_db()
    request.app.state.hook_dispatcher.rebuild_index()
    load_builtin_templates()
    clear_user_templates()
    request.app.state.skill_report = load_user_skill_templates(
        request.app.state.settings.workspace,
        request.app.state.tool_registry,
        request.app.state.plan_validator,
    )


def _error(exc):
    raise domain_error(exc.status, exc.code) from exc


def governance_router(required_kind=None):
    router = APIRouter(prefix="/governance", tags=["capability-governance"])

    @router.get("")
    def overview(request: Request):
        store = _store(request)
        versions = store.versions()
        for item in versions:
            item["permissions"] = json.loads(item.pop("permissions_json"))
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )

            signature_available = bool(Ed25519PublicKey)
        except ImportError:
            signature_available = False
        return {
            "settings": store.settings(),
            "publishers": store.publishers(),
            "versions": versions,
            "signature_schema": "marvis.capability-signature.v1",
            "signature_verification_available": signature_available,
            "boundaries": {
                "identity": "workspace_server_session_and_admin_confirmation",
                "immutable_versions": "read_only_material_with_signature_and_hash_verification",
                "hostile_same_uid_os_isolation": False,
                "independent_skill_runtime": False,
            },
        }

    @router.put("/settings")
    def set_settings(body: ModeRequest, request: Request):
        try:
            result = _store(request).set_mode(
                body.mode, body.expected_revision, _actor(request), body.reason
            )
            _refresh(request)
            return result
        except CapabilityError as exc:
            _error(exc)

    @router.post("/publishers", status_code=201)
    def add_publisher(body: PublisherRequest, request: Request):
        try:
            return _store(request).add_publisher(
                body.id, body.display_name, body.public_key, _actor(request)
            )
        except CapabilityError as exc:
            _error(exc)

    @router.post("/publishers/{publisher_id}/revoke")
    def revoke_publisher(publisher_id: str, body: ReasonRequest, request: Request):
        try:
            result = _store(request).revoke(publisher_id, _actor(request), body.reason)
            _refresh(request)
            return result
        except CapabilityError as exc:
            _error(exc)

    @router.post("/proposals", status_code=201)
    async def propose(
        request: Request,
        expected_revision: int = Query(ge=0),
        kind: Literal["plugin", "workflow"] = required_kind or "plugin",
        expected_name: str | None = Query(default=None, max_length=96),
    ):
        from marvis.routers.plugins import _read_plugin_upload

        try:
            actor = _actor(request)
            require(
                required_kind is None or kind == required_kind,
                "capability_kind_changed",
                status=422,
            )
            _filename, content = await _read_plugin_upload(request)
            with tempfile.TemporaryDirectory(
                prefix="marvis-capability-upload-"
            ) as directory:
                path = Path(directory) / "capability.zip"
                path.write_bytes(content)
                return propose_archive(
                    path,
                    db_path=request.app.state.settings.db_path,
                    registry=request.app.state.plugin_registry,
                    tools=request.app.state.tool_registry,
                    validator=request.app.state.plan_validator,
                    actor=actor,
                    expected_revision=expected_revision,
                    kind=kind,
                    expected_name=expected_name,
                )
        except CapabilityError as exc:
            _error(exc)
        except (PluginError, SkillTemplateError) as exc:
            raise unprocessable("capability_package_invalid") from exc

    @router.get("/proposals/{proposal_id}")
    def get_proposal(proposal_id: str, request: Request):
        try:
            actor = _actor(request)
            result = _store(request).proposal(proposal_id)
            require(
                result["actor"] == actor,
                "capability_proposal_actor_changed",
                status=403,
            )
            require(
                required_kind is None or result["kind"] == required_kind,
                "capability_kind_changed",
                status=422,
            )
            return result
        except CapabilityError as exc:
            _error(exc)

    @router.post("/proposals/{proposal_id}/confirm", status_code=201)
    def confirm(proposal_id: str, body: ConfirmationRequest, request: Request):
        try:
            actor = _actor(request)
            proposal = _store(request).proposal(proposal_id)
            require(
                required_kind is None or proposal["kind"] == required_kind,
                "capability_kind_changed",
                status=422,
            )
            result = confirm_proposal(
                request.app.state.settings.db_path,
                request.app.state.plugin_registry,
                request.app.state.tool_registry,
                request.app.state.plan_validator,
                proposal_id,
                actor=actor,
                **body.model_dump(),
            )
            _refresh(request)
            return result
        except CapabilityError as exc:
            _error(exc)
        except (PluginError, SkillTemplateError) as exc:
            raise unprocessable("capability_package_invalid") from exc

    @router.post("/rollback-proposals", status_code=201)
    def rollback(body: RollbackRequest, request: Request):
        try:
            require(
                required_kind is None or body.kind == required_kind,
                "capability_kind_changed",
                status=422,
            )
            return _store(request).rollback_proposal(
                body.kind,
                body.name,
                body.version,
                body.expected_revision,
                _actor(request),
            )
        except CapabilityError as exc:
            _error(exc)

    @router.post("/enabled")
    def enabled(body: EnabledRequest, request: Request):
        try:
            require(
                required_kind is None or body.kind == required_kind,
                "capability_kind_changed",
                status=422,
            )
            result = _store(request).set_enabled(
                body.kind,
                body.name,
                body.enabled,
                _actor(request),
                body.expected_revision,
            )
            _refresh(request)
            return result
        except CapabilityError as exc:
            _error(exc)

    return router
