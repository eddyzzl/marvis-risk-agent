"""Explicit signed transfer, immutable readback and head-CAS rollback APIs."""

from typing import Literal

from fastapi import APIRouter,Request
from pydantic import Field

from marvis.institution.router import StrictBody,_admin
from marvis.institution.policy import require_human_principal
from marvis.institution.synchronization import SynchronizationStore


router = APIRouter(prefix='/api/instance-sync',tags=['instance-sync'])


class Node(StrictBody):
    id: str
    public_key: str


class Export(StrictBody):
    kind: Literal['model','strategy','workflow']
    native_ref: str
    name: str
    release_version: str
    receiver_node: str


class Import(StrictBody):
    envelope: dict
    material: dict
    task_id: str
    expected_revision: int = Field(ge=0)
    approved_permissions: list[str]


class Rollback(StrictBody):
    version_id: str
    expected_revision: int = Field(ge=1)
    approved_permissions: list[str]


class NodeRevocation(StrictBody):
    expected_heads: list[dict]
    proposal_hash: str


def _store(request):
    return SynchronizationStore(request.app.state.settings)


def _actor(request):
    require_human_principal(request)
    return request.state.local_principal.id


@router.get('')
def catalog(request: Request):
    _admin(request)
    return _store(request).catalog()


@router.get('/node')
def node_identity(request: Request):
    _admin(request)
    return _store(request).node()


@router.post('/trust',status_code=201)
def trust_node(body: Node,request: Request):
    _admin(request)
    return _store(request).trust_node(body.id,body.public_key)


@router.post('/nodes/{node_id}/revoke')
def revoke_node(node_id: str,body: NodeRevocation,request: Request):
    _admin(request)
    return _store(request).revoke_node(node_id,**body.model_dump())


@router.get('/nodes/{node_id}/revocation-proposal')
def node_revocation_proposal(node_id: str,request: Request):
    _admin(request)
    return _store(request).node_revocation_proposal(node_id)


@router.post('/export')
def export(body: Export,request: Request):
    _admin(request)
    return _store(request).export(**body.model_dump(),actor=_actor(request))


@router.post('/import',status_code=201)
def import_resource(body: Import,request: Request):
    _admin(request)
    receipt = _store(request).import_bundle(**body.model_dump(),actor=_actor(request),app=request.app)
    if body.envelope.get('body',{}).get('kind')=='workflow':
        from marvis.routers.governed_capabilities import _refresh
        _refresh(request)
    return receipt


@router.get('/versions/{version_id}/readback')
def readback(version_id: str,request: Request):
    _admin(request)
    return _store(request).readback(version_id)


@router.post('/rollback')
def rollback(body: Rollback,request: Request):
    _admin(request)
    receipt = _store(request).rollback(**body.model_dump(),actor=_actor(request),app=request.app)
    from marvis.routers.governed_capabilities import _refresh
    _refresh(request)
    return receipt
