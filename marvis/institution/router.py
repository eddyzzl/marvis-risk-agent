"""Signed-assertion reference login and server-owned resource grants."""

import hmac
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from marvis.institution.policy import claim_resource, grant_resource, revoke_grant, require_human_principal
from marvis.institution.store import IdentityStore, require


router = APIRouter(prefix='/api/institution',tags=['institution'])


class StrictBody(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)


class Issuer(StrictBody):
    issuer: str
    key_id: str
    public_key: str


class Subject(StrictBody):
    issuer: str
    subject: str
    display_name: str
    role: Literal['member','maker','checker','admin','service']


class Challenge(StrictBody):
    issuer: str
    subject: str


class Login(StrictBody):
    assertion: dict
    signature: str


class Mode(StrictBody):
    mode: Literal['trusted_local','institutional']
    audience: str
    expected_revision: int = Field(ge=0)


class Owner(StrictBody):
    kind: Literal['task','dataset','artifact','plan']
    resource_id: str
    owner_id: str
    expected_revision: int = Field(ge=0)


class Grant(StrictBody):
    kind: Literal['task','dataset','artifact','plan']
    resource_id: str
    subject_id: str
    actions: list[str]
    expected_revision: int = Field(ge=0)
    expires_at: float | None = None


class Revocation(StrictBody):
    kind: Literal['subject','issuer','session']
    resource_id: str
    key_id: str | None = None


class RevokeGrant(StrictBody):
    kind: Literal['task','dataset','artifact','plan']
    resource_id: str
    subject_id: str
    expected_revision: int = Field(ge=0)


def _store(request):
    return IdentityStore(request.app.state.settings.db_path)


def _admin(request):
    actor = getattr(request.state,'institution_principal',None)
    if _store(request).settings()['mode']=='institutional':
        require(actor and actor['role']=='admin' and actor['actor_kind']=='human','identity_admin_required')
        return actor['id']
    expected = str(getattr(request.app.state,'plugin_admin_token','') or '')
    presented = request.headers.get('x-marvis-plugin-admin','')
    require(expected and hmac.compare_digest(expected,presented),'identity_bootstrap_admin_required')
    require_human_principal(request)
    return request.state.local_principal.id


@router.get('/status')
def status(request: Request):
    settings = _store(request).settings()
    return {**settings,'protocol':'marvis.identity-assertion.v1','algorithm':'Ed25519',
            'external_identity_verified':False,'resource_policy':'canonical owner and expiring explicit grants',
            'unsupported':['arbitrary Notebook','unregistered local path intake','unknown HTTP surfaces']}


@router.get('/me')
def me(request: Request):
    actor = getattr(request.state,'institution_principal',None)
    require(actor,'identity_actor_required',401)
    return actor


@router.get('/subjects')
def subjects(request: Request):
    _admin(request)
    return _store(request).list_subjects()


@router.post('/issuers',status_code=201)
def trust_issuer(body: Issuer, request: Request):
    _admin(request)
    return _store(request).trust_issuer(**body.model_dump())


@router.post('/subjects',status_code=201)
def provision_subject(body: Subject, request: Request):
    _admin(request)
    return _store(request).provision_subject(**body.model_dump())


@router.put('/mode')
def set_mode(body: Mode, request: Request):
    _admin(request)
    return _store(request).set_mode(**body.model_dump())


@router.post('/challenge')
def challenge(body: Challenge, request: Request):
    return _store(request).challenge(**body.model_dump())


@router.post('/login')
def login(body: Login, request: Request, response: Response):
    token, actor, ttl = _store(request).login(body.assertion,body.signature)
    from marvis.institution.http import COOKIE
    response.set_cookie(COOKIE,token,max_age=ttl,httponly=True,samesite='strict',
                        secure=request.url.scheme=='https',path='/')
    response.headers['Cache-Control']='no-store'
    return actor


@router.post('/logout')
def logout(request: Request, response: Response):
    from marvis.institution.http import COOKIE
    _store(request).revoke('session',request.cookies.get(COOKIE,''))
    response.delete_cookie(COOKIE,path='/')
    return {'logged_out':True}


@router.post('/owners')
def owner(body: Owner, request: Request):
    actor = _admin(request)
    return claim_resource(_store(request).db_path,actor,**body.model_dump())


@router.post('/grants')
def grant(body: Grant, request: Request):
    actor = getattr(request.state,'institution_principal',None)
    require(actor,'identity_actor_required',401)
    require_human_principal(request)
    payload = body.model_dump()
    payload['target_id'] = payload.pop('subject_id')
    return grant_resource(_store(request).db_path,actor['id'],**payload)


@router.post('/grants/revoke')
def revoke_resource_grant(body: RevokeGrant, request: Request):
    actor = getattr(request.state,'institution_principal',None)
    require(actor,'identity_actor_required',401)
    payload = body.model_dump()
    payload['target_id'] = payload.pop('subject_id')
    return revoke_grant(_store(request).db_path,actor['id'],**payload)


@router.post('/revoke')
def revoke_identity(body: Revocation, request: Request):
    _admin(request)
    return _store(request).revoke(**body.model_dump())
