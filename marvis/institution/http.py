"""Admission for the explicit institution mode, before local session fallback."""

from contextlib import contextmanager
from types import SimpleNamespace
import json

from fastapi.responses import JSONResponse

from marvis.db_schema import connect
from marvis.institution.policy import (
    actor_context, authorize_input_references, authorize_on_connection, task_for_resource,
)
from marvis.institution.store import IdentityError, IdentityStore, identity_mode, require


COOKIE = 'marvis_institution_session'
PUBLIC = frozenset({'/api/institution/challenge','/api/institution/login','/api/institution/status'})


async def prepare_request(request, db_path):
    request.state.institution_principal = None
    request.state.institution_task_id = None
    with connect(db_path) as conn:
        mode = identity_mode(conn)
    if mode != 'institutional' or not request.url.path.startswith('/api/'):
        return None
    path, method = request.url.path, request.method.upper()
    if path in PUBLIC:
        return None
    try:
        actor = IdentityStore(db_path).resolve_session(request.cookies.get(COOKIE,''))
        request.state.institution_principal = actor
        request.state.local_principal = SimpleNamespace(id=actor['id'],kind='institutional',
            display_name=actor['display_name'],status='active',expires_at=None,
            actor_kind=actor['actor_kind'],role=actor['role'])
        pieces = path.strip('/').split('/')
        action = 'read' if method in {'GET','HEAD'} else 'write'
        if any(piece in {'run','retry','resume','resume-completion'} for piece in pieces):
            action = 'execute'
        if any(piece in {'confirm','decide','approve','approval','publish','activate','rollback','revoke'} for piece in pieces):
            require(actor['actor_kind'] == 'human','human_actor_required')
        if 'notebook' in path or path.endswith('/register-path') or '/drafts/' in path:
            raise IdentityError('identity_execution_surface_unsupported')
        with actor_context(db_path,actor['id']):
            with connect(db_path) as conn:
                scoped = False
                if pieces[1] == 'tasks' and len(pieces)>2:
                    task_id = pieces[2]
                    authorize_on_connection(conn,'task',task_id,action)
                    request.state.institution_task_id = task_id
                    scoped = True
                elif len(pieces)>2 and pieces[1] in {'datasets','plans','experiments','strategies','step-outputs','joins'}:
                    kind = {'datasets':'dataset','plans':'plan','experiments':'experiment','strategies':'strategy','step-outputs':'step','joins':'join'}[pieces[1]]
                    task_id = task_for_resource(conn,kind,pieces[2])
                    require(task_id,'identity_resource_not_found',404)
                    authorize_on_connection(conn,kind,pieces[2],action)
                    request.state.institution_task_id = task_id
                    scoped = True
                elif pieces[1] == 'artifacts' and len(pieces)>2:
                    # Native generic artifact paths start with their owning task;
                    # unknown aliases are refused rather than trusting a URI body.
                    task_id = pieces[2]
                    require(task_for_resource(conn,'task',task_id),'identity_resource_not_found',404)
                    authorize_on_connection(conn,'task',task_id,action)
                    request.state.institution_task_id = task_id
                    scoped = True
                elif path in {'/api/tasks','/api/experiments'}:
                    scoped = True
                elif path.startswith('/api/institution/') or path=='/api/instance-sync' or path.startswith('/api/instance-sync/'):
                    scoped = True
                elif path.startswith('/api/production-governance/'):
                    require(not path.endswith('/principals/claim'),'identity_role_claim_forbidden')
                    require(actor['role'] in {'maker','checker','admin'} and actor['actor_kind']=='human','identity_governance_role_required')
                    # Native governance repositories guard exact strategy,
                    # promotion and deployment references and filter lists.
                    scoped = True
                elif path.startswith('/api/risk-sources/'):
                    if method not in {'GET','HEAD'}:
                        require(actor['role']=='admin' and actor['actor_kind']=='human','identity_admin_required')
                        if len(pieces)>3 and pieces[2]=='grants':
                            row=conn.execute('SELECT task_id FROM source_grants WHERE id=?',(pieces[3],)).fetchone()
                            require(row,'identity_resource_not_found',404)
                            authorize_on_connection(conn,'task',row['task_id'],'write')
                    else:
                        require(path in {'/api/risk-sources/capabilities','/api/risk-sources/profiles'},'identity_surface_unsupported')
                    scoped = True
                elif path in {'/api/risk-sources/capabilities','/api/risk-sources/profiles','/api/risk-events/capabilities'} and method=='GET':
                    scoped = True
                elif path.startswith('/api/operations/'):
                    require(not path.startswith(('/api/operations/tick','/api/operations/recover','/api/operations/dataset-source-gc','/api/operations/task-filesystem-gc')),'identity_surface_unsupported')
                    require(actor['role'] in {'maker','checker','admin'} and actor['actor_kind']=='human','identity_governance_role_required')
                    scoped = True
                elif path.startswith(('/api/plugins/','/api/skills/','/api/settings/')):
                    require(actor['role']=='admin' and actor['actor_kind']=='human','identity_admin_required')
                    require(not path.endswith('/principals/claim'),'identity_role_claim_forbidden')
                    scoped = True
                elif path in {'/api/plugins','/api/skills','/api/capability-tiers','/api/health'} or (path=='/api/branding' and method=='GET'):
                    scoped = True
                require(scoped,'identity_surface_unsupported')
                if method not in {'GET','HEAD'} and request.headers.get('content-type','').split(';')[0]=='application/json':
                    raw = await request.body()
                    budget=8*1024*1024 if path=='/api/instance-sync/import' else 2*1024*1024
                    require(len(raw)<=budget,'identity_request_budget',413)
                    try:
                        payload = json.loads(raw)
                    except (ValueError,UnicodeError) as exc:
                        raise IdentityError('identity_invalid_json',422) from exc
                    require(not (path=='/api/tasks' and isinstance(payload,dict) and payload.get('source_dir')),
                            'identity_external_source_unsupported')
                    authorize_input_references(conn,payload)
                    if path.startswith('/api/risk-sources/') and isinstance(payload,dict) and payload.get('task_id'):
                        authorize_on_connection(conn,'task',payload['task_id'],'write')
                authorize_input_references(conn,dict(request.query_params))
    except IdentityError as exc:
        return JSONResponse(status_code=exc.status,content={'detail':{'code':exc.code}})
    return None


@contextmanager
def request_context(request, db_path):
    actor = getattr(request.state,'institution_principal',None)
    if actor:
        with actor_context(db_path,actor['id'],getattr(request.state,'institution_task_id',None)):
            yield
    else:
        yield


def identity_error_response(exc):
    return JSONResponse(status_code=exc.status,content={'detail':{'code':exc.code}})
