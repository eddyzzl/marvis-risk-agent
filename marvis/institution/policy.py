"""One canonical task/resource authorization policy for HTTP and native lookups."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import inspect
from functools import wraps
from pathlib import Path
import time

from marvis.db_schema import connect
from marvis.institution.store import IdentityError, IdentityStore, identity_mode, require


_actor = ContextVar('marvis_institution_actor', default=None)


@contextmanager
def actor_context(db_path, actor_id, task_id=None):
    token = _actor.set((str(Path(db_path).resolve()), actor_id, task_id))
    try:
        yield
    finally:
        _actor.reset(token)


def current_actor(db_path):
    context = _actor.get()
    if context and context[0] == str(Path(db_path).resolve()):
        return context[1]
    return None


def task_for_resource(conn, kind, resource_id):
    if kind == 'task':
        return resource_id if conn.execute('SELECT 1 FROM tasks WHERE id=?', (resource_id,)).fetchone() else None
    query = {
        'dataset': 'SELECT task_id FROM datasets WHERE id=?',
        'join': 'SELECT task_id FROM joins WHERE id=?',
        'run': 'SELECT p.task_id FROM plan_step_runs r JOIN plans p ON p.id=r.plan_id WHERE r.id=?',
        'plan': 'SELECT task_id FROM plans WHERE id=?',
        'experiment': 'SELECT task_id FROM experiments WHERE id=?',
        'strategy': 'SELECT task_id FROM strategies WHERE id=?',
        'artifact': 'SELECT task_id FROM task_artifacts WHERE id=? UNION SELECT e.task_id FROM model_artifacts m JOIN experiments e ON e.id=m.experiment_id WHERE m.id=? UNION SELECT s.task_id FROM strategy_artifacts a JOIN strategies s ON s.id=a.strategy_id WHERE a.id=?',
        'step': 'SELECT p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.id=?',
        'job': 'SELECT task_id FROM jobs WHERE id=?',
        'sub': 'SELECT parent_task_id FROM sub_agents WHERE id=?',
        'promotion': 'SELECT s.task_id FROM production_promotion_requests p JOIN strategies s ON s.id=p.strategy_id WHERE p.id=?',
        'deployment': 'SELECT s.task_id FROM production_deployments d JOIN strategies s ON s.id=d.strategy_id WHERE d.id=?',
    }.get(kind)
    if not query:
        return None
    rows = conn.execute(query, (resource_id,) * query.count('?')).fetchall()
    tasks = {row[0] for row in rows}
    require(len(tasks) <= 1, 'identity_resource_ambiguous', 409)
    return next(iter(tasks), None)


def _allowed(conn, kind, resource_id, actor_id, action):
    subject = IdentityStore._subject(conn, actor_id)
    if action in {'share','approve','admin'} and subject['actor_kind'] != 'human':
        return False
    owner = conn.execute('SELECT owner_id FROM institution_resource_owners WHERE kind=? AND resource_id=?',
                         (kind, resource_id)).fetchone()
    if owner and owner['owner_id'] == actor_id:
        return True
    grant = conn.execute('SELECT * FROM institution_resource_grants WHERE kind=? AND resource_id=? AND subject_id=?',
                         (kind, resource_id, actor_id)).fetchone()
    if grant and grant['revoked_at'] is None and (grant['expires_at'] is None or grant['expires_at'] > time.time()):
        if action in json.loads(grant['actions_json']):
            return True
    if kind != 'task':
        task_id = task_for_resource(conn, kind, resource_id)
        if task_id:
            return _allowed(conn, 'task', task_id, actor_id, action)
    return False


def authorize_on_connection(conn, kind, resource_id, action='read', *, actor_id=None):
    from marvis.institution.synchronization import authorize_synced_ref
    authorize_synced_ref(conn,resource_id)
    if identity_mode(conn) != 'institutional':
        return
    if actor_id is None:
        databases = conn.execute('PRAGMA database_list').fetchall()
        actor_id = current_actor(next(row['file'] for row in databases if row['name'] == 'main'))
    require(actor_id, 'identity_actor_required', 401)
    require(_allowed(conn, kind, resource_id, actor_id, action), 'identity_resource_forbidden')


def authorize(db_path, kind, resource_id, action='read', *, actor_id=None):
    with connect(db_path) as conn:
        authorize_on_connection(conn, kind, resource_id, action, actor_id=actor_id)


def bind_new_task(conn, task_id):
    if identity_mode(conn) != 'institutional':
        return
    db_path = next(row['file'] for row in conn.execute('PRAGMA database_list') if row['name'] == 'main')
    actor = current_actor(db_path)
    require(actor, 'identity_actor_required', 401)
    IdentityStore._subject(conn, actor)
    conn.execute('INSERT INTO institution_resource_owners VALUES(\'task\',?,?,1,?)', (task_id, actor, time.time()))


def bind_new_plan(conn, plan_id, task_id):
    if identity_mode(conn) != 'institutional':
        return
    authorize_on_connection(conn, 'task', task_id, 'execute')
    db_path = next(row['file'] for row in conn.execute('PRAGMA database_list') if row['name'] == 'main')
    actor = current_actor(db_path)
    conn.execute('INSERT INTO institution_plan_actors VALUES(?,?,?)', (plan_id, actor, task_id))


def visible_task_ids(conn):
    if identity_mode(conn) != 'institutional':
        return None
    db_path = next(row['file'] for row in conn.execute('PRAGMA database_list') if row['name'] == 'main')
    actor = current_actor(db_path)
    require(actor, 'identity_actor_required', 401)
    IdentityStore._subject(conn, actor)
    return [row['id'] for row in conn.execute('SELECT id FROM tasks') if _allowed(conn, 'task', row['id'], actor, 'read')]


def require_human_principal(request):
    actor = getattr(request.state, 'institution_principal', None)
    if actor:
        require(actor['actor_kind'] == 'human', 'human_actor_required')
        return actor
    principal = getattr(request.state, 'local_principal', None)
    require(principal is not None, 'identity_actor_required', 401)
    return {'id': principal.id, 'actor_kind': 'human'}


def claim_resource(db_path, actor_id, kind, resource_id, owner_id, expected_revision):
    with connect(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        actor = IdentityStore._subject(conn, actor_id)
        require(actor['role'] == 'admin' and actor['actor_kind'] == 'human', 'identity_admin_required')
        IdentityStore._subject(conn, owner_id)
        require(kind in {'task','dataset','artifact','plan'} and task_for_resource(conn, kind, resource_id),
                'identity_resource_not_found', 404)
        old = conn.execute('SELECT * FROM institution_resource_owners WHERE kind=? AND resource_id=?',
                           (kind, resource_id)).fetchone()
        require((old['revision'] if old else 0) == expected_revision, 'identity_owner_conflict', 409)
        if old:
            conn.execute('UPDATE institution_resource_owners SET owner_id=?,revision=revision+1 WHERE kind=? AND resource_id=?',
                         (owner_id, kind, resource_id))
        else:
            conn.execute('INSERT INTO institution_resource_owners VALUES(?,?,?,1,?)',
                         (kind, resource_id, owner_id, time.time()))
        return dict(conn.execute('SELECT * FROM institution_resource_owners WHERE kind=? AND resource_id=?',
                                 (kind, resource_id)).fetchone())


def grant_resource(db_path, actor_id, kind, resource_id, target_id, actions, expected_revision, expires_at=None):
    require(kind in {'task','dataset','artifact','plan'}, 'identity_invalid_resource', 422)
    require(isinstance(actions,list) and len(actions) == len(set(actions)) and
            set(actions) <= {'read','write','execute','share','approve'} and actions,
            'identity_invalid_actions', 422)
    require(expires_at is None or type(expires_at) in (int,float) and time.time() < expires_at <= time.time()+86400*365,
            'identity_invalid_expiry', 422)
    with connect(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        actor = IdentityStore._subject(conn, actor_id)
        target = IdentityStore._subject(conn, target_id)
        require(actor['actor_kind'] == 'human', 'human_actor_required')
        require(actor['role'] == 'admin' or _allowed(conn, kind, resource_id, actor_id, 'share'),
                'identity_share_forbidden')
        require(task_for_resource(conn, kind, resource_id), 'identity_resource_not_found', 404)
        require(target['actor_kind'] != 'service' or set(actions) <= {'read','write','execute'},
                'identity_service_scope_forbidden')
        old = conn.execute('SELECT * FROM institution_resource_grants WHERE kind=? AND resource_id=? AND subject_id=?',
                           (kind, resource_id, target_id)).fetchone()
        require((old['revision'] if old else 0) == expected_revision, 'identity_grant_conflict', 409)
        conn.execute('''INSERT INTO institution_resource_grants VALUES(?,?,?,?,?,?,?,NULL)
             ON CONFLICT(kind,resource_id,subject_id) DO UPDATE SET actions_json=excluded.actions_json,
             revision=excluded.revision,granted_by=excluded.granted_by,expires_at=excluded.expires_at,revoked_at=NULL''',
             (kind,resource_id,target_id,json.dumps(sorted(actions)),expected_revision+1,actor_id,expires_at))
        return {'kind':kind,'resource_id':resource_id,'subject_id':target_id,'actions':sorted(actions),
                'revision':expected_revision+1,'expires_at':expires_at}


def revoke_grant(db_path, actor_id, kind, resource_id, target_id, expected_revision):
    with connect(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        actor = IdentityStore._subject(conn, actor_id)
        require(actor['actor_kind'] == 'human' and (actor['role'] == 'admin' or _allowed(conn,kind,resource_id,actor_id,'share')),
                'identity_share_forbidden')
        cursor = conn.execute('UPDATE institution_resource_grants SET revoked_at=?,revision=revision+1 WHERE kind=? AND resource_id=? AND subject_id=? AND revision=?',
                              (time.time(),kind,resource_id,target_id,expected_revision))
        require(cursor.rowcount == 1,'identity_grant_conflict',409)
        return {'revoked':True,'revision':expected_revision+1}


def authorize_file(root, path):
    """Registered bytes or the current task's upload staging; never a whole workspace."""
    context = _actor.get()
    db_path = Path(context[0]) if context else next(
        (candidate/'marvis.sqlite' for candidate in (Path(root),*Path(root).parents)
         if (candidate/'marvis.sqlite').is_file()), Path(root).parent/'marvis.sqlite')
    if not db_path.exists():
        return
    with connect(db_path) as conn:
        if identity_mode(conn) != 'institutional':
            return
        resolved = Path(path).resolve()
        require(current_actor(db_path), 'identity_actor_required',401)
        datasets_root = db_path.parent/'datasets'
        registered=False
        for row in conn.execute('SELECT id,source_path FROM datasets'):
            if (datasets_root / row['source_path']).resolve() == resolved:
                registered=True
                try:
                    authorize_on_connection(conn,'dataset',row['id'])
                except IdentityError as exc:
                    if exc.code!='identity_resource_forbidden':
                        raise
                    continue
                return
        require(not registered,'identity_resource_forbidden')
        # Native profiling of a just-uploaded object is scoped by the request's
        # canonical task. External path registration is unsupported in this mode.
        if context and context[2]:
            upload_root = (datasets_root/context[2]).resolve()
            if resolved.is_relative_to(upload_root):
                authorize_on_connection(conn,'task',context[2],'write')
                return
        raise IdentityError('identity_unregistered_file_forbidden')


def invocation_actor(db_path, invocation_id, *, require_dispatch=True, installed=None, inputs=None):
    with connect(db_path) as conn:
        if identity_mode(conn) != 'institutional':
            return None
        row = conn.execute('''SELECT a.subject_id,a.task_id,r.dispatch_started_at FROM institution_plan_actors a
            JOIN plan_step_runs r ON r.plan_id=a.plan_id WHERE r.id=?''',
            (invocation_id,)).fetchone()
        if row is None:
            return scheduled_invocation_actor(conn,invocation_id,installed=installed,inputs=inputs)
        require(not require_dispatch or row['dispatch_started_at'] is not None, 'identity_runtime_binding_required')
        authorize_on_connection(conn,'task',row['task_id'],'execute',actor_id=row['subject_id'])
        return row['subject_id'],row['task_id']


def scheduled_invocation_actor(conn,invocation_id,*,installed,inputs):
    """Accept only a consumed native scheduler dispatch and its current lease."""
    from datetime import datetime
    from marvis.operations.contracts import ScheduleContract
    from marvis.operations.native_invocations import allowed_tools
    from marvis.operations.schedule_proposals import sha,schedule_task
    require(conn.execute("SELECT 1 FROM sqlite_master WHERE name='operations_native_invocations'").fetchone(),
            'identity_runtime_binding_required')
    row=conn.execute("""SELECT i.*,r.lease_token AS claimed_token,p.lease_token AS current_token,
        p.lease_expires_at AS current_expires,p.state AS current_state,s.contract_json,s.contract_hash
        FROM operations_native_invocations i JOIN operations_runs r ON r.run_id=i.run_id
        JOIN operations_periods p ON p.schedule_id=r.schedule_id AND p.period_key=r.period_key
        JOIN operations_schedules s ON s.schedule_id=i.schedule_id AND s.revision=i.schedule_revision
        WHERE i.invocation_id=? AND i.schedule_id=r.schedule_id AND i.schedule_revision=r.schedule_revision""",(invocation_id,)).fetchone()
    require(row and row['dispatch_started_at'] and row['current_state']=='running'
            and row['claimed_token']==row['current_token']
            and datetime.fromisoformat(row['current_expires']).timestamp()>time.time(),'identity_scheduled_lease_invalid')
    contract=ScheduleContract.from_dict(json.loads(row['contract_json']))
    require(sha(contract.to_dict())==row['contract_hash'] and contract.execution_actor_id==row['actor_id']
            and schedule_task(contract)==row['task_id'] and row['tool_ref'] in allowed_tools(contract),
            'identity_scheduled_binding_changed')
    require(isinstance(installed,dict) and installed.get('tool_ref')==row['tool_ref']
            and sha(installed)==row['installed_invocation_hash'] and sha(inputs)==row['inputs_hash'],
            'identity_scheduled_inputs_changed')
    authorize_on_connection(conn,'task',row['task_id'],'execute',actor_id=row['actor_id'])
    return row['actor_id'],row['task_id']


def plan_execution_scope(db_path, plan_id):
    with connect(db_path) as conn:
        if identity_mode(conn) != 'institutional':
            return nullcontext()
        row = conn.execute('SELECT * FROM institution_plan_actors WHERE plan_id=?',(plan_id,)).fetchone()
        require(row,'identity_runtime_binding_required')
        actor = current_actor(db_path) or row['subject_id']
        authorize_on_connection(conn,'task',row['task_id'],'execute',actor_id=actor)
        authorize_on_connection(conn,'task',row['task_id'],'execute',actor_id=row['subject_id'])
        return actor_context(db_path,row['subject_id'],row['task_id'])


def worker_execution_scope(job):
    db_path = Path(job['workspace'])/'marvis.sqlite'
    if not job.get('builtin') or not db_path.exists():
        return nullcontext()
    binding = invocation_actor(db_path,job.get('invocation_id'),installed=job.get('institution_invocation'),inputs=job.get('inputs'))
    if binding is None:
        return nullcontext()
    require(binding[1] == job['task_id'],'identity_runtime_task_mismatch')
    return actor_context(db_path,binding[0],binding[1])


def authorize_tool(db_path, task_id, manifest, ref, inputs, invocation_id, worker_mode, installed=None):
    with connect(db_path) as conn:
        if identity_mode(conn) != 'institutional':
            return
        require(ref.plugin != 'v1_compat','identity_notebook_workflow_unsupported')
        require(manifest.builtin or worker_mode=='os_sandbox','identity_plugin_os_boundary_required')
        authorize_on_connection(conn,'task',task_id,'execute')
        authorize_input_references(conn,inputs)
        # Only the existing native Plan dispatch can hand a builtin the host DB.
        if manifest.builtin:
            row = conn.execute('''SELECT a.subject_id,a.task_id FROM institution_plan_actors a
                JOIN plan_step_runs r ON r.plan_id=a.plan_id WHERE r.id=?''',(invocation_id,)).fetchone()
            binding=(row['subject_id'],row['task_id']) if row else scheduled_invocation_actor(conn,invocation_id,installed=installed,inputs=inputs)
            require(binding[1]==task_id and binding[0]==current_actor(db_path),'identity_runtime_binding_required')


def authorize_input_references(conn, value, action='read'):
    names = {'task_id':'task','dataset_id':'dataset','model_artifact_id':'artifact','artifact_id':'artifact',
             'plan_id':'plan','strategy_id':'strategy','experiment_id':'experiment',
             'anchor_dataset_id':'dataset','sample_dataset_id':'dataset','baseline_dataset_id':'dataset'}
    if isinstance(value,dict):
        for key,item in value.items():
            kind=names.get(key)
            if kind is None:
                kind=next((resource for suffix,resource in (('_dataset_id','dataset'),('_artifact_id','artifact'),('_experiment_id','experiment'),('_strategy_id','strategy'),('_plan_id','plan')) if key.endswith(suffix)),None)
            if kind and isinstance(item,str) and task_for_resource(conn,kind,item):
                authorize_on_connection(conn,kind,item,action)
            elif key.endswith('_ids') and isinstance(item,list):
                singular=key[:-1]
                for reference in item:
                    authorize_input_references(conn,{singular:reference},action)
            else:
                authorize_input_references(conn,item,action)
    elif isinstance(value,list):
        for item in value:
            authorize_input_references(conn,item,action)


def guard_repository(cls):
    """Protect existing named-resource methods, including connection-scoped writes.

    This decorates the existing repository; it creates no storage/runtime facade.
    In local mode it performs no policy decisions. Context never comes from JSON.
    """
    bindings = {'task_id':'task','normalized_task_id':'task','plan_id':'plan','dataset_id':'dataset',
                'artifact_id':'artifact','experiment_id':'experiment','strategy_id':'strategy',
                'step_id':'step','step_ids':'step','task_ids':'task','artifact_ids':'artifact','run_id':'run','join_plan_id':'join','job_id':'job','sub_id':'sub','request_id':'promotion','deployment_id':'deployment','expected_active_deployment_id':'deployment'}
    if cls.__name__=='DatasetRepository':
        bindings['plan_id']='join'
    for name,func in tuple(vars(cls).items()):
        if name=='get_active_job_kinds_for_tasks':
            # Its native batch seam admits on its one existing connection.
            continue
        if name.startswith('_') or not inspect.isfunction(func):
            continue
        signature = inspect.signature(func)
        relevant = {key:kind for key,kind in bindings.items() if key in signature.parameters}
        if not relevant:
            continue
        action = 'approve' if name in {'review_promotion_request','approve_promotion_request','activate_promotion_request','rollback_deployment','confirm_plan','confirm_step','confirm_step_with_inputs'} else 'read' if name.startswith(('get','list','count','find','has','load','is','peek','resolve','latest','workflow')) else 'write'
        def decorate(method, method_signature, refs, required_action):
            @wraps(method)
            def guarded(self,*args,**kwargs):
                bound = method_signature.bind(self,*args,**kwargs)
                conn = bound.arguments.get('conn')
                def admit(connection):
                    if identity_mode(connection)=='institutional' and 'actor_principal_id' in bound.arguments:
                        require(bound.arguments['actor_principal_id']==current_actor(self.db_path),'identity_actor_binding_mismatch')
                    for key,kind in refs.items():
                        value = bound.arguments.get(key)
                        values=value if isinstance(value,(tuple,list)) else [value]
                        for reference in values:
                            if isinstance(reference,str) and reference:
                                authorize_on_connection(connection,kind,reference,required_action)
                if conn is None:
                    with connect(self.db_path) as connection:
                        admit(connection)
                else:
                    admit(conn)
                return method(self,*args,**kwargs)
            return guarded
        setattr(cls,name,decorate(func,signature,relevant,action))
    return cls
