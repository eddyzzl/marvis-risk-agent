"""Central governed lookups; the trusted host OS user remains outside this boundary."""

import hashlib
import json
from collections import OrderedDict
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import Lock
import time

from marvis.db_schema import connect
from marvis.modeling_holdout.contracts import HoldoutError

PRIVATE_DIRECTORY = '.blind_oot'
_HASH_CACHE = OrderedDict()
_HASH_LOCK = Lock()
_EXPOSURE_CONTEXT = ContextVar('holdout_exposure_context', default=None)


def _has_schema(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='blind_oot_cohorts'").fetchone() is not None


def cohort_tx(conn, task_id):
    if not _has_schema(conn):
        return None
    return conn.execute('SELECT * FROM blind_oot_cohorts WHERE task_id=?', (task_id,)).fetchone()


def cohort_for_task(db_path, task_id):
    if db_path is None or not Path(db_path).exists():
        return None
    with connect(Path(db_path)) as conn:
        row = cohort_tx(conn, task_id)
        return None if row is None else dict(row)


def assert_public_path(workspace, path):
    private = (Path(workspace) / PRIVATE_DIRECTORY).resolve()
    candidate = Path(path).resolve()
    if candidate == private or private in candidate.parents:
        raise HoldoutError('sealed_holdout_access_forbidden', 403)
    # Case-insensitive filesystems may preserve an alternate spelling in
    # resolve(); directory identity still forbids the physical private tree.
    if private.exists() and any(ancestor.exists() and ancestor.samefile(private)
                               for ancestor in (candidate, *candidate.parents)):
        raise HoldoutError('sealed_holdout_access_forbidden', 403)


def _hash(path):
    path = Path(path)
    stat = path.stat()
    identity = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with _HASH_LOCK:
        if identity in _HASH_CACHE:
            _HASH_CACHE.move_to_end(identity)
            return _HASH_CACHE[identity]
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    value = digest.hexdigest()
    current = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (
            current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns):
        raise HoldoutError('holdout_source_changed_during_admission')
    with _HASH_LOCK:
        _HASH_CACHE[identity] = value
        while len(_HASH_CACHE) > 256:
            _HASH_CACHE.popitem(last=False)
    return value


def assert_hash_public(conn, content_hash):
    if _has_schema(conn) and conn.execute(
        'SELECT 1 FROM blind_oot_cohorts WHERE sealed_hash=?', (content_hash,),
    ).fetchone():
        raise HoldoutError('sealed_holdout_access_forbidden', 403)


def assert_source_public(db_path, workspace, path, *, content_hash=None, conn=None):
    assert_public_path(workspace, path)
    if db_path is None or not Path(db_path).exists():
        return
    if conn is None:
        with connect(Path(db_path)) as connection:
            return assert_source_public(db_path, workspace, path, content_hash=content_hash, conn=connection)
    if not _has_schema(conn):
        return
    if content_hash is not None:
        assert_hash_public(conn, content_hash)
    elif conn.execute('SELECT 1 FROM blind_oot_cohorts LIMIT 1').fetchone():
        assert_hash_public(conn, _hash(path))


def _members_reserved(conn, table, members):
    values = tuple(members)
    for start in range(0, len(values), 400):
        chunk = values[start:start + 400]
        placeholders = ','.join('?' for _ in chunk)
        if conn.execute(f'SELECT 1 FROM {table} WHERE member_hash IN ({placeholders}) LIMIT 1', chunk).fetchone():  # nosec B608 -- table is restricted by installed callers to blind_oot_row_members/blind_oot_exposed_members; generated placeholders are only ? and all member values are bound
            return True
    return False


def _identity_schema(path):
    from marvis.modeling_holdout.identity import identity_source, source_schema
    with identity_source(path) as (source, _):
        return source_schema(source)


def assert_reserved_projections(conn, path):
    """Check existing custody projections before any optional legacy fallback."""
    from marvis.modeling_holdout.identity import parquet_identity
    projections = {tuple(json.loads(row['identity_columns'])) for row in conn.execute('SELECT identity_columns FROM blind_oot_cohorts')}
    if not projections:
        return
    try:
        available = set(_identity_schema(path).names)
    except Exception as exc:
        raise HoldoutError('holdout_admission_identity_unverifiable', 403) from exc
    for projection in projections:
        if set(projection) <= available:
            try:
                _, members = parquet_identity(Path(path), columns=projection, exclude_columns=())
            except Exception as exc:
                raise HoldoutError('holdout_admission_identity_unverifiable', 403) from exc
            if _members_reserved(conn, 'blind_oot_row_members', members):
                raise HoldoutError('sealed_holdout_access_forbidden', 403)


def admission_identity(conn, path, *, exclude_columns=('split',), primitive_only=False):
    from marvis.modeling_holdout.identity import parquet_identity, supported_columns
    schema = _identity_schema(path)
    available = set(schema.names)
    columns = supported_columns(schema, exclude_columns=exclude_columns) if primitive_only else sorted(available - set(exclude_columns))
    logical_hash, primary_members = parquet_identity(Path(path), columns=columns, exclude_columns=())
    projections = {tuple(columns)}
    projections.update(tuple(json.loads(row['identity_columns'])) for row in conn.execute('SELECT identity_columns FROM blind_oot_cohorts'))
    projections.update(tuple(json.loads(row['columns_json'])) for row in conn.execute('SELECT columns_json FROM blind_oot_public_schemas'))
    members = set(primary_members)
    for projection in projections:
        if set(projection) <= available and tuple(columns) != projection:
            _, projected = parquet_identity(Path(path), columns=projection, exclude_columns=())
            members.update(projected)
    return logical_hash, primary_members, members, columns


def assert_no_unverified_exposure(conn, columns):
    for row in conn.execute('SELECT columns_json FROM blind_oot_unverified_sources'):
        previous = json.loads(row['columns_json'])
        if previous is None or set(columns) <= set(previous):
            raise HoldoutError('prior_exposure_provenance_unknown')


def record_exposure(db_path, workspace, path, *, conn=None, content_hash=None):
    # Reserve normal-source exposure before copying, profiling or parsing. This
    # shares the SQLite write lock with sealing, including failed intake attempts.
    assert_public_path(workspace, path)
    if db_path is None or not Path(db_path).exists() or not Path(path).is_file():
        return
    digest = content_hash or _hash(path)
    if conn is None:
        with connect(Path(db_path)) as connection:
            if not _has_schema(connection):
                return
            assert_hash_public(connection, digest)
            if connection.execute('SELECT 1 FROM blind_oot_exposures WHERE content_hash=?', (digest,)).fetchone():
                return
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('BEGIN IMMEDIATE')
            return record_exposure(db_path, workspace, path, conn=connection, content_hash=digest)
    if _has_schema(conn):
        assert_hash_public(conn, digest)
        if conn.execute('SELECT 1 FROM blind_oot_exposures WHERE content_hash=?', (digest,)).fetchone():
            return
        # An unsupported extra column or ordinary format must never skip a
        # known reserved projection. This check intentionally cannot fall back.
        assert_reserved_projections(conn, path)
        members, columns = (), []
        try:
            _, _, members, columns = admission_identity(conn, path, primitive_only=True)
        except Exception:
            # Preserve ordinary ingestion, but never promote unknown historical
            # identity into blind custody. An overlapping future intake fails.
            try:
                unknown_columns = _identity_schema(path).names
            except Exception:
                unknown_columns = None
            conn.execute('INSERT OR IGNORE INTO blind_oot_unverified_sources VALUES(?,?)',
                         (digest, json.dumps(unknown_columns, separators=(',', ':'))))
        if _members_reserved(conn, 'blind_oot_row_members', members):
            raise HoldoutError('sealed_holdout_access_forbidden', 403)
        conn.execute('INSERT OR IGNORE INTO blind_oot_exposures VALUES(?,?)', (digest, time.time()))
        if columns:
            conn.execute('INSERT OR IGNORE INTO blind_oot_public_schemas VALUES(?)', (json.dumps(columns, separators=(',', ':')),))
        conn.executemany('INSERT OR IGNORE INTO blind_oot_exposed_members VALUES(?,?)', ((member, time.time()) for member in members))


def exposure_connection(function):
    """Reuse registration's caller transaction for backend admission lookups."""
    @wraps(function)
    def wrapped(self, conn, *args, **kwargs):
        token = _EXPOSURE_CONTEXT.set((getattr(self._repo, 'db_path', None), conn))
        try:
            return function(self, conn, *args, **kwargs)
        finally:
            _EXPOSURE_CONTEXT.reset(token)
    return wrapped


def record_backend_exposure(workspace, path):
    context = _EXPOSURE_CONTEXT.get()
    db_path, conn = context if context else (Path(workspace) / 'marvis.sqlite', None)
    record_exposure(db_path, workspace, path, conn=conn)


def assert_development_frame(conn, task_id, frame):
    row = cohort_tx(conn, task_id)
    if row is None:
        return
    split = json.loads(row['contract'])['split_col']
    if split in frame:
        # Blind tasks accept only development train/test partitions. They never
        # reinterpret a row from an uploaded mixed file as sealed data.
        values = frame[split].dropna().astype(str).str.casefold()
        if frame[split].isna().any() or not values.isin(['train', 'test']).all():
            raise HoldoutError('blind_development_requires_train_test_only', 422)


def require_builtin_for_task(db_path, task_id, manifest, ref, inputs):
    cohort = cohort_for_task(db_path, task_id)
    if cohort is None:
        return
    if not manifest.builtin or ref.plugin in {'v1_compat'}:
        raise HoldoutError('blind_mode_requires_governed_builtin_tools', 403)
    # Only audited data/feature/modeling tools are accepted; arbitrary hosted
    # code, draft Python and workflows with unconstrained file inputs are barred.
    if ref.plugin not in {'data_ops', 'feature', 'modeling'}:
        raise HoldoutError('blind_mode_requires_governed_builtin_tools', 403)
    with connect(Path(db_path)) as conn:
        frozen = conn.execute('SELECT 1 FROM blind_oot_freezes WHERE cohort_id=?', (cohort['id'],)).fetchone()
    if frozen and ref.tool.startswith(('train', 'tune', 'prepare', 'make_split', 'select', 'calibrate')):
        raise HoldoutError('blind_development_is_frozen')
    configuration = inputs.get('split_config') or {}
    if (configuration.get('oot_by_time') or configuration.get('random_oot')
            or any(rule.get('split') == 'oot' or rule.get('value') == 'oot' for rule in configuration.get('rules', []))
            or 'oot' in (inputs.get('split_values') or {})):
        raise HoldoutError('blind_development_cannot_create_oot_partition', 422)


def require_http_developer(db_path, task_id, actor_id):
    with connect(Path(db_path)) as conn:
        row = cohort_tx(conn, task_id)
        if row is None:
            return
        from marvis.risk_context.source_repository import require_actor
        from marvis.risk_context.source_contracts import SourceError
        try:
            require_actor(conn, actor_id, {'maker'})
        except SourceError as exc:
            raise HoldoutError('blind_development_scope_forbidden', 403) from exc
        if row['developer_id'] != actor_id:
            raise HoldoutError('blind_development_scope_forbidden', 403)


def reject_arbitrary_execution(db_path, task_id):
    if cohort_for_task(db_path, task_id) is not None:
        raise HoldoutError('blind_mode_arbitrary_execution_forbidden', 403)


def project_task_payloads(db_path, payloads, actor_id):
    """Expose only ordinary task metadata and the bound viewer's purpose.

    Custody/evaluation views can select their task without requesting development
    data, Agent messages or reports. Unrelated principals receive no blind task.
    """
    from marvis.risk_context.source_repository import require_actor
    from marvis.risk_context.source_contracts import SourceError
    result = []
    with connect(Path(db_path)) as conn:
        for payload in payloads:
            cohort = cohort_tx(conn, payload['id'])
            if cohort is None:
                result.append(payload)
                continue
            binding = next(((purpose, role) for purpose, role, column in (
                ('develop', 'maker', 'developer_id'), ('evaluate', 'checker', 'evaluator_id'),
                ('custody', 'admin', 'custodian_id')) if cohort[column] == actor_id), None)
            if binding is None:
                continue
            try:
                require_actor(conn, actor_id, {binding[1]})
            except SourceError:
                continue
            result.append({**payload, 'blind_oot': {'mode': 'governed_blind_oot', 'viewer_role': binding[0]}})
    return result



def related_development_task(db_path, path, query_params):
    """Resolve URL aliases to server-owned task IDs before the route executes."""
    pieces = path.strip('/').split('/')
    if len(pieces) < 2 or pieces[0] != 'api':
        return None
    if query_params.get('task_id'):
        return query_params['task_id']
    if len(pieces) < 3:
        return None
    if pieces[1] == 'artifacts':
        return pieces[2]
    tables = {'plans':'plans','datasets':'datasets','joins':'joins','drafts':'draft_tools','experiments':'experiments'}
    with connect(Path(db_path)) as conn:
        if pieces[1] == 'step-outputs':
            row = conn.execute('SELECT p.task_id FROM plan_steps s JOIN plans p ON p.id=s.plan_id WHERE s.id=?', (pieces[2],)).fetchone()
        elif pieces[1] in tables:
            row = conn.execute(f'SELECT task_id FROM {tables[pieces[1]]} WHERE id=?', (pieces[2],)).fetchone()  # nosec B608 -- identifier comes only from the fixed source-owned route-to-table map; resource id is bound
        else:
            return None
        return row['task_id'] if row else None
