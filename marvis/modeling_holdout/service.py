"""Role-separated custody, immutable model closure and a durable one-read budget."""

from dataclasses import asdict
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
import shutil
import time
import uuid

import numpy as np
import pyarrow.parquet as pq

from marvis.data.authenticated_snapshot import authenticated_file_snapshot
from marvis.db_schema import connect as ordinary_connect
from marvis.modeling_holdout.contracts import HoldoutError
from marvis.modeling_holdout.guards import PRIVATE_DIRECTORY, cohort_tx, admission_identity, _members_reserved
from marvis.modeling_holdout.identity import (
    MAX_SEALED_BYTES, MAX_DECODED_BYTES, MAX_SEALED_ROWS, MAX_SEALED_COLUMNS,
)

from marvis.packs.modeling.contracts import ModelArtifact
from marvis.packs.modeling.producer_receipts import canonical, load_receipt, verify_files
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.repositories.modeling import ModelingRepository
from marvis.risk_context.source_contracts import SourceError
from marvis.risk_context.source_repository import require_actor


SUPPORTED_ALGORITHMS = frozenset({
    'lr', 'lgb', 'xgb', 'catboost', 'mlp', 'scorecard', 'ensemble',
    'lr_regressor', 'lgb_regressor', 'xgb_regressor', 'mlp_regressor',
    'lr_multiclass', 'lgb_multiclass', 'xgb_multiclass', 'mlp_multiclass',
})

def digest(value):
    return hashlib.sha256(value).hexdigest()


@contextmanager
def connect(db_path):
    # The committed reservation/read-start must precede access to the shard,
    # including an application/process restart. WAL NORMAL is insufficient for
    # that durability requirement under an abrupt host power loss.
    with ordinary_connect(db_path) as conn:
        conn.execute('PRAGMA synchronous=FULL')
        yield conn


def _sync_directory(path):
    if os.name != 'posix':
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class HoldoutService:
    def __init__(self, settings):
        self.settings, self.db_path = settings, settings.db_path
        self.private = settings.workspace / PRIVATE_DIRECTORY
        if self.private.is_symlink():
            raise HoldoutError('holdout_storage_unavailable')
        self.private.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.private.chmod(0o700)
        self.secret = settings.plugin_admin_token_path.read_bytes().strip()
        if not self.secret:
            raise HoldoutError('holdout_authentication_unavailable')

    def _actor(self, conn, actor, roles):
        try:
            return require_actor(conn, actor, roles)
        except SourceError as exc:
            raise HoldoutError('holdout_role_forbidden', 403) from exc

    def _cohort(self, conn, task, actor, purpose):
        row = cohort_tx(conn, task)
        if row is None:
            raise HoldoutError('holdout_not_found', 404)
        roles, column = {
            'develop': ({'maker'}, 'developer_id'),
            'evaluate': ({'checker'}, 'evaluator_id'),
            'custody': ({'admin'}, 'custodian_id'),
        }[purpose]
        self._actor(conn, actor, roles)
        if row[column] != actor:
            raise HoldoutError('holdout_scope_forbidden', 403)
        return row

    def _signature(self, body):
        return hmac.new(self.secret, canonical(body).encode(), hashlib.sha256).hexdigest()

    def _path(self, *parts):
        path = self.private.joinpath(*parts)
        if path.resolve() != path.absolute() or self.private.resolve() not in path.resolve().parents:
            raise HoldoutError('holdout_storage_binding_failed')
        return path

    def _publish(self, path, data):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != data:
                raise HoldoutError('holdout_immutable_object_conflict')
            return
        temporary = path.parent / f'.pending-{uuid.uuid4().hex}'
        try:
            with temporary.open('xb') as stream:
                temporary.chmod(0o600)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard link is an atomic publish with O_EXCL semantics; replacement
            # of an immutable object is never allowed, including concurrent writes.
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != data:
                    raise HoldoutError('holdout_immutable_object_conflict') from None
            path.chmod(0o400)
        finally:
            temporary.unlink(missing_ok=True)
            _sync_directory(path.parent)

    def _was_exposed(self, conn, sealed_hash):
        if conn.execute('SELECT 1 FROM blind_oot_exposures WHERE content_hash=?', (sealed_hash,)).fetchone():
            return True
        # Existing pre-migration upload identity sidecars carry original hashes;
        # canonical dataset hashes alone cannot establish first-ever custody.
        for path in self.settings.datasets_dir.glob('*/.source-identities/*.json'):
            try:
                if json.loads(path.read_text())['sha256'] == sealed_hash:
                    return True
            except (OSError, ValueError, KeyError) as exc:
                raise HoldoutError('prior_exposure_provenance_unknown') from exc
        return False

    def intake(self, task, payload, stream, actor):
        contract = canonical(payload.model_dump())
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._actor(conn, actor, {'admin'})
            self._actor(conn, payload.developer_id, {'maker'})
            self._actor(conn, payload.evaluator_id, {'checker'})
            if len({actor, payload.developer_id, payload.evaluator_id}) != 3:
                raise HoldoutError('holdout_requires_separate_actors', 422)
            record = conn.execute('SELECT task_type,status FROM tasks WHERE id=?', (task,)).fetchone()
            if record is None:
                raise HoldoutError('task_not_found', 404)
            if record['task_type'] != 'modeling':
                raise HoldoutError('blind_mode_requires_modeling_task', 422)
            existing = cohort_tx(conn, task)
            if existing is None:
                if record['status'] != 'created' or any(conn.execute(
                    f'SELECT 1 FROM {table} WHERE task_id=? LIMIT 1', (task,),  # nosec B608 -- table is from the inline fixed five-table tuple; task id is bound
                ).fetchone() for table in ('datasets', 'experiments', 'plans', 'jobs', 'draft_tools')):
                    raise HoldoutError('blind_intake_must_precede_development', 409)
            incoming = self._path('incoming', uuid.uuid4().hex)
            incoming.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            size, hasher = 0, hashlib.sha256()
            try:
                with incoming.open('xb') as output:
                    incoming.chmod(0o600)
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        size += len(chunk)
                        if size > min(self.settings.max_csv_upload_bytes, MAX_SEALED_BYTES):
                            raise HoldoutError('holdout_upload_too_large', 413)
                        hasher.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                sealed_hash = hasher.hexdigest()
                if existing:
                    if (existing['contract'], existing['custodian_id'], existing['sealed_hash']) != (contract, actor, sealed_hash):
                        raise HoldoutError('holdout_identity_immutable')
                    return self._summary(conn, existing)
                if self._was_exposed(conn, sealed_hash):
                    raise HoldoutError('already_exposed_source_cannot_be_blind')
                if conn.execute('SELECT 1 FROM blind_oot_cohorts WHERE sealed_hash=? OR data_identity=?',
                                (sealed_hash, payload.data_identity)).fetchone():
                    raise HoldoutError('holdout_identity_already_reserved')
                # First inspect the footer/schema. The later bounded identity decoder
                # is custody-only; development and evaluation have no row access.
                try:
                    parquet = pq.ParquetFile(incoming)
                except Exception as exc:
                    raise HoldoutError('holdout_schema_invalid', 422) from exc
                metadata = parquet.metadata
                if (metadata.num_rows > MAX_SEALED_ROWS or len(parquet.schema_arrow) > MAX_SEALED_COLUMNS
                        or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) > MAX_DECODED_BYTES):
                    raise HoldoutError('holdout_resource_profile_exceeded', 413)
                if parquet.metadata.num_rows < 1 or payload.target_col not in parquet.schema_arrow.names:
                    raise HoldoutError('holdout_schema_invalid', 422)
                # Custody-only decoding establishes immutable row identity. No
                # developer/Agent/evaluator sees or profiles these rows here.
                logical_hash, members, checked_members, identity_columns = admission_identity(conn, incoming, exclude_columns=(payload.split_col, 'split'))
                from marvis.modeling_holdout.guards import assert_no_unverified_exposure
                assert_no_unverified_exposure(conn, identity_columns)
                if conn.execute('SELECT 1 FROM blind_oot_cohorts WHERE logical_hash=?', (logical_hash,)).fetchone():
                    raise HoldoutError('holdout_identity_already_reserved')
                if _members_reserved(conn, 'blind_oot_exposed_members', checked_members):
                    raise HoldoutError('already_exposed_source_cannot_be_blind')
                if _members_reserved(conn, 'blind_oot_row_members', checked_members):
                    raise HoldoutError('holdout_identity_already_reserved')
                object_path = self._path('sealed', sealed_hash, 'original.parquet')
                object_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if object_path.exists():
                    if digest(object_path.read_bytes()) != sealed_hash:
                        raise HoldoutError('holdout_sealed_integrity_failed')
                else:
                    os.link(incoming, object_path)
                    object_path.chmod(0o400)
                    _sync_directory(object_path.parent)
                cohort_id = 'cohort_' + digest((payload.data_identity + sealed_hash).encode())[:32]
                conn.execute('INSERT INTO blind_oot_cohorts VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                    (cohort_id, task, payload.data_identity, sealed_hash, logical_hash, json.dumps(identity_columns, separators=(',', ':')), contract,
                     actor, payload.developer_id, payload.evaluator_id, time.time()))
                conn.executemany('INSERT INTO blind_oot_row_members VALUES(?,?)', ((member, cohort_id) for member in members))
                return self._summary(conn, cohort_tx(conn, task))
            finally:
                incoming.unlink(missing_ok=True)

    def _summary(self, conn, row):
        frozen = conn.execute('SELECT freeze_hash,payload FROM blind_oot_freezes WHERE cohort_id=?', (row['id'],)).fetchone()
        evaluation = conn.execute('SELECT state,read_started_at FROM blind_oot_evaluations WHERE cohort_id=?', (row['id'],)).fetchone()
        return {'cohort_id': row['id'], 'task_id': row['task_id'],
                'mode': 'governed_blind_oot', 'state': evaluation['state'] if evaluation else ('frozen' if frozen else 'sealed'),
                'freeze_hash': frozen['freeze_hash'] if frozen else None,
                'historical_availability': 'unknown', 'label_maturity': 'unknown',
                'host_user_filesystem_isolated': False,
                'model_evidence': ({'algorithm': json.loads(frozen['payload'])['artifact']['algorithm'],
                    'feature_count': len(json.loads(frozen['payload'])['artifact']['feature_list']),
                    'native_receipt_verified': True, 'immutable_dependency_closure': True} if frozen else None),
                'sample_read_started': bool(evaluation and evaluation['read_started_at'] is not None)}

    def summary(self, task, actor):
        with connect(self.db_path) as conn:
            row = cohort_tx(conn, task)
            if row is None:
                raise HoldoutError('holdout_not_found', 404)
            purpose = next((purpose for purpose, col in (
                ('develop', 'developer_id'), ('evaluate', 'evaluator_id'), ('custody', 'custodian_id')) if row[col] == actor), None)
            if purpose is None:
                raise HoldoutError('holdout_scope_forbidden', 403)
            summary = self._summary(conn, self._cohort(conn, task, actor, purpose))
            summary['viewer_role'] = purpose
            summary['recovery_options'] = []
            if purpose == 'custody' and summary['state'] in {'prepared', 'reading', 'uncertain'}:
                if not summary['sample_read_started']:
                    summary['recovery_options'].append('resume_before_read')
                try:
                    self._result(row, summary['freeze_hash'])
                except HoldoutError:
                    pass
                else:
                    summary['recovery_options'].append('publish_result')
            summary['recovery_id'] = None
            if purpose == 'evaluate':
                approval = conn.execute('SELECT r.* FROM blind_oot_recoveries r JOIN blind_oot_evaluations e ON e.cohort_id=r.cohort_id WHERE r.cohort_id=? AND r.freeze_hash=? AND r.owner_token=e.owner_token AND r.consumed_at IS NULL AND r.expires_at>? ORDER BY r.created_at DESC LIMIT 1', (row['id'], summary['freeze_hash'], time.time())).fetchone()
                if approval:
                    try:
                        self._actor(conn, approval['approved_by'], {'admin'})
                    except HoldoutError:
                        pass
                    else:
                        summary['recovery_id'] = approval['id']
            return summary

    def freeze(self, task, payload, actor):
        # Metadata, dependency bytes, preprocessing and source evidence are one
        # closure. Arbitrary pickle uploads and historical missing receipts fail.
        repo = ModelingRepository(self.db_path)
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            cohort = self._cohort(conn, task, actor, 'develop')
            old = conn.execute('SELECT * FROM blind_oot_freezes WHERE cohort_id=?', (cohort['id'],)).fetchone()
            if old:
                if json.loads(old['payload'])['artifact']['id'] != payload.artifact_id:
                    raise HoldoutError('final_model_freeze_immutable')
                return self._summary(conn, cohort)
            artifact = repo.get_model_artifact(payload.artifact_id)
            if artifact is None:
                raise HoldoutError('model_artifact_not_found', 404)
            experiment = repo.get_experiment(artifact.experiment_id)
            contract = json.loads(cohort['contract'])
            if experiment is None or experiment.task_id != task or experiment.artifact_id != artifact.id:
                raise HoldoutError('model_artifact_binding_failed')
            if artifact.algorithm not in SUPPORTED_ALGORITHMS or experiment.config.target_type != contract['target_type']:
                raise HoldoutError('holdout_model_profile_unsupported', 422)
            if experiment.config.target_col != contract['target_col']:
                raise HoldoutError('holdout_target_binding_failed')
            if experiment.metrics is None or any(value is not None for key, value in asdict(experiment.metrics).items()
                                                 if '_oot_' in '_' + key + '_'):
                raise HoldoutError('model_was_not_developed_blind')
            directory = self.settings.tasks_dir / task / 'modeling_artifacts'
            try:
                record, snapshots = load_receipt(self.db_path, experiment, artifact, directory)
            except Exception as exc:
                raise HoldoutError('native_model_receipt_missing_or_invalid') from exc
            dataset = conn.execute('SELECT * FROM datasets WHERE id=? AND task_id=?',
                                   (experiment.config.dataset_id, task)).fetchone()
            if dataset is None:
                raise HoldoutError('frozen_development_source_required')
            source_path = self.settings.datasets_dir / dataset['source_path']
            with authenticated_file_snapshot(source_path, root=self.settings.datasets_dir,
                                              expected_sha256=dataset['content_hash']):
                pass
            # Source facts remain unknown where the existing producer did not
            # establish them; freezing never upgrades historical assertions.
            from marvis.feature.preprocessing import sidecar_path
            lineage = sidecar_path(source_path)
            source = dict(dataset)
            source['lineage'] = json.loads(lineage.read_text()) if lineage.exists() else None
            closure = {'schema_version': 'marvis.blind_oot.freeze.v1', 'cohort_id': cohort['id'],
                       'sealed_hash': cohort['sealed_hash'], 'artifact': asdict(artifact),
                       'experiment': asdict(experiment), 'producer_receipt': record,
                       'files': record['provenance']['files'], 'source': source}
            encoded = canonical(closure).encode()
            freeze_hash = digest(encoded)
            for name, data in snapshots.items():
                self._publish(self._path('freezes', freeze_hash, name), data)
            self._publish(self._path('freezes', freeze_hash, 'closure.json'), encoded)
            conn.execute('INSERT INTO blind_oot_freezes VALUES(?,?,?,?,?)',
                         (cohort['id'], freeze_hash, encoded.decode(), actor, time.time()))
            return self._summary(conn, cohort)

    def _result(self, row, freeze_hash):
        path = self._path('results', row['id'] + '.json')
        if not path.exists():
            raise HoldoutError('holdout_result_not_available')
        try:
            envelope = json.loads(path.read_bytes())
            body = envelope['body']
            signature = envelope['signature']
        except (ValueError, KeyError, TypeError) as exc:
            raise HoldoutError('holdout_result_integrity_failed') from exc
        if (not isinstance(body, dict) or not isinstance(signature, str)
                or body.get('cohort_id') != row['id'] or body.get('freeze_hash') != freeze_hash
                or not hmac.compare_digest(signature, self._signature(body))):
            raise HoldoutError('holdout_result_integrity_failed')
        return body, digest(path.read_bytes())

    def result(self, task, actor):
        with connect(self.db_path) as conn:
            row = self._cohort(conn, task, actor, 'evaluate')
            evaluation = conn.execute('SELECT * FROM blind_oot_evaluations WHERE cohort_id=?', (row['id'],)).fetchone()
            if evaluation is None or evaluation['state'] != 'committed':
                raise HoldoutError('holdout_result_not_available')
            body, result_hash = self._result(row, evaluation['freeze_hash'])
            if result_hash != evaluation['result_hash']:
                raise HoldoutError('holdout_result_integrity_failed')
            return body

    def recover(self, task, payload, actor):
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._cohort(conn, task, actor, 'custody')
            evaluation = conn.execute('SELECT * FROM blind_oot_evaluations WHERE cohort_id=?', (row['id'],)).fetchone()
            if evaluation is None or evaluation['state'] == 'committed':
                raise HoldoutError('holdout_recovery_not_applicable')
            if payload.mode == 'resume_before_read' and evaluation['read_started_at'] is not None:
                raise HoldoutError('holdout_cannot_repeat_sample_read')
            if payload.mode == 'publish_result':
                self._result(row, evaluation['freeze_hash'])
            recovery_id = 'recovery_' + uuid.uuid4().hex
            conn.execute('INSERT INTO blind_oot_recoveries VALUES(?,?,?,?,?,?,?,?,?,NULL)',
                         (recovery_id, row['id'], evaluation['freeze_hash'], evaluation['owner_token'],
                          payload.mode, actor, payload.reason, time.time(), time.time() + 900))
            return {'recovery_id': recovery_id, 'mode': payload.mode}

    def evaluate(self, task, payload, actor):
        owner = uuid.uuid4().hex
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            cohort = self._cohort(conn, task, actor, 'evaluate')
            frozen = conn.execute('SELECT * FROM blind_oot_freezes WHERE cohort_id=?', (cohort['id'],)).fetchone()
            if frozen is None:
                raise HoldoutError('final_model_freeze_required')
            if payload.freeze_hash != frozen['freeze_hash']:
                raise HoldoutError('holdout_freeze_binding_failed')
            old = conn.execute('SELECT * FROM blind_oot_evaluations WHERE cohort_id=?', (cohort['id'],)).fetchone()
            if old and old['state'] == 'committed':
                body, result_hash = self._result(cohort, frozen['freeze_hash'])
                if result_hash != old['result_hash']:
                    raise HoldoutError('holdout_result_integrity_failed')
                return body
            if old:
                if not payload.recovery_id:
                    raise HoldoutError('holdout_evaluation_busy_or_recovery_required')
                recovery = conn.execute('SELECT * FROM blind_oot_recoveries WHERE id=?', (payload.recovery_id,)).fetchone()
                if (recovery is None or recovery['cohort_id'] != cohort['id'] or recovery['freeze_hash'] != frozen['freeze_hash']
                        or recovery['owner_token'] != old['owner_token'] or recovery['consumed_at'] is not None
                        or recovery['expires_at'] <= time.time()):
                    raise HoldoutError('holdout_recovery_binding_failed')
                self._actor(conn, recovery['approved_by'], {'admin'})
                conn.execute('UPDATE blind_oot_recoveries SET consumed_at=? WHERE id=?', (time.time(), recovery['id']))
                if recovery['mode'] == 'publish_result':
                    body, result_hash = self._result(cohort, frozen['freeze_hash'])
                    conn.execute("UPDATE blind_oot_evaluations SET state='committed',result_hash=?,error_code=NULL WHERE cohort_id=?",
                                 (result_hash, cohort['id']))
                    return body
                if old['read_started_at'] is not None:
                    raise HoldoutError('holdout_cannot_repeat_sample_read')
                conn.execute("UPDATE blind_oot_evaluations SET state='prepared',owner_token=?,error_code=NULL WHERE cohort_id=?",
                             (owner, cohort['id']))
            else:
                conn.execute('INSERT INTO blind_oot_evaluations VALUES(?,?,?,?,?,?,?,NULL,NULL,NULL)',
                             (cohort['id'], frozen['freeze_hash'], 'prepared', owner, actor, payload.request_id, time.time()))
        # This durable reservation precedes *all* sealed-row access. Every error
        # keeps its budget; a caller-controlled request id cannot release it.
        try:
            closure = json.loads(frozen['payload'])
            if digest(canonical(closure).encode()) != frozen['freeze_hash']:
                raise HoldoutError('holdout_freeze_integrity_failed')
            directory = self._path('freezes', frozen['freeze_hash'])
            if digest((directory / 'closure.json').read_bytes()) != frozen['freeze_hash']:
                raise HoldoutError('holdout_freeze_integrity_failed')
            snapshots = verify_files(directory, closure['files'])
            # Load only producer-authenticated frozen dependencies. Use a private
            # pinned directory so subsequent OS file changes cannot affect score.
            runtime_dir = self._path('runtime', owner)
            runtime_dir.mkdir(mode=0o700, parents=True)
            try:
                for name, data in snapshots.items():
                    self._publish(runtime_dir / name, data)
                model_data = closure['artifact']
                for name in ('feature_list', 'feature_importance', 'scorecard_table'):
                    model_data[name] = tuple(model_data[name])
                artifact = ModelArtifact(**model_data)
                scorer = _ModelArtifactScorer(artifact, base_dir=runtime_dir, replay_preprocessing=True)
                self._mark_read_started(cohort, frozen['freeze_hash'], owner, actor)
                result = self._score_sealed(cohort, closure, scorer)
            finally:
                for path in runtime_dir.rglob('*'):
                    if path.is_file():
                        path.chmod(0o600)
                shutil.rmtree(runtime_dir)
            body = {'schema_version': 'marvis.blind_oot.result.v1', 'cohort_id': cohort['id'],
                    'task_id': task, 'freeze_hash': frozen['freeze_hash'], 'sealed_hash': cohort['sealed_hash'],
                    'metrics': result, 'historical_availability': 'unknown', 'label_maturity': 'unknown',
                    'external_authority': 'custodian_declared_not_independently_verified'}
            encoded = canonical({'body': body, 'signature': self._signature(body)}).encode()
            self._publish(self._path('results', cohort['id'] + '.json'), encoded)
            self._commit_result(cohort['id'], owner, digest(encoded))
            return body
        except Exception as exc:
            with connect(self.db_path) as conn:
                conn.execute("UPDATE blind_oot_evaluations SET state='uncertain',error_code='evaluation_failed' WHERE cohort_id=? AND owner_token=? AND state!='committed'",
                             (cohort['id'], owner))
            if isinstance(exc, HoldoutError):
                raise
            raise HoldoutError('holdout_evaluation_failed_recovery_required') from exc

    def _mark_read_started(self, cohort, freeze_hash, owner, actor):
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._cohort(conn, cohort['task_id'], actor, 'evaluate')
            changed = conn.execute("UPDATE blind_oot_evaluations SET state='reading',read_started_at=? WHERE cohort_id=? AND freeze_hash=? AND owner_token=? AND state='prepared' AND read_started_at IS NULL",
                                   (time.time(), cohort['id'], freeze_hash, owner)).rowcount
            if changed != 1:
                raise HoldoutError('holdout_read_admission_failed')

    def _commit_result(self, cohort, owner, result_hash):
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT task_id FROM blind_oot_cohorts WHERE id=?', (cohort,)).fetchone()
            actor = conn.execute('SELECT actor_id FROM blind_oot_evaluations WHERE cohort_id=?', (cohort,)).fetchone()
            self._cohort(conn, row['task_id'], actor['actor_id'], 'evaluate')
            changed = conn.execute("UPDATE blind_oot_evaluations SET state='committed',result_hash=? WHERE cohort_id=? AND owner_token=? AND state='reading'",
                                   (result_hash, cohort, owner)).rowcount
            if changed != 1:
                raise HoldoutError('holdout_commit_binding_failed')

    def _score_sealed(self, cohort, closure, scorer):
        contract = json.loads(cohort['contract'])
        path = self._path('sealed', cohort['sealed_hash'], 'original.parquet')
        with authenticated_file_snapshot(path, root=self.private, expected_sha256=cohort['sealed_hash']) as snapshot:
            frame = pq.read_table(snapshot).to_pandas()
        target = frame[contract['target_col']].to_numpy()
        if len(target) < 2 or frame[contract['target_col']].isna().any():
            raise HoldoutError('holdout_labels_invalid')
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss, roc_auc_score
        target_type = contract['target_type']
        if target_type == 'multiclass':
            classes = closure['artifact']['params'].get('classes')
            probabilities = scorer.predict_proba(frame)
            return {'rows': len(frame), 'logloss': float(log_loss(target, probabilities, labels=classes)),
                    'macro_auc': float(roc_auc_score(target, probabilities, multi_class='ovr', average='macro', labels=classes))}
        scores = np.asarray(scorer.score(frame), dtype=float)
        if not np.isfinite(scores).all():
            raise HoldoutError('holdout_scores_invalid')
        if target_type == 'continuous':
            return {'rows': len(frame), 'rmse': float(np.sqrt(mean_squared_error(target, scores))),
                    'mae': float(mean_absolute_error(target, scores)), 'r2': float(r2_score(target, scores))}
        if not set(target) == {0, 1}:
            raise HoldoutError('holdout_binary_labels_invalid')
        from marvis.feature.metrics import feature_ks, feature_auc
        return {'rows': len(frame), 'ks': float(feature_ks(scores, target)), 'auc': float(feature_auc(scores, target))}
