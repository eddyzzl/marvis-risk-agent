"""Signed immutable transfer into the existing model, strategy and Workflow stores.

    Remote model bytes are never unpickled. This carrier admits a typed binary LR
    preprocessing pipeline; other estimator families are explicitly unsupported.
"""

from __future__ import annotations

import base64
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from marvis.db_schema import connect
from marvis.institution.policy import authorize_on_connection
from marvis.institution.store import IdentityError, require
from marvis.plugins.publisher_trust import canonical, decode_key, digest, identifier, version


SCHEMA = 'marvis.instance-resource.v2'
BUDGET = 8*1024*1024


def _encode_material(material):
    # JSON's numeric values cannot preserve canonical spelling in a browser:
    # 0.0 becomes 0 and exponent formatting differs across runtimes. Carry the
    # exact signed UTF-8 bytes instead of weakening the material hash.
    return {'encoding':'base64.canonical-json.v1',
            'payload':base64.b64encode(canonical(material)).decode()}


def _decode_material(wire):
    require(isinstance(wire,dict) and set(wire)=={'encoding','payload'}
            and wire['encoding']=='base64.canonical-json.v1'
            and isinstance(wire['payload'],str),'sync_material_contract',422)
    require(len(wire['payload'])<=BUDGET,'sync_material_budget',413)
    try:
        raw=base64.b64decode(wire['payload'],validate=True)
        material=json.loads(raw)
        # Reject alternate spellings and duplicate keys before native admission.
        valid=isinstance(material,dict) and canonical(material)==raw
    except Exception as exc:
        raise IdentityError('sync_material_contract',422) from exc
    require(valid,'sync_material_contract',422)
    return material


def node_identity(workspace):
    root = Path(workspace)/'.institution'
    root.mkdir(mode=0o700,exist_ok=True)
    path = root/'node-signing.key'
    require(not root.is_symlink() and not path.is_symlink(),'sync_node_key_invalid')
    if not path.exists():
        # Publish complete private bytes atomically: another process must never
        # observe an O_EXCL-created but still empty key file.
        fd,temporary = tempfile.mkstemp(dir=root,prefix='.node-')
        try:
            with os.fdopen(fd,'wb') as stream:
                stream.write(Ed25519PrivateKey.generate().private_bytes_raw())
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary,path)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    require(path.stat().st_mode & 0o077 == 0,'sync_node_key_permissions')
    raw = path.read_bytes()
    require(len(raw)==32,'sync_node_key_invalid')
    key = Ed25519PrivateKey.from_private_bytes(raw)
    public = key.public_key().public_bytes_raw()
    return {'id':'node-'+hashlib.sha256(public).hexdigest(),
            'public_key':base64.b64encode(public).decode(),'algorithm':'Ed25519'},key


def require_trusted_node(conn,node_id):
    row = conn.execute('SELECT * FROM instance_trusted_nodes WHERE id=?',(node_id,)).fetchone()
    require(row and row['revoked_at'] is None,'sync_node_untrusted')
    return row


def authorize_synced_ref(conn,native_ref):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='instance_resource_versions'").fetchone():
        return
    rows = conn.execute('SELECT * FROM instance_resource_versions WHERE native_ref=?',(native_ref,)).fetchall()
    for row in rows:
        require_trusted_node(conn,row['origin_node'])
        head = conn.execute('SELECT enabled FROM instance_resource_heads WHERE origin_node=? AND kind=? AND resource_name=?',
                            (row['origin_node'],row['kind'],row['resource_name'])).fetchone()
        require(head and head['enabled'],'sync_resource_withdrawn')


def _archive(root):
    out = io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(root).rglob('*')):
            require(not path.is_symlink(),'sync_material_symlink')
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(root).as_posix(),date_time=(1980,1,1,0,0,0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info,path.read_bytes())
    require(out.tell()<=BUDGET,'sync_material_budget',413)
    return base64.b64encode(out.getvalue()).decode()


def _lr_material(settings,artifact_id):
    from marvis.repositories.modeling import ModelingRepository
    from marvis.packs.modeling.handoff import _artifact_base_dir
    from marvis.packs.modeling.producer_receipts import load_receipt
    from marvis.packs.modeling.artifact import load_model
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    repo = ModelingRepository(settings.db_path)
    artifact = repo.get_model_artifact(artifact_id)
    require(artifact and artifact.algorithm=='lr' and not artifact.params.get('calibration')
            and not artifact.params.get('preprocessing_steps'),
            'sync_model_family_unsupported',422)
    experiment = repo.get_experiment(artifact.experiment_id)
    directory = _artifact_base_dir(settings,experiment)
    try:
        proof,snapshots = load_receipt(settings.db_path,experiment,artifact,directory)
    except Exception as exc:
        raise IdentityError('sync_native_source_changed',409) from exc
    # The native receipt validates original bytes before this local native load.
    model = load_model(artifact,base_dir=directory)
    require(type(model) is Pipeline and list(model.named_steps)==['impute','scale','lr']
            and type(model.named_steps['impute']) is SimpleImputer
            and type(model.named_steps['scale']) is StandardScaler
            and type(model.named_steps['lr']) is LogisticRegression,'sync_model_family_unsupported',422)
    impute,scale,lr = (model.named_steps[k] for k in ('impute','scale','lr'))
    require(impute.strategy=='median' and impute.add_indicator is False and scale.with_mean and scale.with_std
            and lr.classes_.tolist()==[0,1],'sync_model_preprocessing_unsupported',422)
    material = {'schema_version':'marvis.native-lr-transfer.v1','features':list(artifact.feature_list),
                'statistics':impute.statistics_.tolist(),'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),
                'variance':scale.var_.tolist(),'coefficients':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),
                'n_samples_seen':int(scale.n_samples_seen_), 'source_closure_hash':digest(proof['provenance'])}
    _validate_lr(material)
    return material


def _validate_lr(value):
    import math
    keys = {'schema_version','features','statistics','mean','scale','variance','coefficients','intercept',
            'n_samples_seen','source_closure_hash'}
    require(isinstance(value,dict) and set(value)==keys and value['schema_version']=='marvis.native-lr-transfer.v1',
            'sync_model_contract',422)
    features = value['features']
    require(isinstance(features,list) and 0<len(features)<=4096 and len(features)==len(set(features))
            and all(isinstance(f,str) and 0<len(f)<=128 for f in features),'sync_model_features',422)
    for key in ('statistics','mean','scale','variance','coefficients'):
        numbers = value[key]
        require(isinstance(numbers,list) and len(numbers)==len(features)
                and all(type(v) in (float,int) and math.isfinite(v) for v in numbers),'sync_model_contract',422)
    require(all(v>0 for v in value['scale']) and all(v>=0 for v in value['variance'])
            and type(value['intercept']) in (int,float) and math.isfinite(value['intercept'])
            and type(value['n_samples_seen']) is int and value['n_samples_seen']>0
            and isinstance(value['source_closure_hash'],str) and len(value['source_closure_hash'])==64,
            'sync_model_contract',422)


def _reconstruct_lr(material):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    _validate_lr(material)
    n = len(material['features'])
    impute = SimpleImputer(strategy='median')
    # Initialize the installed sklearn's non-semantic fitted bookkeeping from
    # the declared finite statistic vector; never fit a remote population.
    impute.fit(np.asarray([material['statistics']],dtype=float))
    impute.statistics_ = np.asarray(material['statistics'],dtype=float)
    impute.n_features_in_,impute._fit_dtype,impute.indicator_ = n,np.dtype('float64'),None
    impute.feature_names_in_ = np.asarray(material['features'],dtype=object)
    scale = StandardScaler()
    scale.fit(np.zeros((1,n)))
    scale.n_features_in_,scale.n_samples_seen_ = n,material['n_samples_seen']
    scale.mean_,scale.scale_,scale.var_ = (np.asarray(material[k],dtype=float) for k in ('mean','scale','variance'))
    lr = LogisticRegression()
    lr.classes_,lr.n_features_in_,lr.n_iter_ = np.asarray([0,1]),n,np.asarray([1])
    lr.coef_,lr.intercept_ = np.asarray([material['coefficients']]),np.asarray([material['intercept']])
    return Pipeline([('impute',impute),('scale',scale),('lr',lr)])


class SynchronizationStore:
    def __init__(self,settings):
        self.settings,self.db_path = settings,Path(settings.db_path)

    def node(self):
        return node_identity(self.settings.workspace)[0]

    def trust_node(self,node_id,public_key):
        raw = decode_key(public_key)
        require(node_id=='node-'+hashlib.sha256(raw).hexdigest(),'sync_node_identity_mismatch',422)
        with connect(self.db_path) as conn:
            old = conn.execute('SELECT * FROM instance_trusted_nodes WHERE id=?',(node_id,)).fetchone()
            if old:
                require(old['public_key']==public_key and old['revoked_at'] is None,'sync_node_conflict',409)
            else:
                conn.execute('INSERT INTO instance_trusted_nodes VALUES(?,?,?,?,NULL)',
                             (node_id,public_key,hashlib.sha256(raw).hexdigest(),time.time()))
        return {'id':node_id,'trusted':True}

    def node_revocation_proposal(self,node_id):
        with connect(self.db_path) as conn:
            require_trusted_node(conn,node_id)
            heads=[dict(row) for row in conn.execute('SELECT kind,resource_name,version_id,revision,enabled FROM instance_resource_heads WHERE origin_node=? ORDER BY kind,resource_name',(node_id,))]
        binding={'node_id':node_id,'heads':heads}
        return binding|{'proposal_hash':digest(binding)}

    def revoke_node(self,node_id,expected_heads,proposal_hash):
        binding={'node_id':node_id,'heads':expected_heads}
        require(digest(binding)==proposal_hash,'sync_node_revocation_binding',409)
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            replay=conn.execute('SELECT receipt_json FROM instance_sync_receipts WHERE envelope_hash=?',
                                ('node-revoke:'+proposal_hash,)).fetchone()
            if replay:
                return json.loads(replay['receipt_json'])|{'replayed':True}
            require_trusted_node(conn,node_id)
            actual=[dict(row) for row in conn.execute('SELECT kind,resource_name,version_id,revision,enabled FROM instance_resource_heads WHERE origin_node=? ORDER BY kind,resource_name',(node_id,))]
            require(actual==expected_heads,'sync_node_revocation_head_changed',409)
            cursor = conn.execute('UPDATE instance_trusted_nodes SET revoked_at=COALESCE(revoked_at,?) WHERE id=?',
                                  (time.time(),node_id))
            require(cursor.rowcount,'sync_node_not_found',404)
            conn.execute('UPDATE instance_resource_heads SET enabled=0,revision=revision+1 WHERE origin_node=?',(node_id,))
            receipt={'id':node_id,'revoked':True,'reviewed_heads':expected_heads,'proposal_hash':proposal_hash}
            conn.execute('INSERT INTO instance_sync_receipts VALUES(?,?,?)',
                         ('node-revoke:'+proposal_hash,canonical(receipt).decode(),time.time()))
        return receipt

    def export(self,kind,native_ref,name,release_version,receiver_node,actor):
        identifier(name,'name')
        version(release_version)
        require(kind in {'model','strategy','workflow'},'sync_kind_unsupported',422)
        with connect(self.db_path) as conn:
            require_trusted_node(conn,receiver_node)
            if kind!='workflow':
                authorize_on_connection(conn,'artifact' if kind=='model' else 'strategy',native_ref,'share',actor_id=actor)
        if kind=='model':
            material = _lr_material(self.settings,native_ref)
            permissions = []
        elif kind=='strategy':
            from marvis.repositories.strategy import StrategyRepository
            from marvis.packs.strategy.legacy_adapter import legacy_strategy_to_spec
            strategy = StrategyRepository(self.db_path).get_strategy(native_ref)
            require(strategy,'sync_resource_not_found',404)
            spec = strategy.spec or legacy_strategy_to_spec(strategy)
            material = {'schema_version':'marvis.strategy-transfer.v1','spec':spec.to_dict()}
            permissions = []
        else:
            from marvis.plugins.capability_versions import CapabilityStore
            row = CapabilityStore(self.db_path).resolve('workflow',native_ref,release_version)
            material = {'schema_version':'marvis.workflow-transfer.v1','archive':_archive(row['material_path']),
                        'workflow_name':row['name'],'workflow_version':row['version']}
            require(name==row['name'] and release_version==row['version'],'sync_workflow_identity_mismatch',422)
            permissions = json.loads(row['permissions_json'])
        material = _encode_material(material)
        require(len(canonical(material))<=BUDGET,'sync_material_budget',413)
        node,key = node_identity(self.settings.workspace)
        # A signer cannot silently reuse a public origin version for new bytes,
        # even before any receiver has imported the first envelope.
        export_identity = 'export-version-'+digest([node['id'],kind,name,release_version])
        binding = {'origin_node':node['id'],'kind':kind,'name':name,'version':release_version,
                   'material_hash':digest(material),'native_ref':native_ref}
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            old = conn.execute('SELECT receipt_json FROM instance_sync_receipts WHERE envelope_hash=?',
                               (export_identity,)).fetchone()
            if old:
                require(json.loads(old['receipt_json'])==binding,'sync_export_version_reused',409)
            else:
                conn.execute('INSERT INTO instance_sync_receipts VALUES(?,?,?)',
                             (export_identity,canonical(binding).decode(),time.time()))
        body = {'schema_version':SCHEMA,'origin_node':node['id'],'receiver_node':receiver_node,'kind':kind,
                'name':name,'version':release_version,'material_hash':digest(material),'permissions':permissions,
                'expires_at':time.time()+900}
        return {'envelope':{'body':body,'signature':base64.b64encode(key.sign(canonical(body))).decode()},
                'material':material}

    def verify_bundle(self,envelope,material):
        require(isinstance(envelope,dict) and set(envelope)=={'body','signature'},'sync_signature_contract',422)
        body = envelope['body']
        require(isinstance(body,dict) and set(body)=={'schema_version','origin_node','receiver_node','kind','name',
                'version','material_hash','permissions','expires_at'} and body['schema_version']==SCHEMA,
                'sync_signature_contract',422)
        require(body['receiver_node']==self.node()['id'],'sync_receiver_mismatch')
        identifier(body['name'],'name')
        version(body['version'])
        require(body['kind'] in {'model','strategy','workflow'} and isinstance(body['permissions'],list)
                and all(isinstance(p,str) for p in body['permissions']),'sync_signature_contract',422)
        require(type(body['expires_at']) in (int,float) and time.time()<body['expires_at']<=time.time()+905,
                'sync_bundle_expired')
        with connect(self.db_path) as conn:
            node = require_trusted_node(conn,body['origin_node'])
            try:
                Ed25519PublicKey.from_public_bytes(decode_key(node['public_key'])).verify(
                    base64.b64decode(envelope['signature'],validate=True),canonical(body))
            except Exception as exc:
                raise IdentityError('sync_signature_invalid') from exc
        require(len(canonical(material))<=BUDGET,'sync_material_budget',413)
        require(digest(material)==body['material_hash'],'sync_material_changed',422)
        return body,_decode_material(material)

    def _head(self,conn,body):
        return conn.execute('SELECT * FROM instance_resource_heads WHERE origin_node=? AND kind=? AND resource_name=?',
                            (body['origin_node'],body['kind'],body['name'])).fetchone()

    def _admit_tx(self,conn,body,expected_revision):
        require_trusted_node(conn,body['origin_node'])
        head = self._head(conn,body)
        require((head['revision'] if head else 0)==expected_revision,'sync_head_conflict',409)
        return head

    def _require_new_version(self,conn,body):
        old = conn.execute('SELECT id FROM instance_resource_versions WHERE origin_node=? AND kind=? AND resource_name=? AND version=?',
                           (body['origin_node'],body['kind'],body['name'],body['version'])).fetchone()
        require(not old,'sync_version_already_imported_use_rollback',409)

    def _publish_tx(self,conn,body,envelope,material,native_ref,actor,expected_revision):
        self._admit_tx(conn,body,expected_revision)
        version_id = digest([body['origin_node'],body['kind'],body['name'],body['version']])
        existing = conn.execute('SELECT * FROM instance_resource_versions WHERE id=?',(version_id,)).fetchone()
        require(not existing,'sync_version_already_imported_use_rollback',409)
        conn.execute('INSERT INTO instance_resource_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                     (version_id,body['origin_node'],body['kind'],body['name'],body['version'],body['material_hash'],
                      canonical(envelope).decode(),canonical(material).decode(),native_ref,actor,time.time()))
        conn.execute('''INSERT INTO instance_resource_heads VALUES(?,?,?,?,?,1)
             ON CONFLICT(origin_node,kind,resource_name) DO UPDATE SET version_id=excluded.version_id,
             revision=excluded.revision,enabled=1''',
             (body['origin_node'],body['kind'],body['name'],version_id,expected_revision+1))
        receipt = {'version_id':version_id,'origin_node':body['origin_node'],'kind':body['kind'],'name':body['name'],
                   'version':body['version'],'content_hash':body['material_hash'],'native_ref':native_ref,
                   'revision':expected_revision+1,'envelope_hash':digest(envelope)}
        conn.execute('INSERT INTO instance_sync_receipts VALUES(?,?,?)',
                     (digest(envelope),canonical(receipt).decode(),time.time()))
        return receipt

    def import_bundle(self,envelope,material,*,task_id,actor,expected_revision,approved_permissions,app):
        body,material = self.verify_bundle(envelope,material)
        require(approved_permissions==body['permissions'],'sync_permissions_not_approved')
        with connect(self.db_path) as conn:
            authorize_on_connection(conn,'task',task_id,'write',actor_id=actor)
            replay = conn.execute('SELECT receipt_json FROM instance_sync_receipts WHERE envelope_hash=?',
                                  (digest(envelope),)).fetchone()
            if replay:
                receipt = json.loads(replay['receipt_json'])
                head = self._head(conn,body)
                require(head and head['version_id']==receipt['version_id'] and head['revision']==receipt['revision'],
                        'sync_replay_head_changed',409)
                self._readback(conn,conn.execute('SELECT * FROM instance_resource_versions WHERE id=?',
                                                (receipt['version_id'],)).fetchone())
                return receipt|{'replayed':True}
            self._admit_tx(conn,body,expected_revision)
            self._require_new_version(conn,body)
        if body['kind']=='workflow':
            return self._import_workflow(body,envelope,material,actor,expected_revision,app)
        if body['kind']=='model':
            return self._import_model(body,envelope,material,task_id,actor,expected_revision)
        require(isinstance(material,dict) and set(material)=={'schema_version','spec'}
                and material['schema_version']=='marvis.strategy-transfer.v1','sync_strategy_contract',422)
        from marvis.packs.strategy.dsl import parse_strategy_spec
        from marvis.packs.strategy.strategy import build_strategy_from_spec
        from marvis.repositories.strategy import _insert_strategy_row
        spec = parse_strategy_spec(material['spec'])
        native_ref = 'sync-strategy-'+digest([body['origin_node'],body['name'],body['version']])[:32]
        strategy = replace(build_strategy_from_spec(spec),id=native_ref)
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._admit_tx(conn,body,expected_revision)
            self._require_new_version(conn,body)
            _insert_strategy_row(conn,task_id,strategy,datetime.now(UTC).isoformat())
            return self._publish_tx(conn,body,envelope,material,native_ref,actor,expected_revision)

    def _import_model(self,body,envelope,material,task_id,actor,expected_revision):
        from marvis.artifacts import ArtifactUnitOfWork
        from marvis.packs.modeling.artifact import save_model
        from marvis.packs.modeling.contracts import Experiment,TrainConfig
        from marvis.packs.modeling.producer_receipts import capture_native_writes,finish_capture,register_on_connection
        from marvis.repositories.modeling import _insert_experiment_row,_insert_model_artifact_row
        model = _reconstruct_lr(material)
        native_id = 'sync-experiment-'+digest([body['origin_node'],body['name'],body['version']])[:32]
        directory = self.settings.tasks_dir/task_id/'modeling_artifacts'
        uow = ArtifactUnitOfWork()
        temporary = tempfile.TemporaryDirectory(prefix='marvis-sync-model-')
        staged_path = Path(temporary.name)
        try:
            with capture_native_writes() as writes:
                artifact = save_model(model,'lr',staged_path,feature_list=material['features'],params={
                    'instance_origin_node':body['origin_node'],'instance_content_hash':body['material_hash']})
                artifact = replace(artifact,experiment_id=native_id)
                receipt = finish_capture(writes,artifact,staged_path)
            # Native scorers expect the established modeling_artifacts directory.
            # Each immutable file is promoted individually, never replacing it.
            directory.mkdir(parents=True,exist_ok=True)
            for path in staged_path.iterdir():
                if path.name=='model_meta.json':
                    continue
                target = uow.stage_file(directory,path.name)
                target.path.write_bytes(path.read_bytes())
            config = TrainConfig('',tuple(material['features']),'','',{}, {},0,None,recipe_id='instance_import')
            experiment = Experiment(native_id,task_id,'instance_import',config,None,artifact.id,'imported',
                                    datetime.now(UTC).isoformat())
            with connect(self.db_path) as conn:
                conn.execute('BEGIN IMMEDIATE')
                self._admit_tx(conn,body,expected_revision)
                self._require_new_version(conn,body)
                authorize_on_connection(conn,'task',task_id,'write',actor_id=actor)
                uow.promote_all()
                _insert_experiment_row(conn,experiment)
                _insert_model_artifact_row(conn,artifact)
                receipt['directory']=str(directory)
                register_on_connection(conn,self.db_path,artifact,receipt)
                result = self._publish_tx(conn,body,envelope,material,artifact.id,actor,expected_revision)
            uow.commit()
            return result
        except Exception:
            uow.rollback()
            raise
        finally:
            temporary.cleanup()

    def _import_workflow(self,body,envelope,material,actor,expected_revision,app):
        from marvis.plugins.capability_versions import CapabilityStore
        from marvis.plugins.governed_install import propose_archive,confirm_proposal
        require(isinstance(material,dict) and set(material)=={'schema_version','archive','workflow_name','workflow_version'}
                and material['schema_version']=='marvis.workflow-transfer.v1'
                and material['workflow_name']==body['name'] and material['workflow_version']==body['version'],
                'sync_workflow_contract',422)
        try:
            raw = base64.b64decode(material['archive'],validate=True)
        except Exception as exc:
            raise IdentityError('sync_workflow_archive_invalid',422) from exc
        require(len(raw)<=BUDGET,'sync_material_budget',413)
        capability = CapabilityStore(self.db_path)
        head = capability.head('workflow',body['name'])
        native_revision = head['revision'] if head else 0
        with tempfile.TemporaryDirectory(prefix='marvis-sync-workflow-') as temp:
            path = Path(temp)/'workflow.zip'
            path.write_bytes(raw)
            proposal = propose_archive(path,db_path=self.db_path,registry=app.state.plugin_registry,
                tools=app.state.tool_registry,validator=app.state.plan_validator,actor=actor,
                expected_revision=native_revision,kind='workflow',expected_name=body['name'])
        require(proposal['permissions']==body['permissions'],'sync_workflow_permissions_changed')
        result = {}
        def on_connection(conn,receipt):
            result.update(self._publish_tx(conn,body,envelope,material,'workflow:'+receipt['version_id'],actor,expected_revision))
        confirm_proposal(self.db_path,app.state.plugin_registry,app.state.tool_registry,app.state.plan_validator,
            proposal['id'],proposal_hash=proposal['proposal_hash'],expected_revision=native_revision,
            approved_permissions=body['permissions'],actor=actor,on_connection=on_connection)
        # Same registry refresh is performed by the router after return.
        return result

    def _readback(self,conn,row):
        require_trusted_node(conn,row['origin_node'])
        if row['kind']!='workflow':
            authorize_on_connection(conn,'artifact' if row['kind']=='model' else 'strategy',row['native_ref'])
        material = json.loads(row['material_json'])
        schema=json.loads(row['envelope_json'])['body']['schema_version']
        checked=_encode_material(material) if schema==SCHEMA else material
        require(digest(checked)==row['content_hash'],'sync_stored_material_changed',409)
        if row['kind']=='workflow':
            from marvis.plugins.capability_versions import CapabilityStore
            CapabilityStore(self.db_path)._version(conn,row['native_ref'].split(':',1)[1])
        elif row['kind']=='strategy':
            from marvis.repositories.strategy import _strategy_from_row
            strategy = _strategy_from_row(conn.execute('SELECT * FROM strategies WHERE id=?',(row['native_ref'],)).fetchone())
            require(strategy.spec.to_dict()==material['spec'],'sync_native_readback_changed',409)
        else:
            from marvis.repositories.modeling import ModelingRepository
            from marvis.packs.modeling.producer_receipts import load_receipt
            from marvis.packs.modeling.handoff import _artifact_base_dir
            repo = ModelingRepository(self.db_path)
            artifact = repo.get_model_artifact(row['native_ref'])
            experiment = repo.get_experiment(artifact.experiment_id)
            try:
                load_receipt(self.db_path,experiment,artifact,_artifact_base_dir(self.settings,experiment))
            except Exception as exc:
                raise IdentityError('sync_native_readback_changed',409) from exc
        return {'version_id':row['id'],'native_ref':row['native_ref'],'content_hash':row['content_hash'],'verified':True}

    def readback(self,version_id):
        with connect(self.db_path) as conn:
            row = conn.execute('SELECT * FROM instance_resource_versions WHERE id=?',(version_id,)).fetchone()
            require(row,'sync_version_not_found',404)
            return self._readback(conn,row)

    def catalog(self):
        with connect(self.db_path) as conn:
            versions=[dict(r) for r in conn.execute('''SELECT v.id,v.origin_node,v.kind,v.resource_name,v.version,
                    v.content_hash,v.native_ref,v.envelope_json,(h.version_id=v.id AND h.enabled=1 AND n.revoked_at IS NULL) AS active,h.revision,h.enabled,n.revoked_at IS NOT NULL AS node_revoked
                    FROM instance_resource_versions v JOIN instance_resource_heads h ON h.origin_node=v.origin_node
                    AND h.kind=v.kind AND h.resource_name=v.resource_name JOIN instance_trusted_nodes n ON n.id=v.origin_node''')]
            for item in versions:
                item['permissions']=json.loads(item.pop('envelope_json'))['body']['permissions']
            return {'node':self.node(),'trusted_nodes':[dict(r) for r in conn.execute('SELECT * FROM instance_trusted_nodes')],
                    'versions':versions}

    def rollback(self,version_id,expected_revision,approved_permissions,actor,app):
        from marvis.plugins.capability_versions import CapabilityStore
        from marvis.plugins.governed_install import confirm_proposal
        with connect(self.db_path) as conn:
            row = conn.execute('SELECT * FROM instance_resource_versions WHERE id=?',(version_id,)).fetchone()
            require(row,'sync_version_not_found',404)
            self._readback(conn,row)
            if row['kind']!='workflow':
                authorize_on_connection(conn,'artifact' if row['kind']=='model' else 'strategy',row['native_ref'],'approve',actor_id=actor)
            body = json.loads(row['envelope_json'])['body']
        require(approved_permissions==body['permissions'],'sync_permissions_not_approved')
        def advance(conn,receipt=None):
            self._admit_tx(conn,body,expected_revision)
            cursor = conn.execute('UPDATE instance_resource_heads SET version_id=?,revision=revision+1 WHERE origin_node=? AND kind=? AND resource_name=? AND revision=?',
                (version_id,body['origin_node'],body['kind'],body['name'],expected_revision))
            require(cursor.rowcount==1,'sync_head_conflict',409)
        if row['kind']=='workflow':
            store = CapabilityStore(self.db_path)
            head = store.head('workflow',body['name'])
            proposal = store.rollback_proposal('workflow',body['name'],body['version'],head['revision'],actor)
            confirm_proposal(self.db_path,app.state.plugin_registry,app.state.tool_registry,app.state.plan_validator,
                proposal['id'],proposal_hash=proposal['proposal_hash'],expected_revision=head['revision'],
                approved_permissions=approved_permissions,actor=actor,on_connection=advance)
        else:
            with connect(self.db_path) as conn:
                conn.execute('BEGIN IMMEDIATE')
                advance(conn)
        return self.readback(version_id)|{'revision':expected_revision+1,'rolled_back':True}
