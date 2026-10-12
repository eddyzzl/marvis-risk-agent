"""Private synthetic carrier; never contacts a provider or loads user models."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from marvis.institution.policy import claim_resource
from marvis.institution.store import IdentityStore
from marvis.institution.synchronization import SynchronizationStore
from marvis.plugins.publisher_trust import canonical
from marvis.repositories.strategy import StrategyRepository
from marvis.packs.strategy.contracts import Strategy, StrategyRule


def _private_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))
    path.chmod(0o600)


def create(root):
    # Development fixture helpers themselves call installed native Tools and APIs.
    from tests.test_instance_synchronization import _app,_train,_signed_workflow
    require_empty=not root.exists() or not any(root.iterdir())
    if not require_empty:
        raise SystemExit('fixture directory must be fresh and empty')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    idp=Ed25519PrivateKey.generate()
    key_path=root/'reference-idp.key'
    key_path.write_bytes(idp.private_bytes_raw())
    key_path.chmod(0o600)
    source,destination=_app(root/'source'),_app(root/'destination')
    artifact_id,_=_train(source[0],source[3],root)
    strategies=[]
    for release_version,cutoff in [('1.0.0',0.),('2.0.0',1.)]:
        native_ref='browser-strategy-'+release_version
        StrategyRepository(source[0].state.settings.db_path).create_strategy(source[3],Strategy(
            native_ref,'approval',(StrategyRule('x > '+str(cutoff),'approve',None),),'x','reject','Synthetic browser transfer'))
        strategies.append({'kind':'strategy','native_ref':native_ref,'name':'browser_strategy','release_version':release_version})
    publisher=Ed25519PrivateKey.generate()
    for release_version,revision in [('1.0.0',0),('2.0.0',1)]:
        _signed_workflow(source[0],source[1],source[2],root,publisher,release_version,revision)
    result=destination[1].post('/api/plugins/governance/publishers',headers=destination[2],json={
        'id':'synthetic_vendor','display_name':'Synthetic Vendor',
        'public_key':base64.b64encode(publisher.public_key().public_bytes_raw()).decode()})
    if result.status_code!=201:
        raise RuntimeError('native receiver publisher admission failed')
    nodes=[]
    instances=[]
    for name,item in [('source',source),('destination',destination)]:
        store=IdentityStore(item[0].state.settings.db_path)
        store.trust_issuer('synthetic-reference-idp','reference-key',
            base64.b64encode(idp.public_key().public_bytes_raw()).decode())
        subjects={}
        for subject,role in [('reference-admin','admin'),('reference-maker','maker'),
                             ('reference-checker','checker'),('reference-service','service')]:
            subjects[subject]=store.provision_subject('synthetic-reference-idp',subject,'Synthetic '+subject,role)
        actor=subjects['reference-admin']['id']
        claim_resource(store.db_path,actor,'task',item[3],actor,0)
        store.set_mode('institutional','synthetic-browser-'+name,0)
        token_path=root/(name+'-local-admin-token.txt')
        token_path.write_text(item[0].state.plugin_admin_token)
        token_path.chmod(0o600)
        nodes.append(SynchronizationStore(item[0].state.settings).node())
        instances.append({'name':name,'workspace':str(item[0].state.settings.workspace),'db_path':str(store.db_path),
                          'task_id':item[3],'subjects':subjects,'admin_token_path':str(token_path)})
    metadata={'schema_version':'marvis.synthetic-institution-sync-fixture.v1',
              'material':'authorized synthetic reference only','external_identity_verified':False,
              'idp_private_key_path':str(key_path),'issuer':'synthetic-reference-idp','key_id':'reference-key',
              'instances':instances,'nodes':nodes,'exports':[
                  {'kind':'model','native_ref':artifact_id,'name':'browser_lr','release_version':'1.0.0'},*strategies,
                  *[{'kind':'workflow','native_ref':'sync_flow','name':'sync_flow','release_version':v}
                    for v in ('1.0.0','2.0.0')]],
              'starting_sync_versions':0,'starting_node_trust':0}
    path=root/'metadata.json'
    _private_json(path,metadata)
    return path


def login_file(metadata_path,instance,subject):
    value=json.loads(metadata_path.read_text())
    item=next(i for i in value['instances'] if i['name']==instance)
    key=Ed25519PrivateKey.from_private_bytes(Path(value['idp_private_key_path']).read_bytes())
    assertion={k:v for k,v in IdentityStore(item['db_path']).challenge(value['issuer'],subject).items()
               if k in {'issuer','subject','audience','nonce'}}
    assertion.update(schema_version='marvis.identity-assertion.v1',algorithm='Ed25519',key_id=value['key_id'],
                     issued_at=time.time(),expires_at=time.time()+240)
    path=metadata_path.parent/(instance+'-'+subject+'-login.json')
    _private_json(path,{'assertion':assertion,'signature':base64.b64encode(key.sign(canonical(assertion))).decode()})
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--instance',choices=['source','destination'],default='source')
    parser.add_argument('--subject',default='reference-admin')
    args=parser.parse_args()
    path=login_file(args.metadata,args.instance,args.subject) if args.metadata else create(args.root)
    # Print the path only, never authentication bodies or private key material.
    print(path)
