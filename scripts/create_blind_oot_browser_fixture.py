"""Create authorized synthetic blind OOT cases in a fresh local QA workspace.

Session bindings are saved privately for the local browser test harness, never
printed. No provider calls, production materials or external publication occur.
"""
import argparse
from io import BytesIO
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from marvis.app import create_app
from marvis.plugins.manifest import ToolRef


def parquet(frame):
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    return stream.getvalue()


def samples(seed, n, development=False):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 2
    frame = pd.DataFrame({'x': y * 0.6 + rng.normal(size=n), 'z': rng.normal(size=n), 'y': y})
    if development:
        frame['split'] = ['train'] * (n * 2 // 3) + ['test'] * (n - n * 2 // 3)
    return frame


def create(workspace):
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit('QA workspace must be empty; existing files are preserved')
    app = create_app(workspace)
    clients, bindings = {}, {}
    with TestClient(app) as custodian:
        for name, role, display in [('custodian','admin','演示保管人员'), ('developer','maker','演示开发人员'), ('evaluator','checker','演示复核人员')]:
            client = custodian if name == 'custodian' else TestClient(app)
            claimed = client.post('/api/production-governance/principals/claim',
                headers={'x-marvis-governance-admin':app.state.plugin_admin_token},
                json={'display_name':display,'role':role})
            claimed.raise_for_status()
            clients[name] = client
            bindings[name] = {'principal_id':claimed.json()['id'], 'display_name':display,
                'cookies':dict(client.cookies)}
        tasks = {}
        for index, state in enumerate(['empty', 'sealed', 'frozen', 'committed']):
            source = workspace / f'empty-{state}'
            source.mkdir()
            task = custodian.post('/api/tasks',json={'model_name':f'合成盲 OOT ({state})', 'validator':'软件验收',
                'source_dir':str(source),'task_type':'modeling','run_mode':'manual','recipes':['lr']})
            task.raise_for_status()
            task_id = task.json()['id']
            tasks[state] = task_id
            if state == 'empty':
                continue
            contract = {'data_identity':f'synthetic-browser-{state}', 'developer_id':bindings['developer']['principal_id'],
                'evaluator_id':bindings['evaluator']['principal_id'], 'target_col':'y','split_col':'split',
                'authority_basis':'Authorized synthetic browser acceptance; no institutional or historical claim'}
            seal = custodian.post(f'/api/tasks/{task_id}/blind-oot/cohort', files={'file':('independent.parquet',parquet(samples(831+index,90)),'application/octet-stream')}, data={'contract':json.dumps(contract)})
            seal.raise_for_status()
            dev = clients['developer'].post(f'/api/tasks/{task_id}/blind-oot/development',files={'file':('development.parquet',parquet(samples(441+index,240,True)),'application/octet-stream')})
            dev.raise_for_status()
            trained = app.state.tool_runner.invoke(ToolRef('modeling','train_model'),
                {'dataset_id':dev.json()['dataset_id'],'recipe':'lr','features':['x','z'],'target_col':'y',
                 'split_col':'split','split_values':{'train':'train','test':'test'},'seed':19,'params':{}},task_id=task_id)
            if not trained.ok:
                raise SystemExit('Synthetic native training failed')
            if state == 'sealed':
                continue
            freeze = clients['developer'].post(f'/api/tasks/{task_id}/blind-oot/freeze',json={'artifact_id':trained.output['artifact_id']})
            freeze.raise_for_status()
            if state == 'committed':
                result = clients['evaluator'].post(f'/api/tasks/{task_id}/blind-oot/evaluate',json={'freeze_hash':freeze.json()['freeze_hash']})
                result.raise_for_status()
        binding_file = workspace / '.blind_oot' / 'browser-session-bindings.json'
        binding_file.write_text(json.dumps({'synthetic_only':True,'actors':bindings,'tasks':tasks},ensure_ascii=False,indent=2))
        binding_file.chmod(0o600)
        for state, task_id in tasks.items():
            print(f'{state}: {task_id}')
        print(f'Private browser harness bindings: {binding_file}')
        for client in clients.values():
            if client is not custodian:
                client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    create(parser.parse_args().workspace.resolve())
