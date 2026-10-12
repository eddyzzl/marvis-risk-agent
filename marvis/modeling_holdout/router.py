"""Explicit custody API; sealed data and evaluation results never enter Agent state."""

import json
from pathlib import Path
import uuid

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.modeling_holdout.contracts import Evaluate, Freeze, HoldoutError, Intake, Recover
from marvis.modeling_holdout.guards import assert_development_frame, record_exposure
from marvis.modeling_holdout.service import HoldoutService, SUPPORTED_ALGORITHMS, MAX_SEALED_BYTES, MAX_DECODED_BYTES, MAX_SEALED_ROWS, MAX_SEALED_COLUMNS
from marvis.repositories.modeling import ModelingRepository
from marvis.production_governance.router import _local_principal_id
from marvis.repositories.datasets import DatasetRepository

router = APIRouter(prefix='/api/tasks/{task_id}/blind-oot', tags=['blind-oot'])


def service(request):
    return HoldoutService(request.app.state.settings)


async def body(request, contract):
    data = await request.body()
    if len(data) > 16_000:
        raise HoldoutError('holdout_request_too_large', 413)
    try:
        return contract.model_validate_json(data)
    except ValueError as exc:
        raise HoldoutError('invalid_holdout_contract', 422) from exc


@router.get('/capabilities')
def capabilities(task_id: str, request: Request):
    instance = service(request)
    with connect(instance.db_path) as conn:
        instance._actor(conn, _local_principal_id(request), {'maker', 'checker', 'admin'})
    return {'mode': 'governed_blind_oot', 'intake_format': 'parquet',
            'model_source': 'native_producer_authenticated_dependency_closure',
            'algorithms': sorted(SUPPORTED_ALGORITHMS),
            'arbitrary_notebooks_and_plugins': 'prohibited_in_blind_tasks',
            'evaluation_budget': 'one_per_logical_cohort_and_row_members',
            'identity_inspection': 'custody_admission_only_before_freeze',
            'identity_reencoding': 'primitive_lexical_projection_parquet_csv_feather_excel',
            'unverifiable_source_policy': 'deny_potential_sealed_overlap_and_future_blind_claim',
            'external_subject_identity': 'unknown',
            'host_user_filesystem_isolated': False,
            'historical_availability': 'unknown', 'label_maturity': 'unknown',
            'resource_profile': {'max_upload_bytes': MAX_SEALED_BYTES, 'max_parquet_decoded_bytes': MAX_DECODED_BYTES,
                'max_rows': MAX_SEALED_ROWS, 'max_columns': MAX_SEALED_COLUMNS, 'hard_worker_resource_isolation': False}}


@router.post('/cohort', status_code=201)
def intake(task_id: str, request: Request, file: UploadFile = File(...), contract: str = Form(...)):
    if len(contract) > 16_000 or Path(file.filename or '').suffix.lower() != '.parquet':
        raise HoldoutError('invalid_holdout_intake', 422)
    try:
        payload = Intake.model_validate_json(contract)
    except ValueError as exc:
        raise HoldoutError('invalid_holdout_contract', 422) from exc
    return service(request).intake(task_id, payload, file.file, _local_principal_id(request))


@router.get('')
def summary(task_id: str, request: Request):
    return service(request).summary(task_id, _local_principal_id(request))


@router.post('/development', status_code=201)
def development(task_id: str, request: Request, file: UploadFile = File(...)):
    instance = service(request)
    settings = instance.settings
    actor = _local_principal_id(request)
    with connect(instance.db_path) as conn:
        row = instance._cohort(conn, task_id, actor, 'develop')
        if conn.execute('SELECT 1 FROM blind_oot_freezes WHERE cohort_id=?', (row['id'],)).fetchone():
            raise HoldoutError('blind_development_is_frozen')
    if Path(file.filename or '').suffix.lower() != '.parquet':
        raise HoldoutError('blind_development_requires_parquet', 422)
    directory = settings.tasks_dir / task_id / 'blind_development_uploads'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid.uuid4().hex + '.parquet')
    try:
        with path.open('xb') as output:
            size = 0
            for chunk in iter(lambda: file.file.read(1024 * 1024), b''):
                size += len(chunk)
                if size > settings.max_csv_upload_bytes:
                    raise HoldoutError('holdout_upload_too_large', 413)
                output.write(chunk)
        # Deny sealed bytes before any parser, profiler, preview or feature tool.
        record_exposure(settings.db_path, settings.workspace, path)
        backend = DataBackend(settings.datasets_dir)
        frame = backend.read_frame(path)
        contract = json.loads(row['contract'])
        if not {contract['target_col'], contract['split_col']} <= set(frame):
            raise HoldoutError('blind_development_schema_invalid', 422)
        with connect(settings.db_path) as conn:
            assert_development_frame(conn, task_id, frame)
        if set(frame[contract['split_col']]) != {'train', 'test'}:
            raise HoldoutError('blind_development_requires_train_and_test', 422)
        registry = DatasetRegistry(DatasetRepository(settings.db_path), backend, settings.datasets_dir)
        dataset = registry.register_from_upload(task_id, path, role='sample')
        return {'dataset_id': dataset.id, 'mode': 'governed_blind_oot',
                'split_col': contract['split_col'], 'split_values': {'train': 'train', 'test': 'test'}}
    finally:
        path.unlink(missing_ok=True)


@router.post('/freeze')
async def freeze(task_id: str, request: Request):
    return await run_in_threadpool(service(request).freeze, task_id, await body(request, Freeze), _local_principal_id(request))


@router.post('/evaluate')
async def evaluate(task_id: str, request: Request):
    return await run_in_threadpool(service(request).evaluate, task_id, await body(request, Evaluate), _local_principal_id(request))


@router.get('/result')
def result(task_id: str, request: Request):
    return service(request).result(task_id, _local_principal_id(request))


@router.post('/recoveries', status_code=201)
async def recover(task_id: str, request: Request):
    return await run_in_threadpool(service(request).recover, task_id, await body(request, Recover), _local_principal_id(request))


@router.get('/participants')
def participants(task_id: str, request: Request):
    instance = service(request)
    with connect(instance.db_path) as conn:
        instance._actor(conn, _local_principal_id(request), {'admin'})
        candidates = conn.execute("SELECT p.local_principal_id id,p.display_name,p.role FROM production_principals p JOIN local_principals l ON p.local_principal_id=l.id WHERE p.status='active' AND l.status='active' AND p.role IN ('maker','checker')").fetchall()
        result = []
        for row in candidates:
            try:
                instance._actor(conn, row['id'], {row['role']})
            except HoldoutError:
                continue
            result.append(dict(row))
        return {'participants': result}


@router.get('/candidates')
def candidates(task_id: str, request: Request):
    instance = service(request)
    with connect(instance.db_path) as conn:
        instance._cohort(conn, task_id, _local_principal_id(request), 'develop')
    repo = ModelingRepository(instance.db_path)
    return {'candidates': [{'artifact_id': experiment.artifact_id, 'algorithm': experiment.recipe_id,
            'created_at': experiment.created_at} for experiment in repo.list_experiments(task_id)
            if experiment.artifact_id and experiment.status == 'trained']}
