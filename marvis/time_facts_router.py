"""Data-intake entry point for time-contract facts on ordinary tasks.

Blind OOT tasks record facts only through custody and development intake
(marvis.modeling_holdout), so a fact cannot bypass the role split.
"""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from marvis import time_facts
from marvis.db_schema import connect
from marvis.modeling_holdout.guards import cohort_tx
from marvis.production_governance.router import _local_principal_id

router = APIRouter(prefix='/api/tasks/{task_id}/time-facts', tags=['time-facts'])


class FactRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    fact: Literal['label_available_at', 'historical_parameters', 'candidate_provenance']
    value: str = Field(min_length=1, max_length=40)
    subject_kind: Literal['task', 'dataset']
    subject_id: str | None = Field(default=None, min_length=1, max_length=160)
    basis: str | None = Field(default=None, max_length=2000)


def _error(exc):
    return HTTPException(status_code=exc.status, detail={'code': exc.code})


@router.get('')
def read(task_id: str, request: Request):
    _local_principal_id(request)
    with connect(request.app.state.settings.db_path) as conn:
        if conn.execute('SELECT 1 FROM tasks WHERE id=?', (task_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail={'code': 'task_not_found'})
        if cohort_tx(conn, task_id) is not None:
            # Blind facts are shown per role through the blind OOT summary.
            raise HTTPException(status_code=409, detail={'code': 'blind_task_uses_custody_entry_points'})
        return {'facts': time_facts.list_facts(conn, task_id),
                'conclusion_dependencies': {name: list(spec['facts']) for name, spec in time_facts.CONCLUSIONS.items()}}


@router.post('', status_code=201)
def record(task_id: str, payload: FactRecord, request: Request):
    actor = _local_principal_id(request)
    subject_id = task_id if payload.subject_kind == 'task' else payload.subject_id
    with connect(request.app.state.settings.db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM tasks WHERE id=?', (task_id,)).fetchone() is None:
            raise HTTPException(status_code=404, detail={'code': 'task_not_found'})
        if cohort_tx(conn, task_id) is not None:
            raise HTTPException(status_code=409, detail={'code': 'blind_task_uses_custody_entry_points'})
        if payload.subject_kind == 'dataset' and (not subject_id or conn.execute(
                'SELECT 1 FROM datasets WHERE id=? AND task_id=?', (subject_id, task_id)).fetchone() is None):
            raise HTTPException(status_code=404, detail={'code': 'dataset_not_found'})
        try:
            return time_facts.record_fact_on_connection(
                conn, task_id=task_id, subject_kind=payload.subject_kind, subject_id=subject_id,
                fact=payload.fact, value=payload.value, source='data_intake', actor=actor, basis=payload.basis)
        except time_facts.TimeFactError as exc:
            raise _error(exc) from exc
