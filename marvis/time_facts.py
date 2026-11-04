"""Time-contract facts and the conclusions that depend on them.

Three facts have explicit collection points (label construction, data intake,
blind OOT custody/development intake). A fact without a record is ``unknown``;
nothing here infers, defaults or upgrades a value. Facts never change metrics:
they only label or block conclusions drawn from platform-computed numbers.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import time
import uuid

LABEL_AVAILABLE_AT = 'label_available_at'
HISTORICAL_PARAMETERS = 'historical_parameters'
CANDIDATE_PROVENANCE = 'candidate_provenance'
UNKNOWN = 'unknown'

# fact -> subjects it may describe
FACT_SUBJECTS = {
    LABEL_AVAILABLE_AT: ('dataset', 'blind_cohort'),
    HISTORICAL_PARAMETERS: ('task',),
    CANDIDATE_PROVENANCE: ('task',),
}
FACT_VALUES = {
    HISTORICAL_PARAMETERS: ('existed', 'absent'),
    CANDIDATE_PROVENANCE: ('pre_registered', 'development_data_only', 'holdout_informed'),
}
SOURCES = ('label_construction', 'data_intake', 'holdout_custody', 'holdout_development')
_PLATFORM_DERIVED = frozenset({'label_construction'})

# Which facts each conclusion depends on. A conclusion is "allowed" only when
# every fact is recorded before the reference time and has an admissible value.
CONCLUSIONS = {
    'blind_oot_acceptance': {
        'facts': (LABEL_AVAILABLE_AT, CANDIDATE_PROVENANCE),
        'blocked_label': '不能作为验收结论',
    },
    'historical_period_comparison': {
        'facts': (LABEL_AVAILABLE_AT, HISTORICAL_PARAMETERS),
        'blocked_label': '不能作为历史同期结论',
    },
}


class TimeFactError(ValueError):
    def __init__(self, code, status=422):
        super().__init__(code)
        self.code, self.status = code, status


def normalize_value(fact, value):
    if fact not in FACT_SUBJECTS:
        raise TimeFactError('time_fact_unknown_kind')
    if not isinstance(value, str) or not value.strip():
        raise TimeFactError('time_fact_value_invalid')
    value = value.strip()
    if fact == LABEL_AVAILABLE_AT:
        _available_epoch(value)
        return value
    if value not in FACT_VALUES[fact]:
        raise TimeFactError('time_fact_value_invalid')
    return value


def _available_epoch(value):
    """Epoch from which labels are observable. A bare date means end of that day (UTC)."""
    try:
        if len(value) == 10:
            day = date.fromisoformat(value)
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() + timedelta(days=1).total_seconds()
        moment = datetime.fromisoformat(value)
    except ValueError as exc:
        raise TimeFactError('time_fact_value_invalid') from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def has_schema(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='time_facts'").fetchone() is not None


def record_fact_on_connection(conn, *, task_id, subject_kind, subject_id, fact, value, source, actor, basis=None):
    """Append one immutable fact. Re-recording the same value is idempotent."""
    value = normalize_value(fact, value)
    if subject_kind not in FACT_SUBJECTS[fact]:
        raise TimeFactError('time_fact_subject_invalid')
    if source not in SOURCES or not str(actor or '').strip() or not str(subject_id or '').strip():
        raise TimeFactError('time_fact_source_invalid')
    existing = conn.execute('SELECT * FROM time_facts WHERE subject_kind=? AND subject_id=? AND fact=?',
                            (subject_kind, subject_id, fact)).fetchone()
    if existing is not None:
        if existing['value'] != value:
            raise TimeFactError('time_fact_immutable', 409)
        return dict(existing)
    row = {'id': 'fact_' + uuid.uuid4().hex, 'task_id': task_id, 'subject_kind': subject_kind,
           'subject_id': subject_id, 'fact': fact, 'value': value, 'source': source,
           'actor': str(actor), 'basis': basis, 'recorded_at': time.time()}
    conn.execute('INSERT INTO time_facts(id,task_id,subject_kind,subject_id,fact,value,source,actor,basis,recorded_at) '
                 'VALUES(:id,:task_id,:subject_kind,:subject_id,:fact,:value,:source,:actor,:basis,:recorded_at)', row)
    return row


def list_facts(conn, task_id):
    if not has_schema(conn):
        return []
    return [dict(row) for row in conn.execute(
        'SELECT * FROM time_facts WHERE task_id=? ORDER BY recorded_at,id', (task_id,))]


def facts_for(conn, subjects, *, cutoff=None, trust=None):
    """Fact states for ordered (kind, id) subjects; the first matching record wins.

    A record made after ``cutoff`` (e.g. after sealed rows were read) is kept
    visible but does not establish the fact. ``trust`` maps a subject kind to
    the only (source, actor) allowed to establish facts about it; other
    records are ignored, as if absent.
    """
    rows = {}
    if has_schema(conn):
        for kind, subject_id in subjects:
            for row in conn.execute('SELECT * FROM time_facts WHERE subject_kind=? AND subject_id=?', (kind, subject_id)):
                if trust is not None and (row['source'], row['actor']) != trust.get(kind):
                    continue
                rows.setdefault(row['fact'], row)
    result = {}
    for fact in FACT_SUBJECTS:
        row = rows.get(fact)
        if row is None:
            result[fact] = {'state': UNKNOWN, 'value': UNKNOWN}
            continue
        late = cutoff is not None and row['recorded_at'] > cutoff
        result[fact] = {'state': 'recorded_after_read' if late else 'recorded', 'value': row['value'],
                        'assurance': 'platform_derived' if row['source'] in _PLATFORM_DERIVED else 'declared',
                        'source': row['source'], 'recorded_by': row['actor'], 'recorded_at': row['recorded_at']}
    return result


def label_maturity(facts, reference_time):
    fact = (facts or {}).get(LABEL_AVAILABLE_AT) or {}
    if fact.get('state') != 'recorded':
        return UNKNOWN
    return 'mature' if _available_epoch(fact['value']) <= reference_time else 'immature'


def _reasons(fact, state):
    if state.get('state') == 'recorded_after_read':
        return [f'{fact}_recorded_after_read']
    if state.get('state') != 'recorded':
        return [f'{fact}_unknown']
    value = state['value']
    if fact == HISTORICAL_PARAMETERS and value != 'existed':
        return ['historical_parameters_absent']
    if fact == CANDIDATE_PROVENANCE and value == 'holdout_informed':
        return ['candidate_informed_by_holdout']
    return []


def gate_conclusions(facts, reference_time):
    """Map facts to conclusion states. Never reads or alters metric values."""
    facts = facts or {}
    maturity = label_maturity(facts, reference_time)
    gated = {}
    for name, spec in CONCLUSIONS.items():
        reasons = []
        for fact in spec['facts']:
            state = facts.get(fact) or {'state': UNKNOWN}
            fact_reasons = _reasons(fact, state)
            if fact == LABEL_AVAILABLE_AT and not fact_reasons and maturity == 'immature':
                fact_reasons.append('labels_immature')
            reasons += fact_reasons
        gated[name] = {'status': 'blocked' if reasons else 'allowed', 'depends_on': list(spec['facts']),
                       'reasons': reasons, 'label': spec['blocked_label'] if reasons else None}
    return {'label_maturity': maturity, 'conclusions': gated}
