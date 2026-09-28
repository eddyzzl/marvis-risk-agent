"""Real SQLite admission under concurrent expired-request recovery."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
import time

import pytest

from marvis.db_schema import connect, init_db
from marvis.reference_decision.contracts import DecisionError
from marvis.reference_decision.ledger import DecisionLedger
from marvis.reference_decision.schema import initialize

PACKAGE = "a" * 64
SCOPE = "local-reference:production"


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "admission.sqlite"
    init_db(path)
    initialize(path)
    return DecisionLedger(path)


def _claim(ledger, request_id):
    return ledger.claim(SCOPE, request_id, request_id, PACKAGE, 30)


def _expire(ledger, request_id):
    with connect(ledger.db_path) as conn:
        conn.execute(
            "UPDATE reference_decisions SET lease_until=? WHERE environment=? AND request_id=?",
            (time.time() - 1, SCOPE, request_id),
        )


def _active(ledger):
    with connect(ledger.db_path) as conn:
        return conn.execute(
            "SELECT count(*) FROM reference_decisions WHERE environment=? AND status='running' AND lease_until>?",
            (SCOPE, time.time()),
        ).fetchone()[0]


def test_concurrent_recovery_of_eight_expired_requests_only_claims_four_owners(ledger):
    previous = {}
    for index in range(8):
        request_id = f"expired-{index}"
        previous[request_id] = _claim(ledger, request_id)[0]
        _expire(ledger, request_id)
    barrier = Barrier(8)

    def recover(request_id):
        # Separate repository and SQLite connection for each simultaneous caller.
        caller = DecisionLedger(ledger.db_path)
        barrier.wait(timeout=10)
        try:
            owner, result = _claim(caller, request_id)
            assert result is None and owner != previous[request_id]
            return request_id, owner
        except DecisionError as exc:
            assert exc.status == 429
            return request_id, exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        recovered = list(pool.map(recover, previous))
    assert sum(value == "reference_capacity_exceeded" for _, value in recovered) == 4
    assert _active(ledger) == 4
    for request_id, value in recovered:
        if value != "reference_capacity_exceeded":
            with pytest.raises(DecisionError, match="decision_lease_lost"):
                ledger.finish(
                    SCOPE, request_id, previous[request_id], {"timing_ms": {}}
                )


def test_recovery_and_new_requests_share_slots_but_completed_reads_do_not(ledger):
    complete_owner, _ = _claim(ledger, "completed")
    response = ledger.finish(
        SCOPE, "completed", complete_owner, {"decision": "review", "timing_ms": {}}
    )
    stale_owner, _ = _claim(ledger, "expired")
    _expire(ledger, "expired")
    owners = {str(index): _claim(ledger, str(index))[0] for index in range(4)}
    for request_id in ("expired", "new"):
        with pytest.raises(DecisionError, match="reference_capacity_exceeded"):
            _claim(ledger, request_id)
    assert _claim(ledger, "completed") == (None, response)
    assert _active(ledger) == 4
    ledger.finish(SCOPE, "0", owners["0"], {"timing_ms": {}})
    recovered_owner, result = _claim(ledger, "expired")
    assert recovered_owner != stale_owner and result is None
    assert _active(ledger) == 4


def test_per_minute_limit_counts_new_request_ids_not_recovery_attempts(ledger):
    # One existing, expired request from outside the minute window.
    _claim(ledger, "recover-old")
    _expire(ledger, "recover-old")
    with connect(ledger.db_path) as conn:
        conn.execute(
            "UPDATE reference_decisions SET created_at=? WHERE request_id='recover-old'",
            ((datetime.now(UTC) - timedelta(minutes=2)).isoformat(),),
        )
    for index in range(120):
        request_id = f"new-{index}"
        owner, _ = _claim(ledger, request_id)
        ledger.finish(SCOPE, request_id, owner, {"timing_ms": {}})
    with pytest.raises(DecisionError, match="reference_capacity_exceeded"):
        _claim(ledger, "one-more-new")
    owner, result = _claim(ledger, "recover-old")
    assert owner is not None and result is None
    # Retrying a completed request still reads its immutable result at the cap.
    assert _claim(ledger, "new-119")[0] is None
    with connect(ledger.db_path) as conn:
        assert (
            conn.execute("SELECT count(*) FROM reference_decisions").fetchone()[0]
            == 121
        )
