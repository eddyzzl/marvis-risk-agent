"""Real native rule package → governed activation → decisions → feedback ticks."""

from datetime import datetime, timedelta

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.repositories.datasets import DatasetRepository
from marvis.operations.monitoring import read_monitoring_evidence
from marvis.operations.decision_feedback import feedback_window, import_feedback
from marvis.operations.extension_bindings import DecisionFeedbackBinding, FeedbackImport

from test_reference_rule_packages import rule_runtime
from test_reference_deployment import _promotion, _install_activate
from test_operations_api import _claim_role


@pytest.fixture(scope="module")
def feedback_carrier(tmp_path_factory):
    app, maker, checker, admin, task, package_request, root = rule_runtime.__wrapped__(
        tmp_path_factory
    )
    built = maker.post(
        "/api/reference-decision/packages", json=package_request.model_dump()
    )
    assert built.status_code == 201, built.text
    package_hash = built.json()["package_hash"]
    promotion = _promotion((app, maker, checker, admin), package_hash)
    _install_activate((app, maker, checker, admin), promotion)
    rows = []
    for i, x in enumerate((0.1, 0.9)):
        r = admin.post(
            "/api/reference-decision/decisions",
            json=dict(
                request_id=f"feedback-decision-{i}",
                decision_node="underwriting",
                expected_package_hash=package_hash,
                features={"x1": x, "x2": 2},
            ),
        )
        assert r.status_code == 200, r.text
        value = r.json()
        rows.append(
            dict(
                request_id=value["request_id"],
                output_hash=value["output_hash"],
                package_hash=package_hash,
                label=i,
                event_at=value["created_at"],
                available_at=value["created_at"],
            )
        )
    frame = pd.DataFrame(rows)
    path = root / "declared-outcomes.parquet"
    frame.to_parquet(path, index=False)
    registry = DatasetRegistry(
        DatasetRepository(app.state.settings.db_path),
        DataBackend(app.state.settings.datasets_dir),
        app.state.settings.datasets_dir,
    )
    dataset = registry.register_existing(
        path, task_id=task.id, role="decision_outcomes"
    )
    start = min(datetime.fromisoformat(v["event_at"]) for v in rows).replace(
        second=0, microsecond=0
    )
    end = max(datetime.fromisoformat(v["event_at"]) for v in rows).replace(
        second=0, microsecond=0
    ) + timedelta(minutes=1)
    now = end
    return app, admin, task, registry, dataset, package_hash, start, end, now, rows


def feedback_payload(task, dataset):
    return dict(
        task_id=task.id,
        environment="local-reference:production",
        dataset_id=dataset.id,
        dataset_content_hash=dataset.content_hash,
        timezone="UTC",
        request_id_col="request_id",
        output_hash_col="output_hash",
        package_hash_col="package_hash",
        label_col="label",
        event_at_col="event_at",
        available_at_col="available_at",
    )


def test_actual_decision_feedback_import_monitor_tick_red_inbox_and_retry_no_requery(
    feedback_carrier, monkeypatch
):
    app, client, task, registry, dataset, package_hash, start, end, now, _ = (
        feedback_carrier
    )
    r = client.post(
        "/api/operations/decision-feedback", json=feedback_payload(task, dataset)
    )
    assert r.status_code == 201, r.text
    assert (
        r.json()["inserted"] == 2
        and r.json()["outcome_assurance"] == "publisher_declared"
    )
    r = client.post(
        "/api/operations/decision-feedback", json=feedback_payload(task, dataset)
    )
    assert r.status_code == 201 and r.json()["replayed"] == 2
    timed = create_app(app.state.settings, operations_clock=lambda: now)
    client = TestClient(timed)
    _claim_role(timed, client, "admin")
    binding = dict(
        task_id=task.id,
        environment="local-reference:production",
        package_hash=package_hash,
        complete_through=end.isoformat(),
        label_maturity_seconds=0,
        min_labeled_count=2,
        max_bad_rate=0.25,
        max_fallback_rate=0.1,
    )
    body = dict(
        schedule_id="decision-feedback",
        revision=1,
        monitoring_ref="analysis.decision_ledger_feedback",
        active_from=start.isoformat(),
        calendar=dict(
            anchor_at=start.isoformat(),
            interval_seconds=int((end - start).total_seconds()),
        ),
        catch_up_budget=1,
        lease_seconds=300,
        retry_policy=dict(
            max_attempts=1,
            initial_backoff_seconds=0.0,
            multiplier=1.0,
            max_backoff_seconds=0.0,
        ),
        feedback_binding=binding,
    )
    p = client.post(
        "/api/operations/proposals", json=dict(expected_revision=0, schedule=body)
    )
    assert p.status_code == 201, p.text
    p = p.json()
    c = client.post(
        f"/api/operations/proposals/{p['proposal_id']}/confirm",
        json=dict(expected_contract_hash=p["contract_hash"]),
    )
    assert c.status_code == 200, c.text
    runtime = timed.state.operations_runtime
    runtime.tick(catch_up_budget=1, notification_budget=100)
    contract = runtime.store.get_schedule("decision-feedback").contract
    key = contract.calendar.period(0).key
    record = runtime.store.get_outcome("decision-feedback", key)
    assert record is not None, [
        x for x in runtime.store.list_runs("decision-feedback", key)
    ]
    evidence = read_monitoring_evidence(timed.state.settings.workspace, record.outcome)
    value = evidence["tool_output"]
    assert record.outcome.level == "red"
    assert value["population_count"] == value["labeled_count"] == 2
    assert value["bad_rate"] == 0.5 and value["approval_bad_rate"] == 0
    assert (
        value["fallback_rate"] == 0
        and value["rule_replay"] == "unavailable_original_features_not_retained"
    )
    assert value["score_psi"] is None and value["economic_outcome"] is None
    assert value["source_refs"] == [
        {"dataset_id": dataset.id, "content_hash": dataset.content_hash}
    ]
    assert len(runtime.local_inbox.list()) == 1
    monkeypatch.setattr(
        runtime._executors["analysis.decision_ledger_feedback"].runner,
        "invoke",
        lambda *a, **kw: pytest.fail("committed tick replayed native Tool"),
    )
    runtime.tick(catch_up_budget=1, notification_budget=100)
    assert (
        len(runtime.store.list_runs("decision-feedback", key)) == 1
        and len(runtime.local_inbox.list()) == 1
    )


def test_actual_feedback_unknown_maturity_watermark_and_source_cas(feedback_carrier):
    app, _, task, registry, dataset, package_hash, start, end, now, _ = feedback_carrier
    binding = DecisionFeedbackBinding(
        task_id=task.id,
        environment="local-reference:production",
        package_hash=package_hash,
        complete_through=end,
        label_maturity_seconds=86400,
        min_labeled_count=2,
        max_bad_rate=0.25,
        max_fallback_rate=0.1,
    )
    v = feedback_window(app.state.settings, binding, start, end, now)
    assert (
        v["overall_level"] == "not_available"
        and v["labeled_count"] == 0
        and v["bad_rate"] is None
    )
    assert (
        feedback_window(
            app.state.settings,
            binding.model_copy(update={"complete_through": start}),
            start,
            end,
            now,
        )["reason_code"]
        == "window_incomplete"
    )
    assert (
        feedback_window(
            app.state.settings,
            binding.model_copy(update={"complete_through": now + timedelta(seconds=1)}),
            start,
            end,
            now,
        )["reason_code"]
        == "future_source_watermark"
    )
    path = registry.authenticate_dataset_binding(
        dataset.id, expected_task_id=task.id, expected_content_hash=dataset.content_hash
    ).path
    path.chmod(0o600)
    original = path.read_bytes()
    path.write_bytes(original + b"drift")
    try:
        with pytest.raises(Exception):
            feedback_window(app.state.settings, binding, start, end, now)
    finally:
        path.write_bytes(original)
        path.chmod(0o400)


def test_invalid_unbound_feedback_rows_roll_back_all_rows(feedback_carrier):
    app, _, task, registry, _, package_hash, _, _, _, rows = feedback_carrier
    # Existing immutable decisions are referenced honestly, while one identity is
    # unknown. A separate source identity prevents the earlier valid import from
    # becoming a misleading replay success for the first row.
    frame = pd.DataFrame(
        [dict(rows[0]), {**rows[1], "request_id": "never-completed-native"}]
    )
    path = app.state.settings.workspace / "invalid-feedback.parquet"
    frame.to_parquet(path, index=False)
    dataset = registry.register_existing(
        path, task_id=task.id, role="decision_outcomes"
    )
    with connect(app.state.settings.db_path) as conn:
        before = conn.execute(
            "SELECT count(*) FROM operations_decision_feedback"
        ).fetchone()[0]
    with pytest.raises(ValueError):
        import_feedback(
            app.state.settings,
            FeedbackImport(**feedback_payload(task, dataset)),
            actor_id="test",
        )
    with connect(app.state.settings.db_path) as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM operations_decision_feedback"
            ).fetchone()[0]
            == before
        )
