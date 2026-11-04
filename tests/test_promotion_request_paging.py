"""V3 5.7: the promotion request list pages in the database."""

from fastapi.testclient import TestClient

from tests.test_production_governance_api import (
    _claim_role,
    _create_governed_app,
    _promotion_payload,
    _seed_strategy,
)


def test_promotion_requests_page_in_order_with_count_and_next_offset(tmp_path, monkeypatch):
    app = _create_governed_app(tmp_path)
    maker = TestClient(app)
    _claim_role(app, maker, display_name="Maker", role="maker")
    created = []
    for index in range(5):
        strategy_id = f"strategy-page-{index}"
        _seed_strategy(app, strategy_id)
        environment = "staging" if index == 4 else "production"
        response = maker.post(
            "/api/production-governance/promotion-requests",
            json=_promotion_payload(strategy_id, environment=environment),
        )
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])

    # The page comes from LIMIT/OFFSET: the full table is never loaded.
    import marvis.production_governance.repository as repository_module

    statements = []
    original_connect = repository_module.connect

    from contextlib import contextmanager

    @contextmanager
    def tracing_connect(*args, **kwargs):
        with original_connect(*args, **kwargs) as conn:
            conn.set_trace_callback(statements.append)
            yield conn

    monkeypatch.setattr(repository_module, "connect", tracing_connect)

    base = "/api/production-governance/promotion-requests?environment=production"
    first = maker.get(base + "&limit=2&offset=0").json()
    second = maker.get(base + "&limit=2&offset=2").json()
    everything = maker.get("/api/production-governance/promotion-requests?limit=100").json()

    assert first["count"] == second["count"] == 4
    assert len(first["requests"]) == 2 and first["next_offset"] == 2
    assert len(second["requests"]) == 2 and second["next_offset"] is None
    paged = [row["id"] for row in first["requests"] + second["requests"]]
    newest_first = [row["id"] for row in everything["requests"] if row["environment"] == "production"]
    assert paged == newest_first and set(paged) == set(created[:4])
    assert everything["count"] == 5 and everything["next_offset"] is None
    listing = [sql for sql in statements if "FROM production_promotion_requests p" in sql and "COUNT" not in sql]
    assert listing and all("LIMIT" in sql for sql in listing)


def test_one_malformed_grant_row_does_not_break_the_readable_strategy_filter(monkeypatch):
    import sqlite3

    import marvis.institution.policy as policy
    import marvis.institution.store as store
    import marvis.production_governance.repository as repository_module

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE institution_resource_grants(kind TEXT, resource_id TEXT, subject_id TEXT, actions_json TEXT,
            expires_at REAL, revoked_at REAL);
        CREATE TABLE institution_resource_owners(kind TEXT, resource_id TEXT, owner_id TEXT);
        CREATE TABLE strategies(id TEXT, task_id TEXT);
        CREATE TABLE production_promotion_requests(id TEXT, strategy_id TEXT);
        INSERT INTO institution_resource_grants VALUES ('strategy','s-bad','actor','not json',NULL,NULL),
            ('strategy','s-ok','actor','["read"]',NULL,NULL);
        INSERT INTO production_promotion_requests VALUES ('p-bad','s-bad'),('p-ok','s-ok'),('p-none','s-none');
    """)
    monkeypatch.setattr(repository_module, "identity_mode", lambda _conn: "institutional")
    monkeypatch.setattr(policy, "current_actor", lambda _path: "actor")
    monkeypatch.setattr(store.IdentityStore, "_subject", staticmethod(lambda _conn, _actor: None))
    sql, args = repository_module._readable_strategy_sql(conn)
    rows = conn.execute("SELECT p.id FROM production_promotion_requests p WHERE " + sql, args).fetchall()
    assert [row["id"] for row in rows] == ["p-ok"]
