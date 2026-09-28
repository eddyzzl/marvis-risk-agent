from marvis.db_schema import connect


def initialize(db_path):
    # Owned, additive tables; the shared application's migration version is untouched.
    with connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS reference_decision_packages (
            id TEXT PRIMARY KEY, canonical_json TEXT NOT NULL,
            signature TEXT NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reference_decisions (
            environment TEXT NOT NULL, request_id TEXT NOT NULL, input_hash TEXT NOT NULL,
            package_hash TEXT NOT NULL, owner TEXT NOT NULL, lease_until REAL NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','completed')),
            response_json TEXT, created_at TEXT NOT NULL,
            PRIMARY KEY(environment, request_id)
        );
        CREATE TABLE IF NOT EXISTS reference_installations (
            promotion_id TEXT PRIMARY KEY, package_hash TEXT NOT NULL,
            receipt_json TEXT NOT NULL, installed_at TEXT NOT NULL
        );
        CREATE TRIGGER IF NOT EXISTS reference_package_immutable_update
            BEFORE UPDATE ON reference_decision_packages BEGIN
            SELECT RAISE(ABORT, 'reference packages are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reference_package_immutable_delete
            BEFORE DELETE ON reference_decision_packages BEGIN
            SELECT RAISE(ABORT, 'reference packages are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reference_decision_completed_immutable
            BEFORE UPDATE ON reference_decisions WHEN OLD.status = 'completed' BEGIN
            SELECT RAISE(ABORT, 'completed decisions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reference_decision_no_delete
            BEFORE DELETE ON reference_decisions BEGIN
            SELECT RAISE(ABORT, 'decisions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reference_installation_immutable_update
            BEFORE UPDATE ON reference_installations BEGIN
            SELECT RAISE(ABORT, 'installations are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS reference_installation_immutable_delete
            BEFORE DELETE ON reference_installations BEGIN
            SELECT RAISE(ABORT, 'installations are immutable'); END;
        """)
