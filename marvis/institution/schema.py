"""Migration 44: persistent identities, authority, and signed resource versions."""


def create_schema(conn):
    statements = [
        """CREATE TABLE IF NOT EXISTS institution_settings(
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), mode TEXT NOT NULL
            CHECK(mode IN ('trusted_local','institutional')), audience TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0)""",
        "INSERT OR IGNORE INTO institution_settings VALUES(1,'trusted_local','marvis-local',0)",
        """CREATE TABLE IF NOT EXISTS institution_issuer_keys(
            issuer TEXT NOT NULL, key_id TEXT NOT NULL, algorithm TEXT NOT NULL CHECK(algorithm='Ed25519'),
            public_key TEXT NOT NULL, key_hash TEXT NOT NULL, created_at REAL NOT NULL,
            revoked_at REAL, PRIMARY KEY(issuer,key_id))""",
        """CREATE TABLE IF NOT EXISTS institution_subjects(
            id TEXT PRIMARY KEY, issuer TEXT NOT NULL, subject TEXT NOT NULL, display_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('member','maker','checker','admin','service')),
            actor_kind TEXT NOT NULL CHECK(actor_kind IN ('human','service')), created_at REAL NOT NULL,
            revoked_at REAL, UNIQUE(issuer,subject), FOREIGN KEY(id) REFERENCES local_principals(id))""",
        """CREATE TABLE IF NOT EXISTS institution_nonces(
            nonce_hash TEXT PRIMARY KEY, issuer TEXT NOT NULL, subject TEXT NOT NULL,
            audience TEXT NOT NULL, expires_at REAL NOT NULL, consumed_at REAL)""",
        """CREATE TABLE IF NOT EXISTS institution_sessions(
            token_hash TEXT PRIMARY KEY, subject_id TEXT NOT NULL, issuer TEXT NOT NULL, key_id TEXT NOT NULL,
            created_at REAL NOT NULL, expires_at REAL NOT NULL, revoked_at REAL,
            FOREIGN KEY(subject_id) REFERENCES institution_subjects(id))""",
        """CREATE TABLE IF NOT EXISTS institution_resource_owners(
            kind TEXT NOT NULL CHECK(kind IN ('task','dataset','artifact','plan','alias')),
            resource_id TEXT NOT NULL, owner_id TEXT NOT NULL, revision INTEGER NOT NULL,
            created_at REAL NOT NULL, PRIMARY KEY(kind,resource_id),
            FOREIGN KEY(owner_id) REFERENCES institution_subjects(id))""",
        """CREATE TABLE IF NOT EXISTS institution_resource_grants(
            kind TEXT NOT NULL, resource_id TEXT NOT NULL, subject_id TEXT NOT NULL,
            actions_json TEXT NOT NULL, revision INTEGER NOT NULL, granted_by TEXT NOT NULL,
            expires_at REAL, revoked_at REAL, PRIMARY KEY(kind,resource_id,subject_id),
            FOREIGN KEY(subject_id) REFERENCES institution_subjects(id))""",
        """CREATE TABLE IF NOT EXISTS institution_plan_actors(
            plan_id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, task_id TEXT NOT NULL,
            FOREIGN KEY(plan_id) REFERENCES plans(id), FOREIGN KEY(subject_id) REFERENCES institution_subjects(id))""",
        """CREATE TABLE IF NOT EXISTS instance_trusted_nodes(
            id TEXT PRIMARY KEY, public_key TEXT NOT NULL, key_hash TEXT NOT NULL,
            created_at REAL NOT NULL, revoked_at REAL)""",
        """CREATE TABLE IF NOT EXISTS instance_resource_versions(
            id TEXT PRIMARY KEY, origin_node TEXT NOT NULL, kind TEXT NOT NULL,
            resource_name TEXT NOT NULL, version TEXT NOT NULL, content_hash TEXT NOT NULL,
            envelope_json TEXT NOT NULL, material_json TEXT NOT NULL, native_ref TEXT NOT NULL,
            imported_by TEXT NOT NULL, created_at REAL NOT NULL,
            UNIQUE(origin_node,kind,resource_name,version))""",
        """CREATE TABLE IF NOT EXISTS instance_resource_heads(
            origin_node TEXT NOT NULL, kind TEXT NOT NULL, resource_name TEXT NOT NULL,
            version_id TEXT NOT NULL, revision INTEGER NOT NULL, enabled INTEGER NOT NULL,
            PRIMARY KEY(origin_node,kind,resource_name), FOREIGN KEY(version_id) REFERENCES instance_resource_versions(id))""",
        """CREATE TABLE IF NOT EXISTS instance_sync_receipts(
            envelope_hash TEXT PRIMARY KEY, receipt_json TEXT NOT NULL, created_at REAL NOT NULL)""",
    ]
    for statement in statements:
        conn.execute(statement)
    for table in ('institution_plan_actors', 'instance_resource_versions', 'instance_sync_receipts'):
        for action in ('UPDATE', 'DELETE'):
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_{action.lower()}_immutable "
                         f"BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'identity custody immutable'); END")
    for table, keys in (('institution_issuer_keys', ('issuer','key_id','algorithm','public_key','key_hash','created_at')),
                        ('institution_subjects', ('id','issuer','subject','display_name','role','actor_kind','created_at')),
                        ('instance_trusted_nodes', ('id','public_key','key_hash','created_at'))):
        condition = ' OR '.join(f'NEW.{key} IS NOT OLD.{key}' for key in keys)
        conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_binding_immutable BEFORE UPDATE ON {table} "
                     f"WHEN {condition} OR (OLD.revoked_at IS NOT NULL AND NEW.revoked_at IS NOT OLD.revoked_at) "
                     "BEGIN SELECT RAISE(ABORT,'identity binding immutable'); END")
        conn.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} "
                     "BEGIN SELECT RAISE(ABORT,'identity binding immutable'); END")
