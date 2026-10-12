"""Server-owned subject roles and short-lived signed assertion sessions."""

from __future__ import annotations

import hashlib
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
import base64

from marvis.db_schema import connect
from marvis.plugins.publisher_trust import canonical, decode_key


class IdentityError(PermissionError):
    def __init__(self, code, status=403):
        super().__init__(code)
        self.code, self.status = code, status


def require(condition, code, status=403):
    if not condition:
        raise IdentityError(code, status)


def token_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def identity_mode(conn):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='institution_settings'").fetchone():
        return 'trusted_local'
    return str(conn.execute('SELECT mode FROM institution_settings WHERE singleton=1').fetchone()[0])


def _text(value):
    require(isinstance(value, str) and 0 < len(value) <= 200 and value == value.strip()
            and not any(ord(c) < 32 for c in value), 'identity_invalid_text', 422)
    return value


def subject_id(issuer, subject):
    return 'institution-' + hashlib.sha256(canonical([issuer, subject])).hexdigest()


class IdentityStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    def settings(self):
        with connect(self.db_path) as conn:
            return dict(conn.execute('SELECT * FROM institution_settings').fetchone())

    def trust_issuer(self, issuer, key_id, public_key):
        issuer, key_id = _text(issuer), _text(key_id)
        try:
            key_hash = hashlib.sha256(decode_key(public_key)).hexdigest()
        except Exception as exc:
            raise IdentityError('identity_invalid_key', 422) from exc
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            old = conn.execute('SELECT * FROM institution_issuer_keys WHERE issuer=? AND key_id=?',
                               (issuer, key_id)).fetchone()
            if old:
                require(old['public_key'] == public_key and old['revoked_at'] is None,
                        'identity_key_conflict', 409)
            else:
                conn.execute('INSERT INTO institution_issuer_keys VALUES(?,?,\'Ed25519\',?,?,?,NULL)',
                             (issuer, key_id, public_key, key_hash, time.time()))
        return {'issuer': issuer, 'key_id': key_id, 'algorithm': 'Ed25519', 'key_hash': key_hash}

    def provision_subject(self, issuer, subject, display_name, role):
        issuer, subject, display_name = map(_text, (issuer, subject, display_name))
        require(role in {'member','maker','checker','admin','service'}, 'identity_invalid_role', 422)
        principal_id = subject_id(issuer, subject)
        now = datetime.now(UTC).isoformat()
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            require(conn.execute('SELECT 1 FROM institution_issuer_keys WHERE issuer=? AND revoked_at IS NULL',
                                 (issuer,)).fetchone(), 'identity_issuer_untrusted')
            old = conn.execute('SELECT * FROM institution_subjects WHERE id=?', (principal_id,)).fetchone()
            if old:
                require(old['role'] == role and old['display_name'] == display_name and old['revoked_at'] is None,
                        'identity_subject_conflict', 409)
            else:
                # This principal cannot authenticate through the legacy local cookie.
                conn.execute("""INSERT INTO local_principals(id,kind,display_name,session_token_hash,status,
                             created_at,last_seen_at,expires_at,revoked_at)
                             VALUES(?,'local_session',?,?,'active',?,?,'9999-12-31T00:00:00+00:00',NULL)""",
                             (principal_id, display_name, token_hash(secrets.token_urlsafe(48)), now, now))
                conn.execute('INSERT INTO institution_subjects VALUES(?,?,?,?,?,?,?,NULL)',
                             (principal_id, issuer, subject, display_name, role,
                              'service' if role == 'service' else 'human', time.time()))
                if role in {'maker','checker','admin'}:
                    conn.execute("""INSERT INTO production_principals(local_principal_id,display_name,role,status,
                                 created_at,updated_at) VALUES(?,?,?,'active',?,?)""",
                                 (principal_id, display_name, role, now, now))
        return self.subject(principal_id)

    def subject(self, principal_id):
        with connect(self.db_path) as conn:
            return self._subject(conn, principal_id)

    @staticmethod
    def _subject(conn, principal_id):
        row = conn.execute('SELECT * FROM institution_subjects WHERE id=?', (principal_id,)).fetchone()
        require(row and row['revoked_at'] is None, 'identity_subject_inactive', 401)
        return dict(row)

    def set_mode(self, mode, audience, expected_revision):
        require(mode in {'trusted_local','institutional'}, 'identity_invalid_mode', 422)
        audience = _text(audience)
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            old = conn.execute('SELECT * FROM institution_settings').fetchone()
            require(old['revision'] == expected_revision, 'identity_settings_conflict', 409)
            if mode == 'institutional':
                require(not conn.execute("""SELECT 1 FROM plans p LEFT JOIN institution_plan_actors a ON a.plan_id=p.id
                    WHERE a.plan_id IS NULL AND p.status NOT IN ('done','failed','cancelled') LIMIT 1""").fetchone(),
                    'identity_active_plan_owner_binding_required',409)
                require(conn.execute("""SELECT 1 FROM institution_subjects s JOIN institution_issuer_keys k
                         ON k.issuer=s.issuer WHERE s.role='admin' AND s.revoked_at IS NULL
                         AND k.revoked_at IS NULL""").fetchone(), 'identity_admin_required', 409)
                conn.execute("UPDATE capability_settings SET mode='governed',revision=revision+1 WHERE singleton=1")
            conn.execute('UPDATE institution_settings SET mode=?,audience=?,revision=revision+1 WHERE singleton=1',
                         (mode, audience))
        return self.settings()

    def challenge(self, issuer, subject):
        issuer, subject = _text(issuer), _text(subject)
        nonce = secrets.token_urlsafe(32)
        with connect(self.db_path) as conn:
            row = conn.execute('SELECT audience FROM institution_settings').fetchone()
            # Unknown subjects receive the same shape; authentication reveals no records.
            conn.execute('INSERT INTO institution_nonces VALUES(?,?,?,?,?,NULL)',
                         (token_hash(nonce), issuer, subject, row['audience'], time.time()+120))
        return {'issuer': issuer, 'subject': subject, 'audience': row['audience'], 'nonce': nonce,
                'algorithm': 'Ed25519', 'expires_in_seconds': 120}

    def login(self, assertion, signature):
        required = {'schema_version','issuer','subject','audience','nonce','issued_at','expires_at','key_id','algorithm'}
        require(isinstance(assertion, dict) and set(assertion) == required,
                'identity_assertion_contract', 401)
        require(assertion['schema_version'] == 'marvis.identity-assertion.v1'
                and assertion['algorithm'] == 'Ed25519', 'identity_assertion_contract', 401)
        now = time.time()
        issued, expires = assertion['issued_at'], assertion['expires_at']
        require(type(issued) in (int,float) and type(expires) in (int,float)
                and now-300 <= issued <= now+5 and now < expires <= issued+300,
                'identity_assertion_expired', 401)
        for key in ('issuer','subject','audience','nonce','key_id'):
            _text(assertion[key])
        token = secrets.token_urlsafe(32)
        with connect(self.db_path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            settings = conn.execute('SELECT * FROM institution_settings').fetchone()
            require(settings['audience'] == assertion['audience'], 'identity_audience_mismatch', 401)
            key = conn.execute('SELECT * FROM institution_issuer_keys WHERE issuer=? AND key_id=?',
                               (assertion['issuer'], assertion['key_id'])).fetchone()
            require(key and key['revoked_at'] is None, 'identity_issuer_untrusted', 401)
            try:
                Ed25519PublicKey.from_public_bytes(decode_key(key['public_key'])).verify(
                    base64.b64decode(signature, validate=True), canonical(assertion))
            except Exception as exc:
                raise IdentityError('identity_signature_invalid', 401) from exc
            actor = self._subject(conn, subject_id(assertion['issuer'], assertion['subject']))
            nonce = conn.execute('SELECT * FROM institution_nonces WHERE nonce_hash=?',
                                 (token_hash(assertion['nonce']),)).fetchone()
            require(nonce and nonce['consumed_at'] is None and nonce['expires_at'] > now
                    and all(nonce[k] == assertion[k] for k in ('issuer','subject','audience')),
                    'identity_nonce_invalid', 401)
            conn.execute('UPDATE institution_nonces SET consumed_at=? WHERE nonce_hash=?',
                         (now, token_hash(assertion['nonce'])))
            conn.execute('INSERT INTO institution_sessions VALUES(?,?,?,?,?,?,NULL)',
                         (token_hash(token), actor['id'], assertion['issuer'], assertion['key_id'], now, expires))
        return token, actor, int(expires-now)

    def resolve_session(self, token):
        with connect(self.db_path) as conn:
            row = conn.execute('SELECT * FROM institution_sessions WHERE token_hash=?', (token_hash(token),)).fetchone()
            require(row and row['revoked_at'] is None and row['expires_at'] > time.time(),
                    'identity_session_invalid', 401)
            key = conn.execute('SELECT revoked_at FROM institution_issuer_keys WHERE issuer=? AND key_id=?',
                               (row['issuer'], row['key_id'])).fetchone()
            require(key and key['revoked_at'] is None, 'identity_issuer_untrusted', 401)
            return self._subject(conn, row['subject_id'])

    def revoke(self, kind, resource_id, *, key_id=None):
        require(kind in {'subject','issuer','session'}, 'identity_invalid_revocation', 422)
        with connect(self.db_path) as conn:
            if kind == 'subject':
                cursor = conn.execute('UPDATE institution_subjects SET revoked_at=COALESCE(revoked_at,?) WHERE id=?',
                                      (time.time(), resource_id))
                conn.execute("UPDATE local_principals SET status='revoked',revoked_at=? WHERE id=?",
                             (datetime.now(UTC).isoformat(), resource_id))
                conn.execute("UPDATE production_principals SET status='revoked',updated_at=? WHERE local_principal_id=?",
                             (datetime.now(UTC).isoformat(), resource_id))
            elif kind == 'issuer':
                cursor = conn.execute('UPDATE institution_issuer_keys SET revoked_at=COALESCE(revoked_at,?) WHERE issuer=? AND key_id=?',
                                      (time.time(), resource_id, key_id))
            else:
                cursor = conn.execute('UPDATE institution_sessions SET revoked_at=COALESCE(revoked_at,?) WHERE token_hash=?',
                                      (time.time(), token_hash(resource_id)))
            require(cursor.rowcount, 'identity_not_found', 404)
        return {'revoked': True}

    def list_subjects(self):
        with connect(self.db_path) as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM institution_subjects ORDER BY created_at,id')]
