import hmac
import json
from pathlib import Path
import secrets
import sqlite3
import time

from fastapi import FastAPI, HTTPException, Request

from marvis.risk_context.source_contracts import (
    ProviderQuery,
    SourceEnvelope,
    canonical_json,
    content_hash,
    instant,
)


def create_reference_provider(state_dir, *, records_path=None, fault_mode="normal"):
    root = Path(state_dir).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_path = root / "access_token"
    if not token_path.exists():
        try:
            with token_path.open("x") as stream:
                token_path.chmod(0o600)
                stream.write(secrets.token_urlsafe(48))
        except FileExistsError:
            pass
    token = token_path.read_text().strip()
    if not token:
        raise RuntimeError("reference provider token unavailable")
    db_path = root / "reference.sqlite"

    def connect():
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    with connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS records(namespace TEXT, subject TEXT, version INTEGER,
          payload TEXT NOT NULL, PRIMARY KEY(namespace,subject,version));
        CREATE TABLE IF NOT EXISTS queries(id TEXT PRIMARY KEY, request_hash TEXT NOT NULL,
          response TEXT NOT NULL, created_at REAL NOT NULL);
        """)
        if records_path:
            data = Path(records_path).read_bytes()
            if len(data) > 10_000_000:
                raise RuntimeError("reference fixture too large")
            records = json.loads(data)
            if not isinstance(records, list) or len(records) > 10_000:
                raise RuntimeError("reference fixture must be a bounded array")
            for raw in records:
                record = SourceEnvelope.model_validate(raw)
                payload = canonical_json(record.model_dump())
                key = (
                    record.subject_namespace,
                    record.subject_token,
                    record.record_version,
                )
                existing = conn.execute(
                    "SELECT payload FROM records WHERE namespace=? AND subject=? AND version=?",
                    key,
                ).fetchone()
                if existing and existing["payload"] != payload:
                    raise RuntimeError("reference record version conflict")
                conn.execute(
                    "INSERT OR IGNORE INTO records VALUES(?,?,?,?)", (*key, payload)
                )
    db_path.chmod(0o600)
    app = FastAPI(title="MARVIS local reference source")

    def authorize(request):
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {token}"
        ):
            raise HTTPException(401, "unauthorized")
        if fault_mode == "unauthorized":
            raise HTTPException(403, "unauthorized")
        if fault_mode == "unavailable":
            raise HTTPException(503, "reference_unavailable")

    @app.get("/health")
    def health():
        return {"scope": "local_reference_only", "protocol": "risk-source.response.v1"}

    @app.post("/v1/queries")
    async def query(request: Request):
        authorize(request)
        body = await request.body()
        if len(body) > 16_000:
            raise HTTPException(413, "request_too_large")
        try:
            contract = ProviderQuery.model_validate_json(body)
        except ValueError as exc:
            raise HTTPException(422, "invalid_query_contract") from exc
        payload = contract.model_dump()
        query_hash = content_hash(payload)
        with connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute(
                "SELECT * FROM queries WHERE id=?", (contract.request_id,)
            ).fetchone()
            if old:
                if old["request_hash"] != query_hash:
                    raise HTTPException(409, "request_id_conflict")
                return json.loads(old["response"])
            now = time.time()
            if instant(contract.expires_at) <= now:
                raise HTTPException(410, "expired")
            records = conn.execute(
                "SELECT payload FROM records WHERE namespace=? AND subject=? ORDER BY version DESC",
                (contract.subject_namespace, contract.subject_token),
            ).fetchall()
            visible = [
                json.loads(r["payload"])
                for r in records
                if instant(json.loads(r["payload"])["available_at"]) <= now
            ]
            result = {
                "request_id": contract.request_id,
                "request_hash": query_hash,
                "status": "found" if visible else "no_record",
                "envelope": visible[0] if visible else None,
            }
            conn.execute(
                "INSERT INTO queries VALUES(?,?,?,?)",
                (contract.request_id, query_hash, canonical_json(result), now),
            )
        if fault_mode == "timeout_after_commit":
            # Conformance-only fault; the durable result already exists for GET.
            import asyncio

            await asyncio.sleep(2)
        if fault_mode == "schema_drift":
            return {**result, "undeclared_provider_field": "v2"}
        return result

    @app.get("/v1/queries/{request_id}")
    def readback(request_id: str, request: Request):
        authorize(request)
        with connect() as conn:
            row = conn.execute(
                "SELECT response FROM queries WHERE id=?", (request_id,)
            ).fetchone()
        if not row:
            raise HTTPException(404, "unknown_request")
        return json.loads(row["response"])

    return app


def serve_reference_provider(args):
    import uvicorn

    app = create_reference_provider(
        args.state_dir, records_path=args.records, fault_mode=args.fault_mode
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)
