import http.client
import json
from urllib.parse import urlsplit

from marvis.risk_context.source_contracts import SourceError, canonical_json


def call(profile, token, query, *, readback):
    """No redirects, proxy environment, DNS, arbitrary headers or response logging."""
    endpoint = urlsplit(profile.endpoint)
    conn = http.client.HTTPConnection(
        "127.0.0.1", endpoint.port, timeout=profile.timeout_seconds
    )
    path = "/v1/queries" + (f"/{query['request_id']}" if readback else "")
    try:
        conn.request(
            "GET" if readback else "POST",
            path,
            body=None if readback else canonical_json(query).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        response = conn.getresponse()
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise SourceError("schema_drift")
        if response.status == 404 and readback:
            # Absence is not proof that a previously sent query can never commit.
            raise SourceError("unknown_effect")
        if response.status in (401, 403):
            raise SourceError("unauthorized")
        if response.status == 410:
            raise SourceError("expired")
        if response.status != 200:
            raise SourceError(
                "unavailable" if response.status >= 500 else "protocol_error"
            )
        try:
            return json.loads(data), data
        except (ValueError, UnicodeError) as exc:
            raise SourceError("schema_drift") from exc
    except (OSError, http.client.HTTPException) as exc:
        # Even connection errors are conservative: only provider readback can resolve.
        raise SourceError("unknown_effect") from exc
    finally:
        conn.close()
