"""Fixed routes, local credentials, verified TLS, no proxy/redirect or body logging."""
import http.client
import json
import os
import ssl
import stat
from urllib.parse import urlsplit

from marvis.reference_decision.contracts import canonical

from .contracts import AdapterError, EndpointConfig, Identity, RemoteReceipt, RemoteReadback, RemoteCapabilities
from pydantic import TypeAdapter


def credential(path):
    path = path.absolute()
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise AdapterError("credential_local_permissions_invalid", 503)
            value = handle.read(4097).decode().strip()
    except (OSError, UnicodeError) as exc:
        raise AdapterError("credential_local_unavailable", 503) from exc
    if not value or len(value) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise AdapterError("credential_local_invalid", 503)
    return value


class Transport:
    def __init__(self, config: EndpointConfig):
        self.config = EndpointConfig.model_validate(config.model_dump())

    def request(self, method, operation_id=None, body=None, *, resource="operations"):
        # No caller can supply an HTTP path, hostname, header or callback.
        if operation_id is not None:
            TypeAdapter(Identity).validate_python(operation_id)
        if resource not in {"operations", "registry", "deployment", "capabilities"} or method not in {"GET", "POST"} or (resource != "operations" and method != "GET"):
            raise AdapterError("adapter_fixed_protocol_required", 422)
        path = "/v1/adapter/" + resource + ("/" + operation_id if operation_id else "")
        url = urlsplit(self.config.base_url)
        token = credential(self.config.credential_file)
        if self.config.mode == "local_reference":
            conn = http.client.HTTPConnection("127.0.0.1", url.port, timeout=self.config.timeout_seconds)
        else:
            context = ssl.create_default_context(cafile=str(self.config.ca_file) if self.config.ca_file else None)
            conn = http.client.HTTPSConnection(url.hostname, url.port or 443, context=context, timeout=self.config.timeout_seconds)
        try:
            data = None if body is None else canonical(body).encode()
            if data is not None and len(data) > 64_000_000:
                raise AdapterError("protocol_upload_byte_limit", 422)
            conn.request(method, path, body=data, headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read(2_000_001)
            if response.status in {401, 403}:
                raise AdapterError("remote_credential_rejected", 503)
            if response.status == 404:
                raise AdapterError("remote_effect_unknown", next_action="远端尚未证明操作结果；继续显式读回或人工核查，不能隐式重发")
            if response.status not in {200, 201} or len(data) > 2_000_000:
                raise AdapterError("remote_protocol_unavailable", 503)
            value = json.loads(data)
            model = RemoteReceipt if resource == "operations" else RemoteCapabilities if resource == "capabilities" else RemoteReadback
            receipt = model.model_validate(value)
            if receipt.instance_id != self.config.instance_id:
                raise AdapterError("remote_instance_binding_mismatch")
            if resource in {"registry", "deployment"} and (receipt.resource != resource or receipt.identity != operation_id):
                raise AdapterError("remote_readback_identity_mismatch")
            return receipt
        except (OSError, http.client.HTTPException) as exc:
            raise AdapterError("remote_effect_unknown") from exc
        except (ValueError, UnicodeError) as exc:
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError("remote_schema_mismatch") from exc
        finally:
            conn.close()
