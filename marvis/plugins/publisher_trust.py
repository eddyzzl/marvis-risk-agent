"""Exact Ed25519 publisher admission, separate from platform checksums."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from marvis.plugins.errors import ToolNotFoundError


SIGNATURE_FILE = "publisher-signature.json"
SIGNATURE_SCHEMA = "marvis.capability-signature.v1"
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,95}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$"
)


class CapabilityError(PermissionError, ToolNotFoundError):
    def __init__(self, code: str, *, status: int = 409):
        super().__init__(code)
        self.code = code
        self.status = status


def require(condition, code, *, status=409):
    if not condition:
        raise CapabilityError(code, status=status)


def canonical(value) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (ValueError, TypeError) as exc:
        raise CapabilityError("capability_invalid_json", status=422) from exc


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def identifier(value, label="identity") -> str:
    require(
        isinstance(value, str) and bool(_ID.fullmatch(value)),
        f"capability_invalid_{label}",
        status=422,
    )
    return value


def version(value) -> str:
    require(
        isinstance(value, str) and bool(_VERSION.fullmatch(value)),
        "capability_invalid_version",
        status=422,
    )
    return value


def decode_key(value: str) -> bytes:
    try:
        key = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise CapabilityError("publisher_invalid_key", status=422) from exc
    require(len(key) == 32, "publisher_invalid_key", status=422)
    return key


def tree_hash(root: Path) -> str:
    """Length-framed paths and bytes; signature is the sole excluded file."""
    require(root.is_dir() and not root.is_symlink(), "capability_material_missing")
    out = hashlib.sha256()
    files = sorted(root.rglob("*"))
    total = 0
    count = 0
    for path in files:
        require(not path.is_symlink(), "capability_symlink_forbidden", status=422)
        if path.is_dir():
            continue
        require(path.is_file(), "capability_special_file_forbidden", status=422)
        relative = path.relative_to(root).as_posix()
        if relative == SIGNATURE_FILE:
            continue
        count += 1
        size = path.stat().st_size
        total += size
        require(
            count <= 1000 and total <= 50 * 1024 * 1024,
            "capability_material_budget",
            status=422,
        )
        name = relative.encode()
        out.update(len(name).to_bytes(8, "big"))
        out.update(name)
        out.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                out.update(chunk)
    return out.hexdigest()


def signature_body(
    *, kind, name, release_version, publisher_id, content_hash, declaration_hash
) -> dict:
    require(kind in {"plugin", "workflow"}, "capability_invalid_kind", status=422)
    for value in (content_hash, declaration_hash):
        require(
            isinstance(value, str) and bool(_HASH.fullmatch(value)),
            "capability_invalid_hash",
            status=422,
        )
    return {
        "schema_version": SIGNATURE_SCHEMA,
        "kind": kind,
        "name": identifier(name, "name"),
        "version": version(release_version),
        "publisher_id": identifier(publisher_id, "publisher"),
        "content_hash": content_hash,
        "declaration_hash": declaration_hash,
    }


def verify_signature(envelope: dict, public_key: str) -> dict:
    require(isinstance(envelope, dict), "publisher_signature_missing", status=422)
    require(
        set(envelope) == {"body", "signature"}, "publisher_signature_shape", status=422
    )
    body = envelope["body"]
    require(isinstance(body, dict), "publisher_signature_shape", status=422)
    expected = signature_body(
        kind=body.get("kind"),
        name=body.get("name"),
        release_version=body.get("version"),
        publisher_id=body.get("publisher_id"),
        content_hash=body.get("content_hash"),
        declaration_hash=body.get("declaration_hash"),
    )
    require(body == expected, "publisher_signature_shape", status=422)
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise CapabilityError(
            "publisher_signature_dependency_unavailable", status=503
        ) from exc
    try:
        signature = base64.b64decode(envelope["signature"], validate=True)
        require(len(signature) == 64, "publisher_signature_invalid", status=422)
        Ed25519PublicKey.from_public_bytes(decode_key(public_key)).verify(
            signature, SIGNATURE_SCHEMA.encode() + b"\0" + canonical(body)
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise CapabilityError("publisher_signature_not_verified", status=422) from exc
    return body


def read_json(path: Path) -> dict:
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size <= 2 * 1024 * 1024,
        "capability_document_missing_or_large",
        status=422,
    )

    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "capability_duplicate_json_key", status=422)
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=pairs,
            parse_constant=lambda _: require(
                False, "capability_invalid_json", status=422
            ),
        )
    except (ValueError, UnicodeError) as exc:
        raise CapabilityError("capability_invalid_json", status=422) from exc
    require(isinstance(value, dict), "capability_document_shape", status=422)
    return value
