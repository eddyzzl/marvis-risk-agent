"""The immutable declaration checked before an execution attempt starts."""

import hashlib
import json

from marvis.plugins.manifest import manifest_to_dict


def _hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def manifest_receipt_hash(manifest) -> str:
    checksum = str(manifest.checksum or "").strip()
    return (checksum if checksum.startswith("sha256:") else f"sha256:{checksum}") if checksum else _hash(manifest_to_dict(manifest))


def invocation_contract(manifest, tool, ref) -> dict:
    return {
        "schema_version": "tool_invocation.v1",
        "tool_ref": ref.label(),
        "tool_version": manifest.version,
        "manifest_hash": manifest_receipt_hash(manifest),
        # Package checksum alone does not bind declaration changes.
        "manifest_declaration_hash": _hash(manifest_to_dict(manifest)),
        "side_effects": list(tool.side_effects),
    }


def valid_invocation_contract(contract) -> bool:
    if not isinstance(contract, dict) or contract.get("schema_version") != "tool_invocation.v1":
        return False
    for key in ("tool_ref", "tool_version"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            return False
    for key in ("manifest_hash", "manifest_declaration_hash"):
        value = str(contract.get(key) or "")
        if not value.startswith("sha256:") or len(value) != 71 or any(char not in "0123456789abcdef" for char in value[7:]):
            return False
    effects = contract.get("side_effects")
    return isinstance(effects, list) and all(isinstance(effect, str) for effect in effects)


def retry_safe_invocation(contract) -> bool:
    """Only a complete, frozen declaration proves a read-only attempt."""
    if not valid_invocation_contract(contract):
        return False
    effects = contract["side_effects"]
    return isinstance(effects, list) and all(isinstance(effect, str) and effect.startswith("read:") for effect in effects)


def retry_safe_run(run: dict) -> bool:
    contract = run.get("invocation_contract")
    return valid_invocation_contract(contract) and (
        run.get("dispatch_started_at") is None or retry_safe_invocation(contract)
    )


def load_invocation_contract(raw):
    try:
        value = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None
