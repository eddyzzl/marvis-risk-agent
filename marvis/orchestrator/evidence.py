"""Deterministic reference extraction shared by execution and presentation."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from marvis.orchestrator.references import parse_step_output_ref


def payload_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def dataset_refs(payload: Any) -> list[str]:
    refs: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        normalized = key.lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        if text.startswith("dataset:"):
            _append_unique(refs, text)
        elif normalized.endswith("dataset_id") or normalized.endswith(
            "dataset_ids"
        ):
            _append_unique(refs, f"dataset:{text}")

    visit(payload)
    return refs


def artifact_refs(payload: Any) -> list[str]:
    refs: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        normalized = key.lower()
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if not isinstance(value, str) or not value.strip():
            return
        text = value.strip()
        if text.startswith("artifact:"):
            _append_unique(refs, text)
        elif normalized == "path" or normalized.endswith("_path"):
            _append_unique(refs, f"artifact:{text}")
        elif normalized.endswith("artifact_id") or normalized.endswith(
            "artifact_ref"
        ):
            _append_unique(refs, f"artifact:{text}")

    visit(payload)
    return refs


def artifact_bindings(payload: Any) -> list[dict[str, str]]:
    """Extract immutable artifact identities declared by a Tool output.

    A textual ``artifact_ref`` is useful for navigation, but it is not an
    identity unless the content hash travels with it.  Preserve traversal order
    so the stored evidence can be compared byte-for-byte at presentation time.
    """

    bindings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            artifact_id = value.get("artifact_id")
            content_hash = value.get("content_hash")
            if (
                isinstance(artifact_id, str)
                and artifact_id.strip()
                and _is_sha256_hex(content_hash)
            ):
                identity = (artifact_id.strip(), str(content_hash))
                if identity not in seen:
                    seen.add(identity)
                    binding = {
                        "artifact_id": identity[0],
                        "content_hash": identity[1],
                    }
                    kind = value.get("kind")
                    if isinstance(kind, str) and kind.strip():
                        binding["kind"] = kind.strip()
                    bindings.append(binding)
            for child in value.values():
                visit(child)
            return
        if isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return bindings


def result_dataset_ids(payload: Any) -> list[str]:
    """Return only datasets explicitly declared as materialized results."""

    values: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key)
            return
        if key.lower() != "result_dataset_id":
            return
        if isinstance(value, str) and value.strip():
            _append_unique(values, value.strip())

    visit(payload)
    return values


def step_output_references(payload: Any) -> list[dict[str, str]]:
    """Extract every ``$ref`` with its exact input JSON-pointer location."""

    references: list[dict[str, str]] = []

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
            return
        if not isinstance(value, str) or not value.startswith("$ref:"):
            return
        step_id, field = parse_step_output_ref(value)
        references.append(
            {
                "input_path": "/" + "/".join(_json_pointer_token(part) for part in path),
                "step_id": step_id,
                "field": field,
            }
        )

    visit(payload, ())
    return references


def _append_unique(values: list[str], item: str) -> None:
    if item not in values:
        values.append(item)


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "artifact_bindings",
    "artifact_refs",
    "dataset_refs",
    "payload_hash",
    "result_dataset_ids",
    "step_output_references",
]
