"""Closed text-request ceiling for the explicitly approved formal provider profile.

Local bytes/output limits are enforced before forwarding. Their conversion to a
provider's billed tokens requires separately pinned provider documentation and
approval; returned usage is never used to create additional formal capacity.
"""
from __future__ import annotations

from datetime import date
import json
import math
from typing import Literal

from pydantic import Field, model_validator

from .formal_benchmark import FrozenModel

_HASH = r"^[0-9a-f]{64}$"


class ProviderDocumentRef(FrozenModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_HASH)


class ProviderLimits(FrozenModel):
    protocol: Literal["single_text_utf8_bound_v1"]
    context_window: int = Field(ge=4096, le=2_000_000)
    max_output_tokens: int = Field(ge=1, le=32768)
    framing_token_allowance: int = Field(ge=0, le=65536)
    price_book_sha256: str = Field(pattern=_HASH)
    retain_full_reservations: Literal[True]
    approval: ProviderDocumentRef

    @model_validator(mode="after")
    def nonempty_input_capacity(self):
        if self.framing_token_allowance + self.max_output_tokens >= self.context_window:
            raise ValueError("provider_profile_has_no_input_capacity")
        return self


class ProviderApproval(FrozenModel):
    schema_version: Literal["marvis.provider-budget-approval.v1"] = Field(alias="schema")
    protocol: Literal["single_text_utf8_bound_v1"]
    provider: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_on: str
    source_scope: Literal["local_test_fixture", "provider_documentation"]
    source_document: ProviderDocumentRef
    price_book_sha256: str = Field(pattern=_HASH)
    context_window: int = Field(ge=4096, le=2_000_000)
    max_output_tokens: int = Field(ge=1, le=32768)
    framing_token_allowance: int = Field(ge=0, le=65536)
    input_token_accounting: Literal["at_most_utf8_request_bytes_plus_framing"]
    output_token_accounting: Literal["max_tokens_includes_all_billed_completion"]
    extra_charges: Literal["none_for_allowed_payload"]

    @model_validator(mode="after")
    def explicit_approval(self):
        if not self.approved_by.strip() or date.fromisoformat(self.approved_on).isoformat() != self.approved_on:
            raise ValueError("provider_approval_identity_invalid")
        return self


def validate_formal_payload(payload, *, model_name, limits):
    """Return the exact canonical request bytes and bounded admission metadata."""
    limits = limits if isinstance(limits, ProviderLimits) else ProviderLimits.model_validate(limits)
    allowed = {"model", "messages", "max_tokens", "stream", "stream_options", "temperature", "top_p",
               "seed", "stop", "response_format", "n", "best_of", "thinking"}
    if (type(payload) is not dict or set(payload) - allowed or payload.get("model") != model_name
            or type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= limits.max_output_tokens):
        raise ValueError("formal_payload_outside_provider_contract")
    messages = payload.get("messages")
    if (type(messages) is not list or not 1 <= len(messages) <= 1024
            or any(type(m) is not dict or set(m) != {"role", "content"}
                   or m["role"] not in {"system", "user", "assistant"} or type(m["content"]) is not str for m in messages)):
        raise ValueError("formal_payload_requires_plain_text_messages")
    if any(type(payload.get(k, 1)) is not int or payload.get(k, 1) != 1 for k in ("n", "best_of")):
        raise ValueError("formal_payload_requires_one_completion")
    if "thinking" in payload and payload["thinking"] != {"type": "disabled"}:
        raise ValueError("formal_reasoning_budget_not_bounded")
    if "stream" in payload and type(payload["stream"]) is not bool:
        raise ValueError("formal_stream_flag_invalid")
    if "stream_options" in payload and (type(payload["stream_options"]) is not dict
            or set(payload["stream_options"]) != {"include_usage"}
            or payload["stream_options"]["include_usage"] is not True):
        raise ValueError("formal_stream_options_invalid")
    for key, maximum in (("temperature", 2), ("top_p", 1)):
        if key in payload and (type(payload[key]) not in {float, int} or not math.isfinite(payload[key]) or not 0 <= payload[key] <= maximum):
            raise ValueError("formal_sampling_parameter_invalid")
    if "seed" in payload and type(payload["seed"]) is not int:
        raise ValueError("formal_seed_invalid")
    if "stop" in payload and not (type(payload["stop"]) is str or type(payload["stop"]) is list
                                   and len(payload["stop"]) <= 4 and all(type(v) is str for v in payload["stop"])):
        raise ValueError("formal_stop_invalid")
    if "response_format" in payload:
        value = payload["response_format"]
        if type(value) is not dict or not (value == {"type": "json_object"}
                or set(value) == {"type", "json_schema"} and value["type"] == "json_schema" and type(value["json_schema"]) is dict):
            raise ValueError("formal_response_format_invalid")
        def local_refs(node):
            if isinstance(node, dict):
                for ref in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if ref in node and (type(node[ref]) is not str or not node[ref].startswith("#")):
                        raise ValueError("formal_schema_external_reference")
                for child in node.values():
                    local_refs(child)
            elif isinstance(node, list):
                for child in node:
                    local_refs(child)
        local_refs(value)
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    upper = len(body) + limits.framing_token_allowance
    if upper + payload["max_tokens"] > limits.context_window:
        raise ValueError("formal_input_ceiling_exceeded")
    return body, {"formal_budget_protocol": limits.protocol, "input_utf8_bytes": len(body),
                  "input_token_upper_bound": upper, "framing_token_allowance": limits.framing_token_allowance,
                  "provider_limits_approval_sha256": limits.approval.sha256,
                  "retain_full_reservations": True}
