"""Versioned contracts for driver gates and recoverable failures.

The current UI and driver still carry legacy metadata keys such as ``screen`` and
``dedup``. These envelopes provide a stable contract beside those keys so newer
AUTO decisions, frontend controls, and retry UX can evolve without guessing from
markdown text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

GATE_SCHEMA_VERSION = "gate.v1"
FAILURE_SCHEMA_VERSION = "failure.v1"
EVIDENCE_SCHEMA_VERSION = "evidence.v1"

GATE_ACTIONS = frozenset({"confirm", "adjust", "replan", "clarify", "halt"})
DEFAULT_GATE_ACTIONS = ("confirm", "halt")


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _clean_action(value: Any) -> str | None:
    action = _clean_str(value).lower()
    return action if action in GATE_ACTIONS else None


def _clean_actions(values: Any) -> tuple[str, ...]:
    actions: list[str] = []
    for item in values or []:
        action = _clean_action(item)
        if action and action not in actions:
            actions.append(action)
    return tuple(actions) or DEFAULT_GATE_ACTIONS


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_dicts(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


@dataclass(frozen=True)
class GateControl:
    id: str
    kind: str
    label: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    default: Any = None
    bounds: dict[str, Any] = field(default_factory=dict)
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "schema": dict(self.schema),
            "bounds": dict(self.bounds),
            "required": self.required,
        }
        if self.default is not None:
            data["default"] = self.default
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateControl":
        return cls(
            id=_clean_str(payload.get("id")),
            kind=_clean_str(payload.get("kind") or "input"),
            label=_clean_str(payload.get("label")),
            schema=_dict(payload.get("schema")),
            default=payload.get("default"),
            bounds=_dict(payload.get("bounds")),
            required=bool(payload.get("required", False)),
        )


@dataclass(frozen=True)
class GateRenderBlock:
    kind: str
    title: str = ""
    ref: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {"kind": self.kind, "title": self.title, "payload": dict(self.payload)}
        if self.ref:
            data["ref"] = self.ref
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateRenderBlock":
        return cls(
            kind=_clean_str(payload.get("kind") or "markdown"),
            title=_clean_str(payload.get("title")),
            ref=_clean_str(payload.get("ref")) or None,
            payload=_dict(payload.get("payload")),
        )


@dataclass(frozen=True)
class RetryPolicy:
    retryable: bool = True
    editable_inputs: tuple[str, ...] = ()
    downstream_reset: str = "dependent_steps"

    def to_dict(self) -> dict[str, Any]:
        return {
            "retryable": self.retryable,
            "editable_inputs": list(self.editable_inputs),
            "downstream_reset": self.downstream_reset,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "RetryPolicy":
        data = _dict(payload)
        return cls(
            retryable=bool(data.get("retryable", True)),
            editable_inputs=tuple(_clean_str(item) for item in data.get("editable_inputs") or [] if _clean_str(item)),
            downstream_reset=_clean_str(data.get("downstream_reset") or "dependent_steps"),
        )


@dataclass(frozen=True)
class GateEnvelope:
    kind: str
    target_step_id: str | None = None
    allowed_actions: tuple[str, ...] = DEFAULT_GATE_ACTIONS
    schema_version: str = GATE_SCHEMA_VERSION
    stale_token: str | None = None
    source_output_refs: dict[str, str] = field(default_factory=dict)
    controls: tuple[GateControl, ...] = ()
    render_blocks: tuple[GateRenderBlock, ...] = ()
    risk_flags: tuple[str, ...] = ()
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    downstream_reset_policy: dict[str, Any] = field(default_factory=dict)

    def allows(self, action: str) -> bool:
        return _clean_action(action) in self.allowed_actions

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "target_step_id": self.target_step_id,
            "allowed_actions": list(self.allowed_actions),
            "source_output_refs": dict(self.source_output_refs),
            "controls": [control.to_dict() for control in self.controls],
            "render_blocks": [block.to_dict() for block in self.render_blocks],
            "risk_flags": list(self.risk_flags),
            "retry_policy": self.retry_policy.to_dict(),
            "downstream_reset_policy": dict(self.downstream_reset_policy),
        }
        if self.stale_token:
            data["stale_token"] = self.stale_token
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GateEnvelope":
        controls = tuple(GateControl.from_dict(item) for item in _list_of_dicts(payload.get("controls")))
        render_blocks = tuple(GateRenderBlock.from_dict(item) for item in _list_of_dicts(payload.get("render_blocks")))
        return cls(
            schema_version=_clean_str(payload.get("schema_version") or GATE_SCHEMA_VERSION),
            kind=_clean_str(payload.get("kind") or "gate"),
            target_step_id=_clean_str(payload.get("target_step_id")) or None,
            stale_token=_clean_str(payload.get("stale_token")) or None,
            allowed_actions=_clean_actions(payload.get("allowed_actions") or DEFAULT_GATE_ACTIONS),
            source_output_refs={str(k): str(v) for k, v in _dict(payload.get("source_output_refs")).items()},
            controls=controls,
            render_blocks=render_blocks,
            risk_flags=tuple(_clean_str(item) for item in payload.get("risk_flags") or [] if _clean_str(item)),
            retry_policy=RetryPolicy.from_dict(payload.get("retry_policy")),
            downstream_reset_policy=_dict(payload.get("downstream_reset_policy")),
        )

    @classmethod
    def from_gate_message(cls, gate: Mapping[str, Any]) -> "GateEnvelope":
        meta = _dict(gate.get("metadata"))
        explicit = meta.get("gate_envelope")
        if isinstance(explicit, Mapping):
            return cls.from_dict(explicit)
        return infer_gate_envelope(meta)


@dataclass(frozen=True)
class FailureEnvelope:
    failed_step_id: str | None
    error_kind: str = "execution"
    message: str = ""
    retryable: bool = True
    stale_token: str | None = None
    editable_input_schema: dict[str, Any] = field(default_factory=dict)
    suggested_actions: tuple[str, ...] = ()
    downstream_reset: str = "dependent_steps"
    schema_version: str = FAILURE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "failed_step_id": self.failed_step_id,
            "error_kind": self.error_kind,
            "message": self.message,
            "retryable": self.retryable,
            "editable_input_schema": dict(self.editable_input_schema),
            "suggested_actions": list(self.suggested_actions),
            "downstream_reset": self.downstream_reset,
        }
        if self.stale_token:
            data["stale_token"] = self.stale_token
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FailureEnvelope":
        return cls(
            schema_version=_clean_str(payload.get("schema_version") or FAILURE_SCHEMA_VERSION),
            failed_step_id=_clean_str(payload.get("failed_step_id")) or None,
            error_kind=_clean_str(payload.get("error_kind") or "execution"),
            message=_clean_str(payload.get("message")),
            retryable=bool(payload.get("retryable", True)),
            stale_token=_clean_str(payload.get("stale_token")) or None,
            editable_input_schema=_dict(payload.get("editable_input_schema")),
            suggested_actions=tuple(_clean_str(item) for item in payload.get("suggested_actions") or [] if _clean_str(item)),
            downstream_reset=_clean_str(payload.get("downstream_reset") or "dependent_steps"),
        )


@dataclass(frozen=True)
class EvidenceEnvelope:
    output_ref: str
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    tool_name: str | None = None
    tool_version: str | None = None
    manifest_hash: str | None = None
    input_hash: str | None = None
    input_summary: dict[str, Any] = field(default_factory=dict)
    source_dataset_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    parent_output_refs: tuple[str, ...] = ()
    random_seed: int | None = None
    renderer_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "output_ref": self.output_ref,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "manifest_hash": self.manifest_hash,
            "input_hash": self.input_hash,
            "input_summary": dict(self.input_summary),
            "source_dataset_refs": list(self.source_dataset_refs),
            "artifact_refs": list(self.artifact_refs),
            "parent_output_refs": list(self.parent_output_refs),
            "random_seed": self.random_seed,
            "renderer_hint": self.renderer_hint,
        }


def infer_gate_envelope(meta: Mapping[str, Any]) -> GateEnvelope:
    kind = _clean_str(meta.get("kind") or "gate")
    target_step_id = _clean_str(meta.get("step_id")) or None
    output_refs = {str(k): str(v) for k, v in _dict(meta.get("output_refs")).items()}
    allowed = list(DEFAULT_GATE_ACTIONS)
    controls: list[GateControl] = []
    render_blocks: list[GateRenderBlock] = []

    if kind == "plan_overview":
        allowed = ["confirm", "replan", "clarify", "halt"]
    elif isinstance(meta.get("screen"), Mapping):
        allowed = ["confirm", "adjust", "replan", "clarify", "halt"]
        screen = _dict(meta.get("screen"))
        thresholds = _dict(screen.get("thresholds"))
        for name in ("leakage_ks", "max_missing_rate"):
            controls.append(GateControl(
                id=name,
                kind="number",
                label=name,
                default=thresholds.get(name),
                bounds={"min": 0, "max": 1},
            ))
        controls.append(GateControl(
            id="selection",
            kind="list",
            label="Selected features",
            schema={"items": "string"},
        ))
        render_blocks.append(GateRenderBlock(kind="screen_table", title="Feature screening"))
    elif isinstance(meta.get("dedup"), Mapping):
        allowed = ["confirm", "adjust", "clarify", "halt"]
        controls.append(GateControl(id="dedup_strategies", kind="map", label="Dedup strategies"))
        render_blocks.append(GateRenderBlock(kind="dedup_table", title="Join deduplication"))
    elif kind == "gate":
        allowed = ["confirm", "replan", "clarify", "halt"]

    if isinstance(meta.get("modeling_setup"), Mapping):
        if "adjust" not in allowed:
            allowed = ["confirm", "adjust", "replan", "clarify", "halt"]
        setup = _dict(meta.get("modeling_setup"))
        candidates = [
            _clean_str(item)
            for item in setup.get("sample_weight_candidates") or []
            if _clean_str(item)
        ]
        controls.append(GateControl(
            id="sample_weight_col",
            kind="select",
            label="Sample weight column",
            default=_clean_str(setup.get("sample_weight_col")),
            schema={"enum": ["", *candidates]},
        ))
        render_blocks.append(GateRenderBlock(kind="modeling_setup", title="Modeling setup"))

    return GateEnvelope(
        kind=kind,
        target_step_id=target_step_id,
        stale_token=_stale_token(meta),
        allowed_actions=tuple(allowed),
        source_output_refs=output_refs,
        controls=tuple(controls),
        render_blocks=tuple(render_blocks),
    )


def extract_gate_envelope(gate: Mapping[str, Any]) -> GateEnvelope:
    return GateEnvelope.from_gate_message(gate)


def build_failure_envelope(*, plan_id: str, step_id: str | None, run_seq: int, message: str) -> FailureEnvelope:
    return FailureEnvelope(
        failed_step_id=step_id,
        message=message,
        stale_token=f"{plan_id}:{step_id or 'none'}:{run_seq}",
        suggested_actions=("retry", "adjust", "replan", "halt"),
    )


def _stale_token(meta: Mapping[str, Any]) -> str | None:
    plan_id = _clean_str(meta.get("plan_id"))
    step_id = _clean_str(meta.get("step_id")) or "none"
    run_seq = _clean_str(meta.get("run_seq"))
    if not plan_id:
        return None
    return f"{plan_id}:{step_id}:{run_seq or '0'}"
