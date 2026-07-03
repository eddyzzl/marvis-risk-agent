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
    downstream_reset_steps: tuple[str, ...] = ()
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
            "downstream_reset_steps": list(self.downstream_reset_steps),
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
            downstream_reset_steps=tuple(
                _clean_str(item)
                for item in payload.get("downstream_reset_steps") or []
                if _clean_str(item)
            ),
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


# LT-2: AUTO safety layer. _apply_safety_policy / _gate_risk_reason halt a bare
# AUTO confirm whenever the gate carries a risk flag matching an
# AUTO_HIGH_RISK_FLAG_TOKENS token (irreversible / delivery / handoff / champion
# / deploy / approval / ...). The flags were never populated in production, so
# every forced-confirmation gate slipped through AUTO unblocked. This maps the
# recognizable forced gates to structured risk-flag codes (enum code + Chinese
# gloss recorded where a human reads it) so a declared-risk gate can no longer be
# auto-confirmed. Purely informational gates (plan_overview, plain decision_point
# render gates with an opaque {plan}-step-N id) get no flag and stay AUTO-confirmable.

#: gate source tools whose confirmation is a forced, human-review moment. The
#: value is the risk-flag code emitted; each code contains an
#: AUTO_HIGH_RISK_FLAG_TOKENS token so _gate_risk_reason fires on it.
_HIGH_RISK_GATE_SOURCE_TOOLS: dict[str, tuple[str, ...]] = {
    "post_training_action": ("model_delivery_handoff_champion",),
    "select_experiment": ("champion_model_selection",),
    "compare_experiments": ("champion_model_selection",),
    "confirm_join": ("irreversible_dedup_merge",),
    "propose_join": ("irreversible_dedup_merge",),
    "adopt_strategy": ("irreversible_strategy_approval",),
    "run_strategy_monitoring": ("strategy_monitoring_alarm_approval",),
    "render_monitoring_report": ("strategy_monitoring_alarm_approval",),
    "design_cutoff_bands": ("strategy_direction_approval",),
    "tradeoff_view": ("strategy_direction_approval",),
    "compare_strategies": ("strategy_direction_approval",),
    "vintage_curve": ("strategy_direction_approval",),
    # FIN-3 #1: systematic sweep of every needs_confirmation=True gate step across
    # orchestrator/templates/ found seven forced-confirmation source tools with a
    # red-flag checklist or an irreversible/delivery action that were NOT mapped
    # here, so AUTO silently auto-confirmed them. Each code carries an
    # AUTO_HIGH_RISK_FLAG_TOKENS token (approval / irreversible) so _gate_risk_reason
    # fires. The four modeling-funnel gates (screen_features / select_features /
    # configure_tuning / tune_hyperparameters) are DELIBERATELY not listed: they are
    # AUTO-operable low-consequence reversible gates whose expensive / algorithm-swap
    # controls are already blocked by _apply_safety_policy's control-level guards, and
    # flagging them would over-block the whole modeling auto-drive (INV: pure /
    # reversible gate -> no flag).
    "execute_join": ("irreversible_dedup_merge",),
    "portfolio_gate_summary": ("portfolio_summary_approval",),
    "render_reports": ("validation_report_approval",),
    "monitor_run": ("monitoring_run_alarm_approval",),
    "generate_model_report": ("model_report_approval",),
    "backtest_strategy": ("strategy_direction_approval",),
    "select_rule_set": ("irreversible_strategy_approval",),
}

#: step_id substrings for the same forced gates, used when meta carries only a
#: (test-shaped) semantic step_id with no model_delivery/dedup/source_tool key.
#: Production step_ids are opaque "{plan}-step-N" and never match these, so this
#: only ever fires on gates a caller explicitly named after the forced tool.
_HIGH_RISK_STEP_ID_TOKENS: dict[str, tuple[str, ...]] = {
    "tradeoff": ("strategy_direction_approval",),
    "vintage": ("strategy_direction_approval",),
    "cutoff": ("strategy_direction_approval",),
    "adopt": ("irreversible_strategy_approval",),
    "monitor": ("strategy_monitoring_alarm_approval",),
    "post-training": ("model_delivery_handoff_champion",),
    "select-champion": ("champion_model_selection",),
}


def _infer_risk_flags(meta: Mapping[str, Any]) -> tuple[str, ...]:
    flags: list[str] = []

    def _add(codes: tuple[str, ...]) -> None:
        for code in codes:
            if code not in flags:
                flags.append(code)

    delivery = meta.get("model_delivery")
    if isinstance(delivery, Mapping):
        source_tool = _clean_str(delivery.get("source_tool"))
        _add(_HIGH_RISK_GATE_SOURCE_TOOLS.get(source_tool, ("model_delivery_champion_handoff",)))
    if isinstance(meta.get("dedup"), Mapping):
        _add(("irreversible_dedup_merge",))
    gate_source_tool = _clean_str(meta.get("gate_source_tool"))
    if gate_source_tool in _HIGH_RISK_GATE_SOURCE_TOOLS:
        _add(_HIGH_RISK_GATE_SOURCE_TOOLS[gate_source_tool])
    step_id = _clean_str(meta.get("step_id")).lower()
    for token, codes in _HIGH_RISK_STEP_ID_TOKENS.items():
        if token in step_id:
            _add(codes)
    return tuple(flags)


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
        eligible = [
            _clean_str(item)
            for item in setup.get("eligible_algorithms") or setup.get("recipes") or []
            if _clean_str(item)
        ]
        controls.append(GateControl(
            id="target_type",
            kind="select",
            label="Target type",
            default=_clean_str(setup.get("target_type") or "binary"),
            schema={"enum": ["binary", "continuous", "multiclass"]},
        ))
        controls.append(GateControl(
            id="recipes",
            kind="multi_select",
            label="Algorithms",
            default=list(setup.get("recipes") or []),
            schema={"items": "string", "enum": eligible},
        ))
        controls.append(GateControl(
            id="n_trials",
            kind="number",
            label="Tuning trials",
            default=setup.get("n_trials"),
            bounds={"min": 1, "max": 200},
        ))
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
        risk_flags=_infer_risk_flags(meta),
    )


def extract_gate_envelope(gate: Mapping[str, Any]) -> GateEnvelope:
    return GateEnvelope.from_gate_message(gate)


def build_failure_envelope(
    *,
    plan_id: str,
    step_id: str | None,
    run_seq: int,
    message: str,
    step_inputs: Mapping[str, Any] | None = None,
    downstream_reset_steps: tuple[str, ...] = (),
    error_kind: str = "execution",
    retryable: bool = True,
) -> FailureEnvelope:
    editable_schema = _editable_input_schema(step_inputs or {})
    actions = ("retry", "adjust", "replan", "halt") if retryable else ("replan", "halt")
    return FailureEnvelope(
        failed_step_id=step_id,
        error_kind=error_kind,
        message=message,
        retryable=retryable,
        stale_token=f"{plan_id}:{step_id or 'none'}:{run_seq}",
        editable_input_schema=editable_schema,
        suggested_actions=actions,
        downstream_reset_steps=downstream_reset_steps,
    )


def _stale_token(meta: Mapping[str, Any]) -> str | None:
    plan_id = _clean_str(meta.get("plan_id"))
    step_id = _clean_str(meta.get("step_id")) or "none"
    run_seq = _clean_str(meta.get("run_seq"))
    if not plan_id:
        return None
    return f"{plan_id}:{step_id}:{run_seq or '0'}"


def _editable_input_schema(inputs: Mapping[str, Any]) -> dict[str, Any]:
    properties = {
        str(key): _schema_for_value(value)
        for key, value in inputs.items()
        if str(key)
    }
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }


def _schema_for_value(value: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {"default": value}
    if isinstance(value, bool):
        schema["type"] = "boolean"
    elif isinstance(value, int) and not isinstance(value, bool):
        schema["type"] = "integer"
    elif isinstance(value, float):
        schema["type"] = "number"
    elif isinstance(value, str):
        schema["type"] = "string"
    elif isinstance(value, list):
        schema["type"] = "array"
    elif isinstance(value, Mapping):
        schema["type"] = "object"
    elif value is None:
        schema["type"] = "null"
    else:
        schema["type"] = "string"
        schema["default"] = str(value)
    return schema
