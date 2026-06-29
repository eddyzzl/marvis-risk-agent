from marvis.agent.gates import FailureEnvelope, GateEnvelope, extract_gate_envelope


def test_explicit_gate_envelope_round_trips_allowed_actions_and_controls():
    gate = {
        "metadata": {
            "gate_envelope": {
                "kind": "screen",
                "target_step_id": "gate-1",
                "stale_token": "plan-1:gate-1:0",
                "allowed_actions": ["confirm", "adjust", "halt"],
                "controls": [
                    {
                        "id": "leakage_ks",
                        "kind": "number",
                        "label": "Leakage KS",
                        "bounds": {"min": 0, "max": 1},
                        "default": 0.4,
                    }
                ],
                "source_output_refs": {"screen": "metrics:screen:v1"},
            }
        }
    }

    envelope = extract_gate_envelope(gate)
    payload = envelope.to_dict()

    assert envelope.allows("adjust")
    assert payload["kind"] == "screen"
    assert payload["target_step_id"] == "gate-1"
    assert payload["controls"][0]["id"] == "leakage_ks"
    assert GateEnvelope.from_dict(payload).to_dict() == payload


def test_legacy_screen_metadata_infers_adjustable_gate_envelope():
    envelope = extract_gate_envelope({
        "metadata": {
            "kind": "gate",
            "plan_id": "plan-1",
            "step_id": "gate-screen",
            "run_seq": 2,
            "output_refs": {"screen": "metrics:screen:v1"},
            "screen": {"thresholds": {"leakage_ks": 0.35, "max_missing_rate": 0.9}},
        }
    })

    assert envelope.kind == "gate"
    assert envelope.target_step_id == "gate-screen"
    assert envelope.stale_token == "plan-1:gate-screen:2"
    assert envelope.allowed_actions == ("confirm", "adjust", "replan", "clarify", "halt")
    assert [control.id for control in envelope.controls] == ["leakage_ks", "max_missing_rate", "selection"]


def test_failure_envelope_round_trips_retry_contract():
    envelope = FailureEnvelope(
        failed_step_id="step-1",
        error_kind="audit",
        message="audit failed",
        retryable=False,
        suggested_actions=("replan", "halt"),
    )

    payload = envelope.to_dict()

    assert payload["schema_version"] == "failure.v1"
    assert payload["retryable"] is False
    assert FailureEnvelope.from_dict(payload).to_dict() == payload


def test_failure_envelope_round_trips_editable_inputs_and_reset_steps():
    envelope = FailureEnvelope(
        failed_step_id="train",
        message="training failed",
        editable_input_schema={
            "type": "object",
            "properties": {
                "num_leaves": {"type": "integer", "default": 31},
            },
            "additionalProperties": True,
        },
        downstream_reset_steps=("train", "report"),
        suggested_actions=("retry", "replan", "halt"),
    )

    payload = envelope.to_dict()

    assert payload["editable_input_schema"]["properties"]["num_leaves"]["default"] == 31
    assert payload["downstream_reset_steps"] == ["train", "report"]
    assert FailureEnvelope.from_dict(payload).to_dict() == payload
