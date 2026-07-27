from __future__ import annotations

import numpy as np
import pytest

import marvis.packs.strategy.sample_membership as membership_module
from marvis.packs.strategy.sample_membership import (
    MEMBERSHIP_MASK_ORDER,
    StrategySampleMembershipError,
    canonical_sample_membership_header_json,
    decode_sample_membership,
    encode_sample_membership,
    sample_membership_header_from_json,
)


_DATASET_HASH = "a" * 64


def _membership_masks() -> dict[str, np.ndarray]:
    return {
        "approval/development": np.array(
            [True, False, True, False, False, False, False, False, False, False]
        ),
        "approval/validation": np.array(
            [False, False, False, False, False, False, False, False, True, False]
        ),
        "approval/oot": np.array(
            [False, False, False, False, False, False, False, False, False, True]
        ),
        "risk/development": np.array(
            [False, True, False, False, False, False, False, False, False, False]
        ),
        "risk/validation": np.array(
            [False, False, False, False, False, False, False, False, True, False]
        ),
        "risk/oot": np.zeros(10, dtype=bool),
    }


def test_membership_codec_is_canonical_little_endian_and_round_trips_six_masks():
    masks = _membership_masks()

    encoded = encode_sample_membership(
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash=_DATASET_HASH,
        masks=dict(reversed(list(masks.items()))),
    )
    decoded = decode_sample_membership(encoded)

    assert MEMBERSHIP_MASK_ORDER == (
        "approval/development",
        "approval/validation",
        "approval/oot",
        "risk/development",
        "risk/validation",
        "risk/oot",
    )
    assert decoded["header"]["mask_order"] == list(MEMBERSHIP_MASK_ORDER)
    assert decoded["header"]["row_ordinal"] == {
        "start": 0,
        "stop": 10,
        "step": 1,
    }
    assert decoded["header"]["counts"] == {
        "analysis_universe": 10,
        "approval": {
            "development": 2,
            "validation": 1,
            "oot": 1,
            "total": 4,
        },
        "risk": {
            "development": 1,
            "validation": 1,
            "oot": 0,
            "total": 2,
        },
        "relationship": {
            "risk_within_approval": {
                "development": 0,
                "validation": 1,
                "oot": 0,
                "total": 1,
            },
            "risk_outside_approval": {
                "development": 1,
                "validation": 0,
                "oot": 0,
                "total": 1,
            },
        },
    }
    for name, expected in masks.items():
        np.testing.assert_array_equal(decoded["masks"][name], expected)

    expected_little_endian_payload = bytes(
        [
            0x05,
            0x00,
            0x00,
            0x01,
            0x00,
            0x02,
            0x02,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
        ]
    )
    assert encoded.endswith(expected_little_endian_payload)
    assert encoded == encode_sample_membership(
        task_id="task-1",
        dataset_id="dataset-1",
        dataset_content_hash=_DATASET_HASH,
        masks=masks,
    )


def test_membership_codec_rejects_shape_overlap_and_payload_tampering():
    masks = _membership_masks()

    overlapping = {name: value.copy() for name, value in masks.items()}
    overlapping["approval/validation"][0] = True
    with pytest.raises(StrategySampleMembershipError, match="mutually exclusive"):
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=overlapping,
        )

    wrong_length = {name: value.copy() for name, value in masks.items()}
    wrong_length["risk/oot"] = np.zeros(9, dtype=bool)
    with pytest.raises(StrategySampleMembershipError, match="same row count"):
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=wrong_length,
        )

    encoded = bytearray(
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=masks,
        )
    )
    encoded[-1] ^= 0x80  # an unused high tail bit in the final mask byte
    with pytest.raises(StrategySampleMembershipError, match="payload_hash"):
        decode_sample_membership(encoded)

    truncated = bytes(encoded[:-1])
    with pytest.raises(StrategySampleMembershipError, match="payload length"):
        decode_sample_membership(truncated)

    with pytest.raises(StrategySampleMembershipError, match="codec_version"):
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=masks,
            codec_version="future-codec",
        )


def test_membership_header_has_one_canonical_hash_bound_json_shape():
    decoded = decode_sample_membership(
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=_membership_masks(),
        )
    )
    canonical = canonical_sample_membership_header_json(decoded["header"])

    assert sample_membership_header_from_json(canonical) == decoded["header"]
    with pytest.raises(StrategySampleMembershipError, match="duplicate key"):
        sample_membership_header_from_json(
            canonical.replace("{", '{"schema_version":"duplicate",', 1)
        )

    tampered = dict(decoded["header"])
    tampered["counts"] = {
        **tampered["counts"],
        "risk": {**tampered["counts"]["risk"], "total": 3},
    }
    with pytest.raises(StrategySampleMembershipError, match="do not conserve"):
        canonical_sample_membership_header_json(tampered)


def test_membership_encoder_checks_payload_budget_before_packing(monkeypatch):
    monkeypatch.setattr(membership_module, "MAX_MEMBERSHIP_PAYLOAD_BYTES", 1)

    def _must_not_pack(*_args, **_kwargs):
        raise AssertionError("packbits must not run after the budget is exceeded")

    monkeypatch.setattr(membership_module.np, "packbits", _must_not_pack)
    with pytest.raises(StrategySampleMembershipError, match="payload exceeds"):
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=_membership_masks(),
        )


def test_membership_header_rejects_relationship_tamper_unknown_and_nonfinite():
    decoded = decode_sample_membership(
        encode_sample_membership(
            task_id="task-1",
            dataset_id="dataset-1",
            dataset_content_hash=_DATASET_HASH,
            masks=_membership_masks(),
        )
    )
    tampered = dict(decoded["header"])
    tampered["counts"] = {
        **tampered["counts"],
        "relationship": {
            **tampered["counts"]["relationship"],
            "risk_outside_approval": {
                **tampered["counts"]["relationship"]["risk_outside_approval"],
                "development": 0,
            },
        },
    }
    with pytest.raises(
        StrategySampleMembershipError,
        match="risk_outside_approval counts|relationship counts",
    ):
        canonical_sample_membership_header_json(tampered)

    unknown = {**decoded["header"], "future": True}
    with pytest.raises(StrategySampleMembershipError, match="unknown: future"):
        canonical_sample_membership_header_json(unknown)

    canonical = canonical_sample_membership_header_json(decoded["header"])
    nonfinite = canonical.replace('"row_count":10', '"row_count":NaN')
    with pytest.raises(StrategySampleMembershipError, match="row_count"):
        sample_membership_header_from_json(nonfinite)
