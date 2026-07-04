"""T3-2: minimal lineage tuple (NumberProvenance) tests."""

from __future__ import annotations

from marvis import __version__
from marvis.provenance import (
    NumberProvenance,
    code_version,
    dataset_fingerprint,
    params_digest,
)


def test_code_version_is_app_version():
    assert code_version() == __version__


def test_params_digest_is_stable_and_order_independent():
    a = params_digest({"anchor_id": "ds_1", "feature_ids": ["ds_2"], "seed": 0})
    b = params_digest({"seed": 0, "feature_ids": ["ds_2"], "anchor_id": "ds_1"})
    assert a == b
    assert a.startswith("sha256:")
    # A changed parameter changes the digest.
    c = params_digest({"anchor_id": "ds_1", "feature_ids": ["ds_3"], "seed": 0})
    assert c != a


def test_params_digest_matches_executor_input_hash_convention():
    # Provenance's params_digest must agree with the evidence envelope's input_hash
    # convention (executor._payload_hash) so the two are comparable.
    from marvis.orchestrator.executor import _payload_hash  # noqa: PLC0415

    payload = {"anchor_id": "ds_1", "feature_ids": ["ds_2"], "seed": 3}
    assert params_digest(payload) == _payload_hash(payload)


def test_single_dataset_fingerprint_is_its_content_hash():
    fp = dataset_fingerprint(["abc123"])
    assert fp == "sha256:abc123"


def test_multi_dataset_fingerprint_is_stable_and_order_sensitive():
    ab = dataset_fingerprint(["a", "b"])
    ab2 = dataset_fingerprint(["a", "b"])
    ba = dataset_fingerprint(["b", "a"])
    assert ab == ab2
    assert ab != ba  # anchor vs feature order is part of identity
    assert ab.startswith("sha256:")


def test_missing_content_hash_is_recorded_as_unknown_not_dropped():
    # A dataset registered before content_hash existed must not silently vanish
    # from the fingerprint; it becomes the literal "unknown" so the gap is visible.
    with_missing = dataset_fingerprint(["a", None])
    with_explicit = dataset_fingerprint(["a", "unknown"])
    assert with_missing == with_explicit
    # Still deterministic across calls.
    assert with_missing == dataset_fingerprint(["a", None])


def test_build_assembles_full_tuple():
    prov = NumberProvenance.build(
        content_hashes=["anchor_hash", "feature_hash"],
        params={"anchor_id": "ds_1", "feature_ids": ["ds_2"], "seed": 5},
        seed=5,
    )
    assert prov.code_version == __version__
    assert prov.dataset_fingerprint == dataset_fingerprint(["anchor_hash", "feature_hash"])
    assert prov.params_digest.startswith("sha256:")
    assert prov.seed == 5


def test_to_dict_round_trips_the_four_fields():
    prov = NumberProvenance(
        dataset_fingerprint="sha256:x",
        code_version="2.0.0",
        params_digest="sha256:y",
        seed=None,
    )
    data = prov.to_dict()
    assert data == {
        "dataset_fingerprint": "sha256:x",
        "code_version": "2.0.0",
        "params_digest": "sha256:y",
        "seed": None,
    }
