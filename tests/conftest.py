"""Shared pytest fixtures.

Pre-warm the pypmml JVM gateway once per test session, in the clean startup
environment, before any test runs.

Why: the suite uses pytest-randomly (random test order). pypmml's JVM gateway
is a process-wide singleton that is cold-started lazily on the first PMML use.
If that first use happens *after* another test has mutated the process state
(working directory, environment), the JVM launch can intermittently return an
empty gateway port and fail with ``int(b'')``. Establishing the gateway at
session start sidesteps the order dependency: every later PMML test reuses the
already-running singleton instead of cold-starting it.
"""

from __future__ import annotations

import pytest

from marvis.validation.input_contracts import PmmlInputManifest, StressUnit
from tests.validation_builders import make_ready_contract


@pytest.fixture
def ready_contract():
    return make_ready_contract()


@pytest.fixture
def pmml_contract(ready_contract):
    return ready_contract


@pytest.fixture
def direct_manifest(ready_contract):
    return ready_contract.require_pmml_manifest()


@pytest.fixture
def derived_manifest():
    return PmmlInputManifest(
        schema_version="marvis.pmml_input_manifest.v1",
        raw_required_fields=("age", "income"),
        derived_fields=("age_bucket",),
        model_features=("age_bucket", "income"),
        stress_units=(
            StressUnit("age_bucket", ("age",), ("age_bucket <- age",)),
            StressUnit("income", ("income",), ()),
        ),
        unsupported_derivations=(),
        output_candidates=("probability_1",),
        algorithm="xgb",
    )


@pytest.fixture(scope="session", autouse=True)
def _prewarm_pmml_gateway():
    """Start the pypmml JVM gateway once at session start (best effort)."""
    try:
        from pypmml.base import PMMLContext

        PMMLContext.getOrCreate()
    except Exception:
        # If pypmml or a JVM is unavailable, individual PMML tests will surface
        # that on their own; pre-warming must never block the rest of the suite.
        pass
    yield
