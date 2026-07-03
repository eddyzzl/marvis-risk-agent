def test_settings_routes_live_in_dedicated_module():
    from marvis import api_settings

    route_paths = {route.path for route in api_settings.router.routes}

    assert "/settings/execution-environment" in route_paths
    assert "/settings/execution-environment/options" in route_paths
    assert "/settings/llm" in route_paths
    assert "/settings/llm/test" in route_paths
    assert all(
        route.endpoint.__module__ == "marvis.api_settings"
        for route in api_settings.router.routes
    )


def test_task_payload_helpers_live_in_dedicated_module():
    from marvis import api_task_payloads
    from marvis.api import _task_payload

    assert _task_payload is api_task_payloads.task_payload
    assert api_task_payloads.task_payload.__module__ == "marvis.api_task_payloads"


# --- ARCH-7 error-taxonomy convergence guard --------------------------------
#
# marvis/routers/* should raise HTTP errors through the semantic factories in
# marvis.errors (not_found/conflict/unprocessable/...), not hand-write
# ``raise HTTPException(status_code=..., detail=...)``. ARCH-7 converged 130 such
# call sites onto the factories; this test fails if a NEW bare raise is added, so
# the convergence does not quietly regress. Anything that genuinely cannot use a
# factory must be added to the whitelist below with a one-line justification.
#
# Baseline whitelist -- the only legitimate hand-written raises left after ARCH-7
# (each file has exactly one). A factory takes a plain string ``detail`` and pins
# a single status code, so these three do not fit:
#   plugins.py -- status_code is computed at runtime (422 vs 400 depending on the
#                 error text), so no single fixed-status factory applies.
#   plans.py   -- detail is a structured dict ({"problems": [...]}), which the
#                 string-detail factories intentionally do not accept.
#   drafts.py  -- detail is a structured dict ({"check": ...}), same reason.
_ARCH7_BARE_RAISE_WHITELIST = {
    "plugins.py": 1,
    "plans.py": 1,
    "drafts.py": 1,
}


def test_routers_use_error_factories_not_bare_httpexception():
    import re
    from pathlib import Path

    import marvis.routers as routers_pkg

    routers_dir = Path(routers_pkg.__file__).parent
    # Any bare `raise HTTPException(` in a router is a candidate for a factory.
    bare_pattern = re.compile(r"raise HTTPException\(")

    offenders = {}
    for path in sorted(routers_dir.glob("*.py")):
        count = len(bare_pattern.findall(path.read_text(encoding="utf-8")))
        if count:
            offenders[path.name] = count

    assert offenders == _ARCH7_BARE_RAISE_WHITELIST, (
        "Bare `raise HTTPException(...)` in routers changed from the ARCH-7 "
        "baseline. Use marvis.errors factories (not_found/conflict/...) instead, "
        "or add a justified entry to _ARCH7_BARE_RAISE_WHITELIST.\n"
        f"  expected: {_ARCH7_BARE_RAISE_WHITELIST}\n"
        f"  found:    {offenders}"
    )
