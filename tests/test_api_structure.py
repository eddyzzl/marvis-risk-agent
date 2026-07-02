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
