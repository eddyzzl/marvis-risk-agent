from fastapi.testclient import TestClient

from riskmodel_checker.app import create_app


def test_unsafe_methods_are_limited_to_local_clients(tmp_path):
    client = TestClient(create_app(tmp_path), client=("203.0.113.10", 4321))

    health = client.get("/api/health")
    response = client.put(
        "/api/settings/execution-environment",
        json={"execution_mode": "local", "kernel_name": "python3"},
    )

    assert health.status_code == 200
    assert response.status_code == 403
    assert response.json()["detail"] == "unsafe API methods are limited to local clients"
