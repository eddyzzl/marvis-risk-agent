from __future__ import annotations

from types import SimpleNamespace

from scripts import run_closure_smoke


def test_runner_uses_isolated_workspace_per_flow(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="abc123\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(run_closure_smoke.subprocess, "run", fake_run)
    result = run_closure_smoke.run(
        tmp_path,
        python="python",
        temp_root=tmp_path / "fresh",
    )

    assert result["status"] == "PASS"
    assert [item["flow"] for item in result["results"]] == [
        "data_join",
        "feature_analysis",
        "modeling",
    ]
    roots = {item["fresh_workspace_root"] for item in result["results"]}
    assert len(roots) == 3
    pytest_calls = [command for command, _ in calls if "-m" in command]
    assert len(pytest_calls) == 3
    assert all("--basetemp" in command for command in pytest_calls)
