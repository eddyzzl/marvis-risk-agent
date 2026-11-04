from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

from marvis.orchestrator.eval.perf_gate import (
    DEFAULT_PROFILE,
    formal_blockers,
    load_profile,
    run_formal_gate,
    validate_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _frozen():
    profile = copy.deepcopy(load_profile())
    profile.update(status="frozen", approved_by="tester", approved_on="2026-11-02")
    profile["hardware"] = {"description": "test box", "fingerprint_sha256": "a" * 64}
    profile["load"] = {"data_rows": 1000, "concurrent_users": 4, "sustained_seconds": 60}
    profile["slo"] = {m: {"p95_seconds": 1.0, "minimum_observations": 10, "max_error_rate": 0.01}
                      for m in ("cold_start", "steady", "concurrent", "recovery")}
    return profile


def test_repository_profile_is_pending_and_blocks_formal_gate():
    profile = load_profile()
    assert profile["status"] == "pending_user_freeze"
    assert "performance_profile_pending_user_freeze" in formal_blockers(profile)
    with pytest.raises(RuntimeError, match="formal_performance_gate_refused"):
        run_formal_gate(DEFAULT_PROFILE)


def test_frozen_profile_requires_every_decision():
    assert formal_blockers(validate_profile(_frozen())) == []
    for path in (("approved_by",), ("hardware", "fingerprint_sha256"), ("load", "data_rows"),
                 ("slo", "recovery", "p95_seconds")):
        profile = _frozen()
        target = profile
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = "pending"
        with pytest.raises(ValueError, match="frozen_profile_requires"):
            validate_profile(profile)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=1),
    lambda p: p.update(status="approved"),
    lambda p: p["slo"].pop("recovery"),
    lambda p: p["load"].update(data_rows=-5),
    lambda p: p["local_baseline"].update(concurrency=0),
    lambda p: p["hardware"].update(fingerprint_sha256="not-a-hash"),
])
def test_invalid_profiles_are_rejected(mutate):
    profile = copy.deepcopy(load_profile())
    mutate(profile)
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_manual_checklist_lists_54_blank_items():
    spec = importlib.util.spec_from_file_location("checklist", ROOT / "scripts/manual_acceptance_checklist.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    items = module.parse_items((ROOT / "docs/development/手动验收手册.md").read_text(encoding="utf-8"))
    assert [i["id"] for i in items] == [f"S{n:02d}" for n in range(1, 55)]
    html = module.render_html(items, "0" * 64, "2026-11-02")
    assert html.count("<tr><td class=id>") == 54
    assert html.count("<td class=sign></td>") == 54 + 2 * 4  # every signature cell is blank
    with pytest.raises(ValueError):
        module.parse_items("## 3. 软件操作检查\n| S01 | a | b |\n## 4.")


@pytest.mark.parametrize("url", ["file:///tmp/example", "https://127.0.0.1/", "http://example.com/",
    "http://user:pass@127.0.0.1/", "http://127.0.0.1/#fragment"])
def test_baseline_transport_rejects_nonlocal_or_ambiguous_targets(url):
    from marvis.orchestrator.eval.perf_gate import _request
    with pytest.raises(ValueError, match="literal_http_loopback"):
        _request(url)


def test_baseline_transport_keeps_redirect_as_original_response():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from marvis.orchestrator.eval.perf_gate import _request

    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            calls.append(self.path)
            self.send_response(302)
            self.send_header("Location", "file:///tmp/must-not-read")
            self.end_headers()
            self.wfile.write(b"original redirect")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, elapsed, body = _request(f"http://127.0.0.1:{server.server_port}/probe")
        assert status == 302 and elapsed >= 0 and body == b"original redirect"
        assert calls == ["/probe"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
