"""Performance gate profile validation and a local, non-formal baseline.

The hardware, load and SLO profile is a user decision. While it is
``pending_user_freeze`` the formal gate refuses and only the local baseline may
run; that report measures this machine and never declares the gate met.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import http.client
import math
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

SCHEMA = "marvis.performance-gate-profile.v1"
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = ROOT / "docs/development/performance-gate-profile.json"
LOAD_MODES = ("cold_start", "steady", "concurrent", "recovery")
BASELINE_LABEL = "本机基线，非正式"
_TOP = {"schema", "status", "approved_by", "approved_on", "note", "hardware", "load", "slo", "local_baseline"}
_HARDWARE = {"description", "fingerprint_sha256"}
_LOAD = {"data_rows", "concurrent_users", "sustained_seconds"}
_SLO = {"p95_seconds", "minimum_observations", "max_error_rate"}
_BASELINE = {
    "cold_start_runs": (1, 20), "sustained_seconds": (1, 600), "concurrency": (1, 64),
    "requests_per_worker": (1, 1000), "recovery_runs": (1, 10), "request_timeout_seconds": (1, 120),
    "startup_timeout_seconds": (5, 600),
}


def _is_pending(value):
    return value == "pending"


def _number(value):
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def _positive_integer(value):
    return type(value) is int and value > 0


def _error_rate(value):
    return type(value) in {int, float} and math.isfinite(value) and 0 <= value <= 1


def _approval_date(value):
    try:
        return type(value) is str and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def validate_profile(profile: dict) -> dict:
    """Strict schema; ``frozen`` requires every decision, ``pending`` allows ``"pending"``."""
    def need(condition, reason):
        if not condition:
            raise ValueError(reason)

    need(isinstance(profile, dict) and set(profile) == _TOP, "profile_keys")
    need(profile["schema"] == SCHEMA, "profile_schema")
    need(profile["status"] in {"pending_user_freeze", "frozen"}, "profile_status")
    frozen = profile["status"] == "frozen"
    need(isinstance(profile["hardware"], dict) and set(profile["hardware"]) == _HARDWARE, "hardware_keys")
    need(isinstance(profile["load"], dict) and set(profile["load"]) == _LOAD, "load_keys")
    need(isinstance(profile["slo"], dict) and set(profile["slo"]) == set(LOAD_MODES)
         and all(isinstance(v, dict) and set(v) == _SLO for v in profile["slo"].values()), "slo_keys")
    fingerprint = profile["hardware"]["fingerprint_sha256"]
    decisions = [
        ("approved_by", profile["approved_by"], lambda v: isinstance(v, str) and 0 < len(v.strip()) <= 120),
        ("approved_on", profile["approved_on"], _approval_date),
        ("hardware.description", profile["hardware"]["description"], lambda v: isinstance(v, str) and 0 < len(v.strip()) <= 500),
        ("hardware.fingerprint_sha256", fingerprint,
         lambda v: isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)),
        *((f"load.{k}", profile["load"][k], _positive_integer) for k in sorted(_LOAD)),
        *((f"slo.{m}.{k}", profile["slo"][m][k],
           _error_rate if k == "max_error_rate" else _positive_integer if k == "minimum_observations" else _number)
          for m in LOAD_MODES for k in sorted(_SLO)),
    ]
    for name, value, valid in decisions:
        if frozen:
            need(not _is_pending(value) and valid(value), f"frozen_profile_requires_{name}")
        else:
            need(_is_pending(value) or valid(value), f"invalid_{name}")
    if frozen:
        need(all(profile["slo"][m]["max_error_rate"] <= 1 for m in LOAD_MODES), "slo_error_rate_above_one")
    baseline = profile["local_baseline"]
    need(isinstance(baseline, dict) and set(baseline) == set(_BASELINE), "local_baseline_keys")
    for key, (low, high) in _BASELINE.items():
        need(type(baseline[key]) is int and low <= baseline[key] <= high, f"invalid_local_baseline_{key}")
    return profile


def load_profile(path: Path = DEFAULT_PROFILE) -> dict:
    return validate_profile(json.loads(Path(path).read_text(encoding="utf-8")))


def formal_blockers(profile: dict) -> list[str]:
    validate_profile(profile)
    if profile["status"] == "frozen":
        return []
    pending = ["approved_by"] if _is_pending(profile["approved_by"]) else []
    pending += [f"hardware.{k}" for k in sorted(_HARDWARE) if _is_pending(profile["hardware"][k])]
    pending += [f"load.{k}" for k in sorted(_LOAD) if _is_pending(profile["load"][k])]
    pending += [f"slo.{m}" for m in LOAD_MODES if any(_is_pending(v) for v in profile["slo"][m].values())]
    return ["performance_profile_pending_user_freeze", *(f"performance_{name}_pending" for name in pending)]


def run_formal_gate(profile_path: Path = DEFAULT_PROFILE, *, binding_path=None, out_dir=None):
    profile = load_profile(profile_path)
    blockers = formal_blockers(profile)
    if blockers:
        raise RuntimeError("formal_performance_gate_refused: " + ", ".join(blockers))
    if binding_path is None or out_dir is None:
        raise RuntimeError("formal_performance_gate_refused: frozen_workload_binding_and_output_required")
    from .perf_gate_formal import run_frozen_workload
    return run_frozen_workload(profile, Path(binding_path), Path(out_dir))


# ---- local baseline ---------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _request(url, *, method="GET", body=None, timeout=10):
    target = urlsplit(url)
    if target.scheme != "http" or target.hostname not in {"127.0.0.1", "::1"} or target.username or target.password or target.fragment:
        raise ValueError("baseline_requests_must_use_literal_http_loopback")
    data = None if body is None else json.dumps(body).encode()
    started = time.perf_counter()
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)
    try:
        # HTTPConnection does not follow redirects, including off-loopback ones.
        connection.request(method, target.path + ("?" + target.query if target.query else ""),
                           body=data, headers={"Content-Type": "application/json"} if data else {})
        response = connection.getresponse()
        payload = response.read()
        return response.status, time.perf_counter() - started, payload
    except (http.client.HTTPException, OSError, TimeoutError):
        return None, time.perf_counter() - started, b""
    finally:
        connection.close()


class _Server:
    def __init__(self, workspace: Path, port: int, log: Path):
        self.workspace, self.port, self.log = workspace, port, log
        self.process = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout):
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        handle = open(self.log, "ab")  # noqa: SIM115 - owned by the child for its lifetime
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "marvis", "serve", "--workspace", str(self.workspace),
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        handle.close()
        while time.perf_counter() - started < timeout:
            if self.process.poll() is not None:
                return None
            status, _, _ = _request(self.base + "/api/tasks", timeout=2)
            if status == 200:
                return time.perf_counter() - started
            time.sleep(0.1)
        return None

    def stop(self, *, kill=False):
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGKILL if kill else signal.SIGTERM)
            self.process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=10)


def _stats(latencies, errors, total):
    ordered = sorted(latencies)

    def rank(q):
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)] if ordered else None
    return {"observations": total, "measured": len(ordered), "errors": errors,
            "error_rate": errors / total if total else None, "retries": 0,
            "p50_seconds": rank(0.5), "p95_seconds": rank(0.95), "max_seconds": ordered[-1] if ordered else None}


def _task_body(source: Path, index: int) -> dict:
    return {"task_type": "modeling", "model_name": f"baseline_{index}", "validator": "perf-baseline",
            "source_dir": str(source), "run_mode": "manual"}


def _machine():
    try:
        import psutil
        memory = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - descriptive only
        memory = None
    return {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "memory_bytes": memory,
            "scope": "observed_this_machine_not_a_frozen_hardware_profile"}


def run_local_baseline(profile: dict, out_dir: Path) -> dict:
    validate_profile(profile)
    cfg = profile["local_baseline"]
    timeout, startup = cfg["request_timeout_seconds"], cfg["startup_timeout_seconds"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "server.log"
    results = {}
    with tempfile.TemporaryDirectory(prefix="marvis-perf-") as tmp:
        tmp = Path(tmp)
        cold = []
        for run in range(cfg["cold_start_runs"]):
            server = _Server(tmp / f"cold-{run}", _free_port(), log)
            seconds = server.start(startup)
            server.stop()
            cold.append(seconds)
        results["cold_start"] = _stats([s for s in cold if s is not None], sum(s is None for s in cold), len(cold))

        workspace = tmp / "load"
        source = workspace / "source"  # material roots must sit under the workspace
        source.mkdir(parents=True)
        server = _Server(workspace, _free_port(), log)
        if server.start(startup) is None:
            raise RuntimeError("baseline_server_failed_to_start")
        try:
            latencies, errors, total = [], 0, 0
            deadline = time.perf_counter() + cfg["sustained_seconds"]
            paths = ("/api/tasks", "/", "/api/branding")
            while time.perf_counter() < deadline:
                status, seconds, _ = _request(server.base + paths[total % len(paths)], timeout=timeout)
                total += 1
                if status == 200:
                    latencies.append(seconds)
                else:
                    errors += 1
            results["steady"] = _stats(latencies, errors, total)

            def worker(index):
                rows = []
                for n in range(cfg["requests_per_worker"]):
                    if n % 4 == 0:
                        rows.append(_request(server.base + "/api/tasks", method="POST",
                                             body=_task_body(source, index * 1000 + n), timeout=timeout)[:2])
                    else:
                        rows.append(_request(server.base + "/api/tasks", timeout=timeout)[:2])
                return rows
            with ThreadPoolExecutor(cfg["concurrency"]) as pool:
                rows = [row for chunk in pool.map(worker, range(cfg["concurrency"])) for row in chunk]
            results["concurrent"] = _stats([s for status, s in rows if status == 200],
                                           sum(status != 200 for status, _ in rows), len(rows))

            recovery, consistent = [], []
            for _ in range(cfg["recovery_runs"]):
                status, _, payload = _request(server.base + "/api/tasks", timeout=timeout)
                before = len(json.loads(payload)) if status == 200 else None
                server.stop(kill=True)
                seconds = server.start(startup)
                recovery.append(seconds)
                status, _, payload = _request(server.base + "/api/tasks", timeout=timeout) if seconds else (None, 0, b"")
                after = len(json.loads(payload)) if status == 200 else None
                consistent.append(before is not None and before == after)
            results["recovery"] = _stats([s for s in recovery if s is not None],
                                         sum(s is None for s in recovery), len(recovery))
            results["recovery"]["task_count_preserved_after_sigkill"] = consistent
        finally:
            server.stop()
    blockers = formal_blockers(profile)
    report = {
        "schema": "marvis.performance-local-baseline.v1",
        "label": BASELINE_LABEL,
        "formal": False,
        "gate_verdict": "not_evaluated_slo_not_frozen" if blockers else "not_evaluated_local_baseline_only",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile_status": profile["status"],
        "formal_blockers": blockers,
        "machine": _machine(),
        "parameters": cfg,
        "workloads": {
            "cold_start": "fresh empty workspace; process start until GET /api/tasks returns 200",
            "steady": "sequential GET /api/tasks, /, /api/branding for sustained_seconds",
            "concurrent": "threads mixing GET /api/tasks and POST /api/tasks (manual modeling task)",
            "recovery": "SIGKILL the server process group, restart on the same workspace until healthy",
        },
        "results": results,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report):
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    lines = [f"# 性能门 · {report['label']}", "",
             f"生成时间：{report['generated_at']} ｜ 配置状态：`{report['profile_status']}` ｜ 判定：`{report['gate_verdict']}`",
             "", "> 这是本机测量，不是冻结硬件/负载/SLO 下的正式性能门；不据此宣布达标。", "",
             "| 模式 | 观测 | 错误 | p50 秒 | p95 秒 | 最大 秒 |", "| --- | --- | --- | --- | --- | --- |"]
    for mode in LOAD_MODES:
        row = report["results"][mode]
        lines.append(f"| {mode} | {row['observations']} | {row['errors']} | {fmt(row['p50_seconds'])} | "
                     f"{fmt(row['p95_seconds'])} | {fmt(row['max_seconds'])} |")
    lines += ["", f"恢复后任务数一致：{report['results']['recovery'].get('task_count_preserved_after_sigkill')}",
              "", "## 正式运行仍缺", *(f"- `{b}`" for b in report["formal_blockers"]),
              "", "## 本机", "```json", json.dumps(report["machine"], ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="MARVIS performance gate (formal requires frozen profile and workload).")
    parser.add_argument("mode", choices=("validate", "formal", "baseline"))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--binding", type=Path, default=None)
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    if args.mode == "validate":
        print(json.dumps({"status": profile["status"], "formal_blockers": formal_blockers(profile)}, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "formal":
        try:
            report = run_formal_gate(args.profile, binding_path=args.binding, out_dir=args.out)
        except (RuntimeError, ValueError, OSError) as exc:
            print(str(exc))
            return 2
        print(json.dumps({"out": str(args.out), "gate_verdict": report["gate_verdict"],
                          "acceptance_claim": report["acceptance_claim"]}, ensure_ascii=False))
        return 0 if report["gate_verdict"] == "passed" else 1
    out = args.out or ROOT / ".omc/acceptance" / datetime.now().strftime("%Y-%m-%d") / "perf-baseline"
    report = run_local_baseline(profile, out)
    print(json.dumps({"label": report["label"], "out": str(out),
                      "results": {m: report["results"][m]["p95_seconds"] for m in LOAD_MODES}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
