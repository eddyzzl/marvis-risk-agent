"""Real application benchmark: one isolated HTTP app per case, real tools.

Only the public case and one explicit model configuration reach the application.
The expected file is frozen by the parent and scored *after* the child exits.
This runner does not manufacture plans, tool results, invocations or DB state.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection, HTTPSConnection
import hmac
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from urllib.parse import urlparse

import httpx
import psutil

from .runtime_budget import AdmissionBudgetExceeded, AggregateBudget
from .runtime_contracts import (
    ModelConnection,
    RuntimeCase,
    RuntimeSuite,
    RUNTIME_FINISH_REASONS,
    digest,
)


class RuntimeBudgetExceeded(RuntimeError):
    pass


class RuntimeJourneyError(RuntimeError):
    pass


def _write_new(path: Path, payload) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


class AttemptObserver:
    """Append-only transport receipts, including retries and interrupted attempts."""

    def __init__(self, path: Path, *, max_attempts: int, deadline: float,
                 aggregate_budget: AggregateBudget | None = None):
        self.path = path
        self.max_attempts = max_attempts
        self.deadline = deadline
        self.count = 0
        self.closed = False
        self.failed = False
        self.lock = threading.Lock()
        self.aggregate_budget = aggregate_budget
        self.path.touch(exist_ok=False, mode=0o600)
        if aggregate_budget is not None and aggregate_budget.configured:
            self._append({"event": "aggregate_budget_policy", **aggregate_budget.policy()})

    def _append(self, value: dict):
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            self.failed = True
            raise RuntimeJourneyError("LLM observation persistence failed") from None

    def reject(self, reason):
        with self.lock:
            if not self.closed:
                self._append({"event": "transport_rejected", "reason": reason})

    def before_attempt(self, metadata: dict) -> str:
        with self.lock:
            if self.closed:
                raise RuntimeBudgetExceeded("runtime observation is closed")
            if self.failed:
                raise RuntimeJourneyError("LLM observation persistence failed")
            if time.monotonic() >= self.deadline or self.count >= self.max_attempts:
                self._append(
                    {
                        "event": "budget_blocked",
                        "reason": "llm_attempt_or_wall_limit",
                        "logical_call_id": metadata["logical_call_id"],
                    }
                )
                raise RuntimeBudgetExceeded("runtime LLM budget exhausted")
            ticket = uuid.uuid4().hex
            reservation = {}
            if self.aggregate_budget is not None:
                try:
                    reservation = self.aggregate_budget.reserve(
                        ticket, metadata["forwarded_max_output_tokens"]
                    )
                except AdmissionBudgetExceeded as exc:
                    self._append({"event": "budget_blocked", "reason": str(exc),
                                  "logical_call_id": metadata["logical_call_id"]})
                    raise RuntimeBudgetExceeded(str(exc)) from None
            self.count += 1
            self._append(
                {
                    "event": "started",
                    "attempt_id": ticket,
                    "ordinal": self.count,
                    **metadata,
                    **reservation,
                }
            )
            return ticket

    def after_attempt(self, ticket: str, result: dict):
        with self.lock:
            if not self.closed:
                settlement = self.aggregate_budget.settle(ticket, result) if self.aggregate_budget else {}
                self._append({"event": "finished", "attempt_id": ticket, **result, **settlement})

    def seal(self, reason="case_closed"):
        with self.lock:
            if not self.closed:
                try:
                    if self.aggregate_budget is not None and self.aggregate_budget.configured:
                        self._append({"event": "aggregate_budget_snapshot", **self.aggregate_budget.snapshot()})
                    self._append({"event": "measurement_closed", "reason": reason})
                finally:
                    self.closed = True


def _usage_fields(payload):
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    def count(value):
        return value if type(value) is int and value >= 0 else None

    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        details = {}
    return {
        "prompt_tokens": count(usage.get("prompt_tokens")),
        "completion_tokens": count(usage.get("completion_tokens")),
        "reasoning_tokens": count(details.get("reasoning_tokens")),
    }


class _CompletionObservation:
    """Bounded provider-envelope facts, not application-level success.

    Text is counted transiently and never retained. Content counts describe the
    first provider choice before any client thinking-tag removal or domain JSON
    validation. A nonempty answer therefore does not establish semantic success.
    """

    def __init__(self):
        self.finish_reason = None
        self.content_chars = 0
        self.content_non_whitespace = False
        self.reasoning_chars = 0
        self.observed = False
        self.invalid = False
        self.missing_content = False
        self.final_usage_candidate = False
        self.usage_regressed = False

    def read(self, raw, result, *, streaming):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            self.invalid = True
            return
        if not isinstance(payload, dict):
            self.invalid = True
            return
        # A later usage-only SSE event must not reset prior observed fields.
        usage_fields = _usage_fields(payload)
        complete_usage_chunk = all(usage_fields[key] is not None for key in ("prompt_tokens", "completion_tokens"))
        if payload.get("usage") is not None:
            # Confirmation belongs to one whole envelope. A later partial chunk
            # cannot inherit completeness from an earlier pair of counts.
            self.final_usage_candidate = False
        for key, value in usage_fields.items():
            if value is not None:
                field = f"observed_{key}_max"
                if value < result.get(field, 0):
                    self.usage_regressed = True
                result[field] = max(result.get(field, 0), value)
        result.update(
            {
                key: value
                for key, value in usage_fields.items()
                if value is not None
            }
        )
        choices = payload.get("choices")
        if streaming and choices in (None, []):
            if complete_usage_chunk:
                self.final_usage_candidate = self.finish_reason is not None
            return
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            self.missing_content = True
            return
        choice = choices[0]
        self.observed = True
        finish = choice.get("finish_reason")
        if finish is not None:
            self.finish_reason = (
                finish
                if isinstance(finish, str) and finish in RUNTIME_FINISH_REASONS
                else "other"
            )
        if complete_usage_chunk:
            self.final_usage_candidate = self.finish_reason is not None
        container = choice.get("delta") if streaming else None
        if not isinstance(container, dict):
            container = choice.get("message")
        if not isinstance(container, dict):
            if not streaming:
                self.missing_content = True
            return
        if not streaming and "content" not in container:
            self.missing_content = True
        content = container.get("content")
        if streaming and content is None and isinstance(choice.get("message"), dict):
            content = choice["message"].get("content")
        if content is not None and not isinstance(content, str):
            self.invalid = True
        elif isinstance(content, str):
            self.content_chars += len(content)
            self.content_non_whitespace |= bool(content.strip())
        reasoning = container.get("reasoning_content")
        if isinstance(reasoning, str):
            self.reasoning_chars += len(reasoning)

    def fields(self, result):
        shape = (
            "invalid"
            if self.invalid
            else "missing_content"
            if self.missing_content
            else "observed"
            if self.observed
            else "unknown"
        )
        if not result["transport_ok"]:
            outcome = (
                "http_error"
                if (result["http_status"] or 0) >= 400
                else "transport_error"
            )
        elif shape == "invalid":
            outcome = "invalid_response"
        elif shape == "missing_content":
            outcome = "missing_content"
        elif shape == "unknown":
            outcome = "unknown"
        elif self.finish_reason == "content_filter":
            outcome = "content_filtered"
        elif self.finish_reason == "length":
            outcome = (
                "content_at_output_limit"
                if self.content_non_whitespace
                else "empty_at_output_limit"
            )
        else:
            outcome = (
                "content_present" if self.content_non_whitespace else "empty_content"
            )
        return {
            "finish_reason": self.finish_reason,
            "response_shape": shape,
            "response_content_chars": self.content_chars if self.observed else None,
            "response_content_non_whitespace": self.content_non_whitespace
            if self.observed
            else None,
            "response_reasoning_chars": self.reasoning_chars if self.observed else None,
            "attempt_outcome": outcome,
        }


@contextmanager
def _model_gateway(
    profile, secret_env, attempt_path, *, max_attempts, deadline, max_output_tokens=2048,
    budget=None, price_bytes=None,
):
    """One credential-owning, metered endpoint shared by the app and all workers.

    Only this parent process knows the provider/key. A child profile contains one
    loopback endpoint and an ephemeral bearer token. No inherited role profile or
    model switch can bypass the case's shared attempt budget.
    """
    aggregate_budget = None
    if budget is not None and (budget.max_total_tokens is not None or budget.max_cost is not None):
        price = None
        if budget.max_cost is not None:
            from .runtime_scoring import PriceBook
            if price_bytes is None:
                raise RuntimeJourneyError("cost_budget_requires_frozen_price_book")
            price = PriceBook.model_validate_json(price_bytes)
            if price.model_name != profile["model_name"] or price.currency != budget.currency:
                raise RuntimeJourneyError("cost_budget_price_book_mismatch")
        aggregate_budget = AggregateBudget(
            context_window=profile["context_window"], max_total_tokens=budget.max_total_tokens,
            max_cost=budget.max_cost, currency=budget.currency, price=price,
            price_hash=digest(price_bytes) if price is not None else None,
        )
    observer = AttemptObserver(
        attempt_path, max_attempts=max_attempts, deadline=deadline, aggregate_budget=aggregate_budget
    )
    token = uuid.uuid4().hex + uuid.uuid4().hex
    provider_key = profile.get("api_key") or secret_env.get(profile.get("api_key_env"))
    if not provider_key:
        raise RuntimeJourneyError("missing_explicit_model_credential")
    connections = set()
    upstream_sockets = set()
    client_sockets = set()
    response_lock = threading.Lock()
    closing = threading.Event()
    endpoint = urlparse(profile["api_base_url"])
    output_cap = min(int(profile.get("max_output_tokens") or 2048), max_output_tokens)

    def abort_socket(sock):
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close_transports(reason="case_closed"):
        closing.set()
        if reason == "case_closed" and time.monotonic() >= deadline:
            reason = "wall_deadline"
        try:
            observer.seal(reason)
        except RuntimeJourneyError:
            # A full/unwritable evidence disk must not prevent network shutdown.
            # The context exit reports observer.failed after closing transports.
            pass
        with response_lock:
            active_connections = list(connections)
            active_upstreams = list(upstream_sockets)
            active_clients = list(client_sockets)
        for connection in active_connections:
            abort_socket(connection.sock)
        for sock in [*active_upstreams, *active_clients]:
            abort_socket(sock)
            try:
                sock.close()
            except OSError:
                pass

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(max(0.01, deadline - time.monotonic()))
            with response_lock:
                client_sockets.add(self.connection)

        def finish(self):
            try:
                super().finish()
            finally:
                with response_lock:
                    client_sockets.discard(self.connection)

        def reject(self, code, reason):
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": {"code": reason}}).encode())
            except OSError:
                pass

        def do_POST(self):
            if not hmac.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {token}"
            ):
                self.reject(401, "unauthorized")
                return
            if self.path != "/v1/chat/completions":
                self.reject(404, "unknown_route")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError
                body = self.rfile.read(length)
                payload = json.loads(body)
                if payload.get("model") != profile["model_name"]:
                    observer.reject("model_not_found")
                    self.reject(400, "model_not_found")
                    return
                trace = json.loads(self.headers.get("X-Marvis-Runtime-Trace", "{}"))
                if trace and trace.get("request_sha256") != digest(body):
                    raise ValueError
                fields = (
                    "logical_call_id",
                    "caller",
                    "model_id",
                    "model_name",
                    "prompt_name",
                    "prompt_version",
                    "prompt_chars",
                    "request_sha256",
                    "attempt",
                )
                metadata = {field: trace.get(field) for field in fields}
                metadata.update(
                    logical_call_id=str(
                        trace.get("logical_call_id") or uuid.uuid4().hex
                    ),
                    caller=str(trace.get("caller") or "untraced"),
                    model_id=profile["model_id"],
                    model_name=profile["model_name"],
                    request_sha256=digest(body),
                    attempt=int(trace.get("attempt") or 1),
                    trace_complete=bool(trace),
                )
                requested_output = payload.get("max_tokens")
                if type(requested_output) is not int or requested_output < 1:
                    raise ValueError
                # One sampled completion is the metering contract. Alternative
                # output limit fields or multiplicity cannot bypass reservation.
                if any(type(payload.get(key, 1)) is not int or payload.get(key, 1) != 1 for key in ("n", "best_of")) or "max_completion_tokens" in payload:
                    observer.reject("unsupported_output_budget_parameters")
                    self.reject(400, "unsupported_output_budget_parameters")
                    return
                effective_output = min(requested_output, output_cap)
                metadata["requested_max_output_tokens"] = requested_output
                metadata["forwarded_max_output_tokens"] = effective_output
                if effective_output != requested_output:
                    payload["max_tokens"] = effective_output
                    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                metadata["provider_request_sha256"] = digest(body)
            except (ValueError, TypeError, AttributeError):
                self.reject(400, "invalid_request")
                return
            try:
                ticket = observer.before_attempt(metadata)
            except RuntimeBudgetExceeded:
                self.reject(429, "runtime_budget_exhausted")
                return
            started = time.monotonic()
            result = {
                "transport_ok": False,
                "error_type": None,
                "http_status": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "reasoning_tokens": None,
                "usage_final": False,
            }
            observation = _CompletionObservation()
            response_hash = hashlib.sha256()
            connection = None
            transport_socket = None
            response = None
            sent_headers = False
            try:
                timeout = max(
                    0.01,
                    min(
                        float(profile.get("timeout_seconds") or 30),
                        deadline - time.monotonic(),
                    ),
                )
                connection_type = (
                    HTTPSConnection if endpoint.scheme == "https" else HTTPConnection
                )
                connection = connection_type(
                    endpoint.hostname, endpoint.port, timeout=timeout
                )
                with response_lock:
                    connections.add(connection)
                # Keep the connection/socket under parent control before reading
                # headers or bodies; a slow trickle must not reset a wall budget.
                connection.connect()
                transport_socket = connection.sock
                with response_lock:
                    upstream_sockets.add(transport_socket)
                if closing.is_set() or time.monotonic() >= deadline:
                    raise RuntimeBudgetExceeded("upstream wall budget exhausted")
                connection.request(
                    "POST",
                    endpoint.path.rstrip("/") + "/chat/completions",
                    body=body,
                    headers={
                        "Authorization": f"Bearer {provider_key}",
                        "Content-Type": "application/json",
                    },
                )
                response = connection.getresponse()  # redirects are never followed
                status = response.status
                result["http_status"] = status
                if status >= 400:
                    result["error_type"] = "HTTPError"
                content_type = response.headers.get("Content-Type", "application/json")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                sent_headers = True
                size = 0
                buffered = b""
                is_sse = "text/event-stream" in content_type
                saw_stream_done = False
                while True:
                    if closing.is_set() or time.monotonic() >= deadline:
                        raise RuntimeBudgetExceeded("stream wall budget exhausted")
                    # read(n) may repeatedly refill internally while a peer sends
                    # one byte forever. read1 returns after one underlying read.
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 8 * 1024 * 1024:
                        raise RuntimeBudgetExceeded(
                            "model response byte budget exhausted"
                        )
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    response_hash.update(chunk)
                    buffered += chunk
                    if is_sse:
                        while b"\n" in buffered:
                            line, buffered = buffered.split(b"\n", 1)
                            if line.startswith(b"data:") and line[5:].strip() == b"[DONE]":
                                saw_stream_done = True
                            if (
                                line.startswith(b"data:")
                                and line[5:].strip() != b"[DONE]"
                            ):
                                observation.read(line[5:], result, streaming=True)
                if is_sse:
                    if buffered.startswith(b"data:") and buffered[5:].strip() == b"[DONE]":
                        saw_stream_done = True
                    # The client also consumes a final SSE line without a newline.
                    # Preserve its usage/finish facts instead of silently dropping it.
                    if (
                        buffered.startswith(b"data:")
                        and buffered[5:].strip() != b"[DONE]"
                    ):
                        observation.read(buffered[5:], result, streaming=True)
                else:
                    observation.read(buffered, result, streaming=False)
                result["transport_ok"] = 200 <= status < 300
                result["usage_final"] = (
                    result["transport_ok"] and observation.final_usage_candidate
                    and not observation.invalid and not observation.usage_regressed
                    and (not is_sse or saw_stream_done)
                    and not (type(result["reasoning_tokens"]) is int
                             and type(result["completion_tokens"]) is int
                             and result["reasoning_tokens"] > result["completion_tokens"])
                )
                result["output_limit_exceeded"] = (
                    type(result["completion_tokens"]) is int
                    and result["completion_tokens"] > effective_output
                )
            except Exception as exc:
                result["error_type"] = type(exc).__name__
                if not sent_headers:
                    try:
                        self.reject(502, "upstream_transport_failure")
                    except OSError:
                        pass
            finally:
                if connection is not None:
                    connection.close()
                    with response_lock:
                        connections.discard(connection)
                        upstream_sockets.discard(transport_socket)
                if response is not None:
                    response.close()
                result.update(observation.fields(result))
                result["latency_ms"] = int((time.monotonic() - started) * 1000)
                result["response_sha256"] = response_hash.hexdigest()
                observer.after_attempt(ticket, result)

    class MeteringServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False

    server = MeteringServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.05), daemon=True
    )
    thread.start()
    deadline_timer = threading.Timer(
        max(0, deadline - time.monotonic()), close_transports, args=("wall_deadline",)
    )
    deadline_timer.daemon = True
    deadline_timer.start()
    child_profile = {
        key: value
        for key, value in profile.items()
        if key not in {"api_key", "api_key_env"}
    }
    child_profile.update(
        api_base_url=f"http://127.0.0.1:{server.server_port}/v1",
        api_key=token,
        _runtime_trace=True,
        max_output_tokens=output_cap,
    )
    try:
        yield child_profile
    finally:
        # Seal first: no new attempt and no late receipt may alter frozen evidence.
        deadline_timer.cancel()
        close_transports()
        server.shutdown()
        server.server_close()  # explicitly does not join daemon request handlers
        thread.join(timeout=2)
        if observer.failed:
            raise RuntimeJourneyError("measurement_persistence_failed")


def _serve(config_path: Path):
    # No case/expected file or parent dataset directory is passed to this child.
    import uvicorn
    from marvis.app import create_app

    config = json.loads(config_path.read_text())
    workspace = Path(config["workspace"])
    settings_dir = workspace / "settings"
    settings_dir.mkdir(parents=True)
    profile = config["profile"]
    _write_new(
        settings_dir / "llm.json",
        {"default_model_id": profile["model_id"], "models": [profile]},
    )
    app = create_app(workspace)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config["port"],
        log_level="warning",
        access_log=False,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _terminate(proc: subprocess.Popen):
    # This process group was created solely for this case; includes ToolRunner
    # children so a timed-out tool cannot continue consuming resources afterward.
    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=5)
    finally:
        # ToolRunner gives top-level workers their own process groups. A parent
        # process-group signal alone cannot stop those groups. psutil remembers
        # process creation times and protects against accidentally reused PIDs.
        for child in descendants:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        _, alive = psutil.wait_procs(descendants, timeout=2)
        if any(
            child.is_running() and child.status() != psutil.STATUS_ZOMBIE
            for child in alive
        ):
            raise RuntimeJourneyError("owned_process_cleanup_incomplete")


@contextmanager
def _application(
    root: Path,
    case: RuntimeCase,
    profile: dict,
    evidence_dir: Path,
    secret_env: dict,
    source_sha256: str,
    forbidden_inputs: tuple[Path, ...],
    price_bytes: bytes | None = None,
    process_state: dict | None = None,
):
    deadline = time.monotonic() + case.budget.wall_seconds
    with _model_gateway(
        profile,
        secret_env,
        evidence_dir / "llm-attempts.jsonl",
        max_attempts=case.budget.max_llm_attempts,
        deadline=deadline,
        max_output_tokens=case.budget.max_output_tokens_per_attempt,
        budget=case.budget,
        price_bytes=price_bytes,
    ) as child_profile:
        with _application_process(
            root, case, child_profile, source_sha256, deadline, forbidden_inputs,
            process_state=process_state,
        ) as app:
            yield app


@contextmanager
def _application_process(
    root: Path,
    case: RuntimeCase,
    profile: dict,
    source_sha256: str,
    deadline: float,
    forbidden_inputs: tuple[Path, ...],
    process_state: dict | None = None,
):
    workspace = root / "workspace"
    workspace.mkdir(mode=0o700)
    config = {
        "workspace": str(workspace),
        "profile": profile,
        "port": _free_port(),
    }
    config_path = root / "runtime-config.json"
    _write_new(config_path, config)
    source_root = Path(__file__).resolve().parents[3]
    _validate_package_isolation(source_root / "marvis", forbidden_inputs)
    # ToolRunner intentionally strips PYTHONPATH. Give its unchanged `python -m`
    # worker a package in the isolated cwd, without installing or mutating the
    # developer environment (or running against a different editable checkout).
    shutil.copytree(
        source_root / "marvis",
        root / "marvis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        symlinks=True,
    )
    if _package_identity(root) != source_sha256:
        raise RuntimeJourneyError("source_changed_since_manifest")
    # No user workspace, config root, API keys, proxy settings or profile path is
    # inherited incidentally. The provider credential stays in the parent.
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "CONDA_PREFIX")
        if key in os.environ
    }
    env["PYTHONPATH"] = str(source_root)
    env["PYTHONUNBUFFERED"] = "1"
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "marvis.orchestrator.eval.runtime_runner",
                "--serve-config",
                str(config_path),
            ],
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if process_state is not None:
            process_state["started"] = True
        try:
            base_url = f"http://127.0.0.1:{config['port']}"
            with httpx.Client(base_url=base_url, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeJourneyError("application_startup_failed")
                    try:
                        if client.get("/api/branding", timeout=0.5).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.05)
                else:
                    raise RuntimeBudgetExceeded("application_startup_timeout")
                yield client, workspace, deadline
                saved = json.loads((workspace / "settings" / "llm.json").read_text())
                if saved != {
                    "default_model_id": profile["model_id"],
                    "models": [profile],
                }:
                    raise RuntimeJourneyError("isolated_model_profile_changed")
        finally:
            _terminate(proc)
            if process_state is not None:
                process_state["stopped"] = True


class Journey:
    def __init__(self, client, case, deadline):
        self.client, self.case, self.deadline = client, case, deadline
        self.events: list[dict] = []
        self.count = 0
        self.task_id = None
        self.approval = None
        self.interventions = 0
        self.uploaded_samples: list[dict] = []

    def request(self, method, path, *, label, **kwargs):
        if (
            time.monotonic() >= self.deadline
            or self.count >= self.case.budget.max_http_requests
        ):
            raise RuntimeBudgetExceeded("runtime HTTP or wall budget exhausted")
        self.count += 1
        started = time.monotonic()
        event = {"ordinal": self.count, "stage": label, "method": method}
        try:
            response = self.client.request(
                method, path, timeout=max(0.01, self.deadline - started), **kwargs
            )
            event["status_code"] = response.status_code
            return response
        finally:
            event["duration_ms"] = int((time.monotonic() - started) * 1000)
            self.events.append(event)

    def json_request(self, method, path, *, label, **kwargs):
        response = self.request(method, path, label=label, **kwargs)
        if response.status_code >= 400:
            raise RuntimeJourneyError(f"{label}_http_{response.status_code}")
        return response.json()

    def plans(self):
        return self.json_request(
            "GET", f"/api/tasks/{self.task_id}/plans", label="read_plans"
        )["plans"]

    def wait_idle(self, changed_step=None):
        previous_state = None
        delay = 0.25
        while True:
            plans = self.plans()
            state = [
                (
                    p["id"],
                    p["status"],
                    [(s["id"], s["status"], s.get("output_ref")) for s in p["steps"]],
                )
                for p in plans
            ]
            delay = min(delay * 2, 2.0) if state == previous_state else 0.25
            previous_state = state
            statuses = {p["status"] for p in plans}
            moving = bool(statuses & {"running", "confirmed"})
            if changed_step:
                moving |= any(
                    s["id"] == changed_step and s["status"] == "awaiting_confirm"
                    for p in plans
                    for s in p["steps"]
                )
            if not moving:
                job = self.json_request(
                    "GET", f"/api/tasks/{self.task_id}/jobs/latest", label="read_job"
                ).get("job")
                if not job or job.get("status") not in {"queued", "running"}:
                    return plans
            # Real model calls take seconds. A 50 ms loop exhausted the entire
            # HTTP budget while a healthy workflow was still computing. Pace
            # unchanged state without raising any case budget or hiding failure.
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeBudgetExceeded("runtime wall budget exhausted")
            time.sleep(min(delay, remaining))

    def run(self, dataset_root: Path, workspace: Path):
        if self.case.task.task_type == "validation":
            self.prepare_validation_task(dataset_root, workspace)
        else:
            source = workspace / "source"
            source.mkdir()
            body = self.case.task.model_dump(exclude_none=True)
            body.update(source_dir=str(source), run_mode="agent")
            task = self.json_request("POST", "/api/tasks", label="create_task", json=body)
            self.task_id = task["id"]
            for material in self.case.materials:
                path = (dataset_root / material.path).resolve()
                if not path.is_relative_to(dataset_root.resolve()):
                    raise RuntimeJourneyError("material_identity_mismatch")
                # Upload the exact frozen bytes that were hashed. A source-file edit
                # between a separate hash pass and HTTP upload cannot change input.
                with path.open("rb") as source_stream, tempfile.TemporaryFile() as stream:
                    material_hash = hashlib.sha256()
                    while chunk := source_stream.read(1024 * 1024):
                        if time.monotonic() >= self.deadline:
                            raise RuntimeBudgetExceeded("material snapshot wall budget")
                        material_hash.update(chunk)
                        stream.write(chunk)
                    if material_hash.hexdigest() != material.sha256:
                        raise RuntimeJourneyError("material_identity_mismatch")
                    stream.seek(0)
                    uploaded = self.json_request(
                        "POST",
                        f"/api/tasks/{self.task_id}/datasets/upload",
                        label="upload_material",
                        files={"file": (path.name, stream)},
                        data={"role": material.role},
                    )
                    if material.role == "sample":
                        self.uploaded_samples.extend(uploaded["datasets"])
            payload = {"acceptance_mode": self.case.acceptance_mode}
            route = "start"
            if self.case.initial_message is not None:
                route = "messages"
                payload["content"] = self.case.initial_message
            labeling_entry = bool(self.case.actions and self.case.actions[0].kind == "submit_labeling_request")
            if not labeling_entry:
                self.json_request(
                    "POST",
                    f"/api/tasks/{self.task_id}/agent/{route}",
                    label="agent_initial_turn",
                    json=payload,
                )
                self.wait_idle()
        for action in self.case.actions:
            if action.kind == "start_validation_workflow":
                self.start_validation_workflow(action)
            elif action.kind == "start_validation_agent":
                from .runtime_validation_adapter import start_agent
                start_agent(self, action)
            elif action.kind == "confirm_current_validation_report":
                from .runtime_validation_adapter import confirm_current_report
                confirm_current_report(self, action)
            elif action.kind == "submit_labeling_request":
                from .runtime_labeling import submit_labeling_request
                submit_labeling_request(self, action)
            elif action.kind == "download_labeling_results":
                from .runtime_labeling import download_labeling_results
                download_labeling_results(self)
            elif action.kind == "message":
                self.interventions += 1
                message = {
                    "content": action.content,
                    "acceptance_mode": self.case.acceptance_mode,
                }
                if action.portfolio_request is not None:
                    message["portfolio_request"] = action.portfolio_request.model_dump(
                        exclude_none=True
                    )
                self.json_request(
                    "POST",
                    f"/api/tasks/{self.task_id}/agent/messages",
                    label="agent_user_turn",
                    json=message,
                )
                self.wait_idle()
            elif action.kind == "select_recommended_experiment":
                self.select_recommended_experiment(action)
            elif action.kind == "bind_single_strategy_sample":
                self.bind_single_strategy_sample(action)
            elif action.kind in {"approve_step", "reject_step", "retry_step"}:
                plans = self.plans()
                match = [
                    (p, s)
                    for p in plans
                    for s in p["steps"]
                    if _tool_name(s["tool_ref"]) == action.tool
                    and s["status"]
                    == (
                        "awaiting_confirm"
                        if action.kind != "retry_step"
                        else "failed"
                    )
                ]
                if len(match) != 1:
                    raise RuntimeJourneyError(
                        "declared_action_has_no_unique_current_step"
                    )
                plan, step = match[0]
                prefix = f"/api/plans/{plan['id']}/steps/{step['id']}"
                self.interventions += 1
                if action.kind in {"approve_step", "reject_step"}:
                    approval = {
                        "decision": "approve" if action.kind == "approve_step" else "reject",
                        "reason": action.content,
                        **step["confirmation_snapshot"],
                    }
                    route = prefix + "/decisions"
                    if action.kind == "approve_step":
                        self.approval = (route, approval)
                    self.json_request(
                        "POST", route,
                        label="human_approval" if action.kind == "approve_step" else "human_rejection",
                        json=approval,
                    )
                    self.wait_idle(changed_step=step["id"])
                else:
                    self.json_request(
                        "POST", prefix + "/retry", label="human_retry", json={}
                    )
                    self.wait_idle()
            elif action.kind == "replay_approval":
                if self.approval is None:
                    raise RuntimeJourneyError("no_prior_approval_to_replay")
                self.interventions += 1
                self.request(
                    "POST",
                    self.approval[0],
                    label="stale_approval_probe",
                    json=self.approval[1],
                )
                self.wait_idle()
            elif action.kind == "stop":
                self.interventions += 1
                self.json_request(
                    "POST",
                    f"/api/tasks/{self.task_id}/agent/stop",
                    label="user_stop",
                    json={},
                )

    def prepare_validation_task(self, dataset_root, workspace):
        """Upload only role/hash-bound validation artifacts through the product API."""
        from contextlib import ExitStack

        with ExitStack() as stack:
            files = []
            paths = {}
            for material in self.case.materials:
                path = (dataset_root / material.path).resolve()
                if not path.is_relative_to(dataset_root.resolve()) or not path.is_file():
                    raise RuntimeJourneyError("material_identity_mismatch")
                source = stack.enter_context(path.open("rb"))
                stream = stack.enter_context(tempfile.TemporaryFile())
                sha = hashlib.sha256()
                while chunk := source.read(1024 * 1024):
                    if time.monotonic() >= self.deadline:
                        raise RuntimeBudgetExceeded("material snapshot wall budget")
                    sha.update(chunk)
                    stream.write(chunk)
                if sha.hexdigest() != material.sha256:
                    raise RuntimeJourneyError("material_identity_mismatch")
                stream.seek(0)
                paths[material.role] = material.path
                files.append(("files", (material.path, stream)))
            uploaded = self.json_request(
                "POST", "/api/material-uploads", label="upload_validation_materials",
                files=files, data={"relative_paths": list(paths.values())},
            )
        source_dir = Path(uploaded["source_dir"]).resolve()
        if (not source_dir.is_relative_to((workspace / "material_uploads").resolve())
                or sorted(f["relative_path"] for f in uploaded["files"]) != sorted(paths.values())):
            raise RuntimeJourneyError("validation_upload_binding_mismatch")
        # The public compatibility workflow uses the existing manual/API entry.
        # Do not pretend this exercised the separate V2 validation Agent route.
        body = self.case.task.model_dump(exclude_none=True)
        from .runtime_validation_adapter import validation_entry
        compatibility = validation_entry(self.case) == "manual_compatibility_workflow"
        body.update(source_dir=str(source_dir), run_mode="manual" if compatibility else "agent")
        task = self.json_request("POST", "/api/tasks", label="create_task", json=body)
        self.task_id = task["id"]
        self.interventions += 1
        self.json_request(
            "PUT", f"/api/tasks/{self.task_id}/materials",
            label="human_validation_material_selection",
            json={role + "_path": path for role, path in paths.items()},
        )

    def start_validation_workflow(self, action):
        """A declared human start, bound to the sole current native template plan."""
        if self.case.task.task_type != "validation" or not action.content.strip():
            raise RuntimeJourneyError("validation_start_not_declared")
        if self.plans():
            raise RuntimeJourneyError("validation_start_requires_no_existing_plan")
        created = self.json_request(
            "POST", f"/api/tasks/{self.task_id}/plans", label="create_validation_workflow",
            json={"goal": "模型验证", "slots": {"task_id": self.task_id}},
        )["plan"]
        plan = self.json_request(
            "GET", f"/api/plans/{created['id']}", label="read_validation_start_snapshot",
        )["plan"]
        if (plan["task_id"] != self.task_id or plan["template_id"] != "model_validation"
                or plan["status"] != "validated"
                or [_tool_name(step["tool_ref"]) for step in plan["steps"]] != [
                    "v1_compat.scan_materials", "v1_compat.run_notebook",
                    "v1_compat.compute_validation_metrics", "v1_compat.render_reports",
                ]
                or any(step["inputs"] != {"task_id": self.task_id} for step in plan["steps"])):
            raise RuntimeJourneyError("validation_workflow_contract_changed")
        self.interventions += 1
        confirmed = self.json_request(
            "POST", f"/api/plans/{plan['id']}/confirm", label="human_validation_workflow_start",
            json=plan["confirmation_snapshot"],
        )["plan"]
        self.json_request(
            "POST", f"/api/plans/{plan['id']}/run", label="run_validation_workflow",
            json={"expected_plan_fingerprint": confirmed["confirmation_snapshot"]["expected_plan_fingerprint"]},
        )
        self.wait_idle()

    def bind_single_strategy_sample(self, action):
        """Replay a declared human sample/semantics selection through workspace CAS."""
        from marvis.api_schemas import (
            DataWorkspaceSnapshotResponse,
            DataWorkspaceUpdateRequest,
        )

        if self.case.task.task_type != "strategy" or len(self.uploaded_samples) != 1:
            raise RuntimeJourneyError("strategy_sample_binding_not_unique")
        sample = self.uploaded_samples[0]
        if sample.get("task_id") != self.task_id or sample.get("role") != "sample":
            raise RuntimeJourneyError("strategy_sample_binding_wrong_owner")
        if (
            action.semantic_mapping is None
            or action.semantic_mapping.target_col != self.case.task.target_col
        ):
            raise RuntimeJourneyError("strategy_sample_binding_target_mismatch")
        route = f"/api/tasks/{self.task_id}/data-workspace"
        snapshot = DataWorkspaceSnapshotResponse.model_validate(
            self.json_request(
                "GET",
                route,
                label="read_strategy_workspace",
            )
        )
        if snapshot.task_id != self.task_id:
            raise RuntimeJourneyError("strategy_sample_binding_wrong_owner")
        # A dataset change requires the product's reset payload, then a second
        # CAS-bound write for the human field mapping. Neither write is retried.
        self.interventions += 1
        if (snapshot.active_dataset_id, snapshot.active_dataset_content_hash) != (
            sample["id"],
            sample["content_hash"],
        ):
            reset = DataWorkspaceUpdateRequest(
                active_dataset_id=sample["id"],
                active_dataset_content_hash=sample["content_hash"],
                page="overview",
                selected_field=None,
                semantic_mapping={
                    "target_col": None,
                    "field_roles": {},
                    "business_names": {},
                },
            )
            snapshot = DataWorkspaceSnapshotResponse.model_validate(
                self.json_request(
                    "PUT",
                    route,
                    label="human_strategy_dataset_selection",
                    headers={"If-Match": str(snapshot.revision)},
                    json=reset.model_dump(),
                )
            )
            if snapshot.task_id != self.task_id or (
                snapshot.active_dataset_id,
                snapshot.active_dataset_content_hash,
            ) != (sample["id"], sample["content_hash"]):
                raise RuntimeJourneyError("strategy_sample_binding_wrong_owner")
        request = DataWorkspaceUpdateRequest(
            active_dataset_id=sample["id"],
            active_dataset_content_hash=sample["content_hash"],
            page="semantics",
            selected_field=None,
            semantic_mapping=action.semantic_mapping,
        )
        self.json_request(
            "PUT",
            route,
            label="human_strategy_sample_binding",
            headers={"If-Match": str(snapshot.revision)},
            json=request.model_dump(),
        )

    def select_recommended_experiment(self, action):
        """Replay an explicit public human policy using the currently shown gate.

        The candidate is supplied by the application presentation, never by the
        private expected file or by ranking metrics inside the benchmark.
        """
        if (
            action.tool != "modeling.select_experiment"
            or self.case.task.task_type != "modeling"
        ):
            raise RuntimeJourneyError("recommended_selection_wrong_gate")
        matches = [
            (p, s)
            for p in self.plans()
            for s in p["steps"]
            if _tool_name(s["tool_ref"]) == action.tool
            and s["status"] == "awaiting_confirm"
            and p["status"] == "awaiting_confirm"
        ]
        if len(matches) != 1:
            raise RuntimeJourneyError("declared_action_has_no_unique_current_step")
        plan, step = matches[0]
        messages = self.json_request(
            "GET", f"/api/tasks/{self.task_id}/agent/messages", label="read_selection_gate"
        )["messages"]
        gates = [
            m.get("metadata", {})
            for m in messages
            if m.get("role") == "assistant"
            and m.get("metadata", {}).get("plan_id") == plan["id"]
            and m.get("metadata", {}).get("step_id") == step["id"]
            and m.get("metadata", {}).get("kind") == "gate"
        ]
        if not gates:
            raise RuntimeJourneyError("recommended_selection_missing_display")
        gate = gates[-1]
        from marvis.api_schemas import StepConfirmationRequest
        from pydantic import ValidationError

        snapshot = gate.get("confirmation_snapshot")
        try:
            StepConfirmationRequest.model_validate(snapshot)
        except ValidationError:
            raise RuntimeJourneyError("recommended_selection_stale_display") from None
        if (
            snapshot != step.get("confirmation_snapshot")
            or snapshot["expected_plan_status"] != "awaiting_confirm"
        ):
            raise RuntimeJourneyError("recommended_selection_stale_display")
        delivery = gate.get("model_delivery") or {}
        candidates = delivery.get("candidates") or []
        recommended = delivery.get("recommended_experiment_id")
        ids = [c.get("id") for c in candidates if isinstance(c, dict)]
        marked = [
            c.get("id")
            for c in candidates
            if isinstance(c, dict) and c.get("recommended") is True
        ]
        if (
            not isinstance(recommended, str)
            or not recommended
            or len(ids) != len(candidates)
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)
            or marked != [recommended]
            or recommended not in ids
        ):
            raise RuntimeJourneyError("recommended_selection_not_unique")
        body = {
            "content": action.content,
            "ui_action": "confirm_gate",
            "acceptance_mode": self.case.acceptance_mode,
            "adjust_params": {"selected_experiment_id": recommended},
            "expected_plan_id": plan["id"],
            "expected_step_id": step["id"],
            **snapshot,
        }
        route = f"/api/tasks/{self.task_id}/agent/messages"
        self.interventions += 1
        self.json_request("POST", route, label="human_recommended_selection", json=body)
        self.approval = (route, body)
        self.wait_idle(changed_step=step["id"])


def _tool_name(value):
    if isinstance(value, str):
        return value
    return f"{value['plugin']}.{value['tool']}"


def _output_identity(output, evidence, tool):
    """Preserve finite legacy hashes; authenticate native open-ended band output.

    Native Strategy bands legitimately contain infinite interval endpoints. Their
    persisted producer binding uses the platform payload hash. Do not reclassify
    that authenticated result as corrupt because case identities use strict JSON.
    """
    try:
        return {"output_sha256": digest(output)}
    except ValueError:
        from copy import deepcopy
        from marvis.orchestrator.evidence import payload_hash

        if tool != "strategy.design_cutoff_bands":
            raise
        # Only the native open interval endpoints may be non-finite. A NaN
        # metric, infinite profit, interior boundary or another tool still fails.
        finite_check = deepcopy(output)
        edges = finite_check.get("band_edges", [])
        bands = finite_check.get("bands", [])
        if edges:
            if edges[0] == float("-inf"):
                edges[0] = None
            if edges[-1] == float("inf"):
                edges[-1] = None
        if bands:
            if bands[0].get("lo") == float("-inf"):
                bands[0]["lo"] = None
            if bands[-1].get("hi") == float("inf"):
                bands[-1]["hi"] = None
        digest(finite_check)
        native_hash = payload_hash(output)
        if native_hash != evidence.get("output_hash"):
            raise ValueError("native output hash differs from authenticated evidence") from None
        return {
            "output_sha256": native_hash.removeprefix("sha256:"),
            "output_hash_contract": "marvis.orchestrator.evidence.payload_hash",
        }


def _validation_report_files(workspace, task_id, output):
    """Inspect only the two native report carriers, never caller-selected paths."""
    import io
    import zipfile

    # macOS's system temporary directory may use /var -> /private/var. Resolve
    # the trusted workspace root once, while rejecting links below that root.
    workspace = workspace.resolve()
    result = []
    artifacts = output.get("artifacts", [])
    if not isinstance(artifacts, list):
        return result
    for kind, filename, part in (
        ("excel", "validation.xlsx", "xl/workbook.xml"),
        ("word", "validation_report.docx", "word/document.xml"),
    ):
        relative = f"tasks/{task_id}/outputs/{filename}"
        matches = [a for a in artifacts if isinstance(a, dict) and a.get("kind") == kind]
        if len(matches) != 1 or matches[0].get("path") != relative:
            continue
        path = workspace / relative
        try:
            if path.resolve() != path or not path.is_file() or path.stat().st_size > 32_000_000:
                continue
            raw = path.read_bytes()
            if not raw or len(raw) > 32_000_000:
                continue
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                entries = archive.infolist()
                if (len(entries) > 1024 or sum(i.file_size for i in entries) > 64_000_000
                        or part not in archive.namelist() or "[Content_Types].xml" not in archive.namelist()
                        or archive.testzip() is not None or not archive.read(part)):
                    continue
                if kind == "excel" and not any(i.filename.startswith("xl/worksheets/sheet") for i in entries):
                    continue
            result.append({"kind": kind, "sha256": digest(raw), "size_bytes": len(raw), "format_verified": True})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return result


def _receipts(workspace: Path, task_id: str | None, *, case=None, journey=None) -> tuple[dict, dict]:
    """Read authenticated output bindings only after the application has stopped.

    Raw tool outputs stay in memory for the scorer. The public artifact includes
    only execution identities, integrity hashes, statuses and safe counters.
    """
    from marvis.repositories.plans import PlanRepository
    from marvis.repositories.tasks import TaskRepository
    from marvis.repositories.datasets import DatasetRepository
    from marvis.agent.semantic_diagnostics import sanitize_diagnostics
    from marvis.settings import Settings

    db = Settings(workspace).db_path
    if task_id is None or not db.exists():
        return {"plans": [], "steps": [], "task_id": task_id}, {
            "outputs": {},
            "messages": [],
        }
    repo = PlanRepository(db)
    datasets = DatasetRepository(db)
    tasks = TaskRepository(db)
    evidence = {"task_id": task_id, "plans": [], "steps": []}
    job = tasks.get_latest_job(task_id)
    evidence["latest_job"] = (
        {
            key: job.get(key)
            for key in (
                "id",
                "kind",
                "status",
                "error_name",
                "started_at",
                "finished_at",
            )
        }
        if job
        else None
    )
    private = {"outputs": {}, "messages": tasks.list_agent_messages(task_id)}
    from .runtime_validation_adapter import validation_entry, pipeline_receipt
    if case is not None and validation_entry(case) == "standard_validation_agent_v2":
        evidence["validation_pipeline"] = pipeline_receipt(
            workspace, task_id, case, private["messages"],
            confirmation=getattr(journey, "validation_confirmation", None),
            downloads=getattr(journey, "validation_downloads", ()),
        )
    semantic_observations = []
    for ordinal, message in enumerate(private["messages"], 1):
        metadata = message.get("metadata")
        if message.get("role") != "assistant" or not isinstance(metadata, dict):
            continue
        safe = sanitize_diagnostics(metadata.get("semantic_diagnostics"))
        if safe:
            semantic_observations.append({"message_ordinal": ordinal, **safe})
    if semantic_observations:
        # Diagnostic facts only; neither raw message text nor authorization proof.
        # Preserve them before the isolated workspace is removed after scoring.
        evidence["semantic_observations"] = semantic_observations
    for plan in repo.list_plans_for_task(task_id):
        evidence["plans"].append(
            {
                "id": plan.id,
                "status": plan.status.value,
                "template_id": plan.template_id,
                "source": plan.source,
            }
        )
        for step in plan.steps:
            tool = step.tool_ref.label()
            receipt = {
                "id": step.id,
                "tool": tool,
                "status": step.status.value,
                "output_ref": step.output_ref,
                "runs": [],
            }
            for run in repo.list_step_runs(step.id):
                receipt["runs"].append(
                    {
                        k: run.get(k)
                        for k in (
                            "id",
                            "attempt",
                            "status",
                            "started_at",
                            "finished_at",
                            "error_kind",
                            "invocation_id",
                            "raw_output_hash",
                            "canonical_binding_verified",
                            "tool_version",
                            "manifest_hash",
                            "output_ref",
                        )
                    }
                )
            if step.status.value == "done" and step.output_ref:
                try:
                    bound = repo.load_step_presentation_binding(
                        step.id, step.output_ref
                    )
                    ev = bound["evidence"]
                    receipt["binding_verified"] = True
                    receipt.update(_output_identity(bound["output"], ev, receipt["tool"]))
                    receipt["producer_invocation_id"] = ev.get("producer_invocation_id")
                    receipt["input_hash"] = ev.get("input_hash")
                    receipt["parent_output_refs"] = ev.get("parent_output_refs", [])
                    receipt["result_datasets"] = []
                    for item in ev.get("result_dataset_bindings", []):
                        dataset = datasets.get_dataset(item["dataset_id"])
                        path = (workspace / "datasets" / dataset.source_path).resolve()
                        verified = (
                            path.is_relative_to((workspace / "datasets").resolve())
                            and path.is_file()
                            and digest(path.read_bytes()) == item["content_hash"]
                        )
                        receipt["result_datasets"].append(
                            {
                                "dataset_id": item["dataset_id"],
                                "content_hash": item["content_hash"],
                                "row_count": dataset.row_count,
                                "file_verified": verified,
                            }
                        )
                    receipt["output_files"] = []
                    for key in (
                        "report_path", "pmml_path", "model_card_path", "approval_package_path", "doc_path"
                    ):
                        if not isinstance(bound["output"].get(key), str):
                            continue
                        path = Path(bound["output"][key]).resolve()
                        if path.is_relative_to(workspace.resolve()) and path.is_file():
                            receipt["output_files"].append(
                                {
                                    "field": key,
                                    "sha256": digest(path.read_bytes()),
                                    "size_bytes": path.stat().st_size,
                                    "suffix": path.suffix,
                                }
                            )
                    if receipt["tool"] == "v1_compat.render_reports":
                        receipt["validation_report_files"] = _validation_report_files(
                            workspace, task_id, bound["output"],
                        )
                    private["outputs"][step.id] = bound["output"]
                    if tool == "labeling.define_label":
                        from .runtime_labeling import labeling_receipt
                        receipt["labeling"] = labeling_receipt(workspace, task_id, bound["output"])
                except (ValueError, KeyError, OSError):
                    receipt["binding_verified"] = False
            evidence["steps"].append(receipt)
    return evidence, private


def _read_attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"event": "incomplete_receipt"})
    return events


def _run_case(
    case, dataset_root, profile, case_dir, secret_env, source_sha256, forbidden_inputs,
    price_bytes=None,
    custody_run_dir=None, custody_bindings=None,
):
    started = time.monotonic()
    record = {
        "case_id": case.id,
        "family": case.family,
        "task_type": case.task.task_type,
        "case_set": case.case_set,
        "scenario": case.scenario,
        "case_sha256": digest(case.model_dump()),
        "materials": [
            {"sha256": item.sha256, "source_kind": item.source_kind, "role": item.role}
            for item in case.materials
        ],
        "budget": case.budget.model_dump(),
        "runtime_status": "completed",
        "error_code": None,
    }
    journey = None
    workspace = None
    private = {"outputs": {}, "messages": []}
    process_state = {"started": False, "stopped": False}
    custody_seconds = 0
    with tempfile.TemporaryDirectory(prefix="marvis-runtime-case-") as tmp:
        root = Path(tmp)
        try:
            with _application(
                root,
                case,
                profile,
                case_dir,
                secret_env,
                source_sha256,
                forbidden_inputs,
                price_bytes,
                process_state=process_state,
            ) as (client, workspace, deadline):
                journey = Journey(client, case, deadline)
                try:
                    journey.run(dataset_root, workspace)
                except BaseException:
                    if journey.task_id:
                        try:
                            response = client.post(
                                f"/api/tasks/{journey.task_id}/agent/stop",
                                json={},
                                timeout=2,
                            )
                            journey.events.append(
                                {
                                    "stage": "budget_or_failure_stop",
                                    "status_code": response.status_code,
                                }
                            )
                        except httpx.HTTPError:
                            journey.events.append(
                                {"stage": "budget_or_failure_stop", "status_code": None}
                            )
                    raise
        except KeyboardInterrupt:
            record.update(runtime_status="interrupted", error_code="operator_interrupt")
        except (RuntimeBudgetExceeded, httpx.TimeoutException):
            record.update(
                runtime_status="budget_exceeded", error_code="wall_or_call_budget"
            )
        except Exception as exc:
            # Never archive exception text: it may contain a prompt, file path or
            # model/provider response. Known runner codes are bounded literals.
            record.update(
                runtime_status="error",
                error_code=str(exc)
                if isinstance(exc, RuntimeJourneyError)
                else type(exc).__name__,
            )
        if workspace is None:
            workspace = root / "workspace"
        try:
            evidence, private = _receipts(
                workspace, journey.task_id if journey else None, case=case, journey=journey,
            )
            record["execution"] = evidence
        except Exception as exc:
            record["receipt_error"] = type(exc).__name__
            record["runtime_status"] = "receipt_error"
        record["http_events"] = journey.events if journey else []
        record["human_interventions"] = journey.interventions if journey else 0
        if custody_run_dir is not None:
            from .runtime_custody import capture_case, CustodyError

            custody_started = time.monotonic()
            try:
                record["evidence_custody"] = capture_case(
                    run_dir=custody_run_dir, workspace=workspace, dataset_root=dataset_root,
                    case=case, private=private,
                    bindings={**(custody_bindings or {}), "execution_before_scoring": record},
                    owned_processes_stopped=process_state["stopped"],
                    forbidden_secrets=tuple(secret_env.values()) + (profile.get("api_key", ""),),
                )
            except KeyboardInterrupt:
                record["evidence_custody"] = {
                    "status": "failed", "error_code": "operator_interrupt",
                    "acceptance_claim": "not_established",
                }
                record.update(runtime_status="interrupted", error_code="operator_interrupt")
            except Exception as exc:
                record["evidence_custody"] = {
                    "status": "failed", "error_code": str(exc) if isinstance(exc, CustodyError) else type(exc).__name__,
                    "acceptance_claim": "not_established",
                }
                record["runtime_status_before_custody_failure"] = record["runtime_status"]
                record.update(runtime_status="receipt_error", error_code="evidence_custody_failed")
            finally:
                custody_seconds = time.monotonic() - custody_started
                record["evidence_custody"]["duration_ms"] = int(custody_seconds * 1000)
    if case.task.task_type == "validation":
        from .runtime_validation_adapter import validation_entry
        record["runtime_entry"] = validation_entry(case)
    attempts = _read_attempts(case_dir / "llm-attempts.jsonl")
    if any(
        item["event"] == "budget_blocked"
        or item.get("output_limit_exceeded") is True
        or item.get("aggregate_budget_violation") is True
        or (
            item["event"] == "finished"
            and item.get("error_type") == "RuntimeBudgetExceeded"
        )
        or (
            item["event"] == "measurement_closed"
            and item.get("reason") == "wall_deadline"
        )
        for item in attempts
    ):
        record.update(
            runtime_status="budget_exceeded", error_code="llm_attempt_or_wall_budget"
        )
    elif any(item["event"] == "transport_rejected" for item in attempts):
        record.update(
            runtime_status="error", error_code="model_gateway_rejected_request"
        )
    record["duration_ms"] = int((time.monotonic() - started - custody_seconds) * 1000)
    record["llm_events"] = attempts
    record["isolated_workspace_removed"] = not workspace.exists()
    return record, private


def _validate_package_isolation(package: Path, forbidden_inputs: tuple[Path, ...]):
    """Reject private inputs or filesystem aliases before copying worker code."""
    root = package.resolve()
    for forbidden in forbidden_inputs:
        if forbidden.resolve().is_relative_to(root):
            raise ValueError(
                "evaluation inputs must be outside the application package"
            )
    for entry in package.rglob("*"):
        if entry.is_symlink():
            raise ValueError("runtime package must not contain symlinks")
        if entry.is_file() and any(entry.samefile(path) for path in forbidden_inputs):
            raise ValueError("runtime package contains an evaluation-input hardlink")


def _package_identity(root):
    if any(path.is_symlink() for path in (root / "marvis").rglob("*")):
        raise ValueError("runtime package must not contain symlinks")
    files = sorted(
        path
        for path in (root / "marvis").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    return digest(
        {str(path.relative_to(root)): digest(path.read_bytes()) for path in files}
    )


def _source_identity():
    from marvis.llm_prompts import prompt_version_snapshot

    root = Path(__file__).resolve().parents[3]

    def git(*args):
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False
        )
        return result.stdout if result.returncode == 0 else b""

    return {
        "commit": git("rev-parse", "HEAD").decode().strip() or None,
        "source_sha256": _package_identity(root),
        "dirty_diff_sha256": digest(git("diff", "HEAD", "--", "marvis")),
        "prompt_versions": prompt_version_snapshot(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("fastapi", "uvicorn", "httpx", "pandas", "pydantic")
        },
    }


def run_runtime_suite(
    *,
    cases_path: Path,
    expected_path: Path,
    dataset_root: Path,
    output_dir: Path,
    model: ModelConnection,
    model_source: str,
    price_book_path: Path | None = None,
    secret_env: dict | None = None,
    baseline_path: Path | None = None,
    evidence_custody_dir: Path | None = None,
) -> dict:
    """All cases enter the denominator, including infrastructure/scorer failures.

    The caller explicitly chooses real_model vs fixture_model; this label alone
    is never proof of A. Hidden independence and business approval remain external
    acceptance requirements, even for a successful real-model execution.
    """
    forbidden_inputs = (cases_path, expected_path)
    _validate_package_isolation(
        Path(__file__).resolve().parents[3] / "marvis", forbidden_inputs
    )
    cases_bytes = cases_path.read_bytes()
    suite = RuntimeSuite.model_validate_json(cases_bytes)
    expected_bytes = expected_path.read_bytes()  # freeze only; never sent to child
    baseline_bytes = baseline_path.read_bytes() if baseline_path else None
    if expected_path.resolve().is_relative_to(dataset_root.resolve()):
        raise ValueError("expected must be physically outside the runtime dataset root")
    if cases_path.resolve() == expected_path.resolve():
        raise ValueError("public cases and expected must be separate files")
    profile = model.profile(model_source)
    allowed_env = dict(secret_env or {})
    if model_source == "real_model" and model.api_key_env not in allowed_env:
        if model.api_key_env and os.environ.get(model.api_key_env):
            allowed_env[model.api_key_env] = os.environ[model.api_key_env]
    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:12]
    )
    run_dir = output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    custody_run_dir = None
    if evidence_custody_dir is not None:
        from .runtime_custody import prepare_custody_run

        custody_run_dir = prepare_custody_run(
            evidence_custody_dir, run_id,
            forbidden=(output_dir, dataset_root, cases_path, expected_path,
                       Path(__file__).resolve().parents[3]),
        )
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "execution_mode": "real_http_agent_tools",
        "model_source": model_source,
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_connection_sha256": digest(model.model_dump()),
        "source": _source_identity(),
        "cases_sha256": digest(cases_bytes),
        "expected_sha256": digest(expected_bytes),
        "case_ids": [case.id for case in suite.cases],
        "denominator": len(suite.cases),
        "acceptance_claim": "not_established",
        "cases": [],
    }
    if all(case.task.task_type == "validation" for case in suite.cases):
        from .runtime_validation_adapter import validation_entry
        entries = {validation_entry(case) for case in suite.cases}
        report["execution_mode"] = ("real_http_manual_compatibility_workflow_tools"
                                    if entries == {"manual_compatibility_workflow"}
                                    else "real_http_validation_native_entries")
    report["declared_max_total_llm_attempts"] = sum(
        case.budget.max_llm_attempts for case in suite.cases
    )
    report["declared_max_total_case_seconds"] = sum(
        case.budget.wall_seconds for case in suite.cases
    )
    report["model_source_attestation"] = (
        "operator_selected; not independent acceptance evidence"
    )
    _write_new(
        run_dir / "manifest.json", {k: v for k, v in report.items() if k != "cases"}
    )
    price_bytes = price_book_path.read_bytes() if price_book_path else None
    from .runtime_scoring import compare_baseline, score_case, summarize

    interrupted = False
    for case in suite.cases:
        case_dir = run_dir / case.id
        case_dir.mkdir(mode=0o700)
        if interrupted:
            record = {
                "case_id": case.id,
                "family": case.family,
                "task_type": case.task.task_type,
                "case_set": case.case_set,
                "scenario": case.scenario,
                "runtime_status": "not_run_after_interrupt",
                "error_code": "operator_interrupt",
                "execution": {"plans": [], "steps": []},
                "llm_events": [],
                "http_events": [],
            }
            private = {"outputs": {}, "messages": []}
        else:
            record, private = _run_case(
                case,
                dataset_root,
                profile,
                case_dir,
                allowed_env,
                report["source"]["source_sha256"],
                forbidden_inputs,
                price_bytes,
                custody_run_dir=custody_run_dir,
                custody_bindings={key: report[key] for key in (
                    "run_id", "source", "cases_sha256", "expected_sha256",
                    "model_connection_sha256", "model_source", "model_id", "model_name",
                )} if custody_run_dir is not None else None,
            )
            interrupted = record["runtime_status"] == "interrupted"
        # Commit the original execution outcome before attempting any scoring.
        _write_new(case_dir / "execution.json", record)
        try:
            scored = score_case(
                case,
                record,
                private,
                expected_bytes,
                model_source=model_source,
                price_bytes=price_bytes,
            )
        except Exception as exc:
            scored = {
                "passed": False,
                "scorer_error": type(exc).__name__,
                "a_evidence_eligible": False,
            }
        _write_new(case_dir / "score.json", scored)
        report["cases"].append(
            {
                **record,
                "score": scored,
                "execution_sha256": digest((case_dir / "execution.json").read_bytes()),
            }
        )
    report.update(summarize(report["cases"]))
    if baseline_bytes is not None:
        report["baseline_sha256"] = digest(baseline_bytes)
        report.update(compare_baseline(baseline_bytes, report))
    report["regression_gate_passed"] = report["all_passed"] and report.get(
        "regression_ok", True
    )
    report["report_path"] = str(run_dir / "report.json")
    _write_new(run_dir / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-config", type=Path, required=True)
    args = parser.parse_args()
    _serve(args.serve_config)


if __name__ == "__main__":
    main()
