"""Real loopback transport receipts expose bounded facts, never provider text."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.orchestrator.eval.runtime_runner import _model_gateway
from marvis.orchestrator.eval.runtime_scoring import usage_summary


@contextmanager
def provider(responses):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            status, content_type, body = responses[len(calls)]
            calls.append(True)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # Fragment inside both UTF-8 characters and JSON tokens.
            for start in range(0, len(body), 7):
                self.wfile.write(body[start : start + 7])
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            {
                "model_id": "fixture",
                "model_name": "fixture",
                "api_base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "api_key": "PROVIDER_KEY_SENTINEL",
                "max_output_tokens": 64,
                "transport_max_retries": 0,
                "timeout_seconds": 5,
            },
            calls,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def json_response(payload, status=200):
    return status, "application/json", json.dumps(payload, ensure_ascii=False).encode()


def receipts(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def complete(profile, **kwargs):
    return OpenAICompatibleLLMClient(profile).complete(
        system_prompt="SYSTEM_PROMPT_SENTINEL",
        user_prompt="PRIVATE_DATA_SENTINEL",
        caller="critic",
        **kwargs,
    )


def test_reasoning_only_length_retry_remains_visible_as_two_distinct_attempts(tmp_path):
    first = json_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "私有推理_SENTINEL",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 64,
                "completion_tokens_details": {"reasoning_tokens": 64},
            },
        }
    )
    second = json_response(
        {
            "choices": [
                {"message": {"content": "ANSWER_SENTINEL"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        }
    )
    path = tmp_path / "attempts.jsonl"
    with provider([first, second]) as (profile, calls):
        profile["transport_max_retries"] = 1
        with _model_gateway(
            profile,
            {},
            path,
            max_attempts=2,
            deadline=time.monotonic() + 10,
            max_output_tokens=64,
        ) as child:
            assert complete(child, stream=False) == "ANSWER_SENTINEL"
        assert len(calls) == 2
    events = receipts(path)
    starts = [e for e in events if e["event"] == "started"]
    finished = [e for e in events if e["event"] == "finished"]
    assert starts[0]["logical_call_id"] == starts[1]["logical_call_id"]
    assert [e["attempt"] for e in starts] == [1, 2]
    assert [e["http_status"] for e in finished] == [200, 200]
    assert finished[0]["transport_ok"] is True
    assert finished[0]["attempt_outcome"] == "empty_at_output_limit"
    assert finished[0]["finish_reason"] == "length"
    assert finished[0]["response_content_chars"] == 0
    assert finished[0]["response_reasoning_chars"] == len("私有推理_SENTINEL")
    assert finished[0]["reasoning_tokens"] == 64
    assert finished[1]["attempt_outcome"] == "content_present"
    assert "SENTINEL" not in path.read_text()
    summary = usage_summary(events, model_name="fixture", price_bytes=None)
    assert summary["attempt_outcomes"] == {
        "empty_at_output_limit": 1,
        "content_present": 1,
    }
    assert summary["finish_reasons"] == {"length": 1, "stop": 1}
    assert (
        summary["token_budget_status"]
        == summary["cost_budget_status"]
        == "not_enforced"
    )


def test_stream_finish_and_usage_survive_fragmentation_and_no_final_newline(tmp_path):
    records = [
        {"choices": [{"delta": {"reasoning_content": "私有推理_SENTINEL"}}]},
        {"choices": [{"delta": {"content": ""}, "finish_reason": "length"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 64,
                "completion_tokens_details": {"reasoning_tokens": 64},
            },
        },
    ]
    body = b"\n\n".join(
        b"data: " + json.dumps(x, ensure_ascii=False).encode() for x in records
    )
    path = tmp_path / "attempts.jsonl"
    with provider([(200, "text/event-stream", body)]) as (profile, calls):
        with _model_gateway(
            profile,
            {},
            path,
            max_attempts=1,
            deadline=time.monotonic() + 10,
            max_output_tokens=64,
        ) as child:
            with pytest.raises(LLMClientError) as raised:
                complete(child)
            assert raised.value.finish_reason == "length"
        assert len(calls) == 1
    event = next(e for e in receipts(path) if e["event"] == "finished")
    assert event["prompt_tokens"] == 5
    assert event["completion_tokens"] == event["reasoning_tokens"] == 64
    assert event["finish_reason"] == "length"
    assert event["attempt_outcome"] == "empty_at_output_limit"
    assert event["response_reasoning_chars"] == len("私有推理_SENTINEL")
    assert "SENTINEL" not in path.read_text()


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        ((200, "application/json", b"INVALID_ANSWER_SENTINEL"), "invalid_response"),
        (
            json_response({"choices": [{"message": {}, "finish_reason": "stop"}]}),
            "missing_content",
        ),
        (
            json_response(
                {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
            ),
            "empty_content",
        ),
        (
            json_response({"error": {"message": "PRIVATE_ERROR_SENTINEL"}}, status=503),
            "http_error",
        ),
    ],
)
def test_transport_success_does_not_claim_a_valid_or_nonempty_answer(
    tmp_path, response, outcome
):
    path = tmp_path / "attempts.jsonl"
    with provider([response]) as (profile, calls):
        with _model_gateway(
            profile, {}, path, max_attempts=1, deadline=time.monotonic() + 10
        ) as child:
            with pytest.raises(LLMClientError):
                complete(child, stream=False)
        assert len(calls) == 1
    event = next(e for e in receipts(path) if e["event"] == "finished")
    assert event["attempt_outcome"] == outcome
    assert event["transport_ok"] is (response[0] == 200)
    assert "SENTINEL" not in path.read_text()


def test_unknown_finish_reason_is_mapped_and_invalid_usage_remains_unknown(tmp_path):
    response = json_response(
        {
            "choices": [
                {
                    "message": {"content": "ANSWER_SENTINEL"},
                    "finish_reason": "PRIVATE_FINISH_REASON_SENTINEL",
                }
            ],
            "usage": ["PRIVATE_USAGE_SENTINEL"],
        }
    )
    path = tmp_path / "attempts.jsonl"
    with provider([response]) as (profile, _):
        with _model_gateway(
            profile, {}, path, max_attempts=1, deadline=time.monotonic() + 10
        ) as child:
            assert complete(child, stream=False) == "ANSWER_SENTINEL"
    event = next(e for e in receipts(path) if e["event"] == "finished")
    assert event["finish_reason"] == "other"
    assert event["attempt_outcome"] == "content_present"
    assert event["prompt_tokens"] is event["completion_tokens"] is None
    assert "SENTINEL" not in path.read_text()


def test_old_and_interrupted_receipts_never_invent_completion_outcomes():
    events = [
        {"event": "started", "attempt_id": "a", "logical_call_id": "a", "attempt": 1},
        {
            "event": "finished",
            "attempt_id": "a",
            "http_status": 200,
            "transport_ok": True,
        },
        {"event": "started", "attempt_id": "b", "logical_call_id": "b", "attempt": 1},
    ]
    summary = usage_summary(events, model_name="fixture", price_bytes=None)
    assert summary["attempt_outcomes"] == {"unknown": 1, "incomplete": 1}
    assert summary["finish_reasons"] == {"unknown": 2}
    assert summary["attempt_outcome_scope"] == "provider_transport_and_envelope_only"
    assert summary["usage_status"] == "unknown"
