import json
import subprocess
from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "marvis" / "static" / "app.js"


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    # The poller has a destructured default argument (`{ preserveOptimistic }
    # = {}`), so the first brace after the name is not the function body.
    brace = source.index(") {", start) + 2
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def _run(script: str) -> dict:
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_agent_message_polling_backs_off_and_resets_after_progress():
    source = APP_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        [
            _function(source, "function agentMessagePollSignature"),
            _function(source, "function agentStreamPollDelay"),
            _function(source, "async function pollAgentMessagesUntilSettled"),
        ]
    )
    script = "\n".join(
        [
            "const AGENT_STREAM_POLL_INTERVAL_MS = 180;",
            "const AGENT_STREAM_POLL_IDLE_INTERVAL_MS = 1000;",
            "const AGENT_STREAM_POLL_LONG_INTERVAL_MS = 3000;",
            "const AGENT_STREAM_POLL_IDLE_AFTER_MS = 2000;",
            "const AGENT_STREAM_POLL_LONG_AFTER_MS = 15000;",
            "let selectedTaskId = 'task-A';",
            "let agentMessages = [{ id: 'thinking', role: 'assistant', content: '', metadata: { streaming: true } }];",
            "const delays = [];",
            "let loads = 0;",
            "let resetExpectedAt = -1;",
            "let resolvePending;",
            "const pending = new Promise((resolve) => { resolvePending = resolve; });",
            "async function sleep(delay) { delays.push(delay); }",
            "async function loadAgentMessages() {",
            "  loads += 1;",
            "  if (loads === 15) {",
            "    agentMessages = [{ id: 'progress', role: 'assistant', content: '', metadata: { kind: 'tool_progress', progress: { kind: 'model_tuning', algorithm: 'xgb', trial: 2, trial_total: 40 } } }];",
            "    resetExpectedAt = delays.length;",
            "  }",
            "  if (loads === 43) resolvePending();",
            "}",
            functions,
            "await pollAgentMessagesUntilSettled('task-A', pending);",
            "process.stdout.write(JSON.stringify({ delays, loads, resetExpectedAt }));",
        ]
    )
    payload = _run(script)

    assert payload["loads"] == 43
    assert payload["delays"][0] == 180
    assert 1000 in payload["delays"]
    assert 3000 in payload["delays"]
    assert payload["delays"][payload["resetExpectedAt"]] == 180
    assert len(payload["delays"]) == payload["loads"], (
        "settling during the last load must not schedule another poll"
    )


def test_agent_message_polling_stops_immediately_when_request_settles_during_wait():
    source = APP_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        [
            _function(source, "function agentMessagePollSignature"),
            _function(source, "function agentStreamPollDelay"),
            _function(source, "async function pollAgentMessagesUntilSettled"),
        ]
    )
    script = "\n".join(
        [
            "const AGENT_STREAM_POLL_INTERVAL_MS = 180;",
            "const AGENT_STREAM_POLL_IDLE_INTERVAL_MS = 1000;",
            "const AGENT_STREAM_POLL_LONG_INTERVAL_MS = 3000;",
            "const AGENT_STREAM_POLL_IDLE_AFTER_MS = 2000;",
            "const AGENT_STREAM_POLL_LONG_AFTER_MS = 15000;",
            "let selectedTaskId = 'task-A';",
            "let agentMessages = [];",
            "let loads = 0;",
            "let resolvePending;",
            "const pending = new Promise((resolve) => { resolvePending = resolve; });",
            "async function sleep() { return await new Promise(() => {}); }",
            "async function loadAgentMessages() { loads += 1; }",
            functions,
            "const polling = pollAgentMessagesUntilSettled('task-A', pending);",
            "resolvePending();",
            "await polling;",
            "process.stdout.write(JSON.stringify({ loads }));",
        ]
    )
    payload = _run(script)

    assert payload["loads"] == 0


def test_agent_poll_backoff_constants_keep_fast_first_paint_and_cap_long_jobs():
    source = APP_JS.read_text(encoding="utf-8")

    assert "const AGENT_STREAM_POLL_INTERVAL_MS = 180;" in source
    assert "const AGENT_STREAM_POLL_IDLE_INTERVAL_MS = 1000;" in source
    assert "const AGENT_STREAM_POLL_LONG_INTERVAL_MS = 3000;" in source
    assert "const AGENT_STREAM_POLL_IDLE_AFTER_MS = 2000;" in source
    assert "const AGENT_STREAM_POLL_LONG_AFTER_MS = 15000;" in source
    assert "Promise.race([" in _function(
        source, "async function pollAgentMessagesUntilSettled"
    )
