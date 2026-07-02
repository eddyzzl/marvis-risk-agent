from __future__ import annotations

from collections.abc import Callable
from http.client import HTTPException
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        profile: dict,
        *,
        on_call_recorded: Callable[[dict], None] | None = None,
    ):
        self.profile = profile
        self._on_call_recorded = on_call_recorded

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        response_format: dict | None = None,
        json_schema: dict | None = None,
        on_delta: Callable[[str], None] | None = None,
        stream: bool = True,
        caller: str = "unknown",
        on_call_recorded: Callable[[dict], None] | None = None,
    ) -> str:
        recorder = on_call_recorded or self._on_call_recorded
        api_base_url = str(self.profile.get("api_base_url") or "").rstrip("/")
        model_name = str(self.profile.get("model_name") or "")
        api_key = str(self.profile.get("api_key") or "")
        if not api_base_url or not model_name or not api_key:
            raise LLMClientError("LLM profile is incomplete")
        if not api_base_url.startswith(("http://", "https://")):
            raise LLMClientError("api_base_url must start with http:// or https://")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": bool(stream),
            "temperature": temperature,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if self.profile.get("enable_thinking"):
            payload["reasoning_effort"] = str(self.profile.get("reasoning_effort") or "high")
            payload["thinking"] = {"type": "enabled"}
        extra_request_fields = self.profile.get("extra_request_fields")
        if isinstance(extra_request_fields, dict):
            payload.update(extra_request_fields)
        structured_output = str(self.profile.get("structured_output") or "json_object")
        if json_schema is not None and structured_output == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif response_format:
            payload["response_format"] = response_format
        request = Request(
            f"{api_base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        timeout = int(self.profile.get("timeout_seconds") or 60)
        prompt_chars = len(system_prompt or "") + len(user_prompt or "")
        usage: dict = {}
        started = time.monotonic()
        try:
            with urlopen(request, timeout=timeout) as response:
                content = _read_completion_content(
                    response, on_delta=on_delta, usage_out=usage
                )
        except HTTPError as exc:
            # Drain the body to free the socket, but never surface it: error bodies
            # can echo the rejected prompt and would be persisted into
            # agent_messages.metadata. Only the status code and reason are recorded.
            try:
                exc.read()
            except Exception:
                pass
            self._record_call(
                recorder, caller=caller, model_name=model_name,
                prompt_chars=prompt_chars, usage={}, started=started,
                streamed=bool(stream), ok=False,
                error_kind=_http_error_kind(exc.code),
            )
            raise LLMClientError(f"LLM HTTP {exc.code} {exc.reason}") from exc
        except URLError as exc:
            self._record_call(
                recorder, caller=caller, model_name=model_name,
                prompt_chars=prompt_chars, usage={}, started=started,
                streamed=bool(stream), ok=False, error_kind="connection",
            )
            raise LLMClientError(f"LLM request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            self._record_call(
                recorder, caller=caller, model_name=model_name,
                prompt_chars=prompt_chars, usage={}, started=started,
                streamed=bool(stream), ok=False, error_kind="timeout",
            )
            raise LLMClientError("LLM request timed out") from exc
        except (OSError, HTTPException) as exc:
            self._record_call(
                recorder, caller=caller, model_name=model_name,
                prompt_chars=prompt_chars, usage={}, started=started,
                streamed=bool(stream), ok=False, error_kind="stream_interrupted",
            )
            raise LLMClientError(f"LLM stream interrupted: {exc}") from exc
        self._record_call(
            recorder, caller=caller, model_name=model_name,
            prompt_chars=prompt_chars, usage=usage, started=started,
            streamed=bool(stream), ok=True, error_kind=None,
        )
        return content.strip()

    def _record_call(
        self,
        on_call_recorded: Callable[[dict], None] | None,
        *,
        caller: str,
        model_name: str,
        prompt_chars: int,
        usage: dict,
        started: float,
        streamed: bool,
        ok: bool,
        error_kind: str | None,
    ) -> None:
        if on_call_recorded is None:
            return
        latency_ms = int((time.monotonic() - started) * 1000)
        record = {
            "caller": caller,
            "model_id": (str(self.profile.get("model_id") or "") or None),
            "model_name": model_name,
            "prompt_chars": prompt_chars,
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
            "latency_ms": latency_ms,
            "ok": ok,
            "error_kind": error_kind,
            "streamed": streamed,
        }
        try:
            on_call_recorded(record)
        except Exception:
            # Observability must never break the call path.
            pass


def _http_error_kind(code) -> str:
    try:
        status = int(code)
    except (TypeError, ValueError):
        return "http_error"
    if 400 <= status < 500:
        return "http_4xx"
    if 500 <= status < 600:
        return "http_5xx"
    return "http_error"


def _usage_int(usage: dict, key: str) -> int | None:
    value = usage.get(key) if isinstance(usage, dict) else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_completion_content(
    response,
    *,
    on_delta: Callable[[str], None] | None = None,
    usage_out: dict | None = None,
) -> str:
    raw_parts: list[bytes] = []
    stream_parts: list[str] = []
    saw_stream_event = False

    for raw_line in response:
        line_bytes = (
            raw_line
            if isinstance(raw_line, bytes)
            else str(raw_line).encode("utf-8")
        )
        raw_parts.append(line_bytes)
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        saw_stream_event = True
        event_data = line.removeprefix("data:").strip()
        if event_data == "[DONE]":
            break
        content = _content_from_stream_event(event_data, usage_out=usage_out)
        stream_parts.append(content)
        if content and on_delta:
            on_delta(content)

    if saw_stream_event:
        return "".join(stream_parts)

    if raw_parts:
        raw = b"".join(raw_parts).decode("utf-8", errors="replace")
    else:
        raw = response.read().decode("utf-8")
    return _content_from_json_response(raw, usage_out=usage_out)


def _content_from_stream_event(event_data: str, *, usage_out: dict | None = None) -> str:
    try:
        data = json.loads(event_data)
        if usage_out is not None:
            _merge_usage(usage_out, data.get("usage"))
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is not None:
            return str(content)
        message = choice.get("message") or {}
        message_content = message.get("content")
        if message_content is not None:
            return str(message_content)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMClientError("LLM stream event did not contain valid JSON") from exc
    return ""


def _content_from_json_response(raw: str, *, usage_out: dict | None = None) -> str:
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMClientError("LLM response did not contain message content") from exc
    if usage_out is not None:
        _merge_usage(usage_out, data.get("usage"))
    return str(content or "")


def _merge_usage(usage_out: dict, usage) -> None:
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if usage.get(key) is not None:
                usage_out[key] = usage[key]
