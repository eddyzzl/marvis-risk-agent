from __future__ import annotations

from collections.abc import Callable
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMClientError(RuntimeError):
    pass


class OpenAICompatibleLLMClient:
    def __init__(self, profile: dict):
        self.profile = profile

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        response_format: dict | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        api_base_url = str(self.profile.get("api_base_url") or "").rstrip("/")
        model_name = str(self.profile.get("model_name") or "")
        api_key = str(self.profile.get("api_key") or "")
        if not api_base_url or not model_name or not api_key:
            raise LLMClientError("LLM profile is incomplete")
        if not api_base_url.startswith(("http://", "https://")):
            raise LLMClientError("api_base_url must start with http:// or https://")
        reasoning_effort = str(self.profile.get("reasoning_effort") or "high")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "reasoning_effort": reasoning_effort,
            "thinking": {"type": "enabled"},
            "temperature": temperature,
        }
        if response_format:
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
        try:
            with urlopen(request, timeout=timeout) as response:
                content = _read_completion_content(response, on_delta=on_delta)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
        except URLError as exc:
            raise LLMClientError(f"LLM request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMClientError("LLM request timed out") from exc
        return content.strip()


def _read_completion_content(
    response,
    *,
    on_delta: Callable[[str], None] | None = None,
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
        content = _content_from_stream_event(event_data)
        stream_parts.append(content)
        if content and on_delta:
            on_delta(content)

    if saw_stream_event:
        return "".join(stream_parts)

    if raw_parts:
        raw = b"".join(raw_parts).decode("utf-8", errors="replace")
    else:
        raw = response.read().decode("utf-8")
    return _content_from_json_response(raw)


def _content_from_stream_event(event_data: str) -> str:
    try:
        data = json.loads(event_data)
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


def _content_from_json_response(raw: str) -> str:
    try:
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMClientError("LLM response did not contain message content") from exc
    return str(content or "")
