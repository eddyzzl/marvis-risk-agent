from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPException
import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from marvis.agent.json_reply import strip_thinking

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConnectionTestResult:
    ok: bool
    latency_ms: int
    model_echo: str | None
    error_kind: str | None
    error_detail: str | None


# GAP-8: actionable Chinese hints per failure kind, shown directly by the
# settings-panel "test connection" button so a first-run user can tell "wrong
# port" apart from "wrong model name" apart from "never configured at all".
_CONNECTION_TEST_HINTS: dict[str, str] = {
    "connection": "无法连接到该地址，请检查服务是否已启动、地址与端口是否正确。",
    "timeout": "连接超时，请检查服务是否已启动、网络是否可达，或适当增加超时时间。",
    "http_404": "接口返回 404，请检查 API 地址（Base URL）是否正确。",
    "http_401": "认证失败（401），请检查 API 密钥是否正确。",
    "http_403": "认证被拒绝（403），请检查 API 密钥是否有效。",
    "model_not_found": "模型未找到，请检查模型名称是否正确。",
    "http_4xx": "请求被拒绝，请检查 API 地址、模型名称与密钥是否正确。",
    "http_5xx": "服务端返回错误，请检查服务是否正常运行。",
    "invalid_response": "服务返回内容无法解析，请检查地址是否指向兼容 OpenAI 协议的服务。",
    "incomplete_profile": "请先填写 API 地址、模型名称与密钥。",
}


def test_llm_connection(profile: dict, *, timeout_seconds: float = 5.0) -> LLMConnectionTestResult:
    """Cheapest-possible preflight: send a 1-token ping and report latency/model/error.

    Deliberately independent of OpenAICompatibleLLMClient.complete() -- this is a
    fixed-shape, non-streaming, no-retry, short-timeout probe used only to answer
    "is this reachable at all", not a real call site.
    """
    api_base_url = str(profile.get("api_base_url") or "").rstrip("/")
    model_name = str(profile.get("model_name") or "")
    api_key = str(profile.get("api_key") or "")
    if not api_base_url or not model_name or not api_key:
        return LLMConnectionTestResult(
            ok=False, latency_ms=0, model_echo=None,
            error_kind="incomplete_profile",
            error_detail=_CONNECTION_TEST_HINTS["incomplete_profile"],
        )
    if not api_base_url.startswith(("http://", "https://")):
        return LLMConnectionTestResult(
            ok=False, latency_ms=0, model_echo=None,
            error_kind="invalid_url",
            error_detail="api_base_url must start with http:// or https://",
        )
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "max_tokens": 8,
        "temperature": 0,
    }
    request = Request(
        f"{api_base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        latency_ms = int((time.monotonic() - started) * 1000)
        error_kind = _connection_test_http_error_kind(exc)
        return LLMConnectionTestResult(
            ok=False, latency_ms=latency_ms, model_echo=None,
            error_kind=error_kind,
            error_detail=_CONNECTION_TEST_HINTS.get(error_kind, f"LLM HTTP {exc.code} {exc.reason}"),
        )
    except TimeoutError:
        return LLMConnectionTestResult(
            ok=False, latency_ms=int((time.monotonic() - started) * 1000), model_echo=None,
            error_kind="timeout", error_detail=_CONNECTION_TEST_HINTS["timeout"],
        )
    except URLError as exc:
        return LLMConnectionTestResult(
            ok=False, latency_ms=int((time.monotonic() - started) * 1000), model_echo=None,
            error_kind="connection",
            error_detail=f"{_CONNECTION_TEST_HINTS['connection']}（{exc.reason}）",
        )
    latency_ms = int((time.monotonic() - started) * 1000)
    try:
        data = json.loads(raw)
        content = str(data["choices"][0]["message"]["content"] or "")
        echoed_model = str(data.get("model") or model_name)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return LLMConnectionTestResult(
            ok=False, latency_ms=latency_ms, model_echo=None,
            error_kind="invalid_response", error_detail=_CONNECTION_TEST_HINTS["invalid_response"],
        )
    del content  # only used to prove the round trip parsed; not surfaced.
    return LLMConnectionTestResult(
        ok=True, latency_ms=latency_ms, model_echo=echoed_model,
        error_kind=None, error_detail=None,
    )


def _connection_test_http_error_kind(exc: HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode("utf-8", errors="replace"))
        message = str((body.get("error") or {}).get("message") or "").lower()
        if "model" in message and ("not found" in message or "does not exist" in message):
            return "model_not_found"
    except Exception:
        pass
    if exc.code == 404:
        return "http_404"
    if exc.code == 401:
        return "http_401"
    if exc.code == 403:
        return "http_403"
    return _http_error_kind(exc.code)


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
        max_tokens: int | None = None,
        on_delta: Callable[[str], None] | None = None,
        stream: bool = True,
        caller: str = "unknown",
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        truncated: bool = False,
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
        # LLM-5: client-side context-window budget check before any request is sent.
        # A weak local model's window (default 32768) is easy to punch through as
        # planner catalogs/gate metadata grow; the previous failure mode was an
        # opaque "LLM HTTP 400" with the body deliberately discarded (never surface
        # a rejected prompt — it can echo into agent_messages.metadata). Call sites
        # that can shrink their own prompt should do so *before* calling complete()
        # (see marvis.orchestrator.context.budget.fit_to_budget); this is the last
        # line of defense that turns an inevitable server-side rejection into an
        # explicit, typed, sizes-included error instead.
        effective_max_tokens = int(max_tokens) if max_tokens is not None else _default_max_tokens(self.profile)
        context_window = _context_window(self.profile)
        estimated_prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
        if estimated_prompt_tokens + effective_max_tokens > context_window:
            raise LLMClientError(
                "上下文过长：prompt 约 "
                f"{estimated_prompt_tokens} tokens + max_tokens {effective_max_tokens} "
                f"超过模型窗口 {context_window} tokens，请缩短输入或换用更大窗口的模型。"
            )
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": bool(stream),
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if self.profile.get("enable_thinking"):
            thinking_style = str(self.profile.get("thinking_style") or "none")
            if thinking_style == "openai_reasoning":
                payload["reasoning_effort"] = str(
                    self.profile.get("reasoning_effort") or "high"
                )
            elif thinking_style == "anthropic":
                payload["thinking"] = {"type": "enabled"}
            elif thinking_style == "qwen_chat_template":
                payload["chat_template_kwargs"] = {"enable_thinking": True}
            # thinking_style == "none" (the default) sends no non-standard field.
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
        max_retries = _transport_max_retries(self.profile)
        started = time.monotonic()
        retry_count = 0
        while True:
            usage: dict = {}
            delta_fired = {"value": False}
            wrapped_on_delta = None
            if on_delta is not None:
                def wrapped_on_delta(chunk, _flag=delta_fired, _cb=on_delta):
                    _flag["value"] = True
                    _cb(chunk)
            try:
                with urlopen(request, timeout=timeout) as response:
                    content = _read_completion_content(
                        response, on_delta=wrapped_on_delta, usage_out=usage
                    )
            except HTTPError as exc:
                # Drain the body to extract a whitelisted error.code/error.type enum
                # only (never the message — a rejected-prompt body must never be
                # persisted into agent_messages.metadata), then discard the rest.
                body_error_kind = None
                try:
                    body_error_kind = _classify_http_error_body(exc.read())
                except Exception:
                    pass
                error_kind = body_error_kind or _http_error_kind(exc.code)
                if error_kind == "http_5xx" and retry_count < max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                self._record_call(
                    recorder, caller=caller, model_name=model_name,
                    prompt_chars=prompt_chars, usage={}, started=started,
                    streamed=bool(stream), ok=False, error_kind=error_kind,
                    retry_count=retry_count, prompt_name=prompt_name,
                    prompt_version=prompt_version, truncated=truncated,
                )
                if error_kind == "context_length_exceeded":
                    raise LLMClientError(
                        "上下文过长：模型拒绝了该请求(context_length_exceeded)，"
                        "请缩短输入或换用更大窗口的模型。"
                    ) from exc
                raise LLMClientError(f"LLM HTTP {exc.code} {exc.reason}") from exc
            except URLError as exc:
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                self._record_call(
                    recorder, caller=caller, model_name=model_name,
                    prompt_chars=prompt_chars, usage={}, started=started,
                    streamed=bool(stream), ok=False, error_kind="connection",
                    retry_count=retry_count, prompt_name=prompt_name,
                    prompt_version=prompt_version, truncated=truncated,
                )
                raise LLMClientError(f"LLM request failed: {exc.reason}") from exc
            except TimeoutError as exc:
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                self._record_call(
                    recorder, caller=caller, model_name=model_name,
                    prompt_chars=prompt_chars, usage={}, started=started,
                    streamed=bool(stream), ok=False, error_kind="timeout",
                    retry_count=retry_count, prompt_name=prompt_name,
                    prompt_version=prompt_version, truncated=truncated,
                )
                raise LLMClientError("LLM request timed out") from exc
            except (OSError, HTTPException) as exc:
                # A mid-stream interruption is only retryable if nothing has been
                # forwarded to the caller yet; retrying after on_delta emitted
                # content would make the UI roll back partial output.
                if not delta_fired["value"] and retry_count < max_retries:
                    retry_count += 1
                    time.sleep(_RETRY_BACKOFF_SECONDS)
                    continue
                self._record_call(
                    recorder, caller=caller, model_name=model_name,
                    prompt_chars=prompt_chars, usage={}, started=started,
                    streamed=bool(stream), ok=False,
                    error_kind="stream_interrupted", retry_count=retry_count,
                    prompt_name=prompt_name, prompt_version=prompt_version,
                    truncated=truncated,
                )
                raise LLMClientError(f"LLM stream interrupted: {exc}") from exc
            self._record_call(
                recorder, caller=caller, model_name=model_name,
                prompt_chars=prompt_chars, usage=usage, started=started,
                streamed=bool(stream), ok=True, error_kind=None,
                retry_count=retry_count, prompt_name=prompt_name,
                prompt_version=prompt_version, truncated=truncated,
            )
            return strip_thinking(content).strip()

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
        retry_count: int = 0,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        truncated: bool = False,
    ) -> None:
        latency_ms = int((time.monotonic() - started) * 1000)
        # Single choke point every call path (success, retry-exhausted, and
        # every typed failure kind) funnels through -- independent of whether
        # an on_call_recorded callback happens to be wired for this caller.
        if ok:
            logger.debug(
                "llm call ok caller=%s model=%s latency_ms=%d retry_count=%d",
                caller, model_name, latency_ms, retry_count,
            )
        else:
            logger.warning(
                "llm call failed caller=%s model=%s error_kind=%s retry_count=%d",
                caller, model_name, error_kind, retry_count,
            )
        if on_call_recorded is None:
            return
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
            "retry_count": retry_count,
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "truncated": bool(truncated),
        }
        try:
            on_call_recorded(record)
        except Exception:
            # Observability must never break the call path.
            pass


_RETRY_BACKOFF_SECONDS = 1.0

# LLM-5: defaults used when a profile omits context_window/max_output_tokens.
# 32768 matches the common local-model window cited in the review (Qwen/Llama
# class 32K checkpoints); 2048 is a safe default completion budget for JSON
# decisions/summaries — call sites that need more (e.g. the planner) pass an
# explicit max_tokens.
DEFAULT_CONTEXT_WINDOW = 32768
DEFAULT_MAX_OUTPUT_TOKENS = 2048

# Mixed CJK/ASCII token estimate: CJK glyphs run close to 1.6 chars/token,
# ASCII/latin text closer to 4 chars/token on common BPE tokenizers. This is a
# deliberately conservative (over-)estimate, not a tokenizer replacement — it
# only needs to be good enough to catch a request before the server rejects it.
_CJK_CHARS_PER_TOKEN = 1.6
_ASCII_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Conservative client-side token estimate for a client-window pre-check."""
    if not text:
        return 0
    cjk_chars = sum(1 for ch in text if _is_cjk(ch))
    ascii_chars = len(text) - cjk_chars
    return int(cjk_chars / _CJK_CHARS_PER_TOKEN) + int(ascii_chars / _ASCII_CHARS_PER_TOKEN) + 1


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x3000 <= code <= 0x303F  # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF  # fullwidth forms
    )


def _context_window(profile: dict) -> int:
    try:
        value = int(profile.get("context_window") or DEFAULT_CONTEXT_WINDOW)
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_WINDOW
    return value if value > 0 else DEFAULT_CONTEXT_WINDOW


def _default_max_tokens(profile: dict) -> int:
    try:
        value = int(profile.get("max_output_tokens") or DEFAULT_MAX_OUTPUT_TOKENS)
    except (TypeError, ValueError):
        return DEFAULT_MAX_OUTPUT_TOKENS
    return value if value > 0 else DEFAULT_MAX_OUTPUT_TOKENS


# Whitelisted OpenAI-compatible error.code / error.type enum values only — never
# the message, which can echo the rejected prompt back (the exact leak the
# caller-side "drain and discard" comment above guards against).
_CONTEXT_LENGTH_ERROR_TOKENS = frozenset({
    "context_length_exceeded",
    "context_window_exceeded",
    "string_above_max_length",
    "max_tokens_exceeded",
})


def _classify_http_error_body(raw: bytes | None) -> str | None:
    """Best-effort whitelist parse of an OpenAI-compatible error body.

    Only returns a fixed, non-message enum (`"context_length_exceeded"`) or
    ``None`` — the raw body/message is never returned or logged.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    for key in ("code", "type"):
        token = str(error.get(key) or "").strip().lower()
        if token in _CONTEXT_LENGTH_ERROR_TOKENS:
            return "context_length_exceeded"
    return None


def _transport_max_retries(profile: dict) -> int:
    value = profile.get("transport_max_retries")
    if value is None:
        return 1
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 1


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
