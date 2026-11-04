"""Structured one-click replies for Agent messages (2026-10-14 redesign, D11).

A quick reply is exactly a command the user could type. The frontend renders it
as a chip above the composer and sends ``command`` through the normal message
path, so turn routing, authorization checks and the audit trail are unchanged.
``label`` may spell out the consequence; ``command`` stays the literal phrase
the turn handlers already recognise. High-risk confirmations keep their full
explicit phrase as the command, so one click never weakens the gate.

``reply_hint`` carries an Agent-mode-only instruction (for example how to state
several choices in one message), so message content can stay mode-neutral and
the frontend never has to strip chat instructions out of free text.
"""

from __future__ import annotations

from collections.abc import Iterable

QUICK_REPLIES_KEY = "quick_replies"
REPLY_HINT_KEY = "reply_hint"
MAX_QUICK_REPLIES = 4


def quick_reply(label: str, command: str | None = None) -> dict[str, str]:
    text = str(command if command is not None else label).strip()
    shown = str(label).strip() or text
    if not text:
        raise ValueError("quick reply needs a non-empty command")
    return {"label": shown, "command": text}


def quick_replies(items: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """De-duplicate by command and cap the list; order is preserved."""
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        command = str(item.get("command") or "").strip()
        if not command or command in seen:
            continue
        seen.add(command)
        result.append({"label": str(item.get("label") or command).strip(), "command": command})
        if len(result) == MAX_QUICK_REPLIES:
            break
    return result


def with_quick_replies(
    metadata: dict,
    *items: dict[str, str],
    hint: str | None = None,
) -> dict:
    """Return ``metadata`` extended with quick replies and an optional hint."""
    merged = dict(metadata)
    replies = quick_replies(items)
    if replies:
        merged[QUICK_REPLIES_KEY] = replies
    if hint and hint.strip():
        merged[REPLY_HINT_KEY] = hint.strip()
    return merged
