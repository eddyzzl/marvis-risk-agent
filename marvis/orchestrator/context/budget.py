from __future__ import annotations

import json

from marvis.llm_client import estimate_tokens


def fit_to_budget(items: list[dict], *, max_chars: int) -> list[dict]:
    kept = []
    used = 0
    for item in sorted(items, key=lambda value: -int(value.get("priority", 0))):
        size = len(json.dumps(item, ensure_ascii=False, sort_keys=True))
        if used + size > max_chars:
            continue
        kept.append(item)
        used += size
    return kept


def truncate_text_to_token_budget(text: str, *, max_tokens: int) -> tuple[str, bool]:
    """Trim ``text`` from the tail (oldest content dropped first is the caller's
    responsibility when it orders text old-to-new) to fit within max_tokens,
    keeping the leading portion intact. Returns ``(text, truncated)``.
    """
    if estimate_tokens(text) <= max_tokens:
        return text, False
    # estimate_tokens is char-based; binary-search the char cut point so the
    # kept prefix's estimate lands at or under the budget.
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip(), True
