from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any

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


# LLM-5: shared truncation helper for the three highest-volume prompt touch
# points named in the review (decide_gate's gate content, the planner's tool
# catalog, and cross-task memory injection). Each caller trims its own
# truncatable segment (tables/catalog entries/memory packets) from oldest to
# newest — i.e. drop the tail — while keeping the leading "core instruction"
# portion of the segment intact, then reports whether it had to cut anything
# so the caller can set the audit-visible ``truncated`` flag on the LLM call.
def truncate_items_to_token_budget(
    items: list[Any],
    *,
    max_tokens: int,
    render: Callable[[Any], str],
) -> tuple[list[Any], bool]:
    """Keep items from the front until ``render``ing them would exceed max_tokens.

    ``render(item)`` returns the text used to estimate that item's token cost.
    Returns ``(kept_items, truncated)``.
    """
    kept: list[Any] = []
    used_tokens = 0
    truncated = False
    for item in items:
        item_tokens = estimate_tokens(str(render(item)))
        if kept and used_tokens + item_tokens > max_tokens:
            truncated = True
            continue
        if not kept and used_tokens + item_tokens > max_tokens:
            # Always keep at least one item (the most important one) even if it
            # alone exceeds the budget — an empty catalog/gate/memory section is
            # worse than a single over-budget one; the caller-level complete()
            # pre-flight check is still the final backstop.
            kept.append(item)
            used_tokens += item_tokens
            continue
        kept.append(item)
        used_tokens += item_tokens
    return kept, truncated


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
