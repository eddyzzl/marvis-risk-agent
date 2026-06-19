from __future__ import annotations

import json


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
