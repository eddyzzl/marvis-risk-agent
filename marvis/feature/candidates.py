from __future__ import annotations

import re
from pathlib import Path


# Columns that should not be treated as model/feature-analysis candidates even
# when they are numeric. Keep this shared between setup and runtime tools so a
# JOIN-composed workflow infers the same feature family as the single-table path.
META_TOKENS = re.compile(
    r"(^|_)(id|uid|uuid|idcard|cust|user|order|loan|apply|cert|phone|mobile|name|"
    r"date|time|month|day|dt|ts|created|updated|weight|sample_weight)(_|$)",
    re.IGNORECASE,
)


def candidate_numeric_features(
    backend,
    path: Path,
    *,
    target_col: str,
    split_col: str | None = None,
    sample_rows: int = 4000,
) -> list[str]:
    """Infer numeric candidate features from the actual dataset schema/sample."""
    target = str(target_col)
    split = str(split_col) if split_col else ""
    probe = backend.sample_rows(Path(path), int(sample_rows), seed=0)
    excluded = {target, split}
    return [
        str(column)
        for column in probe.select_dtypes("number").columns
        if str(column) not in excluded and not META_TOKENS.search(str(column))
    ]


__all__ = ["META_TOKENS", "candidate_numeric_features"]
