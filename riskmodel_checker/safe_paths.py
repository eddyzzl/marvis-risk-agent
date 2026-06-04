import re
import unicodedata
from pathlib import Path


_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w一-鿿\-]+", re.UNICODE)
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def safe_filename_component(
    value: str,
    *,
    max_length: int = 64,
    fallback: str = "_",
) -> str:
    normalized = unicodedata.normalize("NFC", value)
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", normalized).strip("._")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:max_length]


def assert_within(parent: Path, candidate: Path) -> Path:
    resolved_parent = parent.resolve()
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate != resolved_parent
        and resolved_parent not in resolved_candidate.parents
    ):
        raise PermissionError(f"path escape: {candidate} not within {parent}")
    return resolved_candidate
