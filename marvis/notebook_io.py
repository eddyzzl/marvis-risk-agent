from pathlib import Path
from typing import Any

import nbformat


NOTEBOOK_TEXT_ENCODINGS = (
    "utf-8",
    "utf-8-sig",
    "gb18030",
    "gbk",
    "cp1252",
)

_STANDARD_NOTEBOOK_TEXT_ENCODINGS = ("utf-8", "utf-8-sig")
_LEGACY_NOTEBOOK_TEXT_ENCODINGS = ("gb18030", "gbk", "cp1252")


def read_notebook(path: str | Path, *, as_version: int = 4) -> Any:
    """Read a notebook from disk with a conservative text-encoding fallback.

    Notebook files should be UTF-8, but real submitted materials sometimes arrive
    saved by Windows tools with GBK/GB18030 or cp1252 bytes. Decode fallback keeps
    the platform from failing before it can inspect or execute the notebook.
    """
    notebook_path = Path(path)
    raw = notebook_path.read_bytes()
    return read_notebook_bytes(
        raw,
        source=str(notebook_path),
        as_version=as_version,
    )


def read_notebook_bytes(
    raw: bytes,
    *,
    source: str = "<notebook bytes>",
    as_version: int = 4,
) -> Any:
    """Parse an immutable byte snapshot using the notebook encoding policy."""
    errors: list[str] = []
    for encoding in _STANDARD_NOTEBOOK_TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        try:
            return nbformat.reads(text, as_version=as_version)
        except Exception as exc:  # noqa: BLE001 - preserve nbformat's parse detail.
            errors.append(f"{encoding}: {exc}")

    legacy_candidates: dict[str, tuple[Any, list[str]]] = {}
    for encoding in _LEGACY_NOTEBOOK_TEXT_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
            continue
        if text in legacy_candidates:
            legacy_candidates[text][1].append(encoding)
            continue
        try:
            notebook = nbformat.reads(text, as_version=as_version)
        except Exception as exc:  # noqa: BLE001 - preserve nbformat's parse detail.
            errors.append(f"{encoding}: {exc}")
            continue
        legacy_candidates[text] = (notebook, [encoding])

    if len(legacy_candidates) == 1:
        return next(iter(legacy_candidates.values()))[0]
    if len(legacy_candidates) > 1:
        candidate_encodings = [
            "/".join(encodings) for _, encodings in legacy_candidates.values()
        ]
        raise ValueError(
            f"Unable to read notebook {source}: ambiguous notebook encoding "
            f"({', '.join(candidate_encodings)})"
        )
    detail = "; ".join(errors[-3:])
    raise ValueError(f"Unable to read notebook {source}: {detail}")
