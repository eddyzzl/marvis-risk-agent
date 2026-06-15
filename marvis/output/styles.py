from __future__ import annotations

from marvis.validation.results import ConsistencyStatus


BRAND_HEADER_FILL = "C00000"
BRAND_HEADER_FONT_COLOR = "FFFFFF"
BORDER_COLOR = "808080"
CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
)
FONT_NAME = CJK_FONT_CANDIDATES[0]
FONT_SIZE_PT = 8

_STATUS_FILLS = {
    ConsistencyStatus.PASS: "C6EFCE",     # green
    ConsistencyStatus.REVIEW: "FFEB9C",    # yellow
    ConsistencyStatus.FAIL: "F4B084",      # red
}

KS_DELTA_WARN_THRESHOLD = 0.01
KS_DELTA_FAIL_THRESHOLD = 0.03


def status_cell_color(status: ConsistencyStatus) -> str:
    return _STATUS_FILLS[status]


def ks_delta_cell_color(delta: float) -> str | None:
    magnitude = abs(delta)
    if magnitude >= KS_DELTA_FAIL_THRESHOLD:
        return "F4B084"
    if magnitude >= KS_DELTA_WARN_THRESHOLD:
        return "FFEB9C"
    return None
