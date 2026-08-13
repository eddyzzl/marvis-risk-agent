"""strategy_report driver-turn handlers (executed into the package namespace by __init__.py)."""
from __future__ import annotations
import re

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # names defined by sibling lanes; merged into one namespace at runtime
    from . import _StrategyV2EvidenceSetupError

_STRATEGY_REPORT_POOL_TYPE_PATTERNS = {
    "approval": re.compile(
        r"(?:审批|准入)|(?<![A-Za-z0-9_])approval(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "reject": re.compile(
        r"拒绝|(?<![A-Za-z0-9_])reject(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "limit": re.compile(
        r"额度|(?<![A-Za-z0-9_])limit(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "pricing": re.compile(
        r"定价|利率|(?<![A-Za-z0-9_])pricing(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
    "segmentation": re.compile(
        r"分群|分层|客群|"
        r"(?<![A-Za-z0-9_])segmentation(?![A-Za-z0-9_])",
        re.IGNORECASE,
    ),
}

_STRATEGY_REPORT_POOL_COMMAND_RE = re.compile(
    r"(?:生成|创建|制作|编制|形成|出一份|出个|给我|导出|构建)"
    r"[^；;。.!?？\n]{0,80}(?:报告|Report)|"
    r"(?<![A-Za-z0-9_])(?:generate|create|build|produce|prepare|render|export)"
    r"[^;.!?\n]{0,80}\breport(?:\s+bundle)?\b",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_TITLE_RE = re.compile(
    r"(?:报告标题|标题|report\s+title|title)\s*"
    r"(?:为|是|叫|is|=|:|：)\s*"
    r"(?:[“\"'《][^”\"'》\n]{1,200}[”\"'》]|"
    r"[^，,；;。.!?？\n]{1,200})",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_SELECTOR_RE = re.compile(
    r"(?:选择|选用|使用|采用|针对|指定|按|基于|改用|就用|要用|而是|"
    r"(?:Pool|策略)\s*类型\s*(?:为|是|=|:|：)|"
    r"(?<![A-Za-z0-9_])(?:select|choose|use|using|for|on|but|instead)"
    r"(?![A-Za-z0-9_]))\s*$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_TYPE_NEGATION_RE = re.compile(
    r"(?:不要|不用|无需|先别|先不|暂不|禁止|排除|剔除|而非|不是|并非|"
    r"不使用|不选|不选择)\s*(?:(?:选择|选用|使用|采用|针对|指定)\s*)?"
    r"[^，,；;。.!?？\n]{0,16}$|"
    r"(?<![A-Za-z0-9_])(?:do\s+not|don't|dont|not|never|without|exclude)"
    r"[^,;.!?\n]{0,20}$",
    re.IGNORECASE,
)

_STRATEGY_REPORT_POOL_HISTORY_RE = re.compile(
    r"(?:昨天|之前|此前|过去|上次|曾经|历史|已归档|已生成)|"
    r"(?<![A-Za-z0-9_])(?:yesterday|previously|earlier|historical|"
    r"last\s+time|archived|already\s+generated)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT = 64

_STRATEGY_REPORT_VOTING_SEARCH_REPLAY_LIMIT = (
    _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
)

_STRATEGY_REPORT_CROSS_SEARCH_REPLAY_LIMIT = (
    _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
)

_STRATEGY_REPORT_POOL_STABILITY_REPLAY_LIMIT = (
    _STRATEGY_REPORT_EVIDENCE_REPLAY_LIMIT
)

def _raise_corrupt_report_optional(
    label: str,
    *,
    cause: Exception | None = None,
) -> None:
    error = _StrategyV2EvidenceSetupError(
        "strategy_report_bundle_v2_optional_evidence_invalid",
        f"最新 {label} artifact 未通过完整认证；平台不会回退到旧证据。",
    )
    if cause is None:
        raise error
    raise error from cause

