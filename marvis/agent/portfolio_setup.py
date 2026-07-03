"""Setup (slot-filling) for the portfolio-analysis task (S3).

Probes a performance (表现期快照) table (by role=performance or column hints),
infers the id / snapshot / bucket columns, enumerates the bucket states from the
data and proposes a deterioration order, then hands the user a C1-style
confirmation gate: **the bucket state semantic order must be confirmed by a
human** (机器不可猜) before any analysis runs. Mirrors join_setup's C1 roles gate
(gate state in message metadata + a parse of the reply), but self-contained under
its own ``portfolio_states`` metadata key rather than reusing join's C1 machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from marvis.domain import FileRole
from marvis.files import scan_source_dir

_PERFORMANCE_ROLES = frozenset({"performance"})
_SAMPLE_ROLES = frozenset({FileRole.SAMPLE.value, "sample"})
_ID_HINTS = ("loan_id", "loanid", "id", "account_id", "acct_id", "contract_id", "cust_id")
_SNAPSHOT_HINTS = ("snapshot_month", "snapshot", "obs_month", "observe_month", "month", "stat_month", "dt")
_BUCKET_HINTS = ("bucket", "delinq", "dpd_bucket", "overdue_bucket", "status", "stage", "state")
_BALANCE_HINTS = ("balance", "bal", "principal", "outstanding", "ead", "exposure")
_SEGMENT_HINTS = ("segment", "product", "channel", "grade", "region", "seg")

# Ordering hint: lower rank = healthier. Any state whose lowercase name contains
# one of these markers is placed by this rank; unmatched states keep input order
# after the ranked ones. This is only a *proposal* -- the user must confirm.
_STATE_RANK_MARKERS = (
    ("current", 0),
    ("c", 1),
    ("m0", 2),
    ("1-30", 3),
    ("m1", 4),
    ("31-60", 5),
    ("m2", 6),
    ("61-90", 7),
    ("m3", 8),
    ("90", 9),
    ("charge", 98),
    ("loss", 99),
    ("writeoff", 99),
    ("write_off", 99),
)


class PortfolioSetupError(ValueError):
    """Raised when a portfolio task cannot infer the performance table/columns."""


@dataclass
class PortfolioProposal:
    dataset_id: str
    dataset_name: str
    id_col: str
    snapshot_col: str
    bucket_col: str
    proposed_states: list[str]
    balance_col: str | None = None
    segment_col: str | None = None
    score_col: str | None = None
    experiment_id: str | None = None
    project_meta: dict = field(default_factory=dict)

    @property
    def template_id(self) -> str:
        # 剪步语义：无 experiment_id 用不含趋势步的变体。
        return "portfolio_analysis" if self.experiment_id else "portfolio_analysis_no_trend"

    def template_slots(self, states: list[str]) -> dict:
        slots = {
            "performance_dataset_id": self.dataset_id,
            "id_col": self.id_col,
            "snapshot_col": self.snapshot_col,
            "bucket_col": self.bucket_col,
            "states": list(states),
        }
        if self.balance_col:
            slots["balance_col"] = self.balance_col
        if self.segment_col:
            slots["segment_col"] = self.segment_col
        if self.score_col:
            slots["score_col"] = self.score_col
        if self.experiment_id:
            slots["experiment_id"] = self.experiment_id
        if self.project_meta:
            slots["project_meta"] = self.project_meta
        return slots


def build_portfolio_proposal(
    registry,
    backend,
    task_id: str,
    source_dir,
    *,
    segment_col: str | None = None,
    score_col: str | None = None,
    experiment_id: str | None = None,
) -> PortfolioProposal:
    dataset = _resolve_performance_dataset(registry, task_id, source_dir)
    path = registry.resolve_path(dataset.id)
    columns = backend.column_names(path)

    id_col = _resolve_named_col(columns, None, _ID_HINTS)
    snapshot_col = _resolve_named_col(columns, None, _SNAPSHOT_HINTS)
    bucket_col = _resolve_named_col(columns, None, _BUCKET_HINTS)
    if not id_col or not snapshot_col or not bucket_col:
        raise PortfolioSetupError(
            "未能识别表现期快照的 贷款id/快照月/逾期桶 列；请确认数据含相应字段"
            f"（已识别：id=`{id_col}` snapshot=`{snapshot_col}` bucket=`{bucket_col}`）。"
        )
    balance_col = _resolve_named_col(columns, None, _BALANCE_HINTS) or None
    resolved_segment = segment_col if segment_col and segment_col in columns else _resolve_named_col(columns, None, _SEGMENT_HINTS) or None

    frame = backend.read_frame(path, columns=[bucket_col])
    observed = [str(value) for value in frame[bucket_col].dropna().unique().tolist()]
    if not observed:
        raise PortfolioSetupError(f"逾期桶列 `{bucket_col}` 无有效取值。")
    proposed_states = _order_states(observed)

    return PortfolioProposal(
        dataset_id=dataset.id,
        dataset_name=_dataset_name(dataset),
        id_col=id_col,
        snapshot_col=snapshot_col,
        bucket_col=bucket_col,
        proposed_states=proposed_states,
        balance_col=balance_col,
        segment_col=resolved_segment,
        score_col=(score_col or None),
        experiment_id=(experiment_id or None),
    )


def build_states_gate_state(proposal: PortfolioProposal) -> dict:
    """Gate metadata payload: the proposed state order the user must confirm."""
    return {
        "dataset_id": proposal.dataset_id,
        "bucket_col": proposal.bucket_col,
        "proposed_states": list(proposal.proposed_states),
        "id_col": proposal.id_col,
        "snapshot_col": proposal.snapshot_col,
        "balance_col": proposal.balance_col,
        "segment_col": proposal.segment_col,
        "score_col": proposal.score_col,
        "experiment_id": proposal.experiment_id,
    }


def parse_states_reply(user_text: str | None, gate_state: dict) -> list[str] | None:
    """Interpret the user's confirmation reply into the final ordered states.

    - bare 确认/confirm -> accept the proposed order verbatim;
    - a comma/、/空格 separated re-ordering that is a permutation of the proposed
      states -> that order;
    - anything else -> None (caller re-prompts, never guesses the order).
    """
    from marvis.agent.plan_driver import is_confirm

    proposed = [str(state) for state in gate_state.get("proposed_states") or []]
    text = (user_text or "").strip()
    if not text:
        return None
    if is_confirm(text):
        return proposed
    tokens = [token.strip() for token in text.replace("，", ",").replace("、", ",").replace(" ", ",").split(",")]
    tokens = [token for token in tokens if token]
    if tokens and set(tokens) == set(proposed) and len(tokens) == len(proposed):
        return tokens
    return None


def _order_states(observed: list[str]) -> list[str]:
    def rank(state: str) -> tuple[int, int]:
        low = state.lower()
        for marker, value in _STATE_RANK_MARKERS:
            if marker in low:
                return (value, observed.index(state))
        return (50, observed.index(state))  # unknown -> middle, keep input order

    return sorted(observed, key=rank)


def _resolve_performance_dataset(registry, task_id: str, source_dir):
    datasets = [d for d in registry.list_for_task(task_id) if d.role in _PERFORMANCE_ROLES]
    if not datasets:
        # fall back to any sample-like table registered under the task, and to
        # scanning the source dir (registered as sample) -- column-hint detection
        # later decides whether it's actually a performance frame.
        datasets = [d for d in registry.list_for_task(task_id) if d.role in _SAMPLE_ROLES]
    if not datasets and source_dir is not None:
        for artifact in scan_source_dir(Path(source_dir)):
            if artifact.role == FileRole.SAMPLE:
                registry.register_from_upload(task_id, Path(artifact.path), role="sample")
        datasets = [
            d
            for d in registry.list_for_task(task_id)
            if d.role in (_PERFORMANCE_ROLES | _SAMPLE_ROLES)
        ]
    if not datasets:
        raise PortfolioSetupError(f"组合分析未找到表现期数据文件:{source_dir}")
    return sorted(datasets, key=lambda d: -int(getattr(d, "row_count", 0) or 0))[0]


def _resolve_named_col(columns: list[str], requested: str | None, hints: tuple[str, ...]) -> str:
    requested = str(requested or "").strip()
    if requested and requested in columns:
        return requested
    lowered = {column.lower(): column for column in columns}
    for hint in hints:
        if hint in lowered:
            return lowered[hint]
    for column in columns:
        low = column.lower()
        if any(hint in low for hint in hints):
            return column
    return ""


def _dataset_name(dataset) -> str:
    source = getattr(dataset, "source_path", None)
    return Path(source).name if source else str(getattr(dataset, "id", ""))


__all__ = [
    "PortfolioProposal",
    "PortfolioSetupError",
    "build_portfolio_proposal",
    "build_states_gate_state",
    "parse_states_reply",
]
