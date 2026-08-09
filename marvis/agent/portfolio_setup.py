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
import hmac
import math
from pathlib import Path
import re

from marvis.data.authenticated_snapshot import (
    AuthenticatedSnapshotError,
    read_authenticated_parquet_snapshot,
)
from marvis.domain import FileRole
from marvis.files import scan_source_dir

_PERFORMANCE_ROLES = frozenset({"performance"})
_SAMPLE_ROLES = frozenset({FileRole.SAMPLE.value, "sample"})
_ID_HINTS = ("loan_id", "loanid", "id", "account_id", "acct_id", "contract_id", "cust_id")
_SNAPSHOT_HINTS = ("snapshot_month", "snapshot", "obs_month", "observe_month", "month", "stat_month", "dt")
_BUCKET_HINTS = ("bucket", "delinq", "dpd_bucket", "overdue_bucket", "status", "stage", "state")
_BALANCE_HINTS = ("balance", "bal", "principal", "outstanding", "ead", "exposure")
_SEGMENT_HINTS = ("segment", "product", "channel", "grade", "region", "seg")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

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
    dataset_content_hash: str
    dataset_name: str
    id_col: str
    snapshot_col: str
    bucket_col: str
    proposed_states: list[str]
    balance_col: str
    segment_col: str
    loss_state: str
    lgd: float
    horizon_months: int
    score_col: str | None = None
    experiment_id: str | None = None
    project_meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(str(self.dataset_content_hash or "")) is None:
            raise PortfolioSetupError(
                "组合分析表现期数据缺少有效的 SHA-256 内容绑定。"
            )
        missing = []
        if not str(self.balance_col or "").strip():
            missing.append("余额/EAD")
        if not str(self.segment_col or "").strip():
            missing.append("业务分群")
        if missing:
            raise PortfolioSetupError(
                "组合分析完整报告缺少必要列："
                + "、".join(missing)
                + "。首版不会用笔数冒充金额，也不会静默省略分群分析。"
            )
        loss_state = str(self.loss_state or "").strip()
        if not loss_state:
            raise PortfolioSetupError("组合分析缺少明确的损失态；不会静默猜测损失事件。")
        if loss_state not in self.proposed_states:
            raise PortfolioSetupError(
                f"损失态 `{loss_state}` 不在逾期桶状态中：{self.proposed_states}。"
            )
        if isinstance(self.lgd, bool):
            raise PortfolioSetupError("LGD 必须是 0 到 1 之间的有限数值。")
        try:
            lgd = float(self.lgd)
        except (TypeError, ValueError) as exc:
            raise PortfolioSetupError("LGD 必须是 0 到 1 之间的有限数值。") from exc
        if not math.isfinite(lgd) or not 0.0 <= lgd <= 1.0:
            raise PortfolioSetupError("LGD 必须是 0 到 1 之间的有限数值。")
        if (
            isinstance(self.horizon_months, bool)
            or not isinstance(self.horizon_months, int)
            or self.horizon_months <= 0
        ):
            raise PortfolioSetupError("预测期限 horizon_months 必须是正整数月数。")
        self.loss_state = loss_state
        self.lgd = lgd

    @property
    def template_id(self) -> str:
        # 剪步语义：无 experiment_id 用不含趋势步的变体。
        return "portfolio_analysis" if self.experiment_id else "portfolio_analysis_no_trend"

    def template_slots(self, states: list[str]) -> dict:
        if self.loss_state not in states:
            raise PortfolioSetupError(
                f"损失态 `{self.loss_state}` 不在已确认的逾期桶状态中：{states}。"
            )
        slots = {
            "performance_dataset_id": self.dataset_id,
            "performance_dataset_content_hash": self.dataset_content_hash,
            "id_col": self.id_col,
            "snapshot_col": self.snapshot_col,
            "bucket_col": self.bucket_col,
            "states": list(states),
            "loss_state": self.loss_state,
            "lgd": self.lgd,
            "horizon_months": self.horizon_months,
        }
        slots["balance_col"] = self.balance_col
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
    loss_state: str | None,
    lgd: float | None,
    horizon_months: int | None,
    id_col: str | None = None,
    snapshot_col: str | None = None,
    bucket_col: str | None = None,
    balance_col: str | None = None,
    segment_col: str | None = None,
    score_col: str | None = None,
    experiment_id: str | None = None,
) -> PortfolioProposal:
    dataset = _resolve_performance_dataset(registry, task_id, source_dir)
    dataset_content_hash = str(getattr(dataset, "content_hash", "") or "")
    path = verify_portfolio_dataset_binding(
        registry,
        task_id=task_id,
        dataset_id=dataset.id,
        expected_content_hash=dataset_content_hash,
        dataset=dataset,
    )
    snapshot_reader = getattr(
        registry,
        "read_authenticated_parquet_snapshot",
        None,
    )
    authenticated_frame = None
    if callable(snapshot_reader):
        try:
            # Infer both the schema and the observed states from one private,
            # hash-authenticated frame.  A later path-based schema probe would
            # reopen mutable bytes and recreate a TOCTOU window.
            authenticated_frame = snapshot_reader(dataset.id)
        except (
            AuthenticatedSnapshotError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise PortfolioSetupError(
                "组合分析表现期数据快照未通过不可变内容校验。"
            ) from exc
        columns = [str(column) for column in authenticated_frame.columns]
    else:
        columns = backend.column_names(path)

    requested_columns = {
        "贷款id": id_col,
        "快照月": snapshot_col,
        "逾期桶": bucket_col,
        "余额/EAD": balance_col,
        "业务分群": segment_col,
        "模型分数": score_col,
    }
    for label, requested in requested_columns.items():
        if requested and requested not in columns:
            raise PortfolioSetupError(
                f"组合分析指定的{label}列 `{requested}` 不在表现期数据中。"
            )

    resolved_id_col = _resolve_named_col(columns, id_col, _ID_HINTS)
    resolved_snapshot_col = _resolve_named_col(
        columns, snapshot_col, _SNAPSHOT_HINTS
    )
    resolved_bucket_col = _resolve_named_col(columns, bucket_col, _BUCKET_HINTS)
    if not resolved_id_col or not resolved_snapshot_col or not resolved_bucket_col:
        raise PortfolioSetupError(
            "未能识别表现期快照的 贷款id/快照月/逾期桶 列；请确认数据含相应字段"
            "（已识别："
            f"id=`{resolved_id_col}` snapshot=`{resolved_snapshot_col}` "
            f"bucket=`{resolved_bucket_col}`）。"
        )
    resolved_balance_col = (
        _resolve_named_col(columns, balance_col, _BALANCE_HINTS) or None
    )
    resolved_segment_col = (
        _resolve_named_col(columns, segment_col, _SEGMENT_HINTS) or None
    )

    if authenticated_frame is not None:
        frame = authenticated_frame
    elif callable(getattr(registry, "resolve_verified_path", None)):
        try:
            frame = read_authenticated_parquet_snapshot(
                path,
                root=path.parent,
                expected_sha256=dataset_content_hash,
                columns=[resolved_bucket_col],
            )
        except AuthenticatedSnapshotError as exc:
            raise PortfolioSetupError(
                "组合分析表现期数据快照未通过不可变内容校验。"
            ) from exc
    else:
        frame = backend.read_frame(path, columns=[resolved_bucket_col])
    observed = [
        str(value)
        for value in frame[resolved_bucket_col].dropna().unique().tolist()
    ]
    if not observed:
        raise PortfolioSetupError(f"逾期桶列 `{resolved_bucket_col}` 无有效取值。")
    proposed_states = _order_states(observed)

    return PortfolioProposal(
        dataset_id=dataset.id,
        dataset_content_hash=dataset_content_hash,
        dataset_name=_dataset_name(dataset),
        id_col=resolved_id_col,
        snapshot_col=resolved_snapshot_col,
        bucket_col=resolved_bucket_col,
        proposed_states=proposed_states,
        balance_col=resolved_balance_col,
        segment_col=resolved_segment_col,
        loss_state=loss_state,
        lgd=lgd,
        horizon_months=horizon_months,
        score_col=(score_col or None),
        experiment_id=(experiment_id or None),
    )


def build_states_gate_state(proposal: PortfolioProposal) -> dict:
    """Gate metadata payload: the proposed state order the user must confirm."""
    return {
        "dataset_id": proposal.dataset_id,
        "dataset_content_hash": proposal.dataset_content_hash,
        "bucket_col": proposal.bucket_col,
        "proposed_states": list(proposal.proposed_states),
        "id_col": proposal.id_col,
        "snapshot_col": proposal.snapshot_col,
        "balance_col": proposal.balance_col,
        "segment_col": proposal.segment_col,
        "loss_state": proposal.loss_state,
        "lgd": proposal.lgd,
        "horizon_months": proposal.horizon_months,
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


def verify_portfolio_dataset_binding(
    registry,
    *,
    task_id: str,
    dataset_id: str,
    expected_content_hash: str,
    dataset=None,
) -> Path:
    """Re-authenticate the exact dataset identity behind a human gate."""

    try:
        bound_dataset = dataset if dataset is not None else registry.get(dataset_id)
    except KeyError as exc:
        raise PortfolioSetupError(
            "组合分析表现期数据在确认前已不存在，请重新配置。"
        ) from exc
    owner = str(getattr(bound_dataset, "task_id", task_id))
    if owner != str(task_id) or str(bound_dataset.id) != str(dataset_id):
        raise PortfolioSetupError("组合分析表现期数据不属于当前任务。")
    registered_hash = str(getattr(bound_dataset, "content_hash", "") or "")
    if (
        _SHA256_RE.fullmatch(str(expected_content_hash or "")) is None
        or _SHA256_RE.fullmatch(registered_hash) is None
        or not hmac.compare_digest(registered_hash, expected_content_hash)
    ):
        raise PortfolioSetupError(
            "组合分析表现期数据内容绑定在确认前发生变化，请重新配置。"
        )
    resolver = getattr(registry, "resolve_verified_path", None)
    try:
        if callable(resolver):
            return Path(resolver(dataset_id))
        return Path(registry.resolve_path(dataset_id))
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise PortfolioSetupError(
            "组合分析表现期数据内容在确认前发生变化，请重新配置。"
        ) from exc


def _order_states(observed: list[str]) -> list[str]:
    def rank(state: str) -> tuple[int, int]:
        low = state.lower()
        for marker, value in _STATE_RANK_MARKERS:
            if (low == marker) if marker == "c" else (marker in low):
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
    "verify_portfolio_dataset_binding",
]
