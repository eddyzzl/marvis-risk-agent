"""Fail-closed business contract and pre-plan proposal for label construction.

The contract deliberately does not infer label semantics from column names.  A
caller must bind the source snapshot, cutoff date, output column, observation /
performance windows, and exactly one explicit DPD or ordered-status rule before
a plan can be created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from marvis.data.errors import DatasetContentDriftError
from marvis.data.label_construction import check_cohort_maturity


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RULE_KINDS = frozenset({"dpd", "status"})


class LabelingContractError(ValueError):
    """The explicit label business contract is absent, stale, or inconsistent."""


@dataclass(frozen=True)
class LabelingRequest:
    dataset_id: str
    expected_content_hash: str
    workspace_revision: int
    analysis_generation: int
    id_col: str
    mob_col: str
    cohort_col: str
    date_col: str
    as_of_date: str
    target_col: str
    observation_window: int
    performance_window: int
    at_mob: int
    rule_kind: str
    dpd_col: str | None = None
    threshold_dpd: float | None = None
    status_col: str | None = None
    threshold_status: str | None = None
    states: tuple[str, ...] | list[str] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "id_col",
            "mob_col",
            "cohort_col",
            "date_col",
            "as_of_date",
            "target_col",
            "rule_kind",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name),
            )
        content_hash = str(self.expected_content_hash or "").strip().lower()
        if _SHA256_RE.fullmatch(content_hash) is None:
            raise LabelingContractError(
                "expected_content_hash must be an explicit lowercase SHA-256"
            )
        object.__setattr__(self, "expected_content_hash", content_hash)
        object.__setattr__(
            self,
            "workspace_revision",
            _non_negative_int(self.workspace_revision, "workspace_revision"),
        )
        object.__setattr__(
            self,
            "analysis_generation",
            _non_negative_int(self.analysis_generation, "analysis_generation"),
        )
        observation_window = _non_negative_int(
            self.observation_window,
            "observation_window",
        )
        performance_window = _positive_int(
            self.performance_window,
            "performance_window",
        )
        at_mob = _positive_int(self.at_mob, "at_mob")
        if at_mob <= observation_window:
            raise LabelingContractError("at_mob must be greater than observation_window")
        if at_mob > observation_window + performance_window:
            raise LabelingContractError(
                "at_mob cannot exceed observation_window + performance_window"
            )
        object.__setattr__(self, "observation_window", observation_window)
        object.__setattr__(self, "performance_window", performance_window)
        object.__setattr__(self, "at_mob", at_mob)
        try:
            normalized_as_of = date.fromisoformat(self.as_of_date).isoformat()
        except ValueError as exc:
            raise LabelingContractError("as_of_date must be an ISO date (YYYY-MM-DD)") from exc
        object.__setattr__(self, "as_of_date", normalized_as_of)

        rule_kind = self.rule_kind.lower()
        if rule_kind not in _RULE_KINDS:
            raise LabelingContractError("rule_kind must be explicitly dpd or status")
        object.__setattr__(self, "rule_kind", rule_kind)
        if rule_kind == "dpd":
            dpd_col = _required_text(self.dpd_col, "dpd_col")
            threshold = _finite_non_negative_number(
                self.threshold_dpd,
                "threshold_dpd",
            )
            if any(
                value not in (None, "", (), [])
                for value in (self.status_col, self.threshold_status, self.states)
            ):
                raise LabelingContractError(
                    "status_col/threshold_status/states must be absent for rule_kind=dpd"
                )
            object.__setattr__(self, "dpd_col", dpd_col)
            object.__setattr__(self, "threshold_dpd", threshold)
            object.__setattr__(self, "status_col", None)
            object.__setattr__(self, "threshold_status", None)
            object.__setattr__(self, "states", None)
        else:
            status_col = _required_text(self.status_col, "status_col")
            threshold_status = _required_text(
                self.threshold_status,
                "threshold_status",
            )
            if self.dpd_col not in (None, "") or self.threshold_dpd is not None:
                raise LabelingContractError(
                    "dpd_col/threshold_dpd must be absent for rule_kind=status"
                )
            states = tuple(
                _required_text(value, "states") for value in (self.states or ())
            )
            if len(states) < 2:
                raise LabelingContractError(
                    "states must explicitly contain at least two buckets in good-to-bad order"
                )
            if len(set(states)) != len(states):
                raise LabelingContractError("states must not contain duplicate buckets")
            if threshold_status not in states:
                raise LabelingContractError(
                    "threshold_status must be present in the explicit states order"
                )
            object.__setattr__(self, "dpd_col", None)
            object.__setattr__(self, "threshold_dpd", None)
            object.__setattr__(self, "status_col", status_col)
            object.__setattr__(self, "threshold_status", threshold_status)
            object.__setattr__(self, "states", states)

    @property
    def value_col(self) -> str:
        return str(self.dpd_col if self.rule_kind == "dpd" else self.status_col)

    @property
    def rule_summary(self) -> str:
        if self.rule_kind == "dpd":
            return f"{self.dpd_col} >= {float(self.threshold_dpd):g}"
        states = tuple(self.states or ())
        threshold_index = states.index(str(self.threshold_status))
        return f"{self.status_col} in [{', '.join(states[threshold_index:])}]"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.states is not None:
            payload["states"] = list(self.states)
        return payload

    @property
    def contract_hash(self) -> str:
        return _canonical_hash(self.to_dict())


@dataclass(frozen=True)
class LabelingProposal:
    request: LabelingRequest
    source_dataset_id: str
    source_dataset_name: str
    source_content_hash: str
    source_row_count: int
    rows_at_as_of: int
    rows_excluded_after_as_of: int
    maturity: dict[str, Any]
    requires_human_confirmation: bool = True

    @property
    def contract_hash(self) -> str:
        return self.request.contract_hash

    @property
    def as_of_date(self) -> str:
        return self.request.as_of_date

    @property
    def rule_summary(self) -> str:
        return self.request.rule_summary

    def to_template_slots(
        self,
        *,
        confirm_immature_cohorts: bool = False,
    ) -> dict[str, Any]:
        return {
            **self.request.to_dict(),
            "proposal_hash": self.contract_hash,
            "confirm_immature_cohorts": bool(confirm_immature_cohorts),
        }

    def to_gate_state(self) -> dict[str, Any]:
        return {
            "schema_version": "labeling-proposal.v1",
            "proposal_hash": self.contract_hash,
            "request": self.request.to_dict(),
            "source_dataset_name": self.source_dataset_name,
            "source_row_count": self.source_row_count,
            "rows_at_as_of": self.rows_at_as_of,
            "rows_excluded_after_as_of": self.rows_excluded_after_as_of,
            "rule_summary": self.rule_summary,
            "maturity": self.maturity,
            "requires_human_confirmation": True,
        }


@dataclass(frozen=True)
class AuthenticatedLabelingSource:
    """One content-addressed source frame shared by proposal and calculation."""

    proposal: LabelingProposal
    frame: pd.DataFrame
    source_path: Path
    registered_source_path: str


def build_labeling_proposal(
    registry,
    backend,
    workspace,
    *,
    task_id: str,
    request: LabelingRequest,
) -> LabelingProposal:
    """Validate a source-bound request and calculate read-only maturity evidence."""

    return build_authenticated_labeling_source(
        registry,
        backend,
        workspace,
        task_id=task_id,
        request=request,
    ).proposal


def build_authenticated_labeling_source(
    registry,
    backend,
    workspace,
    *,
    task_id: str,
    request: LabelingRequest,
) -> AuthenticatedLabelingSource:
    """Bind proposal evidence and downstream labels to one authenticated frame.

    ``backend`` remains in the signature for compatibility with the established
    proposal service boundary.  Reads intentionally bypass it: resolving and
    later reopening a pathname leaves a file-swap window, whereas the registry's
    authenticated reader hashes and parses one retained-descriptor snapshot.
    """

    del backend

    if not isinstance(request, LabelingRequest):
        raise LabelingContractError("request must be a LabelingRequest")
    try:
        dataset = registry.get(request.dataset_id)
    except KeyError as exc:
        raise LabelingContractError(
            f"dataset_id is not available: {request.dataset_id}"
        ) from exc
    if str(getattr(dataset, "task_id", "")) != str(task_id):
        raise LabelingContractError("dataset_id is not owned by this task")
    registered_hash = str(getattr(dataset, "content_hash", "") or "").lower()
    if not hmac.compare_digest(registered_hash, request.expected_content_hash):
        raise LabelingContractError("expected_content_hash does not match the dataset")
    _require_workspace_binding(workspace, request)

    columns = [
        str(getattr(column, "name", "") or "").strip()
        for column in tuple(getattr(dataset, "columns", ()) or ())
    ]
    if not columns or any(not column for column in columns):
        raise LabelingContractError(
            "labeling source registered schema is unavailable or invalid"
        )
    required_columns = [
        request.id_col,
        request.mob_col,
        request.cohort_col,
        request.date_col,
        request.value_col,
    ]
    missing = [column for column in dict.fromkeys(required_columns) if column not in columns]
    if missing:
        raise LabelingContractError(
            "labeling source is missing explicit column(s): " + ", ".join(missing)
        )
    if request.target_col in columns:
        raise LabelingContractError(
            f"target_col `{request.target_col}` already exists; label construction will not overwrite it"
        )
    selected_columns = list(dict.fromkeys(required_columns))
    try:
        path = Path(registry.resolve_verified_path(dataset.id))
        frame = registry.read_authenticated_parquet_snapshot(
            dataset.id,
            columns=selected_columns,
        )
    except DatasetContentDriftError as exc:
        raise LabelingContractError(
            f"labeling source authenticated snapshot failed: {exc.reason}"
        ) from exc
    cutoff_frame, excluded = frame_at_as_of(
        frame,
        date_col=request.date_col,
        as_of_date=request.as_of_date,
    )
    report = check_cohort_maturity(
        cutoff_frame,
        id_col=request.id_col,
        mob_col=request.mob_col,
        cohort_col=request.cohort_col,
        required_mob=request.at_mob,
    )
    maturity = {
        "required_mob": report.required_mob,
        "cohorts": [asdict(cohort) for cohort in report.cohorts],
        "immature_cohorts": list(report.immature_cohorts),
        "all_matured": report.all_matured,
    }
    source_row_count = int(len(frame))
    proposal = LabelingProposal(
        request=request,
        source_dataset_id=str(dataset.id),
        source_dataset_name=Path(str(getattr(dataset, "source_path", dataset.id))).name,
        source_content_hash=registered_hash,
        source_row_count=source_row_count,
        rows_at_as_of=int(len(cutoff_frame)),
        rows_excluded_after_as_of=int(excluded),
        maturity=maturity,
    )
    return AuthenticatedLabelingSource(
        proposal=proposal,
        frame=frame,
        source_path=path,
        registered_source_path=str(getattr(dataset, "source_path", "")),
    )


def frame_at_as_of(
    frame: pd.DataFrame,
    *,
    date_col: str,
    as_of_date: str,
) -> tuple[pd.DataFrame, int]:
    """Return rows observed on/before the explicit cutoff, rejecting bad dates."""

    if date_col not in frame.columns:
        raise LabelingContractError(f"date_col `{date_col}` is missing")
    raw = frame[date_col]
    parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    if (raw.notna() & parsed.isna()).any() or raw.isna().any():
        raise LabelingContractError(
            f"date_col `{date_col}` contains null or unparseable dates"
        )
    cutoff = pd.Timestamp(date.fromisoformat(as_of_date), tz="UTC")
    mask = parsed.dt.normalize() <= cutoff
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise LabelingContractError("as_of_date leaves no source rows for labeling")
    return selected, int((~mask).sum())


def _require_workspace_binding(workspace, request: LabelingRequest) -> None:
    expected = {
        "active_dataset_id": request.dataset_id,
        "active_dataset_content_hash": request.expected_content_hash,
        "revision": request.workspace_revision,
        "analysis_generation": request.analysis_generation,
    }
    for field_name, value in expected.items():
        if getattr(workspace, field_name, None) != value:
            raise LabelingContractError(
                f"data workspace {field_name} changed before proposal confirmation"
            )


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LabelingContractError(f"{field_name} is required")
    return text


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelingContractError(f"{field_name} must be a non-negative integer")
    return int(value)


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LabelingContractError(f"{field_name} must be a positive integer")
    return int(value)


def _finite_non_negative_number(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise LabelingContractError(f"{field_name} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelingContractError(
            f"{field_name} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(number) or number < 0:
        raise LabelingContractError(f"{field_name} must be a finite non-negative number")
    return number


def _canonical_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "AuthenticatedLabelingSource",
    "LabelingContractError",
    "LabelingProposal",
    "LabelingRequest",
    "build_authenticated_labeling_source",
    "build_labeling_proposal",
    "frame_at_as_of",
]
