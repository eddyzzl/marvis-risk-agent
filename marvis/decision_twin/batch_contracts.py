"""Explicit historical contracts; v1's complete observed scenarios stay strict."""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from marvis.decision_twin._canonical import content_hash, parse_datetime
from marvis.decision_twin.event_batch_contracts import HistoricalEventMapping


class StrictContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, str_strip_whitespace=True
    )


class HistoricalFeature(StrictContract):
    name: str = Field(min_length=1, max_length=160)
    value_col: str = Field(min_length=1, max_length=160)
    available_at_col: str = Field(min_length=1, max_length=160)
    # Event time alone does not prove availability. Both must precede decision.
    event_at_col: str = Field(min_length=1, max_length=160)


class HistoricalScenario(StrictContract):
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["baseline", "challenger", "counterfactual"]
    package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class HistoricalEconomics(StrictContract):
    """Fixed-condition estimates, never inferred actual cash flows or causal gains."""

    currency: str = Field(pattern=r"^[A-Z]{3}$")
    ead_col: str = Field(min_length=1, max_length=160)
    available_at_col: str = Field(min_length=1, max_length=160)
    annual_rate: float = Field(ge=0, le=1)
    funding_rate: float = Field(ge=0, le=1)
    lgd: float = Field(ge=0, le=1)
    term_months: float = Field(gt=0)
    operating_cost_per_loan: float = Field(ge=0)
    assumption_sources: dict[str, str]

    @model_validator(mode="after")
    def sources(self):
        required = {
            "ead",
            "pd",
            "annual_rate",
            "funding_rate",
            "lgd",
            "term_months",
            "operating_cost_per_loan",
        }
        if set(self.assumption_sources) != required or any(
            not value.strip() for value in self.assumption_sources.values()
        ):
            raise ValueError("each economic assumption requires an explicit source")
        return self


class HistoricalGroup(StrictContract):
    column: str = Field(min_length=1, max_length=160)
    available_at_col: str = Field(min_length=1, max_length=160)
    governance_ref: str = Field(min_length=1, max_length=500)
    minimum_group_size: int = Field(ge=1)


class HistoricalCapacity(StrictContract):
    unit: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    per_action: dict[Literal["approval", "reject", "review"], float]
    source_ref: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def complete(self):
        if set(self.per_action) != {"approval", "reject", "review"} or any(
            value < 0 for value in self.per_action.values()
        ):
            raise ValueError("capacity requires nonnegative estimates for all actions")
        return self


class HistoricalActions(StrictContract):
    action_col: str = Field(min_length=1, max_length=160)
    recorded_at_col: str = Field(min_length=1, max_length=160)
    source_ref: str = Field(min_length=1, max_length=500)


class HistoricalConstraint(StrictContract):
    metric: Literal[
        "approval_rate",
        "ead",
        "expected_loss",
        "profit",
        "protected_group_disparity",
        "operations_capacity",
    ]
    operator: Literal[">=", "<="]
    threshold: float
    unit: str = Field(min_length=1, max_length=80)


class HistoricalTimeWindow(StrictContract):
    name: str = Field(min_length=1, max_length=80)
    start: str
    end: str

    @model_validator(mode="after")
    def ordered(self):
        if parse_datetime(self.start, "window.start") >= parse_datetime(
            self.end, "window.end"
        ):
            raise ValueError("temporal window must have start < end")
        return self


class TemporalThresholds(StrictContract):
    max_score_psi: float = Field(ge=0)
    max_action_psi: float = Field(ge=0)
    max_absolute_approval_rate_delta: float = Field(ge=0, le=1)


class HistoricalTemporalStability(StrictContract):
    """User-declared, half-open windows for descriptive temporal drift only."""

    timezone: str = Field(min_length=1, max_length=80)
    reference_window: HistoricalTimeWindow
    comparison_windows: list[HistoricalTimeWindow] = Field(min_length=1, max_length=24)
    minimum_reference_rows: int = Field(ge=2)
    minimum_comparison_rows: int = Field(ge=2)
    bin_count: int = Field(ge=2, le=20)
    thresholds: TemporalThresholds
    policy_source_ref: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def unambiguous(self):
        try:
            zone = ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("temporal timezone must be a known IANA timezone") from exc
        windows = [self.reference_window, *self.comparison_windows]
        if len({w.name for w in windows}) != len(windows):
            raise ValueError("temporal window names must be unique")
        intervals = []
        for window in windows:
            bounds = []
            for value in (window.start, window.end):
                at = parse_datetime(value, "window boundary")
                supplied = datetime.fromisoformat(value.replace("Z", "+00:00"))
                local = at.astimezone(zone)
                if supplied.utcoffset() != local.utcoffset() or supplied.replace(
                    tzinfo=None
                ) != local.replace(tzinfo=None):
                    raise ValueError(
                        "temporal boundary offset must match the declared timezone"
                    )
                bounds.append(at)
            intervals.append(tuple(bounds))
        reference_end = intervals[0][1]
        if any(start < reference_end for start, _ in intervals[1:]):
            raise ValueError(
                "comparison windows must follow the complete reference window"
            )
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("temporal windows must not overlap")
        return self


class HistoricalReplayRequest(StrictContract):
    schema_version: Literal["decision_twin.batch_request.v2"] = (
        "decision_twin.batch_request.v2"
    )
    dataset_id: str = Field(min_length=1, max_length=160)
    expected_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_id_col: str = Field(min_length=1, max_length=160)
    decision_at_col: str = Field(min_length=1, max_length=160)
    as_of: str
    source_ref: str = Field(min_length=1, max_length=500)
    population: str = Field(min_length=1, max_length=500)
    features: list[HistoricalFeature] = Field(max_length=500)
    event_mappings: list[HistoricalEventMapping] = Field(
        default_factory=list, max_length=3
    )
    scenarios: list[HistoricalScenario] = Field(min_length=2, max_length=3)
    economics: HistoricalEconomics | None = None
    protected_group: HistoricalGroup | None = None
    capacity: HistoricalCapacity | None = None
    observed_actions: HistoricalActions | None = None
    constraints: list[HistoricalConstraint] = Field(default_factory=list, max_length=20)
    temporal_stability: HistoricalTemporalStability | None = None

    @model_validator(mode="after")
    def unique(self):
        parse_datetime(self.as_of, "as_of")
        if len({f.name for f in self.features}) != len(self.features):
            raise ValueError("feature names must be unique")
        kinds = [s.kind for s in self.scenarios]
        if len(set(kinds)) != len(kinds) or not {"baseline", "challenger"} <= set(
            kinds
        ):
            raise ValueError("exactly one baseline and challenger are required")
        if len({s.name for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("scenario names must be unique")
        mapped = [m.scenario_kind for m in self.event_mappings]
        if len(set(mapped)) != len(mapped) or not set(mapped) <= set(kinds):
            raise ValueError("event mappings must name unique declared scenarios")
        if self.temporal_stability:
            as_of = parse_datetime(self.as_of, "as_of")
            windows = [
                self.temporal_stability.reference_window,
                *self.temporal_stability.comparison_windows,
            ]
            if any(parse_datetime(w.end, "window.end") > as_of for w in windows):
                raise ValueError(
                    "temporal window extends after the historical as_of cutoff"
                )
        return self

    @model_serializer(mode="wrap")
    def historical_wire(self, handler):
        payload = handler(self)
        # Preserve every byte-bearing field of previously published v2 contracts.
        # Absence and explicit null add no temporal assertion or receipt identity.
        if self.temporal_stability is None:
            payload.pop("temporal_stability", None)
        if not self.event_mappings:
            payload.pop("event_mappings", None)
        return payload

    @property
    def contract_hash(self):
        return content_hash(self.model_dump())


class HistoricalReconciliationRequest(StrictContract):
    replay_artifact_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str = Field(min_length=1, max_length=160)
    expected_content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_id_col: str = Field(min_length=1, max_length=160)
    observed_at_col: str = Field(min_length=1, max_length=160)
    actual_loss_col: str = Field(min_length=1, max_length=160)
    actual_profit_col: str = Field(min_length=1, max_length=160)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    maturity_days: int = Field(ge=1, le=36500)
    maturity_source_ref: str = Field(min_length=1, max_length=500)
    source_ref: str = Field(min_length=1, max_length=500)
    reconciled_at: str

    @model_validator(mode="after")
    def timestamp(self):
        parse_datetime(self.reconciled_at, "reconciled_at")
        return self

    @property
    def contract_hash(self):
        return content_hash(self.model_dump())
