from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Any

from marvis.decision_twin._canonical import content_hash, finite_number, required_text
from marvis.decision_twin.replay import ReplayedDecision


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UNIT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}")


class ScenarioKind(StrEnum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    COUNTERFACTUAL = "counterfactual"


@dataclass(frozen=True)
class PopulationDenominator:
    name: str
    count: int
    unit: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", required_text(self.name, "denominator name"))
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("denominator count must be a positive integer")
        unit = required_text(self.unit, "denominator unit", max_length=80)
        if not _UNIT_RE.fullmatch(unit):
            raise ValueError("denominator unit must be an identifier")
        object.__setattr__(self, "unit", unit)


@dataclass(frozen=True)
class CounterfactualAssessment:
    """Causal boundary for an outcome that was not observed under treatment."""

    estimand: str
    identifiability: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimand", required_text(self.estimand, "estimand"))
        if self.identifiability not in {"unidentified", "bounded"}:
            raise ValueError(
                "unobserved counterfactual identifiability must be unidentified or bounded"
            )
        object.__setattr__(self, "unit", required_text(self.unit, "unit"))
        if self.identifiability == "unidentified":
            if self.lower_bound is not None or self.upper_bound is not None:
                raise ValueError(
                    "unidentified counterfactual cannot carry numeric bounds"
                )
            return
        if self.lower_bound is None or self.upper_bound is None:
            raise ValueError("bounded counterfactual requires lower and upper bounds")
        lower = finite_number(self.lower_bound, "lower_bound")
        upper = finite_number(self.upper_bound, "upper_bound")
        if lower > upper:
            raise ValueError("counterfactual lower bound must not exceed upper bound")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "identifiability": self.identifiability,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "unit": self.unit,
            "causal_outcome_observed": False,
        }


@dataclass(frozen=True)
class ScenarioReplay:
    name: str
    kind: ScenarioKind
    decisions: tuple[ReplayedDecision, ...]
    denominator: PopulationDenominator
    observed_execution: bool = False
    counterfactual: CounterfactualAssessment | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", required_text(self.name, "scenario name"))
        if not isinstance(self.kind, ScenarioKind):
            raise ValueError("kind must be a ScenarioKind")
        decisions = tuple(self.decisions)
        if not decisions or not all(
            isinstance(decision, ReplayedDecision) for decision in decisions
        ):
            raise ValueError("decisions must contain ReplayedDecision records")
        record_ids = tuple(decision.record_id for decision in decisions)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("scenario record ids must be unique")
        if not isinstance(self.denominator, PopulationDenominator):
            raise ValueError("denominator must be PopulationDenominator")
        if self.denominator.count != len(decisions):
            raise ValueError("denominator count must equal the replayed population")
        if len({decision.currency for decision in decisions}) != 1:
            raise ValueError("currency unit drift within scenario")
        if len({decision.operations_unit for decision in decisions}) != 1:
            raise ValueError("operations capacity unit drift within scenario")
        if len({decision.lineage.manifest_hash for decision in decisions}) != 1:
            raise ValueError("manifest hash drift within scenario")
        if len({decision.lineage.bindings for decision in decisions}) != 1:
            raise ValueError("version binding lineage drift within scenario")
        if len({decision.lineage.adapter for decision in decisions}) != 1:
            raise ValueError("adapter lineage drift within scenario")
        if len({decision.protected_group.attribute for decision in decisions}) != 1:
            raise ValueError("protected attribute drift within scenario")
        for decision in decisions:
            if decision.lineage.record_id != decision.record_id:
                raise ValueError("decision lineage record binding drifted")
            if (
                decision.lineage.protected_group_source_sha256
                != decision.protected_group.source_sha256
            ):
                raise ValueError("protected group lineage binding drifted")
        if not isinstance(self.observed_execution, bool):
            raise ValueError("observed_execution must be a boolean")
        if self.kind is ScenarioKind.CHAMPION and not self.observed_execution:
            raise ValueError("champion must represent the observed execution")
        if self.kind is not ScenarioKind.CHAMPION and self.observed_execution:
            raise ValueError("only champion may represent the observed execution")
        if self.kind is ScenarioKind.COUNTERFACTUAL:
            if not isinstance(self.counterfactual, CounterfactualAssessment):
                raise ValueError(
                    "counterfactual scenario requires a counterfactual assessment"
                )
        elif self.counterfactual is not None:
            raise ValueError("only counterfactual scenario may carry causal assessment")
        object.__setattr__(self, "decisions", decisions)

    @property
    def manifest_hash(self) -> str:
        return self.decisions[0].lineage.manifest_hash

    @property
    def currency(self) -> str:
        return self.decisions[0].currency

    @property
    def operations_unit(self) -> str:
        return self.decisions[0].operations_unit


@dataclass(frozen=True)
class JointConstraintSet:
    currency: str
    operations_unit: str
    min_approval_rate: float
    max_exposure: float
    max_ead: float
    max_projected_loss: float
    min_projected_profit: float
    min_stability: float
    max_protected_group_disparity: float
    max_operations_capacity: float
    min_protected_group_size: int

    def __post_init__(self) -> None:
        currency = required_text(self.currency, "currency", max_length=3)
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("currency must be a three-letter uppercase code")
        object.__setattr__(self, "currency", currency)
        operations_unit = required_text(
            self.operations_unit, "operations_unit", max_length=80
        )
        if not _UNIT_RE.fullmatch(operations_unit):
            raise ValueError("operations_unit must be an identifier")
        object.__setattr__(self, "operations_unit", operations_unit)
        for field in (
            "min_approval_rate",
            "max_exposure",
            "max_ead",
            "max_projected_loss",
            "min_projected_profit",
            "min_stability",
            "max_protected_group_disparity",
            "max_operations_capacity",
        ):
            object.__setattr__(self, field, finite_number(getattr(self, field), field))
        for field in (
            "min_approval_rate",
            "min_stability",
            "max_protected_group_disparity",
        ):
            if not 0 <= getattr(self, field) <= 1:
                raise ValueError(f"{field} must be between 0 and 1")
        for field in (
            "max_exposure",
            "max_ead",
            "max_projected_loss",
            "max_operations_capacity",
        ):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} must be non-negative")
        if (
            isinstance(self.min_protected_group_size, bool)
            or not isinstance(self.min_protected_group_size, int)
            or self.min_protected_group_size < 1
        ):
            raise ValueError("min_protected_group_size must be a positive integer")


@dataclass(frozen=True)
class GroupApprovalRate:
    group: str
    approvals: int
    denominator: int
    rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "approvals": self.approvals,
            "denominator": self.denominator,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class ScenarioMetrics:
    approval_rate: float
    denominator_name: str
    denominator_count: int
    denominator_unit: str
    exposure: float
    ead: float
    projected_loss: float
    projected_profit: float
    currency: str
    stability: float
    protected_group_attribute: str
    protected_group_rates: tuple[GroupApprovalRate, ...]
    protected_group_disparity: float
    operations_capacity: float
    operations_unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_rate": self.approval_rate,
            "denominator": {
                "name": self.denominator_name,
                "count": self.denominator_count,
                "unit": self.denominator_unit,
            },
            "exposure": self.exposure,
            "ead": self.ead,
            "projected_loss": self.projected_loss,
            "projected_profit": self.projected_profit,
            "currency": self.currency,
            "stability": self.stability,
            "protected_group_attribute": self.protected_group_attribute,
            "protected_group_rates": [
                rate.to_dict() for rate in self.protected_group_rates
            ],
            "protected_group_disparity": self.protected_group_disparity,
            "operations_capacity": self.operations_capacity,
            "operations_unit": self.operations_unit,
        }


@dataclass(frozen=True)
class ConstraintResult:
    metric: str
    actual: float
    operator: str
    threshold: float
    unit: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


@dataclass(frozen=True)
class ScenarioEvaluation:
    name: str
    kind: ScenarioKind
    manifest_hash: str
    metrics: ScenarioMetrics
    constraints: tuple[ConstraintResult, ...]
    jointly_satisfied: bool
    counterfactual: CounterfactualAssessment | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "manifest_hash": self.manifest_hash,
            "metrics": self.metrics.to_dict(),
            "constraints": [result.to_dict() for result in self.constraints],
            "jointly_satisfied": self.jointly_satisfied,
            "counterfactual": (
                None if self.counterfactual is None else self.counterfactual.to_dict()
            ),
        }


@dataclass(frozen=True)
class PromotionEvidenceProposal:
    """Evidence binding only; FSP-8 remains the promotion authority."""

    scenario_name: str
    strategy_id: str
    strategy_version: int
    manifest_hash: str
    evidence_ref: str
    evidence_sha256: str
    comparison_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario_name", required_text(self.scenario_name, "scenario_name")
        )
        object.__setattr__(
            self, "strategy_id", required_text(self.strategy_id, "strategy_id")
        )
        if (
            isinstance(self.strategy_version, bool)
            or not isinstance(self.strategy_version, int)
            or self.strategy_version < 1
        ):
            raise ValueError("strategy_version must be a positive integer")
        for field in ("manifest_hash", "evidence_sha256", "comparison_hash"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
        object.__setattr__(
            self, "evidence_ref", required_text(self.evidence_ref, "evidence_ref")
        )

    @property
    def automatic_action_permitted(self) -> bool:
        return False

    @property
    def requires_maker_checker(self) -> bool:
        return True

    def fsp8_binding(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "manifest_hash": self.manifest_hash,
            "evidence_ref": self.evidence_ref,
            "evidence_sha256": self.evidence_sha256,
            "comparison_hash": self.comparison_hash,
        }

    def fsp8_request_fields(self, *, reason: str) -> dict[str, Any]:
        """Return the non-authority fields accepted by FSP-8's request API.

        The FSP-8 caller must still supply the authenticated actor, environment,
        deployment slot, and expiry under its maker-checker controls.
        """

        human_reason = required_text(reason, "reason", max_length=3000)
        evidence_reason = (
            f"{human_reason} | decision_twin_evidence_ref={self.evidence_ref} "
            f"decision_twin_manifest_sha256={self.manifest_hash} "
            f"evidence_sha256={self.evidence_sha256} "
            f"comparison_sha256={self.comparison_hash}"
        )
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "reason": evidence_reason,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "decision_twin.promotion_evidence_proposal.v1",
            "authority": "proposal_only",
            "scenario_name": self.scenario_name,
            **self.fsp8_binding(),
            "automatic_action_permitted": self.automatic_action_permitted,
            "requires_maker_checker": self.requires_maker_checker,
        }


@dataclass(frozen=True)
class DecisionTwinComparison:
    champion: ScenarioEvaluation
    challenger: ScenarioEvaluation
    counterfactual: ScenarioEvaluation

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "decision_twin.comparison.v1",
            "champion": self.champion.to_dict(),
            "challenger": self.challenger.to_dict(),
            "counterfactual": self.counterfactual.to_dict(),
        }

    @property
    def comparison_hash(self) -> str:
        return content_hash(self.to_dict())

    def proposal_for(
        self,
        *,
        scenario_name: str,
        strategy_id: str,
        strategy_version: int,
        evidence_ref: str,
        evidence_sha256: str,
    ) -> PromotionEvidenceProposal:
        candidates = {
            self.champion.name: self.champion,
            self.challenger.name: self.challenger,
            self.counterfactual.name: self.counterfactual,
        }
        candidate = candidates.get(scenario_name)
        if candidate is None:
            raise ValueError("scenario_name is not part of this comparison")
        if candidate.kind is not ScenarioKind.CHALLENGER:
            raise ValueError("only a challenger may become FSP-8 promotion evidence")
        if not candidate.jointly_satisfied:
            raise ValueError("scenario does not satisfy the joint constraints")
        return PromotionEvidenceProposal(
            scenario_name=candidate.name,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            manifest_hash=candidate.manifest_hash,
            evidence_ref=evidence_ref,
            evidence_sha256=evidence_sha256,
            comparison_hash=self.comparison_hash,
        )


class DecisionTwinComparator:
    """Compute deterministic, matched-population scenario evidence."""

    def __init__(self, constraints: JointConstraintSet) -> None:
        if not isinstance(constraints, JointConstraintSet):
            raise ValueError("constraints must be JointConstraintSet")
        self._constraints = constraints

    def compare(
        self,
        *,
        champion: ScenarioReplay,
        challenger: ScenarioReplay,
        counterfactual: ScenarioReplay,
    ) -> DecisionTwinComparison:
        scenarios = (champion, challenger, counterfactual)
        expected_kinds = (
            ScenarioKind.CHAMPION,
            ScenarioKind.CHALLENGER,
            ScenarioKind.COUNTERFACTUAL,
        )
        if tuple(scenario.kind for scenario in scenarios) != expected_kinds:
            raise ValueError(
                "comparison requires champion, challenger, and counterfactual"
            )
        if len({scenario.name for scenario in scenarios}) != 3:
            raise ValueError("scenario names must be distinct")
        denominators = {
            (
                scenario.denominator.name,
                scenario.denominator.count,
                scenario.denominator.unit,
            )
            for scenario in scenarios
        }
        if len(denominators) != 1:
            raise ValueError("scenario denominator drift")
        reference = {decision.record_id: decision for decision in champion.decisions}
        reference_ids = set(reference)
        reference_groups = {
            record_id: (
                decision.protected_group.attribute,
                decision.protected_group.group,
                decision.protected_group.governance_ref,
                decision.protected_group.source_sha256,
            )
            for record_id, decision in reference.items()
        }
        for scenario in scenarios[1:]:
            decisions = {
                decision.record_id: decision for decision in scenario.decisions
            }
            if set(decisions) != reference_ids:
                raise ValueError("scenario replay populations are not record-matched")
            for record_id, decision in decisions.items():
                group_binding = (
                    decision.protected_group.attribute,
                    decision.protected_group.group,
                    decision.protected_group.governance_ref,
                    decision.protected_group.source_sha256,
                )
                if group_binding != reference_groups[record_id]:
                    raise ValueError(
                        "protected group assignment drifted across scenarios"
                    )
                if decision.decision_at != reference[record_id].decision_at:
                    raise ValueError("decision timestamp drifted across scenarios")
        group_counts: dict[str, int] = {}
        for decision in champion.decisions:
            group_counts[decision.protected_group.group] = (
                group_counts.get(decision.protected_group.group, 0) + 1
            )
        if len(group_counts) < 2:
            raise ValueError("at least two protected groups are required")
        if min(group_counts.values()) < self._constraints.min_protected_group_size:
            raise ValueError("protected group size is below the governed minimum")
        evaluations = tuple(self._evaluate(scenario) for scenario in scenarios)
        return DecisionTwinComparison(*evaluations)

    def _evaluate(self, scenario: ScenarioReplay) -> ScenarioEvaluation:
        constraints = self._constraints
        if scenario.currency != constraints.currency:
            raise ValueError("scenario currency does not match constraint unit")
        if scenario.operations_unit != constraints.operations_unit:
            raise ValueError(
                "scenario operations capacity does not match constraint unit"
            )
        approvals = sum(decision.approved for decision in scenario.decisions)
        approval_rate = approvals / scenario.denominator.count
        group_values: dict[str, list[bool]] = {}
        for decision in scenario.decisions:
            group_values.setdefault(decision.protected_group.group, []).append(
                decision.approved
            )
        group_rates = tuple(
            GroupApprovalRate(
                group=group,
                approvals=sum(values),
                denominator=len(values),
                rate=sum(values) / len(values),
            )
            for group, values in sorted(group_values.items())
        )
        disparity = max(rate.rate for rate in group_rates) - min(
            rate.rate for rate in group_rates
        )
        metrics = ScenarioMetrics(
            approval_rate=approval_rate,
            denominator_name=scenario.denominator.name,
            denominator_count=scenario.denominator.count,
            denominator_unit=scenario.denominator.unit,
            exposure=sum(decision.exposure for decision in scenario.decisions),
            ead=sum(decision.ead for decision in scenario.decisions),
            projected_loss=sum(
                decision.projected_loss for decision in scenario.decisions
            ),
            projected_profit=sum(
                decision.projected_profit for decision in scenario.decisions
            ),
            currency=scenario.currency,
            stability=min(decision.stability for decision in scenario.decisions),
            protected_group_attribute=scenario.decisions[0].protected_group.attribute,
            protected_group_rates=group_rates,
            protected_group_disparity=disparity,
            operations_capacity=sum(
                decision.operations_capacity for decision in scenario.decisions
            ),
            operations_unit=scenario.operations_unit,
        )
        results = (
            ConstraintResult(
                "approval_rate",
                metrics.approval_rate,
                ">=",
                constraints.min_approval_rate,
                "rate",
                metrics.approval_rate >= constraints.min_approval_rate,
            ),
            ConstraintResult(
                "exposure",
                metrics.exposure,
                "<=",
                constraints.max_exposure,
                constraints.currency,
                metrics.exposure <= constraints.max_exposure,
            ),
            ConstraintResult(
                "ead",
                metrics.ead,
                "<=",
                constraints.max_ead,
                constraints.currency,
                metrics.ead <= constraints.max_ead,
            ),
            ConstraintResult(
                "projected_loss",
                metrics.projected_loss,
                "<=",
                constraints.max_projected_loss,
                constraints.currency,
                metrics.projected_loss <= constraints.max_projected_loss,
            ),
            ConstraintResult(
                "projected_profit",
                metrics.projected_profit,
                ">=",
                constraints.min_projected_profit,
                constraints.currency,
                metrics.projected_profit >= constraints.min_projected_profit,
            ),
            ConstraintResult(
                "stability",
                metrics.stability,
                ">=",
                constraints.min_stability,
                "score",
                metrics.stability >= constraints.min_stability,
            ),
            ConstraintResult(
                "protected_group_disparity",
                metrics.protected_group_disparity,
                "<=",
                constraints.max_protected_group_disparity,
                "rate",
                metrics.protected_group_disparity
                <= constraints.max_protected_group_disparity,
            ),
            ConstraintResult(
                "operations_capacity",
                metrics.operations_capacity,
                "<=",
                constraints.max_operations_capacity,
                constraints.operations_unit,
                metrics.operations_capacity <= constraints.max_operations_capacity,
            ),
        )
        return ScenarioEvaluation(
            name=scenario.name,
            kind=scenario.kind,
            manifest_hash=scenario.manifest_hash,
            metrics=metrics,
            constraints=results,
            jointly_satisfied=all(result.passed for result in results),
            counterfactual=scenario.counterfactual,
        )
