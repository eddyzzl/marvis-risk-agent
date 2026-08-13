"""Bounded counterfactual ("what-if") sandbox kernel for the decision twin.

This module turns the decision twin's deterministic replay foundations into a
*proposal-only* counterfactual analysis over one authenticated sample partition.
It reuses the strategy pack's canonical DSL and row evaluator for first-match +
default-action semantics, and layers its own bounded replay orchestration on top
so no shared code changes are required.

Metric definitions (``1 = bad`` target convention, matching the strategy pack's
internal ``1=bad`` normalization):

For a scenario evaluated over ``N`` bound partition rows:

* ``count`` (件数): ``N``, the partition population actually replayed.
* ``approval_count`` (通过件数): number of rows whose first-match decision
  action type is ``approval``.
* ``approval_rate`` (通过率): ``approval_count / N``.
* ``bad_count`` (坏件数): number of *approved* rows whose target equals the bad
  value. A declined bad never realizes loss through this strategy, so bads are
  counted only inside the approved population.
* ``bad_rate`` (坏率): ``bad_count / approval_count`` when ``approval_count > 0``,
  otherwise ``0.0``. This is the approved-population ("through-the-door") bad
  rate, which is the quantity a strategy threshold actually moves.
* ``exposure_sum``: sum of the exposure column over approved rows; present only
  when ``exposure_col`` is supplied.
* ``expected_loss`` (EL): sum of exposure over approved *and* bad rows; present
  only when ``exposure_col`` is supplied. This is the deterministic realized-
  default exposure proxy ``EL = sum(exposure_i * 1{approved_i and bad_i})``.

Determinism and safety contract:

* The replay is pure: it never writes files, never emits per-row detail, and
  never mutates the caller's spec, base strategy, or frame.
* A positive row budget bounds the replay; exceeding it fails closed before any
  row is evaluated.
* The output carries ``counterfactual_only=True`` and ``authority='proposal_only'``
  and exposes ``automatic_action_permitted=False``. It has no authority to
  mutate the pool, the strategy, or adoption state.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from marvis.decision_twin._canonical import (
    content_hash,
    finite_number,
    required_text,
)
from marvis.packs.strategy.dsl import (
    StrategySpec,
    canonicalize_expression,
    parse_strategy_spec,
    strategy_spec_hash,
)
from marvis.packs.strategy.evaluator import evaluate_strategy_row

COUNTERFACTUAL_SCHEMA_VERSION = "decision_twin.counterfactual.v1"
DEFAULT_ROW_BUDGET = 1_000_000

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BASE_REF_KINDS = frozenset({"pool_revision", "canonical_strategy"})
_DELTA_KINDS = frozenset({"threshold", "action_flip"})
_THRESHOLD_ATTRIBUTES = frozenset({"value", "lower", "upper"})
_APPROVAL_STRATEGY_TYPES = frozenset({"approval", "reject"})
_JSON_SCALARS = (str, int, float, bool)


class CounterfactualError(ValueError):
    """Base error for the counterfactual sandbox."""


class CounterfactualSpecError(CounterfactualError):
    """A counterfactual spec or binding is invalid."""


class CounterfactualDeltaError(CounterfactualError):
    """A counterfactual delta violates the closed whitelist."""


class CounterfactualRowBudgetError(CounterfactualError):
    """The replay exceeded its row budget and failed closed."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CounterfactualSpecError(
            f"{field} must be a lowercase SHA-256 hex digest"
        )
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CounterfactualSpecError(f"{field} must be a positive integer")
    return value


def _delta_scalar(value: object, field: str) -> str | int | float | bool:
    """Validate one closed-whitelist threshold value as a finite JSON scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return finite_number(value, field)
    if isinstance(value, str):
        return required_text(value, field, max_length=200)
    raise CounterfactualDeltaError(
        f"{field} must be a scalar string, number, or boolean"
    )


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class PoolRevisionRef:
    """Content-addressed reference to one authenticated Strategy Pool revision."""

    pool_id: str
    revision: int
    revision_id: str
    snapshot_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", required_text(self.pool_id, "pool_id"))
        object.__setattr__(
            self, "revision", _positive_int(self.revision, "pool revision")
        )
        object.__setattr__(
            self,
            "revision_id",
            required_text(self.revision_id, "pool revision_id"),
        )
        object.__setattr__(
            self,
            "snapshot_hash",
            _sha256(self.snapshot_hash, "pool snapshot_hash"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoolRevisionRef":
        if not isinstance(payload, Mapping):
            raise CounterfactualSpecError("pool revision ref must be an object")
        return cls(
            pool_id=payload.get("pool_id"),
            revision=payload.get("revision"),
            revision_id=payload.get("revision_id"),
            snapshot_hash=payload.get("snapshot_hash"),
        )

    def to_ref_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "revision": self.revision,
            "revision_id": self.revision_id,
            "snapshot_hash": self.snapshot_hash,
        }


@dataclass(frozen=True)
class CanonicalStrategyRef:
    """Content-addressed reference to one canonical strategy id/version."""

    strategy_id: str
    strategy_version: int
    strategy_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy_id", required_text(self.strategy_id, "strategy_id")
        )
        object.__setattr__(
            self,
            "strategy_version",
            _positive_int(self.strategy_version, "strategy_version"),
        )
        object.__setattr__(
            self,
            "strategy_content_hash",
            _sha256(self.strategy_content_hash, "strategy_content_hash"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalStrategyRef":
        if not isinstance(payload, Mapping):
            raise CounterfactualSpecError("canonical strategy ref must be an object")
        return cls(
            strategy_id=payload.get("strategy_id"),
            strategy_version=payload.get("strategy_version"),
            strategy_content_hash=payload.get("strategy_content_hash"),
        )

    def to_ref_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "strategy_content_hash": self.strategy_content_hash,
        }


@dataclass(frozen=True)
class BaseStrategyRef:
    """Exactly one of a pool revision or a canonical strategy id.

    A ``pool_revision`` binding is verified against the compiled strategy's
    ``metadata.lineage.pool_ref`` when that lineage is present; a
    ``canonical_strategy`` binding is always verified against the strategy's
    content hash and fails closed on any drift.
    """

    kind: str
    pool_revision: PoolRevisionRef | None = None
    canonical_strategy: CanonicalStrategyRef | None = None

    def __post_init__(self) -> None:
        kind = required_text(self.kind, "base ref kind")
        if kind not in _BASE_REF_KINDS:
            allowed = ", ".join(sorted(_BASE_REF_KINDS))
            raise CounterfactualSpecError(
                f"base ref kind must be one of: {allowed}"
            )
        object.__setattr__(self, "kind", kind)
        if kind == "pool_revision":
            if not isinstance(self.pool_revision, PoolRevisionRef):
                raise CounterfactualSpecError(
                    "pool_revision base ref requires a pool_revision object"
                )
            if self.canonical_strategy is not None:
                raise CounterfactualSpecError(
                    "pool_revision base ref must not carry canonical_strategy"
                )
        else:
            if not isinstance(self.canonical_strategy, CanonicalStrategyRef):
                raise CounterfactualSpecError(
                    "canonical_strategy base ref requires a canonical_strategy object"
                )
            if self.pool_revision is not None:
                raise CounterfactualSpecError(
                    "canonical_strategy base ref must not carry pool_revision"
                )

    @classmethod
    def from_value(cls, value: object) -> "BaseStrategyRef":
        if not isinstance(value, Mapping):
            raise CounterfactualSpecError("base strategy ref must be an object")
        kind = value.get("kind")
        if kind == "pool_revision":
            return cls(kind="pool_revision", pool_revision=PoolRevisionRef.from_dict(
                value.get("pool_revision")
            ))
        if kind == "canonical_strategy":
            return cls(
                kind="canonical_strategy",
                canonical_strategy=CanonicalStrategyRef.from_dict(
                    value.get("canonical_strategy")
                ),
            )
        raise CounterfactualSpecError(
            "base strategy ref kind must be pool_revision or canonical_strategy"
        )

    def to_ref_dict(self) -> dict[str, Any]:
        if self.kind == "pool_revision":
            return {"kind": "pool_revision", "pool_revision": self.pool_revision.to_ref_dict()}
        return {
            "kind": "canonical_strategy",
            "canonical_strategy": self.canonical_strategy.to_ref_dict(),
        }


@dataclass(frozen=True)
class CounterfactualSampleRef:
    """Content-addressed reference to one authenticated sample partition."""

    artifact_id: str
    artifact_content_hash: str
    sample_design_id: str
    sample_design_content_hash: str
    partition: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", required_text(self.artifact_id, "artifact_id")
        )
        object.__setattr__(
            self,
            "artifact_content_hash",
            _sha256(self.artifact_content_hash, "artifact_content_hash"),
        )
        object.__setattr__(
            self,
            "sample_design_id",
            required_text(self.sample_design_id, "sample_design_id"),
        )
        object.__setattr__(
            self,
            "sample_design_content_hash",
            _sha256(self.sample_design_content_hash, "sample_design_content_hash"),
        )
        object.__setattr__(
            self, "partition", required_text(self.partition, "partition")
        )

    @classmethod
    def from_value(cls, value: object) -> "CounterfactualSampleRef":
        if not isinstance(value, Mapping):
            raise CounterfactualSpecError("sample ref must be an object")
        return cls(
            artifact_id=value.get("artifact_id"),
            artifact_content_hash=value.get("artifact_content_hash"),
            sample_design_id=value.get("sample_design_id"),
            sample_design_content_hash=value.get("sample_design_content_hash"),
            partition=value.get("partition"),
        )

    def to_ref_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_content_hash": self.artifact_content_hash,
            "sample_design_id": self.sample_design_id,
            "sample_design_content_hash": self.sample_design_content_hash,
            "partition": self.partition,
        }


@dataclass(frozen=True)
class CounterfactualDelta:
    """One concrete, closed-whitelist strategy mutation.

    Supported kinds (exactly two; anything else fails closed):

    * ``threshold``: adjust one rule's threshold attribute to an exact scalar.
      Fields: ``rule_id``, ``field``, ``attribute`` (``value`` for a ``compare``
      leaf, or ``lower``/``upper`` for a ``between`` leaf), and ``value``.
    * ``action_flip``: flip one rule/entry's approve<->reject action. Field:
      ``rule_id`` only.

    Fuzzy targets such as "lower approval rate to 30%" are structurally
    unrepresentable here; only a concrete, rule-addressed delta is accepted.
    """

    kind: str
    rule_id: str
    field: str | None = None
    attribute: str | None = None
    value: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        kind = required_text(self.kind, "delta kind")
        if kind not in _DELTA_KINDS:
            allowed = ", ".join(sorted(_DELTA_KINDS))
            raise CounterfactualDeltaError(
                f"unsupported delta kind {kind!r}; allowed: {allowed}"
            )
        rule_id = required_text(self.rule_id, "delta rule_id")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rule_id", rule_id)
        if kind == "threshold":
            field = required_text(self.field, "delta field")
            attribute = required_text(self.attribute, "delta attribute")
            if attribute not in _THRESHOLD_ATTRIBUTES:
                allowed = ", ".join(sorted(_THRESHOLD_ATTRIBUTES))
                raise CounterfactualDeltaError(
                    f"threshold delta attribute must be one of: {allowed}"
                )
            object.__setattr__(self, "field", field)
            object.__setattr__(self, "attribute", attribute)
            object.__setattr__(
                self, "value", _delta_scalar(self.value, "delta value")
            )
        else:
            if self.field is not None or self.attribute is not None or self.value is not None:
                raise CounterfactualDeltaError(
                    "action_flip delta accepts only kind and rule_id"
                )

    @classmethod
    def from_value(cls, value: object) -> "CounterfactualDelta":
        if not isinstance(value, Mapping):
            raise CounterfactualDeltaError("delta must be an object")
        kind = value.get("kind")
        if kind == "threshold":
            return cls(
                kind="threshold",
                rule_id=value.get("rule_id"),
                field=value.get("field"),
                attribute=value.get("attribute"),
                value=value.get("value"),
            )
        if kind == "action_flip":
            return cls(kind="action_flip", rule_id=value.get("rule_id"))
        raise CounterfactualDeltaError(
            "delta kind must be threshold or action_flip"
        )

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "threshold":
            return {
                "kind": "threshold",
                "rule_id": self.rule_id,
                "field": self.field,
                "attribute": self.attribute,
                "value": self.value,
            }
        return {"kind": "action_flip", "rule_id": self.rule_id}


@dataclass(frozen=True)
class CounterfactualSpec:
    """Complete typed "what-if" request: base ref + concrete deltas + sample."""

    base: BaseStrategyRef
    deltas: tuple[CounterfactualDelta, ...]
    sample: CounterfactualSampleRef
    row_budget: int = DEFAULT_ROW_BUDGET
    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.base, BaseStrategyRef):
            raise CounterfactualSpecError("base must be a BaseStrategyRef")
        if not isinstance(self.sample, CounterfactualSampleRef):
            raise CounterfactualSpecError("sample must be a CounterfactualSampleRef")
        deltas = tuple(self.deltas)
        if not deltas or not all(
            isinstance(delta, CounterfactualDelta) for delta in deltas
        ):
            raise CounterfactualSpecError(
                "deltas must contain at least one CounterfactualDelta"
            )
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(
            self, "row_budget", _positive_int(self.row_budget, "row_budget")
        )
        if self.schema_version != COUNTERFACTUAL_SCHEMA_VERSION:
            raise CounterfactualSpecError(
                f"unsupported counterfactual schema: {self.schema_version}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base": self.base.to_ref_dict(),
            "deltas": [delta.to_dict() for delta in self.deltas],
            "sample": self.sample.to_ref_dict(),
            "row_budget": self.row_budget,
        }

    @property
    def spec_hash(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True)
class ScenarioSummary:
    """Deterministic aggregates for one strategy over the bound partition."""

    label: str
    count: int
    approval_count: int
    approval_rate: float
    bad_count: int
    bad_rate: float
    exposure_sum: float | None
    expected_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "approval_count": self.approval_count,
            "approval_rate": self.approval_rate,
            "bad_count": self.bad_count,
            "bad_rate": self.bad_rate,
            "exposure_sum": self.exposure_sum,
            "expected_loss": self.expected_loss,
        }


@dataclass(frozen=True)
class CounterfactualComparison:
    """Base -> delta differences (delta minus base)."""

    approval_count_delta: int
    approval_rate_delta: float
    bad_count_delta: int
    bad_rate_delta: float
    exposure_sum_delta: float | None
    expected_loss_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_count_delta": self.approval_count_delta,
            "approval_rate_delta": self.approval_rate_delta,
            "bad_count_delta": self.bad_count_delta,
            "bad_rate_delta": self.bad_rate_delta,
            "exposure_sum_delta": self.exposure_sum_delta,
            "expected_loss_delta": self.expected_loss_delta,
        }


@dataclass(frozen=True)
class CounterfactualResult:
    """Proposal-only evidence: base/delta comparison with no side effects."""

    spec_hash: str
    base_strategy_hash: str
    delta_strategy_hash: str
    base: ScenarioSummary
    delta: ScenarioSummary
    comparison: CounterfactualComparison
    counterfactual_only: bool = True
    authority: str = "proposal_only"
    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION

    @property
    def automatic_action_permitted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority": self.authority,
            "counterfactual_only": self.counterfactual_only,
            "automatic_action_permitted": self.automatic_action_permitted,
            "spec_hash": self.spec_hash,
            "base_strategy_hash": self.base_strategy_hash,
            "delta_strategy_hash": self.delta_strategy_hash,
            "base": self.base.to_dict(),
            "delta": self.delta.to_dict(),
            "comparison": self.comparison.to_dict(),
        }

    @property
    def result_hash(self) -> str:
        return content_hash(self.to_dict())


def _find_rule(rules: list[dict[str, Any]], rule_id: str) -> dict[str, Any]:
    matches = [rule for rule in rules if rule.get("rule_id") == rule_id]
    if not matches:
        raise CounterfactualDeltaError(f"delta targets unknown rule_id: {rule_id!r}")
    if len(matches) > 1:
        raise CounterfactualDeltaError(
            f"delta targets ambiguous rule_id: {rule_id!r}"
        )
    # Return the live rule dict: callers only pass a deep copy of the base spec.
    return matches[0]


def _threshold_attribute_applies(node: Mapping[str, Any], attribute: str) -> bool:
    op = node.get("op")
    if attribute == "value":
        return op == "compare" and node.get("operator") not in {"in", "not_in"}
    if attribute in {"lower", "upper"}:
        return op == "between"
    return False


def _collect_threshold_leaves(
    node: object,
    field: str,
    attribute: str,
    out: list[dict[str, Any]],
) -> None:
    if not isinstance(node, dict):
        return
    if node.get("op") in {"compare", "between"} and node.get("field") == field:
        if _threshold_attribute_applies(node, attribute):
            out.append(node)
    args = node.get("args")
    if isinstance(args, list):
        for argument in args:
            _collect_threshold_leaves(argument, field, attribute, out)
    arg = node.get("arg")
    if isinstance(arg, dict):
        _collect_threshold_leaves(arg, field, attribute, out)


def _apply_threshold_delta(spec_dict: dict[str, Any], delta: CounterfactualDelta) -> None:
    rule = _find_rule(spec_dict["rules"], delta.rule_id)
    condition = rule["condition"]
    matches: list[dict[str, Any]] = []
    _collect_threshold_leaves(condition, delta.field, delta.attribute, matches)
    if len(matches) != 1:
        raise CounterfactualDeltaError(
            f"threshold delta must match exactly one {delta.attribute!r} leaf "
            f"for field {delta.field!r} in rule {delta.rule_id!r}; found {len(matches)}"
        )
    matches[0][delta.attribute] = delta.value
    rule["condition"] = canonicalize_expression(condition)


def _apply_action_flip_delta(spec_dict: dict[str, Any], delta: CounterfactualDelta) -> None:
    rule = _find_rule(spec_dict["rules"], delta.rule_id)
    action = rule["action"]
    action_type = action.get("type")
    if action_type == "approval":
        new_type, new_value = "reject", "reject"
    elif action_type == "reject":
        new_type, new_value = "approval", "approve"
    else:
        raise CounterfactualDeltaError(
            "action_flip requires an approval or reject action; "
            f"rule {delta.rule_id!r} has {action_type!r}"
        )
    rule["action"] = {
        "type": new_type,
        "value": new_value,
        "reason_code": action.get("reason_code"),
        "stop": True,
    }


def _apply_deltas(
    base_dict: dict[str, Any],
    deltas: tuple[CounterfactualDelta, ...],
) -> dict[str, Any]:
    spec_dict = copy.deepcopy(base_dict)
    for delta in deltas:
        if delta.kind == "threshold":
            _apply_threshold_delta(spec_dict, delta)
        else:
            _apply_action_flip_delta(spec_dict, delta)
    return spec_dict


def _verify_base_ref(base_ref: BaseStrategyRef, base_dict: dict[str, Any]) -> None:
    if base_ref.kind == "canonical_strategy":
        expected = base_ref.canonical_strategy.strategy_content_hash
        actual = strategy_spec_hash(base_dict)
        if actual != expected:
            raise CounterfactualSpecError(
                "base strategy content hash does not match canonical_strategy ref"
            )
        return
    lineage = base_dict.get("metadata", {}).get("lineage", {})
    pool_ref = lineage.get("pool_ref")
    if not isinstance(pool_ref, dict):
        return
    reference = base_ref.pool_revision
    observed = (
        pool_ref.get("pool_id"),
        pool_ref.get("revision"),
        pool_ref.get("revision_id"),
        pool_ref.get("snapshot_hash"),
    )
    expected = (
        reference.pool_id,
        reference.revision,
        reference.revision_id,
        reference.snapshot_hash,
    )
    if observed != expected:
        raise CounterfactualSpecError(
            "base strategy pool_ref does not match pool_revision ref"
        )


def _is_bad(value: object, bad_value: object) -> bool:
    value = _json_scalar(value)
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    try:
        return bool(value == bad_value)
    except (TypeError, ValueError):
        return False


def _validate_exposure(frame: pd.DataFrame, exposure_col: str) -> None:
    for raw in frame[exposure_col].tolist():
        value = _json_scalar(raw)
        try:
            number = finite_number(value, f"exposure column {exposure_col!r}")
        except ValueError as exc:
            raise CounterfactualSpecError(str(exc)) from exc
        if number < 0:
            raise CounterfactualSpecError(
                f"exposure column {exposure_col!r} must be non-negative"
            )


def _summarize(
    records: list[dict[str, Any]],
    spec: StrategySpec,
    target_col: str,
    bad_value: object,
    exposure_col: str | None,
    label: str,
) -> ScenarioSummary:
    approval_count = 0
    bad_count = 0
    exposure_sum = 0.0
    expected_loss = 0.0
    for record in records:
        evaluation = evaluate_strategy_row(record, spec)
        if evaluation.action.type != "approval":
            continue
        approval_count += 1
        if exposure_col is not None:
            exposure_sum += float(_json_scalar(record[exposure_col]))
        if _is_bad(record[target_col], bad_value):
            bad_count += 1
            if exposure_col is not None:
                expected_loss += float(_json_scalar(record[exposure_col]))
    count = len(records)
    approval_rate = (approval_count / count) if count else 0.0
    bad_rate = (bad_count / approval_count) if approval_count else 0.0
    has_exposure = exposure_col is not None
    return ScenarioSummary(
        label=label,
        count=count,
        approval_count=approval_count,
        approval_rate=approval_rate,
        bad_count=bad_count,
        bad_rate=bad_rate,
        exposure_sum=(exposure_sum if has_exposure else None),
        expected_loss=(expected_loss if has_exposure else None),
    )


def _maybe_delta(delta_value: float | None, base_value: float | None) -> float | None:
    if delta_value is None or base_value is None:
        return None
    return delta_value - base_value


def run_bounded_counterfactual(
    *,
    spec: CounterfactualSpec,
    base_strategy: Mapping[str, Any] | StrategySpec,
    frame: pd.DataFrame,
    target_col: str,
    bad_value: object = 1,
    exposure_col: str | None = None,
) -> CounterfactualResult:
    """Run base vs delta over the same authenticated partition, fail-closed.

    ``base_strategy`` is the canonical strategy spec dict (the ``strategy_spec``
    produced by pool compilation or a canonical strategy body). The base ref in
    ``spec`` is provenance; a canonical ref is verified against the content hash.
    """
    if not isinstance(spec, CounterfactualSpec):
        raise CounterfactualSpecError("spec must be a CounterfactualSpec")
    if not isinstance(frame, pd.DataFrame):
        raise CounterfactualSpecError("frame must be a pandas DataFrame")
    if not isinstance(base_strategy, (Mapping, StrategySpec)):
        raise CounterfactualSpecError("base_strategy must be a strategy spec")

    base_spec = parse_strategy_spec(base_strategy)
    if base_spec.strategy_type not in _APPROVAL_STRATEGY_TYPES:
        raise CounterfactualSpecError(
            "counterfactual approval metrics require an approval or reject "
            f"strategy_type, got {base_spec.strategy_type!r}"
        )
    base_dict = base_spec.to_dict()
    _verify_base_ref(spec.base, base_dict)

    row_count = len(frame)
    if row_count > spec.row_budget:
        raise CounterfactualRowBudgetError(
            f"counterfactual row budget {spec.row_budget} exceeded by {row_count} rows"
        )

    columns = set(str(column) for column in frame.columns)
    if target_col not in columns:
        raise CounterfactualSpecError(f"target_col {target_col!r} is not a frame column")
    if exposure_col is not None and exposure_col not in columns:
        raise CounterfactualSpecError(
            f"exposure_col {exposure_col!r} is not a frame column"
        )
    if exposure_col is not None:
        _validate_exposure(frame, exposure_col)

    delta_dict = _apply_deltas(base_dict, spec.deltas)
    delta_spec = parse_strategy_spec(delta_dict)
    bad = _json_scalar(bad_value)

    records = frame.reset_index(drop=True).to_dict("records")
    base_summary = _summarize(records, base_spec, target_col, bad, exposure_col, "base")
    delta_summary = _summarize(records, delta_spec, target_col, bad, exposure_col, "delta")

    comparison = CounterfactualComparison(
        approval_count_delta=delta_summary.approval_count - base_summary.approval_count,
        approval_rate_delta=delta_summary.approval_rate - base_summary.approval_rate,
        bad_count_delta=delta_summary.bad_count - base_summary.bad_count,
        bad_rate_delta=delta_summary.bad_rate - base_summary.bad_rate,
        exposure_sum_delta=_maybe_delta(
            delta_summary.exposure_sum, base_summary.exposure_sum
        ),
        expected_loss_delta=_maybe_delta(
            delta_summary.expected_loss, base_summary.expected_loss
        ),
    )
    return CounterfactualResult(
        spec_hash=spec.spec_hash,
        base_strategy_hash=strategy_spec_hash(base_dict),
        delta_strategy_hash=strategy_spec_hash(delta_dict),
        base=base_summary,
        delta=delta_summary,
        comparison=comparison,
    )


class CounterfactualSandbox:
    """Proposal-only bounded counterfactual replay kernel."""

    def run(
        self,
        *,
        spec: CounterfactualSpec,
        base_strategy: Mapping[str, Any] | StrategySpec,
        frame: pd.DataFrame,
        target_col: str,
        bad_value: object = 1,
        exposure_col: str | None = None,
    ) -> CounterfactualResult:
        return run_bounded_counterfactual(
            spec=spec,
            base_strategy=base_strategy,
            frame=frame,
            target_col=target_col,
            bad_value=bad_value,
            exposure_col=exposure_col,
        )


__all__ = [
    "COUNTERFACTUAL_SCHEMA_VERSION",
    "DEFAULT_ROW_BUDGET",
    "BaseStrategyRef",
    "CanonicalStrategyRef",
    "CounterfactualComparison",
    "CounterfactualDelta",
    "CounterfactualDeltaError",
    "CounterfactualError",
    "CounterfactualResult",
    "CounterfactualRowBudgetError",
    "CounterfactualSampleRef",
    "CounterfactualSandbox",
    "CounterfactualSpec",
    "CounterfactualSpecError",
    "PoolRevisionRef",
    "ScenarioSummary",
    "run_bounded_counterfactual",
]
