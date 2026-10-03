"""Business field declarations compile into the existing frozen batch contract.

This convenience surface only prepares a proposal. It grants no contact or
execution authority and keeps the same collection source and human gates.
"""

from typing import Literal

from pydantic import Field, model_validator

from marvis.collection.actions import (
    Channel,
    CollectionAction,
    CollectionCaseInput,
    CollectionPolicy,
    ContactHistory,
)
from marvis.collection.batches import CollectionBatchRequest
from marvis.collection.contracts import Contract, Identity, Minor
from marvis.packs.strategy.dsl import (
    COLLECTION_STRATEGY_DSL_SCHEMA_VERSION,
    StrategyAction,
    StrategySpec,
    StrategyRuleSpec,
)


class ProposedAction(Contract):
    kind: Literal["contact", "review", "hold"]
    queue_id: Identity | None
    priority: int = Field(ge=0, le=1000)
    channel: Channel | None
    estimated_cost_minor: Minor | None


class ProposedCondition(Contract):
    field: str = Field(min_length=1, max_length=128)
    value_type: Literal["number", "text", "boolean"]
    operator: Literal["<", "<=", ">", ">=", "==", "!=", "is_null", "is_not_null"]
    value: str | int | float | bool | None
    missing: Literal["no_match", "match", "error"] | None

    def accepts(self, value):
        if self.value_type == "number":
            return type(value) in {int, float}
        return type(value) is (str if self.value_type == "text" else bool)

    @model_validator(mode="after")
    def coherent(self):
        if not self.field.strip() or self.field != self.field.strip():
            raise ValueError("collection_rule_field_invalid")
        if self.operator in {"is_null", "is_not_null"}:
            if self.value is not None or self.missing is not None:
                raise ValueError("collection_null_condition_requires_inapplicable_values")
        else:
            if self.missing is None:
                raise ValueError("collection_comparison_requires_missing_policy")
            if not self.accepts(self.value):
                raise ValueError("collection_rule_literal_type_mismatch")
        if self.operator in {"<", "<=", ">", ">="} and self.value_type != "number":
            raise ValueError("collection_ordered_condition_requires_number")
        return self

    def expression(self):
        if self.operator in {"is_null", "is_not_null"}:
            return {"op": self.operator, "field": self.field}
        value = {
            "op": "compare",
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "missing": self.missing,
        }
        if self.operator in {"==", "!="}:
            value["coercion"] = "strict"
        return value


class ProposedRule(Contract):
    rule_id: Identity
    combination: Literal["all", "any"]
    conditions: list[ProposedCondition] = Field(min_length=1, max_length=20)
    action: ProposedAction


class CollectionBusinessProposal(Contract):
    batch_id: Identity
    policy: CollectionPolicy
    action: ProposedAction
    rules: list[ProposedRule] = Field(default_factory=list, max_length=50)
    cases: list[CollectionCaseInput] = Field(min_length=1, max_length=10000)
    histories: list[ContactHistory] = Field(default_factory=list, max_length=10000)
    as_of: str
    knowledge_cutoff: str

    def batch(self) -> CollectionBatchRequest:
        def action(value):
            frozen = CollectionAction.model_validate(
                {**value.model_dump(), "policy_hash": self.policy.content_hash}
            )
            return StrategyAction(type="collection", value=frozen.model_dump())

        rules = []
        field_types = {}
        for priority, rule in enumerate(self.rules):
            for condition in rule.conditions:
                previous_type = field_types.setdefault(
                    condition.field, condition.value_type
                )
                if previous_type != condition.value_type:
                    raise ValueError("collection_rule_field_type_conflict")
                for case in self.cases:
                    if condition.field not in case.features:
                        raise ValueError("collection_rule_field_not_declared")
                    value = case.features[condition.field]
                    if value is not None and not condition.accepts(value):
                        raise ValueError("collection_rule_feature_type_mismatch")
            rules.append(
                StrategyRuleSpec(
                    rule_id=rule.rule_id,
                    priority=priority,
                    condition={
                        "op": "and" if rule.combination == "all" else "or",
                        "args": [c.expression() for c in rule.conditions],
                    },
                    action=action(rule.action),
                )
            )
        strategy = StrategySpec(
            strategy_type="collection",
            schema_version=COLLECTION_STRATEGY_DSL_SCHEMA_VERSION,
            rules=tuple(rules),
            default_action=action(self.action),
        )
        return CollectionBatchRequest(
            batch_id=self.batch_id,
            execution_mode="local_reference",
            policy=self.policy,
            strategy=strategy.to_dict(),
            cases=self.cases,
            histories=self.histories,
            as_of=self.as_of,
            knowledge_cutoff=self.knowledge_cutoff,
        )
