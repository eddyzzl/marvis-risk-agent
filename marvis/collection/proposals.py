"""Business field declarations compile into the existing frozen batch contract.

This convenience surface only prepares a proposal. It grants no contact or
execution authority and keeps the same collection source and human gates.
"""

from typing import Literal

from pydantic import Field

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
)


class ProposedAction(Contract):
    kind: Literal["contact", "review", "hold"]
    queue_id: Identity | None
    priority: int = Field(ge=0, le=1000)
    channel: Channel | None
    estimated_cost_minor: Minor | None


class CollectionBusinessProposal(Contract):
    batch_id: Identity
    policy: CollectionPolicy
    action: ProposedAction
    cases: list[CollectionCaseInput] = Field(min_length=1, max_length=10000)
    histories: list[ContactHistory] = Field(default_factory=list, max_length=10000)
    as_of: str
    knowledge_cutoff: str

    def batch(self) -> CollectionBatchRequest:
        action = CollectionAction.model_validate(
            {
                **self.action.model_dump(),
                "policy_hash": self.policy.content_hash,
            }
        )
        strategy = StrategySpec(
            strategy_type="collection",
            schema_version=COLLECTION_STRATEGY_DSL_SCHEMA_VERSION,
            rules=(),
            default_action=StrategyAction(type="collection", value=action.model_dump()),
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
