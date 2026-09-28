"""Column mappings select native event evidence, never caller-made feature values."""

from typing import Literal

from pydantic import Field

from marvis.risk_context.event_contracts import Contract, EventFeatureContract, Hash, Id


class HistoricalEventMapping(Contract):
    scenario_kind: Literal["baseline", "challenger", "counterfactual"]
    reference_col: str = Field(min_length=1, max_length=160)
    subject_namespace_col: str = Field(min_length=1, max_length=160)
    subject_token_col: str = Field(min_length=1, max_length=160)
    grant_id: Id


class HistoricalNativeEventReference(Contract):
    task_id: Id
    request_id: Id
    expected_content_hash: Hash
    contract: EventFeatureContract
