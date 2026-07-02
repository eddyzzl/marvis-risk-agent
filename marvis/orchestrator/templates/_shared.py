from __future__ import annotations

from marvis.orchestrator.contracts import PostCheck

# Modeling templates should not hard-fail every project on one fixed KS target.
# Business acceptance thresholds belong in a configurable policy/report gate; the
# workflow still reports OOT KS/AUC and section status, but completion is based on
# successful execution and explicit user/agent decisions.
BINARY_MODELING_SUCCESS_CRITERIA = ()

JOIN_EXECUTE_POST_CHECKS = (
    PostCheck("nonempty", {"field": "result_dataset_id"}),
    PostCheck("rowcount", {"field": "joined_rows", "min": 1}),
    PostCheck("invariant", {"rule": "joined_rows<=anchor_rows"}),
)
