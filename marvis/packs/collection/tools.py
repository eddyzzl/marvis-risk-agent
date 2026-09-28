from marvis.collection.contracts import Contract, Hash, Identity
from marvis.collection.execution import CollectionExecutor
from marvis.collection.ledger import CollectionEvidenceError
from marvis.risk_context.source_contracts import SourceError
from marvis.settings import build_settings


class CollectionToolInput(Contract):
    batch_id: Identity
    request_hash: Hash
    preview_hash: Hash


def _execute(inputs, ctx, operation):
    payload = CollectionToolInput.model_validate(inputs).model_dump()
    try:
        return CollectionExecutor(build_settings(ctx.workspace)).apply(
            ctx.task_id, payload, operation, ctx
        )
    except (CollectionEvidenceError, SourceError) as exc:
        raise ValueError(str(exc)) from None


def tool_queue_batch(inputs, ctx):
    return _execute(inputs, ctx, "queue_batch")


def tool_execute_reference(inputs, ctx):
    return _execute(inputs, ctx, "execute_reference")


def tool_cancel_batch(inputs, ctx):
    return _execute(inputs, ctx, "cancel_batch")
