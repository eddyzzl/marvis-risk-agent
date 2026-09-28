from marvis.risk_context.source_contracts import SourceError
from marvis.risk_context.source_service import SourceService
from marvis.settings import build_settings


def tool_query_source(inputs, ctx):
    # The server froze this exact intent under a live maker and admin-issued grant.
    # Workers cannot create grants, select subjects, or authorize new request IDs.
    try:
        return SourceService(build_settings(ctx.workspace)).execute(
            ctx.task_id, inputs["request_id"]
        )
    except SourceError as exc:
        # No provider response, identity, URL or credential in exception diagnostics.
        raise ValueError(exc.code) from None
