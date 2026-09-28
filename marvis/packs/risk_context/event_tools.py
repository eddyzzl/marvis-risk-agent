from marvis.governance.native_producers import NativeInvocation
from marvis.risk_context.event_contracts import EventError
from marvis.risk_context.event_service import EventService, EventToolInput
from marvis.risk_context.source_contracts import SourceError
from marvis.settings import build_settings


def tool_replay_events(inputs, ctx):
    try:
        return EventService(build_settings(ctx.workspace)).execute(
            ctx.task_id,
            EventToolInput.model_validate(inputs),
            invocation=NativeInvocation.from_context(
                ctx, "risk_context.replay_events", inputs
            ),
        )
    except (EventError, SourceError) as exc:
        raise ValueError(exc.code) from None
    except (ValueError, RuntimeError, KeyError, OSError):
        raise ValueError("event_material_invalid") from None
