"""HTTP actor adapter for the existing native source read guards."""

from functools import wraps
from inspect import iscoroutinefunction, signature

from marvis.errors import forbidden


def require_native_read(request, *, plan_id=None, task_id=None, proposed_plan=None):
    executor = getattr(request.app.state, "plan_executor", None)
    reconciler = getattr(executor, "reconciler", None)
    # Older embedders without the native producers retain ordinary plan reads.
    if reconciler is None:
        return
    actor_id = getattr(getattr(request.state, "local_principal", None), "id", None)
    try:
        if proposed_plan is not None:
            reconciler.authorize_proposed_plan_read(proposed_plan, actor_id=actor_id)
        elif plan_id is not None:
            reconciler.authorize_plan_read(plan_id, actor_id=actor_id)
        elif task_id is not None:
            reconciler.authorize_task_read(task_id, actor_id=actor_id)
        else:
            raise ValueError("plan or task scope required")
    except PermissionError as exc:
        raise forbidden(str(exc)) from exc


def native_task_read_scope(operation):
    """Recheck before returning derived conversation data after a long call."""
    parameters = signature(operation)

    def context(args, kwargs):
        bound = parameters.bind(*args, **kwargs)
        return bound.arguments["request"], bound.arguments["task_id"]

    if iscoroutinefunction(operation):

        @wraps(operation)
        async def guarded_async(*args, **kwargs):
            request, task_id = context(args, kwargs)
            require_native_read(request, task_id=task_id)
            result = await operation(*args, **kwargs)
            require_native_read(request, task_id=task_id)
            return result

        return guarded_async

    @wraps(operation)
    def guarded(*args, **kwargs):
        request, task_id = context(args, kwargs)
        require_native_read(request, task_id=task_id)
        result = operation(*args, **kwargs)
        require_native_read(request, task_id=task_id)
        return result

    return guarded
