class GovernanceError(RuntimeError):
    """Base class for governance persistence/runtime failures."""


class AuthorizationError(GovernanceError):
    """A governed effect is not authorized to execute.

    ToolRunner may safely map this narrow hierarchy to
    ``error_kind='authorization'``. Unexpected programming/database errors are
    deliberately not wrapped as AuthorizationError.
    """


class PrincipalNotFound(AuthorizationError):
    pass


class PrincipalInactive(AuthorizationError):
    pass


class ApprovalNotFound(AuthorizationError):
    pass


class ApprovalBindingError(AuthorizationError):
    pass


class ApprovalStateError(AuthorizationError):
    pass


class ApprovalExpired(ApprovalStateError):
    pass


class EffectExecutionNotFound(AuthorizationError):
    pass
