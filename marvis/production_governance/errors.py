class ProductionGovernanceError(RuntimeError):
    """Base error for the local production-governance boundary."""


class GovernanceAuthenticationRequired(ProductionGovernanceError):
    pass


class GovernanceForbidden(ProductionGovernanceError):
    pass


class GovernanceNotFound(ProductionGovernanceError):
    pass


class GovernanceConflict(ProductionGovernanceError):
    pass
