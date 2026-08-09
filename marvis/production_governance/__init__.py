"""Local production-governance foundation.

This package deliberately reuses MARVIS' server-issued local session principal.
It is not an external identity provider or a production deployment adapter.
"""

from marvis.production_governance.repository import ProductionGovernanceRepository

__all__ = ["ProductionGovernanceRepository"]
