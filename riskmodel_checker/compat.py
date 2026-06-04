try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised by user Notebook kernels on py<3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass
