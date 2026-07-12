from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALIDATION_NON_REGRESSION_CONTRACT_PATH = (
    Path(__file__).parent / "fixtures" / "validation_non_regression_contract.json"
)


def load_validation_non_regression_contract() -> dict[str, Any]:
    return json.loads(
        VALIDATION_NON_REGRESSION_CONTRACT_PATH.read_text(encoding="utf-8")
    )
