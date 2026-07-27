"""Task-scoped, per-recipe checkpoints for long model-tuning runs.

The tuning search itself remains owned by :mod:`marvis.packs.modeling.tune`.
This module only persists the completed result of one whole recipe so a later
retry can skip work that was already finished.  A checkpoint is reusable only
when every input that can affect that recipe's search is identical.
"""

from __future__ import annotations

import hashlib
import hmac
from functools import lru_cache
from importlib import metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

from marvis.artifacts import TransactionalArtifactStore
from marvis.files import sha256_file
from marvis.packs.modeling.defaults import (
    DEFAULT_TRAIN_NUM_THREADS,
    DEFAULT_TUNE_NUM_THREADS,
)


TUNE_CHECKPOINT_CONTRACT_VERSION = "modeling.tune-checkpoint.v2"
TUNE_CHECKPOINT_DIR_NAME = "tuning_checkpoints"
_RECIPE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMPLEMENTATION_FILES = (
    "train_tools.py",
    "training_dataset.py",
    "tune.py",
    "tune_isolation.py",
    "tune_checkpoint.py",
    "defaults.py",
    "recipes/common.py",
    "../../data/backend.py",
    "../../data/labels.py",
    "../../feature/metrics.py",
)
_RECIPE_DEPENDENCIES = {
    "lgb": ("lightgbm",),
    "lgb_regressor": ("lightgbm",),
    "lgb_multiclass": ("lightgbm",),
    "xgb": ("xgboost",),
    "xgb_regressor": ("xgboost",),
    "xgb_multiclass": ("xgboost",),
    "catboost": ("catboost",),
    "lr": ("scikit-learn",),
    "lr_regressor": ("scikit-learn",),
    "lr_multiclass": ("scikit-learn",),
    "scorecard": ("scikit-learn",),
    "mlp": ("scikit-learn",),
    "mlp_regressor": ("scikit-learn",),
    "mlp_multiclass": ("scikit-learn",),
}
_THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def dataset_content_identity(_dataset: object, dataset_path: Path) -> str:
    """Hash the current bytes instead of trusting possibly stale registry metadata."""

    return f"sha256:{sha256_file(Path(dataset_path))}"


def tuning_runtime_fingerprint(recipe: str) -> dict[str, Any]:
    """Fingerprint tuning code, pack contract, and recipe-specific learners."""

    dependencies = tuple(dict.fromkeys((
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        *_RECIPE_DEPENDENCIES.get(str(recipe), ()),
    )))
    payload = {
        "implementation_sha256": _implementation_sha256(),
        "pack_version": _pack_version(),
        # Retain the historical top-level field for audit readers while the
        # v2 contract records the complete platform identity below.
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "cpu": {
            "logical_count": int(os.cpu_count() or 1),
            "affinity_count": _cpu_affinity_count(),
        },
        "threading": {
            "default_train_num_threads": int(DEFAULT_TRAIN_NUM_THREADS),
            "default_tune_num_threads": int(DEFAULT_TUNE_NUM_THREADS),
            "environment": {
                name: os.environ.get(name)
                for name in _THREAD_ENVIRONMENT_KEYS
            },
        },
        "dependencies": {
            distribution: _dependency_version(distribution)
            for distribution in dependencies
        },
    }
    return {
        **payload,
        "fingerprint": "sha256:"
        + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    }


def _cpu_affinity_count() -> int | None:
    affinity = getattr(os, "sched_getaffinity", None)
    if not callable(affinity):
        return None
    try:
        count = len(affinity(0))
    except (OSError, TypeError, ValueError):
        return None
    return int(count) if count > 0 else None


def build_tune_checkpoint_identity(
    *,
    task_id: str,
    dataset_id: str,
    dataset_content_hash: str,
    features: list[str],
    target_col: str,
    split_col: str,
    split_values: dict,
    sample_weight_col: str,
    drop_nan_labels: bool,
    recipe: str,
    n_trials: int,
    seed: int,
    cv_folds: int | None,
    early_stopping_rounds: int,
    max_boost_round: int,
    overfit_penalty: float,
    base_params: dict,
    control_params: dict,
) -> dict[str, Any]:
    """Build the complete identity of one recipe's deterministic search.

    ``features`` deliberately preserves caller order.  ``recipes`` is not an
    argument: adding/removing another arena participant must not invalidate an
    otherwise identical completed recipe.
    """

    return {
        "contract_version": TUNE_CHECKPOINT_CONTRACT_VERSION,
        "task_id": str(task_id),
        "dataset": {
            "id": str(dataset_id),
            "content_hash": str(dataset_content_hash),
        },
        "features": [str(feature) for feature in features],
        "target_col": str(target_col),
        "split_col": str(split_col),
        "split_values": _json_value(split_values),
        "sample_weight_col": str(sample_weight_col or ""),
        "drop_nan_labels": bool(drop_nan_labels),
        "recipe": str(recipe),
        "runtime_fingerprint": tuning_runtime_fingerprint(recipe),
        "n_trials": int(n_trials),
        "seed": int(seed),
        "cv_folds": None if cv_folds is None else int(cv_folds),
        "early_stopping_rounds": int(early_stopping_rounds),
        "max_boost_round": int(max_boost_round),
        "overfit_penalty": float(overfit_penalty),
        "base_params": _json_value(base_params),
        "control_params": _json_value(control_params),
    }


def checkpoint_identity_hash(identity: dict[str, Any]) -> str:
    """Stable, non-sensitive handle for audit correlation."""

    normalized = _json_value(identity)
    return "sha256:" + hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")
    ).hexdigest()


class TuneCheckpointStore:
    """Atomic JSON checkpoint store rooted inside one task directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load(self, recipe: str, identity: dict[str, Any]) -> dict | None:
        """Return an exact result on a valid hit; otherwise silently miss."""

        try:
            path = self._path(recipe)
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict):
                return None
            if envelope.get("contract_version") != TUNE_CHECKPOINT_CONTRACT_VERSION:
                return None
            stored_identity = envelope.get("identity")
            result = envelope.get("result")
            checksum = str(envelope.get("checksum") or "")
            expected_identity = _json_value(identity)
            if stored_identity != expected_identity or not isinstance(result, dict):
                return None
            expected_checksum = _payload_checksum(stored_identity, result)
            if not hmac.compare_digest(checksum, expected_checksum):
                return None
            return _json_value(result)
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def save(self, recipe: str, identity: dict[str, Any], result: dict) -> Path:
        """Atomically replace one recipe checkpoint after full serialization."""

        normalized_identity = _json_value(identity)
        normalized_result = _json_value(result)
        envelope = {
            "contract_version": TUNE_CHECKPOINT_CONTRACT_VERSION,
            "identity": normalized_identity,
            "result": normalized_result,
            "checksum": _payload_checksum(normalized_identity, normalized_result),
        }
        encoded = _canonical_json(envelope).encode("utf-8")
        filename = self._filename(recipe)
        staged = TransactionalArtifactStore(self.root).stage(filename)
        try:
            staged.path.write_bytes(encoded)
            final_path = staged.promote()
            staged.commit()
            return final_path
        except Exception:
            staged.rollback()
            raise

    def _path(self, recipe: str) -> Path:
        return self.root / self._filename(recipe)

    @staticmethod
    def _filename(recipe: str) -> str:
        normalized = str(recipe).strip()
        if not normalized or _RECIPE_NAME.fullmatch(normalized) is None:
            raise ValueError(f"invalid tuning checkpoint recipe: {recipe!r}")
        return f"{normalized}.json"


def _payload_checksum(identity: dict, result: dict) -> str:
    payload = {"identity": identity, "result": result}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _implementation_sha256() -> str:
    base = Path(__file__).parent
    digest = hashlib.sha256()
    for relative in _IMPLEMENTATION_FILES:
        path = (base / relative).resolve()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


@lru_cache(maxsize=1)
def _pack_version() -> str:
    manifest = json.loads((Path(__file__).parent / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["version"])


def _dependency_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not-installed"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: Any) -> Any:
    """Normalize tuples/numpy scalars through strict canonical JSON."""

    return json.loads(_canonical_json(value))


__all__ = [
    "TUNE_CHECKPOINT_CONTRACT_VERSION",
    "TUNE_CHECKPOINT_DIR_NAME",
    "TuneCheckpointStore",
    "build_tune_checkpoint_identity",
    "checkpoint_identity_hash",
    "dataset_content_identity",
    "tuning_runtime_fingerprint",
]
