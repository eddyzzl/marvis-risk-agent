"""Fit the complete candidate-selection policy inside an explicit CV fold.

This numerical provider proves only the supplied candidate algorithm's scope.
Its caller owns candidate provenance, fold construction, publication and final
held-out evaluation. No OOT frame or labels are exposed by this interface.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from marvis.data.authenticated_snapshot import authenticated_file_snapshot, arrow_integer_dtype
from marvis.feature.screen import screen_features, screen_features_non_binary
from marvis.feature.transform import normalize_sentinel_value, mask_sentinel_values
from marvis.packs.modeling.errors import ModelingError, SpecialValueDecisionRequiredError
from marvis.packs.modeling.select import select_features


@dataclass(frozen=True)
class FoldSelectionResult:
    features: tuple[str, ...]
    train: pd.DataFrame
    fit: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    evidence: dict


@dataclass(frozen=True)
class FinalSelectionResult:
    features: tuple[str, ...]
    train: pd.DataFrame
    evidence: dict


def _positions(value, row_count, *, role):
    positions = np.asarray(value)
    if (positions.ndim != 1 or positions.dtype.kind not in "iu"
            or not len(positions) or (positions < 0).any()
            or (positions >= row_count).any()
            or len(np.unique(positions)) != len(positions)):
        raise ModelingError(f"{role} requires unique native source row positions")
    return positions.astype(np.int64, copy=True)


def _members(positions, row_count):
    mask = np.zeros(row_count, dtype=bool)
    mask[positions] = True
    raw = np.packbits(mask, bitorder="little").tobytes()
    return {"rows": len(positions), "membership": base64.b64encode(raw).decode(),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _effective_parameters(function, parameters):
    bound = inspect.signature(function).bind(None, None, **parameters)
    bound.apply_defaults()
    return {key: value for key, value in bound.arguments.items() if key not in {"backend", "dataset_path"}}


class _ProjectedBackend:
    def __init__(self, parquet, positions, columns, policies):
        self.parquet = parquet
        self.positions = positions
        self.columns = tuple(columns)
        self.policies = policies

    def column_names(self, _path):
        return self.columns

    def read_frame(self, _path, *, columns=None):
        columns = list(self.columns if columns is None else columns)
        if not set(columns) <= set(self.columns):
            raise ModelingError("fold selection requested an undeclared column")
        # Project rows before conversion to pandas. A wide candidate universe
        # is read in the selector's bounded column batches, never one full frame.
        ordered = np.sort(self.positions)
        chunks, offset = [], 0
        for batch in self.parquet.iter_batches(batch_size=4096, columns=columns, use_threads=False):
            end = offset + batch.num_rows
            left, right = np.searchsorted(ordered, [offset, end])
            members = ordered[left:right]
            if len(members):
                frame = batch.take(members - offset).to_pandas(types_mapper=arrow_integer_dtype)
                frame.index = members
                chunks.append(frame)
            offset = end
        frame = pd.concat(chunks).loc[self.positions, columns]
        for column in columns:
            policy = self.policies.get(column, {})
            if policy.get("action") == "mask":
                frame[column] = mask_sentinel_values(frame[column], policy["values"])
        return frame


class FoldSelectionSession:
    """One authenticated private snapshot, reused by each fold's fit-only policy."""

    def __init__(self, registry, dataset_id, *, candidates, target_col,
                 target_type="binary", sample_weight_col=None, control_columns=(),
                 screen_parameters=None, selection_parameters=None, special_value_policy=None):
        self.registry, self.source = registry, registry.get(dataset_id)
        self.target_col, self.target_type = target_col, target_type
        self.weight_col = sample_weight_col
        controls = {target_col, sample_weight_col, *control_columns}
        self.candidates = tuple(candidates)
        if (not self.candidates or len(set(self.candidates)) != len(self.candidates)
                or any(not isinstance(name, str) or not name for name in self.candidates)
                or set(self.candidates) & controls):
            raise ModelingError("fold selection requires distinct non-control candidate columns")
        if target_type not in {"binary", "continuous", "multiclass"}:
            raise ModelingError("unsupported fold selection target type")
        # JSON copies isolate nested caller mutation and reject non-finite policy
        # settings rather than hashing a different effective configuration.
        self.screen_parameters = json.loads(json.dumps(screen_parameters or {}, allow_nan=False))
        self.selection_parameters = json.loads(json.dumps(selection_parameters or {}, allow_nan=False))
        self.policies = json.loads(json.dumps(special_value_policy or {}, allow_nan=False))
        allowed_screen = {"leakage_ks", "max_missing_rate", "top_k", "batch_size", "drop_nan_labels"}
        allowed_selection = {"iv_min", "corr_max", "vif_max", "top_k", "seed", "drop_nan_labels",
                             "space", "scorecard_max_bins", "enforce_monotonic",
                             "monotonic_direction_request", "sign_check", "batch_size",
                             "multivariate_sample_rows"}
        if (set(self.screen_parameters) - allowed_screen
                or set(self.selection_parameters) - allowed_selection):
            raise ModelingError("fold selection cannot override row scope or diagnostic partitions")
        self.screen_parameters.setdefault("top_k", 200)
        width = self.screen_parameters["top_k"]
        if type(width) is not int or not 1 <= width <= 200:
            raise ModelingError("fold screen top_k must be between 1 and 200")
        for parameters in (self.screen_parameters, self.selection_parameters):
            parameters.setdefault("batch_size", 16)
            width = parameters["batch_size"]
            if type(width) is not int or not 1 <= width <= 32:
                raise ModelingError("fold selection batch_size must be between 1 and 32")
        for name, policy in self.policies.items():
            if (name not in self.candidates or not isinstance(policy, dict)
                    or policy.get("action") not in {"mask", "retain", "drop"}
                    or not isinstance(policy.get("values"), list)
                    or not policy["values"]
                    or any(type(value) not in {float, int} or not np.isfinite(value)
                           for value in policy["values"])
                    or (policy["action"] == "retain" and
                        (policy.get("confirmed") is not True or not str(policy.get("reason") or "").strip()))):
                raise ModelingError("fold special-value policy is incomplete")
        self._snapshot, self._parquet = None, None

    def __enter__(self):
        if self._parquet is not None:
            raise ModelingError("fold selection session is already open")
        self._snapshot = authenticated_file_snapshot(
            self.registry.resolve_verified_path(self.source.id),
            root=self.registry.datasets_root, expected_sha256=self.source.content_hash,
        )
        stream = self._snapshot.__enter__()
        try:
            self._parquet = pq.ParquetFile(stream)
            required = {*self.candidates, self.target_col}
            if self.weight_col:
                required.add(self.weight_col)
            if (self._parquet.metadata.num_rows != self.source.row_count
                    or not required <= set(self._parquet.schema_arrow.names)):
                raise ModelingError("fold source schema or row identity differs from its registration")
        except Exception:
            import sys
            self._snapshot.__exit__(*sys.exc_info())
            self._parquet = None
            raise
        return self

    def __exit__(self, *exception):
        self._parquet = None
        return self._snapshot.__exit__(*exception)

    def training_controls(self, *, split_col, train_value, group_columns=()):
        """Read split membership first, then labels/controls only for outer train.

        Group keys keep their source dtype and source row index. Missing optional
        group columns retain the existing row-wise CV fallback contract.
        """
        if self._parquet is None:
            raise ModelingError("fold selection requires an open authenticated session")
        names = set(self._parquet.schema_arrow.names)
        if split_col not in names:
            raise ModelingError("fold source is missing its declared split column")
        all_positions = np.arange(self.source.row_count)
        splits = _ProjectedBackend(self._parquet, all_positions, [split_col], {}).read_frame(None)
        positions = np.flatnonzero(splits[split_col].eq(train_value).fillna(False).to_numpy(dtype=bool))
        if not len(positions):
            raise ModelingError("fold source has no rows in the declared training partition")
        columns = list(dict.fromkeys([
            self.target_col, *([self.weight_col] if self.weight_col else []),
            *[name for name in group_columns if name in names],
        ]))
        return _ProjectedBackend(self._parquet, positions, columns, {}).read_frame(None)

    def prepare_fold(self, *, fit_positions, train_positions, valid_positions, test_positions):
        if self._parquet is None:
            raise ModelingError("fold selection requires an open authenticated session")
        roles = {role: _positions(value, self.source.row_count, role=role) for role, value in {
            "fit": fit_positions, "train": train_positions,
            "valid": valid_positions, "test": test_positions,
        }.items()}
        fit, train, valid, test = (set(roles[key]) for key in ("fit", "train", "valid", "test"))
        if (not fit <= train or fit & valid or train & test
                or not ((valid <= train and train == fit | valid) or (valid == test and fit == train))):
            raise ModelingError("fold fit, early-stopping and validation memberships overlap or disagree")
        features, parameters = self._select_fit(roles["fit"])
        columns = self._model_columns(features)
        train_frame = _ProjectedBackend(self._parquet, roles["train"], columns, self.policies).read_frame(None)
        test_frame = _ProjectedBackend(self._parquet, roles["test"], columns, self.policies).read_frame(None)
        valid_frame = train_frame.loc[roles["valid"]] if valid <= train else test_frame.loc[roles["valid"]]
        evidence = self._evidence(features, parameters, roles)
        return FoldSelectionResult(features, train_frame,
                                   train_frame.loc[roles["fit"]], valid_frame, test_frame, evidence)

    def prepare_final_fit(self, positions):
        """Refit the frozen selection algorithm on outer train, without a holdout read."""
        if self._parquet is None:
            raise ModelingError("fold selection requires an open authenticated session")
        positions = _positions(positions, self.source.row_count, role="final_fit")
        features, parameters = self._select_fit(positions)
        frame = _ProjectedBackend(self._parquet, positions,
            self._model_columns(features), self.policies).read_frame(None)
        return FinalSelectionResult(features, frame,
            self._evidence(features, parameters, {"fit": positions, "train": positions}))

    def _model_columns(self, features):
        return list(dict.fromkeys([*features, self.target_col, *([self.weight_col] if self.weight_col else [])]))

    def _select_fit(self, positions):
        candidates = [name for name in self.candidates if self.policies.get(name, {}).get("action") != "drop"]
        controls = [self.target_col, *([self.weight_col] if self.weight_col else [])]
        backend = _ProjectedBackend(self._parquet, positions, [*candidates, *controls], {})
        parameters = dict(self.screen_parameters)
        if self.target_type == "binary":
            screen = screen_features
        else:
            screen = screen_features_non_binary
            parameters.pop("leakage_ks", None)
            parameters["target_type"] = self.target_type
        screen_arguments = {"features": candidates, "target_col": self.target_col,
                            "sample_weight_col": self.weight_col, **parameters}
        result = screen(backend, None, **screen_arguments)
        problems = {}
        for name in result.selected:
            observed = {normalize_sentinel_value(value[0]) for value in result.sentinel_columns.get(name, [])}
            policy = self.policies.get(name, {})
            if observed - set(policy.get("values", [])):
                problems[name] = "fold_detection_not_covered_by_frozen_policy"
        if problems:
            raise SpecialValueDecisionRequiredError(columns=sorted(problems),
                sentinel_columns=result.sentinel_columns, problems=problems)
        backend.policies = self.policies
        selection_arguments = {"features": list(result.selected), "target_col": self.target_col,
            "target_type": self.target_type, "allow_full_fit": True, **self.selection_parameters}
        selected = select_features(backend, None, **selection_arguments)
        if not selected.selected:
            raise ModelingError("fold selection produced no usable features")
        return tuple(selected.selected), {
            "screen_parameters": _effective_parameters(screen, screen_arguments),
            "selection_parameters": _effective_parameters(select_features, selection_arguments),
        }

    def _evidence(self, features, parameters, roles):
        evidence = {
            "schema_version": "marvis.fold_selection.v1", "source_dataset_id": self.source.id,
            "source_content_hash": self.source.content_hash, "row_count": self.source.row_count,
            "candidates": list(self.candidates), "selected": list(features), **parameters,
            "special_value_policy": self.policies,
            "memberships": {role: _members(positions, self.source.row_count) for role, positions in roles.items()},
            "scope": "supplied_candidate_algorithm_fit_only",
            "upstream_candidate_selection": "unknown", "historical_availability": "unknown",
        }
        # Returned reports must not be able to mutate this session's next fold.
        evidence = json.loads(json.dumps(evidence, allow_nan=False))
        evidence["sha256"] = hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return evidence
