from __future__ import annotations

import uuid
from pathlib import Path

from marvis.artifacts import ArtifactUnitOfWork, TransactionalArtifactStore
from marvis.data.align import ColumnAligner
from marvis.data.backend import DataBackend
from marvis.data.contracts import (
    LARGE_ROW_THRESHOLD,
    SHRINK_WARN_THRESHOLD,
    SMALL_SAMPLE_N,
    ColumnProfile,
    Dataset,
    JoinDiagnostics,
    JoinPlan,
    JoinSpec,
    KeyAlternative,
    KeyPair,
)
from marvis.data.dedup import two_level_dedup
from marvis.data.errors import (
    DataBackendError,
    DedupRequiredError,
    FanOutError,
    JoinNotConfirmedError,
)


class JoinEngine:
    def __init__(
        self,
        backend: DataBackend,
        aligner: ColumnAligner,
        registry,
        repo,
    ):
        if not callable(getattr(repo, "write_audit", None)):
            raise TypeError("JoinEngine repo must provide write_audit")
        self._backend = backend
        self._aligner = aligner
        self._registry = registry
        self._repo = repo

    def propose_join_plan(
        self,
        anchor_id: str,
        feature_ids: list[str],
        task_id: str,
        *,
        seed: int = 0,
    ) -> JoinPlan:
        anchor = self._registry.get(anchor_id)
        anchor_path = self._registry.resolve_path(anchor_id)
        specs = []
        for feature_id in feature_ids:
            feature = self._registry.get(feature_id)
            feature_path = self._registry.resolve_path(feature_id)
            key_pairs = self._aligner.align(
                anchor,
                anchor_path,
                feature,
                feature_path,
                seed=seed,
            )
            diagnostics = self.diagnose_join(
                anchor,
                anchor_path,
                feature,
                feature_path,
                key_pairs,
                seed=seed,
            )
            specs.append(
                JoinSpec(
                    feature_dataset_id=feature_id,
                    key_pairs=key_pairs,
                    diagnostics=diagnostics,
                    dedup_strategy=None,
                    confirmed=False,
                ),
            )
        plan = JoinPlan(
            id=_new_id("join_plan"),
            task_id=task_id,
            anchor_dataset_id=anchor_id,
            joins=specs,
            status="draft",
        )
        self._repo.create_join_plan(plan)
        return plan

    def diagnose_join(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        key_pairs: list[KeyPair],
        *,
        seed: int,
    ) -> JoinDiagnostics:
        anchor_rows = anchor.row_count
        feature_rows = feature.row_count
        if not key_pairs:
            return JoinDiagnostics(
                anchor_rows=anchor_rows,
                feature_rows=feature_rows,
                feature_key_unique=False,
                matched_rows=0,
                match_rate=0.0,
                joined_rows_preview=0,
                fan_out_detected=False,
                shrink_detected=True,
                new_columns=0,
                new_columns_null_rate=1.0,
            )

        anchor_keys = [pair.anchor_col for pair in key_pairs]
        feature_keys = [pair.feature_col for pair in key_pairs]
        key_unique = self._backend.is_key_unique(feature_path, feature_keys)
        if len(key_pairs) == 1:
            match_rate = key_pairs[0].match_rate
            sampled = min(SMALL_SAMPLE_N, anchor_rows)
            matched = int(round(match_rate * sampled))
        else:
            matched, sampled = self._backend.match_rate_for_method(
                anchor_path,
                anchor_keys,
                feature_path,
                feature_keys,
                method=[pair.match_method for pair in key_pairs],
                key_fingerprints=_key_fps(anchor, feature, key_pairs),
                sample_n=SMALL_SAMPLE_N,
                seed=seed,
            )
            match_rate = matched / sampled if sampled else 0.0

        conflict_report = None
        if key_unique:
            joined_preview = anchor_rows
            fan_out = False
        else:
            distinct_keys = self._backend.distinct_count(feature_path, feature_keys)
            duplicate_factor = feature_rows / max(1, distinct_keys)
            joined_preview = int(
                anchor_rows * match_rate * duplicate_factor
                + anchor_rows * (1 - match_rate),
            )
            fan_out = joined_preview > anchor_rows
            # Break the non-unique key down (spec §6): how many duplicates are safe
            # (whole-row identical) vs genuine same-key value conflicts that must not be
            # silently dropped. Surfaced at the C2 gate so the user resolves consciously.
            if feature_rows <= LARGE_ROW_THRESHOLD:
                _deduped, conflict_report = two_level_dedup(
                    self._backend.read_frame(feature_path), list(feature_keys)
                )
            else:
                try:
                    conflict_report = self._backend.conflict_report(
                        feature_path,
                        list(feature_keys),
                    )
                except DataBackendError:
                    conflict_report = None

        # Dynamic key relaxation (spec §4/§5): the full identity key matched poorly — propose
        # dropping one element to raise the match (with the reduced key's re-checked fan-out).
        # Proposal only; the engine never swaps the key silently.
        key_alternatives: tuple[KeyAlternative, ...] = ()
        if match_rate < SHRINK_WARN_THRESHOLD and len(key_pairs) >= 2:
            key_alternatives = self._relaxation_alternatives(
                anchor, anchor_path, feature, feature_path, key_pairs,
                seed=seed, current_match_rate=match_rate,
            )

        anchor_column_names = {column.name for column in anchor.columns}
        new_columns = len([
            column
            for column in feature.columns
            if column.name not in anchor_column_names
        ])
        return JoinDiagnostics(
            anchor_rows=anchor_rows,
            feature_rows=feature_rows,
            feature_key_unique=key_unique,
            matched_rows=matched,
            match_rate=round(match_rate, 4),
            joined_rows_preview=joined_preview,
            fan_out_detected=fan_out,
            shrink_detected=match_rate < SHRINK_WARN_THRESHOLD,
            new_columns=new_columns,
            new_columns_null_rate=round(1 - match_rate, 4),
            conflict_report=conflict_report,
            key_alternatives=key_alternatives,
        )

    def _relaxation_alternatives(
        self,
        anchor: Dataset,
        anchor_path: Path,
        feature: Dataset,
        feature_path: Path,
        key_pairs: list[KeyPair],
        *,
        seed: int,
        current_match_rate: float,
    ) -> tuple[KeyAlternative, ...]:
        """Drop one element at a time and keep the reduced keys that IMPROVE the match rate
        (spec §4/§5). Each candidate re-checks key-uniqueness + fan-out so the user sees the
        trade-off (a name-only key matches more rows but may fan out)."""
        alternatives: list[KeyAlternative] = []
        for i in range(len(key_pairs)):
            reduced = key_pairs[:i] + key_pairs[i + 1:]
            anchor_keys = [pair.anchor_col for pair in reduced]
            feature_keys = [pair.feature_col for pair in reduced]
            if len(reduced) == 1:
                match_rate = reduced[0].match_rate
            else:
                matched, sampled = self._backend.match_rate_for_method(
                    anchor_path,
                    anchor_keys,
                    feature_path,
                    feature_keys,
                    method=[pair.match_method for pair in reduced],
                    key_fingerprints=_key_fps(anchor, feature, reduced),
                    sample_n=SMALL_SAMPLE_N,
                    seed=seed,
                )
                match_rate = matched / sampled if sampled else 0.0
            # Only propose a relaxation that actually raises the match (else it is strictly worse).
            if match_rate <= current_match_rate:
                continue
            key_unique = self._backend.is_key_unique(feature_path, feature_keys)
            if key_unique:
                fan_out = False
            else:
                distinct_keys = self._backend.distinct_count(feature_path, feature_keys)
                duplicate_factor = feature.row_count / max(1, distinct_keys)
                preview = int(
                    anchor.row_count * match_rate * duplicate_factor
                    + anchor.row_count * (1 - match_rate),
                )
                fan_out = preview > anchor.row_count
            alternatives.append(KeyAlternative(
                key_pairs=tuple((pair.anchor_col, pair.feature_col) for pair in reduced),
                dropped=key_pairs[i].anchor_col,
                match_rate=round(match_rate, 4),
                feature_key_unique=key_unique,
                fan_out_detected=fan_out,
            ))
        alternatives.sort(key=lambda alt: alt.match_rate, reverse=True)
        return tuple(alternatives)

    def confirm_join_spec(
        self,
        join_plan_id: str,
        feature_dataset_id: str,
        *,
        dedup_strategy: str | None,
    ) -> None:
        plan = self._repo.load_join_plan(join_plan_id)
        spec = _find_spec(plan, feature_dataset_id)
        if not spec.diagnostics.feature_key_unique and dedup_strategy in (None, "abort"):
            raise DedupRequiredError(
                f"feature {feature_dataset_id} key is not unique; choose dedup strategy",
            )
        spec.dedup_strategy = dedup_strategy
        spec.confirmed = True
        audit = self._audit_payload(
            kind="join.confirmed",
            target_ref=join_plan_id,
            outcome="confirmed",
            detail={
                "task_id": plan.task_id,
                "anchor_dataset_id": plan.anchor_dataset_id,
                "feature_dataset_id": feature_dataset_id,
                "dedup_strategy": dedup_strategy,
                "match_rate": spec.diagnostics.match_rate,
                "matched_rows": spec.diagnostics.matched_rows,
                "key_pairs": [
                    {
                        "anchor_col": pair.anchor_col,
                        "feature_col": pair.feature_col,
                        "match_method": pair.match_method,
                        "transform_side": pair.transform_side,
                    }
                    for pair in spec.key_pairs
                ],
            },
        )
        update_with_audit = getattr(self._repo, "update_join_spec_with_audit", None)
        if callable(update_with_audit):
            update_with_audit(join_plan_id, spec, audit=audit)
        else:
            self._repo.update_join_spec(join_plan_id, spec)
            self._write_audit(**audit)

    def execute_join_plan(self, join_plan_id: str, *, out_dir: Path) -> Dataset:
        plan = self._repo.load_join_plan(join_plan_id)
        if any(not join.confirmed for join in plan.joins):
            raise JoinNotConfirmedError("all joins must be confirmed before execute")

        anchor = self._registry.get(plan.anchor_dataset_id)
        anchor_rows = anchor.row_count
        current_path = self._registry.resolve_path(plan.anchor_dataset_id)
        artifact_store = TransactionalArtifactStore(Path(out_dir))
        staged_artifacts = []
        for spec in plan.joins:
            if (
                not spec.diagnostics.feature_key_unique
                and spec.dedup_strategy in (None, "abort")
            ):
                raise DedupRequiredError(
                    f"feature {spec.feature_dataset_id} key is not unique; choose dedup strategy",
                )
            feature_path = self._registry.resolve_path(spec.feature_dataset_id)
            artifact = artifact_store.stage(f"join_{uuid.uuid4().hex}.parquet")
            staged_artifacts.append(artifact)
            try:
                joined_rows = self._backend.left_join(
                    current_path,
                    feature_path,
                    spec.key_pairs,
                    dedup_strategy=spec.dedup_strategy,
                    out_path=artifact.path,
                )
            except DataBackendError as exc:
                self._rollback_artifacts(staged_artifacts)
                if "produced" in str(exc) and "anchor" in str(exc):
                    raise FanOutError(str(exc)) from exc
                raise
            # Spec §7: the joined sample must equal the anchor exactly (1:1). The backend
            # already asserts this; re-check defensively, distinguishing fan-out (grow)
            # from silent row loss (shrink) so a mismatch routes back to C2.
            if joined_rows != anchor_rows:
                self._rollback_artifacts(staged_artifacts)
                kind = "fan-out" if joined_rows > anchor_rows else "row loss (shrink)"
                raise FanOutError(
                    f"join {kind}: {joined_rows} rows vs anchor {anchor_rows} (must be 1:1)",
                )
            current_path = artifact.path

        def audit_for(result) -> dict:
            return self._audit_payload(
                kind="join.executed",
                target_ref=join_plan_id,
                outcome="succeeded",
                detail={
                    "task_id": plan.task_id,
                    "anchor_dataset_id": plan.anchor_dataset_id,
                    "result_dataset_id": result.id,
                    "anchor_rows": anchor_rows,
                    "joined_rows": result.row_count,
                    "feature_dataset_ids": [spec.feature_dataset_id for spec in plan.joins],
                    # Column provenance (spec §11): which feature table each contributed column
                    # came from, so downstream FEATURE/MODELING can trace a column's origin.
                    "provenance": [
                        {
                            "feature_dataset_id": spec.feature_dataset_id,
                            "columns": [
                                column.name
                                for column in self._registry.get(spec.feature_dataset_id).columns
                                if column.name not in {pair.feature_col for pair in spec.key_pairs}
                            ],
                        }
                        for spec in plan.joins
                    ],
                },
            )

        final_artifact = staged_artifacts[-1] if staged_artifacts else None
        if final_artifact is None:
            raise DataBackendError("join plan contains no confirmed feature joins")
        for intermediate in staged_artifacts[:-1]:
            intermediate.rollback()
        uow = ArtifactUnitOfWork()
        uow.track(final_artifact)

        def register_result() -> Dataset:
            final_path = final_artifact.final_path
            register_join_result = getattr(self._registry, "register_join_result_with_audit", None)
            if callable(register_join_result):
                return register_join_result(
                    final_path,
                    join_plan_id=join_plan_id,
                    audit_factory=audit_for,
                    task_id=plan.task_id,
                    role="derived",
                    anchor_target=plan.anchor_dataset_id,
                )
            result = self._registry.register_existing(
                final_path,
                task_id=plan.task_id,
                role="derived",
                anchor_target=plan.anchor_dataset_id,
            )
            audit = audit_for(result)
            set_executed_with_audit = getattr(self._repo, "set_join_plan_executed_with_audit", None)
            if callable(set_executed_with_audit):
                set_executed_with_audit(join_plan_id, result.id, audit=audit)
            else:
                self._repo.set_join_plan_executed(join_plan_id, result.id)
                self._write_audit(**audit)
            return result

        result = uow.finalize(register_result)
        plan.status = "executed"
        plan.result_dataset_id = result.id
        return result

    @staticmethod
    def _rollback_artifacts(staged_artifacts) -> None:
        for artifact in reversed(staged_artifacts):
            artifact.rollback()

    def _audit_payload(self, *, kind: str, target_ref: str, outcome: str, detail: dict) -> dict:
        return {
            "kind": kind,
            "target_ref": target_ref,
            "actor": "system",
            "outcome": outcome,
            "detail": detail,
        }

    def _write_audit(
        self,
        *,
        kind: str,
        target_ref: str,
        actor: str = "system",
        outcome: str,
        detail: dict,
    ) -> None:
        self._repo.write_audit(
            kind=kind,
            target_ref=target_ref,
            actor=actor,
            outcome=outcome,
            detail=detail,
        )


def _find_spec(plan: JoinPlan, feature_dataset_id: str) -> JoinSpec:
    for spec in plan.joins:
        if spec.feature_dataset_id == feature_dataset_id:
            return spec
    raise KeyError(f"join spec not found for feature dataset: {feature_dataset_id}")


def _key_fps(
    anchor: Dataset,
    feature: Dataset,
    key_pairs: list[KeyPair],
) -> list[tuple]:
    anchor_profiles = _profiles_by_name(anchor.columns)
    feature_profiles = _profiles_by_name(feature.columns)
    return [
        (
            anchor_profiles[pair.anchor_col].fingerprint,
            feature_profiles[pair.feature_col].fingerprint,
        )
        for pair in key_pairs
    ]


def _profiles_by_name(columns: tuple[ColumnProfile, ...]) -> dict[str, ColumnProfile]:
    return {column.name: column for column in columns}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


__all__ = ["JoinEngine"]
