from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from filelock import FileLock
import numpy as np
import pytest

from marvis.artifacts import ArtifactUnitOfWork
from marvis.db_schema import connect
from marvis.feature.metrics import feature_auc
from marvis.files import sha256_file
from marvis.packs.modeling import evidence_tools
from marvis.packs.modeling import score_evidence_tools
from marvis.packs.modeling.errors import ModelingError
from marvis.packs.modeling.evidence import (
    MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
)
from marvis.packs.modeling.score_evidence import (
    MODEL_SCORE_INPUT_SPACE,
    ModelScoreEvidenceError,
    build_model_score_evidence_envelope,
    build_single_model_score_evidence,
    validate_model_score_evidence_envelope,
)
from marvis.packs.modeling.score_evidence_tools import (
    MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND,
    MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
    MODEL_SCORE_VECTOR_ARTIFACT_KIND,
    _model_score_task_lock_path,
    load_model_score_evidence_artifacts,
    require_model_score_evidence_artifact_binding_on_connection,
    run_materialize_model_score_evidence_v2,
    validate_materialize_model_score_evidence_v2_tool_output,
)
from marvis.packs.modeling.scoring import _ModelArtifactScorer
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_modeling_training_evidence_tool import (
    _binding,
    _fixture,
    _run as run_training,
)


def _score_inputs(fx: dict, training_output: dict) -> dict:
    return {
        "training_evidence_ref": evidence_tools.build_training_evidence_ref(
            _binding(fx, training_output)
        )
    }


def _run_score(fx: dict, training_output: dict) -> dict:
    return run_materialize_model_score_evidence_v2(
        _score_inputs(fx, training_output),
        fx["ctx"],
        fx["runtime"],
    )


def _score_records(fx: dict) -> list[dict]:
    return [
        item
        for item in TaskArtifactRepository(fx["settings"].db_path).list_for_task(
            fx["task"].id
        )
        if item["kind"]
        in {
            MODEL_SCORE_VECTOR_ARTIFACT_KIND,
            MODEL_SCORE_EVIDENCE_ARTIFACT_KIND,
        }
    ]


def test_lr_score_evidence_matches_direct_same_space_scorer(tmp_path: Path) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)
    training = _binding(fx, training_output)

    output = _run_score(fx, training_output)

    assert output["input_space"] == MODEL_SCORE_INPUT_SPACE
    assert output["governance"] == {
        "not_compared": True,
        "not_selected": True,
        "not_adopted": True,
        "not_deployed": True,
    }
    assert len(_score_records(fx)) == 2
    assert (
        validate_materialize_model_score_evidence_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )
        == output
    )
    loaded = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"]["score_evidence"][
            "content_hash"
        ],
        score_vector_artifact_id=output["artifacts"]["score_vector"]["artifact_id"],
        expected_score_vector_artifact_content_hash=output["artifacts"]["score_vector"][
            "content_hash"
        ],
    )
    frame = fx["runtime"].backend.read_frame(
        training.sample.source_binding.dataset_path
    )
    direct = np.asarray(
        _ModelArtifactScorer(
            training.model_artifact,
            base_dir=training.model_binary_path.parent,
            load_calibration=False,
            replay_preprocessing=False,
        ).score(frame, use_calibration=False),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(loaded.vector.scores, direct)
    assert loaded.envelope["scoring_contract"] == {
        "input_space": MODEL_SCORE_INPUT_SPACE,
        "load_calibration": False,
        "replay_preprocessing": False,
        "rows_scored_exactly_once": True,
        "row_ordinal": {
            "start": 0,
            "stop": len(frame),
            "step": 1,
        },
        "score_direction": "higher_is_riskier",
    }


def test_score_evidence_maps_reversed_raw_target_to_bad_probability(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path, target_bad_value=0)
    training_output = run_training(fx)
    output = _run_score(fx, training_output)
    loaded = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"]["score_evidence"][
            "content_hash"
        ],
    )
    training = loaded.training
    frame = fx["runtime"].backend.read_frame(
        training.sample.source_binding.dataset_path
    )
    mask = training.sample.membership["masks"]["risk/development"]
    expected_auc = feature_auc(
        loaded.vector.scores[mask],
        (frame.loc[mask, "bad"].to_numpy() == 0).astype(np.int8),
        direction_agnostic=False,
    )
    observations = loaded.envelope["single_model_evidence"]["observations"]
    observed_auc = next(
        item["value"]
        for item in observations
        if item["metric_key"] == "auc"
        and item["period"] is None
        and item["sample_ref"]["population"] == "risk"
        and item["sample_ref"]["partition"] == "development"
    )
    assert observed_auc == pytest.approx(expected_auc)


def test_envelope_validator_rejects_substituted_training_artifact_id(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    output = _run_score(fx, run_training(fx))
    binding = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"]["score_evidence"][
            "content_hash"
        ],
    )
    authenticated_ref = evidence_tools.build_training_evidence_ref(binding.training)
    forged_ref = deepcopy(authenticated_ref)
    forged_ref["evidence_artifact_id"] = "f" * 64
    if forged_ref["evidence_artifact_id"] == authenticated_ref["evidence_artifact_id"]:
        forged_ref["evidence_artifact_id"] = "e" * 64
    frame = fx["runtime"].backend.read_frame(
        binding.training.sample.source_binding.dataset_path
    )
    forged_single = build_single_model_score_evidence(
        sample_design_bundle=binding.training.sample.bundle,
        membership_masks=binding.training.sample.membership["masks"],
        frame=frame,
        scores=binding.vector.scores,
        training_evidence_ref={
            "kind": MODELING_TRAINING_EVIDENCE_ARTIFACT_KIND,
            "ref_id": forged_ref["evidence_artifact_id"],
            "content_hash": forged_ref["expected_evidence_artifact_content_hash"],
        },
        model_ref=binding.envelope["model_ref"],
        score_ref=binding.envelope["score_vector_ref"],
        features=binding.training.evidence["training_contract"]["features"],
    )
    forged = build_model_score_evidence_envelope(
        task_id=binding.task_id,
        training_evidence_ref=forged_ref,
        training_evidence=binding.training.evidence,
        sample_design_bundle=binding.training.sample.bundle,
        model_ref=binding.envelope["model_ref"],
        score_ref=binding.envelope["score_vector_ref"],
        score_vector=binding.vector,
        single_model_evidence=forged_single,
    )

    with pytest.raises(ModelScoreEvidenceError, match="authenticated evidence"):
        validate_model_score_evidence_envelope(
            forged,
            sample_design_bundle=binding.training.sample.bundle,
            training_evidence=binding.training.evidence,
            expected_training_evidence_ref=authenticated_ref,
            score_vector=binding.vector,
        )


def test_model_is_deserialized_only_from_verified_private_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)
    training = _binding(fx, training_output)
    original_init = score_evidence_tools._ModelArtifactScorer.__init__
    checked = False

    def checking_init(self, artifact, *, base_dir, **kwargs):
        nonlocal checked
        private_path = Path(base_dir) / artifact.model_path
        assert Path(base_dir) != training.model_binary_path.parent
        assert Path(base_dir).name.startswith("run.")
        assert private_path.exists()
        assert sha256_file(private_path) == training.model_binary_record["content_hash"]
        assert kwargs == {
            "load_calibration": False,
            "replay_preprocessing": False,
        }
        checked = True
        return original_init(self, artifact, base_dir=base_dir, **kwargs)

    monkeypatch.setattr(
        score_evidence_tools._ModelArtifactScorer,
        "__init__",
        checking_init,
    )
    _run_score(fx, training_output)
    assert checked is True


def test_model_source_toctou_fails_before_publication_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)
    training = _binding(fx, training_output)
    original_score = score_evidence_tools._ModelArtifactScorer.score

    def score_then_replace_source(self, dataframe, *, use_calibration=True):
        result = original_score(
            self,
            dataframe,
            use_calibration=use_calibration,
        )
        training.model_binary_path.write_bytes(b"changed-after-private-load")
        return result

    monkeypatch.setattr(
        score_evidence_tools._ModelArtifactScorer,
        "score",
        score_then_replace_source,
    )
    with pytest.raises(ModelingError, match="model binary|artifact bytes"):
        _run_score(fx, training_output)
    assert _score_records(fx) == []
    out_dir = Path(fx["settings"].tasks_dir) / fx["task"].id / "model_score_evidence"
    assert not list(out_dir.glob("*.parquet"))
    assert not list(out_dir.glob("*.json"))


def test_live_output_reload_rejects_cached_scalar_or_vector_tamper(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    output = _run_score(fx, run_training(fx))
    forged = deepcopy(output)
    forged["single_model_evidence_id"] = "strategy-model-evidence-forged"
    with pytest.raises(ModelingError, match="drift|invalid|match"):
        validate_materialize_model_score_evidence_v2_tool_output(
            forged,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )

    vector_record = next(
        item
        for item in _score_records(fx)
        if item["kind"] == MODEL_SCORE_VECTOR_ARTIFACT_KIND
    )
    Path(vector_record["path"]).write_bytes(
        Path(vector_record["path"]).read_bytes() + b"tamper"
    )
    with pytest.raises(ModelingError, match="hash|changed|size|Parquet"):
        validate_materialize_model_score_evidence_v2_tool_output(
            output,
            runtime=fx["runtime"],
            task_id=fx["task"].id,
        )


def test_transaction_revalidator_rejects_task_artifact_row_toctou_and_rolls_back(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    output = _run_score(fx, run_training(fx))
    binding = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"]["score_evidence"][
            "content_hash"
        ],
    )

    with connect(fx["settings"].db_path) as conn:
        with pytest.raises(ModelingError, match="active transaction"):
            require_model_score_evidence_artifact_binding_on_connection(conn, binding)

    with pytest.raises(ModelingError, match="disappeared before commit"):
        with fx["runtime"].task_artifacts.transaction() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM task_artifacts WHERE id = ?",
                (binding.vector_record["id"],),
            )
            require_model_score_evidence_artifact_binding_on_connection(conn, binding)

    with fx["runtime"].task_artifacts.transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        require_model_score_evidence_artifact_binding_on_connection(conn, binding)
        conn.rollback()
    assert (
        load_model_score_evidence_artifacts(
            fx["runtime"],
            task_id=fx["task"].id,
            evidence_artifact_id=binding.evidence_record["id"],
            expected_evidence_artifact_content_hash=binding.evidence_record[
                "content_hash"
            ],
        ).envelope
        == binding.envelope
    )


def test_transaction_revalidator_rejects_score_evidence_file_toctou(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    output = _run_score(fx, run_training(fx))
    binding = load_model_score_evidence_artifacts(
        fx["runtime"],
        task_id=fx["task"].id,
        evidence_artifact_id=output["artifacts"]["score_evidence"]["artifact_id"],
        expected_evidence_artifact_content_hash=output["artifacts"]["score_evidence"][
            "content_hash"
        ],
    )
    original = binding.evidence_path.read_bytes()
    try:
        binding.evidence_path.write_bytes(original + b" ")
        with pytest.raises(ModelingError, match="hash|changed"):
            with fx["runtime"].task_artifacts.transaction() as conn:
                conn.execute("BEGIN IMMEDIATE")
                require_model_score_evidence_artifact_binding_on_connection(
                    conn,
                    binding,
                )
    finally:
        binding.evidence_path.write_bytes(original)

    assert (
        load_model_score_evidence_artifacts(
            fx["runtime"],
            task_id=fx["task"].id,
            evidence_artifact_id=binding.evidence_record["id"],
            expected_evidence_artifact_content_hash=binding.evidence_record[
                "content_hash"
            ],
        ).envelope
        == binding.envelope
    )


@pytest.mark.parametrize("failure", ["vector_register", "json_register", "audit"])
def test_score_evidence_precommit_failure_rolls_back_files_rows_and_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)
    original_register = fx["runtime"].task_artifacts.register_on_connection
    calls = 0

    def maybe_fail_register(conn, **kwargs):
        nonlocal calls
        calls += 1
        if failure == "vector_register" and calls == 1:
            raise RuntimeError("vector register down")
        if failure == "json_register" and calls == 2:
            raise RuntimeError("json register down")
        return original_register(conn, **kwargs)

    if failure in {"vector_register", "json_register"}:
        monkeypatch.setattr(
            fx["runtime"].task_artifacts,
            "register_on_connection",
            maybe_fail_register,
        )
    else:
        monkeypatch.setattr(
            fx["runtime"].repo,
            "write_audit_on_connection",
            lambda conn, **kwargs: (_ for _ in ()).throw(
                RuntimeError("score evidence audit down")
            ),
        )

    with pytest.raises(RuntimeError, match="down"):
        _run_score(fx, training_output)

    assert _score_records(fx) == []
    out_dir = Path(fx["settings"].tasks_dir) / fx["task"].id / "model_score_evidence"
    assert not list(out_dir.glob("*.parquet"))
    assert not list(out_dir.glob("*.json"))
    with connect(fx["settings"].db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND,),
        ).fetchone()[0]
    assert count == 0


def test_score_evidence_is_idempotent_and_same_task_lock_fails_fast(
    tmp_path: Path,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)
    first = _run_score(fx, training_output)
    second = _run_score(fx, training_output)

    assert second == first
    assert len(_score_records(fx)) == 2
    with connect(fx["settings"].db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM audit WHERE kind = ?",
            (MATERIALIZE_MODEL_SCORE_EVIDENCE_V2_AUDIT_KIND,),
        ).fetchone()[0]
    assert count == 1

    lock_path = _model_score_task_lock_path(
        fx["settings"].tasks_dir,
        task_id=fx["task"].id,
    )
    with FileLock(str(lock_path), timeout=0):
        with pytest.raises(ModelingError, match="already running"):
            _run_score(fx, training_output)


def test_database_commit_failure_rolls_back_score_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)

    class FailingCommitConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def commit(self):
            raise RuntimeError("score evidence commit down")

    @contextmanager
    def failing_transaction():
        with connect(fx["settings"].db_path) as conn:
            yield FailingCommitConnection(conn)

    monkeypatch.setattr(
        fx["runtime"].task_artifacts,
        "transaction",
        failing_transaction,
    )
    with pytest.raises(RuntimeError, match="commit down"):
        _run_score(fx, training_output)

    assert _score_records(fx) == []


def test_postcommit_cleanup_warning_does_not_reverse_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fx = _fixture(tmp_path)
    training_output = run_training(fx)

    monkeypatch.setattr(
        ArtifactUnitOfWork,
        "commit",
        lambda self: (_ for _ in ()).throw(RuntimeError("cleanup down")),
    )

    output = _run_score(fx, training_output)
    assert output["evidence_id"].startswith("model-score-evidence-")
    assert len(_score_records(fx)) == 2
