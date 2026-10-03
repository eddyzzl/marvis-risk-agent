from __future__ import annotations

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
import stat
import time

import pandas as pd
import pytest

from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_custody import (
    CustodyError, capture_case, prepare_custody_run, verify_case_archive,
)
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from test_runtime_agent_benchmark import fixture_model


def simple_capture(tmp_path, **overrides):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "tasks" / "task" / "outputs").mkdir(parents=True)
    (workspace / "tasks" / "task" / "outputs" / "raw.json").write_text('{"metric": 0.73}')
    (workspace / "settings").mkdir()
    (workspace / "settings" / "llm.json").write_text(json.dumps({"models": [{"api_key": "ephemeral-gateway-credential"}]}))
    (workspace / "plugin_admin_token").write_text("native-workspace-verification-key")
    with closing(sqlite3.connect(workspace / "marvis.sqlite")) as conn:
        conn.execute("CREATE TABLE records(id TEXT PRIMARY KEY, value REAL)")
        conn.execute("INSERT INTO records VALUES('original', 0.73)")
        conn.commit()
    data = tmp_path / "data"
    data.mkdir()
    (data / "rows.csv").write_text("x,y\n1,0\n2,1\n")
    case = RuntimeCase.model_validate({
        "id": "one", "revision": "1", "family": "feature_analysis",
        "task": {"task_type": "feature_analysis", "target_col": "y"},
        "materials": [{"path": "rows.csv", "sha256": digest((data / "rows.csv").read_bytes())}],
        "business_constraints_source": "public synthetic test",
    })
    run = prepare_custody_run(tmp_path / "private", "run-one", forbidden=(data,))
    args = dict(run_dir=run, workspace=workspace, dataset_root=data, case=case,
                private={"outputs": {"first": {"metric": 0.73}}, "messages": [{"content": "private narrative"}]},
                bindings={"source": "frozen-source", "expected_sha256": "a" * 64},
                owned_processes_stopped=True)
    args.update(overrides)
    return args


def test_raw_originals_survive_deletion_and_manifest_is_not_a_trust_receipt(tmp_path):
    import shutil

    args = simple_capture(tmp_path)
    receipt = capture_case(**args)
    path = args["run_dir"] / "one"
    shutil.rmtree(args["workspace"])
    shutil.rmtree(args["dataset_root"])
    verified = verify_case_archive(path, expected_manifest_sha256=receipt["manifest_sha256"])
    assert verified["archived_bytes_match_manifest"] is True
    assert verified["acceptance_claim"] == "not_established"
    assert verified["native_key_dependent_reverification"] == "unavailable_original_key_not_exported"
    assert json.loads((path / "private.json").read_text()) == args["private"]
    with closing(sqlite3.connect((path / "workspace/marvis.sqlite").as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT * FROM records").fetchall() == [("original", 0.73)]
    assert not (path / "workspace/settings").exists()
    assert not (path / "workspace/plugin_admin_token").exists()
    for p in path.rglob("*"):
        assert stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600)
        if p.is_file():
            assert b"ephemeral-gateway-credential" not in p.read_bytes()
            assert b"native-workspace-verification-key" not in p.read_bytes()

    raw = path / "workspace/tasks/task/outputs/raw.json"
    raw.write_text('{"metric": 0.99}')
    with pytest.raises(CustodyError, match="custody_originals_changed"):
        verify_case_archive(path, expected_manifest_sha256=receipt["manifest_sha256"])
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["workspace/tasks/task/outputs/raw.json"]["sha256"] = digest(raw.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(CustodyError, match="custody_manifest_changed"):
        verify_case_archive(path, expected_manifest_sha256=receipt["manifest_sha256"])


@pytest.mark.parametrize("mutation", ["symlink", "hardlink", "secret", "credential_column", "live_approval", "live_upload", "live_lease"])
def test_sensitive_or_aliased_originals_fail_without_publishing(tmp_path, mutation):
    args = simple_capture(tmp_path)
    workspace = args["workspace"]
    raw = workspace / "tasks/task/outputs/raw.json"
    if mutation == "symlink":
        (raw.parent / "escape").symlink_to(tmp_path)
    elif mutation == "hardlink":
        (raw.parent / "alias.json").hardlink_to(raw)
    elif mutation == "secret":
        raw.write_text("ephemeral-gateway-credential")
    else:
        with closing(sqlite3.connect(workspace / "marvis.sqlite")) as conn:
            statement = {
                "credential_column": "CREATE TABLE unexpected(api_key TEXT)",
                "live_approval": "CREATE TABLE approval_records(nonce TEXT, status TEXT)",
                "live_upload": "CREATE TABLE validation_batch_material_uploads(upload_token TEXT)",
                "live_lease": "CREATE TABLE operations_periods(lease_token TEXT)",
            }[mutation]
            conn.execute(statement)
            insert = {
                "credential_column": "INSERT INTO unexpected VALUES('do-not-copy')",
                "live_approval": "INSERT INTO approval_records VALUES('capability', 'issued')",
                "live_upload": "INSERT INTO validation_batch_material_uploads VALUES('capability')",
                "live_lease": "INSERT INTO operations_periods VALUES('capability')",
            }[mutation]
            conn.execute(insert)
            conn.commit()
    with pytest.raises(CustodyError):
        capture_case(**args)
    assert list(args["run_dir"].iterdir()) == []


def test_capture_requires_verified_process_stop_and_disjoint_private_destination(tmp_path):
    args = simple_capture(tmp_path)
    args["owned_processes_stopped"] = False
    with pytest.raises(CustodyError, match="custody_processes_not_stopped"):
        capture_case(**args)
    with pytest.raises(CustodyError, match="custody_location_must_be_separate"):
        prepare_custody_run(args["dataset_root"] / "archive", "run", forbidden=(args["dataset_root"],))
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(CustodyError, match="custody_directory_must_be_private"):
        prepare_custody_run(public, "run", forbidden=())
    link = tmp_path / "link"
    link.symlink_to(args["run_dir"])
    with pytest.raises(CustodyError, match="custody_path_alias_rejected"):
        prepare_custody_run(link, "run", forbidden=())


def test_existing_case_and_material_changes_are_not_overwritten(tmp_path):
    args = simple_capture(tmp_path)
    receipt = capture_case(**args)
    with pytest.raises(CustodyError, match="custody_case_collision"):
        capture_case(**args)
    verify_case_archive(args["run_dir"] / "one", expected_manifest_sha256=receipt["manifest_sha256"])
    args["run_dir"] = prepare_custody_run(tmp_path / "other-private", "run-two", forbidden=())
    (args["dataset_root"] / "rows.csv").write_text("x,y\n9,1\n")
    with pytest.raises(CustodyError, match="custody_material_changed"):
        capture_case(**args)
    assert list(args["run_dir"].iterdir()) == []


def test_workspace_mutation_during_snapshot_fails_closed(tmp_path, monkeypatch):
    from marvis.orchestrator.eval import runtime_custody

    args = simple_capture(tmp_path)
    snapshot = runtime_custody._sqlite_snapshot

    def change(source, destination, secrets):
        snapshot(source, destination, secrets)
        (args["workspace"] / "tasks/task/outputs/raw.json").write_text('{"metric": 0.99}')

    monkeypatch.setattr(runtime_custody, "_sqlite_snapshot", change)
    with pytest.raises(CustodyError, match="custody_workspace_changed_during_capture"):
        capture_case(**args)
    assert list(args["run_dir"].iterdir()) == []


def test_actual_http_tools_private_dataset_and_messages_remain_after_temporary_cleanup(tmp_path):
    from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite

    paths = write_synthetic_suite(tmp_path / "suite")
    with fixture_model() as (model, calls):
        report = run_runtime_suite(
            cases_path=paths["cases"], expected_path=paths["expected"], dataset_root=paths["dataset_root"],
            output_dir=tmp_path / "public-runs", evidence_custody_dir=tmp_path / "private-custody",
            model=model, model_source="fixture_model",
        )
    assert calls
    assert report["all_passed"], [(c["case_id"], c.get("evidence_custody"), c["runtime_status"]) for c in report["cases"]]
    assert report["acceptance_claim"] == "not_established"
    assert report["a_evidence_case_count"] == 0
    for case in report["cases"]:
        assert case["isolated_workspace_removed"]
        archive = tmp_path / "private-custody" / report["run_id"] / case["case_id"]
        verify_case_archive(archive, expected_manifest_sha256=case["evidence_custody"]["manifest_sha256"])
        manifest = json.loads((archive / "manifest.json").read_text())
        assert not Path(manifest["original_workspace"]).exists()
        assert json.loads((archive / "private.json").read_text())["messages"]
        bindings = json.loads((archive / "bindings.json").read_text())
        assert bindings["source"] == report["source"]
        assert bindings["cases_sha256"] == report["cases_sha256"]
        assert bindings["expected_sha256"] == report["expected_sha256"]
        assert bindings["execution_before_scoring"]["execution"]["task_id"] == case["execution"]["task_id"]
        originals = list((archive / "workspace/datasets").rglob("*.parquet"))
        assert originals
        # Re-read native raw data after deletion, not a saved success flag.
        assert all(len(pd.read_parquet(p)) == 80 for p in originals)
        with closing(sqlite3.connect((archive / "workspace/marvis.sqlite").as_uri() + "?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM plan_steps WHERE status='done'").fetchone()[0] > 0
        public = Path(report["report_path"]).read_text()
        assert "private narrative" not in public
        assert str(archive) not in public


def test_wal_committed_rows_survive_without_changing_source_database(tmp_path):
    args = simple_capture(tmp_path)
    database = args["workspace"] / "marvis.sqlite"
    with closing(sqlite3.connect(database)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO records VALUES('committed-in-wal', 0.91)")
        writer.commit()
        before = {p.name: p.read_bytes() for p in database.parent.glob("marvis.sqlite*") if not p.name.endswith("-shm")}
        receipt = capture_case(**args)
        assert {p.name: p.read_bytes() for p in database.parent.glob("marvis.sqlite*") if not p.name.endswith("-shm")} == before
    archive = args["run_dir"] / "one"
    verify_case_archive(archive, expected_manifest_sha256=receipt["manifest_sha256"])
    with closing(sqlite3.connect(archive / "workspace/marvis.sqlite")) as conn:
        assert conn.execute("SELECT value FROM records WHERE id='committed-in-wal'").fetchone() == (0.91,)


def test_cli_custody_is_explicit_and_is_forwarded_without_changing_defaults(tmp_path, monkeypatch):
    from marvis.__main__ import _parse_args
    from marvis.orchestrator.eval.cli import run_eval_agent_cli
    from marvis.orchestrator.eval import runtime_runner
    from marvis.orchestrator.eval.runtime_contracts import ModelConnection

    config = tmp_path / "model.json"
    config.write_text(ModelConnection(model_id="fixture", model_name="fixture", api_base_url="http://127.0.0.1:1234/v1").model_dump_json())
    argv = ["eval-agent", "--cases", str(tmp_path / "cases.json"), "--expected", str(tmp_path / "expected.json"),
            "--dataset-root", str(tmp_path / "data"), "--output-dir", str(tmp_path / "public"),
            "--model-config", str(config), "--model-source", "fixture_model"]
    captured = []
    monkeypatch.setattr(runtime_runner, "run_runtime_suite", lambda **kwargs: captured.append(kwargs) or {})
    run_eval_agent_cli(_parse_args(argv))
    run_eval_agent_cli(_parse_args(argv + ["--evidence-custody-dir", str(tmp_path / "private")]))
    assert captured[0]["evidence_custody_dir"] is None
    assert captured[1]["evidence_custody_dir"] == tmp_path / "private"
    assert {k: v for k, v in captured[0].items() if k != "evidence_custody_dir"} == {k: v for k, v in captured[1].items() if k != "evidence_custody_dir"}


@pytest.mark.parametrize("failure", ["unconfirmed_stop", "archive_io", "interrupt"])
def test_capture_failure_keeps_execution_record_and_temporary_cleanup(tmp_path, monkeypatch, failure):
    from marvis.orchestrator.eval import runtime_runner, runtime_custody

    args = simple_capture(tmp_path)

    @contextmanager
    def application(root, *unused, process_state, **kwargs):
        workspace = root / "workspace"
        workspace.mkdir()
        process_state["started"] = True
        try:
            yield None, workspace, time.monotonic() + 10
        finally:
            process_state["stopped"] = failure != "unconfirmed_stop"

    if failure != "unconfirmed_stop":
        def fail_capture(**kwargs):
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise OSError("sensitive path or body must not be published")
        monkeypatch.setattr(runtime_custody, "capture_case", fail_capture)
    monkeypatch.setattr(runtime_runner, "_application", application)
    monkeypatch.setattr(runtime_runner.Journey, "run", lambda *args: None)
    monkeypatch.setattr(runtime_runner, "_receipts", lambda *args, **kwargs: ({"plans": [], "steps": []}, {"outputs": {}, "messages": []}))
    case_dir = tmp_path / "public-case"
    case_dir.mkdir()
    record, _ = runtime_runner._run_case(
        args["case"], args["dataset_root"], {}, case_dir, {}, "a" * 64, (),
        custody_run_dir=args["run_dir"], custody_bindings={},
    )
    assert record["evidence_custody"]["status"] == "failed"
    assert record["runtime_status"] == ("interrupted" if failure == "interrupt" else "receipt_error")
    assert record["isolated_workspace_removed"]
    assert "sensitive path" not in json.dumps(record)
    assert list(args["run_dir"].iterdir()) == []


@pytest.mark.parametrize("limit", ["declared_count", "declared_bytes", "actual_bytes"])
def test_verification_limits_apply_before_hashing_large_or_excess_originals(tmp_path, monkeypatch, limit):
    from marvis.orchestrator.eval import runtime_custody

    args = simple_capture(tmp_path)
    receipt = capture_case(**args)
    archive = args["run_dir"] / "one"
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original_hash = runtime_custody._file_identity
    if limit == "declared_count":
        monkeypatch.setattr(runtime_custody, "MAX_FILES", len(manifest["files"]) - 1)
    elif limit == "declared_bytes":
        manifest["files"]["private.json"]["size_bytes"] = runtime_custody.MAX_BYTES + 1
        manifest_path.write_text(json.dumps(manifest))
        receipt["manifest_sha256"] = digest(manifest_path.read_bytes())
    else:
        too_large = archive / "private.json"
        too_large.write_bytes(too_large.read_bytes() + b"unexpected growth")

    def guarded_hash(path, *args):
        assert limit == "actual_bytes" and path != archive / "private.json", "oversized input must be rejected before hashing"
        return original_hash(path, *args)

    monkeypatch.setattr(runtime_custody, "_file_identity", guarded_hash)
    with pytest.raises(CustodyError, match="custody_inventory_limit|custody_originals_changed"):
        verify_case_archive(archive, expected_manifest_sha256=receipt["manifest_sha256"])


def test_physical_database_and_wal_copy_use_byte_bound(tmp_path, monkeypatch):
    from marvis.orchestrator.eval import runtime_custody

    args = simple_capture(tmp_path)
    db = args["workspace"] / "marvis.sqlite"
    calls = []
    original_copy = runtime_custody._copy_bounded

    def tracked_copy(source, destination, expected_size):
        calls.append((source.name, expected_size))
        return original_copy(source, destination, expected_size)

    monkeypatch.setattr(runtime_custody, "_copy_bounded", tracked_copy)
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("INSERT INTO records VALUES('wal-copy', 0.52)")
        writer.commit()
        capture_case(**args)
    assert {name for name, _ in calls} >= {"marvis.sqlite", "marvis.sqlite-wal"}
    with pytest.raises(CustodyError, match="custody_source_changed"):
        original_copy(db, tmp_path / "bounded-copy", db.stat().st_size - 1)
