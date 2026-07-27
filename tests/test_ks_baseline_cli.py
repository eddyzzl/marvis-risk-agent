from __future__ import annotations

import json
from pathlib import Path

from scripts import ks_baseline
from support.ks_harness import KSRunResult


def _baseline_file(path: Path) -> Path:
    payload = {
        "_meta": {},
        "give_me_some_credit": {"baseline_ks": None},
        "home_credit": {"baseline_ks": None},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_status_is_blocking_when_public_files_and_ground_truth_are_missing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(ks_baseline, "_DATASETS_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(
        ks_baseline, "_BASELINES_PATH", _baseline_file(tmp_path / "baselines.json")
    )

    assert ks_baseline.main(["--status"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked"
    assert set(payload["datasets"]) == {"give_me_some_credit", "home_credit"}
    assert payload["datasets"]["give_me_some_credit"]["missing"] == [
        "public_dataset_file",
        "human_tuned_baseline",
    ]
    assert "--tuned-by" in payload["datasets"]["give_me_some_credit"]["next_commands"][0]


def test_record_refuses_anonymous_agent_run(tmp_path, monkeypatch, capsys):
    data_path = tmp_path / "cs-training.csv"
    data_path.write_text("SeriousDlqin2yrs,age\n0,30\n1,40\n", encoding="utf-8")
    monkeypatch.setattr(
        ks_baseline, "_BASELINES_PATH", _baseline_file(tmp_path / "baselines.json")
    )

    exit_code = ks_baseline.main(
        [
            "--dataset",
            "give_me_some_credit",
            "--input",
            str(data_path),
            "--record",
        ]
    )
    assert exit_code == 2
    assert "anonymous agent run" in capsys.readouterr().err


def test_record_writes_traceable_provenance(tmp_path, monkeypatch):
    data_path = tmp_path / "cs-training.csv"
    data_path.write_text("SeriousDlqin2yrs,age\n0,30\n1,40\n", encoding="utf-8")
    baseline_path = _baseline_file(tmp_path / "baselines.json")
    monkeypatch.setattr(ks_baseline, "_BASELINES_PATH", baseline_path)
    monkeypatch.setattr(
        ks_baseline,
        "_run_public",
        lambda *args, **kwargs: KSRunResult(
            dataset="give_me_some_credit",
            recipe="lgb",
            n_rows=2,
            n_features=1,
            train_ks=0.6,
            test_ks=0.5,
            oot_ks=0.4,
            test_auc=0.7,
            nan_labels_dropped=0,
        ),
    )

    exit_code = ks_baseline.main(
        [
            "--dataset",
            "give_me_some_credit",
            "--input",
            str(data_path),
            "--record",
            "--tuned-by",
            "risk-team",
            "--tuning-note",
            "reviewed one-round reference",
            "--params-json",
            '{"num_leaves": 15}',
            "--workdir",
            str(tmp_path / "work"),
        ]
    )
    assert exit_code == 0
    entry = json.loads(baseline_path.read_text(encoding="utf-8"))[
        "give_me_some_credit"
    ]
    assert entry["baseline_ks"] == 0.5
    assert entry["status"] == "recorded"
    assert entry["provenance"]["dataset_sha256"].startswith("sha256:")
    assert entry["provenance"]["params"] == {"num_leaves": 15}
    assert entry["provenance"]["tuned_by"] == "risk-team"
