from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(importlib.util.find_spec("bandit") is None, reason="bandit is optional")
def test_incremental_bandit_matches_recursive_baseline_and_rejects_new_findings(tmp_path):
    root = tmp_path / "repo"
    source = root / "marvis" / "known.py"
    source.parent.mkdir(parents=True)
    source.write_text("import pickle\npickle.loads(b'data')\n", encoding="utf-8")
    shutil.copytree(ROOT / "scripts", root / "scripts")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "baseline"],
        cwd=root, check=True,
    )
    baseline = root / ".bandit-baseline.json"
    scanned = subprocess.run(
        [sys.executable, "-m", "bandit", "-r", "marvis", "-ll", "-ii",
         "-f", "json", "-o", str(baseline)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert scanned.returncode == 1, scanned.stdout + scanned.stderr
    original_baseline = baseline.read_bytes()
    assert json.loads(original_baseline)["results"][0]["filename"] == "marvis/known.py"
    source.write_text("import pickle\npickle.loads(b'data')\n# harmless change\n", encoding="utf-8")
    env = {**os.environ, "PYTHON": sys.executable}
    env.pop("CHECK_DIFF_RANGE", None)
    command = ["bash", "scripts/check", "--skip-pytest", "--skip-ruff", "--skip-node", "--skip-diff"]

    accepted = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert baseline.read_bytes() == original_baseline

    # A new medium/high finding must still fail, including in an untracked file.
    (source.parent / "new.py").write_text("eval(input())\n", encoding="utf-8")
    rejected = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
    assert rejected.returncode == 1, rejected.stdout + rejected.stderr
    assert "B307" in rejected.stdout
    assert baseline.read_bytes() == original_baseline
