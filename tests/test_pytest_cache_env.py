from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("use_external_override", [False, True])
def test_check_provides_persistent_matplotlib_cache(
    tmp_path: Path,
    use_external_override: bool,
):
    project_root = tmp_path / "project"
    shutil.copytree(ROOT / "scripts", project_root / "scripts")
    capture = tmp_path / "matplotlib-cache.txt"
    fake_python = tmp_path / "python"
    _write(
        fake_python,
        "#!/bin/sh\n"
        'printf "%s\\n" "${MPLCONFIGDIR-}" > "$CHECK_CAPTURE"\n',
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env.pop("MPLCONFIGDIR", None)
    env.update(
        {
            "PYTHON": str(fake_python),
            "CHECK_CAPTURE": str(capture),
        }
    )
    if use_external_override:
        expected = tmp_path / "caller-cache"
        env["MPLCONFIGDIR"] = str(expected)
    else:
        expected = project_root / ".pytest_cache" / "matplotlib"

    completed = subprocess.run(
        [
            str(project_root / "scripts" / "check"),
            "--skip-ruff",
            "--skip-node",
            "--skip-diff",
        ],
        cwd=project_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert capture.read_text(encoding="utf-8").strip() == str(expected)
    assert expected.is_dir()


@pytest.mark.parametrize("use_external_override", [False, True])
def test_direct_pytest_bootstrap_provides_persistent_matplotlib_cache(
    tmp_path: Path,
    use_external_override: bool,
):
    env = os.environ.copy()
    env.pop("MPLCONFIGDIR", None)
    if use_external_override:
        expected = tmp_path / "caller-cache"
        env["MPLCONFIGDIR"] = str(expected)
    else:
        expected = ROOT / ".pytest_cache" / "matplotlib"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import tests.conftest; "
                "print(os.environ['MPLCONFIGDIR'])"
            ),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == str(expected)
    assert expected.is_dir()
