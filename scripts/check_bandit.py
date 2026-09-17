"""Run file-scoped Bandit using the recursive CI baseline's path convention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=".bandit-baseline.json")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args(argv)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    # Bandit prefixes explicitly selected files with './', but not files
    # discovered via '-r marvis'. Preserve every finding and its severity;
    # adapt paths only in a temporary copy, leaving the CI baseline untouched.
    for issue in baseline["results"]:
        issue["filename"] = os.path.join(".", os.path.normpath(issue["filename"]))
    with tempfile.TemporaryDirectory(prefix="marvis-bandit-") as directory:
        path = Path(directory) / "baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, "-m", "bandit", "-ll", "-ii", "-b", str(path),
             *(os.path.normpath(name) for name in args.files)],
            check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
