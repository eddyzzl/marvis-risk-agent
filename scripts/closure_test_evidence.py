#!/usr/bin/env python3
"""Run the closure regression net and emit commit-bound machine evidence.

The evidence is intentionally external to the acceptance report.  It is only
green when the exact command succeeds against a clean worktree, and the report
consumer independently verifies the log hash and current commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


REQUIRED_TESTS = (
    "tests/test_dirty_shape_regression.py",
    "tests/test_reconcile_reference_numbers.py",
)


def build_command(python: str) -> list[str]:
    return [python, "-m", "pytest", "-q", *REQUIRED_TESTS]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def run(
    *,
    repo: Path,
    output_path: Path,
    log_path: Path,
    python: str,
) -> dict:
    repo = repo.resolve()
    output_path = output_path.resolve()
    log_path = log_path.resolve()
    commit_sha = _git(repo, "rev-parse", "HEAD")
    worktree_clean_before = not bool(
        _git(repo, "status", "--porcelain", "--untracked-files=normal")
    )
    command = build_command(python)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_stream:
        completed = subprocess.run(
            command,
            cwd=repo,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )

    commit_sha_after = _git(repo, "rev-parse", "HEAD")
    worktree_clean_after = not bool(
        _git(repo, "status", "--porcelain", "--untracked-files=normal")
    )
    worktree_clean = (
        worktree_clean_before
        and worktree_clean_after
        and commit_sha_after == commit_sha
    )
    try:
        serialized_log_path = str(log_path.relative_to(output_path.parent))
    except ValueError:
        serialized_log_path = str(log_path)
    payload = {
        "schema_version": "closure-test-evidence.v1",
        "command": command,
        "exit_code": int(completed.returncode),
        "commit_sha": commit_sha,
        "commit_sha_after": commit_sha_after,
        "timestamp": datetime.now(UTC).isoformat(),
        "log_path": serialized_log_path,
        "log_hash": _sha256_file(log_path),
        "worktree_clean": worktree_clean,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--output",
        default="workspace/closure_test_evidence.json",
    )
    parser.add_argument(
        "--log",
        default="workspace/closure-regression.log",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo / output_path
    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = repo / log_path
    try:
        payload = run(
            repo=repo,
            output_path=output_path,
            log_path=log_path,
            python=str(args.python),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error": str(exc)},
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["exit_code"] == 0 and payload["worktree_clean"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
