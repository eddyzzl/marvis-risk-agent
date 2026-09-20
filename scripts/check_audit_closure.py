#!/usr/bin/env python3
"""Validate audit ledger bindings; never turn self-reported receipts into proof.

The standalone CLI has no trusted environment/reviewer adapters. It can check
an open ledger and evidence integrity, but cannot authenticate closure. Actual
acceptance integrations use check_ledger with independent trusted verifiers.
Exit 0 means a valid ledger, not completed development; --require-complete
also fails while any required verified evidence remains unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from marvis.orchestrator.eval.acceptance import check_ledger, load_acceptance_spec  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--spec", type=Path, help="Independently approved acceptance requirements")
    parser.add_argument("--spec-sha256", help="Approved spec digest, supplied independently of the ledger")
    args = parser.parse_args(argv)
    if bool(args.spec) != bool(args.spec_sha256):
        parser.error("--spec and --spec-sha256 must be supplied together")
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError("ledger must be an object")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
        result = check_ledger(
            ledger, source_root=REPO_ROOT, evidence_root=args.evidence_root,
            expected_commit=commit,
            trusted_spec=load_acceptance_spec(args.spec, expected_sha256=args.spec_sha256) if args.spec else None,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"valid": False, "complete": False, "errors": [str(exc)]}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result["valid"] or (args.require_complete and not result["complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
