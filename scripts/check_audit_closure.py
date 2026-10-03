#!/usr/bin/env python3
"""Validate audit ledger bindings; never turn self-reported receipts into proof.

Without an independently pinned trust configuration this checks integrity only.
Optional installed adapters verify external authority signatures and their exact
source, criteria and complete-run bindings; candidate flags are never proof.
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
from marvis.orchestrator.eval.acceptance_trust import TrustInputError, load_trusted_environment  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--spec", type=Path, help="Independently approved acceptance requirements")
    parser.add_argument("--spec-sha256", help="Approved spec digest, supplied independently of the ledger")
    parser.add_argument("--trust-config", type=Path, help="External authority configuration; read only")
    parser.add_argument("--trust-config-sha256", help="Configuration digest held independently of candidate evidence")
    args = parser.parse_args(argv)
    if bool(args.spec) != bool(args.spec_sha256):
        parser.error("--spec and --spec-sha256 must be supplied together")
    if bool(args.trust_config) != bool(args.trust_config_sha256):
        parser.error("--trust-config and --trust-config-sha256 must be supplied together")
    if args.trust_config and not args.spec:
        parser.error("a trusted environment also requires the independently pinned acceptance spec")
    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError("ledger must be an object")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
        spec = load_acceptance_spec(args.spec, expected_sha256=args.spec_sha256) if args.spec else None
        environment = load_trusted_environment(
            args.trust_config, expected_sha256=args.trust_config_sha256,
            source_root=REPO_ROOT, evidence_root=args.evidence_root,
            expected_commit=commit, spec_sha256=args.spec_sha256,
        ) if args.trust_config else None
        result = check_ledger(
            ledger, source_root=REPO_ROOT, evidence_root=args.evidence_root,
            expected_commit=commit,
            trusted_spec=spec, trusted_environment=environment,
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.CalledProcessError) as exc:
        reason = str(exc) if isinstance(exc, TrustInputError) or not args.trust_config else "input_or_environment_rejected"
        print(json.dumps({"valid": False, "complete": False, "errors": [reason]}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result["valid"] or (args.require_complete and not result["complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
