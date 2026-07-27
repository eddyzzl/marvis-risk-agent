#!/usr/bin/env python3
"""T4-2: end-to-end KS-baseline runner (CLI).

Point this at a public credit dataset you have dropped into the convention
directory and it drives the modeling stack (ingest → split → train → KS) and
compares the result to the stored ground-truth baseline.

    # compare a dataset's agent-produced KS to its recorded baseline
    python scripts/ks_baseline.py --dataset give_me_some_credit

    # capture a fresh ground-truth baseline (after a human-tuned run)
    python scripts/ks_baseline.py --dataset give_me_some_credit --record

    # run the built-in synthetic smoke anchor (no external data needed)
    python scripts/ks_baseline.py --dataset synthetic_smoke

Dataset conventions live in docs/ks_baseline/README.md. This runner is NOT part
of the CI fast tier — the public-dataset paths require user-provided data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The reusable harness core lives under tests/support so the smoke test and this
# CLI share exactly one implementation; make it importable when run as a script.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

_BASELINES_PATH = _REPO_ROOT / "docs" / "ks_baseline" / "baselines.json"
_DATASETS_ROOT = _REPO_ROOT / "datasets"
DEFAULT_KS_TOLERANCE = 0.005

# Per public dataset: the on-disk file (relative to datasets/<name>/), its target
# column, and the feature columns to model. The user drops the raw Kaggle file at
# the named path; everything else is fixed here so a baseline run is reproducible.
_DATASET_SPECS: dict[str, dict] = {
    "give_me_some_credit": {
        "file": "cs-training.csv",
        "target_col": "SeriousDlqin2yrs",
        "features": [
            "RevolvingUtilizationOfUnsecuredLines",
            "age",
            "NumberOfTime30-59DaysPastDueNotWorse",
            "DebtRatio",
            "MonthlyIncome",
            "NumberOfOpenCreditLinesAndLoans",
            "NumberOfTimes90DaysLate",
            "NumberRealEstateLoansOrLines",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfDependents",
        ],
        "recipe": "lgb",
    },
    "home_credit": {
        "file": "application_train.csv",
        "target_col": "TARGET",
        # a compact, always-present numeric subset; extend once the file is present.
        "features": [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "CNT_CHILDREN",
            "REGION_POPULATION_RELATIVE",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
        ],
        "recipe": "lgb",
    },
}


def _load_baselines() -> dict:
    return json.loads(_BASELINES_PATH.read_text(encoding="utf-8"))


def _write_baselines(data: dict) -> None:
    _BASELINES_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _run_synthetic(workdir: Path):
    from marvis.sample_data import generate_sample_frame
    from support.ks_harness import SplitSpec, run_ks

    frame = generate_sample_frame(n_rows=2000, seed=20260701)
    features = [
        "credit_score",
        "debt_income_ratio",
        "monthly_income",
        "loan_amount",
        "history_overdue_count",
        "account_age_months",
    ]
    return run_ks(
        frame,
        features=features,
        target_col="y",
        dataset_name="synthetic_smoke",
        recipe="lr",
        split=SplitSpec(seed=20260701),
        workdir=workdir,
    )


def _public_data_path(name: str, input_path: str | Path | None = None) -> Path:
    if input_path is not None:
        return Path(input_path).expanduser().resolve()
    spec = _DATASET_SPECS[name]
    return _DATASETS_ROOT / name / spec["file"]


def _run_public(
    name: str,
    workdir: Path,
    *,
    recipe: str | None,
    input_path: str | Path | None = None,
    params: dict[str, Any] | None = None,
):
    from support.ks_harness import SplitSpec, run_ks

    spec = _DATASET_SPECS[name]
    data_path = _public_data_path(name, input_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {data_path}\n"
            f"Drop the raw file there (see docs/ks_baseline/README.md), or pass "
            f"--input /absolute/path/to/{spec['file']}."
        )
    return run_ks(
        data_path,
        features=spec["features"],
        target_col=spec["target_col"],
        dataset_name=name,
        recipe=recipe or spec["recipe"],
        split=SplitSpec(seed=20260705),
        workdir=workdir,
        drop_nan_labels=True,  # public files carry unlabeled rows; drop for the metric
        params=params,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parse_params(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    if raw.startswith("@"):
        payload = Path(raw[1:]).read_text(encoding="utf-8")
    else:
        payload = raw
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("--params-json must decode to a JSON object")
    return value


def _status_payload() -> dict[str, Any]:
    baselines = _load_baselines()
    datasets: dict[str, Any] = {}
    for name, spec in sorted(_DATASET_SPECS.items()):
        data_path = _public_data_path(name)
        entry = baselines.get(name) or {}
        file_present = data_path.is_file()
        baseline_recorded = entry.get("baseline_ks") is not None
        datasets[name] = {
            "status": "ready" if file_present and baseline_recorded else "blocked",
            "data_path": str(data_path),
            "file_present": file_present,
            "baseline_recorded": baseline_recorded,
            "missing": [
                item
                for item, present in (
                    ("public_dataset_file", file_present),
                    ("human_tuned_baseline", baseline_recorded),
                )
                if not present
            ],
            "next_commands": [
                (
                    f"python scripts/ks_baseline.py --dataset {name} "
                    f"--input /absolute/path/to/{spec['file']} --record "
                    '--tuned-by "<name/team>" --tuning-note "<method and review>"'
                ),
                (
                    f"python scripts/ks_baseline.py --dataset {name} "
                    f"--input /absolute/path/to/{spec['file']}"
                ),
            ],
        }
    return {
        "gate": "T4-2",
        "status": (
            "ready"
            if datasets and all(item["status"] == "ready" for item in datasets.values())
            else "blocked"
        ),
        "checked_at": datetime.now(UTC).isoformat(),
        "datasets": datasets,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end KS-baseline harness.")
    parser.add_argument(
        "--dataset",
        choices=["synthetic_smoke", *sorted(_DATASET_SPECS)],
        help="which dataset to run",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="print machine-readable T4-2 readiness; exits 2 while any public gate is blocked",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="absolute public-dataset path (otherwise uses datasets/<name>/<file>)",
    )
    parser.add_argument("--recipe", default=None, help="override the recipe (lr/lgb/scorecard)")
    parser.add_argument(
        "--params-json",
        default=None,
        help="human-reviewed recipe params as inline JSON or @/path/to/params.json",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_KS_TOLERANCE,
        help=f"KS tolerance below baseline that still passes (default {DEFAULT_KS_TOLERANCE})",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="record a reviewed public-data run as baseline (requires provenance arguments)",
    )
    parser.add_argument(
        "--tuned-by",
        default=None,
        help="required with --record: analyst/team that tuned and reviewed the baseline",
    )
    parser.add_argument(
        "--tuning-note",
        default=None,
        help="required with --record: reproducible tuning/review method",
    )
    parser.add_argument("--workdir", default=None, help="working directory (default: a temp dir)")
    args = parser.parse_args(argv)
    if args.status:
        if args.dataset or args.record or args.input:
            parser.error("--status cannot be combined with --dataset/--record/--input")
    elif not args.dataset:
        parser.error("--dataset is required unless --status is used")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        payload = _status_payload()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if payload["status"] == "ready" else 2

    if args.record:
        if args.dataset == "synthetic_smoke":
            print(
                "--record is reserved for reviewed public-data baselines; "
                "the synthetic anchor is maintained in source control.",
                file=sys.stderr,
            )
            return 2
        if not args.tuned_by or not args.tuning_note:
            print(
                "--record requires both --tuned-by and --tuning-note; "
                "an anonymous agent run is not a human-tuned ground truth.",
                file=sys.stderr,
            )
            return 2

    try:
        params = _parse_params(args.params_json)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid --params-json: {exc}", file=sys.stderr)
        return 2

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="ks_baseline_"))

    try:
        if args.dataset == "synthetic_smoke":
            result = _run_synthetic(workdir)
        else:
            result = _run_public(
                args.dataset,
                workdir,
                recipe=args.recipe,
                input_path=args.input,
                params=params,
            )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    baselines = _load_baselines()
    entry = baselines.get(args.dataset, {})

    if args.record:
        data_path = _public_data_path(args.dataset, args.input)
        entry = dict(entry)
        entry["baseline_ks"] = result.test_ks
        entry["recipe"] = result.recipe
        entry["recorded_test_ks"] = result.test_ks
        entry["status"] = "recorded"
        entry["note"] = args.tuning_note
        entry["provenance"] = {
            "dataset": str(data_path),
            "dataset_sha256": _file_sha256(data_path),
            "target_col": _DATASET_SPECS[args.dataset]["target_col"],
            "n_rows": result.n_rows,
            "n_features": result.n_features,
            "seed": 20260705,
            "split": "positional 60/20/20 (deterministic)",
            "recipe": result.recipe,
            "params": params,
            "tuned_by": args.tuned_by,
            "tuning_note": args.tuning_note,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        baselines[args.dataset] = entry
        _write_baselines(baselines)
        print(f"\nRecorded baseline_ks={result.test_ks} for {args.dataset} in {_BASELINES_PATH}")
        return 0

    baseline_ks = entry.get("baseline_ks")
    if baseline_ks is None:
        spec = _DATASET_SPECS.get(args.dataset, {})
        input_hint = args.input or f"/absolute/path/to/{spec.get('file', 'dataset.csv')}"
        print(
            f"\nNo human-tuned baseline recorded for {args.dataset}; T4-2 remains BLOCKED.\n"
            f"Record (after review): python scripts/ks_baseline.py --dataset "
            f"{args.dataset} --input {input_hint} --record "
            f'--tuned-by "<name/team>" --tuning-note "<method and review>"\n'
            f"Then compare: python scripts/ks_baseline.py --dataset {args.dataset} "
            f"--input {input_hint}"
        )
        return 2

    from support.ks_harness import compare_to_baseline

    verdict = compare_to_baseline(result, float(baseline_ks), tolerance=args.tolerance)
    print("\n" + verdict.render())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
