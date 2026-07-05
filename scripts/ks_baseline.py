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
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
# The reusable harness core lives under tests/support so the smoke test and this
# CLI share exactly one implementation; make it importable when run as a script.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from support.ks_harness import (  # noqa: E402
    DEFAULT_KS_TOLERANCE,
    SplitSpec,
    compare_to_baseline,
    run_ks,
)

_BASELINES_PATH = _REPO_ROOT / "docs" / "ks_baseline" / "baselines.json"
_DATASETS_ROOT = _REPO_ROOT / "datasets"

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


def _run_public(name: str, workdir: Path, *, recipe: str | None):
    spec = _DATASET_SPECS[name]
    data_path = _DATASETS_ROOT / name / spec["file"]
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset file not found: {data_path}\n"
            f"Drop the raw file there (see docs/ks_baseline/README.md) and retry."
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
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end KS-baseline harness.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["synthetic_smoke", *sorted(_DATASET_SPECS)],
        help="which dataset to run",
    )
    parser.add_argument("--recipe", default=None, help="override the recipe (lr/lgb/scorecard)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_KS_TOLERANCE,
        help=f"KS tolerance below baseline that still passes (default {DEFAULT_KS_TOLERANCE})",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="capture the run's test_ks as the new ground-truth baseline for this dataset",
    )
    parser.add_argument("--workdir", default=None, help="working directory (default: a temp dir)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="ks_baseline_"))

    try:
        if args.dataset == "synthetic_smoke":
            result = _run_synthetic(workdir)
        else:
            result = _run_public(args.dataset, workdir, recipe=args.recipe)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    baselines = _load_baselines()
    entry = baselines.get(args.dataset, {})

    if args.record:
        entry = dict(entry)
        entry["baseline_ks"] = result.test_ks
        entry.setdefault("recipe", result.recipe)
        entry["recorded_test_ks"] = result.test_ks
        baselines[args.dataset] = entry
        _write_baselines(baselines)
        print(f"\nRecorded baseline_ks={result.test_ks} for {args.dataset} in {_BASELINES_PATH}")
        return 0

    baseline_ks = entry.get("baseline_ks")
    if baseline_ks is None:
        print(
            f"\nNo baseline recorded for {args.dataset}. Run with --record after a "
            f"human-tuned run to capture ground truth."
        )
        return 2

    verdict = compare_to_baseline(result, float(baseline_ks), tolerance=args.tolerance)
    print("\n" + verdict.render())
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
