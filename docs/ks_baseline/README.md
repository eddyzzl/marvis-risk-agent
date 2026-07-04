# KS-Baseline Harness (T4-2)

End-to-end harness that drives the MARVIS modeling stack (ingest → split → train →
KS) on a tabular credit dataset and compares the resulting KS to a stored
**ground-truth baseline**. It is the second of the three data-judgement layers in
[`docs/plans/v2-trust-first-plan.md`](../plans/v2-trust-first-plan.md) (T4).

There are two ways to run it:

| Mode | Data | Entry point | In CI fast tier? |
|------|------|-------------|------------------|
| **Smoke anchor** | built-in deterministic synthetic data | `tests/test_ks_baseline_smoke.py` | **Yes** — proves the harness runs |
| **KS baseline** | a real public dataset **you** drop in | `scripts/ks_baseline.py` | No — needs user data |

> The smoke anchor is **not** a KS standard. It only proves the plumbing works
> end-to-end in a sandbox with no external data. The real KS gate is the
> public-dataset path below.

---

## The pass/fail contract

For a given dataset:

```
PASS   when   agent_ks  >=  baseline_ks − tolerance      (default tolerance = 0.005)
```

- `agent_ks` — the `test_ks` the harness's `run_ks` produces by driving the real
  training recipe (`marvis/packs/modeling/recipes/*`).
- `baseline_ks` — a **one-off, human-tuned** KS captured as ground truth in
  [`baselines.json`](./baselines.json). It represents "the best a careful analyst
  reached on this data"; the agent must not regress below it by more than the
  tolerance.
- `tolerance` — `0.005` by default (per plan §0), overridable with `--tolerance`.

The tolerance and the metric (`test_ks`) are recorded in `baselines.json`'s
`_meta` block.

---

## Running the built-in smoke anchor

No data needed:

```bash
python scripts/ks_baseline.py --dataset synthetic_smoke
```

or via the test suite (fast tier):

```bash
PYTHONPATH=. /opt/miniconda3/envs/py_313/bin/python -m pytest tests/test_ks_baseline_smoke.py -q
```

---

## Running a real public dataset

The harness supports two public benchmarks out of the box. **You** provide the
raw files — they are not shipped (licensing / size).

### 1. Give Me Some Credit (Kaggle)

- Download `cs-training.csv` from the Kaggle *Give Me Some Credit* competition.
- Drop it at:

  ```
  datasets/give_me_some_credit/cs-training.csv
  ```

- Target column: `SeriousDlqin2yrs`. Feature columns are fixed in
  `scripts/ks_baseline.py` (`_DATASET_SPECS`).

### 2. Home Credit Default Risk (Kaggle)

- Download `application_train.csv` from the Kaggle *Home Credit Default Risk*
  competition.
- Drop it at:

  ```
  datasets/home_credit/application_train.csv
  ```

- Target column: `TARGET`. A compact always-present numeric feature subset is
  used by default; extend `_DATASET_SPECS["home_credit"]["features"]` once the
  file is present.

The `datasets/` directory is git-ignored — your raw data never enters the repo.

### Capture a ground-truth baseline (one-off)

After a careful, human-tuned run you consider the reference, record its KS as the
ground truth:

```bash
python scripts/ks_baseline.py --dataset give_me_some_credit --record
```

This writes `baseline_ks` for that dataset into `baselines.json`. Commit that
number (and fill in the `provenance` block: seed, split, who tuned it) so the
baseline is a reviewable ground truth, per the plan's risk table.

### Check the agent against the baseline

Once a baseline exists:

```bash
python scripts/ks_baseline.py --dataset give_me_some_credit
```

Exit codes: `0` = PASS, `1` = FAIL (agent KS below the floor), `2` = no baseline
recorded yet. Sample output:

```
[PASS] give_me_some_credit: agent KS 0.5312 vs baseline 0.5330 (floor 0.5280, tolerance 0.005)
```

---

## Files

| Path | Role |
|------|------|
| `tests/support/ks_harness.py` | reusable runner + baseline-compare core (shared by CLI and smoke test) |
| `scripts/ks_baseline.py` | CLI: run a dataset, `--record` a baseline, or compare |
| `tests/test_ks_baseline_smoke.py` | CI smoke anchor (fast tier) |
| `docs/ks_baseline/baselines.json` | ground-truth baselines (synthetic anchor + public-dataset placeholders) |
| `datasets/<name>/…` | **user-provided** raw public datasets (git-ignored) |

## Adding another dataset

1. Add a `_DATASET_SPECS` entry in `scripts/ks_baseline.py` (file name, target,
   features, recipe).
2. Add a placeholder entry in `baselines.json`.
3. Drop the raw file under `datasets/<name>/`, run `--record`, commit the number.
