# Hidden benchmark execution bindings

The formal hidden entry is executable after its separately held inputs and approvals are frozen. The repository's pending ledger/profile remain blocking. The entry produces **unsigned candidate originals**, never independent executor/reviewer signatures or an acceptance verdict.

Use a dedicated source checkout. Keep the held inputs, output batch and private custody outside that checkout and separate from one another. The acceptance source inventory covers the actual checkout (including untracked files), not only the runtime package. A large development checkout containing experimental outputs is not an acceptance checkout.

## Frozen materialization

The original hidden freeze contains 133 scenario templates and 193 attempt slots. Its `case_sha256` is a template hash. `marvis.hidden-execution-binding.v1` separately binds every template to an actual `RuntimeCase`, `ExpectedCase`, fault contract, material inventory, model connection, price book, provider accounting approval and source inventory. Case/slot/repetition identities and frozen thresholds cannot be changed. The second freeze may explicitly assign load modes; it never edits the original proposal.

Validate bindings using `hidden_formal_contract.prepare_hidden_execution`. This is read-only and creates no tasks, outputs or provider requests. A valid binding additionally requires the decided total cost cap, frozen hardware/load/SLO profile, and the explicitly named credential environment variable. Credentials are neither copied into originals nor written in reports.

The fixed price book and provider accounting approval must agree with the ledger. The approval also binds the underlying source document bytes and price digest. Local-fixture approvals only authorize loopback fixtures. Provider accounting is an externally approved assumption, not an inference from returned usage.

## Execution

```sh
python -m marvis.orchestrator.eval.hidden_benchmark formal \
  --freeze-dir /held/proposal \
  --binding /held/materialization/binding.json \
  --ledger /held/materialization/ledger.json \
  --profile /held/materialization/profile.json \
  --out /evidence/public-runs \
  --custody /evidence/private-custody
```

All 193 full per-attempt cost ceilings are checked before any execution. Every formal request is restricted to the frozen plain-text protocol and input-byte/output-token ceilings before forwarding. Full reservations are retained after each response; lower or missing reported usage never releases new spending capacity. The report does not claim a provider billing guarantee.

Cold-start timings include process startup. Steady mode executes the exact frozen read-only warmup requests and minimum duration in the same PID/workspace, then measures from the recorded warmup completion through the original case end. Originals preserve startup and warmup; a producer cannot select a shorter measurement window.

Concurrent slots execute in a bounded thread pool, each using its own real application process. Candidate proof references finalized run manifests. The independent reader reconstructs the actual journey intervals, matching host boot/monotonic clock, source, model, load and materials, and requires the frozen number of simultaneously running journeys. Merely assigning a `concurrent` label is insufficient. This verifies actual simultaneous journeys; it does not claim constant concurrency throughout the profile's sustained-load duration. The separate performance runner covers sustained load. Declared profile data rows are not treated as equal to every case's multi-file material size.

Recovery faults are executed and recorded independently from whether the product recovers. A failed recovery or a required human review remains a failed/unavailable expectation in the complete denominator. Service restart uses a new actual process with the same workspace and retains separate process epochs. A crashed, unsealed timing epoch remains incomplete; the later service journal cannot replace it to claim a shorter SLO.

## Evidence boundary

Each batch retains its planned 193 slots, frozen input bytes and hashes, per-slot originals, finalized runtime manifests, fixed-collector telemetry, concurrency references, and a final candidate report. Private native archives remain in the separate custody directory. The final report distinguishes local fixtures (`local_fixture/protocol_only`) from unsigned formal candidates and has `acceptance_claim=none`.

The existing trusted acceptance adapter still requires independently held execution/review signatures and independently verified native/domain evidence. No candidate report signs itself or upgrades historical source labels into business approval. Local development tests use controlled fixtures only; they are not a paid-model or historical benchmark result.

The original sixteen recovery slots use family-specific native boundaries. Labeling stale CAS recovery requires the actual 409 with unchanged target originals, a separately declared current approval, one native invocation/output, and independent label recomputation. Validation stale revision recovery similarly binds the rejected draft revision, explicit current confirmation, unique report job, unchanged scoring originals and actual downloaded reports. A 409 alone is never recovery.

Portfolio after-commit injection binds the explicit `analysis.portfolio_gate_summary` approval to its unique fresh downstream `analysis.portfolio_report`; modeling binds `modeling.configure_tuning` to the native `modeling.train_models` batch. These are source-owned pairs, not arbitrary downstream tools. Fault execution and recovery completion are separate observations; unknown effects and rejected retries remain visible.
