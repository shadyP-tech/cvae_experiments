# Consumed-test utility-aligned endpoint router

This package implements the target-static Stage-90 diagnostic registered as
`midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router.v1`.
It reuses the already-consumed MIDOG++ test surface under a dedicated ledger
amendment. Its outputs are terminal exploratory diagnostics: they cannot be
used as fresh routing evidence, update a policy/model/expert, select another
experiment, or feed Stage 50, 60, 70, or another Stage-90 run.

## Scientific contract

- The first eight lexically sorted whole cases in each center are label-free
  support; the remaining whole cases are terminal evaluation.
- Development responses are 504 strict `H/q/e` exact-nine probability-
  ensemble BACC deltas. The nine seed cells are averaged before thresholding
  and are never treated as independent observations.
- `M0` contains one source-global control. `M1` adds exactly one label-free,
  ensemble-first support action-probability-shift scalar. `P` is its fixed
  same-capacity cyclic permutation control.
- The feature runtime also computes posterior-mean reconstruction MSE,
  latent-dimension-normalized analytic PS KL, and a linear-kernel squared-mean
  discrepancy as descriptive CVAE diagnostics. They do not enter `M0`, `M1`,
  or `P`, are not an exact NELBO, and are not downstream utility. The local
  probability-shift predictor is unsigned classifier sensitivity, not a
  generative compatibility score.
- Every `Hxe` candidate (and any `G/R/P` alias selecting it) is an equal-union
  `B` action plus one single-source tail. Its endpoint is not standalone expert
  utility.
- The router emits one static plan per target center. `R` is accepted only
  through the frozen source-transfer and bootstrap lower-bound gates;
  otherwise the executed action is exact `B`.
- All target plans, action identities, and probability arrays are sealed before
  same-target evaluation labels become accessible. Terminal scoring cannot
  change any plan.

## Module boundaries

- `contracts.py`, `partitions.py`, `features.py`, `models.py`, `policy.py`, and
  `inference.py` contain typed scientific contracts and pure transformations.
- `feature_*`, `prediction_*`, `development_runtime.py`, and
  `target_runtime.py` own workstation computation, task hashes, atomic
  checkpoints, and immutable probability stores.
- `inputs.py`, `label_capabilities.py`, `protocol.py`, and `seals.py` own data
  admission and label ordering.
- `persistence.py` and `bundle.py` own closed-world publication. `validation.py`
  orchestrates the phase-specific `validation_science_*` replay modules for
  development responses, feature surfaces, fitted models and policies, action
  aliases, and terminal endpoint formulas.
- `runner.py` only orders phases. Scientific callbacks are injectable for
  focused tests, while workspace, provenance, label, preflight, persistence,
  and final-validation gates remain production-owned.
- `run_lock.py` records a tokenized PID/host owner and may recover only a dead
  same-host owner, so a killed workstation process can resume without stealing
  a live or remote lock.

The validator reconstructs all persisted scientific rows and formula/hash
lineage. Target bootstrap productions are cross-bound by their persisted
production, row, and surface hashes; their raw per-case shift tensors are not
published, so that one intermediate cannot be independently regenerated from
the terminal bundle. This is an explicit diagnostic replay limit, not evidence
for a fresh or promotable routing claim.

## Workstation launch

Use the registered workspace command from a clean checkout. Do not use
`--force` and do not invoke the diagnostic runner directly.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_target_static_endpoint_router.v1
```

The run requires two RTX A5000 GPUs for the phase-disjoint generation and
label-free compatibility phase, followed by four CPU classifier workers with
three BLAS threads each. Task checkpoints are hash-bound and resumable within
the same workspace snapshot; a completed bundle is reusable only after full
closed-world validation.

The original pre-compute failure `Endpoint-router test-cache identity drifted`
was caused by expecting `representation_id` at the frozen protocol's top level;
the canonical producer stores it in the nested extractor protocol, builder
report, and shard extractors. A second pre-label bug compared the partition-
aware row hash (`support`) with the physical cache row hash (`unassigned`);
physical embedding lookup now checks every immutable row field while partition
seals remain role-aware. A third pre-label task-contract bug classified the
16-hex config, expert-bank, and frozen-source semantic identities as 64-hex
file digests. The contract now keeps those hash families distinct and binds
each source block's semantic SHA-256 into the GPU task before scoring.

The registered exact-snapshot retry recognizes only those three failures: the
original `FAILED/INITIALIZING` three-file boundary, the exact 11-file
`FAILED/SOURCE_AND_LABEL_FREE_FEATURES` embedding-identity boundary with its
sealed 81-stream source cache, or the feature-task boundary containing those
11 files plus exactly nine hash-validated staged support arrays. It preserves
the original resolved config and input provenance and rejects `--force`, extra
arguments, or any inventory/input drift.
