# MIDOG++ Uniform-B v2 consumed-test fixed-bank LOO directional-shrinkage ensemble

## Status and claim boundary

Experiment:
`midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_directional_shrinkage_ensemble.v1`

Status: registered and runnable, but not executed. The maximum evidence role is
`POST_HOC_CONSUMED_TEST_SENSITIVITY`; `fresh_evidence=false` and the immutable
decision is `TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE`.

The bounded interpretation is **no stable incremental target-support advantage
versus G**. This diagnostic may describe mechanism and stability on fixed,
already-consumed test bytes. It cannot establish fresh routing quality,
downstream utility, target-support generalization, or a promotable policy.

## Six-input isolation

The experiment has exactly six ordered inputs:

1. promoted Uniform-B v2 expert bank;
2. immutable GenerationLock;
3. dedicated label-free consumed-test cache alias;
4. dedicated capability-gated manifest alias;
5. byte-exact alias of the immutable original consumption ledger; and
6. this experiment's direct, single-consumer amendment.

There is no earlier Stage-90 output or amendment, prior prediction surface,
Stage-50/60/70 result, foreign scratch directory, or foreign checkpoint. All
probabilities and decisions are rematerialized under the new experiment and
artifact namespaces.

## Frozen physical surface and LOO roles

The physical surface contains 810 cells: nine targets by nine exact
train/generation seed pairs by ten actions. Actions are B, internal control U,
and eight non-target A1 source actions. Exact-nine probabilities are averaged
before directional scoring.

The held unit is each whole case or group `c`, not an arbitrary fold. Across
the nine centers there are 218 held units. For `(H,c)`, support contains all
other whole cases in H and terminal evaluation contains only c. Route-scoped
support labels cannot update an expert, shared model, geometry, strength,
threshold, or feature set.

All physical probabilities seal globally before any label opens. Donor labels
may contribute to `G_d(H,e)` only for query centers `q` outside `{H,e}`.
Support labels open only for their `(H,c)` route. All 218 endpoint plans and
the aggregate method-decision seal must exist before any terminal held-case
label capability opens.

## Directional ensemble

The two direction IDs are `zero_to_one` and `one_to_zero`. They are defined by
B's hard prediction before any candidate flip. Support gain uses pooled
additive confusion counts separately on each branch; per-case BACC is not a
selection statistic. G is the equal-center mean directional gain over legal
donor query centers.

For each direction, the eight legal sources are ranked by G. The only arm grid
is:

| K | exact w values |
|---:|:---------------|
| 4 | 1/2, 3/5, 7/10 |
| 5 | 1/2, 3/5, 7/10 |
| 6 | 1/2, 3/5, 7/10 |

Within an arm, each top-K source receives `w*S + (1-w)*G`. OFF has score zero,
is ordered before numeric sources for ties, and contributes B probability.
Arithmetic remains rational until the final `1e-12` tie check. The nine arm
identities remain distinct even when they choose identical endpoints.

For each held case, DCSE averages the nine selected endpoint probabilities on
their respective B hard branches. Probability averaging precedes the sole
threshold `0.5`, and equality maps to class 1. `G_directional_matched` executes
the same nine-arm pipeline with `S:=G`; it is not given target-support labels.

## Controls and descriptive checks

Registered methods are B, U, `DCSE_LOO`, `G_directional_matched`, `DLOO_raw`,
`LOO_frequency_committee`, `O_directional_static`, and
`O_case_directional`. The two O methods are terminal oracles, not pre-terminal
methods. Additional controls are hard vote, unique-action mean, uniform A1,
direction decomposition, nested delete-one-support frequency, leave-one-arm,
and whole-pipeline delete-one-center recomputation.

The predeclared descriptive success checks require:

- full-sample DCSE-B and DCSE-U to be strictly positive;
- both contrasts to remain positive in all nine whole-pipeline center
  deletions;
- at least eight of nine center-level DCSE-B deltas to be nonnegative; and
- every leave-one-arm DCSE-B contrast to remain strictly positive.

DCSE-G, nominal t intervals, jackknife summaries, and the null are not gates.
The candidate-identity null has 10,000 replicates with seed `20260813`. Its
frozen `splitmix64_route_candidate_block_permutation_v1` plan applies one
route-local eight-candidate permutation to support-S identities in both
directions, while B, G, physical probabilities, and canonical decisions stay
fixed. Null-specific endpoint selections are recomputed, but the summary has
no exchangeability or confirmatory p-value interpretation. No descriptive
check can widen the claim boundary.

## Execution and canonical output

Two persistent spawned RTX A5000 workers materialize the frozen generation
surface. CUDA visibility is then cleared before a phase-disjoint CPU pool of
four workers with three BLAS threads each. Source/probability storage is
float32, confusion counts are int64, and reductions are float64. Dedicated
`/data/local/fixed_bank_loo_directional_shrinkage_ensemble_v1` scratch is a
throughput-only surface. Intra-launch atomic task checkpoints are cleaned after
their validated global seal; owned-task replay, terminal recovery, and
cross-run recovery are forbidden.

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_loo_directional_shrinkage_ensemble.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_fixed_bank_loo_directional_shrinkage_ensemble/v1/
```

Registration identities:

- config contract: `500dc61f9f8d3bd0`;
- protocol contract: `d3dfdfb4d612a97b`;
- direct-amendment SHA-256:
  `05f800f1bd053528477abd1e67163612c01d44f56418f98961bcdf64677bdc52`.

The canonical bundle is closed-world with 43 required members. Until it exists
and independently validates, there is no experiment result. Regardless of its
future values, it cannot feed Stage 50, 60, 70, another Stage-90 diagnostic,
another experiment, recipe selection, deployable selection, or promotion.
