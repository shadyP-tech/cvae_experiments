# MIDOG++ Uniform-B V2 Utility-Aligned Residual Router

## Status

Implemented as a fresh, modular Stage-60/70 experiment family. All new
workspace entries remain `planned`: the checkout does not contain the required
fresh development reservation, fresh target-support surface, or fresh target
evaluation reservation/cache/manifest. Historical Stage-70 or Stage-90 bytes
cannot activate these experiments.

This family is the follow-up to the consumed-validation B/U/G/S case-OOF
diagnostic. That diagnostic found single-source-tail oracle headroom, but its
unlabeled support proxy did not identify the useful source. The new family
therefore learns the response of the exact action that will be deployed; it
does not tune Borda ranks or reinterpret compatibility energy as utility.

## Scientific contract

For each outer target `H`, source-inner pseudoquery `q`, and legal source `e`,
Stage 60 now measures the same probability-ensemble construction and BACC
endpoint used by Stage 70, over a cardinality-matched inner action family. It
does not pretend that a six-to-seven-source inner action is literally the same
deployed action as a seven-to-eight-source target action:

```text
p_bar_a = (1/9) * sum_{training seed, generation seed} p_a(y=1)
delta(H,q,e) = BACC(1[p_bar_tail >= 0.5]) - BACC(1[p_bar_B >= 0.5])
```

with `H != q` and `e` excluded from both `H` and `q`. The inner action uses the
cardinality-matched 12.5% residual geometry: seven sources at 144 samples per
source/class plus one exact 126-sample/class tail. The nine fixed
training/generation seed pairs are aggregated into one candidate response;
they are technical repeats, not nine independent utility observations. The
legacy 4,536 seed-cell rows remain inspectable and descriptive, while only the
504 `(H,q,e)` probability-ensemble rows may enter the model. Predictions and
both evaluation and support probabilities are globally sealed before the
dedicated development scoring labels open.

The model is deliberately limited to two predeclared predictors. `M0` contains
one source-global, label-free metadata-similarity control derived only from
the `H`-excluded source-inner feature surface. `M1` adds exactly one
target-local, action-proximal scalar. On each unlabeled support row it first
averages the canonical 3x3 seed-cell probabilities for exact `B` and the
corresponding exact-tail action, then takes their absolute difference, and
finally averages those row differences. The previous mean of nine per-seed
absolute shifts is retained only as descriptive technical-seed spread and may
not enter the model. `P` cyclically permutes only the ensemble-first local
scalar. No
source/query one-hot columns, broad proxy vector, target label, or utility
value may enter the feature surface. Source and query exclusions are nested.
Six-candidate training to seven-candidate validation is only an eligibility
test for the later seven-to-eight transfer; it is not target-routing evidence.

Activation is deliberately conservative:

- a deterministic 10,000-draw query-domain bootstrap must put routed top-1
  above `1/7`, routed Spearman above zero, and the normalized-oracle-gap upper
  bound below the frozen `0.46` ceiling;
- paired bootstrap lower bounds must improve top-1, Spearman, and normalized
  oracle gap over both `M0` and the permuted-local-scalar control;
- the global `G_delta` source-quality control has its own positive-gain gate;
- each target needs at least eight independent unlabeled support cases and at
  least 32 typed whole-case bootstrap resamples;
- model covariance/residual variance and independent whole-case support
  bootstrap dispersion are the only target-decision uncertainty components;
  descriptive technical-seed spread remains report-only and cannot change the
  combined standard error, LCB, selected source/action, or fallback;
- an uncertain or non-positive decision returns the exact base action `B`.

The final action geometry is exact and non-negotiable: eight source blocks of
128 samples per class plus, when authorized, one 128-sample/class source tail.

## Stage-60 artifact boundaries

Stage 60 is deliberately three separate jobs rather than one mutable runner:

1. the exact-tail development producer creates the sealed 504-row
   probability-ensemble response, retains the 4,536 descriptive seed-cell
   rows, and creates a separate 4,536-row label-free support-action-shift
   component table from the same classifier fits;
2. the target-support producer creates label-free eight-source features and
   per-case action-shift components from its dedicated support-only
   reservation/cache. It runs 81 resumable target/seed tasks and 729 exact
   classifier fits, then reconstructs the point and all 32 whole-case
   bootstrap feature surfaces;
3. the CPU-only policy-lock producer independently validates both surfaces,
   the support-only parent reservation, and the distinct Stage-70 reservation
   before fitting models and freezing actions.

The policy requires the support case map in the two target reservations to
match exactly. It also reconstructs the exact-tail development reservation,
requires its declared held-out target-evaluation map to equal the Stage-70
evaluation map, and proves that every development support/evaluation case is
globally disjoint from fresh target support/evaluation. These maps, the
per-query partition hashes, and their aggregate manifest hash are frozen into
the target-policy, action-library, and final-policy locks. The target-support
producer never receives target-evaluation embeddings, while the policy never
receives target labels. This keeps data production, model fitting, and blind
downstream evaluation independently reconstructable.

## Fresh Stage-70 evaluation

Stage 70 freezes and evaluates:

- `B`: immutable equal-union base;
- `U`: uniform residual tail;
- `G_delta`: independently gated global single-source tail;
- `R`: utility-aligned target-specific residual tail, or exact `B` on abstention;
- `P`: deterministic target-feature permutation control;
- all eight `H x e` single-source tails as terminal oracle diagnostics.

Every logical target/action/training-seed/generation-seed prediction is sealed
before labels open. The primary endpoint is the all-nine-seed probability
ensemble BACC, with target center (`n=9`) as the inference unit. Confirmation
requires positive one-sided lower confidence bounds for `R-B`, `R-G_delta`,
`R-U`, and `R-P`. Oracle rows cannot alter the policy.

## Workstation execution

The frozen runtime matches `delli2`: Xeon W-2265 (12C/24T), 125 GiB RAM, and
two 24 GiB RTX A5000 GPUs.

- one persistent spawned generation worker on each of `cuda:0` and `cuda:1`;
- parent process remains CUDA-free; TF32 and AMP are disabled;
- GPU generation and CPU classifier phases are disjoint;
- four classifier processes use three BLAS threads each;
- source and prediction caches are float32 NPY memmaps with hash-validated
  resume;
- optional `/data/local` scratch is copied atomically to canonical NFS output
  only after validation.

The exact-tail producer plans 81 source streams, 648 coarse prediction tasks,
and 5,184 exact classifier fits. Support probabilities are obtained by
concatenating support and evaluation embeddings inside each existing fit, so
the endpoint repair adds no classifier fits. The target-support producer uses
its own two persistent GPU workers, followed by 81 CPU action-probe tasks and
729 fits on four workers with three BLAS threads each. The policy-lock producer
is CPU-only and uses four model workers with three BLAS threads each. Stage 70
seals 1,053 logical action cells while deduplicating identical compositions
for compute only.

## Activation order

1. Materialize and validate the fresh development reservation/cache/manifest,
   the support-only reservation/cache, and the separate Stage-70 reservation.
   The latter reserves case identities but does not expose evaluation labels.
2. Activate and run the exact-tail producer, then require its reconstructive
   validator to pass.
3. Activate and run the independent target-support producer, then require its
   label-free surface validator to pass.
4. Activate the CPU-only policy-lock producer. It always emits a frozen policy:
   failed transfer or uncertainty gates produce exact `B`, which is a valid
   fail-closed scientific result rather than an execution failure.
5. Only after the policy lock exists, extract the separate Stage-70 target
   cache/scoring manifest and activate Stage 70. Never repoint consumed
   surfaces to these IDs.

The workspace runners fail closed while any entry is `planned`, any required
artifact is absent, or any split/hash/label-access contract drifts.

After each corresponding registry entry has been deliberately activated and
its fresh inputs have been hash-registered, the workstation launch order is:

```bash
env PYTHONPATH=.:src /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.routing_and_composition.uniform_b_v2_exact_tail_utility_surface.v1
env PYTHONPATH=.:src /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.routing_and_composition.uniform_b_v2_utility_aligned_target_support_surface.v1
env PYTHONPATH=.:src /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.routing_and_composition.uniform_b_v2_utility_aligned_residual_policy_lock.v1
env PYTHONPATH=.:src /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.frozen_policy_downstream.uniform_b_v2_utility_aligned_residual_fresh.v1
```

Do not use `--force` to bypass a planned status or unresolved provenance drift;
activation is an explicit data-contract step, not a runtime convenience flag.

## Consumed Stage-90 diagnostic

The separate
`midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router.v1`
runner applies the same exact-tail question to the already-consumed MIDOG++
validation surface. It is runnable now, but it is not a substitute for the
fresh Stage-60/70 chain.

For each outer target `H`, it seals exact `B` and single-tail predictions for
every `q != H`, then opens development labels and fits only rows whose query
and candidate source both exclude `H`. The all-center development-label phase
is disclosed explicitly. The fixed target proposal `R2` uses exactly two
unlabeled support cases, so its status is always
`INSUFFICIENT_SUPPORT_FOR_POLICY`; it cannot be promoted even if its mean BACC
is promising. Target `B/U/G_delta/R2/P` and all eight `Hxe` actions are frozen
and globally predicted before terminal scoring.

```bash
cd /home/stud/spark/cvae_experiments && env PYTHONPATH=.:src /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router.v1
```

The run creates a dedicated 270-row/class source cache, stages the finalized
memmap to experiment-scoped `/data/local` storage when available, uses one
persistent worker per A5000, and then switches to four CPU workers with three
BLAS threads each. Hash-bound task checkpoints make interruption recoverable.
Its output is terminal exploratory Stage-90 evidence and is forbidden from
feeding Stage 60, Stage 70, recipe selection, promotion, or deployment.

The completed diagnostic explains why the endpoint-aligned redesign is
necessary. Equal-union `B` reached mean ensemble BACC `0.770276`, while `R2`
reached `0.762182`; `R2-B` was `-0.008093` with 95% center-level interval
`[-0.034655, 0.018468]`. The terminal single-source oracle reached
`0.791928`, a descriptive `+0.021652` over `B`, but `R2` selected the exact
best source only `1/9` times, had mean target-wise Spearman `-0.000860`, and a
normalized oracle gap of `0.513228`. Moreover, the response used to fit the
old router was the mean of seed-cell BACC deltas, whereas Stage 70 evaluates
BACC after probability ensembling. The new family fixes both the response
unit and the aggregation functional; it does not reinterpret the oracle as an
attainable policy result.

This is still the MIDOG++ dataset family, not a cross-dataset experiment. A
fresh claim nevertheless requires new hash-bound MIDOG++ development,
support, and evaluation aliases with whole-case disjointness. Repointing any
of those aliases at the consumed Stage-90 rows would convert the run back into
a terminal diagnostic and is rejected by the loaders.
