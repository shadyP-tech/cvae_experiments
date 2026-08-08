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
Stage 60 measures

```text
delta(H,q,e) = BACC(B_q + exact_tail_e; E_q) - BACC(B_q; E_q)
```

with `H != q` and `e` excluded from both `H` and `q`. The inner action uses the
cardinality-matched 12.5% residual geometry: seven sources at 144 samples per
source/class plus one exact 126-sample/class tail. All nine fixed
training/generation seed pairs are retained. Predictions are globally sealed
before the dedicated development scoring labels open.

The model is a low-dimensional, utility-aligned source-preference model. Its
features are label-free reconstruction/KL components, technical-replica
disagreement, linear-kernel MMD-squared, metadata similarity, and within-query
rank features. Source and query exclusions are nested. Six-candidate training
to seven-candidate validation is only an eligibility test for the later
seven-to-eight transfer; it is not target-routing evidence.

Activation is deliberately conservative:

- query-domain bootstrap lower bounds must beat chance top-1, zero Spearman,
  and zero selected gain;
- normalized oracle gap must improve and remain below the frozen ceiling;
- the global `G_delta` source-quality control has its own positive-gain gate;
- each target needs at least eight independent unlabeled support cases and at
  least 32 typed whole-case bootstrap resamples;
- model covariance, held-out query-domain residual uncertainty, technical seed
  spread, and support-bootstrap uncertainty all enter the decision;
- an uncertain or non-positive decision returns the exact base action `B`.

The final action geometry is exact and non-negotiable: eight source blocks of
128 samples per class plus, when authorized, one 128-sample/class source tail.

## Stage-60 artifact boundaries

Stage 60 is deliberately three separate jobs rather than one mutable runner:

1. the exact-tail development producer creates the sealed, label-scored
   source-inner utility surface;
2. the target-support producer creates only label-free eight-source point and
   whole-case-bootstrap feature surfaces from its dedicated support-only
   reservation/cache;
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
and 5,184 exact classifier fits. The target-support producer uses its own two
persistent GPU workers; the policy-lock producer is CPU-only and uses four
model workers with three BLAS threads each. Stage 70 seals 1,053 logical action
cells while deduplicating identical compositions for compute only.

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
