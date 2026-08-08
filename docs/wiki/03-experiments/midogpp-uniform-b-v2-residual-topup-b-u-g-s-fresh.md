# MIDOG++ Uniform-B V2 Residual Top-Up B/U/G/S Fresh Study

## Status

`IMPLEMENTED AS A FAIL-CLOSED PLANNED STUDY; FRESH DATA SURFACE ABSENT`.

The code and registered contracts define the decisive fixed-policy
decomposition of the promising Stage-90 residual-top-up mechanism. The
current checkout cannot execute or promote this study because it has no new,
unconsumed, case-disjoint pseudoquery/support/evaluation surface. Consumed
Stage-70 test rows and Stage-90 validation rows are explicitly ineligible.

## Frozen Stage-60 actions

For held-out target center `H`, all actions exclude expert `H` and retain the
same eight-source equal-union base of 128 generated rows per source and class
(1,024/class):

- `B` (`base_equal_union`): the base only.
- `U` (`uniform_residual_topup`): B plus an equal 128/class tail.
- `G` (`global_rank_residual_topup`): B plus a 128/class tail allocated from
  leave-target-out global proxy ballots.
- `S` (`support_rank_residual_topup`): B plus a 128/class tail allocated from
  the target's unlabeled, evaluation-disjoint support ballots.
- `P` (`support_rank_permutation_control`): the fixed source-identity
  permutation control for S: canonical candidate order rotated by exactly one
  position, frozen before any downstream outcome.
- `single_source_tail::<e>`: one sealed diagnostic action for every legal
  `H x e` pair.

The score is the existing label-blind proxy: shared-space posterior-mean
reconstruction MSE plus latent-dimension-normalized analytic KL to the PS
prior, marginalized over fixed class hypotheses `(0.5, 0.5)`. It is not
NELBO, likelihood, downstream utility, or a BACC prediction.

Every case ballot first averages all three training replicas. True midranks
are normalized as

\[
b_{q,e}=\frac{\operatorname{midrank}_{q}(e)-1}{m_q-1},
\]

with lower values better. Hamilton top-up priority is `1 - mean(b)`. Tied
scores stay tied; canonical source order is used only for deterministic
integer remainder allocation. G removes both `H` and pseudo-target `q` before
each ballot. S uses only unlabeled support from `H`. There is no utility
selector, fallback gate, temperature, tuned residual strength, or
empirical-Bayes shrinkage.

## Frozen Stage-70 endpoint

The policy lock and all B/U/G/S, P, and `H x e` actions must exist before the
fresh target cache is extracted. Every action/seed prediction is sealed before
evaluation labels open.

The operational endpoint is the all-nine-seed probability-ensemble BACC at a
fixed 0.5 threshold. The target center is the inference unit (`n = 9`); seed
cells are technical repeats. The primary contrasts are `S-U` and `S-G`.
Secondary contrasts are `G-U`, `U-B`, and `S-B`; `S-P` is a diagnostic
permutation contrast. Reports include center-paired means, 95% t intervals,
one-sided 95% lower bounds, and wins/ties/losses.

The sealed `H x e` matrix reports oracle headroom, support-score/utility
Spearman, top-1 agreement, and normalized oracle gap. Those labels and oracle
identities are terminal evaluation outputs and may never update Stage 60.

## Workstation execution contract

The live `xai-master` profile is a 12-core/24-thread Xeon W-2265 workstation
with 125 GiB RAM and two 24 GiB RTX A5000 GPUs. The frozen schedule uses:

- one persistent spawned process per GPU and one expert per GPU at a time;
- deterministic 256/class float32 source-prefix caches with hash-validated
  resume checkpoints;
- a phase-disjoint CPU pool of four classifier workers with three BLAS threads
  each;
- TF32 and AMP disabled, a CUDA-free parent process, and explicit GPU/RAM/disk
  launch gates;
- artifact-specific `/data/local` scratch enabled by the registered workstation
  command only as a performance cache, followed by hash-validated atomic
  publication to the canonical artifact root.

The fresh proxy producer is deliberately an API, not an ad-hoc fallback CLI:
`residual_topup_policy.materialize_fresh_proxy_inputs` accepts only validated
`FreshQueryShard` objects and writes exactly `tables/proxy_scores.csv` plus
`manifests/fresh_surface_attestation.json`. The attestation binds all 81 shard
hashes, the canonical surface hash, upstream locks, and the CSV. A future data
reservation must own those shards under the registered artifact identity before
either experiment can be changed from `planned` to `active`.

The registry entries remain `planned`. They become runnable only after the
fresh proxy surface, Stage-70 reservation, label-blind target cache, and
scoring manifest are materialized and independently validated under their
registered identities.
