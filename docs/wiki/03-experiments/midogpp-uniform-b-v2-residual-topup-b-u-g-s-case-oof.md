# MIDOG++ Uniform-B v2 Residual Top-up B/U/G/S Case-OOF

## Status

- Experiment: `midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1`
- Stage: `90_oracles_and_diagnostics`
- Status: `EXPLORATORY_CONSUMED_DATA_ONLY`
- Claim: terminal mechanism decomposition only
- Fresh confirmation or promotion: forbidden

## Question

The earlier fixed residual top-up improved over matched uniform allocation, but
its score was dominated by source identity. This diagnostic asks whether the
mechanism is explained by a global source preference (`G`) or whether fixed
unlabeled target support adds value (`S-G`). It does not fit or select a router.

## Frozen actions

- `B`: original equal-union base, 1,024 rows per class.
- `U`: `B` plus an exact uniform 128-row/class tail.
- `G`: `B` plus a Borda tail from leave-`H`/leave-`q` global support ballots.
- `S`: `B` plus a Borda tail from the two fixed unlabeled support cases `S_H`.
- `P`: fixed source-identity permutation of `S`.
- `H x e`: eight sealed single-source tails for diagnostic headroom only.

All three expert replicas are averaged before each case ballot. Ballots use
true normalized midranks with lower proxy energy better, and routing weights
use the explicit direction transform `1-b`. No temperature, strength, budget,
seed, expert, source, selector, or fallback is fitted.

## Split and label boundary

The input is the already-consumed 2,615-row, 44-case MIDOG++ validation surface.
Two whole cases per center are frozen as label-free support. The other 26 cases
are whole-case OOF scoring folds and each appears exactly once. `S` never uses
another evaluation embedding; `G` uses only fixed support cases from `q != H`
and excludes both `H` and `q`. Frozen source experts are never updated.

All target/action/training-seed/generation-seed predictions are persisted and
covered by one global seal before the label-bearing manifest opens. Labels are
then available only to terminal scoring. The `H x e` matrix emits no action or
policy update.

## Endpoint and inference

The primary endpoint is BACC from the arithmetic mean of all nine seed-cell
probabilities. Primary center-level contrasts are `S-U` and `S-G`. Secondary
contrasts are `G-U`, `U-B`, and `S-B`; `S-P` is a permutation diagnostic.
Confidence intervals, wins, ties, and losses use the nine target centers as the
independent units. Seed cells are technical repeats.

## Workstation command

```bash
cd /home/stud/spark/cvae_experiments && env PYTHONPATH=.:src \
  /home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_residual_topup_b_u_g_s_case_oof.v1
```

The runner uses one persistent spawned worker per RTX A5000, a phase-disjoint
four-worker classifier pool with three BLAS threads per worker, TF32 disabled,
float32 memmaps, and hash-validated resumable checkpoints.

## Interpretation boundary

A positive `S-U` and `S-G` would strengthen post-hoc mechanism evidence only.
If `S-U` is positive but `S-G` is not, the supported interpretation is a global
source-prior mechanism, not target-specific routing. Any promotion still needs
the separately planned fresh Stage-60 policy lock and Stage-70 evaluation on a
new unconsumed case-disjoint surface.
