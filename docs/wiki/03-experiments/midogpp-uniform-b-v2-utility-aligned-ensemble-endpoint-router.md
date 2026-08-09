# MIDOG++ Uniform-B v2 Utility-Aligned Ensemble-Endpoint Router

## Status and question

`midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_router.v1`
is an active, runnable Stage-90 mechanism diagnostic. It asks whether a
candidate-specific ridge model can explain utility at the same exact-nine
probability-ensemble endpoint used for terminal evaluation, then form the
label-free fixed-support proposal `R2E`.

This is not fresh evidence. The MIDOG++ validation cases were already consumed,
method development is post-hoc, and two fixed unlabeled support cases per center
remain below the predeclared eight-case minimum for policy authorization. The
result is therefore always terminal `EXPLORATORY_CONSUMED_DATA_ONLY` evidence
with status `INSUFFICIENT_SUPPORT_FOR_POLICY`, regardless of the diagnostic
metrics.

## Frozen units and exclusions

The primary development table has 504 `(H,q,e)` rows. Each response first
averages the exact Cartesian product of three training and three generation
seed probabilities, applies one threshold, and computes the candidate-minus-
base BACC delta. The corresponding 4,536 seed-cell rows are descriptive only:
they are not independent observations and may not enter model fitting or
authorization uncertainty.

For each outer target `H`, development excludes `H` from both query and source
roles, and candidate source `e` also differs from pseudoquery `q`. Target labels,
target utility, the target expert, previous Stage-90 outputs, and every Stage-60
or Stage-70 output are unavailable to model fitting and action selection. All
development predictions are sealed before development labels open; all target
predictions are globally sealed before terminal target scoring.

The only inputs are the immutable expert bank, generation lock, experiment-
fenced consumed-validation cache and manifest aliases, and immutable metadata
profiles. The new aliases authorize only this experiment and do not create
fresh evidence.

## Execution contract

The registered workstation topology uses one generation worker on each of
`cuda:0` and `cuda:1`, followed by four classifier workers with three BLAS
threads each. TF32 and AMP are disabled. The frozen workload includes 81 source
streams, 504 primary ensemble-endpoint rows, 4,536 descriptive development seed
rows, 117 target actions, and 1,053 descriptive target action/seed identities.

Run the registered diagnostic with:

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_router.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_validation_utility_aligned_ensemble_endpoint_router/v1/
```

The output cannot update a policy, train another router, feed Stage 60 or 70,
select a recipe, authorize promotion or deployment, or establish routing or
target-performance claims.
