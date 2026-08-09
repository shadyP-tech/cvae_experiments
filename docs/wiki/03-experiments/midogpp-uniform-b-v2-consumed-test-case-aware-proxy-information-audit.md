# MIDOG++ Consumed-Test Case-Aware Proxy-Information Audit

## Status

`midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_case_aware_proxy_information_audit.v1`
is a registered terminal Stage-90 diagnostic. The user explicitly authorizes
repurposing the already-consumed MIDOG++ test split for this better-powered,
post-hoc audit. It is always `EXPLORATORY_CONSUMED_DATA_ONLY`; no result from
it is fresh evidence.

## Frozen data and label boundary

The six inputs are the frozen expert bank, GenerationLock, a dedicated
label-free test-cache alias, a dedicated post-seal response-manifest alias,
immutable metadata profiles, and the test-consumption ledger. Stage-60 outputs,
Stage-70 prediction/scoring/policy outputs, and all earlier Stage-90 outputs
are forbidden.

The hash-pinned full manifest contains 218 eligible test cases across nine
centers. Seed `20260809` freezes eight whole support cases per center; the
remaining 146 cases are evaluation-only. Support features never receive
labels. Test labels open only after the complete prediction seal and are used
to construct the exact and smooth response rows. Those label-derived outcomes
feed separate strict diagnostic cross-fits; labels never enter features,
policy fitting, or action selection.

## Fixed screen

The primary response is the exact-nine probability-ensemble BACC delta for
each of 504 legal `(H,q,e)` candidates. SoftBACC is a post-seal descriptive
response. It is cross-fitted only in separate descriptive models and cannot
affect the primary exact-BACC model, gate, selection, or decision.

Three case-aware candidate families are compared with equal-union,
metadata-only, pooled row-weighted shift, and cyclic-permutation controls.
Every family has at most three predictors, ridge alpha is fixed at `1.0`, and
there is no tuning. Each cross-fit prediction excludes `H`, `q`, and `e` from
all three training roles, leaving exactly 120 rows. The nine outer centers are
the inference units; query, seed, case, and patch rows are not independent
replicates.

## Runtime and claim boundary

The frozen workload uses one source worker on each of two A5000 GPUs, then
four CPU workers with three BLAS threads each: 27 source jobs, 81 streams, 648
development tasks, and 5,184 fits. Resume is hash-validated and prefers
`/data/local` with the artifact parent as fallback; AMP and TF32 are disabled.

Even a positive screen cannot build or update a policy or target action, feed
Stage 60, Stage 70, recipe/deployable selection, or another Stage-90
experiment, or establish routing or downstream-utility success.

```bash
cd /home/stud/spark/cvae_experiments && \
env PYTHONPATH=.:src \
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v2_consumed_test_utility_aligned_case_aware_proxy_information_audit.v1
```

Canonical output:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v2_consumed_test_utility_aligned_case_aware_proxy_information_audit/v1/
```
