# MIDOG++ Uniform-B canonical real-feature reference v1

Status: `PROMOTED_AS_NEW_CANONICAL_REFERENCE`

Canonical output:

```text
artifacts/midogpp/10_real_feature_reference/
  uniform_b_canonical_real_feature_reference_v1/seed42/
```

## Purpose

This separately reviewed Stage-10 experiment promotes the fixed Uniform-B
representation `annotation_jpeg_fixed_center_b_v3` as a new canonical
real-feature reference. It does not overwrite, rename, or invalidate canonical
A or any completed A-based experiment.

The promotion is supported by the prospective Phase-B result:

- equal-center test BACC: `0.799159` for B versus `0.735733` for A;
- paired delta: `+0.063426`;
- strict wins: `9/9`;
- paired case-bootstrap 95% interval: `[0.050709, 0.073650]`.

Those test outcomes authorize the representation decision only. The MIDOG++
test split is now recorded as consumed for representation adoption and cannot
be presented as fresh representation-selection evidence again.

## Operational reference

The promotion runner does not import the Phase-A or Phase-B classifier locks.
It standardizes the validated train shards into one exact-identity 3,840
dimensional cache, then reruns the canonical ten-spec classifier grid using
source-inner center LODO separately for every outer center.

The resulting train-LODO reference has:

- nine held-out centers;
- 90 tuning rows;
- 9,648 held-out predictions;
- equal-center mean BACC `0.792087`;
- validator status `PASS`.

Selected classifiers use `C=0.01`; centers 0, 7, and 8 select balanced class
weights, while the other six select no class weighting.

## Claim and migration boundary

The allowed claim is `real_feature_transfer_only` for new cases in the
observed MIDOG++ center population. New-center and external-dataset
generalization are not claimed.

Canonical A remains available. This experiment does not automatically edit
Stage-20 or later configs. A downstream experiment adopts B only by explicitly
declaring both:

1. `midogpp_virchow2_uniform_b_canonical_train_cache_seed42`; and
2. `midogpp_output_uniform_b_canonical_reference_v1`.

That explicit migration must also update the expected feature dimension from
2,560 to 3,840 and must preserve the test-consumption boundary.

## Reproduction

```bash
python -m midogpp_thesis real-feature-classifier \
  build-uniform-b-canonical-train-cache \
  --config datasets/midogpp/configs/uniform_b_canonical_train_cache_v1.yaml

python -m midogpp_thesis workspace run \
  midogpp.real_feature.uniform_b_canonical_reference.v1
```

The registered run uses eight independent process workers while each
individual numerical fit remains single-threaded. Serial and process-worker
selection paths are regression-tested to produce identical selected
configurations and aggregate scores.
