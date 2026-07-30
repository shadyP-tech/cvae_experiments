# MIDOG++ Uniform-B v3 Prospective Test Confirmation v1

## Purpose and design

Phase A discovered that applying the 3,840-dimensional canonical-JPEG
fixed-center B representation uniformly improved mean BACC on the nine train
outer folds. Because that hypothesis was chosen after those outcomes were
known, Phase B was frozen before any B feature or prediction was generated for
the independent test split.

The confirmation surface contains 9,928 eligible test rows from the same nine
MIDOG++ centers. It has zero sample and case overlap with the 9,648 train rows
used for discovery. Validation rows are unused. Test labels are carried only
for final scoring; they do not enter feature extraction, preprocessing,
classifier fitting, lock selection, or the representation choice.

The test B cache uses the identical pinned Virchow2 revision, preprocessing,
fixed token window `(rows 6:10, columns 6:10)`, and 3,840-dimensional pooling
recipe as Phase A. Its 2,560-dimensional prefix passes the frozen numeric
bridge against the canonical A test cache. The cache is independently
validated, content-indexed, and fully hash-promoted before scoring.

## Frozen estimand and gate

For outer center `H`, both A and B classifiers are refit on train rows from the
other eight centers using the representation-specific classifier specification
already frozen in the source-v3 lock. They are then evaluated on the untouched
test rows from `H`.

The primary estimand is equal-center mean test BACC for B minus A. Confirmation
requires all four conditions:

1. mean BACC delta at least `+0.02`;
2. at least six of nine strict center wins;
3. worst-center delta at least `-0.01`;
4. conditional paired case-bootstrap 95% lower bound greater than zero.

No p-value, threshold tuning, alternate representation, validation-set
selection, or post-hoc gate is permitted.

## Result

Decision: `CONFIRMED_WITHIN_CENTER`.

| Held-out center | A test BACC | B test BACC | B minus A |
| --- | ---: | ---: | ---: |
| 0 | `0.696593` | `0.768267` | `+0.071673` |
| 1 | `0.705686` | `0.775043` | `+0.069357` |
| 2 | `0.716511` | `0.803115` | `+0.086604` |
| 3 | `0.713615` | `0.808294` | `+0.094679` |
| 5 | `0.786624` | `0.810510` | `+0.023885` |
| 6 | `0.754717` | `0.807278` | `+0.052561` |
| 7 | `0.794326` | `0.804965` | `+0.010638` |
| 8 | `0.739669` | `0.833333` | `+0.093664` |
| 9 | `0.713855` | `0.781627` | `+0.067771` |
| equal-center mean | `0.735733` | `0.799159` | `+0.063426` |

B wins on all nine centers, and even the smallest improvement remains positive.
The 2,000-replicate conditional paired case bootstrap has mean delta
`+0.062661` and a 95% percentile interval of `[+0.050709, +0.073650]`. Every
predeclared gate check passes. The independent validator reconstructs 18
result rows, 19,856 predictions, nine paired comparisons, and the final
decision.

## Interpretation

The train discovery (`+0.051775`) reproduces and strengthens on new,
case-disjoint test examples (`+0.063426`). The gain is not driven solely by a
few centers: B improves all nine test centers. This supports B as a uniform
representation for new cases drawn from the currently observed center
populations under the frozen classifier policy.

The experiment does not sample a new center, lab, scanner population, or
external dataset. The nine test center identities match the nine discovery
center identities, so the result does not estimate new-center uncertainty.
It also does not establish calibration, deployment utility, CVAE preservation,
routing, generation, or downstream synthetic utility.

The output remains Stage-90 `DIAGNOSTIC ONLY`. The existing workspace firewall
prevents the discovery lineage from automatically revising the canonical
Stage-10 reference. A separate protocol review is required before defining a
new B-based canonical reference, and external or new-center evidence remains
the next validation step.

## Reproduction

```bash
/home/stud/spark/.venvs/cvae-breakhis/bin/python -m midogpp_thesis workspace run \
  midogpp.oracle.uniform_b_v3_prospective_test_confirmation.v1
```

Canonical artifact:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_v3_prospective_test_confirmation_v1/seed42/
```
