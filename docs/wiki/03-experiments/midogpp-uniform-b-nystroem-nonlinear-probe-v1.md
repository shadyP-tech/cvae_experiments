# MIDOG++ Canonical Uniform-B Nyström Nonlinear Probe v1

## Status

`COMPLETE — NONLINEAR_B_DIAGNOSTIC_GATE_PASS`

This is a Stage-90 diagnostic over the already inspected 9,648-row train
surface. It does not replace the canonical B reference, authorize validation
scoring, or establish new-center generalization.

## Question

Does canonical B already contain useful class information that its frozen
linear logistic decision boundary cannot use?

The only modeled change was:

```text
StandardScaler
  -> Nyström approximation to an RBF kernel
  -> L2 logistic regression
```

For every outer center, all scaling, median-distance estimation, landmarks,
and fitting were performed only on the source-inner rows. The canonical B
class-weight choice for that outer center was inherited unchanged.

## Frozen search

- RBF width multiplier: `0.5`, `1`, `2`
- Nyström components: `256`, `512`, `1024`
- Logistic C: `0.01`, `0.1`, `1`, `10`
- Candidate count: `36`
- Primary landmark seed: `42`
- Stability landmark seeds: `17`, `101`
- Median-distance sample: deterministic source-only sample of at most `512`
- Width mapping: `sigma = multiplier * median_distance`
- Kernel mapping: `gamma = 1 / (2 * sigma^2)`

The runtime reused each of 36 unordered seven-center preprocessing frames for
both ordered outer/inner orientations. This produced 324 primary Nyström
transforms and 2,592 selector cells without changing the exact nested LODO
estimand.

## Primary result

| Quantity | Linear B | Nonlinear B | Delta |
| --- | ---: | ---: | ---: |
| Equal-center mean BACC | 0.792087 | 0.815278 | +0.023192 |
| Equal-center positive recall | — | — | +0.018308 |
| Equal-center specificity | — | — | +0.028075 |

The nonlinear model won strictly on all `9/9` centers. The worst primary
center delta was `+0.002008`. The case-within-center paired bootstrap was
supportive, with 2,000-replicate interval `[+0.011774, +0.032390]`.

Center deltas were:

| Center | Delta BACC |
| --- | ---: |
| 0 | +0.024396 |
| 1 | +0.029650 |
| 2 | +0.026439 |
| 3 | +0.034031 |
| 5 | +0.020767 |
| 6 | +0.008197 |
| 7 | +0.002008 |
| 8 | +0.019885 |
| 9 | +0.043353 |

All frozen progression checks passed, including the two supplemental landmark
seeds. Their equal-center gains were `+0.017142` and `+0.017197`; their worst
center deltas remained within the frozen `-0.01` floor.

## Error exchange

| Outcome | Rows |
| --- | ---: |
| Linear wrong, nonlinear correct | 751 |
| Linear correct, nonlinear wrong | 550 |
| Both wrong | 1,318 |
| Both correct | 7,029 |
| Net rescue | +201 |

Among 1,025 baseline errors whose source-only class centroid was already
closer to the true class, the nonlinear boundary rescued 613 (`59.8%`).
Among the 517 such errors made with baseline confidence at least 0.75, it
rescued 214 (`41.4%`).

The class exchange is directional: positives gained 285 net correct rows,
while negatives lost 84 net correct rows. Equal-center specificity still
improved overall because the tradeoff varied substantially by center, but
center 2 specificity declined by `-0.123972`. This is an important follow-up
target even though the predeclared aggregate uniformity gate passed.

## Interpretation

The result supports the classifier-limitation hypothesis: B contains useful
information that is not linearly separable. It therefore does not justify
replacing B with C or immediately constructing B-spatial. The next candidate
should keep B fixed and freeze a regularized nonlinear B+ decision function.

This is not proof that B is sufficient. There are still 1,318 rows that both
models miss, seed-to-seed regressions occur at centers 1 and 7, and the
center-2 sensitivity/specificity exchange needs explicit center-by-class
review.

## Validation reservation

The validation split was not featurized or scored. Its 2,615 eligible rows
cover all nine centers and have zero sample/case overlap with train and test,
but comprise only 44 cases and 3–7 cases per center. It remains reserved and
below the existing ten-case-per-center threshold for formal confirmation.

## Runtime and artifact

The CPU-optimized run used four processes with three threads each and completed
in `195.64` seconds. Both GPUs were intentionally unused.

Canonical diagnostic bundle:

```text
artifacts/midogpp/90_oracles_and_diagnostics/
  uniform_b_nystroem_nonlinear_probe_v1/seed42/
```

The bundle passed independent reconstruction of selector coverage, exact
canonical-B baseline row/hash identity, outer metrics, case bootstrap,
progression decision, split isolation, and content hashes.
