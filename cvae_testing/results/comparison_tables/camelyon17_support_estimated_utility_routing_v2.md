# Camelyon17 Support-Estimated Utility Routing v2

Protocol status: completed run, protocol-compliant from checked artifacts
Result status: PASS

Method under test: `support_set_nelbo_conservative`: mean support NELBO plus source-inner-selected alpha times support NELBO standard error.

Short label: `support_set_nelbo_conservative`

## Evidence source

- Protocol version: `support_response_candidate_specific_v1`
- Aggregation unit: `seed_x_heldout_center_x_support_seed_x_support_size`
- Runs inspected: support_utility_v2_seed42, support_utility_v2_seed43, support_utility_v2_seed44
- Compared against metadata routing, static embedding routing, direct support-set NELBO top1, risk-constrained response routing, unrestricted learned response routing, and candidate oracle diagnostics.

## Protocol checks

| Check | Status | Evidence |
|---|---:|---|
| Support/evaluation disjoint | ok | 900 split rows, 0 bad disjoint rows |
| Split status | ok | 0 non-ok split rows |
| Target expert exclusion | ok | 0 bad exclusion rows, 0 candidate-pool violations |
| Support-utility alpha selection | ok | 60 rows, source-inner-only and pre-scoring flags checked |
| Frozen thresholds | ok | 15 rows, source-inner-only and pre-scoring flags checked |
| Leakage report | ok | 0 duplicate paths, 0 patient-overlap entries |

## Aggregate metrics

| Method | Tier | Top1 | Spearman | Oracle gap pct | High-regret rate | Catastrophic rate | Harmful override | Override rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metadata routing | baseline | 0.289 | 0.000 | 4.112 | 0.561 | 0.561 | n/a | n/a |
| static embedding | pass | 0.511 | 0.560 | 3.389 | 0.433 | 0.433 | n/a | n/a |
| direct support-set NELBO | pass | 0.822 | 0.848 | 0.622 | 0.100 | 0.100 | n/a | n/a |
| conservative support-set NELBO | pass | 0.822 | 0.847 | 0.716 | 0.106 | 0.106 | n/a | n/a |
| risk-constrained response | weak_pass | 0.400 | 0.326 | 3.522 | 0.461 | 0.461 | 0.306 | 0.183 |
| unrestricted learned response | fail | 0.378 | 0.326 | 6.323 | 0.528 | 0.528 | 0.675 | 0.656 |
| candidate oracle | reference_only | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Seed stability

| Seed | Top1 | Spearman | Oracle gap pct | Gap reduction vs metadata | High-regret rate | Catastrophic rate |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.767 | 0.820 | 0.680 | 4.631 | 0.100 | 0.100 |
| 43 | 0.883 | 0.840 | 0.788 | 3.578 | 0.117 | 0.117 |
| 44 | 0.817 | 0.880 | 0.680 | 1.979 | 0.100 | 0.100 |

## Decision

Classification: `PASS`.

Classification follows the support-utility v2 rule: conservative support NELBO must match direct support NELBO within strict non-regression tolerance and improve stability or high-regret/catastrophic selection rate.

Selected methods: support_set_nelbo_conservative.

Claim boundary: Support-estimated utility routing treats compatibility as expected utility from unlabeled target-local support NELBO. Conservative support scoring may be claimed only if it satisfies direct-support non-regression and improves stability or high-regret selection rate. Do not claim learned response routing is the main solution for this experiment.

Risk-constrained response comparator:
the support-regret gate fell back to metadata for 3 of 15 seed-by-held-out-center units.
