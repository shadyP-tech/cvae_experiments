# Camelyon17 Risk-Constrained Response Routing v1

Protocol status: completed run, protocol-compliant from checked artifacts
Result status: WEAK PASS

Method under test: `metadata_anchored_response_routing_with_support_regret_gate`

Short label: `risk_constrained_response_routing`

## Evidence source

- Protocol version: `support_response_candidate_specific_v1`
- Aggregation unit: `seed_x_heldout_center_x_support_seed_x_support_size`
- Runs inspected: support_utility_v2_seed42, support_utility_v2_seed43, support_utility_v2_seed44
- Compared against metadata routing, static embedding routing, unrestricted learned response routing, direct support-set NELBO top1, and candidate oracle diagnostics.

## Protocol checks

| Check | Status | Evidence |
|---|---:|---|
| Support/evaluation disjoint | ok | 900 split rows, 0 bad disjoint rows |
| Split status | ok | 0 non-ok split rows |
| Target expert exclusion | ok | 0 bad exclusion rows, 0 candidate-pool violations |
| Frozen thresholds | ok | 15 rows, source-inner-only and pre-scoring flags checked |
| Leakage report | ok | 0 duplicate paths, 0 patient-overlap entries |

## Aggregate metrics

| Method | Tier | Top1 | Spearman | Oracle gap pct | Harmful override | Override rate |
|---|---:|---:|---:|---:|---:|---:|
| metadata routing | baseline | 0.289 | 0.000 | 4.112 | n/a | n/a |
| risk-constrained response | weak_pass | 0.400 | 0.326 | 3.522 | 0.306 | 0.183 |
| unrestricted learned response | fail | 0.378 | 0.326 | 6.323 | 0.675 | 0.656 |
| static embedding | pass | 0.511 | 0.560 | 3.389 | n/a | n/a |
| direct support-set NELBO | pass | 0.822 | 0.848 | 0.622 | n/a | n/a |
| conservative support-set NELBO | pass | 0.822 | 0.847 | 0.716 | 0.000 | 0.000 |

## Seed stability

| Seed | Top1 | Spearman | Oracle gap pct | Gap reduction vs metadata | Harmful override | Override rate | Fallback centers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.217 | 0.213 | 5.023 | 0.288 | 0.214 | 0.233 | 2 |
| 43 | 0.500 | 0.400 | 3.338 | 1.028 | 0.455 | 0.183 | 1 |
| 44 | 0.483 | 0.363 | 2.205 | 0.454 | 0.250 | 0.133 | 0 |

## Decision

Classification: `WEAK PASS`.

The risk-constrained policy improves metadata routing on aggregate top1, Spearman, and mean oracle-gap percentage, and reduces harmful overrides relative to unrestricted learned-response routing. It is not result-level `PASS` because it fails the static-embedding non-regression check and remains far below direct support-set NELBO.

The gate fell back to metadata for 3 of 15 seed-by-held-out-center units. This partial collapse is thesis-useful evidence that learned response proposals are safer with a support-NELBO regret gate but remain fragile under domain shift.

Claim boundary: do not claim learned response routing beats metadata. The tested method is a metadata-anchored learned-response proposal with a support-NELBO regret gate.
