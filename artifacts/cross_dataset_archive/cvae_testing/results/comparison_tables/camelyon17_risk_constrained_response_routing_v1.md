# Camelyon17 Risk-Constrained Response Routing v1

Protocol status: completed run, protocol-compliant from checked artifacts
Result status: WEAK PASS

Method under test: `metadata_anchored_response_routing_with_support_regret_gate`

Short label: `risk_constrained_response_routing`

## Evidence source

- Protocol version: `support_response_candidate_specific_v1`
- Aggregation unit: `seed_x_heldout_center_x_support_seed_x_support_size`
- Runs inspected: risk_constrained_response_seed42, risk_constrained_response_seed43, risk_constrained_response_seed44
- Compared against metadata routing, static embedding routing, unrestricted learned response routing, direct support-set NELBO top1, and candidate oracle diagnostics.

## Protocol checks

| Check | Status | Evidence |
|---|---:|---|
| Support/evaluation disjoint | ok | 450 split rows, 0 bad disjoint rows |
| Split status | ok | 0 non-ok split rows |
| Target expert exclusion | ok | 0 bad exclusion rows, 0 candidate-pool violations |
| Frozen thresholds | ok | 15 rows, source-inner-only and pre-scoring flags checked |
| Leakage report | ok | 0 duplicate paths, 0 patient-overlap entries |

## Aggregate metrics

| Method | Tier | Top1 | Spearman | Oracle gap pct | Harmful override | Override rate |
|---|---:|---:|---:|---:|---:|---:|
| metadata routing | baseline | 0.278 | 0.000 | 4.136 | n/a | n/a |
| risk-constrained response | weak_pass | 0.344 | 0.338 | 3.857 | 0.417 | 0.100 |
| unrestricted learned response | fail | 0.389 | 0.338 | 5.872 | 0.622 | 0.611 |
| static embedding | pass | 0.444 | 0.542 | 3.756 | n/a | n/a |
| direct support-set NELBO | pass | 0.833 | 0.889 | 0.471 | n/a | n/a |

## Seed stability

| Seed | Top1 | Spearman | Oracle gap pct | Gap reduction vs metadata | Harmful override | Override rate | Fallback centers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.167 | 0.200 | 4.931 | 0.355 | 0.000 | 0.133 | 3 |
| 43 | 0.400 | 0.440 | 4.484 | -0.077 | 1.000 | 0.033 | 3 |
| 44 | 0.467 | 0.373 | 2.156 | 0.557 | 0.250 | 0.133 | 0 |

## Decision

Classification: `WEAK PASS`.

The risk-constrained policy improves metadata routing on aggregate top1, Spearman, and mean oracle-gap percentage, and reduces harmful overrides relative to unrestricted learned-response routing. It is not result-level `PASS` because it fails the static-embedding non-regression check and remains far below direct support-set NELBO.

The gate fell back to metadata for 6 of 15 seed-by-held-out-center units. This partial collapse is thesis-useful evidence that learned response proposals are safer with a support-NELBO regret gate but remain fragile under domain shift.

Claim boundary: do not claim learned response routing beats metadata. The tested method is a metadata-anchored learned-response proposal with a support-NELBO regret gate.
