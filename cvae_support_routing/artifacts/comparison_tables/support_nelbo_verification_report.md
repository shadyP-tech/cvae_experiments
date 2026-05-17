# Support-NELBO Verification Report

## Decision

- Classification: `Strong`
- Protocol gate: `pass`
- Count gate: `pass`
- Uncertainty support: `Moderate uncertainty support`
- Allowed claim: the support-NELBO routing result passes protocol and count gates; target-support resampling supports selected-expert stability, margin-dependent reliability, and low oracle-gap regret under support-selection uncertainty when the bootstrap gates pass.
- Disallowed claim: this bootstrap proves full held-out test-set uncertainty or general robustness across support regimes and domains.

Reasons:
- all Strong gates passed

## Headline Metrics

| Method | Rows | Top1 | Spearman | Oracle gap pct | High regret >2% |
| --- | --- | --- | --- | --- | --- |
| metadata_routing | 180 | 0.278 | 0.000 | 4.110 | 0.556 |
| static_embedding_routing | 180 | 0.489 | 0.569 | 3.419 | 0.433 |
| source_global_prior_routing | 180 | 0.194 | 0.039 | 7.888 | 0.700 |
| direct_support_nelbo | 180 | 0.833 | 0.853 | 0.547 | 0.089 |
| conservative_support_nelbo | 180 | 0.828 | 0.850 | 0.651 | 0.094 |

Direct headline check: top1=0.833, Spearman=0.853, oracle gap pct=0.547.

## Protocol And Counts

- Protocol failures: 0
- Expected direct decisions: 180
- Observed direct decisions: 180
- Expected direct candidate rows: 720
- Observed direct candidate rows: 720
- Expected raw support-NELBO rows: 10800
- Observed raw support-NELBO rows: 10800

## Support-Size Evidence

| Top1 k4 | Top1 k32 | Gap k4 | Gap k32 | Directional | Severe k16->k32 |
| --- | --- | --- | --- | --- | --- |
| 0.778 | 0.867 | 0.917 | 0.140 | 1 | 0 |

## Center 1 Stress Case

| k | Top1 | Spearman | Gap pct | High regret >2% |
| --- | --- | --- | --- | --- |
| 4 | 0.778 | 0.844 | 1.691 | 0.222 |
| 8 | 0.778 | 0.844 | 2.320 | 0.222 |
| 16 | 0.889 | 0.844 | 1.236 | 0.111 |
| 32 | 1.000 | 1.000 | 0.000 | 0.000 |

## Failure Anatomy

- Top1 failure count: 30
- Regret classes: {'catastrophic': 7, 'high_regret': 9, 'moderate_regret': 3, 'near_miss': 11}
- Support-confidence classes among failures: {'ambiguous_support': 15, 'normal_margin': 15}
- P(eval oracle support-rank <= 2): 0.967
- P(selected expert eval-rank <= 2): 0.961

| center | k | support_seed | selected | oracle | gap pct | support margin | regret | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16 | 17 | 2 | 3 | 11.125 | 4.326 | catastrophic | ambiguous_support |
| 1 | 8 | 17 | 2 | 3 | 10.736 | 3.101 | catastrophic | ambiguous_support |
| 1 | 8 | 31 | 2 | 3 | 10.139 | 24.808 | catastrophic | normal_margin |
| 1 | 4 | 31 | 2 | 3 | 9.859 | 27.208 | catastrophic | normal_margin |
| 0 | 8 | 17 | 1 | 3 | 6.966 | 9.885 | catastrophic | ambiguous_support |
| 0 | 4 | 17 | 1 | 3 | 6.740 | 14.984 | catastrophic | normal_margin |
| 1 | 4 | 23 | 0 | 3 | 5.363 | 2.883 | catastrophic | ambiguous_support |
| 4 | 4 | 17 | 2 | 1 | 4.337 | 17.721 | high_regret | normal_margin |
| 4 | 4 | 23 | 2 | 1 | 4.218 | 33.861 | high_regret | normal_margin |
| 2 | 4 | 23 | 3 | 4 | 3.848 | 7.070 | high_regret | ambiguous_support |
| 4 | 8 | 31 | 1 | 0 | 2.562 | 11.432 | high_regret | ambiguous_support |
| 4 | 4 | 23 | 1 | 0 | 2.437 | 9.435 | high_regret | ambiguous_support |

## Margin Reliability

| k | bin | n | top1 | gap pct | high regret >2% | wrong confident |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | q1_low | 12 | 0.750 | 0.971 | 0.250 | 0.000 |
| 4 | q2_mid_low | 11 | 0.636 | 1.231 | 0.182 | 0.000 |
| 4 | q3_mid_high | 11 | 0.818 | 1.280 | 0.182 | 0.000 |
| 4 | q4_high | 11 | 0.909 | 0.181 | 0.000 | 0.000 |
| 8 | q1_low | 12 | 0.500 | 1.904 | 0.250 | 0.000 |
| 8 | q2_mid_low | 11 | 0.818 | 0.237 | 0.091 | 0.000 |
| 8 | q3_mid_high | 11 | 0.909 | 0.922 | 0.091 | 0.000 |
| 8 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 16 | q1_low | 12 | 0.833 | 0.974 | 0.083 | 0.000 |
| 16 | q2_mid_low | 11 | 0.727 | 0.320 | 0.091 | 0.000 |
| 16 | q3_mid_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 16 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 32 | q1_low | 12 | 0.667 | 0.444 | 0.167 | 0.000 |
| 32 | q2_mid_low | 11 | 0.818 | 0.090 | 0.000 | 0.000 |
| 32 | q3_mid_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 32 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |

## Adaptivity And Conservative Check

| expert | count | share |
| --- | --- | --- |
| 0 | 57 | 0.317 |
| 1 | 25 | 0.139 |
| 2 | 6 | 0.033 |
| 3 | 66 | 0.367 |
| 4 | 26 | 0.144 |

Direct/conservative disagreements: 1; direct wins: 1; conservative wins: 0; ties: 179.

## Bootstrap Stability

- Bootstrap status: `pass`
- Bootstrap reps: 10000
- Bootstrap seed: 1337

| Method | k | n | Top1 | Top1 lo | Gap | Gap hi | High regret hi | Stability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_support_nelbo | 4 | 45 | 0.632 | 0.533 | 2.134 | 3.299 | 0.378 | 0.775 |
| conservative_support_nelbo | 8 | 45 | 0.739 | 0.644 | 1.451 | 2.364 | 0.289 | 0.822 |
| conservative_support_nelbo | 16 | 45 | 0.767 | 0.667 | 1.092 | 1.952 | 0.222 | 0.821 |
| conservative_support_nelbo | 32 | 45 | 0.815 | 0.733 | 0.408 | 0.917 | 0.156 | 0.886 |
| direct_support_nelbo | 4 | 45 | 0.641 | 0.533 | 2.068 | 3.187 | 0.356 | 0.774 |
| direct_support_nelbo | 8 | 45 | 0.745 | 0.644 | 1.343 | 2.220 | 0.267 | 0.823 |
| direct_support_nelbo | 16 | 45 | 0.766 | 0.667 | 1.097 | 1.915 | 0.222 | 0.821 |
| direct_support_nelbo | 32 | 45 | 0.815 | 0.733 | 0.407 | 0.903 | 0.156 | 0.886 |
