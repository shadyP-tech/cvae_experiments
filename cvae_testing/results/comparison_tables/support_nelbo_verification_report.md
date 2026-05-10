# Support-NELBO Verification Report

## Decision

- Classification: `Strong`
- Protocol gate: `pass`
- Count gate: `pass`
- Allowed claim: direct support-set NELBO is the primary support-estimated utility router and strongest tested support-estimated utility variant in the current Camelyon17 support experiment.
- Disallowed claim: support-estimated utility routing is generally robust across support regimes and domains.

Reasons:
- all Strong gates passed

## Headline Metrics

| Method | Rows | Top1 | Spearman | Oracle gap pct | High regret >2% |
| --- | --- | --- | --- | --- | --- |
| metadata_routing | 180 | 0.289 | 0.000 | 4.112 | 0.561 |
| static_embedding_routing | 180 | 0.511 | 0.560 | 3.389 | 0.433 |
| source_global_prior_routing | 180 | 0.178 | 0.006 | 7.981 | 0.706 |
| direct_support_nelbo | 180 | 0.822 | 0.848 | 0.622 | 0.100 |
| conservative_support_nelbo | 180 | 0.822 | 0.847 | 0.716 | 0.106 |

Direct headline check: top1=0.822, Spearman=0.848, oracle gap pct=0.622.

## Protocol And Counts

- Protocol failures: 0
- Expected direct decisions: 180
- Observed direct decisions: 180
- Expected direct candidate rows: 720
- Observed direct candidate rows: 720

## Support-Size Evidence

| Top1 k4 | Top1 k32 | Gap k4 | Gap k32 | Directional | Severe k16->k32 |
| --- | --- | --- | --- | --- | --- |
| 0.733 | 0.889 | 1.220 | 0.158 | 1 | 0 |

## Center 1 Stress Case

| k | Top1 | Spearman | Gap pct | High regret >2% |
| --- | --- | --- | --- | --- |
| 4 | 0.667 | 0.822 | 3.196 | 0.333 |
| 8 | 0.778 | 0.867 | 2.319 | 0.222 |
| 16 | 0.889 | 0.889 | 1.212 | 0.111 |
| 32 | 1.000 | 1.000 | 0.000 | 0.000 |

## Failure Anatomy

- Top1 failure count: 32
- Regret classes: {'catastrophic': 7, 'high_regret': 11, 'moderate_regret': 6, 'near_miss': 8}
- Support-confidence classes among failures: {'ambiguous_support': 17, 'normal_margin': 15}
- P(eval oracle support-rank <= 2): 0.967
- P(selected expert eval-rank <= 2): 0.956

| center | k | support_seed | selected | oracle | gap pct | support margin | regret | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 23 | 4 | 3 | 13.919 | 8.133 | catastrophic | ambiguous_support |
| 1 | 16 | 17 | 2 | 3 | 10.910 | 4.754 | catastrophic | ambiguous_support |
| 1 | 8 | 17 | 2 | 3 | 10.537 | 5.002 | catastrophic | ambiguous_support |
| 1 | 8 | 31 | 2 | 3 | 10.334 | 21.655 | catastrophic | normal_margin |
| 1 | 4 | 31 | 2 | 3 | 10.020 | 14.826 | catastrophic | normal_margin |
| 0 | 8 | 17 | 1 | 3 | 6.752 | 6.717 | catastrophic | ambiguous_support |
| 0 | 4 | 17 | 1 | 3 | 6.545 | 14.435 | catastrophic | normal_margin |
| 1 | 4 | 23 | 0 | 3 | 4.829 | 7.522 | high_regret | ambiguous_support |
| 4 | 4 | 17 | 2 | 1 | 4.314 | 25.722 | high_regret | normal_margin |
| 4 | 4 | 23 | 2 | 1 | 4.188 | 35.030 | high_regret | normal_margin |
| 2 | 4 | 23 | 0 | 4 | 3.868 | 1.243 | high_regret | ambiguous_support |
| 2 | 32 | 17 | 0 | 4 | 2.374 | 3.259 | high_regret | ambiguous_support |

## Margin Reliability

| k | bin | n | top1 | gap pct | high regret >2% | wrong confident |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | q1_low | 12 | 0.583 | 2.053 | 0.333 | 0.000 |
| 4 | q2_mid_low | 11 | 0.636 | 1.805 | 0.273 | 0.000 |
| 4 | q3_mid_high | 11 | 0.818 | 0.773 | 0.182 | 0.000 |
| 4 | q4_high | 11 | 0.909 | 0.174 | 0.000 | 0.000 |
| 8 | q1_low | 12 | 0.583 | 1.726 | 0.250 | 0.000 |
| 8 | q2_mid_low | 11 | 0.727 | 0.241 | 0.091 | 0.000 |
| 8 | q3_mid_high | 11 | 0.909 | 0.939 | 0.091 | 0.000 |
| 8 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 16 | q1_low | 12 | 0.750 | 1.013 | 0.083 | 0.000 |
| 16 | q2_mid_low | 11 | 0.818 | 0.335 | 0.091 | 0.000 |
| 16 | q3_mid_high | 11 | 0.909 | 0.047 | 0.000 | 0.000 |
| 16 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 32 | q1_low | 12 | 0.667 | 0.517 | 0.167 | 0.000 |
| 32 | q2_mid_low | 11 | 0.909 | 0.081 | 0.000 | 0.000 |
| 32 | q3_mid_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |
| 32 | q4_high | 11 | 1.000 | 0.000 | 0.000 | 0.000 |

## Adaptivity And Conservative Check

| expert | count | share |
| --- | --- | --- |
| 0 | 59 | 0.328 |
| 1 | 24 | 0.133 |
| 2 | 6 | 0.033 |
| 3 | 64 | 0.356 |
| 4 | 27 | 0.150 |

Direct/conservative disagreements: 2; direct wins: 1; conservative wins: 1; ties: 178.

## Bootstrap Stability

Skipped: current exported artifacts contain support means/stderr, not per-support-sample NELBO rows.
