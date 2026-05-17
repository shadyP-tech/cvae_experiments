# Support-NELBO Consolidation Report

This report supersedes earlier selection summaries for thesis interpretation. It does not invalidate their protocol checks; it revises the claim level based on per-k support-seed stability and alpha-selection diagnostics.

## Thesis-facing decision

- Primary method: `direct_support_nelbo`
- Conservative method: `diagnostic ablation only`
- Result wording: Direct support-set NELBO is the primary support-estimated utility router.
- Claim boundary: direct support-set NELBO is the strongest support-estimated utility variant in the current Camelyon17 support experiment; this report does not make a broader overall router-ranking claim.
- Reason: conservative scoring is protocol-safe, but alpha selection is mostly degenerate and does not demonstrate a stable small-k improvement.

## Decision layers

### Protocol validity

- Overall protocol validity: `pass`
- Support/eval disjointness, target expert exclusion, candidate-pool exclusion, eval-NELBO isolation, eval-statistics isolation, and alpha preselection are audited in `support_nelbo_protocol_audit.csv`.

### Utility performance

| Method | Role | Rows | Top1 | Spearman | Oracle gap pct | Selected eval NELBO |
| --- | --- | --- | --- | --- | --- | --- |
| metadata_routing | baseline | 180 | 0.278 | 0.000 | 4.110 | 434.238 |
| static_embedding_routing | baseline | 180 | 0.489 | 0.569 | 3.419 | 432.194 |
| direct_support_nelbo | primary | 180 | 0.833 | 0.853 | 0.547 | 421.352 |
| conservative_support_nelbo | diagnostic_ablation | 180 | 0.828 | 0.850 | 0.651 | 421.682 |

Direct support NELBO: top1=0.833, Spearman=0.853, oracle gap pct=0.547.
Conservative support NELBO: top1=0.828, Spearman=0.850, oracle gap pct=0.651.

### Stability diagnostics

Conservative scoring does not demonstrate stable small-k improvement; oracle-gap support-seed variance is higher than direct support NELBO for k=8.

| k | Method | Groups | Gap var | NELBO var | Spearman groups | Spearman dropped |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | direct_support_nelbo | 15 | 3.236 | 53.141 | 15 | 0 |
| 4 | conservative_support_nelbo | 15 | 3.236 | 53.141 | 15 | 0 |
| 8 | direct_support_nelbo | 15 | 4.165 | 79.319 | 15 | 0 |
| 8 | conservative_support_nelbo | 15 | 9.444 | 118.059 | 15 | 0 |
| 16 | direct_support_nelbo | 15 | 1.915 | 77.958 | 15 | 0 |
| 16 | conservative_support_nelbo | 15 | 1.915 | 77.958 | 15 | 0 |
| 32 | direct_support_nelbo | 15 | 0.161 | 74.680 | 15 | 0 |
| 32 | conservative_support_nelbo | 15 | 0.161 | 74.680 | 15 | 0 |

Spearman handling: precomputed Spearman values are used; rows with constant predicted or eval score vectors are treated as undefined for variance. Ties are otherwise retained from the upstream rank calculation.

Warnings:
- No Spearman variance groups were dropped.

## Direct vs conservative by k

| k | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap | Direct high-regret | Cons high-regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.778 | 0.778 | 0.917 | 0.917 | 0.000 | 0.156 | 0.156 |
| 8 | 0.800 | 0.778 | 0.791 | 1.211 | -0.419 | 0.111 | 0.133 |
| 16 | 0.889 | 0.889 | 0.338 | 0.338 | 0.000 | 0.044 | 0.044 |
| 32 | 0.867 | 0.867 | 0.140 | 0.140 | 0.000 | 0.044 | 0.044 |

## Alpha degeneracy

| alpha | count | pct |
| --- | --- | --- |
| 0.0 | 53 | 88.333 |
| nonzero | 7 | 11.667 |

Alpha mostly collapses to direct support NELBO, so conservative scoring is not meaningfully regularizing routing in this run.

## Per-center oracle gap

| Center | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.944 | 0.917 | 0.381 | 0.905 | -0.524 |
| 1 | 0.861 | 0.861 | 1.312 | 1.312 | 0.000 |
| 2 | 0.694 | 0.694 | 0.436 | 0.436 | 0.000 |
| 3 | 0.917 | 0.917 | 0.162 | 0.162 | 0.000 |
| 4 | 0.750 | 0.750 | 0.443 | 0.443 | 0.000 |

## Earlier-artifact cross-check

Changed conclusion is a claim-level revision based on per-k support-seed stability and alpha diagnostics, not a protocol invalidation.

Allowed thesis claim: Direct support-set NELBO is the primary support-estimated utility router and the strongest support-estimated utility variant in the current Camelyon17 support experiment.

Not allowed: Conservative support NELBO improves small-k stability, or alpha regularization is meaningful in this run.
