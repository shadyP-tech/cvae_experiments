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
- Support/eval disjointness, target expert exclusion, candidate-pool exclusion, eval-NELBO isolation, eval-statistics isolation, and alpha preselection are audited in `midogpp_scanner_patch_support_estimated_utility_routing_v1_protocol_audit.csv`.

### Utility performance

| Method | Role | Rows | Top1 | Spearman | Oracle gap pct | Selected eval NELBO |
| --- | --- | --- | --- | --- | --- | --- |
| metadata_routing | baseline | 144 | 0.174 | -0.132 | 11.159 | 407.136 |
| static_embedding_routing | baseline | 144 | 0.528 | 0.181 | 5.491 | 385.387 |
| random_expert_floor | diagnostic_floor | 144 | 0.382 | 0.056 | 8.640 | 397.840 |
| direct_support_nelbo | primary | 144 | 0.757 | 0.764 | 0.693 | 367.925 |
| conservative_support_nelbo | diagnostic_ablation | 144 | 0.757 | 0.760 | 0.693 | 367.925 |

Direct support NELBO: top1=0.757, Spearman=0.764, oracle gap pct=0.693.
Conservative support NELBO: top1=0.757, Spearman=0.760, oracle gap pct=0.693.

### Stability diagnostics

Conservative scoring does not improve small-k stability beyond direct support NELBO in the inspected groups.

| k | Method | Groups | Gap var | NELBO var | Spearman groups | Spearman dropped |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | direct_support_nelbo | 12 | 8.406 | 115.771 | 12 | 0 |
| 4 | conservative_support_nelbo | 12 | 8.406 | 115.771 | 12 | 0 |
| 8 | direct_support_nelbo | 12 | 0.596 | 18.420 | 12 | 0 |
| 8 | conservative_support_nelbo | 12 | 0.596 | 18.420 | 12 | 0 |
| 16 | direct_support_nelbo | 12 | 2.278 | 56.527 | 12 | 0 |
| 16 | conservative_support_nelbo | 12 | 2.278 | 56.527 | 12 | 0 |
| 32 | direct_support_nelbo | 12 | 2.353 | 40.756 | 12 | 0 |
| 32 | conservative_support_nelbo | 12 | 2.353 | 40.756 | 12 | 0 |

Spearman handling: precomputed Spearman values are used; rows with constant predicted or eval score vectors are treated as undefined for variance. Ties are otherwise retained from the upstream rank calculation.

Warnings:
- No Spearman variance groups were dropped.

## Direct vs conservative by k

| k | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap | Direct high-regret | Cons high-regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.694 | 0.694 | 1.320 | 1.320 | 0.000 | 0.167 | 0.167 |
| 8 | 0.806 | 0.806 | 0.240 | 0.240 | 0.000 | 0.028 | 0.028 |
| 16 | 0.806 | 0.806 | 0.530 | 0.530 | 0.000 | 0.056 | 0.056 |
| 32 | 0.722 | 0.722 | 0.681 | 0.681 | 0.000 | 0.111 | 0.111 |

## Alpha degeneracy

| alpha | count | pct |
| --- | --- | --- |
| 0.0 | 42 | 87.500 |
| nonzero | 6 | 12.500 |

Alpha mostly collapses to direct support NELBO, so conservative scoring is not meaningfully regularizing routing in this run.

## Per-center oracle gap

| Center | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 1 | 0.972 | 0.972 | 0.471 | 0.471 | 0.000 |
| 2 | 0.500 | 0.500 | 1.470 | 1.470 | 0.000 |
| 3 | 0.556 | 0.556 | 0.829 | 0.829 | 0.000 |

## Earlier-artifact cross-check



Allowed thesis claim: Direct support-set NELBO is the primary support-estimated utility router and the strongest support-estimated utility variant in the current Camelyon17 support experiment.

Not allowed: Conservative support NELBO improves small-k stability, or alpha regularization is meaningful in this run.
