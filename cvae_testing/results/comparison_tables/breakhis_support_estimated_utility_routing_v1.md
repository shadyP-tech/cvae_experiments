# BreakHis Cross-Dataset Stress Test Of Direct Support-NELBO

This report supersedes earlier selection summaries for thesis interpretation. It does not invalidate their protocol checks; it revises the claim level based on per-k support-seed stability and alpha-selection diagnostics.

## Thesis-facing decision

- Primary method: `direct_support_nelbo`
- Conservative method: `diagnostic ablation only`
- Result wording: Direct support-set NELBO is the primary support-estimated utility router.
- PASS-rule verdict: `PASS`
- Claim boundary: direct support-set NELBO is stress-tested under BreakHis magnification-domain shift; this does not prove robustness across hospital, scanner, staining, lab, or patient-population shifts.
- Reason: conservative scoring is protocol-safe, but alpha selection is mostly degenerate and does not demonstrate a stable small-k improvement.

## Decision layers

### Protocol validity

- Overall protocol validity: `pass`
- Support/eval disjointness, target expert exclusion, candidate-pool exclusion, eval-NELBO isolation, eval-statistics isolation, and alpha preselection are audited in `breakhis_support_estimated_utility_routing_v1_protocol_audit.csv`.

### Utility performance

| Method | Role | Rows | Top1 | Spearman | Oracle gap pct | Selected eval NELBO |
| --- | --- | --- | --- | --- | --- | --- |
| metadata_routing | baseline | 144 | 0.257 | 0.000 | 24.686 | 497.943 |
| static_embedding_routing | baseline | 144 | 0.722 | 0.837 | 1.909 | 404.019 |
| random_expert_floor | diagnostic_floor | 144 | 0.389 | -0.017 | 14.881 | 457.898 |
| direct_support_nelbo | primary | 144 | 0.840 | 0.917 | 0.972 | 400.626 |
| conservative_support_nelbo | diagnostic_ablation | 144 | 0.840 | 0.917 | 0.972 | 400.626 |

Direct support NELBO: top1=0.840, Spearman=0.917, oracle gap pct=0.972.
Conservative support NELBO: top1=0.840, Spearman=0.917, oracle gap pct=0.972.

### Stability diagnostics

Conservative scoring does not improve small-k stability beyond direct support NELBO in the inspected groups.

| k | Method | Groups | Gap var | NELBO var | Spearman groups | Spearman dropped |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | direct_support_nelbo | 12 | 9.690 | 135.271 | 12 | 0 |
| 4 | conservative_support_nelbo | 12 | 9.690 | 135.271 | 12 | 0 |
| 8 | direct_support_nelbo | 12 | 3.651 | 107.646 | 12 | 0 |
| 8 | conservative_support_nelbo | 12 | 3.651 | 107.646 | 12 | 0 |
| 16 | direct_support_nelbo | 12 | 7.228 | 143.227 | 12 | 0 |
| 16 | conservative_support_nelbo | 12 | 7.228 | 143.227 | 12 | 0 |
| 32 | direct_support_nelbo | 12 | 2.934 | 714.837 | 12 | 0 |
| 32 | conservative_support_nelbo | 12 | 2.934 | 714.837 | 12 | 0 |

Spearman handling: precomputed Spearman values are used; rows with constant predicted or eval score vectors are treated as undefined for variance. Ties are otherwise retained from the upstream rank calculation.

Warnings:
- No Spearman variance groups were dropped.

## Direct vs conservative by k

| k | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap | Direct high-regret | Cons high-regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.806 | 0.806 | 1.355 | 1.355 | 0.000 | 0.139 | 0.139 |
| 8 | 0.833 | 0.833 | 0.866 | 0.866 | 0.000 | 0.139 | 0.139 |
| 16 | 0.861 | 0.861 | 1.126 | 1.126 | 0.000 | 0.139 | 0.139 |
| 32 | 0.861 | 0.861 | 0.541 | 0.541 | 0.000 | 0.083 | 0.083 |

## Alpha degeneracy

| alpha | count | pct |
| --- | --- | --- |
| 0.0 | 47 | 97.917 |
| nonzero | 1 | 2.083 |

Alpha mostly collapses to direct support NELBO, so conservative scoring is not meaningfully regularizing routing in this run.

## Per-center oracle gap

| Center | Direct top1 | Cons top1 | Direct gap | Cons gap | Direct-cons gap |
| --- | --- | --- | --- | --- | --- |
| 40 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 100 | 0.750 | 0.750 | 1.856 | 1.856 | 0.000 |
| 200 | 0.611 | 0.611 | 2.033 | 2.033 | 0.000 |
| 400 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## Earlier-artifact cross-check



Allowed thesis claim: Direct support-set NELBO was stress-tested as a target-local utility estimator under BreakHis leave-one-magnification-out routing using unlabeled patient-disjoint target support and held-out NELBO utility evaluation.

Not allowed: This experiment does not prove general support-NELBO robustness across all medical domain shifts; BreakHis magnification shift is narrower than hospital, scanner, staining, lab, or patient-population shift.
