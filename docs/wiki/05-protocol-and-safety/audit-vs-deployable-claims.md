# Audit Vs Deployable Claims

## Purpose

Define evidence labels and claim boundaries.

## Key Claims

| Evidence type | Meaning |
| --- | --- |
| Deployable evidence | Chosen without target evaluation labels. |
| Deployable diagnostic | Protocol-clean diagnostic evidence, not necessarily a complete deployable method. |
| Audit-only evidence | Diagnostic; cannot justify deployable claim. |
| Posthoc evidence | Uses target outcomes; feasibility only. |
| Oracle evidence | Upper bound using target outcomes. |
| Centralized diagnostic upper bound | Uses pooled or source-union information to diagnose feasibility or bottlenecks; not deployable as decentralized routing. |
| Dense aggregation evidence | Protocol-clean evidence that all available non-target experts can be reweighted or budgeted better than equal weighting; not evidence of sparse expert selection. |
| Raw-data-free summary exchange | Shares source-local summaries/scores rather than raw images or embeddings; not a formal privacy claim by itself. |
| Negative result | Valid result showing no useful gain. |
| Assumption | Plausible but not yet verified. |
| TODO | Missing or unverified evidence. |

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../PROTOCOL_STATUS.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/reports/decision_summary.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/manifests/protocol_manifest.json`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/leakage_report.json`

## Interpretation

Posthoc and oracle rows are valuable because they show headroom and diagnose regret. They cannot be used to choose a deployable method.

Likewise, centralized source-union GMM prior rows can diagnose that latent prior
sampling is the bottleneck, but they cannot by themselves support a
decentralized MoErging or raw-data-free deployment claim.

Raw-data-free source-local summary exchange is stronger than raw pooling, but
it remains a data-minimizing protocol, not formal differential privacy.

The paired dense-all4 reliability confirmation is a positive dense aggregation
result: heldout-excluded source-local reliability improves how all four
non-target source experts are weighted and budgeted. It must not be described
as sparse routing, because no source expert is excluded.

## Implication For Thesis

Every result section should state the strongest allowable claim and the forbidden overclaim.

## Limitations

Evidence labels depend on artifact fields. If labels are missing, classify the claim as incomplete until verified.

## Next Checks

- Add evidence labels to new wiki result pages as SAIL artifacts arrive.
- Keep dense aggregation labels separate from sparse routing labels.
