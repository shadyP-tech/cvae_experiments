# CVAE Generative Experts

## Purpose

Summarize the role of CVAEs as source-domain generative experts.

## Key Claims

- CVAEs remain the intended generative expert family.
- CVAE experts should be judged by downstream utility preservation, not only reconstruction.
- A strong real-feature classifier result does not prove that a CVAE can model the feature space without utility loss.
- The latest Virchow2 CVAE artifacts show that latent prior sampling and
  decentralized composition are the main generated-embedding bottlenecks.

## Evidence / Source Artifacts

- `../../context/thesis_project_context.md`
- `../../../cvae_downstream_evaluation/README.md`
- `../../../cvae_downstream_evaluation/docs/protocol.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_preservation_repair_v1/reports/decision_summary.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_source_union_gmm_prior_v1/tables/gmm_prior_gap_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_weighted_gmm_prior_v1/tables/decentralized_reliability_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/reports/decision_summary.md`

## Interpretation

The CVAE question is whether generated Virchow2 embeddings preserve the utility
observed in real-feature Virchow2 transfer. The latest artifacts partially
answer this:

```text
decode(mu) / source-pool capacity can be useful
standard prior sampling is weak
source-union K16 GMM is a strong centralized diagnostic
paired dense all4 reliability weighting improves dense aggregation
```

## Implication For Thesis

The thesis should frame Virchow2 CVAE work as utility-preservation and
composition evidence. Current results justify discussing prior/composition
bottlenecks and a dense aggregation win, but not claiming a solved
target-conditioned router or sparse expert selector.

## Limitations

The D-series CVAE artifacts used here are synced under this working repo's
`cvae_rebuild/artifacts/` directory.

## Next Checks

- Keep generated-embedding utility and fidelity diagnostics separate.
- Use the D-series page for current decentralized composition evidence.
- Do not promote source-union K16 from diagnostic upper bound to deployable
  method.
