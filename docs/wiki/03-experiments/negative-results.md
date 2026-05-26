# Negative Results

## Purpose

Preserve negative findings as part of the research record.

## Key Claims

- Naive metadata/domain-ID conditioning did not solve compatibility-aligned utility under the locked historical protocol.
- Historical BreakHis routed CVAE metadata routing was technically correct but underperformed the global CVAE.
- Support-distance and learned/sparse top-1 routing signals were often too brittle.
- C7.1a source-probe CE is negative diagnostic evidence in the C6.3 synthesis.
- D1.3/D1.3.1 support-NELBO showed some alignment but did not validate a final
  target-conditioned compatibility selector because shuffled-support controls
  remained competitive.
- D1.4 reliability-only sparse top3 and D1.5 source-inner transfer drop-one
  selection are negative or diagnostic-only evidence for sparse source
  exclusion.
- The paired dense all4 reliability confirmation is positive for dense
  aggregation, but it does not overturn the negative sparse-selection evidence.

## Evidence / Source Artifacts

- `../../../cvae_testing/thesis_outline.txt`
- `../../../cvae_testing/results/compact_interpretation_summary.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `../../../cvae_testing/results/comparison_tables/learned_utility_decision_summary_v2.md`
- `../../../cvae_testing/results/comparison_tables/learned_utility_decision_summary_v2_strict.md`
- `../../../PROTOCOL_STATUS.md`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_support8_top3_tau05_gmm_prior_v1/tables/decentralized_support8_top3_tau05_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_reliability_top3_gmm_prior_v1/tables/decentralized_reliability_top3_summary.csv`
- `/Users/stephpark/Documents/Master/Thesis/cvae_experiments/cvae_rebuild/artifacts/virchow2_cvae_decentralized_source_inner_transfer_top3_gmm_prior_v1/tables/decentralized_source_inner_transfer_summary.csv`
- `../../../cvae_rebuild/artifacts/virchow2_cvae_paired_dense_all4_reliability_confirmation_v1/tables/paired_dense_all4_summary.csv`

## Interpretation

Negative results drove the pivot: metadata and similarity proxies are not enough
unless they predict expected utility. Dense aggregation is useful because the
problem often appears as high regret from brittle top-1 selection.

The latest D-series negative results sharpen this further:

```text
support-NELBO alignment alone is not enough if shuffled-support controls match
or beat the primary method
```

```text
source-inner off-diagonal transfer is not sufficient if its score is negatively
correlated with target subset utility
```

## Implication For Thesis

The thesis should present negative findings as evidence for claim discipline and for the move toward utility-aligned compatibility.

D1.5 in particular supports a thesis limitation statement: source-only
off-diagonal transfer among the remaining source centers did not identify the
best drop-one source subset for the heldout target.

The paired dense all4 reliability result narrows this limitation: reliability
has useful direction for dense aggregation, but the sparse top3/drop-one
formulations tested so far are not reliable.

## Limitations

Some older artifacts are quarantined or superseded. Do not use quarantined results for method-selection claims.

## Next Checks

- For each final negative claim, cite the current non-quarantined decision table or report.
- Keep unsafe claims out of thesis prose.
- Before adding another sparse selector, reuse paired generation/prediction
  invariants so identical source sets are not confounded by method-specific
  sampling seeds.
