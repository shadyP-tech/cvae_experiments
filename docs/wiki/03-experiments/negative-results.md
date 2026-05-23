# Negative Results

## Purpose

Preserve negative findings as part of the research record.

## Key Claims

- Naive metadata/domain-ID conditioning did not solve compatibility-aligned utility under the locked historical protocol.
- Historical BreakHis routed CVAE metadata routing was technically correct but underperformed the global CVAE.
- Support-distance and learned/sparse top-1 routing signals were often too brittle.
- C7.1a source-probe CE is negative diagnostic evidence in the C6.3 synthesis.

## Evidence / Source Artifacts

- `../../../cvae_testing/thesis_outline.txt`
- `../../../cvae_testing/results/compact_interpretation_summary.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `../../../cvae_testing/results/comparison_tables/learned_utility_decision_summary_v2.md`
- `../../../cvae_testing/results/comparison_tables/learned_utility_decision_summary_v2_strict.md`
- `../../../PROTOCOL_STATUS.md`

## Interpretation

Negative results drove the pivot: metadata and similarity proxies are not enough unless they predict expected utility. Dense aggregation is useful because the problem often appears as high regret from brittle top-1 selection.

## Implication For Thesis

The thesis should present negative findings as evidence for claim discipline and for the move toward utility-aligned compatibility.

## Limitations

Some older artifacts are quarantined or superseded. Do not use quarantined results for method-selection claims.

## Next Checks

- For each final negative claim, cite the current non-quarantined decision table or report.
- Keep unsafe claims out of thesis prose.
