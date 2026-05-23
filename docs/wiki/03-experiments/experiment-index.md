# Experiment Index

## Purpose

Summarize major experiment families and their relation to the current thesis direction.

## Key Claims

| Family | Purpose | Current conclusion | Relation to current best approach | Open TODOs |
| --- | --- | --- | --- | --- |
| Z-series | Representation ceiling and synthetic preservation audits. | Z1.1 local report says real-feature ceiling rows were missing and synthetic evidence was missing. | Baseline context for why R1.2/R1.2b were needed. | Verify any newer Z-series artifacts before quoting numbers. |
| R-series | Real-feature pathology embedding routing/source-selected diagnostics. | R1.2b makes Virchow2 the current strongest source-selected backbone but top-1 selection is brittle. | Directly motivates SAIL. | Run or sync SAIL outputs. |
| C-series | CVAE generator and aggregation experiments. | C6.3 synthesis supports dense late aggregation over frozen CVAE components, but not Virchow2 CVAE preservation. | Provides aggregation precedent and the CVAE preservation question. | Verify C6.3 numbers against raw/synced artifacts. |
| F-series | Residual or empirical transfer generator diagnostics, if present. | Residual tables exist, but current docs treat them as diagnostic unless protocol-clean adoption is established. | Baseline/diagnostic context, not current best path. | Verify exact F-series naming and claim status. |

## Evidence / Source Artifacts

- `../../../cvae_downstream_evaluation/artifacts/reports/z11_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/reports/r12_decision_report.md`
- `../../../cvae_downstream_evaluation/artifacts/r12b_source_selector_pathology_screen/reports/r12b_decision_report.md`
- `../../../cvae_downstream_evaluation/docs/c63_conceptual_synthesis.md`
- `../../../cvae_testing/results/comparison_tables/residual_routing_decision_table.csv`
- `../../../cvae_testing/results/comparison_tables/residual_safe_v2_decision_table.csv`

## Interpretation

The experiment history supports a pivot from metadata-first or sparse top-1 routing toward utility-driven compatibility and dense risk reduction.

## Implication For Thesis

The current thesis narrative should center SAIL as the real-feature source-only aggregation gate: it tests whether Virchow2 transfer is stable enough to justify CVAE preservation testing.

## Limitations

The index is a synthesis. It does not replace individual decision reports or leakage reports.

## Next Checks

- Add SAIL result rows when available.
- Decide whether F-series should be renamed once exact artifact provenance is verified.
